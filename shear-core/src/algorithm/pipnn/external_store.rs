use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use memmap2::{Advice, Mmap, MmapOptions};
use rayon::prelude::*;

use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::hash_prune::{SketchAccessor, SketchStore, generate_hyperplanes};
use super::params::PiPNNParams;
use super::point_store::{
    InmemDatasetPointStore, PointStore, WindowedGatherOptions, WindowedGatherStats,
    build_gather_plan, scatter_window_rows,
};
use crate::common::{AnnError, AnnResult};
use crate::model::InmemDataset;
use crate::utils::thread_pool::with_rayon_thread_pool;

const HEADER_BYTES: usize = size_of::<u64>() * 2;
const DEFAULT_SKETCH_BUILD_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const MIN_SKETCH_BUILD_CHUNK_ROWS: usize = 1024;

#[derive(Debug)]
pub struct DiskSketchStore {
    path: PathBuf,
    backing: SketchBacking,
    rows: usize,
    width: usize,
}

#[derive(Debug)]
pub(crate) struct ResidentSketchChunkBuilder {
    path: PathBuf,
    rows: usize,
    dim: usize,
    width: usize,
    hyperplanes: Arc<Vec<f32>>,
    data: Vec<f32>,
    threads: usize,
    compute_wall: Duration,
}

impl ResidentSketchChunkBuilder {
    pub(crate) fn new(path: &Path, rows: usize, dim: usize, params: &PiPNNParams) -> Self {
        Self::new_with_threads(path, rows, dim, params, 1)
    }

    pub(crate) fn new_with_threads(
        path: &Path,
        rows: usize,
        dim: usize,
        params: &PiPNNParams,
        requested_threads: u32,
    ) -> Self {
        let width = params.m_hash_bits;
        Self {
            path: path.to_path_buf(),
            rows,
            dim,
            width,
            hyperplanes: Arc::new(generate_hyperplanes(dim, width, params.random_seed)),
            data: vec![0.0; rows.saturating_mul(width)],
            threads: sketch_build_threads(params, dim, requested_threads),
            compute_wall: Duration::ZERO,
        }
    }

    pub(crate) fn write_raw_chunk(
        &mut self,
        row_start: usize,
        raw_vectors_row_major: &[f32],
    ) -> AnnResult<()> {
        if self.dim == 0 {
            return Err(AnnError::log_index_error(
                "Resident sketch raw chunk builder requires dim > 0".to_string(),
            ));
        }
        if raw_vectors_row_major.len() % self.dim != 0 {
            return Err(AnnError::log_index_error(format!(
                "Resident sketch raw chunk has {} values, not divisible by dim {}",
                raw_vectors_row_major.len(),
                self.dim
            )));
        }
        let chunk_rows = raw_vectors_row_major.len() / self.dim;
        let row_end = row_start.checked_add(chunk_rows).ok_or_else(|| {
            AnnError::log_index_error("Resident sketch row range overflow".to_string())
        })?;
        if row_end > self.rows {
            return Err(AnnError::log_index_error(format!(
                "Resident sketch raw chunk rows [{}..{}) exceed {} rows",
                row_start, row_end, self.rows
            )));
        }

        let compute_start = Instant::now();
        if self.width > 0 {
            let value_start = row_start.saturating_mul(self.width);
            let value_end = row_end.saturating_mul(self.width);
            let output = &mut self.data[value_start..value_end];
            if self.threads <= 1 || chunk_rows <= 1 {
                for (point, row) in raw_vectors_row_major
                    .chunks_exact(self.dim)
                    .zip(output.chunks_exact_mut(self.width))
                {
                    fill_sketch_row_from_slice(point, self.hyperplanes.as_slice(), row)?;
                }
            } else {
                let hyperplanes = Arc::clone(&self.hyperplanes);
                let dim = self.dim;
                let width = self.width;
                with_rayon_thread_pool(self.threads as u32, || {
                    raw_vectors_row_major
                        .par_chunks_exact(dim)
                        .zip(output.par_chunks_exact_mut(width))
                        .try_for_each(|(point, row)| {
                            fill_sketch_row_from_slice(point, hyperplanes.as_slice(), row)
                        })
                })??;
            }
        }
        self.compute_wall += compute_start.elapsed();
        Ok(())
    }

    pub(crate) fn compute_wall(&self) -> Duration {
        self.compute_wall
    }

    pub(crate) fn finish(self) -> AnnResult<DiskSketchStore> {
        DiskSketchStore::from_resident_row_major(&self.path, self.rows, self.width, self.data)
    }
}

#[derive(Debug)]
enum SketchBacking {
    Mmap { file: File, mmap: Mmap },
    Direct { file: DirectIoFile },
    Resident { data: Vec<f32> },
}

impl DiskSketchStore {
    pub(crate) fn from_resident_row_major(
        path: &Path,
        rows: usize,
        width: usize,
        data: Vec<f32>,
    ) -> AnnResult<Self> {
        if data.len() != rows.saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Resident sketch payload size mismatch: rows={} width={} values={}",
                rows,
                width,
                data.len()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            backing: SketchBacking::Resident { data },
            rows,
            width,
        })
    }

    pub fn write_from_memory(path: &Path, sketches: &SketchStore) -> AnnResult<Self> {
        Self::write_from_memory_with_config(path, sketches, DirectIoConfig::disabled())
    }

    pub fn write_from_memory_with_config(
        path: &Path,
        sketches: &SketchStore,
        io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        if io_cfg.enabled {
            write_sketch_file_direct(
                path,
                sketches.rows(),
                sketches.width(),
                sketches.data(),
                io_cfg,
            )?;
            return Self::open_with_config(path, io_cfg);
        }

        let mut out = BufWriter::new(File::create(path)?);
        let width = sketches.width();
        let rows = sketches.rows();

        out.write_all(&(rows as u64).to_le_bytes())?;
        out.write_all(&(width as u64).to_le_bytes())?;
        for value in sketches.data() {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;

        Self::open_with_config(path, io_cfg)
    }

    pub fn build_from_dataset(
        path: &Path,
        dataset: &InmemDataset<f32>,
        num_points: usize,
        params: &PiPNNParams,
        num_threads: u32,
    ) -> AnnResult<Self> {
        let store = InmemDatasetPointStore::new(dataset, num_points);
        Self::build_from_point_store(path, &store, num_points, params, num_threads)
    }

    pub fn build_from_point_store(
        path: &Path,
        dataset: &dyn PointStore,
        num_points: usize,
        params: &PiPNNParams,
        requested_threads: u32,
    ) -> AnnResult<Self> {
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let dim = dataset.dim();
        let width = params.m_hash_bits;
        let hyperplanes = generate_hyperplanes(dim, width, params.random_seed);
        let chunk_rows = sketch_build_chunk_rows(params, width, dim);
        let sketch_threads = sketch_build_threads(params, dim, requested_threads);
        if sketch_threads <= 1 {
            let mut all_data = if io_cfg.enabled {
                Vec::with_capacity(num_points.saturating_mul(width))
            } else {
                Vec::new()
            };
            let mut out = if io_cfg.enabled {
                None
            } else {
                let mut writer = BufWriter::new(File::create(path)?);
                writer.write_all(&(num_points as u64).to_le_bytes())?;
                writer.write_all(&(width as u64).to_le_bytes())?;
                Some(writer)
            };
            for chunk_start in (0..num_points).step_by(chunk_rows) {
                let rows = (num_points - chunk_start).min(chunk_rows);
                let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;

                for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                    let point = &point_chunk[offset * dim..(offset + 1) * dim];
                    fill_sketch_row_from_slice(point, &hyperplanes, row)?;
                }

                if let Some(writer) = out.as_mut() {
                    for value in &chunk_data {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                } else {
                    all_data.extend_from_slice(&chunk_data);
                }
            }
            if let Some(mut writer) = out {
                writer.flush()?;
            } else {
                write_sketch_file_direct(path, num_points, width, &all_data, io_cfg)?;
            }
        } else {
            let chunk_starts: Vec<usize> = (0..num_points).step_by(chunk_rows).collect();
            let hyperplanes = Arc::new(hyperplanes);

            if io_cfg.enabled {
                let writer = DirectIoFile::create_write(path, io_cfg)?;
                writer.write_all_at(&(num_points as u64).to_le_bytes(), 0)?;
                writer.write_all_at(&(width as u64).to_le_bytes(), size_of::<u64>() as u64)?;
                let expected_bytes = HEADER_BYTES.saturating_add(
                    num_points
                        .saturating_mul(width)
                        .saturating_mul(size_of::<f32>()),
                );

                with_rayon_thread_pool(sketch_threads as u32, || {
                    chunk_starts
                        .par_iter()
                        .try_for_each(|&chunk_start| -> AnnResult<()> {
                            let rows = (num_points - chunk_start).min(chunk_rows);
                            let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                            let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                            dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;

                            for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                                let point = &point_chunk[offset * dim..(offset + 1) * dim];
                                fill_sketch_row_from_slice(point, hyperplanes.as_slice(), row)?;
                            }

                            let byte_offset = HEADER_BYTES as u64
                                + (chunk_start as u64)
                                    .saturating_mul(width as u64)
                                    .saturating_mul(size_of::<f32>() as u64);
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    chunk_data.as_ptr() as *const u8,
                                    std::mem::size_of_val(chunk_data.as_slice()),
                                )
                            };
                            writer.write_all_at(bytes, byte_offset)?;
                            Ok(())
                        })
                })??;
                writer.set_len(expected_bytes as u64)?;
                writer.sync_all()?;
            } else {
                let chunk_results = with_rayon_thread_pool(sketch_threads as u32, || {
                    chunk_starts
                        .par_iter()
                        .map(|&chunk_start| -> AnnResult<(usize, Vec<f32>)> {
                            let rows = (num_points - chunk_start).min(chunk_rows);
                            let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                            let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                            dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;

                            for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                                let point = &point_chunk[offset * dim..(offset + 1) * dim];
                                fill_sketch_row_from_slice(point, hyperplanes.as_slice(), row)?;
                            }

                            Ok((chunk_start, chunk_data))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })??;

                let mut chunk_results = chunk_results;
                chunk_results.sort_unstable_by_key(|(chunk_start, _)| *chunk_start);
                let mut writer = BufWriter::new(File::create(path)?);
                writer.write_all(&(num_points as u64).to_le_bytes())?;
                writer.write_all(&(width as u64).to_le_bytes())?;
                for (_, chunk_data) in chunk_results {
                    for value in &chunk_data {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                writer.flush()?;
            }
        }

        Self::open_with_config(path, io_cfg)
    }

    pub fn build_resident_from_point_store(
        path: &Path,
        dataset: &dyn PointStore,
        num_points: usize,
        params: &PiPNNParams,
        requested_threads: u32,
    ) -> AnnResult<Self> {
        let dim = dataset.dim();
        let width = params.m_hash_bits;
        let hyperplanes = generate_hyperplanes(dim, width, params.random_seed);
        let chunk_rows = sketch_build_chunk_rows(params, width, dim);
        let sketch_threads = sketch_build_threads(params, dim, requested_threads);
        let mut data = Vec::with_capacity(num_points.saturating_mul(width));
        if sketch_threads <= 1 {
            for chunk_start in (0..num_points).step_by(chunk_rows) {
                let rows = (num_points - chunk_start).min(chunk_rows);
                let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;

                for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                    let point = &point_chunk[offset * dim..(offset + 1) * dim];
                    fill_sketch_row_from_slice(point, &hyperplanes, row)?;
                }
                data.extend_from_slice(&chunk_data);
            }
        } else {
            let chunk_starts: Vec<usize> = (0..num_points).step_by(chunk_rows).collect();
            let hyperplanes = Arc::new(hyperplanes);
            let chunk_results = with_rayon_thread_pool(sketch_threads as u32, || {
                chunk_starts
                    .par_iter()
                    .map(|&chunk_start| -> AnnResult<(usize, Vec<f32>)> {
                        let rows = (num_points - chunk_start).min(chunk_rows);
                        let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                        let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                        dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;

                        for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                            let point = &point_chunk[offset * dim..(offset + 1) * dim];
                            fill_sketch_row_from_slice(point, hyperplanes.as_slice(), row)?;
                        }

                        Ok((chunk_start, chunk_data))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })??;

            let mut chunk_results = chunk_results;
            chunk_results.sort_unstable_by_key(|(chunk_start, _)| *chunk_start);
            for (_, chunk_data) in chunk_results {
                data.extend_from_slice(&chunk_data);
            }
        }

        Self::from_resident_row_major(path, num_points, width, data)
    }

    pub fn open(path: &Path) -> AnnResult<Self> {
        Self::open_with_config(path, DirectIoConfig::disabled())
    }

    pub fn open_with_config(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        let rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let width = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

        let file_bytes = file.metadata()?.len() as usize;
        let expected_bytes = HEADER_BYTES
            .checked_add(
                rows.checked_mul(width)
                    .and_then(|values| values.checked_mul(size_of::<f32>()))
                    .ok_or_else(|| {
                        AnnError::log_index_error("Sketch file size overflow".to_string())
                    })?,
            )
            .ok_or_else(|| AnnError::log_index_error("Sketch file size overflow".to_string()))?;

        if file_bytes != expected_bytes {
            return Err(AnnError::log_index_error(format!(
                "Sketch file size mismatch: expected {expected_bytes}, found {file_bytes}"
            )));
        }

        let backing = if io_cfg.enabled {
            SketchBacking::Direct {
                file: DirectIoFile::open_read(path, io_cfg)?,
            }
        } else {
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let _ = mmap.advise(Advice::Sequential);
            SketchBacking::Mmap { file, mmap }
        };

        Ok(Self {
            path: path.to_path_buf(),
            backing,
            rows,
            width,
        })
    }

    fn row_range(&self, idx: usize) -> AnnResult<(usize, usize)> {
        if idx >= self.rows {
            return Err(AnnError::log_index_error(format!(
                "Sketch row {idx} is out of bounds for {} rows",
                self.rows
            )));
        }

        let row_bytes = self
            .width
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| AnnError::log_index_error("Sketch row size overflow".to_string()))?;
        let start = HEADER_BYTES
            .checked_add(idx.checked_mul(row_bytes).ok_or_else(|| {
                AnnError::log_index_error("Sketch row offset overflow".to_string())
            })?)
            .ok_or_else(|| AnnError::log_index_error("Sketch row offset overflow".to_string()))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| AnnError::log_index_error("Sketch row end overflow".to_string()))?;

        Ok((start, end))
    }

    fn read_sketch_range_into(&self, start: usize, count: usize, out: &mut [f32]) -> AnnResult<()> {
        let row_bytes = self.width * size_of::<f32>();
        let byte_offset = HEADER_BYTES + start * row_bytes;
        let byte_len = count * row_bytes;
        match &self.backing {
            SketchBacking::Mmap { mmap, .. } => {
                let bytes = &mmap[byte_offset..byte_offset + byte_len];
                for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            SketchBacking::Direct { file } => {
                let mut bytes = vec![0u8; byte_len];
                file.read_exact_at(&mut bytes, byte_offset as u64)?;
                for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            SketchBacking::Resident { data } => {
                let start_value = start.checked_mul(self.width).ok_or_else(|| {
                    AnnError::log_index_error("Resident sketch range overflow".to_string())
                })?;
                let value_len = count.checked_mul(self.width).ok_or_else(|| {
                    AnnError::log_index_error("Resident sketch range overflow".to_string())
                })?;
                out.copy_from_slice(&data[start_value..start_value + value_len]);
            }
        }
        Ok(())
    }

    pub fn row(&self, idx: usize) -> AnnResult<Vec<f32>> {
        let mut row = vec![0.0f32; self.width];
        self.row_copy_into(idx, &mut row)?;
        Ok(row)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn file_bytes(&self) -> usize {
        if matches!(self.backing, SketchBacking::Resident { .. }) {
            return 0;
        }
        std::fs::metadata(&self.path)
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0)
    }

    pub(crate) fn is_resident(&self) -> bool {
        matches!(self.backing, SketchBacking::Resident { .. })
    }

    #[cfg(test)]
    fn uses_mmap(&self) -> bool {
        matches!(self.backing, SketchBacking::Mmap { .. })
    }
}

pub(crate) struct ResidentPrefixSketchAccessor<'a> {
    prefix: SketchStore,
    backing: &'a DiskSketchStore,
}

impl<'a> ResidentPrefixSketchAccessor<'a> {
    pub(crate) fn from_disk_prefix(
        backing: &'a DiskSketchStore,
        prefix_rows: usize,
    ) -> AnnResult<Self> {
        if prefix_rows > backing.rows {
            return Err(AnnError::log_index_error(format!(
                "Resident sketch prefix rows out of bounds: prefix_rows={} rows={}",
                prefix_rows, backing.rows
            )));
        }

        let width = backing.width;
        let mut data = if backing.is_resident() {
            Vec::new()
        } else {
            vec![0.0f32; prefix_rows.saturating_mul(width)]
        };
        if !backing.is_resident() && prefix_rows > 0 {
            backing.read_sketch_range_into(0, prefix_rows, &mut data)?;
        }

        Ok(Self {
            prefix: SketchStore::from_row_major(data, width),
            backing,
        })
    }

    #[inline]
    fn prefix_rows(&self) -> usize {
        self.prefix.rows()
    }

    fn scatter_subset_rows(
        subset_positions: &[usize],
        subset_out: &[f32],
        row_width: usize,
        out: &mut [f32],
    ) {
        for (subset_idx, &out_row) in subset_positions.iter().enumerate() {
            let src_start = subset_idx * row_width;
            let src_end = src_start + row_width;
            let dst_start = out_row * row_width;
            let dst_end = dst_start + row_width;
            out[dst_start..dst_end].copy_from_slice(&subset_out[src_start..src_end]);
        }
    }
}

impl SketchAccessor for ResidentPrefixSketchAccessor<'_> {
    fn width(&self) -> usize {
        self.backing.width()
    }

    fn rows(&self) -> usize {
        self.backing.rows()
    }

    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()> {
        if idx < self.prefix_rows() {
            self.prefix.row_copy_into(idx, dst)
        } else {
            self.backing.row_copy_into(idx, dst)
        }
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let width = self.width();
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Resident prefix sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        for (row, &id) in ids.iter().enumerate() {
            let start = row * width;
            let end = start + width;
            self.row_copy_into(id as usize, &mut out[start..end])?;
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let width = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Resident prefix sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        let prefix_rows = self.prefix_rows() as u32;
        if prefix_rows == 0 {
            return self
                .backing
                .read_rows_windowed_into_stats(ids, out, options, stats);
        }

        let mut prefix_ids = Vec::new();
        let mut prefix_positions = Vec::new();
        let mut tail_ids = Vec::new();
        let mut tail_positions = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            if id < prefix_rows {
                prefix_ids.push(id);
                prefix_positions.push(pos);
            } else {
                tail_ids.push(id);
                tail_positions.push(pos);
            }
        }

        if !prefix_ids.is_empty() {
            let mut prefix_out = vec![0.0f32; prefix_ids.len() * width];
            let mut prefix_stats = WindowedGatherStats::default();
            self.prefix.read_rows_windowed_into_stats(
                &prefix_ids,
                &mut prefix_out,
                options,
                &mut prefix_stats,
            )?;
            prefix_stats.physical_bytes = 0;
            prefix_stats.alignment_waste_bytes = 0;
            Self::scatter_subset_rows(&prefix_positions, &prefix_out, width, out);
            stats.merge(&prefix_stats);
        }

        if !tail_ids.is_empty() {
            let mut tail_out = vec![0.0f32; tail_ids.len() * width];
            let mut tail_stats = WindowedGatherStats::default();
            self.backing.read_rows_windowed_into_stats(
                &tail_ids,
                &mut tail_out,
                options,
                &mut tail_stats,
            )?;
            Self::scatter_subset_rows(&tail_positions, &tail_out, width, out);
            stats.merge(&tail_stats);
        }

        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        size_of::<Self>() + self.prefix.resident_bytes()
    }
}

fn sketch_build_chunk_rows(params: &PiPNNParams, width: usize, dim: usize) -> usize {
    let bytes_per_row = width
        .saturating_mul(size_of::<f32>())
        .saturating_add(dim.saturating_mul(size_of::<f32>()))
        .max(1);
    let budget_bytes = if params.oom_sketch_cache_bytes == 0 {
        DEFAULT_SKETCH_BUILD_CHUNK_BYTES
    } else {
        params
            .oom_sketch_cache_bytes
            .max(DEFAULT_SKETCH_BUILD_CHUNK_BYTES / 8)
    };
    (budget_bytes / bytes_per_row).max(MIN_SKETCH_BUILD_CHUNK_ROWS)
}

fn sketch_build_threads(params: &PiPNNParams, dim: usize, requested_threads: u32) -> usize {
    let budget_bytes = params.effective_oom_memory_budget_bytes();
    let requested = if requested_threads == 0 {
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
    } else {
        requested_threads as usize
    };
    let per_worker_bytes = dim
        .saturating_mul(params.m_hash_bits)
        .saturating_mul(size_of::<f32>())
        .saturating_add(dim.saturating_mul(size_of::<f32>()))
        .saturating_add(4 * 1024 * 1024)
        .max(1);
    let budget_threads = (budget_bytes / per_worker_bytes).max(1);
    requested.min(budget_threads).max(1)
}

pub(crate) fn fill_sketch_row_from_slice(
    point: &[f32],
    hyperplanes: &[f32],
    out_row: &mut [f32],
) -> AnnResult<()> {
    let dim = point.len();
    if hyperplanes.len() != out_row.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Hyperplane shape mismatch: hyperplanes={} out_row={} dim={}",
            hyperplanes.len(),
            out_row.len(),
            dim
        )));
    }

    for (h_idx, h) in hyperplanes.chunks_exact(dim).enumerate() {
        let mut acc = 0.0f32;
        for (a, b) in point.iter().zip(h.iter()) {
            acc += *a * *b;
        }
        out_row[h_idx] = acc;
    }
    Ok(())
}

impl SketchAccessor for DiskSketchStore {
    fn width(&self) -> usize {
        self.width
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()> {
        if dst.len() != self.width {
            return Err(AnnError::log_index_error(format!(
                "Sketch row copy width mismatch: dst={} expected={}",
                dst.len(),
                self.width
            )));
        }

        if let SketchBacking::Resident { data } = &self.backing {
            if idx >= self.rows {
                return Err(AnnError::log_index_error(format!(
                    "Sketch row {idx} is out of bounds for {} rows",
                    self.rows
                )));
            }
            let start = idx * self.width;
            dst.copy_from_slice(&data[start..start + self.width]);
            return Ok(());
        }

        let (start, end) = self.row_range(idx)?;
        match &self.backing {
            SketchBacking::Mmap { mmap, .. } => {
                let bytes = mmap.get(start..end).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Sketch row {idx} byte range is out of bounds for mapped file"
                    ))
                })?;
                for (slot, chunk) in dst.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            SketchBacking::Direct { file } => {
                let mut bytes = vec![0u8; end - start];
                file.read_exact_at(&mut bytes, start as u64)?;
                for (slot, chunk) in dst.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            SketchBacking::Resident { .. } => unreachable!("resident backing handled above"),
        }
        Ok(())
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let w = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        // Coalesce contiguous runs of consecutive IDs into range reads
        let mut run_start = 0usize;
        while run_start < ids.len() {
            let mut run_end = run_start + 1;
            while run_end < ids.len() && ids[run_end] == ids[run_end - 1] + 1 {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            let start_idx = ids[run_start] as usize;
            self.read_sketch_range_into(start_idx, run_len, &mut out[run_start * w..run_end * w])?;
            run_start = run_end;
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let width = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        let row_bytes = width * size_of::<f32>();
        let plan = build_gather_plan(ids, row_bytes, *options, stats);
        let mut window_buf = vec![
            0.0f32;
            plan.windows
                .iter()
                .map(|window| window.row_count as usize)
                .max()
                .unwrap_or(0)
                * width
        ];
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let rows = window.row_count as usize;
            let elems = rows * width;
            let window_slice = &mut window_buf[..elems];
            self.read_sketch_range_into(window.start_row as usize, rows, window_slice)?;
            scatter_window_rows(&plan, window_idx, width, window_slice, out);
        }
        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + match &self.backing {
                SketchBacking::Mmap { .. } => 0,
                SketchBacking::Direct { .. } => 0,
                SketchBacking::Resident { data } => {
                    data.capacity().saturating_mul(size_of::<f32>())
                }
            }
    }
}

fn write_sketch_file_direct(
    path: &Path,
    rows: usize,
    width: usize,
    data: &[f32],
    io_cfg: DirectIoConfig,
) -> AnnResult<()> {
    if data.len() != rows.saturating_mul(width) {
        return Err(AnnError::log_index_error(format!(
            "Sketch payload size mismatch: rows={} width={} values={}",
            rows,
            width,
            data.len()
        )));
    }

    let file = DirectIoFile::create_write(path, io_cfg)?;
    let mut bytes = Vec::with_capacity(HEADER_BYTES + data.len().saturating_mul(size_of::<f32>()));
    bytes.extend_from_slice(&(rows as u64).to_le_bytes());
    bytes.extend_from_slice(&(width as u64).to_le_bytes());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all_at(&bytes, 0)?;
    file.set_len(bytes.len() as u64)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::{DiskSketchStore, ResidentPrefixSketchAccessor, ResidentSketchChunkBuilder};
    use crate::algorithm::pipnn::PiPNNParams;
    use crate::algorithm::pipnn::direct_io::DirectIoConfig;
    use crate::algorithm::pipnn::hash_prune::{SketchAccessor, compute_sketches};
    use crate::algorithm::pipnn::point_store::{
        DirectPointStore, InmemDatasetPointStore, PointStore, WindowedGatherOptions,
        WindowedGatherStats,
    };
    use crate::model::InmemDataset;
    use crate::utils::save_bin_f32;

    fn build_test_dataset(num_points: usize, dim: usize) -> InmemDataset<f32> {
        let data: Vec<f32> = (0..num_points)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.25 + col as f32 * 0.5))
            .collect();
        build_test_dataset_from_data(num_points, dim, &data)
    }

    fn build_test_dataset_from_data(
        num_points: usize,
        dim: usize,
        data: &[f32],
    ) -> InmemDataset<f32> {
        assert_eq!(data.len(), num_points * dim);
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        dataset.data.copy_from_slice(data);
        dataset
    }

    struct CountingRangePointStore {
        data: Vec<f32>,
        rows: usize,
        dim: usize,
        point_reads: Arc<AtomicUsize>,
        range_reads: Arc<AtomicUsize>,
    }

    impl CountingRangePointStore {
        fn new(
            rows: usize,
            dim: usize,
            data: Vec<f32>,
        ) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let point_reads = Arc::new(AtomicUsize::new(0));
            let range_reads = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    data,
                    rows,
                    dim,
                    point_reads: Arc::clone(&point_reads),
                    range_reads: Arc::clone(&range_reads),
                },
                point_reads,
                range_reads,
            )
        }
    }

    struct BlockingRangePointStore {
        data: Vec<f32>,
        rows: usize,
        dim: usize,
        inflight_reads: Arc<AtomicUsize>,
        max_inflight_reads: Arc<AtomicUsize>,
    }

    impl BlockingRangePointStore {
        fn new(rows: usize, dim: usize, data: Vec<f32>) -> (Self, Arc<AtomicUsize>) {
            let inflight_reads = Arc::new(AtomicUsize::new(0));
            let max_inflight_reads = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    data,
                    rows,
                    dim,
                    inflight_reads,
                    max_inflight_reads: Arc::clone(&max_inflight_reads),
                },
                max_inflight_reads,
            )
        }
    }

    impl PointStore for BlockingRangePointStore {
        fn len(&self) -> usize {
            self.rows
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn read_point_into(&self, pid: u32, out: &mut [f32]) -> crate::common::AnnResult<()> {
            let point = pid as usize;
            let start = point * self.dim;
            let end = start + self.dim;
            out.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn read_range_into(
            &self,
            start_pid: u32,
            count: usize,
            out: &mut [f32],
        ) -> crate::common::AnnResult<()> {
            let inflight = self.inflight_reads.fetch_add(1, Ordering::AcqRel) + 1;
            let mut observed = self.max_inflight_reads.load(Ordering::Acquire);
            while inflight > observed {
                match self.max_inflight_reads.compare_exchange_weak(
                    observed,
                    inflight,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(next) => observed = next,
                }
            }

            thread::sleep(Duration::from_millis(25));
            let start = start_pid as usize * self.dim;
            let end = start + count * self.dim;
            out.copy_from_slice(&self.data[start..end]);
            self.inflight_reads.fetch_sub(1, Ordering::AcqRel);
            Ok(())
        }

        fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> crate::common::AnnResult<()> {
            for (row, &pid) in ids.iter().enumerate() {
                let start = row * self.dim;
                let end = start + self.dim;
                self.read_point_into(pid, &mut out[start..end])?;
            }
            Ok(())
        }
    }

    impl PointStore for CountingRangePointStore {
        fn len(&self) -> usize {
            self.rows
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn read_point_into(&self, pid: u32, out: &mut [f32]) -> crate::common::AnnResult<()> {
            self.point_reads.fetch_add(1, Ordering::Relaxed);
            let point = pid as usize;
            let start = point * self.dim;
            let end = start + self.dim;
            out.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn read_range_into(
            &self,
            start_pid: u32,
            count: usize,
            out: &mut [f32],
        ) -> crate::common::AnnResult<()> {
            self.range_reads.fetch_add(1, Ordering::Relaxed);
            let start = start_pid as usize * self.dim;
            let end = start + count * self.dim;
            out.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> crate::common::AnnResult<()> {
            for (row, &pid) in ids.iter().enumerate() {
                let start = row * self.dim;
                let end = start + self.dim;
                self.read_point_into(pid, &mut out[start..end])?;
            }
            Ok(())
        }
    }

    #[test]
    fn disk_sketch_store_round_trips_rows_from_dataset() {
        let dataset = build_test_dataset(16, 8);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 6;
        let sketches = compute_sketches(&dataset, 16, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches.bin");

        let store = DiskSketchStore::write_from_memory(&path, &sketches).unwrap();

        assert_eq!(store.width(), params.m_hash_bits);
        for point in 0..16 {
            let mut row = vec![0.0f32; params.m_hash_bits];
            store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, sketches.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_reports_file_backed_bytes() {
        let dataset = build_test_dataset(8, 4);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 5;
        let sketches = compute_sketches(&dataset, 8, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches.bin");

        let store = DiskSketchStore::write_from_memory(&path, &sketches).unwrap();

        assert_eq!(store.path(), PathBuf::from(&path));
        assert!(store.file_bytes() >= 8 * 5 * std::mem::size_of::<f32>());
    }

    #[test]
    fn disk_sketch_store_reads_rows_without_full_clone() {
        let dataset = build_test_dataset(12, 6);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 7;
        let expected = compute_sketches(&dataset, 12, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-direct.bin");

        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 12, &params, 1).unwrap();

        assert_eq!(store.rows(), 12);
        assert_eq!(store.width(), 7);
        assert!(
            store.resident_bytes() < store.file_bytes(),
            "disk sketch store should avoid cloning the full sketch matrix into heap memory"
        );

        for point in 0..12 {
            let mut row = vec![0.0f32; params.m_hash_bits];
            store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_builds_from_direct_point_store() {
        let rows = 12usize;
        let dim = 6usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.25 + col as f32 * 0.5))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("tiny.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open(&dataset_path).unwrap();
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 7;

        let sketch_path = dir.path().join("sketches-direct-store.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 1)
                .unwrap();

        let expected_dataset = build_test_dataset(rows, dim);
        let expected = compute_sketches(&expected_dataset, rows, &params, 1).unwrap();
        for point in 0..rows {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_parallel_build_matches_reference_sketches() {
        let rows = 257usize;
        let dim = 32usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.03125 + col as f32 * 0.125))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("parallel-tiny.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open(&dataset_path).unwrap();
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 11;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 64 * 1024 * 1024;

        let sketch_path = dir.path().join("sketches-parallel-direct-store.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        let expected_dataset = build_test_dataset_from_data(rows, dim, &data);
        let expected = compute_sketches(&expected_dataset, rows, &params, 4).unwrap();
        for point in 0..rows {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn resident_sketch_chunk_builder_matches_point_store_build_for_raw_chunks() {
        let rows = 9usize;
        let dim = 5usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| (row as f32 * 0.375) - (col as f32 * 0.125)))
            .collect();
        let dataset = build_test_dataset_from_data(rows, dim, &data);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 7;

        let dir = tempdir().unwrap();
        let expected_path = dir.path().join("expected-sketches.bin");
        let expected_store = InmemDatasetPointStore::new(&dataset, rows);
        let expected = DiskSketchStore::build_resident_from_point_store(
            &expected_path,
            &expected_store,
            rows,
            &params,
            1,
        )
        .unwrap();

        let actual_path = dir.path().join("actual-sketches.bin");
        let mut builder = ResidentSketchChunkBuilder::new(&actual_path, rows, dim, &params);
        builder.write_raw_chunk(4, &data[4 * dim..7 * dim]).unwrap();
        builder.write_raw_chunk(0, &data[0..4 * dim]).unwrap();
        builder
            .write_raw_chunk(7, &data[7 * dim..rows * dim])
            .unwrap();
        assert!(builder.compute_wall() > Duration::ZERO);
        let actual = builder.finish().unwrap();

        assert_eq!(actual.rows(), expected.rows());
        assert_eq!(actual.width(), expected.width());
        for row in 0..rows {
            let mut expected_row = vec![0.0f32; params.m_hash_bits];
            let mut actual_row = vec![0.0f32; params.m_hash_bits];
            expected.row_copy_into(row, &mut expected_row).unwrap();
            actual.row_copy_into(row, &mut actual_row).unwrap();
            assert_eq!(actual_row, expected_row);
        }
    }

    #[test]
    fn disk_sketch_store_prefers_range_reads_when_building_from_point_store() {
        let rows = 129usize;
        let dim = 24usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.125 + col as f32 * 0.0625))
            .collect();
        let (store, point_reads, range_reads) =
            CountingRangePointStore::new(rows, dim, data.clone());
        let dir = tempdir().unwrap();
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 9;
        params.oom_memory_budget_bytes = 256 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 8 * 1024 * 1024;

        let sketch_path = dir.path().join("sketches-range-first.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        assert!(range_reads.load(Ordering::Relaxed) > 0);
        assert_eq!(point_reads.load(Ordering::Relaxed), 0);

        let expected_dataset = build_test_dataset_from_data(rows, dim, &data);
        let expected = compute_sketches(&expected_dataset, rows, &params, 4).unwrap();
        for point in 0..rows {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_can_overlap_range_reads_during_parallel_build() {
        let rows = 65_536usize;
        let dim = 64usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.015625 + col as f32 * 0.03125))
            .collect();
        let (store, max_inflight_reads) = BlockingRangePointStore::new(rows, dim, data);
        let dir = tempdir().unwrap();
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 11;
        params.oom_enable = true;
        params.oom_strict_io = true;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 256 * 1024;

        let sketch_path = dir.path().join("sketches-overlap.bin");
        let _disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        assert!(
            max_inflight_reads.load(Ordering::Relaxed) > 1,
            "expected sketch build to overlap multiple range reads when parallel workers are available"
        );
    }

    #[test]
    fn disk_sketch_store_parallel_strict_io_matches_reference_sketches() {
        let rows = 4096usize;
        let dim = 64usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.0625 + col as f32 * 0.015625))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("parallel-strict.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open_with_config(
            &dataset_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 11;
        params.oom_enable = true;
        params.oom_strict_io = true;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 256 * 1024;

        let sketch_path = dir.path().join("sketches-parallel-strict.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();
        assert!(
            !disk_store.uses_mmap(),
            "strict sketch path must stay off mmap"
        );

        let expected_dataset = build_test_dataset_from_data(rows, dim, &data);
        let expected = compute_sketches(&expected_dataset, rows, &params, 4).unwrap();
        for point in [0usize, 7, 255, 1024, rows - 1] {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_strict_io_avoids_mmap_and_round_trips_rows() {
        let dataset = build_test_dataset(19, 9);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 5;
        params.oom_enable = true;
        params.oom_strict_io = true;
        let expected = compute_sketches(&dataset, 19, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-strict.bin");

        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 19, &params, 1).unwrap();
        assert!(
            !store.uses_mmap(),
            "strict sketch path must not reopen the file via mmap"
        );

        for point in 0..19 {
            let mut row = vec![0.0f32; params.m_hash_bits];
            store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }

        let reopened =
            DiskSketchStore::open_with_config(&path, DirectIoConfig::enabled_with_alignment(4096))
                .unwrap();
        assert!(!reopened.uses_mmap());
        let mut row = vec![0.0f32; params.m_hash_bits];
        reopened.row_copy_into(7, &mut row).unwrap();
        assert_eq!(row, expected.row(7));
    }

    #[test]
    fn disk_sketch_store_windowed_reads_preserve_logical_order_for_unsorted_ids() {
        let dataset = build_test_dataset(16, 8);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 6;
        let expected = compute_sketches(&dataset, 16, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-windowed-unsorted.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 16, &params, 1).unwrap();

        let ids = vec![9_u32, 2, 10, 3];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        store
            .read_rows_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 2);
        for (row, &pid) in ids.iter().enumerate() {
            let start = row * params.m_hash_bits;
            let end = start + params.m_hash_bits;
            assert_eq!(&out[start..end], expected.row(pid as usize));
        }
    }

    #[test]
    fn disk_sketch_store_windowed_reads_merge_small_gaps_into_single_window() {
        let dataset = build_test_dataset(16, 8);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 6;

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-windowed-gap.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 16, &params, 1).unwrap();

        let ids = vec![4_u32, 6];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        store
            .read_rows_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 1);
        assert_eq!(stats.overread_rows, 1);
    }

    #[test]
    fn resident_prefix_sketch_accessor_preserves_logical_order_for_mixed_cached_and_tail_ids() {
        let dataset = build_test_dataset(16, 8);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 6;
        let expected = compute_sketches(&dataset, 16, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-prefix-cache.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 16, &params, 1).unwrap();
        let cached = ResidentPrefixSketchAccessor::from_disk_prefix(&store, 8).unwrap();

        let ids = vec![9_u32, 2, 10, 3, 7, 12, 0];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        cached
            .read_rows_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        for (row, &pid) in ids.iter().enumerate() {
            let start = row * params.m_hash_bits;
            let end = start + params.m_hash_bits;
            assert_eq!(&out[start..end], expected.row(pid as usize));
        }
    }
}
