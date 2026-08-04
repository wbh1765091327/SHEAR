use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::bridge::{Rabitq1Codec, generate_rotator, rotate_vectors};
use super::checksum::{Crc64, base_window_fingerprint, crc64, crc64_reader};
use super::format::{RABITQ1_HEADER_BYTES, Rabitq1Header, Rabitq1Sidecar, Section};
use super::kmeans::{deterministic_sample_ids, train_centers};
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug)]
pub struct BuildRabitq1Options {
    pub center_count: usize,
    pub sample_rows: usize,
    pub kmeans_iterations: usize,
    pub seed: u64,
    pub batch_rows: usize,
    pub threads: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct BuildRabitq1Report {
    pub schema: &'static str,
    pub rows: u64,
    pub dim: u32,
    pub padded_dim: u32,
    pub center_count: u32,
    pub code_bytes: u32,
    pub bin_stride: u32,
    pub ex_data_bytes: u64,
    pub sidecar_bytes: u64,
    pub base_full_crc64: u64,
    pub base_window_fingerprint: u64,
    pub output_crc64: u64,
    pub elapsed_seconds: f64,
}

impl BuildRabitq1Report {
    pub fn to_pretty_json(&self) -> AnnResult<String> {
        serde_json::to_string_pretty(self).map_err(|error| {
            AnnError::log_index_error(format!("serialize graph RaBitQ1 build report: {error}"))
        })
    }
}

fn builder_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(format!("build graph RaBitQ1 sidecar: {}", message.into()))
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value: OsString = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}

fn read_base_metadata(path: &Path) -> AnnResult<(u32, u32, u64)> {
    let mut file = File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut header = [0_u8; 8];
    file.read_exact(&mut header)?;
    let rows = u32::from_le_bytes(header[0..4].try_into().expect("fixed fbin field"));
    let dim = u32::from_le_bytes(header[4..8].try_into().expect("fixed fbin field"));
    if rows == 0 || dim == 0 {
        return Err(builder_error(
            "base fbin rows and dimension must be nonzero",
        ));
    }
    let expected_bytes = 8_u64
        .checked_add(
            u64::from(rows)
                .checked_mul(u64::from(dim))
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| builder_error("base fbin length overflow"))?,
        )
        .ok_or_else(|| builder_error("base fbin length overflow"))?;
    if file_bytes != expected_bytes {
        return Err(builder_error(format!(
            "base fbin byte length {file_bytes} does not match {expected_bytes}"
        )));
    }
    Ok((rows, dim, file_bytes))
}

fn gather_sample(path: &Path, dim: usize, ids: &[u64]) -> AnnResult<Vec<f32>> {
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| builder_error("sample row length overflow"))?;
    let capacity = ids
        .len()
        .checked_mul(dim)
        .ok_or_else(|| builder_error("sample allocation overflow"))?;
    let mut file = File::open(path)?;
    let mut raw = vec![0_u8; row_bytes];
    let mut sample = Vec::with_capacity(capacity);
    for &id in ids {
        let offset = 8_u64
            .checked_add(
                id.checked_mul(row_bytes as u64)
                    .ok_or_else(|| builder_error("sample offset overflow"))?,
            )
            .ok_or_else(|| builder_error("sample offset overflow"))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut raw)?;
        for chunk in raw.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().expect("four-byte sample value"));
            if !value.is_finite() {
                return Err(builder_error(format!(
                    "sample row {id} contains a non-finite value"
                )));
            }
            sample.push(value);
        }
    }
    Ok(sample)
}

fn section_offset(section: Section, relative: u64) -> AnnResult<u64> {
    section
        .offset
        .checked_add(relative)
        .ok_or_else(|| builder_error("section write offset overflow"))
}

fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> AnnResult<()> {
    while !bytes.is_empty() {
        let written = file.write_at(bytes, offset)?;
        if written == 0 {
            return Err(builder_error("short positioned write"));
        }
        bytes = &bytes[written..];
        offset = offset
            .checked_add(written as u64)
            .ok_or_else(|| builder_error("positioned write offset overflow"))?;
    }
    Ok(())
}

fn centers_to_bytes(centers: &[f32]) -> Vec<u8> {
    centers
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn fsync_parent(path: &Path) -> AnnResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| builder_error("output path has no parent directory"))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn publish_manifest(path: &Path, report: &BuildRabitq1Report) -> AnnResult<()> {
    let temporary = append_suffix(path, ".partial");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(report.to_pretty_json()?.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    fsync_parent(path)
}

fn validate_options(options: &BuildRabitq1Options, rows: usize) -> AnnResult<()> {
    if options.center_count != 16 {
        return Err(builder_error("center_count must equal 16"));
    }
    if options.kmeans_iterations != 20 {
        return Err(builder_error("kmeans_iterations must equal 20"));
    }
    if options.sample_rows == 0 || options.sample_rows > rows {
        return Err(builder_error("sample_rows must be in 1..=rows"));
    }
    if options.sample_rows < options.center_count {
        return Err(builder_error("sample_rows must be at least center_count"));
    }
    if options.batch_rows == 0 || options.threads == 0 {
        return Err(builder_error("batch_rows and threads must be positive"));
    }
    Ok(())
}

pub fn build_rabitq1_sidecar(
    base_path: &Path,
    output_path: &Path,
    options: &BuildRabitq1Options,
) -> AnnResult<BuildRabitq1Report> {
    let started = Instant::now();
    let manifest_path = append_suffix(output_path, ".json");
    let partial_path = append_suffix(output_path, ".partial");
    let manifest_partial_path = append_suffix(&manifest_path, ".partial");
    for path in [
        output_path,
        manifest_path.as_path(),
        partial_path.as_path(),
        manifest_partial_path.as_path(),
    ] {
        if path.exists() {
            return Err(builder_error(format!(
                "refusing to overwrite {}",
                path.display()
            )));
        }
    }

    let (rows_u32, dim_u32, base_file_bytes) = read_base_metadata(base_path)?;
    let rows = rows_u32 as usize;
    let dim = dim_u32 as usize;
    validate_options(options, rows)?;
    let sample_ids = deterministic_sample_ids(rows, options.sample_rows, options.seed)?;
    let sample = gather_sample(base_path, dim, &sample_ids)?;
    let raw_centers = train_centers(
        &sample,
        options.sample_rows,
        dim,
        options.center_count,
        options.kmeans_iterations,
        options.seed,
    )?;
    let rotator = generate_rotator(dim, options.seed)?;
    let rotated_centers = rotate_vectors(dim, &rotator, &raw_centers, options.center_count)?;
    let codec = Rabitq1Codec::new(
        dim,
        rotator.clone(),
        rotated_centers.clone(),
        options.center_count,
    )?;

    let mut fingerprint_file = File::open(base_path)?;
    let fingerprint = base_window_fingerprint(&mut fingerprint_file, base_file_bytes)?;
    let mut header = Rabitq1Header::new_layout(
        u64::from(rows_u32),
        dim_u32,
        options.center_count as u32,
        options.seed,
        base_file_bytes,
        0,
        fingerprint,
    )?;

    let output = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&partial_path)?;
    output.set_len(header.file_bytes)?;
    write_all_at(&output, &rotator, header.rotator.offset)?;
    let center_bytes = centers_to_bytes(&rotated_centers);
    write_all_at(&output, &center_bytes, header.centers.offset)?;

    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| builder_error("base row length overflow"))?;
    let mut base = File::open(base_path)?;
    let mut fbin_header = [0_u8; 8];
    base.read_exact(&mut fbin_header)?;
    let mut base_crc = Crc64::default();
    base_crc.update(&fbin_header);
    let mut center_ids_crc = Crc64::default();
    let mut bin_data_crc = Crc64::default();
    let mut row_start = 0_usize;
    while row_start < rows {
        let chunk_rows = options.batch_rows.min(rows - row_start);
        let chunk_bytes = chunk_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| builder_error("base chunk length overflow"))?;
        let mut raw = vec![0_u8; chunk_bytes];
        base.read_exact(&mut raw)?;
        base_crc.update(&raw);
        let mut points = Vec::with_capacity(chunk_rows * dim);
        for (index, chunk) in raw.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes(chunk.try_into().expect("four-byte base value"));
            if !value.is_finite() {
                return Err(builder_error(format!(
                    "base value {} is non-finite",
                    row_start * dim + index
                )));
            }
            points.push(value);
        }
        let quantized = codec.quantize(&points, chunk_rows, options.threads)?;
        center_ids_crc.update(&quantized.center_ids);
        bin_data_crc.update(&quantized.bin_data);
        write_all_at(
            &output,
            &quantized.center_ids,
            section_offset(header.center_ids, row_start as u64)?,
        )?;
        let bin_relative = (row_start as u64)
            .checked_mul(u64::from(header.bin_stride))
            .ok_or_else(|| builder_error("BinData write offset overflow"))?;
        write_all_at(
            &output,
            &quantized.bin_data,
            section_offset(header.bin_data, bin_relative)?,
        )?;
        row_start += chunk_rows;
    }

    header.base_full_crc64 = base_crc.finish();
    header.rotator.crc64 = crc64(&rotator);
    header.centers.crc64 = crc64(&center_bytes);
    header.center_ids.crc64 = center_ids_crc.finish();
    header.bin_data.crc64 = bin_data_crc.finish();
    output.sync_all()?;

    let mut payload_reader = File::open(&partial_path)?;
    payload_reader.seek(SeekFrom::Start(RABITQ1_HEADER_BYTES as u64))?;
    header.payload_crc64 = crc64_reader(&mut payload_reader)?;
    header.seal_header_crc()?;
    write_all_at(&output, &header.encode()?, 0)?;
    output.sync_all()?;

    let loaded = Rabitq1Sidecar::load(&partial_path, base_path)?;
    let validation_count = rows.min(128);
    let validation_ids =
        deterministic_sample_ids(rows, validation_count, options.seed ^ 0xA5A5_A5A5)?
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>();
    let validation_query = gather_sample(base_path, dim, &[0])?;
    let mut prepared = loaded.prepare_query(&validation_query)?;
    let estimates =
        prepared.estimate_ids(loaded.center_ids(), loaded.bin_data(), &validation_ids)?;
    let first_id = validation_ids[0] as usize;
    let stride = loaded.header().bin_stride as usize;
    let reference = prepared.reference_estimate(
        loaded.center_ids()[first_id],
        &loaded.bin_data()[first_id * stride..(first_id + 1) * stride],
    )?;
    if estimates[0] != reference {
        return Err(builder_error(
            "batch and reference estimates differ during validation",
        ));
    }
    drop(prepared);
    drop(loaded);

    let mut output_reader = File::open(&partial_path)?;
    let output_crc64 = crc64_reader(&mut output_reader)?;
    let report = BuildRabitq1Report {
        schema: "graph_rabitq1_manifest_v1",
        rows: u64::from(rows_u32),
        dim: dim_u32,
        padded_dim: header.padded_dim,
        center_count: header.centroid_count,
        code_bytes: header.code_bytes,
        bin_stride: header.bin_stride,
        ex_data_bytes: 0,
        sidecar_bytes: header.file_bytes,
        base_full_crc64: header.base_full_crc64,
        base_window_fingerprint: header.base_window_fingerprint,
        output_crc64,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    fs::rename(&partial_path, output_path)?;
    fsync_parent(output_path)?;
    publish_manifest(&manifest_path, &report)?;
    Ok(report)
}
