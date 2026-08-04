use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::common::{AnnError, AnnResult};

const REPLAY_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaCompactReplayBatchFiles {
    pub x_row_major: String,
    pub point_ids: String,
    pub norms: String,
    pub leaf_starts: String,
    pub leaf_sizes: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaCompactReplayBatchMetadata {
    pub version: u32,
    pub batch_index: usize,
    pub rows: usize,
    pub dim: usize,
    pub k: usize,
    pub leaves: usize,
    pub files: CudaCompactReplayBatchFiles,
}

pub struct CudaCompactReplayBatchView<'a> {
    pub x_row_major: &'a [f32],
    pub point_ids: &'a [u32],
    pub norms: &'a [f32],
    pub leaf_starts: &'a [u64],
    pub leaf_sizes: &'a [u64],
    pub rows: usize,
    pub dim: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub struct CudaCompactReplayBatch {
    pub metadata: CudaCompactReplayBatchMetadata,
    pub x_row_major: Vec<f32>,
    pub point_ids: Vec<u32>,
    pub norms: Vec<f32>,
    pub leaf_starts: Vec<u64>,
    pub leaf_sizes: Vec<u64>,
}

impl CudaCompactReplayBatch {
    pub fn validate(&self) -> AnnResult<()> {
        validate_batch_shape(
            self.metadata.rows,
            self.metadata.dim,
            self.metadata.k,
            self.metadata.leaves,
            &self.x_row_major,
            &self.point_ids,
            &self.norms,
            &self.leaf_starts,
            &self.leaf_sizes,
        )
    }
}

pub fn write_cuda_compact_replay_batch(
    dir: &Path,
    batch_index: usize,
    batch: &CudaCompactReplayBatchView<'_>,
) -> AnnResult<PathBuf> {
    validate_batch_shape(
        batch.rows,
        batch.dim,
        batch.k,
        batch.leaf_starts.len(),
        batch.x_row_major,
        batch.point_ids,
        batch.norms,
        batch.leaf_starts,
        batch.leaf_sizes,
    )?;

    fs::create_dir_all(dir).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create CUDA compact replay dump dir {}: {err}",
            dir.display()
        ))
    })?;

    let prefix = format!("batch_{batch_index:06}");
    let files = CudaCompactReplayBatchFiles {
        x_row_major: format!("{prefix}.x.f32.bin"),
        point_ids: format!("{prefix}.point_ids.u32.bin"),
        norms: format!("{prefix}.norms.f32.bin"),
        leaf_starts: format!("{prefix}.leaf_starts.u64.bin"),
        leaf_sizes: format!("{prefix}.leaf_sizes.u64.bin"),
    };

    write_f32_file(&dir.join(&files.x_row_major), batch.x_row_major)?;
    write_u32_file(&dir.join(&files.point_ids), batch.point_ids)?;
    write_f32_file(&dir.join(&files.norms), batch.norms)?;
    write_u64_file(&dir.join(&files.leaf_starts), batch.leaf_starts)?;
    write_u64_file(&dir.join(&files.leaf_sizes), batch.leaf_sizes)?;

    let metadata = CudaCompactReplayBatchMetadata {
        version: REPLAY_VERSION,
        batch_index,
        rows: batch.rows,
        dim: batch.dim,
        k: batch.k,
        leaves: batch.leaf_starts.len(),
        files,
    };
    let metadata_path = dir.join(format!("{prefix}.json"));
    let metadata_json = serde_json::to_vec_pretty(&metadata).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to serialize CUDA compact replay metadata: {err}"
        ))
    })?;
    fs::write(&metadata_path, [metadata_json, b"\n".to_vec()].concat()).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to write CUDA compact replay metadata {}: {err}",
            metadata_path.display()
        ))
    })?;
    Ok(metadata_path)
}

pub fn load_cuda_compact_replay_batches(dir: &Path) -> AnnResult<Vec<CudaCompactReplayBatch>> {
    let mut metadata_paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to read CUDA compact replay dir {}: {err}",
            dir.display()
        ))
    })? {
        let entry = entry.map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to read CUDA compact replay dir entry {}: {err}",
                dir.display()
            ))
        })?;
        let path = entry.path();
        if is_cuda_compact_replay_batch_metadata_path(&path) {
            metadata_paths.push(path);
        }
    }
    metadata_paths.sort();

    metadata_paths
        .iter()
        .map(|path| load_cuda_compact_replay_batch(path))
        .collect()
}

fn is_cuda_compact_replay_batch_metadata_path(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(batch_index) = file_name
        .strip_prefix("batch_")
        .and_then(|name| name.strip_suffix(".json"))
    else {
        return false;
    };
    !batch_index.is_empty() && batch_index.bytes().all(|byte| byte.is_ascii_digit())
}

pub fn load_cuda_compact_replay_batch(metadata_path: &Path) -> AnnResult<CudaCompactReplayBatch> {
    let metadata_bytes = fs::read(metadata_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to read CUDA compact replay metadata {}: {err}",
            metadata_path.display()
        ))
    })?;
    let metadata: CudaCompactReplayBatchMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to parse CUDA compact replay metadata {}: {err}",
                metadata_path.display()
            ))
        })?;
    if metadata.version != REPLAY_VERSION {
        return Err(AnnError::log_index_error(format!(
            "Unsupported CUDA compact replay version {} in {}",
            metadata.version,
            metadata_path.display()
        )));
    }

    let dir = metadata_path.parent().ok_or_else(|| {
        AnnError::log_index_error(format!(
            "CUDA compact replay metadata has no parent dir: {}",
            metadata_path.display()
        ))
    })?;
    let batch = CudaCompactReplayBatch {
        x_row_major: read_f32_file(&dir.join(&metadata.files.x_row_major))?,
        point_ids: read_u32_file(&dir.join(&metadata.files.point_ids))?,
        norms: read_f32_file(&dir.join(&metadata.files.norms))?,
        leaf_starts: read_u64_file(&dir.join(&metadata.files.leaf_starts))?,
        leaf_sizes: read_u64_file(&dir.join(&metadata.files.leaf_sizes))?,
        metadata,
    };
    batch.validate()?;
    Ok(batch)
}

fn validate_batch_shape(
    rows: usize,
    dim: usize,
    k: usize,
    leaves: usize,
    x_row_major: &[f32],
    point_ids: &[u32],
    norms: &[f32],
    leaf_starts: &[u64],
    leaf_sizes: &[u64],
) -> AnnResult<()> {
    if rows == 0 || dim == 0 || k == 0 {
        return Err(AnnError::log_index_error(format!(
            "Invalid CUDA compact replay shape: rows={rows} dim={dim} k={k}"
        )));
    }
    if x_row_major.len() != rows.saturating_mul(dim)
        || point_ids.len() != rows
        || norms.len() != rows
        || leaf_starts.len() != leaves
        || leaf_sizes.len() != leaves
        || leaf_starts.len() != leaf_sizes.len()
    {
        return Err(AnnError::log_index_error(format!(
            "Invalid CUDA compact replay array lengths: rows={rows} dim={dim} leaves={leaves} x={} point_ids={} norms={} leaf_starts={} leaf_sizes={}",
            x_row_major.len(),
            point_ids.len(),
            norms.len(),
            leaf_starts.len(),
            leaf_sizes.len()
        )));
    }
    let total_leaf_rows: u64 = leaf_sizes.iter().sum();
    if total_leaf_rows != rows as u64 {
        return Err(AnnError::log_index_error(format!(
            "Invalid CUDA compact replay leaf sizes: sum={total_leaf_rows} rows={rows}"
        )));
    }
    Ok(())
}

fn write_f32_file(path: &Path, values: &[f32]) -> AnnResult<()> {
    write_binary_file(path, values, |value, out| {
        out.extend_from_slice(&value.to_le_bytes())
    })
}

fn write_u32_file(path: &Path, values: &[u32]) -> AnnResult<()> {
    write_binary_file(path, values, |value, out| {
        out.extend_from_slice(&value.to_le_bytes())
    })
}

fn write_u64_file(path: &Path, values: &[u64]) -> AnnResult<()> {
    write_binary_file(path, values, |value, out| {
        out.extend_from_slice(&value.to_le_bytes())
    })
}

fn write_binary_file<T>(
    path: &Path,
    values: &[T],
    mut append: impl FnMut(&T, &mut Vec<u8>),
) -> AnnResult<()> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        append(value, &mut bytes);
    }
    let mut file = File::create(path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create CUDA compact replay file {}: {err}",
            path.display()
        ))
    })?;
    file.write_all(&bytes).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to write CUDA compact replay file {}: {err}",
            path.display()
        ))
    })
}

fn read_f32_file(path: &Path) -> AnnResult<Vec<f32>> {
    read_binary_file(path, 4, |chunk| {
        f32::from_le_bytes(chunk.try_into().expect("chunk length checked"))
    })
}

fn read_u32_file(path: &Path) -> AnnResult<Vec<u32>> {
    read_binary_file(path, 4, |chunk| {
        u32::from_le_bytes(chunk.try_into().expect("chunk length checked"))
    })
}

fn read_u64_file(path: &Path) -> AnnResult<Vec<u64>> {
    read_binary_file(path, 8, |chunk| {
        u64::from_le_bytes(chunk.try_into().expect("chunk length checked"))
    })
}

fn read_binary_file<T>(
    path: &Path,
    width: usize,
    convert: impl Fn(&[u8]) -> T,
) -> AnnResult<Vec<T>> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to read CUDA compact replay file {}: {err}",
                path.display()
            ))
        })?;
    if bytes.len() % width != 0 {
        return Err(AnnError::log_index_error(format!(
            "CUDA compact replay file {} has {} bytes, not divisible by element width {}",
            path.display(),
            bytes.len(),
            width
        )));
    }
    Ok(bytes.chunks_exact(width).map(convert).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_batch_roundtrip_preserves_arrays() {
        let dir = std::env::temp_dir().join(format!(
            "pipnn_compact_replay_roundtrip_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let x_row_major = vec![0.0, 1.0, 2.0, 3.0];
        let point_ids = vec![7, 9];
        let norms = vec![1.0, 13.0];
        let leaf_starts = vec![0];
        let leaf_sizes = vec![2];

        let metadata_path = write_cuda_compact_replay_batch(
            &dir,
            3,
            &CudaCompactReplayBatchView {
                x_row_major: &x_row_major,
                point_ids: &point_ids,
                norms: &norms,
                leaf_starts: &leaf_starts,
                leaf_sizes: &leaf_sizes,
                rows: 2,
                dim: 2,
                k: 1,
            },
        )
        .unwrap();

        let loaded = load_cuda_compact_replay_batch(&metadata_path).unwrap();
        assert_eq!(loaded.metadata.batch_index, 3);
        assert_eq!(loaded.metadata.rows, 2);
        assert_eq!(loaded.x_row_major, x_row_major);
        assert_eq!(loaded.point_ids, point_ids);
        assert_eq!(loaded.norms, norms);
        assert_eq!(loaded.leaf_starts, leaf_starts);
        assert_eq!(loaded.leaf_sizes, leaf_sizes);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn replay_batches_ignore_non_batch_json_outputs() {
        let dir = std::env::temp_dir().join(format!(
            "pipnn_compact_replay_extra_json_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let x_row_major = vec![0.0, 1.0, 2.0, 3.0];
        let point_ids = vec![7, 9];
        let norms = vec![1.0, 13.0];
        let leaf_starts = vec![0];
        let leaf_sizes = vec![2];
        write_cuda_compact_replay_batch(
            &dir,
            3,
            &CudaCompactReplayBatchView {
                x_row_major: &x_row_major,
                point_ids: &point_ids,
                norms: &norms,
                leaf_starts: &leaf_starts,
                leaf_sizes: &leaf_sizes,
                rows: 2,
                dim: 2,
                k: 1,
            },
        )
        .unwrap();
        fs::write(
            dir.join("replay_real_vectors.json"),
            b"{\"batch_count\":1}\n",
        )
        .unwrap();

        let loaded = load_cuda_compact_replay_batches(&dir).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].metadata.batch_index, 3);

        fs::remove_dir_all(&dir).unwrap();
    }
}
