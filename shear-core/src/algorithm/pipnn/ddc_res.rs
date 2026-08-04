use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug)]
pub(crate) struct ResInferDdcResPcaArtifact {
    pub(crate) origin_dim: usize,
    pub(crate) project_dim: usize,
    pub(crate) mean: Vec<f32>,
    pub(crate) extra_mean: Vec<f32>,
    pub(crate) variance: Vec<f32>,
    pub(crate) projection_matrix: Vec<f32>,
}

impl ResInferDdcResPcaArtifact {
    pub(crate) fn load_fvecs(path: &Path) -> AnnResult<Self> {
        let mut file = File::open(path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open ResInfer DDC_res PCA matrix '{}': {err}",
                path.display()
            ))
        })?;
        let mut dim_bytes = [0_u8; 4];
        file.read_exact(&mut dim_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to read ResInfer DDC_res PCA matrix header '{}': {err}",
                path.display()
            ))
        })?;
        let project_dim = u32::from_le_bytes(dim_bytes) as usize;
        if project_dim == 0 {
            return Err(AnnError::log_index_error(format!(
                "Invalid ResInfer DDC_res PCA matrix '{}': project_dim is zero",
                path.display()
            )));
        }
        let file_size = file.metadata()?.len() as usize;
        let row_bytes = 4usize
            .checked_add(
                project_dim
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        AnnError::log_index_error(format!(
                            "Invalid ResInfer DDC_res PCA matrix '{}': row byte size overflows",
                            path.display()
                        ))
                    })?,
            )
            .ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Invalid ResInfer DDC_res PCA matrix '{}': row byte size overflows",
                    path.display()
                ))
            })?;
        if file_size == 0 || file_size % row_bytes != 0 {
            return Err(AnnError::log_index_error(format!(
                "Invalid ResInfer DDC_res PCA matrix '{}': size {} is not a multiple of fvecs row bytes {}",
                path.display(),
                file_size,
                row_bytes
            )));
        }
        let rows = file_size / row_bytes;
        if rows < 4 {
            return Err(AnnError::log_index_error(format!(
                "Invalid ResInfer DDC_res PCA matrix '{}': expected at least 4 fvecs rows, got {rows}",
                path.display()
            )));
        }
        let origin_dim = rows - 3;
        if origin_dim != project_dim {
            return Err(AnnError::log_index_error(format!(
                "ResInfer DDC_res PCA matrix '{}' has origin_dim={} project_dim={}; SHEAR DDC_res currently supports only full-dimensional PCA artifacts",
                path.display(),
                origin_dim,
                project_dim
            )));
        }

        file.seek(SeekFrom::Start(0))?;
        let mut values = vec![0.0_f32; rows.saturating_mul(project_dim)];
        for row in 0..rows {
            file.read_exact(&mut dim_bytes)?;
            let row_dim = u32::from_le_bytes(dim_bytes) as usize;
            if row_dim != project_dim {
                return Err(AnnError::log_index_error(format!(
                    "Invalid ResInfer DDC_res PCA matrix '{}': row {row} has dim {row_dim}, expected {project_dim}",
                    path.display()
                )));
            }
            let row_values =
                &mut values[row.saturating_mul(project_dim)..(row + 1).saturating_mul(project_dim)];
            let row_bytes = unsafe {
                std::slice::from_raw_parts_mut(row_values.as_mut_ptr() as *mut u8, project_dim * 4)
            };
            file.read_exact(row_bytes)?;
        }

        let mean = values[0..project_dim].to_vec();
        let extra_mean = values[project_dim..2 * project_dim].to_vec();
        let variance = values[2 * project_dim..3 * project_dim].to_vec();
        let projection_matrix = values[3 * project_dim..].to_vec();
        if projection_matrix.len() != origin_dim.saturating_mul(project_dim) {
            return Err(AnnError::log_index_error(format!(
                "Invalid ResInfer DDC_res PCA matrix '{}': projection matrix has {} values, expected {}",
                path.display(),
                projection_matrix.len(),
                origin_dim.saturating_mul(project_dim)
            )));
        }

        Ok(Self {
            origin_dim,
            project_dim,
            mean,
            extra_mean,
            variance,
            projection_matrix,
        })
    }

    pub(crate) fn project_raw_vector_minus_extra_mean(&self, raw: &[f32]) -> AnnResult<Vec<f32>> {
        if raw.len() != self.origin_dim {
            return Err(AnnError::log_index_error(format!(
                "DDC_res PCA projection shape mismatch: raw dim {} expected {}",
                raw.len(),
                self.origin_dim
            )));
        }
        let mut out = vec![0.0_f32; self.project_dim];
        for (dst_axis, out_value) in out.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (src_axis, &raw_value) in raw.iter().enumerate() {
                let centered = raw_value - self.mean[src_axis];
                sum += centered * self.projection_matrix[src_axis * self.project_dim + dst_axis];
            }
            *out_value = sum - self.extra_mean[dst_axis];
        }
        Ok(out)
    }

    pub(crate) fn subtract_extra_mean_from_projected(
        &self,
        projected: &[f32],
        out: &mut [f32],
    ) -> AnnResult<()> {
        if projected.len() != self.project_dim || out.len() != self.project_dim {
            return Err(AnnError::log_index_error(format!(
                "DDC_res projected vector shape mismatch: input dim {} output dim {} expected {}",
                projected.len(),
                out.len(),
                self.project_dim
            )));
        }
        for axis in 0..self.project_dim {
            out[axis] = projected[axis] - self.extra_mean[axis];
        }
        Ok(())
    }

    pub(crate) fn weighted_query_square(
        &self,
        projected_minus_extra_mean: &[f32],
    ) -> AnnResult<f32> {
        if projected_minus_extra_mean.len() != self.project_dim {
            return Err(AnnError::log_index_error(format!(
                "DDC_res weighted query square shape mismatch: dim {} expected {}",
                projected_minus_extra_mean.len(),
                self.project_dim
            )));
        }
        Ok(projected_minus_extra_mean
            .iter()
            .zip(&self.variance)
            .map(|(&value, &var)| value * value * var)
            .sum())
    }

    pub(crate) fn norm_square(&self, projected_minus_extra_mean: &[f32]) -> AnnResult<f32> {
        if projected_minus_extra_mean.len() != self.project_dim {
            return Err(AnnError::log_index_error(format!(
                "DDC_res norm shape mismatch: dim {} expected {}",
                projected_minus_extra_mean.len(),
                self.project_dim
            )));
        }
        Ok(projected_minus_extra_mean
            .iter()
            .map(|&value| value * value)
            .sum())
    }
}

pub(crate) fn resinfer_uniform_fast_inference(
    query: &[f32],
    base: &[f32],
    variance: &[f32],
    base_square: f32,
    threshold: f32,
    base_dim: usize,
    sigma_count: f32,
) -> AnnResult<(f32, bool)> {
    if query.len() != base.len() || query.len() != variance.len() {
        return Err(AnnError::log_index_error(format!(
            "DDC_res inference shape mismatch: query={} base={} variance={}",
            query.len(),
            base.len(),
            variance.len()
        )));
    }
    if query.is_empty() {
        return Ok((0.0, false));
    }
    if base_dim == 0 || !sigma_count.is_finite() || sigma_count <= 0.0 {
        return Err(AnnError::log_index_error(format!(
            "Invalid DDC_res inference parameters: base_dim={base_dim} sigma_count={sigma_count}"
        )));
    }

    let query_square: f32 = query.iter().map(|&value| value * value).sum();
    let mut weighted_tail: f32 = query
        .iter()
        .zip(variance)
        .map(|(&value, &var)| value * value * var)
        .sum();
    let mut res = base_square + query_square + 1.0e-5;
    let step = base_dim.min(query.len()).max(1);
    for axis_begin in (0..query.len()).step_by(step) {
        let axis_end = (axis_begin + step).min(query.len());
        let mut dot = 0.0_f32;
        let mut weighted_block = 0.0_f32;
        for axis in axis_begin..axis_end {
            dot += query[axis] * base[axis];
            weighted_block += query[axis] * query[axis] * variance[axis];
        }
        res -= 2.0 * dot;
        weighted_tail = (weighted_tail - weighted_block).max(0.0);
        let pre_query = 2.0 * sigma_count * weighted_tail.sqrt();
        if res - pre_query > threshold {
            return Ok((res.max(0.0), true));
        }
    }
    Ok((res.max(0.0), res > threshold))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn write_fvecs(path: &Path, rows: &[Vec<f32>]) {
        let mut file = File::create(path).unwrap();
        for row in rows {
            file.write_all(&(row.len() as u32).to_le_bytes()).unwrap();
            for &value in row {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn resinfer_ddc_res_pca_artifact_loads_fvecs_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pca.fvecs");
        write_fvecs(
            &path,
            &[
                vec![1.0, 2.0],
                vec![0.5, -0.5],
                vec![4.0, 9.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
        );

        let artifact = ResInferDdcResPcaArtifact::load_fvecs(&path).unwrap();

        assert_eq!(artifact.origin_dim, 2);
        assert_eq!(artifact.project_dim, 2);
        assert_eq!(artifact.mean, vec![1.0, 2.0]);
        assert_eq!(artifact.extra_mean, vec![0.5, -0.5]);
        assert_eq!(artifact.variance, vec![4.0, 9.0]);
        assert_eq!(artifact.projection_matrix, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn resinfer_ddc_res_projection_and_uniform_inference_match_formula() {
        let artifact = ResInferDdcResPcaArtifact {
            origin_dim: 2,
            project_dim: 2,
            mean: vec![1.0, 2.0],
            extra_mean: vec![0.5, -0.5],
            variance: vec![4.0, 9.0],
            projection_matrix: vec![1.0, 0.0, 0.0, 1.0],
        };
        let query = artifact
            .project_raw_vector_minus_extra_mean(&[3.0, 5.0])
            .unwrap();
        let base = artifact
            .project_raw_vector_minus_extra_mean(&[2.0, 4.0])
            .unwrap();
        assert_eq!(query, vec![1.5, 3.5]);
        assert_eq!(base, vec![0.5, 2.5]);

        let base_square = artifact.norm_square(&base).unwrap();
        let (dist, pruned) = resinfer_uniform_fast_inference(
            &query,
            &base,
            &artifact.variance,
            base_square,
            f32::INFINITY,
            1,
            3.0,
        )
        .unwrap();

        assert!(!pruned);
        assert!((dist - 2.00001).abs() <= 1.0e-5, "dist={dist}");
    }
}
