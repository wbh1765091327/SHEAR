use std::mem::size_of;

use super::dco_rabitq::{
    RABITQ_DEFAULT_K_ALIGN, RABITQ_DEFAULT_TOTAL_BITS, RabitqMatrix, quantize_rabitq_matrix,
};
use super::params::{PiPNNGpuSubtreeDcoMode, PiPNNGpuSubtreeDcoRerankMode};
use super::rbc_partition::GpuSubtreeLeafDcoTelemetry;
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeafDcoShadowResult {
    pub(crate) topk_ids: Vec<Vec<u32>>,
    pub(crate) telemetry: GpuSubtreeLeafDcoTelemetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeafDcoShadowConfig {
    pub(crate) mode: PiPNNGpuSubtreeDcoMode,
    pub(crate) rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    pub(crate) proj_dim: usize,
    pub(crate) survivors: usize,
    pub(crate) audit: bool,
}

pub(crate) fn leaf_dco_shadow_l2(
    points_row_major: &[f32],
    point_ids: &[u32],
    dim: usize,
    requested_k: usize,
    cfg: LeafDcoShadowConfig,
) -> AnnResult<LeafDcoShadowResult> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "leaf DCO shadow requires dim > 0".to_string(),
        ));
    }
    if points_row_major.len() != point_ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "leaf DCO shadow point shape mismatch: rows={} dim={} values={}",
            point_ids.len(),
            dim,
            points_row_major.len()
        )));
    }
    if cfg.mode == PiPNNGpuSubtreeDcoMode::DdcRes {
        return Err(AnnError::log_index_error(
            "leaf DCO ddc-res shadow requires ResInfer PCA metadata; raw-space DDC_res shadow is disabled"
                .to_string(),
        ));
    }
    if cfg.mode == PiPNNGpuSubtreeDcoMode::SqInt8Filter {
        return Err(AnnError::log_index_error(
            "leaf DCO sq-int8-filter is D0-only; use wmma-filter for leaf SQ-int8 experiments"
                .to_string(),
        ));
    }

    let rows = point_ids.len();
    if rows == 0 {
        return Ok(LeafDcoShadowResult {
            topk_ids: Vec::new(),
            telemetry: GpuSubtreeLeafDcoTelemetry::default(),
        });
    }

    let k = requested_k.min(rows.saturating_sub(1));
    let candidate_count = rows.saturating_sub(1);
    let projection_dims = cfg.proj_dim.clamp(1, dim);
    let rabitq_shadow = if cfg.mode == PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter {
        let matrix = quantize_rabitq_matrix(
            points_row_major,
            rows,
            dim,
            RABITQ_DEFAULT_TOTAL_BITS,
            RABITQ_DEFAULT_K_ALIGN,
        )?;
        let norms = row_norm_sqr(points_row_major, rows, dim);
        Some((matrix, norms))
    } else {
        None
    };
    let fixed_survivor_mode = matches!(
        cfg.mode,
        PiPNNGpuSubtreeDcoMode::PrefixProjection
            | PiPNNGpuSubtreeDcoMode::Quantized
            | PiPNNGpuSubtreeDcoMode::WmmaFilter
            | PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter
            | PiPNNGpuSubtreeDcoMode::ShapeAware
    );
    let survivor_count = if fixed_survivor_mode {
        cfg.survivors.max(k).min(candidate_count)
    } else {
        candidate_count
    };
    let pair_count = rows.saturating_mul(candidate_count);
    let metadata_bytes = leaf_shadow_metadata_bytes(
        cfg.mode,
        rows,
        projection_dims,
        rabitq_shadow.as_ref().map(|(matrix, _)| matrix),
    );
    let mut telemetry = GpuSubtreeLeafDcoTelemetry {
        leaf_count: 1,
        rows,
        pair_count,
        prefilter_pairs: usize::from(cfg.mode.is_enabled()).saturating_mul(pair_count),
        pruned_pairs: if fixed_survivor_mode {
            rows.saturating_mul(candidate_count.saturating_sub(survivor_count))
        } else {
            0
        },
        survivor_pairs: if fixed_survivor_mode {
            rows.saturating_mul(survivor_count)
        } else if cfg.mode.is_enabled() {
            0
        } else {
            pair_count
        },
        metadata_bytes,
        ..GpuSubtreeLeafDcoTelemetry::default()
    };

    let mut topk_ids = Vec::with_capacity(rows);
    for row in 0..rows {
        let point = row_slice(points_row_major, dim, row);
        let mut approx_candidates = Vec::with_capacity(candidate_count);
        for col in 0..rows {
            if row == col {
                continue;
            }
            let other = row_slice(points_row_major, dim, col);
            let dist = match cfg.mode {
                PiPNNGpuSubtreeDcoMode::None => exact_l2(point, other),
                PiPNNGpuSubtreeDcoMode::PrefixProjection
                | PiPNNGpuSubtreeDcoMode::WmmaFilter
                | PiPNNGpuSubtreeDcoMode::ShapeAware => {
                    projected_l2(point, other, projection_dims, dim)
                }
                PiPNNGpuSubtreeDcoMode::DdcRes => unreachable!("ddc-res rejected above"),
                PiPNNGpuSubtreeDcoMode::RabitqSplitFilter => {
                    unreachable!(
                        "rabitq-split-filter shadow is implemented in the GPU persistent path"
                    )
                }
                PiPNNGpuSubtreeDcoMode::SqInt8Filter => {
                    unreachable!("sq-int8-filter is D0-only")
                }
                PiPNNGpuSubtreeDcoMode::Quantized => quantized_l2(point, other, projection_dims),
                PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter => {
                    let (matrix, norms) = rabitq_shadow
                        .as_ref()
                        .expect("RaBitQ shadow metadata should be initialized");
                    matrix.approx_l2(row, norms[row], col, norms[col])
                }
            };
            approx_candidates.push((col, dist));
        }
        sort_candidates(&mut approx_candidates, point_ids);
        let survivors = &approx_candidates[..survivor_count];

        let row_topk = match cfg.rerank_mode {
            PiPNNGpuSubtreeDcoRerankMode::Exact => {
                let mut exact_survivors = Vec::with_capacity(survivor_count);
                for &(col, _) in survivors {
                    exact_survivors
                        .push((col, exact_l2(point, row_slice(points_row_major, dim, col))));
                }
                telemetry.exact_pairs += survivor_count;
                sort_candidates(&mut exact_survivors, point_ids);
                exact_survivors
                    .iter()
                    .take(k)
                    .map(|&(col, _)| point_ids[col])
                    .collect::<Vec<_>>()
            }
            PiPNNGpuSubtreeDcoRerankMode::ExactStamp => {
                let stamp_count = k.min(survivors.len());
                telemetry.exact_stamp_pairs += stamp_count;
                approx_candidates
                    .iter()
                    .take(k)
                    .map(|&(col, _)| point_ids[col])
                    .collect::<Vec<_>>()
            }
            PiPNNGpuSubtreeDcoRerankMode::None => {
                telemetry.approx_output_pairs += k;
                approx_candidates
                    .iter()
                    .take(k)
                    .map(|&(col, _)| point_ids[col])
                    .collect::<Vec<_>>()
            }
        };

        if cfg.audit {
            telemetry.false_prune_audit_pairs += candidate_count;
            let exact_row_topk = exact_topk_for_row(points_row_major, point_ids, dim, row, k);
            telemetry.false_prune_audit_misses += exact_row_topk
                .iter()
                .filter(|target| !row_topk.contains(target))
                .count();
        }
        topk_ids.push(row_topk);
    }

    Ok(LeafDcoShadowResult {
        topk_ids,
        telemetry,
    })
}

fn leaf_shadow_metadata_bytes(
    mode: PiPNNGpuSubtreeDcoMode,
    rows: usize,
    projection_dims: usize,
    rabitq_matrix: Option<&RabitqMatrix>,
) -> usize {
    match mode {
        PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter => rabitq_matrix
            .map(|matrix| {
                matrix
                    .codes
                    .len()
                    .saturating_mul(size_of::<i8>())
                    .saturating_add(matrix.deltas.len().saturating_mul(size_of::<f32>()))
                    .saturating_add(matrix.code_sums.len().saturating_mul(size_of::<f32>()))
            })
            .unwrap_or(0),
        PiPNNGpuSubtreeDcoMode::PrefixProjection
        | PiPNNGpuSubtreeDcoMode::Quantized
        | PiPNNGpuSubtreeDcoMode::WmmaFilter
        | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        | PiPNNGpuSubtreeDcoMode::SqInt8Filter
        | PiPNNGpuSubtreeDcoMode::ShapeAware => rows
            .saturating_mul(projection_dims)
            .saturating_mul(size_of::<f32>()),
        PiPNNGpuSubtreeDcoMode::None | PiPNNGpuSubtreeDcoMode::DdcRes => 0,
    }
}

fn exact_topk_for_row(
    points_row_major: &[f32],
    point_ids: &[u32],
    dim: usize,
    row: usize,
    k: usize,
) -> Vec<u32> {
    let point = row_slice(points_row_major, dim, row);
    let mut candidates = Vec::with_capacity(point_ids.len().saturating_sub(1));
    for col in 0..point_ids.len() {
        if row == col {
            continue;
        }
        candidates.push((col, exact_l2(point, row_slice(points_row_major, dim, col))));
    }
    sort_candidates(&mut candidates, point_ids);
    candidates
        .iter()
        .take(k)
        .map(|&(col, _)| point_ids[col])
        .collect()
}

fn row_slice(points_row_major: &[f32], dim: usize, row: usize) -> &[f32] {
    &points_row_major[row * dim..(row + 1) * dim]
}

fn sort_candidates(candidates: &mut [(usize, f32)], point_ids: &[u32]) {
    candidates.sort_unstable_by(|left, right| {
        let left_id = point_ids[left.0];
        let right_id = point_ids[right.0];
        super::stable_distance_order(left.1, left_id, right.1, right_id)
    });
}

fn exact_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&lhs, &rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum()
}

fn projected_l2(left: &[f32], right: &[f32], projection_dims: usize, full_dim: usize) -> f32 {
    let partial: f32 = left
        .iter()
        .zip(right)
        .take(projection_dims)
        .map(|(&lhs, &rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum();
    partial * full_dim as f32 / projection_dims.max(1) as f32
}

fn quantized_l2(left: &[f32], right: &[f32], projection_dims: usize) -> f32 {
    left.iter()
        .zip(right)
        .take(projection_dims)
        .map(|(&lhs, &rhs)| {
            let delta = quantize_4bit_step(lhs) - quantize_4bit_step(rhs);
            delta * delta
        })
        .sum()
}

fn quantize_4bit_step(value: f32) -> f32 {
    (value * 16.0).round() / 16.0
}

fn row_norm_sqr(points_row_major: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut norms = Vec::with_capacity(rows);
    for row in 0..rows {
        norms.push(
            row_slice(points_row_major, dim, row)
                .iter()
                .map(|value| value * value)
                .sum(),
        );
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_dco_exact_rerank_recovers_exact_topk_when_survivor_contains_winner() {
        let point_ids = [10_u32, 11_u32, 12_u32];
        let points = [
            0.0_f32, 100.0, // row 0: projection favors row 1, exact favors row 2
            0.0_f32, 0.0, 1.0_f32, 100.0,
        ];

        let result = leaf_dco_shadow_l2(
            &points,
            &point_ids,
            2,
            1,
            LeafDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::PrefixProjection,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
                proj_dim: 1,
                survivors: 2,
                audit: true,
            },
        )
        .expect("leaf DCO shadow");

        assert_eq!(result.topk_ids[0], vec![12]);
        assert_eq!(result.telemetry.leaf_count, 1);
        assert_eq!(result.telemetry.rows, 3);
        assert_eq!(result.telemetry.pair_count, 6);
        assert_eq!(result.telemetry.prefilter_pairs, 6);
        assert_eq!(result.telemetry.pruned_pairs, 0);
        assert_eq!(result.telemetry.survivor_pairs, 6);
        assert_eq!(result.telemetry.exact_pairs, 6);
        assert_eq!(result.telemetry.approx_output_pairs, 0);
        assert_eq!(result.telemetry.false_prune_audit_pairs, 6);
        assert_eq!(result.telemetry.false_prune_audit_misses, 0);
        assert_eq!(result.telemetry.metadata_bytes, 12);
    }

    #[test]
    fn leaf_dco_no_rerank_exposes_approx_topk_miss() {
        let point_ids = [10_u32, 11_u32, 12_u32];
        let points = [
            0.0_f32, 100.0, // row 0: projection favors row 1, exact favors row 2
            0.0_f32, 0.0, 1.0_f32, 100.0,
        ];

        let result = leaf_dco_shadow_l2(
            &points,
            &point_ids,
            2,
            1,
            LeafDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::PrefixProjection,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::None,
                proj_dim: 1,
                survivors: 1,
                audit: true,
            },
        )
        .expect("leaf DCO shadow");

        assert_eq!(result.topk_ids[0], vec![11]);
        assert_eq!(result.telemetry.prefilter_pairs, 6);
        assert_eq!(result.telemetry.pruned_pairs, 3);
        assert_eq!(result.telemetry.survivor_pairs, 3);
        assert_eq!(result.telemetry.exact_pairs, 0);
        assert_eq!(result.telemetry.approx_output_pairs, 3);
        assert_eq!(result.telemetry.false_prune_audit_pairs, 6);
        assert_eq!(result.telemetry.false_prune_audit_misses, 1);
    }

    #[test]
    fn leaf_dco_rabitq_shadow_uses_quantized_metadata() {
        let point_ids = [10_u32, 11_u32, 12_u32];
        let points = [
            0.2_f32, -1.0, 0.7, //
            -0.9_f32, 0.5, 1.4, //
            1.1_f32, 0.2, -0.3,
        ];

        let result = leaf_dco_shadow_l2(
            &points,
            &point_ids,
            3,
            1,
            LeafDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
                proj_dim: 0,
                survivors: 2,
                audit: true,
            },
        )
        .expect("dense-i8 RabitQ+ leaf DCO shadow");

        assert_eq!(result.topk_ids.len(), 3);
        assert!(result.topk_ids.iter().all(|row| row.len() == 1));
        assert_eq!(result.telemetry.prefilter_pairs, 6);
        assert_eq!(result.telemetry.survivor_pairs, 6);
        assert_eq!(result.telemetry.exact_pairs, 6);
        assert_eq!(
            result.telemetry.metadata_bytes,
            3 * RABITQ_DEFAULT_K_ALIGN * size_of::<i8>() + 3 * 2 * size_of::<f32>()
        );
    }

    #[test]
    fn leaf_dco_shadow_rejects_ddc_res_without_pca_metadata() {
        let point_ids = [10_u32, 11_u32, 12_u32, 13_u32];
        let points = [
            0.0_f32, 0.0, 0.0, 0.0, // row 0
            0.0_f32, 0.0, 0.0, 0.1, // close to row 0
            100.0_f32, 0.0, 0.0, 0.0, // far enough to prune once threshold is set
            200.0_f32, 0.0, 0.0, 0.0,
        ];

        let err = leaf_dco_shadow_l2(
            &points,
            &point_ids,
            4,
            1,
            LeafDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::DdcRes,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
                proj_dim: 1,
                survivors: 0,
                audit: true,
            },
        )
        .expect_err("raw-space DDC_res shadow should be rejected");

        assert!(
            err.to_string().contains("PCA metadata"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn leaf_dco_shadow_rejects_mismatched_shapes() {
        let err = leaf_dco_shadow_l2(
            &[0.0, 1.0, 2.0],
            &[0],
            2,
            1,
            LeafDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::PrefixProjection,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
                proj_dim: 1,
                survivors: 1,
                audit: false,
            },
        )
        .expect_err("shape mismatch should fail");

        assert!(
            err.to_string().contains("point shape mismatch"),
            "unexpected error: {err:?}"
        );
    }
}
