use std::mem::size_of;

use super::params::{PiPNNGpuSubtreeDcoMode, PiPNNGpuSubtreeDcoRerankMode};
use super::rbc_partition::GpuSubtreeAssignmentDcoTelemetry;
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AssignmentDcoShadowResult {
    pub(crate) assignments: Vec<Vec<usize>>,
    pub(crate) telemetry: GpuSubtreeAssignmentDcoTelemetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AssignmentDcoShadowConfig {
    pub(crate) mode: PiPNNGpuSubtreeDcoMode,
    pub(crate) rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    pub(crate) proj_dim: usize,
    pub(crate) survivors: usize,
    pub(crate) audit: bool,
}

pub(crate) fn assignment_dco_shadow_l2(
    points_row_major: &[f32],
    point_ids: &[u32],
    leaders_row_major: &[f32],
    leader_ids: &[u32],
    dim: usize,
    fanout: usize,
    cfg: AssignmentDcoShadowConfig,
) -> AnnResult<AssignmentDcoShadowResult> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "assignment DCO shadow requires dim > 0".to_string(),
        ));
    }
    if point_ids.is_empty() {
        return Ok(AssignmentDcoShadowResult {
            assignments: Vec::new(),
            telemetry: GpuSubtreeAssignmentDcoTelemetry::default(),
        });
    }
    if leaders_row_major.is_empty() || leader_ids.is_empty() || fanout == 0 {
        return Ok(AssignmentDcoShadowResult {
            assignments: vec![Vec::new(); point_ids.len()],
            telemetry: GpuSubtreeAssignmentDcoTelemetry {
                rows: point_ids.len(),
                nodes: 1,
                ..GpuSubtreeAssignmentDcoTelemetry::default()
            },
        });
    }
    if points_row_major.len() != point_ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "assignment DCO shadow point shape mismatch: rows={} dim={} values={}",
            point_ids.len(),
            dim,
            points_row_major.len()
        )));
    }
    if leaders_row_major.len() != leader_ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "assignment DCO shadow leader shape mismatch: rows={} dim={} values={}",
            leader_ids.len(),
            dim,
            leaders_row_major.len()
        )));
    }
    if cfg.mode == PiPNNGpuSubtreeDcoMode::DdcRes {
        return Err(AnnError::log_index_error(
            "assignment DCO ddc-res is not wired; ResInfer DDC_res is leaf top-k only".to_string(),
        ));
    }
    if cfg.mode == PiPNNGpuSubtreeDcoMode::SqInt8Filter {
        return Err(AnnError::log_index_error(
            "assignment DCO sq-int8-filter is D0-only; use wmma-filter for leaf/assignment SQ-int8 experiments"
                .to_string(),
        ));
    }

    let rows = point_ids.len();
    let leader_count = leader_ids.len();
    let fanout = fanout.min(leader_count);
    let projection_dims = cfg.proj_dim.clamp(1, dim);
    let survivor_count = if cfg.mode.is_enabled() {
        cfg.survivors.max(fanout).min(leader_count)
    } else {
        leader_count
    };
    let mut telemetry = GpuSubtreeAssignmentDcoTelemetry {
        nodes: 1,
        rows,
        leader_pairs: rows.saturating_mul(leader_count),
        prefilter_pairs: usize::from(cfg.mode.is_enabled())
            .saturating_mul(rows.saturating_mul(leader_count)),
        pruned_pairs: if cfg.mode.is_enabled() {
            rows.saturating_mul(leader_count.saturating_sub(survivor_count))
        } else {
            0
        },
        survivor_pairs: if cfg.mode.is_enabled() {
            rows.saturating_mul(survivor_count)
        } else {
            rows.saturating_mul(leader_count)
        },
        metadata_bytes: if cfg.mode.is_enabled() {
            rows.saturating_add(leader_count)
                .saturating_mul(projection_dims)
                .saturating_mul(size_of::<f32>())
        } else {
            0
        },
        ..GpuSubtreeAssignmentDcoTelemetry::default()
    };
    let mut assignments = Vec::with_capacity(rows);

    for row in 0..rows {
        let point = &points_row_major[row * dim..(row + 1) * dim];
        let mut approx_candidates = Vec::with_capacity(leader_count);
        for leader_idx in 0..leader_count {
            let leader = &leaders_row_major[leader_idx * dim..(leader_idx + 1) * dim];
            let dist = match cfg.mode {
                PiPNNGpuSubtreeDcoMode::None => exact_l2(point, leader),
                PiPNNGpuSubtreeDcoMode::PrefixProjection
                | PiPNNGpuSubtreeDcoMode::WmmaFilter
                | PiPNNGpuSubtreeDcoMode::ShapeAware => {
                    projected_l2(point, leader, projection_dims, dim)
                }
                PiPNNGpuSubtreeDcoMode::DdcRes => unreachable!("ddc-res rejected above"),
                PiPNNGpuSubtreeDcoMode::RabitqSplitFilter => {
                    unreachable!("rabitq-split-filter is D0-only")
                }
                PiPNNGpuSubtreeDcoMode::SqInt8Filter => {
                    unreachable!("sq-int8-filter is D0-only")
                }
                PiPNNGpuSubtreeDcoMode::Quantized | PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter => {
                    quantized_l2(point, leader, projection_dims)
                }
            };
            approx_candidates.push((leader_idx, dist));
        }
        sort_candidates(&mut approx_candidates, leader_ids);
        let survivors = &approx_candidates[..survivor_count];

        let row_assignment = match cfg.rerank_mode {
            PiPNNGpuSubtreeDcoRerankMode::Exact | PiPNNGpuSubtreeDcoRerankMode::ExactStamp => {
                let mut exact_survivors = Vec::with_capacity(survivor_count);
                for &(leader_idx, _) in survivors {
                    let leader = &leaders_row_major[leader_idx * dim..(leader_idx + 1) * dim];
                    exact_survivors.push((leader_idx, exact_l2(point, leader)));
                }
                telemetry.exact_pairs += survivor_count;
                sort_candidates(&mut exact_survivors, leader_ids);
                exact_survivors
                    .iter()
                    .take(fanout)
                    .map(|&(leader_idx, _)| leader_idx)
                    .collect::<Vec<_>>()
            }
            PiPNNGpuSubtreeDcoRerankMode::None => {
                telemetry.approx_output_pairs += fanout;
                approx_candidates
                    .iter()
                    .take(fanout)
                    .map(|&(leader_idx, _)| leader_idx)
                    .collect::<Vec<_>>()
            }
        };

        if cfg.audit {
            telemetry.false_prune_audit_pairs += leader_count;
            let exact_assignment =
                exact_assignment_for_row(point, leaders_row_major, leader_ids, dim, fanout);
            telemetry.false_prune_audit_misses += exact_assignment
                .iter()
                .filter(|leader_idx| !row_assignment.contains(leader_idx))
                .count();
        }
        assignments.push(row_assignment);
    }

    Ok(AssignmentDcoShadowResult {
        assignments,
        telemetry,
    })
}

fn exact_assignment_for_row(
    point: &[f32],
    leaders_row_major: &[f32],
    leader_ids: &[u32],
    dim: usize,
    fanout: usize,
) -> Vec<usize> {
    let mut candidates = Vec::with_capacity(leader_ids.len());
    for leader_idx in 0..leader_ids.len() {
        let leader = &leaders_row_major[leader_idx * dim..(leader_idx + 1) * dim];
        candidates.push((leader_idx, exact_l2(point, leader)));
    }
    sort_candidates(&mut candidates, leader_ids);
    candidates
        .iter()
        .take(fanout.min(leader_ids.len()))
        .map(|&(leader_idx, _)| leader_idx)
        .collect()
}

fn sort_candidates(candidates: &mut [(usize, f32)], leader_ids: &[u32]) {
    candidates.sort_unstable_by(|left, right| {
        let left_id = leader_ids[left.0];
        let right_id = leader_ids[right.0];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_dco_exact_rerank_recovers_exact_assignment_when_survivor_contains_winner() {
        let point_ids = [10_u32];
        let leader_ids = [1_u32, 2_u32];
        let points = [0.0_f32, 100.0];
        let leaders = [
            0.0_f32, 0.0, // projection winner, exact loser
            1.0_f32, 100.0, // exact winner
        ];

        let result = assignment_dco_shadow_l2(
            &points,
            &point_ids,
            &leaders,
            &leader_ids,
            2,
            1,
            AssignmentDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::PrefixProjection,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
                proj_dim: 1,
                survivors: 2,
                audit: true,
            },
        )
        .expect("assignment DCO shadow");

        assert_eq!(result.assignments, vec![vec![1]]);
        assert_eq!(result.telemetry.nodes, 1);
        assert_eq!(result.telemetry.rows, 1);
        assert_eq!(result.telemetry.leader_pairs, 2);
        assert_eq!(result.telemetry.prefilter_pairs, 2);
        assert_eq!(result.telemetry.pruned_pairs, 0);
        assert_eq!(result.telemetry.survivor_pairs, 2);
        assert_eq!(result.telemetry.exact_pairs, 2);
        assert_eq!(result.telemetry.approx_output_pairs, 0);
        assert_eq!(result.telemetry.false_prune_audit_pairs, 2);
        assert_eq!(result.telemetry.false_prune_audit_misses, 0);
        assert_eq!(result.telemetry.metadata_bytes, 12);
    }

    #[test]
    fn assignment_dco_no_rerank_exposes_approx_assignment_miss() {
        let point_ids = [10_u32];
        let leader_ids = [1_u32, 2_u32];
        let points = [0.0_f32, 100.0];
        let leaders = [
            0.0_f32, 0.0, // projection winner, exact loser
            1.0_f32, 100.0, // exact winner
        ];

        let result = assignment_dco_shadow_l2(
            &points,
            &point_ids,
            &leaders,
            &leader_ids,
            2,
            1,
            AssignmentDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::PrefixProjection,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::None,
                proj_dim: 1,
                survivors: 1,
                audit: true,
            },
        )
        .expect("assignment DCO shadow");

        assert_eq!(result.assignments, vec![vec![0]]);
        assert_eq!(result.telemetry.prefilter_pairs, 2);
        assert_eq!(result.telemetry.pruned_pairs, 1);
        assert_eq!(result.telemetry.survivor_pairs, 1);
        assert_eq!(result.telemetry.exact_pairs, 0);
        assert_eq!(result.telemetry.approx_output_pairs, 1);
        assert_eq!(result.telemetry.false_prune_audit_pairs, 2);
        assert_eq!(result.telemetry.false_prune_audit_misses, 1);
    }

    #[test]
    fn assignment_dco_quantized_exact_rerank_uses_quantized_shortlist() {
        let point_ids = [10_u32];
        let leader_ids = [1_u32, 2_u32, 3_u32];
        let points = [0.03_f32, 0.03];
        let leaders = [
            0.02_f32, 0.02, // same quantized cell and exact winner
            0.26_f32, 0.26, 1.0_f32, 1.0,
        ];

        let result = assignment_dco_shadow_l2(
            &points,
            &point_ids,
            &leaders,
            &leader_ids,
            2,
            1,
            AssignmentDcoShadowConfig {
                mode: PiPNNGpuSubtreeDcoMode::Quantized,
                rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
                proj_dim: 2,
                survivors: 2,
                audit: true,
            },
        )
        .expect("assignment quantized DCO shadow");

        assert_eq!(result.assignments, vec![vec![0]]);
        assert_eq!(result.telemetry.prefilter_pairs, 3);
        assert_eq!(result.telemetry.pruned_pairs, 1);
        assert_eq!(result.telemetry.survivor_pairs, 2);
        assert_eq!(result.telemetry.exact_pairs, 2);
        assert_eq!(result.telemetry.approx_output_pairs, 0);
        assert_eq!(result.telemetry.false_prune_audit_misses, 0);
    }

    #[test]
    fn assignment_dco_shadow_rejects_mismatched_shapes() {
        let err = assignment_dco_shadow_l2(
            &[0.0, 1.0, 2.0],
            &[0],
            &[0.0, 1.0],
            &[0],
            2,
            1,
            AssignmentDcoShadowConfig {
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
