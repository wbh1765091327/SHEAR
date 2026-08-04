use std::cmp::Reverse;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::d0_capacity::{D0CapacityConfig, compute_thresholds, plan_root_leaders};
use super::params::PiPNNGpuSubtreeD0ThroughputModelMode;

const CANDIDATE_RATIOS_X1000: [usize; 5] = [1000, 850, 750, 600, 500];
const LEADER_TARGET_MULTIPLIERS_X1000: [usize; 8] =
    [1000, 1250, 1500, 2000, 2250, 2500, 3000, 4000];
const CHILD_P95_TARGET_ROWS: [usize; 6] = [
    2_500_000, 3_000_000, 3_500_000, 4_000_000, 5_000_000, 6_000_000,
];
const CANDIDATE_GENERATION_VERSION: &str = "leader-aware-v1";
const CHILD_SKEW_FACTOR_X1000: usize = 1500;
const LEAF_MULTIPLIER: usize = 8;
const TOPK_MULTIPLIER: usize = 1;
const PRELOAD_HIDDEN_RATIO_X1000: usize = 750;
const ROW_WEIGHT: u128 = 1;
const LEAF_WEIGHT: u128 = 1024;
const TOPK_WEIGHT: u128 = 256;
const SPLIT_WEIGHT: u128 = 4096;
const D0_LEADER_WEIGHT: u128 = 64;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct D0ThroughputCalibration {
    pub(crate) source: Option<String>,
    pub(crate) persistent_ms_per_score_x1000: Option<u128>,
    pub(crate) preload_ms_per_row_x1000: Option<u128>,
    pub(crate) d0_ms_per_leader_x1000: Option<u128>,
    pub(crate) child_skew_factor_x1000: Option<usize>,
}

impl D0ThroughputCalibration {
    pub(crate) fn missing() -> Self {
        Self::default()
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.persistent_ms_per_score_x1000.is_some()
            && self.preload_ms_per_row_x1000.is_some()
            && self.d0_ms_per_leader_x1000.is_some()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct D0ThroughputCalibrationSeed {
    dataset_label: String,
    total_points: usize,
    dim: usize,
    root_fanout: usize,
    sidecar_code_row_bytes: usize,
    observed_c_gpu: usize,
    observed_root_leaders: usize,
    observed_child_p50_rows: usize,
    observed_child_p95_rows: usize,
    observed_child_p99_rows: usize,
    observed_d0_wall_ms: u128,
    observed_persistent_total_ms: u128,
    observed_sidecar_gather_ms: u128,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum D0ThroughputCandidateSource {
    #[default]
    RowRatio,
    LeaderTarget,
    ChildP95Target,
}

impl D0ThroughputCandidateSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::RowRatio => "row_ratio",
            Self::LeaderTarget => "leader_target",
            Self::ChildP95Target => "child_p95_target",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct D0ThroughputModelInput {
    pub(crate) total_points: usize,
    pub(crate) dim: usize,
    pub(crate) root_fanout: usize,
    pub(crate) fanout_second: usize,
    pub(crate) leaf_target: usize,
    pub(crate) leaf_tile_rows: usize,
    pub(crate) min_efficient_rows: usize,
    pub(crate) memory_safe_rows: usize,
    pub(crate) sidecar_code_row_bytes: usize,
    pub(crate) persistent_growth_limit: f64,
    pub(crate) preload_visible_limit_ms: usize,
    pub(crate) min_ratio: f64,
    pub(crate) child_p95_guardrail_rows: Option<usize>,
    pub(crate) calibration: D0ThroughputCalibration,
}

impl D0ThroughputModelInput {
    fn child_skew_factor_x1000(&self) -> usize {
        self.calibration
            .child_skew_factor_x1000
            .unwrap_or(CHILD_SKEW_FACTOR_X1000)
            .max(1)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct D0ThroughputCandidate {
    pub(crate) source: D0ThroughputCandidateSource,
    pub(crate) rows: usize,
    pub(crate) ratio_x1000: usize,
    pub(crate) target_root_leaders: Option<usize>,
    pub(crate) root_leaders: usize,
    pub(crate) predicted_avg_child_rows: usize,
    pub(crate) predicted_p95_child_rows: usize,
    pub(crate) predicted_leaf_count: usize,
    pub(crate) predicted_topk_pops: usize,
    pub(crate) predicted_lazy_split_count: usize,
    pub(crate) predicted_persistent_score: u128,
    pub(crate) predicted_preload_visible_score: u128,
    pub(crate) predicted_d0_penalty_score: u128,
    pub(crate) structural_total_score: u128,
    pub(crate) predicted_persistent_ms: Option<u128>,
    pub(crate) predicted_preload_visible_ms: Option<u128>,
    pub(crate) predicted_d0_penalty_ms: Option<u128>,
    pub(crate) predicted_child_ms: Option<u128>,
    pub(crate) predicted_total_ms: Option<u128>,
    pub(crate) admitted: bool,
    pub(crate) reject_reason: Option<&'static str>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct D0ThroughputModelDecision {
    pub(crate) mode: PiPNNGpuSubtreeD0ThroughputModelMode,
    pub(crate) memory_safe_rows: usize,
    pub(crate) memory_safe_root_leaders: usize,
    pub(crate) selected_rows: usize,
    pub(crate) selected_root_leaders: usize,
    pub(crate) score_preferred_rows: usize,
    pub(crate) score_preferred_root_leaders: usize,
    pub(crate) selected_ratio_x1000: usize,
    pub(crate) candidate_count: usize,
    pub(crate) candidate_generation_version: &'static str,
    pub(crate) candidate_sources: Vec<&'static str>,
    pub(crate) child_skew_factor_x1000: usize,
    pub(crate) row_weight: u128,
    pub(crate) leaf_weight: u128,
    pub(crate) topk_weight: u128,
    pub(crate) split_weight: u128,
    pub(crate) d0_leader_weight: u128,
    pub(crate) predicted_persistent_ms: Option<u128>,
    pub(crate) predicted_preload_visible_ms: Option<u128>,
    pub(crate) predicted_d0_penalty_ms: Option<u128>,
    pub(crate) predicted_total_ms: Option<u128>,
    pub(crate) predicted_persistent_score: u128,
    pub(crate) predicted_preload_visible_score: u128,
    pub(crate) predicted_d0_penalty_score: u128,
    pub(crate) structural_total_score: u128,
    pub(crate) predicted_p95_child_rows: usize,
    pub(crate) predicted_leaf_count: usize,
    pub(crate) predicted_topk_pops: usize,
    pub(crate) predicted_lazy_split_count: usize,
    pub(crate) calibration_state: &'static str,
    pub(crate) calibration_source: Option<String>,
    pub(crate) reason: &'static str,
    pub(crate) candidates: Vec<D0ThroughputCandidate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct D0ThroughputModelError {
    pub(crate) reason: &'static str,
    pub(crate) memory_safe_rows: usize,
    pub(crate) candidate_count: usize,
    pub(crate) detail: Option<String>,
}

pub(crate) fn decide_d0_throughput_capacity(
    mode: PiPNNGpuSubtreeD0ThroughputModelMode,
    input: D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) -> Result<D0ThroughputModelDecision, D0ThroughputModelError> {
    if input.memory_safe_rows == 0 {
        return Err(D0ThroughputModelError {
            reason: "zero_memory_safe_rows",
            memory_safe_rows: 0,
            candidate_count: 0,
            detail: None,
        });
    }

    let candidates = build_candidates(&input, d0_cfg);
    if candidates.is_empty() {
        return Err(D0ThroughputModelError {
            reason: "empty_candidate_set",
            memory_safe_rows: input.memory_safe_rows,
            candidate_count: 0,
            detail: None,
        });
    }

    let baseline = candidates
        .iter()
        .find(|candidate| candidate.rows == input.memory_safe_rows)
        .unwrap_or(&candidates[0]);
    let score_preferred = candidates
        .iter()
        .filter(|candidate| candidate.admitted)
        .min_by_key(|candidate| candidate_rank_key(candidate))
        .unwrap_or(baseline);
    let has_calibration = input.calibration.is_complete();

    if mode == PiPNNGpuSubtreeD0ThroughputModelMode::Enforce && !has_calibration {
        return Err(D0ThroughputModelError {
            reason: "missing_calibration",
            memory_safe_rows: input.memory_safe_rows,
            candidate_count: candidates.len(),
            detail: None,
        });
    }

    let selected = if mode == PiPNNGpuSubtreeD0ThroughputModelMode::Enforce {
        select_calibrated_candidate(&candidates, baseline, &input).unwrap_or(baseline)
    } else {
        baseline
    };

    Ok(D0ThroughputModelDecision {
        mode,
        memory_safe_rows: input.memory_safe_rows,
        memory_safe_root_leaders: baseline.root_leaders,
        selected_rows: selected.rows,
        selected_root_leaders: selected.root_leaders,
        score_preferred_rows: score_preferred.rows,
        score_preferred_root_leaders: score_preferred.root_leaders,
        selected_ratio_x1000: selected.ratio_x1000,
        candidate_count: candidates.len(),
        candidate_generation_version: CANDIDATE_GENERATION_VERSION,
        candidate_sources: candidate_sources(&candidates),
        child_skew_factor_x1000: input.child_skew_factor_x1000(),
        row_weight: ROW_WEIGHT,
        leaf_weight: LEAF_WEIGHT,
        topk_weight: TOPK_WEIGHT,
        split_weight: SPLIT_WEIGHT,
        d0_leader_weight: D0_LEADER_WEIGHT,
        predicted_persistent_ms: selected.predicted_persistent_ms,
        predicted_preload_visible_ms: selected.predicted_preload_visible_ms,
        predicted_d0_penalty_ms: selected.predicted_d0_penalty_ms,
        predicted_total_ms: selected.predicted_total_ms,
        predicted_persistent_score: selected.predicted_persistent_score,
        predicted_preload_visible_score: selected.predicted_preload_visible_score,
        predicted_d0_penalty_score: selected.predicted_d0_penalty_score,
        structural_total_score: selected.structural_total_score,
        predicted_p95_child_rows: selected.predicted_p95_child_rows,
        predicted_leaf_count: selected.predicted_leaf_count,
        predicted_topk_pops: selected.predicted_topk_pops,
        predicted_lazy_split_count: selected.predicted_lazy_split_count,
        calibration_state: if has_calibration {
            "complete"
        } else {
            "missing"
        },
        calibration_source: input.calibration.source.clone(),
        reason: decision_reason(
            mode,
            has_calibration,
            selected.rows,
            score_preferred.rows,
            input.memory_safe_rows,
        ),
        candidates,
    })
}

pub(crate) fn calibration_from_json_path(
    path: &Path,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) -> Result<D0ThroughputCalibration, D0ThroughputModelError> {
    let text = fs::read_to_string(path).map_err(|err| D0ThroughputModelError {
        reason: "calibration_read_failed",
        memory_safe_rows: input.memory_safe_rows,
        candidate_count: 0,
        detail: Some(format!("{}: {err}", path.display())),
    })?;
    calibration_from_json_str(&text, input, d0_cfg)
}

fn calibration_from_json_str(
    text: &str,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) -> Result<D0ThroughputCalibration, D0ThroughputModelError> {
    let seed: D0ThroughputCalibrationSeed =
        serde_json::from_str(text).map_err(|err| D0ThroughputModelError {
            reason: "calibration_parse_failed",
            memory_safe_rows: input.memory_safe_rows,
            candidate_count: 0,
            detail: Some(err.to_string()),
        })?;

    if !calibration_seed_is_valid(&seed, input) {
        return Err(D0ThroughputModelError {
            reason: "calibration_invalid",
            memory_safe_rows: input.memory_safe_rows,
            candidate_count: 0,
            detail: Some(
                "calibration seed does not match model input or has invalid observed fields"
                    .to_string(),
            ),
        });
    }

    let observed_p95 = seed.observed_child_p95_rows;
    let observed_leaf_count = ceil_saturating_usize(
        (observed_p95 as u128).saturating_mul(input.fanout_second.max(1) as u128),
        input.leaf_target.max(1) as u128,
    )
    .saturating_mul(LEAF_MULTIPLIER);
    let observed_topk_pops = observed_p95
        .div_ceil(input.leaf_tile_rows.max(1))
        .saturating_mul(input.fanout_second.max(1))
        .saturating_mul(TOPK_MULTIPLIER);
    let thresholds = compute_thresholds(seed.observed_c_gpu, d0_cfg);
    let observed_lazy_split_count = if thresholds.c_high == 0 {
        0
    } else {
        observed_p95.div_ceil(thresholds.c_high).saturating_sub(1)
    };
    let observed_persistent_score = persistent_score(
        observed_p95,
        observed_leaf_count,
        observed_topk_pops,
        observed_lazy_split_count,
    );
    let persistent_scale = seed
        .observed_persistent_total_ms
        .saturating_mul(1000)
        .div_ceil(observed_persistent_score.max(1))
        .max(1);
    let preload_scale = seed
        .observed_sidecar_gather_ms
        .saturating_mul(1000)
        .div_ceil(seed.observed_c_gpu.max(1) as u128)
        .max(1);
    let d0_scale = seed
        .observed_d0_wall_ms
        .saturating_mul(1000)
        .div_ceil(seed.observed_root_leaders.max(1) as u128)
        .max(1);
    let avg_child_rows = (seed.total_points as u128)
        .saturating_mul(seed.root_fanout.max(1) as u128)
        .div_ceil(seed.observed_root_leaders.max(1) as u128)
        .max(1);
    let child_skew = (seed.observed_child_p95_rows as u128)
        .saturating_mul(1000)
        .div_ceil(avg_child_rows)
        .max(1)
        .min(usize::MAX as u128) as usize;

    Ok(D0ThroughputCalibration {
        source: Some(seed.dataset_label),
        persistent_ms_per_score_x1000: Some(persistent_scale),
        preload_ms_per_row_x1000: Some(preload_scale),
        d0_ms_per_leader_x1000: Some(d0_scale),
        child_skew_factor_x1000: Some(child_skew),
    })
}

fn calibration_seed_is_valid(
    seed: &D0ThroughputCalibrationSeed,
    input: &D0ThroughputModelInput,
) -> bool {
    seed.total_points > 0
        && seed.dim > 0
        && seed.root_fanout > 0
        && seed.sidecar_code_row_bytes > 0
        && seed.observed_c_gpu > 0
        && seed.observed_root_leaders > 0
        && seed.observed_child_p50_rows > 0
        && seed.observed_child_p95_rows > 0
        && seed.observed_child_p99_rows > 0
        && seed.observed_d0_wall_ms > 0
        && seed.observed_persistent_total_ms > 0
        && seed.observed_sidecar_gather_ms > 0
        && seed.total_points == input.total_points
        && seed.root_fanout == input.root_fanout
        && seed.dim == input.dim
        && seed.sidecar_code_row_bytes == input.sidecar_code_row_bytes
        && seed.observed_child_p50_rows <= seed.observed_child_p95_rows
        && seed.observed_child_p95_rows <= seed.observed_child_p99_rows
}

fn ceil_saturating_usize(numerator: u128, denominator: u128) -> usize {
    numerator
        .div_ceil(denominator.max(1))
        .min(usize::MAX as u128) as usize
}

fn decision_reason(
    mode: PiPNNGpuSubtreeD0ThroughputModelMode,
    has_calibration: bool,
    selected_rows: usize,
    score_preferred_rows: usize,
    memory_safe_rows: usize,
) -> &'static str {
    match mode {
        PiPNNGpuSubtreeD0ThroughputModelMode::Off => "off",
        PiPNNGpuSubtreeD0ThroughputModelMode::Report if !has_calibration => {
            if score_preferred_rows < memory_safe_rows {
                "report_only_missing_calibration_score_prefers_smaller"
            } else {
                "report_only_missing_calibration"
            }
        }
        PiPNNGpuSubtreeD0ThroughputModelMode::Report => "report_only",
        PiPNNGpuSubtreeD0ThroughputModelMode::Enforce if selected_rows < memory_safe_rows => {
            "enforced_smaller_capacity"
        }
        PiPNNGpuSubtreeD0ThroughputModelMode::Enforce => "enforced_memory_safe_capacity",
    }
}

fn build_candidates(
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) -> Vec<D0ThroughputCandidate> {
    let min_ratio_x1000 = ratio_to_x1000(input.min_ratio, 500);
    let mut candidates = Vec::new();

    for ratio_x1000 in CANDIDATE_RATIOS_X1000 {
        if ratio_x1000 < min_ratio_x1000 {
            continue;
        }
        let rows = scale_by_x1000(input.memory_safe_rows, ratio_x1000)
            .max(1)
            .min(input.memory_safe_rows);
        add_candidate(
            &mut candidates,
            D0ThroughputCandidateSource::RowRatio,
            rows,
            ratio_x1000,
            None,
            input,
            d0_cfg,
        );
    }

    push_leader_target_candidates(&mut candidates, input, d0_cfg);
    push_child_p95_target_candidates(&mut candidates, input, d0_cfg);
    add_candidate(
        &mut candidates,
        D0ThroughputCandidateSource::RowRatio,
        input.memory_safe_rows,
        1000,
        None,
        input,
        d0_cfg,
    );
    candidates.sort_by(|left, right| {
        left.rows
            .cmp(&right.rows)
            .then_with(|| left.root_leaders.cmp(&right.root_leaders))
            .then_with(|| left.source.as_str().cmp(right.source.as_str()))
    });
    candidates
}

fn add_candidate(
    candidates: &mut Vec<D0ThroughputCandidate>,
    source: D0ThroughputCandidateSource,
    rows: usize,
    ratio_x1000: usize,
    target_root_leaders: Option<usize>,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) {
    let candidate = score_candidate(
        source,
        rows,
        ratio_x1000,
        target_root_leaders,
        input,
        d0_cfg,
    );
    if candidates.iter().any(|existing| {
        existing.rows == candidate.rows && existing.root_leaders == candidate.root_leaders
    }) {
        return;
    }
    candidates.push(candidate);
}

fn candidate_rows_for_target_leaders(
    target_root_leaders: usize,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) -> usize {
    let assignments = input.total_points.saturating_mul(input.root_fanout.max(1));
    let target_child_rows = assignments.div_ceil(target_root_leaders.max(1));
    let ratio = d0_cfg.target_ratio.clamp(0.01, 1.0);
    ((target_child_rows as f64) / ratio).ceil().max(1.0) as usize
}

fn push_leader_target_candidates(
    candidates: &mut Vec<D0ThroughputCandidate>,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) {
    let memory_safe_candidate = score_candidate(
        D0ThroughputCandidateSource::LeaderTarget,
        input.memory_safe_rows.max(1),
        1000,
        None,
        input,
        d0_cfg,
    );
    let base_root_leaders = memory_safe_candidate.root_leaders.max(1);
    for multiplier_x1000 in LEADER_TARGET_MULTIPLIERS_X1000 {
        let target_root_leaders = scale_by_x1000(base_root_leaders, multiplier_x1000).max(1);
        let rows = candidate_rows_for_target_leaders(target_root_leaders, input, d0_cfg);
        add_candidate(
            candidates,
            D0ThroughputCandidateSource::LeaderTarget,
            rows,
            ratio_against_safe_x1000(rows, input.memory_safe_rows),
            Some(target_root_leaders),
            input,
            d0_cfg,
        );
    }
}

fn push_child_p95_target_candidates(
    candidates: &mut Vec<D0ThroughputCandidate>,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) {
    let assignments = input.total_points.saturating_mul(input.root_fanout.max(1));
    for target_child_p95_rows in CHILD_P95_TARGET_ROWS {
        let target_avg_child_rows = (target_child_p95_rows as u128)
            .saturating_mul(1000)
            .div_ceil(input.child_skew_factor_x1000() as u128);
        let target_root_leaders = (assignments as u128)
            .div_ceil(target_avg_child_rows.max(1))
            .min(usize::MAX as u128) as usize;
        let rows = candidate_rows_for_target_leaders(target_root_leaders, input, d0_cfg);
        add_candidate(
            candidates,
            D0ThroughputCandidateSource::ChildP95Target,
            rows,
            ratio_against_safe_x1000(rows, input.memory_safe_rows),
            Some(target_root_leaders.max(1)),
            input,
            d0_cfg,
        );
    }
}

fn ratio_against_safe_x1000(rows: usize, memory_safe_rows: usize) -> usize {
    if memory_safe_rows == 0 {
        return 1000;
    }
    (rows as u128)
        .saturating_mul(1000)
        .div_ceil(memory_safe_rows as u128)
        .min(1000) as usize
}

fn score_candidate(
    source: D0ThroughputCandidateSource,
    rows: usize,
    ratio_x1000: usize,
    target_root_leaders: Option<usize>,
    input: &D0ThroughputModelInput,
    d0_cfg: D0CapacityConfig,
) -> D0ThroughputCandidate {
    let effective_rows = rows.max(1).min(input.memory_safe_rows);
    let thresholds = compute_thresholds(effective_rows, d0_cfg);
    let leader_plan = plan_root_leaders(input.total_points, input.root_fanout, thresholds, d0_cfg);
    let root_leaders = leader_plan.k_initial.max(1);
    let assignments = input.total_points.saturating_mul(input.root_fanout.max(1));
    let predicted_avg_child_rows = assignments.div_ceil(root_leaders);
    let predicted_p95_child_rows =
        scale_by_x1000(predicted_avg_child_rows, input.child_skew_factor_x1000()).max(1);
    let fanout_amplification = input.fanout_second.max(1);
    let leaf_target = input.leaf_target.max(1);
    let leaf_tile_rows = input.leaf_tile_rows.max(1);
    let predicted_leaf_count = predicted_p95_child_rows
        .saturating_mul(fanout_amplification)
        .div_ceil(leaf_target)
        .max(1)
        .saturating_mul(LEAF_MULTIPLIER);
    let predicted_topk_pops = predicted_p95_child_rows
        .div_ceil(leaf_tile_rows)
        .saturating_mul(fanout_amplification)
        .saturating_mul(TOPK_MULTIPLIER);
    let predicted_lazy_split_count = if thresholds.c_high == 0 {
        0
    } else {
        predicted_p95_child_rows
            .div_ceil(thresholds.c_high)
            .saturating_sub(1)
    };
    let predicted_persistent_score = persistent_score(
        predicted_p95_child_rows,
        predicted_leaf_count,
        predicted_topk_pops,
        predicted_lazy_split_count,
    );
    let sidecar_rows_score =
        effective_rows.saturating_mul(input.sidecar_code_row_bytes.max(1)) as u128;
    let hidden = (predicted_persistent_score * PRELOAD_HIDDEN_RATIO_X1000 as u128) / 1000;
    let predicted_preload_visible_score = sidecar_rows_score.saturating_sub(hidden);
    let predicted_d0_penalty_score = (root_leaders as u128).saturating_mul(D0_LEADER_WEIGHT);
    let structural_total_score = predicted_persistent_score
        .max(predicted_preload_visible_score)
        .saturating_add(predicted_d0_penalty_score);
    let predicted_persistent_ms = input
        .calibration
        .persistent_ms_per_score_x1000
        .map(|scale| {
            predicted_persistent_score
                .saturating_mul(scale)
                .div_ceil(1000)
        });
    let predicted_preload_visible_ms = input.calibration.preload_ms_per_row_x1000.map(|scale| {
        (effective_rows as u128)
            .saturating_mul(scale)
            .div_ceil(1000)
    });
    let predicted_child_ms = match (predicted_persistent_ms, predicted_preload_visible_ms) {
        (Some(persistent), Some(preload)) => Some(persistent.max(preload)),
        _ => None,
    };
    let predicted_d0_penalty_ms = input
        .calibration
        .d0_ms_per_leader_x1000
        .map(|scale| (root_leaders as u128).saturating_mul(scale).div_ceil(1000));
    let predicted_total_ms = match (predicted_child_ms, predicted_d0_penalty_ms) {
        (Some(child), Some(d0)) => Some(child.saturating_add(d0)),
        _ => None,
    };
    let min_admitted_rows = input.min_efficient_rows.min(input.memory_safe_rows);
    let min_ratio_x1000 = ratio_to_x1000(input.min_ratio, 500);
    let (admitted, reject_reason) = if rows > input.memory_safe_rows {
        (false, Some("above_memory_safe_rows"))
    } else if rows < min_admitted_rows && rows != input.memory_safe_rows {
        (false, Some("below_min_efficient_rows"))
    } else if ratio_x1000 < min_ratio_x1000 && rows != input.memory_safe_rows {
        (false, Some("below_min_ratio"))
    } else if input
        .child_p95_guardrail_rows
        .is_some_and(|limit| predicted_p95_child_rows > limit)
    {
        (false, Some("above_child_p95_guardrail"))
    } else {
        (true, None)
    };

    D0ThroughputCandidate {
        source,
        rows,
        ratio_x1000,
        target_root_leaders,
        root_leaders,
        predicted_avg_child_rows,
        predicted_p95_child_rows,
        predicted_leaf_count,
        predicted_topk_pops,
        predicted_lazy_split_count,
        predicted_persistent_score,
        predicted_preload_visible_score,
        predicted_d0_penalty_score,
        structural_total_score,
        predicted_persistent_ms,
        predicted_preload_visible_ms,
        predicted_d0_penalty_ms,
        predicted_child_ms,
        predicted_total_ms,
        admitted,
        reject_reason,
    }
}

fn candidate_rank_key(candidate: &D0ThroughputCandidate) -> (u128, usize, Reverse<usize>) {
    let total = candidate
        .predicted_total_ms
        .unwrap_or(candidate.structural_total_score);
    (
        total,
        candidate.predicted_p95_child_rows,
        Reverse(candidate.rows),
    )
}

fn select_calibrated_candidate<'a>(
    candidates: &'a [D0ThroughputCandidate],
    baseline: &'a D0ThroughputCandidate,
    input: &D0ThroughputModelInput,
) -> Option<&'a D0ThroughputCandidate> {
    let baseline_persistent_ms = baseline.predicted_persistent_ms?;
    let persistent_limit_ms =
        ((baseline_persistent_ms as f64) * input.persistent_growth_limit).ceil() as u128;
    let preload_limit_ms = input.preload_visible_limit_ms as u128;
    candidates
        .iter()
        .filter(|candidate| {
            candidate.admitted
                && candidate.predicted_total_ms.is_some()
                && candidate.predicted_persistent_ms.unwrap_or(u128::MAX) <= persistent_limit_ms
                && candidate.predicted_preload_visible_ms.unwrap_or(u128::MAX) <= preload_limit_ms
        })
        .min_by_key(|candidate| candidate_rank_key(candidate))
}

fn persistent_score(
    rows: usize,
    leaf_count: usize,
    topk_pops: usize,
    lazy_split_count: usize,
) -> u128 {
    (rows as u128)
        .saturating_mul(ROW_WEIGHT)
        .saturating_add((leaf_count as u128).saturating_mul(LEAF_WEIGHT))
        .saturating_add((topk_pops as u128).saturating_mul(TOPK_WEIGHT))
        .saturating_add((lazy_split_count as u128).saturating_mul(SPLIT_WEIGHT))
}

fn candidate_sources(candidates: &[D0ThroughputCandidate]) -> Vec<&'static str> {
    if candidates.is_empty() {
        return Vec::new();
    }
    vec![
        D0ThroughputCandidateSource::RowRatio.as_str(),
        D0ThroughputCandidateSource::LeaderTarget.as_str(),
        D0ThroughputCandidateSource::ChildP95Target.as_str(),
    ]
}

fn ratio_to_x1000(ratio: f64, default_x1000: usize) -> usize {
    if ratio.is_finite() && ratio > 0.0 {
        ((ratio * 1000.0).round() as usize).clamp(1, 1000)
    } else {
        default_x1000
    }
}

fn scale_by_x1000(value: usize, ratio_x1000: usize) -> usize {
    let scaled = (value as u128)
        .saturating_mul(ratio_x1000 as u128)
        .div_ceil(1000);
    scaled.min(usize::MAX as u128) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::pipnn::d0_capacity::D0CapacityConfig;
    use crate::algorithm::pipnn::params::PiPNNGpuSubtreeD0ThroughputModelMode;

    fn input(memory_safe_rows: usize) -> D0ThroughputModelInput {
        D0ThroughputModelInput {
            total_points: 35_000_000,
            dim: 1024,
            root_fanout: 6,
            fanout_second: 6,
            leaf_target: 2048,
            leaf_tile_rows: 4096,
            min_efficient_rows: 32 * 1024,
            memory_safe_rows,
            sidecar_code_row_bytes: 128,
            persistent_growth_limit: 1.10,
            preload_visible_limit_ms: 5000,
            min_ratio: 0.50,
            child_p95_guardrail_rows: None,
            calibration: D0ThroughputCalibration::missing(),
        }
    }

    fn wiki35m_input(memory_safe_rows: usize) -> D0ThroughputModelInput {
        D0ThroughputModelInput {
            total_points: 35_000_000,
            dim: 1024,
            root_fanout: 8,
            fanout_second: 6,
            leaf_target: 2048,
            leaf_tile_rows: 4096,
            min_efficient_rows: 32 * 1024,
            memory_safe_rows,
            sidecar_code_row_bytes: 128,
            persistent_growth_limit: 1.10,
            preload_visible_limit_ms: 5000,
            min_ratio: 0.25,
            child_p95_guardrail_rows: None,
            calibration: D0ThroughputCalibration::missing(),
        }
    }

    #[test]
    fn d0_throughput_report_keeps_memory_safe_rows_without_calibration() {
        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Report,
            input(1_000_000),
            D0CapacityConfig::default(),
        )
        .unwrap();
        assert_eq!(decision.selected_rows, 1_000_000);
        assert!(decision.score_preferred_rows <= decision.memory_safe_rows);
        assert_eq!(decision.calibration_state, "missing");
        assert!(decision.predicted_persistent_ms.is_none());
    }

    #[test]
    fn d0_throughput_report_ranks_smaller_leader_aware_candidate_without_enforcing() {
        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Report,
            wiki35m_input(12_145_001),
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.selected_rows, 12_145_001);
        assert_eq!(
            decision.selected_root_leaders,
            decision.memory_safe_root_leaders
        );
        assert!(decision.score_preferred_root_leaders >= 128);
        assert!(decision.score_preferred_rows < decision.memory_safe_rows);
        assert_eq!(
            decision.reason,
            "report_only_missing_calibration_score_prefers_smaller"
        );
    }

    #[test]
    fn d0_throughput_enforce_rejects_missing_calibration() {
        let err = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Enforce,
            input(1_000_000),
            D0CapacityConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err.reason, "missing_calibration");
        assert_eq!(err.memory_safe_rows, 1_000_000);
    }

    #[test]
    fn d0_throughput_enforce_respects_child_p95_guardrail() {
        let mut input = wiki35m_input(12_145_001);
        input.child_p95_guardrail_rows = Some(4_000_000);
        input.calibration = D0ThroughputCalibration {
            source: Some("unit".to_string()),
            persistent_ms_per_score_x1000: Some(1),
            preload_ms_per_row_x1000: Some(1),
            d0_ms_per_leader_x1000: Some(1),
            child_skew_factor_x1000: Some(1500),
        };

        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Enforce,
            input,
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert!(decision.selected_rows < decision.memory_safe_rows);
        assert!(decision.selected_root_leaders >= 128);
        assert!(decision.predicted_p95_child_rows <= 4_000_000);
        assert_eq!(decision.calibration_state, "complete");
    }

    #[test]
    fn d0_throughput_enforce_rejects_all_candidates_below_min_ratio() {
        let mut input = wiki35m_input(12_145_001);
        input.min_ratio = 0.95;
        input.calibration = D0ThroughputCalibration {
            source: Some("unit".to_string()),
            persistent_ms_per_score_x1000: Some(1),
            preload_ms_per_row_x1000: Some(1),
            d0_ms_per_leader_x1000: Some(1),
            child_skew_factor_x1000: Some(1500),
        };

        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Enforce,
            input,
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert!(
            decision
                .candidates
                .iter()
                .all(|candidate| { !candidate.admitted || candidate.ratio_x1000 >= 950 })
        );
        assert!(decision.candidates.iter().any(|candidate| {
            candidate.source != D0ThroughputCandidateSource::RowRatio
                && candidate.ratio_x1000 < 950
                && candidate.reject_reason == Some("below_min_ratio")
        }));
        assert_eq!(decision.selected_rows, decision.memory_safe_rows);
    }

    #[test]
    fn d0_throughput_enforce_falls_back_to_baseline_when_limits_reject_smaller_candidates() {
        let mut input = wiki35m_input(12_145_001);
        input.calibration = D0ThroughputCalibration {
            source: Some("unit".to_string()),
            persistent_ms_per_score_x1000: Some(1),
            preload_ms_per_row_x1000: Some(1),
            d0_ms_per_leader_x1000: Some(1),
            child_skew_factor_x1000: Some(1500),
        };
        input.preload_visible_limit_ms = 1;

        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Enforce,
            input,
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.selected_rows, decision.memory_safe_rows);
        assert_eq!(
            decision.selected_root_leaders,
            decision.memory_safe_root_leaders
        );
    }

    #[test]
    fn d0_throughput_candidates_include_memory_safe_rows() {
        let input = input(1_000_000);
        let candidates = build_candidates(&input, D0CapacityConfig::default());
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.rows == 1_000_000)
        );
        assert!(candidates.iter().all(|candidate| candidate.rows > 0));
        assert!(
            candidates
                .iter()
                .filter(|candidate| candidate.admitted)
                .all(|candidate| candidate.rows <= 1_000_000)
        );
    }

    #[test]
    fn d0_throughput_add_candidate_records_above_memory_rejection() {
        let input = input(1_000_000);
        let mut candidates = Vec::new();
        add_candidate(
            &mut candidates,
            D0ThroughputCandidateSource::LeaderTarget,
            1_250_000,
            1250,
            Some(1),
            &input,
            D0CapacityConfig::default(),
        );

        let candidate = candidates
            .iter()
            .find(|candidate| candidate.rows == 1_250_000)
            .expect("above-memory requested rows should be kept for telemetry");
        assert!(!candidate.admitted);
        assert_eq!(candidate.reject_reason, Some("above_memory_safe_rows"));
    }

    #[test]
    fn d0_throughput_add_candidate_records_below_min_rejection() {
        let input = input(1_000_000);
        let mut candidates = Vec::new();
        add_candidate(
            &mut candidates,
            D0ThroughputCandidateSource::LeaderTarget,
            16 * 1024,
            16,
            Some(1),
            &input,
            D0CapacityConfig::default(),
        );

        let candidate = candidates
            .iter()
            .find(|candidate| candidate.rows == 16 * 1024)
            .expect("below-min requested rows should be kept for telemetry");
        assert!(!candidate.admitted);
        assert_eq!(candidate.reject_reason, Some("below_min_efficient_rows"));
    }

    #[test]
    fn d0_leader_aware_candidates_cover_wiki35m_147_region() {
        let input = wiki35m_input(12_145_001);
        let candidates = build_candidates(&input, D0CapacityConfig::default());
        assert!(
            candidates.iter().any(|candidate| candidate.source
                == D0ThroughputCandidateSource::LeaderTarget
                && (128..=160).contains(&candidate.root_leaders)),
            "expected a leader-target candidate in the 128..=160 band, got {:?}",
            candidates
                .iter()
                .map(|candidate| (candidate.source, candidate.rows, candidate.root_leaders))
                .collect::<Vec<_>>()
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.source == D0ThroughputCandidateSource::ChildP95Target),
            "expected child-p95 target candidates"
        );
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.rows <= 12_145_001)
        );
    }

    #[test]
    fn d0_throughput_decision_reports_all_wiki35m_generator_sources() {
        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Report,
            wiki35m_input(12_145_001),
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert_eq!(
            decision.candidate_sources,
            vec!["row_ratio", "leader_target", "child_p95_target"]
        );
    }

    #[test]
    fn d0_throughput_candidate_sources_survive_deduped_generators() {
        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Report,
            input(32 * 1024),
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert_eq!(
            decision
                .candidates
                .iter()
                .filter(|candidate| candidate.admitted)
                .count(),
            1
        );
        assert_eq!(
            decision.candidate_sources,
            vec!["row_ratio", "leader_target", "child_p95_target"]
        );
    }

    #[test]
    fn d0_throughput_candidates_are_sorted_deterministically() {
        let input = wiki35m_input(12_145_001);
        let candidates = build_candidates(&input, D0CapacityConfig::default());
        let mut sorted = candidates.clone();
        sorted.sort_by(|left, right| {
            left.rows
                .cmp(&right.rows)
                .then_with(|| left.root_leaders.cmp(&right.root_leaders))
                .then_with(|| left.source.as_str().cmp(right.source.as_str()))
        });

        assert_eq!(candidates, sorted);
    }

    #[test]
    fn d0_throughput_candidate_serial_metadata_defaults_are_populated() {
        let input = input(1_000_000);
        let candidates = build_candidates(&input, D0CapacityConfig::default());
        let baseline = candidates
            .iter()
            .find(|candidate| candidate.rows == 1_000_000)
            .expect("memory-safe candidate must be present");

        assert_eq!(baseline.source, D0ThroughputCandidateSource::RowRatio);
        assert_eq!(baseline.target_root_leaders, None);
        assert!(baseline.predicted_avg_child_rows > 0);
        assert!(baseline.predicted_d0_penalty_score > 0);
        assert!(baseline.structural_total_score >= baseline.predicted_persistent_score);
        assert!(baseline.predicted_total_ms.is_none());
        assert!(baseline.admitted);
        assert_eq!(baseline.reject_reason, None);
    }

    #[test]
    fn d0_throughput_enforce_selects_calibrated_candidate() {
        let mut calibrated = input(1_000_000);
        calibrated.calibration = D0ThroughputCalibration {
            source: Some("unit".to_string()),
            persistent_ms_per_score_x1000: Some(1),
            preload_ms_per_row_x1000: Some(1),
            d0_ms_per_leader_x1000: Some(1),
            child_skew_factor_x1000: Some(1500),
        };
        calibrated.preload_visible_limit_ms = 900;
        let decision = decide_d0_throughput_capacity(
            PiPNNGpuSubtreeD0ThroughputModelMode::Enforce,
            calibrated,
            D0CapacityConfig::default(),
        )
        .unwrap();
        assert!(decision.selected_rows <= 1_000_000);
        assert_eq!(decision.calibration_state, "complete");
        assert!(decision.predicted_persistent_ms.is_some());
        assert!(decision.predicted_preload_visible_ms.is_some());
    }

    #[test]
    fn d0_throughput_calibration_seed_derives_scales() {
        let text = r#"{
  "dataset_label": "wiki35m-entrypoint-no-medoid",
  "total_points": 35000000,
  "dim": 1024,
  "root_fanout": 8,
  "sidecar_code_row_bytes": 128,
  "observed_c_gpu": 3832598,
  "observed_root_leaders": 147,
  "observed_child_p50_rows": 1839510,
  "observed_child_p95_rows": 3138345,
  "observed_child_p99_rows": 3492873,
  "observed_d0_wall_ms": 75909,
  "observed_persistent_total_ms": 296873,
  "observed_sidecar_gather_ms": 131258
}"#;

        let calibration = calibration_from_json_str(
            text,
            &wiki35m_input(12_145_001),
            D0CapacityConfig::default(),
        )
        .unwrap();

        assert!(calibration.is_complete());
        assert_eq!(
            calibration.source.as_deref(),
            Some("wiki35m-entrypoint-no-medoid")
        );
        assert!(calibration.persistent_ms_per_score_x1000.unwrap() > 0);
        assert!(calibration.preload_ms_per_row_x1000.unwrap() > 0);
        assert!(calibration.d0_ms_per_leader_x1000.unwrap() > 0);
        assert!(calibration.child_skew_factor_x1000.unwrap() >= 1000);
    }

    #[test]
    fn d0_throughput_calibration_seed_rejects_zero_observed_metric() {
        let text = r#"{
  "dataset_label": "wiki35m-entrypoint-no-medoid",
  "total_points": 35000000,
  "dim": 1024,
  "root_fanout": 8,
  "sidecar_code_row_bytes": 128,
  "observed_c_gpu": 3832598,
  "observed_root_leaders": 147,
  "observed_child_p50_rows": 0,
  "observed_child_p95_rows": 3138345,
  "observed_child_p99_rows": 3492873,
  "observed_d0_wall_ms": 75909,
  "observed_persistent_total_ms": 296873,
  "observed_sidecar_gather_ms": 131258
}"#;

        let err = calibration_from_json_str(
            text,
            &wiki35m_input(12_145_001),
            D0CapacityConfig::default(),
        )
        .unwrap_err();

        assert_eq!(err.reason, "calibration_invalid");
    }

    #[test]
    fn d0_throughput_calibration_seed_rejects_dimension_mismatch() {
        let text = r#"{
  "dataset_label": "wiki35m-entrypoint-no-medoid",
  "total_points": 35000000,
  "dim": 999,
  "root_fanout": 8,
  "sidecar_code_row_bytes": 128,
  "observed_c_gpu": 3832598,
  "observed_root_leaders": 147,
  "observed_child_p50_rows": 1839510,
  "observed_child_p95_rows": 3138345,
  "observed_child_p99_rows": 3492873,
  "observed_d0_wall_ms": 75909,
  "observed_persistent_total_ms": 296873,
  "observed_sidecar_gather_ms": 131258
}"#;

        let err = calibration_from_json_str(
            text,
            &wiki35m_input(12_145_001),
            D0CapacityConfig::default(),
        )
        .unwrap_err();

        assert_eq!(err.reason, "calibration_invalid");
    }

    #[test]
    fn d0_throughput_calibration_seed_uses_observed_p95_score() {
        let text = r#"{
  "dataset_label": "wiki35m-entrypoint-no-medoid",
  "total_points": 35000000,
  "dim": 1024,
  "root_fanout": 8,
  "sidecar_code_row_bytes": 128,
  "observed_c_gpu": 3832598,
  "observed_root_leaders": 147,
  "observed_child_p50_rows": 1839510,
  "observed_child_p95_rows": 3138345,
  "observed_child_p99_rows": 3492873,
  "observed_d0_wall_ms": 75909,
  "observed_persistent_total_ms": 296873,
  "observed_sidecar_gather_ms": 131258
}"#;
        let input = wiki35m_input(12_145_001);
        let d0_cfg = D0CapacityConfig::default();
        let thresholds = compute_thresholds(3_832_598, d0_cfg);
        let observed_p95 = 3_138_345usize;
        let observed_leaf_count = observed_p95
            .saturating_mul(input.fanout_second)
            .div_ceil(input.leaf_target)
            .saturating_mul(LEAF_MULTIPLIER);
        let observed_topk_pops = observed_p95
            .div_ceil(input.leaf_tile_rows)
            .saturating_mul(input.fanout_second)
            .saturating_mul(TOPK_MULTIPLIER);
        let observed_lazy_split_count = observed_p95
            .div_ceil(thresholds.c_high.max(1))
            .saturating_sub(1);
        let observed_persistent_score = persistent_score(
            observed_p95,
            observed_leaf_count,
            observed_topk_pops,
            observed_lazy_split_count,
        );
        let expected_persistent_scale = 296_873_u128
            .saturating_mul(1000)
            .div_ceil(observed_persistent_score.max(1));

        let calibration = calibration_from_json_str(text, &input, d0_cfg).unwrap();

        assert_eq!(
            calibration.persistent_ms_per_score_x1000,
            Some(expected_persistent_scale)
        );
        assert_eq!(
            calibration.d0_ms_per_leader_x1000,
            Some(75_909_u128.saturating_mul(1000).div_ceil(147))
        );
    }
}
