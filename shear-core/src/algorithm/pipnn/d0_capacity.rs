use super::params::PiPNNParams;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct D0CapacityThresholds {
    pub(crate) c_gpu: usize,
    pub(crate) c_high: usize,
    pub(crate) c_target: usize,
    pub(crate) c_low: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct D0CapacityConfig {
    pub(crate) enabled: bool,
    pub(crate) high_ratio: f64,
    pub(crate) target_ratio: f64,
    pub(crate) low_ratio: f64,
    pub(crate) min_efficient_rows: usize,
    pub(crate) quality_min_root_leaders: usize,
    pub(crate) root_leader_cap: usize,
    pub(crate) split_recursion_limit: usize,
}

impl Default for D0CapacityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            high_ratio: 0.60,
            target_ratio: 0.50,
            low_ratio: 0.15,
            min_efficient_rows: 32 * 1024,
            quality_min_root_leaders: 64,
            root_leader_cap: 10_000,
            split_recursion_limit: 8,
        }
    }
}

pub(crate) fn config_from_params(params: &PiPNNParams) -> D0CapacityConfig {
    D0CapacityConfig {
        enabled: params.gpu_subtree_d0_capacity_enable,
        high_ratio: params.gpu_subtree_d0_capacity_high_ratio,
        target_ratio: params.gpu_subtree_d0_capacity_target_ratio,
        low_ratio: params.gpu_subtree_d0_capacity_low_ratio,
        min_efficient_rows: params.gpu_subtree_d0_capacity_min_efficient_rows,
        quality_min_root_leaders: params.gpu_subtree_d0_capacity_quality_min_root_leaders,
        root_leader_cap: params.gpu_subtree_d0_capacity_root_leader_cap,
        split_recursion_limit: params.gpu_subtree_d0_capacity_split_recursion_limit,
    }
}

#[cfg(any(feature = "pipnn_cuda", test))]
use super::compute_backend::plan_gpu_persistent_subtree_resident_admission;

#[cfg(any(feature = "pipnn_cuda", test))]
pub(crate) fn find_max_admissible_rows(
    dim: usize,
    leaf_knn: usize,
    max_leaders_per_node: usize,
    max_fanout: usize,
    workspace_budget_bytes: Option<usize>,
    device_budget_bytes: Option<usize>,
    params: &PiPNNParams,
) -> usize {
    if dim == 0 {
        return 0;
    }
    let supported = |rows: usize| {
        plan_gpu_persistent_subtree_resident_admission(
            rows,
            dim,
            leaf_knn,
            max_leaders_per_node,
            max_fanout,
            workspace_budget_bytes,
            device_budget_bytes,
            params,
        )
        .supported
    };
    if !supported(1) {
        return 0;
    }
    let mut lo = 1usize;
    let mut hi = 2usize;
    while supported(hi) && hi < (1usize << 31) {
        lo = hi;
        hi = hi.saturating_mul(2);
        if hi == 0 {
            hi = usize::MAX;
            break;
        }
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if supported(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(not(any(feature = "pipnn_cuda", test)))]
pub(crate) fn find_max_admissible_rows(
    dim: usize,
    _leaf_knn: usize,
    _max_leaders_per_node: usize,
    _max_fanout: usize,
    _workspace_budget_bytes: Option<usize>,
    _device_budget_bytes: Option<usize>,
    _params: &PiPNNParams,
) -> usize {
    if dim == 0 {
        return 0;
    }
    0
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct D0LeaderPlan {
    pub(crate) k_capacity: usize,
    pub(crate) k_initial: usize,
    pub(crate) root_leader_cap: usize,
    pub(crate) root_leader_source: &'static str,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct D1AdmissionPreflight {
    pub(crate) supported: bool,
    pub(crate) required_bytes: usize,
    pub(crate) budget_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct D0CapacityTelemetry {
    pub(crate) enabled: bool,
    pub(crate) c_gpu: usize,
    pub(crate) c_high: usize,
    pub(crate) c_target: usize,
    pub(crate) c_low: usize,
    pub(crate) threshold_policy: &'static str,
    pub(crate) high_budget_bytes: usize,
    pub(crate) high_required_bytes: usize,
    pub(crate) target_budget_bytes: usize,
    pub(crate) target_required_bytes: usize,
    pub(crate) k_capacity: usize,
    pub(crate) quality_min_root_leaders: usize,
    pub(crate) root_leader_cap: usize,
    pub(crate) root_leader_source: &'static str,
    pub(crate) root_leaders_initial: usize,
    pub(crate) root_leaders_final: usize,
    pub(crate) oversized_child_count: usize,
    pub(crate) oversized_split_count: usize,
    pub(crate) oversized_split_rows: usize,
    pub(crate) small_child_count: usize,
    pub(crate) small_child_rows: usize,
    pub(crate) actual_child_max_unique_rows: usize,
    pub(crate) actual_child_p50_unique_rows: usize,
    pub(crate) actual_child_p95_unique_rows: usize,
    pub(crate) actual_child_p99_unique_rows: usize,
    pub(crate) admission_failures: usize,
}

impl D0CapacityTelemetry {
    pub(crate) fn record_thresholds(&mut self, thresholds: D0CapacityThresholds,
                                  dim: usize, params: &PiPNNParams) {
        self.c_gpu = thresholds.c_gpu;
        self.c_high = thresholds.c_high;
        self.c_target = thresholds.c_target;
        self.c_low = thresholds.c_low;
        self.threshold_policy = "byte-admission";
        let high = d1_admission_preflight(thresholds.c_high, dim, params,
                                         params.gpu_subtree_d0_capacity_high_ratio);
        let target = d1_admission_preflight(thresholds.c_target, dim, params,
                                           params.gpu_subtree_d0_capacity_target_ratio);
        self.high_budget_bytes = high.budget_bytes;
        self.high_required_bytes = high.required_bytes;
        self.target_budget_bytes = target.budget_bytes;
        self.target_required_bytes = target.required_bytes;
    }
}

#[cfg(any(feature = "pipnn_cuda", test))]
pub(crate) fn d1_admission_preflight(
    rows: usize,
    dim: usize,
    params: &PiPNNParams,
    device_budget_ratio: f64,
) -> D1AdmissionPreflight {
    let workspace_budget = params.effective_gpu_workspace_budget_bytes();
    let ratio = device_budget_ratio.clamp(0.01, 1.0);
    let device_budget = ((workspace_budget as f64) * ratio).floor().max(1.0) as usize;
    let plan = plan_gpu_persistent_subtree_resident_admission(
        rows,
        dim,
        params.leaf_knn,
        super::compute_backend::gpu_persistent_subtree_max_leaders_for_params(params),
        params.fanout_second.max(1),
        Some(workspace_budget),
        Some(device_budget),
        params,
    );
    D1AdmissionPreflight {
        supported: plan.supported,
        required_bytes: plan.device_admission_required_bytes,
        budget_bytes: plan.device_admission_budget_bytes,
    }
}

#[cfg(not(any(feature = "pipnn_cuda", test)))]
pub(crate) fn d1_admission_preflight(
    _rows: usize,
    _dim: usize,
    _params: &PiPNNParams,
    _device_budget_ratio: f64,
) -> D1AdmissionPreflight {
    D1AdmissionPreflight {
        supported: true,
        required_bytes: 0,
        budget_bytes: 0,
    }
}

pub(crate) fn compute_thresholds(c_gpu: usize, cfg: D0CapacityConfig) -> D0CapacityThresholds {
    if c_gpu == 0 {
        return D0CapacityThresholds {
            c_gpu: 0,
            c_high: 0,
            c_target: 0,
            c_low: 0,
        };
    }
    let bounded_high = cfg.high_ratio.clamp(0.01, 1.0);
    let bounded_target = cfg.target_ratio.clamp(0.01, bounded_high);
    let bounded_low = cfg.low_ratio.clamp(0.0, bounded_target);
    let c_high = ((c_gpu as f64) * bounded_high).floor().max(1.0) as usize;
    let c_target = ((c_gpu as f64) * bounded_target).floor().max(1.0) as usize;
    let ratio_low = ((c_gpu as f64) * bounded_low).floor().max(1.0) as usize;
    let c_low = ratio_low.max(cfg.min_efficient_rows).min(c_target);
    D0CapacityThresholds {
        c_gpu,
        c_high,
        c_target,
        c_low,
    }
}

/// Scaling a full-budget row count does not scale fixed workspace overhead.
/// Keep throughput-model limits, but also fit the actual fractional byte budget.
pub(crate) fn compute_admission_thresholds(
    c_gpu: usize, dim: usize, params: &PiPNNParams, cfg: D0CapacityConfig,
) -> D0CapacityThresholds {
    let mut thresholds = compute_thresholds(c_gpu, cfg);
    if c_gpu == 0 {
        return thresholds;
    }
    let high_ratio = cfg.high_ratio.clamp(0.01, 1.0);
    let target_ratio = cfg.target_ratio.clamp(0.01, high_ratio);
    let safe_rows = |ratio: f64| {
        let workspace = params.effective_gpu_workspace_budget_bytes();
        let budget = ((workspace as f64) * ratio).floor().max(1.0) as usize;
        find_max_admissible_rows(dim, params.leaf_knn,
            super::compute_backend::gpu_persistent_subtree_max_leaders_for_params(params),
            params.fanout_second.max(1), Some(workspace), Some(budget), params)
    };
    thresholds.c_high = thresholds.c_high.min(safe_rows(high_ratio));
    thresholds.c_target = thresholds.c_target.min(thresholds.c_high).min(safe_rows(target_ratio));
    thresholds.c_low = thresholds.c_low.min(thresholds.c_target);
    thresholds
}

pub(crate) fn plan_root_leaders(
    total_points: usize,
    root_fanout: usize,
    thresholds: D0CapacityThresholds,
    cfg: D0CapacityConfig,
) -> D0LeaderPlan {
    if total_points == 0 || root_fanout == 0 {
        return D0LeaderPlan::default();
    }
    let assignments = total_points.saturating_mul(root_fanout);
    let k_capacity = if thresholds.c_target == 0 {
        0
    } else {
        assignments.div_ceil(thresholds.c_target).max(1)
    };
    let root_leader_cap = cfg.root_leader_cap.max(1).min(total_points.max(1));
    let capacity_target = k_capacity.max(1);
    let quality_floor = cfg.quality_min_root_leaders.max(1).min(total_points.max(1));
    let desired_leaders = capacity_target.max(quality_floor);
    let root_leader_source = if desired_leaders > root_leader_cap {
        "root_leader_cap"
    } else if quality_floor > capacity_target {
        "quality_floor"
    } else {
        "capacity"
    };
    let k_initial = desired_leaders.clamp(1, root_leader_cap);
    D0LeaderPlan {
        k_capacity,
        k_initial,
        root_leader_cap,
        root_leader_source,
    }
}

pub(crate) fn split_leader_count(rows: usize, thresholds: D0CapacityThresholds) -> usize {
    if rows <= thresholds.c_high || thresholds.c_target == 0 {
        return 1;
    }
    rows.div_ceil(thresholds.c_target).max(2)
}

pub(crate) fn percentile_nearest_rank(sorted: &[usize], pct: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let rank = sorted.len().saturating_mul(pct).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

pub(crate) fn observe_child_unique_sizes(
    telemetry: &mut D0CapacityTelemetry,
    thresholds: D0CapacityThresholds,
    sizes: &mut [usize],
) {
    sizes.sort_unstable();
    telemetry.actual_child_max_unique_rows = sizes.last().copied().unwrap_or(0);
    telemetry.actual_child_p50_unique_rows = percentile_nearest_rank(sizes, 50);
    telemetry.actual_child_p95_unique_rows = percentile_nearest_rank(sizes, 95);
    telemetry.actual_child_p99_unique_rows = percentile_nearest_rank(sizes, 99);
    telemetry.oversized_child_count = sizes
        .iter()
        .filter(|&&size| size > thresholds.c_high)
        .count();
    telemetry.small_child_count = sizes
        .iter()
        .filter(|&&size| size < thresholds.c_low)
        .count();
    telemetry.small_child_rows = sizes
        .iter()
        .filter(|&&size| size < thresholds.c_low)
        .copied()
        .sum();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d0_capacity_config_default_uses_strict_ratios() {
        let cfg = D0CapacityConfig::default();
        assert_eq!(cfg.high_ratio, 0.60);
        assert_eq!(cfg.target_ratio, 0.50);
        assert_eq!(cfg.low_ratio, 0.15);
    }

    #[test]
    fn thresholds_keep_target_below_high_and_low_below_target() {
        let cfg = D0CapacityConfig {
            high_ratio: 0.85,
            target_ratio: 0.75,
            low_ratio: 0.20,
            min_efficient_rows: 32,
            ..D0CapacityConfig::default()
        };
        let thresholds = compute_thresholds(1_000, cfg);
        assert_eq!(thresholds.c_gpu, 1_000);
        assert_eq!(thresholds.c_high, 850);
        assert_eq!(thresholds.c_target, 750);
        assert_eq!(thresholds.c_low, 200);
        assert!(thresholds.c_target <= thresholds.c_high);
        assert!(thresholds.c_low <= thresholds.c_target);
    }

    #[test]
    fn thresholds_clamp_target_when_config_exceeds_high() {
        let cfg = D0CapacityConfig {
            high_ratio: 0.60,
            target_ratio: 0.90,
            low_ratio: 0.20,
            min_efficient_rows: 1,
            ..D0CapacityConfig::default()
        };
        let thresholds = compute_thresholds(1_000, cfg);
        assert_eq!(thresholds.c_high, 600);
        assert_eq!(thresholds.c_target, 600);
        assert_eq!(thresholds.c_low, 200);
    }

    #[test]
    fn root_leader_plan_lifts_configured_leaders_to_capacity_target() {
        let cfg = D0CapacityConfig {
            quality_min_root_leaders: 64,
            root_leader_cap: 10_000,
            ..D0CapacityConfig::default()
        };
        let thresholds = D0CapacityThresholds {
            c_gpu: 400_000,
            c_high: 340_000,
            c_target: 300_000,
            c_low: 64_000,
        };
        let plan = plan_root_leaders(10_000_000, 6, thresholds, cfg);
        assert_eq!(plan.k_capacity, 200);
        assert_eq!(plan.k_initial, 200);
        assert_eq!(plan.root_leader_cap, 10_000);
        assert_eq!(plan.root_leader_source, "capacity");
    }

    #[test]
    fn root_leader_plan_preserves_quality_floor_when_capacity_target_is_lower() {
        let thresholds = D0CapacityThresholds {
            c_gpu: 1_000_000,
            c_high: 850_000,
            c_target: 750_000,
            c_low: 64_000,
        };
        let floor_cfg = D0CapacityConfig {
            quality_min_root_leaders: 128,
            root_leader_cap: 10_000,
            ..D0CapacityConfig::default()
        };
        let floor_plan = plan_root_leaders(1_000_000, 6, thresholds, floor_cfg);
        assert_eq!(floor_plan.k_capacity, 8);
        assert_eq!(floor_plan.k_initial, 128);
        assert_eq!(floor_plan.root_leader_source, "quality_floor");

        let capped_cfg = D0CapacityConfig {
            quality_min_root_leaders: 160,
            root_leader_cap: 100,
            ..D0CapacityConfig::default()
        };
        let capped_plan = plan_root_leaders(20_000_000, 6, thresholds, capped_cfg);
        assert_eq!(capped_plan.k_capacity, 160);
        assert_eq!(capped_plan.k_initial, 100);
        assert_eq!(capped_plan.root_leader_cap, 100);
        assert_eq!(capped_plan.root_leader_source, "root_leader_cap");
    }

    #[test]
    fn resident_device_budget_lifts_root_plan_to_capacity_target() {
        let mut params = crate::algorithm::pipnn::PiPNNParams::default();
        params.leaf_knn = 2;
        params.c_min = 256;
        params.c_max = 1280;
        params.fanout_top = 6;
        params.fanout_second = 3;
        params.max_leaders = 5000;
        params.gpu_workspace_budget_bytes = 16 * 1024 * 1024 * 1024;

        let max_leaders_per_node =
            crate::algorithm::pipnn::compute_backend::gpu_persistent_subtree_max_leaders_for_params(
                &params,
            );
        let workspace_budget = Some(params.effective_gpu_workspace_budget_bytes());
        let workspace_only = find_max_admissible_rows(
            1024,
            params.leaf_knn,
            max_leaders_per_node,
            params.fanout_second.max(1),
            workspace_budget,
            None,
            &params,
        );
        let resident_aware = find_max_admissible_rows(
            1024,
            params.leaf_knn,
            max_leaders_per_node,
            params.fanout_second.max(1),
            workspace_budget,
            workspace_budget,
            &params,
        );

        assert!(
            resident_aware < workspace_only,
            "resident-aware admission should be tighter than workspace-only admission: workspace_only={workspace_only} resident_aware={resident_aware}"
        );

        let cfg = D0CapacityConfig::default();
        let thresholds = compute_thresholds(resident_aware, cfg);
        let plan = plan_root_leaders(35_000_000, params.fanout_top, thresholds, cfg);

        assert!(
            plan.k_capacity > cfg.quality_min_root_leaders,
            "Wiki35M capacity estimate should still expose pressure above the configured leader count: plan={plan:?} thresholds={thresholds:?}"
        );
        assert_eq!(plan.k_initial, plan.k_capacity);
        assert_eq!(plan.root_leader_source, "capacity");
    }

    #[test]
    fn sq4_100m_fractional_byte_admission_boundary() {
        let mut params = PiPNNParams::default();
        params.gpu_subtree_sq4_plus = true;
        params.gpu_subtree_sq4_primary_center = true;
        params.gpu_subtree_sq_native_int4 = true;
        params.gpu_subtree_sq_bits = 4;
        params.gpu_subtree_sq_wmma_mode = super::super::PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_workspace_budget_bytes = 16usize << 30;
        params.leaf_knn = 2;
        params.c_min = 256;
        params.c_max = 1280;
        params.fanout_top = 8;
        params.fanout_second = 3;
        params.max_leaders = 5000;
        let cfg = D0CapacityConfig { enabled: true, high_ratio: 0.08,
            target_ratio: 0.08, low_ratio: 0.08, ..D0CapacityConfig::default() };
        let failed = d1_admission_preflight(1_515_897, 512, &params, 0.08);
        assert!(!failed.supported);
        // The current counter header has 168 u64 words (the historical failure had 167).
        assert_eq!(failed.required_bytes - failed.budget_bytes, 17_334);
        let thresholds = compute_admission_thresholds(18_950_395, 512, &params, cfg);
        assert!(thresholds.c_high < 1_515_897);
        assert_eq!(thresholds.c_high, thresholds.c_target);
        assert!(d1_admission_preflight(thresholds.c_high, 512, &params, 0.08).supported);
        assert!(!d1_admission_preflight(thresholds.c_high + 1, 512, &params, 0.08).supported);
        assert_eq!(plan_root_leaders(100_000_000, 8, thresholds, cfg).k_initial, 528);
        eprintln!("100M byte-admission regression: old_failed_rows=1515897 safe_rows={} budget_bytes={}",
            thresholds.c_high, failed.budget_bytes);
    }

    #[test]
    fn sq4_planning_budget_is_independent_of_upload_threshold() {
        let mut params = PiPNNParams::default();
        params.gpu_subtree_sq4_plus = true;
        params.gpu_subtree_sq4_primary_center = true;
        params.gpu_subtree_sq_native_int4 = true;
        params.gpu_subtree_sq_bits = 4;
        params.gpu_subtree_sq_wmma_mode = super::super::PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_workspace_budget_bytes = 16usize << 30;
        params.leaf_knn = 2;
        params.c_min = 256;
        params.c_max = 1280;
        params.fanout_top = 8;
        params.fanout_second = 3;
        params.max_leaders = 5000;
        let old_cfg = D0CapacityConfig { enabled: true, high_ratio: 0.08,
            target_ratio: 0.08, low_ratio: 0.08, ..D0CapacityConfig::default() };
        let cfg = D0CapacityConfig { high_ratio: 0.50, ..old_cfg };
        let before = compute_admission_thresholds(18_950_395, 512, &params, old_cfg);
        let after = compute_admission_thresholds(18_950_395, 512, &params, cfg);

        assert_eq!(before.c_target, after.c_target);
        assert_eq!(before.c_low, after.c_low);
        assert!(after.c_high > after.c_target);
        for (rows, fanout, leaders) in [(10_000_000, 6, 64), (100_000_000, 8, 528)] {
            let a = plan_root_leaders(rows, fanout, before, old_cfg);
            let b = plan_root_leaders(rows, fanout, after, cfg);
            assert_eq!(a, b);
            assert_eq!(b.k_initial, leaders);
        }

        let above_target = after.c_target + 1;
        assert!(!d1_admission_preflight(above_target, 512, &params, 0.08).supported);
        assert!(d1_admission_preflight(above_target, 512, &params, 0.50).supported);
        assert_eq!(split_leader_count(above_target, after), 1);
        let at_limit = d1_admission_preflight(after.c_high, 512, &params, 0.50);
        let over_limit = d1_admission_preflight(after.c_high + 1, 512, &params, 0.50);
        assert_eq!(at_limit.budget_bytes, 8usize << 30);
        assert!(at_limit.supported);
        assert!(!over_limit.supported);
        assert!(over_limit.required_bytes > over_limit.budget_bytes);
        assert_eq!(split_leader_count(after.c_high + 1, after),
                   (after.c_high + 1).div_ceil(after.c_target));
        eprintln!("SQ4 separate limits: target_rows={} threshold_rows={} threshold_bytes={} leaders_100m=528",
                  after.c_target, after.c_high, at_limit.budget_bytes);
    }

    #[test]
    fn split_leader_count_is_minimum_capacity_cut() {
        let thresholds = D0CapacityThresholds {
            c_gpu: 400_000,
            c_high: 340_000,
            c_target: 300_000,
            c_low: 64_000,
        };
        assert_eq!(split_leader_count(300_000, thresholds), 1);
        assert_eq!(split_leader_count(341_000, thresholds), 2);
        assert_eq!(split_leader_count(900_001, thresholds), 4);
    }

    #[test]
    fn child_size_observation_records_tail_and_small_counts() {
        let thresholds = D0CapacityThresholds {
            c_gpu: 400_000,
            c_high: 340_000,
            c_target: 300_000,
            c_low: 64,
        };
        let mut telemetry = D0CapacityTelemetry::default();
        let mut sizes = vec![10, 100, 200, 300, 400_000];
        observe_child_unique_sizes(&mut telemetry, thresholds, &mut sizes);
        assert_eq!(telemetry.actual_child_max_unique_rows, 400_000);
        assert_eq!(telemetry.actual_child_p50_unique_rows, 200);
        assert_eq!(telemetry.actual_child_p95_unique_rows, 400_000);
        assert_eq!(telemetry.actual_child_p99_unique_rows, 400_000);
        assert_eq!(telemetry.oversized_child_count, 1);
        assert_eq!(telemetry.small_child_count, 1);
        assert_eq!(telemetry.small_child_rows, 10);
    }
}
