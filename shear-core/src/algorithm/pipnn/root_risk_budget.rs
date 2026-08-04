use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RootRiskSigmaMode {
    #[default]
    CellQ90Q50,
}

impl RootRiskSigmaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CellQ90Q50 => "cell-q90-q50",
        }
    }
}

impl std::fmt::Display for RootRiskSigmaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RootRiskSigmaMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cell-q90-q50" | "cell_q90_q50" => Ok(Self::CellQ90Q50),
            other => Err(format!(
                "Unsupported root risk sigma mode '{other}', expected one of: cell-q90-q50"
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootRiskBudgetConfig {
    pub fixed_fanout: usize,
    pub min_fanout: usize,
    pub candidate_rs: Vec<usize>,
    pub target_avg_fanout: f32,
    pub warmup_points: usize,
    pub sigma_lambda: f32,
    pub budget_lambda_override: Option<f32>,
    pub sigma_mode: RootRiskSigmaMode,
}

impl Default for RootRiskBudgetConfig {
    fn default() -> Self {
        Self {
            fixed_fanout: 10,
            min_fanout: 4,
            candidate_rs: vec![4, 6, 8, 10],
            target_avg_fanout: 8.0,
            warmup_points: 500_000,
            sigma_lambda: 1.0,
            budget_lambda_override: None,
            sigma_mode: RootRiskSigmaMode::CellQ90Q50,
        }
    }
}

impl RootRiskBudgetConfig {
    pub fn normalized_candidate_rs(&self) -> Vec<usize> {
        let fixed_fanout = self.fixed_fanout.max(1).min(32);
        let min_fanout = self.min_fanout.max(1).min(fixed_fanout);
        let mut candidates: Vec<_> = self
            .candidate_rs
            .iter()
            .copied()
            .filter(|&r| r >= min_fanout && r <= fixed_fanout)
            .collect();
        candidates.push(fixed_fanout);
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            candidates.push(fixed_fanout);
        }
        candidates
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootCellStats {
    pub count: usize,
    pub q50_owner_dist: f32,
    pub q90_owner_dist: f32,
    pub q95_owner_dist: f32,
    pub sigma: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RootRiskStatsSource {
    Cell,
    Global,
    #[default]
    Unavailable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootRiskBudgetStore {
    cells: Vec<Vec<f32>>,
}

impl RootRiskBudgetStore {
    pub fn new(cell_count: usize) -> Self {
        Self {
            cells: vec![Vec::new(); cell_count],
        }
    }

    pub fn insert(&mut self, owner: usize, owner_distance: f32) {
        if !owner_distance.is_finite() {
            return;
        }
        if let Some(cell) = self.cells.get_mut(owner) {
            cell.push(owner_distance.max(0.0));
        }
    }

    pub fn insert_batch_sorted<I>(&mut self, batch: I)
    where
        I: IntoIterator<Item = (usize, f32)>,
    {
        let mut batch: Vec<_> = batch.into_iter().collect();
        batch.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
        for (owner, owner_distance) in batch {
            self.insert(owner, owner_distance);
        }
    }

    pub fn snapshot_for_config(&self, config: &RootRiskBudgetConfig) -> RootRiskBudgetSnapshot {
        RootRiskBudgetSnapshot::from_cell_samples_with_config(self.cells.clone(), config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootRiskBudgetSnapshot {
    stats: Vec<Option<RootCellStats>>,
    global_stats: Option<RootCellStats>,
    ready_cells: usize,
}

impl RootRiskBudgetSnapshot {
    pub fn from_cell_samples_with_config(
        cell_samples: Vec<Vec<f32>>,
        config: &RootRiskBudgetConfig,
    ) -> Self {
        let mut global_samples = Vec::new();
        let stats: Vec<_> = cell_samples
            .into_iter()
            .map(|samples| {
                global_samples.extend(samples.iter().copied());
                cell_stats_from_samples(samples, config)
            })
            .collect();
        let ready_cells = stats.iter().filter(|stats| stats.is_some()).count();
        let global_stats = cell_stats_from_samples(global_samples, config);
        Self {
            stats,
            global_stats,
            ready_cells,
        }
    }

    pub fn cell_stats(&self, cell: usize) -> Option<RootCellStats> {
        self.stats.get(cell).copied().flatten()
    }

    pub fn global_stats(&self) -> Option<RootCellStats> {
        self.global_stats
    }

    pub fn stats_for_cell(&self, cell: usize) -> (Option<RootCellStats>, RootRiskStatsSource) {
        if let Some(stats) = self.cell_stats(cell) {
            return (Some(stats), RootRiskStatsSource::Cell);
        }
        if let Some(stats) = self.global_stats {
            return (Some(stats), RootRiskStatsSource::Global);
        }
        (None, RootRiskStatsSource::Unavailable)
    }

    pub fn ready_cells(&self) -> usize {
        self.ready_cells
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootRiskUtilityRecord {
    utilities: Vec<f64>,
}

impl RootRiskUtilityRecord {
    pub fn from_utilities(utilities: Vec<f64>) -> Self {
        let utilities = utilities
            .into_iter()
            .map(|utility| {
                if utility.is_finite() {
                    utility.max(0.0)
                } else {
                    0.0
                }
            })
            .collect();
        Self { utilities }
    }

    pub fn utilities(&self) -> &[f64] {
        &self.utilities
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RootRiskBudgetDecision {
    pub kept: usize,
    pub fallback: bool,
    pub insufficient_stats: bool,
    pub stats_source: RootRiskStatsSource,
    pub retained_utility: f64,
    pub omitted_utility: f64,
    pub score: f64,
    pub budget_lambda: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RootRiskBudgetCalibration {
    pub budget_lambda: f32,
    pub avg_fanout: f64,
    pub points: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootRiskBudgetTelemetry {
    pub total_points: usize,
    pub fallback_points: usize,
    pub insufficient_stats_points: usize,
    pub kept_hist: Vec<usize>,
    pub selector_ms: u128,
    pub retained_utility_sum: f64,
    pub omitted_utility_sum: f64,
    child_counts: Vec<usize>,
}

impl RootRiskBudgetTelemetry {
    pub fn new(config: &RootRiskBudgetConfig, total_points: usize, cell_count: usize) -> Self {
        Self {
            total_points,
            fallback_points: 0,
            insufficient_stats_points: 0,
            kept_hist: vec![0; config.fixed_fanout.max(1).min(32) + 1],
            selector_ms: 0,
            retained_utility_sum: 0.0,
            omitted_utility_sum: 0.0,
            child_counts: vec![0; cell_count],
        }
    }

    pub fn record(&mut self, decision: &RootRiskBudgetDecision, top_leaders: &[usize]) {
        if decision.fallback {
            self.fallback_points += 1;
        }
        if decision.insufficient_stats {
            self.insufficient_stats_points += 1;
        }
        if self.kept_hist.len() <= decision.kept {
            self.kept_hist.resize(decision.kept + 1, 0);
        }
        self.kept_hist[decision.kept] += 1;
        self.retained_utility_sum += decision.retained_utility.max(0.0);
        self.omitted_utility_sum += decision.omitted_utility.max(0.0);
        for &leader in top_leaders.iter().take(decision.kept) {
            if let Some(count) = self.child_counts.get_mut(leader) {
                *count += 1;
            }
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.fallback_points += other.fallback_points;
        self.insufficient_stats_points += other.insufficient_stats_points;
        if self.kept_hist.len() < other.kept_hist.len() {
            self.kept_hist.resize(other.kept_hist.len(), 0);
        }
        for (kept, count) in other.kept_hist.into_iter().enumerate() {
            self.kept_hist[kept] += count;
        }
        self.selector_ms += other.selector_ms;
        self.retained_utility_sum += other.retained_utility_sum;
        self.omitted_utility_sum += other.omitted_utility_sum;
        if self.child_counts.len() < other.child_counts.len() {
            self.child_counts.resize(other.child_counts.len(), 0);
        }
        for (idx, count) in other.child_counts.into_iter().enumerate() {
            self.child_counts[idx] += count;
        }
    }

    pub fn omitted_utility_avg(&self) -> f64 {
        let total: usize = self.kept_hist.iter().sum();
        if total == 0 {
            0.0
        } else {
            self.omitted_utility_sum / total as f64
        }
    }

    pub fn sum_child_size_squared_ratio(&self) -> f64 {
        child_sum_squared_ratio(&self.child_counts)
    }

    pub fn largest_child_ratio(&self) -> f64 {
        child_largest_ratio(&self.child_counts)
    }

    pub fn child_size_cv(&self) -> f64 {
        child_size_cv(&self.child_counts)
    }

    pub fn predicted_work_score(&self) -> f64 {
        self.sum_child_size_squared_ratio() * root_fanout_avg(&self.kept_hist)
    }

    pub fn child_counts(&self) -> &[usize] {
        &self.child_counts
    }
}

pub fn select_risk_budgeted_fanout(
    config: &RootRiskBudgetConfig,
    snapshot: &RootRiskBudgetSnapshot,
    top_leaders: &[usize],
    top_distances_sq: &[f32],
    leader_pair_distances: &[f32],
    budget_lambda: f32,
) -> RootRiskBudgetDecision {
    let fixed_fanout = config
        .fixed_fanout
        .max(1)
        .min(32)
        .min(top_leaders.len().max(1))
        .min(top_distances_sq.len().max(1));
    if top_leaders.is_empty() || top_distances_sq.is_empty() {
        return RootRiskBudgetDecision {
            kept: fixed_fanout,
            fallback: true,
            stats_source: RootRiskStatsSource::Unavailable,
            budget_lambda,
            ..RootRiskBudgetDecision::default()
        };
    }

    let owner = top_leaders[0];
    let (stats, stats_source) = snapshot.stats_for_cell(owner);
    let Some(stats) = stats else {
        return RootRiskBudgetDecision {
            kept: fixed_fanout,
            fallback: true,
            insufficient_stats: true,
            stats_source,
            budget_lambda,
            ..RootRiskBudgetDecision::default()
        };
    };
    if stats.count < 16 || !stats.sigma.is_finite() || stats.sigma <= 0.0 {
        return RootRiskBudgetDecision {
            kept: fixed_fanout,
            fallback: true,
            insufficient_stats: true,
            stats_source,
            budget_lambda,
            ..RootRiskBudgetDecision::default()
        };
    }

    let utilities: Vec<_> = top_distances_sq
        .iter()
        .take(fixed_fanout)
        .enumerate()
        .map(|(idx, &distance_sq)| {
            if idx == 0 {
                1.0
            } else {
                let leader_distance = leader_pair_distances
                    .get(idx)
                    .copied()
                    .filter(|value| value.is_finite() && *value > 0.0);
                membership_utility_from_margin(
                    top_distances_sq[0],
                    distance_sq,
                    leader_distance,
                    stats.sigma,
                )
            }
        })
        .collect();
    let record = RootRiskUtilityRecord::from_utilities(utilities);
    let mut decision = select_budgeted_utility_fanout(config, &record, budget_lambda);
    decision.stats_source = stats_source;
    decision
}

pub fn select_budgeted_utility_fanout(
    config: &RootRiskBudgetConfig,
    record: &RootRiskUtilityRecord,
    budget_lambda: f32,
) -> RootRiskBudgetDecision {
    let fixed_fanout = config
        .fixed_fanout
        .max(1)
        .min(32)
        .min(record.utilities.len().max(1));
    let candidates = config.normalized_candidate_rs();
    let mut best_kept = fixed_fanout;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_retained = 0.0;
    let budget_lambda = if budget_lambda.is_finite() {
        budget_lambda.max(0.0)
    } else {
        0.0
    };

    for r in candidates {
        let kept = r.min(fixed_fanout).max(1);
        let retained = record.utilities.iter().take(kept).sum::<f64>();
        let score = retained - f64::from(budget_lambda) * kept as f64;
        if score > best_score + 1e-12 || ((score - best_score).abs() <= 1e-12 && kept < best_kept) {
            best_kept = kept;
            best_score = score;
            best_retained = retained;
        }
    }

    let omitted = record
        .utilities
        .iter()
        .take(fixed_fanout)
        .skip(best_kept)
        .sum::<f64>();
    RootRiskBudgetDecision {
        kept: best_kept,
        retained_utility: best_retained,
        omitted_utility: omitted,
        score: best_score,
        budget_lambda,
        ..RootRiskBudgetDecision::default()
    }
}

pub fn calibrate_budget_lambda(
    config: &RootRiskBudgetConfig,
    records: &[RootRiskUtilityRecord],
) -> RootRiskBudgetCalibration {
    if let Some(lambda) = config.budget_lambda_override {
        let budget_lambda = lambda.max(0.0);
        return RootRiskBudgetCalibration {
            budget_lambda,
            avg_fanout: avg_selected_fanout(config, records, budget_lambda),
            points: records.len(),
        };
    }
    if records.is_empty() {
        return RootRiskBudgetCalibration {
            budget_lambda: 0.0,
            avg_fanout: 0.0,
            points: 0,
        };
    }

    let target = f64::from(config.target_avg_fanout);
    let mut lo = 0.0f32;
    let mut hi = records
        .iter()
        .flat_map(|record| record.utilities.iter().copied())
        .filter(|value| value.is_finite())
        .fold(1.0f64, f64::max) as f32;
    while avg_selected_fanout(config, records, hi) > target && hi < 1.0e6 {
        hi *= 2.0;
    }

    for _ in 0..48 {
        let mid = lo + (hi - lo) * 0.5;
        let avg = avg_selected_fanout(config, records, mid);
        if avg > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let lo_avg = avg_selected_fanout(config, records, lo);
    let hi_avg = avg_selected_fanout(config, records, hi);
    let (budget_lambda, avg_fanout) = if (lo_avg - target).abs() <= (hi_avg - target).abs() {
        (lo, lo_avg)
    } else {
        (hi, hi_avg)
    };
    RootRiskBudgetCalibration {
        budget_lambda,
        avg_fanout,
        points: records.len(),
    }
}

pub fn avg_selected_fanout(
    config: &RootRiskBudgetConfig,
    records: &[RootRiskUtilityRecord],
    budget_lambda: f32,
) -> f64 {
    if records.is_empty() {
        return 0.0;
    }
    let total = records
        .iter()
        .map(|record| select_budgeted_utility_fanout(config, record, budget_lambda).kept)
        .sum::<usize>();
    total as f64 / records.len() as f64
}

pub fn utility_record_from_top_distances(
    config: &RootRiskBudgetConfig,
    snapshot: &RootRiskBudgetSnapshot,
    top_leaders: &[usize],
    top_distances_sq: &[f32],
    leader_pair_distances: &[f32],
) -> Option<(RootRiskUtilityRecord, RootRiskStatsSource)> {
    if top_leaders.is_empty() || top_distances_sq.is_empty() {
        return None;
    }
    let (stats, source) = snapshot.stats_for_cell(top_leaders[0]);
    let stats = stats?;
    if stats.count < 16 || !stats.sigma.is_finite() || stats.sigma <= 0.0 {
        return None;
    }
    let fixed_fanout = config
        .fixed_fanout
        .max(1)
        .min(32)
        .min(top_distances_sq.len());
    let utilities = top_distances_sq
        .iter()
        .take(fixed_fanout)
        .enumerate()
        .map(|(idx, &distance_sq)| {
            if idx == 0 {
                1.0
            } else {
                let leader_distance = leader_pair_distances
                    .get(idx)
                    .copied()
                    .filter(|value| value.is_finite() && *value > 0.0);
                membership_utility_from_margin(
                    top_distances_sq[0],
                    distance_sq,
                    leader_distance,
                    stats.sigma,
                )
            }
        })
        .collect();
    Some((RootRiskUtilityRecord::from_utilities(utilities), source))
}

pub fn membership_utility_from_margin(
    nearest_sq: f32,
    candidate_sq: f32,
    leader_pair_distance: Option<f32>,
    sigma: f32,
) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 {
        return 0.0;
    }
    let nearest_sq = nearest_sq.max(0.0);
    let candidate_sq = candidate_sq.max(0.0);
    let delta = (candidate_sq - nearest_sq).max(0.0) as f64;
    let denominator = leader_pair_distance
        .filter(|distance| distance.is_finite() && *distance > 0.0)
        .map(|distance| 2.0 * f64::from(distance))
        .unwrap_or_else(|| {
            let nearest = f64::from(nearest_sq.sqrt());
            let candidate = f64::from(candidate_sq.sqrt());
            2.0 * (nearest + candidate)
        });
    let margin = if denominator > 0.0 {
        delta / denominator
    } else {
        0.0
    };
    let sigma = f64::from(sigma);
    (-(margin * margin) / (2.0 * sigma * sigma)).exp()
}

fn cell_stats_from_samples(
    samples: Vec<f32>,
    config: &RootRiskBudgetConfig,
) -> Option<RootCellStats> {
    let mut samples: Vec<_> = samples
        .into_iter()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect();
    if samples.len() < 16 {
        return None;
    }
    samples.sort_by(f32::total_cmp);
    let q50 = quantile_sorted(&samples, 0.50)?;
    let q90 = quantile_sorted(&samples, 0.90)?;
    let q95 = quantile_sorted(&samples, 0.95)?;
    let spread = match config.sigma_mode {
        RootRiskSigmaMode::CellQ90Q50 => (q90 - q50).max(1.0e-6 * q50.max(1.0)),
    };
    let sigma = config.sigma_lambda * spread;
    if !sigma.is_finite() || sigma <= 0.0 {
        return None;
    }
    Some(RootCellStats {
        count: samples.len(),
        q50_owner_dist: q50,
        q90_owner_dist: q90,
        q95_owner_dist: q95,
        sigma,
    })
}

fn quantile_sorted(sorted: &[f32], q: f32) -> Option<f32> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return sorted.first().copied();
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted.get(lo).copied();
    }
    let weight = pos - lo as f32;
    Some(sorted[lo] * (1.0 - weight) + sorted[hi] * weight)
}

fn root_fanout_avg(hist: &[usize]) -> f64 {
    let total: usize = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let weighted: usize = hist
        .iter()
        .enumerate()
        .map(|(fanout, count)| fanout * count)
        .sum();
    weighted as f64 / total as f64
}

fn child_sum_squared_ratio(child_counts: &[usize]) -> f64 {
    let total = child_counts.iter().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    let sum_sq = child_counts
        .iter()
        .map(|&count| {
            let count = count as f64;
            count * count
        })
        .sum::<f64>();
    sum_sq / (total as f64 * total as f64)
}

fn child_largest_ratio(child_counts: &[usize]) -> f64 {
    let total = child_counts.iter().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    child_counts.iter().copied().max().unwrap_or(0) as f64 / total as f64
}

fn child_size_cv(child_counts: &[usize]) -> f64 {
    if child_counts.is_empty() {
        return 0.0;
    }
    let total = child_counts.iter().sum::<usize>() as f64;
    if total == 0.0 {
        return 0.0;
    }
    let mean = total / child_counts.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = child_counts
        .iter()
        .map(|&count| {
            let delta = count as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / child_counts.len() as f64;
    variance.sqrt() / mean
}
