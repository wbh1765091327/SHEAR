use std::collections::HashSet;

use ndarray::ArrayView2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::Serialize;

use super::params::PiPNNParams;
use super::root_radius_witness::{root_leader_fingerprint, select_root_leaders};
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug, PartialEq)]
pub struct RootOpportunityCalibrationConfig {
    pub seed: u64,
    pub psamp_fraction: f64,
    pub max_leaders: usize,
    pub fanout_top: usize,
    pub knn: usize,
    pub uniform_sources: usize,
    pub boundary_sources: usize,
    pub cell_sources: usize,
    pub boundary_pool: usize,
    pub core_rs: Vec<usize>,
    pub gap_quantiles: Vec<f64>,
    pub delta: f64,
    pub threads: usize,
    pub query_batch: usize,
    pub base_block: usize,
}

impl Default for RootOpportunityCalibrationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            psamp_fraction: 0.001,
            max_leaders: 1000,
            fanout_top: 10,
            knn: 10,
            uniform_sources: 4096,
            boundary_sources: 2048,
            cell_sources: 4096,
            boundary_pool: 200_000,
            core_rs: vec![4, 6, 8],
            gap_quantiles: vec![0.50, 0.70, 0.80, 0.90, 0.95],
            delta: 0.05,
            threads: 0,
            query_batch: 0,
            base_block: 0,
        }
    }
}

impl RootOpportunityCalibrationConfig {
    pub fn validate(&self) -> AnnResult<()> {
        if !self.psamp_fraction.is_finite() || self.psamp_fraction <= 0.0 {
            return Err(config_error(
                "psamp_fraction",
                "psamp_fraction must be finite and > 0",
            ));
        }
        if self.max_leaders == 0 {
            return Err(config_error("max_leaders", "max_leaders must be > 0"));
        }
        if self.fanout_top == 0 {
            return Err(config_error("fanout_top", "fanout_top must be > 0"));
        }
        if self.knn == 0 {
            return Err(config_error("knn", "knn must be > 0"));
        }
        if self.core_rs.is_empty() || self.core_rs.iter().any(|&r| r == 0) {
            return Err(config_error(
                "core_rs",
                "core_rs must contain at least one positive integer",
            ));
        }
        if self.gap_quantiles.is_empty()
            || self
                .gap_quantiles
                .iter()
                .any(|&q| !q.is_finite() || q <= 0.0 || q >= 1.0)
        {
            return Err(config_error(
                "gap_quantiles",
                "gap_quantiles must contain values in (0, 1)",
            ));
        }
        if !self.delta.is_finite() || self.delta <= 0.0 || self.delta >= 1.0 {
            return Err(config_error("delta", "delta must be in (0, 1)"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KnnNeighbor {
    pub point_id: usize,
    pub distance_sq: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpportunityObservation {
    pub fixed_opportunity: usize,
    pub candidate_opportunity: usize,
    pub critical_miss_vs_fixed: bool,
    pub opportunity_loss_vs_fixed: f64,
    pub rank_weight: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpportunitySummary {
    pub observations: usize,
    pub critical_miss_vs_fixed: f64,
    pub critical_miss_upper_vs_fixed: f64,
    pub mean_opportunity_loss_vs_fixed: f64,
    pub mean_opportunity_loss_upper_vs_fixed: f64,
    pub weighted_opportunity_loss_vs_fixed: f64,
    pub weighted_opportunity_loss_upper_vs_fixed: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChildWorkMetrics {
    pub sum_child_size_squared_ratio: f64,
    pub largest_child_ratio: f64,
    pub child_size_cv: f64,
    pub predicted_work_score: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SourceCounts {
    pub uniform: usize,
    pub boundary: usize,
    pub cell: usize,
    pub total_unique: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FixedBaselineSummary {
    pub fanout: usize,
    pub avg_fanout: f64,
    pub kept_hist: Vec<usize>,
    pub sum_child_size_squared_ratio: f64,
    pub largest_child_ratio: f64,
    pub child_size_cv: f64,
    pub predicted_work_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateEvaluation {
    pub core_r: usize,
    pub gap_quantile: f64,
    pub gap_threshold: f32,
    pub source_count: usize,
    pub neighbor_pairs: usize,
    pub critical_miss_vs_fixed: f64,
    pub critical_miss_upper_vs_fixed: f64,
    pub mean_opportunity_loss_vs_fixed: f64,
    pub mean_opportunity_loss_upper_vs_fixed: f64,
    pub weighted_opportunity_loss_vs_fixed: f64,
    pub weighted_opportunity_loss_upper_vs_fixed: f64,
    pub kept_hist: Vec<usize>,
    pub avg_fanout: f64,
    pub sum_child_size_squared_ratio: f64,
    pub largest_child_ratio: f64,
    pub child_size_cv: f64,
    pub predicted_work_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Advisory {
    pub selected: Option<AdvisoryCandidate>,
    pub max_avg_fanout_ratio: f64,
    pub max_critical_miss_upper_vs_fixed: f64,
    pub max_mean_opportunity_loss_upper_vs_fixed: f64,
    pub max_predicted_work_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdvisoryCandidate {
    pub core_r: usize,
    pub gap_quantile: f64,
    pub gap_threshold: f32,
    pub avg_fanout: f64,
    pub predicted_work_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CalibrationReport {
    pub schema_version: u32,
    pub selection_semantics: String,
    pub dataset_len: usize,
    pub dim: usize,
    pub worker_threads: usize,
    pub root_leader_count: usize,
    pub root_leader_hash: u64,
    pub source_counts: SourceCounts,
    pub fixed_baseline: FixedBaselineSummary,
    pub candidate_evaluations: Vec<CandidateEvaluation>,
    pub advisory: Advisory,
}

#[derive(Clone, Debug)]
struct PointRootTopK {
    roots: Vec<usize>,
    distances_sq: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceKind {
    Uniform,
    Boundary,
    Cell,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SampledSource {
    point_id: usize,
    kind: SourceKind,
}

pub fn core_drop_kept_len(
    distances_sq: &[f32],
    core_r: usize,
    threshold: f32,
    fixed_fanout: usize,
) -> usize {
    let fixed = fixed_fanout.max(1).min(distances_sq.len().max(1));
    if core_r == 0 || core_r >= fixed || core_r >= distances_sq.len() {
        return fixed;
    }

    let boundary = distances_sq[core_r].max(0.0).sqrt();
    let previous = distances_sq[core_r - 1].max(0.0).sqrt();
    if boundary - previous >= threshold {
        core_r
    } else {
        fixed
    }
}

pub fn membership_intersection_count(left: &[usize], right: &[usize]) -> usize {
    left.iter()
        .filter(|&&left_root| right.contains(&left_root))
        .count()
}

pub fn opportunity_observation(
    fixed_source: &[usize],
    fixed_neighbor: &[usize],
    candidate_source: &[usize],
    candidate_neighbor: &[usize],
    neighbor_rank: usize,
) -> OpportunityObservation {
    let fixed_opportunity = membership_intersection_count(fixed_source, fixed_neighbor);
    let candidate_opportunity = membership_intersection_count(candidate_source, candidate_neighbor);
    let lost = fixed_opportunity.saturating_sub(candidate_opportunity);
    let opportunity_loss_vs_fixed = if fixed_opportunity == 0 {
        0.0
    } else {
        lost as f64 / fixed_opportunity as f64
    };

    OpportunityObservation {
        fixed_opportunity,
        candidate_opportunity,
        critical_miss_vs_fixed: fixed_opportunity > 0 && candidate_opportunity == 0,
        opportunity_loss_vs_fixed,
        rank_weight: 1.0 / (neighbor_rank + 1) as f64,
    }
}

pub fn summarize_opportunity_observations(
    observations: &[OpportunityObservation],
    delta: f64,
) -> OpportunitySummary {
    if observations.is_empty() {
        return OpportunitySummary {
            observations: 0,
            critical_miss_upper_vs_fixed: 1.0,
            mean_opportunity_loss_upper_vs_fixed: 1.0,
            weighted_opportunity_loss_upper_vs_fixed: 1.0,
            ..OpportunitySummary::default()
        };
    }

    let n = observations.len();
    let critical = observations
        .iter()
        .filter(|obs| obs.critical_miss_vs_fixed)
        .count() as f64
        / n as f64;
    let mean_loss = observations
        .iter()
        .map(|obs| obs.opportunity_loss_vs_fixed)
        .sum::<f64>()
        / n as f64;
    let weight_sum = observations.iter().map(|obs| obs.rank_weight).sum::<f64>();
    let weighted_loss = if weight_sum == 0.0 {
        0.0
    } else {
        observations
            .iter()
            .map(|obs| obs.opportunity_loss_vs_fixed * obs.rank_weight)
            .sum::<f64>()
            / weight_sum
    };

    OpportunitySummary {
        observations: n,
        critical_miss_vs_fixed: critical,
        critical_miss_upper_vs_fixed: hoeffding_upper(critical, n, delta),
        mean_opportunity_loss_vs_fixed: mean_loss,
        mean_opportunity_loss_upper_vs_fixed: hoeffding_upper(mean_loss, n, delta),
        weighted_opportunity_loss_vs_fixed: weighted_loss,
        weighted_opportunity_loss_upper_vs_fixed: hoeffding_upper(weighted_loss, n, delta),
    }
}

pub fn child_work_metrics(child_sizes: &[usize], baseline_sum_squares: f64) -> ChildWorkMetrics {
    let total: usize = child_sizes.iter().sum();
    let sum_squares = child_sizes
        .iter()
        .map(|&size| {
            let size = size as f64;
            size * size
        })
        .sum::<f64>();
    let ratio = if baseline_sum_squares > 0.0 {
        sum_squares / baseline_sum_squares
    } else {
        0.0
    };
    let largest_child_ratio = if total == 0 {
        0.0
    } else {
        child_sizes.iter().copied().max().unwrap_or(0) as f64 / total as f64
    };
    let child_size_cv = if child_sizes.is_empty() {
        0.0
    } else {
        let mean = total as f64 / child_sizes.len() as f64;
        if mean == 0.0 {
            0.0
        } else {
            let variance = child_sizes
                .iter()
                .map(|&size| {
                    let diff = size as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / child_sizes.len() as f64;
            variance.sqrt() / mean
        }
    };

    ChildWorkMetrics {
        sum_child_size_squared_ratio: ratio,
        largest_child_ratio,
        child_size_cv,
        predicted_work_score: ratio,
    }
}

pub fn exact_l2_knn(
    data: &[f32],
    dataset_len: usize,
    dim: usize,
    source_id: usize,
    knn: usize,
) -> AnnResult<Vec<KnnNeighbor>> {
    validate_data(data, dataset_len, dim)?;
    if source_id >= dataset_len {
        return Err(config_error("source_id", "source_id is out of bounds"));
    }
    if knn == 0 {
        return Ok(Vec::new());
    }

    let mut neighbors = Vec::with_capacity(knn.min(dataset_len.saturating_sub(1)));
    for point_id in 0..dataset_len {
        if point_id == source_id {
            continue;
        }
        insert_bounded_neighbor(
            &mut neighbors,
            KnnNeighbor {
                point_id,
                distance_sq: l2_distance_sq(data, dim, source_id, point_id),
            },
            knn,
        );
    }
    Ok(neighbors)
}

fn insert_bounded_neighbor(neighbors: &mut Vec<KnnNeighbor>, candidate: KnnNeighbor, knn: usize) {
    if neighbors.len() == knn
        && compare_neighbors(&candidate, neighbors.last().expect("non-empty neighbors"))
            != std::cmp::Ordering::Less
    {
        return;
    }

    let insert_at = neighbors
        .binary_search_by(|neighbor| compare_neighbors(neighbor, &candidate))
        .unwrap_or_else(|idx| idx);
    neighbors.insert(insert_at, candidate);
    neighbors.truncate(knn);
}

fn compare_neighbors(left: &KnnNeighbor, right: &KnnNeighbor) -> std::cmp::Ordering {
    left.distance_sq
        .total_cmp(&right.distance_sq)
        .then_with(|| left.point_id.cmp(&right.point_id))
}

pub fn run_root_opportunity_calibration(
    data: &[f32],
    dataset_len: usize,
    dim: usize,
    config: &RootOpportunityCalibrationConfig,
) -> AnnResult<CalibrationReport> {
    config.validate()?;
    validate_data(data, dataset_len, dim)?;

    if config.threads == 0 {
        return run_root_opportunity_calibration_in_pool(data, dataset_len, dim, config);
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .map_err(|err| {
            AnnError::log_index_error(format!("failed to build calibration thread pool: {err}"))
        })?;
    pool.install(|| run_root_opportunity_calibration_in_pool(data, dataset_len, dim, config))
}

fn run_root_opportunity_calibration_in_pool(
    data: &[f32],
    dataset_len: usize,
    dim: usize,
    config: &RootOpportunityCalibrationConfig,
) -> AnnResult<CalibrationReport> {
    if dataset_len == 0 {
        return Err(config_error(
            "data_path",
            "dataset must contain at least one point",
        ));
    }
    if dataset_len > u32::MAX as usize {
        return Err(config_error(
            "dataset_len",
            "dataset_len must fit in u32 root point identifiers",
        ));
    }

    let root_leaders = select_calibration_root_leaders(dataset_len, config);
    let root_leader_hash = root_leader_fingerprint(&root_leaders);
    let fixed_fanout = config.fanout_top.max(1).min(root_leaders.len().max(1));
    let topk_width = (fixed_fanout + 1).min(root_leaders.len().max(1));
    let root_topks = compute_root_topks(
        data,
        dataset_len,
        dim,
        &root_leaders,
        topk_width,
        config.query_batch.max(config.base_block),
    );

    let fixed_child_sizes = child_sizes_for_fixed(&root_topks, root_leaders.len(), fixed_fanout);
    let fixed_sum_squares = sum_child_size_squares(&fixed_child_sizes);
    let mut fixed_child_metrics = child_work_metrics(&fixed_child_sizes, fixed_sum_squares);
    fixed_child_metrics.sum_child_size_squared_ratio = 1.0;
    fixed_child_metrics.predicted_work_score = 1.0;

    let sampled_sources = select_sources(dataset_len, &root_topks, config);
    let source_counts = count_sources(&sampled_sources);
    let knn_by_source = compute_sample_knn(data, dataset_len, dim, &sampled_sources, config.knn)?;

    let candidate_evaluations = candidate_specs(config, &root_topks, fixed_fanout, topk_width)
        .into_par_iter()
        .map(|(core_r, gap_quantile, gap_threshold)| {
            evaluate_candidate(
                &root_topks,
                &sampled_sources,
                &knn_by_source,
                root_leaders.len(),
                fixed_fanout,
                fixed_sum_squares,
                core_r,
                gap_quantile,
                gap_threshold,
                config.delta,
            )
        })
        .collect::<Vec<_>>();

    let advisory = select_advisory_candidate(&candidate_evaluations, fixed_fanout as f64);

    Ok(CalibrationReport {
        schema_version: 1,
        selection_semantics: "relative_fixed_root_opportunity_calibration_v1".to_string(),
        dataset_len,
        dim,
        worker_threads: rayon::current_num_threads(),
        root_leader_count: root_leaders.len(),
        root_leader_hash,
        source_counts,
        fixed_baseline: FixedBaselineSummary {
            fanout: fixed_fanout,
            avg_fanout: fixed_fanout as f64,
            kept_hist: fixed_kept_hist(dataset_len, fixed_fanout),
            sum_child_size_squared_ratio: fixed_child_metrics.sum_child_size_squared_ratio,
            largest_child_ratio: fixed_child_metrics.largest_child_ratio,
            child_size_cv: fixed_child_metrics.child_size_cv,
            predicted_work_score: fixed_child_metrics.predicted_work_score,
        },
        candidate_evaluations,
        advisory,
    })
}

fn candidate_specs(
    config: &RootOpportunityCalibrationConfig,
    root_topks: &[PointRootTopK],
    fixed_fanout: usize,
    topk_width: usize,
) -> Vec<(usize, f64, f32)> {
    let mut specs = Vec::new();
    let mut core_rs = config.core_rs.clone();
    core_rs.sort_unstable();
    core_rs.dedup();

    for core_r in core_rs {
        if core_r == 0 || core_r >= fixed_fanout || core_r >= topk_width {
            continue;
        }
        let gap_values = core_gap_values(root_topks, core_r);
        for &gap_quantile in &config.gap_quantiles {
            let gap_threshold = quantile(&gap_values, gap_quantile).unwrap_or(f32::INFINITY);
            specs.push((core_r, gap_quantile, gap_threshold));
        }
    }
    specs
}

fn select_calibration_root_leaders(
    dataset_len: usize,
    config: &RootOpportunityCalibrationConfig,
) -> Vec<u32> {
    let mut params = PiPNNParams::default();
    params.psamp_fraction = config.psamp_fraction;
    params.max_leaders = config.max_leaders;
    params.fanout_top = config.fanout_top;
    params.random_seed = config.seed;

    let indices: Vec<u32> = (0..dataset_len as u32).collect();
    let mut rng = StdRng::seed_from_u64(config.seed);
    select_root_leaders(&indices, &params, &mut rng)
}

fn compute_root_topks(
    data: &[f32],
    dataset_len: usize,
    dim: usize,
    leaders: &[u32],
    topk_width: usize,
    block_rows_hint: usize,
) -> Vec<PointRootTopK> {
    if leaders.is_empty() {
        return Vec::new();
    }

    let mut leader_data = Vec::with_capacity(leaders.len() * dim);
    for &leader in leaders {
        let leader = leader as usize;
        leader_data.extend_from_slice(&data[leader * dim..(leader + 1) * dim]);
    }
    let leader_norms: Vec<f32> = leader_data
        .chunks_exact(dim)
        .map(|row| row.iter().map(|&value| value * value).sum())
        .collect();
    let block_rows = if block_rows_hint == 0 {
        1024
    } else {
        block_rows_hint
    }
    .clamp(1, 4096);
    let starts: Vec<_> = (0..dataset_len).step_by(block_rows).collect();

    starts
        .into_par_iter()
        .map(|start| {
            let end = (start + block_rows).min(dataset_len);
            compute_root_topk_block(
                data,
                dim,
                leaders,
                &leader_data,
                &leader_norms,
                topk_width,
                start,
                end,
            )
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compute_root_topk_block(
    data: &[f32],
    dim: usize,
    leaders: &[u32],
    leader_data: &[f32],
    leader_norms: &[f32],
    topk_width: usize,
    start: usize,
    end: usize,
) -> Vec<PointRootTopK> {
    let rows = end - start;
    let point_block = &data[start * dim..end * dim];
    let point_norms: Vec<f32> = point_block
        .chunks_exact(dim)
        .map(|row| row.iter().map(|&value| value * value).sum())
        .collect();
    let point_matrix = ArrayView2::from_shape((rows, dim), point_block).unwrap();
    let leader_matrix = ArrayView2::from_shape((leaders.len(), dim), leader_data).unwrap();
    let dots = point_matrix.dot(&leader_matrix.t());

    (0..rows)
        .map(|row| {
            let mut entries: Vec<_> = leaders
                .iter()
                .enumerate()
                .map(|(root, &leader_point_id)| {
                    let distance_sq =
                        (point_norms[row] + leader_norms[root] - 2.0 * dots[(row, root)]).max(0.0);
                    (root, leader_point_id as usize, distance_sq)
                })
                .collect();
            entries.sort_by(|left, right| {
                left.2
                    .total_cmp(&right.2)
                    .then_with(|| left.1.cmp(&right.1))
            });
            entries.truncate(topk_width.min(entries.len()));

            PointRootTopK {
                roots: entries.iter().map(|entry| entry.0).collect(),
                distances_sq: entries.iter().map(|entry| entry.2).collect(),
            }
        })
        .collect()
}

fn select_sources(
    dataset_len: usize,
    root_topks: &[PointRootTopK],
    config: &RootOpportunityCalibrationConfig,
) -> Vec<SampledSource> {
    let uniform = sample_ids(
        dataset_len,
        config.uniform_sources,
        config.seed ^ 0xA076_1D64_78BD_642F,
    );
    let boundary = select_boundary_sources(dataset_len, root_topks, config);
    let cell = select_cell_sources(dataset_len, root_topks, config);

    let mut seen = HashSet::new();
    let mut sampled = Vec::new();
    append_sources(&mut sampled, &mut seen, &uniform, SourceKind::Uniform);
    append_sources(&mut sampled, &mut seen, &boundary, SourceKind::Boundary);
    append_sources(&mut sampled, &mut seen, &cell, SourceKind::Cell);

    sampled
}

fn append_sources(
    sampled: &mut Vec<SampledSource>,
    seen: &mut HashSet<usize>,
    ids: &[usize],
    kind: SourceKind,
) {
    for &point_id in ids {
        if seen.insert(point_id) {
            sampled.push(SampledSource { point_id, kind });
        }
    }
}

fn sample_ids(dataset_len: usize, requested: usize, seed: u64) -> Vec<usize> {
    let take = requested.min(dataset_len);
    if take == 0 {
        return Vec::new();
    }
    if take == dataset_len {
        return (0..dataset_len).collect();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ids: Vec<_> = index::sample(&mut rng, dataset_len, take).iter().collect();
    ids.sort_unstable();
    ids
}

fn select_boundary_sources(
    dataset_len: usize,
    root_topks: &[PointRootTopK],
    config: &RootOpportunityCalibrationConfig,
) -> Vec<usize> {
    let pool = sample_ids(
        dataset_len,
        config.boundary_pool,
        config.seed ^ 0xE703_7ED1_A0B4_28DB,
    );
    let mut scored: Vec<_> = pool
        .into_iter()
        .map(|point_id| {
            let gap = top2_root_gap(root_topks.get(point_id));
            (point_id, gap)
        })
        .collect();
    scored.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(config.boundary_sources.min(scored.len()));
    scored.into_iter().map(|(point_id, _)| point_id).collect()
}

fn select_cell_sources(
    dataset_len: usize,
    root_topks: &[PointRootTopK],
    config: &RootOpportunityCalibrationConfig,
) -> Vec<usize> {
    if config.cell_sources == 0 || root_topks.is_empty() {
        return Vec::new();
    }
    let root_count = root_topks
        .iter()
        .flat_map(|topk| topk.roots.first().copied())
        .max()
        .map(|root| root + 1)
        .unwrap_or(0);
    if root_count == 0 {
        return Vec::new();
    }
    let mut buckets = vec![Vec::new(); root_count];
    for point_id in 0..dataset_len {
        if let Some(&owner) = root_topks[point_id].roots.first() {
            buckets[owner].push(point_id);
        }
    }

    let mut out = Vec::new();
    let mut offsets = vec![0usize; root_count];
    while out.len() < config.cell_sources.min(dataset_len) {
        let start_len = out.len();
        for root in 0..root_count {
            if out.len() >= config.cell_sources.min(dataset_len) {
                break;
            }
            if offsets[root] < buckets[root].len() {
                out.push(buckets[root][offsets[root]]);
                offsets[root] += 1;
            }
        }
        if out.len() == start_len {
            break;
        }
    }
    out
}

fn count_sources(sources: &[SampledSource]) -> SourceCounts {
    let mut counts = SourceCounts::default();
    for source in sources {
        match source.kind {
            SourceKind::Uniform => counts.uniform += 1,
            SourceKind::Boundary => counts.boundary += 1,
            SourceKind::Cell => counts.cell += 1,
        }
    }
    counts.total_unique = sources.len();
    counts
}

fn compute_sample_knn(
    data: &[f32],
    dataset_len: usize,
    dim: usize,
    sampled_sources: &[SampledSource],
    knn: usize,
) -> AnnResult<Vec<Vec<KnnNeighbor>>> {
    let results: Vec<_> = sampled_sources
        .par_iter()
        .map(|source| exact_l2_knn(data, dataset_len, dim, source.point_id, knn))
        .collect();
    results.into_iter().collect()
}

fn evaluate_candidate(
    root_topks: &[PointRootTopK],
    sampled_sources: &[SampledSource],
    knn_by_source: &[Vec<KnnNeighbor>],
    root_count: usize,
    fixed_fanout: usize,
    fixed_sum_squares: f64,
    core_r: usize,
    gap_quantile: f64,
    gap_threshold: f32,
    delta: f64,
) -> CandidateEvaluation {
    let (child_sizes, kept_hist, avg_fanout) =
        child_sizes_for_candidate(root_topks, root_count, fixed_fanout, core_r, gap_threshold);
    let child_metrics = child_work_metrics(&child_sizes, fixed_sum_squares);

    let mut observations = Vec::new();
    for (source_idx, source) in sampled_sources.iter().enumerate() {
        let source_fixed = membership_for_fixed(&root_topks[source.point_id], fixed_fanout);
        let source_candidate = membership_for_candidate(
            &root_topks[source.point_id],
            fixed_fanout,
            core_r,
            gap_threshold,
        );
        for (rank, neighbor) in knn_by_source[source_idx].iter().enumerate() {
            let neighbor_fixed = membership_for_fixed(&root_topks[neighbor.point_id], fixed_fanout);
            let neighbor_candidate = membership_for_candidate(
                &root_topks[neighbor.point_id],
                fixed_fanout,
                core_r,
                gap_threshold,
            );
            observations.push(opportunity_observation(
                &source_fixed,
                &neighbor_fixed,
                &source_candidate,
                &neighbor_candidate,
                rank,
            ));
        }
    }

    let summary = summarize_opportunity_observations(&observations, delta);
    CandidateEvaluation {
        core_r,
        gap_quantile,
        gap_threshold,
        source_count: sampled_sources.len(),
        neighbor_pairs: summary.observations,
        critical_miss_vs_fixed: summary.critical_miss_vs_fixed,
        critical_miss_upper_vs_fixed: summary.critical_miss_upper_vs_fixed,
        mean_opportunity_loss_vs_fixed: summary.mean_opportunity_loss_vs_fixed,
        mean_opportunity_loss_upper_vs_fixed: summary.mean_opportunity_loss_upper_vs_fixed,
        weighted_opportunity_loss_vs_fixed: summary.weighted_opportunity_loss_vs_fixed,
        weighted_opportunity_loss_upper_vs_fixed: summary.weighted_opportunity_loss_upper_vs_fixed,
        kept_hist,
        avg_fanout,
        sum_child_size_squared_ratio: child_metrics.sum_child_size_squared_ratio,
        largest_child_ratio: child_metrics.largest_child_ratio,
        child_size_cv: child_metrics.child_size_cv,
        predicted_work_score: child_metrics.predicted_work_score,
    }
}

fn fixed_kept_hist(dataset_len: usize, fixed_fanout: usize) -> Vec<usize> {
    let mut hist = vec![0; fixed_fanout + 1];
    hist[fixed_fanout] = dataset_len;
    hist
}

fn child_sizes_for_fixed(
    root_topks: &[PointRootTopK],
    root_count: usize,
    fixed_fanout: usize,
) -> Vec<usize> {
    let mut child_sizes = vec![0usize; root_count];
    for topk in root_topks {
        for &root in topk.roots.iter().take(fixed_fanout) {
            child_sizes[root] += 1;
        }
    }
    child_sizes
}

fn child_sizes_for_candidate(
    root_topks: &[PointRootTopK],
    root_count: usize,
    fixed_fanout: usize,
    core_r: usize,
    threshold: f32,
) -> (Vec<usize>, Vec<usize>, f64) {
    let mut child_sizes = vec![0usize; root_count];
    let mut kept_hist = vec![0usize; fixed_fanout + 1];
    let mut assignment_count = 0usize;
    for topk in root_topks {
        let kept = core_drop_kept_len(&topk.distances_sq, core_r, threshold, fixed_fanout)
            .min(topk.roots.len());
        if kept >= kept_hist.len() {
            kept_hist.resize(kept + 1, 0);
        }
        kept_hist[kept] += 1;
        assignment_count += kept;
        for &root in topk.roots.iter().take(kept) {
            child_sizes[root] += 1;
        }
    }
    let avg_fanout = if root_topks.is_empty() {
        0.0
    } else {
        assignment_count as f64 / root_topks.len() as f64
    };
    (child_sizes, kept_hist, avg_fanout)
}

fn membership_for_fixed(topk: &PointRootTopK, fixed_fanout: usize) -> Vec<usize> {
    topk.roots.iter().take(fixed_fanout).copied().collect()
}

fn membership_for_candidate(
    topk: &PointRootTopK,
    fixed_fanout: usize,
    core_r: usize,
    threshold: f32,
) -> Vec<usize> {
    let kept = core_drop_kept_len(&topk.distances_sq, core_r, threshold, fixed_fanout)
        .min(topk.roots.len());
    topk.roots.iter().take(kept).copied().collect()
}

fn core_gap_values(root_topks: &[PointRootTopK], core_r: usize) -> Vec<f32> {
    let mut values = Vec::new();
    for topk in root_topks {
        if core_r < topk.distances_sq.len() {
            values.push(
                topk.distances_sq[core_r].max(0.0).sqrt()
                    - topk.distances_sq[core_r - 1].max(0.0).sqrt(),
            );
        }
    }
    values
}

fn top2_root_gap(topk: Option<&PointRootTopK>) -> f32 {
    let Some(topk) = topk else {
        return f32::INFINITY;
    };
    if topk.distances_sq.len() < 2 {
        return f32::INFINITY;
    }
    topk.distances_sq[1].max(0.0).sqrt() - topk.distances_sq[0].max(0.0).sqrt()
}

fn quantile(values: &[f32], q: f64) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let idx = ((sorted.len() - 1) as f64 * q).ceil() as usize;
    sorted.get(idx.min(sorted.len() - 1)).copied()
}

fn sum_child_size_squares(child_sizes: &[usize]) -> f64 {
    child_sizes
        .iter()
        .map(|&size| {
            let size = size as f64;
            size * size
        })
        .sum()
}

fn select_advisory_candidate(evaluations: &[CandidateEvaluation], fixed_fanout: f64) -> Advisory {
    const MAX_AVG_FANOUT_RATIO: f64 = 0.90;
    const MAX_CRITICAL_MISS_UPPER: f64 = 0.005;
    const MAX_MEAN_LOSS_UPPER: f64 = 0.02;
    const MAX_PREDICTED_WORK_SCORE: f64 = 0.80;

    let selected = evaluations
        .iter()
        .filter(|evaluation| {
            evaluation.avg_fanout <= MAX_AVG_FANOUT_RATIO * fixed_fanout
                && evaluation.critical_miss_upper_vs_fixed <= MAX_CRITICAL_MISS_UPPER
                && evaluation.mean_opportunity_loss_upper_vs_fixed <= MAX_MEAN_LOSS_UPPER
                && evaluation.predicted_work_score <= MAX_PREDICTED_WORK_SCORE
        })
        .min_by(|left, right| {
            left.predicted_work_score
                .total_cmp(&right.predicted_work_score)
                .then_with(|| {
                    left.mean_opportunity_loss_upper_vs_fixed
                        .total_cmp(&right.mean_opportunity_loss_upper_vs_fixed)
                })
                .then_with(|| left.core_r.cmp(&right.core_r))
                .then_with(|| left.gap_quantile.total_cmp(&right.gap_quantile))
        })
        .map(|evaluation| AdvisoryCandidate {
            core_r: evaluation.core_r,
            gap_quantile: evaluation.gap_quantile,
            gap_threshold: evaluation.gap_threshold,
            avg_fanout: evaluation.avg_fanout,
            predicted_work_score: evaluation.predicted_work_score,
        });

    Advisory {
        selected,
        max_avg_fanout_ratio: MAX_AVG_FANOUT_RATIO,
        max_critical_miss_upper_vs_fixed: MAX_CRITICAL_MISS_UPPER,
        max_mean_opportunity_loss_upper_vs_fixed: MAX_MEAN_LOSS_UPPER,
        max_predicted_work_score: MAX_PREDICTED_WORK_SCORE,
    }
}

fn hoeffding_upper(empirical: f64, n: usize, delta: f64) -> f64 {
    if n == 0 || !delta.is_finite() || delta <= 0.0 {
        return 1.0;
    }
    (empirical + (1.0 / delta).ln().sqrt() / (2.0 * n as f64).sqrt()).min(1.0)
}

fn validate_data(data: &[f32], dataset_len: usize, dim: usize) -> AnnResult<()> {
    if dim == 0 {
        return Err(config_error("dim", "dim must be > 0"));
    }
    let expected_len = dataset_len
        .checked_mul(dim)
        .ok_or_else(|| config_error("data", "dataset size overflow"))?;
    if data.len() != expected_len {
        return Err(config_error(
            "data",
            "data length must equal dataset_len * dim",
        ));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(config_error("data", "data values must be finite"));
    }
    Ok(())
}

fn l2_distance_sq(data: &[f32], dim: usize, left_id: usize, right_id: usize) -> f32 {
    let left = &data[left_id * dim..(left_id + 1) * dim];
    let right = &data[right_id * dim..(right_id + 1) * dim];
    left.iter()
        .zip(right)
        .map(|(&left, &right)| {
            let diff = left - right;
            diff * diff
        })
        .sum()
}

fn config_error(parameter: &str, err: &str) -> AnnError {
    AnnError::log_index_config_error(parameter.to_string(), err.to_string())
}
