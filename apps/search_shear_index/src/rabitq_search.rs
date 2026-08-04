use std::cmp::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use shear_core::model::rabitq::graph_search::Rabitq1Query;

pub trait GraphAccess {
    fn num_nodes(&self) -> usize;
    fn neighbors(&self, id: usize) -> Option<&[u32]>;

    fn primary_count(&self, id: usize) -> Option<usize> {
        self.neighbors(id).map(<[u32]>::len)
    }
}

pub trait VectorAccess {
    fn dim(&self) -> usize;
    fn row(&self, id: usize) -> Option<&[f32]>;
}

pub trait OneBitEstimator {
    fn estimate_batch(&mut self, ids: &[u32], output: &mut Vec<Estimate>) -> Result<()>;
}

pub struct Rabitq1Estimator<'sidecar> {
    query: Rabitq1Query<'sidecar>,
    center_ids: &'sidecar [u8],
    bin_data: &'sidecar [u8],
}

impl<'sidecar> Rabitq1Estimator<'sidecar> {
    pub fn new(
        query: Rabitq1Query<'sidecar>,
        center_ids: &'sidecar [u8],
        bin_data: &'sidecar [u8],
    ) -> Self {
        Self {
            query,
            center_ids,
            bin_data,
        }
    }
}

impl OneBitEstimator for Rabitq1Estimator<'_> {
    fn estimate_batch(&mut self, ids: &[u32], output: &mut Vec<Estimate>) -> Result<()> {
        let estimates = self
            .query
            .estimate_ids(self.center_ids, self.bin_data, ids)?;
        output.clear();
        output.extend(estimates.into_iter().map(|estimate| Estimate {
            est_dist: estimate.est_dist,
            low_dist: estimate.low_dist,
        }));
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DistanceKey {
    distance: f32,
    id: u32,
}

impl DistanceKey {
    fn new(distance: f32, id: u32) -> Result<Self> {
        ensure!(distance.is_finite(), "non-finite distance for point {id}");
        Ok(Self { distance, id })
    }
}

impl Eq for DistanceKey {}

impl Ord for DistanceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for DistanceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
struct BeamCandidate {
    key: DistanceKey,
    checked: bool,
    first_discovery_via_backfill: bool,
}

pub struct BeamBuffer {
    entries: Vec<BeamCandidate>,
    cursor: usize,
    capacity: usize,
}

impl BeamBuffer {
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "beam capacity must be positive");
        Ok(Self {
            entries: Vec::with_capacity(capacity),
            cursor: 0,
            capacity,
        })
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
    }

    pub fn insert(
        &mut self,
        id: u32,
        distance: f32,
        first_discovery_via_backfill: bool,
    ) -> Result<bool> {
        let key = DistanceKey::new(distance, id)?;
        let insertion_index = self.entries.partition_point(|entry| entry.key < key);
        self.entries.insert(
            insertion_index,
            BeamCandidate {
                key,
                checked: false,
                first_discovery_via_backfill,
            },
        );
        self.cursor = self.cursor.min(insertion_index);
        let retained = insertion_index < self.capacity;
        if self.entries.len() > self.capacity {
            self.entries.pop();
        }
        Ok(retained)
    }

    pub fn pop_next(&mut self) -> Option<(u32, bool)> {
        while self.cursor < self.entries.len() && self.entries[self.cursor].checked {
            self.cursor += 1;
        }
        let entry = self.entries.get_mut(self.cursor)?;
        entry.checked = true;
        self.cursor += 1;
        Some((entry.key.id, entry.first_discovery_via_backfill))
    }

    pub fn retained_ids(&self) -> Vec<u32> {
        self.entries.iter().map(|entry| entry.key.id).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub est_dist: f32,
    pub low_dist: f32,
}

impl Estimate {
    pub const fn new(est_dist: f32, low_dist: f32) -> Self {
        Self { est_dist, low_dist }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub visited_nodes: usize,
    pub expanded_nodes: usize,
    pub distance_comparisons: usize,
    pub early_stopped: bool,
    pub one_bit_evaluations: u64,
    pub lower_bound_rejects: u64,
    pub navigation_exact_evaluations: u64,
    pub adaptive_exact_refinements: u64,
    pub post_rerank_exact_evaluations: u64,
    pub total_exact_evaluations: u64,
    pub query_preprocess_ns: u64,
    pub one_bit_estimate_ns: u64,
    pub exact_l2_ns: u64,
    pub navigation_ns: u64,
    pub backfill_edges_scanned: u64,
    pub backfill_first_discoveries: u64,
    pub backfill_beam_admissions: u64,
    pub backfill_nodes_expanded: u64,
    pub backfill_exact_refinements: u64,
    pub backfill_lower_bound_rejects: u64,
    pub backfill_final_top10: u64,
    pub first_backfill_admission_expansion_ordinal: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchOutcome {
    pub ids: Vec<u32>,
    pub first_discovery_via_backfill: Vec<bool>,
    pub stats: SearchStats,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StandardSearchStatsSnapshot {
    visited_nodes: usize,
    expanded_nodes: usize,
    distance_comparisons: usize,
    early_stopped: bool,
    one_bit_evaluations: u64,
    lower_bound_rejects: u64,
    navigation_exact_evaluations: u64,
    adaptive_exact_refinements: u64,
    post_rerank_exact_evaluations: u64,
    total_exact_evaluations: u64,
}

#[cfg(test)]
impl SearchStats {
    fn without_backfill_fields(&self) -> StandardSearchStatsSnapshot {
        StandardSearchStatsSnapshot {
            visited_nodes: self.visited_nodes,
            expanded_nodes: self.expanded_nodes,
            distance_comparisons: self.distance_comparisons,
            early_stopped: self.early_stopped,
            one_bit_evaluations: self.one_bit_evaluations,
            lower_bound_rejects: self.lower_bound_rejects,
            navigation_exact_evaluations: self.navigation_exact_evaluations,
            adaptive_exact_refinements: self.adaptive_exact_refinements,
            post_rerank_exact_evaluations: self.post_rerank_exact_evaluations,
            total_exact_evaluations: self.total_exact_evaluations,
        }
    }
}

pub struct RabitqSearchScratch {
    visited_epoch: Vec<u32>,
    epoch: u32,
    beam: BeamBuffer,
    neighbor_ids: Vec<u32>,
    neighbor_first_discovery_backfill: Vec<bool>,
    estimate_output: Vec<Estimate>,
    exact_candidates: Vec<DistanceKey>,
    exact_topk: ExactTopK,
    exact_capacity: usize,
}

impl RabitqSearchScratch {
    pub fn new(num_points: usize, beam_capacity: usize, exact_capacity: usize) -> Result<Self> {
        ensure!(
            num_points > 0,
            "search graph must contain at least one point"
        );
        ensure!(exact_capacity > 0, "exact result capacity must be positive");
        Ok(Self {
            visited_epoch: vec![0; num_points],
            epoch: 0,
            beam: BeamBuffer::new(beam_capacity)?,
            neighbor_ids: Vec::new(),
            neighbor_first_discovery_backfill: Vec::new(),
            estimate_output: Vec::new(),
            exact_candidates: Vec::with_capacity(beam_capacity),
            exact_topk: ExactTopK::new(exact_capacity)?,
            exact_capacity,
        })
    }

    fn reset(
        &mut self,
        num_points: usize,
        beam_capacity: usize,
        exact_capacity: usize,
    ) -> Result<()> {
        ensure!(
            self.visited_epoch.len() == num_points,
            "search scratch graph size mismatch"
        );
        ensure!(
            self.beam.capacity == beam_capacity,
            "search scratch beam capacity mismatch"
        );
        ensure!(
            self.exact_capacity == exact_capacity,
            "search scratch exact capacity mismatch"
        );
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited_epoch.fill(0);
            self.epoch = 1;
        }
        self.beam.clear();
        self.neighbor_ids.clear();
        self.neighbor_first_discovery_backfill.clear();
        self.estimate_output.clear();
        self.exact_candidates.clear();
        self.exact_topk.clear();
        Ok(())
    }

    fn mark_visited(&mut self, id: usize) -> bool {
        if id >= self.visited_epoch.len() || self.visited_epoch[id] == self.epoch {
            return false;
        }
        self.visited_epoch[id] = self.epoch;
        true
    }
}

#[derive(Clone, Copy, Debug)]
struct ExactCandidate {
    key: DistanceKey,
    first_discovery_via_backfill: bool,
}

struct ExactTopK {
    entries: Vec<ExactCandidate>,
    capacity: usize,
}

impl ExactTopK {
    fn new(capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "exact top-K capacity must be positive");
        Ok(Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        })
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn insert(&mut self, id: u32, distance: f32, first_discovery_via_backfill: bool) -> Result<()> {
        let key = DistanceKey::new(distance, id)?;
        let insertion_index = self.entries.partition_point(|entry| entry.key < key);
        self.entries.insert(
            insertion_index,
            ExactCandidate {
                key,
                first_discovery_via_backfill,
            },
        );
        if self.entries.len() > self.capacity {
            self.entries.pop();
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn dist_k(&self) -> f32 {
        if self.entries.len() < self.capacity {
            f32::INFINITY
        } else {
            self.entries
                .last()
                .expect("a full exact top-K is nonempty")
                .key
                .distance
        }
    }

    fn ids(&self) -> Vec<u32> {
        self.entries.iter().map(|entry| entry.key.id).collect()
    }

    fn provenance(&self) -> Vec<bool> {
        self.entries
            .iter()
            .map(|entry| entry.first_discovery_via_backfill)
            .collect()
    }
}

fn should_refine(topk: &ExactTopK, low_dist: f32) -> Result<bool> {
    ensure!(low_dist.is_finite(), "non-finite RaBitQ lower bound");
    Ok(topk.len() < topk.capacity() || low_dist < topk.dist_k())
}

pub fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn add_duration_ns(total: &mut u64, duration: Duration) {
    *total = total.saturating_add(duration_ns(duration));
}

fn exclusive_navigation_ns(policy_elapsed: Duration, stats: &SearchStats) -> u64 {
    duration_ns(policy_elapsed)
        .saturating_sub(stats.one_bit_estimate_ns)
        .saturating_sub(stats.exact_l2_ns)
}

fn exact_squared_l2(query: &[f32], row: &[f32]) -> Result<f32> {
    ensure!(query.len() == row.len(), "exact L2 dimension mismatch");
    let distance = query
        .iter()
        .zip(row)
        .map(|(&left, &right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f32>();
    ensure!(distance.is_finite(), "non-finite exact L2 distance");
    Ok(distance)
}

fn validate_search_inputs<G: GraphAccess, V: VectorAccess>(
    graph: &G,
    vectors: &V,
    query: &[f32],
    start: u32,
    k: usize,
    l: usize,
) -> Result<()> {
    ensure!(k > 0, "K must be positive");
    ensure!(l >= k, "search list size L must be at least K");
    ensure!(
        graph.num_nodes() > 0,
        "graph must contain at least one node"
    );
    ensure!(
        (start as usize) < graph.num_nodes(),
        "start node is outside the graph"
    );
    ensure!(query.len() == vectors.dim(), "query dimension mismatch");
    ensure!(
        query.iter().all(|value| value.is_finite()),
        "query contains a non-finite value"
    );
    Ok(())
}

pub fn search_post_rerank<G, V, E>(
    graph: &G,
    vectors: &V,
    query: &[f32],
    start: u32,
    k: usize,
    l: usize,
    estimator: &mut E,
    scratch: &mut RabitqSearchScratch,
) -> Result<SearchOutcome>
where
    G: GraphAccess,
    V: VectorAccess,
    E: OneBitEstimator,
{
    validate_search_inputs(graph, vectors, query, start, k, l)?;
    scratch.reset(graph.num_nodes(), l, k)?;
    let navigation_started = Instant::now();
    let mut stats = SearchStats::default();

    scratch.mark_visited(start as usize);
    stats.visited_nodes = 1;
    scratch.neighbor_ids.push(start);
    let estimate_started = Instant::now();
    estimator.estimate_batch(&scratch.neighbor_ids, &mut scratch.estimate_output)?;
    add_duration_ns(&mut stats.one_bit_estimate_ns, estimate_started.elapsed());
    ensure!(
        scratch.estimate_output.len() == 1,
        "1-bit estimator returned an unexpected result count"
    );
    let start_estimate = scratch.estimate_output[0];
    ensure!(
        start_estimate.est_dist.is_finite() && start_estimate.low_dist.is_finite(),
        "1-bit estimator returned a non-finite distance"
    );
    scratch.beam.insert(start, start_estimate.est_dist, false)?;
    stats.one_bit_evaluations = 1;
    stats.distance_comparisons = 1;

    while let Some((id, _)) = scratch.beam.pop_next() {
        stats.expanded_nodes += 1;
        let neighbors = graph
            .neighbors(id as usize)
            .ok_or_else(|| anyhow::anyhow!("missing graph row for point {id}"))?;
        scratch.neighbor_ids.clear();
        for &neighbor in neighbors {
            ensure!(
                (neighbor as usize) < graph.num_nodes(),
                "neighbor {neighbor} is outside the graph"
            );
            if scratch.mark_visited(neighbor as usize) {
                scratch.neighbor_ids.push(neighbor);
            }
        }
        stats.visited_nodes = stats
            .visited_nodes
            .saturating_add(scratch.neighbor_ids.len());
        if scratch.neighbor_ids.is_empty() {
            continue;
        }
        let estimate_started = Instant::now();
        estimator.estimate_batch(&scratch.neighbor_ids, &mut scratch.estimate_output)?;
        add_duration_ns(&mut stats.one_bit_estimate_ns, estimate_started.elapsed());
        ensure!(
            scratch.estimate_output.len() == scratch.neighbor_ids.len(),
            "1-bit estimator returned an unexpected result count"
        );
        stats.one_bit_evaluations = stats
            .one_bit_evaluations
            .saturating_add(scratch.neighbor_ids.len() as u64);
        stats.distance_comparisons = stats
            .distance_comparisons
            .saturating_add(scratch.neighbor_ids.len());
        for (&neighbor, &estimate) in scratch.neighbor_ids.iter().zip(&scratch.estimate_output) {
            ensure!(
                estimate.est_dist.is_finite() && estimate.low_dist.is_finite(),
                "1-bit estimator returned a non-finite distance"
            );
            scratch.beam.insert(neighbor, estimate.est_dist, false)?;
        }
    }
    let retained = scratch.beam.retained_ids();
    scratch.exact_candidates.clear();
    for id in retained {
        let row = vectors
            .row(id as usize)
            .ok_or_else(|| anyhow::anyhow!("missing base vector for point {id}"))?;
        let exact_started = Instant::now();
        let distance = exact_squared_l2(query, row)?;
        add_duration_ns(&mut stats.exact_l2_ns, exact_started.elapsed());
        scratch
            .exact_candidates
            .push(DistanceKey::new(distance, id)?);
    }
    stats.post_rerank_exact_evaluations = scratch.exact_candidates.len() as u64;
    stats.total_exact_evaluations = stats.post_rerank_exact_evaluations;
    stats.distance_comparisons = stats
        .distance_comparisons
        .saturating_add(scratch.exact_candidates.len());
    scratch.exact_candidates.sort_unstable();
    ensure!(
        scratch.exact_candidates.len() >= k,
        "search retained fewer than K candidates"
    );
    let ids: Vec<u32> = scratch.exact_candidates[..k]
        .iter()
        .map(|key| key.id)
        .collect();
    stats.navigation_ns = exclusive_navigation_ns(navigation_started.elapsed(), &stats);
    Ok(SearchOutcome {
        first_discovery_via_backfill: vec![false; ids.len()],
        ids,
        stats,
    })
}

pub fn search_adaptive_exact<G, V, E>(
    graph: &G,
    vectors: &V,
    query: &[f32],
    start: u32,
    k: usize,
    l: usize,
    estimator: &mut E,
    scratch: &mut RabitqSearchScratch,
) -> Result<SearchOutcome>
where
    G: GraphAccess,
    V: VectorAccess,
    E: OneBitEstimator,
{
    validate_search_inputs(graph, vectors, query, start, k, l)?;
    scratch.reset(graph.num_nodes(), l, k)?;
    let navigation_started = Instant::now();
    let mut stats = SearchStats::default();

    scratch.mark_visited(start as usize);
    stats.visited_nodes = 1;
    let start_row = vectors
        .row(start as usize)
        .ok_or_else(|| anyhow::anyhow!("missing base vector for start point {start}"))?;
    let exact_started = Instant::now();
    let start_distance = exact_squared_l2(query, start_row)?;
    add_duration_ns(&mut stats.exact_l2_ns, exact_started.elapsed());
    scratch.exact_topk.insert(start, start_distance, false)?;
    scratch.beam.insert(start, start_distance, false)?;
    stats.navigation_exact_evaluations = 1;
    stats.total_exact_evaluations = 1;
    stats.distance_comparisons = 1;

    while let Some((id, expanded_via_backfill)) = scratch.beam.pop_next() {
        stats.expanded_nodes += 1;
        stats.backfill_nodes_expanded = stats
            .backfill_nodes_expanded
            .saturating_add(u64::from(expanded_via_backfill));
        let neighbors = graph
            .neighbors(id as usize)
            .ok_or_else(|| anyhow::anyhow!("missing graph row for point {id}"))?;
        let primary_count = graph
            .primary_count(id as usize)
            .ok_or_else(|| anyhow::anyhow!("missing primary boundary for point {id}"))?;
        ensure!(
            primary_count <= neighbors.len(),
            "primary boundary exceeds adjacency degree for point {id}"
        );
        scratch.neighbor_ids.clear();
        scratch.neighbor_first_discovery_backfill.clear();
        for (position, &neighbor) in neighbors.iter().enumerate() {
            let via_backfill = position >= primary_count;
            stats.backfill_edges_scanned = stats
                .backfill_edges_scanned
                .saturating_add(u64::from(via_backfill));
            ensure!(
                (neighbor as usize) < graph.num_nodes(),
                "neighbor {neighbor} is outside the graph"
            );
            if scratch.mark_visited(neighbor as usize) {
                scratch.neighbor_ids.push(neighbor);
                scratch.neighbor_first_discovery_backfill.push(via_backfill);
                stats.backfill_first_discoveries = stats
                    .backfill_first_discoveries
                    .saturating_add(u64::from(via_backfill));
            }
        }
        stats.visited_nodes = stats
            .visited_nodes
            .saturating_add(scratch.neighbor_ids.len());
        if scratch.neighbor_ids.is_empty() {
            continue;
        }
        let estimate_started = Instant::now();
        estimator.estimate_batch(&scratch.neighbor_ids, &mut scratch.estimate_output)?;
        add_duration_ns(&mut stats.one_bit_estimate_ns, estimate_started.elapsed());
        ensure!(
            scratch.estimate_output.len() == scratch.neighbor_ids.len(),
            "1-bit estimator returned an unexpected result count"
        );
        stats.one_bit_evaluations = stats
            .one_bit_evaluations
            .saturating_add(scratch.neighbor_ids.len() as u64);
        stats.distance_comparisons = stats
            .distance_comparisons
            .saturating_add(scratch.neighbor_ids.len());

        for index in 0..scratch.neighbor_ids.len() {
            let neighbor = scratch.neighbor_ids[index];
            let via_backfill = scratch.neighbor_first_discovery_backfill[index];
            let estimate = scratch.estimate_output[index];
            ensure!(
                estimate.est_dist.is_finite() && estimate.low_dist.is_finite(),
                "1-bit estimator returned a non-finite distance"
            );
            if should_refine(&scratch.exact_topk, estimate.low_dist)? {
                let row = vectors
                    .row(neighbor as usize)
                    .ok_or_else(|| anyhow::anyhow!("missing base vector for point {neighbor}"))?;
                let exact_started = Instant::now();
                let exact_distance = exact_squared_l2(query, row)?;
                add_duration_ns(&mut stats.exact_l2_ns, exact_started.elapsed());
                scratch
                    .exact_topk
                    .insert(neighbor, exact_distance, via_backfill)?;
                let retained = scratch
                    .beam
                    .insert(neighbor, exact_distance, via_backfill)?;
                if via_backfill {
                    stats.backfill_exact_refinements =
                        stats.backfill_exact_refinements.saturating_add(1);
                }
                if retained && via_backfill {
                    stats.backfill_beam_admissions =
                        stats.backfill_beam_admissions.saturating_add(1);
                    stats
                        .first_backfill_admission_expansion_ordinal
                        .get_or_insert(stats.expanded_nodes as u64);
                }
                stats.navigation_exact_evaluations =
                    stats.navigation_exact_evaluations.saturating_add(1);
                stats.adaptive_exact_refinements =
                    stats.adaptive_exact_refinements.saturating_add(1);
                stats.total_exact_evaluations = stats.total_exact_evaluations.saturating_add(1);
                stats.distance_comparisons = stats.distance_comparisons.saturating_add(1);
            } else {
                let retained = scratch
                    .beam
                    .insert(neighbor, estimate.est_dist, via_backfill)?;
                stats.lower_bound_rejects = stats.lower_bound_rejects.saturating_add(1);
                if via_backfill {
                    stats.backfill_lower_bound_rejects =
                        stats.backfill_lower_bound_rejects.saturating_add(1);
                }
                if retained && via_backfill {
                    stats.backfill_beam_admissions =
                        stats.backfill_beam_admissions.saturating_add(1);
                    stats
                        .first_backfill_admission_expansion_ordinal
                        .get_or_insert(stats.expanded_nodes as u64);
                }
            }
        }
    }
    ensure!(
        scratch.exact_topk.len() >= k,
        "adaptive search found fewer than K exact candidates"
    );
    stats.navigation_ns = exclusive_navigation_ns(navigation_started.elapsed(), &stats);
    let ids = scratch.exact_topk.ids();
    let first_discovery_via_backfill = scratch.exact_topk.provenance();
    stats.backfill_final_top10 = first_discovery_via_backfill
        .iter()
        .take(10)
        .filter(|&&flag| flag)
        .count() as u64;
    Ok(SearchOutcome {
        ids,
        first_discovery_via_backfill,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    struct TestGraph {
        rows: Vec<Vec<u32>>,
        primary_counts: Vec<usize>,
    }

    impl GraphAccess for TestGraph {
        fn num_nodes(&self) -> usize {
            self.rows.len()
        }

        fn neighbors(&self, id: usize) -> Option<&[u32]> {
            self.rows.get(id).map(Vec::as_slice)
        }

        fn primary_count(&self, id: usize) -> Option<usize> {
            self.primary_counts.get(id).copied()
        }
    }

    struct TestVectors {
        dim: usize,
        values: Vec<f32>,
    }

    impl TestVectors {
        fn new(dim: usize, values: Vec<f32>) -> Self {
            Self { dim, values }
        }
    }

    impl VectorAccess for TestVectors {
        fn dim(&self) -> usize {
            self.dim
        }

        fn row(&self, id: usize) -> Option<&[f32]> {
            self.values.get(id * self.dim..(id + 1) * self.dim)
        }
    }

    struct TableEstimator(BTreeMap<u32, Estimate>);

    impl TableEstimator {
        fn new<const N: usize>(entries: [(u32, Estimate); N]) -> Self {
            Self(entries.into_iter().collect())
        }
    }

    impl OneBitEstimator for TableEstimator {
        fn estimate_batch(&mut self, ids: &[u32], output: &mut Vec<Estimate>) -> Result<()> {
            output.clear();
            for id in ids {
                output.push(*self.0.get(id).unwrap());
            }
            Ok(())
        }
    }

    fn complete_graph(nodes: usize) -> TestGraph {
        let rows: Vec<Vec<u32>> = (0..nodes)
            .map(|id| {
                (0..nodes)
                    .filter(|neighbor| *neighbor != id)
                    .map(|neighbor| neighbor as u32)
                    .collect()
            })
            .collect();
        let primary_counts = rows.iter().map(Vec::len).collect();
        TestGraph {
            rows,
            primary_counts,
        }
    }

    #[test]
    fn beam_orders_by_distance_then_global_id() {
        let mut beam = BeamBuffer::new(4).unwrap();
        beam.insert(8, 2.0, false).unwrap();
        beam.insert(9, 1.0, false).unwrap();
        beam.insert(3, 2.0, false).unwrap();
        beam.insert(4, 1.0, false).unwrap();
        assert_eq!(beam.retained_ids(), vec![4, 9, 3, 8]);
    }

    #[test]
    fn beam_truncates_to_capacity() {
        let mut beam = BeamBuffer::new(2).unwrap();
        beam.insert(1, 3.0, false).unwrap();
        beam.insert(2, 1.0, false).unwrap();
        beam.insert(3, 2.0, false).unwrap();
        assert_eq!(beam.retained_ids(), vec![2, 3]);
    }

    #[test]
    fn beam_reopens_the_cursor_for_a_better_new_candidate() {
        let mut beam = BeamBuffer::new(3).unwrap();
        beam.insert(7, 7.0, false).unwrap();
        beam.insert(8, 8.0, false).unwrap();
        assert_eq!(beam.pop_next(), Some((7, false)));
        beam.insert(2, 2.0, false).unwrap();
        assert_eq!(beam.pop_next(), Some((2, false)));
        assert_eq!(beam.pop_next(), Some((8, false)));
        assert_eq!(beam.pop_next(), None);
    }

    #[test]
    fn beam_never_returns_checked_entries_twice() {
        let mut beam = BeamBuffer::new(3).unwrap();
        beam.insert(1, 1.0, false).unwrap();
        beam.insert(2, 2.0, false).unwrap();
        assert_eq!(beam.pop_next(), Some((1, false)));
        assert_eq!(beam.pop_next(), Some((2, false)));
        assert_eq!(beam.pop_next(), None);
        assert_eq!(beam.pop_next(), None);
    }

    #[test]
    fn beam_rejects_non_finite_scores() {
        let mut beam = BeamBuffer::new(2).unwrap();
        assert!(beam.insert(1, f32::NAN, false).is_err());
        assert!(beam.insert(2, f32::INFINITY, false).is_err());
    }

    #[test]
    fn beam_retention_reports_admission_without_ordering_on_provenance() {
        let mut beam = BeamBuffer::new(2).unwrap();
        assert!(beam.insert(1, 1.0, false).unwrap());
        assert!(beam.insert(2, 2.0, true).unwrap());
        assert!(!beam.insert(3, 3.0, true).unwrap());
        assert_eq!(beam.pop_next(), Some((1, false)));
        assert_eq!(beam.pop_next(), Some((2, true)));
    }

    #[test]
    fn post_rerank_uses_no_exact_distance_until_navigation_finishes() {
        let graph = complete_graph(4);
        let vectors = TestVectors::new(2, vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let query = [2.8, 0.0];
        let mut estimator = TableEstimator::new([
            (0, Estimate::new(0.1, 0.0)),
            (1, Estimate::new(0.2, 0.1)),
            (2, Estimate::new(0.4, 0.2)),
            (3, Estimate::new(0.3, 0.2)),
        ]);
        let mut scratch = RabitqSearchScratch::new(graph.num_nodes(), 4, 2).unwrap();
        let outcome = search_post_rerank(
            &graph,
            &vectors,
            &query,
            0,
            2,
            4,
            &mut estimator,
            &mut scratch,
        )
        .unwrap();

        assert_eq!(outcome.ids, vec![3, 2]);
        assert_eq!(outcome.stats.one_bit_evaluations, 4);
        assert_eq!(outcome.stats.navigation_exact_evaluations, 0);
        assert_eq!(outcome.stats.post_rerank_exact_evaluations, 4);
        assert_eq!(outcome.stats.total_exact_evaluations, 4);
    }

    struct TracingGraph {
        rows: Vec<Vec<u32>>,
        expanded: RefCell<Vec<u32>>,
    }

    impl TracingGraph {
        fn new(rows: Vec<Vec<u32>>) -> Self {
            Self {
                rows,
                expanded: RefCell::new(Vec::new()),
            }
        }
    }

    impl GraphAccess for TracingGraph {
        fn num_nodes(&self) -> usize {
            self.rows.len()
        }

        fn neighbors(&self, id: usize) -> Option<&[u32]> {
            self.expanded.borrow_mut().push(id as u32);
            self.rows.get(id).map(Vec::as_slice)
        }

        fn primary_count(&self, id: usize) -> Option<usize> {
            self.rows.get(id).map(Vec::len)
        }
    }

    #[test]
    fn adaptive_exact_refines_strictly_and_expands_rejected_candidates() {
        let graph = TracingGraph::new(vec![vec![1, 2, 3], vec![], vec![4], vec![], vec![]]);
        let vectors = TestVectors::new(1, vec![10.0, 3.0, 99.0, 50.0_f32.sqrt(), 0.0]);
        let query = [0.0];
        let mut estimator = TableEstimator::new([
            (1, Estimate::new(20.0, 0.0)),
            (2, Estimate::new(2.0, 100.0)),
            (3, Estimate::new(1.0, 1.0)),
            (4, Estimate::new(0.0, 0.0)),
        ]);
        let mut scratch = RabitqSearchScratch::new(graph.num_nodes(), 5, 2).unwrap();

        let outcome = search_adaptive_exact(
            &graph,
            &vectors,
            &query,
            0,
            2,
            5,
            &mut estimator,
            &mut scratch,
        )
        .unwrap();

        assert_eq!(outcome.ids, vec![4, 1]);
        assert_eq!(*graph.expanded.borrow(), vec![0, 2, 4, 1, 3]);
        assert_eq!(outcome.stats.one_bit_evaluations, 4);
        assert_eq!(outcome.stats.lower_bound_rejects, 1);
        assert_eq!(outcome.stats.adaptive_exact_refinements, 3);
        assert_eq!(outcome.stats.navigation_exact_evaluations, 4);
        assert_eq!(outcome.stats.post_rerank_exact_evaluations, 0);
        assert_eq!(outcome.stats.total_exact_evaluations, 4);
    }

    #[test]
    fn adaptive_strict_boundary_rejects_low_distance_equal_to_dist_k() {
        let mut topk = ExactTopK::new(2).unwrap();
        topk.insert(1, 4.0, false).unwrap();
        topk.insert(2, 9.0, false).unwrap();
        assert!(!should_refine(&topk, 9.0).unwrap());
        assert!(should_refine(&topk, 8.999).unwrap());
    }

    fn deterministic_adaptive_fixture(with_provenance: bool) -> SearchOutcome {
        let rows = vec![vec![1, 2], vec![3], vec![3, 4], vec![], vec![]];
        let primary_counts = if with_provenance {
            vec![1, 1, 1, 0, 0]
        } else {
            rows.iter().map(Vec::len).collect()
        };
        let graph = TestGraph {
            rows,
            primary_counts,
        };
        let vectors = TestVectors::new(1, vec![10.0, 2.0, 3.0, 1.0, 0.5]);
        let mut estimator = TableEstimator::new([
            (1, Estimate::new(4.0, 0.0)),
            (2, Estimate::new(9.0, 100.0)),
            (3, Estimate::new(1.0, 0.0)),
            (4, Estimate::new(0.25, 0.0)),
        ]);
        let mut scratch = RabitqSearchScratch::new(graph.num_nodes(), 5, 2).unwrap();
        search_adaptive_exact(
            &graph,
            &vectors,
            &[0.0],
            0,
            2,
            5,
            &mut estimator,
            &mut scratch,
        )
        .unwrap()
    }

    #[test]
    fn adaptive_exact_attributes_first_discovery_without_changing_results() {
        let outcome = deterministic_adaptive_fixture(true);

        assert_eq!(outcome.ids, vec![4, 3]);
        assert_eq!(outcome.first_discovery_via_backfill, vec![true, false]);
        assert_eq!(outcome.stats.backfill_edges_scanned, 2);
        assert_eq!(outcome.stats.backfill_first_discoveries, 2);
        assert_eq!(outcome.stats.backfill_beam_admissions, 2);
        assert_eq!(outcome.stats.backfill_nodes_expanded, 2);
        assert_eq!(outcome.stats.backfill_exact_refinements, 1);
        assert_eq!(outcome.stats.backfill_lower_bound_rejects, 1);
        assert_eq!(outcome.stats.backfill_final_top10, 1);
        assert_eq!(
            outcome.stats.first_backfill_admission_expansion_ordinal,
            Some(1)
        );
    }

    #[test]
    fn provenance_disabled_graph_preserves_results_and_standard_counters() {
        let plain = deterministic_adaptive_fixture(false);
        let annotated = deterministic_adaptive_fixture(true);
        assert_eq!(plain.ids, annotated.ids);
        assert_eq!(
            plain.stats.without_backfill_fields(),
            annotated.stats.without_backfill_fields()
        );
        assert!(plain.first_discovery_via_backfill.iter().all(|flag| !flag));
    }
}
