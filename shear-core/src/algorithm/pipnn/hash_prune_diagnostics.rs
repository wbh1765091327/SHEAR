use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[cfg(test)]
use super::hash_prune::Slot;
use super::hash_prune::{FinalizedHashPruneRow, HashPruneReservoir};
use super::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use super::params::{PiPNNHashPruneBackfillMode, PiPNNParams};
use crate::common::{AnnError, AnnResult};
use crate::model::hashprune_provenance::{
    HashPruneGraphEncoding, HashPrunePrimaryCountSidecar, HashPrunePrimaryCountWriter,
};

pub(crate) const DIAGNOSTIC_SAMPLE_SEED: u64 = 42;
pub(crate) const DIAGNOSTIC_SAMPLE_DENOMINATOR: u64 = 4096;
pub(crate) const DIAGNOSTIC_TRACE_MEMORY_CEILING: u64 = 2 * 1024 * 1024 * 1024;
const TRACE_SHARDS: usize = 64;
const TRACE_CHUNK_CANDIDATES: usize = 256;
const TRACE_COLLECTOR_BASE_ACCOUNTED_BYTES: u64 = 1024 * 1024;
const TRACE_ROW_ACCOUNTED_BYTES: u64 = 1024;
const TRACE_CHUNK_ALLOCATOR_HEADROOM: u64 = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiagnosticCandidate {
    pub(crate) target: u32,
    pub(crate) hash: u16,
    pub(crate) squared_distance: f32,
}

impl DiagnosticCandidate {
    pub(crate) const fn new(target: u32, hash: u16, squared_distance: f32) -> Self {
        Self {
            target,
            hash,
            squared_distance,
        }
    }
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub(crate) fn source_is_sampled(source: u64) -> bool {
    splitmix64(source ^ DIAGNOSTIC_SAMPLE_SEED).is_multiple_of(DIAGNOSTIC_SAMPLE_DENOMINATOR)
}

#[derive(Default)]
struct TraceRow {
    chunks: Vec<Vec<DiagnosticCandidate>>,
}

impl TraceRow {
    fn charge(&self) -> u64 {
        TRACE_ROW_ACCOUNTED_BYTES + self.chunks.len() as u64 * trace_chunk_accounted_bytes()
    }
}

pub(crate) struct HashPruneTraceCollector {
    shards: Vec<Mutex<HashMap<usize, TraceRow>>>,
    accounted_bytes: AtomicU64,
    memory_ceiling: u64,
}

impl HashPruneTraceCollector {
    pub(crate) fn new() -> Self {
        Self::with_memory_ceiling(DIAGNOSTIC_TRACE_MEMORY_CEILING)
    }

    fn with_memory_ceiling(memory_ceiling: u64) -> Self {
        assert!(memory_ceiling >= TRACE_COLLECTOR_BASE_ACCOUNTED_BYTES);
        Self {
            shards: (0..TRACE_SHARDS)
                .map(|_| Mutex::new(HashMap::new()))
                .collect(),
            accounted_bytes: AtomicU64::new(TRACE_COLLECTOR_BASE_ACCOUNTED_BYTES),
            memory_ceiling,
        }
    }

    #[cfg(test)]
    fn with_memory_ceiling_for_test(memory_ceiling: u64) -> Self {
        Self::with_memory_ceiling(memory_ceiling)
    }

    pub(crate) fn record(
        &self,
        source: usize,
        target: usize,
        hash: u16,
        squared_distance: f32,
    ) -> AnnResult<()> {
        if !source_is_sampled(source as u64) {
            return Ok(());
        }
        if !squared_distance.is_finite() || squared_distance < 0.0 {
            return Err(diagnostic_error(format!(
                "invalid sampled squared distance for source {source}"
            )));
        }
        let target = u32::try_from(target)?;
        let shard_index = splitmix64(source as u64) as usize % self.shards.len();
        let mut shard = self.shards[shard_index].lock();
        if let Entry::Vacant(entry) = shard.entry(source) {
            self.reserve(TRACE_ROW_ACCOUNTED_BYTES)?;
            entry.insert(TraceRow::default());
        }
        let row = shard.get_mut(&source).expect("trace row inserted above");
        let needs_chunk = row
            .chunks
            .last()
            .is_none_or(|chunk| chunk.len() == TRACE_CHUNK_CANDIDATES);
        if needs_chunk {
            if let Err(error) = self.reserve(trace_chunk_accounted_bytes()) {
                if row.chunks.is_empty() {
                    let removed = shard
                        .remove(&source)
                        .expect("new trace row remains present");
                    self.release(removed.charge());
                }
                return Err(error);
            }
            row.chunks.push(Vec::with_capacity(TRACE_CHUNK_CANDIDATES));
        }
        row.chunks
            .last_mut()
            .expect("trace chunk inserted above")
            .push(DiagnosticCandidate::new(target, hash, squared_distance));
        Ok(())
    }

    pub(crate) fn take(&self, source: usize) -> AnnResult<Vec<DiagnosticCandidate>> {
        if !source_is_sampled(source as u64) {
            return Ok(Vec::new());
        }
        let shard_index = splitmix64(source as u64) as usize % self.shards.len();
        let row = self.shards[shard_index].lock().remove(&source);
        let Some(row) = row else {
            return Ok(Vec::new());
        };
        self.release(row.charge());
        let candidate_count = row.chunks.iter().map(Vec::len).sum();
        let mut candidates = Vec::with_capacity(candidate_count);
        for chunk in row.chunks {
            candidates.extend(chunk);
        }
        Ok(candidates)
    }

    pub(crate) fn accounted_bytes(&self) -> u64 {
        self.accounted_bytes.load(Ordering::Acquire)
    }

    pub(crate) fn live_trace_bytes(&self) -> u64 {
        self.accounted_bytes()
            .saturating_sub(TRACE_COLLECTOR_BASE_ACCOUNTED_BYTES)
    }

    fn reserve(&self, bytes: u64) -> AnnResult<()> {
        let mut current = self.accounted_bytes.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .ok_or_else(|| diagnostic_error("trace memory accounting overflow"))?;
            if next > self.memory_ceiling {
                return Err(diagnostic_error(format!(
                    "HashPrune diagnostic trace memory ceiling exceeded: requested={bytes} accounted={current} ceiling={} (2-GiB trace memory ceiling in production)",
                    self.memory_ceiling
                )));
            }
            match self.accounted_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, bytes: u64) {
        let previous = self.accounted_bytes.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(previous >= bytes + TRACE_COLLECTOR_BASE_ACCOUNTED_BYTES);
    }
}

impl Default for HashPruneTraceCollector {
    fn default() -> Self {
        Self::new()
    }
}

const fn trace_chunk_accounted_bytes() -> u64 {
    (TRACE_CHUNK_CANDIDATES * size_of::<DiagnosticCandidate>()) as u64
        + TRACE_CHUNK_ALLOCATOR_HEADROOM
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum DiagnosticPolicy {
    DistanceFirst,
    SoftTier { alpha: f32 },
    HardTier,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiagnosticReplayRow {
    pub(crate) primary_slots: Vec<(u32, u16, u32)>,
    pub(crate) backfill_slots: Vec<(u32, u16, u32, u8)>,
    pub(crate) ordered_slots: Vec<(u32, u16, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProductionRowSnapshot {
    pub(crate) primary_slots: Vec<(u32, u16, u32)>,
    pub(crate) ordered_slots: Vec<(u32, u16, u32)>,
}

impl ProductionRowSnapshot {
    #[cfg(test)]
    fn from_replay(row: &DiagnosticReplayRow) -> Self {
        Self {
            primary_slots: row.primary_slots.clone(),
            ordered_slots: row.ordered_slots.clone(),
        }
    }

    pub(crate) fn from_finalized(row: &FinalizedHashPruneRow) -> Self {
        let ordered_slots = row
            .slots
            .iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect::<Vec<_>>();
        Self {
            primary_slots: ordered_slots[..row.primary_count].to_vec(),
            ordered_slots,
        }
    }
}

pub(crate) struct DiagnosticReplayReport {
    pub(crate) distance_first: DiagnosticReplayRow,
    pub(crate) soft_100: DiagnosticReplayRow,
    pub(crate) soft_102: DiagnosticReplayRow,
    pub(crate) soft_105: DiagnosticReplayRow,
    pub(crate) soft_110: DiagnosticReplayRow,
    pub(crate) soft_120: DiagnosticReplayRow,
    pub(crate) soft_140: DiagnosticReplayRow,
    pub(crate) hard_tier: DiagnosticReplayRow,
}

impl DiagnosticReplayReport {
    pub(crate) fn all_replays(&self) -> [&DiagnosticReplayRow; 8] {
        [
            &self.distance_first,
            &self.soft_100,
            &self.soft_102,
            &self.soft_105,
            &self.soft_110,
            &self.soft_120,
            &self.soft_140,
            &self.hard_tier,
        ]
    }
}

pub(crate) fn replay_trace(
    l_max: usize,
    trace: &[DiagnosticCandidate],
    policy: DiagnosticPolicy,
) -> AnnResult<DiagnosticReplayRow> {
    replay_trace_with_row_cap(l_max, trace, policy, None)
}

fn replay_trace_with_row_cap(
    l_max: usize,
    trace: &[DiagnosticCandidate],
    policy: DiagnosticPolicy,
    row_cap: Option<usize>,
) -> AnnResult<DiagnosticReplayRow> {
    if l_max == 0 {
        return Err(diagnostic_error("diagnostic replay requires l_max > 0"));
    }
    let policy = replay_policy_with_row_cap(l_max, policy, row_cap)?;
    let mut reservoir = HashPruneReservoir::new(l_max);
    let mut profile = HashPruneBackfillProfile::default();
    for candidate in trace {
        reservoir.insert_backfill_prevalidated_with_policy(
            0,
            candidate.target,
            candidate.hash,
            candidate.squared_distance,
            policy,
            &mut profile,
        )?;
    }
    let prefinal = reservoir.diagnostic_prefinal_snapshot()?;
    let finalized = reservoir.into_finalized_slots_with_backfill(0, policy, &mut profile)?;
    if finalized.primary_count != prefinal.primary_count {
        return Err(diagnostic_error(
            "diagnostic replay primary boundary changed during finalization",
        ));
    }
    let ordered_slots = finalized
        .slots
        .iter()
        .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
        .collect::<Vec<_>>();
    let primary_slots = ordered_slots[..finalized.primary_count].to_vec();
    let layers = prefinal.slots[prefinal.primary_count..]
        .iter()
        .map(|&(target, hash, distance, layer)| ((target, hash, distance), layer))
        .collect::<HashMap<_, _>>();
    let backfill_slots = ordered_slots[finalized.primary_count..]
        .iter()
        .map(|&(target, hash, distance)| {
            let layer = layers
                .get(&(target, hash, distance))
                .copied()
                .ok_or_else(|| {
                    diagnostic_error(format!(
                        "diagnostic replay lost hit layer for target={target} hash={hash}"
                    ))
                })?;
            if !(1..=3).contains(&layer) {
                return Err(diagnostic_error(format!(
                    "diagnostic replay produced invalid hit layer {layer}"
                )));
            }
            Ok((target, hash, distance, layer))
        })
        .collect::<AnnResult<Vec<_>>>()?;
    Ok(DiagnosticReplayRow {
        primary_slots,
        backfill_slots,
        ordered_slots,
    })
}

pub(crate) fn replay_all_policies(
    l_max: usize,
    trace: &[DiagnosticCandidate],
    actual: &ProductionRowSnapshot,
) -> AnnResult<DiagnosticReplayReport> {
    let production_policy = replay_policy(l_max, DiagnosticPolicy::SoftTier { alpha: 1.10 })?;
    replay_all_policies_with_policy(l_max, trace, actual, production_policy)
}

fn replay_all_policies_with_policy(
    l_max: usize,
    trace: &[DiagnosticCandidate],
    actual: &ProductionRowSnapshot,
    production_policy: HashPruneBackfillPolicy,
) -> AnnResult<DiagnosticReplayReport> {
    let row_cap = production_policy.requested_max_backfill_per_row();
    let report = DiagnosticReplayReport {
        distance_first: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::DistanceFirst,
            row_cap,
        )?,
        soft_100: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::SoftTier { alpha: 1.00 },
            row_cap,
        )?,
        soft_102: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::SoftTier { alpha: 1.02 },
            row_cap,
        )?,
        soft_105: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::SoftTier { alpha: 1.05 },
            row_cap,
        )?,
        soft_110: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::SoftTier { alpha: 1.10 },
            row_cap,
        )?,
        soft_120: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::SoftTier { alpha: 1.20 },
            row_cap,
        )?,
        soft_140: replay_trace_with_row_cap(
            l_max,
            trace,
            DiagnosticPolicy::SoftTier { alpha: 1.40 },
            row_cap,
        )?,
        hard_tier: replay_trace_with_row_cap(l_max, trace, DiagnosticPolicy::HardTier, row_cap)?,
    };
    if report.soft_110.ordered_slots != actual.ordered_slots {
        return Err(diagnostic_error(
            "diagnostic alpha 1.10 replay differs from the production SoftTier row",
        ));
    }
    if report.soft_100.ordered_slots != report.distance_first.ordered_slots {
        return Err(diagnostic_error(
            "diagnostic alpha 1.00 replay differs from DistanceFirst",
        ));
    }
    for replay in report.all_replays() {
        if replay.primary_slots != actual.primary_slots {
            return Err(diagnostic_error(
                "diagnostic replay changed the policy-independent primary prefix",
            ));
        }
    }
    Ok(report)
}

fn replay_policy(l_max: usize, policy: DiagnosticPolicy) -> AnnResult<HashPruneBackfillPolicy> {
    replay_policy_with_row_cap(l_max, policy, None)
}

fn replay_policy_with_row_cap(
    l_max: usize,
    policy: DiagnosticPolicy,
    row_cap: Option<usize>,
) -> AnnResult<HashPruneBackfillPolicy> {
    let (mode, priority_alpha) = match policy {
        DiagnosticPolicy::DistanceFirst => (PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3, None),
        DiagnosticPolicy::SoftTier { alpha } => {
            (PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, Some(alpha))
        }
        DiagnosticPolicy::HardTier => (PiPNNHashPruneBackfillMode::RHalfHardTierQ3, None),
    };
    let params = PiPNNParams {
        l_max,
        hashprune_backfill_mode: mode,
        hashprune_rhalf_max_backfill_per_row: row_cap,
        hashprune_soft_tier_priority_alpha: priority_alpha,
        ..PiPNNParams::default()
    };
    HashPruneBackfillPolicy::from_params(&params)
}

fn diagnostic_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(message.into())
}

const RHO_UPPER_BOUNDS: [f64; 9] = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.21,
    1.4641,
    2.00,
    4.00,
    f64::INFINITY,
];
const RELATIVE_MARGIN_UPPER_BOUNDS: [f64; 11] = [
    0.0,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    5e-2,
    1e-1,
    2.5e-1,
    5e-1,
    1.0,
    f64::INFINITY,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SelectionClass {
    Selected,
    Rejected,
    LeftOnly,
    RightOnly,
    BoundarySelected,
    BoundaryRejected,
}

#[derive(Clone, Debug, Default)]
struct DistanceLayerHistograms {
    rho_bins: BTreeMap<(SelectionClass, u8), [u64; RHO_UPPER_BOUNDS.len()]>,
    relative_margin_bins: BTreeMap<String, [u64; RELATIVE_MARGIN_UPPER_BOUNDS.len()]>,
    zero_dmax_rows: u64,
    boundary_not_applicable: u64,
}

#[derive(Clone, Copy, Debug)]
struct EligibleCandidate {
    target: u32,
    hash: u16,
    squared_distance: f32,
    primary_squared_distance: f32,
    hit_layer: u8,
}

impl EligibleCandidate {
    fn key(self) -> (u32, u16, u32) {
        (self.target, self.hash, self.squared_distance.to_bits())
    }
}

impl DistanceLayerHistograms {
    fn record(
        &mut self,
        primary_distance: f64,
        candidate_distance: f64,
        hit_layer: u8,
        class: SelectionClass,
    ) -> AnnResult<()> {
        if !primary_distance.is_finite()
            || primary_distance < 0.0
            || !candidate_distance.is_finite()
            || candidate_distance < 0.0
            || !(1..=3).contains(&hit_layer)
        {
            return Err(diagnostic_error(
                "distance histogram requires finite nonnegative squared distances and layer 1..=3",
            ));
        }
        if primary_distance == 0.0 {
            self.zero_dmax_rows += 1;
            return Ok(());
        }
        let rho = candidate_distance / primary_distance;
        let bin = RHO_UPPER_BOUNDS
            .iter()
            .position(|&upper| rho <= upper)
            .expect("infinite terminal rho bin");
        self.rho_bins.entry((class, hit_layer)).or_insert([0; 9])[bin] += 1;
        Ok(())
    }

    fn record_margin(&mut self, policy: &str, margin: f64) -> AnnResult<()> {
        if !margin.is_finite() || margin < -1e-12 {
            return Err(diagnostic_error(format!(
                "invalid diagnostic priority boundary margin for {policy}: {margin}"
            )));
        }
        let margin = margin.max(0.0);
        let bin = RELATIVE_MARGIN_UPPER_BOUNDS
            .iter()
            .position(|&upper| margin <= upper)
            .expect("infinite terminal margin bin");
        self.relative_margin_bins
            .entry(policy.to_string())
            .or_insert([0; RELATIVE_MARGIN_UPPER_BOUNDS.len()])[bin] += 1;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct PolicyPairAccumulator {
    rows_compared: u64,
    rows_different_selected_sets: u64,
    rows_different_ordered_suffixes: u64,
    edge_swaps: u64,
    intersection_edges: u64,
    union_edges: u64,
    macro_jaccard_sum: f64,
    selected_by_left_layer: [u64; 3],
    selected_by_right_layer: [u64; 3],
    shared_layer_matrix: [[u64; 3]; 3],
}

impl PolicyPairAccumulator {
    fn record(&mut self, left: &DiagnosticReplayRow, right: &DiagnosticReplayRow) {
        let left_set = left
            .backfill_slots
            .iter()
            .map(|&(target, hash, distance, _)| (target, hash, distance))
            .collect::<BTreeSet<_>>();
        let right_set = right
            .backfill_slots
            .iter()
            .map(|&(target, hash, distance, _)| (target, hash, distance))
            .collect::<BTreeSet<_>>();
        let intersection = left_set.intersection(&right_set).count() as u64;
        let union = left_set.union(&right_set).count() as u64;
        let left_only = left_set.difference(&right_set).count() as u64;
        let right_only = right_set.difference(&left_set).count() as u64;
        self.rows_compared += 1;
        self.rows_different_selected_sets += u64::from(left_set != right_set);
        self.rows_different_ordered_suffixes +=
            u64::from(left.backfill_slots != right.backfill_slots);
        self.edge_swaps += left_only.max(right_only);
        self.intersection_edges += intersection;
        self.union_edges += union;
        self.macro_jaccard_sum += if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        };
        for &(_, _, _, layer) in &left.backfill_slots {
            self.selected_by_left_layer[usize::from(layer - 1)] += 1;
        }
        for &(_, _, _, layer) in &right.backfill_slots {
            self.selected_by_right_layer[usize::from(layer - 1)] += 1;
        }
        let right_layers = right
            .backfill_slots
            .iter()
            .map(|&(target, hash, distance, layer)| ((target, hash, distance), layer))
            .collect::<BTreeMap<_, _>>();
        for &(target, hash, distance, left_layer) in &left.backfill_slots {
            if let Some(&right_layer) = right_layers.get(&(target, hash, distance)) {
                self.shared_layer_matrix[usize::from(left_layer - 1)]
                    [usize::from(right_layer - 1)] += 1;
            }
        }
    }
}

fn canonical_eligible_universe(
    trace: &[DiagnosticCandidate],
    primary_slots: &[(u32, u16, u32)],
) -> (Vec<EligibleCandidate>, u64) {
    let primary_targets = primary_slots
        .iter()
        .map(|&(target, _, _)| target)
        .collect::<BTreeSet<_>>();
    let primary_by_hash = primary_slots
        .iter()
        .map(|&(_, hash, distance)| (hash, f32::from_bits(distance)))
        .collect::<BTreeMap<_, _>>();
    let mut best = HashMap::<(u16, u32), f32>::new();
    for candidate in trace {
        if primary_targets.contains(&candidate.target)
            || !primary_by_hash.contains_key(&candidate.hash)
        {
            continue;
        }
        best.entry((candidate.hash, candidate.target))
            .and_modify(|distance| {
                if candidate.squared_distance.total_cmp(distance).is_lt() {
                    *distance = candidate.squared_distance;
                }
            })
            .or_insert(candidate.squared_distance);
    }
    let mut groups = BTreeMap::<u16, Vec<(u32, f32)>>::new();
    for ((hash, target), distance) in best {
        groups.entry(hash).or_default().push((target, distance));
    }
    let mut eligible = Vec::new();
    let mut beyond_hash_quota = 0_u64;
    for (hash, mut candidates) in groups {
        candidates.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (index, (target, squared_distance)) in candidates.into_iter().enumerate() {
            if index >= 3 {
                beyond_hash_quota += 1;
                continue;
            }
            eligible.push(EligibleCandidate {
                target,
                hash,
                squared_distance,
                primary_squared_distance: primary_by_hash[&hash],
                hit_layer: (index + 1) as u8,
            });
        }
    }
    (eligible, beyond_hash_quota)
}

fn selected_set(row: &DiagnosticReplayRow) -> BTreeSet<(u32, u16, u32)> {
    row.backfill_slots
        .iter()
        .map(|&(target, hash, distance, _)| (target, hash, distance))
        .collect()
}

fn validate_replay_universe(
    replay: &DiagnosticReplayRow,
    eligible: &[EligibleCandidate],
) -> AnnResult<()> {
    let universe = eligible
        .iter()
        .map(|candidate| (candidate.key(), candidate.hit_layer))
        .collect::<BTreeMap<_, _>>();
    for &(target, hash, distance, layer) in &replay.backfill_slots {
        if universe.get(&(target, hash, distance)) != Some(&layer) {
            return Err(diagnostic_error(format!(
                "replayed backfill edge is absent from canonical eligible universe: target={target} hash={hash} layer={layer}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct ConstructionMetrics {
    schema: &'static str,
    rows: u64,
    degree_cap: usize,
    capacity_rule: &'static str,
    requested_max_backfill_per_row: Option<usize>,
    effective_max_backfill_per_row: usize,
    primary_degree_hist: Vec<u64>,
    degree_deficit_hist: Vec<u64>,
    retained_backfill_hist: Vec<u64>,
    rows_capacity_bound: u64,
    capacity_binding_rate: Option<f64>,
    primary_edges: u64,
    backfill_edges: u64,
    shortlist_admissions: u64,
    shortlist_replacements: u64,
    shortlist_unchanged: u64,
    shortlist_capacity_rejections: u64,
    shortlist_event_denominator: u64,
    shortlist_admission_rate: Option<f64>,
    shortlist_replacement_rate: Option<f64>,
    shortlist_unchanged_rate: Option<f64>,
    shortlist_capacity_rejection_rate: Option<f64>,
    sampled_rows: u64,
    sampled_candidate_records: u64,
    distinct_collision_hashes_hist: Vec<u64>,
    max_hit_layer_hist: [u64; 4],
    zero_dmax_rows: u64,
    beyond_hash_quota: u64,
    boundary_not_applicable: u64,
}

pub(crate) struct HashPruneDiagnosticFinalizer {
    prefix: PathBuf,
    rows: usize,
    degree_cap: usize,
    production_policy: HashPruneBackfillPolicy,
    next_source: usize,
    primary_count_writer: HashPrunePrimaryCountWriter,
    primary_degree_hist: Vec<u64>,
    degree_deficit_hist: Vec<u64>,
    retained_backfill_hist: Vec<u64>,
    rows_capacity_bound: u64,
    sampled_rows: u64,
    sampled_candidate_records: u64,
    distinct_collision_hashes_hist: Vec<u64>,
    max_hit_layer_hist: [u64; 4],
    policy_pairs: BTreeMap<String, PolicyPairAccumulator>,
    distance_histograms: DistanceLayerHistograms,
    beyond_hash_quota: u64,
}

pub(crate) struct HashPruneDiagnosticsSession {
    collector: Arc<HashPruneTraceCollector>,
    finalizer: HashPruneDiagnosticFinalizer,
}

impl HashPruneDiagnosticsSession {
    pub(crate) fn create(
        prefix: &Path,
        rows: usize,
        degree_cap: usize,
        production_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Self> {
        Ok(Self {
            collector: Arc::new(HashPruneTraceCollector::new()),
            finalizer: HashPruneDiagnosticFinalizer::create_with_policy(
                prefix,
                rows,
                degree_cap,
                production_policy,
            )?,
        })
    }

    pub(crate) fn collector(&self) -> &HashPruneTraceCollector {
        &self.collector
    }

    pub(crate) fn collector_arc(&self) -> Arc<HashPruneTraceCollector> {
        Arc::clone(&self.collector)
    }

    pub(crate) fn parts(
        &mut self,
    ) -> (&HashPruneTraceCollector, &mut HashPruneDiagnosticFinalizer) {
        (&self.collector, &mut self.finalizer)
    }

    pub(crate) fn finish(
        self,
        graph_start: u32,
        profile: &HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        if self.collector.live_trace_bytes() != 0 {
            return Err(diagnostic_error(format!(
                "diagnostic trace accounting did not drain: live_bytes={}",
                self.collector.live_trace_bytes()
            )));
        }
        self.finalizer.finish(graph_start, profile)
    }
}

impl HashPruneDiagnosticFinalizer {
    #[cfg(test)]
    pub(crate) fn create(prefix: &Path, rows: usize, degree_cap: usize) -> AnnResult<Self> {
        let production_policy =
            replay_policy(degree_cap, DiagnosticPolicy::SoftTier { alpha: 1.10 })?;
        Self::create_with_policy(prefix, rows, degree_cap, production_policy)
    }

    pub(crate) fn create_with_policy(
        prefix: &Path,
        rows: usize,
        degree_cap: usize,
        production_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Self> {
        if degree_cap == 0 || degree_cap > usize::from(u8::MAX) {
            return Err(diagnostic_error("diagnostic degree cap must be in 1..=255"));
        }
        if production_policy.mode() != PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
            || production_policy.priority_alpha().to_bits() != 1.10_f32.to_bits()
        {
            return Err(diagnostic_error(
                "construction diagnostics require the production SoftTier-Q3 alpha-1.10 policy",
            ));
        }
        fs::create_dir_all(prefix)?;
        for name in [
            "primary_count.sidecar",
            "construction_metrics.json",
            "counterfactual_policy_summary.tsv",
            "alpha_sensitivity.tsv",
            "distance_layer_histograms.tsv",
            "primary_count_manifest.json",
        ] {
            if prefix.join(name).exists() {
                return Err(diagnostic_error(format!(
                    "refusing to overwrite diagnostic artifact {}",
                    prefix.join(name).display()
                )));
            }
        }
        let primary_count_writer = HashPrunePrimaryCountWriter::create(
            &prefix.join("primary_count.sidecar"),
            rows,
            degree_cap,
        )?;
        Ok(Self {
            prefix: prefix.to_path_buf(),
            rows,
            degree_cap,
            production_policy,
            next_source: 0,
            primary_count_writer,
            primary_degree_hist: vec![0; degree_cap + 1],
            degree_deficit_hist: vec![0; degree_cap + 1],
            retained_backfill_hist: vec![0; degree_cap + 1],
            rows_capacity_bound: 0,
            sampled_rows: 0,
            sampled_candidate_records: 0,
            distinct_collision_hashes_hist: vec![0; degree_cap + 1],
            max_hit_layer_hist: [0; 4],
            policy_pairs: BTreeMap::new(),
            distance_histograms: DistanceLayerHistograms::default(),
            beyond_hash_quota: 0,
        })
    }

    pub(crate) fn observe_row(
        &mut self,
        source: usize,
        collector: &HashPruneTraceCollector,
        actual: &FinalizedHashPruneRow,
    ) -> AnnResult<()> {
        if source != self.next_source {
            return Err(diagnostic_error(format!(
                "diagnostic row order mismatch: source={source} expected={}",
                self.next_source
            )));
        }
        let degree = actual.slots.len();
        let primary = actual.primary_count;
        if primary > degree || degree > self.degree_cap {
            return Err(diagnostic_error(format!(
                "invalid finalized diagnostic row source={source} primary={primary} degree={degree}"
            )));
        }
        self.primary_count_writer
            .write_row(source as u64, primary, degree)?;
        self.primary_degree_hist[primary] += 1;
        self.degree_deficit_hist[self.degree_cap - primary] += 1;
        self.retained_backfill_hist[degree - primary] += 1;
        let capacity = self
            .production_policy
            .candidate_capacity_for(self.degree_cap, primary);
        self.rows_capacity_bound += u64::from(capacity != 0 && degree - primary == capacity);

        if source_is_sampled(source as u64) {
            let trace = collector.take(source)?;
            let actual_snapshot = ProductionRowSnapshot::from_finalized(actual);
            let replay = replay_all_policies_with_policy(
                self.degree_cap,
                &trace,
                &actual_snapshot,
                self.production_policy,
            )?;
            self.record_sampled_row(&trace, &replay)?;
        }
        self.next_source += 1;
        Ok(())
    }

    fn record_sampled_row(
        &mut self,
        trace: &[DiagnosticCandidate],
        replay: &DiagnosticReplayReport,
    ) -> AnnResult<()> {
        self.sampled_rows += 1;
        self.sampled_candidate_records += trace.len() as u64;
        let mut hash_counts = HashMap::<u16, usize>::new();
        for candidate in trace {
            *hash_counts.entry(candidate.hash).or_default() += 1;
        }
        let collision_hashes = hash_counts.values().filter(|&&count| count > 1).count();
        self.distinct_collision_hashes_hist[collision_hashes.min(self.degree_cap)] += 1;
        let max_layer = replay
            .soft_110
            .backfill_slots
            .iter()
            .map(|slot| usize::from(slot.3))
            .max()
            .unwrap_or(0);
        self.max_hit_layer_hist[max_layer] += 1;
        let (eligible, beyond_hash_quota) =
            canonical_eligible_universe(trace, &replay.soft_110.primary_slots);
        self.beyond_hash_quota = self
            .beyond_hash_quota
            .checked_add(beyond_hash_quota)
            .ok_or_else(|| diagnostic_error("beyond-quota count overflow"))?;
        for policy in replay.all_replays() {
            validate_replay_universe(policy, &eligible)?;
        }
        for (name, policy, row) in [
            (
                "distance_first",
                DiagnosticPolicy::DistanceFirst,
                &replay.distance_first,
            ),
            (
                "soft_100",
                DiagnosticPolicy::SoftTier { alpha: 1.00 },
                &replay.soft_100,
            ),
            (
                "soft_102",
                DiagnosticPolicy::SoftTier { alpha: 1.02 },
                &replay.soft_102,
            ),
            (
                "soft_105",
                DiagnosticPolicy::SoftTier { alpha: 1.05 },
                &replay.soft_105,
            ),
            (
                "soft_110",
                DiagnosticPolicy::SoftTier { alpha: 1.10 },
                &replay.soft_110,
            ),
            (
                "soft_120",
                DiagnosticPolicy::SoftTier { alpha: 1.20 },
                &replay.soft_120,
            ),
            (
                "soft_140",
                DiagnosticPolicy::SoftTier { alpha: 1.40 },
                &replay.soft_140,
            ),
            ("hard_tier", DiagnosticPolicy::HardTier, &replay.hard_tier),
        ] {
            self.record_policy_distribution(name, policy, row, &eligible)?;
        }
        for (name, left, right) in [
            (
                "distance_first__soft_110",
                &replay.distance_first,
                &replay.soft_110,
            ),
            ("soft_110__hard_tier", &replay.soft_110, &replay.hard_tier),
            ("soft_100__soft_110", &replay.soft_100, &replay.soft_110),
            ("soft_102__soft_110", &replay.soft_102, &replay.soft_110),
            ("soft_105__soft_110", &replay.soft_105, &replay.soft_110),
            ("soft_120__soft_110", &replay.soft_120, &replay.soft_110),
            ("soft_140__soft_110", &replay.soft_140, &replay.soft_110),
        ] {
            self.policy_pairs
                .entry(name.to_string())
                .or_default()
                .record(left, right);
            self.record_pair_only_distributions(left, right, &eligible)?;
        }
        Ok(())
    }

    fn record_policy_distribution(
        &mut self,
        name: &str,
        policy: DiagnosticPolicy,
        row: &DiagnosticReplayRow,
        eligible: &[EligibleCandidate],
    ) -> AnnResult<()> {
        let selected = selected_set(row);
        for candidate in eligible {
            self.distance_histograms.record(
                f64::from(candidate.primary_squared_distance),
                f64::from(candidate.squared_distance),
                candidate.hit_layer,
                if selected.contains(&candidate.key()) {
                    SelectionClass::Selected
                } else {
                    SelectionClass::Rejected
                },
            )?;
        }
        if matches!(policy, DiagnosticPolicy::HardTier) {
            self.distance_histograms.boundary_not_applicable += 1;
            return Ok(());
        }
        let priority = replay_policy_with_row_cap(
            self.degree_cap,
            policy,
            self.production_policy.requested_max_backfill_per_row(),
        )?;
        let scored = |candidate: &EligibleCandidate| {
            priority.priority_score(candidate.squared_distance, usize::from(candidate.hit_layer))
        };
        let last_selected = eligible
            .iter()
            .filter(|candidate| selected.contains(&candidate.key()))
            .max_by(|left, right| scored(left).total_cmp(&scored(right)));
        let first_rejected = eligible
            .iter()
            .filter(|candidate| !selected.contains(&candidate.key()))
            .min_by(|left, right| scored(left).total_cmp(&scored(right)));
        let (Some(last_selected), Some(first_rejected)) = (last_selected, first_rejected) else {
            self.distance_histograms.boundary_not_applicable += 1;
            return Ok(());
        };
        self.distance_histograms.record(
            f64::from(last_selected.primary_squared_distance),
            f64::from(last_selected.squared_distance),
            last_selected.hit_layer,
            SelectionClass::BoundarySelected,
        )?;
        self.distance_histograms.record(
            f64::from(first_rejected.primary_squared_distance),
            f64::from(first_rejected.squared_distance),
            first_rejected.hit_layer,
            SelectionClass::BoundaryRejected,
        )?;
        let last_score = scored(last_selected);
        let margin = (scored(first_rejected) - last_score) / last_score.max(f64::MIN_POSITIVE);
        self.distance_histograms.record_margin(name, margin)
    }

    fn record_pair_only_distributions(
        &mut self,
        left: &DiagnosticReplayRow,
        right: &DiagnosticReplayRow,
        eligible: &[EligibleCandidate],
    ) -> AnnResult<()> {
        let left_selected = selected_set(left);
        let right_selected = selected_set(right);
        for candidate in eligible {
            let class = match (
                left_selected.contains(&candidate.key()),
                right_selected.contains(&candidate.key()),
            ) {
                (true, false) => Some(SelectionClass::LeftOnly),
                (false, true) => Some(SelectionClass::RightOnly),
                _ => None,
            };
            if let Some(class) = class {
                self.distance_histograms.record(
                    f64::from(candidate.primary_squared_distance),
                    f64::from(candidate.squared_distance),
                    candidate.hit_layer,
                    class,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(
        self,
        graph_start: u32,
        profile: &HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        let primary_edges = weighted_sum(&self.primary_degree_hist)?;
        let backfill_edges = weighted_sum(&self.retained_backfill_hist)?;
        if self.next_source != self.rows
            || sum_hist(&self.primary_degree_hist)? != self.rows as u64
            || profile.rows_seen != self.rows as u64
            || primary_edges != profile.primary_edges
            || backfill_edges != profile.backfill_edges
            || profile.degree_before_sum != profile.primary_edges
            || profile.degree_after_sum
                != profile
                    .primary_edges
                    .checked_add(profile.backfill_edges)
                    .ok_or_else(|| diagnostic_error("profile edge sum overflow"))?
        {
            return Err(diagnostic_error(
                "construction diagnostic histogram/profile conservation failed",
            ));
        }
        let shortlist_event_denominator = profile
            .shortlist_admissions
            .checked_add(profile.shortlist_replacements)
            .and_then(|value| value.checked_add(profile.shortlist_unchanged))
            .and_then(|value| value.checked_add(profile.shortlist_capacity_rejections))
            .ok_or_else(|| diagnostic_error("shortlist event denominator overflow"))?;
        let shortlist_rate = |value: u64| {
            (shortlist_event_denominator != 0)
                .then(|| value as f64 / shortlist_event_denominator as f64)
        };
        let metrics = ConstructionMetrics {
            schema: "hashprune_construction_diagnostics_v1",
            rows: self.rows as u64,
            degree_cap: self.degree_cap,
            capacity_rule: self.production_policy.capacity_rule(),
            requested_max_backfill_per_row: self.production_policy.requested_max_backfill_per_row(),
            effective_max_backfill_per_row: self
                .production_policy
                .effective_max_backfill_per_row(self.degree_cap),
            primary_degree_hist: self.primary_degree_hist.clone(),
            degree_deficit_hist: self.degree_deficit_hist.clone(),
            retained_backfill_hist: self.retained_backfill_hist.clone(),
            rows_capacity_bound: self.rows_capacity_bound,
            capacity_binding_rate: (self.rows != 0)
                .then_some(self.rows_capacity_bound as f64 / self.rows as f64),
            primary_edges,
            backfill_edges,
            shortlist_admissions: profile.shortlist_admissions,
            shortlist_replacements: profile.shortlist_replacements,
            shortlist_unchanged: profile.shortlist_unchanged,
            shortlist_capacity_rejections: profile.shortlist_capacity_rejections,
            shortlist_event_denominator,
            shortlist_admission_rate: shortlist_rate(profile.shortlist_admissions),
            shortlist_replacement_rate: shortlist_rate(profile.shortlist_replacements),
            shortlist_unchanged_rate: shortlist_rate(profile.shortlist_unchanged),
            shortlist_capacity_rejection_rate: shortlist_rate(
                profile.shortlist_capacity_rejections,
            ),
            sampled_rows: self.sampled_rows,
            sampled_candidate_records: self.sampled_candidate_records,
            distinct_collision_hashes_hist: self.distinct_collision_hashes_hist.clone(),
            max_hit_layer_hist: self.max_hit_layer_hist,
            zero_dmax_rows: self.distance_histograms.zero_dmax_rows,
            beyond_hash_quota: self.beyond_hash_quota,
            boundary_not_applicable: self.distance_histograms.boundary_not_applicable,
        };

        self.primary_count_writer
            .finish(graph_start, HashPruneGraphEncoding::SequentialDegreeOnly)?;
        let sidecar_path = self.prefix.join("primary_count.sidecar");
        let sidecar = HashPrunePrimaryCountSidecar::load(&sidecar_path, self.rows)?;
        let manifest = serde_json::json!({
            "schema": "hashprune_primary_count_manifest_v1",
            "bytes": fs::metadata(&sidecar_path)?.len(),
            "sha256": sha256_file(&sidecar_path)?,
            "rows": sidecar.header().rows,
            "degree_cap": sidecar.header().degree_cap,
            "graph_start": sidecar.header().graph_start,
            "encoding": "sequential-degree-only",
            "policy": "rhalf-soft-tier-q3",
            "capacity_rule": self.production_policy.capacity_rule(),
            "requested_max_backfill_per_row": self.production_policy.requested_max_backfill_per_row(),
            "effective_max_backfill_per_row": self.production_policy.effective_max_backfill_per_row(self.degree_cap),
            "sample_seed": sidecar.header().sample_seed,
            "sample_denominator": sidecar.header().sample_denominator,
            "element_width": sidecar.header().element_width,
        });
        atomic_write(
            &self.prefix.join("construction_metrics.json"),
            &serde_json::to_vec_pretty(&metrics)
                .map_err(|error| diagnostic_error(error.to_string()))?,
        )?;
        atomic_write(
            &self.prefix.join("primary_count_manifest.json"),
            &serde_json::to_vec_pretty(&manifest)
                .map_err(|error| diagnostic_error(error.to_string()))?,
        )?;
        atomic_write(
            &self.prefix.join("counterfactual_policy_summary.tsv"),
            policy_pair_tsv(&self.policy_pairs).as_bytes(),
        )?;
        atomic_write(
            &self.prefix.join("alpha_sensitivity.tsv"),
            alpha_tsv(&self.policy_pairs).as_bytes(),
        )?;
        atomic_write(
            &self.prefix.join("distance_layer_histograms.tsv"),
            distance_histogram_tsv(&self.distance_histograms).as_bytes(),
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn next_source_for_test(&self) -> usize {
        self.next_source
    }
}

fn sum_hist(histogram: &[u64]) -> AnnResult<u64> {
    histogram.iter().try_fold(0_u64, |sum, &value| {
        sum.checked_add(value)
            .ok_or_else(|| diagnostic_error("histogram sum overflow"))
    })
}

fn weighted_sum(histogram: &[u64]) -> AnnResult<u64> {
    histogram
        .iter()
        .enumerate()
        .try_fold(0_u64, |sum, (index, &count)| {
            let contribution = (index as u64)
                .checked_mul(count)
                .ok_or_else(|| diagnostic_error("weighted histogram overflow"))?;
            sum.checked_add(contribution)
                .ok_or_else(|| diagnostic_error("weighted histogram sum overflow"))
        })
}

fn policy_pair_tsv(pairs: &BTreeMap<String, PolicyPairAccumulator>) -> String {
    let mut text = "# schema=hashprune_counterfactual_policy_summary_v1\npair\trows_compared\trows_different_selected_sets\trows_different_ordered_suffixes\tedge_swaps\tmicro_jaccard\tmacro_jaccard\tselected_by_left_layer\tselected_by_right_layer\tshared_layer_matrix\n".to_string();
    for (name, pair) in pairs {
        let micro = if pair.union_edges == 0 {
            1.0
        } else {
            pair.intersection_edges as f64 / pair.union_edges as f64
        };
        let macro_value = if pair.rows_compared == 0 {
            0.0
        } else {
            pair.macro_jaccard_sum / pair.rows_compared as f64
        };
        text.push_str(&format!(
            "{name}\t{}\t{}\t{}\t{}\t{micro:.12}\t{macro_value:.12}\t{},{},{}\t{},{},{}\t{},{},{};{},{},{};{},{},{}\n",
            pair.rows_compared,
            pair.rows_different_selected_sets,
            pair.rows_different_ordered_suffixes,
            pair.edge_swaps,
            pair.selected_by_left_layer[0],
            pair.selected_by_left_layer[1],
            pair.selected_by_left_layer[2],
            pair.selected_by_right_layer[0],
            pair.selected_by_right_layer[1],
            pair.selected_by_right_layer[2],
            pair.shared_layer_matrix[0][0],
            pair.shared_layer_matrix[0][1],
            pair.shared_layer_matrix[0][2],
            pair.shared_layer_matrix[1][0],
            pair.shared_layer_matrix[1][1],
            pair.shared_layer_matrix[1][2],
            pair.shared_layer_matrix[2][0],
            pair.shared_layer_matrix[2][1],
            pair.shared_layer_matrix[2][2],
        ));
    }
    text
}

fn alpha_tsv(pairs: &BTreeMap<String, PolicyPairAccumulator>) -> String {
    let mut text =
        "# schema=hashprune_alpha_sensitivity_v1\nalpha\trows_compared\tchanged_rows\tedge_swaps\n"
            .to_string();
    let sampled_rows = pairs
        .get("soft_100__soft_110")
        .map_or(0, |pair| pair.rows_compared);
    for (alpha, name) in [
        ("1.00", "soft_100__soft_110"),
        ("1.02", "soft_102__soft_110"),
        ("1.05", "soft_105__soft_110"),
        ("1.10", "soft_110__soft_110"),
        ("1.20", "soft_120__soft_110"),
        ("1.40", "soft_140__soft_110"),
    ] {
        let pair = if alpha == "1.10" {
            PolicyPairAccumulator {
                rows_compared: sampled_rows,
                ..PolicyPairAccumulator::default()
            }
        } else {
            pairs.get(name).cloned().unwrap_or_default()
        };
        text.push_str(&format!(
            "{alpha}\t{}\t{}\t{}\n",
            pair.rows_compared, pair.rows_different_selected_sets, pair.edge_swaps
        ));
    }
    text
}

fn distance_histogram_tsv(histograms: &DistanceLayerHistograms) -> String {
    let mut text = "# schema=hashprune_distance_layer_histograms_v1\nmetric\tclass_or_policy\thit_layer\tbin\tupper_bound\tcount\n".to_string();
    for ((class, layer), bins) in &histograms.rho_bins {
        for (bin, count) in bins.iter().enumerate() {
            text.push_str(&format!(
                "rho\t{class:?}\t{layer}\t{bin}\t{}\t{count}\n",
                RHO_UPPER_BOUNDS[bin]
            ));
        }
    }
    for (policy, bins) in &histograms.relative_margin_bins {
        for (bin, count) in bins.iter().enumerate() {
            text.push_str(&format!(
                "relative_margin\t{policy}\tNA\t{bin}\t{}\t{count}\n",
                RELATIVE_MARGIN_UPPER_BOUNDS[bin]
            ));
        }
    }
    text.push_str(&format!(
        "boundary_not_applicable\tall\tNA\tNA\tNA\t{}\n",
        histograms.boundary_not_applicable
    ));
    text
}

fn atomic_write(path: &Path, bytes: &[u8]) -> AnnResult<()> {
    let mut partial = path.as_os_str().to_os_string();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    file.write_all(bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&partial, path)?;
    File::open(
        path.parent()
            .ok_or_else(|| diagnostic_error("artifact has no parent"))?,
    )?
    .sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> AnnResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rayon::prelude::*;

    use super::*;

    fn first_sampled_source() -> u64 {
        (0_u64..).find(|&source| source_is_sampled(source)).unwrap()
    }

    fn soft_policy_with_row_cap(l_max: usize, cap: usize) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.l_max = l_max;
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_rhalf_max_backfill_per_row = Some(cap);
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    fn diagnostic_fixture_with_replacements_duplicates_and_three_layers() -> Vec<DiagnosticCandidate>
    {
        [
            (100, 10, 1.00),
            (101, 10, 1.10),
            (102, 10, 1.20),
            (103, 10, 1.30),
            (200, 20, 1.00),
            (201, 20, 1.05),
            (202, 20, 1.15),
            (300, 30, 0.90),
            (301, 30, 1.40),
            (101, 10, 1.08),
        ]
        .into_iter()
        .map(|(target, hash, squared_distance)| {
            DiagnosticCandidate::new(target, hash, squared_distance)
        })
        .collect()
    }

    #[test]
    fn sampled_sources_are_seeded_and_order_independent() {
        let forward = (0..100_000_u64)
            .filter(|&source| source_is_sampled(source))
            .collect::<Vec<_>>();
        let mut reverse = (0..100_000_u64)
            .rev()
            .filter(|&source| source_is_sampled(source))
            .collect::<Vec<_>>();
        reverse.reverse();
        assert_eq!(forward, reverse);
        assert!(forward.iter().all(|&source| {
            splitmix64(source ^ DIAGNOSTIC_SAMPLE_SEED)
                .is_multiple_of(DIAGNOSTIC_SAMPLE_DENOMINATOR)
        }));
        let parallel = (0..100_000_u64)
            .into_par_iter()
            .filter(|&source| source_is_sampled(source))
            .collect::<BTreeSet<_>>();
        assert_eq!(parallel, forward.iter().copied().collect());
    }

    #[test]
    fn trace_collector_fails_before_exceeding_its_accounted_ceiling() {
        const TEST_CEILING: u64 = 2 * 1024 * 1024;
        let collector = HashPruneTraceCollector::with_memory_ceiling_for_test(TEST_CEILING);
        let source = first_sampled_source();
        let mut accepted = 0;
        loop {
            match collector.record(source as usize, accepted, 7, accepted as f32 + 1.0) {
                Ok(()) => accepted += 1,
                Err(error) => {
                    assert!(
                        error.to_string().contains("2-GiB trace memory ceiling")
                            || error.to_string().contains("trace memory ceiling")
                    );
                    break;
                }
            }
        }
        assert!(collector.accounted_bytes() <= TEST_CEILING);
    }

    #[test]
    fn sampled_replays_satisfy_the_three_conservation_identities() {
        let trace = diagnostic_fixture_with_replacements_duplicates_and_three_layers();
        let actual_replay =
            replay_trace(8, &trace, DiagnosticPolicy::SoftTier { alpha: 1.10 }).unwrap();
        let actual = ProductionRowSnapshot::from_replay(&actual_replay);
        let report = replay_all_policies(8, &trace, &actual).unwrap();
        assert_eq!(report.soft_110.ordered_slots, actual_replay.ordered_slots);
        assert_eq!(
            report.soft_100.ordered_slots,
            report.distance_first.ordered_slots
        );
        for replay in report.all_replays() {
            assert_eq!(replay.primary_slots, report.soft_110.primary_slots);
        }
    }

    #[test]
    fn diagnostic_rhalf_row_cap_is_shared_by_every_counterfactual_policy() {
        let trace = diagnostic_fixture_with_replacements_duplicates_and_three_layers();
        let production_policy = soft_policy_with_row_cap(8, 2);
        let production = replay_trace_with_row_cap(
            8,
            &trace,
            DiagnosticPolicy::SoftTier { alpha: 1.10 },
            Some(2),
        )
        .unwrap();
        let report = replay_all_policies_with_policy(
            8,
            &trace,
            &ProductionRowSnapshot::from_replay(&production),
            production_policy,
        )
        .unwrap();
        assert_eq!(report.soft_110.ordered_slots, production.ordered_slots);
        assert_eq!(
            report.soft_100.ordered_slots,
            report.distance_first.ordered_slots
        );
        assert!(report.all_replays().iter().all(|row| {
            row.ordered_slots
                .len()
                .saturating_sub(row.primary_slots.len())
                <= 2
        }));
        assert_eq!(report.all_replays().len(), 8);
        let _ = &report.soft_102;
    }

    #[test]
    fn diagnostic_rhalf_row_cap_controls_capacity_binding_and_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("diagnostics");
        let collector = HashPruneTraceCollector::new();
        let policy = soft_policy_with_row_cap(8, 2);
        let mut finalizer =
            HashPruneDiagnosticFinalizer::create_with_policy(&prefix, 1, 8, policy).unwrap();
        finalizer
            .observe_row(0, &collector, &finalized_fixture_row(3, 2))
            .unwrap();
        assert_eq!(finalizer.rows_capacity_bound, 1);
        let profile = HashPruneBackfillProfile {
            rows_seen: 1,
            primary_edges: 3,
            backfill_edges: 2,
            degree_before_sum: 3,
            degree_after_sum: 5,
            ..HashPruneBackfillProfile::default()
        };
        finalizer.finish(0, &profile).unwrap();
        let metrics: serde_json::Value = serde_json::from_slice(
            &std::fs::read(prefix.join("construction_metrics.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metrics["requested_max_backfill_per_row"], 2);
        assert_eq!(metrics["effective_max_backfill_per_row"], 2);
        assert_eq!(metrics["capacity_rule"], "min-derived-budget-row-cap-spare");
        let alpha = std::fs::read_to_string(prefix.join("alpha_sensitivity.tsv")).unwrap();
        assert!(alpha.lines().any(|line| line.starts_with("1.02\t")));
    }

    #[test]
    fn diagnostic_replay_matches_production_for_seeded_traces() {
        let mut rng = StdRng::seed_from_u64(0x4449_4147_5245_504c);
        for _ in 0..512 {
            let l_max = rng.random_range(1_usize..=32);
            let candidate_count = rng.random_range(0_usize..=8 * l_max);
            let mut trace = Vec::with_capacity(candidate_count);
            for ordinal in 0..candidate_count {
                let target = u32::try_from(ordinal + 1).unwrap();
                let hash = rng.random_range(0_u16..u16::try_from(2 * l_max).unwrap());
                let squared_distance = rng.random_range(1_u32..=100_000) as f32 / 1_000.0;
                trace.push(DiagnosticCandidate::new(target, hash, squared_distance));
            }

            for policy in [
                DiagnosticPolicy::DistanceFirst,
                DiagnosticPolicy::SoftTier { alpha: 1.00 },
                DiagnosticPolicy::SoftTier { alpha: 1.10 },
                DiagnosticPolicy::SoftTier { alpha: 1.40 },
                DiagnosticPolicy::HardTier,
            ] {
                let actual = replay_trace(l_max, &trace, policy).unwrap();
                let production_policy = replay_policy(l_max, policy).unwrap();
                let mut reservoir = HashPruneReservoir::new(l_max);
                let mut profile = HashPruneBackfillProfile::default();
                for candidate in &trace {
                    reservoir
                        .insert_backfill_prevalidated_with_policy(
                            0,
                            candidate.target,
                            candidate.hash,
                            candidate.squared_distance,
                            production_policy,
                            &mut profile,
                        )
                        .unwrap();
                }
                let prefinal = reservoir.diagnostic_prefinal_snapshot().unwrap();
                let row = reservoir
                    .into_finalized_slots_with_backfill(0, production_policy, &mut profile)
                    .unwrap();
                let ordered_slots = row
                    .slots
                    .iter()
                    .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                    .collect::<Vec<_>>();
                let layers = prefinal.slots[prefinal.primary_count..]
                    .iter()
                    .map(|&(target, hash, distance, layer)| ((target, hash, distance), layer))
                    .collect::<HashMap<_, _>>();
                let expected_backfill = ordered_slots[row.primary_count..]
                    .iter()
                    .map(|&(target, hash, distance)| {
                        (
                            target,
                            hash,
                            distance,
                            *layers.get(&(target, hash, distance)).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual.ordered_slots, ordered_slots);
                assert_eq!(actual.backfill_slots, expected_backfill);
            }
        }
    }

    fn finalized_fixture_row(primary: usize, backfill: usize) -> FinalizedHashPruneRow {
        assert!(primary > 0 || backfill == 0);
        let mut slots = (0..primary)
            .map(|index| Slot::new(1_000 + index as u32, index as u16, index as f32 + 1.0))
            .collect::<Vec<_>>();
        slots.extend((0..backfill).map(|index| {
            Slot::new(
                2_000 + index as u32,
                (index % primary) as u16,
                index as f32 + 2.0,
            )
        }));
        FinalizedHashPruneRow {
            slots,
            primary_count: primary,
        }
    }

    fn sum_json_array(value: &serde_json::Value) -> u64 {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry.as_u64().unwrap())
            .sum()
    }

    fn weighted_json_array(value: &serde_json::Value) -> u64 {
        value
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, entry)| index as u64 * entry.as_u64().unwrap())
            .sum()
    }

    #[test]
    fn construction_histograms_conserve_rows_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("diagnostics");
        let collector = HashPruneTraceCollector::new();
        let mut finalizer = HashPruneDiagnosticFinalizer::create(&prefix, 3, 8).unwrap();

        for (source, primary, backfill) in [(0, 3, 2), (1, 8, 0), (2, 1, 1)] {
            let actual = finalized_fixture_row(primary, backfill);
            finalizer.observe_row(source, &collector, &actual).unwrap();
        }
        let profile = HashPruneBackfillProfile {
            rows_seen: 3,
            primary_edges: 12,
            backfill_edges: 3,
            degree_before_sum: 12,
            degree_after_sum: 15,
            ..HashPruneBackfillProfile::default()
        };
        finalizer.finish(17, &profile).unwrap();

        let value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(prefix.join("construction_metrics.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(sum_json_array(&value["primary_degree_hist"]), 3);
        assert_eq!(weighted_json_array(&value["primary_degree_hist"]), 12);
        assert_eq!(sum_json_array(&value["retained_backfill_hist"]), 3);
        assert_eq!(weighted_json_array(&value["retained_backfill_hist"]), 3);
    }

    #[test]
    fn construction_finalizer_rejects_profile_or_replay_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("diagnostics");
        let collector = HashPruneTraceCollector::new();
        let mut finalizer = HashPruneDiagnosticFinalizer::create(&prefix, 1, 8).unwrap();
        finalizer
            .observe_row(0, &collector, &finalized_fixture_row(2, 1))
            .unwrap();
        let wrong = HashPruneBackfillProfile {
            rows_seen: 1,
            primary_edges: 2,
            backfill_edges: 0,
            degree_before_sum: 2,
            degree_after_sum: 2,
            ..HashPruneBackfillProfile::default()
        };
        assert!(finalizer.finish(0, &wrong).is_err());
        assert!(!prefix.join("construction_metrics.json").exists());
    }

    #[test]
    fn rho_zero_primary_distance_uses_the_exact_match_bucket() {
        let mut histogram = DistanceLayerHistograms::default();
        histogram
            .record(0.0, 0.0, 1, SelectionClass::Selected)
            .unwrap();
        assert_eq!(histogram.zero_dmax_rows, 1);
        assert_eq!(
            histogram
                .rho_bins
                .values()
                .flat_map(|bins| bins.iter())
                .sum::<u64>(),
            0
        );
    }

    #[test]
    fn distance_histograms_reject_invalid_squared_distances() {
        let mut histogram = DistanceLayerHistograms::default();
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                histogram
                    .record(1.0, invalid, 1, SelectionClass::Rejected)
                    .is_err()
            );
        }
        assert!(
            histogram
                .record(f64::NAN, 1.0, 1, SelectionClass::Selected)
                .is_err()
        );
    }

    #[test]
    fn sampled_row_records_selected_rejected_and_boundary_distributions() {
        let dir = tempfile::tempdir().unwrap();
        let trace = diagnostic_fixture_with_replacements_duplicates_and_three_layers();
        let production =
            replay_trace(8, &trace, DiagnosticPolicy::SoftTier { alpha: 1.10 }).unwrap();
        let replay =
            replay_all_policies(8, &trace, &ProductionRowSnapshot::from_replay(&production))
                .unwrap();
        let mut finalizer =
            HashPruneDiagnosticFinalizer::create(&dir.path().join("diagnostics"), 1, 8).unwrap();

        finalizer.record_sampled_row(&trace, &replay).unwrap();

        let count = |class| {
            finalizer
                .distance_histograms
                .rho_bins
                .iter()
                .filter(|((observed, _), _)| *observed == class)
                .flat_map(|(_, bins)| bins)
                .sum::<u64>()
        };
        assert!(count(SelectionClass::Selected) > 0);
        assert!(count(SelectionClass::Rejected) > 0);
        assert!(count(SelectionClass::BoundarySelected) > 0);
        assert!(count(SelectionClass::BoundaryRejected) > 0);
        assert!(
            !finalizer
                .distance_histograms
                .relative_margin_bins
                .is_empty()
        );
    }

    fn completed_finalizer_fixture() -> (
        tempfile::TempDir,
        PathBuf,
        HashPruneDiagnosticFinalizer,
        HashPruneBackfillProfile,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("diagnostics");
        let collector = HashPruneTraceCollector::new();
        let mut finalizer = HashPruneDiagnosticFinalizer::create(&prefix, 3, 8).unwrap();
        for (source, primary, backfill) in [(0, 3, 2), (1, 8, 0), (2, 1, 1)] {
            finalizer
                .observe_row(
                    source,
                    &collector,
                    &finalized_fixture_row(primary, backfill),
                )
                .unwrap();
        }
        let profile = HashPruneBackfillProfile {
            rows_seen: 3,
            primary_edges: 12,
            backfill_edges: 3,
            degree_before_sum: 12,
            degree_after_sum: 15,
            ..HashPruneBackfillProfile::default()
        };
        (dir, prefix, finalizer, profile)
    }

    #[test]
    fn construction_artifacts_are_parseable_and_leave_no_partial_files() {
        let (_dir, prefix, finalizer, profile) = completed_finalizer_fixture();
        finalizer.finish(7, &profile).unwrap();
        for name in ["construction_metrics.json", "primary_count_manifest.json"] {
            let bytes = std::fs::read(prefix.join(name)).unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        }
        for (name, schema) in [
            (
                "counterfactual_policy_summary.tsv",
                "hashprune_counterfactual_policy_summary_v1",
            ),
            ("alpha_sensitivity.tsv", "hashprune_alpha_sensitivity_v1"),
            (
                "distance_layer_histograms.tsv",
                "hashprune_distance_layer_histograms_v1",
            ),
        ] {
            let text = std::fs::read_to_string(prefix.join(name)).unwrap();
            let mut lines = text.lines();
            let expected_schema = format!("# schema={schema}");
            assert_eq!(lines.next(), Some(expected_schema.as_str()));
            assert!(lines.next().unwrap().contains('\t'));
        }
        assert!(std::fs::read_dir(&prefix).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".partial")
        }));
    }

    #[test]
    fn diagnostic_session_accepts_resident_prefix_then_external_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let collector = HashPruneTraceCollector::new();
        let mut finalizer =
            HashPruneDiagnosticFinalizer::create(&dir.path().join("diag"), 4, 8).unwrap();
        for source in [0_usize, 1] {
            finalizer
                .observe_row(source, &collector, &finalized_fixture_row(2, 1))
                .unwrap();
        }
        for source in [2_usize, 3] {
            finalizer
                .observe_row(source, &collector, &finalized_fixture_row(1, 0))
                .unwrap();
        }
        assert_eq!(finalizer.next_source_for_test(), 4);
    }
}
