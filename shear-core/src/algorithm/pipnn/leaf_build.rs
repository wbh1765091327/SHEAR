use std::collections::HashMap;
use std::mem::size_of;
use std::time::{Duration, Instant};
use std::{cmp, env};

use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};
use parking_lot::Mutex;
use rayon::prelude::*;

use super::compute_backend::{PiPNNComputeBackend, cpu_backend};
use super::flat_resident_reservoir::FlatResidentReservoirStore;
use super::hash_prune::{HashPruneInsertProfile, HashPruneReservoir, SketchAccessor, compute_hash};
use super::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use super::hash_prune_diagnostics::HashPruneTraceCollector;
use super::hash_prune_evidence::{
    HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneEvidenceProfile, HashPruneMeta,
};
use super::lrpf::{LrpfCandidate, LrpfConfig, LrpfWitness, compress_lrpf_row_before_spill};
use super::lrpf_spill::LrpfFragmentSink;
use super::params::{PiPNNParams, PiPNNPruneBackend};
use super::point_store::{PointBatchStats, PointStore, WindowedGatherOptions};
use super::stable_distance_order;
use crate::common::{AnnError, AnnResult, Metric};

/// Prefetched leaf-local sketch data for fast in-memory hash computation.
pub(crate) struct LeafSketchCache {
    data: Vec<f32>,
    width: usize,
}

impl LeafSketchCache {
    pub(crate) fn prefetch(sketches: &dyn SketchAccessor, leaf: &[u32]) -> AnnResult<Self> {
        let width = sketches.width();
        let mut data = vec![0.0f32; leaf.len() * width];
        let row_bytes = width.saturating_mul(size_of::<f32>());
        let options = WindowedGatherOptions {
            max_gap_rows: if row_bytes <= 4096 { 1 } else { 0 },
            max_window_bytes: row_bytes
                .saturating_mul(leaf.len().max(1))
                .min(2 * 1024 * 1024)
                .max(128 * 1024),
            alignment_bytes: 4096,
            sort_ids: true,
        };
        let mut stats = super::point_store::WindowedGatherStats::default();
        sketches.read_rows_windowed_into_stats(leaf, &mut data, &options, &mut stats)?;
        Ok(Self { data, width })
    }

    #[inline]
    pub(crate) fn row(&self, local_idx: usize) -> &[f32] {
        let start = local_idx * self.width;
        &self.data[start..start + self.width]
    }
}

/// 暂存待插入的边，用于减少锁竞争
#[derive(Clone, Debug)]
pub(crate) struct PendingEdge {
    pub(crate) p: usize,
    pub(crate) c: u32,
    pub(crate) hash: u16,
    pub(crate) dist: f32,
    pub(crate) meta: HashPruneMeta,
}

pub(crate) trait PendingEdgeSink: Sync {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        HashPruneEvidencePolicy::off()
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        HashPruneBackfillPolicy::off()
    }

    fn prepare_for_d1(&self) -> AnnResult<()> {
        Ok(())
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<PendingEdgeFlushProfile>;
}

pub(crate) struct ReservoirEdgeSink<'a> {
    reservoirs: &'a [Mutex<HashPruneReservoir>],
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostics: Option<&'a HashPruneTraceCollector>,
}

impl<'a> ReservoirEdgeSink<'a> {
    pub(crate) fn new(reservoirs: &'a [Mutex<HashPruneReservoir>]) -> Self {
        Self::new_with_policies(
            reservoirs,
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
    }

    pub(crate) fn new_with_policy(
        reservoirs: &'a [Mutex<HashPruneReservoir>],
        policy: HashPruneEvidencePolicy,
    ) -> Self {
        Self::new_with_policies(reservoirs, policy, HashPruneBackfillPolicy::off())
    }

    pub(crate) fn new_with_policies(
        reservoirs: &'a [Mutex<HashPruneReservoir>],
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> Self {
        debug_assert!(policy.is_off() || backfill_policy.is_off());
        Self {
            reservoirs,
            policy,
            backfill_policy,
            diagnostics: None,
        }
    }

    pub(crate) fn new_with_policies_and_diagnostics(
        reservoirs: &'a [Mutex<HashPruneReservoir>],
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostics: Option<&'a HashPruneTraceCollector>,
    ) -> Self {
        debug_assert!(policy.is_off() || backfill_policy.is_off());
        Self {
            reservoirs,
            policy,
            backfill_policy,
            diagnostics,
        }
    }
}

impl PendingEdgeSink for ReservoirEdgeSink<'_> {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        self.policy
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        self.backfill_policy
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<PendingEdgeFlushProfile> {
        flush_pending_edges_to_reservoirs_with_policies_and_diagnostics(
            edges,
            self.reservoirs,
            self.policy,
            self.backfill_policy,
            self.diagnostics,
        )
    }
}

pub(crate) struct FlatReservoirEdgeSink<'a> {
    store: &'a FlatResidentReservoirStore,
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostics: Option<&'a HashPruneTraceCollector>,
}

impl<'a> FlatReservoirEdgeSink<'a> {
    pub(crate) fn new(store: &'a FlatResidentReservoirStore) -> Self {
        Self::new_with_policies(
            store,
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
    }

    pub(crate) fn new_with_policies(
        store: &'a FlatResidentReservoirStore,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> Self {
        Self::new_with_policies_and_diagnostics(store, policy, backfill_policy, None)
    }

    pub(crate) fn new_with_policies_and_diagnostics(
        store: &'a FlatResidentReservoirStore,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostics: Option<&'a HashPruneTraceCollector>,
    ) -> Self {
        debug_assert!(policy.is_off() || backfill_policy.is_off());
        Self {
            store,
            policy,
            backfill_policy,
            diagnostics,
        }
    }
}

impl PendingEdgeSink for FlatReservoirEdgeSink<'_> {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        self.policy
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        self.backfill_policy
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<PendingEdgeFlushProfile> {
        flush_pending_edges_to_access_with_policies_and_diagnostics(
            edges,
            self.store,
            self.policy,
            self.backfill_policy,
            self.diagnostics,
        )
    }
}

pub struct LeafScratch {
    pub dmat: Vec<f32>,
    pub block_dmat: Vec<f32>,
    pub x: Array2<f32>,
    pub(crate) edges: Vec<PendingEdge>,
    row_topk: Vec<RowTopK>,
}

pub(crate) struct ParallelBlockResult {
    pub(crate) block_start: usize,
    pub(crate) profile: LeafProfile,
    pub(crate) edges: Vec<PendingEdge>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PendingEdgeFlushProfile {
    pub(crate) edges: usize,
    pub(crate) sort_wall: Duration,
    pub(crate) insert_wall: Duration,
    pub(crate) lock_wall: Duration,
    pub(crate) insert_body_wall: Duration,
    pub(crate) flushes: usize,
    pub(crate) groups: usize,
    pub(crate) max_group_edges: usize,
    pub(crate) insert_profile: HashPruneInsertProfile,
    pub(crate) evidence: HashPruneEvidenceProfile,
    pub(crate) backfill: HashPruneBackfillProfile,
}

impl PendingEdgeFlushProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        self.edges += other.edges;
        self.sort_wall += other.sort_wall;
        self.insert_wall += other.insert_wall;
        self.lock_wall += other.lock_wall;
        self.insert_body_wall += other.insert_body_wall;
        self.flushes += other.flushes;
        self.groups += other.groups;
        self.max_group_edges = self.max_group_edges.max(other.max_group_edges);
        self.insert_profile.merge(other.insert_profile);
        self.evidence.merge(other.evidence);
        self.backfill.merge(other.backfill);
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CompactTopKBatch {
    pub(crate) leaf_id: u64,
    pub(crate) point_ids: Vec<u32>,
    pub(crate) rows: usize,
    pub(crate) k: usize,
    pub(crate) topk_ids: Vec<u32>,
    pub(crate) topk_dists: Vec<f32>,
    pub(crate) lune_witness_pivot_ranks: Vec<u32>,
    pub(crate) lune_witness_margins: Vec<f32>,
    pub(crate) estimated_dtoh_bytes: usize,
    pub(crate) estimated_lune_witness_dtoh_bytes: usize,
}

pub(crate) const COMPACT_LUNE_NO_WITNESS: u32 = u32::MAX;

impl CompactTopKBatch {
    pub(crate) fn estimated_total_dtoh_bytes(&self) -> usize {
        self.estimated_dtoh_bytes
            .saturating_add(self.estimated_lune_witness_dtoh_bytes)
    }
}

pub(crate) trait CompactTopKSink: Sync {
    fn prepare_for_d1(&self) -> AnnResult<()> {
        Ok(())
    }

    fn flush_compact_topk(&self, batch: CompactTopKBatch) -> AnnResult<LeafProfile>;

    fn flush_compact_topk_batches(&self, batches: Vec<CompactTopKBatch>) -> AnnResult<LeafProfile> {
        let mut merged = LeafProfile::default();
        for batch in batches {
            merged.merge(self.flush_compact_topk(batch)?);
        }
        Ok(merged)
    }
}

const COMPACT_TOPK_EDGE_FLUSH_THRESHOLD: usize = 16_384;
const COMPACT_TOPK_SUPERBATCH_EDGE_FLUSH_THRESHOLD: usize = 262_144;
const COMPACT_TOPK_LINEAR_LOCAL_INDEX_MAX_ROWS: usize = 64;
const COMPACT_TOPK_LINEAR_LOCAL_INDEX_MAX_K: usize = 8;

enum CompactLocalIndex<'a> {
    Linear(&'a [u32]),
    Hash(HashMap<u32, usize>),
}

impl CompactLocalIndex<'_> {
    fn lookup(&self, point_id: u32) -> Option<usize> {
        match self {
            Self::Linear(point_ids) => point_ids
                .iter()
                .position(|&candidate| candidate == point_id),
            Self::Hash(local_by_point) => local_by_point.get(&point_id).copied(),
        }
    }
}

fn compact_local_index(batch: &CompactTopKBatch) -> CompactLocalIndex<'_> {
    if batch.rows <= COMPACT_TOPK_LINEAR_LOCAL_INDEX_MAX_ROWS
        && batch.k <= COMPACT_TOPK_LINEAR_LOCAL_INDEX_MAX_K
    {
        CompactLocalIndex::Linear(&batch.point_ids)
    } else {
        CompactLocalIndex::Hash(
            batch
                .point_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(idx, point_id)| (point_id, idx))
                .collect(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct CompactTopKEvidenceSlot {
    target_local: u32,
    forward_meta: HashPruneMeta,
    reverse_meta: HashPruneMeta,
}

fn compact_reciprocal_rank(
    batch: &CompactTopKBatch,
    target_local: usize,
    source_global: u32,
) -> Option<usize> {
    let offset = target_local.checked_mul(batch.k)?;
    let end = offset.checked_add(batch.k)?;
    batch
        .topk_ids
        .get(offset..end)?
        .iter()
        .position(|&candidate| candidate == source_global)
}

fn annotate_compact_topk_evidence(
    batch: &CompactTopKBatch,
    local_index: &CompactLocalIndex<'_>,
    policy: HashPruneEvidencePolicy,
) -> AnnResult<(
    Option<Vec<CompactTopKEvidenceSlot>>,
    HashPruneEvidenceProfile,
)> {
    if !policy.annotates() {
        return Ok((None, HashPruneEvidenceProfile::default()));
    }

    let annotation_start = Instant::now();
    let mut profile = HashPruneEvidenceProfile::default();
    let mut annotations = Vec::with_capacity(batch.rows.saturating_mul(batch.k));
    for (source_local, &source_global) in batch.point_ids.iter().enumerate() {
        let offset = source_local * batch.k;
        for rank in 0..batch.k {
            let target_global = batch.topk_ids[offset + rank];
            let distance = batch.topk_dists[offset + rank];
            if target_global == u32::MAX || !distance.is_finite() {
                annotations.push(CompactTopKEvidenceSlot {
                    target_local: u32::MAX,
                    forward_meta: HashPruneMeta::empty(),
                    reverse_meta: HashPruneMeta::empty(),
                });
                continue;
            }

            let Some(target_local) = local_index.lookup(target_global) else {
                return Err(AnnError::log_index_error(format!(
                    "Compact top-k batch references point {target_global} outside leaf {}",
                    batch.leaf_id
                )));
            };
            let target_local_u32 = u32::try_from(target_local).map_err(|_| {
                AnnError::log_index_error(format!(
                    "Compact top-k local index {target_local} exceeds u32 for leaf {}",
                    batch.leaf_id
                ))
            })?;

            profile.reciprocal_checks += 1;
            let reciprocal_rank = compact_reciprocal_rank(batch, target_local, source_global);
            profile.reciprocal_hits += usize::from(reciprocal_rank.is_some());

            let forward_class = HashPruneEvidenceClass::from_ranks(Some(rank), reciprocal_rank)
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "Missing forward compact top-k evidence class".to_string(),
                    )
                })?;
            let reverse_class = HashPruneEvidenceClass::from_ranks(reciprocal_rank, Some(rank))
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "Missing reverse compact top-k evidence class".to_string(),
                    )
                })?;
            profile.class_counts[forward_class.index()] += 1;
            profile.class_counts[reverse_class.index()] += 1;

            if reciprocal_rank.is_some() {
                if source_global < target_global {
                    profile.support_twice_candidates += 2;
                }
            } else {
                profile.support_once_candidates += 2;
            }

            annotations.push(CompactTopKEvidenceSlot {
                target_local: target_local_u32,
                forward_meta: HashPruneMeta::with_evidence(forward_class),
                reverse_meta: HashPruneMeta::with_evidence(reverse_class),
            });
        }
    }
    profile.annotation_wall = annotation_start.elapsed();
    Ok((Some(annotations), profile))
}

pub(crate) struct OomCompactTopKSink<'a> {
    sketches: &'a dyn SketchAccessor,
    params: &'a PiPNNParams,
    edge_sink: &'a dyn PendingEdgeSink,
}

pub(crate) struct OomLrpfCompactTopKSink<'a> {
    params: &'a PiPNNParams,
    fragment_sink: &'a dyn LrpfFragmentSink,
}

impl<'a> OomCompactTopKSink<'a> {
    pub(crate) fn new(
        sketches: &'a dyn SketchAccessor,
        params: &'a PiPNNParams,
        edge_sink: &'a dyn PendingEdgeSink,
    ) -> Self {
        Self {
            sketches,
            params,
            edge_sink,
        }
    }
}

impl<'a> OomLrpfCompactTopKSink<'a> {
    pub(crate) fn new(params: &'a PiPNNParams, fragment_sink: &'a dyn LrpfFragmentSink) -> Self {
        Self {
            params,
            fragment_sink,
        }
    }
}

impl CompactTopKSink for OomCompactTopKSink<'_> {
    fn prepare_for_d1(&self) -> AnnResult<()> {
        self.edge_sink.prepare_for_d1()
    }

    fn flush_compact_topk(&self, batch: CompactTopKBatch) -> AnnResult<LeafProfile> {
        flush_compact_topk_batch(&batch, self.sketches, self.params, self.edge_sink)
    }

    fn flush_compact_topk_batches(&self, batches: Vec<CompactTopKBatch>) -> AnnResult<LeafProfile> {
        flush_compact_topk_batches(&batches, self.sketches, self.params, self.edge_sink)
    }
}

impl CompactTopKSink for OomLrpfCompactTopKSink<'_> {
    fn flush_compact_topk(&self, batch: CompactTopKBatch) -> AnnResult<LeafProfile> {
        flush_compact_topk_batch_lrpf(&batch, self.params, self.fragment_sink)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LeafProfile {
    pub total_wall: Duration,
    pub load: Duration,
    pub distance: Duration,
    pub topk: Duration,
    pub hash: Duration,
    pub flush: Duration,
    pub leaves: usize,
    pub points: usize,
    pub blockwise_leaves: usize,
    pub io_stats: PointBatchStats,
    pub sketch_prefetch: Duration,
    pub sketch_rows_prefetched: usize,
    pub gpu_h2d_bytes: usize,
    pub gpu_d2h_bytes: usize,
    pub gpu_h2d: Duration,
    pub gpu_d2h: Duration,
    pub gpu_kernel: Duration,
    pub gpu_sync: Duration,
    pub gpu_workspace_peak_bytes: usize,
    pub gpu_workspace_budget_bytes: usize,
    pub gpu_workspace_pressure_events: usize,
    pub gpu_cpu_fallbacks: usize,
    pub host_load_wall: Duration,
    pub backend_compute_wall: Duration,
    pub compact_materialize_wall: Duration,
    pub hash_wall: Duration,
    pub edge_sink_wall: Duration,
    pub edge_sink_sort_wall: Duration,
    pub edge_sink_insert_wall: Duration,
    pub edge_sink_lock_wall: Duration,
    pub edge_sink_insert_body_wall: Duration,
    pub edge_sink_edges: usize,
    pub edge_sink_flushes: usize,
    pub edge_sink_groups: usize,
    pub edge_sink_max_group_edges: usize,
    pub edge_sink_insert_candidates: usize,
    pub edge_sink_hash_hits: usize,
    pub edge_sink_hash_replacements: usize,
    pub edge_sink_hash_rejects: usize,
    pub edge_sink_not_full_inserts: usize,
    pub edge_sink_full_replacement_checks: usize,
    pub edge_sink_full_replacements: usize,
    pub edge_sink_full_rejects: usize,
    pub edge_sink_farthest_recomputes: usize,
    pub edge_sink_rotate_lefts: usize,
    pub edge_sink_rotate_rights: usize,
    pub flush_wall: Duration,
    pub gpu_h2d_submit_wall: Duration,
    pub gpu_h2d_event_elapsed: Duration,
    pub gpu_kernel_launch: Duration,
    pub gpu_kernel_elapsed: Duration,
    pub gpu_d2h_submit_wall: Duration,
    pub gpu_d2h_event_elapsed: Duration,
    pub gpu_post_d2h_sync_wall: Duration,
    pub gpu_stream_total_elapsed: Duration,
    pub compact_lune_witness_enabled: bool,
    pub compact_lune_witness_slots: usize,
    pub compact_lune_witnesses: usize,
    pub compact_lune_prune_enabled: bool,
    pub compact_lune_prune_protected_core: usize,
    pub compact_lune_pruned_slots: usize,
    pub compact_lune_pruned_rows: usize,
    pub lrpf_pre_spill_pruned_candidates: usize,
    pub(crate) hashprune_evidence: HashPruneEvidenceProfile,
    pub(crate) hashprune_backfill: HashPruneBackfillProfile,
}

impl LeafProfile {
    pub fn merge(&mut self, other: LeafProfile) {
        self.total_wall += other.total_wall;
        self.load += other.load;
        self.distance += other.distance;
        self.topk += other.topk;
        self.hash += other.hash;
        self.flush += other.flush;
        self.leaves += other.leaves;
        self.points += other.points;
        self.blockwise_leaves += other.blockwise_leaves;
        self.io_stats.merge(&other.io_stats);
        self.sketch_prefetch += other.sketch_prefetch;
        self.sketch_rows_prefetched += other.sketch_rows_prefetched;
        self.gpu_h2d_bytes += other.gpu_h2d_bytes;
        self.gpu_d2h_bytes += other.gpu_d2h_bytes;
        self.gpu_h2d += other.gpu_h2d;
        self.gpu_d2h += other.gpu_d2h;
        self.gpu_kernel += other.gpu_kernel;
        self.gpu_sync += other.gpu_sync;
        self.gpu_workspace_peak_bytes = self
            .gpu_workspace_peak_bytes
            .max(other.gpu_workspace_peak_bytes);
        self.gpu_workspace_budget_bytes = self
            .gpu_workspace_budget_bytes
            .max(other.gpu_workspace_budget_bytes);
        self.gpu_workspace_pressure_events += other.gpu_workspace_pressure_events;
        self.gpu_cpu_fallbacks += other.gpu_cpu_fallbacks;
        self.host_load_wall += other.host_load_wall;
        self.backend_compute_wall += other.backend_compute_wall;
        self.compact_materialize_wall += other.compact_materialize_wall;
        self.hash_wall += other.hash_wall;
        self.edge_sink_wall += other.edge_sink_wall;
        self.edge_sink_sort_wall += other.edge_sink_sort_wall;
        self.edge_sink_insert_wall += other.edge_sink_insert_wall;
        self.edge_sink_lock_wall += other.edge_sink_lock_wall;
        self.edge_sink_insert_body_wall += other.edge_sink_insert_body_wall;
        self.edge_sink_edges += other.edge_sink_edges;
        self.edge_sink_flushes += other.edge_sink_flushes;
        self.edge_sink_groups += other.edge_sink_groups;
        self.edge_sink_max_group_edges = self
            .edge_sink_max_group_edges
            .max(other.edge_sink_max_group_edges);
        self.edge_sink_insert_candidates += other.edge_sink_insert_candidates;
        self.edge_sink_hash_hits += other.edge_sink_hash_hits;
        self.edge_sink_hash_replacements += other.edge_sink_hash_replacements;
        self.edge_sink_hash_rejects += other.edge_sink_hash_rejects;
        self.edge_sink_not_full_inserts += other.edge_sink_not_full_inserts;
        self.edge_sink_full_replacement_checks += other.edge_sink_full_replacement_checks;
        self.edge_sink_full_replacements += other.edge_sink_full_replacements;
        self.edge_sink_full_rejects += other.edge_sink_full_rejects;
        self.edge_sink_farthest_recomputes += other.edge_sink_farthest_recomputes;
        self.edge_sink_rotate_lefts += other.edge_sink_rotate_lefts;
        self.edge_sink_rotate_rights += other.edge_sink_rotate_rights;
        self.flush_wall += other.flush_wall;
        self.gpu_h2d_submit_wall += other.gpu_h2d_submit_wall;
        self.gpu_h2d_event_elapsed += other.gpu_h2d_event_elapsed;
        self.gpu_kernel_launch += other.gpu_kernel_launch;
        self.gpu_kernel_elapsed += other.gpu_kernel_elapsed;
        self.gpu_d2h_submit_wall += other.gpu_d2h_submit_wall;
        self.gpu_d2h_event_elapsed += other.gpu_d2h_event_elapsed;
        self.gpu_post_d2h_sync_wall += other.gpu_post_d2h_sync_wall;
        self.gpu_stream_total_elapsed += other.gpu_stream_total_elapsed;
        self.compact_lune_witness_enabled |= other.compact_lune_witness_enabled;
        self.compact_lune_witness_slots += other.compact_lune_witness_slots;
        self.compact_lune_witnesses += other.compact_lune_witnesses;
        self.compact_lune_prune_enabled |= other.compact_lune_prune_enabled;
        self.compact_lune_prune_protected_core = self
            .compact_lune_prune_protected_core
            .max(other.compact_lune_prune_protected_core);
        self.compact_lune_pruned_slots += other.compact_lune_pruned_slots;
        self.compact_lune_pruned_rows += other.compact_lune_pruned_rows;
        self.lrpf_pre_spill_pruned_candidates += other.lrpf_pre_spill_pruned_candidates;
        self.hashprune_evidence.merge(other.hashprune_evidence);
        self.hashprune_backfill.merge(other.hashprune_backfill);
    }

    pub(crate) fn record_pending_edge_flush(
        &mut self,
        elapsed: Duration,
        flush_profile: PendingEdgeFlushProfile,
    ) {
        self.flush += elapsed;
        self.flush_wall += elapsed;
        self.edge_sink_wall += elapsed;
        self.edge_sink_sort_wall += flush_profile.sort_wall;
        self.edge_sink_insert_wall += flush_profile.insert_wall;
        self.edge_sink_lock_wall += flush_profile.lock_wall;
        self.edge_sink_insert_body_wall += flush_profile.insert_body_wall;
        self.edge_sink_edges += flush_profile.edges;
        self.edge_sink_flushes += flush_profile.flushes;
        self.edge_sink_groups += flush_profile.groups;
        self.edge_sink_max_group_edges = self
            .edge_sink_max_group_edges
            .max(flush_profile.max_group_edges);
        self.edge_sink_insert_candidates += flush_profile.insert_profile.candidates;
        self.edge_sink_hash_hits += flush_profile.insert_profile.hash_hits;
        self.edge_sink_hash_replacements += flush_profile.insert_profile.hash_replacements;
        self.edge_sink_hash_rejects += flush_profile.insert_profile.hash_rejects;
        self.edge_sink_not_full_inserts += flush_profile.insert_profile.not_full_inserts;
        self.edge_sink_full_replacement_checks +=
            flush_profile.insert_profile.full_replacement_checks;
        self.edge_sink_full_replacements += flush_profile.insert_profile.full_replacements;
        self.edge_sink_full_rejects += flush_profile.insert_profile.full_rejects;
        self.edge_sink_farthest_recomputes += flush_profile.insert_profile.farthest_recomputes;
        self.edge_sink_rotate_lefts += flush_profile.insert_profile.rotate_lefts;
        self.edge_sink_rotate_rights += flush_profile.insert_profile.rotate_rights;
        self.hashprune_evidence.merge(flush_profile.evidence);
        self.hashprune_backfill.merge(flush_profile.backfill);
    }
}

pub(crate) fn estimate_compact_lune_witness_dtoh_bytes(rows: usize, k: usize) -> usize {
    rows.saturating_mul(k)
        .saturating_mul(size_of::<u32>().saturating_add(size_of::<f32>()))
}

pub(crate) fn compact_lune_witness_slots_are_valid(batch: &CompactTopKBatch) -> bool {
    let expected = batch.rows.saturating_mul(batch.k);
    if batch.lune_witness_pivot_ranks.is_empty() && batch.lune_witness_margins.is_empty() {
        return true;
    }
    batch.lune_witness_pivot_ranks.len() == expected && batch.lune_witness_margins.len() == expected
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn effective_compact_lune_prune_enabled(params: &PiPNNParams) -> bool {
    params.compact_lune_prune_enable || env_flag_enabled("SHEAR_PIPNN_COMPACT_LUNE_PRUNE")
}

pub(crate) fn effective_compact_lune_witness_enabled(params: &PiPNNParams) -> bool {
    params.compact_lune_witness_enable
        || env_flag_enabled("SHEAR_PIPNN_COMPACT_LUNE_WITNESS")
        || effective_compact_lune_prune_enabled(params)
        || params.prune_backend == PiPNNPruneBackend::LRPF
}

fn effective_compact_lune_prune_protected_core(params: &PiPNNParams) -> usize {
    env::var("SHEAR_PIPNN_COMPACT_LUNE_PRUNE_PROTECTED_CORE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(params.compact_lune_prune_protected_core)
}

fn compact_lune_prune_candidate(
    batch: &CompactTopKBatch,
    local_row: usize,
    rank: usize,
    protected_core: usize,
) -> bool {
    if protected_core == 0
        || rank < protected_core
        || batch.lune_witness_pivot_ranks.is_empty()
        || batch.lune_witness_margins.is_empty()
    {
        return false;
    }
    let slot = local_row.saturating_mul(batch.k).saturating_add(rank);
    let Some(&pivot_rank) = batch.lune_witness_pivot_ranks.get(slot) else {
        return false;
    };
    if pivot_rank == COMPACT_LUNE_NO_WITNESS {
        return false;
    }
    let pivot_rank = pivot_rank as usize;
    pivot_rank < rank
        && pivot_rank < protected_core
        && batch.lune_witness_margins.get(slot).copied().unwrap_or(0.0) > 0.0
}

pub(crate) fn derive_compact_lune_witnesses_from_topk(
    batch: &mut CompactTopKBatch,
    alpha: f32,
    max_witnesses_per_victim: usize,
) -> AnnResult<()> {
    if batch.point_ids.len() != batch.rows
        || batch.topk_ids.len() != batch.rows.saturating_mul(batch.k)
        || batch.topk_dists.len() != batch.rows.saturating_mul(batch.k)
    {
        return Err(AnnError::log_index_error(format!(
            "Invalid compact top-k batch shape before lune witness derivation: rows={} k={} point_ids={} topk_ids={} topk_dists={}",
            batch.rows,
            batch.k,
            batch.point_ids.len(),
            batch.topk_ids.len(),
            batch.topk_dists.len()
        )));
    }

    let slots = batch.rows.saturating_mul(batch.k);
    batch
        .lune_witness_pivot_ranks
        .resize(slots, COMPACT_LUNE_NO_WITNESS);
    batch.lune_witness_margins.resize(slots, 0.0);
    batch.lune_witness_pivot_ranks.fill(COMPACT_LUNE_NO_WITNESS);
    batch.lune_witness_margins.fill(0.0);
    batch.estimated_lune_witness_dtoh_bytes =
        estimate_compact_lune_witness_dtoh_bytes(batch.rows, batch.k);

    if batch.rows == 0 || batch.k == 0 || alpha <= 0.0 || max_witnesses_per_victim == 0 {
        return Ok(());
    }

    let local_by_point: HashMap<u32, usize> = batch
        .point_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(local, point)| (point, local))
        .collect();

    for src_row in 0..batch.rows {
        let src_offset = src_row.saturating_mul(batch.k);
        for victim_rank in 1..batch.k {
            let victim_slot = src_offset + victim_rank;
            let victim_id = batch.topk_ids[victim_slot];
            let src_victim_dist = batch.topk_dists[victim_slot];
            if victim_id == u32::MAX || !src_victim_dist.is_finite() {
                continue;
            }

            for pivot_rank in 0..victim_rank {
                let pivot_id = batch.topk_ids[src_offset + pivot_rank];
                if pivot_id == u32::MAX {
                    continue;
                }
                let Some(&pivot_row) = local_by_point.get(&pivot_id) else {
                    continue;
                };
                let pivot_offset = pivot_row.saturating_mul(batch.k);
                let Some(pivot_victim_slot) = batch.topk_ids[pivot_offset..pivot_offset + batch.k]
                    .iter()
                    .position(|&candidate| candidate == victim_id)
                else {
                    continue;
                };
                let pivot_victim_dist = batch.topk_dists[pivot_offset + pivot_victim_slot];
                if !pivot_victim_dist.is_finite() {
                    continue;
                }
                let threshold = src_victim_dist / alpha;
                if pivot_victim_dist < threshold {
                    batch.lune_witness_pivot_ranks[victim_slot] = pivot_rank as u32;
                    batch.lune_witness_margins[victim_slot] = threshold - pivot_victim_dist;
                    break;
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn maybe_derive_compact_lune_witnesses_from_topk(
    batch: &mut CompactTopKBatch,
    params: &PiPNNParams,
) -> AnnResult<()> {
    if effective_compact_lune_witness_enabled(params) {
        derive_compact_lune_witnesses_from_topk(batch, 1.2, 4)
    } else {
        batch.lune_witness_pivot_ranks.clear();
        batch.lune_witness_margins.clear();
        batch.estimated_lune_witness_dtoh_bytes = 0;
        Ok(())
    }
}

fn default_leaf_point_read_options(dataset: &dyn PointStore, rows: usize) -> WindowedGatherOptions {
    let row_bytes = dataset.dim().saturating_mul(size_of::<f32>());
    WindowedGatherOptions {
        max_gap_rows: if row_bytes <= 4096 { 1 } else { 0 },
        max_window_bytes: row_bytes
            .saturating_mul(rows.max(1))
            .min(4 * 1024 * 1024)
            .max(256 * 1024),
        alignment_bytes: 4096,
        sort_ids: true,
    }
}

impl LeafScratch {
    pub fn new(c_max: usize, dim: usize, leaf_knn: usize) -> Self {
        let edge_capacity = c_max.saturating_mul(leaf_knn.max(1)).saturating_mul(2);
        Self {
            dmat: Vec::with_capacity(c_max * c_max),
            block_dmat: Vec::new(),
            x: Array2::zeros((c_max, dim)),
            edges: Vec::with_capacity(edge_capacity),
            row_topk: Vec::new(),
        }
    }

    /// 将暂存的边写入全局 reservoirs。
    /// 为了减少锁竞争，我们可以对边进行排序，使得对同一个 reservoir 的操作集中在一起，
    /// 或者仅仅是利用批量写入来平摊锁开销。
    pub(crate) fn flush_edges(
        &mut self,
        edge_sink: &dyn PendingEdgeSink,
    ) -> AnnResult<PendingEdgeFlushProfile> {
        flush_pending_edges(&mut self.edges, edge_sink)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RowTopK {
    buf: [(usize, u32, f32); 32],
    len: usize,
}

impl Default for RowTopK {
    fn default() -> Self {
        Self {
            buf: [(0, u32::MAX, f32::INFINITY); 32],
            len: 0,
        }
    }
}

impl RowTopK {
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn is_candidate_better(
        candidate_dist: f32,
        candidate_target_id: u32,
        current: (usize, u32, f32),
    ) -> bool {
        stable_distance_order(candidate_dist, candidate_target_id, current.2, current.1).is_lt()
    }

    pub(crate) fn push(&mut self, k: usize, idx: usize, target_id: u32, dist: f32) {
        debug_assert!(k <= 32, "RowTopK: k={k} exceeds fixed buffer size 32");
        if k == 0 {
            return;
        }

        if self.len < k {
            // Buffer not full yet: insertion sort into buf[0..len+1]
            let mut i = self.len;
            while i > 0 && Self::is_candidate_better(dist, target_id, self.buf[i - 1]) {
                self.buf[i] = self.buf[i - 1];
                i -= 1;
            }
            self.buf[i] = (idx, target_id, dist);
            self.len += 1;
        } else {
            // Buffer full: only insert if closer than the farthest (buf[k-1])
            if !Self::is_candidate_better(dist, target_id, self.buf[k - 1]) {
                return;
            }
            // Insertion sort: shift right from the end, stop when we find the right spot
            let mut i = k - 1;
            while i > 0 && Self::is_candidate_better(dist, target_id, self.buf[i - 1]) {
                self.buf[i] = self.buf[i - 1];
                i -= 1;
            }
            self.buf[i] = (idx, target_id, dist);
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.buf[..self.len]
            .iter()
            .map(|&(idx, _target_id, dist)| (idx, dist))
    }
}

/// 对单个叶子执行：
///  - 计算叶内所有点对距离（O(|b|^2)）
///  - 为每个点选取 leaf_knn 个最近邻
///  - 对每一条候选边 (p, c) 分别插入 p 和 c 的 HashPrune 水库
///
/// 当度量为 L2 时，使用 ndarray+BLAS 的 GEMM 计算点积矩阵 X·X^T，
/// 再通过行范数组装距离平方矩阵 dist2 = ||x||^2 + ||y||^2 - 2·(X·X^T)；
/// 非 L2 度量则回退到逐对距离计算，以兼容 Cosine/IP 等扩展。
pub fn process_leaf(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    scratch: &mut LeafScratch,
) -> AnnResult<()> {
    process_leaf_profiled(dataset, metric, sketches, leaf, params, reservoirs, scratch).map(|_| ())
}

pub fn process_leaf_profiled(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    scratch: &mut LeafScratch,
) -> AnnResult<LeafProfile> {
    let edge_sink = ReservoirEdgeSink::new(reservoirs);
    process_leaf_profiled_with_sink(dataset, metric, sketches, leaf, params, &edge_sink, scratch)
}

pub(crate) fn process_leaf_profiled_with_sink(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    scratch: &mut LeafScratch,
) -> AnnResult<LeafProfile> {
    let total_start = Instant::now();
    let mut profile = LeafProfile {
        leaves: usize::from(!leaf.is_empty()),
        points: leaf.len(),
        ..LeafProfile::default()
    };
    let leaf_size = leaf.len();
    if leaf_size <= 1 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    let max_leaf_size = params.kernel_safe_leaf_size();
    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    if k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    if leaf_size > max_leaf_size {
        profile.blockwise_leaves = 1;
        process_leaf_blockwise(
            dataset,
            metric,
            sketches,
            leaf,
            params,
            edge_sink,
            scratch,
            &mut profile,
        )?;
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    // 重置并调整 scratch 大小
    scratch.dmat.clear();
    scratch.dmat.resize(leaf_size * leaf_size, 0.0f32);

    if metric == Metric::L2 {
        let dim = dataset.dim();
        // 确保 scratch.x 足够大，如果不匹配则重新分配
        if scratch.x.nrows() < leaf_size || scratch.x.ncols() < dim {
            scratch.x = Array2::<f32>::zeros((leaf_size.max(scratch.x.nrows()), dim));
        }

        // 局部视图，避免分配
        let mut x_view = scratch.x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);

        let load_start = Instant::now();
        let mut flat = vec![0.0f32; leaf_size * dim];
        let read_options = default_leaf_point_read_options(dataset, leaf_size);
        dataset.read_points_windowed_into_batch_stats(
            leaf,
            &mut flat,
            &read_options,
            &mut profile.io_stats,
        )?;
        for (i_local, chunk) in flat.chunks(dim).enumerate() {
            x_view
                .row_mut(i_local)
                .assign(&ndarray::ArrayView1::from(chunk));
        }
        let elapsed = load_start.elapsed();
        profile.load += elapsed;
        profile.host_load_wall += elapsed;

        // 行范数 ||x_i||^2
        let distance_start = Instant::now();
        let norms: Vec<f32> = x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec();

        // 使用 dmat 作为 Gram 矩阵的存储空间
        let mut dmat_view =
            ArrayViewMut2::from_shape((leaf_size, leaf_size), &mut scratch.dmat).unwrap();
        ndarray::linalg::general_mat_mul(1.0, &x_view, &x_view.t(), 0.0, &mut dmat_view);

        // 现在 scratch.dmat 中存储的是 gram[i, j] = x_i · x_j
        // 转换为 dist2 = norms[i] + norms[j] - 2.0 * gram[i, j]
        for i in 0..leaf_size {
            let offset = i * leaf_size;
            let norm_i = norms[i];
            for j in 0..leaf_size {
                let gram_ij = scratch.dmat[offset + j];
                let mut dist2 = norm_i + norms[j] - 2.0 * gram_ij;
                if dist2 <= 0.0 {
                    dist2 = 0.0;
                }
                scratch.dmat[offset + j] = dist2;
            }
        }
        let elapsed = distance_start.elapsed();
        profile.distance += elapsed;
        profile.backend_compute_wall += elapsed;
    } else {
        let distance_start = Instant::now();
        for (i_local, &i_global) in leaf.iter().enumerate() {
            for (j_local, &j_global) in leaf.iter().enumerate().skip(i_local + 1) {
                let d = dataset.get_distance(i_global, j_global, metric)?;
                scratch.dmat[i_local * leaf_size + j_local] = d;
                scratch.dmat[j_local * leaf_size + i_local] = d;
            }
        }
        let elapsed = distance_start.elapsed();
        profile.distance += elapsed;
        profile.backend_compute_wall += elapsed;
    }

    // 对每个局部点做 kNN 并存入 scratch.edges
    if scratch.row_topk.len() < leaf_size {
        scratch.row_topk.resize_with(leaf_size, RowTopK::default);
    }

    let topk_start = Instant::now();
    for (i_local, &_i_global) in leaf.iter().enumerate() {
        let row = &scratch.dmat[i_local * leaf_size..(i_local + 1) * leaf_size];
        let topk = &mut scratch.row_topk[i_local];
        topk.clear();
        for (j_local, &d) in row.iter().enumerate() {
            if j_local != i_local {
                topk.push(k, j_local, leaf[j_local], d);
            }
        }
    }
    let elapsed = topk_start.elapsed();
    profile.topk += elapsed;
    profile.backend_compute_wall += elapsed;

    let hash_total_start = Instant::now();
    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };
    let mut hash_wall = Duration::ZERO;
    let mut hash_wall_start = Instant::now();
    for (i_local, &i_global) in leaf.iter().enumerate() {
        let topk = &scratch.row_topk[i_local];
        let i_idx = i_global as usize;
        let sketch_i = sketch_cache.row(i_local);

        for (j_local, d) in topk.iter() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);

            scratch.edges.push(PendingEdge {
                p: i_idx,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                meta: HashPruneMeta::empty(),
            });
            scratch.edges.push(PendingEdge {
                p: j_global,
                c: i_global,
                hash: hash_ji,
                dist: d,
                meta: HashPruneMeta::empty(),
            });
        }

        if scratch.edges.len() >= 16384 {
            hash_wall += hash_wall_start.elapsed();
            let flush_start = Instant::now();
            let flush_profile = scratch.flush_edges(edge_sink)?;
            let elapsed = flush_start.elapsed();
            profile.record_pending_edge_flush(elapsed, flush_profile);
            hash_wall_start = Instant::now();
        }
    }
    hash_wall += hash_wall_start.elapsed();
    let elapsed = hash_total_start.elapsed();
    profile.hash += elapsed;
    profile.hash_wall += hash_wall;

    // 确保处理完当前叶子后，所有剩余的边都写入 reservoirs
    let flush_start = Instant::now();
    let flush_profile = scratch.flush_edges(edge_sink)?;
    let elapsed = flush_start.elapsed();
    profile.record_pending_edge_flush(elapsed, flush_profile);

    profile.total_wall = total_start.elapsed();
    Ok(profile)
}

pub fn process_leaf_parallel_large_profiled(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    process_leaf_parallel_large_profiled_with_backend(
        cpu_backend(),
        dataset,
        metric,
        sketches,
        leaf,
        params,
        reservoirs,
        block_rows,
    )
}

pub(crate) fn process_leaf_parallel_large_profiled_with_backend(
    compute_backend: &dyn PiPNNComputeBackend,
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    let edge_sink = ReservoirEdgeSink::new(reservoirs);
    process_leaf_parallel_large_profiled_with_backend_and_sink(
        compute_backend,
        dataset,
        metric,
        sketches,
        leaf,
        params,
        &edge_sink,
        block_rows,
    )
}

pub(crate) fn process_leaf_parallel_large_profiled_with_backend_and_sink(
    compute_backend: &dyn PiPNNComputeBackend,
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    if metric == Metric::L2 {
        return compute_backend.process_leaf_parallel_large_l2(
            dataset, sketches, leaf, params, edge_sink, block_rows,
        );
    }

    process_leaf_parallel_large_profiled_impl(
        dataset, metric, sketches, leaf, params, edge_sink, block_rows,
    )
}

pub(super) fn process_leaf_parallel_large_profiled_cpu(
    dataset: &dyn PointStore,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    process_leaf_parallel_large_profiled_impl(
        dataset,
        Metric::L2,
        sketches,
        leaf,
        params,
        edge_sink,
        block_rows,
    )
}

fn process_leaf_parallel_large_profiled_impl(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    let total_start = Instant::now();
    let leaf_size = leaf.len();
    let mut profile = LeafProfile {
        leaves: usize::from(!leaf.is_empty()),
        points: leaf_size,
        blockwise_leaves: usize::from(!leaf.is_empty()),
        ..LeafProfile::default()
    };

    if leaf_size <= 1 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    if k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    let dim = dataset.dim();
    let block_rows = block_rows.max(1);
    let mut x = Array2::<f32>::zeros((leaf_size, dim));

    let load_start = Instant::now();
    {
        let mut x_view = x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);
        let mut flat = vec![0.0f32; leaf_size * dim];
        let read_options = default_leaf_point_read_options(dataset, leaf_size);
        dataset.read_points_windowed_into_batch_stats(
            leaf,
            &mut flat,
            &read_options,
            &mut profile.io_stats,
        )?;
        for (i_local, chunk) in flat.chunks(dim).enumerate() {
            x_view
                .row_mut(i_local)
                .assign(&ndarray::ArrayView1::from(chunk));
        }
    }
    let elapsed = load_start.elapsed();
    profile.load += elapsed;
    profile.host_load_wall += elapsed;

    let x_view = x.view();
    let norms: Vec<f32> = if metric == Metric::L2 {
        x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec()
    } else {
        Vec::new()
    };

    let block_starts: Vec<usize> = (0..leaf_size).step_by(block_rows).collect();
    let wave_width = choose_parallel_block_wave_width(
        leaf_size,
        block_rows,
        params.leaf_knn,
        block_starts.len(),
    );

    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };

    for wave in block_starts.chunks(wave_width) {
        let mut results: Vec<_> = wave
            .par_iter()
            .map(|&block_start| {
                process_leaf_parallel_large_block(
                    dataset,
                    metric,
                    &sketch_cache,
                    leaf,
                    params,
                    &x_view,
                    &norms,
                    block_start,
                    block_rows,
                    k,
                )
            })
            .collect::<AnnResult<Vec<_>>>()?;
        results.sort_unstable_by_key(|result| result.block_start);

        for mut result in results {
            let flush_start = Instant::now();
            let flush_profile = flush_pending_edges(&mut result.edges, edge_sink)?;
            let elapsed = flush_start.elapsed();
            result
                .profile
                .record_pending_edge_flush(elapsed, flush_profile);
            profile.merge(result.profile);
        }
    }

    profile.total_wall = total_start.elapsed();
    Ok(profile)
}

const BLOCK_SIZE: usize = 1_024;

pub(crate) fn choose_parallel_block_wave_width(
    leaf_size: usize,
    block_rows: usize,
    leaf_knn: usize,
    total_blocks: usize,
) -> usize {
    const TARGET_BLOCK_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

    let bytes_per_block = block_rows
        .saturating_mul(leaf_size)
        .saturating_mul(size_of::<f32>())
        .saturating_add(
            block_rows
                .saturating_mul(leaf_knn.max(1))
                .saturating_mul(2)
                .saturating_mul(size_of::<PendingEdge>()),
        );
    let by_memory = if bytes_per_block == 0 {
        rayon::current_num_threads().max(1)
    } else {
        cmp::max(1, TARGET_BLOCK_SCRATCH_BYTES / bytes_per_block)
    };

    cmp::max(
        1,
        total_blocks.min(rayon::current_num_threads().max(1).min(by_memory)),
    )
}

pub(crate) fn process_leaf_parallel_large_block(
    dataset: &dyn PointStore,
    metric: Metric,
    sketch_cache: &LeafSketchCache,
    leaf: &[u32],
    params: &PiPNNParams,
    x_view: &ArrayView2<'_, f32>,
    norms: &[f32],
    block_start: usize,
    block_rows: usize,
    k: usize,
) -> AnnResult<ParallelBlockResult> {
    let leaf_size = leaf.len();
    let block_end = (block_start + block_rows).min(leaf_size);
    let block_size = block_end - block_start;
    let dim = dataset.dim();
    let mut profile = LeafProfile::default();
    let mut row_topk = vec![RowTopK::default(); block_size];

    let distance_start = Instant::now();
    if metric == Metric::L2 {
        let mut block_dmat = vec![0.0f32; block_size * leaf_size];
        let x_block = x_view.slice(ndarray::s![block_start..block_end, 0..dim]);
        let mut dmat_view =
            ArrayViewMut2::from_shape((block_size, leaf_size), &mut block_dmat).unwrap();
        ndarray::linalg::general_mat_mul(1.0, &x_block, &x_view.t(), 0.0, &mut dmat_view);

        for i_block in 0..block_size {
            let i_local = block_start + i_block;
            let norm_i = norms[i_local];
            let topk = &mut row_topk[i_block];
            let row = &block_dmat[i_block * leaf_size..(i_block + 1) * leaf_size];

            for (j_local, &gram_ij) in row.iter().enumerate() {
                if i_local == j_local {
                    continue;
                }

                let mut dist2 = norm_i + norms[j_local] - 2.0 * gram_ij;
                if dist2 <= 0.0 {
                    dist2 = 0.0;
                }
                topk.push(k, j_local, leaf[j_local], dist2);
            }
        }
    } else {
        for i_local in block_start..block_end {
            let i_global = leaf[i_local];
            let topk = &mut row_topk[i_local - block_start];
            for (j_local, &j_global) in leaf.iter().enumerate() {
                if i_local == j_local {
                    continue;
                }
                let d = dataset.get_distance(i_global, j_global, metric)?;
                topk.push(k, j_local, j_global, d);
            }
        }
    }
    let elapsed = distance_start.elapsed();
    profile.distance += elapsed;
    profile.backend_compute_wall += elapsed;

    let hash_start = Instant::now();
    let mut edges = Vec::with_capacity(block_size.saturating_mul(k.max(1)).saturating_mul(2));
    for i_local in block_start..block_end {
        let i_idx = leaf[i_local] as usize;
        let sketch_i = sketch_cache.row(i_local);

        for (j_local, d) in row_topk[i_local - block_start].iter() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);

            edges.push(PendingEdge {
                p: i_idx,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                meta: HashPruneMeta::empty(),
            });
            edges.push(PendingEdge {
                p: j_global,
                c: leaf[i_local],
                hash: hash_ji,
                dist: d,
                meta: HashPruneMeta::empty(),
            });
        }
    }
    let elapsed = hash_start.elapsed();
    profile.hash += elapsed;
    profile.hash_wall += elapsed;

    Ok(ParallelBlockResult {
        block_start,
        profile,
        edges,
    })
}

pub(crate) fn compute_leaf_topk_l2_cpu_shadow(
    dataset: &dyn PointStore,
    leaf: &[u32],
    params: &PiPNNParams,
) -> AnnResult<(CompactTopKBatch, LeafProfile)> {
    let total_start = Instant::now();
    let leaf_size = leaf.len();
    let mut profile = LeafProfile {
        leaves: usize::from(!leaf.is_empty()),
        points: leaf_size,
        ..LeafProfile::default()
    };
    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    let mut batch = CompactTopKBatch {
        point_ids: leaf.to_vec(),
        rows: leaf_size,
        k,
        topk_ids: vec![u32::MAX; leaf_size.saturating_mul(k)],
        topk_dists: vec![f32::INFINITY; leaf_size.saturating_mul(k)],
        estimated_dtoh_bytes: leaf_size
            .saturating_mul(k)
            .saturating_mul(size_of::<u32>().saturating_add(size_of::<f32>())),
        estimated_lune_witness_dtoh_bytes: 0,
        ..CompactTopKBatch::default()
    };

    if leaf_size <= 1 || k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok((batch, profile));
    }

    let dim = dataset.dim();
    let mut x = vec![0.0f32; leaf_size.saturating_mul(dim)];

    let load_start = Instant::now();
    {
        let read_options = default_leaf_point_read_options(dataset, leaf_size);
        dataset.read_points_windowed_into_batch_stats(
            leaf,
            &mut x,
            &read_options,
            &mut profile.io_stats,
        )?;
    }
    let elapsed = load_start.elapsed();
    profile.load += elapsed;
    profile.host_load_wall += elapsed;

    let distance_start = Instant::now();
    let norms: Vec<f32> = x
        .chunks(dim)
        .map(|row| row.iter().map(|&v| v * v).sum())
        .collect();
    let elapsed = distance_start.elapsed();
    profile.distance += elapsed;
    profile.backend_compute_wall += elapsed;

    let topk_start = Instant::now();
    for i_local in 0..leaf_size {
        let norm_i = norms[i_local];
        let mut topk = RowTopK::default();
        let row_vec = &x[i_local * dim..(i_local + 1) * dim];
        for j_local in 0..leaf_size {
            if i_local == j_local {
                continue;
            }
            let col_vec = &x[j_local * dim..(j_local + 1) * dim];
            let dot = row_vec
                .iter()
                .zip(col_vec.iter())
                .fold(0.0f32, |acc, (&left, &right)| acc + left * right);
            let mut dist2 = norm_i + norms[j_local] - 2.0 * dot;
            if dist2 <= 0.0 {
                dist2 = 0.0;
            }
            topk.push(k, j_local, leaf[j_local], dist2);
        }
        let out_offset = i_local * k;
        for (rank, (j_local, dist)) in topk.iter().enumerate() {
            batch.topk_ids[out_offset + rank] = leaf[j_local];
            batch.topk_dists[out_offset + rank] = dist;
        }
    }
    maybe_derive_compact_lune_witnesses_from_topk(&mut batch, params)?;
    let elapsed = topk_start.elapsed();
    profile.topk += elapsed;
    profile.backend_compute_wall += elapsed;

    profile.total_wall = total_start.elapsed();
    Ok((batch, profile))
}

pub(crate) fn flush_compact_topk_batch(
    batch: &CompactTopKBatch,
    sketches: &dyn SketchAccessor,
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
) -> AnnResult<LeafProfile> {
    let policy = edge_sink.hashprune_evidence_policy();
    flush_compact_topk_batch_with_policy(batch, sketches, params, edge_sink, policy)
}

fn flush_compact_topk_batch_with_policy(
    batch: &CompactTopKBatch,
    sketches: &dyn SketchAccessor,
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    policy: HashPruneEvidencePolicy,
) -> AnnResult<LeafProfile> {
    let mut edges = Vec::with_capacity(batch.rows.saturating_mul(batch.k.max(1)).saturating_mul(2));
    let mut profile = flush_compact_topk_batch_into_edges(
        batch,
        sketches,
        params,
        edge_sink,
        policy,
        &mut edges,
        COMPACT_TOPK_EDGE_FLUSH_THRESHOLD,
    )?;

    let flush_start = Instant::now();
    let flush_profile = flush_pending_edges(&mut edges, edge_sink)?;
    let elapsed = flush_start.elapsed();
    profile.record_pending_edge_flush(elapsed, flush_profile);
    profile.total_wall += elapsed;
    Ok(profile)
}

pub(crate) fn flush_compact_topk_batches(
    batches: &[CompactTopKBatch],
    sketches: &dyn SketchAccessor,
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
) -> AnnResult<LeafProfile> {
    let policy = edge_sink.hashprune_evidence_policy();
    flush_compact_topk_batches_with_policy(batches, sketches, params, edge_sink, policy)
}

fn flush_compact_topk_batches_with_policy(
    batches: &[CompactTopKBatch],
    sketches: &dyn SketchAccessor,
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    policy: HashPruneEvidencePolicy,
) -> AnnResult<LeafProfile> {
    let mut profile = LeafProfile::default();
    let mut edges = Vec::new();
    for batch in batches {
        let batch_profile = flush_compact_topk_batch_into_edges(
            batch,
            sketches,
            params,
            edge_sink,
            policy,
            &mut edges,
            COMPACT_TOPK_SUPERBATCH_EDGE_FLUSH_THRESHOLD,
        )?;
        profile.merge(batch_profile);
    }
    let flush_start = Instant::now();
    let flush_profile = flush_pending_edges(&mut edges, edge_sink)?;
    let elapsed = flush_start.elapsed();
    profile.record_pending_edge_flush(elapsed, flush_profile);
    profile.total_wall += elapsed;
    Ok(profile)
}

fn flush_compact_topk_batch_into_edges(
    batch: &CompactTopKBatch,
    sketches: &dyn SketchAccessor,
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    evidence_policy: HashPruneEvidencePolicy,
    edges: &mut Vec<PendingEdge>,
    edge_flush_threshold: usize,
) -> AnnResult<LeafProfile> {
    let total_start = Instant::now();
    let mut profile = LeafProfile {
        leaves: usize::from(batch.rows > 0),
        points: batch.rows,
        ..LeafProfile::default()
    };

    if batch.rows == 0 || batch.k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }
    if batch.point_ids.len() != batch.rows
        || batch.topk_ids.len() != batch.rows.saturating_mul(batch.k)
        || batch.topk_dists.len() != batch.rows.saturating_mul(batch.k)
        || !compact_lune_witness_slots_are_valid(batch)
    {
        return Err(AnnError::log_index_error(format!(
            "Invalid compact top-k batch shape: rows={} k={} point_ids={} topk_ids={} topk_dists={} witness_ranks={} witness_margins={}",
            batch.rows,
            batch.k,
            batch.point_ids.len(),
            batch.topk_ids.len(),
            batch.topk_dists.len(),
            batch.lune_witness_pivot_ranks.len(),
            batch.lune_witness_margins.len()
        )));
    }
    profile.compact_lune_witness_enabled =
        !batch.lune_witness_pivot_ranks.is_empty() && !batch.lune_witness_margins.is_empty();
    profile.compact_lune_witness_slots = batch.lune_witness_pivot_ranks.len();
    profile.compact_lune_witnesses = batch
        .lune_witness_pivot_ranks
        .iter()
        .filter(|&&pivot_rank| pivot_rank != COMPACT_LUNE_NO_WITNESS)
        .count();
    let compact_lune_prune_requested = effective_compact_lune_prune_enabled(params);
    let lune_prune_enabled = compact_lune_prune_requested
        && !batch.lune_witness_pivot_ranks.is_empty()
        && !batch.lune_witness_margins.is_empty();
    if evidence_policy.annotates() && compact_lune_prune_requested {
        return Err(AnnError::log_index_error(
            "HashPrune evidence annotation is incompatible with compact lune pruning".to_string(),
        ));
    }
    let lune_prune_protected_core = if lune_prune_enabled {
        effective_compact_lune_prune_protected_core(params).min(batch.k)
    } else {
        0
    };
    profile.compact_lune_prune_enabled = lune_prune_enabled;
    profile.compact_lune_prune_protected_core = lune_prune_protected_core;

    let local_index = compact_local_index(batch);
    let (evidence_slots, evidence_profile) =
        annotate_compact_topk_evidence(batch, &local_index, evidence_policy)?;
    profile.hashprune_evidence = evidence_profile;

    let hash_total_start = Instant::now();
    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, &batch.point_ids)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += batch.point_ids.len();
        cache
    };
    let mut hash_wall = Duration::ZERO;
    let mut hash_wall_start = Instant::now();
    edges.reserve(batch.rows.saturating_mul(batch.k.max(1)).saturating_mul(2));
    for (i_local, &i_global) in batch.point_ids.iter().enumerate() {
        let i_idx = i_global as usize;
        let sketch_i = sketch_cache.row(i_local);
        let offset = i_local * batch.k;
        let mut row_pruned = false;
        for rank in 0..batch.k {
            let j_global_u32 = batch.topk_ids[offset + rank];
            let d = batch.topk_dists[offset + rank];
            if j_global_u32 == u32::MAX || !d.is_finite() {
                continue;
            }
            if lune_prune_enabled
                && compact_lune_prune_candidate(batch, i_local, rank, lune_prune_protected_core)
            {
                profile.compact_lune_pruned_slots += 1;
                row_pruned = true;
                continue;
            }
            let (j_local, forward_meta, reverse_meta) =
                if let Some(evidence_slots) = &evidence_slots {
                    let evidence = evidence_slots[offset + rank];
                    (
                        evidence.target_local as usize,
                        evidence.forward_meta,
                        evidence.reverse_meta,
                    )
                } else {
                    let Some(j_local) = local_index.lookup(j_global_u32) else {
                        return Err(AnnError::log_index_error(format!(
                            "Compact top-k batch references point {j_global_u32} outside leaf {}",
                            batch.leaf_id
                        )));
                    };
                    (j_local, HashPruneMeta::empty(), HashPruneMeta::empty())
                };
            let j_global = j_global_u32 as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);

            edges.push(PendingEdge {
                p: i_idx,
                c: j_global_u32,
                hash: hash_ij,
                dist: d,
                meta: forward_meta,
            });
            edges.push(PendingEdge {
                p: j_global,
                c: i_global,
                hash: hash_ji,
                dist: d,
                meta: reverse_meta,
            });

            if edges.len() >= edge_flush_threshold.max(1) {
                hash_wall += hash_wall_start.elapsed();
                let flush_start = Instant::now();
                let flush_profile = flush_pending_edges(edges, edge_sink)?;
                let elapsed = flush_start.elapsed();
                profile.record_pending_edge_flush(elapsed, flush_profile);
                hash_wall_start = Instant::now();
            }
        }
        profile.compact_lune_pruned_rows += usize::from(row_pruned);
    }
    hash_wall += hash_wall_start.elapsed();
    let elapsed = hash_total_start.elapsed();
    profile.hash += elapsed;
    profile.hash_wall += hash_wall;
    profile.total_wall = total_start.elapsed();
    Ok(profile)
}

pub(crate) fn flush_compact_topk_batch_lrpf(
    batch: &CompactTopKBatch,
    params: &PiPNNParams,
    fragment_sink: &dyn LrpfFragmentSink,
) -> AnnResult<LeafProfile> {
    let total_start = Instant::now();
    let mut profile = LeafProfile {
        leaves: usize::from(batch.rows > 0),
        points: batch.rows,
        ..LeafProfile::default()
    };

    if params.prune_backend != PiPNNPruneBackend::LRPF {
        return Err(AnnError::log_index_error(
            "LRPF compact flush requires prune_backend=lrpf".to_string(),
        ));
    }
    if batch.rows == 0 || batch.k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }
    if batch.point_ids.len() != batch.rows
        || batch.topk_ids.len() != batch.rows.saturating_mul(batch.k)
        || batch.topk_dists.len() != batch.rows.saturating_mul(batch.k)
        || !compact_lune_witness_slots_are_valid(batch)
    {
        return Err(AnnError::log_index_error(format!(
            "Invalid compact top-k batch shape for LRPF: rows={} k={} point_ids={} topk_ids={} topk_dists={} witness_ranks={} witness_margins={}",
            batch.rows,
            batch.k,
            batch.point_ids.len(),
            batch.topk_ids.len(),
            batch.topk_dists.len(),
            batch.lune_witness_pivot_ranks.len(),
            batch.lune_witness_margins.len()
        )));
    }

    profile.compact_lune_witness_enabled =
        !batch.lune_witness_pivot_ranks.is_empty() && !batch.lune_witness_margins.is_empty();
    profile.compact_lune_witness_slots = batch.lune_witness_pivot_ranks.len();
    profile.compact_lune_witnesses = batch
        .lune_witness_pivot_ranks
        .iter()
        .filter(|&&pivot_rank| pivot_rank != COMPACT_LUNE_NO_WITNESS)
        .count();

    #[derive(Default)]
    struct LrpfFlushRow {
        candidates: Vec<LrpfCandidate>,
        witnesses: Vec<LrpfWitness>,
    }

    let local_by_point: HashMap<u32, usize> = batch
        .point_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, point_id)| (point_id, idx))
        .collect();
    let mut rows: Vec<LrpfFlushRow> = (0..batch.rows).map(|_| LrpfFlushRow::default()).collect();

    let flush_start = Instant::now();
    for (row, &src) in batch.point_ids.iter().enumerate() {
        let row_offset = row.saturating_mul(batch.k);
        let mut rank_to_candidate_idx = vec![None; batch.k];
        for rank in 0..batch.k {
            let slot = row_offset + rank;
            let dst = batch.topk_ids[slot];
            let dist = batch.topk_dists[slot];
            if dst == u32::MAX || !dist.is_finite() {
                continue;
            }
            let Some(&dst_row) = local_by_point.get(&dst) else {
                return Err(AnnError::log_index_error(format!(
                    "LRPF compact top-k batch references point {dst} outside leaf {}",
                    batch.leaf_id
                )));
            };
            let local_idx = u16::try_from(rows[row].candidates.len()).map_err(|_| {
                AnnError::log_index_error(format!(
                    "LRPF compact flush has too many candidates for src={src}"
                ))
            })?;
            rank_to_candidate_idx[rank] = Some(local_idx);
            rows[row].candidates.push(LrpfCandidate::new(dst, dist));
            rows[dst_row].candidates.push(LrpfCandidate::new(src, dist));
        }

        if profile.compact_lune_witness_enabled {
            for victim_rank in 0..batch.k {
                let slot = row_offset + victim_rank;
                let pivot_rank = batch.lune_witness_pivot_ranks[slot];
                let margin = batch.lune_witness_margins[slot];
                if pivot_rank == COMPACT_LUNE_NO_WITNESS || margin <= 0.0 {
                    continue;
                }
                let pivot_rank = pivot_rank as usize;
                let Some(Some(pivot_idx)) = rank_to_candidate_idx.get(pivot_rank).copied() else {
                    continue;
                };
                let Some(victim_idx) = rank_to_candidate_idx[victim_rank] else {
                    continue;
                };
                rows[row]
                    .witnesses
                    .push(LrpfWitness::new(pivot_idx, victim_idx, margin));
            }
        }
    }

    for (row, &src) in batch.point_ids.iter().enumerate() {
        let candidates = &rows[row].candidates;
        let witnesses = &rows[row].witnesses;
        if params.lrpf_pre_spill_compress_enable {
            let pre_spill_width = effective_lrpf_pre_spill_width(params);
            if candidates.len() > pre_spill_width {
                let compressed = compress_lrpf_row_before_spill(
                    candidates,
                    witnesses,
                    LrpfConfig {
                        degree: params.l_max,
                        protected_core: params.lrpf_protected_core,
                        pre_spill_width,
                        long_edge_quota: params.lrpf_long_edge_quota,
                    },
                );
                profile.lrpf_pre_spill_pruned_candidates = profile
                    .lrpf_pre_spill_pruned_candidates
                    .saturating_add(candidates.len().saturating_sub(compressed.candidates.len()));
                fragment_sink.push_lrpf_fragment(
                    src,
                    &compressed.candidates,
                    &compressed.witnesses,
                )?;
                continue;
            }
        }

        fragment_sink.push_lrpf_fragment(src, candidates, witnesses)?;
    }
    let elapsed = flush_start.elapsed();
    profile.flush += elapsed;
    profile.flush_wall += elapsed;
    profile.edge_sink_wall += elapsed;
    profile.total_wall = total_start.elapsed();
    Ok(profile)
}

pub(crate) fn effective_lrpf_pre_spill_width(params: &PiPNNParams) -> usize {
    let l_max = params.l_max.max(1);
    let scaled = ((l_max as f32) * params.lrpf_pre_spill_width_multiplier).ceil() as usize;
    scaled.clamp(l_max, l_max.saturating_mul(4))
}

pub(crate) fn effective_lrpf_target_degree(params: &PiPNNParams) -> usize {
    params.lrpf_target_degree.max(1).min(params.l_max.max(1))
}

pub(crate) fn flush_pending_edges(
    edges: &mut Vec<PendingEdge>,
    edge_sink: &dyn PendingEdgeSink,
) -> AnnResult<PendingEdgeFlushProfile> {
    edge_sink.flush_pending_edges(edges)
}

trait ReservoirRowAccess {
    fn with_reservoir<T>(
        &self,
        source: usize,
        measure_lock: bool,
        update: impl FnOnce(&mut HashPruneReservoir) -> AnnResult<T>,
    ) -> AnnResult<(T, Duration)>;
}

struct LegacyReservoirAccess<'a> {
    reservoirs: &'a [Mutex<HashPruneReservoir>],
}

impl ReservoirRowAccess for LegacyReservoirAccess<'_> {
    fn with_reservoir<T>(
        &self,
        source: usize,
        measure_lock: bool,
        update: impl FnOnce(&mut HashPruneReservoir) -> AnnResult<T>,
    ) -> AnnResult<(T, Duration)> {
        let reservoir = self.reservoirs.get(source).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "source {source} exceeds resident reservoir length {}",
                self.reservoirs.len()
            ))
        })?;
        let lock_started = measure_lock.then(Instant::now);
        let mut reservoir = reservoir.lock();
        let lock_wall = lock_started.map_or(Duration::ZERO, |started| started.elapsed());
        let value = update(&mut reservoir)?;
        Ok((value, lock_wall))
    }
}

impl ReservoirRowAccess for FlatResidentReservoirStore {
    fn with_reservoir<T>(
        &self,
        source: usize,
        measure_lock: bool,
        update: impl FnOnce(&mut HashPruneReservoir) -> AnnResult<T>,
    ) -> AnnResult<(T, Duration)> {
        self.with_reservoir_profiled(source, measure_lock, update)
    }
}

pub(crate) fn flush_pending_edges_to_reservoirs(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
) -> PendingEdgeFlushProfile {
    let access = LegacyReservoirAccess { reservoirs };
    flush_pending_edges_to_access_impl(edges, &access, edge_sink_microscope_enabled())
        .expect("legacy resident reservoir access must remain in bounds")
}

#[inline]
fn pending_edge_report_pair_key(edge: &PendingEdge) -> u64 {
    debug_assert!(edge.p <= u32::MAX as usize);
    ((edge.p as u64) << 32) | u64::from(edge.c)
}

pub(crate) fn prepare_pending_edges_for_sink(
    edges: &mut Vec<PendingEdge>,
    policy: HashPruneEvidencePolicy,
) -> AnnResult<HashPruneEvidenceProfile> {
    if policy.is_off() {
        edges.sort_unstable_by_key(|edge| edge.p);
        return Ok(HashPruneEvidenceProfile::default());
    }
    if policy.enforces() {
        for edge in edges.iter() {
            edge.meta.to_spill_flags()?;
        }
    }

    let mut profile = HashPruneEvidenceProfile {
        candidates_before_coalesce: edges.len(),
        ..HashPruneEvidenceProfile::default()
    };

    let sort_start = Instant::now();
    if policy.enforces() {
        edges.sort_unstable_by(|left, right| {
            left.p.cmp(&right.p).then_with(|| left.c.cmp(&right.c))
        });
    } else {
        edges.sort_unstable_by_key(pending_edge_report_pair_key);
    }
    profile.coalesce_sort_wall = sort_start.elapsed();

    let merge_start = Instant::now();
    let mut read = 0usize;
    let mut write = 0usize;
    while read < edges.len() {
        let mut merged = edges[read].clone();
        read += 1;
        while read < edges.len() && edges[read].p == merged.p && edges[read].c == merged.c {
            let duplicate = &edges[read];
            if duplicate.hash != merged.hash {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune duplicate hash mismatch for source {} destination {}: left={} right={}",
                    merged.p, merged.c, merged.hash, duplicate.hash
                )));
            }
            if duplicate.dist.total_cmp(&merged.dist).is_lt() {
                merged.dist = duplicate.dist;
            }
            // PendingEdge metadata is typed by compact preparation; serialized spill flags cross
            // HashPruneMeta::from_spill_flags before they can reach this coalescing path.
            merged.meta = merged.meta.merge_prevalidated_candidate(duplicate.meta);
            read += 1;
        }

        if merged.meta.evidence().is_none() {
            profile.unannotated_candidates += 1;
            if policy.enforces() {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune enforce received an unannotated candidate for source {} destination {}",
                    merged.p, merged.c
                )));
            }
        }
        edges[write] = merged;
        write += 1;
    }
    edges.truncate(write);
    profile.coalesce_merge_wall = merge_start.elapsed();
    profile.candidates_after_coalesce = write;
    profile.duplicate_records_removed = profile.candidates_before_coalesce.saturating_sub(write);
    profile.coalesce_wall = profile.coalesce_sort_wall + profile.coalesce_merge_wall;
    Ok(profile)
}

pub(crate) fn flush_pending_edges_to_reservoirs_with_policy(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
    policy: HashPruneEvidencePolicy,
) -> AnnResult<PendingEdgeFlushProfile> {
    let access = LegacyReservoirAccess { reservoirs };
    flush_pending_edges_to_access_with_policy(
        edges,
        &access,
        policy,
        edge_sink_microscope_enabled(),
    )
}

fn flush_pending_edges_to_access_with_policy<A: ReservoirRowAccess>(
    edges: &mut Vec<PendingEdge>,
    access: &A,
    policy: HashPruneEvidencePolicy,
    microscope: bool,
) -> AnnResult<PendingEdgeFlushProfile> {
    if policy.is_off() {
        return flush_pending_edges_to_access_impl(edges, access, microscope);
    }
    flush_pending_edges_to_access_with_policy_core(edges, access, policy, microscope)
}

pub(crate) fn flush_pending_edges_to_reservoirs_with_policies(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
) -> AnnResult<PendingEdgeFlushProfile> {
    flush_pending_edges_to_reservoirs_with_policies_and_diagnostics(
        edges,
        reservoirs,
        evidence_policy,
        backfill_policy,
        None,
    )
}

pub(crate) fn flush_pending_edges_to_reservoirs_with_policies_and_diagnostics(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostics: Option<&HashPruneTraceCollector>,
) -> AnnResult<PendingEdgeFlushProfile> {
    let access = LegacyReservoirAccess { reservoirs };
    flush_pending_edges_to_access_with_policies_and_diagnostics(
        edges,
        &access,
        evidence_policy,
        backfill_policy,
        diagnostics,
    )
}

fn flush_pending_edges_to_access_with_policies_and_diagnostics<A: ReservoirRowAccess>(
    edges: &mut Vec<PendingEdge>,
    access: &A,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostics: Option<&HashPruneTraceCollector>,
) -> AnnResult<PendingEdgeFlushProfile> {
    if backfill_policy.is_off() {
        if diagnostics.is_some() {
            return Err(AnnError::log_index_error(
                "HashPrune diagnostics require an enabled named backfill policy".to_string(),
            ));
        }
        return flush_pending_edges_to_access_with_policy(
            edges,
            access,
            evidence_policy,
            edge_sink_microscope_enabled(),
        );
    }
    if !evidence_policy.is_off() {
        return Err(AnnError::log_index_error(
            "distance-gated backfill cannot share the Evidence-Aware sink path".to_string(),
        ));
    }
    if edges.is_empty() {
        return Ok(PendingEdgeFlushProfile::default());
    }

    let edge_count = edges.len();
    let sort_start = Instant::now();
    edges.sort_unstable_by_key(|edge| edge.p);
    let sort_wall = sort_start.elapsed();
    let insert_start = Instant::now();
    let mut backfill = HashPruneBackfillProfile::default();
    let mut i = 0usize;
    while i < edges.len() {
        let source = edges[i].p;
        let group_start = i;
        while i < edges.len() && edges[i].p == source {
            i += 1;
        }
        access.with_reservoir(source, false, |reservoir| {
            for edge in &edges[group_start..i] {
                if let Some(collector) = diagnostics {
                    collector.record(source, edge.c as usize, edge.hash, edge.dist)?;
                }
                reservoir.insert_backfill_prevalidated_with_policy(
                    source,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    backfill_policy,
                    &mut backfill,
                )?;
            }
            Ok(())
        })?;
    }
    let insert_wall = insert_start.elapsed();
    edges.clear();
    Ok(PendingEdgeFlushProfile {
        edges: edge_count,
        sort_wall,
        insert_wall,
        flushes: 1,
        backfill,
        ..PendingEdgeFlushProfile::default()
    })
}

fn flush_pending_edges_to_reservoirs_with_policy_core(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
    policy: HashPruneEvidencePolicy,
    microscope: bool,
) -> AnnResult<PendingEdgeFlushProfile> {
    let access = LegacyReservoirAccess { reservoirs };
    flush_pending_edges_to_access_with_policy_core(edges, &access, policy, microscope)
}

fn flush_pending_edges_to_access_with_policy_core<A: ReservoirRowAccess>(
    edges: &mut Vec<PendingEdge>,
    access: &A,
    policy: HashPruneEvidencePolicy,
    microscope: bool,
) -> AnnResult<PendingEdgeFlushProfile> {
    if policy.is_off() {
        return flush_pending_edges_to_access_impl(edges, access, microscope);
    }
    if edges.is_empty() {
        return Ok(PendingEdgeFlushProfile::default());
    }

    let edge_count = edges.len();
    let mut evidence = prepare_pending_edges_for_sink(edges, policy)?;
    let sort_wall = evidence.coalesce_wall;
    let insert_start = Instant::now();
    let mut lock_wall = Duration::ZERO;
    let mut insert_body_wall = Duration::ZERO;
    let mut groups = 0usize;
    let mut max_group_edges = 0usize;
    let mut insert_profile = HashPruneInsertProfile::default();
    let canonical_fast_report = !policy.is_off() && !policy.enforces() && !microscope;
    if canonical_fast_report {
        insert_profile.candidates = insert_profile.candidates.saturating_add(edges.len());
    }
    let mut i = 0usize;
    while i < edges.len() {
        let group_start = i;
        let p_idx = edges[i].p;
        while i < edges.len() && edges[i].p == p_idx {
            i += 1;
        }
        if microscope {
            groups += 1;
            max_group_edges = max_group_edges.max(i - group_start);
        }

        let ((body_wall, ()), group_lock_wall) =
            access.with_reservoir(p_idx, microscope, |reservoir| {
                let body_start = microscope.then(Instant::now);
                if canonical_fast_report {
                    for edge in &edges[group_start..i] {
                        reservoir.insert_report_prevalidated(
                            p_idx, edge.c, edge.hash, edge.dist, edge.meta,
                        );
                    }
                } else {
                    for edge in &edges[group_start..i] {
                        reservoir.insert_with_policy_profiled(
                            p_idx,
                            edge.c,
                            edge.hash,
                            edge.dist,
                            edge.meta,
                            policy,
                            &mut insert_profile,
                        )?;
                    }
                }
                Ok((
                    body_start.map_or(Duration::ZERO, |started| started.elapsed()),
                    (),
                ))
            })?;
        if microscope {
            lock_wall += group_lock_wall;
            insert_body_wall += body_wall;
        }
    }
    let insert_wall = insert_start.elapsed();
    evidence.merge(std::mem::take(&mut insert_profile.evidence));
    edges.clear();
    Ok(PendingEdgeFlushProfile {
        edges: edge_count,
        sort_wall,
        insert_wall,
        lock_wall,
        insert_body_wall,
        flushes: 1,
        groups,
        max_group_edges,
        insert_profile,
        evidence,
        backfill: HashPruneBackfillProfile::default(),
    })
}

#[cfg(test)]
pub(crate) fn flush_pending_edges_to_reservoirs_microscope(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
) -> PendingEdgeFlushProfile {
    let access = LegacyReservoirAccess { reservoirs };
    flush_pending_edges_to_access_impl(edges, &access, true)
        .expect("legacy resident reservoir access must remain in bounds")
}

pub(crate) fn edge_sink_microscope_enabled() -> bool {
    matches!(
        env::var("SHEAR_PIPNN_EDGE_SINK_MICROSCOPE").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn flush_pending_edges_to_access_impl<A: ReservoirRowAccess>(
    edges: &mut Vec<PendingEdge>,
    access: &A,
    microscope: bool,
) -> AnnResult<PendingEdgeFlushProfile> {
    if edges.is_empty() {
        return Ok(PendingEdgeFlushProfile::default());
    }

    let edge_count = edges.len();
    let sort_start = Instant::now();
    edges.sort_unstable_by_key(|e| e.p);
    let sort_wall = sort_start.elapsed();

    let insert_start = Instant::now();
    if !microscope {
        let mut i = 0;
        while i < edges.len() {
            let p_idx = edges[i].p;
            let group_start = i;
            while i < edges.len() && edges[i].p == p_idx {
                i += 1;
            }
            access.with_reservoir(p_idx, false, |reservoir| {
                for edge in &edges[group_start..i] {
                    reservoir.insert(edge.c, edge.hash, edge.dist);
                }
                Ok(())
            })?;
        }
        let insert_wall = insert_start.elapsed();
        edges.clear();
        return Ok(PendingEdgeFlushProfile {
            edges: edge_count,
            sort_wall,
            insert_wall,
            flushes: 1,
            ..PendingEdgeFlushProfile::default()
        });
    }

    let mut lock_wall = Duration::ZERO;
    let mut insert_body_wall = Duration::ZERO;
    let mut groups = 0usize;
    let mut max_group_edges = 0usize;
    let mut insert_profile = HashPruneInsertProfile::default();
    let mut i = 0;
    while i < edges.len() {
        let group_start = i;
        let p_idx = edges[i].p;
        while i < edges.len() && edges[i].p == p_idx {
            i += 1;
        }
        groups += 1;
        max_group_edges = max_group_edges.max(i - group_start);

        let (body_wall, group_lock_wall) = access.with_reservoir(p_idx, true, |reservoir| {
            let body_start = Instant::now();
            for edge in &edges[group_start..i] {
                reservoir.insert_profiled(edge.c, edge.hash, edge.dist, &mut insert_profile);
            }
            Ok(body_start.elapsed())
        })?;
        lock_wall += group_lock_wall;
        insert_body_wall += body_wall;
    }
    let insert_wall = insert_start.elapsed();
    edges.clear();
    Ok(PendingEdgeFlushProfile {
        edges: edge_count,
        sort_wall,
        insert_wall,
        lock_wall,
        insert_body_wall,
        flushes: 1,
        groups,
        max_group_edges,
        insert_profile,
        evidence: HashPruneEvidenceProfile::default(),
        backfill: HashPruneBackfillProfile::default(),
    })
}

fn process_leaf_blockwise(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &PiPNNParams,
    edge_sink: &dyn PendingEdgeSink,
    scratch: &mut LeafScratch,
    profile: &mut LeafProfile,
) -> AnnResult<()> {
    let leaf_size = leaf.len();
    if leaf_size <= 1 {
        return Ok(());
    }

    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    if k == 0 {
        return Ok(());
    }

    let dim = dataset.dim();
    if scratch.x.nrows() < leaf_size || scratch.x.ncols() < dim {
        scratch.x = Array2::<f32>::zeros((leaf_size.max(scratch.x.nrows()), dim));
    }

    let load_start = Instant::now();
    let mut x_view = scratch.x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);
    let mut flat = vec![0.0f32; leaf_size * dim];
    let read_options = default_leaf_point_read_options(dataset, leaf_size);
    dataset.read_points_windowed_into_batch_stats(
        leaf,
        &mut flat,
        &read_options,
        &mut profile.io_stats,
    )?;
    for (i_local, chunk) in flat.chunks(dim).enumerate() {
        x_view
            .row_mut(i_local)
            .assign(&ndarray::ArrayView1::from(chunk));
    }
    let elapsed = load_start.elapsed();
    profile.load += elapsed;
    profile.host_load_wall += elapsed;

    let distance_start = Instant::now();
    let norms: Vec<f32> = if metric == Metric::L2 {
        x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec()
    } else {
        Vec::new()
    };

    if scratch.row_topk.len() < leaf_size {
        scratch.row_topk.resize_with(leaf_size, RowTopK::default);
    }
    for row in scratch.row_topk.iter_mut().take(leaf_size) {
        row.clear();
    }

    for block_start in (0..leaf_size).step_by(BLOCK_SIZE) {
        let block_end = (block_start + BLOCK_SIZE).min(leaf_size);
        let block_size = block_end - block_start;

        if metric == Metric::L2 {
            scratch.block_dmat.clear();
            scratch.block_dmat.resize(block_size * leaf_size, 0.0f32);

            let x_block = x_view.slice(ndarray::s![block_start..block_end, 0..dim]);
            let mut dmat_view =
                ArrayViewMut2::from_shape((block_size, leaf_size), &mut scratch.block_dmat)
                    .unwrap();
            ndarray::linalg::general_mat_mul(1.0, &x_block, &x_view.t(), 0.0, &mut dmat_view);

            for i_block in 0..block_size {
                let i_local = block_start + i_block;
                let norm_i = norms[i_local];
                let row = &scratch.block_dmat[i_block * leaf_size..(i_block + 1) * leaf_size];
                let topk = &mut scratch.row_topk[i_local];

                for (j_local, &gram_ij) in row.iter().enumerate() {
                    if i_local == j_local {
                        continue;
                    }

                    let mut dist2 = norm_i + norms[j_local] - 2.0 * gram_ij;
                    if dist2 <= 0.0 {
                        dist2 = 0.0;
                    }
                    topk.push(k, j_local, leaf[j_local], dist2);
                }
            }
        } else {
            for i_local in block_start..block_end {
                let i_global = leaf[i_local];
                let topk = &mut scratch.row_topk[i_local];
                for (j_local, &j_global) in leaf.iter().enumerate() {
                    if i_local == j_local {
                        continue;
                    }
                    let d = dataset.get_distance(i_global, j_global, metric)?;
                    topk.push(k, j_local, j_global, d);
                }
            }
        }
    }
    let elapsed = distance_start.elapsed();
    profile.distance += elapsed;
    profile.backend_compute_wall += elapsed;

    scratch.edges.clear();
    let hash_total_start = Instant::now();
    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };
    let mut hash_wall = Duration::ZERO;
    let mut hash_wall_start = Instant::now();
    for (i_local, &i_global) in leaf.iter().enumerate() {
        let i_idx = i_global as usize;
        let sketch_i = sketch_cache.row(i_local);
        for (j_local, d) in scratch.row_topk[i_local].iter() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);

            scratch.edges.push(PendingEdge {
                p: i_idx,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                meta: HashPruneMeta::empty(),
            });
            scratch.edges.push(PendingEdge {
                p: j_global,
                c: i_global,
                hash: hash_ji,
                dist: d,
                meta: HashPruneMeta::empty(),
            });
        }

        if scratch.edges.len() >= 16384 {
            hash_wall += hash_wall_start.elapsed();
            let flush_start = Instant::now();
            let flush_profile = scratch.flush_edges(edge_sink)?;
            let elapsed = flush_start.elapsed();
            profile.record_pending_edge_flush(elapsed, flush_profile);
            hash_wall_start = Instant::now();
        }
    }
    hash_wall += hash_wall_start.elapsed();
    let elapsed = hash_total_start.elapsed();
    profile.hash += elapsed;
    profile.hash_wall += hash_wall;

    let flush_start = Instant::now();
    let flush_profile = scratch.flush_edges(edge_sink)?;
    let elapsed = flush_start.elapsed();
    profile.record_pending_edge_flush(elapsed, flush_profile);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::mem::size_of;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use parking_lot::Mutex;

    use super::{
        CompactTopKBatch, CompactTopKSink, HashPruneReservoir, LeafScratch, PendingEdge,
        PendingEdgeFlushProfile, PendingEdgeSink, RowTopK, compute_leaf_topk_l2_cpu_shadow,
        effective_compact_lune_witness_enabled, flush_compact_topk_batch,
        flush_compact_topk_batch_lrpf, flush_pending_edges_to_reservoirs_microscope,
        flush_pending_edges_to_reservoirs_with_policies,
        flush_pending_edges_to_reservoirs_with_policies_and_diagnostics, process_leaf,
        process_leaf_parallel_large_profiled,
    };
    use crate::algorithm::pipnn::compute_backend::cpu_backend;
    use crate::algorithm::pipnn::flat_resident_reservoir::FlatResidentReservoirStore;
    use crate::algorithm::pipnn::hash_prune::{HashPruneInsertProfile, SketchStore, Slot};
    use crate::algorithm::pipnn::hash_prune_backfill::{
        HashPruneBackfillPolicy, HashPruneBackfillProfile,
    };
    use crate::algorithm::pipnn::hash_prune_diagnostics::{
        DiagnosticCandidate, HashPruneTraceCollector, source_is_sampled,
    };
    use crate::algorithm::pipnn::hash_prune_evidence::{
        HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneEvidenceProfile, HashPruneMeta,
    };
    use crate::algorithm::pipnn::lrpf::LrpfCandidate;
    use crate::algorithm::pipnn::lrpf_spill::{LrpfFragment, LrpfFragmentSink};
    use crate::algorithm::pipnn::point_store::InmemDatasetPointStore;
    use crate::algorithm::pipnn::{
        PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNParams, PiPNNPruneBackend,
    };
    use crate::common::{AnnError, AnnResult, Metric};
    use crate::model::InmemDataset;

    fn build_test_dataset(num_points: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        for point in 0..num_points {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = point as f32 * 0.25 + axis as f32 * 0.5;
            }
        }
        dataset
    }

    #[test]
    fn flat_sink_matches_legacy_sink_across_flushes() {
        let reservoirs = (0..4)
            .map(|_| Mutex::new(HashPruneReservoir::new(4)))
            .collect::<Vec<_>>();
        let flat = FlatResidentReservoirStore::new_for_test(4, 4, 2, 2).unwrap();
        let legacy_sink = super::ReservoirEdgeSink::new(&reservoirs);
        let flat_sink = super::FlatReservoirEdgeSink::new(&flat);
        let batches = [
            vec![
                PendingEdge {
                    p: 2,
                    c: 21,
                    hash: 3,
                    dist: 5.0,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 0,
                    c: 11,
                    hash: 2,
                    dist: 4.0,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 3.0,
                    meta: HashPruneMeta::empty(),
                },
            ],
            vec![
                PendingEdge {
                    p: 0,
                    c: 12,
                    hash: 2,
                    dist: 2.0,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 0,
                    c: 11,
                    hash: 2,
                    dist: 3.5,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 2,
                    c: 22,
                    hash: 4,
                    dist: 1.0,
                    meta: HashPruneMeta::empty(),
                },
            ],
        ];

        for batch in batches {
            let mut legacy_edges = batch.clone();
            let mut flat_edges = batch;
            let legacy_profile = legacy_sink.flush_pending_edges(&mut legacy_edges).unwrap();
            let flat_profile = flat_sink.flush_pending_edges(&mut flat_edges).unwrap();
            assert_eq!(flat_profile.edges, legacy_profile.edges);
            assert_eq!(flat_profile.flushes, legacy_profile.flushes);
        }

        for source in 0..4 {
            let legacy = reservoirs[source].lock();
            let flat_slots = flat.snapshot_slots(source).unwrap();
            let keys = |slots: &[Slot]| {
                slots
                    .iter()
                    .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(keys(&flat_slots), keys(legacy.slots()));
        }
    }

    #[test]
    fn flat_sink_preserves_backfill_state_across_flushes() {
        let mut reservoirs = vec![Mutex::new(HashPruneReservoir::new(4))];
        let flat = FlatResidentReservoirStore::new_for_test(1, 4, 1, 1).unwrap();
        let policy = HashPruneBackfillPolicy::distance_gated(8, 1.1);
        let legacy_sink = super::ReservoirEdgeSink::new_with_policies(
            &reservoirs,
            HashPruneEvidencePolicy::off(),
            policy,
        );
        let flat_sink = super::FlatReservoirEdgeSink::new_with_policies(
            &flat,
            HashPruneEvidencePolicy::off(),
            policy,
        );
        let batches = [
            vec![
                PendingEdge {
                    p: 0,
                    c: 101,
                    hash: 10,
                    dist: 1.5,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 0,
                    c: 100,
                    hash: 10,
                    dist: 1.0,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 0,
                    c: 200,
                    hash: 20,
                    dist: 2.0,
                    meta: HashPruneMeta::empty(),
                },
            ],
            vec![
                PendingEdge {
                    p: 0,
                    c: 102,
                    hash: 10,
                    dist: 1.2,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 0,
                    c: 99,
                    hash: 10,
                    dist: 0.9,
                    meta: HashPruneMeta::empty(),
                },
            ],
        ];

        for batch in batches {
            let mut legacy_edges = batch.clone();
            let mut flat_edges = batch;
            let legacy_profile = legacy_sink.flush_pending_edges(&mut legacy_edges).unwrap();
            let flat_profile = flat_sink.flush_pending_edges(&mut flat_edges).unwrap();
            assert_eq!(flat_profile.backfill, legacy_profile.backfill);
        }
        drop(legacy_sink);

        let mut legacy_profile = HashPruneBackfillProfile::default();
        let legacy_row = reservoirs
            .pop()
            .unwrap()
            .into_inner()
            .into_finalized_slots_with_backfill(0, policy, &mut legacy_profile)
            .unwrap();
        let mut flat_profile = HashPruneBackfillProfile::default();
        let flat_row = flat
            .finalize_row_with_logical_source(0, 0, policy, &mut flat_profile)
            .unwrap();
        let signature = |slots: &[Slot]| {
            slots
                .iter()
                .map(|slot| {
                    (
                        slot.target,
                        slot.hash,
                        slot.dist.to_bits(),
                        slot.meta.to_spill_flags().unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(flat_row.primary_count, legacy_row.primary_count);
        assert_eq!(signature(&flat_row.slots), signature(&legacy_row.slots));
        assert_eq!(flat_profile, legacy_profile);
    }

    #[test]
    fn diagnostic_resident_trace_matches_the_post_coalesce_insertion_order() {
        let source = (0_u64..)
            .find(|&candidate| source_is_sampled(candidate))
            .unwrap() as usize;
        let collector = HashPruneTraceCollector::new();
        let reservoirs = (0..=source)
            .map(|_| Mutex::new(HashPruneReservoir::new(8)))
            .collect::<Vec<_>>();
        let mut edges = vec![
            PendingEdge {
                p: source,
                c: 30,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: source,
                c: 10,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: source,
                c: 20,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        let mut expected_edges = edges.clone();
        expected_edges.sort_unstable_by_key(|edge| edge.p);
        let expected_trace = expected_edges
            .iter()
            .map(|edge| DiagnosticCandidate::new(edge.c, edge.hash, edge.dist))
            .collect::<Vec<_>>();
        let mut params = PiPNNParams::default();
        params.l_max = 8;
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        let soft_policy = HashPruneBackfillPolicy::from_params(&params).unwrap();
        flush_pending_edges_to_reservoirs_with_policies_and_diagnostics(
            &mut edges,
            &reservoirs,
            HashPruneEvidencePolicy::off(),
            soft_policy,
            Some(&collector),
        )
        .unwrap();
        assert_eq!(collector.take(source).unwrap(), expected_trace);
    }

    fn build_test_sketches(num_points: usize, width: usize) -> SketchStore {
        let mut data = Vec::with_capacity(num_points * width);
        for point in 0..num_points {
            for bit in 0..width {
                data.push(point as f32 * 0.1 + bit as f32);
            }
        }
        SketchStore::from_test_data(data, width)
    }

    fn collect_neighbors(reservoirs: Vec<Mutex<HashPruneReservoir>>) -> Vec<Vec<u32>> {
        reservoirs
            .into_iter()
            .map(|reservoir| {
                let mut neighbors = reservoir.into_inner().into_neighbors();
                neighbors.sort_unstable();
                neighbors
            })
            .collect()
    }

    #[test]
    fn hashprune_metadata_layout_pending_edge_is_24_bytes_on_64_bit_targets() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(size_of::<PendingEdge>(), 24);

        let edge = PendingEdge {
            p: 1,
            c: 2,
            hash: 3,
            dist: 4.0,
            meta: HashPruneMeta::mandatory(),
        };
        assert!(edge.meta.is_mandatory());
    }

    #[test]
    fn hashprune_backfill_resident_flush_bypasses_evidence_preparation() {
        let mut reservoirs = vec![Mutex::new(HashPruneReservoir::new(4))];
        let evidence = HashPruneEvidencePolicy::off();
        let backfill = HashPruneBackfillPolicy::distance_gated(8, 1.1);
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 101,
                hash: 10,
                dist: 1.5,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 100,
                hash: 10,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 200,
                hash: 20,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        let flush = flush_pending_edges_to_reservoirs_with_policies(
            &mut edges,
            &reservoirs,
            evidence,
            backfill,
        )
        .unwrap();
        assert!(edges.is_empty());
        assert_eq!(flush.edges, 3);
        assert_eq!(flush.evidence, HashPruneEvidenceProfile::default());

        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoirs
            .pop()
            .unwrap()
            .into_inner()
            .into_neighbors_with_backfill(0, backfill, &mut profile)
            .unwrap();
        assert_eq!(neighbors, vec![100, 200, 101]);
    }

    #[test]
    fn hashprune_tiered_rhalf_resident_flush_uses_policy_aware_admission() {
        let mut params = PiPNNParams::default();
        params.l_max = 6;
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf;
        params.hashprune_backfill_ablation = PiPNNHashPruneBackfillAblation::NoVeto;
        params.hashprune_backfill_alpha = 1.1;
        let policy = HashPruneBackfillPolicy::from_params(&params).unwrap();
        let mut reservoirs = vec![Mutex::new(HashPruneReservoir::new(params.l_max))];
        let mut edges = [
            (100, 10, 1.0),
            (101, 20, 1.0),
            (102, 30, 1.0),
            (200, 10, 1.1),
            (201, 20, 1.2),
            (202, 30, 1.3),
            (300, 10, 9.0),
        ]
        .into_iter()
        .map(|(c, hash, dist)| PendingEdge {
            p: 0,
            c,
            hash,
            dist,
            meta: HashPruneMeta::empty(),
        })
        .collect::<Vec<_>>();

        let flush = flush_pending_edges_to_reservoirs_with_policies(
            &mut edges,
            &reservoirs,
            HashPruneEvidencePolicy::off(),
            policy,
        )
        .unwrap();
        assert_eq!(flush.backfill.shortlist_admissions, 3);
        assert_eq!(flush.backfill.shortlist_capacity_rejections, 1);

        let mut profile = flush.backfill;
        let slots = reservoirs
            .pop()
            .unwrap()
            .into_inner()
            .into_slots_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(slots.len(), 6);
        assert_eq!(profile.shortlisted_by_layer, [3, 0, 0]);
    }

    struct SleepingEdgeSink {
        delay: Duration,
        flushes: AtomicUsize,
    }

    impl PendingEdgeSink for SleepingEdgeSink {
        fn flush_pending_edges(
            &self,
            edges: &mut Vec<PendingEdge>,
        ) -> AnnResult<PendingEdgeFlushProfile> {
            let edge_count = edges.len();
            if !edges.is_empty() {
                self.flushes.fetch_add(1, Ordering::Relaxed);
                thread::sleep(self.delay);
            }
            edges.clear();
            Ok(PendingEdgeFlushProfile {
                edges: edge_count,
                flushes: usize::from(edge_count > 0),
                ..PendingEdgeFlushProfile::default()
            })
        }
    }

    struct CountingEdgeSink {
        flushes: AtomicUsize,
        edges: Mutex<Vec<PendingEdge>>,
    }

    impl Default for CountingEdgeSink {
        fn default() -> Self {
            Self {
                flushes: AtomicUsize::new(0),
                edges: Mutex::new(Vec::new()),
            }
        }
    }

    impl PendingEdgeSink for CountingEdgeSink {
        fn flush_pending_edges(
            &self,
            edges: &mut Vec<PendingEdge>,
        ) -> AnnResult<PendingEdgeFlushProfile> {
            let edge_count = edges.len();
            if !edges.is_empty() {
                self.flushes.fetch_add(1, Ordering::Relaxed);
                self.edges.lock().extend(edges.iter().cloned());
            }
            edges.clear();
            Ok(PendingEdgeFlushProfile {
                edges: edge_count,
                flushes: usize::from(edge_count > 0),
                ..PendingEdgeFlushProfile::default()
            })
        }
    }

    #[derive(Default)]
    struct CapturingEdgeSink {
        edges: Mutex<Vec<PendingEdge>>,
        policy: HashPruneEvidencePolicy,
    }

    impl CapturingEdgeSink {
        fn with_policy(policy: HashPruneEvidencePolicy) -> Self {
            Self {
                policy,
                ..Self::default()
            }
        }
    }

    impl PendingEdgeSink for CapturingEdgeSink {
        fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
            self.policy
        }

        fn flush_pending_edges(
            &self,
            edges: &mut Vec<PendingEdge>,
        ) -> AnnResult<PendingEdgeFlushProfile> {
            let edge_count = edges.len();
            self.edges.lock().extend(edges.iter().cloned());
            edges.clear();
            Ok(PendingEdgeFlushProfile {
                edges: edge_count,
                flushes: usize::from(edge_count > 0),
                ..PendingEdgeFlushProfile::default()
            })
        }
    }

    fn compact_topk_evidence_fixture() -> CompactTopKBatch {
        CompactTopKBatch {
            leaf_id: 17,
            point_ids: vec![10, 20, 30],
            rows: 3,
            k: 2,
            topk_ids: vec![20, 30, 30, 10, 20, u32::MAX],
            topk_dists: vec![1.0, 2.0, 1.5, 1.0, 1.5, f32::INFINITY],
            estimated_dtoh_bytes: 3 * 2 * (size_of::<u32>() + size_of::<f32>()),
            ..CompactTopKBatch::default()
        }
    }

    fn assert_edge_evidence(
        edges: &[PendingEdge],
        src: usize,
        dst: u32,
        expected: HashPruneEvidenceClass,
    ) {
        let matching = edges
            .iter()
            .filter(|edge| edge.p == src && edge.c == dst)
            .collect::<Vec<_>>();
        assert!(!matching.is_empty(), "missing edge {src}->{dst}");
        assert!(
            matching
                .iter()
                .all(|edge| edge.meta.evidence() == Some(expected)),
            "unexpected evidence for {src}->{dst}: {matching:?}"
        );
    }

    #[test]
    fn compact_topk_evidence_classifies_both_directions_from_batch_ids() {
        let mut params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        params.m_hash_bits = 12;
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let sink = CapturingEdgeSink::with_policy(HashPruneEvidencePolicy::report());
        let batch = compact_topk_evidence_fixture();
        let dtoh_bytes = batch.estimated_total_dtoh_bytes();

        let profile = flush_compact_topk_batch(&batch, &sketches, &params, &sink).unwrap();
        let edges = sink.edges.lock();

        assert_edge_evidence(
            &edges,
            10,
            20,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        assert_edge_evidence(&edges, 20, 10, HashPruneEvidenceClass::Bidirectional);
        assert_edge_evidence(&edges, 10, 30, HashPruneEvidenceClass::Direct);
        assert_edge_evidence(&edges, 30, 10, HashPruneEvidenceClass::Reverse);
        assert_edge_evidence(
            &edges,
            20,
            30,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        assert_edge_evidence(
            &edges,
            30,
            20,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        assert_eq!(profile.hashprune_evidence.reciprocal_checks, 5);
        assert_eq!(profile.hashprune_evidence.reciprocal_hits, 4);
        assert_eq!(profile.hashprune_evidence.support_once_candidates, 2);
        assert_eq!(profile.hashprune_evidence.support_twice_candidates, 4);
        assert_eq!(profile.hashprune_evidence.class_counts, [1, 0, 1, 2, 0, 6]);
        assert_eq!(batch.estimated_total_dtoh_bytes(), dtoh_bytes);
    }

    #[test]
    fn compact_topk_evidence_off_emits_unannotated_edges_without_reciprocal_checks() {
        let params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let sink = CapturingEdgeSink::default();
        let batch = compact_topk_evidence_fixture();

        let profile = flush_compact_topk_batch(&batch, &sketches, &params, &sink).unwrap();
        let edges = sink.edges.lock();

        assert!(edges.iter().all(|edge| edge.meta.evidence().is_none()));
        assert_eq!(profile.hashprune_evidence.reciprocal_checks, 0);
        assert_eq!(profile.hashprune_evidence.reciprocal_hits, 0);
        assert_eq!(profile.hashprune_evidence.class_counts, [0; 6]);
    }

    #[test]
    fn compact_topk_evidence_rejects_effective_lune_pruning() {
        let params = PiPNNParams {
            leaf_knn: 2,
            compact_lune_prune_enable: true,
            ..PiPNNParams::default()
        };
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let sink = CapturingEdgeSink::with_policy(HashPruneEvidencePolicy::report());
        let batch = compact_topk_evidence_fixture();

        let error = flush_compact_topk_batch(&batch, &sketches, &params, &sink).unwrap_err();

        assert!(error.to_string().contains("compact lune pruning"));
    }

    fn pending_edge_tuples(edges: &[PendingEdge]) -> Vec<(usize, u32, u16, u32, HashPruneMeta)> {
        edges
            .iter()
            .map(|edge| (edge.p, edge.c, edge.hash, edge.dist.to_bits(), edge.meta))
            .collect()
    }

    fn evidence_counter_snapshot(profile: &HashPruneEvidenceProfile) -> Vec<usize> {
        let mut counters = vec![
            profile.reciprocal_checks,
            profile.reciprocal_hits,
            profile.support_once_candidates,
            profile.support_twice_candidates,
        ];
        counters.extend(profile.class_counts);
        counters.extend([
            profile.candidates_before_coalesce,
            profile.candidates_after_coalesce,
            profile.duplicate_records_removed,
            profile.unannotated_candidates,
            profile.qualified_collision_losers,
            profile.overflow_inserts,
            profile.overflow_replacements,
            profile.overflow_rejects,
            profile.overflow_evictions_by_new_hash,
            profile.rows_with_overflow,
            profile.spill_records_avoided,
            profile.spill_bytes_avoided,
        ]);
        counters
    }

    #[inline]
    fn pending_edge_packed_pair_key_for_test(edge: &PendingEdge) -> u64 {
        debug_assert!(edge.p <= u32::MAX as usize);
        ((edge.p as u64) << 32) | u64::from(edge.c)
    }

    fn sort_pending_edges_by_packed_pair_key_for_test(edges: &mut [PendingEdge]) {
        edges.sort_unstable_by_key(pending_edge_packed_pair_key_for_test);
    }

    fn prepare_pending_edges_for_sink_with_sort_for_test(
        edges: &mut Vec<PendingEdge>,
        policy: HashPruneEvidencePolicy,
        packed_pair_key: bool,
    ) -> AnnResult<HashPruneEvidenceProfile> {
        if policy.is_off() {
            edges.sort_unstable_by_key(|edge| edge.p);
            return Ok(HashPruneEvidenceProfile::default());
        }
        if policy.enforces() {
            for edge in edges.iter() {
                edge.meta.to_spill_flags()?;
            }
        }

        let mut profile = HashPruneEvidenceProfile {
            candidates_before_coalesce: edges.len(),
            ..HashPruneEvidenceProfile::default()
        };

        let sort_start = Instant::now();
        if packed_pair_key {
            sort_pending_edges_by_packed_pair_key_for_test(edges);
        } else {
            edges.sort_unstable_by(|left, right| {
                left.p.cmp(&right.p).then_with(|| left.c.cmp(&right.c))
            });
        }
        profile.coalesce_sort_wall = sort_start.elapsed();

        let merge_start = Instant::now();
        let mut read = 0usize;
        let mut write = 0usize;
        while read < edges.len() {
            let mut merged = edges[read].clone();
            read += 1;
            while read < edges.len() && edges[read].p == merged.p && edges[read].c == merged.c {
                let duplicate = &edges[read];
                if duplicate.hash != merged.hash {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune duplicate hash mismatch for source {} destination {}: left={} right={}",
                        merged.p, merged.c, merged.hash, duplicate.hash
                    )));
                }
                if duplicate.dist.total_cmp(&merged.dist).is_lt() {
                    merged.dist = duplicate.dist;
                }
                merged.meta = merged.meta.merge_prevalidated_candidate(duplicate.meta);
                read += 1;
            }

            if merged.meta.evidence().is_none() {
                profile.unannotated_candidates += 1;
                if policy.enforces() {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune enforce received an unannotated candidate for source {} destination {}",
                        merged.p, merged.c
                    )));
                }
            }
            edges[write] = merged;
            write += 1;
        }
        edges.truncate(write);
        profile.coalesce_merge_wall = merge_start.elapsed();
        profile.candidates_after_coalesce = write;
        profile.duplicate_records_removed =
            profile.candidates_before_coalesce.saturating_sub(write);
        profile.coalesce_wall = profile.coalesce_sort_wall + profile.coalesce_merge_wall;
        Ok(profile)
    }

    fn prepare_pending_edges_for_sink_comparator_for_test(
        edges: &mut Vec<PendingEdge>,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<HashPruneEvidenceProfile> {
        prepare_pending_edges_for_sink_with_sort_for_test(edges, policy, false)
    }

    fn prepare_pending_edges_for_sink_packed_for_test(
        edges: &mut Vec<PendingEdge>,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<HashPruneEvidenceProfile> {
        prepare_pending_edges_for_sink_with_sort_for_test(edges, policy, true)
    }

    #[derive(Clone, Copy)]
    enum PendingEdgePairSortVariant {
        Comparator,
        Packed,
    }

    struct PendingEdgePairSortBenchmarkResult {
        elapsed: Duration,
        edges: Vec<PendingEdge>,
        profile: HashPruneEvidenceProfile,
        profile_counters: Vec<usize>,
        checksum: u64,
    }

    fn pending_edge_benchmark_meta(index: usize) -> HashPruneMeta {
        let class = match index % 6 {
            0 => HashPruneEvidenceClass::Reverse,
            1 => HashPruneEvidenceClass::ReverseRank0,
            2 => HashPruneEvidenceClass::Direct,
            3 => HashPruneEvidenceClass::Bidirectional,
            4 => HashPruneEvidenceClass::DirectRank0,
            _ => HashPruneEvidenceClass::BidirectionalDirectRank0,
        };
        HashPruneMeta::with_evidence(class)
    }

    fn build_pending_edge_pair_sort_benchmark_workload() -> Vec<PendingEdge> {
        const RECORDS: usize = super::COMPACT_TOPK_SUPERBATCH_EDGE_FLUSH_THRESHOLD;
        const DUPLICATE_RECORDS: usize = 41_077;
        const UNIQUE_RECORDS: usize = RECORDS - DUPLICATE_RECORDS;
        const SOURCES: usize = 4_096;

        let mut edges = Vec::with_capacity(RECORDS);
        for index in 0..UNIQUE_RECORDS {
            let source = index % SOURCES;
            let ordinal = index / SOURCES;
            let destination = ((ordinal as u32) << 12) | ((source as u32) ^ 0x0a5a);
            edges.push(PendingEdge {
                p: source,
                c: destination,
                hash: (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) as u16,
                dist: ((index % 65_521) + 1) as f32 / 257.0,
                meta: pending_edge_benchmark_meta(index),
            });
        }

        for duplicate_index in 0..DUPLICATE_RECORDS {
            let source_index = duplicate_index * UNIQUE_RECORDS / DUPLICATE_RECORDS;
            let mut duplicate = edges[source_index].clone();
            duplicate.dist = -duplicate.dist;
            duplicate.meta = duplicate.meta.merge_prevalidated_candidate(
                HashPruneMeta::mandatory().merge_prevalidated_candidate(
                    HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0),
                ),
            );
            edges.push(duplicate);
        }

        let mut state = 0xd1b5_4a32_d192_ed03u64;
        for index in (1..edges.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            edges.swap(index, state as usize % (index + 1));
        }
        edges
    }

    fn pending_edge_pair_sort_benchmark_checksum(
        edges: &[PendingEdge],
        profile_counters: &[usize],
    ) -> u64 {
        fn mix(checksum: &mut u64, value: u64) {
            *checksum ^= value;
            *checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
        }

        let mut checksum = 0xcbf2_9ce4_8422_2325u64;
        mix(&mut checksum, edges.len() as u64);
        for edge in edges {
            mix(&mut checksum, edge.p as u64);
            mix(&mut checksum, u64::from(edge.c));
            mix(&mut checksum, u64::from(edge.hash));
            mix(&mut checksum, u64::from(edge.dist.to_bits()));
            mix(
                &mut checksum,
                u64::from(edge.meta.to_spill_flags().unwrap()),
            );
        }
        for &counter in profile_counters {
            mix(&mut checksum, counter as u64);
        }
        checksum
    }

    fn run_pending_edge_pair_sort_benchmark_variant(
        mut edges: Vec<PendingEdge>,
        variant: PendingEdgePairSortVariant,
    ) -> PendingEdgePairSortBenchmarkResult {
        let start = Instant::now();
        let profile = match variant {
            PendingEdgePairSortVariant::Comparator => {
                prepare_pending_edges_for_sink_comparator_for_test(
                    &mut edges,
                    HashPruneEvidencePolicy::report(),
                )
            }
            PendingEdgePairSortVariant::Packed => prepare_pending_edges_for_sink_packed_for_test(
                &mut edges,
                HashPruneEvidencePolicy::report(),
            ),
        }
        .unwrap();
        let elapsed = start.elapsed();

        let profile_counters = evidence_counter_snapshot(&profile);
        let checksum = pending_edge_pair_sort_benchmark_checksum(&edges, &profile_counters);
        black_box(&edges);
        black_box(&profile);
        black_box(&profile_counters);
        black_box(checksum);
        PendingEdgePairSortBenchmarkResult {
            elapsed,
            edges,
            profile,
            profile_counters,
            checksum,
        }
    }

    fn assert_pending_edge_pair_sort_benchmark_equivalent(
        comparator: &PendingEdgePairSortBenchmarkResult,
        packed: &PendingEdgePairSortBenchmarkResult,
    ) {
        assert_eq!(
            pending_edge_tuples(&packed.edges),
            pending_edge_tuples(&comparator.edges)
        );
        assert_eq!(packed.profile_counters, comparator.profile_counters);
        assert_eq!(packed.checksum, comparator.checksum);
        assert_eq!(comparator.profile.candidates_before_coalesce, 262_144);
        assert_eq!(comparator.profile.candidates_after_coalesce, 221_067);
        assert_eq!(comparator.profile.duplicate_records_removed, 41_077);
        assert_eq!(comparator.profile.unannotated_candidates, 0);
        assert_eq!(
            comparator.profile.coalesce_wall,
            comparator.profile.coalesce_sort_wall + comparator.profile.coalesce_merge_wall
        );
        assert_eq!(
            packed.profile.coalesce_wall,
            packed.profile.coalesce_sort_wall + packed.profile.coalesce_merge_wall
        );
    }

    fn median_f64(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    #[test]
    fn pending_edge_packed_pair_key_matches_comparator_order() {
        let direct = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let reverse = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse);
        let mut comparator_edges = vec![
            PendingEdge {
                p: u32::MAX as usize,
                c: u32::MAX,
                hash: 61,
                dist: 6.25,
                meta: reverse,
            },
            PendingEdge {
                p: 0,
                c: u32::MAX,
                hash: 11,
                dist: 1.25,
                meta: direct,
            },
            PendingEdge {
                p: 7,
                c: u32::MAX,
                hash: 43,
                dist: 4.25,
                meta: reverse,
            },
            PendingEdge {
                p: 7,
                c: 0,
                hash: 31,
                dist: 3.25,
                meta: direct,
            },
            PendingEdge {
                p: u32::MAX as usize,
                c: 0,
                hash: 53,
                dist: 5.25,
                meta: direct,
            },
            PendingEdge {
                p: 0,
                c: 0,
                hash: 2,
                dist: 0.25,
                meta: reverse,
            },
        ];
        let mut packed_edges = comparator_edges.clone();

        comparator_edges.sort_unstable_by(|left, right| {
            left.p.cmp(&right.p).then_with(|| left.c.cmp(&right.c))
        });
        sort_pending_edges_by_packed_pair_key_for_test(&mut packed_edges);

        let expected = vec![
            (0, 0, 2, 0.25f32.to_bits(), reverse),
            (0, u32::MAX, 11, 1.25f32.to_bits(), direct),
            (7, 0, 31, 3.25f32.to_bits(), direct),
            (7, u32::MAX, 43, 4.25f32.to_bits(), reverse),
            (u32::MAX as usize, 0, 53, 5.25f32.to_bits(), direct),
            (u32::MAX as usize, u32::MAX, 61, 6.25f32.to_bits(), reverse),
        ];
        assert_eq!(pending_edge_tuples(&comparator_edges), expected);
        assert_eq!(pending_edge_tuples(&packed_edges), expected);
    }

    #[test]
    fn pending_edge_packed_pair_key_coalesces_identically() {
        let direct = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let reverse = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse);
        let bidirectional = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional);
        let mandatory_direct_rank0 = HashPruneMeta::mandatory()
            .merge_candidate(HashPruneMeta::with_evidence(
                HashPruneEvidenceClass::DirectRank0,
            ))
            .unwrap();
        let input = vec![
            PendingEdge {
                p: u32::MAX as usize,
                c: u32::MAX,
                hash: 19,
                dist: 9.0,
                meta: reverse,
            },
            PendingEdge {
                p: 0,
                c: u32::MAX,
                hash: 7,
                dist: 4.0,
                meta: direct,
            },
            PendingEdge {
                p: 0,
                c: 0,
                hash: 1,
                dist: 8.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: u32::MAX,
                hash: 7,
                dist: 1.0,
                meta: mandatory_direct_rank0,
            },
            PendingEdge {
                p: u32::MAX as usize,
                c: 0,
                hash: 17,
                dist: 7.0,
                meta: direct,
            },
            PendingEdge {
                p: 0,
                c: 17,
                hash: 5,
                dist: 5.0,
                meta: reverse,
            },
            PendingEdge {
                p: 0,
                c: 17,
                hash: 5,
                dist: 3.0,
                meta: bidirectional,
            },
        ];
        let mut comparator_edges = input.clone();
        let mut packed_edges = input.clone();
        let mut production_edges = input;

        let comparator_profile = prepare_pending_edges_for_sink_comparator_for_test(
            &mut comparator_edges,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        let packed_profile = prepare_pending_edges_for_sink_packed_for_test(
            &mut packed_edges,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        let production_profile = super::prepare_pending_edges_for_sink(
            &mut production_edges,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();

        let expected = vec![
            (0, 0, 1, 8.0f32.to_bits(), HashPruneMeta::empty()),
            (0, 17, 5, 3.0f32.to_bits(), bidirectional),
            (0, u32::MAX, 7, 1.0f32.to_bits(), mandatory_direct_rank0),
            (u32::MAX as usize, 0, 17, 7.0f32.to_bits(), direct),
            (u32::MAX as usize, u32::MAX, 19, 9.0f32.to_bits(), reverse),
        ];
        assert_eq!(pending_edge_tuples(&comparator_edges), expected);
        assert_eq!(pending_edge_tuples(&packed_edges), expected);
        assert_eq!(pending_edge_tuples(&production_edges), expected);
        assert_eq!(
            evidence_counter_snapshot(&packed_profile),
            evidence_counter_snapshot(&comparator_profile)
        );
        assert_eq!(
            evidence_counter_snapshot(&production_profile),
            evidence_counter_snapshot(&comparator_profile)
        );
        assert_eq!(comparator_profile.candidates_before_coalesce, 7);
        assert_eq!(comparator_profile.candidates_after_coalesce, 5);
        assert_eq!(comparator_profile.duplicate_records_removed, 2);
        assert_eq!(comparator_profile.unannotated_candidates, 1);

        let mismatched = vec![
            PendingEdge {
                p: u32::MAX as usize,
                c: u32::MAX,
                hash: 23,
                dist: 2.0,
                meta: direct,
            },
            PendingEdge {
                p: u32::MAX as usize,
                c: u32::MAX,
                hash: 29,
                dist: 1.0,
                meta: mandatory_direct_rank0,
            },
        ];
        let mut comparator_mismatched = mismatched.clone();
        let mut packed_mismatched = mismatched.clone();
        let mut production_mismatched = mismatched;
        let comparator_error = prepare_pending_edges_for_sink_comparator_for_test(
            &mut comparator_mismatched,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap_err()
        .to_string();
        let packed_error = prepare_pending_edges_for_sink_packed_for_test(
            &mut packed_mismatched,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap_err()
        .to_string();
        let production_error = super::prepare_pending_edges_for_sink(
            &mut production_mismatched,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap_err()
        .to_string();

        assert_eq!(packed_error, comparator_error);
        assert_eq!(production_error, comparator_error);
        assert_eq!(
            comparator_error,
            "IndexError: HashPrune duplicate hash mismatch for source 4294967295 destination 4294967295: left=23 right=29"
        );
    }

    #[test]
    #[ignore = "manual release microbenchmark"]
    fn pending_edge_pair_sort_release_microbenchmark() {
        const RECORDS: usize = super::COMPACT_TOPK_SUPERBATCH_EDGE_FLUSH_THRESHOLD;
        const DUPLICATE_RECORDS: usize = 41_077;
        const UNIQUE_RECORDS: usize = RECORDS - DUPLICATE_RECORDS;
        const OBSERVED_PROFILE_RECORDS: usize = 719_981_780;
        const OBSERVED_PROFILE_DUPLICATES: usize = 112_819_728;
        const ROUNDS: usize = 8;

        assert_eq!(size_of::<PendingEdge>(), 24);
        let grounded_duplicate_records = ((RECORDS as u128 * OBSERVED_PROFILE_DUPLICATES as u128)
            + (OBSERVED_PROFILE_RECORDS as u128 / 2))
            / OBSERVED_PROFILE_RECORDS as u128;
        assert_eq!(grounded_duplicate_records as usize, DUPLICATE_RECORDS);

        let workload = build_pending_edge_pair_sort_benchmark_workload();
        assert_eq!(workload.len(), RECORDS);
        println!(
            "pending_edge_pair_sort workload_records={RECORDS} unique_records={UNIQUE_RECORDS} duplicate_records={DUPLICATE_RECORDS} duplicate_ratio={:.6}% observed_wiki10m_duplicate_ratio={:.6}% pending_edge_bytes={} rounds={ROUNDS}",
            DUPLICATE_RECORDS as f64 * 100.0 / RECORDS as f64,
            OBSERVED_PROFILE_DUPLICATES as f64 * 100.0 / OBSERVED_PROFILE_RECORDS as f64,
            size_of::<PendingEdge>()
        );

        let packed_warmup = run_pending_edge_pair_sort_benchmark_variant(
            workload.clone(),
            PendingEdgePairSortVariant::Packed,
        );
        let comparator_warmup = run_pending_edge_pair_sort_benchmark_variant(
            workload.clone(),
            PendingEdgePairSortVariant::Comparator,
        );
        assert_pending_edge_pair_sort_benchmark_equivalent(&comparator_warmup, &packed_warmup);
        println!("pending_edge_pair_sort warmup_order=packed->comparator");

        let mut comparator_ns_per_record = Vec::with_capacity(ROUNDS);
        let mut packed_ns_per_record = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let (order, comparator, packed) = if round % 2 == 0 {
                let comparator = run_pending_edge_pair_sort_benchmark_variant(
                    workload.clone(),
                    PendingEdgePairSortVariant::Comparator,
                );
                let packed = run_pending_edge_pair_sort_benchmark_variant(
                    workload.clone(),
                    PendingEdgePairSortVariant::Packed,
                );
                ("comparator->packed", comparator, packed)
            } else {
                let packed = run_pending_edge_pair_sort_benchmark_variant(
                    workload.clone(),
                    PendingEdgePairSortVariant::Packed,
                );
                let comparator = run_pending_edge_pair_sort_benchmark_variant(
                    workload.clone(),
                    PendingEdgePairSortVariant::Comparator,
                );
                ("packed->comparator", comparator, packed)
            };
            assert_pending_edge_pair_sort_benchmark_equivalent(&comparator, &packed);

            let comparator_ns = comparator.elapsed.as_nanos() as f64 / RECORDS as f64;
            let packed_ns = packed.elapsed.as_nanos() as f64 / RECORDS as f64;
            comparator_ns_per_record.push(comparator_ns);
            packed_ns_per_record.push(packed_ns);
            println!(
                "pending_edge_pair_sort round={} order={order} comparator_ms={:.6} comparator_ns_per_record={comparator_ns:.6} packed_ms={:.6} packed_ns_per_record={packed_ns:.6} checksum={:#018x}",
                round + 1,
                comparator.elapsed.as_secs_f64() * 1_000.0,
                packed.elapsed.as_secs_f64() * 1_000.0,
                comparator.checksum
            );
        }

        let comparator_median = median_f64(&comparator_ns_per_record);
        let packed_median = median_f64(&packed_ns_per_record);
        let speedup = comparator_median / packed_median;
        let improvement_pct = (1.0 - packed_median / comparator_median) * 100.0;
        println!(
            "pending_edge_pair_sort comparator_round_ns_per_record={comparator_ns_per_record:?}"
        );
        println!("pending_edge_pair_sort packed_round_ns_per_record={packed_ns_per_record:?}");
        println!(
            "pending_edge_pair_sort comparator_median_ns_per_record={comparator_median:.6} packed_median_ns_per_record={packed_median:.6} speedup_x={speedup:.6} improvement_pct={improvement_pct:.6} meets_3pct_gate={}",
            improvement_pct >= 3.0
        );
    }

    #[test]
    fn pending_edge_coalesce_merges_min_distance_mandatory_and_max_class() {
        let mandatory_direct_rank0 = HashPruneMeta::mandatory()
            .merge_candidate(HashPruneMeta::with_evidence(
                HashPruneEvidenceClass::DirectRank0,
            ))
            .unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 7,
                c: 9,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            },
            PendingEdge {
                p: 7,
                c: 9,
                hash: 3,
                dist: 2.0,
                meta: mandatory_direct_rank0,
            },
        ];

        let profile =
            super::prepare_pending_edges_for_sink(&mut edges, HashPruneEvidencePolicy::report())
                .unwrap();

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].p, 7);
        assert_eq!(edges[0].c, 9);
        assert_eq!(edges[0].hash, 3);
        assert_eq!(edges[0].dist, 2.0);
        assert!(edges[0].meta.is_mandatory());
        assert_eq!(
            edges[0].meta.evidence(),
            Some(HashPruneEvidenceClass::DirectRank0)
        );
        assert_eq!(profile.candidates_before_coalesce, 2);
        assert_eq!(profile.candidates_after_coalesce, 1);
        assert_eq!(profile.duplicate_records_removed, 1);
        assert_eq!(
            profile.coalesce_wall,
            profile.coalesce_sort_wall + profile.coalesce_merge_wall
        );
    }

    #[test]
    fn pending_edge_coalesce_off_keeps_all_evidence_durations_zero() {
        let mut edges = vec![
            PendingEdge {
                p: 7,
                c: 9,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 2,
                c: 4,
                hash: 5,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        let profile =
            super::prepare_pending_edges_for_sink(&mut edges, HashPruneEvidencePolicy::off())
                .unwrap();

        assert_eq!(edges.iter().map(|edge| edge.p).collect::<Vec<_>>(), [2, 7]);
        assert_eq!(profile.coalesce_sort_wall, Duration::ZERO);
        assert_eq!(profile.coalesce_merge_wall, Duration::ZERO);
        assert_eq!(profile.coalesce_wall, Duration::ZERO);
    }

    #[test]
    fn pending_edge_coalesce_rejects_hash_mismatch_with_endpoint_ids() {
        let mut edges = vec![
            PendingEdge {
                p: 7,
                c: 9,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            },
            PendingEdge {
                p: 7,
                c: 9,
                hash: 4,
                dist: 2.0,
                meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0),
            },
        ];

        let error =
            super::prepare_pending_edges_for_sink(&mut edges, HashPruneEvidencePolicy::report())
                .unwrap_err();
        let message = error.to_string();

        assert!(message.contains("source 7"), "{message}");
        assert!(message.contains("destination 9"), "{message}");
    }

    #[test]
    fn pending_edge_coalesce_enforce_rejects_unannotated_survivor() {
        let mut edges = vec![PendingEdge {
            p: 7,
            c: 9,
            hash: 3,
            dist: 2.0,
            meta: HashPruneMeta::empty(),
        }];

        let error = super::prepare_pending_edges_for_sink(
            &mut edges,
            HashPruneEvidencePolicy::enforce_one(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("unannotated candidate"));
    }

    #[test]
    fn pending_edge_coalesce_enforce_rejects_overflow_metadata() {
        let mut edges = vec![
            PendingEdge {
                p: 7,
                c: 9,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            },
            PendingEdge {
                p: 7,
                c: 9,
                hash: 3,
                dist: 2.0,
                meta: HashPruneMeta::empty().mark_overflow(),
            },
        ];
        let before = edges
            .iter()
            .map(|edge| (edge.p, edge.c, edge.hash, edge.dist.to_bits(), edge.meta))
            .collect::<Vec<_>>();

        let error = super::prepare_pending_edges_for_sink(
            &mut edges,
            HashPruneEvidencePolicy::enforce_one(),
        )
        .unwrap_err();
        let after = edges
            .iter()
            .map(|edge| (edge.p, edge.c, edge.hash, edge.dist.to_bits(), edge.meta))
            .collect::<Vec<_>>();

        assert!(error.to_string().contains("overflow"));
        assert_eq!(after, before);
    }

    fn ordered_reservoir_digest(
        reservoirs: &[Mutex<HashPruneReservoir>],
    ) -> Vec<Vec<(u32, u16, u32)>> {
        reservoirs
            .iter()
            .map(|reservoir| {
                reservoir
                    .lock()
                    .slots()
                    .iter()
                    .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                    .collect()
            })
            .collect()
    }

    fn ordered_reservoir_state(
        reservoirs: &[Mutex<HashPruneReservoir>],
    ) -> Vec<Vec<(u32, u16, u32, u16)>> {
        reservoirs
            .iter()
            .map(|reservoir| {
                reservoir
                    .lock()
                    .slots()
                    .iter()
                    .map(|slot| {
                        (
                            slot.target,
                            slot.hash,
                            slot.dist.to_bits(),
                            slot.meta.to_spill_flags().unwrap(),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    fn evidence_telemetry(profile: &HashPruneEvidenceProfile) -> [usize; 22] {
        [
            profile.reciprocal_checks,
            profile.reciprocal_hits,
            profile.support_once_candidates,
            profile.support_twice_candidates,
            profile.class_counts[0],
            profile.class_counts[1],
            profile.class_counts[2],
            profile.class_counts[3],
            profile.class_counts[4],
            profile.class_counts[5],
            profile.candidates_before_coalesce,
            profile.candidates_after_coalesce,
            profile.duplicate_records_removed,
            profile.unannotated_candidates,
            profile.qualified_collision_losers,
            profile.overflow_inserts,
            profile.overflow_replacements,
            profile.overflow_rejects,
            profile.overflow_evictions_by_new_hash,
            profile.rows_with_overflow,
            profile.spill_records_avoided,
            profile.spill_bytes_avoided,
        ]
    }

    fn report_fast_path_candidates() -> Vec<PendingEdge> {
        let direct = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let bidirectional = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional);
        let direct_rank0 = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        let bidirectional_rank0 =
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0);
        vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 2,
                dist: 5.0,
                meta: direct,
            },
            PendingEdge {
                p: 0,
                c: 10,
                hash: 2,
                dist: 4.0,
                meta: direct_rank0,
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 2,
                dist: 1.0,
                meta: bidirectional,
            },
            PendingEdge {
                p: 1,
                c: 20,
                hash: 10,
                dist: 1.0,
                meta: direct,
            },
            PendingEdge {
                p: 1,
                c: 21,
                hash: 20,
                dist: 5.0,
                meta: direct_rank0,
            },
            PendingEdge {
                p: 1,
                c: 22,
                hash: 30,
                dist: 2.0,
                meta: bidirectional,
            },
            PendingEdge {
                p: 1,
                c: 23,
                hash: 5,
                dist: 0.5,
                meta: bidirectional_rank0,
            },
            PendingEdge {
                p: 2,
                c: 30,
                hash: 10,
                dist: 1.0,
                meta: direct,
            },
            PendingEdge {
                p: 2,
                c: 31,
                hash: 20,
                dist: 5.0,
                meta: direct_rank0,
            },
            PendingEdge {
                p: 2,
                c: 32,
                hash: 30,
                dist: 2.0,
                meta: bidirectional,
            },
            PendingEdge {
                p: 2,
                c: 33,
                hash: 40,
                dist: 0.5,
                meta: bidirectional_rank0,
            },
        ]
    }

    fn run_fully_profiled_report_reference(
        edges: &mut Vec<PendingEdge>,
        reservoirs: &[Mutex<HashPruneReservoir>],
    ) -> (usize, HashPruneInsertProfile, HashPruneEvidenceProfile) {
        let mut evidence =
            super::prepare_pending_edges_for_sink(edges, HashPruneEvidencePolicy::report())
                .unwrap();
        let prepared_edge_count = edges.len();
        let mut insert_profile = HashPruneInsertProfile::default();
        let mut i = 0usize;
        while i < edges.len() {
            let group_start = i;
            let source = edges[i].p;
            while i < edges.len() && edges[i].p == source {
                i += 1;
            }
            let mut reservoir = reservoirs[source].lock();
            for edge in &edges[group_start..i] {
                reservoir
                    .insert_with_policy_profiled(
                        source,
                        edge.c,
                        edge.hash,
                        edge.dist,
                        edge.meta,
                        HashPruneEvidencePolicy::report(),
                        &mut insert_profile,
                    )
                    .unwrap();
            }
        }
        evidence.merge(std::mem::take(&mut insert_profile.evidence));
        edges.clear();
        (prepared_edge_count, insert_profile, evidence)
    }

    #[test]
    fn hashprune_report_fast_path_matches_profiled_reference_resident() {
        let canonical_reservoirs = (0..3)
            .map(|_| Mutex::new(HashPruneReservoir::new(3)))
            .collect::<Vec<_>>();
        let reference_reservoirs = (0..3)
            .map(|_| Mutex::new(HashPruneReservoir::new(3)))
            .collect::<Vec<_>>();
        let mut canonical_edges = report_fast_path_candidates();
        let mut reference_edges = canonical_edges.clone();

        let canonical_profile = super::flush_pending_edges_to_reservoirs_with_policy_core(
            &mut canonical_edges,
            &canonical_reservoirs,
            HashPruneEvidencePolicy::report(),
            false,
        )
        .unwrap();
        let (_, _, reference_evidence) =
            run_fully_profiled_report_reference(&mut reference_edges, &reference_reservoirs);

        assert_eq!(
            ordered_reservoir_state(&canonical_reservoirs),
            ordered_reservoir_state(&reference_reservoirs)
        );
        assert_eq!(
            evidence_telemetry(&canonical_profile.evidence),
            evidence_telemetry(&reference_evidence)
        );
        assert_eq!(canonical_profile.evidence.overflow_inserts, 0);
        assert_eq!(canonical_profile.evidence.overflow_replacements, 0);
        assert_eq!(canonical_profile.evidence.overflow_rejects, 0);
        assert_eq!(canonical_profile.evidence.overflow_evictions_by_new_hash, 0);
        assert_eq!(canonical_profile.evidence.rows_with_overflow, 0);
    }

    #[test]
    fn hashprune_report_fast_path_skips_transition_counters_without_microscope() {
        let canonical_reservoirs = (0..3)
            .map(|_| Mutex::new(HashPruneReservoir::new(3)))
            .collect::<Vec<_>>();
        let reference_reservoirs = (0..3)
            .map(|_| Mutex::new(HashPruneReservoir::new(3)))
            .collect::<Vec<_>>();
        let mut canonical_edges = report_fast_path_candidates();
        let mut reference_edges = canonical_edges.clone();

        let canonical_profile = super::flush_pending_edges_to_reservoirs_with_policy_core(
            &mut canonical_edges,
            &canonical_reservoirs,
            HashPruneEvidencePolicy::report(),
            false,
        )
        .unwrap();
        let (prepared_edge_count, reference_profile, _) =
            run_fully_profiled_report_reference(&mut reference_edges, &reference_reservoirs);

        assert_eq!(
            canonical_profile.insert_profile.candidates,
            prepared_edge_count
        );
        assert_eq!(canonical_profile.insert_profile.hash_hits, 0);
        assert_eq!(canonical_profile.insert_profile.hash_replacements, 0);
        assert_eq!(canonical_profile.insert_profile.hash_rejects, 0);
        assert_eq!(canonical_profile.insert_profile.not_full_inserts, 0);
        assert_eq!(canonical_profile.insert_profile.full_replacement_checks, 0);
        assert_eq!(canonical_profile.insert_profile.full_replacements, 0);
        assert_eq!(canonical_profile.insert_profile.full_rejects, 0);
        assert_eq!(canonical_profile.insert_profile.farthest_recomputes, 0);
        assert_eq!(canonical_profile.insert_profile.rotate_lefts, 0);
        assert_eq!(canonical_profile.insert_profile.rotate_rights, 0);
        assert!(reference_profile.hash_hits > 0);
        assert!(reference_profile.full_replacements > 0);
        assert!(reference_profile.rotate_lefts > 0);
        assert!(reference_profile.rotate_rights > 0);
    }

    #[test]
    fn hashprune_report_fast_path_microscope_uses_profiled_path() {
        let reservoirs = (0..3)
            .map(|_| Mutex::new(HashPruneReservoir::new(3)))
            .collect::<Vec<_>>();
        let mut edges = report_fast_path_candidates();

        let profile = super::flush_pending_edges_to_reservoirs_with_policy_core(
            &mut edges,
            &reservoirs,
            HashPruneEvidencePolicy::report(),
            true,
        )
        .unwrap();

        assert_eq!(profile.insert_profile.candidates, 10);
        assert!(profile.insert_profile.hash_hits > 0);
        assert!(profile.insert_profile.full_replacements > 0);
        assert!(profile.insert_profile.rotate_lefts > 0);
        assert!(profile.insert_profile.rotate_rights > 0);
        assert_eq!(profile.groups, 3);
        assert_eq!(profile.max_group_edges, 4);
    }

    #[test]
    fn hashprune_cross_flush_resident_canonical_merges_same_pair_metadata() {
        let reservoirs = vec![Mutex::new(HashPruneReservoir::new(4))];
        let first_meta = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let second_meta = HashPruneMeta::mandatory()
            .merge_candidate(HashPruneMeta::with_evidence(
                HashPruneEvidenceClass::BidirectionalDirectRank0,
            ))
            .unwrap();

        for (dist, meta) in [(2.0, first_meta), (5.0, second_meta)] {
            let mut edges = vec![PendingEdge {
                p: 0,
                c: 9,
                hash: 3,
                dist,
                meta,
            }];
            super::flush_pending_edges_to_reservoirs_with_policy_core(
                &mut edges,
                &reservoirs,
                HashPruneEvidencePolicy::report(),
                false,
            )
            .unwrap();
        }

        let expected_meta = first_meta.merge_candidate(second_meta).unwrap();
        let reservoir = reservoirs[0].lock();
        assert_eq!(reservoir.slots().len(), 1);
        let slot = reservoir.slots()[0];
        assert_eq!(
            (slot.target, slot.hash, slot.dist.to_bits(), slot.meta),
            (9, 3, 2.0f32.to_bits(), expected_meta)
        );
        assert!(slot.meta.is_mandatory());
        assert_eq!(
            slot.meta.evidence(),
            Some(HashPruneEvidenceClass::BidirectionalDirectRank0)
        );
    }

    #[test]
    fn hashprune_cross_flush_resident_microscope_rejects_hash_mismatch_before_mutation() {
        let reservoirs = vec![Mutex::new(HashPruneReservoir::new(4))];
        let mut first = vec![PendingEdge {
            p: 0,
            c: 9,
            hash: 3,
            dist: 2.0,
            meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
        }];
        super::flush_pending_edges_to_reservoirs_with_policy_core(
            &mut first,
            &reservoirs,
            HashPruneEvidencePolicy::report(),
            true,
        )
        .unwrap();
        let before = ordered_reservoir_state(&reservoirs);

        let mut mismatch = vec![PendingEdge {
            p: 0,
            c: 9,
            hash: 4,
            dist: 1.0,
            meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0),
        }];
        let error = super::flush_pending_edges_to_reservoirs_with_policy_core(
            &mut mismatch,
            &reservoirs,
            HashPruneEvidencePolicy::report(),
            true,
        )
        .unwrap_err();
        let message = error.to_string();

        assert!(message.contains("source 0"), "{message}");
        assert!(message.contains("destination 9"), "{message}");
        assert!(message.contains("existing hash=3"), "{message}");
        assert!(message.contains("new hash=4"), "{message}");
        assert_eq!(ordered_reservoir_state(&reservoirs), before);
        assert_eq!(reservoirs[0].lock().slots().len(), 1);
    }

    #[test]
    fn hashprune_report_matches_off_for_same_candidate_multiset() {
        let direct = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let direct_rank0 = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        let candidates = vec![
            PendingEdge {
                p: 0,
                c: 11,
                hash: 2,
                dist: 5.0,
                meta: direct,
            },
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 4.0,
                meta: direct_rank0,
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 2,
                dist: 3.0,
                meta: direct_rank0,
            },
            PendingEdge {
                p: 0,
                c: 12,
                hash: 2,
                dist: 2.0,
                meta: direct,
            },
            PendingEdge {
                p: 0,
                c: 13,
                hash: 3,
                dist: 1.0,
                meta: direct_rank0,
            },
        ];
        let off_reservoirs = vec![Mutex::new(HashPruneReservoir::new(2))];
        let report_reservoirs = vec![Mutex::new(HashPruneReservoir::new(2))];
        let mut off_edges = candidates.clone();
        let mut report_edges = candidates;

        super::flush_pending_edges_to_reservoirs(&mut off_edges, &off_reservoirs);
        super::flush_pending_edges_to_reservoirs_with_policy(
            &mut report_edges,
            &report_reservoirs,
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();

        assert_eq!(
            ordered_reservoir_digest(&report_reservoirs),
            ordered_reservoir_digest(&off_reservoirs)
        );
    }

    struct PolicyReservoirSink<'a> {
        reservoirs: &'a [Mutex<HashPruneReservoir>],
        policy: HashPruneEvidencePolicy,
    }

    impl PendingEdgeSink for PolicyReservoirSink<'_> {
        fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
            self.policy
        }

        fn flush_pending_edges(
            &self,
            edges: &mut Vec<PendingEdge>,
        ) -> AnnResult<PendingEdgeFlushProfile> {
            super::flush_pending_edges_to_reservoirs_with_policy(
                edges,
                self.reservoirs,
                self.policy,
            )
        }
    }

    fn compact_report_digest_with_threshold(
        edge_flush_threshold: usize,
    ) -> Vec<Vec<(u32, u16, u32)>> {
        let params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let reservoirs = (0..31)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect::<Vec<_>>();
        let sink = PolicyReservoirSink {
            reservoirs: &reservoirs,
            policy: HashPruneEvidencePolicy::report(),
        };
        let mut edges = Vec::new();
        super::flush_compact_topk_batch_into_edges(
            &compact_topk_evidence_fixture(),
            &sketches,
            &params,
            &sink,
            sink.policy,
            &mut edges,
            edge_flush_threshold,
        )
        .unwrap();
        super::flush_pending_edges(&mut edges, &sink).unwrap();
        ordered_reservoir_digest(&reservoirs)
    }

    #[test]
    fn flush_pending_edges_report_is_stable_across_runtime_and_superbatch_boundaries() {
        let threshold_one = compact_report_digest_with_threshold(1);
        let threshold_three = compact_report_digest_with_threshold(3);
        let threshold_all = compact_report_digest_with_threshold(usize::MAX);
        assert_eq!(threshold_one, threshold_three);
        assert_eq!(threshold_one, threshold_all);

        let params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let separate_reservoirs = (0..31)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect::<Vec<_>>();
        let combined_reservoirs = (0..31)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect::<Vec<_>>();
        let separate_sink = PolicyReservoirSink {
            reservoirs: &separate_reservoirs,
            policy: HashPruneEvidencePolicy::report(),
        };
        let combined_sink = PolicyReservoirSink {
            reservoirs: &combined_reservoirs,
            policy: HashPruneEvidencePolicy::report(),
        };
        let batch = compact_topk_evidence_fixture();

        super::flush_compact_topk_batch(&batch, &sketches, &params, &separate_sink).unwrap();
        super::flush_compact_topk_batch(&batch, &sketches, &params, &separate_sink).unwrap();
        super::flush_compact_topk_batches(
            &[batch.clone(), batch],
            &sketches,
            &params,
            &combined_sink,
        )
        .unwrap();

        assert_eq!(
            ordered_reservoir_digest(&separate_reservoirs),
            ordered_reservoir_digest(&combined_reservoirs)
        );
    }

    #[derive(Default)]
    struct TestLrpfFragmentSink {
        fragments: Mutex<Vec<LrpfFragment>>,
    }

    impl LrpfFragmentSink for TestLrpfFragmentSink {
        fn push_lrpf_fragment(
            &self,
            src: u32,
            candidates: &[LrpfCandidate],
            witnesses: &[crate::algorithm::pipnn::lrpf::LrpfWitness],
        ) -> AnnResult<()> {
            self.fragments.lock().push(LrpfFragment {
                src,
                candidates: candidates.to_vec(),
                witnesses: witnesses.to_vec(),
            });
            Ok(())
        }
    }

    #[test]
    fn reservoir_flush_profile_splits_lock_and_insert_outcomes() {
        let reservoirs = (0..2)
            .map(|_| Mutex::new(HashPruneReservoir::new(2)))
            .collect::<Vec<_>>();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 70,
                hash: 7,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 20,
                hash: 2,
                dist: 5.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 30,
                hash: 2,
                dist: 4.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 40,
                hash: 4,
                dist: 9.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 80,
                hash: 8,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        let profile = flush_pending_edges_to_reservoirs_microscope(&mut edges, &reservoirs);

        assert!(edges.is_empty());
        assert_eq!(profile.edges, 6);
        assert_eq!(profile.groups, 2);
        assert_eq!(profile.max_group_edges, 4);
        assert_eq!(profile.insert_profile.candidates, 6);
        assert_eq!(profile.insert_profile.hash_hits, 1);
        assert_eq!(profile.insert_profile.hash_replacements, 1);
        assert_eq!(profile.insert_profile.full_replacement_checks, 1);
        assert_eq!(profile.insert_profile.full_rejects, 1);
        assert!(profile.insert_wall >= profile.lock_wall);
        assert!(profile.insert_wall >= profile.insert_body_wall);
    }

    #[test]
    fn lrpf_compact_flush_does_not_prefetch_sketch_or_compute_hash() {
        let mut params = PiPNNParams::default();
        params.prune_backend = PiPNNPruneBackend::LRPF;
        params.lrpf_protected_core = 1;

        let batch = CompactTopKBatch {
            leaf_id: 0,
            rows: 3,
            k: 2,
            point_ids: vec![10, 20, 30],
            topk_ids: vec![20, 30, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            topk_dists: vec![
                1.0,
                2.0,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
            ],
            lune_witness_pivot_ranks: vec![
                super::COMPACT_LUNE_NO_WITNESS,
                0,
                super::COMPACT_LUNE_NO_WITNESS,
                super::COMPACT_LUNE_NO_WITNESS,
                super::COMPACT_LUNE_NO_WITNESS,
                super::COMPACT_LUNE_NO_WITNESS,
            ],
            lune_witness_margins: vec![0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            estimated_lune_witness_dtoh_bytes: 16,
            ..CompactTopKBatch::default()
        };
        let sink = TestLrpfFragmentSink::default();

        flush_compact_topk_batch_lrpf(&batch, &params, &sink).unwrap();
        let fragments = sink.fragments.lock();

        assert_eq!(fragments.len(), 3);
        assert_eq!(fragments[0].src, 10);
        assert_eq!(fragments[0].candidates.len(), 2);
        assert_eq!(fragments[0].witnesses.len(), 1);
        assert_eq!(fragments[1].src, 20);
        assert!(
            fragments[1]
                .candidates
                .iter()
                .any(|candidate| candidate.dst == 10)
        );
        assert_eq!(fragments[2].src, 30);
        assert!(
            fragments[2]
                .candidates
                .iter()
                .any(|candidate| candidate.dst == 10)
        );
    }

    #[test]
    fn lrpf_compact_flush_adds_reverse_candidates_to_match_hashprune_pool() {
        let mut params = PiPNNParams::default();
        params.prune_backend = PiPNNPruneBackend::LRPF;

        let batch = CompactTopKBatch {
            leaf_id: 0,
            rows: 2,
            k: 1,
            point_ids: vec![10, 20],
            topk_ids: vec![20, u32::MAX],
            topk_dists: vec![1.0, f32::INFINITY],
            ..CompactTopKBatch::default()
        };
        let sink = TestLrpfFragmentSink::default();

        flush_compact_topk_batch_lrpf(&batch, &params, &sink).unwrap();
        let fragments = sink.fragments.lock();
        let reverse_row = fragments
            .iter()
            .find(|fragment| fragment.src == 20)
            .expect("LRPF flush should emit a row for every batch point");

        assert!(
            reverse_row
                .candidates
                .iter()
                .any(|candidate| candidate.dst == 10 && candidate.dist == 1.0),
            "LRPF must preserve HashPrune's bidirectional leaf-candidate pool"
        );
    }

    #[test]
    fn lrpf_backend_enables_compact_lune_witness_metadata() {
        let mut params = PiPNNParams::default();
        params.prune_backend = PiPNNPruneBackend::LRPF;

        assert!(effective_compact_lune_witness_enabled(&params));
    }

    #[test]
    fn lrpf_compact_flush_applies_pre_spill_compression_width() {
        let mut params = PiPNNParams::default();
        params.prune_backend = PiPNNPruneBackend::LRPF;
        params.l_max = 2;
        params.lrpf_pre_spill_compress_enable = true;
        params.lrpf_pre_spill_width_multiplier = 1.0;
        params.lrpf_protected_core = 1;

        let batch = CompactTopKBatch {
            leaf_id: 0,
            rows: 5,
            k: 4,
            point_ids: vec![10, 20, 30, 40, 50],
            topk_ids: vec![
                20,
                30,
                40,
                50,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
            ],
            topk_dists: vec![
                1.0,
                2.0,
                3.0,
                4.0,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::INFINITY,
            ],
            ..CompactTopKBatch::default()
        };
        let sink = TestLrpfFragmentSink::default();

        let profile = flush_compact_topk_batch_lrpf(&batch, &params, &sink).unwrap();
        let fragments = sink.fragments.lock();

        assert_eq!(fragments[0].candidates.len(), 2);
        assert!(
            fragments[0]
                .candidates
                .iter()
                .any(|candidate| candidate.dst == 20)
        );
        assert_eq!(profile.lrpf_pre_spill_pruned_candidates, 2);
    }

    fn lune_prune_fixture_batch() -> CompactTopKBatch {
        CompactTopKBatch {
            leaf_id: 11,
            point_ids: vec![10, 20, 30],
            rows: 3,
            k: 2,
            topk_ids: vec![
                20,
                30, // src 10: victim 30 is witnessed by protected pivot 20.
                10,
                u32::MAX,
                20,
                u32::MAX,
            ],
            topk_dists: vec![1.0, 4.0, 1.0, f32::INFINITY, 2.0, f32::INFINITY],
            lune_witness_pivot_ranks: vec![
                super::COMPACT_LUNE_NO_WITNESS,
                0,
                super::COMPACT_LUNE_NO_WITNESS,
                super::COMPACT_LUNE_NO_WITNESS,
                super::COMPACT_LUNE_NO_WITNESS,
                super::COMPACT_LUNE_NO_WITNESS,
            ],
            lune_witness_margins: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            estimated_dtoh_bytes: 3 * 2 * (size_of::<u32>() + size_of::<f32>()),
            estimated_lune_witness_dtoh_bytes: 3 * 2 * (size_of::<u32>() + size_of::<f32>()),
        }
    }

    #[test]
    fn compact_lune_prune_filters_witnessed_victim_before_hashprune() {
        let mut params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        params.compact_lune_prune_enable = true;
        params.compact_lune_prune_protected_core = 1;
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let sink = CapturingEdgeSink::default();
        let batch = lune_prune_fixture_batch();

        let profile = flush_compact_topk_batch(&batch, &sketches, &params, &sink).unwrap();
        let edges = sink.edges.lock();

        assert_eq!(profile.compact_lune_pruned_slots, 1);
        assert_eq!(profile.compact_lune_pruned_rows, 1);
        assert!(!edges.iter().any(|edge| edge.p == 10 && edge.c == 30));
        assert!(!edges.iter().any(|edge| edge.p == 30 && edge.c == 10));
        assert!(edges.iter().any(|edge| edge.p == 10 && edge.c == 20));
        assert!(edges.iter().any(|edge| edge.p == 30 && edge.c == 20));
    }

    #[test]
    fn compact_lune_prune_disabled_keeps_witnessed_victim_candidate() {
        let params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        let sketches = build_test_sketches(31, params.m_hash_bits);
        let sink = CapturingEdgeSink::default();
        let batch = lune_prune_fixture_batch();

        let profile = flush_compact_topk_batch(&batch, &sketches, &params, &sink).unwrap();
        let edges = sink.edges.lock();

        assert_eq!(profile.compact_lune_pruned_slots, 0);
        assert_eq!(profile.compact_lune_pruned_rows, 0);
        assert!(edges.iter().any(|edge| edge.p == 10 && edge.c == 30));
        assert!(edges.iter().any(|edge| edge.p == 30 && edge.c == 10));
    }

    #[test]
    fn leaf_profile_merge_accumulates_all_phases() {
        let mut left = super::LeafProfile::default();
        left.total_wall = Duration::from_millis(10);
        left.distance = Duration::from_millis(6);
        left.host_load_wall = Duration::from_millis(1);
        left.backend_compute_wall = Duration::from_millis(2);
        left.gpu_h2d_event_elapsed = Duration::from_millis(3);
        left.gpu_kernel_launch = Duration::from_millis(3);
        left.gpu_kernel_elapsed = Duration::from_millis(4);
        left.gpu_d2h_event_elapsed = Duration::from_millis(5);
        left.gpu_stream_total_elapsed = Duration::from_millis(5);

        let mut right = super::LeafProfile::default();
        right.total_wall = Duration::from_millis(7);
        right.load = Duration::from_millis(2);
        right.flush = Duration::from_millis(3);
        right.leaves = 2;
        right.host_load_wall = Duration::from_millis(11);
        right.backend_compute_wall = Duration::from_millis(13);
        right.compact_materialize_wall = Duration::from_millis(17);
        right.hash_wall = Duration::from_millis(19);
        right.edge_sink_wall = Duration::from_millis(23);
        right.edge_sink_sort_wall = Duration::from_millis(29);
        right.edge_sink_insert_wall = Duration::from_millis(31);
        right.edge_sink_edges = 37;
        right.edge_sink_flushes = 2;
        right.flush_wall = Duration::from_millis(41);
        right.gpu_h2d_submit_wall = Duration::from_millis(43);
        right.gpu_h2d_event_elapsed = Duration::from_millis(47);
        right.gpu_kernel_launch = Duration::from_millis(53);
        right.gpu_kernel_elapsed = Duration::from_millis(59);
        right.gpu_d2h_submit_wall = Duration::from_millis(61);
        right.gpu_d2h_event_elapsed = Duration::from_millis(67);
        right.gpu_post_d2h_sync_wall = Duration::from_millis(71);
        right.gpu_stream_total_elapsed = Duration::from_millis(73);
        right.compact_lune_witness_enabled = true;
        right.hashprune_backfill.shortlist_admissions = 5;

        left.merge(right);

        assert_eq!(left.total_wall, Duration::from_millis(17));
        assert_eq!(left.distance, Duration::from_millis(6));
        assert_eq!(left.load, Duration::from_millis(2));
        assert_eq!(left.flush, Duration::from_millis(3));
        assert_eq!(left.leaves, 2);
        assert_eq!(left.host_load_wall, Duration::from_millis(12));
        assert_eq!(left.backend_compute_wall, Duration::from_millis(15));
        assert_eq!(left.compact_materialize_wall, Duration::from_millis(17));
        assert_eq!(left.hash_wall, Duration::from_millis(19));
        assert_eq!(left.edge_sink_wall, Duration::from_millis(23));
        assert_eq!(left.edge_sink_sort_wall, Duration::from_millis(29));
        assert_eq!(left.edge_sink_insert_wall, Duration::from_millis(31));
        assert_eq!(left.edge_sink_edges, 37);
        assert_eq!(left.edge_sink_flushes, 2);
        assert_eq!(left.flush_wall, Duration::from_millis(41));
        assert_eq!(left.gpu_h2d_submit_wall, Duration::from_millis(43));
        assert_eq!(left.gpu_h2d_event_elapsed, Duration::from_millis(50));
        assert_eq!(left.gpu_kernel_launch, Duration::from_millis(56));
        assert_eq!(left.gpu_kernel_elapsed, Duration::from_millis(63));
        assert_eq!(left.gpu_d2h_submit_wall, Duration::from_millis(61));
        assert_eq!(left.gpu_d2h_event_elapsed, Duration::from_millis(72));
        assert_eq!(left.gpu_post_d2h_sync_wall, Duration::from_millis(71));
        assert_eq!(left.gpu_stream_total_elapsed, Duration::from_millis(78));
        assert!(left.compact_lune_witness_enabled);
        assert_eq!(left.hashprune_backfill.shortlist_admissions, 5);
    }

    #[test]
    fn parallel_large_leaf_matches_serial_leaf_neighbors() {
        let num_points = 96usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = PiPNNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;
        params.max_full_matrix_leaf_size = 32;

        let serial_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let parallel_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();

        let mut serial_scratch = LeafScratch::new(leaf.len(), dim, params.leaf_knn);
        process_leaf(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &serial_reservoirs,
            &mut serial_scratch,
        )
        .unwrap();

        let parallel_profile = process_leaf_parallel_large_profiled(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &parallel_reservoirs,
            16,
        )
        .unwrap();

        let serial_neighbors = collect_neighbors(serial_reservoirs);
        let parallel_neighbors = collect_neighbors(parallel_reservoirs);

        assert_eq!(parallel_profile.blockwise_leaves, 1);
        assert_eq!(serial_neighbors, parallel_neighbors);
    }

    #[test]
    fn cpu_backend_leaf_path_matches_serial_leaf_neighbors() {
        let num_points = 96usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = PiPNNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;
        params.max_full_matrix_leaf_size = 32;

        let serial_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let backend_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();

        let mut serial_scratch = LeafScratch::new(leaf.len(), dim, params.leaf_knn);
        process_leaf(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &serial_reservoirs,
            &mut serial_scratch,
        )
        .unwrap();

        let backend_profile = super::process_leaf_parallel_large_profiled_with_backend(
            cpu_backend(),
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &backend_reservoirs,
            16,
        )
        .unwrap();

        let serial_neighbors = collect_neighbors(serial_reservoirs);
        let backend_neighbors = collect_neighbors(backend_reservoirs);

        assert_eq!(backend_profile.blockwise_leaves, 1);
        assert_eq!(serial_neighbors, backend_neighbors);
    }

    #[test]
    fn cpu_shadow_compact_topk_flush_matches_serial_leaf_neighbors() {
        let num_points = 12usize;
        let dim = 4usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = PiPNNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;

        let serial_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let shadow_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();

        let mut serial_scratch = LeafScratch::new(leaf.len(), dim, params.leaf_knn);
        process_leaf(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &serial_reservoirs,
            &mut serial_scratch,
        )
        .unwrap();

        let edge_sink = super::ReservoirEdgeSink::new(&shadow_reservoirs);
        let (compact, compute_profile) =
            compute_leaf_topk_l2_cpu_shadow(&store, &leaf, &params).unwrap();
        let flush_profile =
            flush_compact_topk_batch(&compact, &sketches, &params, &edge_sink).unwrap();

        let serial_neighbors = collect_neighbors(serial_reservoirs);
        let shadow_neighbors = collect_neighbors(shadow_reservoirs);

        assert_eq!(compact.rows, leaf.len());
        assert_eq!(compact.k, params.leaf_knn);
        assert!(compact.lune_witness_pivot_ranks.is_empty());
        assert!(compact.lune_witness_margins.is_empty());
        assert_eq!(compact.estimated_lune_witness_dtoh_bytes, 0);
        assert_eq!(compute_profile.points, leaf.len());
        assert_eq!(flush_profile.points, leaf.len());
        assert_eq!(serial_neighbors, shadow_neighbors);
    }

    #[test]
    fn cpu_shadow_derives_lune_witnesses_when_compact_lune_prune_is_enabled() {
        let num_points = 12usize;
        let dim = 4usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = PiPNNParams {
            leaf_knn: 2,
            ..PiPNNParams::default()
        };
        params.compact_lune_prune_enable = true;

        let (compact, _profile) = compute_leaf_topk_l2_cpu_shadow(&store, &leaf, &params).unwrap();

        assert_eq!(
            compact.lune_witness_pivot_ranks.len(),
            leaf.len() * params.leaf_knn
        );
        assert_eq!(
            compact.lune_witness_margins.len(),
            leaf.len() * params.leaf_knn
        );
        assert_eq!(
            compact.estimated_lune_witness_dtoh_bytes,
            leaf.len() * params.leaf_knn * (size_of::<u32>() + size_of::<f32>())
        );
    }

    #[test]
    fn compact_lune_witness_marks_pivot_for_farther_victim() {
        let mut batch = CompactTopKBatch {
            leaf_id: 11,
            point_ids: vec![10, 20, 30],
            rows: 3,
            k: 2,
            topk_ids: vec![
                20, 30, // src 10: pivot 20 is closer than victim 30.
                10, 30, // pivot row 20 confirms 30 is closer to pivot than to src 10.
                20, 10,
            ],
            topk_dists: vec![
                1.0, 4.0, // dist(10,20), dist(10,30)
                1.0, 2.0, // dist(20,10), dist(20,30)
                2.0, 4.0,
            ],
            estimated_dtoh_bytes: 3 * 2 * (size_of::<u32>() + size_of::<f32>()),
            ..CompactTopKBatch::default()
        };

        super::derive_compact_lune_witnesses_from_topk(&mut batch, 1.2, 4).unwrap();

        assert_eq!(batch.lune_witness_pivot_ranks.len(), batch.rows * batch.k);
        assert_eq!(batch.lune_witness_margins.len(), batch.rows * batch.k);
        assert_eq!(
            batch.lune_witness_pivot_ranks[0],
            super::COMPACT_LUNE_NO_WITNESS
        );
        assert_eq!(batch.lune_witness_pivot_ranks[1], 0);
        assert!(batch.lune_witness_margins[1] > 0.0);
    }

    #[test]
    fn oom_compact_topk_sink_flushes_batch_into_edge_sink() {
        let params = PiPNNParams {
            leaf_knn: 1,
            final_prune: false,
            ..PiPNNParams::default()
        };
        let sketches = build_test_sketches(2, params.m_hash_bits);
        let reservoirs: Vec<_> = (0..2)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(&reservoirs);
        let compact_sink = super::OomCompactTopKSink::new(&sketches, &params, &edge_sink);
        let batch = super::CompactTopKBatch {
            leaf_id: 7,
            point_ids: vec![0, 1],
            rows: 2,
            k: 1,
            topk_ids: vec![1, 0],
            topk_dists: vec![1.0, 1.0],
            estimated_dtoh_bytes: 16,
            ..super::CompactTopKBatch::default()
        };

        let profile = super::CompactTopKSink::flush_compact_topk(&compact_sink, batch).unwrap();

        assert_eq!(profile.leaves, 1);
        assert_eq!(profile.points, 2);
        assert!(
            reservoirs
                .iter()
                .all(|reservoir| !reservoir.lock().slots().is_empty())
        );
    }

    #[test]
    fn cpu_shadow_compact_profiles_mirror_semantic_leaf_flow_fields() {
        let num_points = 8usize;
        let dim = 4usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = PiPNNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;

        let reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(&reservoirs);

        let (compact, compute_profile) =
            compute_leaf_topk_l2_cpu_shadow(&store, &leaf, &params).unwrap();
        let flush_profile =
            flush_compact_topk_batch(&compact, &sketches, &params, &edge_sink).unwrap();

        assert_eq!(compute_profile.host_load_wall, compute_profile.load);
        assert_eq!(
            compute_profile.backend_compute_wall,
            compute_profile.distance + compute_profile.topk
        );
        assert_eq!(compute_profile.compact_materialize_wall, Duration::ZERO);
        assert_eq!(compute_profile.hash_wall, Duration::ZERO);
        assert_eq!(compute_profile.flush_wall, Duration::ZERO);
        assert_eq!(compute_profile.edge_sink_wall, Duration::ZERO);

        assert!(flush_profile.hash_wall <= flush_profile.hash);
        assert_eq!(flush_profile.flush_wall, flush_profile.flush);
        assert_eq!(flush_profile.edge_sink_wall, flush_profile.flush);
        assert_eq!(flush_profile.host_load_wall, Duration::ZERO);
        assert_eq!(flush_profile.backend_compute_wall, Duration::ZERO);
        assert_eq!(flush_profile.compact_materialize_wall, Duration::ZERO);
    }

    #[test]
    fn compact_flush_hash_wall_excludes_edge_sink_wall() {
        let sketches = build_test_sketches(1, 1);
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 1;
        let sink = SleepingEdgeSink {
            delay: Duration::from_millis(20),
            flushes: AtomicUsize::new(0),
        };
        let k = 8192usize;
        let batch = CompactTopKBatch {
            point_ids: vec![0],
            rows: 1,
            k,
            topk_ids: vec![0; k],
            topk_dists: vec![1.0; k],
            estimated_dtoh_bytes: k.saturating_mul(size_of::<u32>() + size_of::<f32>()),
            ..CompactTopKBatch::default()
        };

        let profile = flush_compact_topk_batch(&batch, &sketches, &params, &sink).unwrap();

        assert_eq!(sink.flushes.load(Ordering::Relaxed), 1);
        assert_eq!(profile.flush_wall, profile.flush);
        assert_eq!(profile.edge_sink_wall, profile.flush);
        assert!(profile.edge_sink_wall >= sink.delay);
        assert!(
            profile.hash > profile.hash_wall,
            "legacy hash should include the delayed in-loop flush while hash_wall excludes it: {profile:?}"
        );
        assert!(
            profile.hash_wall < profile.hash,
            "hash_wall should be narrower than legacy hash when edge sink blocks: {profile:?}"
        );
    }

    #[test]
    fn compact_superbatch_flush_combines_edge_sink_writes_across_batches() {
        let sketches = build_test_sketches(4, 1);
        let mut params = PiPNNParams {
            leaf_knn: 1,
            ..PiPNNParams::default()
        };
        params.m_hash_bits = 1;
        let sink = CountingEdgeSink::default();
        let compact_sink = super::OomCompactTopKSink::new(&sketches, &params, &sink);
        let first = CompactTopKBatch {
            leaf_id: 1,
            rows: 2,
            k: 1,
            point_ids: vec![0, 1],
            topk_ids: vec![1, 0],
            topk_dists: vec![1.0, 1.0],
            estimated_dtoh_bytes: 2 * (size_of::<u32>() + size_of::<f32>()),
            ..CompactTopKBatch::default()
        };
        let second = CompactTopKBatch {
            leaf_id: 2,
            rows: 2,
            k: 1,
            point_ids: vec![2, 3],
            topk_ids: vec![3, 2],
            topk_dists: vec![1.0, 1.0],
            estimated_dtoh_bytes: 2 * (size_of::<u32>() + size_of::<f32>()),
            ..CompactTopKBatch::default()
        };

        let profile = compact_sink
            .flush_compact_topk_batches(vec![first, second])
            .unwrap();

        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert_eq!(sink.flushes.load(Ordering::Relaxed), 1);
        assert_eq!(sink.edges.lock().len(), 8);
    }

    #[test]
    fn compact_local_index_uses_linear_lookup_for_small_batches() {
        let batch = CompactTopKBatch {
            rows: 3,
            k: 2,
            point_ids: vec![10, 42, 7],
            topk_ids: vec![42, 7, 10, 7, 10, 42],
            topk_dists: vec![1.0; 6],
            ..CompactTopKBatch::default()
        };

        let local_index = super::compact_local_index(&batch);

        assert!(matches!(local_index, super::CompactLocalIndex::Linear(_)));
        assert_eq!(local_index.lookup(10), Some(0));
        assert_eq!(local_index.lookup(42), Some(1));
        assert_eq!(local_index.lookup(99), None);
    }

    #[test]
    fn compact_local_index_keeps_hash_lookup_for_large_batches() {
        let rows = super::COMPACT_TOPK_LINEAR_LOCAL_INDEX_MAX_ROWS + 1;
        let batch = CompactTopKBatch {
            rows,
            k: 2,
            point_ids: (0..rows as u32).map(|point| point * 3).collect(),
            topk_ids: vec![0; rows * 2],
            topk_dists: vec![1.0; rows * 2],
            ..CompactTopKBatch::default()
        };

        let local_index = super::compact_local_index(&batch);

        assert!(matches!(local_index, super::CompactLocalIndex::Hash(_)));
        assert_eq!(local_index.lookup(12), Some(4));
        assert_eq!(local_index.lookup(13), None);
    }

    #[test]
    fn cpu_shadow_compact_topk_uses_target_id_tie_break_and_excludes_self() {
        let num_points = 4usize;
        let dim = 1usize;
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        dataset.data.copy_from_slice(&[0.0, 1.0, -1.0, 5.0]);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let leaf = vec![0_u32, 2, 1, 3];

        let mut params = PiPNNParams::default();
        params.leaf_knn = 2;

        let (compact, _profile) = compute_leaf_topk_l2_cpu_shadow(&store, &leaf, &params).unwrap();

        assert_eq!(compact.point_ids, leaf);
        assert_eq!(compact.rows, leaf.len());
        assert_eq!(compact.k, 2);
        assert_eq!(
            compact.estimated_dtoh_bytes,
            leaf.len() * 2 * (size_of::<u32>() + size_of::<f32>())
        );

        let row0 = &compact.topk_ids[0..2];
        let row0_dists = &compact.topk_dists[0..2];
        assert_eq!(row0, &[1, 2]);
        assert_eq!(row0_dists, &[1.0, 1.0]);
        assert!(!row0.contains(&0));
    }

    #[test]
    fn row_topk_uses_target_id_tie_break_for_near_equal_distances() {
        let mut topk = RowTopK::default();

        topk.push(1, 0, 30, 1.0);
        topk.push(1, 1, 20, 1.0 + 1.5e-5);

        let neighbors: Vec<_> = topk.iter().collect();
        assert_eq!(neighbors, vec![(1, 1.0 + 1.5e-5)]);
    }

    #[test]
    fn cpu_shadow_compact_topk_truncates_k_and_handles_empty_or_singleton_leaf() {
        let num_points = 3usize;
        let dim = 1usize;
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        dataset.data.copy_from_slice(&[0.0, 1.0, 2.0]);
        let store = InmemDatasetPointStore::new(&dataset, num_points);

        let mut params = PiPNNParams::default();
        params.leaf_knn = 8;

        let leaf = vec![0_u32, 1, 2];
        let (compact, _profile) = compute_leaf_topk_l2_cpu_shadow(&store, &leaf, &params).unwrap();
        assert_eq!(compact.rows, leaf.len());
        assert_eq!(compact.k, leaf.len() - 1);
        assert_eq!(compact.topk_ids.len(), leaf.len() * (leaf.len() - 1));
        assert_eq!(compact.topk_dists.len(), leaf.len() * (leaf.len() - 1));

        let (empty, _profile) = compute_leaf_topk_l2_cpu_shadow(&store, &[], &params).unwrap();
        assert_eq!(empty.point_ids, Vec::<u32>::new());
        assert_eq!(empty.rows, 0);
        assert_eq!(empty.k, 0);
        assert!(empty.topk_ids.is_empty());
        assert!(empty.topk_dists.is_empty());
        assert_eq!(empty.estimated_dtoh_bytes, 0);

        let (single, _profile) = compute_leaf_topk_l2_cpu_shadow(&store, &[1], &params).unwrap();
        assert_eq!(single.point_ids, vec![1]);
        assert_eq!(single.rows, 1);
        assert_eq!(single.k, 0);
        assert!(single.topk_ids.is_empty());
        assert!(single.topk_dists.is_empty());
        assert_eq!(single.estimated_dtoh_bytes, 0);
    }
}
