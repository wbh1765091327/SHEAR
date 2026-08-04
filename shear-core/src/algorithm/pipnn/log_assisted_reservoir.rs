use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender, bounded};
use memmap2::{Advice, Mmap, MmapOptions};
use parking_lot::{Condvar, Mutex};

use super::hash_prune::{HashPruneInsertProfile, HashPruneReservoir, HashPruneReservoirState};
use super::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use super::hash_prune_diagnostics::HashPruneTraceCollector;
use super::hash_prune_evidence::HashPruneEvidencePolicy;
use super::leaf_build::PendingEdge;
use super::online_suffix_hashprune::{
    OnlineFallbackReaderGuard, OnlineFallbackRegistry, OnlineSpillSegment, apply_edge,
};
use super::range_reservoir_store::{
    RangeBaseReader, RangeBaseRow, RangeBaseStore, RangeBaseStoreConfig,
};
use super::reservoir_delta_log::{
    CheckpointRunDesc, CheckpointRunReader, DeltaRunDesc, DeltaRunReader,
    ReservoirCheckpointRecord, ReservoirDeltaRecord, write_checkpoint_run_with_id,
    write_delta_run_with_id,
};
use super::spill_hashprune::{RUN_RECORD_BYTES, decode_spill_record16};
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug)]
pub(crate) struct FrozenRangeEpoch {
    pub(crate) range_id: usize,
    pub(crate) epoch: u64,
    pub(crate) cut_ticket: u64,
    pub(crate) delta_runs: Vec<DeltaRunDesc>,
    borrowed_delta_runs: Vec<BorrowedDeltaRunDesc>,
    pub(crate) checkpoint_runs: Vec<CheckpointRunDesc>,
    cache_snapshots: Vec<CacheSnapshotMarker>,
    pub(crate) logical_bytes: usize,
    log_only: bool,
}

#[derive(Clone, Debug)]
struct BorrowedDeltaRunDesc {
    registry: OnlineFallbackRegistry,
    segment: OnlineSpillSegment,
    path: PathBuf,
    range_id: usize,
    epoch: u64,
    run_id: u64,
    record_count: u64,
    byte_len: u64,
    min_ticket: u64,
    max_ticket: u64,
}

#[derive(Clone, Debug)]
struct CacheSnapshotMarker {
    source: u32,
    covered_ticket: u64,
    cache_generation: u32,
}

#[derive(Clone, Debug)]
struct PromotionCandidate {
    source: u32,
    update_count: u64,
    latest_ticket: u64,
    state: HashPruneReservoirState,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RangeCompactionResult {
    pub(crate) base_generation: u64,
    pub(crate) cut_ticket: u64,
    pub(crate) rows_compacted: u64,
    pub(crate) rows_changed: u64,
    pub(crate) delta_records_seen: u64,
    pub(crate) delta_records_applied: u64,
    pub(crate) delta_records_covered: u64,
    pub(crate) checkpoint_records_seen: u64,
    pub(crate) logical_bytes: usize,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) read_wall: Duration,
    pub(crate) merge_wall: Duration,
    pub(crate) write_wall: Duration,
    pub(crate) hot_sources: Vec<(u32, u64)>,
    promotion_candidates: Vec<PromotionCandidate>,
    pub(crate) insert_profile: HashPruneInsertProfile,
    pub(crate) backfill_profile: HashPruneBackfillProfile,
}

#[derive(Clone)]
pub(crate) struct LogAssistedReservoirConfig {
    pub(crate) root: PathBuf,
    pub(crate) source_start: usize,
    pub(crate) source_end: usize,
    pub(crate) shard_points: usize,
    pub(crate) range_points: usize,
    pub(crate) l_max: usize,
    pub(crate) delta_budget_bytes: usize,
    pub(crate) delta_buffer_bytes: usize,
    pub(crate) range_trigger_bytes: usize,
    pub(crate) max_runs_per_range: usize,
    pub(crate) compaction_workers: usize,
    pub(crate) drain_compaction_workers: usize,
    pub(crate) cache_capacity_entries: usize,
    pub(crate) cache_hot_updates: usize,
    pub(crate) promotion_limit_per_compaction: usize,
    pub(crate) policy_key: [u8; 32],
    pub(crate) evidence_policy: HashPruneEvidencePolicy,
    pub(crate) backfill_policy: HashPruneBackfillPolicy,
    pub(crate) diagnostic_collector: Option<Arc<HashPruneTraceCollector>>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LogAssistedApplyProfiles {
    pub(crate) delta_records: u64,
    pub(crate) buffered_bytes: u64,
    pub(crate) buffer_flushes: u64,
    pub(crate) cache_hits: u64,
    pub(crate) insert_profile: HashPruneInsertProfile,
    pub(crate) backfill_profile: HashPruneBackfillProfile,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LogAssistedStats {
    pub(crate) base_reads: u64,
    pub(crate) base_writes: u64,
    pub(crate) cold_miss_base_reads: u64,
    pub(crate) cold_batch_commits: u64,
    pub(crate) cold_batch_records: u64,
    pub(crate) delta_records: u64,
    pub(crate) delta_runs: u64,
    pub(crate) delta_bytes_written: u64,
    pub(crate) borrowed_spill_records: u64,
    pub(crate) borrowed_spill_runs: u64,
    pub(crate) borrowed_spill_bytes: u64,
    pub(crate) delta_buffer_flushes: u64,
    pub(crate) run_pressure_crossings: u64,
    pub(crate) run_pressure_compactions: u64,
    pub(crate) pre_gpu_materializations: u64,
    pub(crate) shard_finalizations: u64,
    pub(crate) ingress_drained_compactions: u64,
    pub(crate) peak_active_delta_runs_per_range: u64,
    pub(crate) compactions: u64,
    pub(crate) compactions_inflight: u64,
    pub(crate) peak_compactions_inflight: u64,
    pub(crate) post_gpu_peak_compactions_inflight: u64,
    pub(crate) rows_compacted: u64,
    pub(crate) rows_changed: u64,
    pub(crate) compaction_input_bytes: u64,
    pub(crate) compaction_output_bytes: u64,
    pub(crate) compaction_read_wall: Duration,
    pub(crate) compaction_merge_wall: Duration,
    pub(crate) compaction_write_wall: Duration,
    pub(crate) compaction_hidden_wall: Duration,
    pub(crate) manifest_switches: u64,
    pub(crate) current_uncompacted_bytes: u64,
    pub(crate) peak_uncompacted_bytes: u64,
    pub(crate) delta_backpressure_count: u64,
    pub(crate) delta_backpressure_wall: Duration,
    pub(crate) checkpoint_records: u64,
    pub(crate) checkpoint_runs: u64,
    pub(crate) checkpoint_bytes_written: u64,
    pub(crate) cache_promotions: u64,
    pub(crate) cache_promotion_skipped_post_cut: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_dirty_checkpoints: u64,
    pub(crate) post_gpu_uncompacted_bytes: u64,
    pub(crate) post_gpu_compaction_wall: Duration,
    pub(crate) insert_profile: HashPruneInsertProfile,
    pub(crate) backfill_profile: HashPruneBackfillProfile,
}

struct RangeMutable {
    epoch: u64,
    next_ticket: u64,
    next_run_id: u64,
    active_delta_runs: Vec<DeltaRunDesc>,
    active_borrowed_delta_runs: Vec<BorrowedDeltaRunDesc>,
    active_checkpoint_runs: Vec<CheckpointRunDesc>,
    pending_checkpoint_records: Vec<ReservoirCheckpointRecord>,
    active_latest: BTreeMap<u32, u64>,
    active_bytes: usize,
    pre_gpu_materialized: bool,
    frozen: Option<FrozenRangeEpoch>,
    compaction_state: CompactionState,
    log_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactionState {
    Idle,
    Scheduled,
    Running,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RangeCompactionTrigger {
    RunPressure,
    BytePressure,
}

struct RangeState {
    base: RangeBaseStore,
    run_trigger_bytes: usize,
    mutable: Mutex<RangeMutable>,
}

#[derive(Clone, Copy)]
struct BufferedDelta {
    range_id: usize,
    record: ReservoirDeltaRecord,
}

#[derive(Default)]
struct OwnerDeltaBuffer {
    records: Vec<BufferedDelta>,
    logical_bytes: usize,
}

struct ActiveCache {
    source_start: usize,
    shard_points: usize,
    checkpoint_bytes: usize,
    shards: Vec<Mutex<CacheShard>>,
}

struct CacheShard {
    capacity: usize,
    entries: HashMap<u32, CacheEntry>,
    clock: VecDeque<ClockToken>,
    next_token: u64,
}

struct CacheEntry {
    reservoir: HashPruneReservoir,
    latest_ticket: u64,
    base_generation: u64,
    state_generation: u32,
    dirty_reservation_bytes: usize,
    inflight_snapshot: Option<CacheSnapshotMarker>,
    referenced: bool,
    clock_token: u64,
    evicting: bool,
}

#[derive(Clone, Copy)]
struct ClockToken {
    source: u32,
    token: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheRoute {
    Miss,
    HitReserved,
    HitNeedsReservation,
}

enum ReservedApplyOutcome {
    Applied {
        consumed_bytes: usize,
        range_id: usize,
        cache_hit: bool,
        insert_profile: HashPruneInsertProfile,
        backfill_profile: HashPruneBackfillProfile,
    },
    Retry,
}

pub(crate) struct LogAssistedReservoir {
    config: Arc<LogAssistedReservoirConfig>,
    ranges: Arc<Vec<Arc<RangeState>>>,
    cache: Arc<ActiveCache>,
    owner_buffers: Vec<Mutex<OwnerDeltaBuffer>>,
    owner_operations: Vec<Mutex<()>>,
    drained_shards: Vec<AtomicBool>,
    owner_flush_bytes: usize,
    // Epoch cuts take this lock, then every owner operation lock in ascending order.
    operation: Mutex<()>,
    stats: Arc<Mutex<LogAssistedStats>>,
    budget: Arc<DeltaBudget>,
    first_error: Arc<FirstError>,
    compaction_tx: Sender<CompactionJob>,
    compaction_worker_gate: Arc<CompactionWorkerGate>,
    idle_signal: Arc<IdleSignal>,
    gpu_complete_at: Arc<Mutex<Option<Instant>>>,
    worker_handles: Mutex<Option<Vec<JoinHandle<()>>>>,
    accepting: Arc<AtomicBool>,
    promotions_enabled: Arc<AtomicBool>,
    finished: AtomicBool,
    #[cfg(test)]
    compaction_gate: Arc<TestCompactionGate>,
    #[cfg(test)]
    admission_gate: Arc<TestAdmissionGate>,
}

struct DeltaBudgetState {
    used: usize,
    peak: usize,
}

struct DeltaBudget {
    limit: usize,
    high: usize,
    state: Mutex<DeltaBudgetState>,
    changed: Condvar,
}

#[derive(Default)]
struct FirstError {
    message: Mutex<Option<String>>,
}

#[derive(Default)]
struct IdleSignal {
    generation: Mutex<u64>,
    changed: Condvar,
}

struct CompactionWorkerGate {
    state: Mutex<CompactionWorkerGateState>,
    changed: Condvar,
}

struct CompactionWorkerGateState {
    active_limit: usize,
    total_workers: usize,
    shutdown: bool,
}

#[derive(Clone, Copy)]
enum CompactionJob {
    Range(usize),
    Shutdown,
}

#[derive(Clone)]
struct CompactionWorkerContext {
    config: Arc<LogAssistedReservoirConfig>,
    ranges: Arc<Vec<Arc<RangeState>>>,
    cache: Arc<ActiveCache>,
    stats: Arc<Mutex<LogAssistedStats>>,
    budget: Arc<DeltaBudget>,
    first_error: Arc<FirstError>,
    compaction_worker_gate: Arc<CompactionWorkerGate>,
    idle_signal: Arc<IdleSignal>,
    gpu_complete_at: Arc<Mutex<Option<Instant>>>,
    accepting: Arc<AtomicBool>,
    promotions_enabled: Arc<AtomicBool>,
    #[cfg(test)]
    compaction_gate: Arc<TestCompactionGate>,
}

struct CompactionInFlightGuard {
    stats: Arc<Mutex<LogAssistedStats>>,
}

#[cfg(test)]
#[derive(Default)]
struct TestCompactionGate {
    blocked: Mutex<bool>,
    changed: Condvar,
}

#[cfg(test)]
#[derive(Default)]
struct TestAdmissionGate {
    state: Mutex<TestAdmissionGateState>,
    changed: Condvar,
}

#[cfg(test)]
#[derive(Default)]
struct TestAdmissionGateState {
    blocked: bool,
    entered: usize,
}

impl ActiveCache {
    fn new(
        source_start: usize,
        shard_points: usize,
        shard_count: usize,
        capacity: usize,
        checkpoint_bytes: usize,
    ) -> Self {
        let base_capacity = capacity / shard_count;
        let extra = capacity % shard_count;
        Self {
            source_start,
            shard_points,
            checkpoint_bytes,
            shards: (0..shard_count)
                .map(|shard_id| {
                    Mutex::new(CacheShard {
                        capacity: base_capacity + usize::from(shard_id < extra),
                        entries: HashMap::new(),
                        clock: VecDeque::new(),
                        next_token: 1,
                    })
                })
                .collect(),
        }
    }

    fn shard_id(&self, source: u32) -> AnnResult<usize> {
        let source = source as usize;
        let relative = source.checked_sub(self.source_start).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "cache source {source} precedes configured source start {}",
                self.source_start
            ))
        })?;
        let shard_id = relative / self.shard_points;
        if shard_id >= self.shards.len() {
            return Err(AnnError::log_index_error(format!(
                "cache source {source} maps outside {} shards",
                self.shards.len()
            )));
        }
        Ok(shard_id)
    }

    fn route(&self, source: u32) -> AnnResult<CacheRoute> {
        let shard_id = self.shard_id(source)?;
        Ok(cache_route_locked(&self.shards[shard_id].lock(), source))
    }

    fn route_runs(
        &self,
        shard_id: usize,
        edges: &[PendingEdge],
    ) -> AnnResult<Vec<(usize, usize, bool)>> {
        let shard = self.shards.get(shard_id).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "cache route run references unknown shard {shard_id}"
            ))
        })?;
        let shard = shard.lock();
        let mut runs = Vec::new();
        let mut start = 0;
        while start < edges.len() {
            let source = u32::try_from(edges[start].p)?;
            let mut end = start + 1;
            while end < edges.len() && edges[end].p == edges[start].p {
                end += 1;
            }
            let is_miss = cache_route_locked(&shard, source) == CacheRoute::Miss;
            if let Some((_, run_end, run_is_miss)) = runs.last_mut()
                && *run_is_miss == is_miss
            {
                *run_end = end;
            } else {
                runs.push((start, end, is_miss));
            }
            start = end;
        }
        Ok(runs)
    }

    fn contains(&self, source: u32) -> AnnResult<bool> {
        let shard_id = self.shard_id(source)?;
        Ok(self.shards[shard_id]
            .lock()
            .entries
            .get(&source)
            .is_some_and(|entry| !entry.evicting))
    }

    fn snapshot_range(
        &self,
        source_start: u32,
        source_end: u32,
    ) -> AnnResult<(Vec<ReservoirCheckpointRecord>, Vec<CacheSnapshotMarker>)> {
        let mut records = Vec::new();
        let mut markers = Vec::new();
        for shard in &self.shards {
            let mut shard = shard.lock();
            let mut sources = shard
                .entries
                .keys()
                .copied()
                .filter(|source| *source >= source_start && *source < source_end)
                .collect::<Vec<_>>();
            sources.sort_unstable();
            for source in sources {
                let entry = shard
                    .entries
                    .get_mut(&source)
                    .expect("cache source was collected from this shard");
                if entry.evicting || entry.dirty_reservation_bytes == 0 {
                    continue;
                }
                let marker = CacheSnapshotMarker {
                    source,
                    covered_ticket: entry.latest_ticket,
                    cache_generation: entry.state_generation,
                };
                records.push(ReservoirCheckpointRecord::from_reservoir(
                    source,
                    marker.covered_ticket,
                    marker.cache_generation,
                    &entry.reservoir,
                )?);
                entry.dirty_reservation_bytes = 0;
                entry.inflight_snapshot = Some(marker.clone());
                markers.push(marker);
            }
        }
        Ok((records, markers))
    }

    fn reconcile_snapshots(
        &self,
        snapshots: &[CacheSnapshotMarker],
        base_generation: u64,
    ) -> AnnResult<()> {
        for snapshot in snapshots {
            let shard_id = self.shard_id(snapshot.source)?;
            let mut shard = self.shards[shard_id].lock();
            let Some(entry) = shard.entries.get_mut(&snapshot.source) else {
                continue;
            };
            if entry.inflight_snapshot.as_ref().is_some_and(|current| {
                current.cache_generation == snapshot.cache_generation
                    && current.covered_ticket == snapshot.covered_ticket
            }) {
                entry.inflight_snapshot = None;
            }
            if entry.state_generation == snapshot.cache_generation
                && entry.latest_ticket == snapshot.covered_ticket
            {
                if entry.dirty_reservation_bytes != 0 {
                    return Err(AnnError::log_index_error(format!(
                        "cache source {} retained a dirty reservation without a post-cut update",
                        snapshot.source
                    )));
                }
                entry.base_generation = base_generation;
            } else if entry.dirty_reservation_bytes == 0 {
                return Err(AnnError::log_index_error(format!(
                    "cache source {} changed after a cut without reserving its next checkpoint",
                    snapshot.source
                )));
            }
        }
        Ok(())
    }

    fn has_dirty_state(&self) -> bool {
        self.shards.iter().any(|shard| {
            shard.lock().entries.values().any(|entry| {
                entry.dirty_reservation_bytes != 0 || entry.inflight_snapshot.is_some()
            })
        })
    }
}

impl CacheShard {
    fn select_clean_victim(&mut self, excluded_source: u32) -> Option<u32> {
        let max_scans = self
            .clock
            .len()
            .saturating_mul(2)
            .saturating_add(self.entries.len())
            .saturating_add(1);
        for _ in 0..max_scans {
            let token = self.clock.pop_front()?;
            let Some(entry) = self.entries.get_mut(&token.source) else {
                continue;
            };
            if entry.clock_token != token.token {
                continue;
            }
            if token.source == excluded_source
                || entry.evicting
                || entry.dirty_reservation_bytes != 0
                || entry.inflight_snapshot.is_some()
            {
                self.clock.push_back(token);
                continue;
            }
            if entry.referenced {
                entry.referenced = false;
                self.clock.push_back(token);
                continue;
            }
            self.clock.push_back(token);
            return Some(token.source);
        }
        None
    }

    fn insert_candidate(
        &mut self,
        candidate: PromotionCandidate,
        base_generation: u64,
        l_max: usize,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<()> {
        let token = self.next_token;
        self.next_token = token.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error("active-cache CLOCK token overflows u64".to_string())
        })?;
        let source = candidate.source;
        let entry = CacheEntry {
            reservoir: HashPruneReservoir::from_state(l_max, candidate.state, backfill_policy)?,
            latest_ticket: candidate.latest_ticket,
            base_generation,
            state_generation: 1,
            dirty_reservation_bytes: 0,
            inflight_snapshot: None,
            referenced: true,
            clock_token: token,
            evicting: false,
        };
        if self.entries.insert(source, entry).is_some() {
            return Err(AnnError::log_index_error(format!(
                "duplicate active-cache promotion for source {source}"
            )));
        }
        self.clock.push_back(ClockToken { source, token });
        Ok(())
    }
}

fn cache_route_locked(shard: &CacheShard, source: u32) -> CacheRoute {
    match shard.entries.get(&source) {
        Some(entry) if !entry.evicting && entry.dirty_reservation_bytes != 0 => {
            CacheRoute::HitReserved
        }
        Some(entry) if !entry.evicting => CacheRoute::HitNeedsReservation,
        _ => CacheRoute::Miss,
    }
}

impl DeltaBudget {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            high: limit.saturating_sub(limit / 5),
            state: Mutex::new(DeltaBudgetState { used: 0, peak: 0 }),
            changed: Condvar::new(),
        }
    }

    fn reserve(
        &self,
        bytes: usize,
        accepting: &AtomicBool,
        first_error: &FirstError,
        mut make_progress: impl FnMut() -> AnnResult<()>,
    ) -> AnnResult<(bool, Duration)> {
        if bytes > self.limit {
            return Err(AnnError::log_index_error(format!(
                "delta reservation {bytes} exceeds hard budget {}",
                self.limit
            )));
        }
        let started = Instant::now();
        let mut blocked = false;
        let mut state = self.state.lock();
        loop {
            first_error.check()?;
            if !accepting.load(AtomicOrdering::Acquire) {
                return Err(AnnError::log_index_error(
                    "log-assisted admission is closed".to_string(),
                ));
            }
            if state.used <= self.limit - bytes {
                state.used += bytes;
                state.peak = state.peak.max(state.used);
                return Ok((blocked, started.elapsed()));
            }
            blocked = true;
            self.changed.wait_for(&mut state, Duration::from_millis(10));
            if state.used > self.limit - bytes {
                drop(state);
                make_progress()?;
                state = self.state.lock();
            }
        }
    }

    fn release(&self, bytes: usize) -> AnnResult<()> {
        let mut state = self.state.lock();
        state.used = state.used.checked_sub(bytes).ok_or_else(|| {
            AnnError::log_index_error("delta budget accounting underflow".to_string())
        })?;
        self.changed.notify_all();
        Ok(())
    }

    fn used(&self) -> usize {
        self.state.lock().used
    }

    fn is_high(&self) -> bool {
        self.used() >= self.high
    }

    fn would_block(&self, bytes: usize) -> bool {
        let state = self.state.lock();
        bytes > self.limit || state.used > self.limit.saturating_sub(bytes)
    }

    fn notify_all(&self) {
        self.changed.notify_all();
    }
}

impl FirstError {
    fn record(&self, error: impl std::fmt::Display) {
        let mut message = self.message.lock();
        if message.is_none() {
            *message = Some(error.to_string());
        }
    }

    fn check(&self) -> AnnResult<()> {
        if let Some(message) = self.message.lock().as_ref() {
            return Err(AnnError::log_index_error(format!(
                "log-assisted background failure: {message}"
            )));
        }
        Ok(())
    }

    fn has_error(&self) -> bool {
        self.message.lock().is_some()
    }
}

impl IdleSignal {
    fn notify(&self) {
        let mut generation = self.generation.lock();
        *generation = generation.wrapping_add(1);
        self.changed.notify_all();
    }

    fn wait_for_change(&self, observed: &mut u64) {
        let mut generation = self.generation.lock();
        while *generation == *observed {
            self.changed.wait(&mut generation);
        }
        *observed = *generation;
    }
}

impl CompactionWorkerGate {
    fn new(active_limit: usize, total_workers: usize) -> Self {
        debug_assert!(active_limit != 0 && active_limit <= total_workers);
        Self {
            state: Mutex::new(CompactionWorkerGateState {
                active_limit,
                total_workers,
                shutdown: false,
            }),
            changed: Condvar::new(),
        }
    }

    fn wait_until_enabled(&self, worker_id: usize, first_error: &FirstError) -> bool {
        let mut state = self.state.lock();
        loop {
            if state.shutdown || first_error.has_error() {
                return false;
            }
            if worker_id < state.active_limit {
                return true;
            }
            self.changed.wait(&mut state);
        }
    }

    fn activate(&self, active_limit: usize) {
        let mut state = self.state.lock();
        state.active_limit = active_limit.min(state.total_workers);
        self.changed.notify_all();
    }

    fn notify_error(&self) {
        let _state = self.state.lock();
        self.changed.notify_all();
    }

    fn shutdown(&self) {
        let mut state = self.state.lock();
        state.shutdown = true;
        self.changed.notify_all();
    }
}

impl CompactionInFlightGuard {
    fn start(context: &CompactionWorkerContext) -> Self {
        let post_gpu = context.gpu_complete_at.lock().is_some();
        let mut stats = context.stats.lock();
        stats.compactions_inflight = stats.compactions_inflight.saturating_add(1);
        stats.peak_compactions_inflight = stats
            .peak_compactions_inflight
            .max(stats.compactions_inflight);
        if post_gpu {
            stats.post_gpu_peak_compactions_inflight = stats
                .post_gpu_peak_compactions_inflight
                .max(stats.compactions_inflight);
        }
        drop(stats);
        Self {
            stats: Arc::clone(&context.stats),
        }
    }
}

impl Drop for CompactionInFlightGuard {
    fn drop(&mut self) {
        let mut stats = self.stats.lock();
        debug_assert_ne!(stats.compactions_inflight, 0);
        stats.compactions_inflight = stats.compactions_inflight.saturating_sub(1);
    }
}

#[cfg(test)]
impl TestCompactionGate {
    fn wait(&self) {
        let mut blocked = self.blocked.lock();
        self.changed.notify_all();
        while *blocked {
            self.changed.wait(&mut blocked);
        }
    }

    fn pause(&self) {
        *self.blocked.lock() = true;
    }

    fn release(&self) {
        *self.blocked.lock() = false;
        self.changed.notify_all();
    }
}

#[cfg(test)]
impl TestAdmissionGate {
    fn enter(&self) {
        let mut state = self.state.lock();
        state.entered += 1;
        self.changed.notify_all();
        while state.blocked {
            self.changed.wait(&mut state);
        }
        state.entered -= 1;
        self.changed.notify_all();
    }

    fn pause(&self) {
        let mut state = self.state.lock();
        assert_eq!(state.entered, 0);
        state.blocked = true;
    }

    fn release(&self) {
        let mut state = self.state.lock();
        state.blocked = false;
        self.changed.notify_all();
    }

    fn wait_for_entered(&self, expected: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock();
        while state.entered < expected {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            self.changed.wait_for(&mut state, remaining);
        }
        true
    }
}

impl LogAssistedReservoir {
    pub(crate) fn create(config: LogAssistedReservoirConfig) -> AnnResult<Self> {
        validate_log_assisted_config(&config)?;
        let base_root = config.root.join("base");
        let run_root = config.root.join("runs");
        fs::create_dir_all(&base_root)?;
        fs::create_dir_all(&run_root)?;

        let source_count = config.source_end - config.source_start;
        let owner_count = source_count.div_ceil(config.shard_points).max(1);
        let owner_flush_bytes = (config.delta_buffer_bytes / owner_count)
            .max(super::reservoir_delta_log::DELTA_RECORD_BYTES);
        let mut ranges = Vec::new();
        let mut source_start = config.source_start;
        while source_start < config.source_end {
            let source_end = source_start
                .saturating_add(config.range_points)
                .min(config.source_end);
            let range_id = ranges.len();
            let base = RangeBaseStore::create(RangeBaseStoreConfig {
                root: base_root.clone(),
                range_id,
                source_start: u32::try_from(source_start)?,
                source_end: u32::try_from(source_end)?,
                l_max: config.l_max,
                policy_key: config.policy_key,
                backfill_policy: config.backfill_policy,
            })?;
            let run_trigger_bytes =
                usize::try_from(base.expected_base_bytes()?)?.min(config.range_trigger_bytes);
            ranges.push(Arc::new(RangeState {
                base,
                run_trigger_bytes,
                mutable: Mutex::new(RangeMutable {
                    epoch: 0,
                    next_ticket: 1,
                    next_run_id: 0,
                    active_delta_runs: Vec::new(),
                    active_borrowed_delta_runs: Vec::new(),
                    active_checkpoint_runs: Vec::new(),
                    pending_checkpoint_records: Vec::new(),
                    active_latest: BTreeMap::new(),
                    active_bytes: 0,
                    pre_gpu_materialized: false,
                    frozen: None,
                    compaction_state: CompactionState::Idle,
                    log_only: false,
                }),
            }));
            source_start = source_end;
        }

        let config = Arc::new(config);
        let ranges = Arc::new(ranges);
        let checkpoint_bytes = ReservoirCheckpointRecord::encoded_bytes(
            &ranges
                .first()
                .expect("validated source range creates at least one range")
                .base
                .layout(),
        )?;
        if config.cache_capacity_entries != 0 && checkpoint_bytes > config.delta_budget_bytes {
            return Err(AnnError::log_index_config_error(
                "online_suffix_delta_budget_bytes".to_string(),
                format!(
                    "delta budget {} cannot hold one cache checkpoint of {checkpoint_bytes} bytes",
                    config.delta_budget_bytes
                ),
            ));
        }
        let cache = Arc::new(ActiveCache::new(
            config.source_start,
            config.shard_points,
            owner_count,
            config.cache_capacity_entries,
            checkpoint_bytes,
        ));
        let stats = Arc::new(Mutex::new(LogAssistedStats::default()));
        let budget = Arc::new(DeltaBudget::new(config.delta_budget_bytes));
        let first_error = Arc::new(FirstError::default());
        let idle_signal = Arc::new(IdleSignal::default());
        let gpu_complete_at = Arc::new(Mutex::new(None));
        let accepting = Arc::new(AtomicBool::new(true));
        let promotions_enabled = Arc::new(AtomicBool::new(true));
        let total_compaction_workers = config
            .compaction_workers
            .max(config.drain_compaction_workers);
        let compaction_worker_gate = Arc::new(CompactionWorkerGate::new(
            config.compaction_workers,
            total_compaction_workers,
        ));
        #[cfg(test)]
        let compaction_gate = Arc::new(TestCompactionGate::default());
        #[cfg(test)]
        let admission_gate = Arc::new(TestAdmissionGate::default());
        let (compaction_tx, compaction_rx) =
            bounded(ranges.len().max(total_compaction_workers).max(1));
        let worker_context = CompactionWorkerContext {
            config: Arc::clone(&config),
            ranges: Arc::clone(&ranges),
            cache: Arc::clone(&cache),
            stats: Arc::clone(&stats),
            budget: Arc::clone(&budget),
            first_error: Arc::clone(&first_error),
            compaction_worker_gate: Arc::clone(&compaction_worker_gate),
            idle_signal: Arc::clone(&idle_signal),
            gpu_complete_at: Arc::clone(&gpu_complete_at),
            accepting: Arc::clone(&accepting),
            promotions_enabled: Arc::clone(&promotions_enabled),
            #[cfg(test)]
            compaction_gate: Arc::clone(&compaction_gate),
        };
        let mut worker_handles = Vec::with_capacity(total_compaction_workers);
        for worker_id in 0..total_compaction_workers {
            let context = worker_context.clone();
            let rx = compaction_rx.clone();
            match thread::Builder::new()
                .name(format!("hashprune-log-compact-{worker_id}"))
                .spawn(move || compaction_worker_loop(worker_id, context, rx))
            {
                Ok(handle) => worker_handles.push(handle),
                Err(error) => {
                    compaction_worker_gate.shutdown();
                    drop(compaction_tx);
                    drop(compaction_rx);
                    for handle in worker_handles {
                        let _ = handle.join();
                    }
                    return Err(error.into());
                }
            }
        }

        Ok(Self {
            config,
            ranges,
            cache,
            owner_buffers: (0..owner_count)
                .map(|_| Mutex::new(OwnerDeltaBuffer::default()))
                .collect(),
            owner_operations: (0..owner_count).map(|_| Mutex::new(())).collect(),
            drained_shards: (0..owner_count).map(|_| AtomicBool::new(false)).collect(),
            owner_flush_bytes,
            operation: Mutex::new(()),
            stats,
            budget,
            first_error,
            compaction_tx,
            compaction_worker_gate,
            idle_signal,
            gpu_complete_at,
            worker_handles: Mutex::new(Some(worker_handles)),
            accepting,
            promotions_enabled,
            finished: AtomicBool::new(false),
            #[cfg(test)]
            compaction_gate,
            #[cfg(test)]
            admission_gate,
        })
    }

    pub(crate) fn apply_edges(
        &self,
        shard_id: usize,
        edges: &[PendingEdge],
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<()> {
        let result = self.apply_edges_inner(shard_id, edges, profiles);
        if let Err(error) = &result {
            self.record_failure(error);
        }
        result
    }

    pub(crate) fn adopt_spill_segment(
        &self,
        shard_id: usize,
        registry: &OnlineFallbackRegistry,
        segment: &OnlineSpillSegment,
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<bool> {
        let result = self.adopt_spill_segment_inner(shard_id, registry, segment, profiles);
        if let Err(error) = &result {
            self.record_failure(error);
        }
        result
    }

    fn adopt_spill_segment_inner(
        &self,
        shard_id: usize,
        registry: &OnlineFallbackRegistry,
        segment: &OnlineSpillSegment,
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<bool> {
        if shard_id >= self.owner_buffers.len() || segment.shard_id != shard_id {
            return Err(AnnError::log_index_error(format!(
                "borrowed spill shard {} does not match log-assisted shard {shard_id}",
                segment.shard_id
            )));
        }
        self.ensure_shard_accepting(shard_id)?;
        if segment.desc.record_count == 0
            || !segment.desc.sorted_by_src
            || segment.desc.byte_len
                != u64::from(segment.desc.record_count) * RUN_RECORD_BYTES as u64
        {
            return Err(AnnError::log_index_error(
                "borrowed spill segment is empty, unsorted, or has an invalid byte length"
                    .to_string(),
            ));
        }
        let source_min = segment.desc.src_min as usize;
        let source_max = segment.desc.src_max as usize;
        if source_min > source_max
            || source_min < self.config.source_start
            || source_max >= self.config.source_end
        {
            return Err(AnnError::log_index_error(format!(
                "borrowed spill source interval [{source_min}, {source_max}] lies outside [{}, {})",
                self.config.source_start, self.config.source_end
            )));
        }
        let range_id = self.range_id(source_min);
        if self.range_id(source_max) != range_id
            || self.cache.shard_id(segment.desc.src_min)? != shard_id
            || self.cache.shard_id(segment.desc.src_max)? != shard_id
        {
            return Err(AnnError::log_index_error(
                "borrowed spill segment crosses a range or shard boundary".to_string(),
            ));
        }
        let reserved_bytes = usize::try_from(segment.desc.byte_len)?;
        if reserved_bytes > self.budget.limit {
            return Ok(false);
        }
        let path = registry.part_path(segment.part_id).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "borrowed spill part {} is no longer registered",
                segment.part_id
            ))
        })?;

        if self.budget.would_block(reserved_bytes) {
            self.schedule_budget_reclaim(profiles)?;
        }
        let (blocked, wait) =
            self.budget
                .reserve(reserved_bytes, &self.accepting, &self.first_error, || {
                    self.schedule_budget_reclaim(profiles)
                })?;
        if blocked {
            let mut stats = self.stats.lock();
            stats.delta_backpressure_count += 1;
            stats.delta_backpressure_wall += wait;
        }
        self.refresh_budget_stats()?;

        let adopted = {
            let _owner_operation = self.owner_operations[shard_id].lock();
            self.first_error.check().and_then(|()| {
                self.ensure_shard_accepting(shard_id)?;
                let mut range = self.ranges[range_id].mutable.lock();
                let record_count = u64::from(segment.desc.record_count);
                let min_ticket = range.next_ticket;
                let next_ticket = min_ticket.checked_add(record_count).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "range {range_id} borrowed ticket interval overflows u64"
                    ))
                })?;
                let max_ticket = next_ticket - 1;
                let run_id = range.next_run_id;
                range.next_run_id = run_id.checked_add(1).ok_or_else(|| {
                    AnnError::log_index_error(format!("range {range_id} run id overflows u64"))
                })?;
                range.active_bytes =
                    range
                        .active_bytes
                        .checked_add(reserved_bytes)
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "range borrowed spill bytes overflow usize".to_string(),
                            )
                        })?;
                let epoch = range.epoch;
                range.active_borrowed_delta_runs.push(BorrowedDeltaRunDesc {
                    registry: registry.clone(),
                    segment: segment.clone(),
                    path,
                    range_id,
                    epoch,
                    run_id,
                    record_count,
                    byte_len: u64::try_from(reserved_bytes)?,
                    min_ticket,
                    max_ticket,
                });
                range.next_ticket = next_ticket;
                range.log_only = true;
                Ok((
                    record_count,
                    range.active_delta_runs.len() + range.active_borrowed_delta_runs.len(),
                ))
            })
        };
        let (record_count, active_run_count) = match adopted {
            Ok(adopted) => adopted,
            Err(error) => {
                self.budget.release(reserved_bytes)?;
                self.refresh_budget_stats()?;
                return Err(error);
            }
        };

        profiles.delta_records = profiles.delta_records.saturating_add(record_count);
        profiles.buffered_bytes = profiles
            .buffered_bytes
            .saturating_add(u64::try_from(reserved_bytes)?);
        {
            let mut stats = self.stats.lock();
            stats.delta_records = stats.delta_records.saturating_add(record_count);
            stats.cold_batch_commits = stats.cold_batch_commits.saturating_add(1);
            stats.cold_batch_records = stats.cold_batch_records.saturating_add(record_count);
            stats.borrowed_spill_records =
                stats.borrowed_spill_records.saturating_add(record_count);
            stats.borrowed_spill_runs = stats.borrowed_spill_runs.saturating_add(1);
            stats.borrowed_spill_bytes = stats
                .borrowed_spill_bytes
                .saturating_add(u64::try_from(reserved_bytes)?);
            stats.peak_active_delta_runs_per_range = stats
                .peak_active_delta_runs_per_range
                .max(u64::try_from(active_run_count)?);
            if active_run_count == self.config.max_runs_per_range {
                stats.run_pressure_crossings = stats.run_pressure_crossings.saturating_add(1);
            }
        }
        self.after_accept(shard_id, range_id, profiles)?;
        Ok(true)
    }

    fn apply_edges_inner(
        &self,
        shard_id: usize,
        edges: &[PendingEdge],
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<()> {
        if shard_id >= self.owner_buffers.len() {
            return Err(AnnError::log_index_error(format!(
                "log-assisted shard {shard_id} is outside owner count {}",
                self.owner_buffers.len()
            )));
        }
        self.ensure_shard_accepting(shard_id)?;
        for edge in edges {
            self.validate_edge(shard_id, edge)?;
        }

        let record_bytes = super::reservoir_delta_log::DELTA_RECORD_BYTES;
        let max_batch_records =
            (self.owner_flush_bytes.min(self.budget.limit) / record_bytes).max(1);
        let mut range_start = 0;
        while range_start < edges.len() {
            let range_id = self.range_id(edges[range_start].p);
            let mut range_end = range_start + 1;
            while range_end < edges.len() && self.range_id(edges[range_end].p) == range_id {
                range_end += 1;
            }
            let route_runs = if self.ranges[range_id].mutable.lock().log_only {
                vec![(0, range_end - range_start, true)]
            } else {
                self.cache
                    .route_runs(shard_id, &edges[range_start..range_end])?
            };
            for (run_start, run_end, is_miss) in route_runs {
                let mut batch_start = range_start + run_start;
                let absolute_run_end = range_start + run_end;
                while batch_start < absolute_run_end {
                    let batch_end =
                        absolute_run_end.min(batch_start.saturating_add(max_batch_records));
                    let batch = &edges[batch_start..batch_end];
                    if !is_miss || !self.apply_cold_batch(shard_id, batch, profiles)? {
                        for edge in batch {
                            self.apply_one_edge(shard_id, edge, profiles)?;
                        }
                    }
                    batch_start = batch_end;
                }
            }
            range_start = range_end;
        }
        Ok(())
    }

    fn apply_cold_batch(
        &self,
        shard_id: usize,
        edges: &[PendingEdge],
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<bool> {
        if edges.is_empty() {
            return Ok(true);
        }
        if !self.cold_batch_routes_are_misses(shard_id, edges)? {
            return Ok(false);
        }
        let reserved_bytes = edges
            .len()
            .checked_mul(super::reservoir_delta_log::DELTA_RECORD_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error("cold delta batch bytes overflow usize".to_string())
            })?;
        loop {
            self.first_error.check()?;
            if !self.cold_batch_routes_are_misses(shard_id, edges)? {
                return Ok(false);
            }
            if self.budget.would_block(reserved_bytes) {
                self.schedule_budget_reclaim(profiles)?;
            }
            let (blocked, wait) =
                self.budget
                    .reserve(reserved_bytes, &self.accepting, &self.first_error, || {
                        self.schedule_budget_reclaim(profiles)
                    })?;
            if blocked {
                let mut stats = self.stats.lock();
                stats.delta_backpressure_count += 1;
                stats.delta_backpressure_wall += wait;
            }
            self.refresh_budget_stats()?;

            let outcome = {
                let _owner_operation = self.owner_operations[shard_id].lock();
                #[cfg(test)]
                self.admission_gate.enter();
                self.first_error.check().and_then(|()| {
                    self.ensure_shard_accepting(shard_id)?;
                    self.apply_reserved_cold_batch(shard_id, edges)
                })
            };
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.budget.release(reserved_bytes)?;
                    self.refresh_budget_stats()?;
                    return Err(error);
                }
            };
            let ReservedApplyOutcome::Applied {
                consumed_bytes,
                range_id,
                cache_hit,
                ..
            } = outcome
            else {
                self.budget.release(reserved_bytes)?;
                self.refresh_budget_stats()?;
                continue;
            };
            debug_assert!(!cache_hit);
            if consumed_bytes != reserved_bytes {
                self.budget.release(reserved_bytes)?;
                self.refresh_budget_stats()?;
                return Err(AnnError::log_index_error(format!(
                    "cold delta batch consumed {consumed_bytes} bytes from a {reserved_bytes}-byte reservation"
                )));
            }
            self.refresh_budget_stats()?;

            let record_count = u64::try_from(edges.len())?;
            profiles.delta_records = profiles.delta_records.saturating_add(record_count);
            profiles.buffered_bytes = profiles
                .buffered_bytes
                .saturating_add(u64::try_from(reserved_bytes)?);
            {
                let mut stats = self.stats.lock();
                stats.delta_records = stats.delta_records.saturating_add(record_count);
                stats.cold_batch_commits = stats.cold_batch_commits.saturating_add(1);
                stats.cold_batch_records = stats.cold_batch_records.saturating_add(record_count);
            }
            self.after_accept(shard_id, range_id, profiles)?;
            return Ok(true);
        }
    }

    fn apply_reserved_cold_batch(
        &self,
        shard_id: usize,
        edges: &[PendingEdge],
    ) -> AnnResult<ReservedApplyOutcome> {
        let first = edges.first().ok_or_else(|| {
            AnnError::log_index_error("cannot apply an empty cold delta batch".to_string())
        })?;
        let range_id = self.range_id(first.p);
        if edges.iter().any(|edge| self.range_id(edge.p) != range_id) {
            return Err(AnnError::log_index_error(
                "cold delta batch spans multiple ranges".to_string(),
            ));
        }
        let mut range = self.ranges[range_id].mutable.lock();
        let cache = self.cache.shards[shard_id].lock();
        let mut previous_source = None;
        for edge in edges {
            let source = u32::try_from(edge.p)?;
            if previous_source != Some(source) {
                if !range.log_only && cache_route_locked(&cache, source) != CacheRoute::Miss {
                    return Ok(ReservedApplyOutcome::Retry);
                }
                previous_source = Some(source);
            }
        }
        drop(cache);

        let record_count = u64::try_from(edges.len())?;
        let next_ticket = range.next_ticket.checked_add(record_count).ok_or_else(|| {
            AnnError::log_index_error(format!("range {range_id} ticket counter overflows u64"))
        })?;
        let mut records = Vec::with_capacity(edges.len());
        let mut latest_tickets = Vec::new();
        for (offset, edge) in edges.iter().enumerate() {
            let ticket = range
                .next_ticket
                .checked_add(u64::try_from(offset)?)
                .ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "range {range_id} ticket counter overflows u64"
                    ))
                })?;
            records.push(BufferedDelta {
                range_id,
                record: ReservoirDeltaRecord::from_edge(ticket, edge)?,
            });
            let source = u32::try_from(edge.p)?;
            if latest_tickets
                .last()
                .is_some_and(|(previous, _)| *previous == source)
            {
                latest_tickets.last_mut().unwrap().1 = ticket;
            } else {
                latest_tickets.push((source, ticket));
            }
        }
        let consumed_bytes = edges
            .len()
            .checked_mul(super::reservoir_delta_log::DELTA_RECORD_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error("cold delta batch bytes overflow usize".to_string())
            })?;
        let owner = self.owner_buffers.get(shard_id).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "log-assisted shard {shard_id} is outside owner count {}",
                self.owner_buffers.len()
            ))
        })?;
        let mut buffer = owner.lock();
        let next_buffer_bytes = buffer
            .logical_bytes
            .checked_add(consumed_bytes)
            .ok_or_else(|| {
                AnnError::log_index_error("owner delta buffer bytes overflow usize".to_string())
            })?;
        buffer.records.extend(records);
        buffer.logical_bytes = next_buffer_bytes;
        range.next_ticket = next_ticket;
        for (source, ticket) in latest_tickets {
            range.active_latest.insert(source, ticket);
        }
        Ok(ReservedApplyOutcome::Applied {
            consumed_bytes,
            range_id,
            cache_hit: false,
            insert_profile: HashPruneInsertProfile::default(),
            backfill_profile: HashPruneBackfillProfile::default(),
        })
    }

    fn cold_batch_routes_are_misses(
        &self,
        shard_id: usize,
        edges: &[PendingEdge],
    ) -> AnnResult<bool> {
        let shard = self.cache.shards.get(shard_id).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "cold delta batch references unknown shard {shard_id}"
            ))
        })?;
        let range_id = self.range_id(edges[0].p);
        if self.ranges[range_id].mutable.lock().log_only {
            return Ok(true);
        }
        let shard = shard.lock();
        let mut previous_source = None;
        for edge in edges {
            let source = u32::try_from(edge.p)?;
            if previous_source != Some(source) {
                if cache_route_locked(&shard, source) != CacheRoute::Miss {
                    return Ok(false);
                }
                previous_source = Some(source);
            }
        }
        Ok(true)
    }

    fn validate_edge(&self, shard_id: usize, edge: &PendingEdge) -> AnnResult<()> {
        if edge.p < self.config.source_start || edge.p >= self.config.source_end {
            return Err(AnnError::log_index_error(format!(
                "log-assisted source {} lies outside [{}, {})",
                edge.p, self.config.source_start, self.config.source_end
            )));
        }
        let source = u32::try_from(edge.p)?;
        let expected_shard = self.cache.shard_id(source)?;
        if expected_shard != shard_id {
            return Err(AnnError::log_index_error(format!(
                "source {} maps to log-assisted shard {expected_shard}, received on {shard_id}",
                edge.p
            )));
        }
        Ok(())
    }

    fn record_failure(&self, error: &impl std::fmt::Display) {
        self.first_error.record(error);
        self.accepting.store(false, AtomicOrdering::Release);
        self.budget.notify_all();
        self.compaction_worker_gate.notify_error();
        self.idle_signal.notify();
        #[cfg(test)]
        self.compaction_gate.release();
        #[cfg(test)]
        self.admission_gate.release();
    }

    fn apply_one_edge(
        &self,
        shard_id: usize,
        edge: &PendingEdge,
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<()> {
        let source = u32::try_from(edge.p)?;
        loop {
            self.first_error.check()?;
            let range_id = self.range_id(edge.p);
            let route = if self.ranges[range_id].mutable.lock().log_only {
                CacheRoute::Miss
            } else {
                self.cache.route(source)?
            };
            let reserved_bytes = match route {
                CacheRoute::Miss => super::reservoir_delta_log::DELTA_RECORD_BYTES,
                CacheRoute::HitReserved => 0,
                CacheRoute::HitNeedsReservation => self.cache.checkpoint_bytes,
            };
            if reserved_bytes != 0 && self.budget.would_block(reserved_bytes) {
                self.schedule_budget_reclaim(profiles)?;
            }
            let (blocked, wait) = if reserved_bytes == 0 {
                (false, Duration::ZERO)
            } else {
                self.budget
                    .reserve(reserved_bytes, &self.accepting, &self.first_error, || {
                        self.schedule_budget_reclaim(profiles)
                    })?
            };
            if blocked {
                let mut stats = self.stats.lock();
                stats.delta_backpressure_count += 1;
                stats.delta_backpressure_wall += wait;
            }
            self.refresh_budget_stats()?;

            let outcome = {
                let _owner_operation = self.owner_operations[shard_id].lock();
                #[cfg(test)]
                self.admission_gate.enter();
                self.first_error.check().and_then(|()| {
                    self.ensure_shard_accepting(shard_id)?;
                    self.apply_reserved_edge(shard_id, edge, route)
                })
            };
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    if reserved_bytes != 0 {
                        self.budget.release(reserved_bytes)?;
                        self.refresh_budget_stats()?;
                    }
                    return Err(error);
                }
            };
            let ReservedApplyOutcome::Applied {
                consumed_bytes,
                range_id,
                cache_hit,
                insert_profile,
                backfill_profile,
            } = outcome
            else {
                if reserved_bytes != 0 {
                    self.budget.release(reserved_bytes)?;
                    self.refresh_budget_stats()?;
                }
                continue;
            };
            if consumed_bytes > reserved_bytes {
                return Err(AnnError::log_index_error(format!(
                    "log-assisted apply consumed {consumed_bytes} bytes from a {reserved_bytes}-byte reservation"
                )));
            }
            if reserved_bytes != consumed_bytes {
                self.budget.release(reserved_bytes - consumed_bytes)?;
            }
            self.refresh_budget_stats()?;

            if cache_hit {
                profiles.cache_hits += 1;
                profiles.insert_profile.merge(insert_profile.clone());
                profiles.backfill_profile.merge(backfill_profile.clone());
                let mut stats = self.stats.lock();
                stats.cache_hits += 1;
            } else {
                profiles.delta_records += 1;
                profiles.buffered_bytes += super::reservoir_delta_log::DELTA_RECORD_BYTES as u64;
                self.stats.lock().delta_records += 1;
            }
            self.after_accept(shard_id, range_id, profiles)?;
            return Ok(());
        }
    }

    fn apply_reserved_edge(
        &self,
        shard_id: usize,
        edge: &PendingEdge,
        expected_route: CacheRoute,
    ) -> AnnResult<ReservedApplyOutcome> {
        let source = u32::try_from(edge.p)?;
        let range_id = self.range_id(edge.p);
        let mut range = self.ranges[range_id].mutable.lock();
        let mut cache = self.cache.shards[shard_id].lock();
        let actual_route = if range.log_only {
            CacheRoute::Miss
        } else {
            cache_route_locked(&cache, source)
        };
        let compatible = matches!(
            (expected_route, actual_route),
            (CacheRoute::Miss, CacheRoute::Miss)
                | (CacheRoute::HitReserved, CacheRoute::HitReserved)
                | (
                    CacheRoute::HitNeedsReservation,
                    CacheRoute::HitNeedsReservation | CacheRoute::HitReserved
                )
        );
        if !compatible {
            return Ok(ReservedApplyOutcome::Retry);
        }

        let ticket = range.next_ticket;
        let next_ticket = ticket.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error(format!("range {range_id} ticket counter overflows u64"))
        })?;
        if actual_route == CacheRoute::Miss {
            drop(cache);
            let record = ReservoirDeltaRecord::from_edge(ticket, edge)?;
            let owner = self.owner_buffers.get(shard_id).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "log-assisted shard {shard_id} is outside owner count {}",
                    self.owner_buffers.len()
                ))
            })?;
            let mut buffer = owner.lock();
            let next_buffer_bytes = buffer
                .logical_bytes
                .checked_add(super::reservoir_delta_log::DELTA_RECORD_BYTES)
                .ok_or_else(|| {
                    AnnError::log_index_error("owner delta buffer bytes overflow usize".to_string())
                })?;
            buffer.records.push(BufferedDelta { range_id, record });
            buffer.logical_bytes = next_buffer_bytes;
            range.next_ticket = next_ticket;
            range.active_latest.insert(source, ticket);
            return Ok(ReservedApplyOutcome::Applied {
                consumed_bytes: super::reservoir_delta_log::DELTA_RECORD_BYTES,
                range_id,
                cache_hit: false,
                insert_profile: HashPruneInsertProfile::default(),
                backfill_profile: HashPruneBackfillProfile::default(),
            });
        }

        let entry = cache.entries.get_mut(&source).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "cache route for source {source} changed while its shard was locked"
            ))
        })?;
        let next_generation = entry.state_generation.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "cache generation for source {source} overflows u32"
            ))
        })?;
        let mut insert_profile = HashPruneInsertProfile::default();
        let mut backfill_profile = HashPruneBackfillProfile::default();
        apply_edge(
            &mut entry.reservoir,
            edge.clone(),
            self.config.evidence_policy,
            self.config.backfill_policy,
            self.config.diagnostic_collector.as_deref(),
            &mut insert_profile,
            &mut backfill_profile,
        )?;
        let consumed_bytes = if actual_route == CacheRoute::HitNeedsReservation {
            range.active_bytes = range
                .active_bytes
                .checked_add(self.cache.checkpoint_bytes)
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "range active checkpoint bytes overflow usize".to_string(),
                    )
                })?;
            entry.dirty_reservation_bytes = self.cache.checkpoint_bytes;
            self.cache.checkpoint_bytes
        } else {
            0
        };
        entry.latest_ticket = ticket;
        entry.state_generation = next_generation;
        entry.referenced = true;
        range.next_ticket = next_ticket;
        range.active_latest.insert(source, ticket);
        Ok(ReservedApplyOutcome::Applied {
            consumed_bytes,
            range_id,
            cache_hit: true,
            insert_profile,
            backfill_profile,
        })
    }

    fn after_accept(
        &self,
        shard_id: usize,
        range_id: usize,
        profiles: &mut LogAssistedApplyProfiles,
    ) -> AnnResult<()> {
        let owner_full =
            self.owner_buffers[shard_id].lock().logical_bytes >= self.owner_flush_bytes;
        let pre_gpu = self.promotions_enabled.load(AtomicOrdering::Acquire);
        let range_triggered = {
            let range = self.ranges[range_id].mutable.lock();
            self.range_compaction_trigger(range_id, &range, pre_gpu)
                .is_some()
        };
        if owner_full || range_triggered || self.budget.is_high() {
            let _operation = self.operation.lock();
            let _owners = self.lock_all_owner_operations();
            profiles.buffer_flushes += u64::try_from(self.flush_all_owners()?)?;
            if self.budget.is_high() {
                self.schedule_largest_idle_range()?;
            }
            self.schedule_triggered_ranges()?;
        }
        Ok(())
    }

    fn schedule_budget_reclaim(&self, profiles: &mut LogAssistedApplyProfiles) -> AnnResult<()> {
        let _operation = self.operation.lock();
        let _owners = self.lock_all_owner_operations();
        profiles.buffer_flushes += u64::try_from(self.flush_all_owners()?)?;
        self.schedule_largest_idle_range()?;
        self.schedule_triggered_ranges()
    }

    fn range_compaction_trigger(
        &self,
        range_id: usize,
        range: &RangeMutable,
        pre_gpu: bool,
    ) -> Option<RangeCompactionTrigger> {
        if range.active_bytes >= self.config.range_trigger_bytes {
            return Some(RangeCompactionTrigger::BytePressure);
        }
        if !pre_gpu || range.pre_gpu_materialized || range.log_only {
            return None;
        }
        let run_count = range.active_delta_runs.len()
            + range.active_borrowed_delta_runs.len()
            + range.active_checkpoint_runs.len()
            + usize::from(!range.pending_checkpoint_records.is_empty());
        (run_count >= self.config.max_runs_per_range
            && range.active_bytes >= self.ranges[range_id].run_trigger_bytes)
            .then_some(RangeCompactionTrigger::RunPressure)
    }

    fn refresh_budget_stats(&self) -> AnnResult<()> {
        let mut stats = self.stats.lock();
        stats.current_uncompacted_bytes = u64::try_from(self.budget.used())?;
        stats.peak_uncompacted_bytes = stats
            .peak_uncompacted_bytes
            .max(stats.current_uncompacted_bytes);
        Ok(())
    }

    pub(crate) fn mark_gpu_complete(&self) -> AnnResult<()> {
        let result = self.mark_gpu_complete_inner();
        if let Err(error) = &result {
            self.record_failure(error);
        }
        result
    }

    fn mark_gpu_complete_inner(&self) -> AnnResult<()> {
        let now = Instant::now();
        {
            let mut completed = self.gpu_complete_at.lock();
            if completed.is_some() {
                return Ok(());
            }
            *completed = Some(now);
        }
        self.promotions_enabled
            .store(false, AtomicOrdering::Release);
        {
            let mut stats = self.stats.lock();
            stats.post_gpu_uncompacted_bytes = u64::try_from(self.budget.used())?;
            stats.post_gpu_peak_compactions_inflight = stats
                .post_gpu_peak_compactions_inflight
                .max(stats.compactions_inflight);
        }
        self.compaction_worker_gate
            .activate(self.config.drain_compaction_workers);
        Ok(())
    }

    pub(crate) fn finalize_ingress_shards(&self, shard_ids: &[usize]) -> AnnResult<usize> {
        let result = self.finalize_ingress_shards_inner(shard_ids);
        if let Err(error) = &result {
            self.record_failure(error);
        }
        result
    }

    fn finalize_ingress_shards_inner(&self, shard_ids: &[usize]) -> AnnResult<usize> {
        if shard_ids.is_empty() {
            return Ok(0);
        }
        let mut shard_ids = shard_ids.to_vec();
        shard_ids.sort_unstable();
        shard_ids.dedup();
        for &shard_id in &shard_ids {
            if shard_id >= self.owner_buffers.len() {
                return Err(AnnError::log_index_error(format!(
                    "cannot finalize unknown log-assisted shard {shard_id}"
                )));
            }
        }

        let _operation = self.operation.lock();
        let _owner_operations = shard_ids
            .iter()
            .map(|&shard_id| self.owner_operations[shard_id].lock())
            .collect::<Vec<_>>();
        self.first_error.check()?;
        for &shard_id in &shard_ids {
            if self.drained_shards[shard_id].load(AtomicOrdering::Acquire) {
                return Err(AnnError::log_index_error(format!(
                    "log-assisted shard {shard_id} was finalized more than once"
                )));
            }
        }
        for &shard_id in &shard_ids {
            self.drained_shards[shard_id].store(true, AtomicOrdering::Release);
        }

        let mut flushed = 0_u64;
        for &shard_id in &shard_ids {
            if !self.owner_buffers[shard_id].lock().records.is_empty() {
                self.flush_owner(shard_id)?;
                flushed = flushed.saturating_add(1);
            }
        }
        {
            let mut stats = self.stats.lock();
            stats.delta_buffer_flushes = stats.delta_buffer_flushes.saturating_add(flushed);
            stats.shard_finalizations = stats
                .shard_finalizations
                .saturating_add(u64::try_from(shard_ids.len())?);
        }

        let mut scheduled = 0_usize;
        for range_id in 0..self.ranges.len() {
            if self.range_ingress_drained(range_id) && self.schedule_range(range_id, true)? {
                scheduled += 1;
            }
        }
        let mut stats = self.stats.lock();
        stats.ingress_drained_compactions = stats
            .ingress_drained_compactions
            .saturating_add(u64::try_from(scheduled)?);
        Ok(scheduled)
    }

    pub(crate) fn finish(&self) -> AnnResult<()> {
        let result = self.finish_inner();
        if let Err(error) = &result {
            self.record_failure(error);
        }
        result
    }

    fn finish_inner(&self) -> AnnResult<()> {
        if self.finished.load(AtomicOrdering::Acquire) {
            return Ok(());
        }
        let close_result = (|| -> AnnResult<()> {
            let _operation = self.operation.lock();
            let _owners = self.lock_all_owner_operations();
            self.accepting.store(false, AtomicOrdering::Release);
            self.budget.notify_all();
            self.first_error.check()?;
            Ok(())
        })();
        let wait_result = close_result.and_then(|()| self.drain_background_to_zero());
        let stop_result = self.stop_workers();
        wait_result?;
        stop_result?;
        self.first_error.check()?;
        if self.budget.used() != 0
            || self.uncompacted_bytes_locked() != 0
            || self.cache.has_dirty_state()
        {
            return Err(AnnError::log_index_error(format!(
                "log-assisted finish left budget={} stats={} uncompacted bytes or dirty cache state",
                self.budget.used(),
                self.uncompacted_bytes_locked()
            )));
        }
        self.finished.store(true, AtomicOrdering::Release);
        Ok(())
    }

    pub(crate) fn for_each_source<F>(&self, mut callback: F) -> AnnResult<()>
    where
        F: FnMut(u32, HashPruneReservoir) -> AnnResult<()>,
    {
        if !self.finished.load(AtomicOrdering::Acquire) {
            return Err(AnnError::log_index_error(
                "log-assisted base cannot be emitted before finish".to_string(),
            ));
        }
        for range in self.ranges.iter() {
            for row in range.base.read_current()? {
                let row = row?;
                callback(row.source, row.reservoir)?;
            }
        }
        Ok(())
    }

    pub(crate) fn record_bytes(&self) -> usize {
        self.ranges[0].base.layout().record_bytes
    }

    pub(crate) fn cleanup(self) -> AnnResult<()> {
        if !self.finished.load(AtomicOrdering::Acquire) {
            return Err(AnnError::log_index_error(
                "cannot clean an unfinished log-assisted reservoir".to_string(),
            ));
        }
        match fs::remove_dir_all(&self.config.root) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    pub(crate) fn stats(&self) -> LogAssistedStats {
        self.stats.lock().clone()
    }

    pub(crate) fn uncompacted_bytes(&self) -> u64 {
        let _operation = self.operation.lock();
        self.uncompacted_bytes_locked()
    }

    fn uncompacted_bytes_locked(&self) -> u64 {
        self.stats.lock().current_uncompacted_bytes
    }

    fn range_id(&self, source: usize) -> usize {
        (source - self.config.source_start) / self.config.range_points
    }

    fn ensure_shard_accepting(&self, shard_id: usize) -> AnnResult<()> {
        if !self.accepting.load(AtomicOrdering::Acquire) {
            return Err(AnnError::log_index_error(
                "cannot append to a closed log-assisted reservoir".to_string(),
            ));
        }
        if self.drained_shards[shard_id].load(AtomicOrdering::Acquire) {
            return Err(AnnError::log_index_error(format!(
                "cannot append to finalized log-assisted shard {shard_id}"
            )));
        }
        Ok(())
    }

    fn range_ingress_drained(&self, range_id: usize) -> bool {
        let (source_start, source_end) = self.ranges[range_id].base.source_range();
        let first_shard =
            (source_start as usize - self.config.source_start) / self.config.shard_points;
        let last_shard =
            (source_end as usize - 1 - self.config.source_start) / self.config.shard_points;
        (first_shard..=last_shard)
            .all(|shard_id| self.drained_shards[shard_id].load(AtomicOrdering::Acquire))
    }

    fn lock_all_owner_operations(&self) -> Vec<parking_lot::MutexGuard<'_, ()>> {
        self.owner_operations.iter().map(Mutex::lock).collect()
    }

    fn flush_owner(&self, owner_id: usize) -> AnnResult<()> {
        let records = {
            let mut owner = self.owner_buffers[owner_id].lock();
            if owner.records.is_empty() {
                return Ok(());
            }
            owner.logical_bytes = 0;
            std::mem::take(&mut owner.records)
        };
        let mut groups = BTreeMap::<usize, Vec<ReservoirDeltaRecord>>::new();
        for buffered in records {
            groups
                .entry(buffered.range_id)
                .or_default()
                .push(buffered.record);
        }
        for (range_id, records) in groups {
            let (epoch, run_id) = {
                let mut range = self.ranges[range_id].mutable.lock();
                let run_id = range.next_run_id;
                range.next_run_id = run_id.checked_add(1).ok_or_else(|| {
                    AnnError::log_index_error(format!("range {range_id} run id overflows u64"))
                })?;
                (range.epoch, run_id)
            };
            let path = self.config.root.join("runs").join(format!(
                "range-{range_id:08}-epoch-{epoch:020}-run-{run_id:020}.delta"
            ));
            let desc = write_delta_run_with_id(&path, range_id, epoch, run_id, records)?;
            let active_run_count = {
                let mut range = self.ranges[range_id].mutable.lock();
                range.active_bytes = range
                    .active_bytes
                    .checked_add(usize::try_from(desc.byte_len)?)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "range active delta bytes overflow usize".to_string(),
                        )
                    })?;
                range.active_delta_runs.push(desc.clone());
                range.active_delta_runs.len() + range.active_borrowed_delta_runs.len()
            };
            let mut stats = self.stats.lock();
            stats.delta_runs += 1;
            stats.delta_bytes_written = stats
                .delta_bytes_written
                .checked_add(desc.byte_len)
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "delta bytes-written counter overflows u64".to_string(),
                    )
                })?;
            stats.peak_active_delta_runs_per_range = stats
                .peak_active_delta_runs_per_range
                .max(u64::try_from(active_run_count)?);
            if active_run_count == self.config.max_runs_per_range {
                stats.run_pressure_crossings = stats.run_pressure_crossings.saturating_add(1);
            }
        }
        Ok(())
    }

    fn flush_all_owners(&self) -> AnnResult<usize> {
        let mut flushed = 0;
        for owner_id in 0..self.owner_buffers.len() {
            if !self.owner_buffers[owner_id].lock().records.is_empty() {
                self.flush_owner(owner_id)?;
                flushed += 1;
            }
        }
        let mut stats = self.stats.lock();
        stats.delta_buffer_flushes = stats
            .delta_buffer_flushes
            .saturating_add(u64::try_from(flushed)?);
        Ok(flushed)
    }

    fn schedule_triggered_ranges(&self) -> AnnResult<()> {
        let available = self.available_ordinary_compaction_slots();
        if available == 0 {
            return Ok(());
        }
        let pre_gpu = self.promotions_enabled.load(AtomicOrdering::Acquire);
        let mut candidates = self
            .ranges
            .iter()
            .enumerate()
            .filter_map(|(range_id, range)| {
                let mutable = range.mutable.lock();
                if mutable.frozen.is_some() || mutable.compaction_state != CompactionState::Idle {
                    return None;
                }
                self.range_compaction_trigger(range_id, &mutable, pre_gpu)
                    .map(|trigger| (trigger, mutable.active_bytes, range_id))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| right.cmp(left));
        for (_trigger, _bytes, range_id) in candidates.into_iter().take(available) {
            self.schedule_range(range_id, false)?;
        }
        Ok(())
    }

    fn schedule_largest_idle_range(&self) -> AnnResult<()> {
        if self.available_ordinary_compaction_slots() == 0 {
            return Ok(());
        }
        let candidate = self
            .ranges
            .iter()
            .enumerate()
            .filter_map(|(range_id, range)| {
                let mutable = range.mutable.lock();
                (mutable.frozen.is_none() && mutable.active_bytes != 0)
                    .then_some((mutable.active_bytes, range_id))
            })
            .max();
        if let Some((_bytes, range_id)) = candidate {
            self.schedule_range(range_id, true)?;
        }
        Ok(())
    }

    fn available_ordinary_compaction_slots(&self) -> usize {
        let limit = if self.promotions_enabled.load(AtomicOrdering::Acquire) {
            self.config.compaction_workers
        } else {
            self.config.drain_compaction_workers
        };
        let frozen = self
            .ranges
            .iter()
            .filter(|range| range.mutable.lock().frozen.is_some())
            .count();
        limit.saturating_sub(frozen)
    }

    fn schedule_range(&self, range_id: usize, force: bool) -> AnnResult<bool> {
        let pre_gpu = self.promotions_enabled.load(AtomicOrdering::Acquire);
        let mut trigger = None;
        let scheduled = {
            let range_state = &self.ranges[range_id];
            let mut mutable = range_state.mutable.lock();
            if mutable.frozen.is_some() || mutable.active_bytes == 0 {
                return Ok(false);
            }
            if !force {
                let Some(observed) = self.range_compaction_trigger(range_id, &mutable, pre_gpu)
                else {
                    return Ok(false);
                };
                trigger = Some(observed);
            }
            if mutable.compaction_state != CompactionState::Idle {
                return Err(AnnError::log_index_error(format!(
                    "range {range_id} has no frozen epoch but compaction state is {:?}",
                    mutable.compaction_state
                )));
            }
            let epoch = mutable.epoch;
            let next_epoch = epoch.checked_add(1).ok_or_else(|| {
                AnnError::log_index_error(format!("range {range_id} epoch overflows u64"))
            })?;
            let cut_ticket = mutable.next_ticket.checked_sub(1).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "range {range_id} ticket counter has no valid cut"
                ))
            })?;
            let (source_start, source_end) = range_state.base.source_range();
            let mut checkpoint_records = std::mem::take(&mut mutable.pending_checkpoint_records);
            let (snapshot_records, cache_snapshots) =
                self.cache.snapshot_range(source_start, source_end)?;
            let snapshot_count = snapshot_records.len();
            checkpoint_records.extend(snapshot_records);
            if !checkpoint_records.is_empty() {
                let checkpoint_count = checkpoint_records.len();
                let run_id = mutable.next_run_id;
                mutable.next_run_id = run_id.checked_add(1).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "range {range_id} checkpoint run id overflows u64"
                    ))
                })?;
                let path = self.config.root.join("runs").join(format!(
                    "range-{range_id:08}-epoch-{epoch:020}-run-{run_id:020}.checkpoint"
                ));
                let desc = write_checkpoint_run_with_id(
                    &path,
                    range_id,
                    epoch,
                    run_id,
                    &range_state.base.layout(),
                    checkpoint_records,
                )?;
                let expected_bytes = checkpoint_count
                    .checked_mul(self.cache.checkpoint_bytes)
                    .ok_or_else(|| {
                        AnnError::log_index_error("cache snapshot bytes overflow usize".to_string())
                    })?;
                if usize::try_from(desc.byte_len)? != expected_bytes {
                    return Err(AnnError::log_index_error(format!(
                        "checkpoint run bytes {} do not match reserved bytes {expected_bytes}",
                        desc.byte_len
                    )));
                }
                mutable.active_checkpoint_runs.push(desc.clone());
                let mut stats = self.stats.lock();
                stats.checkpoint_records += u64::try_from(checkpoint_count)?;
                stats.checkpoint_runs += 1;
                stats.checkpoint_bytes_written = stats
                    .checkpoint_bytes_written
                    .checked_add(desc.byte_len)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "checkpoint byte counter overflows u64".to_string(),
                        )
                    })?;
                stats.cache_dirty_checkpoints += u64::try_from(snapshot_count)?;
            }
            mutable.active_latest.clear();
            let frozen = FrozenRangeEpoch {
                range_id,
                epoch,
                cut_ticket,
                delta_runs: std::mem::take(&mut mutable.active_delta_runs),
                borrowed_delta_runs: std::mem::take(&mut mutable.active_borrowed_delta_runs),
                checkpoint_runs: std::mem::take(&mut mutable.active_checkpoint_runs),
                cache_snapshots,
                logical_bytes: std::mem::take(&mut mutable.active_bytes),
                log_only: mutable.log_only,
            };
            mutable.epoch = next_epoch;
            if pre_gpu {
                mutable.pre_gpu_materialized = true;
            }
            mutable.frozen = Some(frozen);
            mutable.compaction_state = CompactionState::Scheduled;
            true
        };
        if scheduled {
            let mut stats = self.stats.lock();
            if trigger == Some(RangeCompactionTrigger::RunPressure) {
                stats.run_pressure_compactions = stats.run_pressure_compactions.saturating_add(1);
            }
            if pre_gpu {
                stats.pre_gpu_materializations = stats.pre_gpu_materializations.saturating_add(1);
            }
            drop(stats);
            if let Err(error) = self.compaction_tx.send(CompactionJob::Range(range_id)) {
                self.ranges[range_id].mutable.lock().compaction_state = CompactionState::Failed;
                self.first_error.record(error);
                self.budget.notify_all();
                self.idle_signal.notify();
                return self.first_error.check().map(|()| true);
            }
            self.idle_signal.notify();
        }
        Ok(scheduled)
    }

    pub(crate) fn wait_for_background_idle(&self) -> AnnResult<()> {
        let mut observed = *self.idle_signal.generation.lock();
        loop {
            self.first_error.check()?;
            let idle = self.ranges.iter().all(|range| {
                let mutable = range.mutable.lock();
                mutable.frozen.is_none() && mutable.compaction_state == CompactionState::Idle
            });
            if idle {
                return Ok(());
            }
            self.idle_signal.wait_for_change(&mut observed);
        }
    }

    fn drain_background_to_zero(&self) -> AnnResult<()> {
        loop {
            {
                let _operation = self.operation.lock();
                let _owners = self.lock_all_owner_operations();
                self.first_error.check()?;
                self.flush_all_owners()?;
                for range_id in 0..self.ranges.len() {
                    self.schedule_range(range_id, true)?;
                }
            }
            self.wait_for_background_idle()?;
            let drained = {
                let _operation = self.operation.lock();
                let _owners = self.lock_all_owner_operations();
                let buffers_empty = self
                    .owner_buffers
                    .iter()
                    .all(|buffer| buffer.lock().records.is_empty());
                let ranges_empty = self.ranges.iter().all(|range| {
                    let mutable = range.mutable.lock();
                    mutable.active_delta_runs.is_empty()
                        && mutable.active_borrowed_delta_runs.is_empty()
                        && mutable.active_checkpoint_runs.is_empty()
                        && mutable.pending_checkpoint_records.is_empty()
                        && mutable.active_bytes == 0
                        && mutable.frozen.is_none()
                        && mutable.compaction_state == CompactionState::Idle
                });
                buffers_empty && ranges_empty && !self.cache.has_dirty_state()
            };
            if drained {
                return Ok(());
            }
        }
    }

    fn stop_workers(&self) -> AnnResult<()> {
        let mut handles_guard = self.worker_handles.lock();
        let Some(handles) = handles_guard.take() else {
            return Ok(());
        };
        self.compaction_worker_gate.shutdown();
        #[cfg(test)]
        self.compaction_gate.release();
        for _ in 0..handles.len() {
            let _ = self.compaction_tx.try_send(CompactionJob::Shutdown);
        }
        for handle in handles {
            handle.join().map_err(|_| {
                AnnError::log_index_error("log-assisted compaction worker panicked".to_string())
            })?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn delta_tickets(&self, source: u32) -> AnnResult<Vec<u64>> {
        let _operation = self.operation.lock();
        let _owners = self.lock_all_owner_operations();
        let mut tickets = Vec::new();
        for owner in &self.owner_buffers {
            tickets.extend(
                owner
                    .lock()
                    .records
                    .iter()
                    .filter(|buffered| buffered.record.source == source)
                    .map(|buffered| buffered.record.ticket),
            );
        }
        let range_id = self.range_id(source as usize);
        for run in &self.ranges[range_id].mutable.lock().active_delta_runs {
            for record in DeltaRunReader::open(run)? {
                let record = record?;
                if record.source == source {
                    tickets.push(record.ticket);
                }
            }
        }
        for run in &self.ranges[range_id]
            .mutable
            .lock()
            .active_borrowed_delta_runs
        {
            for record in BorrowedDeltaRunReader::open(run)? {
                let record = record?;
                if record.source == source {
                    tickets.push(record.ticket);
                }
            }
        }
        tickets.sort_unstable();
        Ok(tickets)
    }

    #[cfg(test)]
    fn pause_compaction_for_test(&self) {
        self.compaction_gate.pause();
    }

    #[cfg(test)]
    fn release_compaction_for_test(&self) {
        self.compaction_gate.release();
    }

    #[cfg(test)]
    fn pause_admission_for_test(&self) {
        self.admission_gate.pause();
    }

    #[cfg(test)]
    fn release_admission_for_test(&self) {
        self.admission_gate.release();
    }

    #[cfg(test)]
    fn wait_for_admissions_for_test(&self, expected: usize, timeout: Duration) -> bool {
        self.admission_gate.wait_for_entered(expected, timeout)
    }

    #[cfg(test)]
    fn running_compactions_for_test(&self) -> usize {
        self.ranges
            .iter()
            .filter(|range| range.mutable.lock().compaction_state == CompactionState::Running)
            .count()
    }

    #[cfg(test)]
    fn wait_for_running_compactions_for_test(
        &self,
        expected: usize,
        timeout: Duration,
    ) -> AnnResult<()> {
        let deadline = Instant::now() + timeout;
        loop {
            self.first_error.check()?;
            let running = self.running_compactions_for_test();
            if running >= expected {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(AnnError::log_index_error(format!(
                    "timed out waiting for {expected} running compactions; observed {running}"
                )));
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[cfg(test)]
    fn schedule_range_for_test(&self, range_id: usize) -> AnnResult<()> {
        let _operation = self.operation.lock();
        let _owners = self.lock_all_owner_operations();
        self.flush_all_owners()?;
        self.schedule_range(range_id, true)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn compact_range_for_test(&self, range_id: usize) -> AnnResult<()> {
        self.schedule_range_for_test(range_id)?;
        self.wait_for_background_idle()
    }

    #[cfg(test)]
    pub(crate) fn cache_contains_for_test(&self, source: u32) -> bool {
        self.cache.contains(source).unwrap()
    }

    #[cfg(test)]
    fn checkpoint_count_for_test(&self, source: u32) -> usize {
        let range_id = self.range_id(source as usize);
        let (pending, runs) = {
            let range = self.ranges[range_id].mutable.lock();
            let mut runs = range.active_checkpoint_runs.clone();
            if let Some(frozen) = &range.frozen {
                runs.extend(frozen.checkpoint_runs.iter().cloned());
            }
            let pending = range
                .pending_checkpoint_records
                .iter()
                .filter(|record| record.source == source)
                .count();
            (pending, runs)
        };
        pending
            + runs
                .iter()
                .map(|run| {
                    CheckpointRunReader::open(run, &self.ranges[range_id].base.layout())
                        .unwrap()
                        .filter(|record| {
                            record.as_ref().is_ok_and(|record| record.source == source)
                        })
                        .count()
                })
                .sum::<usize>()
    }

    #[cfg(test)]
    fn wait_for_frozen_epoch_for_test(&self, range_id: usize) -> AnnResult<()> {
        let mut observed = *self.idle_signal.generation.lock();
        loop {
            self.first_error.check()?;
            if self.ranges[range_id].mutable.lock().frozen.is_some() {
                return Ok(());
            }
            self.idle_signal.wait_for_change(&mut observed);
        }
    }

    #[cfg(test)]
    fn frozen_targets_for_test(&self, range_id: usize) -> AnnResult<Vec<u32>> {
        let (runs, borrowed_runs) = self.ranges[range_id]
            .mutable
            .lock()
            .frozen
            .as_ref()
            .map(|frozen| {
                (
                    frozen.delta_runs.clone(),
                    frozen.borrowed_delta_runs.clone(),
                )
            })
            .unwrap_or_default();
        read_targets(&runs, &borrowed_runs)
    }

    #[cfg(test)]
    fn active_targets_for_test(&self, range_id: usize) -> AnnResult<Vec<u32>> {
        let (runs, borrowed_runs) = {
            let range = self.ranges[range_id].mutable.lock();
            (
                range.active_delta_runs.clone(),
                range.active_borrowed_delta_runs.clone(),
            )
        };
        let mut targets = read_targets(&runs, &borrowed_runs)?;
        for owner in &self.owner_buffers {
            targets.extend(
                owner
                    .lock()
                    .records
                    .iter()
                    .filter(|buffered| buffered.range_id == range_id)
                    .map(|buffered| buffered.record.target),
            );
        }
        targets.sort_unstable();
        Ok(targets)
    }
}

#[cfg(test)]
fn read_targets(
    runs: &[DeltaRunDesc],
    borrowed_runs: &[BorrowedDeltaRunDesc],
) -> AnnResult<Vec<u32>> {
    let mut targets = Vec::new();
    for run in runs {
        for record in DeltaRunReader::open(run)? {
            targets.push(record?.target);
        }
    }
    for run in borrowed_runs {
        for record in BorrowedDeltaRunReader::open(run)? {
            targets.push(record?.target);
        }
    }
    targets.sort_unstable();
    Ok(targets)
}

impl Drop for LogAssistedReservoir {
    fn drop(&mut self) {
        self.accepting.store(false, AtomicOrdering::Release);
        self.budget.notify_all();
        self.compaction_worker_gate.shutdown();
        #[cfg(test)]
        self.compaction_gate.release();
        #[cfg(test)]
        self.admission_gate.release();
        let _ = self.stop_workers();
    }
}

fn compaction_worker_loop(
    worker_id: usize,
    context: CompactionWorkerContext,
    receiver: Receiver<CompactionJob>,
) {
    while context
        .compaction_worker_gate
        .wait_until_enabled(worker_id, &context.first_error)
    {
        let Ok(job) = receiver.recv() else {
            break;
        };
        let CompactionJob::Range(range_id) = job else {
            break;
        };
        let _inflight = CompactionInFlightGuard::start(&context);
        if let Err(error) = compact_frozen_range(&context, range_id) {
            context.first_error.record(error);
            context.accepting.store(false, AtomicOrdering::Release);
            if let Some(range) = context.ranges.get(range_id) {
                range.mutable.lock().compaction_state = CompactionState::Failed;
            }
            context.budget.notify_all();
            context.compaction_worker_gate.notify_error();
            context.idle_signal.notify();
            break;
        }
    }
}

fn compact_frozen_range(context: &CompactionWorkerContext, range_id: usize) -> AnnResult<()> {
    let range = context.ranges.get(range_id).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "compaction job references unknown range {range_id}"
        ))
    })?;
    let frozen = {
        let mut mutable = range.mutable.lock();
        if mutable.compaction_state != CompactionState::Scheduled {
            return Err(AnnError::log_index_error(format!(
                "range {range_id} compaction started from state {:?}",
                mutable.compaction_state
            )));
        }
        let frozen = mutable.frozen.clone().ok_or_else(|| {
            AnnError::log_index_error(format!("range {range_id} scheduled without a frozen epoch"))
        })?;
        mutable.compaction_state = CompactionState::Running;
        frozen
    };
    context.idle_signal.notify();
    #[cfg(test)]
    context.compaction_gate.wait();
    context.first_error.check()?;

    let reads_base = range.base.manifest().base_path.is_some();
    let compaction_started = Instant::now();
    let promotion_limit = if frozen.log_only {
        0
    } else {
        context
            .config
            .promotion_limit_per_compaction
            .min(context.config.cache_capacity_entries)
    };
    let result = compact_range_with_promotion(
        &range.base,
        &frozen,
        context.config.evidence_policy,
        context.config.backfill_policy,
        promotion_limit,
        context.config.cache_hot_updates,
        context.config.diagnostic_collector.clone(),
    )?;
    let compaction_finished = Instant::now();
    let compaction_wall = compaction_finished.saturating_duration_since(compaction_started);
    let hidden_wall = match *context.gpu_complete_at.lock() {
        Some(gpu_complete) if gpu_complete <= compaction_started => Duration::ZERO,
        Some(gpu_complete) if gpu_complete < compaction_finished => {
            gpu_complete.saturating_duration_since(compaction_started)
        }
        _ => compaction_wall,
    };
    let post_gpu_wall = compaction_wall.saturating_sub(hidden_wall);
    for run in &frozen.delta_runs {
        remove_consumed_run(&run.path)?;
    }
    for run in &frozen.borrowed_delta_runs {
        run.registry.mark_consumed(&run.segment)?;
    }
    for run in &frozen.checkpoint_runs {
        remove_consumed_run(&run.path)?;
    }
    context
        .cache
        .reconcile_snapshots(&frozen.cache_snapshots, result.base_generation)?;
    context.budget.release(frozen.logical_bytes)?;
    for candidate in result.promotion_candidates.iter().cloned() {
        try_promote_candidate(
            context,
            candidate,
            result.base_generation,
            frozen.cut_ticket,
        )?;
    }
    {
        let mut mutable = range.mutable.lock();
        let current = mutable.frozen.as_ref().ok_or_else(|| {
            AnnError::log_index_error(format!(
                "range {range_id} lost its frozen epoch during compaction"
            ))
        })?;
        if mutable.compaction_state != CompactionState::Running
            || current.epoch != frozen.epoch
            || current.cut_ticket != frozen.cut_ticket
        {
            return Err(AnnError::log_index_error(format!(
                "range {range_id} frozen epoch changed during compaction"
            )));
        }
        mutable.frozen = None;
        mutable.compaction_state = CompactionState::Idle;
    }
    {
        let mut stats = context.stats.lock();
        stats.base_reads += u64::from(reads_base);
        stats.base_writes += 1;
        stats.compactions += 1;
        stats.manifest_switches += 1;
        stats.rows_compacted += result.rows_compacted;
        stats.rows_changed += result.rows_changed;
        stats.compaction_input_bytes = stats
            .compaction_input_bytes
            .saturating_add(result.input_bytes);
        stats.compaction_output_bytes = stats
            .compaction_output_bytes
            .saturating_add(result.output_bytes);
        stats.compaction_read_wall += result.read_wall;
        stats.compaction_merge_wall += result.merge_wall;
        stats.compaction_write_wall += result.write_wall;
        stats.compaction_hidden_wall += hidden_wall;
        stats.post_gpu_compaction_wall += post_gpu_wall;
        stats.current_uncompacted_bytes = u64::try_from(context.budget.used())?;
        stats.insert_profile.merge(result.insert_profile);
        stats.backfill_profile.merge(result.backfill_profile);
    }
    context.idle_signal.notify();
    Ok(())
}

fn try_promote_candidate(
    context: &CompactionWorkerContext,
    candidate: PromotionCandidate,
    base_generation: u64,
    cut_ticket: u64,
) -> AnnResult<()> {
    if !context.accepting.load(AtomicOrdering::Acquire)
        || !context.promotions_enabled.load(AtomicOrdering::Acquire)
    {
        return Ok(());
    }
    let candidate_range_id =
        (candidate.source as usize - context.config.source_start) / context.config.range_points;
    if !context.accepting.load(AtomicOrdering::Acquire)
        || !context.promotions_enabled.load(AtomicOrdering::Acquire)
    {
        return Ok(());
    }
    let range = context.ranges[candidate_range_id].mutable.lock();
    if range.log_only
        || range
            .active_latest
            .get(&candidate.source)
            .is_some_and(|ticket| *ticket > cut_ticket)
    {
        context.stats.lock().cache_promotion_skipped_post_cut += 1;
        return Ok(());
    }
    let shard_id = context.cache.shard_id(candidate.source)?;
    let mut shard = context.cache.shards[shard_id].lock();
    if shard.capacity == 0
        || shard
            .entries
            .get(&candidate.source)
            .is_some_and(|entry| !entry.evicting)
    {
        return Ok(());
    }
    if shard.entries.len() == shard.capacity {
        let Some(victim_source) = shard.select_clean_victim(candidate.source) else {
            return Ok(());
        };
        shard.entries.remove(&victim_source);
    }
    shard.insert_candidate(
        candidate,
        base_generation,
        context.config.l_max,
        context.config.backfill_policy,
    )?;
    context.stats.lock().cache_promotions += 1;
    Ok(())
}

fn remove_consumed_run(path: &std::path::Path) -> AnnResult<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn validate_log_assisted_config(config: &LogAssistedReservoirConfig) -> AnnResult<()> {
    if config.source_start >= config.source_end
        || config.source_end > u32::MAX as usize
        || config.shard_points == 0
        || config.range_points == 0
        || config.l_max == 0
        || config.delta_buffer_bytes < super::reservoir_delta_log::DELTA_RECORD_BYTES
        || config.delta_buffer_bytes > config.delta_budget_bytes
        || config.range_trigger_bytes == 0
        || config.range_trigger_bytes > config.delta_budget_bytes
        || config.max_runs_per_range == 0
        || config.compaction_workers == 0
        || config.drain_compaction_workers < config.compaction_workers
        || (config.cache_capacity_entries != 0 && config.cache_hot_updates == 0)
    {
        return Err(AnnError::log_index_config_error(
            "log_assisted_reservoir".to_string(),
            "invalid source range or bounded log-assisted configuration".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn compact_range(
    base: &RangeBaseStore,
    frozen: &FrozenRangeEpoch,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
) -> AnnResult<RangeCompactionResult> {
    compact_range_with_promotion(
        base,
        frozen,
        evidence_policy,
        backfill_policy,
        0,
        usize::MAX,
        None,
    )
}

fn compact_range_with_promotion(
    base: &RangeBaseStore,
    frozen: &FrozenRangeEpoch,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    promotion_limit: usize,
    promotion_min_updates: usize,
    diagnostic_collector: Option<Arc<HashPruneTraceCollector>>,
) -> AnnResult<RangeCompactionResult> {
    let setup_started = Instant::now();
    validate_frozen_epoch(base, frozen)?;
    let manifest = base.manifest();
    if frozen.cut_ticket < manifest.base_watermark {
        return Err(AnnError::log_index_error(format!(
            "range {} compaction cut {} precedes base watermark {}",
            frozen.range_id, frozen.cut_ticket, manifest.base_watermark
        )));
    }

    let old_base_bytes = base.current_base_bytes()?;
    let delta_reader = MergedDeltaReader::open(&frozen.delta_runs, &frozen.borrowed_delta_runs)?;
    let checkpoint_reader = MergedCheckpointReader::open(&frozen.checkpoint_runs, &base.layout())?;
    let mut rows = CompactionRows::new(
        base.read_current()?,
        delta_reader,
        checkpoint_reader,
        base.source_range(),
        base.layout().l_max,
        manifest.base_watermark,
        frozen.cut_ticket,
        evidence_policy,
        backfill_policy,
        promotion_limit,
        u64::try_from(promotion_min_updates)?,
        diagnostic_collector,
    )?;
    let setup_read_wall = setup_started.elapsed();
    let publish_started = Instant::now();
    let base_generation =
        base.publish_iter(manifest.base_generation, frozen.cut_ticket, &mut rows)?;
    let publish_wall = publish_started.elapsed();
    let read_wall = setup_read_wall.saturating_add(rows.read_wall);
    let merge_wall = rows.merge_wall;
    let write_wall = publish_wall
        .saturating_sub(rows.read_wall)
        .saturating_sub(rows.merge_wall);
    let output_bytes = base.current_base_bytes()?;
    let input_bytes = old_base_bytes
        .checked_add(u64::try_from(frozen.logical_bytes)?)
        .ok_or_else(|| {
            AnnError::log_index_error("range compaction input bytes overflow u64".to_string())
        })?;

    let mut promotion_candidates = rows
        .promotion_candidates
        .into_iter()
        .map(|item| item.candidate)
        .collect::<Vec<_>>();
    promotion_candidates.sort_unstable_by(|left, right| {
        right
            .update_count
            .cmp(&left.update_count)
            .then_with(|| left.source.cmp(&right.source))
    });
    let hot_sources = promotion_candidates
        .iter()
        .map(|candidate| (candidate.source, candidate.update_count))
        .collect();
    Ok(RangeCompactionResult {
        base_generation,
        cut_ticket: frozen.cut_ticket,
        rows_compacted: rows.rows_compacted,
        rows_changed: rows.rows_changed,
        delta_records_seen: rows.delta_records_seen,
        delta_records_applied: rows.delta_records_applied,
        delta_records_covered: rows.delta_records_covered,
        checkpoint_records_seen: rows.checkpoint_records_seen,
        logical_bytes: frozen.logical_bytes,
        input_bytes,
        output_bytes,
        read_wall,
        merge_wall,
        write_wall,
        hot_sources,
        promotion_candidates,
        insert_profile: rows.insert_profile,
        backfill_profile: rows.backfill_profile,
    })
}

fn validate_frozen_epoch(base: &RangeBaseStore, frozen: &FrozenRangeEpoch) -> AnnResult<()> {
    if frozen.range_id != base.range_id() {
        return Err(AnnError::log_index_error(format!(
            "frozen range {} does not match base range {}",
            frozen.range_id,
            base.range_id()
        )));
    }
    let mut delta_run_ids = BTreeSet::new();
    let mut checkpoint_run_ids = BTreeSet::new();
    let mut logical_bytes = 0_u64;
    for run in &frozen.delta_runs {
        if run.range_id != frozen.range_id || run.epoch != frozen.epoch {
            return Err(AnnError::log_index_error(
                "delta run belongs to a different range epoch".to_string(),
            ));
        }
        if !delta_run_ids.insert(run.run_id) {
            return Err(AnnError::log_index_error(format!(
                "duplicate delta run id {}",
                run.run_id
            )));
        }
        if run.max_ticket > frozen.cut_ticket {
            return Err(AnnError::log_index_error(format!(
                "delta run {} exceeds compaction cut {}",
                run.run_id, frozen.cut_ticket
            )));
        }
        logical_bytes = logical_bytes.checked_add(run.byte_len).ok_or_else(|| {
            AnnError::log_index_error("frozen delta bytes overflow u64".to_string())
        })?;
    }
    for run in &frozen.borrowed_delta_runs {
        if run.range_id != frozen.range_id || run.epoch != frozen.epoch {
            return Err(AnnError::log_index_error(
                "borrowed delta run belongs to a different range epoch".to_string(),
            ));
        }
        if !delta_run_ids.insert(run.run_id) {
            return Err(AnnError::log_index_error(format!(
                "duplicate delta run id {}",
                run.run_id
            )));
        }
        if run.max_ticket > frozen.cut_ticket {
            return Err(AnnError::log_index_error(format!(
                "borrowed delta run {} exceeds compaction cut {}",
                run.run_id, frozen.cut_ticket
            )));
        }
        logical_bytes = logical_bytes.checked_add(run.byte_len).ok_or_else(|| {
            AnnError::log_index_error("frozen borrowed delta bytes overflow u64".to_string())
        })?;
    }
    for run in &frozen.checkpoint_runs {
        if run.range_id != frozen.range_id || run.epoch != frozen.epoch {
            return Err(AnnError::log_index_error(
                "checkpoint run belongs to a different range epoch".to_string(),
            ));
        }
        if !checkpoint_run_ids.insert(run.run_id) {
            return Err(AnnError::log_index_error(format!(
                "duplicate checkpoint run id {}",
                run.run_id
            )));
        }
        if run.max_ticket > frozen.cut_ticket {
            return Err(AnnError::log_index_error(format!(
                "checkpoint run {} exceeds compaction cut {}",
                run.run_id, frozen.cut_ticket
            )));
        }
        logical_bytes = logical_bytes.checked_add(run.byte_len).ok_or_else(|| {
            AnnError::log_index_error("frozen checkpoint bytes overflow u64".to_string())
        })?;
    }
    if logical_bytes != u64::try_from(frozen.logical_bytes)? {
        return Err(AnnError::log_index_error(format!(
            "frozen logical byte mismatch: descriptor sum={logical_bytes}, epoch={}",
            frozen.logical_bytes
        )));
    }
    Ok(())
}

struct PromotionHeapItem {
    candidate: PromotionCandidate,
}

impl PartialEq for PromotionHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.candidate.update_count == other.candidate.update_count
            && self.candidate.source == other.candidate.source
    }
}

impl Eq for PromotionHeapItem {}

impl PartialOrd for PromotionHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PromotionHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .candidate
            .update_count
            .cmp(&self.candidate.update_count)
            .then_with(|| self.candidate.source.cmp(&other.candidate.source))
    }
}

struct CompactionRows {
    base_rows: RangeBaseReader,
    deltas: MergedDeltaReader,
    checkpoints: MergedCheckpointReader,
    next_delta: Option<ReservoirDeltaRecord>,
    next_checkpoint: Option<ReservoirCheckpointRecord>,
    source_start: u32,
    source_end: u32,
    l_max: usize,
    base_watermark: u64,
    cut_ticket: u64,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<Arc<HashPruneTraceCollector>>,
    failed: bool,
    rows_compacted: u64,
    rows_changed: u64,
    delta_records_seen: u64,
    delta_records_applied: u64,
    delta_records_covered: u64,
    checkpoint_records_seen: u64,
    promotion_limit: usize,
    promotion_min_updates: u64,
    promotion_candidates: BinaryHeap<PromotionHeapItem>,
    insert_profile: HashPruneInsertProfile,
    backfill_profile: HashPruneBackfillProfile,
    read_wall: Duration,
    merge_wall: Duration,
}

impl CompactionRows {
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_rows: RangeBaseReader,
        mut deltas: MergedDeltaReader,
        mut checkpoints: MergedCheckpointReader,
        source_range: (u32, u32),
        l_max: usize,
        base_watermark: u64,
        cut_ticket: u64,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        promotion_limit: usize,
        promotion_min_updates: u64,
        diagnostic_collector: Option<Arc<HashPruneTraceCollector>>,
    ) -> AnnResult<Self> {
        let next_delta = deltas.next_record()?;
        let next_checkpoint = checkpoints.next_record()?;
        Ok(Self {
            base_rows,
            deltas,
            checkpoints,
            next_delta,
            next_checkpoint,
            source_start: source_range.0,
            source_end: source_range.1,
            l_max,
            base_watermark,
            cut_ticket,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            failed: false,
            rows_compacted: 0,
            rows_changed: 0,
            delta_records_seen: 0,
            delta_records_applied: 0,
            delta_records_covered: 0,
            checkpoint_records_seen: 0,
            promotion_limit,
            promotion_min_updates,
            promotion_candidates: BinaryHeap::new(),
            insert_profile: HashPruneInsertProfile::default(),
            backfill_profile: HashPruneBackfillProfile::default(),
            read_wall: Duration::ZERO,
            merge_wall: Duration::ZERO,
        })
    }

    fn fail(&mut self, message: impl Into<String>) -> Option<AnnResult<RangeBaseRow>> {
        self.failed = true;
        Some(Err(AnnError::log_index_error(message.into())))
    }

    fn advance_delta(&mut self) -> AnnResult<()> {
        let started = Instant::now();
        self.next_delta = self.deltas.next_record()?;
        self.read_wall += started.elapsed();
        Ok(())
    }

    fn advance_checkpoint(&mut self) -> AnnResult<()> {
        let started = Instant::now();
        self.next_checkpoint = self.checkpoints.next_record()?;
        self.read_wall += started.elapsed();
        Ok(())
    }
}

impl Iterator for CompactionRows {
    type Item = AnnResult<RangeBaseRow>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        let row_started = Instant::now();
        let read_before = self.read_wall;
        let base_read_started = Instant::now();
        let base_row = self.base_rows.next();
        self.read_wall += base_read_started.elapsed();
        let mut row = match base_row {
            Some(Ok(row)) => row,
            Some(Err(error)) => {
                self.failed = true;
                return Some(Err(error));
            }
            None => {
                if let Some(record) = self.next_delta {
                    return self.fail(format!(
                        "delta source {} lies outside range [{}, {})",
                        record.source, self.source_start, self.source_end
                    ));
                }
                if let Some(record) = &self.next_checkpoint {
                    return self.fail(format!(
                        "checkpoint source {} lies outside range [{}, {})",
                        record.source, self.source_start, self.source_end
                    ));
                }
                return None;
            }
        };
        let source = row.source;
        if self.next_delta.is_some_and(|record| record.source < source) {
            return self.fail(format!("delta source precedes range row {source}"));
        }
        if self
            .next_checkpoint
            .as_ref()
            .is_some_and(|record| record.source < source)
        {
            return self.fail(format!("checkpoint source precedes range row {source}"));
        }

        let mut coverage = self.base_watermark;
        let mut source_updates = 0_u64;
        let mut latest_source_ticket = coverage;
        let mut latest_checkpoint = None;
        while self
            .next_checkpoint
            .as_ref()
            .is_some_and(|record| record.source == source)
        {
            let checkpoint = self.next_checkpoint.take().unwrap();
            if let Err(error) = self.advance_checkpoint() {
                self.failed = true;
                return Some(Err(error));
            }
            self.checkpoint_records_seen += 1;
            if checkpoint.covered_ticket > self.cut_ticket {
                return self.fail(format!(
                    "checkpoint ticket {} exceeds cut {}",
                    checkpoint.covered_ticket, self.cut_ticket
                ));
            }
            if checkpoint.covered_ticket > coverage {
                coverage = checkpoint.covered_ticket;
                latest_checkpoint = Some(checkpoint);
            }
        }
        let checkpoint_applied = latest_checkpoint.is_some();
        if let Some(checkpoint) = latest_checkpoint {
            row.reservoir = match checkpoint.into_reservoir(self.l_max, self.backfill_policy) {
                Ok(reservoir) => reservoir,
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            };
        }

        while self
            .next_delta
            .is_some_and(|record| record.source == source)
        {
            let delta = self.next_delta.take().unwrap();
            if let Err(error) = self.advance_delta() {
                self.failed = true;
                return Some(Err(error));
            }
            self.delta_records_seen += 1;
            if delta.ticket > self.cut_ticket {
                return self.fail(format!(
                    "delta ticket {} exceeds cut {}",
                    delta.ticket, self.cut_ticket
                ));
            }
            if delta.ticket <= coverage {
                self.delta_records_covered += 1;
                continue;
            }
            let edge = match delta.to_edge() {
                Ok(edge) => edge,
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            };
            if let Err(error) = apply_edge(
                &mut row.reservoir,
                edge,
                self.evidence_policy,
                self.backfill_policy,
                self.diagnostic_collector.as_deref(),
                &mut self.insert_profile,
                &mut self.backfill_profile,
            ) {
                self.failed = true;
                return Some(Err(error));
            }
            self.delta_records_applied += 1;
            source_updates += 1;
            latest_source_ticket = delta.ticket;
        }
        if self.promotion_limit != 0 && source_updates >= self.promotion_min_updates {
            self.promotion_candidates.push(PromotionHeapItem {
                candidate: PromotionCandidate {
                    source,
                    update_count: source_updates,
                    latest_ticket: latest_source_ticket,
                    state: row.reservoir.state_snapshot(),
                },
            });
            if self.promotion_candidates.len() > self.promotion_limit {
                self.promotion_candidates.pop();
            }
        }
        if checkpoint_applied || source_updates != 0 {
            self.rows_changed += 1;
        }
        self.rows_compacted += 1;
        let row_read_wall = self.read_wall.saturating_sub(read_before);
        self.merge_wall += row_started.elapsed().saturating_sub(row_read_wall);
        Some(Ok(row))
    }
}

struct DeltaHeapItem {
    record: ReservoirDeltaRecord,
    run_id: u64,
    reader_index: usize,
}

impl DeltaHeapItem {
    fn key(&self) -> (u32, u64, u64, usize) {
        (
            self.record.source,
            self.record.ticket,
            self.run_id,
            self.reader_index,
        )
    }
}

impl PartialEq for DeltaHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for DeltaHeapItem {}

impl PartialOrd for DeltaHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeltaHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key().cmp(&self.key())
    }
}

struct BorrowedDeltaRunReader {
    mmap: Mmap,
    _reader_guard: OnlineFallbackReaderGuard,
    offset: usize,
    end: usize,
    next_ticket: u64,
    remaining: u64,
    source_min: u32,
    source_max: u32,
    previous_source: Option<u32>,
    failed: bool,
}

impl BorrowedDeltaRunReader {
    fn open(desc: &BorrowedDeltaRunDesc) -> AnnResult<Self> {
        if desc.record_count == 0
            || desc.byte_len != desc.record_count * RUN_RECORD_BYTES as u64
            || desc.min_ticket > desc.max_ticket
            || desc.max_ticket - desc.min_ticket + 1 != desc.record_count
        {
            return Err(AnnError::log_index_error(
                "borrowed delta descriptor has inconsistent record or ticket bounds".to_string(),
            ));
        }
        let reader_guard = desc.registry.acquire_reader(desc.segment.part_id)?;
        let registered_path = reader_guard.path();
        if registered_path != desc.path {
            return Err(AnnError::log_index_error(
                "borrowed delta path changed after adoption".to_string(),
            ));
        }
        let segment_end = desc
            .segment
            .desc
            .byte_offset
            .checked_add(desc.byte_len)
            .ok_or_else(|| {
                AnnError::log_index_error("borrowed delta file interval overflows u64".to_string())
            })?;
        let file = File::open(reader_guard.path())?;
        if file.metadata()?.len() < segment_end {
            return Err(AnnError::log_index_error(
                "borrowed delta segment exceeds its fallback part".to_string(),
            ));
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(AnnError::log_index_error(
                "cannot determine mmap page size for borrowed delta".to_string(),
            ));
        }
        let page_size = u64::try_from(page_size)?;
        let map_offset = desc.segment.desc.byte_offset / page_size * page_size;
        let prefix = desc.segment.desc.byte_offset - map_offset;
        let map_len = prefix.checked_add(desc.byte_len).ok_or_else(|| {
            AnnError::log_index_error("borrowed delta mapping length overflows u64".to_string())
        })?;
        let mmap = unsafe {
            MmapOptions::new()
                .offset(map_offset)
                .len(usize::try_from(map_len)?)
                .map(&file)?
        };
        let _ = mmap.advise(Advice::Sequential);
        let offset = usize::try_from(prefix)?;
        let end = offset
            .checked_add(usize::try_from(desc.byte_len)?)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "borrowed delta mapped interval overflows usize".to_string(),
                )
            })?;
        Ok(Self {
            mmap,
            _reader_guard: reader_guard,
            offset,
            end,
            next_ticket: desc.min_ticket,
            remaining: desc.record_count,
            source_min: desc.segment.desc.src_min,
            source_max: desc.segment.desc.src_max,
            previous_source: None,
            failed: false,
        })
    }
}

impl Iterator for BorrowedDeltaRunReader {
    type Item = AnnResult<ReservoirDeltaRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.remaining == 0 {
            return None;
        }
        let end = match self.offset.checked_add(RUN_RECORD_BYTES) {
            Some(end) if end <= self.end => end,
            _ => {
                self.failed = true;
                return Some(Err(AnnError::log_index_error(
                    "borrowed delta segment ended before its descriptor".to_string(),
                )));
            }
        };
        let edge = match decode_spill_record16(&self.mmap[self.offset..end]) {
            Ok(edge) => edge,
            Err(error) => {
                self.failed = true;
                return Some(Err(error));
            }
        };
        self.offset = end;
        let source = match u32::try_from(edge.p) {
            Ok(source) => source,
            Err(error) => {
                self.failed = true;
                return Some(Err(error.into()));
            }
        };
        if source < self.source_min
            || source > self.source_max
            || self
                .previous_source
                .is_some_and(|previous| previous > source)
        {
            self.failed = true;
            return Some(Err(AnnError::log_index_error(
                "borrowed delta records violate their sorted source interval".to_string(),
            )));
        }
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.remaining -= 1;
        self.previous_source = Some(source);
        match ReservoirDeltaRecord::from_edge(ticket, &edge) {
            Ok(record) => Some(Ok(record)),
            Err(error) => {
                self.failed = true;
                Some(Err(error))
            }
        }
    }
}

enum DeltaRecordReader {
    Owned(DeltaRunReader),
    Borrowed(BorrowedDeltaRunReader),
}

impl DeltaRecordReader {
    fn next_record(&mut self) -> Option<AnnResult<ReservoirDeltaRecord>> {
        match self {
            Self::Owned(reader) => reader.next(),
            Self::Borrowed(reader) => reader.next(),
        }
    }
}

struct MergedDeltaReader {
    readers: Vec<(u64, DeltaRecordReader)>,
    heap: BinaryHeap<DeltaHeapItem>,
    previous_key: Option<(u32, u64)>,
}

impl MergedDeltaReader {
    fn open(runs: &[DeltaRunDesc], borrowed_runs: &[BorrowedDeltaRunDesc]) -> AnnResult<Self> {
        let mut readers = Vec::with_capacity(runs.len() + borrowed_runs.len());
        let mut heap = BinaryHeap::new();
        for run in runs {
            let mut reader = DeltaRecordReader::Owned(DeltaRunReader::open(run)?);
            if let Some(record) = reader.next_record().transpose()? {
                let reader_index = readers.len();
                heap.push(DeltaHeapItem {
                    record,
                    run_id: run.run_id,
                    reader_index,
                });
            }
            readers.push((run.run_id, reader));
        }
        for run in borrowed_runs {
            let mut reader = DeltaRecordReader::Borrowed(BorrowedDeltaRunReader::open(run)?);
            if let Some(record) = reader.next_record().transpose()? {
                let reader_index = readers.len();
                heap.push(DeltaHeapItem {
                    record,
                    run_id: run.run_id,
                    reader_index,
                });
            }
            readers.push((run.run_id, reader));
        }
        Ok(Self {
            readers,
            heap,
            previous_key: None,
        })
    }

    fn next_record(&mut self) -> AnnResult<Option<ReservoirDeltaRecord>> {
        let Some(item) = self.heap.pop() else {
            return Ok(None);
        };
        let key = (item.record.source, item.record.ticket);
        if self.previous_key.is_some_and(|previous| previous >= key) {
            return Err(AnnError::log_index_error(format!(
                "duplicate or out-of-order delta key: source={}, ticket={}",
                key.0, key.1
            )));
        }
        self.previous_key = Some(key);
        let (run_id, reader) = &mut self.readers[item.reader_index];
        if let Some(record) = reader.next_record().transpose()? {
            self.heap.push(DeltaHeapItem {
                record,
                run_id: *run_id,
                reader_index: item.reader_index,
            });
        }
        Ok(Some(item.record))
    }
}

struct CheckpointHeapItem {
    record: ReservoirCheckpointRecord,
    run_id: u64,
    reader_index: usize,
}

impl CheckpointHeapItem {
    fn key(&self) -> (u32, u64, u64, usize) {
        (
            self.record.source,
            self.record.covered_ticket,
            self.run_id,
            self.reader_index,
        )
    }
}

impl PartialEq for CheckpointHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for CheckpointHeapItem {}

impl PartialOrd for CheckpointHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CheckpointHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key().cmp(&self.key())
    }
}

struct MergedCheckpointReader {
    readers: Vec<(u64, CheckpointRunReader)>,
    heap: BinaryHeap<CheckpointHeapItem>,
    previous_key: Option<(u32, u64)>,
}

impl MergedCheckpointReader {
    fn open(
        runs: &[CheckpointRunDesc],
        layout: &super::disk_reservoir_store::DiskReservoirLayout,
    ) -> AnnResult<Self> {
        let mut readers = Vec::with_capacity(runs.len());
        let mut heap = BinaryHeap::new();
        for run in runs {
            let mut reader = CheckpointRunReader::open(run, layout)?;
            if let Some(record) = reader.next().transpose()? {
                let reader_index = readers.len();
                heap.push(CheckpointHeapItem {
                    record,
                    run_id: run.run_id,
                    reader_index,
                });
            }
            readers.push((run.run_id, reader));
        }
        Ok(Self {
            readers,
            heap,
            previous_key: None,
        })
    }

    fn next_record(&mut self) -> AnnResult<Option<ReservoirCheckpointRecord>> {
        let Some(item) = self.heap.pop() else {
            return Ok(None);
        };
        let key = (item.record.source, item.record.covered_ticket);
        if self.previous_key.is_some_and(|previous| previous >= key) {
            return Err(AnnError::log_index_error(format!(
                "duplicate or out-of-order checkpoint key: source={}, ticket={}",
                key.0, key.1
            )));
        }
        self.previous_key = Some(key);
        let (run_id, reader) = &mut self.readers[item.reader_index];
        if let Some(record) = reader.next().transpose()? {
            self.heap.push(CheckpointHeapItem {
                record,
                run_id: *run_id,
                reader_index: item.reader_index,
            });
        }
        Ok(Some(item.record))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::path::PathBuf;
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::TempDir;

    use super::{
        BorrowedDeltaRunDesc, FrozenRangeEpoch, LogAssistedApplyProfiles, LogAssistedReservoir,
        LogAssistedReservoirConfig, MergedCheckpointReader, MergedDeltaReader, RUN_RECORD_BYTES,
        compact_range,
    };
    use crate::algorithm::pipnn::hash_prune::{HashPruneReservoir, HashPruneReservoirState};
    use crate::algorithm::pipnn::hash_prune_backfill::HashPruneBackfillPolicy;
    use crate::algorithm::pipnn::hash_prune_evidence::{HashPruneEvidencePolicy, HashPruneMeta};
    use crate::algorithm::pipnn::leaf_build::PendingEdge;
    use crate::algorithm::pipnn::online_suffix_hashprune::OnlineFallbackWriter;
    use crate::algorithm::pipnn::range_reservoir_store::{
        RangeBaseRow, RangeBaseStore, RangeBaseStoreConfig,
    };
    use crate::algorithm::pipnn::reservoir_delta_log::{
        CheckpointRunDesc, DeltaRunDesc, ReservoirCheckpointRecord, ReservoirDeltaRecord,
        write_checkpoint_run_with_id, write_delta_run_with_id,
    };
    use crate::common::AnnResult;

    struct CompactionHarness {
        root: TempDir,
        store: RangeBaseStore,
    }

    impl CompactionHarness {
        fn new(source_start: u32, source_end: u32, l_max: usize) -> Self {
            let root = tempfile::tempdir().unwrap();
            let store = RangeBaseStore::create(RangeBaseStoreConfig {
                root: root.path().to_path_buf(),
                range_id: 0,
                source_start,
                source_end,
                l_max,
                policy_key: [3; 32],
                backfill_policy: HashPruneBackfillPolicy::off(),
            })
            .unwrap();
            Self { root, store }
        }

        fn compact(
            &self,
            delta_groups: Vec<Vec<ReservoirDeltaRecord>>,
            checkpoint_groups: Vec<Vec<ReservoirCheckpointRecord>>,
            cut_ticket: u64,
        ) -> AnnResult<Vec<RangeBaseRow>> {
            let mut delta_runs = Vec::<DeltaRunDesc>::new();
            for (run_id, records) in delta_groups.into_iter().enumerate() {
                if records.is_empty() {
                    continue;
                }
                delta_runs.push(write_delta_run_with_id(
                    &self.root.path().join(format!("delta-{run_id}.run")),
                    0,
                    0,
                    run_id as u64,
                    records,
                )?);
            }
            let mut checkpoint_runs = Vec::<CheckpointRunDesc>::new();
            for (run_id, records) in checkpoint_groups.into_iter().enumerate() {
                if records.is_empty() {
                    continue;
                }
                checkpoint_runs.push(write_checkpoint_run_with_id(
                    &self.root.path().join(format!("checkpoint-{run_id}.run")),
                    0,
                    0,
                    run_id as u64,
                    &self.store.layout(),
                    records,
                )?);
            }
            let logical_bytes = delta_runs
                .iter()
                .map(|run| run.byte_len as usize)
                .chain(checkpoint_runs.iter().map(|run| run.byte_len as usize))
                .sum();
            compact_range(
                &self.store,
                &FrozenRangeEpoch {
                    range_id: 0,
                    epoch: 0,
                    cut_ticket,
                    delta_runs,
                    borrowed_delta_runs: Vec::new(),
                    checkpoint_runs,
                    cache_snapshots: Vec::new(),
                    logical_bytes,
                    log_only: false,
                },
                HashPruneEvidencePolicy::off(),
                HashPruneBackfillPolicy::off(),
            )?;
            self.store.read_current()?.collect()
        }
    }

    fn edge_record(
        source: u32,
        target: u32,
        hash: u16,
        dist: f32,
        ticket: u64,
    ) -> ReservoirDeltaRecord {
        ReservoirDeltaRecord {
            source,
            target,
            hash,
            flags: 0,
            dist_bits: dist.to_bits(),
            ticket,
        }
    }

    fn checkpoint_with_target(
        source: u32,
        covered_ticket: u64,
        target: u32,
        l_max: usize,
    ) -> ReservoirCheckpointRecord {
        let mut reservoir = HashPruneReservoir::new(l_max);
        reservoir.insert(target, target as u16, target as f32);
        ReservoirCheckpointRecord::from_reservoir(source, covered_ticket, 1, &reservoir).unwrap()
    }

    #[test]
    fn log_assisted_many_checkpoint_runs_do_not_exhaust_file_descriptors() {
        let root = tempfile::tempdir().unwrap();
        let layout =
            crate::algorithm::pipnn::disk_reservoir_store::DiskReservoirLayout::new(1, 4096)
                .unwrap();
        let mut runs = Vec::new();
        for source in 0..1_050_u32 {
            runs.push(
                write_checkpoint_run_with_id(
                    &root.path().join(format!("checkpoint-{source}.run")),
                    0,
                    0,
                    u64::from(source),
                    &layout,
                    vec![checkpoint_with_target(
                        source,
                        u64::from(source) + 1,
                        source + 10_000,
                        1,
                    )],
                )
                .unwrap(),
            );
        }

        let before = fs::read_dir("/proc/self/fd").unwrap().count();
        let mut merged = MergedCheckpointReader::open(&runs, &layout).unwrap();
        let after_open = fs::read_dir("/proc/self/fd").unwrap().count();
        let mut observed = Vec::new();
        while let Some(record) = merged.next_record().unwrap() {
            observed.push(record.source);
        }

        assert_eq!(observed, (0..1_050_u32).collect::<Vec<_>>());
        assert!(
            after_open <= before + 64,
            "checkpoint merge retained {} file descriptors",
            after_open.saturating_sub(before)
        );
    }

    #[test]
    fn borrowed_spill_merge_does_not_retain_one_fd_per_segment() {
        let root = tempfile::tempdir().unwrap();
        let mut writer =
            OnlineFallbackWriter::for_test_range_local(root.path(), 4096, 0, 2_000).unwrap();
        let mut segments = Vec::new();
        for source in 0..1_050_usize {
            segments.push(
                writer
                    .append(
                        0,
                        source as u64,
                        &[PendingEdge {
                            p: source,
                            c: source as u32 + 10_000,
                            hash: source as u16,
                            dist: source as f32,
                            meta: HashPruneMeta::empty(),
                        }],
                    )
                    .unwrap()
                    .remove(0),
            );
        }
        writer.seal_all().unwrap();
        let registry = writer.registry();
        let path = writer.part_path(segments[0].part_id).unwrap();
        let runs = segments
            .iter()
            .enumerate()
            .map(|(index, segment)| BorrowedDeltaRunDesc {
                registry: registry.clone(),
                segment: segment.clone(),
                path: path.clone(),
                range_id: 0,
                epoch: 0,
                run_id: index as u64,
                record_count: 1,
                byte_len: RUN_RECORD_BYTES as u64,
                min_ticket: index as u64 + 1,
                max_ticket: index as u64 + 1,
            })
            .collect::<Vec<_>>();

        let before = fs::read_dir("/proc/self/fd").unwrap().count();
        let mut merged = MergedDeltaReader::open(&[], &runs).unwrap();
        let after_open = fs::read_dir("/proc/self/fd").unwrap().count();
        let mut observed = Vec::new();
        while let Some(record) = merged.next_record().unwrap() {
            observed.push(record.source);
        }

        assert_eq!(observed, (0..1_050_u32).collect::<Vec<_>>());
        assert!(
            after_open <= before + 64,
            "borrowed merge retained {} file descriptors",
            after_open.saturating_sub(before)
        );
        for segment in &segments {
            registry.mark_consumed(segment).unwrap();
        }
        assert!(
            path.exists(),
            "an active borrowed reader must delay backing-part reclamation"
        );
        drop(merged);
        assert!(!path.exists());
    }

    #[test]
    fn log_only_compaction_does_not_build_discarded_promotion_candidates() {
        let (root, backend) = cache_backend(1, 2);
        let mut writer =
            OnlineFallbackWriter::for_test_range_local(&root.path().join("borrowed"), 4096, 0, 4)
                .unwrap();
        let segment = writer
            .append(0, 0, &[backend_edge(0, 10), backend_edge(0, 11)])
            .unwrap()
            .remove(0);
        writer.seal_all().unwrap();
        assert!(
            backend
                .adopt_spill_segment(
                    0,
                    &writer.registry(),
                    &segment,
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap()
        );

        backend.compact_range_for_test(0).unwrap();

        assert_eq!(backend.stats().cache_promotions, 0);
        assert_eq!(backend.stats().cache_promotion_skipped_post_cut, 0);
        assert!(!backend.cache_contains_for_test(0));
        backend.finish().unwrap();
    }

    #[test]
    fn log_only_range_defers_run_count_materialization_until_byte_pressure_or_drain() {
        let (root, backend) = cache_backend(0, 2);
        backend.pause_compaction_for_test();
        let mut writer =
            OnlineFallbackWriter::for_test_range_local(&root.path().join("borrowed"), 4096, 0, 4)
                .unwrap();
        let base_bytes =
            usize::try_from(backend.ranges[0].base.expected_base_bytes().unwrap()).unwrap();
        let records_per_segment = 64usize;
        let segment_bytes = records_per_segment * RUN_RECORD_BYTES;
        let segment_count = backend
            .config
            .max_runs_per_range
            .max(base_bytes.div_ceil(segment_bytes));

        for segment_id in 0..segment_count {
            let edges = (0..records_per_segment)
                .map(|record_id| {
                    backend_edge(
                        0,
                        10_000 + (segment_id * records_per_segment + record_id) as u32,
                    )
                })
                .collect::<Vec<_>>();
            let segment = writer
                .append(0, segment_id as u64, &edges)
                .unwrap()
                .remove(0);
            assert!(
                backend
                    .adopt_spill_segment(
                        0,
                        &writer.registry(),
                        &segment,
                        &mut LogAssistedApplyProfiles::default(),
                    )
                    .unwrap()
            );
        }
        writer.seal_all().unwrap();

        let range = backend.ranges[0].mutable.lock();
        assert!(range.log_only);
        assert!(range.active_bytes < backend.config.range_trigger_bytes);
        assert!(
            range.frozen.is_none(),
            "run-count pressure should not rewrite a log-only range before the final drain"
        );
        drop(range);

        backend.release_compaction_for_test();
        backend.finish().unwrap();
        assert_eq!(backend.stats().base_writes, 1);
    }

    fn apply_in_memory(
        source_start: u32,
        source_end: u32,
        l_max: usize,
        deltas: &[ReservoirDeltaRecord],
    ) -> Vec<HashPruneReservoir> {
        let mut reservoirs = (source_start..source_end)
            .map(|_| HashPruneReservoir::new(l_max))
            .collect::<Vec<_>>();
        let mut ordered = deltas.to_vec();
        ordered.sort_unstable_by_key(|record| record.ticket);
        for record in ordered {
            reservoirs[(record.source - source_start) as usize].insert(
                record.target,
                record.hash,
                f32::from_bits(record.dist_bits),
            );
        }
        reservoirs
    }

    type StateSignature = (u16, Vec<(u32, u16, u32, u16)>);

    fn state_signature(state: HashPruneReservoirState) -> StateSignature {
        (
            state.backup_count,
            state
                .slots
                .into_iter()
                .map(|slot| {
                    (
                        slot.target,
                        slot.hash,
                        slot.dist.to_bits(),
                        slot.meta.to_state_flags().unwrap(),
                    )
                })
                .collect(),
        )
    }

    fn snapshot_rows(rows: Vec<RangeBaseRow>) -> Vec<StateSignature> {
        rows.into_iter()
            .map(|row| state_signature(row.reservoir.state_snapshot()))
            .collect()
    }

    fn snapshot_reservoirs(rows: Vec<HashPruneReservoir>) -> Vec<StateSignature> {
        rows.into_iter()
            .map(|reservoir| state_signature(reservoir.state_snapshot()))
            .collect()
    }

    #[test]
    fn compaction_matches_in_memory_ticket_order() {
        let harness = CompactionHarness::new(0, 4, 4);
        let deltas = vec![
            edge_record(0, 10, 1, 4.0, 1),
            edge_record(1, 20, 2, 3.0, 2),
            edge_record(0, 11, 3, 2.0, 3),
        ];
        let expected = apply_in_memory(0, 4, 4, &deltas);
        let actual = harness.compact(vec![deltas], Vec::new(), 3).unwrap();
        assert_eq!(snapshot_rows(actual), snapshot_reservoirs(expected));
    }

    #[test]
    fn latest_checkpoint_supersedes_covered_deltas() {
        let harness = CompactionHarness::new(0, 1, 4);
        let checkpoint = checkpoint_with_target(0, 2, 10, 4);
        let rows = harness
            .compact(
                vec![vec![
                    edge_record(0, 99, 9, 9.0, 1),
                    edge_record(0, 11, 1, 1.0, 3),
                ]],
                vec![vec![checkpoint]],
                3,
            )
            .unwrap();
        let mut targets = rows[0]
            .reservoir
            .slots()
            .iter()
            .map(|slot| slot.target)
            .collect::<Vec<_>>();
        targets.sort_unstable();
        assert_eq!(targets, vec![10, 11]);
    }

    #[test]
    fn compaction_rejects_delta_after_cut() {
        let harness = CompactionHarness::new(0, 1, 4);
        assert!(
            harness
                .compact(vec![vec![edge_record(0, 10, 1, 1.0, 8)]], Vec::new(), 7)
                .is_err()
        );
    }

    #[test]
    fn randomized_compaction_matches_ticket_order_across_runs_and_checkpoints() {
        let source_end = 13;
        let l_max = 8;
        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let deltas = (1..=240_u64)
            .map(|ticket| {
                let source = rng.random_range(0..source_end);
                let target = rng.random_range(100..10_000);
                let hash = rng.random_range(0..32);
                let dist = rng.random_range(0.0..100.0);
                edge_record(source, target, hash, dist, ticket)
            })
            .collect::<Vec<_>>();
        let expected = apply_in_memory(0, source_end, l_max, &deltas);

        let mut delta_groups = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for record in deltas.iter().copied() {
            delta_groups[(record.ticket as usize * 7) % 4].push(record);
        }

        let checkpoint_cut = 127;
        let mut checkpoint_groups = vec![Vec::new(), Vec::new()];
        for source in [1_u32, 4, 9] {
            let mut reservoir = HashPruneReservoir::new(l_max);
            for record in deltas
                .iter()
                .filter(|record| record.source == source && record.ticket <= checkpoint_cut)
            {
                reservoir.insert(record.target, record.hash, f32::from_bits(record.dist_bits));
            }
            checkpoint_groups[source as usize % 2].push(
                ReservoirCheckpointRecord::from_reservoir(source, checkpoint_cut, 1, &reservoir)
                    .unwrap(),
            );
        }

        let harness = CompactionHarness::new(0, source_end, l_max);
        let actual = harness
            .compact(delta_groups, checkpoint_groups, 240)
            .unwrap();
        assert_eq!(snapshot_rows(actual), snapshot_reservoirs(expected));
    }

    fn backend_edge(source: usize, target: u32) -> PendingEdge {
        PendingEdge {
            p: source,
            c: target,
            hash: (target % 17) as u16,
            dist: target as f32,
            meta: HashPruneMeta::empty(),
        }
    }

    fn test_backend(
        source_end: usize,
        l_max: usize,
        delta_buffer_bytes: usize,
    ) -> (TempDir, LogAssistedReservoir) {
        let root = tempfile::tempdir().unwrap();
        let backend = LogAssistedReservoir::create(LogAssistedReservoirConfig {
            root: root.path().to_path_buf(),
            source_start: 0,
            source_end,
            shard_points: source_end.max(1),
            range_points: 4,
            l_max,
            delta_budget_bytes: 1 << 20,
            delta_buffer_bytes,
            range_trigger_bytes: 4096,
            max_runs_per_range: 4,
            compaction_workers: 1,
            drain_compaction_workers: 1,
            cache_capacity_entries: 0,
            cache_hot_updates: 2,
            promotion_limit_per_compaction: 0,
            policy_key: [9; 32],
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        })
        .unwrap();
        (root, backend)
    }

    fn multi_owner_backend() -> (TempDir, Arc<LogAssistedReservoir>) {
        let root = tempfile::tempdir().unwrap();
        let backend = LogAssistedReservoir::create(LogAssistedReservoirConfig {
            root: root.path().to_path_buf(),
            source_start: 0,
            source_end: 8,
            shard_points: 4,
            range_points: 4,
            l_max: 4,
            delta_budget_bytes: 1 << 20,
            delta_buffer_bytes: 4096,
            range_trigger_bytes: 1 << 18,
            max_runs_per_range: 8,
            compaction_workers: 1,
            drain_compaction_workers: 1,
            cache_capacity_entries: 0,
            cache_hot_updates: 2,
            promotion_limit_per_compaction: 0,
            policy_key: [13; 32],
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        })
        .unwrap();
        (root, Arc::new(backend))
    }

    #[test]
    fn different_owners_overlap_fast_admission() {
        let (_root, backend) = multi_owner_backend();
        backend.pause_admission_for_test();

        let first = Arc::clone(&backend);
        let first_handle = thread::spawn(move || {
            first.apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
        });
        let second = Arc::clone(&backend);
        let second_handle = thread::spawn(move || {
            second.apply_edges(
                1,
                &[backend_edge(4, 40)],
                &mut LogAssistedApplyProfiles::default(),
            )
        });

        let overlapped = backend.wait_for_admissions_for_test(2, Duration::from_millis(100));
        backend.release_admission_for_test();
        first_handle.join().unwrap().unwrap();
        second_handle.join().unwrap().unwrap();
        backend.finish().unwrap();

        assert!(
            overlapped,
            "different reducer owners remained serialized at fast admission"
        );
    }

    #[test]
    fn epoch_cut_waits_for_inflight_owner_admission() {
        let (_root, backend) = multi_owner_backend();
        backend.pause_admission_for_test();
        backend.pause_compaction_for_test();

        let producer = Arc::clone(&backend);
        let producer_handle = thread::spawn(move || {
            producer.apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
        });
        assert!(backend.wait_for_admissions_for_test(1, Duration::from_secs(2)));

        let (cut_tx, cut_rx) = mpsc::channel();
        let scheduler = Arc::clone(&backend);
        let scheduler_handle =
            thread::spawn(move || cut_tx.send(scheduler.schedule_range_for_test(0)).unwrap());
        assert!(cut_rx.recv_timeout(Duration::from_millis(30)).is_err());

        backend.release_admission_for_test();
        producer_handle.join().unwrap().unwrap();
        cut_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        scheduler_handle.join().unwrap();
        assert_eq!(backend.frozen_targets_for_test(0).unwrap(), vec![10]);

        backend.release_compaction_for_test();
        backend.finish().unwrap();
        let mut source_zero_targets = Vec::new();
        backend
            .for_each_source(|source, reservoir| {
                if source == 0 {
                    source_zero_targets =
                        reservoir.slots().iter().map(|slot| slot.target).collect();
                    source_zero_targets.sort_unstable();
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(source_zero_targets, vec![10]);
    }

    #[test]
    fn shard_finalization_schedules_only_fully_drained_ranges() {
        let root = tempfile::tempdir().unwrap();
        let backend = LogAssistedReservoir::create(LogAssistedReservoirConfig {
            root: root.path().to_path_buf(),
            source_start: 0,
            source_end: 8,
            shard_points: 4,
            range_points: 3,
            l_max: 4,
            delta_budget_bytes: 1 << 20,
            delta_buffer_bytes: 4096,
            range_trigger_bytes: 1 << 18,
            max_runs_per_range: 8,
            compaction_workers: 2,
            drain_compaction_workers: 2,
            cache_capacity_entries: 0,
            cache_hot_updates: 2,
            promotion_limit_per_compaction: 0,
            policy_key: [17; 32],
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        })
        .unwrap();
        backend.pause_compaction_for_test();
        backend.mark_gpu_complete().unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10), backend_edge(3, 30)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend
            .apply_edges(
                1,
                &[backend_edge(4, 40), backend_edge(6, 60)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();

        assert_eq!(backend.finalize_ingress_shards(&[0]).unwrap(), 1);
        assert!(backend.ranges[0].mutable.lock().frozen.is_some());
        assert!(backend.ranges[1].mutable.lock().frozen.is_none());
        assert!(backend.ranges[2].mutable.lock().frozen.is_none());
        assert!(backend.ensure_shard_accepting(0).is_err());

        assert_eq!(backend.finalize_ingress_shards(&[1]).unwrap(), 2);
        assert!(backend.ranges[1].mutable.lock().frozen.is_some());
        assert!(backend.ranges[2].mutable.lock().frozen.is_some());
        assert_eq!(backend.stats().shard_finalizations, 2);
        assert_eq!(backend.stats().ingress_drained_compactions, 3);

        backend.release_compaction_for_test();
        backend.finish().unwrap();
    }

    #[test]
    fn cold_updates_append_delta_without_base_read() {
        let (_root, backend) = test_backend(8, 4, 4096);
        backend
            .apply_edges(
                0,
                &[
                    backend_edge(2, 20),
                    backend_edge(2, 21),
                    backend_edge(3, 30),
                    backend_edge(3, 31),
                ],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert_eq!(backend.stats().base_reads, 0);
        assert_eq!(backend.stats().delta_records, 4);
        assert_eq!(backend.stats().cold_batch_commits, 1);
        assert_eq!(backend.stats().cold_batch_records, 4);
        assert_eq!(backend.delta_tickets(2).unwrap(), vec![1, 2]);
        assert_eq!(backend.delta_tickets(3).unwrap(), vec![3, 4]);
    }

    #[test]
    fn cold_batches_obey_small_buffer_and_range_boundaries() {
        let (_root, backend) = test_backend(
            8,
            4,
            2 * super::super::reservoir_delta_log::DELTA_RECORD_BYTES,
        );
        backend
            .apply_edges(
                0,
                &[
                    backend_edge(0, 10),
                    backend_edge(0, 11),
                    backend_edge(1, 20),
                    backend_edge(4, 40),
                    backend_edge(5, 50),
                ],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert_eq!(backend.stats().cold_batch_commits, 3);
        assert_eq!(backend.stats().cold_batch_records, 5);
        assert_eq!(backend.delta_tickets(0).unwrap(), vec![1, 2]);
        assert_eq!(backend.delta_tickets(1).unwrap(), vec![3]);
        assert_eq!(backend.delta_tickets(4).unwrap(), vec![1]);
        assert_eq!(backend.delta_tickets(5).unwrap(), vec![2]);
        backend.finish().unwrap();
    }

    #[test]
    fn finish_compacts_every_overlay_and_matches_oracle() {
        let (_root, backend) = test_backend(8, 4, 96);
        let edges = vec![
            backend_edge(3, 30),
            backend_edge(0, 8),
            backend_edge(7, 70),
            backend_edge(3, 31),
            backend_edge(0, 9),
        ];
        backend
            .apply_edges(0, &edges, &mut LogAssistedApplyProfiles::default())
            .unwrap();
        backend.finish().unwrap();
        assert_eq!(backend.uncompacted_bytes(), 0);
        assert_eq!(backend.stats().rows_changed, 3);

        let mut actual = Vec::new();
        backend
            .for_each_source(|_source, reservoir| {
                actual.push(state_signature(reservoir.state_snapshot()));
                Ok(())
            })
            .unwrap();
        let expected_records = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                edge_record(
                    edge.p as u32,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    index as u64 + 1,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            snapshot_reservoirs(apply_in_memory(0, 8, 4, &expected_records))
        );
    }

    #[test]
    fn finish_does_not_promote_compacted_sources_into_cache() {
        let (_root, backend) = cache_backend(1, 2);
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10), backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();

        backend.finish().unwrap();

        assert_eq!(backend.stats().cache_promotions, 0);
        assert!(!backend.cache_contains_for_test(0));
    }

    #[test]
    fn gpu_complete_disables_promotion_but_accepts_queued_updates() {
        let (_root, backend) = cache_backend(1, 2);
        backend.mark_gpu_complete().unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10), backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.compact_range_for_test(0).unwrap();

        assert_eq!(backend.stats().cache_promotions, 0);
        assert!(!backend.cache_contains_for_test(0));
        backend.finish().unwrap();
    }

    #[test]
    fn gpu_complete_activates_parked_compaction_workers() {
        let root = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 12,
                shard_points: 12,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: 1 << 20,
                delta_buffer_bytes: 4096,
                range_trigger_bytes: 1 << 18,
                max_runs_per_range: 8,
                compaction_workers: 1,
                drain_compaction_workers: 3,
                cache_capacity_entries: 0,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 0,
                policy_key: [12; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        backend.pause_compaction_for_test();
        backend
            .apply_edges(
                0,
                &[
                    backend_edge(0, 10),
                    backend_edge(4, 40),
                    backend_edge(8, 80),
                ],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        for range_id in 0..3 {
            backend.schedule_range_for_test(range_id).unwrap();
        }

        backend
            .wait_for_running_compactions_for_test(1, Duration::from_secs(2))
            .unwrap();
        assert_eq!(backend.running_compactions_for_test(), 1);

        backend.mark_gpu_complete().unwrap();
        backend
            .wait_for_running_compactions_for_test(3, Duration::from_secs(2))
            .unwrap();
        assert_eq!(backend.running_compactions_for_test(), 3);

        backend.release_compaction_for_test();
        backend.finish().unwrap();
        let stats = backend.stats();
        assert_eq!(stats.compactions_inflight, 0);
        assert_eq!(stats.peak_compactions_inflight, 3);
        assert_eq!(stats.post_gpu_peak_compactions_inflight, 3);
    }

    #[test]
    fn tickets_remain_source_ordered_across_batch_boundaries() {
        let (_root, backend) = test_backend(4, 4, 4096);
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert_eq!(backend.delta_tickets(0).unwrap(), vec![1, 2]);
    }

    fn background_backend(
        delta_budget_bytes: usize,
        range_trigger_bytes: usize,
    ) -> (TempDir, Arc<LogAssistedReservoir>) {
        let root = tempfile::tempdir().unwrap();
        let backend = LogAssistedReservoir::create(LogAssistedReservoirConfig {
            root: root.path().to_path_buf(),
            source_start: 0,
            source_end: 8,
            shard_points: 8,
            range_points: 4,
            l_max: 4,
            delta_budget_bytes,
            delta_buffer_bytes: 24,
            range_trigger_bytes,
            max_runs_per_range: 8,
            compaction_workers: 1,
            drain_compaction_workers: 1,
            cache_capacity_entries: 0,
            cache_hot_updates: 2,
            promotion_limit_per_compaction: 0,
            policy_key: [10; 32],
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        })
        .unwrap();
        (root, Arc::new(backend))
    }

    fn cache_backend(
        cache_capacity_entries: usize,
        cache_hot_updates: usize,
    ) -> (TempDir, Arc<LogAssistedReservoir>) {
        let root = tempfile::tempdir().unwrap();
        let backend = LogAssistedReservoir::create(LogAssistedReservoirConfig {
            root: root.path().to_path_buf(),
            source_start: 0,
            source_end: 8,
            shard_points: 8,
            range_points: 4,
            l_max: 4,
            delta_budget_bytes: 1 << 20,
            delta_buffer_bytes: 24,
            range_trigger_bytes: 1 << 18,
            max_runs_per_range: 8,
            compaction_workers: 1,
            drain_compaction_workers: 1,
            cache_capacity_entries,
            cache_hot_updates,
            promotion_limit_per_compaction: cache_capacity_entries,
            policy_key: [11; 32],
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        })
        .unwrap();
        (root, Arc::new(backend))
    }

    fn next_run_temporary_path(
        backend: &LogAssistedReservoir,
        range_id: usize,
        kind: &str,
    ) -> PathBuf {
        let range = backend.ranges[range_id].mutable.lock();
        backend.config.root.join("runs").join(format!(
            "range-{range_id:08}-epoch-{:020}-run-{:020}.{kind}.tmp",
            range.epoch, range.next_run_id
        ))
    }

    fn base_temporary_path(root: &TempDir, range_id: usize, generation: u64) -> PathBuf {
        root.path().join("base").join(format!(
            "range-{range_id:08}-base-g{generation:020}.bin.tmp"
        ))
    }

    fn manifest_temporary_path(root: &TempDir, range_id: usize) -> PathBuf {
        root.path()
            .join("base")
            .join(format!("range-{range_id:08}.manifest.tmp"))
    }

    fn finish_with_timeout(backend: Arc<LogAssistedReservoir>) -> AnnResult<()> {
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || tx.send(backend.finish()).unwrap());
        let result = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("log-assisted finish did not complete after a fatal error");
        handle.join().unwrap();
        result
    }

    #[test]
    fn log_assisted_fault_first_delta_write_stops_admission_and_preserves_artifact() {
        let (root, backend) = background_backend(4096, 4096);
        let blocker = next_run_temporary_path(&backend, 0, "delta");
        fs::create_dir(&blocker).unwrap();

        assert!(
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 10)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .is_err()
        );
        assert!(
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 11)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .is_err()
        );
        assert!(backend.finish().is_err());
        assert!(blocker.is_dir());
        assert!(root.path().join("runs").is_dir());
    }

    #[test]
    fn log_assisted_fault_checkpoint_write_stops_admission() {
        let (_root, backend) = cache_backend(1, 2);
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10), backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.compact_range_for_test(0).unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 12)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        let blocker = next_run_temporary_path(&backend, 0, "checkpoint");
        fs::create_dir(&blocker).unwrap();

        backend.mark_gpu_complete().unwrap();
        assert!(backend.finish().is_err());
        assert!(blocker.is_dir());
    }

    #[test]
    fn log_assisted_fault_base_write_does_not_publish_manifest() {
        let (root, backend) = background_backend(4096, 4096);
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        let blocker = base_temporary_path(&root, 0, 1);
        fs::create_dir(&blocker).unwrap();

        backend.mark_gpu_complete().unwrap();
        assert!(finish_with_timeout(Arc::clone(&backend)).is_err());
        assert!(!root.path().join("base/range-00000000.manifest").exists());
        assert!(blocker.is_dir());
    }

    #[test]
    fn log_assisted_fault_manifest_publish_keeps_previous_generation() {
        let (root, backend) = background_backend(4096, 4096);
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        let blocker = manifest_temporary_path(&root, 0);
        fs::create_dir(&blocker).unwrap();

        backend.mark_gpu_complete().unwrap();
        assert!(finish_with_timeout(Arc::clone(&backend)).is_err());
        assert!(!root.path().join("base/range-00000000.manifest").exists());
        assert!(
            root.path()
                .join("base/range-00000000-base-g00000000000000000001.bin")
                .is_file()
        );
        assert!(blocker.is_dir());
    }

    #[test]
    fn log_assisted_fault_truncated_run_is_reported_by_compactor() {
        let (root, backend) = background_backend(4096, 24);
        backend.pause_compaction_for_test();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        let run_path = backend.ranges[0]
            .mutable
            .lock()
            .frozen
            .as_ref()
            .unwrap()
            .delta_runs[0]
            .path
            .clone();
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&run_path)
            .unwrap();
        backend.release_compaction_for_test();

        assert!(finish_with_timeout(Arc::clone(&backend)).is_err());
        assert!(!root.path().join("base/range-00000000.manifest").exists());
    }

    #[test]
    fn log_assisted_fault_compactor_failure_wakes_blocked_budget_waiter() {
        let (_root, backend) = background_backend(2 * 24, 24);
        backend.pause_compaction_for_test();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        let run_path = backend.ranges[0]
            .mutable
            .lock()
            .frozen
            .as_ref()
            .unwrap()
            .delta_runs[0]
            .path
            .clone();

        let (tx, rx) = mpsc::channel();
        let producer = Arc::clone(&backend);
        let producer_handle = thread::spawn(move || {
            tx.send(producer.apply_edges(
                0,
                &[backend_edge(0, 12)],
                &mut LogAssistedApplyProfiles::default(),
            ))
            .unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(run_path)
            .unwrap();
        backend.release_compaction_for_test();

        assert!(rx.recv_timeout(Duration::from_secs(2)).unwrap().is_err());
        producer_handle.join().unwrap();
        assert!(finish_with_timeout(Arc::clone(&backend)).is_err());
    }

    #[test]
    fn range_threshold_schedules_compaction_before_finish() {
        let (_root, backend) = background_backend(4096, 3 * 24);
        backend
            .apply_edges(
                0,
                &[
                    backend_edge(0, 10),
                    backend_edge(0, 11),
                    backend_edge(0, 12),
                    backend_edge(0, 13),
                ],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.wait_for_background_idle().unwrap();
        assert!(backend.stats().compactions >= 1);
        backend.finish().unwrap();
    }

    #[test]
    fn run_pressure_scheduling_is_bounded_by_active_workers() {
        let root = tempfile::tempdir().unwrap();
        let record_bytes = super::super::reservoir_delta_log::DELTA_RECORD_BYTES;
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 12,
                shard_points: 4,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: 1 << 20,
                delta_buffer_bytes: 3 * record_bytes,
                range_trigger_bytes: 1 << 18,
                max_runs_per_range: 4,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 0,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 0,
                policy_key: [14; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        backend.pause_compaction_for_test();
        let run_reference_bytes =
            usize::try_from(backend.ranges[0].base.expected_base_bytes().unwrap())
                .unwrap()
                .min(backend.config.range_trigger_bytes);
        let record_count = run_reference_bytes.div_ceil(record_bytes).max(8);
        for range_id in 0..3 {
            let source = range_id * 4;
            for index in 0..record_count {
                backend
                    .apply_edges(
                        range_id,
                        &[backend_edge(
                            source,
                            100 + source as u32 * 1_000 + index as u32,
                        )],
                        &mut LogAssistedApplyProfiles::default(),
                    )
                    .unwrap();
            }
        }
        let frozen_ranges = backend
            .ranges
            .iter()
            .filter(|range| range.mutable.lock().frozen.is_some())
            .count();
        assert_eq!(frozen_ranges, 1);
        let stats = backend.stats();
        assert_eq!(stats.base_reads, 0);
        assert_eq!(stats.base_writes, 0);
        assert_eq!(stats.run_pressure_crossings, 3);
        assert_eq!(stats.run_pressure_compactions, 1);
        assert_eq!(stats.peak_active_delta_runs_per_range, record_count as u64);

        backend.release_compaction_for_test();
        backend.finish().unwrap();
        let final_stats = backend.stats();
        assert_eq!(final_stats.base_writes, 3);
        assert_eq!(backend.uncompacted_bytes(), 0);
    }

    #[test]
    fn run_pressure_materializes_each_range_only_once_before_gpu_completion() {
        let root = tempfile::tempdir().unwrap();
        let record_bytes = super::super::reservoir_delta_log::DELTA_RECORD_BYTES;
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 4,
                shard_points: 4,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: 1 << 20,
                delta_buffer_bytes: record_bytes,
                range_trigger_bytes: 1 << 18,
                max_runs_per_range: 4,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 0,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 0,
                policy_key: [18; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        let run_reference_bytes =
            usize::try_from(backend.ranges[0].base.expected_base_bytes().unwrap())
                .unwrap()
                .min(backend.config.range_trigger_bytes);
        let record_count = run_reference_bytes.div_ceil(record_bytes).max(8);

        backend.pause_compaction_for_test();
        for index in 0..record_count {
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 10_000 + index as u32)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
        }
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        backend.release_compaction_for_test();
        backend.wait_for_background_idle().unwrap();
        assert_eq!(backend.stats().base_writes, 1);

        backend.pause_compaction_for_test();
        for index in 0..record_count {
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 20_000 + index as u32)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
        }
        assert!(backend.ranges[0].mutable.lock().frozen.is_none());
        let stats = backend.stats();
        assert_eq!(stats.run_pressure_crossings, 2);
        assert_eq!(stats.run_pressure_compactions, 1);
        assert_eq!(stats.pre_gpu_materializations, 1);

        backend.mark_gpu_complete().unwrap();
        assert!(backend.ranges[0].mutable.lock().frozen.is_none());
        backend.release_compaction_for_test();
        backend.finish().unwrap();
        assert_eq!(backend.stats().base_writes, 2);
    }

    #[test]
    fn budget_high_forces_base_materialization() {
        let probe_root = tempfile::tempdir().unwrap();
        let probe = RangeBaseStore::create(RangeBaseStoreConfig {
            root: probe_root.path().to_path_buf(),
            range_id: 0,
            source_start: 0,
            source_end: 4,
            l_max: 4,
            policy_key: [15; 32],
            backfill_policy: HashPruneBackfillPolicy::off(),
        })
        .unwrap();
        let expected_base_bytes = usize::try_from(probe.expected_base_bytes().unwrap()).unwrap();
        let delta_bytes = super::super::reservoir_delta_log::DELTA_RECORD_BYTES;
        let budget_bytes = expected_base_bytes
            .div_ceil(4)
            .checked_mul(5)
            .unwrap()
            .div_ceil(delta_bytes)
            .checked_mul(delta_bytes)
            .unwrap();
        let budget_high = budget_bytes - budget_bytes / 5;

        let root = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 4,
                shard_points: 4,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: budget_bytes,
                delta_buffer_bytes: delta_bytes,
                range_trigger_bytes: budget_bytes,
                max_runs_per_range: 2,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 0,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 0,
                policy_key: [15; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        backend.pause_compaction_for_test();
        let record_count = budget_high.div_ceil(delta_bytes);
        for index in 0..record_count {
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 1_000 + index as u32)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
        }
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        assert!(backend.ranges[0].mutable.lock().frozen.is_some());

        backend.release_compaction_for_test();
        backend.finish().unwrap();
    }

    #[test]
    fn blocked_admission_wakes_after_forced_materialization() {
        let probe_root = tempfile::tempdir().unwrap();
        let probe = RangeBaseStore::create(RangeBaseStoreConfig {
            root: probe_root.path().to_path_buf(),
            range_id: 0,
            source_start: 0,
            source_end: 4,
            l_max: 4,
            policy_key: [16; 32],
            backfill_policy: HashPruneBackfillPolicy::off(),
        })
        .unwrap();
        let expected_base_bytes = usize::try_from(probe.expected_base_bytes().unwrap()).unwrap();
        let delta_bytes = super::super::reservoir_delta_log::DELTA_RECORD_BYTES;
        let budget_bytes = expected_base_bytes
            .checked_mul(4)
            .unwrap()
            .div_ceil(delta_bytes)
            .checked_mul(delta_bytes)
            .unwrap();

        let root = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 4,
                shard_points: 4,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: budget_bytes,
                delta_buffer_bytes: delta_bytes,
                range_trigger_bytes: budget_bytes,
                max_runs_per_range: 2,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 0,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 0,
                policy_key: [16; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        backend.pause_compaction_for_test();
        let budget_high = budget_bytes - budget_bytes / 5;
        let initial_records = budget_high.div_ceil(delta_bytes);
        for index in 0..initial_records {
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 2_000 + index as u32)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
        }
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        assert!(backend.ranges[0].mutable.lock().frozen.is_some());

        let capacity_records = budget_bytes / delta_bytes;
        for index in initial_records..capacity_records {
            backend
                .apply_edges(
                    0,
                    &[backend_edge(0, 2_000 + index as u32)],
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
        }
        assert_eq!(backend.budget.used(), budget_bytes);

        let (tx, rx) = mpsc::channel();
        let producer = Arc::clone(&backend);
        let handle = thread::spawn(move || {
            tx.send(producer.apply_edges(
                0,
                &[backend_edge(0, 9_999)],
                &mut LogAssistedApplyProfiles::default(),
            ))
            .unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
        backend.release_compaction_for_test();
        let result = rx.recv_timeout(Duration::from_secs(3));
        if result.is_err() {
            backend.record_failure(&"budget retry test timed out");
            handle.join().unwrap();
            panic!("blocked admission did not wake after forced materialization");
        }
        result.unwrap().unwrap();
        handle.join().unwrap();

        assert!(backend.stats().base_writes >= 1);
        assert!(backend.stats().delta_backpressure_count >= 1);
        backend.finish().unwrap();
        assert_eq!(backend.uncompacted_bytes(), 0);
    }

    #[test]
    fn deferred_run_pressure_preserves_dirty_cache_until_final_major() {
        let root = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 4,
                shard_points: 4,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: 1 << 20,
                delta_buffer_bytes: super::super::reservoir_delta_log::DELTA_RECORD_BYTES,
                range_trigger_bytes: 1 << 18,
                max_runs_per_range: 2,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 1,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 1,
                policy_key: [17; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        let mut edges = vec![backend_edge(0, 10), backend_edge(0, 11)];
        backend
            .apply_edges(0, &edges, &mut LogAssistedApplyProfiles::default())
            .unwrap();
        backend.compact_range_for_test(0).unwrap();
        assert!(backend.cache_contains_for_test(0));

        let dirty_update = backend_edge(0, 12);
        backend
            .apply_edges(
                0,
                std::slice::from_ref(&dirty_update),
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        edges.push(dirty_update);
        let cold_records = 4;
        for index in 0..cold_records {
            let edge = backend_edge(1, 3_000 + index as u32);
            backend
                .apply_edges(
                    0,
                    std::slice::from_ref(&edge),
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
            edges.push(edge);
        }
        {
            let range = backend.ranges[0].mutable.lock();
            assert!(range.frozen.is_none());
            assert!(range.active_delta_runs.len() >= 2);
        }
        assert!(backend.cache.has_dirty_state());
        assert_eq!(backend.stats().base_writes, 1);

        backend.finish().unwrap();
        let mut actual = Vec::new();
        backend
            .for_each_source(|_source, reservoir| {
                actual.push(state_signature(reservoir.state_snapshot()));
                Ok(())
            })
            .unwrap();
        let expected = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                edge_record(
                    edge.p as u32,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    index as u64 + 1,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            snapshot_reservoirs(apply_in_memory(0, 4, 4, &expected))
        );
        assert_eq!(backend.stats().base_writes, 2);
    }

    #[test]
    fn epoch_cut_keeps_post_cut_delta_out_of_frozen_input() {
        let (_root, backend) = background_backend(4096, 24);
        backend.pause_compaction_for_test();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert_eq!(backend.frozen_targets_for_test(0).unwrap(), vec![10]);
        assert_eq!(backend.active_targets_for_test(0).unwrap(), vec![11]);
        backend.release_compaction_for_test();
        backend.finish().unwrap();
    }

    #[test]
    fn hard_budget_blocks_then_wakes_after_compaction() {
        let (_root, backend) = background_backend(2 * 24, 24);
        backend.pause_compaction_for_test();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10), backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();

        let (tx, rx) = mpsc::channel();
        let producer = Arc::clone(&backend);
        let handle = thread::spawn(move || {
            let result = producer.apply_edges(
                0,
                &[backend_edge(0, 12)],
                &mut LogAssistedApplyProfiles::default(),
            );
            tx.send(result).unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
        backend.release_compaction_for_test();
        rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        handle.join().unwrap();
        assert!(backend.stats().delta_backpressure_count >= 1);
        backend.finish().unwrap();
    }

    #[test]
    fn log_assisted_fault_finish_races_last_compaction_completion() {
        let (_root, backend) = background_backend(4096, 24);
        backend.pause_compaction_for_test();
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();

        let (tx, rx) = mpsc::channel();
        let finisher = Arc::clone(&backend);
        let handle = thread::spawn(move || tx.send(finisher.finish()).unwrap());
        assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
        backend.release_compaction_for_test();
        rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        handle.join().unwrap();
        assert_eq!(backend.uncompacted_bytes(), 0);
    }

    #[test]
    fn gpu_complete_defers_final_cut_until_ingress_drains() {
        let root = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 4,
                shard_points: 4,
                range_points: 4,
                l_max: 4,
                delta_budget_bytes: 1 << 20,
                delta_buffer_bytes: super::super::reservoir_delta_log::DELTA_RECORD_BYTES,
                range_trigger_bytes: 1 << 18,
                max_runs_per_range: 2,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 0,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 0,
                policy_key: [13; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );
        backend.pause_compaction_for_test();
        for edge in [backend_edge(0, 10), backend_edge(1, 20)] {
            backend
                .apply_edges(
                    0,
                    std::slice::from_ref(&edge),
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
        }
        {
            let range = backend.ranges[0].mutable.lock();
            assert_eq!(range.active_delta_runs.len(), 2);
            assert!(range.frozen.is_none());
        }

        backend.mark_gpu_complete().unwrap();
        assert!(backend.ranges[0].mutable.lock().frozen.is_none());
        backend
            .apply_edges(
                0,
                &[backend_edge(2, 30)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert!(backend.ranges[0].mutable.lock().frozen.is_none());
        backend.release_compaction_for_test();
        backend.finish().unwrap();
        assert_eq!(backend.stats().base_writes, 1);
    }

    #[test]
    fn compaction_promotes_hot_source_without_base_read() {
        let (_root, backend) = cache_backend(1, 2);
        backend
            .apply_edges(
                0,
                &[backend_edge(2, 20), backend_edge(2, 21)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.compact_range_for_test(0).unwrap();
        assert!(backend.cache_contains_for_test(2));

        let reads = backend.stats().base_reads;
        backend
            .apply_edges(
                0,
                &[backend_edge(2, 22)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert_eq!(backend.stats().base_reads, reads);
        assert_eq!(backend.stats().cache_hits, 1);
        backend.finish().unwrap();
    }

    #[test]
    fn post_cut_update_skips_promotion() {
        let (_root, backend) = cache_backend(1, 2);
        backend
            .apply_edges(
                0,
                &[backend_edge(2, 20), backend_edge(2, 21)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.pause_compaction_for_test();
        backend.schedule_range_for_test(0).unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        backend
            .apply_edges(
                0,
                &[backend_edge(2, 22)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.release_compaction_for_test();
        backend.wait_for_background_idle().unwrap();

        assert!(!backend.cache_contains_for_test(2));
        assert_eq!(backend.stats().cache_promotion_skipped_post_cut, 1);
        backend.finish().unwrap();
    }

    #[test]
    fn promotion_does_not_evict_dirty_cache_state() {
        let (_root, backend) = cache_backend(1, 2);
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 10), backend_edge(0, 11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.compact_range_for_test(0).unwrap();
        assert!(backend.cache_contains_for_test(0));
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 12)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();

        backend
            .apply_edges(
                0,
                &[backend_edge(4, 40), backend_edge(4, 41)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.compact_range_for_test(1).unwrap();

        assert!(backend.cache_contains_for_test(0));
        assert!(!backend.cache_contains_for_test(4));
        assert_eq!(backend.checkpoint_count_for_test(0), 0);
        assert_eq!(backend.stats().cache_dirty_checkpoints, 0);
        let reads = backend.stats().base_reads;
        let hits = backend.stats().cache_hits;
        backend
            .apply_edges(
                0,
                &[backend_edge(0, 13)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        assert_eq!(backend.stats().base_reads, reads);
        assert_eq!(backend.stats().cache_hits, hits + 1);
        backend.finish().unwrap();
    }

    #[test]
    fn dirty_evictions_are_batched_into_one_checkpoint_run_per_range_cut() {
        let root = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LogAssistedReservoir::create(LogAssistedReservoirConfig {
                root: root.path().to_path_buf(),
                source_start: 0,
                source_end: 32,
                shard_points: 32,
                range_points: 16,
                l_max: 4,
                delta_budget_bytes: 1 << 24,
                delta_buffer_bytes: 1 << 20,
                range_trigger_bytes: 1 << 23,
                max_runs_per_range: 1024,
                compaction_workers: 1,
                drain_compaction_workers: 1,
                cache_capacity_entries: 8,
                cache_hot_updates: 2,
                promotion_limit_per_compaction: 8,
                policy_key: [12; 32],
                evidence_policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
                diagnostic_collector: None,
            })
            .unwrap(),
        );

        let first_range = (0..8)
            .flat_map(|source| {
                [
                    backend_edge(source, 100 + source as u32 * 2),
                    backend_edge(source, 101 + source as u32 * 2),
                ]
            })
            .collect::<Vec<_>>();
        backend
            .apply_edges(0, &first_range, &mut LogAssistedApplyProfiles::default())
            .unwrap();
        backend.compact_range_for_test(0).unwrap();

        let dirty_updates = (0..8)
            .map(|source| backend_edge(source, 200 + source as u32))
            .collect::<Vec<_>>();
        backend
            .apply_edges(0, &dirty_updates, &mut LogAssistedApplyProfiles::default())
            .unwrap();

        let second_range = (16..24)
            .flat_map(|source| {
                [
                    backend_edge(source, 300 + source as u32 * 2),
                    backend_edge(source, 301 + source as u32 * 2),
                ]
            })
            .collect::<Vec<_>>();
        backend
            .apply_edges(0, &second_range, &mut LogAssistedApplyProfiles::default())
            .unwrap();
        backend.compact_range_for_test(1).unwrap();

        let checkpoint_files = fs::read_dir(root.path().join("runs"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "checkpoint")
            })
            .count();
        assert_eq!(
            checkpoint_files, 0,
            "dirty eviction must not create one checkpoint run per source"
        );

        backend.pause_compaction_for_test();
        backend.schedule_range_for_test(0).unwrap();
        backend.wait_for_frozen_epoch_for_test(0).unwrap();
        {
            let range = backend.ranges[0].mutable.lock();
            let frozen = range.frozen.as_ref().unwrap();
            assert_eq!(frozen.checkpoint_runs.len(), 1);
            assert_eq!(frozen.checkpoint_runs[0].record_count, 8);
        }
        assert_eq!(backend.stats().checkpoint_runs, 1);
        backend.release_compaction_for_test();
        backend.wait_for_background_idle().unwrap();
        backend.finish().unwrap();
    }

    #[test]
    fn randomized_log_cache_matches_in_memory() {
        let (_root, backend) = cache_backend(2, 2);
        let mut rng = StdRng::seed_from_u64(0xacc4_cafe);
        let mut edges = Vec::new();
        for index in 0..160 {
            let source = rng.random_range(0..8);
            let target = 1000 + index as u32;
            let mut edge = backend_edge(source, target);
            edge.hash = rng.random_range(0..12);
            edge.dist = rng.random_range(0.0..100.0);
            backend
                .apply_edges(
                    0,
                    std::slice::from_ref(&edge),
                    &mut LogAssistedApplyProfiles::default(),
                )
                .unwrap();
            edges.push(edge);
            if index % 17 == 16 {
                backend.compact_range_for_test((index / 17) % 2).unwrap();
            }
        }
        backend.finish().unwrap();

        let mut actual = Vec::new();
        backend
            .for_each_source(|_source, reservoir| {
                actual.push(state_signature(reservoir.state_snapshot()));
                Ok(())
            })
            .unwrap();
        let expected = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                edge_record(
                    edge.p as u32,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    index as u64 + 1,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            snapshot_reservoirs(apply_in_memory(0, 8, 4, &expected))
        );
    }
}
