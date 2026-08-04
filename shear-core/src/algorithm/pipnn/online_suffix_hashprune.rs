use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::direct_io::DirectIoConfig;
use super::disk_reservoir_store::{DiskReservoirStore, DiskReservoirStoreConfig};
use super::hash_prune::{HashPruneInsertProfile, HashPruneReservoir};
use super::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use super::hash_prune_diagnostics::{HashPruneDiagnosticFinalizer, HashPruneTraceCollector};
use super::hash_prune_evidence::{HashPruneEvidencePolicy, HashPruneEvidenceProfile};
use super::leaf_build::{
    PendingEdge, PendingEdgeFlushProfile, PendingEdgeSink, prepare_pending_edges_for_sink,
};
use super::log_assisted_reservoir::{
    LogAssistedApplyProfiles, LogAssistedReservoir, LogAssistedReservoirConfig,
};
use super::params::PiPNNOnlineSuffixReduceMode;
use super::reservoir_writeback_cache::{
    ShardedReservoirWritebackCache, ShardedReservoirWritebackCacheConfig,
    estimated_writeback_cache_entry_bytes,
};
use super::spill_hashprune::{
    CandidateRunWriter, RUN_RECORD_BYTES, SpillSegmentDesc, replay_single_spill_segment,
};
use crate::common::{AnnError, AnnResult};

const ONLINE_FALLBACK_PART_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const ONLINE_SPILL_SOURCE_BATCH_EDGES: usize = 4096;

#[derive(Clone, Debug)]
pub(crate) struct OnlineSpillSegment {
    pub(crate) shard_id: usize,
    pub(crate) ingress_seq: u64,
    pub(crate) part_id: u16,
    pub(crate) source_count: u32,
    pub(crate) desc: SpillSegmentDesc,
}

struct OnlineFallbackPart {
    part_id: u16,
    path: PathBuf,
    writer: CandidateRunWriter,
    byte_len: u64,
}

#[derive(Debug)]
struct FallbackPartLifecycle {
    path: Arc<PathBuf>,
    live_segments: BTreeSet<(u64, u64)>,
    readers: usize,
    sealed: bool,
}

#[derive(Debug, Default)]
struct OnlineFallbackRegistryInner {
    parts: BTreeMap<u16, FallbackPartLifecycle>,
}

#[derive(Clone, Debug)]
pub(crate) struct OnlineFallbackRegistry {
    inner: Arc<Mutex<OnlineFallbackRegistryInner>>,
    io_cfg: DirectIoConfig,
}

impl OnlineFallbackRegistry {
    fn register_part(&self, part_id: u16, path: PathBuf) -> AnnResult<()> {
        let mut inner = self.inner.lock();
        if inner
            .parts
            .insert(
                part_id,
                FallbackPartLifecycle {
                    path: Arc::new(path),
                    live_segments: BTreeSet::new(),
                    readers: 0,
                    sealed: false,
                },
            )
            .is_some()
        {
            return Err(AnnError::log_index_error(format!(
                "online fallback part {part_id} was registered twice"
            )));
        }
        Ok(())
    }

    fn add_live_segment(&self, segment: &OnlineSpillSegment) -> AnnResult<()> {
        let mut inner = self.inner.lock();
        let part = inner.parts.get_mut(&segment.part_id).ok_or_else(|| {
            AnnError::log_index_error(format!("unknown online fallback part {}", segment.part_id))
        })?;
        let key = (segment.desc.byte_offset, segment.desc.byte_len);
        if !part.live_segments.insert(key) {
            return Err(AnnError::log_index_error(format!(
                "online fallback part {} segment [{}, {}) was registered twice",
                segment.part_id,
                key.0,
                key.0.saturating_add(key.1)
            )));
        }
        Ok(())
    }

    fn seal_part(&self, part_id: u16) -> AnnResult<()> {
        let mut inner = self.inner.lock();
        let part = inner.parts.get_mut(&part_id).ok_or_else(|| {
            AnnError::log_index_error(format!("unknown online fallback part {part_id}"))
        })?;
        part.sealed = true;
        reclaim_part_if_ready(&mut inner, part_id)
    }

    pub(crate) fn mark_consumed(&self, segment: &OnlineSpillSegment) -> AnnResult<()> {
        let mut inner = self.inner.lock();
        let part = inner.parts.get_mut(&segment.part_id).ok_or_else(|| {
            AnnError::log_index_error(format!("unknown online fallback part {}", segment.part_id))
        })?;
        let key = (segment.desc.byte_offset, segment.desc.byte_len);
        if !part.live_segments.remove(&key) {
            return Err(AnnError::log_index_error(format!(
                "online fallback part {} segment [{}, {}) is unknown or was consumed twice",
                segment.part_id,
                key.0,
                key.0.saturating_add(key.1)
            )));
        }
        reclaim_part_if_ready(&mut inner, segment.part_id)
    }

    pub(crate) fn replay_segment<F>(
        &self,
        segment: &OnlineSpillSegment,
        on_edge: F,
    ) -> AnnResult<()>
    where
        F: FnMut(PendingEdge) -> AnnResult<()>,
    {
        let guard = self.acquire_reader(segment.part_id)?;
        replay_single_spill_segment(guard.path(), segment.desc, self.io_cfg, on_edge)
    }

    pub(crate) fn part_path(&self, part_id: u16) -> Option<PathBuf> {
        self.inner
            .lock()
            .parts
            .get(&part_id)
            .map(|part| part.path.as_ref().clone())
    }

    pub(crate) fn artifact_paths(&self) -> Vec<PathBuf> {
        self.inner
            .lock()
            .parts
            .values()
            .map(|part| part.path.as_ref().clone())
            .collect()
    }

    pub(crate) fn acquire_reader(&self, part_id: u16) -> AnnResult<OnlineFallbackReaderGuard> {
        let path = {
            let mut inner = self.inner.lock();
            let part = inner.parts.get_mut(&part_id).ok_or_else(|| {
                AnnError::log_index_error(format!("unknown online fallback part {part_id}"))
            })?;
            part.readers = part.readers.checked_add(1).ok_or_else(|| {
                AnnError::log_index_error("online fallback reader-guard count overflow".to_string())
            })?;
            Arc::clone(&part.path)
        };
        Ok(OnlineFallbackReaderGuard {
            registry: self.clone(),
            part_id,
            path,
        })
    }

    fn release_reader(&self, part_id: u16) -> AnnResult<()> {
        let mut inner = self.inner.lock();
        let part = inner.parts.get_mut(&part_id).ok_or_else(|| {
            AnnError::log_index_error(format!("unknown online fallback part {part_id}"))
        })?;
        if part.readers == 0 {
            return Err(AnnError::log_index_error(
                "online fallback reader guard underflow".to_string(),
            ));
        }
        part.readers -= 1;
        reclaim_part_if_ready(&mut inner, part_id)
    }
}

pub(crate) struct OnlineFallbackReaderGuard {
    registry: OnlineFallbackRegistry,
    part_id: u16,
    path: Arc<PathBuf>,
}

impl OnlineFallbackReaderGuard {
    pub(crate) fn path(&self) -> &Path {
        self.path.as_path()
    }
}

impl Drop for OnlineFallbackReaderGuard {
    fn drop(&mut self) {
        let _ = self.registry.release_reader(self.part_id);
    }
}

fn reclaim_part_if_ready(inner: &mut OnlineFallbackRegistryInner, part_id: u16) -> AnnResult<()> {
    let Some(part) = inner.parts.get(&part_id) else {
        return Ok(());
    };
    if !part.sealed || !part.live_segments.is_empty() || part.readers != 0 {
        return Ok(());
    }
    let path = part.path.as_ref().clone();
    fs::remove_file(&path)?;
    inner.parts.remove(&part_id);
    Ok(())
}

pub(crate) struct OnlineFallbackWriter {
    root: PathBuf,
    io_cfg: DirectIoConfig,
    buffer_bytes: usize,
    part_max_bytes: u64,
    range_split: Option<(usize, usize)>,
    next_part_id: u32,
    current: Option<OnlineFallbackPart>,
    registry: OnlineFallbackRegistry,
    #[cfg(test)]
    fail_next_append: bool,
}

impl OnlineFallbackWriter {
    pub(crate) fn create(
        root: &Path,
        io_cfg: DirectIoConfig,
        buffer_bytes: usize,
    ) -> AnnResult<Self> {
        Self::create_with_options(
            root,
            io_cfg,
            buffer_bytes,
            ONLINE_FALLBACK_PART_MAX_BYTES,
            None,
        )
    }

    fn create_range_local(
        root: &Path,
        io_cfg: DirectIoConfig,
        buffer_bytes: usize,
        source_start: usize,
        range_points: usize,
    ) -> AnnResult<Self> {
        Self::create_with_options(
            root,
            io_cfg,
            buffer_bytes,
            ONLINE_FALLBACK_PART_MAX_BYTES,
            Some((source_start, range_points)),
        )
    }

    fn create_with_part_limit(
        root: &Path,
        io_cfg: DirectIoConfig,
        buffer_bytes: usize,
        part_max_bytes: u64,
    ) -> AnnResult<Self> {
        Self::create_with_options(root, io_cfg, buffer_bytes, part_max_bytes, None)
    }

    fn create_with_options(
        root: &Path,
        io_cfg: DirectIoConfig,
        buffer_bytes: usize,
        part_max_bytes: u64,
        range_split: Option<(usize, usize)>,
    ) -> AnnResult<Self> {
        if buffer_bytes < RUN_RECORD_BYTES {
            return Err(AnnError::log_index_config_error(
                "spill_buffer_bytes".to_string(),
                format!("online fallback buffer must hold at least {RUN_RECORD_BYTES} bytes"),
            ));
        }
        if part_max_bytes < RUN_RECORD_BYTES as u64 || part_max_bytes % RUN_RECORD_BYTES as u64 != 0
        {
            return Err(AnnError::log_index_config_error(
                "part_max_bytes".to_string(),
                format!(
                    "online fallback part limit must be a positive multiple of {RUN_RECORD_BYTES}"
                ),
            ));
        }
        if range_split.is_some_and(|(_, range_points)| range_points == 0) {
            return Err(AnnError::log_index_config_error(
                "range_points".to_string(),
                "online fallback range split must be nonzero".to_string(),
            ));
        }
        fs::create_dir_all(root)?;
        Ok(Self {
            root: root.to_path_buf(),
            io_cfg,
            buffer_bytes,
            part_max_bytes,
            range_split,
            next_part_id: 0,
            current: None,
            registry: OnlineFallbackRegistry {
                inner: Arc::new(Mutex::new(OnlineFallbackRegistryInner::default())),
                io_cfg,
            },
            #[cfg(test)]
            fail_next_append: false,
        })
    }

    #[cfg(test)]
    fn for_test(root: &Path, buffer_bytes: usize) -> AnnResult<Self> {
        Self::create(root, DirectIoConfig::disabled(), buffer_bytes)
    }

    #[cfg(test)]
    fn for_test_with_part_limit(
        root: &Path,
        buffer_bytes: usize,
        part_max_bytes: u64,
    ) -> AnnResult<Self> {
        Self::create_with_part_limit(
            root,
            DirectIoConfig::disabled(),
            buffer_bytes,
            part_max_bytes,
        )
    }

    #[cfg(test)]
    pub(crate) fn for_test_range_local(
        root: &Path,
        buffer_bytes: usize,
        source_start: usize,
        range_points: usize,
    ) -> AnnResult<Self> {
        Self::create_range_local(
            root,
            DirectIoConfig::disabled(),
            buffer_bytes,
            source_start,
            range_points,
        )
    }

    pub(crate) fn append(
        &mut self,
        shard_id: usize,
        ingress_seq: u64,
        edges: &[PendingEdge],
    ) -> AnnResult<Vec<OnlineSpillSegment>> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_append) {
            return Err(AnnError::log_index_error(
                "injected fallback write failure".to_string(),
            ));
        }
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        if edges.windows(2).any(|pair| pair[0].p > pair[1].p) {
            return Err(AnnError::log_index_error(
                "online fallback batch must be sorted by source".to_string(),
            ));
        }
        let shard_id_u32 = u32::try_from(shard_id)?;
        let mut groups = Vec::new();
        let mut group_start = 0usize;
        for index in 1..=edges.len() {
            if index == edges.len() || edges[index].p != edges[group_start].p {
                groups.push(group_start..index);
                group_start = index;
            }
        }

        let mut segments = Vec::new();
        let mut group_index = 0usize;
        while group_index < groups.len() {
            self.ensure_current_part()?;
            let remaining = self
                .part_max_bytes
                .checked_sub(self.current.as_ref().expect("current part").byte_len)
                .expect("part length cannot exceed limit");
            let segment_group_start = group_index;
            let segment_range_id = self.range_id(edges[groups[group_index].start].p)?;
            let mut segment_records = 0usize;
            while group_index < groups.len() {
                if self.range_id(edges[groups[group_index].start].p)? != segment_range_id {
                    break;
                }
                let group_records = groups[group_index].len();
                let group_bytes = u64::try_from(group_records)?
                    .checked_mul(RUN_RECORD_BYTES as u64)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "online fallback source group size overflows u64".to_string(),
                        )
                    })?;
                let accumulated_bytes = u64::try_from(segment_records)?
                    .checked_mul(RUN_RECORD_BYTES as u64)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "online fallback segment size overflows u64".to_string(),
                        )
                    })?;
                if accumulated_bytes + group_bytes > remaining {
                    break;
                }
                segment_records += group_records;
                group_index += 1;
            }
            if segment_records == 0 {
                if self.current.as_ref().expect("current part").byte_len != 0 {
                    self.seal_current_part()?;
                    continue;
                }
                return Err(AnnError::log_index_error(format!(
                    "one source group exceeds the {}-byte online fallback part limit",
                    self.part_max_bytes
                )));
            }

            let edge_start = groups[segment_group_start].start;
            let edge_end = groups[group_index - 1].end;
            let part = self.current.as_mut().expect("current fallback part");
            let byte_offset = part.byte_len;
            for edge in &edges[edge_start..edge_end] {
                part.writer.push(edge)?;
            }
            let byte_len = u64::try_from(segment_records)?
                .checked_mul(RUN_RECORD_BYTES as u64)
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "online fallback segment byte length overflows u64".to_string(),
                    )
                })?;
            part.byte_len += byte_len;
            let segment = OnlineSpillSegment {
                shard_id,
                ingress_seq,
                part_id: part.part_id,
                source_count: u32::try_from(group_index - segment_group_start)?,
                desc: SpillSegmentDesc {
                    shard_id: shard_id_u32,
                    file_id: part.part_id,
                    byte_offset,
                    byte_len,
                    record_count: u32::try_from(segment_records)?,
                    src_min: u32::try_from(edges[edge_start].p)?,
                    src_max: u32::try_from(edges[edge_end - 1].p)?,
                    sorted_by_src: true,
                },
            };
            self.registry.add_live_segment(&segment)?;
            segments.push(segment);
            if part.byte_len == self.part_max_bytes && group_index < groups.len() {
                self.seal_current_part()?;
            }
        }
        if let Some(part) = self.current.as_mut() {
            part.writer.flush_visible()?;
        }
        Ok(segments)
    }

    fn range_id(&self, source: usize) -> AnnResult<Option<usize>> {
        let Some((source_start, range_points)) = self.range_split else {
            return Ok(None);
        };
        let relative = source.checked_sub(source_start).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "online fallback source {source} precedes range start {source_start}"
            ))
        })?;
        Ok(Some(relative / range_points))
    }

    pub(crate) fn seal_all(&mut self) -> AnnResult<()> {
        self.seal_current_part()
    }

    pub(crate) fn mark_consumed(&self, segment: &OnlineSpillSegment) -> AnnResult<()> {
        self.registry.mark_consumed(segment)
    }

    pub(crate) fn replay_segment<F>(
        &self,
        segment: &OnlineSpillSegment,
        on_edge: F,
    ) -> AnnResult<()>
    where
        F: FnMut(PendingEdge) -> AnnResult<()>,
    {
        self.registry.replay_segment(segment, on_edge)
    }

    pub(crate) fn part_path(&self, part_id: u16) -> Option<PathBuf> {
        self.registry.part_path(part_id)
    }

    pub(crate) fn registry(&self) -> OnlineFallbackRegistry {
        self.registry.clone()
    }

    #[cfg(test)]
    fn fail_next_append_for_test(&mut self) {
        self.fail_next_append = true;
    }

    fn ensure_current_part(&mut self) -> AnnResult<()> {
        if self.current.is_some() {
            return Ok(());
        }
        let part_id = u16::try_from(self.next_part_id).map_err(|_| {
            AnnError::log_index_error("online fallback exhausted u16 part ids".to_string())
        })?;
        self.next_part_id = self.next_part_id.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error("online fallback part id overflow".to_string())
        })?;
        let path = self
            .root
            .join(format!("online-fallback-part-{part_id:05}.run"));
        let writer = CandidateRunWriter::create_with_config_and_buffer(
            &path,
            self.io_cfg,
            self.buffer_bytes,
        )?;
        self.registry.register_part(part_id, path.clone())?;
        self.current = Some(OnlineFallbackPart {
            part_id,
            path,
            writer,
            byte_len: 0,
        });
        Ok(())
    }

    fn seal_current_part(&mut self) -> AnnResult<()> {
        let Some(part) = self.current.take() else {
            return Ok(());
        };
        part.writer.finish()?;
        debug_assert_eq!(
            self.registry.part_path(part.part_id).as_deref(),
            Some(part.path.as_path())
        );
        self.registry.seal_part(part.part_id)
    }
}

const REDUCER_ITEM_ACCOUNTING_OVERHEAD: usize = 64;

#[derive(Clone)]
pub(crate) struct OnlineSuffixReduceConfig {
    pub(crate) source_start: usize,
    pub(crate) source_end: usize,
    pub(crate) shard_points: usize,
    pub(crate) l_max: usize,
    pub(crate) direct_queue_bytes: usize,
    pub(crate) spill_buffer_bytes: usize,
    pub(crate) reorder_window_bytes: usize,
    pub(crate) reducer_workers: usize,
    pub(crate) mode: PiPNNOnlineSuffixReduceMode,
    pub(crate) state_cache_bytes: usize,
    pub(crate) read_window_bytes: usize,
    pub(crate) pinned_prefix_points: usize,
    pub(crate) delta_budget_bytes: usize,
    pub(crate) delta_buffer_bytes: usize,
    pub(crate) compaction_bytes: usize,
    pub(crate) range_points: usize,
    pub(crate) range_trigger_bytes: usize,
    pub(crate) max_runs_per_range: usize,
    pub(crate) compaction_workers: usize,
    pub(crate) drain_compaction_workers: usize,
    pub(crate) cache_hot_updates: usize,
    pub(crate) store: DiskReservoirStoreConfig,
    pub(crate) evidence_policy: HashPruneEvidencePolicy,
    pub(crate) backfill_policy: HashPruneBackfillPolicy,
    pub(crate) diagnostic_collector: Option<Arc<HashPruneTraceCollector>>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OnlineSuffixReduceStats {
    pub(crate) direct_candidates: u64,
    pub(crate) direct_batches: u64,
    pub(crate) direct_queue_peak_bytes: usize,
    pub(crate) spill_fallback_candidates: u64,
    pub(crate) spill_fallback_bytes: u64,
    pub(crate) spill_segments_consumed: u64,
    pub(crate) state_records_loaded: u64,
    pub(crate) state_records_updated: u64,
    pub(crate) state_page_cache_hits: u64,
    pub(crate) state_page_cache_misses: u64,
    pub(crate) state_pages_read: u64,
    pub(crate) state_pages_written: u64,
    pub(crate) state_slots_reused: u64,
    pub(crate) state_write_amplification: f64,
    pub(crate) state_cache_partitions: usize,
    pub(crate) state_cache_capacity_entries: usize,
    pub(crate) state_cache_peak_entries: usize,
    pub(crate) state_cache_hits: u64,
    pub(crate) state_cache_misses: u64,
    pub(crate) state_cache_clean_evictions: u64,
    pub(crate) state_cache_dirty_evictions: u64,
    pub(crate) state_cache_coalesced_updates: u64,
    pub(crate) state_read_windows: u64,
    pub(crate) state_read_pages_requested: u64,
    pub(crate) state_read_pages_coalesced: u64,
    pub(crate) state_writeback_cohorts: u64,
    pub(crate) state_writeback_records: u64,
    pub(crate) state_cache_updates_per_commit: f64,
    pub(crate) state_alloc_empty_page: u64,
    pub(crate) state_alloc_resident_partial_page: u64,
    pub(crate) state_alloc_cold_partial_page: u64,
    pub(crate) cold_batch_commits: u64,
    pub(crate) cold_batch_records: u64,
    pub(crate) delta_records: u64,
    pub(crate) delta_runs: u64,
    pub(crate) delta_bytes: u64,
    pub(crate) borrowed_spill_records: u64,
    pub(crate) borrowed_spill_runs: u64,
    pub(crate) borrowed_spill_bytes: u64,
    pub(crate) delta_buffer_flushes: u64,
    pub(crate) delta_uncompacted_bytes: u64,
    pub(crate) delta_uncompacted_peak_bytes: u64,
    pub(crate) delta_backpressure_count: u64,
    pub(crate) delta_backpressure_wall: Duration,
    pub(crate) checkpoint_records: u64,
    pub(crate) checkpoint_runs: u64,
    pub(crate) checkpoint_bytes: u64,
    pub(crate) range_compactions: u64,
    pub(crate) delta_run_pressure_crossings: u64,
    pub(crate) delta_run_pressure_compactions: u64,
    pub(crate) range_pre_gpu_materializations: u64,
    pub(crate) log_shards_finalized: u64,
    pub(crate) range_ingress_drained_compactions: u64,
    pub(crate) delta_peak_active_runs_per_range: u64,
    pub(crate) range_compactions_inflight: u64,
    pub(crate) range_compactions_peak_inflight: u64,
    pub(crate) post_gpu_range_compactions_peak_inflight: u64,
    pub(crate) range_rows_compacted: u64,
    pub(crate) range_rows_changed: u64,
    pub(crate) range_compaction_input_bytes: u64,
    pub(crate) range_compaction_output_bytes: u64,
    pub(crate) range_compaction_read_wall: Duration,
    pub(crate) range_compaction_merge_wall: Duration,
    pub(crate) range_compaction_write_wall: Duration,
    pub(crate) range_compaction_hidden_wall: Duration,
    pub(crate) range_manifest_switches: u64,
    pub(crate) cold_miss_base_reads: u64,
    pub(crate) log_base_reads: u64,
    pub(crate) log_base_writes: u64,
    pub(crate) post_gpu_uncompacted_bytes: u64,
    pub(crate) post_gpu_compaction_wall: Duration,
    pub(crate) log_cache_promotions: u64,
    pub(crate) log_cache_promotion_skipped_post_cut: u64,
    pub(crate) log_cache_hits: u64,
    pub(crate) log_cache_dirty_checkpoints: u64,
    pub(crate) pinned_prefix_points: usize,
    pub(crate) backlog_bytes_at_last_gpu: usize,
    pub(crate) post_gpu_drain_wall: Duration,
    pub(crate) online_reduce_wall: Duration,
    pub(crate) online_reduce_hidden_wall: Duration,
    pub(crate) evidence: HashPruneEvidenceProfile,
    pub(crate) backfill: HashPruneBackfillProfile,
}

#[derive(Default)]
struct FirstError {
    message: Mutex<Option<String>>,
}

impl FirstError {
    fn record(&self, error: impl std::fmt::Display) {
        let mut message = self.message.lock();
        if message.is_none() {
            *message = Some(error.to_string());
        }
    }

    fn check(&self) -> AnnResult<()> {
        match self.message.lock().clone() {
            Some(message) => Err(AnnError::log_index_error(message)),
            None => Ok(()),
        }
    }
}

#[derive(Default)]
struct ByteBudgetState {
    bytes: usize,
    peak_bytes: usize,
}

struct ByteBudget {
    limit: usize,
    state: Mutex<ByteBudgetState>,
    changed: Condvar,
}

impl ByteBudget {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            state: Mutex::new(ByteBudgetState::default()),
            changed: Condvar::new(),
        }
    }

    fn try_reserve(&self, bytes: usize) -> bool {
        let mut state = self.state.lock();
        let Some(next) = state.bytes.checked_add(bytes) else {
            return false;
        };
        if next > self.limit {
            return false;
        }
        state.bytes = next;
        state.peak_bytes = state.peak_bytes.max(next);
        true
    }

    fn reserve_wait(&self, bytes: usize, first_error: &FirstError) -> AnnResult<()> {
        if bytes > self.limit {
            return Err(AnnError::log_index_config_error(
                "reorder_window_bytes".to_string(),
                format!(
                    "one reducer item requires {bytes} bytes, limit is {}",
                    self.limit
                ),
            ));
        }
        let mut state = self.state.lock();
        loop {
            first_error.check()?;
            if let Some(next) = state.bytes.checked_add(bytes)
                && next <= self.limit
            {
                state.bytes = next;
                state.peak_bytes = state.peak_bytes.max(next);
                return Ok(());
            }
            self.changed.wait(&mut state);
        }
    }

    fn release(&self, bytes: usize) {
        let mut state = self.state.lock();
        debug_assert!(state.bytes >= bytes);
        state.bytes -= bytes;
        self.changed.notify_all();
    }

    fn current(&self) -> usize {
        self.state.lock().bytes
    }

    fn peak(&self) -> usize {
        self.state.lock().peak_bytes
    }

    fn wake_all(&self) {
        let _state = self.state.lock();
        self.changed.notify_all();
    }
}

enum ReducerPayload {
    Direct(Vec<PendingEdge>),
    Spill(Vec<OnlineSpillSegment>),
}

#[derive(Clone, Copy)]
enum ReducerCharge {
    Direct,
    Reorder,
}

struct ReducerItem {
    shard_id: usize,
    ingress_seq: u64,
    accounted_bytes: usize,
    charge: ReducerCharge,
    payload: ReducerPayload,
}

#[derive(Default)]
struct ReducerMailboxState {
    queue: VecDeque<ReducerItem>,
    closed: bool,
}

struct ReducerMailbox {
    state: Mutex<ReducerMailboxState>,
    available: Condvar,
}

impl ReducerMailbox {
    fn new() -> Self {
        Self {
            state: Mutex::new(ReducerMailboxState::default()),
            available: Condvar::new(),
        }
    }

    fn enqueue(&self, item: ReducerItem) -> AnnResult<()> {
        let mut state = self.state.lock();
        if state.closed {
            return Err(AnnError::log_index_error(
                "online suffix reducer mailbox is closed".to_string(),
            ));
        }
        state.queue.push_back(item);
        self.available.notify_one();
        Ok(())
    }

    fn receive(&self) -> Option<ReducerItem> {
        let mut state = self.state.lock();
        loop {
            if let Some(item) = state.queue.pop_front() {
                return Some(item);
            }
            if state.closed {
                return None;
            }
            self.available.wait(&mut state);
        }
    }

    fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        self.available.notify_all();
    }
}

#[derive(Default)]
struct ShardOrderState {
    next_expected_seq: u64,
    pending: BTreeMap<u64, ReducerItem>,
}

#[derive(Default)]
struct ReducerWorkerStats {
    insert: HashPruneInsertProfile,
    backfill: HashPruneBackfillProfile,
    spill_segments_consumed: u64,
    state_records_updated: u64,
    delta_buffer_flushes: u64,
    #[cfg(test)]
    applied_sequences: BTreeMap<usize, Vec<u64>>,
}

struct OnlineSuffixInner {
    config: OnlineSuffixReduceConfig,
    backend: OnlineReservoirBackend,
    fallback: Mutex<OnlineFallbackWriter>,
    fallback_registry: OnlineFallbackRegistry,
    mailboxes: Vec<Arc<ReducerMailbox>>,
    sequencers: Vec<AtomicU64>,
    ingress_locks: Vec<Mutex<()>>,
    direct_budget: ByteBudget,
    reorder_budget: ByteBudget,
    first_error: FirstError,
    ingress_closed: AtomicBool,
    stats: Mutex<OnlineSuffixReduceStats>,
    first_candidate_accepted_at: Mutex<Option<Instant>>,
    gpu_complete_at: Mutex<Option<Instant>>,
}

enum OnlineReservoirBackend {
    DirectFirst(Arc<DiskReservoirStore>),
    WritebackCache {
        store: Arc<DiskReservoirStore>,
        cache: Arc<ShardedReservoirWritebackCache>,
    },
    LogAssisted(Arc<LogAssistedReservoir>),
}

impl OnlineSuffixInner {
    fn shard_count(&self) -> usize {
        self.sequencers.len()
    }

    fn shard_for_source(&self, source: usize) -> AnnResult<usize> {
        if source < self.config.source_start || source >= self.config.source_end {
            return Err(AnnError::log_index_error(format!(
                "online suffix source {source} is outside [{}, {})",
                self.config.source_start, self.config.source_end
            )));
        }
        Ok((source - self.config.source_start) / self.config.shard_points)
    }

    fn owner_for_shard(&self, shard_id: usize) -> usize {
        shard_id % self.mailboxes.len()
    }

    fn accept_shard_edges(&self, shard_id: usize, edges: Vec<PendingEdge>) -> AnnResult<()> {
        let _ingress = self.ingress_locks[shard_id].lock();
        self.first_error.check()?;
        if self.ingress_closed.load(Ordering::Acquire) {
            return Err(AnnError::log_index_error(
                "online suffix ingress is closed".to_string(),
            ));
        }
        if edges.is_empty() {
            return Ok(());
        }
        let ingress_seq = self.sequencers[shard_id].fetch_add(1, Ordering::AcqRel);
        let direct_bytes = direct_item_bytes(&edges);
        if self.direct_budget.try_reserve(direct_bytes) {
            let candidate_count = edges.len() as u64;
            let item = ReducerItem {
                shard_id,
                ingress_seq,
                accounted_bytes: direct_bytes,
                charge: ReducerCharge::Direct,
                payload: ReducerPayload::Direct(edges),
            };
            if let Err(error) = self.mailboxes[self.owner_for_shard(shard_id)].enqueue(item) {
                self.direct_budget.release(direct_bytes);
                return Err(error);
            }
            let mut stats = self.stats.lock();
            stats.direct_candidates += candidate_count;
            stats.direct_batches += 1;
        } else {
            let candidate_count = edges.len();
            let segments = self.fallback.lock().append(shard_id, ingress_seq, &edges)?;
            let spill_bytes = candidate_count.saturating_mul(RUN_RECORD_BYTES);
            let accounted_bytes = spill_item_bytes(&segments);
            self.reorder_budget
                .reserve_wait(accounted_bytes, &self.first_error)?;
            let item = ReducerItem {
                shard_id,
                ingress_seq,
                accounted_bytes,
                charge: ReducerCharge::Reorder,
                payload: ReducerPayload::Spill(segments),
            };
            if let Err(error) = self.mailboxes[self.owner_for_shard(shard_id)].enqueue(item) {
                self.reorder_budget.release(accounted_bytes);
                return Err(error);
            }
            let mut stats = self.stats.lock();
            stats.spill_fallback_candidates += candidate_count as u64;
            stats.spill_fallback_bytes += spill_bytes as u64;
        }
        let mut first = self.first_candidate_accepted_at.lock();
        if first.is_none() {
            *first = Some(Instant::now());
        }
        Ok(())
    }

    fn close_ingress(&self) {
        self.ingress_closed.store(true, Ordering::Release);
        for mailbox in &self.mailboxes {
            mailbox.close();
        }
        self.direct_budget.wake_all();
        self.reorder_budget.wake_all();
    }

    fn record_error(&self, error: impl std::fmt::Display) {
        self.first_error.record(error);
        self.close_ingress();
    }
}

pub(crate) struct OnlineSuffixPipeline {
    inner: Arc<OnlineSuffixInner>,
    worker_handles: Mutex<Option<Vec<JoinHandle<AnnResult<ReducerWorkerStats>>>>>,
    finished: AtomicBool,
}

struct OnlineSuffixArtifactManifest {
    root: PathBuf,
    fallback_root: PathBuf,
    state_paths: Vec<PathBuf>,
    fallback_paths: Vec<PathBuf>,
}

impl OnlineSuffixArtifactManifest {
    fn remove_all(self) -> AnnResult<()> {
        for path in self.fallback_paths.iter().chain(self.state_paths.iter()) {
            remove_file_if_present(path)?;
        }
        remove_dir_if_present(&self.fallback_root)?;
        remove_dir_if_present(&self.root)
    }
}

pub(crate) struct OnlineSuffixState {
    backend: OnlineSuffixStateBackend,
    source_start: usize,
    source_end: usize,
    artifacts: OnlineSuffixArtifactManifest,
    backfill_policy: HashPruneBackfillPolicy,
    stats: OnlineSuffixReduceStats,
    #[cfg(test)]
    applied_sequences: BTreeMap<usize, Vec<u64>>,
}

enum OnlineSuffixStateBackend {
    Disk {
        store: Arc<DiskReservoirStore>,
        writeback_cache: Option<Arc<ShardedReservoirWritebackCache>>,
    },
    LogAssisted(Arc<LogAssistedReservoir>),
}

impl OnlineSuffixState {
    pub(crate) fn stats(&self) -> &OnlineSuffixReduceStats {
        &self.stats
    }

    pub(crate) fn record_bytes(&self) -> usize {
        match &self.backend {
            OnlineSuffixStateBackend::Disk { store, .. } => store.record_bytes(),
            OnlineSuffixStateBackend::LogAssisted(backend) => backend.record_bytes(),
        }
    }

    pub(crate) fn emit_rows<F>(
        &self,
        window_sources: usize,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        on_node_done: &mut F,
    ) -> AnnResult<HashPruneBackfillProfile>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        let mut profile = HashPruneBackfillProfile::default();
        self.for_each_logical_source_window(window_sources, |source, mut reservoir| {
            let point = source as usize;
            let neighbors = if let (Some(collector), Some(finalizer)) =
                (diagnostic_collector, diagnostic_finalizer.as_deref_mut())
            {
                let row = reservoir.drain_finalized_slots_with_backfill(
                    point,
                    self.backfill_policy,
                    &mut profile,
                )?;
                finalizer.observe_row(point, collector, &row)?;
                row.slots.into_iter().map(|slot| slot.target).collect()
            } else {
                reservoir.copy_neighbors_with_backfill_and_clear(
                    point,
                    self.backfill_policy,
                    &mut profile,
                )?
            };
            on_node_done(source, neighbors)
        })?;
        Ok(profile)
    }

    fn for_each_logical_source_window<F>(
        &self,
        window_sources: usize,
        mut callback: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, HashPruneReservoir) -> AnnResult<()>,
    {
        let OnlineSuffixStateBackend::Disk {
            store,
            writeback_cache,
        } = &self.backend
        else {
            let OnlineSuffixStateBackend::LogAssisted(backend) = &self.backend else {
                unreachable!()
            };
            return backend.for_each_source(callback);
        };
        let Some(cache) = writeback_cache else {
            return store.for_each_source_window(window_sources, callback);
        };
        if window_sources == 0 {
            return Err(AnnError::log_index_config_error(
                "window_sources".to_string(),
                "online suffix source window must be nonzero".to_string(),
            ));
        }
        for window_start in (self.source_start..self.source_end).step_by(window_sources) {
            let window_end = window_start
                .saturating_add(window_sources)
                .min(self.source_end);
            let sources = (window_start..window_end)
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let mut reservoirs = BTreeMap::new();
            let mut disk_sources = Vec::new();
            for &source in &sources {
                if let Some(reservoir) = cache.clone_cached_reservoir(source)? {
                    reservoirs.insert(source, reservoir);
                } else {
                    disk_sources.push(source);
                }
            }
            for loaded in store.load_sources_grouped(&disk_sources)? {
                reservoirs.insert(loaded.source, loaded.reservoir);
            }
            for source in sources {
                let reservoir = reservoirs.remove(&source).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "online suffix finalizer did not resolve source {source}"
                    ))
                })?;
                callback(source, reservoir)?;
            }
        }
        Ok(())
    }

    pub(crate) fn cleanup(self) -> AnnResult<()> {
        let Self {
            backend,
            source_start: _,
            source_end: _,
            artifacts,
            backfill_policy: _,
            stats: _,
            #[cfg(test)]
                applied_sequences: _,
        } = self;
        match backend {
            OnlineSuffixStateBackend::Disk {
                store,
                writeback_cache,
            } => {
                drop(writeback_cache);
                let store = Arc::try_unwrap(store).map_err(|_| {
                    AnnError::log_index_error(
                        "online suffix state still has live store references".to_string(),
                    )
                })?;
                store.close()?;
            }
            OnlineSuffixStateBackend::LogAssisted(backend) => {
                let backend = Arc::try_unwrap(backend).map_err(|_| {
                    AnnError::log_index_error(
                        "online suffix state still has live log-assisted references".to_string(),
                    )
                })?;
                backend.cleanup()?;
            }
        }
        artifacts.remove_all()
    }

    #[cfg(test)]
    fn collect_rows_for_test(&self) -> AnnResult<Vec<Vec<u32>>> {
        let mut rows = Vec::new();
        self.for_each_logical_source_window(1, |_source, reservoir| {
            rows.push(reservoir.slots().iter().map(|slot| slot.target).collect());
            Ok(())
        })?;
        Ok(rows)
    }

    #[cfg(test)]
    fn state_snapshots_for_test(
        &self,
    ) -> AnnResult<Vec<super::hash_prune::HashPruneReservoirState>> {
        let mut states = Vec::new();
        self.for_each_logical_source_window(1, |_source, reservoir| {
            states.push(reservoir.state_snapshot());
            Ok(())
        })?;
        Ok(states)
    }

    #[cfg(test)]
    fn applied_sequences_for_test(&self, shard_id: usize) -> Vec<u64> {
        self.applied_sequences
            .get(&shard_id)
            .cloned()
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn writeback_cache_partition_count_for_test(&self) -> usize {
        match &self.backend {
            OnlineSuffixStateBackend::Disk {
                writeback_cache: Some(cache),
                ..
            } => cache.partition_count(),
            _ => 0,
        }
    }
}

impl OnlineSuffixPipeline {
    pub(crate) fn start(config: OnlineSuffixReduceConfig) -> AnnResult<Self> {
        validate_online_suffix_config(&config)?;
        if config.mode.uses_log_assisted() {
            Self::start_with_optional_store(config, None)
        } else {
            let store = Arc::new(DiskReservoirStore::create(config.store.clone())?);
            Self::start_with_optional_store(config, Some(store))
        }
    }

    #[cfg(test)]
    fn start_with_failing_store_for_test(config: OnlineSuffixReduceConfig) -> AnnResult<Self> {
        validate_online_suffix_config(&config)?;
        let store = Arc::new(
            DiskReservoirStore::create_with_failing_first_write_for_test(config.store.clone())?,
        );
        Self::start_with_store(config, store)
    }

    #[cfg(test)]
    fn start_with_failing_read_store_for_test(config: OnlineSuffixReduceConfig) -> AnnResult<Self> {
        validate_online_suffix_config(&config)?;
        let store = Arc::new(DiskReservoirStore::create_with_failing_first_read_for_test(
            config.store.clone(),
        )?);
        Self::start_with_store(config, store)
    }

    #[cfg(test)]
    fn start_with_failing_fallback_for_test(config: OnlineSuffixReduceConfig) -> AnnResult<Self> {
        let pipeline = Self::start(config)?;
        pipeline.inner.fallback.lock().fail_next_append_for_test();
        Ok(pipeline)
    }

    fn start_with_store(
        config: OnlineSuffixReduceConfig,
        store: Arc<DiskReservoirStore>,
    ) -> AnnResult<Self> {
        Self::start_with_optional_store(config, Some(store))
    }

    fn start_with_optional_store(
        config: OnlineSuffixReduceConfig,
        store: Option<Arc<DiskReservoirStore>>,
    ) -> AnnResult<Self> {
        let fallback_root = config.store.root.join("fallback");
        let fallback = if config.mode.uses_log_assisted() {
            OnlineFallbackWriter::create_range_local(
                &fallback_root,
                config.store.io,
                config.spill_buffer_bytes,
                config.source_start,
                config.range_points,
            )?
        } else {
            OnlineFallbackWriter::create(
                &fallback_root,
                config.store.io,
                config.spill_buffer_bytes,
            )?
        };
        let fallback_registry = fallback.registry();
        let shard_count = (config.source_end - config.source_start).div_ceil(config.shard_points);
        let worker_count = config.reducer_workers.max(1).min(shard_count.max(1));
        let backend = match config.mode {
            PiPNNOnlineSuffixReduceMode::DirectFirst => {
                OnlineReservoirBackend::DirectFirst(store.ok_or_else(|| {
                    AnnError::log_index_error(
                        "direct-first backend was started without a disk store".to_string(),
                    )
                })?)
            }
            PiPNNOnlineSuffixReduceMode::WritebackCache => {
                let store = store.ok_or_else(|| {
                    AnnError::log_index_error(
                        "write-back cache backend was started without a disk store".to_string(),
                    )
                })?;
                let entry_bytes = estimated_writeback_cache_entry_bytes(config.l_max)?;
                let dynamic_entries = config.state_cache_bytes / entry_bytes;
                let cache = ShardedReservoirWritebackCache::new(
                    ShardedReservoirWritebackCacheConfig {
                        source_start: u32::try_from(config.source_start)?,
                        source_end: u32::try_from(config.source_end)?,
                        shard_points: config.shard_points,
                        pinned_prefix_points: u32::try_from(config.pinned_prefix_points)?,
                        dynamic_entry_capacity: dynamic_entries,
                        read_window_bytes: config.read_window_bytes,
                        l_max: config.l_max,
                        backfill_policy: config.backfill_policy,
                    },
                    Arc::clone(&store),
                )?;
                OnlineReservoirBackend::WritebackCache {
                    store,
                    cache: Arc::new(cache),
                }
            }
            PiPNNOnlineSuffixReduceMode::LogAssisted => {
                let entry_bytes = estimated_writeback_cache_entry_bytes(config.l_max)?;
                let cache_capacity_entries = config.state_cache_bytes / entry_bytes;
                let compaction_worker_bytes =
                    config.compaction_bytes / config.compaction_workers.max(1);
                let promotion_limit_per_compaction =
                    (compaction_worker_bytes / entry_bytes).min(cache_capacity_entries);
                OnlineReservoirBackend::LogAssisted(Arc::new(LogAssistedReservoir::create(
                    LogAssistedReservoirConfig {
                        root: config.store.root.join("log-assisted"),
                        source_start: config.source_start,
                        source_end: config.source_end,
                        shard_points: config.shard_points,
                        range_points: config.range_points,
                        l_max: config.l_max,
                        delta_budget_bytes: config.delta_budget_bytes,
                        delta_buffer_bytes: config.delta_buffer_bytes,
                        range_trigger_bytes: config.range_trigger_bytes,
                        max_runs_per_range: config.max_runs_per_range,
                        compaction_workers: config.compaction_workers,
                        drain_compaction_workers: config.drain_compaction_workers,
                        cache_capacity_entries,
                        cache_hot_updates: config.cache_hot_updates,
                        promotion_limit_per_compaction,
                        policy_key: config.store.policy_key,
                        evidence_policy: config.evidence_policy,
                        backfill_policy: config.backfill_policy,
                        diagnostic_collector: config.diagnostic_collector.clone(),
                    },
                )?))
            }
            PiPNNOnlineSuffixReduceMode::Off => {
                return Err(AnnError::log_index_config_error(
                    "online_suffix_reduce".to_string(),
                    "online suffix pipeline cannot run in off mode".to_string(),
                ));
            }
        };
        let mailboxes = (0..worker_count)
            .map(|_| Arc::new(ReducerMailbox::new()))
            .collect();
        let inner = Arc::new(OnlineSuffixInner {
            direct_budget: ByteBudget::new(config.direct_queue_bytes),
            reorder_budget: ByteBudget::new(config.reorder_window_bytes),
            sequencers: (0..shard_count).map(|_| AtomicU64::new(0)).collect(),
            ingress_locks: (0..shard_count).map(|_| Mutex::new(())).collect(),
            config,
            backend,
            fallback: Mutex::new(fallback),
            fallback_registry,
            mailboxes,
            first_error: FirstError::default(),
            ingress_closed: AtomicBool::new(false),
            stats: Mutex::new(OnlineSuffixReduceStats::default()),
            first_candidate_accepted_at: Mutex::new(None),
            gpu_complete_at: Mutex::new(None),
        });
        let handles = (0..worker_count)
            .map(|owner_id| {
                let inner = Arc::clone(&inner);
                thread::Builder::new()
                    .name(format!("hashprune-online-reducer-{owner_id}"))
                    .spawn(move || {
                        let result = reducer_worker(&inner, owner_id);
                        if let Err(error) = &result {
                            inner.record_error(error);
                        }
                        result
                    })
                    .map_err(AnnError::from)
            })
            .collect::<AnnResult<Vec<_>>>()?;
        Ok(Self {
            inner,
            worker_handles: Mutex::new(Some(handles)),
            finished: AtomicBool::new(false),
        })
    }

    pub(crate) fn mark_gpu_complete(&self) {
        let now = Instant::now();
        let mut completed = self.inner.gpu_complete_at.lock();
        if completed.is_none() {
            *completed = Some(now);
            let backend_bytes = match &self.inner.backend {
                OnlineReservoirBackend::DirectFirst(store)
                | OnlineReservoirBackend::WritebackCache { store, .. } => {
                    let usage = store.memory_usage();
                    usage
                        .page_cache_bytes
                        .saturating_add(usage.writeback_bytes)
                        .saturating_add(usage.decode_bytes)
                }
                OnlineReservoirBackend::LogAssisted(backend) => {
                    if let Err(error) = backend.mark_gpu_complete() {
                        self.inner.record_error(error);
                    }
                    usize::try_from(backend.uncompacted_bytes()).unwrap_or(usize::MAX)
                }
            };
            self.inner.stats.lock().backlog_bytes_at_last_gpu = self
                .inner
                .direct_budget
                .current()
                .saturating_add(self.inner.reorder_budget.current())
                .saturating_add(backend_bytes);
        }
    }

    pub(crate) fn finish(self) -> AnnResult<OnlineSuffixState> {
        self.inner.ingress_closed.store(true, Ordering::Release);
        if let Err(error) = self.inner.fallback.lock().seal_all() {
            self.inner.record_error(error);
        }
        for mailbox in &self.inner.mailboxes {
            mailbox.close();
        }
        let worker_stats = self.join_workers();
        let first_error = self.inner.first_error.check();
        let worker_stats = worker_stats?;
        first_error?;
        let (disk_store, writeback_cache, log_backend) = match &self.inner.backend {
            OnlineReservoirBackend::DirectFirst(store) => {
                store.flush_all()?;
                (Some(Arc::clone(store)), None, None)
            }
            OnlineReservoirBackend::WritebackCache { store, cache } => {
                cache.flush_unpinned()?;
                store.flush_all()?;
                (Some(Arc::clone(store)), Some(Arc::clone(cache)), None)
            }
            OnlineReservoirBackend::LogAssisted(backend) => {
                backend.finish()?;
                (None, None, Some(Arc::clone(backend)))
            }
        };

        let drained_at = Instant::now();
        let mut stats = self.inner.stats.lock().clone();
        #[cfg(test)]
        let applied_sequences = {
            let mut merged = BTreeMap::new();
            for worker in &worker_stats {
                for (&shard_id, sequences) in &worker.applied_sequences {
                    merged
                        .entry(shard_id)
                        .or_insert_with(Vec::new)
                        .extend(sequences.iter().copied());
                }
            }
            merged
        };
        for worker in worker_stats {
            stats.evidence.merge(worker.insert.evidence);
            stats.backfill.merge(worker.backfill);
            stats.spill_segments_consumed += worker.spill_segments_consumed;
            stats.state_records_updated += worker.state_records_updated;
            stats.delta_buffer_flushes += worker.delta_buffer_flushes;
        }
        if let Some(store) = &disk_store {
            let store_stats = store.io_stats();
            stats.state_records_loaded = store_stats.records_loaded;
            stats.state_page_cache_hits = store_stats.page_cache_hits;
            stats.state_page_cache_misses = store_stats.page_cache_misses;
            stats.state_pages_read = store_stats.pages_read;
            stats.state_pages_written = store_stats.pages_written;
            stats.state_slots_reused = store_stats.slots_reused;
            stats.state_alloc_empty_page = store_stats.alloc_empty_page;
            stats.state_alloc_resident_partial_page = store_stats.alloc_resident_partial_page;
            stats.state_alloc_cold_partial_page = store_stats.alloc_cold_partial_page;
        }
        if let Some(cache) = &writeback_cache {
            let cache_stats = cache.stats();
            stats.state_cache_partitions = cache.partition_count();
            stats.state_cache_capacity_entries = cache_stats.capacity_entries;
            stats.state_cache_peak_entries = cache_stats.peak_entries;
            stats.state_cache_hits = cache_stats.hits;
            stats.state_cache_misses = cache_stats.misses;
            stats.state_cache_clean_evictions = cache_stats.clean_evictions;
            stats.state_cache_dirty_evictions = cache_stats.dirty_evictions;
            stats.state_cache_coalesced_updates = cache_stats.coalesced_updates;
            stats.state_read_windows = cache_stats.read_windows;
            stats.state_read_pages_requested = cache_stats.read_pages_requested;
            stats.state_read_pages_coalesced = cache_stats.read_pages_coalesced;
            stats.state_writeback_cohorts = cache_stats.writeback_cohorts;
            stats.state_writeback_records = cache_stats.writeback_records;
            stats.state_cache_updates_per_commit = cache_stats.updates_per_commit();
            stats.pinned_prefix_points = self.inner.config.pinned_prefix_points;
        }
        if let Some(backend) = &log_backend {
            let log = backend.stats();
            stats.cold_batch_commits = log.cold_batch_commits;
            stats.cold_batch_records = log.cold_batch_records;
            stats.delta_records = log.delta_records;
            stats.delta_runs = log.delta_runs;
            stats.delta_bytes = log.delta_bytes_written;
            stats.borrowed_spill_records = log.borrowed_spill_records;
            stats.borrowed_spill_runs = log.borrowed_spill_runs;
            stats.borrowed_spill_bytes = log.borrowed_spill_bytes;
            stats.delta_buffer_flushes = log.delta_buffer_flushes;
            stats.delta_uncompacted_bytes = log.current_uncompacted_bytes;
            stats.delta_uncompacted_peak_bytes = log.peak_uncompacted_bytes;
            stats.delta_backpressure_count = log.delta_backpressure_count;
            stats.delta_backpressure_wall = log.delta_backpressure_wall;
            stats.checkpoint_records = log.checkpoint_records;
            stats.checkpoint_runs = log.checkpoint_runs;
            stats.checkpoint_bytes = log.checkpoint_bytes_written;
            stats.range_compactions = log.compactions;
            stats.delta_run_pressure_crossings = log.run_pressure_crossings;
            stats.delta_run_pressure_compactions = log.run_pressure_compactions;
            stats.range_pre_gpu_materializations = log.pre_gpu_materializations;
            stats.log_shards_finalized = log.shard_finalizations;
            stats.range_ingress_drained_compactions = log.ingress_drained_compactions;
            stats.delta_peak_active_runs_per_range = log.peak_active_delta_runs_per_range;
            stats.range_compactions_inflight = log.compactions_inflight;
            stats.range_compactions_peak_inflight = log.peak_compactions_inflight;
            stats.post_gpu_range_compactions_peak_inflight = log.post_gpu_peak_compactions_inflight;
            stats.range_rows_compacted = log.rows_compacted;
            stats.range_rows_changed = log.rows_changed;
            stats.range_compaction_input_bytes = log.compaction_input_bytes;
            stats.range_compaction_output_bytes = log.compaction_output_bytes;
            stats.range_compaction_read_wall = log.compaction_read_wall;
            stats.range_compaction_merge_wall = log.compaction_merge_wall;
            stats.range_compaction_write_wall = log.compaction_write_wall;
            stats.range_compaction_hidden_wall = log.compaction_hidden_wall;
            stats.range_manifest_switches = log.manifest_switches;
            stats.cold_miss_base_reads = log.cold_miss_base_reads;
            stats.log_base_reads = log.base_reads;
            stats.log_base_writes = log.base_writes;
            stats.post_gpu_uncompacted_bytes = log.post_gpu_uncompacted_bytes;
            stats.post_gpu_compaction_wall = log.post_gpu_compaction_wall;
            stats.log_cache_promotions = log.cache_promotions;
            stats.log_cache_promotion_skipped_post_cut = log.cache_promotion_skipped_post_cut;
            stats.log_cache_hits = log.cache_hits;
            stats.log_cache_dirty_checkpoints = log.cache_dirty_checkpoints;
            stats.evidence.merge(log.insert_profile.evidence);
            stats.backfill.merge(log.backfill_profile);
        }
        if stats.state_records_updated != 0
            && let Some(store) = &disk_store
        {
            stats.state_write_amplification = stats.state_pages_written as f64
                * store.page_bytes() as f64
                / (stats.state_records_updated as f64 * store.record_bytes() as f64);
        }
        stats.direct_queue_peak_bytes = self.inner.direct_budget.peak();
        if let Some(start) = *self.inner.first_candidate_accepted_at.lock() {
            stats.online_reduce_wall = drained_at.saturating_duration_since(start);
        }
        if let Some(gpu_complete) = *self.inner.gpu_complete_at.lock() {
            stats.post_gpu_drain_wall = drained_at.saturating_duration_since(gpu_complete);
        }
        stats.online_reduce_hidden_wall = stats
            .online_reduce_wall
            .saturating_sub(stats.post_gpu_drain_wall);

        self.finished.store(true, Ordering::Release);
        let artifacts = OnlineSuffixArtifactManifest {
            root: self.inner.config.store.root.clone(),
            fallback_root: self.inner.config.store.root.join("fallback"),
            state_paths: disk_store
                .as_ref()
                .map_or_else(Vec::new, |store| store.artifact_paths()),
            fallback_paths: self.inner.fallback_registry.artifact_paths(),
        };
        let backend = match (disk_store, log_backend) {
            (Some(store), None) => OnlineSuffixStateBackend::Disk {
                store,
                writeback_cache,
            },
            (None, Some(backend)) => OnlineSuffixStateBackend::LogAssisted(backend),
            _ => {
                return Err(AnnError::log_index_error(
                    "online suffix final state resolved multiple or no backends".to_string(),
                ));
            }
        };
        Ok(OnlineSuffixState {
            backend,
            source_start: self.inner.config.source_start,
            source_end: self.inner.config.source_end,
            artifacts,
            backfill_policy: self.inner.config.backfill_policy,
            stats,
            #[cfg(test)]
            applied_sequences,
        })
    }

    fn join_workers(&self) -> AnnResult<Vec<ReducerWorkerStats>> {
        let handles = self.worker_handles.lock().take().unwrap_or_default();
        let mut stats = Vec::with_capacity(handles.len());
        let mut first_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(worker)) => stats.push(worker),
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(_) => {
                    if first_error.is_none() {
                        first_error = Some(AnnError::log_index_error(
                            "online suffix reducer worker panicked".to_string(),
                        ));
                    }
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(stats)
    }

    #[cfg(test)]
    fn enqueue_out_of_order_for_test(
        &self,
        shard_id: usize,
        later_direct: Vec<PendingEdge>,
        earlier_spill: Vec<PendingEdge>,
    ) -> AnnResult<()> {
        if shard_id >= self.inner.shard_count() {
            return Err(AnnError::log_index_error(
                "test shard is outside online suffix range".to_string(),
            ));
        }
        self.inner.sequencers[shard_id].store(2, Ordering::Release);
        let segments = self
            .inner
            .fallback
            .lock()
            .append(shard_id, 0, &earlier_spill)?;
        let spill_accounted = spill_item_bytes(&segments);
        self.inner
            .reorder_budget
            .reserve_wait(spill_accounted, &self.inner.first_error)?;
        let direct_accounted = direct_item_bytes(&later_direct);
        if !self.inner.direct_budget.try_reserve(direct_accounted) {
            self.inner.reorder_budget.release(spill_accounted);
            return Err(AnnError::log_index_error(
                "test direct item does not fit direct queue".to_string(),
            ));
        }
        let mailbox = &self.inner.mailboxes[self.inner.owner_for_shard(shard_id)];
        if let Err(error) = mailbox.enqueue(ReducerItem {
            shard_id,
            ingress_seq: 1,
            accounted_bytes: direct_accounted,
            charge: ReducerCharge::Direct,
            payload: ReducerPayload::Direct(later_direct),
        }) {
            self.inner.direct_budget.release(direct_accounted);
            self.inner.reorder_budget.release(spill_accounted);
            return Err(error);
        }
        if let Err(error) = mailbox.enqueue(ReducerItem {
            shard_id,
            ingress_seq: 0,
            accounted_bytes: spill_accounted,
            charge: ReducerCharge::Reorder,
            payload: ReducerPayload::Spill(segments),
        }) {
            self.inner.reorder_budget.release(spill_accounted);
            return Err(error);
        }
        let mut stats = self.inner.stats.lock();
        stats.direct_candidates += 1;
        stats.direct_batches += 1;
        stats.spill_fallback_candidates += earlier_spill.len() as u64;
        stats.spill_fallback_bytes += (earlier_spill.len() * RUN_RECORD_BYTES) as u64;
        Ok(())
    }

    #[cfg(test)]
    fn drop_ticket_for_test(&self, shard_id: usize) {
        assert!(shard_id < self.inner.shard_count());
        self.inner.sequencers[shard_id].fetch_add(1, Ordering::AcqRel);
    }
}

fn remove_file_if_present(path: &Path) -> AnnResult<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn remove_dir_if_present(path: &Path) -> AnnResult<()> {
    match fs::remove_dir(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

impl PendingEdgeSink for OnlineSuffixPipeline {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        self.inner.config.evidence_policy
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        self.inner.config.backfill_policy
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<PendingEdgeFlushProfile> {
        if edges.is_empty() {
            return Ok(PendingEdgeFlushProfile::default());
        }
        self.inner.first_error.check()?;
        let edge_count = edges.len();
        let sort_start = Instant::now();
        let mut evidence =
            prepare_pending_edges_for_sink(edges, self.inner.config.evidence_policy)?;
        let sort_wall = sort_start.elapsed();
        let removed = evidence.duplicate_records_removed;
        evidence.spill_records_avoided = evidence.spill_records_avoided.saturating_add(removed);
        evidence.spill_bytes_avoided = evidence
            .spill_bytes_avoided
            .saturating_add(removed.saturating_mul(RUN_RECORD_BYTES));
        for edge in edges.iter() {
            self.inner.shard_for_source(edge.p)?;
        }

        let insert_start = Instant::now();
        let normalized = std::mem::take(edges);
        let mut current_shard = None;
        let mut current_edges = Vec::new();
        for edge in normalized {
            let shard_id = self.inner.shard_for_source(edge.p)?;
            if current_shard != Some(shard_id) {
                if let Some(flush_shard) = current_shard {
                    self.inner
                        .accept_shard_edges(flush_shard, std::mem::take(&mut current_edges))?;
                }
                current_shard = Some(shard_id);
            }
            current_edges.push(edge);
        }
        if let Some(shard_id) = current_shard {
            self.inner.accept_shard_edges(shard_id, current_edges)?;
        }
        let insert_wall = insert_start.elapsed();
        Ok(PendingEdgeFlushProfile {
            edges: edge_count,
            sort_wall,
            insert_wall,
            insert_body_wall: insert_wall,
            flushes: 1,
            evidence,
            ..PendingEdgeFlushProfile::default()
        })
    }
}

impl Drop for OnlineSuffixPipeline {
    fn drop(&mut self) {
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        self.inner.close_ingress();
        let handles = self.worker_handles.get_mut().take().unwrap_or_default();
        for handle in handles {
            let _ = handle.join();
        }
    }
}

fn validate_online_suffix_config(config: &OnlineSuffixReduceConfig) -> AnnResult<()> {
    if config.source_start >= config.source_end || config.source_end > u32::MAX as usize {
        return Err(AnnError::log_index_config_error(
            "source_range".to_string(),
            "online suffix reducer requires a non-empty u32 source range".to_string(),
        ));
    }
    if config.shard_points == 0 || config.l_max == 0 {
        return Err(AnnError::log_index_config_error(
            "online_suffix".to_string(),
            "shard_points and l_max must be nonzero".to_string(),
        ));
    }
    match config.mode {
        PiPNNOnlineSuffixReduceMode::Off => {
            return Err(AnnError::log_index_config_error(
                "online_suffix_reduce".to_string(),
                "online suffix pipeline cannot run in off mode".to_string(),
            ));
        }
        PiPNNOnlineSuffixReduceMode::DirectFirst => {}
        PiPNNOnlineSuffixReduceMode::LogAssisted => {}
        PiPNNOnlineSuffixReduceMode::WritebackCache => {
            let entry_bytes = estimated_writeback_cache_entry_bytes(config.l_max)?;
            if config.state_cache_bytes < entry_bytes {
                return Err(AnnError::log_index_config_error(
                    "online_suffix_state_cache_bytes".to_string(),
                    format!("must hold at least one {entry_bytes}-byte cache entry"),
                ));
            }
            let layout = super::disk_reservoir_store::DiskReservoirLayout::new(
                config.l_max,
                config.store.io.effective_alignment(),
            )?;
            if config.read_window_bytes < layout.page_bytes {
                return Err(AnnError::log_index_config_error(
                    "online_suffix_read_window_bytes".to_string(),
                    format!(
                        "must hold at least one {}-byte state page",
                        layout.page_bytes
                    ),
                ));
            }
            if config.pinned_prefix_points > config.source_end - config.source_start {
                return Err(AnnError::log_index_config_error(
                    "online_suffix_pinned_prefix_points".to_string(),
                    "pinned prefix exceeds the online source range".to_string(),
                ));
            }
        }
    }
    if config.spill_buffer_bytes < RUN_RECORD_BYTES {
        return Err(AnnError::log_index_config_error(
            "spill_buffer_bytes".to_string(),
            format!("must hold at least one {RUN_RECORD_BYTES}-byte spill record"),
        ));
    }
    let minimum_reorder = REDUCER_ITEM_ACCOUNTING_OVERHEAD
        + size_of::<ReducerItem>()
        + size_of::<OnlineSpillSegment>();
    if config.reorder_window_bytes < minimum_reorder {
        return Err(AnnError::log_index_config_error(
            "reorder_window_bytes".to_string(),
            format!("must hold at least one reducer ticket ({minimum_reorder} bytes)"),
        ));
    }
    if !config.evidence_policy.is_off() && !config.backfill_policy.is_off() {
        return Err(AnnError::log_index_config_error(
            "hashprune_policy".to_string(),
            "online suffix evidence and backfill policies cannot both be enabled".to_string(),
        ));
    }
    if config.evidence_policy.enforces() {
        return Err(AnnError::log_index_config_error(
            "hashprune_evidence_mode".to_string(),
            "online suffix reduction does not support enforce mode".to_string(),
        ));
    }
    if config.store.source_start != config.source_start
        || config.store.source_end != config.source_end
        || config.store.shard_points != config.shard_points
        || config.store.l_max != config.l_max
        || config.store.backfill_policy != config.backfill_policy
    {
        return Err(AnnError::log_index_config_error(
            "store".to_string(),
            "online suffix store geometry or backfill policy does not match the pipeline"
                .to_string(),
        ));
    }
    Ok(())
}

fn direct_item_bytes(edges: &Vec<PendingEdge>) -> usize {
    REDUCER_ITEM_ACCOUNTING_OVERHEAD
        .saturating_add(size_of::<ReducerItem>())
        .saturating_add(edges.capacity().saturating_mul(size_of::<PendingEdge>()))
}

fn spill_item_bytes(segments: &Vec<OnlineSpillSegment>) -> usize {
    REDUCER_ITEM_ACCOUNTING_OVERHEAD
        .saturating_add(size_of::<ReducerItem>())
        .saturating_add(
            segments
                .capacity()
                .saturating_mul(size_of::<OnlineSpillSegment>()),
        )
}

fn reducer_worker(
    inner: &Arc<OnlineSuffixInner>,
    owner_id: usize,
) -> AnnResult<ReducerWorkerStats> {
    let mut stats = ReducerWorkerStats::default();
    let mut shard_states: BTreeMap<usize, ShardOrderState> = BTreeMap::new();
    while let Some(item) = inner.mailboxes[owner_id].receive() {
        inner.first_error.check()?;
        if inner.owner_for_shard(item.shard_id) != owner_id {
            return Err(AnnError::log_index_error(
                "reducer item reached the wrong shard owner".to_string(),
            ));
        }
        let state = shard_states.entry(item.shard_id).or_default();
        if item.ingress_seq < state.next_expected_seq
            || state.pending.insert(item.ingress_seq, item).is_some()
        {
            return Err(AnnError::log_index_error(
                "duplicate or stale online suffix ingress sequence".to_string(),
            ));
        }
        loop {
            let Some(item) = state.pending.remove(&state.next_expected_seq) else {
                break;
            };
            #[cfg(test)]
            let (shard_id, ingress_seq) = (item.shard_id, item.ingress_seq);
            let charge = item.charge;
            let accounted_bytes = item.accounted_bytes;
            let result = apply_reducer_item(inner, item, &mut stats);
            match charge {
                ReducerCharge::Direct => inner.direct_budget.release(accounted_bytes),
                ReducerCharge::Reorder => inner.reorder_budget.release(accounted_bytes),
            }
            result?;
            #[cfg(test)]
            stats
                .applied_sequences
                .entry(shard_id)
                .or_default()
                .push(ingress_seq);
            state.next_expected_seq += 1;
        }
    }

    let mut drained_shards = Vec::new();
    for shard_id in 0..inner.shard_count() {
        if inner.owner_for_shard(shard_id) != owner_id {
            continue;
        }
        let expected_tickets = inner.sequencers[shard_id].load(Ordering::Acquire);
        let state = shard_states.entry(shard_id).or_default();
        if state.next_expected_seq != expected_tickets || !state.pending.is_empty() {
            return Err(AnnError::log_index_error(format!(
                "missing ingress sequence for shard {shard_id}: next={} issued={expected_tickets}",
                state.next_expected_seq
            )));
        }
        drained_shards.push(shard_id);
    }
    if let OnlineReservoirBackend::LogAssisted(backend) = &inner.backend {
        backend.finalize_ingress_shards(&drained_shards)?;
    }
    Ok(stats)
}

fn apply_reducer_item(
    inner: &OnlineSuffixInner,
    item: ReducerItem,
    stats: &mut ReducerWorkerStats,
) -> AnnResult<()> {
    let shard_id = item.shard_id;
    match item.payload {
        ReducerPayload::Direct(edges) => apply_direct_edges(inner, shard_id, &edges, stats),
        ReducerPayload::Spill(segments) => {
            for segment in segments {
                if segment.shard_id != shard_id || segment.ingress_seq != item.ingress_seq {
                    return Err(AnnError::log_index_error(
                        "online spill segment ticket does not match reducer item".to_string(),
                    ));
                }
                if let OnlineReservoirBackend::LogAssisted(backend) = &inner.backend {
                    let mut profiles = LogAssistedApplyProfiles::default();
                    if backend.adopt_spill_segment(
                        shard_id,
                        &inner.fallback_registry,
                        &segment,
                        &mut profiles,
                    )? {
                        stats.insert.merge(profiles.insert_profile);
                        stats.backfill.merge(profiles.backfill_profile);
                        stats.delta_buffer_flushes = stats
                            .delta_buffer_flushes
                            .saturating_add(profiles.buffer_flushes);
                        stats.state_records_updated = stats
                            .state_records_updated
                            .saturating_add(u64::from(segment.source_count));
                        stats.spill_segments_consumed += 1;
                        continue;
                    }
                }
                let batch_capacity = usize::try_from(segment.desc.record_count)
                    .unwrap_or(ONLINE_SPILL_SOURCE_BATCH_EDGES)
                    .min(ONLINE_SPILL_SOURCE_BATCH_EDGES);
                let batches_across_sources =
                    matches!(&inner.backend, OnlineReservoirBackend::LogAssisted(_));
                let mut replay_batch: Vec<PendingEdge> = Vec::with_capacity(batch_capacity);
                inner.fallback_registry.replay_segment(&segment, |edge| {
                    if !replay_batch.is_empty()
                        && (replay_batch.len() == ONLINE_SPILL_SOURCE_BATCH_EDGES
                            || (!batches_across_sources && replay_batch[0].p != edge.p))
                    {
                        apply_replay_edges(inner, shard_id, &replay_batch, stats)?;
                        replay_batch.clear();
                    }
                    replay_batch.push(edge);
                    Ok(())
                })?;
                if !replay_batch.is_empty() {
                    apply_replay_edges(inner, shard_id, &replay_batch, stats)?;
                }
                if let OnlineReservoirBackend::DirectFirst(store) = &inner.backend {
                    store.flush_update_batch()?;
                }
                inner.fallback_registry.mark_consumed(&segment)?;
                stats.spill_segments_consumed += 1;
            }
            Ok(())
        }
    }
}

fn apply_replay_edges(
    inner: &OnlineSuffixInner,
    shard_id: usize,
    edges: &[PendingEdge],
    stats: &mut ReducerWorkerStats,
) -> AnnResult<()> {
    if let OnlineReservoirBackend::LogAssisted(backend) = &inner.backend {
        apply_log_assisted_edges(backend, shard_id, edges, stats)
    } else {
        apply_source_edges(inner, shard_id, edges, stats)
    }
}

fn apply_direct_edges(
    inner: &OnlineSuffixInner,
    shard_id: usize,
    edges: &[PendingEdge],
    stats: &mut ReducerWorkerStats,
) -> AnnResult<()> {
    if let OnlineReservoirBackend::LogAssisted(backend) = &inner.backend {
        return apply_log_assisted_edges(backend, shard_id, edges, stats);
    }
    if let OnlineReservoirBackend::WritebackCache { cache, .. } = &inner.backend {
        let mut groups = Vec::new();
        let mut start = 0usize;
        while start < edges.len() {
            let source = edges[start].p;
            let mut end = start + 1;
            while end < edges.len() && edges[end].p == source {
                end += 1;
            }
            groups.push((u32::try_from(source)?, start, end));
            start = end;
        }
        let sources = groups
            .iter()
            .map(|(source, _, _)| *source)
            .collect::<Vec<_>>();
        let mut group_index = 0usize;
        cache.update_sources(shard_id, &sources, |source, reservoir| {
            let (expected_source, start, end) = groups[group_index];
            if source != expected_source {
                return Err(AnnError::log_index_error(
                    "write-back cache changed source update order".to_string(),
                ));
            }
            for edge in edges[start..end].iter().cloned() {
                apply_edge(
                    reservoir,
                    edge,
                    inner.config.evidence_policy,
                    inner.config.backfill_policy,
                    inner.config.diagnostic_collector.as_deref(),
                    &mut stats.insert,
                    &mut stats.backfill,
                )?;
            }
            group_index += 1;
            Ok(())
        })?;
        stats.state_records_updated = stats
            .state_records_updated
            .saturating_add(u64::try_from(groups.len())?);
        return Ok(());
    }

    let mut start = 0usize;
    while start < edges.len() {
        let source = edges[start].p;
        let mut end = start + 1;
        while end < edges.len() && edges[end].p == source {
            end += 1;
        }
        apply_source_edges(inner, shard_id, &edges[start..end], stats)?;
        start = end;
    }
    let OnlineReservoirBackend::DirectFirst(store) = &inner.backend else {
        unreachable!("non-direct backend returned to direct flush path")
    };
    store.flush_update_batch()
}

fn apply_source_edges(
    inner: &OnlineSuffixInner,
    shard_id: usize,
    edges: &[PendingEdge],
    stats: &mut ReducerWorkerStats,
) -> AnnResult<()> {
    let source = edges
        .first()
        .ok_or_else(|| AnnError::log_index_error("empty online source batch".to_string()))?
        .p;
    debug_assert!(edges.iter().all(|edge| edge.p == source));
    if let OnlineReservoirBackend::LogAssisted(backend) = &inner.backend {
        return apply_log_assisted_edges(backend, shard_id, edges, stats);
    }
    let update = |reservoir: &mut HashPruneReservoir| {
        for edge in edges.iter().cloned() {
            apply_edge(
                reservoir,
                edge,
                inner.config.evidence_policy,
                inner.config.backfill_policy,
                inner.config.diagnostic_collector.as_deref(),
                &mut stats.insert,
                &mut stats.backfill,
            )?;
        }
        Ok(())
    };
    match &inner.backend {
        OnlineReservoirBackend::DirectFirst(store) => {
            store.update_source_deferred(u32::try_from(source)?, update)?;
        }
        OnlineReservoirBackend::WritebackCache { cache, .. } => {
            let mut update = Some(update);
            cache.update_sources(shard_id, &[u32::try_from(source)?], |_source, reservoir| {
                update.take().expect("source update runs once")(reservoir)
            })?;
        }
        OnlineReservoirBackend::LogAssisted(_) => unreachable!(),
    }
    stats.state_records_updated += 1;
    Ok(())
}

fn apply_log_assisted_edges(
    backend: &LogAssistedReservoir,
    shard_id: usize,
    edges: &[PendingEdge],
    stats: &mut ReducerWorkerStats,
) -> AnnResult<()> {
    let mut profiles = LogAssistedApplyProfiles::default();
    backend.apply_edges(shard_id, edges, &mut profiles)?;
    stats.insert.merge(profiles.insert_profile);
    stats.backfill.merge(profiles.backfill_profile);
    stats.delta_buffer_flushes = stats
        .delta_buffer_flushes
        .saturating_add(profiles.buffer_flushes);
    let source_groups = edges
        .iter()
        .enumerate()
        .filter(|(index, edge)| *index == 0 || edges[*index - 1].p != edge.p)
        .count();
    stats.state_records_updated = stats
        .state_records_updated
        .saturating_add(u64::try_from(source_groups)?);
    Ok(())
}

pub(crate) fn apply_edge(
    reservoir: &mut HashPruneReservoir,
    edge: PendingEdge,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    insert_profile: &mut HashPruneInsertProfile,
    backfill_profile: &mut HashPruneBackfillProfile,
) -> AnnResult<()> {
    if let Some(collector) = diagnostic_collector {
        collector.record(edge.p, edge.c as usize, edge.hash, edge.dist)?;
    }
    if !backfill_policy.is_off() {
        return reservoir.insert_backfill_prevalidated_with_policy(
            edge.p,
            edge.c,
            edge.hash,
            edge.dist,
            backfill_policy,
            backfill_profile,
        );
    }
    if evidence_policy.is_off() {
        reservoir.insert_profiled(edge.c, edge.hash, edge.dist, insert_profile);
        return Ok(());
    }
    reservoir.insert_with_policy_profiled(
        edge.p,
        edge.c,
        edge.hash,
        edge.dist,
        edge.meta,
        evidence_policy,
        insert_profile,
    )
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::mem::size_of;

    use tempfile::tempdir;

    use super::*;
    use crate::algorithm::pipnn::direct_io::{
        DirectIoConfig, direct_io_test_write_calls, reset_direct_io_test_counters,
    };
    use crate::algorithm::pipnn::disk_reservoir_store::DiskReservoirStoreConfig;
    use crate::algorithm::pipnn::hash_prune_backfill::HashPruneBackfillPolicy;
    use crate::algorithm::pipnn::hash_prune_evidence::{
        HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneMeta,
    };
    use crate::algorithm::pipnn::leaf_build::{PendingEdge, PendingEdgeSink};
    use crate::algorithm::pipnn::params::{
        PiPNNHashPruneBackfillMode, PiPNNOnlineSuffixReduceMode, PiPNNParams,
    };
    use crate::algorithm::pipnn::spill_hashprune::{ExternalHashPruneReducer, SpillEdgeSink};

    fn test_edges(source_start: usize, count: usize) -> Vec<PendingEdge> {
        (source_start..source_start + count)
            .map(|source| PendingEdge {
                p: source,
                c: source as u32 + 100,
                hash: source as u16,
                dist: source as f32,
                meta: HashPruneMeta::empty(),
            })
            .collect()
    }

    fn pipeline_config(
        root: &Path,
        source_end: usize,
        l_max: usize,
        direct_queue_bytes: usize,
    ) -> OnlineSuffixReduceConfig {
        let store_root = root.join("state");
        OnlineSuffixReduceConfig {
            source_start: 0,
            source_end,
            shard_points: 4,
            l_max,
            direct_queue_bytes,
            spill_buffer_bytes: 4096,
            reorder_window_bytes: 1024 * 1024,
            reducer_workers: 2,
            mode: PiPNNOnlineSuffixReduceMode::DirectFirst,
            state_cache_bytes: 0,
            read_window_bytes: 4096,
            pinned_prefix_points: 0,
            delta_budget_bytes: 1024 * 1024,
            delta_buffer_bytes: 4096,
            compaction_bytes: 64 * 1024 * 1024,
            range_points: source_end.max(1),
            range_trigger_bytes: 4096,
            max_runs_per_range: 4,
            compaction_workers: 1,
            drain_compaction_workers: 1,
            cache_hot_updates: 2,
            store: DiskReservoirStoreConfig {
                root: store_root,
                source_start: 0,
                source_end,
                shard_points: 4,
                l_max,
                page_cache_bytes: 4096 * 2,
                writeback_bytes: 4096,
                decode_bytes: 4096,
                writeback_workers: 1,
                overprovision_ratio: 2.0,
                io: DirectIoConfig::disabled(),
                policy_key: [0; 32],
                backfill_policy: HashPruneBackfillPolicy::off(),
            },
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        }
    }

    fn expected_rows(batches: &[Vec<PendingEdge>], rows: usize, l_max: usize) -> Vec<Vec<u32>> {
        let mut reservoirs = (0..rows)
            .map(|_| super::super::hash_prune::HashPruneReservoir::new(l_max))
            .collect::<Vec<_>>();
        for edge in batches.iter().flatten() {
            reservoirs[edge.p].insert(edge.c, edge.hash, edge.dist);
        }
        reservoirs
            .into_iter()
            .map(|reservoir| reservoir.into_neighbors())
            .collect()
    }

    type ReservoirStateKey = (u16, u16, Vec<(u32, u16, u32, u16)>);

    #[derive(Clone, Copy, Debug)]
    enum OnlineRoute {
        DirectOnly,
        SpillOnly,
        Mixed,
    }

    fn state_key(
        state: &super::super::hash_prune::HashPruneReservoirState,
    ) -> AnnResult<ReservoirStateKey> {
        let backup_count = state.backup_count;
        let primary_count =
            u16::try_from(state.slots.len().saturating_sub(usize::from(backup_count)))
                .map_err(|_| AnnError::log_index_error("primary count exceeds u16".to_string()))?;
        let slots = state
            .slots
            .iter()
            .map(|slot| {
                Ok((
                    slot.target,
                    slot.hash,
                    slot.dist.to_bits(),
                    slot.meta.to_state_flags()?,
                ))
            })
            .collect::<AnnResult<Vec<_>>>()?;
        Ok((primary_count, backup_count, slots))
    }

    fn state_keys(
        states: &[super::super::hash_prune::HashPruneReservoirState],
    ) -> AnnResult<Vec<ReservoirStateKey>> {
        states.iter().map(state_key).collect()
    }

    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_candidate_stream(
        seed: u64,
        num_points: usize,
        candidate_count: usize,
    ) -> Vec<PendingEdge> {
        let mut random = seed ^ 0xd1b5_4a32_d192_ed03;
        let mut source_ordinals = vec![0usize; num_points];
        (0..candidate_count)
            .map(|index| {
                let source = next_random(&mut random) as usize % num_points;
                let ordinal = source_ordinals[source];
                source_ordinals[source] += 1;
                let target = num_points
                    .saturating_add(source.saturating_mul(candidate_count.saturating_add(1)))
                    .saturating_add(ordinal);
                let class = HashPruneEvidenceClass::ALL
                    [next_random(&mut random) as usize % HashPruneEvidenceClass::ALL.len()];
                PendingEdge {
                    p: source,
                    c: target as u32,
                    hash: (next_random(&mut random) % 16) as u16,
                    dist: 1.0
                        + (next_random(&mut random) % 1_000_000) as f32 / 1024.0
                        + index as f32 / 1_000_000.0,
                    meta: HashPruneMeta::with_evidence(class),
                }
            })
            .collect()
    }

    fn random_batch_fragmentation(candidates: &[PendingEdge], seed: u64) -> Vec<Vec<PendingEdge>> {
        let mut random = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut batches = Vec::new();
        let mut start = 0usize;
        let mut batch_index = 0usize;
        while start < candidates.len() {
            let requested = if batch_index % 2 == 0 {
                1
            } else {
                17 + next_random(&mut random) as usize % 24
            };
            let end = start.saturating_add(requested).min(candidates.len());
            batches.push(candidates[start..end].to_vec());
            start = end;
            batch_index += 1;
        }
        batches
    }

    fn reduce_candidates_in_memory(
        batches: &[Vec<PendingEdge>],
        num_points: usize,
        l_max: usize,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Vec<ReservoirStateKey>> {
        let mut reservoirs = (0..num_points)
            .map(|_| HashPruneReservoir::new(l_max))
            .collect::<Vec<_>>();
        let mut insert_profile = HashPruneInsertProfile::default();
        let mut backfill_profile = HashPruneBackfillProfile::default();
        for batch in batches {
            let mut normalized = batch.clone();
            prepare_pending_edges_for_sink(&mut normalized, evidence_policy)?;
            for edge in normalized {
                let source = edge.p;
                apply_edge(
                    &mut reservoirs[source],
                    edge,
                    evidence_policy,
                    backfill_policy,
                    None,
                    &mut insert_profile,
                    &mut backfill_profile,
                )?;
            }
        }
        reservoirs
            .iter()
            .map(|reservoir| state_key(&reservoir.state_snapshot()))
            .collect()
    }

    fn reduce_candidates_external(
        batches: &[Vec<PendingEdge>],
        num_points: usize,
        l_max: usize,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Vec<ReservoirStateKey>> {
        let dir = tempdir()?;
        let sink = SpillEdgeSink::create_sharded_with_config_and_policies(
            &dir.path().join("candidate.run"),
            1,
            16,
            DirectIoConfig::disabled(),
            evidence_policy,
            backfill_policy,
        )?;
        for batch in batches {
            sink.flush_pending_edges(&mut batch.clone())?;
        }
        let artifacts = sink.finish()?;
        let states = ExternalHashPruneReducer::reduce_spill_to_state_snapshots_for_test(
            num_points,
            l_max,
            16,
            &artifacts,
            evidence_policy,
            backfill_policy,
        )?;
        state_keys(&states)
    }

    fn reduce_candidates_online(
        batches: &[Vec<PendingEdge>],
        num_points: usize,
        l_max: usize,
        route: OnlineRoute,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Vec<ReservoirStateKey>> {
        reduce_candidates_online_with_mode(
            batches,
            num_points,
            l_max,
            route,
            evidence_policy,
            backfill_policy,
            PiPNNOnlineSuffixReduceMode::DirectFirst,
            0,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_candidates_online_with_mode(
        batches: &[Vec<PendingEdge>],
        num_points: usize,
        l_max: usize,
        route: OnlineRoute,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        mode: PiPNNOnlineSuffixReduceMode,
        pinned_prefix_points: usize,
        dynamic_entry_capacity: usize,
    ) -> AnnResult<Vec<ReservoirStateKey>> {
        let dir = tempdir()?;
        let mut config = pipeline_config(dir.path(), num_points, l_max, 1024 * 1024);
        config.shard_points = num_points;
        config.reducer_workers = 1;
        config.store.shard_points = num_points;
        config.store.page_cache_bytes = 32 * 1024;
        config.store.writeback_bytes = 32 * 1024;
        config.store.decode_bytes = 32 * 1024;
        config.store.overprovision_ratio = 3.0;
        config.evidence_policy = evidence_policy;
        config.backfill_policy = backfill_policy;
        config.store.backfill_policy = backfill_policy;
        config.mode = mode;
        if mode.uses_writeback_cache() {
            config.state_cache_bytes = estimated_writeback_cache_entry_bytes(l_max)?
                .saturating_mul(dynamic_entry_capacity.max(1));
            config.read_window_bytes = 4096;
            config.pinned_prefix_points = pinned_prefix_points;
            config.store.decode_bytes = 0;
        }
        if mode.uses_log_assisted() {
            config.state_cache_bytes = estimated_writeback_cache_entry_bytes(l_max)?
                .saturating_mul(dynamic_entry_capacity);
            config.delta_budget_bytes = 1024 * 1024;
            config.delta_buffer_bytes = 4096;
            config.range_points = (num_points / 4).max(1);
            config.range_trigger_bytes = 4096;
            config.max_runs_per_range = 4;
            config.compaction_workers = 2;
            config.drain_compaction_workers = 2;
            config.cache_hot_updates = 2;
        }
        config.direct_queue_bytes = match route {
            OnlineRoute::DirectOnly => 1024 * 1024,
            OnlineRoute::SpillOnly => 0,
            OnlineRoute::Mixed => 256,
        };
        let pipeline = OnlineSuffixPipeline::start(config)?;
        for batch in batches {
            pipeline.flush_pending_edges(&mut batch.clone())?;
        }
        pipeline.mark_gpu_complete();
        let state = pipeline.finish()?;
        match route {
            OnlineRoute::DirectOnly => {
                assert_eq!(state.stats().spill_fallback_candidates, 0);
            }
            OnlineRoute::SpillOnly => {
                assert_eq!(state.stats().direct_candidates, 0);
                assert!(state.stats().spill_fallback_candidates > 0);
            }
            OnlineRoute::Mixed => {
                assert!(state.stats().direct_candidates > 0);
                assert!(state.stats().spill_fallback_candidates > 0);
            }
        }
        let keys = state_keys(&state.state_snapshots_for_test()?)?;
        state.cleanup()?;
        Ok(keys)
    }

    fn backfill_policy_for_mode(
        l_max: usize,
        mode: PiPNNHashPruneBackfillMode,
    ) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.l_max = l_max;
        params.hashprune_backfill_mode = mode;
        match mode {
            PiPNNHashPruneBackfillMode::Off => {}
            PiPNNHashPruneBackfillMode::DistanceGated
            | PiPNNHashPruneBackfillMode::DistanceGatedStratified
            | PiPNNHashPruneBackfillMode::DistanceGatedMultihit => {
                params.hashprune_backfill_budget = 4;
                params.hashprune_backfill_alpha = 1.1;
            }
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52 => {
                params.hashprune_backfill_budget = 8;
                params.hashprune_backfill_alpha = 1.1;
            }
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf => {
                params.hashprune_backfill_alpha = 1.1;
            }
            PiPNNHashPruneBackfillMode::RHalfHashSlot1
            | PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
            | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
            | PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => {}
        }
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    #[test]
    fn randomized_online_suffix_matches_memory_and_external_reducers() {
        const NUM_POINTS: usize = 64;
        const L_MAX: usize = 12;
        for seed in 0_u64..32 {
            let candidates = random_candidate_stream(seed, NUM_POINTS, 600);
            let batches = random_batch_fragmentation(&candidates, seed);
            let memory = reduce_candidates_in_memory(
                &batches,
                NUM_POINTS,
                L_MAX,
                HashPruneEvidencePolicy::off(),
                HashPruneBackfillPolicy::off(),
            )
            .unwrap();
            let external = reduce_candidates_external(
                &batches,
                NUM_POINTS,
                L_MAX,
                HashPruneEvidencePolicy::off(),
                HashPruneBackfillPolicy::off(),
            )
            .unwrap();
            assert_eq!(external, memory, "seed={seed} external");
            for route in [
                OnlineRoute::DirectOnly,
                OnlineRoute::SpillOnly,
                OnlineRoute::Mixed,
            ] {
                let configurations =
                    std::iter::once((PiPNNOnlineSuffixReduceMode::DirectFirst, 0, 0))
                        .chain(std::iter::once((
                            PiPNNOnlineSuffixReduceMode::LogAssisted,
                            0,
                            2,
                        )))
                        .chain([1, 2, 8].into_iter().flat_map(|capacity| {
                            [
                                (PiPNNOnlineSuffixReduceMode::WritebackCache, 0, capacity),
                                (
                                    PiPNNOnlineSuffixReduceMode::WritebackCache,
                                    NUM_POINTS / 4,
                                    capacity,
                                ),
                            ]
                        }));
                for (mode, pinned_prefix_points, dynamic_entry_capacity) in configurations {
                    let online = reduce_candidates_online_with_mode(
                        &batches,
                        NUM_POINTS,
                        L_MAX,
                        route,
                        HashPruneEvidencePolicy::off(),
                        HashPruneBackfillPolicy::off(),
                        mode,
                        pinned_prefix_points,
                        dynamic_entry_capacity,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "seed={seed} route={route:?} mode={mode:?} pinned={pinned_prefix_points} capacity={dynamic_entry_capacity}: {error}"
                        )
                    });
                    assert_eq!(
                        online, memory,
                        "seed={seed} route={route:?} mode={mode:?} pinned={pinned_prefix_points} capacity={dynamic_entry_capacity}"
                    );
                }
            }
        }
    }

    #[test]
    fn sharded_writeback_cache_matches_memory_across_pinned_boundary_and_routes() {
        const NUM_POINTS: usize = 12;
        const L_MAX: usize = 8;
        const SHARD_POINTS: usize = 4;
        const PINNED_PREFIX_POINTS: usize = 5;
        const DYNAMIC_ENTRIES: usize = 6;

        let candidates = random_candidate_stream(73, NUM_POINTS, 360);
        let batches = random_batch_fragmentation(&candidates, 73);
        let memory = reduce_candidates_in_memory(
            &batches,
            NUM_POINTS,
            L_MAX,
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
        .unwrap();

        for route in [OnlineRoute::DirectOnly, OnlineRoute::SpillOnly] {
            let dir = tempdir().unwrap();
            let mut config = pipeline_config(dir.path(), NUM_POINTS, L_MAX, 1024 * 1024);
            config.shard_points = SHARD_POINTS;
            config.reducer_workers = 3;
            config.store.shard_points = SHARD_POINTS;
            config.store.page_cache_bytes = 32 * 1024;
            config.store.writeback_bytes = 32 * 1024;
            config.store.decode_bytes = 0;
            config.store.overprovision_ratio = 3.0;
            config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
            config.state_cache_bytes =
                estimated_writeback_cache_entry_bytes(L_MAX).unwrap() * DYNAMIC_ENTRIES;
            config.read_window_bytes = 4096;
            config.pinned_prefix_points = PINNED_PREFIX_POINTS;
            config.direct_queue_bytes = match route {
                OnlineRoute::DirectOnly => 1024 * 1024,
                OnlineRoute::SpillOnly => 0,
                OnlineRoute::Mixed => unreachable!(),
            };

            let pipeline = OnlineSuffixPipeline::start(config).unwrap();
            for batch in &batches {
                pipeline.flush_pending_edges(&mut batch.clone()).unwrap();
            }
            pipeline.mark_gpu_complete();
            let state = pipeline.finish().unwrap();

            assert_eq!(state.writeback_cache_partition_count_for_test(), 3);
            assert_eq!(state.stats().state_cache_partitions, 3);
            assert_eq!(
                state.stats().state_cache_capacity_entries,
                PINNED_PREFIX_POINTS + DYNAMIC_ENTRIES
            );
            match route {
                OnlineRoute::DirectOnly => {
                    assert_eq!(state.stats().spill_fallback_candidates, 0);
                }
                OnlineRoute::SpillOnly => {
                    assert_eq!(state.stats().direct_candidates, 0);
                    assert!(state.stats().spill_fallback_candidates > 0);
                }
                OnlineRoute::Mixed => unreachable!(),
            }
            assert_eq!(
                state_keys(&state.state_snapshots_for_test().unwrap()).unwrap(),
                memory,
                "route={route:?}"
            );
            state.cleanup().unwrap();
        }
    }

    #[test]
    fn online_suffix_supported_policy_states_match_external_and_memory() {
        const NUM_POINTS: usize = 32;
        const L_MAX: usize = 12;
        let candidates = random_candidate_stream(41, NUM_POINTS, 240);
        let batches = random_batch_fragmentation(&candidates, 41);
        let cases = std::iter::once((
            "off",
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        ))
        .chain(std::iter::once((
            "report",
            HashPruneEvidencePolicy::report(),
            HashPruneBackfillPolicy::off(),
        )))
        .chain(
            [
                PiPNNHashPruneBackfillMode::DistanceGated,
                PiPNNHashPruneBackfillMode::DistanceGatedStratified,
                PiPNNHashPruneBackfillMode::DistanceGatedMultihit,
                PiPNNHashPruneBackfillMode::DistanceGatedTieredD52,
                PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf,
                PiPNNHashPruneBackfillMode::RHalfHashSlot1,
                PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
                PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
                PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
            ]
            .into_iter()
            .map(|mode| {
                (
                    mode.as_str(),
                    HashPruneEvidencePolicy::off(),
                    backfill_policy_for_mode(L_MAX, mode),
                )
            }),
        );

        for (name, evidence_policy, backfill_policy) in cases {
            let memory = reduce_candidates_in_memory(
                &batches,
                NUM_POINTS,
                L_MAX,
                evidence_policy,
                backfill_policy,
            )
            .unwrap();
            let external = reduce_candidates_external(
                &batches,
                NUM_POINTS,
                L_MAX,
                evidence_policy,
                backfill_policy,
            )
            .unwrap();
            assert_eq!(external, memory, "policy={name} external");
            for route in [
                OnlineRoute::DirectOnly,
                OnlineRoute::SpillOnly,
                OnlineRoute::Mixed,
            ] {
                for (mode, pinned_prefix_points, dynamic_entry_capacity) in [
                    (PiPNNOnlineSuffixReduceMode::DirectFirst, 0, 0),
                    (PiPNNOnlineSuffixReduceMode::WritebackCache, 0, 2),
                    (
                        PiPNNOnlineSuffixReduceMode::WritebackCache,
                        NUM_POINTS / 4,
                        2,
                    ),
                ] {
                    let online = reduce_candidates_online_with_mode(
                        &batches,
                        NUM_POINTS,
                        L_MAX,
                        route,
                        evidence_policy,
                        backfill_policy,
                        mode,
                        pinned_prefix_points,
                        dynamic_entry_capacity,
                    )
                    .unwrap();
                    assert_eq!(
                        online, memory,
                        "policy={name} route={route:?} mode={mode:?} pinned={pinned_prefix_points}"
                    );
                }
            }
        }
    }

    #[test]
    fn fallback_part_is_reclaimed_only_after_every_segment_is_consumed() {
        let dir = tempdir().unwrap();
        let mut writer = OnlineFallbackWriter::for_test(dir.path(), 64).unwrap();
        let first = writer.append(0, 0, &test_edges(0, 2)).unwrap();
        let second = writer.append(0, 1, &test_edges(2, 2)).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        writer.seal_all().unwrap();
        let part_path = writer.part_path(first[0].part_id).unwrap();
        assert!(part_path.exists());
        writer.mark_consumed(&first[0]).unwrap();
        assert!(part_path.exists());
        writer.mark_consumed(&second[0]).unwrap();
        assert!(!part_path.exists());
    }

    #[test]
    fn fallback_registry_rejects_duplicate_segment_consumption() {
        let dir = tempdir().unwrap();
        let mut writer = OnlineFallbackWriter::for_test(dir.path(), 64).unwrap();
        let first = writer.append(0, 0, &test_edges(0, 2)).unwrap();
        let second = writer.append(0, 1, &test_edges(2, 2)).unwrap();
        writer.seal_all().unwrap();
        let part_path = writer.part_path(first[0].part_id).unwrap();

        writer.mark_consumed(&first[0]).unwrap();
        assert!(writer.mark_consumed(&first[0]).is_err());
        assert!(part_path.exists());

        let mut replayed = Vec::new();
        writer
            .replay_segment(&second[0], |edge| {
                replayed.push(edge.p);
                Ok(())
            })
            .unwrap();
        assert_eq!(replayed, vec![2, 3]);
        writer.mark_consumed(&second[0]).unwrap();
        assert!(!part_path.exists());
    }

    #[test]
    fn log_assisted_fallback_segments_do_not_cross_range_boundaries() {
        let dir = tempdir().unwrap();
        let mut writer = OnlineFallbackWriter::for_test_range_local(dir.path(), 128, 0, 4).unwrap();
        let edges = vec![
            PendingEdge {
                p: 3,
                c: 30,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 4,
                c: 40,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 7,
                c: 70,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 8,
                c: 80,
                hash: 4,
                dist: 4.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        let segments = writer.append(0, 0, &edges).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!((segments[0].desc.src_min, segments[0].desc.src_max), (3, 3));
        assert_eq!((segments[1].desc.src_min, segments[1].desc.src_max), (4, 7));
        assert_eq!((segments[2].desc.src_min, segments[2].desc.src_max), (8, 8));
    }

    #[test]
    fn range_local_fallback_flushes_once_per_append_not_once_per_segment() {
        let dir = tempdir().unwrap();
        let mut writer = OnlineFallbackWriter::create_range_local(
            dir.path(),
            DirectIoConfig::enabled_with_alignment(4096),
            4096,
            0,
            4,
        )
        .unwrap();
        let edges = vec![
            PendingEdge {
                p: 3,
                c: 30,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 4,
                c: 40,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 8,
                c: 80,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        reset_direct_io_test_counters();
        let segments = writer.append(0, 0, &edges).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(direct_io_test_write_calls(), 1);

        for segment in segments {
            let mut replayed = Vec::new();
            writer
                .replay_segment(&segment, |edge| {
                    replayed.push((edge.p, edge.c));
                    Ok(())
                })
                .unwrap();
            assert_eq!(replayed.len(), 1);
        }
    }

    #[test]
    fn online_suffix_fault_truncated_fallback_segment_is_an_error() {
        let dir = tempdir().unwrap();
        let mut writer = OnlineFallbackWriter::for_test(dir.path(), 64).unwrap();
        let segment = writer.append(0, 0, &test_edges(0, 2)).unwrap().remove(0);
        writer.seal_all().unwrap();
        let path = writer.part_path(segment.part_id).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(segment.desc.byte_len - 1)
            .unwrap();
        assert!(writer.replay_segment(&segment, |_| Ok(())).is_err());
    }

    #[test]
    fn fallback_rotation_splits_only_between_sources() {
        let dir = tempdir().unwrap();
        let mut writer =
            OnlineFallbackWriter::for_test_with_part_limit(dir.path(), 64, 48).unwrap();
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 0,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 1,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 12,
                hash: 2,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 2,
                c: 13,
                hash: 3,
                dist: 4.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        let segments = writer.append(0, 0, &edges).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!((segments[0].desc.src_min, segments[0].desc.src_max), (0, 1));
        assert_eq!((segments[1].desc.src_min, segments[1].desc.src_max), (2, 2));
        assert_ne!(segments[0].part_id, segments[1].part_id);
    }

    #[test]
    fn direct_only_pipeline_matches_in_memory_reservoir() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 8, 4, 1024 * 1024);
        let batches = vec![
            vec![
                PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 3.0,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 5,
                    c: 50,
                    hash: 5,
                    dist: 1.0,
                    meta: HashPruneMeta::empty(),
                },
            ],
            vec![
                PendingEdge {
                    p: 0,
                    c: 11,
                    hash: 2,
                    dist: 2.0,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 3,
                    c: 30,
                    hash: 3,
                    dist: 1.0,
                    meta: HashPruneMeta::empty(),
                },
            ],
        ];
        let expected = expected_rows(&batches, 8, 4);
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        for mut batch in batches {
            pipeline.flush_pending_edges(&mut batch).unwrap();
            assert!(batch.is_empty());
        }
        pipeline.mark_gpu_complete();
        let state = pipeline.finish().unwrap();
        assert_eq!(state.collect_rows_for_test().unwrap(), expected);
        assert_eq!(state.stats().spill_fallback_candidates, 0);
    }

    #[test]
    fn direct_reducer_combines_same_page_source_updates() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 4, 4, 1024 * 1024);
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 20,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        pipeline.flush_pending_edges(&mut edges).unwrap();
        let state = pipeline.finish().unwrap();

        assert_eq!(state.stats().state_records_updated, 2);
        assert_eq!(state.stats().state_pages_written, 1);
    }

    #[test]
    fn writeback_cache_coalesces_same_source_across_ingress_items() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 16, 4, 1024 * 1024);
        config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        config.state_cache_bytes = 1024 * 1024;
        config.read_window_bytes = 4096;
        config.store.decode_bytes = 0;
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        for (target, hash, dist) in [(30, 1, 3.0), (31, 2, 2.0), (32, 3, 1.0)] {
            pipeline
                .flush_pending_edges(&mut vec![PendingEdge {
                    p: 3,
                    c: target,
                    hash,
                    dist,
                    meta: HashPruneMeta::empty(),
                }])
                .unwrap();
        }

        let state = pipeline.finish().unwrap();

        assert_eq!(state.stats().state_cache_misses, 1);
        assert_eq!(state.stats().state_cache_hits, 2);
        assert_eq!(state.stats().state_cache_coalesced_updates, 2);
        assert_eq!(state.stats().state_writeback_records, 1);
        assert_eq!(state.collect_rows_for_test().unwrap()[3], vec![30, 31, 32]);
    }

    #[test]
    fn writeback_cache_pipeline_builds_one_cache_partition_per_source_shard() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 8, 4, 1024 * 1024);
        config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        config.state_cache_bytes = estimated_writeback_cache_entry_bytes(4).unwrap() * 2;
        config.read_window_bytes = 4096;
        config.store.decode_bytes = 0;

        let state = OnlineSuffixPipeline::start(config)
            .unwrap()
            .finish()
            .unwrap();

        assert_eq!(state.writeback_cache_partition_count_for_test(), 2);
    }

    #[test]
    fn writeback_cache_pipeline_updates_pinned_and_evictable_sources_in_one_stream() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 8, 4, 1024 * 1024);
        config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        config.state_cache_bytes = 1024 * 1024;
        config.read_window_bytes = 4096;
        config.pinned_prefix_points = 2;
        config.store.decode_bytes = 0;
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 3,
                c: 13,
                hash: 3,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 1,
                dist: 0.5,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 4,
                c: 14,
                hash: 4,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        pipeline.flush_pending_edges(&mut edges).unwrap();

        let state = pipeline.finish().unwrap();
        let rows = state.collect_rows_for_test().unwrap();

        assert_eq!(rows[0], vec![11]);
        assert_eq!(rows[3], vec![13]);
        assert_eq!(rows[4], vec![14]);
        assert_eq!(state.stats().pinned_prefix_points, 2);
    }

    #[test]
    fn spill_ticket_prevents_later_direct_batch_from_overtaking() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 4, 2, 1024);
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        pipeline
            .enqueue_out_of_order_for_test(
                0,
                vec![PendingEdge {
                    p: 0,
                    c: 11,
                    hash: 1,
                    dist: 2.0,
                    meta: HashPruneMeta::empty(),
                }],
                vec![PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 1.0,
                    meta: HashPruneMeta::empty(),
                }],
            )
            .unwrap();
        let state = pipeline.finish().unwrap();
        assert_eq!(state.applied_sequences_for_test(0), vec![0, 1]);
        assert!(state.stats().spill_fallback_candidates > 0);
    }

    #[test]
    fn full_direct_queue_uses_spill_without_growing_candidate_memory() {
        let dir = tempdir().unwrap();
        let direct_queue_bytes = size_of::<PendingEdge>() * 2;
        let config = pipeline_config(dir.path(), 32, 4, direct_queue_bytes);
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let mut edges = test_edges(0, 16);
        pipeline.flush_pending_edges(&mut edges).unwrap();
        let state = pipeline.finish().unwrap();
        assert!(state.stats().direct_queue_peak_bytes <= direct_queue_bytes);
        assert!(state.stats().spill_fallback_bytes > 0);
    }

    #[test]
    fn spill_replay_updates_each_source_once_per_segment() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 4, 4, 0);
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 12,
                hash: 3,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 20,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 21,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        pipeline.flush_pending_edges(&mut edges).unwrap();
        let state = pipeline.finish().unwrap();
        assert_eq!(state.stats().spill_fallback_candidates, 5);
        assert_eq!(state.stats().state_records_updated, 2);
        assert_eq!(state.stats().state_pages_written, 1);
    }

    #[test]
    fn log_assisted_spill_replay_batches_cold_sources_in_one_range() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 4, 0);
        config.mode = PiPNNOnlineSuffixReduceMode::LogAssisted;
        config.range_points = 4;
        config.state_cache_bytes = 0;
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 20,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 21,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        pipeline.flush_pending_edges(&mut edges).unwrap();
        pipeline.mark_gpu_complete();
        let state = pipeline.finish().unwrap();
        assert_eq!(state.stats().spill_fallback_candidates, 4);
        assert_eq!(state.stats().state_records_updated, 2);
        assert_eq!(state.stats().cold_batch_commits, 1);
        assert_eq!(state.stats().cold_batch_records, 4);
        assert_eq!(state.stats().borrowed_spill_records, 4);
        assert_eq!(
            state.stats().borrowed_spill_bytes,
            4 * RUN_RECORD_BYTES as u64
        );
        assert_eq!(state.stats().borrowed_spill_runs, 1);
        assert_eq!(state.stats().delta_bytes, 0);
        assert_eq!(state.stats().log_shards_finalized, 1);
        assert_eq!(state.stats().range_ingress_drained_compactions, 1);
        assert!(
            fs::read_dir(dir.path().join("state/fallback"))
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| entry.path().extension().is_none_or(|ext| ext != "run"))
        );
    }

    #[test]
    fn borrowed_spill_turns_promoted_range_log_only_without_losing_order() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 8, 1024 * 1024);
        config.mode = PiPNNOnlineSuffixReduceMode::LogAssisted;
        config.range_points = 4;
        config.state_cache_bytes = 1024 * 1024;
        config.cache_hot_updates = 2;
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let backend = match &pipeline.inner.backend {
            OnlineReservoirBackend::LogAssisted(backend) => Arc::clone(backend),
            _ => panic!("expected log-assisted backend"),
        };

        let make_edge = |target: u32| PendingEdge {
            p: 0,
            c: target,
            hash: target as u16,
            dist: target as f32,
            meta: HashPruneMeta::empty(),
        };
        backend
            .apply_edges(
                0,
                &[make_edge(10), make_edge(11)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();
        backend.compact_range_for_test(0).unwrap();
        assert!(backend.cache_contains_for_test(0));
        backend
            .apply_edges(
                0,
                &[make_edge(12)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();

        let mut writer =
            OnlineFallbackWriter::for_test_range_local(&dir.path().join("borrowed"), 128, 0, 4)
                .unwrap();
        let segment = writer
            .append(0, 0, &[make_edge(13), make_edge(14)])
            .unwrap()
            .remove(0);
        let part_path = writer.part_path(segment.part_id).unwrap();
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
        backend
            .apply_edges(
                0,
                &[make_edge(15)],
                &mut LogAssistedApplyProfiles::default(),
            )
            .unwrap();

        pipeline.mark_gpu_complete();
        let state = pipeline.finish().unwrap();
        assert_eq!(
            state.collect_rows_for_test().unwrap()[0],
            vec![10, 11, 12, 13, 14, 15]
        );
        assert_eq!(state.stats().borrowed_spill_records, 2);
        assert_eq!(
            state.stats().log_cache_promotion_skipped_post_cut,
            0,
            "log-only compaction must not build candidates that are discarded later"
        );
        assert!(!part_path.exists());
        drop(backend);
    }

    #[test]
    fn online_suffix_fault_reducer_failure_wakes_blocked_ingress() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 2, 0);
        let state_root = config.store.root.clone();
        config.shard_points = 4;
        config.store.shard_points = 4;
        config.reorder_window_bytes = 512;
        let pipeline = OnlineSuffixPipeline::start_with_failing_store_for_test(config).unwrap();

        let producer_failed = std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|producer| {
                    let pipeline = &pipeline;
                    scope.spawn(move || {
                        for round in 0..64 {
                            let mut batch = vec![PendingEdge {
                                p: (producer + round) % 4,
                                c: (1000 + producer * 64 + round) as u32,
                                hash: round as u16,
                                dist: round as f32,
                                meta: HashPruneMeta::empty(),
                            }];
                            pipeline.flush_pending_edges(&mut batch)?;
                        }
                        Ok::<_, AnnError>(())
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .any(|handle| handle.join().unwrap().is_err())
        });

        assert!(producer_failed);
        let error = match pipeline.finish() {
            Ok(_) => panic!("injected page write must fail the reducer"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected page write failure"));
        assert!(state_root.exists());
    }

    #[test]
    fn writeback_cache_fault_write_failure_wakes_blocked_ingress() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 2, 0);
        let state_root = config.store.root.clone();
        config.shard_points = 4;
        config.store.shard_points = 4;
        config.reducer_workers = 1;
        config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        config.state_cache_bytes = estimated_writeback_cache_entry_bytes(2).unwrap();
        config.read_window_bytes = 4096;
        config.store.decode_bytes = 0;
        let pipeline = OnlineSuffixPipeline::start_with_failing_store_for_test(config).unwrap();

        let producer_failed = std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|producer| {
                    let pipeline = &pipeline;
                    scope.spawn(move || {
                        for round in 0..64 {
                            let mut batch = vec![PendingEdge {
                                p: (producer + round) % 4,
                                c: (2000 + producer * 64 + round) as u32,
                                hash: round as u16,
                                dist: round as f32,
                                meta: HashPruneMeta::empty(),
                            }];
                            pipeline.flush_pending_edges(&mut batch)?;
                        }
                        Ok::<_, AnnError>(())
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .any(|handle| handle.join().unwrap().is_err())
        });

        assert!(producer_failed);
        let error = match pipeline.finish() {
            Ok(_) => panic!("injected B1 page write must fail the reducer"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected page write failure"));
        assert!(state_root.exists());
    }

    #[test]
    fn online_suffix_fault_finish_rejects_missing_sequence_ticket() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 4, 2, 1024);
        let state_root = config.store.root.clone();
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        pipeline.drop_ticket_for_test(0);
        let error = match pipeline.finish() {
            Ok(_) => panic!("missing ticket must fail finalization"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("missing ingress sequence"));
        assert!(state_root.exists());
    }

    #[test]
    fn online_suffix_fault_page_read_failure_preserves_artifacts() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 4, 2, 1024);
        let state_root = config.store.root.clone();
        let pipeline =
            OnlineSuffixPipeline::start_with_failing_read_store_for_test(config).unwrap();
        let mut edges = vec![PendingEdge {
            p: 0,
            c: 10,
            hash: 1,
            dist: 1.0,
            meta: HashPruneMeta::empty(),
        }];
        pipeline.flush_pending_edges(&mut edges).unwrap();
        let error = match pipeline.finish() {
            Ok(_) => panic!("injected page read must fail the reducer"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected page read failure"));
        assert!(state_root.exists());
    }

    #[test]
    fn writeback_cache_fault_page_read_failure_preserves_artifacts() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 2, 1024);
        let state_root = config.store.root.clone();
        config.shard_points = 4;
        config.store.shard_points = 4;
        config.reducer_workers = 1;
        config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        config.state_cache_bytes = estimated_writeback_cache_entry_bytes(2).unwrap();
        config.read_window_bytes = 4096;
        config.store.decode_bytes = 0;
        let pipeline =
            OnlineSuffixPipeline::start_with_failing_read_store_for_test(config).unwrap();
        for source in [0, 1, 0] {
            let mut edges = vec![PendingEdge {
                p: source,
                c: 10 + source as u32,
                hash: source as u16,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            }];
            if pipeline.flush_pending_edges(&mut edges).is_err() {
                break;
            }
        }
        let error = match pipeline.finish() {
            Ok(_) => panic!("injected B1 page read must fail the reducer"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected page read failure"));
        assert!(state_root.exists());
    }

    #[test]
    fn writeback_cache_fault_finalizer_failure_preserves_artifacts() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 2, 1024);
        let state_root = config.store.root.clone();
        config.mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        config.state_cache_bytes = estimated_writeback_cache_entry_bytes(2).unwrap();
        config.read_window_bytes = 4096;
        config.pinned_prefix_points = 1;
        config.store.decode_bytes = 0;
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        let mut edges = vec![PendingEdge {
            p: 0,
            c: 10,
            hash: 1,
            dist: 1.0,
            meta: HashPruneMeta::empty(),
        }];
        pipeline.flush_pending_edges(&mut edges).unwrap();
        let state = pipeline.finish().unwrap();

        let error = state
            .emit_rows(2, None, None, &mut |_source, _neighbors| {
                Err(AnnError::log_index_error(
                    "injected finalizer failure".to_string(),
                ))
            })
            .unwrap_err();

        assert!(error.to_string().contains("injected finalizer failure"));
        assert!(state_root.exists());
    }

    #[test]
    fn online_suffix_fault_fallback_write_failure_preserves_artifacts() {
        let dir = tempdir().unwrap();
        let config = pipeline_config(dir.path(), 4, 2, 0);
        let state_root = config.store.root.clone();
        let pipeline = OnlineSuffixPipeline::start_with_failing_fallback_for_test(config).unwrap();
        let mut edges = vec![PendingEdge {
            p: 0,
            c: 10,
            hash: 1,
            dist: 1.0,
            meta: HashPruneMeta::empty(),
        }];
        let error = pipeline.flush_pending_edges(&mut edges).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected fallback write failure")
        );
        drop(pipeline);
        assert!(state_root.exists());
    }

    #[test]
    fn online_suffix_fault_reducer_insertion_failure_preserves_artifacts() {
        let dir = tempdir().unwrap();
        let mut config = pipeline_config(dir.path(), 4, 2, 1024);
        let state_root = config.store.root.clone();
        config.evidence_policy = HashPruneEvidencePolicy::report();
        let pipeline = OnlineSuffixPipeline::start(config).unwrap();
        for hash in [1_u16, 2] {
            let mut edges = vec![PendingEdge {
                p: 0,
                c: 10,
                hash,
                dist: f32::from(hash),
                meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            }];
            pipeline.flush_pending_edges(&mut edges).unwrap();
        }
        let error = match pipeline.finish() {
            Ok(_) => panic!("duplicate target with mismatched hashes must fail insertion"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("duplicate hash mismatch"));
        assert!(state_root.exists());
    }
}
