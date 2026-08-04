use std::collections::{BTreeSet, HashMap, VecDeque};
use std::mem::size_of;
use std::sync::Arc;

use parking_lot::Mutex;

use super::disk_reservoir_store::{
    DiskReservoirRecord, DiskReservoirStore, PackedLocation, ReservoirWriteback,
    ReservoirWritebackCompletion,
};
use super::hash_prune::{HashPruneReservoir, Slot};
use super::hash_prune_backfill::HashPruneBackfillPolicy;
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug)]
pub(crate) struct ReservoirWritebackCacheConfig {
    pub(crate) store_shard_id: usize,
    pub(crate) source_start: u32,
    pub(crate) source_end: u32,
    pub(crate) pinned_prefix_points: u32,
    pub(crate) entry_capacity: usize,
    pub(crate) read_window_bytes: usize,
    pub(crate) l_max: usize,
    pub(crate) backfill_policy: HashPruneBackfillPolicy,
}

#[derive(Clone, Debug)]
pub(crate) struct ShardedReservoirWritebackCacheConfig {
    pub(crate) source_start: u32,
    pub(crate) source_end: u32,
    pub(crate) shard_points: usize,
    pub(crate) pinned_prefix_points: u32,
    pub(crate) dynamic_entry_capacity: usize,
    pub(crate) read_window_bytes: usize,
    pub(crate) l_max: usize,
    pub(crate) backfill_policy: HashPruneBackfillPolicy,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ReservoirWritebackCacheStats {
    pub(crate) capacity_entries: usize,
    pub(crate) peak_entries: usize,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) clean_evictions: u64,
    pub(crate) dirty_evictions: u64,
    pub(crate) coalesced_updates: u64,
    pub(crate) read_windows: u64,
    pub(crate) read_pages_requested: u64,
    pub(crate) read_pages_coalesced: u64,
    pub(crate) writeback_cohorts: u64,
    pub(crate) writeback_records: u64,
    pub(crate) committed_updates: u64,
}

impl ReservoirWritebackCacheStats {
    pub(crate) fn updates_per_commit(&self) -> f64 {
        if self.writeback_records == 0 {
            0.0
        } else {
            self.committed_updates as f64 / self.writeback_records as f64
        }
    }
}

#[derive(Debug)]
struct CachedReservoir {
    reservoir: HashPruneReservoir,
    committed_location: PackedLocation,
    committed_generation: u32,
    latest_generation: u32,
    dirty_generation: Option<u32>,
    inflight_generation: Option<u32>,
    pinned: bool,
    referenced: bool,
    evict_after_clean: bool,
    clock_token: u64,
}

impl CachedReservoir {
    fn loaded(
        reservoir: HashPruneReservoir,
        location: PackedLocation,
        generation: u32,
        pinned: bool,
        clock_token: u64,
    ) -> Self {
        Self {
            reservoir,
            committed_location: location,
            committed_generation: generation,
            latest_generation: generation,
            dirty_generation: None,
            inflight_generation: None,
            pinned,
            referenced: true,
            evict_after_clean: false,
            clock_token,
        }
    }

    fn is_clean_and_idle(&self) -> bool {
        self.dirty_generation.is_none() && self.inflight_generation.is_none()
    }

    fn is_dirty_and_idle(&self) -> bool {
        self.dirty_generation.is_some() && self.inflight_generation.is_none()
    }
}

pub(crate) struct ReservoirWritebackCache {
    config: ReservoirWritebackCacheConfig,
    entries: HashMap<u32, CachedReservoir>,
    clock: VecDeque<(u32, u64)>,
    next_clock_token: u64,
    store: Arc<DiskReservoirStore>,
    dirty_idle_unpinned: usize,
    cache_pressure_active: bool,
    stats: ReservoirWritebackCacheStats,
}

pub(crate) struct ShardedReservoirWritebackCache {
    source_start: u32,
    source_end: u32,
    shard_points: usize,
    partitions: Vec<Mutex<ReservoirWritebackCache>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachePartitionLayout {
    source_start: u32,
    source_end: u32,
    pinned_prefix_points: u32,
    dynamic_entries: usize,
}

fn partition_cache_layouts(
    source_start: u32,
    source_end: u32,
    shard_points: usize,
    pinned_prefix_points: u32,
    dynamic_entries: usize,
) -> AnnResult<Vec<CachePartitionLayout>> {
    let source_count = source_end.saturating_sub(source_start);
    if source_count == 0
        || shard_points == 0
        || pinned_prefix_points > source_count
        || usize::try_from(source_count)?.div_ceil(shard_points) == 0
    {
        return Err(AnnError::log_index_config_error(
            "online_suffix_state_cache".to_string(),
            "invalid sharded cache source range, shard width, or pinned prefix".to_string(),
        ));
    }
    let pinned_end = source_start
        .checked_add(pinned_prefix_points)
        .ok_or_else(|| {
            AnnError::log_index_config_error(
                "online_suffix_pinned_prefix_points".to_string(),
                "pinned prefix endpoint overflows u32".to_string(),
            )
        })?;
    let shard_count = usize::try_from(source_count)?.div_ceil(shard_points);
    let mut layouts = Vec::with_capacity(shard_count);
    let mut unpinned_counts = Vec::with_capacity(shard_count);
    for shard_id in 0..shard_count {
        let offset = shard_id.checked_mul(shard_points).ok_or_else(|| {
            AnnError::log_index_config_error(
                "shard_points".to_string(),
                "write-back cache shard offset overflows usize".to_string(),
            )
        })?;
        let shard_start = source_start
            .checked_add(u32::try_from(offset)?)
            .ok_or_else(|| {
                AnnError::log_index_config_error(
                    "source_range".to_string(),
                    "write-back cache shard start overflows u32".to_string(),
                )
            })?;
        let shard_end = shard_start
            .checked_add(u32::try_from(shard_points)?)
            .unwrap_or(u32::MAX)
            .min(source_end);
        let local_pinned_end = shard_end.min(pinned_end).max(shard_start);
        let local_pinned = local_pinned_end.saturating_sub(shard_start);
        let unpinned = usize::try_from(shard_end - shard_start - local_pinned)?;
        layouts.push(CachePartitionLayout {
            source_start: shard_start,
            source_end: shard_end,
            pinned_prefix_points: local_pinned,
            dynamic_entries: 0,
        });
        unpinned_counts.push(unpinned);
    }

    let unpinned_partitions = unpinned_counts.iter().filter(|&&count| count != 0).count();
    if dynamic_entries < unpinned_partitions {
        return Err(AnnError::log_index_config_error(
            "online_suffix_state_cache_bytes".to_string(),
            format!(
                "write-back cache needs at least one dynamic entry per unpinned source shard: entries={dynamic_entries}, shards={unpinned_partitions}"
            ),
        ));
    }
    if unpinned_partitions == 0 {
        layouts[0].dynamic_entries = dynamic_entries;
        return Ok(layouts);
    }

    for (layout, &unpinned) in layouts.iter_mut().zip(&unpinned_counts) {
        if unpinned != 0 {
            layout.dynamic_entries = 1;
        }
    }
    let remaining = dynamic_entries - unpinned_partitions;
    let total_unpinned = unpinned_counts.iter().sum::<usize>();
    let total_unpinned_u128 = total_unpinned as u128;
    let mut distributed = 0usize;
    let mut remainders = Vec::with_capacity(layouts.len());
    for (shard_id, (&unpinned, layout)) in
        unpinned_counts.iter().zip(layouts.iter_mut()).enumerate()
    {
        let numerator = (remaining as u128) * (unpinned as u128);
        let share = usize::try_from(numerator / total_unpinned_u128)?;
        layout.dynamic_entries = layout.dynamic_entries.saturating_add(share);
        distributed = distributed.saturating_add(share);
        remainders.push((numerator % total_unpinned_u128, shard_id));
    }
    remainders.sort_unstable_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, shard_id) in remainders
        .into_iter()
        .take(remaining.saturating_sub(distributed))
    {
        layouts[shard_id].dynamic_entries += 1;
    }
    debug_assert_eq!(
        layouts
            .iter()
            .map(|layout| layout.dynamic_entries)
            .sum::<usize>(),
        dynamic_entries
    );
    Ok(layouts)
}

pub(crate) fn estimated_writeback_cache_entry_bytes(l_max: usize) -> AnnResult<usize> {
    let bytes = l_max
        .checked_mul(size_of::<Slot>())
        .and_then(|bytes| bytes.checked_add(size_of::<CachedReservoir>()))
        .and_then(|bytes| bytes.checked_add(size_of::<u32>() * 2))
        .and_then(|bytes| bytes.checked_add(32))
        .ok_or_else(|| {
            AnnError::log_index_config_error(
                "l_max".to_string(),
                "write-back cache entry size overflows usize".to_string(),
            )
        })?;
    Ok(bytes.next_multiple_of(64))
}

impl ReservoirWritebackCache {
    pub(crate) fn new(
        config: ReservoirWritebackCacheConfig,
        store: Arc<DiskReservoirStore>,
    ) -> AnnResult<Self> {
        if config.source_start >= config.source_end
            || config.l_max == 0
            || config.entry_capacity == 0
            || config.pinned_prefix_points > config.source_end.saturating_sub(config.source_start)
            || usize::try_from(config.pinned_prefix_points)? > config.entry_capacity
        {
            return Err(AnnError::log_index_config_error(
                "online_suffix_state_cache".to_string(),
                "invalid source range, capacity, pinned prefix, or reservoir width".to_string(),
            ));
        }
        if config.read_window_bytes < store.page_bytes() {
            return Err(AnnError::log_index_config_error(
                "online_suffix_read_window_bytes".to_string(),
                format!(
                    "must hold at least one {}-byte reservoir page",
                    store.page_bytes()
                ),
            ));
        }
        let entry_capacity = config.entry_capacity;
        Ok(Self {
            config,
            entries: HashMap::with_capacity(entry_capacity),
            clock: VecDeque::with_capacity(entry_capacity),
            next_clock_token: 1,
            store,
            dirty_idle_unpinned: 0,
            cache_pressure_active: false,
            stats: ReservoirWritebackCacheStats {
                capacity_entries: entry_capacity,
                ..ReservoirWritebackCacheStats::default()
            },
        })
    }

    pub(crate) fn update_sources<F>(&mut self, sources: &[u32], mut update: F) -> AnnResult<()>
    where
        F: FnMut(u32, &mut HashPruneReservoir) -> AnnResult<()>,
    {
        let mut unique = BTreeSet::new();
        for &source in sources {
            self.validate_source(source)?;
            if !unique.insert(source) {
                return Err(AnnError::log_index_error(
                    "write-back cache source batch contains duplicates".to_string(),
                ));
            }
        }

        let dynamic_capacity = self
            .config
            .entry_capacity
            .saturating_sub(usize::try_from(self.config.pinned_prefix_points)?);
        let pinned_capacity = usize::try_from(self.config.pinned_prefix_points)?;
        let read_window_limit = (self.config.read_window_bytes / self.store.record_bytes()).max(1);
        let mut window_start = 0usize;
        while window_start < sources.len() {
            let pinned = self.is_pinned(sources[window_start]);
            let class_capacity = if pinned {
                pinned_capacity
            } else {
                dynamic_capacity
            }
            .max(1);
            let class_end = sources[window_start..]
                .iter()
                .position(|source| self.is_pinned(*source) != pinned)
                .map_or(sources.len(), |offset| window_start + offset);
            let window_end = window_start
                .saturating_add(read_window_limit.min(class_capacity))
                .min(class_end);
            let window = &sources[window_start..window_end];
            let protected = window.iter().copied().collect::<BTreeSet<_>>();
            for source in window {
                if let Some(entry) = self.entries.get_mut(source) {
                    // A prior capacity eviction may still be in flight. Reacquiring the source
                    // cancels its deferred eviction before ensure_capacity processes completions.
                    entry.evict_after_clean = false;
                }
            }
            let misses = window
                .iter()
                .copied()
                .filter(|source| !self.entries.contains_key(source))
                .collect::<Vec<_>>();
            self.stats.hits = self
                .stats
                .hits
                .saturating_add(u64::try_from(window.len() - misses.len())?);
            self.stats.misses = self
                .stats
                .misses
                .saturating_add(u64::try_from(misses.len())?);
            if !misses.is_empty() {
                self.ensure_capacity(misses.len(), &protected)?;
                let before = self.store.io_stats();
                let loaded = self.store.load_sources_grouped(&misses)?;
                let after = self.store.io_stats();
                self.stats.read_windows = self.stats.read_windows.saturating_add(1);
                let valid_records = loaded
                    .iter()
                    .filter(|entry| !entry.location.is_invalid())
                    .count() as u64;
                let unique_pages = after
                    .pages_read
                    .saturating_sub(before.pages_read)
                    .saturating_add(after.page_cache_hits.saturating_sub(before.page_cache_hits));
                self.stats.read_pages_requested = self
                    .stats
                    .read_pages_requested
                    .saturating_add(valid_records);
                self.stats.read_pages_coalesced = self
                    .stats
                    .read_pages_coalesced
                    .saturating_add(valid_records.saturating_sub(unique_pages));
                for entry in loaded {
                    let source = entry.source;
                    let clock_token = self.allocate_clock_token()?;
                    let cached = CachedReservoir::loaded(
                        entry.reservoir,
                        entry.location,
                        entry.generation,
                        self.is_pinned(source),
                        clock_token,
                    );
                    if self.entries.insert(source, cached).is_some() {
                        return Err(AnnError::log_index_error(
                            "write-back cache admitted a duplicate source".to_string(),
                        ));
                    }
                    self.clock.push_back((source, clock_token));
                }
                self.stats.peak_entries = self.stats.peak_entries.max(self.entries.len());
                debug_assert!(self.entries.len() <= self.config.entry_capacity);
            }

            for &source in window {
                let (was_dirty_idle, is_dirty_idle) = {
                    let entry = self.entries.get_mut(&source).ok_or_else(|| {
                        AnnError::log_index_error(
                            "write-back cache update source was not admitted".to_string(),
                        )
                    })?;
                    let was_dirty_idle = Self::is_counted_dirty_idle(entry);
                    let snapshot = entry.reservoir.state_snapshot();
                    if let Err(error) = update(source, &mut entry.reservoir) {
                        entry.reservoir = HashPruneReservoir::from_state(
                            self.config.l_max,
                            snapshot,
                            self.config.backfill_policy,
                        )?;
                        return Err(error);
                    }
                    if entry.latest_generation > entry.committed_generation {
                        self.stats.coalesced_updates =
                            self.stats.coalesced_updates.saturating_add(1);
                    }
                    let generation = entry.latest_generation.checked_add(1).ok_or_else(|| {
                        AnnError::log_index_error(
                            "write-back cache source generation overflow".to_string(),
                        )
                    })?;
                    entry.latest_generation = generation;
                    entry.dirty_generation = Some(generation);
                    entry.referenced = true;
                    (was_dirty_idle, Self::is_counted_dirty_idle(entry))
                };
                self.record_dirty_idle_transition(was_dirty_idle, is_dirty_idle)?;
            }
            window_start = window_end;
        }
        Ok(())
    }

    pub(crate) fn maybe_writeback(&mut self) -> AnnResult<()> {
        let pinned_capacity = usize::try_from(self.config.pinned_prefix_points)?;
        let dynamic_capacity = self.config.entry_capacity.saturating_sub(pinned_capacity);
        if dynamic_capacity == 0 {
            return Ok(());
        }
        let dirty_high = (dynamic_capacity / 2).max(1);
        let cache_low = pinned_capacity.saturating_add(dynamic_capacity.saturating_mul(3) / 4);
        let cache_high =
            pinned_capacity.saturating_add(dynamic_capacity.saturating_mul(9).div_ceil(10));
        if self.cache_pressure_active && self.entries.len() <= cache_low {
            self.cache_pressure_active = false;
        }
        let crossed_cache_high = !self.cache_pressure_active && self.entries.len() >= cache_high;
        self.cache_pressure_active |= crossed_cache_high;
        if self.dirty_idle_unpinned == 0
            || (self.dirty_idle_unpinned < dirty_high && !crossed_cache_high)
        {
            return Ok(());
        }
        let sources = self
            .entries
            .iter()
            .filter_map(|(&source, entry)| {
                (entry.is_dirty_and_idle() && !entry.pinned).then_some(source)
            })
            .collect::<Vec<_>>();
        self.stage_dirty_sources(&sources, false)
    }

    pub(crate) fn flush_unpinned(&mut self) -> AnnResult<()> {
        loop {
            self.process_completions()?;
            let dirty = self
                .entries
                .iter()
                .filter_map(|(&source, entry)| {
                    (entry.is_dirty_and_idle() && !entry.pinned).then_some(source)
                })
                .collect::<Vec<_>>();
            if !dirty.is_empty() {
                self.stage_dirty_sources(&dirty, true)?;
            }
            let has_unpinned_work = self.entries.values().any(|entry| {
                !entry.pinned
                    && (entry.dirty_generation.is_some() || entry.inflight_generation.is_some())
            });
            if !has_unpinned_work {
                break;
            }
            self.store.flush_all()?;
        }
        self.process_completions()?;
        Ok(())
    }

    pub(crate) fn get_cached(&self, source: u32) -> Option<&HashPruneReservoir> {
        self.entries.get(&source).map(|entry| &entry.reservoir)
    }

    pub(crate) fn clone_cached_reservoir(
        &self,
        source: u32,
    ) -> AnnResult<Option<HashPruneReservoir>> {
        self.get_cached(source)
            .map(|reservoir| {
                HashPruneReservoir::from_state(
                    self.config.l_max,
                    reservoir.state_snapshot(),
                    self.config.backfill_policy,
                )
            })
            .transpose()
    }

    pub(crate) fn stats(&self) -> &ReservoirWritebackCacheStats {
        &self.stats
    }

    pub(crate) fn store(&self) -> &Arc<DiskReservoirStore> {
        &self.store
    }

    fn ensure_capacity(&mut self, additional: usize, protected: &BTreeSet<u32>) -> AnnResult<()> {
        if additional > self.config.entry_capacity {
            return Err(AnnError::log_index_error(
                "one grouped read exceeds write-back cache capacity".to_string(),
            ));
        }
        while self.entries.len().saturating_add(additional) > self.config.entry_capacity {
            self.process_completions()?;
            if self.entries.len().saturating_add(additional) <= self.config.entry_capacity {
                break;
            }
            if let Some(source) = self.select_clean_victim(protected) {
                self.entries.remove(&source);
                self.stats.clean_evictions = self.stats.clean_evictions.saturating_add(1);
                continue;
            }

            let pinned_capacity = usize::try_from(self.config.pinned_prefix_points)?;
            let dynamic_capacity = self.config.entry_capacity.saturating_sub(pinned_capacity);
            let low_water = pinned_capacity.saturating_add(dynamic_capacity.saturating_mul(3) / 4);
            let target_free = self
                .entries
                .len()
                .saturating_add(additional)
                .saturating_sub(low_water.min(self.config.entry_capacity));
            let dirty = self.select_dirty_victims(target_free.max(1), protected);
            if !dirty.is_empty() {
                self.stage_dirty_sources(&dirty, true)?;
                self.store.wait_for_writeback_progress()?;
                continue;
            }
            if self
                .entries
                .values()
                .any(|entry| entry.inflight_generation.is_some())
            {
                self.store.wait_for_writeback_progress()?;
                continue;
            }
            let pinned = self.entries.values().filter(|entry| entry.pinned).count();
            let dirty = self
                .entries
                .values()
                .filter(|entry| entry.dirty_generation.is_some())
                .count();
            let inflight = self
                .entries
                .values()
                .filter(|entry| entry.inflight_generation.is_some())
                .count();
            return Err(AnnError::log_index_error(format!(
                "write-back cache cannot admit {additional} sources: entries={}, pinned={pinned}, dirty={dirty}, inflight={inflight}, clock={}",
                self.entries.len(),
                self.clock.len()
            )));
        }
        Ok(())
    }

    fn select_clean_victim(&mut self, protected: &BTreeSet<u32>) -> Option<u32> {
        let mut remaining = self.clock.len().saturating_mul(2);
        while remaining != 0 {
            remaining -= 1;
            let (source, clock_token) = self.clock.pop_front()?;
            let Some(entry) = self.entries.get_mut(&source) else {
                continue;
            };
            if entry.clock_token != clock_token {
                continue;
            }
            if entry.pinned || protected.contains(&source) || entry.inflight_generation.is_some() {
                self.clock.push_back((source, clock_token));
                continue;
            }
            if entry.referenced {
                entry.referenced = false;
                self.clock.push_back((source, clock_token));
                continue;
            }
            if entry.is_clean_and_idle() {
                return Some(source);
            }
            self.clock.push_back((source, clock_token));
        }
        None
    }

    fn select_dirty_victims(&mut self, count: usize, protected: &BTreeSet<u32>) -> Vec<u32> {
        let mut selected = Vec::new();
        let mut selected_sources = BTreeSet::new();
        let mut remaining = self.clock.len().saturating_mul(2);
        while remaining != 0 && selected.len() < count {
            remaining -= 1;
            let Some((source, clock_token)) = self.clock.pop_front() else {
                break;
            };
            let Some(entry) = self.entries.get_mut(&source) else {
                continue;
            };
            if entry.clock_token != clock_token {
                continue;
            }
            if entry.pinned || protected.contains(&source) || entry.inflight_generation.is_some() {
                self.clock.push_back((source, clock_token));
                continue;
            }
            if entry.referenced {
                entry.referenced = false;
                self.clock.push_back((source, clock_token));
                continue;
            }
            if entry.is_dirty_and_idle() && selected_sources.insert(source) {
                selected.push(source);
            }
            self.clock.push_back((source, clock_token));
        }
        selected
    }

    fn stage_dirty_sources(&mut self, sources: &[u32], evict_after_clean: bool) -> AnnResult<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let mut writes = Vec::with_capacity(sources.len());
        let mut retiring_dirty_idle = 0usize;
        for &source in sources {
            let entry = self.entries.get(&source).ok_or_else(|| {
                AnnError::log_index_error(
                    "write-back cohort references an uncached source".to_string(),
                )
            })?;
            let generation = entry.dirty_generation.ok_or_else(|| {
                AnnError::log_index_error("write-back cohort references a clean source".to_string())
            })?;
            if entry.inflight_generation.is_some() {
                return Err(AnnError::log_index_error(
                    "write-back cohort references an in-flight source".to_string(),
                ));
            }
            if Self::is_counted_dirty_idle(entry) {
                retiring_dirty_idle += 1;
            }
            writes.push(ReservoirWriteback {
                source,
                generation,
                old_location: entry.committed_location,
                record: DiskReservoirRecord::from_reservoir(source, generation, &entry.reservoir)?,
            });
        }
        if retiring_dirty_idle > self.dirty_idle_unpinned {
            return Err(AnnError::log_index_error(
                "write-back cache dirty-idle accounting underflow".to_string(),
            ));
        }
        self.store.stage_writeback_cohort(writes)?;
        for &source in sources {
            let entry = self
                .entries
                .get_mut(&source)
                .expect("validated cache source");
            let generation = entry
                .dirty_generation
                .take()
                .expect("validated dirty source");
            entry.inflight_generation = Some(generation);
            entry.evict_after_clean |= evict_after_clean;
        }
        self.dirty_idle_unpinned -= retiring_dirty_idle;
        self.stats.writeback_cohorts = self.stats.writeback_cohorts.saturating_add(1);
        self.stats.writeback_records = self
            .stats
            .writeback_records
            .saturating_add(u64::try_from(sources.len())?);
        Ok(())
    }

    fn process_completions(&mut self) -> AnnResult<()> {
        for completion in self
            .store
            .take_writeback_completions_for_shard(self.config.store_shard_id)?
        {
            self.apply_completion(completion)?;
        }
        Ok(())
    }

    fn apply_completion(&mut self, completion: ReservoirWritebackCompletion) -> AnnResult<()> {
        let (became_dirty_idle, evict) = {
            let entry = self.entries.get_mut(&completion.source).ok_or_else(|| {
                AnnError::log_index_error(
                    "write-back completion references an evicted cache entry".to_string(),
                )
            })?;
            if entry.inflight_generation != Some(completion.generation)
                || entry.committed_location != completion.old_location
                || completion.generation <= entry.committed_generation
            {
                return Err(AnnError::log_index_error(
                    "write-back completion does not match cached source state".to_string(),
                ));
            }
            self.stats.committed_updates = self.stats.committed_updates.saturating_add(u64::from(
                completion
                    .generation
                    .saturating_sub(entry.committed_generation),
            ));
            entry.committed_generation = completion.generation;
            entry.committed_location = completion.new_location;
            entry.inflight_generation = None;
            let became_dirty_idle = Self::is_counted_dirty_idle(entry);
            let evict = entry.evict_after_clean
                && entry.is_clean_and_idle()
                && entry.latest_generation == entry.committed_generation
                && !entry.pinned;
            (became_dirty_idle, evict)
        };
        if became_dirty_idle {
            self.record_dirty_idle_transition(false, true)?;
        }
        if evict {
            self.entries.remove(&completion.source);
            self.stats.dirty_evictions = self.stats.dirty_evictions.saturating_add(1);
        }
        Ok(())
    }

    fn validate_source(&self, source: u32) -> AnnResult<()> {
        if source < self.config.source_start || source >= self.config.source_end {
            return Err(AnnError::log_index_error(format!(
                "source {source} is outside write-back cache range [{}, {})",
                self.config.source_start, self.config.source_end
            )));
        }
        Ok(())
    }

    fn is_pinned(&self, source: u32) -> bool {
        source
            < self
                .config
                .source_start
                .saturating_add(self.config.pinned_prefix_points)
    }

    fn is_counted_dirty_idle(entry: &CachedReservoir) -> bool {
        !entry.pinned && entry.is_dirty_and_idle()
    }

    fn allocate_clock_token(&mut self) -> AnnResult<u64> {
        let token = self.next_clock_token;
        self.next_clock_token = token.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error("write-back cache CLOCK token overflow".to_string())
        })?;
        Ok(token)
    }

    fn record_dirty_idle_transition(&mut self, before: bool, after: bool) -> AnnResult<()> {
        match (before, after) {
            (false, true) => {
                self.dirty_idle_unpinned =
                    self.dirty_idle_unpinned.checked_add(1).ok_or_else(|| {
                        AnnError::log_index_error(
                            "write-back cache dirty-idle accounting overflow".to_string(),
                        )
                    })?;
            }
            (true, false) => {
                self.dirty_idle_unpinned =
                    self.dirty_idle_unpinned.checked_sub(1).ok_or_else(|| {
                        AnnError::log_index_error(
                            "write-back cache dirty-idle accounting underflow".to_string(),
                        )
                    })?;
            }
            _ => {}
        }
        Ok(())
    }

    #[cfg(test)]
    fn new_for_state_test(
        entry_capacity: usize,
        pinned_prefix_points: u32,
        l_max: usize,
    ) -> AnnResult<Self> {
        use super::direct_io::DirectIoConfig;
        use super::disk_reservoir_store::DiskReservoirStoreConfig;

        let dir = tempfile::tempdir()?;
        let store = Arc::new(DiskReservoirStore::create(DiskReservoirStoreConfig {
            root: dir.path().join("state"),
            source_start: 0,
            source_end: 16,
            shard_points: 16,
            l_max,
            page_cache_bytes: 4096 * 2,
            writeback_bytes: 4096,
            decode_bytes: 4096,
            writeback_workers: 1,
            overprovision_ratio: 2.0,
            io: DirectIoConfig::disabled(),
            policy_key: [0; 32],
            backfill_policy: HashPruneBackfillPolicy::off(),
        })?);
        Self::new(
            ReservoirWritebackCacheConfig {
                store_shard_id: 0,
                source_start: 0,
                source_end: 16,
                pinned_prefix_points,
                entry_capacity,
                read_window_bytes: 4096,
                l_max,
                backfill_policy: HashPruneBackfillPolicy::off(),
            },
            store,
        )
    }

    #[cfg(test)]
    fn insert_empty_for_test(&mut self, source: u32) -> AnnResult<()> {
        self.validate_source(source)?;
        if self.entries.len() == self.config.entry_capacity {
            return Err(AnnError::log_index_error(
                "test cache capacity exceeded".to_string(),
            ));
        }
        let clock_token = self.allocate_clock_token()?;
        self.entries.insert(
            source,
            CachedReservoir::loaded(
                HashPruneReservoir::new(self.config.l_max),
                PackedLocation::INVALID,
                0,
                self.is_pinned(source),
                clock_token,
            ),
        );
        self.clock.push_back((source, clock_token));
        Ok(())
    }

    #[cfg(test)]
    fn mark_inflight_for_test(&mut self, source: u32, generation: u32) {
        let entry = self.entries.get_mut(&source).expect("test source");
        let was_dirty_idle = Self::is_counted_dirty_idle(entry);
        entry.latest_generation = generation;
        entry.inflight_generation = Some(generation);
        entry.dirty_generation = None;
        let is_dirty_idle = Self::is_counted_dirty_idle(entry);
        self.record_dirty_idle_transition(was_dirty_idle, is_dirty_idle)
            .expect("test dirty-idle transition");
    }

    #[cfg(test)]
    fn select_clean_victim_for_test(&mut self) -> Option<u32> {
        self.select_clean_victim(&BTreeSet::new())
    }

    #[cfg(test)]
    fn update_target_for_test(&mut self, source: u32, target: u32) -> AnnResult<()> {
        if !self.entries.contains_key(&source) {
            self.insert_empty_for_test(source)?;
        }
        let (was_dirty_idle, is_dirty_idle) = {
            let entry = self.entries.get_mut(&source).expect("inserted test source");
            let was_dirty_idle = Self::is_counted_dirty_idle(entry);
            entry.reservoir.insert(target, target as u16, target as f32);
            let generation = entry
                .latest_generation
                .checked_add(1)
                .ok_or_else(|| AnnError::log_index_error("test generation overflow".to_string()))?;
            entry.latest_generation = generation;
            entry.dirty_generation = Some(generation);
            (was_dirty_idle, Self::is_counted_dirty_idle(entry))
        };
        self.record_dirty_idle_transition(was_dirty_idle, is_dirty_idle)?;
        Ok(())
    }

    #[cfg(test)]
    fn begin_writeback_for_test(&mut self, source: u32) -> AnnResult<TestWritebackSnapshot> {
        let (generation, was_dirty_idle, is_dirty_idle) = {
            let entry = self
                .entries
                .get_mut(&source)
                .ok_or_else(|| AnnError::log_index_error("missing test source".to_string()))?;
            let was_dirty_idle = Self::is_counted_dirty_idle(entry);
            let generation = entry
                .dirty_generation
                .take()
                .ok_or_else(|| AnnError::log_index_error("clean test source".to_string()))?;
            entry.inflight_generation = Some(generation);
            (
                generation,
                was_dirty_idle,
                Self::is_counted_dirty_idle(entry),
            )
        };
        self.record_dirty_idle_transition(was_dirty_idle, is_dirty_idle)?;
        Ok(TestWritebackSnapshot { source, generation })
    }

    #[cfg(test)]
    fn complete_for_test(&mut self, snapshot: TestWritebackSnapshot) -> AnnResult<()> {
        let (was_dirty_idle, is_dirty_idle) = {
            let entry = self
                .entries
                .get_mut(&snapshot.source)
                .ok_or_else(|| AnnError::log_index_error("missing test source".to_string()))?;
            let was_dirty_idle = Self::is_counted_dirty_idle(entry);
            if entry.inflight_generation != Some(snapshot.generation) {
                return Err(AnnError::log_index_error(
                    "mismatched test completion".to_string(),
                ));
            }
            entry.committed_generation = snapshot.generation;
            entry.inflight_generation = None;
            (was_dirty_idle, Self::is_counted_dirty_idle(entry))
        };
        self.record_dirty_idle_transition(was_dirty_idle, is_dirty_idle)?;
        Ok(())
    }

    #[cfg(test)]
    fn targets_for_test(&self, source: u32) -> Vec<u32> {
        self.entries[&source]
            .reservoir
            .slots()
            .iter()
            .map(|slot| slot.target)
            .collect()
    }

    #[cfg(test)]
    fn is_dirty_for_test(&self, source: u32) -> bool {
        self.entries[&source].dirty_generation.is_some()
    }

    #[cfg(test)]
    fn dirty_idle_unpinned_for_test(&self) -> usize {
        assert_eq!(
            self.dirty_idle_unpinned,
            self.entries
                .values()
                .filter(|entry| Self::is_counted_dirty_idle(entry))
                .count()
        );
        self.dirty_idle_unpinned
    }

    #[cfg(test)]
    fn remove_entry_without_clock_for_test(&mut self, source: u32) {
        let entry = self.entries.remove(&source).expect("test source");
        assert!(!Self::is_counted_dirty_idle(&entry));
    }

    #[cfg(test)]
    fn clock_token_validity_for_test(&self, source: u32) -> Vec<bool> {
        let current = self.entries.get(&source).map(|entry| entry.clock_token);
        self.clock
            .iter()
            .filter_map(|(queued_source, token)| {
                (*queued_source == source).then_some(Some(*token) == current)
            })
            .collect()
    }
}

impl ShardedReservoirWritebackCache {
    pub(crate) fn new(
        config: ShardedReservoirWritebackCacheConfig,
        store: Arc<DiskReservoirStore>,
    ) -> AnnResult<Self> {
        let layouts = partition_cache_layouts(
            config.source_start,
            config.source_end,
            config.shard_points,
            config.pinned_prefix_points,
            config.dynamic_entry_capacity,
        )?;
        if layouts.len() != store.shard_count() {
            return Err(AnnError::log_index_config_error(
                "online_suffix_state_cache".to_string(),
                format!(
                    "cache partitions do not match reservoir store shards: cache={} store={}",
                    layouts.len(),
                    store.shard_count()
                ),
            ));
        }
        let partitions = layouts
            .into_iter()
            .enumerate()
            .map(|(store_shard_id, layout)| {
                let entry_capacity = usize::try_from(layout.pinned_prefix_points)?
                    .checked_add(layout.dynamic_entries)
                    .ok_or_else(|| {
                        AnnError::log_index_config_error(
                            "online_suffix_state_cache_bytes".to_string(),
                            "partition cache entry capacity overflows usize".to_string(),
                        )
                    })?;
                ReservoirWritebackCache::new(
                    ReservoirWritebackCacheConfig {
                        store_shard_id,
                        source_start: layout.source_start,
                        source_end: layout.source_end,
                        pinned_prefix_points: layout.pinned_prefix_points,
                        entry_capacity,
                        read_window_bytes: config.read_window_bytes,
                        l_max: config.l_max,
                        backfill_policy: config.backfill_policy,
                    },
                    Arc::clone(&store),
                )
                .map(Mutex::new)
            })
            .collect::<AnnResult<Vec<_>>>()?;
        Ok(Self {
            source_start: config.source_start,
            source_end: config.source_end,
            shard_points: config.shard_points,
            partitions,
        })
    }

    pub(crate) fn update_sources<F>(
        &self,
        shard_id: usize,
        sources: &[u32],
        update: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, &mut HashPruneReservoir) -> AnnResult<()>,
    {
        for &source in sources {
            if self.shard_for_source(source)? != shard_id {
                return Err(AnnError::log_index_error(
                    "write-back cache update crosses source shard boundaries".to_string(),
                ));
            }
        }
        let partition = self.partitions.get(shard_id).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "write-back cache shard {shard_id} is outside {} partitions",
                self.partitions.len()
            ))
        })?;
        let mut partition = partition.lock();
        partition.update_sources(sources, update)?;
        partition.maybe_writeback()
    }

    pub(crate) fn flush_unpinned(&self) -> AnnResult<()> {
        for partition in &self.partitions {
            partition.lock().flush_unpinned()?;
        }
        Ok(())
    }

    pub(crate) fn clone_cached_reservoir(
        &self,
        source: u32,
    ) -> AnnResult<Option<HashPruneReservoir>> {
        let shard_id = self.shard_for_source(source)?;
        self.partitions[shard_id]
            .lock()
            .clone_cached_reservoir(source)
    }

    pub(crate) fn stats(&self) -> ReservoirWritebackCacheStats {
        let mut total = ReservoirWritebackCacheStats::default();
        for partition in &self.partitions {
            let partition = partition.lock();
            let stats = partition.stats();
            total.capacity_entries = total
                .capacity_entries
                .saturating_add(stats.capacity_entries);
            total.peak_entries = total.peak_entries.saturating_add(stats.peak_entries);
            total.hits = total.hits.saturating_add(stats.hits);
            total.misses = total.misses.saturating_add(stats.misses);
            total.clean_evictions = total.clean_evictions.saturating_add(stats.clean_evictions);
            total.dirty_evictions = total.dirty_evictions.saturating_add(stats.dirty_evictions);
            total.coalesced_updates = total
                .coalesced_updates
                .saturating_add(stats.coalesced_updates);
            total.read_windows = total.read_windows.saturating_add(stats.read_windows);
            total.read_pages_requested = total
                .read_pages_requested
                .saturating_add(stats.read_pages_requested);
            total.read_pages_coalesced = total
                .read_pages_coalesced
                .saturating_add(stats.read_pages_coalesced);
            total.writeback_cohorts = total
                .writeback_cohorts
                .saturating_add(stats.writeback_cohorts);
            total.writeback_records = total
                .writeback_records
                .saturating_add(stats.writeback_records);
            total.committed_updates = total
                .committed_updates
                .saturating_add(stats.committed_updates);
        }
        total
    }

    pub(crate) fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    fn shard_for_source(&self, source: u32) -> AnnResult<usize> {
        if source < self.source_start || source >= self.source_end {
            return Err(AnnError::log_index_error(format!(
                "source {source} is outside sharded write-back cache range [{}, {})",
                self.source_start, self.source_end
            )));
        }
        Ok(usize::try_from(source - self.source_start)? / self.shard_points)
    }

    #[cfg(test)]
    fn new_for_state_test(
        source_end: usize,
        shard_points: usize,
        dynamic_entries: usize,
        l_max: usize,
    ) -> AnnResult<Self> {
        use super::direct_io::DirectIoConfig;
        use super::disk_reservoir_store::DiskReservoirStoreConfig;

        let shard_count = source_end.div_ceil(shard_points);
        if shard_count == 0 || dynamic_entries < shard_count {
            return Err(AnnError::log_index_config_error(
                "online_suffix_state_cache_bytes".to_string(),
                "test cache needs at least one dynamic entry per source shard".to_string(),
            ));
        }
        let dir = tempfile::tempdir()?;
        let store = Arc::new(DiskReservoirStore::create(DiskReservoirStoreConfig {
            root: dir.path().join("state"),
            source_start: 0,
            source_end,
            shard_points,
            l_max,
            page_cache_bytes: 4096 * shard_count,
            writeback_bytes: 4096 * shard_count,
            decode_bytes: 4096 * shard_count,
            writeback_workers: shard_count,
            overprovision_ratio: 2.0,
            io: DirectIoConfig::disabled(),
            policy_key: [0; 32],
            backfill_policy: HashPruneBackfillPolicy::off(),
        })?);
        Self::new(
            ShardedReservoirWritebackCacheConfig {
                source_start: 0,
                source_end: u32::try_from(source_end)?,
                shard_points,
                pinned_prefix_points: 0,
                dynamic_entry_capacity: dynamic_entries,
                read_window_bytes: 4096,
                l_max,
                backfill_policy: HashPruneBackfillPolicy::off(),
            },
            store,
        )
    }
}

#[cfg(test)]
struct TestWritebackSnapshot {
    source: u32,
    generation: u32,
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn clock_never_evicts_pinned_or_inflight_entries() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(3, 1, 4).unwrap();
        cache.insert_empty_for_test(0).unwrap();
        cache.insert_empty_for_test(1).unwrap();
        cache.insert_empty_for_test(2).unwrap();
        cache.mark_inflight_for_test(1, 1);

        assert_eq!(cache.select_clean_victim_for_test(), Some(2));
    }

    #[test]
    fn stale_clock_token_does_not_alias_reloaded_source() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(2, 0, 4).unwrap();
        cache.insert_empty_for_test(1).unwrap();
        cache.remove_entry_without_clock_for_test(1);
        cache.insert_empty_for_test(1).unwrap();

        assert_eq!(cache.clock_token_validity_for_test(1), vec![false, true]);
    }

    #[test]
    fn dirty_state_remains_authoritative_while_older_generation_is_inflight() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(1, 0, 4).unwrap();
        cache.update_target_for_test(7, 10).unwrap();
        let snapshot = cache.begin_writeback_for_test(7).unwrap();
        cache.update_target_for_test(7, 11).unwrap();
        cache.complete_for_test(snapshot).unwrap();

        assert_eq!(cache.targets_for_test(7), vec![10, 11]);
        assert!(cache.is_dirty_for_test(7));
    }

    #[test]
    fn dirty_idle_counter_tracks_update_writeback_and_completion_transitions() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(8, 0, 4).unwrap();
        cache.update_target_for_test(0, 10).unwrap();
        cache.update_target_for_test(1, 11).unwrap();
        assert_eq!(cache.dirty_idle_unpinned_for_test(), 2);

        cache.update_target_for_test(0, 12).unwrap();
        assert_eq!(cache.dirty_idle_unpinned_for_test(), 2);

        let snapshot = cache.begin_writeback_for_test(0).unwrap();
        assert_eq!(cache.dirty_idle_unpinned_for_test(), 1);

        cache.update_target_for_test(0, 13).unwrap();
        assert_eq!(cache.dirty_idle_unpinned_for_test(), 1);

        cache.complete_for_test(snapshot).unwrap();
        assert_eq!(cache.dirty_idle_unpinned_for_test(), 2);
    }

    #[test]
    fn writeback_cache_coalesces_repeated_updates_before_flush() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(8, 0, 4).unwrap();
        for target in [30_u32, 31, 32] {
            cache
                .update_sources(&[3], |_source, reservoir| {
                    reservoir.insert(target, target as u16, target as f32);
                    Ok(())
                })
                .unwrap();
            cache.maybe_writeback().unwrap();
        }

        cache.flush_unpinned().unwrap();

        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().coalesced_updates, 2);
        assert_eq!(cache.stats().writeback_records, 1);
        assert_eq!(cache.stats().committed_updates, 3);
        assert_eq!(cache.stats().updates_per_commit(), 3.0);
        assert_eq!(
            cache.store().load_sources_grouped(&[3]).unwrap()[0]
                .reservoir
                .slots()
                .iter()
                .map(|slot| slot.target)
                .collect::<Vec<_>>(),
            vec![30, 31, 32]
        );
    }

    #[test]
    fn writeback_cache_pressure_evicts_one_page_cohort() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(2, 0, 4).unwrap();
        cache
            .update_sources(&[0, 1], |source, reservoir| {
                reservoir.insert(source + 10, source as u16, source as f32);
                Ok(())
            })
            .unwrap();
        cache
            .update_sources(&[2], |source, reservoir| {
                reservoir.insert(source + 10, source as u16, source as f32);
                Ok(())
            })
            .unwrap();
        cache.flush_unpinned().unwrap();

        assert_eq!(cache.stats().dirty_evictions, 3);
        assert_eq!(cache.stats().writeback_cohorts, 2);
        assert_eq!(cache.store().io_stats().pages_written, 2);
    }

    #[test]
    fn writeback_cache_capacity_one_stages_each_dirty_source_once() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(1, 0, 4).unwrap();
        cache
            .update_sources(&[0], |_source, reservoir| {
                reservoir.insert(10, 0, 1.0);
                Ok(())
            })
            .unwrap();

        cache
            .update_sources(&[1], |_source, reservoir| {
                reservoir.insert(11, 1, 1.0);
                Ok(())
            })
            .unwrap();
        cache.flush_unpinned().unwrap();

        assert_eq!(cache.stats().writeback_records, 2);
    }

    #[test]
    fn writeback_cache_admission_preserves_hits_in_current_window() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(2, 0, 4).unwrap();
        cache
            .update_sources(&[0, 1], |source, reservoir| {
                reservoir.insert(source + 10, source as u16, 1.0);
                Ok(())
            })
            .unwrap();

        cache
            .update_sources(&[0, 2], |source, reservoir| {
                reservoir.insert(source + 20, source as u16, 2.0);
                Ok(())
            })
            .unwrap();
        cache.flush_unpinned().unwrap();

        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn fully_pinned_cache_admits_one_grouped_read_window() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(4, 4, 4).unwrap();

        cache
            .update_sources(&[0, 1, 2, 3], |source, reservoir| {
                reservoir.insert(source + 10, source as u16, 1.0);
                Ok(())
            })
            .unwrap();

        assert_eq!(cache.stats().read_windows, 1);
        assert_eq!(cache.stats().misses, 4);
    }

    #[test]
    fn cache_pressure_applies_low_water_to_dynamic_entries_only() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(12, 8, 4).unwrap();
        cache
            .update_sources(&(0_u32..12).collect::<Vec<_>>(), |source, reservoir| {
                reservoir.insert(source + 20, source as u16, 1.0);
                Ok(())
            })
            .unwrap();

        cache
            .update_sources(&[12], |source, reservoir| {
                reservoir.insert(source + 20, source as u16, 1.0);
                Ok(())
            })
            .unwrap();

        assert_eq!(cache.stats().writeback_records, 2);
    }

    #[test]
    fn cache_high_water_writeback_is_hysteretic() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(10, 0, 4).unwrap();
        cache
            .update_sources(&(0_u32..9).collect::<Vec<_>>(), |source, reservoir| {
                reservoir.insert(source + 20, source as u16, 1.0);
                Ok(())
            })
            .unwrap();
        cache.maybe_writeback().unwrap();
        cache.store().flush_all().unwrap();
        cache.process_completions().unwrap();
        assert_eq!(cache.stats().writeback_records, 9);

        cache
            .update_sources(&[0], |_source, reservoir| {
                reservoir.insert(100, 100, 0.5);
                Ok(())
            })
            .unwrap();
        cache.maybe_writeback().unwrap();

        assert_eq!(cache.stats().writeback_records, 9);
        assert!(cache.is_dirty_for_test(0));
    }

    #[test]
    fn cache_hit_defers_completion_poll_until_pressure_or_flush() {
        let mut cache = ReservoirWritebackCache::new_for_state_test(4, 0, 4).unwrap();
        cache
            .update_sources(&[0, 1], |source, reservoir| {
                reservoir.insert(source + 10, source as u16, 1.0);
                Ok(())
            })
            .unwrap();
        cache.maybe_writeback().unwrap();
        cache.store().flush_all().unwrap();

        cache
            .update_sources(&[0], |_source, reservoir| {
                reservoir.insert(20, 20, 0.5);
                Ok(())
            })
            .unwrap();
        cache.maybe_writeback().unwrap();

        let entry = cache.entries.get(&0).unwrap();
        assert_eq!(entry.inflight_generation, Some(1));
        assert_eq!(entry.dirty_generation, Some(2));
        cache.flush_unpinned().unwrap();
    }

    #[test]
    fn writeback_cache_different_source_shards_progress_independently() {
        let cache =
            Arc::new(ShardedReservoirWritebackCache::new_for_state_test(4, 2, 2, 4).unwrap());
        let (first_entered_tx, first_entered_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let (second_entered_tx, second_entered_rx) = mpsc::channel();

        std::thread::scope(|scope| {
            let first_cache = Arc::clone(&cache);
            let first = scope.spawn(move || {
                first_cache.update_sources(0, &[0], |_source, reservoir| {
                    first_entered_tx.send(()).unwrap();
                    release_first_rx.recv().unwrap();
                    reservoir.insert(10, 1, 1.0);
                    Ok(())
                })
            });
            first_entered_rx.recv().unwrap();

            let second_cache = Arc::clone(&cache);
            let second = scope.spawn(move || {
                second_cache.update_sources(1, &[2], |_source, reservoir| {
                    second_entered_tx.send(()).unwrap();
                    reservoir.insert(12, 2, 1.0);
                    Ok(())
                })
            });

            second_entered_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("source shard 1 must not wait for source shard 0's cache lock");
            release_first_tx.send(()).unwrap();

            first.join().unwrap().unwrap();
            second.join().unwrap().unwrap();
        });
    }

    #[test]
    fn writeback_cache_partition_layout_preserves_global_pinned_and_dynamic_budgets() {
        let layouts = partition_cache_layouts(0, 10, 4, 5, 5).unwrap();

        assert_eq!(
            layouts
                .iter()
                .map(|layout| (
                    layout.source_start,
                    layout.source_end,
                    layout.pinned_prefix_points,
                    layout.dynamic_entries,
                ))
                .collect::<Vec<_>>(),
            vec![(0, 4, 4, 0), (4, 8, 1, 3), (8, 10, 0, 2)]
        );
        assert_eq!(
            layouts
                .iter()
                .map(|layout| layout.pinned_prefix_points as usize)
                .sum::<usize>(),
            5
        );
        assert_eq!(
            layouts
                .iter()
                .map(|layout| layout.dynamic_entries)
                .sum::<usize>(),
            5
        );
    }

    #[test]
    fn writeback_cache_partition_layout_rejects_dynamic_budget_starvation() {
        let error = partition_cache_layouts(0, 10, 4, 5, 1).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("one dynamic entry per unpinned source shard")
        );
    }
}
