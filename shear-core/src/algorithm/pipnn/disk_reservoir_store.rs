use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

use crossbeam::channel::{Receiver, Sender, bounded};
use parking_lot::{Condvar, Mutex};

use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::hash_prune::{HashPruneReservoir, HashPruneReservoirState, Slot};
use super::hash_prune_backfill::HashPruneBackfillPolicy;
use super::hash_prune_evidence::HashPruneMeta;
use crate::common::{AnnError, AnnResult};

const STORE_MAGIC: [u8; 8] = *b"HPSTATE1";
const STORE_VERSION: u16 = 1;
const RECORD_HEADER_BYTES: usize = 16;
const DISK_SLOT_BYTES: usize = 12;
const HEADER_FIELDS_BYTES: usize = 72;
const DISK_ALIGNMENT: usize = 4096;
const INVALID_TARGET: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiskReservoirLayout {
    pub(crate) l_max: usize,
    pub(crate) record_bytes: usize,
    pub(crate) page_bytes: usize,
    pub(crate) records_per_page: usize,
}

impl DiskReservoirLayout {
    pub(crate) fn new(l_max: usize, minimum_page_bytes: usize) -> AnnResult<Self> {
        if l_max == 0 || l_max > u16::MAX as usize {
            return Err(AnnError::log_index_config_error(
                "l_max".to_string(),
                "disk reservoir layout requires 1 <= l_max <= 65535".to_string(),
            ));
        }
        let record_bytes = l_max
            .checked_mul(DISK_SLOT_BYTES)
            .and_then(|bytes| bytes.checked_add(RECORD_HEADER_BYTES))
            .ok_or_else(|| {
                AnnError::log_index_config_error(
                    "l_max".to_string(),
                    "disk reservoir record size overflows usize".to_string(),
                )
            })?;
        let page_bytes =
            round_up_to_alignment(minimum_page_bytes.max(record_bytes), DISK_ALIGNMENT)?;
        let records_per_page = page_bytes / record_bytes;
        debug_assert!(records_per_page > 0);
        Ok(Self {
            l_max,
            record_bytes,
            page_bytes,
            records_per_page,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DiskReservoirRecord {
    pub(crate) source_id: u32,
    pub(crate) generation: u32,
    pub(crate) slots: Vec<Slot>,
    pub(crate) primary_count: u16,
    pub(crate) backup_count: u16,
}

impl DiskReservoirRecord {
    pub(crate) fn from_reservoir(
        source_id: u32,
        generation: u32,
        reservoir: &HashPruneReservoir,
    ) -> AnnResult<Self> {
        let state = reservoir.state_snapshot();
        let slot_count = state.slots.len();
        let backup_count = usize::from(state.backup_count);
        let primary_count = slot_count.checked_sub(backup_count).ok_or_else(|| {
            AnnError::log_index_error(
                "invalid HashPrune reservoir boundary while encoding disk state".to_string(),
            )
        })?;
        for slot in &state.slots {
            if slot.target == INVALID_TARGET {
                return Err(AnnError::log_index_error(
                    "reserved target cannot be encoded in disk reservoir state".to_string(),
                ));
            }
            slot.meta.to_state_flags()?;
        }
        Ok(Self {
            source_id,
            generation,
            slots: state.slots,
            primary_count: u16::try_from(primary_count)?,
            backup_count: state.backup_count,
        })
    }

    pub(crate) fn encode_into(
        &self,
        layout: &DiskReservoirLayout,
        output: &mut [u8],
    ) -> AnnResult<()> {
        if output.len() != layout.record_bytes {
            return Err(AnnError::log_index_error(format!(
                "disk reservoir output length mismatch: expected {}, got {}",
                layout.record_bytes,
                output.len()
            )));
        }
        let slot_count = self.slots.len();
        if slot_count > layout.l_max
            || usize::from(self.primary_count) + usize::from(self.backup_count) != slot_count
        {
            return Err(AnnError::log_index_error(
                "invalid disk reservoir record boundary".to_string(),
            ));
        }

        output.fill(0);
        write_u32(output, 0, self.source_id);
        write_u32(output, 4, self.generation);
        write_u16(output, 8, u16::try_from(slot_count)?);
        write_u16(output, 10, self.primary_count);
        write_u16(output, 12, self.backup_count);

        for index in 0..layout.l_max {
            let offset = RECORD_HEADER_BYTES + index * DISK_SLOT_BYTES;
            write_u32(output, offset, INVALID_TARGET);
        }
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.target == INVALID_TARGET {
                return Err(AnnError::log_index_error(
                    "reserved target cannot be encoded in disk reservoir state".to_string(),
                ));
            }
            let offset = RECORD_HEADER_BYTES + index * DISK_SLOT_BYTES;
            write_u32(output, offset, slot.target);
            write_u32(output, offset + 4, slot.dist.to_bits());
            write_u16(output, offset + 8, slot.hash);
            write_u16(output, offset + 10, slot.meta.to_state_flags()?);
        }
        Ok(())
    }

    pub(crate) fn decode(layout: &DiskReservoirLayout, input: &[u8]) -> AnnResult<Self> {
        if input.len() != layout.record_bytes {
            return Err(AnnError::log_index_error(format!(
                "disk reservoir input length mismatch: expected {}, got {}",
                layout.record_bytes,
                input.len()
            )));
        }
        let source_id = read_u32(input, 0);
        let generation = read_u32(input, 4);
        let slot_count = usize::from(read_u16(input, 8));
        let primary_count = read_u16(input, 10);
        let backup_count = read_u16(input, 12);
        if input[14..16] != [0, 0]
            || slot_count > layout.l_max
            || usize::from(primary_count) + usize::from(backup_count) != slot_count
        {
            return Err(AnnError::log_index_error(
                "corrupt disk reservoir record boundary".to_string(),
            ));
        }

        let mut slots = Vec::with_capacity(slot_count);
        for index in 0..layout.l_max {
            let offset = RECORD_HEADER_BYTES + index * DISK_SLOT_BYTES;
            let target = read_u32(input, offset);
            let distance_bits = read_u32(input, offset + 4);
            let hash = read_u16(input, offset + 8);
            let state_flags = read_u16(input, offset + 10);
            if index < slot_count {
                if target == INVALID_TARGET {
                    return Err(AnnError::log_index_error(
                        "used disk reservoir slot contains the invalid target marker".to_string(),
                    ));
                }
                let meta = HashPruneMeta::from_state_flags(state_flags)?;
                slots.push(Slot::new_with_meta(
                    target,
                    hash,
                    f32::from_bits(distance_bits),
                    meta,
                ));
            } else if target != INVALID_TARGET
                || distance_bits != 0
                || hash != 0
                || state_flags != 0
            {
                return Err(AnnError::log_index_error(
                    "unused disk reservoir slot is not empty".to_string(),
                ));
            }
        }
        Ok(Self {
            source_id,
            generation,
            slots,
            primary_count,
            backup_count,
        })
    }

    pub(crate) fn into_reservoir(
        self,
        l_max: usize,
        policy: HashPruneBackfillPolicy,
    ) -> AnnResult<HashPruneReservoir> {
        HashPruneReservoir::from_state(
            l_max,
            HashPruneReservoirState {
                slots: self.slots,
                backup_count: self.backup_count,
            },
            policy,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackedLocation(u64);

impl PackedLocation {
    pub(crate) const INVALID: Self = Self(u64::MAX);

    #[inline]
    const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    #[inline]
    const fn as_raw(self) -> u64 {
        self.0
    }

    pub(crate) fn new(page_id: u64, slot_index: usize, records_per_page: usize) -> AnnResult<Self> {
        if records_per_page == 0 || slot_index >= records_per_page {
            return Err(AnnError::log_index_error(
                "invalid packed disk reservoir location".to_string(),
            ));
        }
        let records_per_page = u64::try_from(records_per_page)?;
        let slot_index = u64::try_from(slot_index)?;
        let physical_slot = page_id
            .checked_mul(records_per_page)
            .and_then(|base| base.checked_add(slot_index))
            .filter(|&value| value != Self::INVALID.0)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "packed disk reservoir location overflows u64".to_string(),
                )
            })?;
        Ok(Self(physical_slot))
    }

    pub(crate) const fn is_invalid(self) -> bool {
        self.0 == Self::INVALID.0
    }

    pub(crate) fn decode(self, records_per_page: usize) -> AnnResult<(u64, usize)> {
        if self.is_invalid() || records_per_page == 0 {
            return Err(AnnError::log_index_error(
                "cannot decode invalid disk reservoir location".to_string(),
            ));
        }
        let records_per_page_u64 = u64::try_from(records_per_page)?;
        let page_id = self.0 / records_per_page_u64;
        let slot_index = usize::try_from(self.0 % records_per_page_u64)?;
        Ok((page_id, slot_index))
    }

    fn physical_slot(self, records_per_page: usize) -> AnnResult<u64> {
        self.decode(records_per_page)?;
        Ok(self.0)
    }
}

#[derive(Debug)]
pub(crate) struct LoadedReservoir {
    pub(crate) source: u32,
    pub(crate) generation: u32,
    pub(crate) location: PackedLocation,
    pub(crate) reservoir: HashPruneReservoir,
}

#[derive(Clone, Debug)]
pub(crate) struct ReservoirWriteback {
    pub(crate) source: u32,
    pub(crate) generation: u32,
    pub(crate) old_location: PackedLocation,
    pub(crate) record: DiskReservoirRecord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReservoirWritebackCompletion {
    pub(crate) source: u32,
    pub(crate) generation: u32,
    pub(crate) old_location: PackedLocation,
    pub(crate) new_location: PackedLocation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PageAllocationStats {
    empty_page: u64,
    resident_partial_page: u64,
    cold_partial_page: u64,
}

#[derive(Debug)]
struct SlotAllocator {
    next_unused: u64,
    capacity: u64,
    records_per_page: usize,
    recycled_by_page: BTreeMap<u64, BTreeSet<usize>>,
    reserved_by_page: BTreeMap<u64, BTreeSet<usize>>,
    recycled_count: u64,
    reused_allocations: u64,
    allocation_stats: PageAllocationStats,
}

impl SlotAllocator {
    fn new(capacity: u64, records_per_page: usize) -> Self {
        debug_assert!(records_per_page > 0);
        Self {
            next_unused: 0,
            capacity,
            records_per_page,
            recycled_by_page: BTreeMap::new(),
            reserved_by_page: BTreeMap::new(),
            recycled_count: 0,
            reused_allocations: 0,
            allocation_stats: PageAllocationStats::default(),
        }
    }

    fn allocate(&mut self) -> AnnResult<u64> {
        if let Some(page_id) = self
            .recycled_by_page
            .first_key_value()
            .map(|(&page_id, _)| page_id)
        {
            let mut locations = Vec::with_capacity(1);
            let reused = self.take_recycled_from_page(page_id, 1, &mut locations);
            self.allocation_stats.cold_partial_page = self
                .allocation_stats
                .cold_partial_page
                .saturating_add(reused as u64);
            return Ok(locations[0]);
        }
        if self.next_unused < self.capacity {
            let slot = self.next_unused;
            self.next_unused += 1;
            self.reserve_slot(slot);
            self.allocation_stats.empty_page = self.allocation_stats.empty_page.saturating_add(1);
            return Ok(slot);
        }
        Err(AnnError::log_index_error(
            "disk reservoir state has no free record slots".to_string(),
        ))
    }

    fn allocate_batch(
        &mut self,
        count: usize,
        resident_pages: &BTreeSet<u64>,
    ) -> AnnResult<Vec<u64>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let count_u64 = u64::try_from(count)?;
        let free_slots = self
            .capacity
            .saturating_sub(self.next_unused)
            .saturating_add(self.recycled_count);
        if count_u64 > free_slots {
            return Err(AnnError::log_index_error(
                "disk reservoir state has no free record slots".to_string(),
            ));
        }

        let mut locations = Vec::with_capacity(count);
        let records_per_page = u64::try_from(self.records_per_page)?;

        // Finish the never-used tail of an already-open page before touching
        // another page.
        let tail_offset = self.next_unused % records_per_page;
        if tail_offset != 0 {
            let tail_slots = (records_per_page - tail_offset)
                .min(self.capacity.saturating_sub(self.next_unused))
                .min(u64::try_from(count - locations.len())?);
            self.allocate_unused(tail_slots, &mut locations);
            self.allocation_stats.empty_page =
                self.allocation_stats.empty_page.saturating_add(tail_slots);
        }

        // Reuse free slots from pages already present in the page cache.
        let resident_candidates = self.recycled_pages_by_free_count(
            resident_pages
                .iter()
                .copied()
                .filter(|page_id| self.recycled_by_page.contains_key(page_id)),
        );
        for page_id in resident_candidates {
            let remaining = count - locations.len();
            if remaining == 0 {
                break;
            }
            let reused = self.take_recycled_from_page(page_id, remaining, &mut locations);
            self.allocation_stats.resident_partial_page = self
                .allocation_stats
                .resident_partial_page
                .saturating_add(reused as u64);
        }

        // New slots are contiguous and therefore naturally page-packed.
        let unused = self
            .capacity
            .saturating_sub(self.next_unused)
            .min(u64::try_from(count - locations.len())?);
        self.allocate_unused(unused, &mut locations);
        self.allocation_stats.empty_page = self.allocation_stats.empty_page.saturating_add(unused);

        // Only after sequential space is exhausted do we reopen cold partial
        // pages, preferring the page that satisfies most of the cohort.
        let cold_candidates = self.recycled_pages_by_free_count(
            self.recycled_by_page
                .keys()
                .copied()
                .filter(|page_id| !resident_pages.contains(page_id)),
        );
        for page_id in cold_candidates {
            let remaining = count - locations.len();
            if remaining == 0 {
                break;
            }
            let reused = self.take_recycled_from_page(page_id, remaining, &mut locations);
            self.allocation_stats.cold_partial_page = self
                .allocation_stats
                .cold_partial_page
                .saturating_add(reused as u64);
        }

        debug_assert_eq!(locations.len(), count);
        Ok(locations)
    }

    fn recycle(&mut self, slot: u64) -> AnnResult<()> {
        let (page_id, slot_index) = self.slot_parts(slot)?;
        if self
            .reserved_by_page
            .get(&page_id)
            .is_some_and(|slots| slots.contains(&slot_index))
        {
            return Err(AnnError::log_index_error(
                "cannot recycle a reserved reservoir slot".to_string(),
            ));
        }
        if !self
            .recycled_by_page
            .entry(page_id)
            .or_default()
            .insert(slot_index)
        {
            return Err(AnnError::log_index_error(
                "disk reservoir slot was recycled twice".to_string(),
            ));
        }
        self.recycled_count = self.recycled_count.saturating_add(1);
        Ok(())
    }

    fn commit(&mut self, slots: &[u64]) -> AnnResult<()> {
        self.validate_reserved_slots(slots)?;
        for &slot in slots {
            self.remove_reserved(slot)?;
        }
        Ok(())
    }

    fn cancel(&mut self, slots: &[u64]) -> AnnResult<()> {
        self.validate_reserved_slots(slots)?;
        for &slot in slots {
            let (page_id, slot_index) = self.slot_parts(slot)?;
            self.remove_reserved(slot)?;
            let inserted = self
                .recycled_by_page
                .entry(page_id)
                .or_default()
                .insert(slot_index);
            debug_assert!(inserted);
            self.recycled_count = self.recycled_count.saturating_add(1);
        }
        Ok(())
    }

    fn reused_allocations(&self) -> u64 {
        self.reused_allocations
    }

    fn allocation_stats(&self) -> PageAllocationStats {
        self.allocation_stats
    }

    fn page_has_committed_slots(&self, page_id: u64) -> bool {
        let records_per_page = self.records_per_page as u64;
        let page_start = page_id.saturating_mul(records_per_page);
        let page_end = page_start
            .saturating_add(records_per_page)
            .min(self.next_unused);
        (page_start..page_end).any(|slot| {
            let slot_index = (slot - page_start) as usize;
            !self
                .recycled_by_page
                .get(&page_id)
                .is_some_and(|slots| slots.contains(&slot_index))
                && !self
                    .reserved_by_page
                    .get(&page_id)
                    .is_some_and(|slots| slots.contains(&slot_index))
        })
    }

    fn allocate_unused(&mut self, count: u64, locations: &mut Vec<u64>) {
        let end = self.next_unused + count;
        for slot in self.next_unused..end {
            self.reserve_slot(slot);
            locations.push(slot);
        }
        self.next_unused = end;
    }

    fn recycled_pages_by_free_count(&self, page_ids: impl Iterator<Item = u64>) -> Vec<u64> {
        let mut pages = page_ids
            .filter_map(|page_id| {
                self.recycled_by_page
                    .get(&page_id)
                    .map(|slots| (page_id, slots.len()))
            })
            .collect::<Vec<_>>();
        pages.sort_unstable_by(|(left_page, left_count), (right_page, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_page.cmp(right_page))
        });
        pages.into_iter().map(|(page_id, _)| page_id).collect()
    }

    fn take_recycled_from_page(
        &mut self,
        page_id: u64,
        count: usize,
        locations: &mut Vec<u64>,
    ) -> usize {
        let Some(slots) = self.recycled_by_page.get_mut(&page_id) else {
            return 0;
        };
        let selected = slots.iter().copied().take(count).collect::<Vec<_>>();
        for slot_index in &selected {
            slots.remove(slot_index);
        }
        if slots.is_empty() {
            self.recycled_by_page.remove(&page_id);
        }
        for slot_index in &selected {
            let slot = page_id * self.records_per_page as u64 + *slot_index as u64;
            self.reserve_slot(slot);
            locations.push(slot);
        }
        let reused = selected.len() as u64;
        self.recycled_count -= reused;
        self.reused_allocations = self.reused_allocations.saturating_add(reused);
        selected.len()
    }

    fn reserve_slot(&mut self, slot: u64) {
        let page_id = slot / self.records_per_page as u64;
        let slot_index = (slot % self.records_per_page as u64) as usize;
        let inserted = self
            .reserved_by_page
            .entry(page_id)
            .or_default()
            .insert(slot_index);
        debug_assert!(inserted);
    }

    fn validate_reserved_slots(&self, slots: &[u64]) -> AnnResult<()> {
        let mut unique = BTreeSet::new();
        for &slot in slots {
            let (page_id, slot_index) = self.slot_parts(slot)?;
            if !unique.insert(slot)
                || !self
                    .reserved_by_page
                    .get(&page_id)
                    .is_some_and(|reserved| reserved.contains(&slot_index))
            {
                return Err(AnnError::log_index_error(
                    "disk reservoir slot is not reserved".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn remove_reserved(&mut self, slot: u64) -> AnnResult<()> {
        let (page_id, slot_index) = self.slot_parts(slot)?;
        let slots = self.reserved_by_page.get_mut(&page_id).ok_or_else(|| {
            AnnError::log_index_error("disk reservoir slot is not reserved".to_string())
        })?;
        if !slots.remove(&slot_index) {
            return Err(AnnError::log_index_error(
                "disk reservoir slot is not reserved".to_string(),
            ));
        }
        if slots.is_empty() {
            self.reserved_by_page.remove(&page_id);
        }
        Ok(())
    }

    fn slot_parts(&self, slot: u64) -> AnnResult<(u64, usize)> {
        if slot >= self.next_unused {
            return Err(AnnError::log_index_error(
                "reservoir slot has not been allocated".to_string(),
            ));
        }
        let records_per_page = u64::try_from(self.records_per_page)?;
        Ok((
            slot / records_per_page,
            usize::try_from(slot % records_per_page)?,
        ))
    }
}

#[derive(Clone, Debug)]
struct WriteCompletion {
    source_id: u32,
    generation: u32,
    new_location: PackedLocation,
    old_location: PackedLocation,
    owner: WriteCompletionOwner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteCompletionOwner {
    ActiveSource,
    Detached,
}

impl WriteCompletion {
    fn success(
        source_id: u32,
        generation: u32,
        new_location: PackedLocation,
        old_location: PackedLocation,
    ) -> Self {
        Self {
            source_id,
            generation,
            new_location,
            old_location,
            owner: WriteCompletionOwner::ActiveSource,
        }
    }

    fn detached(
        source_id: u32,
        generation: u32,
        new_location: PackedLocation,
        old_location: PackedLocation,
    ) -> Self {
        Self {
            source_id,
            generation,
            new_location,
            old_location,
            owner: WriteCompletionOwner::Detached,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionDisposition {
    Published { old_location: PackedLocation },
    Stale,
}

#[derive(Debug)]
struct SourceWriteState {
    committed_location: PackedLocation,
    latest_generation: u32,
    dirty_generation: Option<u32>,
    inflight_generation: Option<u32>,
    inflight_location: PackedLocation,
}

impl SourceWriteState {
    fn new(committed_location: PackedLocation) -> Self {
        Self::with_generation(committed_location, 0)
    }

    fn with_generation(committed_location: PackedLocation, generation: u32) -> Self {
        Self {
            committed_location,
            latest_generation: generation,
            dirty_generation: None,
            inflight_generation: None,
            inflight_location: PackedLocation::INVALID,
        }
    }

    fn mark_dirty(&mut self, generation: u32) {
        debug_assert!(generation > self.latest_generation);
        self.latest_generation = generation;
        self.dirty_generation = Some(generation);
    }

    fn mark_inflight(&mut self, generation: u32, new_location: PackedLocation) -> AnnResult<()> {
        if self.inflight_generation.is_some()
            || self.dirty_generation != Some(generation)
            || new_location.is_invalid()
        {
            return Err(AnnError::log_index_error(
                "invalid disk reservoir writeback transition".to_string(),
            ));
        }
        self.inflight_generation = Some(generation);
        self.inflight_location = new_location;
        self.dirty_generation = None;
        Ok(())
    }

    fn apply_completion(
        &mut self,
        completion: &WriteCompletion,
    ) -> AnnResult<CompletionDisposition> {
        if self.inflight_generation != Some(completion.generation)
            || self.inflight_location != completion.new_location
            || self.committed_location != completion.old_location
        {
            return Err(AnnError::log_index_error(
                "disk reservoir writeback completion does not match in-flight state".to_string(),
            ));
        }
        self.inflight_generation = None;
        self.inflight_location = PackedLocation::INVALID;
        if self.latest_generation != completion.generation {
            return Ok(CompletionDisposition::Stale);
        }
        self.committed_location = completion.new_location;
        Ok(CompletionDisposition::Published {
            old_location: completion.old_location,
        })
    }

    #[inline]
    fn committed_location(&self) -> PackedLocation {
        self.committed_location
    }

    #[inline]
    fn is_dirty_and_idle(&self) -> bool {
        self.dirty_generation.is_some() && self.inflight_generation.is_none()
    }

    #[inline]
    fn is_clean(&self) -> bool {
        self.dirty_generation.is_none() && self.inflight_generation.is_none()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiskReservoirHeader {
    pub(crate) source_start: u32,
    pub(crate) source_end: u32,
    pub(crate) layout: DiskReservoirLayout,
    pub(crate) policy_key: [u8; 32],
}

impl DiskReservoirHeader {
    pub(crate) fn encode_page(&self) -> AnnResult<Vec<u8>> {
        self.validate()?;
        let mut output = vec![0_u8; self.layout.page_bytes];
        output[0..8].copy_from_slice(&STORE_MAGIC);
        write_u16(&mut output, 8, STORE_VERSION);
        write_u32(&mut output, 12, u32::try_from(self.layout.page_bytes)?);
        write_u32(&mut output, 16, self.source_start);
        write_u32(&mut output, 20, self.source_end);
        write_u32(&mut output, 24, u32::try_from(self.layout.l_max)?);
        write_u32(&mut output, 28, u32::try_from(self.layout.record_bytes)?);
        write_u32(&mut output, 32, u32::try_from(self.layout.page_bytes)?);
        write_u32(
            &mut output,
            36,
            u32::try_from(self.layout.records_per_page)?,
        );
        output[40..72].copy_from_slice(&self.policy_key);
        Ok(output)
    }

    pub(crate) fn decode(input: &[u8], expected: &Self) -> AnnResult<Self> {
        expected.validate()?;
        if input.len() != expected.layout.page_bytes
            || input.len() < HEADER_FIELDS_BYTES
            || input[0..8] != STORE_MAGIC
            || read_u16(input, 8) != STORE_VERSION
            || input[10..12] != [0, 0]
            || usize::try_from(read_u32(input, 12))? != input.len()
            || input[HEADER_FIELDS_BYTES..].iter().any(|&byte| byte != 0)
        {
            return Err(header_mismatch_error());
        }

        let l_max = usize::try_from(read_u32(input, 24))?;
        let page_bytes = usize::try_from(read_u32(input, 32))?;
        let layout = DiskReservoirLayout::new(l_max, page_bytes)?;
        if layout.record_bytes != usize::try_from(read_u32(input, 28))?
            || layout.page_bytes != page_bytes
            || layout.records_per_page != usize::try_from(read_u32(input, 36))?
        {
            return Err(header_mismatch_error());
        }
        let mut policy_key = [0_u8; 32];
        policy_key.copy_from_slice(&input[40..72]);
        let decoded = Self {
            source_start: read_u32(input, 16),
            source_end: read_u32(input, 20),
            layout,
            policy_key,
        };
        decoded.validate()?;
        if decoded != *expected {
            return Err(header_mismatch_error());
        }
        Ok(decoded)
    }

    fn validate(&self) -> AnnResult<()> {
        if self.source_start > self.source_end {
            return Err(AnnError::log_index_config_error(
                "source_range".to_string(),
                "disk reservoir source range is reversed".to_string(),
            ));
        }
        let canonical = DiskReservoirLayout::new(self.layout.l_max, self.layout.page_bytes)?;
        if canonical != self.layout {
            return Err(AnnError::log_index_config_error(
                "layout".to_string(),
                "disk reservoir layout is inconsistent".to_string(),
            ));
        }
        Ok(())
    }
}

trait ReservoirPageIo: Send + Sync {
    fn read_page(&self, page_id: u64, output: &mut [u8]) -> AnnResult<()>;
    fn write_page(&self, page_id: u64, bytes: &[u8]) -> AnnResult<()>;
    fn sync_all(&self) -> AnnResult<()>;
}

struct DirectReservoirPageIo {
    file: DirectIoFile,
    data_offset: u64,
    page_bytes: usize,
}

impl ReservoirPageIo for DirectReservoirPageIo {
    fn read_page(&self, page_id: u64, output: &mut [u8]) -> AnnResult<()> {
        if output.len() != self.page_bytes {
            return Err(AnnError::log_index_error(
                "disk reservoir page read length mismatch".to_string(),
            ));
        }
        self.file.read_exact_at(output, self.page_offset(page_id)?)
    }

    fn write_page(&self, page_id: u64, bytes: &[u8]) -> AnnResult<()> {
        if bytes.len() != self.page_bytes {
            return Err(AnnError::log_index_error(
                "disk reservoir page write length mismatch".to_string(),
            ));
        }
        self.file.write_all_at(bytes, self.page_offset(page_id)?)
    }

    fn sync_all(&self) -> AnnResult<()> {
        self.file.sync_all()
    }
}

impl DirectReservoirPageIo {
    fn page_offset(&self, page_id: u64) -> AnnResult<u64> {
        page_id
            .checked_mul(u64::try_from(self.page_bytes)?)
            .and_then(|offset| offset.checked_add(self.data_offset))
            .ok_or_else(|| {
                AnnError::log_index_error("disk reservoir page offset overflows u64".to_string())
            })
    }
}

#[cfg(test)]
struct FailingFirstWritePageIo {
    writes: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
struct FailingFirstReadPageIo {
    reads: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl ReservoirPageIo for FailingFirstWritePageIo {
    fn read_page(&self, _page_id: u64, output: &mut [u8]) -> AnnResult<()> {
        output.fill(0);
        Ok(())
    }

    fn write_page(&self, _page_id: u64, _bytes: &[u8]) -> AnnResult<()> {
        let write = self.writes.fetch_add(1, Ordering::SeqCst);
        if write == 0 {
            return Err(AnnError::log_index_error(
                "injected page write failure".to_string(),
            ));
        }
        Ok(())
    }

    fn sync_all(&self) -> AnnResult<()> {
        Ok(())
    }
}

#[cfg(test)]
impl ReservoirPageIo for FailingFirstReadPageIo {
    fn read_page(&self, _page_id: u64, output: &mut [u8]) -> AnnResult<()> {
        let read = self.reads.fetch_add(1, Ordering::SeqCst);
        if read == 0 {
            return Err(AnnError::log_index_error(
                "injected page read failure".to_string(),
            ));
        }
        output.fill(0);
        Ok(())
    }

    fn write_page(&self, _page_id: u64, _bytes: &[u8]) -> AnnResult<()> {
        Ok(())
    }

    fn sync_all(&self) -> AnnResult<()> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DiskReservoirStoreConfig {
    pub(crate) root: PathBuf,
    pub(crate) source_start: usize,
    pub(crate) source_end: usize,
    pub(crate) shard_points: usize,
    pub(crate) l_max: usize,
    pub(crate) page_cache_bytes: usize,
    pub(crate) writeback_bytes: usize,
    pub(crate) decode_bytes: usize,
    pub(crate) writeback_workers: usize,
    pub(crate) overprovision_ratio: f64,
    pub(crate) io: DirectIoConfig,
    pub(crate) policy_key: [u8; 32],
    pub(crate) backfill_policy: HashPruneBackfillPolicy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiskReservoirMemoryPeaks {
    pub(crate) page_cache_bytes: usize,
    pub(crate) writeback_bytes: usize,
    pub(crate) decode_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiskReservoirMemoryUsage {
    pub(crate) page_cache_bytes: usize,
    pub(crate) writeback_bytes: usize,
    pub(crate) decode_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiskReservoirIoStats {
    pub(crate) records_loaded: u64,
    pub(crate) page_cache_hits: u64,
    pub(crate) page_cache_misses: u64,
    pub(crate) pages_read: u64,
    pub(crate) pages_written: u64,
    pub(crate) slots_reused: u64,
    pub(crate) alloc_empty_page: u64,
    pub(crate) alloc_resident_partial_page: u64,
    pub(crate) alloc_cold_partial_page: u64,
}

#[derive(Default)]
struct DiskReservoirIoCounters {
    records_loaded: AtomicU64,
    page_cache_hits: AtomicU64,
    page_cache_misses: AtomicU64,
    pages_read: AtomicU64,
    pages_written: AtomicU64,
}

#[derive(Debug, Default)]
struct MemoryAccountingState {
    page_cache_bytes: usize,
    page_cache_peak: usize,
    decode_bytes: usize,
    decode_peak: usize,
}

struct MemoryAccounting {
    page_cache_limit: usize,
    decode_limit: usize,
    state: Mutex<MemoryAccountingState>,
    changed: Condvar,
}

impl MemoryAccounting {
    fn new(page_cache_limit: usize, decode_limit: usize) -> Self {
        Self {
            page_cache_limit,
            decode_limit,
            state: Mutex::new(MemoryAccountingState::default()),
            changed: Condvar::new(),
        }
    }

    fn try_reserve_page(&self, bytes: usize) -> bool {
        let mut state = self.state.lock();
        let Some(next) = state.page_cache_bytes.checked_add(bytes) else {
            return false;
        };
        if next > self.page_cache_limit {
            return false;
        }
        state.page_cache_bytes = next;
        state.page_cache_peak = state.page_cache_peak.max(next);
        true
    }

    fn release_page(&self, bytes: usize) {
        let mut state = self.state.lock();
        debug_assert!(state.page_cache_bytes >= bytes);
        state.page_cache_bytes -= bytes;
        self.changed.notify_all();
    }

    fn try_reserve_decode(&self, bytes: usize) -> bool {
        let mut state = self.state.lock();
        let Some(next) = state.decode_bytes.checked_add(bytes) else {
            return false;
        };
        if next > self.decode_limit {
            return false;
        }
        state.decode_bytes = next;
        state.decode_peak = state.decode_peak.max(next);
        true
    }

    fn release_decode(&self, bytes: usize) {
        let mut state = self.state.lock();
        debug_assert!(state.decode_bytes >= bytes);
        state.decode_bytes -= bytes;
        self.changed.notify_all();
    }

    fn peaks(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.page_cache_peak, state.decode_peak)
    }

    fn usage(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.page_cache_bytes, state.decode_bytes)
    }
}

struct PageWriteTask {
    shard_id: usize,
    page_id: u64,
    io: Arc<dyn ReservoirPageIo>,
    bytes: Arc<Vec<u8>>,
    record_completions: Vec<WriteCompletion>,
}

struct PageWriteResult {
    shard_id: usize,
    page_id: u64,
    record_completions: Vec<WriteCompletion>,
    error: Option<String>,
}

enum WriteMessage {
    Page(PageWriteTask),
    Shutdown,
}

#[derive(Debug, Default)]
struct WriteExecutorState {
    bytes: usize,
    peak_bytes: usize,
    pending_tasks: usize,
    first_error: Option<String>,
}

struct WriteExecutorShared {
    state: Mutex<WriteExecutorState>,
    changed: Condvar,
    completions: Mutex<VecDeque<PageWriteResult>>,
}

struct WriteExecutor {
    sender: Sender<WriteMessage>,
    shared: Arc<WriteExecutorShared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    worker_count: usize,
    byte_limit: usize,
}

impl WriteExecutor {
    fn new(
        worker_count: usize,
        byte_limit: usize,
        io_counters: Arc<DiskReservoirIoCounters>,
    ) -> Self {
        let worker_count = worker_count.max(1);
        let (sender, receiver) = bounded::<WriteMessage>(worker_count.saturating_mul(2).max(1));
        let shared = Arc::new(WriteExecutorShared {
            state: Mutex::new(WriteExecutorState::default()),
            changed: Condvar::new(),
            completions: Mutex::new(VecDeque::new()),
        });
        let workers = (0..worker_count)
            .map(|worker_id| {
                let receiver = receiver.clone();
                let shared = Arc::clone(&shared);
                let io_counters = Arc::clone(&io_counters);
                thread::Builder::new()
                    .name(format!("hashprune-state-writer-{worker_id}"))
                    .spawn(move || writeback_worker(receiver, shared, io_counters))
                    .expect("failed to spawn disk reservoir writeback worker")
            })
            .collect();
        Self {
            sender,
            shared,
            workers: Mutex::new(workers),
            worker_count,
            byte_limit,
        }
    }

    fn submit(&self, task: PageWriteTask) -> AnnResult<()> {
        let bytes = task.bytes.len();
        let mut state = self.shared.state.lock();
        while state
            .bytes
            .checked_add(bytes)
            .is_none_or(|next| next > self.byte_limit)
            && state.first_error.is_none()
        {
            self.shared.changed.wait(&mut state);
        }
        if let Some(error) = &state.first_error {
            return Err(AnnError::log_index_error(format!(
                "disk reservoir writeback failed: {error}"
            )));
        }
        state.bytes += bytes;
        state.peak_bytes = state.peak_bytes.max(state.bytes);
        state.pending_tasks += 1;
        drop(state);

        if self.sender.send(WriteMessage::Page(task)).is_err() {
            let mut state = self.shared.state.lock();
            state.bytes -= bytes;
            state.pending_tasks -= 1;
            self.shared.changed.notify_all();
            return Err(AnnError::log_index_error(
                "disk reservoir writeback workers stopped unexpectedly".to_string(),
            ));
        }
        Ok(())
    }

    fn take_completions(&self) -> Vec<PageWriteResult> {
        self.shared.completions.lock().drain(..).collect()
    }

    fn wait_for_change(&self) {
        let mut state = self.shared.state.lock();
        if state.pending_tasks != 0 {
            self.shared.changed.wait(&mut state);
        }
    }

    fn pending_tasks(&self) -> usize {
        self.shared.state.lock().pending_tasks
    }

    fn first_error(&self) -> Option<String> {
        self.shared.state.lock().first_error.clone()
    }

    fn peak_bytes(&self) -> usize {
        self.shared.state.lock().peak_bytes
    }

    fn current_bytes(&self) -> usize {
        self.shared.state.lock().bytes
    }
}

impl Drop for WriteExecutor {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock();
            while state.pending_tasks != 0 {
                self.shared.changed.wait(&mut state);
            }
        }
        for _ in 0..self.worker_count {
            let _ = self.sender.send(WriteMessage::Shutdown);
        }
        for worker in self.workers.get_mut().drain(..) {
            let _ = worker.join();
        }
    }
}

fn writeback_worker(
    receiver: Receiver<WriteMessage>,
    shared: Arc<WriteExecutorShared>,
    io_counters: Arc<DiskReservoirIoCounters>,
) {
    while let Ok(message) = receiver.recv() {
        let WriteMessage::Page(task) = message else {
            break;
        };
        let bytes = task.bytes.len();
        let error = task
            .io
            .write_page(task.page_id, task.bytes.as_slice())
            .err()
            .map(|error| error.to_string());
        if error.is_none() {
            io_counters.pages_written.fetch_add(1, Ordering::Relaxed);
        }
        shared.completions.lock().push_back(PageWriteResult {
            shard_id: task.shard_id,
            page_id: task.page_id,
            record_completions: task.record_completions,
            error: error.clone(),
        });
        let mut state = shared.state.lock();
        state.bytes -= bytes;
        state.pending_tasks -= 1;
        if state.first_error.is_none() {
            state.first_error = error;
        }
        shared.changed.notify_all();
    }
}

struct ActiveSource {
    reservoir: HashPruneReservoir,
    write_state: SourceWriteState,
}

#[derive(Default)]
struct PageState {
    inflight: Option<Arc<Vec<u8>>>,
    dirty: Option<Vec<u8>>,
    dirty_completions: Vec<WriteCompletion>,
}

struct ShardMutable {
    allocator: SlotAllocator,
    active_sources: HashMap<u32, ActiveSource>,
    pages: HashMap<u64, PageState>,
}

struct ShardStore {
    header: DiskReservoirHeader,
    io: Arc<dyn ReservoirPageIo>,
    locations: Vec<AtomicU64>,
    mutable: Mutex<ShardMutable>,
}

enum StageOutcome {
    Staged,
    AlreadyInflight,
    WouldBlock,
}

pub(crate) struct DiskReservoirStore {
    config: DiskReservoirStoreConfig,
    layout: DiskReservoirLayout,
    shards: Vec<Arc<ShardStore>>,
    accounting: Arc<MemoryAccounting>,
    executor: WriteExecutor,
    io_counters: Arc<DiskReservoirIoCounters>,
    detached_completions: Vec<Mutex<VecDeque<ReservoirWritebackCompletion>>>,
}

impl DiskReservoirStore {
    pub(crate) fn create(config: DiskReservoirStoreConfig) -> AnnResult<Self> {
        let layout = validate_store_config(&config)?;
        fs::create_dir_all(&config.root)?;
        let io_counters = Arc::new(DiskReservoirIoCounters::default());
        let executor = WriteExecutor::new(
            config.writeback_workers,
            config.writeback_bytes,
            Arc::clone(&io_counters),
        );
        let accounting = Arc::new(MemoryAccounting::new(
            config.page_cache_bytes,
            config.decode_bytes,
        ));
        let mut shards = Vec::new();
        let mut shard_start = config.source_start;
        while shard_start < config.source_end {
            let shard_end = shard_start
                .saturating_add(config.shard_points)
                .min(config.source_end);
            let header = DiskReservoirHeader {
                source_start: u32::try_from(shard_start)?,
                source_end: u32::try_from(shard_end)?,
                layout,
                policy_key: config.policy_key,
            };
            let (physical_capacity, page_count) =
                physical_capacity(shard_end - shard_start, layout, config.overprovision_ratio)?;
            let path = shard_path(&config.root, shard_start, shard_end);
            let file = DirectIoFile::create_write(&path, config.io)?;
            let file_bytes = u64::try_from(page_count.checked_add(1).ok_or_else(|| {
                AnnError::log_index_error(
                    "disk reservoir shard page count overflows usize".to_string(),
                )
            })?)?
            .checked_mul(u64::try_from(layout.page_bytes)?)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "disk reservoir shard file length overflows u64".to_string(),
                )
            })?;
            file.set_len(file_bytes)?;
            file.write_all_at(&header.encode_page()?, 0)?;
            file.sync_all()?;
            let io: Arc<dyn ReservoirPageIo> = Arc::new(DirectReservoirPageIo {
                file,
                data_offset: u64::try_from(layout.page_bytes)?,
                page_bytes: layout.page_bytes,
            });
            shards.push(Arc::new(new_shard_store(header, io, physical_capacity)?));
            shard_start = shard_end;
        }
        let detached_completions = (0..shards.len())
            .map(|_| Mutex::new(VecDeque::new()))
            .collect();
        Ok(Self {
            config,
            layout,
            shards,
            accounting,
            executor,
            io_counters,
            detached_completions,
        })
    }

    #[cfg(test)]
    pub(crate) fn create_with_failing_first_write_for_test(
        config: DiskReservoirStoreConfig,
    ) -> AnnResult<Self> {
        Self::create_for_test(
            config,
            Arc::new(FailingFirstWritePageIo {
                writes: std::sync::atomic::AtomicUsize::new(0),
            }),
        )
    }

    #[cfg(test)]
    pub(crate) fn create_with_failing_first_read_for_test(
        config: DiskReservoirStoreConfig,
    ) -> AnnResult<Self> {
        Self::create_for_test(
            config,
            Arc::new(FailingFirstReadPageIo {
                reads: std::sync::atomic::AtomicUsize::new(0),
            }),
        )
    }

    #[cfg(test)]
    fn create_for_test(
        config: DiskReservoirStoreConfig,
        io: Arc<dyn ReservoirPageIo>,
    ) -> AnnResult<Self> {
        let layout = validate_store_config(&config)?;
        if config.shard_points < config.source_end - config.source_start {
            return Err(AnnError::log_index_config_error(
                "shard_points".to_string(),
                "injected test page I/O supports exactly one shard".to_string(),
            ));
        }
        let header = DiskReservoirHeader {
            source_start: u32::try_from(config.source_start)?,
            source_end: u32::try_from(config.source_end)?,
            layout,
            policy_key: config.policy_key,
        };
        let (physical_capacity, _) = physical_capacity(
            config.source_end - config.source_start,
            layout,
            config.overprovision_ratio,
        )?;
        let shard = Arc::new(new_shard_store(header, io, physical_capacity)?);
        let io_counters = Arc::new(DiskReservoirIoCounters::default());
        Ok(Self {
            accounting: Arc::new(MemoryAccounting::new(
                config.page_cache_bytes,
                config.decode_bytes,
            )),
            executor: WriteExecutor::new(
                config.writeback_workers,
                config.writeback_bytes,
                Arc::clone(&io_counters),
            ),
            config,
            layout,
            shards: vec![shard],
            io_counters,
            detached_completions: vec![Mutex::new(VecDeque::new())],
        })
    }

    pub(crate) fn update_source<F>(&self, source: u32, update: F) -> AnnResult<()>
    where
        F: FnOnce(&mut HashPruneReservoir) -> AnnResult<()>,
    {
        self.update_source_deferred(source, update)?;
        self.flush_update_batch()
    }

    pub(crate) fn load_sources_grouped(&self, sources: &[u32]) -> AnnResult<Vec<LoadedReservoir>> {
        self.process_completions()?;
        let mut loaded = (0..sources.len()).map(|_| None).collect::<Vec<_>>();
        let mut by_shard: BTreeMap<usize, Vec<(usize, u32)>> = BTreeMap::new();
        for (output_index, &source) in sources.iter().enumerate() {
            by_shard
                .entry(self.shard_id(source)?)
                .or_default()
                .push((output_index, source));
        }

        for (shard_id, shard_sources) in by_shard {
            let shard = &self.shards[shard_id];
            let mutable = shard.mutable.lock();
            if shard_sources
                .iter()
                .any(|(_, source)| mutable.active_sources.contains_key(source))
            {
                return Err(AnnError::log_index_error(
                    "grouped reservoir read cannot observe an active direct-first source"
                        .to_string(),
                ));
            }

            let mut page_requests: BTreeMap<u64, Vec<(usize, u32, usize, PackedLocation)>> =
                BTreeMap::new();
            for (output_index, source) in shard_sources {
                let local = self.shard_local_index(shard_id, source)?;
                let location =
                    PackedLocation::from_raw(shard.locations[local].load(Ordering::Acquire));
                if location.is_invalid() {
                    loaded[output_index] = Some(LoadedReservoir {
                        source,
                        generation: 0,
                        location,
                        reservoir: HashPruneReservoir::new(self.layout.l_max),
                    });
                    continue;
                }
                let (page_id, slot_index) = location.decode(self.layout.records_per_page)?;
                page_requests.entry(page_id).or_default().push((
                    output_index,
                    source,
                    slot_index,
                    location,
                ));
            }

            for (page_id, requests) in page_requests {
                if !self.accounting.try_reserve_page(self.layout.page_bytes) {
                    return Err(AnnError::log_index_error(
                        "disk reservoir grouped-read page window exceeds the page-cache budget"
                            .to_string(),
                    ));
                }
                let result = (|| {
                    let page = if let Some(bytes) = mutable
                        .pages
                        .get(&page_id)
                        .and_then(|page| page.dirty.as_ref())
                    {
                        self.io_counters
                            .page_cache_hits
                            .fetch_add(1, Ordering::Relaxed);
                        bytes.clone()
                    } else if let Some(bytes) = mutable
                        .pages
                        .get(&page_id)
                        .and_then(|page| page.inflight.as_ref())
                    {
                        self.io_counters
                            .page_cache_hits
                            .fetch_add(1, Ordering::Relaxed);
                        bytes.as_ref().clone()
                    } else {
                        self.io_counters
                            .page_cache_misses
                            .fetch_add(1, Ordering::Relaxed);
                        let mut bytes = vec![0_u8; self.layout.page_bytes];
                        shard.io.read_page(page_id, &mut bytes)?;
                        self.io_counters.pages_read.fetch_add(1, Ordering::Relaxed);
                        bytes
                    };
                    for (output_index, source, slot_index, location) in requests {
                        let start = slot_index * self.layout.record_bytes;
                        let end = start + self.layout.record_bytes;
                        let record = DiskReservoirRecord::decode(&self.layout, &page[start..end])?;
                        if record.source_id != source {
                            return Err(AnnError::log_index_error(format!(
                                "grouped reservoir read expected source {source}, found {}",
                                record.source_id
                            )));
                        }
                        let generation = record.generation;
                        loaded[output_index] = Some(LoadedReservoir {
                            source,
                            generation,
                            location,
                            reservoir: record
                                .into_reservoir(self.layout.l_max, self.config.backfill_policy)?,
                        });
                        self.io_counters
                            .records_loaded
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(())
                })();
                self.accounting.release_page(self.layout.page_bytes);
                result?;
            }
        }

        loaded
            .into_iter()
            .map(|entry| {
                entry.ok_or_else(|| {
                    AnnError::log_index_error(
                        "grouped reservoir read did not produce every requested source".to_string(),
                    )
                })
            })
            .collect()
    }

    pub(crate) fn stage_writeback_cohort(&self, writes: Vec<ReservoirWriteback>) -> AnnResult<()> {
        self.process_completions()?;
        let mut seen = BTreeSet::new();
        let mut by_shard: BTreeMap<usize, Vec<ReservoirWriteback>> = BTreeMap::new();
        for write in writes {
            if !seen.insert(write.source)
                || write.record.source_id != write.source
                || write.record.generation != write.generation
            {
                return Err(AnnError::log_index_error(
                    "invalid or duplicate reservoir writeback record".to_string(),
                ));
            }
            by_shard
                .entry(self.shard_id(write.source)?)
                .or_default()
                .push(write);
        }
        for (shard_id, shard_writes) in by_shard {
            self.stage_shard_writeback_cohort(shard_id, shard_writes)?;
        }
        self.submit_ready_pages()
    }

    pub(crate) fn take_writeback_completions_for_shard(
        &self,
        shard_id: usize,
    ) -> AnnResult<Vec<ReservoirWritebackCompletion>> {
        self.process_completions()?;
        let pending = self.detached_completions.get(shard_id).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "writeback completion shard {shard_id} is outside {} store shards",
                self.shards.len()
            ))
        })?;
        Ok(pending.lock().drain(..).collect())
    }

    #[cfg(test)]
    fn detached_completion_queue_lengths_for_test(&self) -> Vec<usize> {
        self.detached_completions
            .iter()
            .map(|pending| pending.lock().len())
            .collect()
    }

    pub(crate) fn wait_for_writeback_progress(&self) -> AnnResult<()> {
        self.wait_for_progress()
    }

    pub(crate) fn update_source_deferred<F>(&self, source: u32, update: F) -> AnnResult<()>
    where
        F: FnOnce(&mut HashPruneReservoir) -> AnnResult<()>,
    {
        self.process_completions()?;
        let shard_id = self.shard_id(source)?;
        let shard = &self.shards[shard_id];
        let mut update = Some(update);

        loop {
            let mut mutable = shard.mutable.lock();
            if mutable.active_sources.contains_key(&source) {
                let active = mutable
                    .active_sources
                    .get_mut(&source)
                    .expect("checked active source");
                let snapshot = active.reservoir.state_snapshot();
                let result = update
                    .take()
                    .expect("source update closure must run exactly once")(
                    &mut active.reservoir
                );
                if let Err(error) = result {
                    active.reservoir = HashPruneReservoir::from_state(
                        self.layout.l_max,
                        snapshot,
                        self.config.backfill_policy,
                    )?;
                    return Err(error);
                }
                let generation = active
                    .write_state
                    .latest_generation
                    .checked_add(1)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "disk reservoir source generation overflow".to_string(),
                        )
                    })?;
                active.write_state.mark_dirty(generation);
                break;
            }

            if !self.accounting.try_reserve_decode(self.layout.record_bytes) {
                drop(mutable);
                self.flush_update_batch()?;
                self.wait_for_progress()?;
                continue;
            }

            let local = self.shard_local_index(shard_id, source)?;
            let location = PackedLocation::from_raw(shard.locations[local].load(Ordering::Acquire));
            let loaded = if location.is_invalid() {
                Ok((HashPruneReservoir::new(self.layout.l_max), 0))
            } else if !self.accounting.try_reserve_page(self.layout.page_bytes) {
                Err(None)
            } else {
                let result = self.read_reservoir_at(shard, source, location);
                self.accounting.release_page(self.layout.page_bytes);
                result.map_err(Some)
            };

            let (reservoir, generation) = match loaded {
                Ok(loaded) => loaded,
                Err(None) => {
                    self.accounting.release_decode(self.layout.record_bytes);
                    drop(mutable);
                    self.flush_update_batch()?;
                    self.wait_for_progress()?;
                    continue;
                }
                Err(Some(error)) => {
                    self.accounting.release_decode(self.layout.record_bytes);
                    return Err(error);
                }
            };
            mutable.active_sources.insert(
                source,
                ActiveSource {
                    reservoir,
                    write_state: SourceWriteState::with_generation(location, generation),
                },
            );
            let active = mutable
                .active_sources
                .get_mut(&source)
                .expect("new active source");
            let snapshot = active.reservoir.state_snapshot();
            let result = update
                .take()
                .expect("source update closure must run exactly once")(
                &mut active.reservoir
            );
            if let Err(error) = result {
                active.reservoir = HashPruneReservoir::from_state(
                    self.layout.l_max,
                    snapshot,
                    self.config.backfill_policy,
                )?;
                mutable.active_sources.remove(&source);
                self.accounting.release_decode(self.layout.record_bytes);
                return Err(error);
            }
            let generation = generation.checked_add(1).ok_or_else(|| {
                AnnError::log_index_error("disk reservoir source generation overflow".to_string())
            })?;
            active.write_state.mark_dirty(generation);
            break;
        }

        Ok(())
    }

    pub(crate) fn flush_update_batch(&self) -> AnnResult<()> {
        loop {
            self.process_completions()?;
            let mut blocked = false;
            for (shard_id, shard) in self.shards.iter().enumerate() {
                let sources: Vec<u32> = shard
                    .mutable
                    .lock()
                    .active_sources
                    .iter()
                    .filter_map(|(&source, active)| {
                        active.write_state.is_dirty_and_idle().then_some(source)
                    })
                    .collect();
                for source in sources {
                    if matches!(
                        self.stage_source_once(shard_id, source)?,
                        StageOutcome::WouldBlock
                    ) {
                        blocked = true;
                        break;
                    }
                }
                if blocked {
                    break;
                }
            }
            self.submit_ready_pages()?;
            if !blocked {
                return Ok(());
            }
            self.wait_for_progress()?;
        }
    }

    pub(crate) fn flush_all(&self) -> AnnResult<()> {
        loop {
            let iteration = (|| {
                self.flush_update_batch()?;
                if self.executor.pending_tasks() == 0 && !self.has_active_work() {
                    return Ok(true);
                }
                self.wait_for_progress()?;
                Ok(false)
            })();
            match iteration {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => {
                    while self.executor.pending_tasks() != 0 {
                        self.executor.wait_for_change();
                    }
                    return Err(error);
                }
            }
        }
        self.process_completions()?;
        if let Some(error) = self.executor.first_error() {
            return Err(AnnError::log_index_error(format!(
                "disk reservoir writeback failed: {error}"
            )));
        }
        for shard in &self.shards {
            shard.io.sync_all()?;
        }
        Ok(())
    }

    pub(crate) fn for_each_source_window<F>(
        &self,
        window_sources: usize,
        mut callback: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, HashPruneReservoir) -> AnnResult<()>,
    {
        if window_sources == 0 {
            return Err(AnnError::log_index_config_error(
                "window_sources".to_string(),
                "disk reservoir source window must be nonzero".to_string(),
            ));
        }
        if self.config.decode_bytes == 0 {
            for window_start in
                (self.config.source_start..self.config.source_end).step_by(window_sources)
            {
                let window_end = window_start
                    .saturating_add(window_sources)
                    .min(self.config.source_end);
                let sources = (window_start..window_end)
                    .map(u32::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                for loaded in self.load_sources_grouped(&sources)? {
                    callback(loaded.source, loaded.reservoir)?;
                }
            }
            return Ok(());
        }
        let max_window = self.config.decode_bytes / self.layout.record_bytes;
        if window_sources > max_window {
            return Err(AnnError::log_index_config_error(
                "window_sources".to_string(),
                format!(
                    "requested source window {window_sources} exceeds decoded-state capacity {max_window}"
                ),
            ));
        }
        if self.has_active_work() || self.executor.pending_tasks() != 0 {
            return Err(AnnError::log_index_error(
                "disk reservoir must be flushed before final source-order reads".to_string(),
            ));
        }

        for window_start in
            (self.config.source_start..self.config.source_end).step_by(window_sources)
        {
            let window_end = window_start
                .saturating_add(window_sources)
                .min(self.config.source_end);
            let window_len = window_end - window_start;
            let decode_charge = window_len
                .checked_mul(self.layout.record_bytes)
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "disk reservoir finalization window size overflows usize".to_string(),
                    )
                })?;
            if !self.accounting.try_reserve_decode(decode_charge) {
                return Err(AnnError::log_index_error(
                    "disk reservoir decode budget is still occupied during finalization"
                        .to_string(),
                ));
            }
            let result = self.read_source_window(window_start, window_end, &mut callback);
            self.accounting.release_decode(decode_charge);
            result?;
        }
        Ok(())
    }

    pub(crate) fn memory_peaks(&self) -> DiskReservoirMemoryPeaks {
        let (page_cache_bytes, decode_bytes) = self.accounting.peaks();
        DiskReservoirMemoryPeaks {
            page_cache_bytes,
            writeback_bytes: self.executor.peak_bytes(),
            decode_bytes,
        }
    }

    pub(crate) fn memory_usage(&self) -> DiskReservoirMemoryUsage {
        let (page_cache_bytes, decode_bytes) = self.accounting.usage();
        DiskReservoirMemoryUsage {
            page_cache_bytes,
            writeback_bytes: self.executor.current_bytes(),
            decode_bytes,
        }
    }

    pub(crate) fn io_stats(&self) -> DiskReservoirIoStats {
        let (slots_reused, allocation_stats) = self.shards.iter().fold(
            (0_u64, PageAllocationStats::default()),
            |(slots_reused, mut allocation_stats), shard| {
                let mutable = shard.mutable.lock();
                let shard_stats = mutable.allocator.allocation_stats();
                allocation_stats.empty_page = allocation_stats
                    .empty_page
                    .saturating_add(shard_stats.empty_page);
                allocation_stats.resident_partial_page = allocation_stats
                    .resident_partial_page
                    .saturating_add(shard_stats.resident_partial_page);
                allocation_stats.cold_partial_page = allocation_stats
                    .cold_partial_page
                    .saturating_add(shard_stats.cold_partial_page);
                (
                    slots_reused.saturating_add(mutable.allocator.reused_allocations()),
                    allocation_stats,
                )
            },
        );
        DiskReservoirIoStats {
            records_loaded: self.io_counters.records_loaded.load(Ordering::Relaxed),
            page_cache_hits: self.io_counters.page_cache_hits.load(Ordering::Relaxed),
            page_cache_misses: self.io_counters.page_cache_misses.load(Ordering::Relaxed),
            pages_read: self.io_counters.pages_read.load(Ordering::Relaxed),
            pages_written: self.io_counters.pages_written.load(Ordering::Relaxed),
            slots_reused,
            alloc_empty_page: allocation_stats.empty_page,
            alloc_resident_partial_page: allocation_stats.resident_partial_page,
            alloc_cold_partial_page: allocation_stats.cold_partial_page,
        }
    }

    pub(crate) fn record_bytes(&self) -> usize {
        self.layout.record_bytes
    }

    pub(crate) fn page_bytes(&self) -> usize {
        self.layout.page_bytes
    }

    pub(crate) fn artifact_paths(&self) -> Vec<PathBuf> {
        self.shards
            .iter()
            .map(|shard| {
                shard_path(
                    &self.config.root,
                    shard.header.source_start as usize,
                    shard.header.source_end as usize,
                )
            })
            .collect()
    }

    pub(crate) fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub(crate) fn close(self) -> AnnResult<()> {
        self.flush_all()?;
        drop(self);
        Ok(())
    }

    fn stage_shard_writeback_cohort(
        &self,
        shard_id: usize,
        writes: Vec<ReservoirWriteback>,
    ) -> AnnResult<()> {
        let shard = &self.shards[shard_id];
        let allocated = {
            let mut mutable = shard.mutable.lock();
            for write in &writes {
                if mutable.active_sources.contains_key(&write.source) {
                    return Err(AnnError::log_index_error(
                        "detached writeback cannot overlap an active direct-first source"
                            .to_string(),
                    ));
                }
                let local = self.shard_local_index(shard_id, write.source)?;
                let committed =
                    PackedLocation::from_raw(shard.locations[local].load(Ordering::Acquire));
                if committed != write.old_location {
                    return Err(AnnError::log_index_error(format!(
                        "reservoir writeback for source {} has a stale old location",
                        write.source
                    )));
                }
            }
            let resident_pages = mutable.pages.keys().copied().collect::<BTreeSet<_>>();
            let slots = mutable
                .allocator
                .allocate_batch(writes.len(), &resident_pages)?;
            writes
                .into_iter()
                .zip(slots)
                .map(|(write, physical_slot)| {
                    let page_id = physical_slot / u64::try_from(self.layout.records_per_page)?;
                    let slot_index = usize::try_from(
                        physical_slot % u64::try_from(self.layout.records_per_page)?,
                    )?;
                    Ok((
                        page_id,
                        slot_index,
                        physical_slot,
                        PackedLocation::new(page_id, slot_index, self.layout.records_per_page)?,
                        write,
                    ))
                })
                .collect::<AnnResult<Vec<_>>>()?
        };

        let mut remaining_slots = allocated
            .iter()
            .map(|(_, _, physical_slot, _, _)| *physical_slot)
            .collect::<BTreeSet<_>>();
        let mut pages: BTreeMap<u64, Vec<(usize, u64, PackedLocation, ReservoirWriteback)>> =
            BTreeMap::new();
        for (page_id, slot_index, physical_slot, location, write) in allocated {
            pages
                .entry(page_id)
                .or_default()
                .push((slot_index, physical_slot, location, write));
        }

        let result = (|| {
            for (page_id, page_writes) in pages {
                loop {
                    let mut mutable = shard.mutable.lock();
                    let needs_page = mutable
                        .pages
                        .get(&page_id)
                        .is_none_or(|page| page.dirty.is_none());
                    if needs_page && !self.accounting.try_reserve_page(self.layout.page_bytes) {
                        drop(mutable);
                        self.submit_ready_pages()?;
                        self.wait_for_progress()?;
                        continue;
                    }

                    if needs_page {
                        let seed = mutable
                            .pages
                            .get(&page_id)
                            .and_then(|page| page.inflight.as_ref())
                            .map(|bytes| bytes.as_ref().clone());
                        let page_bytes = match seed {
                            Some(bytes) => {
                                self.io_counters
                                    .page_cache_hits
                                    .fetch_add(1, Ordering::Relaxed);
                                bytes
                            }
                            None if !mutable.allocator.page_has_committed_slots(page_id) => {
                                vec![0_u8; self.layout.page_bytes]
                            }
                            None => {
                                self.io_counters
                                    .page_cache_misses
                                    .fetch_add(1, Ordering::Relaxed);
                                let mut bytes = vec![0_u8; self.layout.page_bytes];
                                if let Err(error) = shard.io.read_page(page_id, &mut bytes) {
                                    self.accounting.release_page(self.layout.page_bytes);
                                    return Err(error);
                                }
                                self.io_counters.pages_read.fetch_add(1, Ordering::Relaxed);
                                bytes
                            }
                        };
                        mutable.pages.entry(page_id).or_default().dirty = Some(page_bytes);
                    }

                    {
                        let page = mutable.pages.get_mut(&page_id).expect("staged cohort page");
                        let bytes = page.dirty.as_mut().expect("mutable cohort page image");
                        for (slot_index, physical_slot, location, write) in &page_writes {
                            let start = slot_index
                                .checked_mul(self.layout.record_bytes)
                                .ok_or_else(|| {
                                    AnnError::log_index_error(
                                        "disk reservoir page record offset overflows usize"
                                            .to_string(),
                                    )
                                })?;
                            let end = start + self.layout.record_bytes;
                            write
                                .record
                                .encode_into(&self.layout, &mut bytes[start..end])?;
                            page.dirty_completions.push(WriteCompletion::detached(
                                write.source,
                                write.generation,
                                *location,
                                write.old_location,
                            ));
                            remaining_slots.remove(physical_slot);
                        }
                    }
                    drop(mutable);
                    self.submit_ready_pages()?;
                    break;
                }
            }
            Ok(())
        })();

        if result.is_err() && !remaining_slots.is_empty() {
            let mut mutable = shard.mutable.lock();
            mutable
                .allocator
                .cancel(&remaining_slots.into_iter().collect::<Vec<_>>())?;
        }
        result
    }

    fn stage_source_once(&self, shard_id: usize, source: u32) -> AnnResult<StageOutcome> {
        let shard = &self.shards[shard_id];
        let mut mutable = shard.mutable.lock();
        let (generation, old_location, record) = {
            let Some(active) = mutable.active_sources.get(&source) else {
                return Ok(StageOutcome::AlreadyInflight);
            };
            if !active.write_state.is_dirty_and_idle() {
                return Ok(StageOutcome::AlreadyInflight);
            }
            let generation = active
                .write_state
                .dirty_generation
                .expect("dirty source generation");
            (
                generation,
                active.write_state.committed_location(),
                DiskReservoirRecord::from_reservoir(source, generation, &active.reservoir)?,
            )
        };

        let physical_slot = mutable.allocator.allocate()?;
        let page_id = physical_slot / u64::try_from(self.layout.records_per_page)?;
        let slot_index =
            usize::try_from(physical_slot % u64::try_from(self.layout.records_per_page)?)?;
        let location = PackedLocation::new(page_id, slot_index, self.layout.records_per_page)?;
        let needs_page = mutable
            .pages
            .get(&page_id)
            .is_none_or(|page| page.dirty.is_none());
        if needs_page && !self.accounting.try_reserve_page(self.layout.page_bytes) {
            mutable.allocator.cancel(&[physical_slot])?;
            return Ok(StageOutcome::WouldBlock);
        }

        if needs_page {
            let seed = mutable
                .pages
                .get(&page_id)
                .and_then(|page| page.inflight.as_ref())
                .map(|bytes| bytes.as_ref().clone());
            let page_bytes = match seed {
                Some(bytes) => {
                    self.io_counters
                        .page_cache_hits
                        .fetch_add(1, Ordering::Relaxed);
                    bytes
                }
                None => {
                    self.io_counters
                        .page_cache_misses
                        .fetch_add(1, Ordering::Relaxed);
                    let mut bytes = vec![0_u8; self.layout.page_bytes];
                    if let Err(error) = shard.io.read_page(page_id, &mut bytes) {
                        self.accounting.release_page(self.layout.page_bytes);
                        mutable.allocator.cancel(&[physical_slot])?;
                        return Err(error);
                    }
                    self.io_counters.pages_read.fetch_add(1, Ordering::Relaxed);
                    bytes
                }
            };
            mutable.pages.entry(page_id).or_default().dirty = Some(page_bytes);
        }

        {
            let page = mutable.pages.get_mut(&page_id).expect("staged page state");
            let bytes = page.dirty.as_mut().expect("mutable staged page image");
            let start = slot_index
                .checked_mul(self.layout.record_bytes)
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "disk reservoir page record offset overflows usize".to_string(),
                    )
                })?;
            let end = start + self.layout.record_bytes;
            record.encode_into(&self.layout, &mut bytes[start..end])?;
            page.dirty_completions.push(WriteCompletion::success(
                source,
                generation,
                location,
                old_location,
            ));
        }
        mutable
            .active_sources
            .get_mut(&source)
            .expect("active source while staging")
            .write_state
            .mark_inflight(generation, location)?;
        Ok(StageOutcome::Staged)
    }

    fn submit_ready_pages(&self) -> AnnResult<()> {
        for (shard_id, shard) in self.shards.iter().enumerate() {
            let mut mutable = shard.mutable.lock();
            let page_ids = mutable
                .pages
                .iter()
                .filter_map(|(&page_id, page)| {
                    (page.inflight.is_none() && page.dirty.is_some()).then_some(page_id)
                })
                .collect::<Vec<_>>();
            for page_id in page_ids {
                self.submit_page_locked(shard_id, page_id, shard, &mut mutable)?;
            }
        }
        Ok(())
    }

    fn submit_page_locked(
        &self,
        shard_id: usize,
        page_id: u64,
        shard: &Arc<ShardStore>,
        mutable: &mut ShardMutable,
    ) -> AnnResult<()> {
        let page = mutable
            .pages
            .get_mut(&page_id)
            .expect("submitted page state");
        if page.inflight.is_some() {
            return Ok(());
        }
        let bytes = Arc::new(page.dirty.take().ok_or_else(|| {
            AnnError::log_index_error(
                "disk reservoir page submission has no dirty image".to_string(),
            )
        })?);
        let record_completions = std::mem::take(&mut page.dirty_completions);
        page.inflight = Some(Arc::clone(&bytes));
        let result = self.executor.submit(PageWriteTask {
            shard_id,
            page_id,
            io: Arc::clone(&shard.io),
            bytes,
            record_completions,
        });
        self.accounting.release_page(self.layout.page_bytes);
        result
    }

    fn process_completions(&self) -> AnnResult<usize> {
        let results = self.executor.take_completions();
        let count = results.len();
        for result in results {
            let shard = self.shards.get(result.shard_id).ok_or_else(|| {
                AnnError::log_index_error(
                    "disk reservoir completion references an unknown shard".to_string(),
                )
            })?;
            let mut mutable = shard.mutable.lock();
            let page = mutable.pages.get_mut(&result.page_id).ok_or_else(|| {
                AnnError::log_index_error(
                    "disk reservoir completion references an inactive page".to_string(),
                )
            })?;
            page.inflight.take().ok_or_else(|| {
                AnnError::log_index_error(
                    "disk reservoir completion has no matching in-flight page".to_string(),
                )
            })?;
            if result.error.is_some() {
                let slots = result
                    .record_completions
                    .iter()
                    .map(|completion| {
                        completion
                            .new_location
                            .physical_slot(self.layout.records_per_page)
                    })
                    .collect::<AnnResult<Vec<_>>>()?;
                mutable.allocator.cancel(&slots)?;
                continue;
            }

            let mut clean_sources = Vec::new();
            for completion in result.record_completions {
                let new_slot = completion
                    .new_location
                    .physical_slot(self.layout.records_per_page)?;
                mutable.allocator.commit(&[new_slot])?;
                let local = self.shard_local_index(result.shard_id, completion.source_id)?;
                match completion.owner {
                    WriteCompletionOwner::ActiveSource => {
                        let (disposition, source_is_clean) = {
                            let active = mutable
                                .active_sources
                                .get_mut(&completion.source_id)
                                .ok_or_else(|| {
                                    AnnError::log_index_error(
                                        "disk reservoir completion references an inactive source"
                                            .to_string(),
                                    )
                                })?;
                            let disposition = active.write_state.apply_completion(&completion)?;
                            (disposition, active.write_state.is_clean())
                        };
                        match disposition {
                            CompletionDisposition::Published { old_location } => {
                                shard.locations[local]
                                    .store(completion.new_location.as_raw(), Ordering::Release);
                                if !old_location.is_invalid() {
                                    mutable.allocator.recycle(
                                        old_location.physical_slot(self.layout.records_per_page)?,
                                    )?;
                                }
                            }
                            CompletionDisposition::Stale => {
                                mutable.allocator.recycle(new_slot)?;
                            }
                        }
                        if source_is_clean {
                            clean_sources.push(completion.source_id);
                        }
                    }
                    WriteCompletionOwner::Detached => {
                        let committed = PackedLocation::from_raw(
                            shard.locations[local].load(Ordering::Acquire),
                        );
                        if committed != completion.old_location {
                            return Err(AnnError::log_index_error(format!(
                                "detached reservoir completion for source {} has a stale old location",
                                completion.source_id
                            )));
                        }
                        shard.locations[local]
                            .store(completion.new_location.as_raw(), Ordering::Release);
                        if !completion.old_location.is_invalid() {
                            mutable.allocator.recycle(
                                completion
                                    .old_location
                                    .physical_slot(self.layout.records_per_page)?,
                            )?;
                        }
                        self.detached_completions[result.shard_id].lock().push_back(
                            ReservoirWritebackCompletion {
                                source: completion.source_id,
                                generation: completion.generation,
                                old_location: completion.old_location,
                                new_location: completion.new_location,
                            },
                        );
                    }
                }
            }
            for source in clean_sources {
                if mutable.active_sources.remove(&source).is_some() {
                    self.accounting.release_decode(self.layout.record_bytes);
                }
            }

            let has_dirty_successor = mutable
                .pages
                .get(&result.page_id)
                .is_some_and(|page| page.dirty.is_some());
            if has_dirty_successor {
                self.submit_page_locked(result.shard_id, result.page_id, shard, &mut mutable)?;
            } else {
                mutable.pages.remove(&result.page_id);
            }
        }
        if let Some(error) = self.executor.first_error() {
            return Err(AnnError::log_index_error(format!(
                "disk reservoir writeback failed: {error}"
            )));
        }
        Ok(count)
    }

    fn wait_for_progress(&self) -> AnnResult<()> {
        if self.executor.pending_tasks() != 0 {
            self.executor.wait_for_change();
        } else {
            thread::yield_now();
        }
        self.process_completions()?;
        Ok(())
    }

    fn has_active_work(&self) -> bool {
        self.shards
            .iter()
            .any(|shard| !shard.mutable.lock().active_sources.is_empty())
    }

    fn read_reservoir_at(
        &self,
        shard: &ShardStore,
        source: u32,
        location: PackedLocation,
    ) -> AnnResult<(HashPruneReservoir, u32)> {
        let (page_id, slot_index) = location.decode(self.layout.records_per_page)?;
        let mut page = vec![0_u8; self.layout.page_bytes];
        self.io_counters
            .page_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        shard.io.read_page(page_id, &mut page)?;
        self.io_counters.pages_read.fetch_add(1, Ordering::Relaxed);
        let start = slot_index * self.layout.record_bytes;
        let end = start + self.layout.record_bytes;
        let record = DiskReservoirRecord::decode(&self.layout, &page[start..end])?;
        if record.source_id != source {
            return Err(AnnError::log_index_error(format!(
                "disk reservoir location for source {source} contains source {}",
                record.source_id
            )));
        }
        self.io_counters
            .records_loaded
            .fetch_add(1, Ordering::Relaxed);
        let generation = record.generation;
        let reservoir = record.into_reservoir(self.layout.l_max, self.config.backfill_policy)?;
        Ok((reservoir, generation))
    }

    fn read_source_window<F>(
        &self,
        window_start: usize,
        window_end: usize,
        callback: &mut F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, HashPruneReservoir) -> AnnResult<()>,
    {
        let mut records: Vec<Option<DiskReservoirRecord>> =
            (window_start..window_end).map(|_| None).collect();
        let mut requests: BTreeMap<(usize, u64), Vec<(usize, u32, usize)>> = BTreeMap::new();
        for source_usize in window_start..window_end {
            let source = u32::try_from(source_usize)?;
            let shard_id = self.shard_id(source)?;
            let local = self.shard_local_index(shard_id, source)?;
            let location = PackedLocation::from_raw(
                self.shards[shard_id].locations[local].load(Ordering::Acquire),
            );
            if location.is_invalid() {
                continue;
            }
            let (page_id, slot_index) = location.decode(self.layout.records_per_page)?;
            requests.entry((shard_id, page_id)).or_default().push((
                source_usize - window_start,
                source,
                slot_index,
            ));
        }

        for ((shard_id, page_id), page_requests) in requests {
            if !self.accounting.try_reserve_page(self.layout.page_bytes) {
                return Err(AnnError::log_index_error(
                    "disk reservoir page-cache budget is occupied during finalization".to_string(),
                ));
            }
            let result = (|| {
                let mut page = vec![0_u8; self.layout.page_bytes];
                self.shards[shard_id].io.read_page(page_id, &mut page)?;
                for (window_offset, source, slot_index) in page_requests {
                    let start = slot_index * self.layout.record_bytes;
                    let end = start + self.layout.record_bytes;
                    let record = DiskReservoirRecord::decode(&self.layout, &page[start..end])?;
                    if record.source_id != source {
                        return Err(AnnError::log_index_error(format!(
                            "disk reservoir final read expected source {source}, found {}",
                            record.source_id
                        )));
                    }
                    records[window_offset] = Some(record);
                }
                Ok(())
            })();
            self.accounting.release_page(self.layout.page_bytes);
            result?;
        }

        for (offset, record) in records.into_iter().enumerate() {
            let source = u32::try_from(window_start + offset)?;
            let reservoir = match record {
                Some(record) => {
                    record.into_reservoir(self.layout.l_max, self.config.backfill_policy)?
                }
                None => HashPruneReservoir::new(self.layout.l_max),
            };
            callback(source, reservoir)?;
        }
        Ok(())
    }

    fn shard_id(&self, source: u32) -> AnnResult<usize> {
        let source = source as usize;
        if source < self.config.source_start || source >= self.config.source_end {
            return Err(AnnError::log_index_error(format!(
                "source {source} is outside disk reservoir range [{}, {})",
                self.config.source_start, self.config.source_end
            )));
        }
        let shard_id = (source - self.config.source_start) / self.config.shard_points;
        if shard_id >= self.shards.len() {
            return Err(AnnError::log_index_error(
                "disk reservoir source maps beyond shard table".to_string(),
            ));
        }
        Ok(shard_id)
    }

    fn shard_local_index(&self, shard_id: usize, source: u32) -> AnnResult<usize> {
        let shard = self
            .shards
            .get(shard_id)
            .ok_or_else(|| AnnError::log_index_error("unknown disk reservoir shard".to_string()))?;
        let source = source as usize;
        let start = shard.header.source_start as usize;
        let end = shard.header.source_end as usize;
        if source < start || source >= end {
            return Err(AnnError::log_index_error(
                "source is outside selected disk reservoir shard".to_string(),
            ));
        }
        Ok(source - start)
    }
}

fn validate_store_config(config: &DiskReservoirStoreConfig) -> AnnResult<DiskReservoirLayout> {
    if config.source_start >= config.source_end || config.source_end > u32::MAX as usize {
        return Err(AnnError::log_index_config_error(
            "source_range".to_string(),
            "disk reservoir requires a non-empty u32 source range".to_string(),
        ));
    }
    if config.shard_points == 0 {
        return Err(AnnError::log_index_config_error(
            "shard_points".to_string(),
            "disk reservoir shard size must be nonzero".to_string(),
        ));
    }
    if !config.overprovision_ratio.is_finite() || config.overprovision_ratio <= 1.0 {
        return Err(AnnError::log_index_config_error(
            "overprovision_ratio".to_string(),
            "disk reservoir overprovision ratio must be finite and > 1".to_string(),
        ));
    }
    let layout = DiskReservoirLayout::new(config.l_max, config.io.effective_alignment())?;
    if config.page_cache_bytes < layout.page_bytes {
        return Err(AnnError::log_index_config_error(
            "page_cache_bytes".to_string(),
            format!(
                "must hold at least one {}-byte state page",
                layout.page_bytes
            ),
        ));
    }
    if config.writeback_bytes < layout.page_bytes {
        return Err(AnnError::log_index_config_error(
            "writeback_bytes".to_string(),
            format!(
                "must hold at least one {}-byte state page",
                layout.page_bytes
            ),
        ));
    }
    if config.decode_bytes != 0 && config.decode_bytes < layout.record_bytes {
        return Err(AnnError::log_index_config_error(
            "decode_bytes".to_string(),
            format!(
                "must hold at least one {}-byte decoded state",
                layout.record_bytes
            ),
        ));
    }
    Ok(layout)
}

fn physical_capacity(
    source_count: usize,
    layout: DiskReservoirLayout,
    overprovision_ratio: f64,
) -> AnnResult<(u64, usize)> {
    let requested = ((source_count as f64) * overprovision_ratio).ceil();
    if !requested.is_finite() || requested > u64::MAX as f64 {
        return Err(AnnError::log_index_config_error(
            "overprovision_ratio".to_string(),
            "disk reservoir physical capacity overflows u64".to_string(),
        ));
    }
    let requested = (requested as u64).max(u64::try_from(source_count)?);
    let records_per_page = u64::try_from(layout.records_per_page)?;
    let page_count_u64 = requested.div_ceil(records_per_page);
    let capacity = page_count_u64
        .checked_mul(records_per_page)
        .ok_or_else(|| {
            AnnError::log_index_error(
                "disk reservoir rounded physical capacity overflows u64".to_string(),
            )
        })?;
    Ok((capacity, usize::try_from(page_count_u64)?))
}

fn new_shard_store(
    header: DiskReservoirHeader,
    io: Arc<dyn ReservoirPageIo>,
    physical_capacity: u64,
) -> AnnResult<ShardStore> {
    let source_count = usize::try_from(header.source_end - header.source_start)?;
    let records_per_page = header.layout.records_per_page;
    let locations = (0..source_count)
        .map(|_| AtomicU64::new(PackedLocation::INVALID.as_raw()))
        .collect();
    Ok(ShardStore {
        header,
        io,
        locations,
        mutable: Mutex::new(ShardMutable {
            allocator: SlotAllocator::new(physical_capacity, records_per_page),
            active_sources: HashMap::new(),
            pages: HashMap::new(),
        }),
    })
}

fn shard_path(root: &Path, source_start: usize, source_end: usize) -> PathBuf {
    root.join(format!(
        "hashprune-state-{source_start:010}-{source_end:010}.bin"
    ))
}

fn round_up_to_alignment(value: usize, alignment: usize) -> AnnResult<usize> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| {
            AnnError::log_index_config_error(
                "page_bytes".to_string(),
                "disk reservoir page size overflows usize".to_string(),
            )
        })
}

fn header_mismatch_error() -> AnnError {
    AnnError::log_index_config_error(
        "disk_reservoir_header".to_string(),
        "stored header does not match the expected configuration".to_string(),
    )
}

#[inline]
fn read_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([input[offset], input[offset + 1]])
}

#[inline]
fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
    ])
}

#[inline]
fn write_u16(output: &mut [u8], offset: usize, value: u16) {
    output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[inline]
fn write_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use tempfile::tempdir;

    use super::super::direct_io::DirectIoConfig;
    use super::*;

    struct FaultPageIo {
        fail_write: usize,
        writes: AtomicUsize,
        pages: Mutex<HashMap<u64, Vec<u8>>>,
    }

    struct PermutedPageIo {
        seed: u64,
        writes: AtomicUsize,
        pages: Mutex<HashMap<u64, Vec<u8>>>,
    }

    impl PermutedPageIo {
        fn new(seed: u64) -> Self {
            Self {
                seed,
                writes: AtomicUsize::new(0),
                pages: Mutex::new(HashMap::new()),
            }
        }
    }

    impl ReservoirPageIo for PermutedPageIo {
        fn read_page(&self, page_id: u64, output: &mut [u8]) -> AnnResult<()> {
            if let Some(bytes) = self.pages.lock().get(&page_id) {
                output.copy_from_slice(bytes);
            } else {
                output.fill(0);
            }
            Ok(())
        }

        fn write_page(&self, page_id: u64, bytes: &[u8]) -> AnnResult<()> {
            let sequence = self.writes.fetch_add(1, AtomicOrdering::SeqCst) as u64;
            let mut mixed = self.seed ^ page_id.rotate_left(17) ^ sequence.rotate_left(31);
            mixed ^= mixed << 13;
            mixed ^= mixed >> 7;
            mixed ^= mixed << 17;
            std::thread::sleep(std::time::Duration::from_micros(mixed % 200));
            self.pages.lock().insert(page_id, bytes.to_vec());
            Ok(())
        }

        fn sync_all(&self) -> AnnResult<()> {
            Ok(())
        }
    }

    impl FaultPageIo {
        fn fail_write_number(fail_write: usize) -> Self {
            Self {
                fail_write,
                writes: AtomicUsize::new(0),
                pages: Mutex::new(HashMap::new()),
            }
        }
    }

    impl ReservoirPageIo for FaultPageIo {
        fn read_page(&self, page_id: u64, output: &mut [u8]) -> AnnResult<()> {
            if let Some(bytes) = self.pages.lock().get(&page_id) {
                output.copy_from_slice(bytes);
            } else {
                output.fill(0);
            }
            Ok(())
        }

        fn write_page(&self, page_id: u64, bytes: &[u8]) -> AnnResult<()> {
            let write = self.writes.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            if write == self.fail_write {
                return Err(AnnError::log_index_error(
                    "injected page write failure".to_string(),
                ));
            }
            self.pages.lock().insert(page_id, bytes.to_vec());
            Ok(())
        }

        fn sync_all(&self) -> AnnResult<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct BlockingPageIoState {
        submitted: usize,
        release_first: bool,
        pages: HashMap<u64, Vec<u8>>,
    }

    struct BlockingPageIo {
        state: Mutex<BlockingPageIoState>,
        changed: Condvar,
    }

    impl BlockingPageIo {
        fn new() -> Self {
            Self {
                state: Mutex::new(BlockingPageIoState::default()),
                changed: Condvar::new(),
            }
        }

        fn wait_until_write_submitted(&self, expected: usize) {
            let mut state = self.state.lock();
            while state.submitted < expected {
                self.changed.wait(&mut state);
            }
        }

        fn submitted_write_count(&self) -> usize {
            self.state.lock().submitted
        }

        fn release_next_write(&self) {
            let mut state = self.state.lock();
            state.release_first = true;
            self.changed.notify_all();
        }
    }

    impl ReservoirPageIo for BlockingPageIo {
        fn read_page(&self, page_id: u64, output: &mut [u8]) -> AnnResult<()> {
            if let Some(bytes) = self.state.lock().pages.get(&page_id) {
                output.copy_from_slice(bytes);
            } else {
                output.fill(0);
            }
            Ok(())
        }

        fn write_page(&self, page_id: u64, bytes: &[u8]) -> AnnResult<()> {
            let mut state = self.state.lock();
            state.submitted += 1;
            let submitted = state.submitted;
            self.changed.notify_all();
            while submitted == 1 && !state.release_first {
                self.changed.wait(&mut state);
            }
            state.pages.insert(page_id, bytes.to_vec());
            Ok(())
        }

        fn sync_all(&self) -> AnnResult<()> {
            Ok(())
        }
    }

    fn test_store_config(
        root: &std::path::Path,
        source_start: usize,
        source_end: usize,
        l_max: usize,
    ) -> DiskReservoirStoreConfig {
        DiskReservoirStoreConfig {
            root: root.to_path_buf(),
            source_start,
            source_end,
            shard_points: source_end - source_start,
            l_max,
            page_cache_bytes: 4096 * 2,
            writeback_bytes: 4096,
            decode_bytes: 4096,
            writeback_workers: 1,
            overprovision_ratio: 2.0,
            io: DirectIoConfig::disabled(),
            policy_key: [0; 32],
            backfill_policy: HashPruneBackfillPolicy::off(),
        }
    }

    fn read_targets(store: &DiskReservoirStore, wanted: u32) -> AnnResult<Vec<u32>> {
        let mut targets = Vec::new();
        store.for_each_source_window(2, |source, reservoir| {
            if source == wanted {
                targets = reservoir.slots().iter().map(|slot| slot.target).collect();
            }
            Ok(())
        })?;
        Ok(targets)
    }

    #[test]
    fn disk_reservoir_record_round_trips_internal_state() {
        let policy = HashPruneBackfillPolicy::off();
        let layout = DiskReservoirLayout::new(4, 4096).unwrap();
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert(10, 1, 4.0);
        reservoir.insert(20, 2, 3.0);
        let record = DiskReservoirRecord::from_reservoir(7, 3, &reservoir).unwrap();
        let mut bytes = vec![0_u8; layout.record_bytes];
        record.encode_into(&layout, &mut bytes).unwrap();
        let decoded = DiskReservoirRecord::decode(&layout, &bytes).unwrap();
        assert_eq!(decoded.source_id, 7);
        assert_eq!(decoded.generation, 3);
        let restored = decoded.into_reservoir(4, policy).unwrap();
        assert_eq!(
            reservoir
                .slots()
                .iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                .collect::<Vec<_>>(),
            restored
                .slots()
                .iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn disk_layout_never_crosses_page_boundaries() {
        for l_max in [1_usize, 64, 128, 512] {
            let layout = DiskReservoirLayout::new(l_max, 4096).unwrap();
            assert_eq!(layout.page_bytes % 4096, 0);
            assert!(layout.records_per_page >= 1);
            assert!(layout.records_per_page * layout.record_bytes <= layout.page_bytes);
        }
    }

    #[test]
    fn packed_location_round_trips_and_reserves_invalid() {
        let location = PackedLocation::new(1234, 5, 8).unwrap();
        assert_eq!(location.decode(8).unwrap(), (1234, 5));
        assert!(PackedLocation::INVALID.is_invalid());
    }

    #[test]
    fn disk_record_rejects_corrupt_boundaries() {
        let layout = DiskReservoirLayout::new(4, 4096).unwrap();
        let mut bytes = vec![0_u8; layout.record_bytes];
        bytes[8..10].copy_from_slice(&5_u16.to_le_bytes());
        assert!(DiskReservoirRecord::decode(&layout, &bytes).is_err());
    }

    #[test]
    fn disk_reservoir_header_round_trips_and_rejects_mismatch() {
        let expected = DiskReservoirHeader {
            source_start: 100,
            source_end: 200,
            layout: DiskReservoirLayout::new(64, 4096).unwrap(),
            policy_key: [7; 32],
        };
        let bytes = expected.encode_page().unwrap();
        assert_eq!(bytes.len(), expected.layout.page_bytes);
        assert_eq!(
            DiskReservoirHeader::decode(&bytes, &expected).unwrap(),
            expected
        );

        let mut mismatched = expected.clone();
        mismatched.policy_key[0] ^= 1;
        assert!(DiskReservoirHeader::decode(&bytes, &mismatched).is_err());
    }

    #[test]
    fn slot_allocator_reuses_only_marked_locations() {
        let mut allocator = SlotAllocator::new(8, 4);
        let first = allocator.allocate().unwrap();
        let second = allocator.allocate().unwrap();
        assert_ne!(first, second);
        allocator.commit(&[first, second]).unwrap();
        allocator.recycle(first).unwrap();
        assert_eq!(allocator.allocate().unwrap(), first);
    }

    #[test]
    fn page_allocator_packs_one_cohort_before_opening_another_page() {
        let mut allocator = SlotAllocator::new(16, 4);
        let locations = allocator.allocate_batch(6, &BTreeSet::new()).unwrap();
        let pages = locations
            .iter()
            .map(|slot| slot / 4)
            .collect::<BTreeSet<_>>();
        assert_eq!(pages, BTreeSet::from([0, 1]));
    }

    #[test]
    fn page_allocator_prefers_resident_partial_page_over_new_page() {
        let mut allocator = SlotAllocator::new(12, 4);
        let first = allocator.allocate_batch(4, &BTreeSet::new()).unwrap();
        allocator.commit(&first).unwrap();
        allocator.recycle(first[1]).unwrap();
        allocator.recycle(first[2]).unwrap();

        assert_eq!(
            allocator.allocate_batch(2, &BTreeSet::from([0])).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn page_allocator_accounts_record_placement_by_page_class() {
        let mut allocator = SlotAllocator::new(8, 4);
        let first = allocator.allocate_batch(4, &BTreeSet::new()).unwrap();
        allocator.commit(&first).unwrap();
        allocator.recycle(first[1]).unwrap();
        allocator.recycle(first[2]).unwrap();
        let resident = allocator.allocate_batch(2, &BTreeSet::from([0])).unwrap();
        allocator.commit(&resident).unwrap();
        allocator.recycle(resident[0]).unwrap();
        allocator.recycle(resident[1]).unwrap();
        let second_page = allocator.allocate_batch(4, &BTreeSet::new()).unwrap();
        allocator.commit(&second_page).unwrap();
        let cold = allocator.allocate_batch(2, &BTreeSet::new()).unwrap();
        allocator.commit(&cold).unwrap();

        assert_eq!(
            allocator.allocation_stats(),
            PageAllocationStats {
                empty_page: 8,
                resident_partial_page: 2,
                cold_partial_page: 2,
            }
        );
    }

    #[test]
    fn page_allocator_cancel_returns_only_reserved_slots() {
        let mut allocator = SlotAllocator::new(8, 4);
        let reserved = allocator.allocate_batch(3, &BTreeSet::new()).unwrap();
        allocator.cancel(&reserved).unwrap();

        assert_eq!(
            allocator.allocate_batch(4, &BTreeSet::from([0])).unwrap(),
            vec![3, 0, 1, 2]
        );
        assert!(allocator.recycle(reserved[0]).is_err());
    }

    #[test]
    fn stale_writeback_completion_cannot_publish_location() {
        let mut state = SourceWriteState::new(PackedLocation::INVALID);
        state.mark_dirty(1);
        state
            .mark_inflight(1, PackedLocation::new(2, 0, 4).unwrap())
            .unwrap();
        state.mark_dirty(2);
        let completion = WriteCompletion::success(
            9,
            1,
            PackedLocation::new(2, 0, 4).unwrap(),
            PackedLocation::INVALID,
        );
        assert_eq!(
            state.apply_completion(&completion).unwrap(),
            CompletionDisposition::Stale
        );
        assert!(state.committed_location().is_invalid());
    }

    #[test]
    fn disk_store_updates_flushes_and_reads_in_source_order() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 100, 108, 4);
        let store = DiskReservoirStore::create(config).unwrap();
        for source in [107_u32, 101, 104] {
            store
                .update_source(source, |reservoir| {
                    reservoir.insert(source + 1000, source as u16, source as f32);
                    Ok(())
                })
                .unwrap();
        }
        store.flush_all().unwrap();

        let mut rows = Vec::new();
        store
            .for_each_source_window(3, |source, reservoir| {
                rows.push((
                    source,
                    reservoir
                        .slots()
                        .iter()
                        .map(|slot| slot.target)
                        .collect::<Vec<_>>(),
                ));
                Ok(())
            })
            .unwrap();
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[1], (101, vec![1101]));
        assert_eq!(rows[4], (104, vec![1104]));
        assert_eq!(rows[7], (107, vec![1107]));
    }

    #[test]
    fn disk_store_update_batch_combines_records_on_the_same_page() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 4, 4);
        let io = Arc::new(FaultPageIo::fail_write_number(usize::MAX));
        let store = DiskReservoirStore::create_for_test(config, io.clone()).unwrap();

        for source in 0_u32..2 {
            store
                .update_source_deferred(source, |reservoir| {
                    reservoir.insert(source + 10, source as u16, source as f32);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(io.writes.load(AtomicOrdering::SeqCst), 0);

        store.flush_update_batch().unwrap();
        store.flush_all().unwrap();

        assert_eq!(io.writes.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(read_targets(&store, 0).unwrap(), vec![10]);
        assert_eq!(read_targets(&store, 1).unwrap(), vec![11]);
    }

    #[test]
    fn grouped_read_reads_one_page_for_multiple_requested_sources() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 4, 4);
        let store = DiskReservoirStore::create(config).unwrap();
        for source in 0_u32..3 {
            store
                .update_source_deferred(source, |reservoir| {
                    reservoir.insert(source + 10, source as u16, source as f32);
                    Ok(())
                })
                .unwrap();
        }
        store.flush_all().unwrap();

        let before = store.io_stats().pages_read;
        let loaded = store.load_sources_grouped(&[0, 1, 2]).unwrap();

        assert_eq!(loaded.len(), 3);
        assert_eq!(store.io_stats().pages_read - before, 1);
        assert_eq!(
            loaded
                .iter()
                .map(|entry| entry.reservoir.slots()[0].target)
                .collect::<Vec<_>>(),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn cohort_write_places_new_records_in_the_minimum_empty_pages() {
        let dir = tempdir().unwrap();
        let mut config = test_store_config(dir.path(), 0, 8, 84);
        config.page_cache_bytes = 4096 * 2;
        config.writeback_bytes = 4096 * 2;
        let store = DiskReservoirStore::create(config).unwrap();
        assert_eq!(store.layout.records_per_page, 4);
        let writes = (0_u32..6)
            .map(|source| {
                let mut reservoir = HashPruneReservoir::new(84);
                reservoir.insert(source + 100, source as u16, source as f32);
                Ok(ReservoirWriteback {
                    source,
                    generation: 1,
                    old_location: PackedLocation::INVALID,
                    record: DiskReservoirRecord::from_reservoir(source, 1, &reservoir)?,
                })
            })
            .collect::<AnnResult<Vec<_>>>()
            .unwrap();

        store.stage_writeback_cohort(writes).unwrap();
        store.flush_all().unwrap();

        assert_eq!(store.io_stats().pages_written, 2);
        assert_eq!(
            store
                .load_sources_grouped(&[0, 1, 2, 3, 4, 5])
                .unwrap()
                .into_iter()
                .map(|entry| entry.reservoir.slots()[0].target)
                .collect::<Vec<_>>(),
            vec![100, 101, 102, 103, 104, 105]
        );
    }

    #[test]
    fn detached_writeback_completions_are_consumed_by_source_shard() {
        let dir = tempdir().unwrap();
        let mut config = test_store_config(dir.path(), 0, 4, 4);
        config.shard_points = 2;
        config.page_cache_bytes = 4096 * 2;
        config.writeback_bytes = 4096 * 2;
        config.writeback_workers = 2;
        let store = DiskReservoirStore::create(config).unwrap();
        let writes = [0_u32, 2]
            .into_iter()
            .map(|source| {
                let mut reservoir = HashPruneReservoir::new(4);
                reservoir.insert(source + 10, source as u16, source as f32);
                Ok(ReservoirWriteback {
                    source,
                    generation: 1,
                    old_location: PackedLocation::INVALID,
                    record: DiskReservoirRecord::from_reservoir(source, 1, &reservoir)?,
                })
            })
            .collect::<AnnResult<Vec<_>>>()
            .unwrap();

        store.stage_writeback_cohort(writes).unwrap();
        store.flush_all().unwrap();

        assert_eq!(
            store.detached_completion_queue_lengths_for_test(),
            vec![1, 1]
        );
        let shard_zero = store.take_writeback_completions_for_shard(0).unwrap();
        assert_eq!(
            store.detached_completion_queue_lengths_for_test(),
            vec![0, 1]
        );
        assert!(
            store
                .take_writeback_completions_for_shard(0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            store.detached_completion_queue_lengths_for_test(),
            vec![0, 1]
        );
        let shard_one = store.take_writeback_completions_for_shard(1).unwrap();
        assert_eq!(
            shard_zero
                .into_iter()
                .map(|completion| completion.source)
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(
            shard_one
                .into_iter()
                .map(|completion| completion.source)
                .collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn disk_store_propagates_page_write_failure() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 4, 4);
        let io = Arc::new(FaultPageIo::fail_write_number(1));
        let store = DiskReservoirStore::create_for_test(config, io).unwrap();
        store
            .update_source(0, |reservoir| {
                reservoir.insert(1, 1, 1.0);
                Ok(())
            })
            .unwrap();
        assert!(store.flush_all().is_err());
    }

    #[test]
    fn disk_store_cache_and_writeback_never_exceed_configured_bytes() {
        let dir = tempdir().unwrap();
        let mut config = test_store_config(dir.path(), 0, 32, 4);
        config.page_cache_bytes = 4096 * 2;
        config.writeback_bytes = 4096;
        let store = DiskReservoirStore::create(config).unwrap();
        for source in 0_u32..32 {
            store
                .update_source(source, |reservoir| {
                    reservoir.insert(source + 100, source as u16, source as f32);
                    Ok(())
                })
                .unwrap();
        }
        store.flush_all().unwrap();
        let peaks = store.memory_peaks();
        assert!(peaks.page_cache_bytes <= 4096 * 2);
        assert!(peaks.writeback_bytes <= 4096);
    }

    #[test]
    fn recycled_slot_page_write_preserves_neighbor_records() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 2, 4);
        let store = DiskReservoirStore::create(config).unwrap();
        store
            .update_source(0, |reservoir| {
                reservoir.insert(10, 10, 1.0);
                Ok(())
            })
            .unwrap();
        store
            .update_source(1, |reservoir| {
                reservoir.insert(20, 20, 1.0);
                Ok(())
            })
            .unwrap();
        store.flush_all().unwrap();

        store
            .update_source(0, |reservoir| {
                reservoir.insert(11, 11, 0.5);
                Ok(())
            })
            .unwrap();
        store.flush_all().unwrap();
        store
            .update_source(0, |reservoir| {
                reservoir.insert(12, 12, 0.25);
                Ok(())
            })
            .unwrap();
        store.flush_all().unwrap();

        assert_eq!(read_targets(&store, 1).unwrap(), vec![20]);
    }

    #[test]
    fn same_page_writes_are_submitted_in_generation_order() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 2, 4);
        let io = Arc::new(BlockingPageIo::new());
        let store = DiskReservoirStore::create_for_test(config, io.clone()).unwrap();
        store
            .update_source(0, |reservoir| {
                reservoir.insert(10, 10, 1.0);
                Ok(())
            })
            .unwrap();
        io.wait_until_write_submitted(1);
        store
            .update_source(1, |reservoir| {
                reservoir.insert(20, 20, 1.0);
                Ok(())
            })
            .unwrap();
        assert_eq!(io.submitted_write_count(), 1);
        io.release_next_write();
        store.flush_all().unwrap();
        assert_eq!(io.submitted_write_count(), 2);
    }

    #[test]
    fn newer_source_generation_survives_stale_writeback_completion() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 2, 4);
        let io = Arc::new(BlockingPageIo::new());
        let store = DiskReservoirStore::create_for_test(config, io.clone()).unwrap();
        store
            .update_source(0, |reservoir| {
                reservoir.insert(10, 10, 1.0);
                Ok(())
            })
            .unwrap();
        io.wait_until_write_submitted(1);
        store
            .update_source(0, |reservoir| {
                reservoir.insert(11, 11, 0.5);
                Ok(())
            })
            .unwrap();
        assert_eq!(io.submitted_write_count(), 1);

        io.release_next_write();
        store.flush_all().unwrap();
        assert_eq!(io.submitted_write_count(), 2);
        assert_eq!(read_targets(&store, 0).unwrap(), vec![10, 11]);
    }

    #[test]
    fn flush_all_combines_dirty_successors_on_the_same_page() {
        let dir = tempdir().unwrap();
        let config = test_store_config(dir.path(), 0, 2, 4);
        let io = Arc::new(BlockingPageIo::new());
        let store = DiskReservoirStore::create_for_test(config, io.clone()).unwrap();

        for source in 0_u32..2 {
            store
                .update_source_deferred(source, |reservoir| {
                    reservoir.insert(source + 10, source as u16, 2.0);
                    Ok(())
                })
                .unwrap();
        }
        store.flush_update_batch().unwrap();
        io.wait_until_write_submitted(1);

        for source in 0_u32..2 {
            store
                .update_source_deferred(source, |reservoir| {
                    reservoir.insert(source + 20, source as u16 + 10, 1.0);
                    Ok(())
                })
                .unwrap();
        }
        store.flush_update_batch().unwrap();

        io.release_next_write();
        store.flush_all().unwrap();

        assert_eq!(io.submitted_write_count(), 2);
        assert_eq!(read_targets(&store, 0).unwrap(), vec![10, 20]);
        assert_eq!(read_targets(&store, 1).unwrap(), vec![11, 21]);
    }

    #[test]
    fn online_suffix_async_completion_permutation_preserves_final_state() {
        let mut expected = None;
        for seed in 0_u64..8 {
            let dir = tempdir().unwrap();
            let mut config = test_store_config(dir.path(), 0, 96, 4);
            config.shard_points = 96;
            config.page_cache_bytes = 4096 * 4;
            config.writeback_bytes = 4096 * 4;
            config.decode_bytes = 4096;
            config.writeback_workers = 4;
            let io = Arc::new(PermutedPageIo::new(seed));
            let store = DiskReservoirStore::create_for_test(config, io).unwrap();
            for round in 0..2_u32 {
                for source in 0_u32..96 {
                    store
                        .update_source(source, |reservoir| {
                            reservoir.insert(
                                1000 + round * 100 + source,
                                (round + 1) as u16,
                                (round + 1) as f32,
                            );
                            Ok(())
                        })
                        .unwrap();
                }
            }
            store.flush_all().unwrap();
            let mut actual = Vec::new();
            store
                .for_each_source_window(16, |source, reservoir| {
                    actual.push((source, reservoir.state_snapshot()));
                    Ok(())
                })
                .unwrap();
            let actual = actual
                .iter()
                .map(|(source, state)| {
                    (
                        *source,
                        state
                            .slots
                            .iter()
                            .map(|slot| {
                                (
                                    slot.target,
                                    slot.hash,
                                    slot.dist.to_bits(),
                                    slot.meta.to_state_flags().unwrap(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            match &expected {
                Some(expected) => assert_eq!(&actual, expected, "seed={seed}"),
                None => expected = Some(actual),
            }
        }
    }
}
