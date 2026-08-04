use std::cmp::Ordering;
use std::mem::size_of;

use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rayon::prelude::*;

use super::hash_prune_backfill::{
    HashPruneBackfillPolicy, HashPruneBackfillProfile, MAX_INTERNAL_BACKUPS,
};
use super::hash_prune_evidence::{
    HashPruneEvidencePolicy, HashPruneEvidenceProfile, HashPruneMeta,
};
use super::params::{PiPNNHashPruneBackfillMode, PiPNNParams};
use super::point_store::{PointStore, WindowedGatherOptions, WindowedGatherStats};
use crate::common::{AnnError, AnnResult};
use crate::model::InmemDataset;
use crate::utils::thread_pool::with_rayon_thread_pool;

pub trait SketchAccessor: Sync {
    fn width(&self) -> usize;
    fn rows(&self) -> usize;
    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()>;
    fn resident_bytes(&self) -> usize;

    /// Batched row read — default implementation calls `row_copy_into` per ID.
    /// Implementations should override to coalesce contiguous runs.
    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let w = self.width();
        for (i, &id) in ids.iter().enumerate() {
            self.row_copy_into(id as usize, &mut out[i * w..(i + 1) * w])?;
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            let row_bytes = self.width() * size_of::<f32>();
            stats.rows_requested += ids.len() as u64;
            stats.logical_bytes += (ids.len() * row_bytes) as u64;
            stats.physical_bytes += (ids.len() * row_bytes) as u64;
            stats.scatter_ops += ids.len() as u64;
            stats.singleton_windows += ids.len() as u64;
            stats.windows_submitted += ids.len() as u64;
            stats.rows_per_window_sum += ids.len() as u64;
        }
        self.read_rows_into(ids, out)
    }
}

/// 行优先存储的 sketch 矩阵，避免为每个点分配一个独立的小 Vec。
#[derive(Debug, Clone)]
pub struct SketchStore {
    data: Vec<f32>,
    width: usize,
}

impl SketchStore {
    #[inline]
    pub fn from_row_major(data: Vec<f32>, width: usize) -> Self {
        Self { data, width }
    }

    #[inline]
    pub fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.width;
        let end = start + self.width;
        &self.data[start..end]
    }

    #[inline]
    pub fn estimate_bytes(&self) -> usize {
        size_of::<Self>() + self.data.capacity().saturating_mul(size_of::<f32>())
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn rows(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.data.len() / self.width
        }
    }

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn from_test_data(data: Vec<f32>, width: usize) -> Self {
        Self::from_row_major(data, width)
    }
}

impl SketchAccessor for SketchStore {
    #[inline]
    fn width(&self) -> usize {
        self.width()
    }

    #[inline]
    fn rows(&self) -> usize {
        self.rows()
    }

    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()> {
        if dst.len() != self.width {
            return Err(AnnError::log_index_error(format!(
                "Sketch row copy width mismatch: dst={} expected={}",
                dst.len(),
                self.width
            )));
        }
        let row = self.row(idx);
        dst.copy_from_slice(row);
        Ok(())
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let w = self.width();
        for (i, &id) in ids.iter().enumerate() {
            out[i * w..(i + 1) * w].copy_from_slice(self.row(id as usize));
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let row_bytes = self.width * size_of::<f32>();
        let plan = super::point_store::build_gather_plan(ids, row_bytes, *options, stats);
        let mut window_buf = vec![
            0.0f32;
            plan.windows
                .iter()
                .map(|window| window.row_count as usize)
                .max()
                .unwrap_or(0)
                * self.width
        ];
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let rows = window.row_count as usize;
            let elems = rows * self.width;
            let window_slice = &mut window_buf[..elems];
            let start = window.start_row as usize * self.width;
            let end = start + elems;
            window_slice.copy_from_slice(&self.data[start..end]);
            super::point_store::scatter_window_rows(
                &plan,
                window_idx,
                self.width,
                window_slice,
                out,
            );
        }
        Ok(())
    }

    #[inline]
    fn resident_bytes(&self) -> usize {
        self.estimate_bytes()
    }
}

/// 简单的 bfloat16 实现：截断 float32 的高 16 位。
#[derive(Clone, Copy, Debug, Default)]
pub struct BFloat16(pub u16);

impl BFloat16 {
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Self((v.to_bits() >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// HashPrune 水库中的单个槽位。
///
/// `dist` intentionally stores the exact f32 candidate distance. HashPrune selection must be a
/// function of the candidate set, not of insertion order; BF16-compressed distances are too coarse
/// for the reservoir's replacement decisions.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub target: u32,
    pub dist: f32,
    pub hash: u16,
    pub(crate) meta: HashPruneMeta,
}

impl Slot {
    #[inline]
    pub fn new(target: u32, hash: u16, dist: f32) -> Self {
        Self::new_with_meta(target, hash, dist, HashPruneMeta::empty())
    }

    #[inline]
    pub(crate) fn new_with_meta(target: u32, hash: u16, dist: f32, meta: HashPruneMeta) -> Self {
        Self {
            target,
            dist,
            hash,
            meta,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct HashPruneInsertProfile {
    pub(crate) candidates: usize,
    pub(crate) hash_hits: usize,
    pub(crate) hash_replacements: usize,
    pub(crate) hash_rejects: usize,
    pub(crate) not_full_inserts: usize,
    pub(crate) full_replacement_checks: usize,
    pub(crate) full_replacements: usize,
    pub(crate) full_rejects: usize,
    pub(crate) farthest_recomputes: usize,
    pub(crate) rotate_lefts: usize,
    pub(crate) rotate_rights: usize,
    pub(crate) evidence: HashPruneEvidenceProfile,
}

impl HashPruneInsertProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        self.candidates += other.candidates;
        self.hash_hits += other.hash_hits;
        self.hash_replacements += other.hash_replacements;
        self.hash_rejects += other.hash_rejects;
        self.not_full_inserts += other.not_full_inserts;
        self.full_replacement_checks += other.full_replacement_checks;
        self.full_replacements += other.full_replacements;
        self.full_rejects += other.full_rejects;
        self.farthest_recomputes += other.farthest_recomputes;
        self.rotate_lefts += other.rotate_lefts;
        self.rotate_rights += other.rotate_rights;
        self.evidence.merge(other.evidence);
    }
}

/// 单个点对应的 HashPrune 水库。
///
/// 槽按 hash 有序，便于二分查找；容量固定为 l_max。
#[derive(Debug)]
pub struct HashPruneReservoir {
    l_max: usize,
    slots: Vec<Slot>,
    /// 当前最远槽的下标（仅在 slots 已满时有效）。
    farthest_idx: usize,
    /// 当前最远槽的距离缓存（仅在 slots 已满时有效）。
    farthest_dist: f32,
    /// Backfill suffix length. This occupies existing tail padding on 64-bit targets.
    backup_count: u16,
    /// Relative index of the worst backup in the unsorted suffix.
    backup_worst: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct HashPruneReservoirState {
    pub(crate) slots: Vec<Slot>,
    pub(crate) backup_count: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HashPruneRuntimeState {
    pub(crate) farthest_idx: usize,
    pub(crate) farthest_dist: f32,
    pub(crate) backup_count: u16,
    pub(crate) backup_worst: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct FinalizedHashPruneRow {
    pub(crate) slots: Vec<Slot>,
    pub(crate) primary_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HashPrunePrefinalSnapshot {
    pub(crate) primary_count: usize,
    pub(crate) slots: Vec<(u32, u16, u32, u8)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RHalfAdmissionEvent {
    Unchanged,
    Admitted,
    Replaced,
    CapacityRejected,
}

impl HashPruneReservoir {
    pub fn new(l_max: usize) -> Self {
        assert!(l_max > 0, "l_max must be > 0");
        Self {
            l_max,
            slots: Vec::with_capacity(l_max),
            farthest_idx: 0,
            farthest_dist: 0.0,
            backup_count: 0,
            backup_worst: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.slots.clear();
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
        self.backup_count = 0;
        self.backup_worst = 0;
    }

    #[inline]
    pub(crate) fn runtime_state(&self) -> HashPruneRuntimeState {
        HashPruneRuntimeState {
            farthest_idx: self.farthest_idx,
            farthest_dist: self.farthest_dist,
            backup_count: self.backup_count,
            backup_worst: self.backup_worst,
        }
    }

    pub(crate) fn load_runtime_row(
        &mut self,
        slots: &[Slot],
        state: HashPruneRuntimeState,
    ) -> AnnResult<()> {
        let backup_count = usize::from(state.backup_count);
        let primary_count = slots.len().checked_sub(backup_count).ok_or_else(|| {
            AnnError::log_index_error("invalid flat HashPrune backup boundary".to_string())
        })?;
        let invalid_farthest = if primary_count == 0 {
            state.farthest_idx != 0
        } else {
            state.farthest_idx >= primary_count
        };
        let invalid_backup_worst = if backup_count == 0 {
            state.backup_worst != 0
        } else {
            usize::from(state.backup_worst) >= backup_count
        };
        if slots.len() > self.l_max
            || (primary_count == 0 && backup_count != 0)
            || invalid_farthest
            || invalid_backup_worst
        {
            return Err(AnnError::log_index_error(
                "invalid flat HashPrune runtime state".to_string(),
            ));
        }

        self.slots.clear();
        if self.slots.capacity() < self.l_max {
            self.slots.reserve_exact(self.l_max - self.slots.capacity());
        }
        self.slots.extend_from_slice(slots);
        self.farthest_idx = state.farthest_idx;
        self.farthest_dist = state.farthest_dist;
        self.backup_count = state.backup_count;
        self.backup_worst = state.backup_worst;
        Ok(())
    }

    pub(crate) fn state_snapshot(&self) -> HashPruneReservoirState {
        HashPruneReservoirState {
            slots: self.slots.clone(),
            backup_count: self.backup_count,
        }
    }

    pub(crate) fn from_state(
        l_max: usize,
        state: HashPruneReservoirState,
        policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Self> {
        let HashPruneReservoirState {
            mut slots,
            backup_count,
        } = state;
        if l_max == 0 || slots.len() > l_max || usize::from(backup_count) > slots.len() {
            return Err(AnnError::log_index_error(
                "invalid serialized HashPrune reservoir boundary".to_string(),
            ));
        }
        for slot in &slots {
            slot.meta.to_state_flags()?;
        }
        let primary_count = slots.len() - usize::from(backup_count);
        if slots[..primary_count]
            .windows(2)
            .any(|pair| pair[0].hash >= pair[1].hash)
        {
            return Err(AnnError::log_index_error(
                "serialized HashPrune primary prefix is not strictly hash ordered".to_string(),
            ));
        }

        let max_backup_count = if policy.is_off() {
            0
        } else if policy.is_rhalf() {
            policy.candidate_capacity_for(l_max, primary_count)
        } else {
            MAX_INTERNAL_BACKUPS.min(l_max.saturating_sub(primary_count))
        };
        if usize::from(backup_count) > max_backup_count {
            return Err(AnnError::log_index_error(
                "serialized HashPrune backup count is incompatible with policy".to_string(),
            ));
        }
        if slots.capacity() < l_max {
            slots.reserve_exact(l_max - slots.len());
        }

        let mut reservoir = Self {
            l_max,
            slots,
            farthest_idx: 0,
            farthest_dist: 0.0,
            backup_count,
            backup_worst: 0,
        };
        if primary_count > 0 {
            reservoir.recompute_farthest_prefix(primary_count);
        }
        if backup_count > 0 {
            if policy.is_rhalf() {
                reservoir.recompute_rhalf_backup_worst(primary_count, policy);
            } else {
                reservoir.recompute_backup_worst(primary_count);
            }
        }
        Ok(reservoir)
    }

    #[inline]
    pub fn capacity_hint(&self) -> usize {
        self.l_max
    }

    /// 重新扫描所有槽，更新 farthest 缓存。O(l_max)，仅在替换后调用。
    #[inline]
    fn recompute_farthest(&mut self) {
        let mut idx = 0usize;
        let mut farthest = self.slots[0];
        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if slot_is_worse(*slot, farthest) {
                farthest = *slot;
                idx = i;
            }
        }
        self.farthest_idx = idx;
        self.farthest_dist = farthest.dist;
    }

    pub(crate) fn backfill_count(&self) -> AnnResult<usize> {
        let count = usize::from(self.backup_count);
        if count > self.l_max {
            return Err(AnnError::log_index_error(format!(
                "HashPrune internal backup count exceeds l_max={}: {count}",
                self.l_max
            )));
        }
        Ok(count)
    }

    pub(crate) fn diagnostic_prefinal_snapshot(&self) -> AnnResult<HashPrunePrefinalSnapshot> {
        let backup_count = self.backfill_count()?;
        let primary_count = self.slots.len().checked_sub(backup_count).ok_or_else(|| {
            AnnError::log_index_error("invalid diagnostic prefinal boundary".to_string())
        })?;
        let slots = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let hit_layer = if index < primary_count {
                    0
                } else {
                    slot.meta.backfill_hit_layer() as u8
                };
                (slot.target, slot.hash, slot.dist.to_bits(), hit_layer)
            })
            .collect();
        Ok(HashPrunePrefinalSnapshot {
            primary_count,
            slots,
        })
    }

    fn recompute_farthest_prefix(&mut self, primary_count: usize) {
        debug_assert!(primary_count > 0);
        let mut idx = 0usize;
        let mut farthest = self.slots[0];
        for (i, slot) in self.slots[..primary_count].iter().enumerate().skip(1) {
            if slot_is_worse(*slot, farthest) {
                farthest = *slot;
                idx = i;
            }
        }
        self.farthest_idx = idx;
        self.farthest_dist = farthest.dist;
    }

    #[inline(always)]
    fn recompute_backup_worst(&mut self, primary_count: usize) {
        let suffix = &self.slots[primary_count..];
        if suffix.is_empty() {
            self.backup_worst = 0;
            return;
        }
        let mut worst = 0usize;
        for index in 1..suffix.len() {
            if slot_is_worse(suffix[index], suffix[worst]) {
                worst = index;
            }
        }
        self.backup_worst = worst as u16;
    }

    #[inline(always)]
    fn retain_backup_candidate(
        &mut self,
        primary_count: usize,
        candidate: Slot,
        backup_limit: usize,
        backup_hash_quota: usize,
    ) {
        let capacity = backup_limit.min(self.l_max.saturating_sub(primary_count));
        if capacity == 0 {
            return;
        }

        let suffix_end = self.slots.len();
        if backup_hash_quota > 1 {
            let mut same_hash_count = 0usize;
            let mut same_hash_worst = primary_count;
            for index in primary_count..suffix_end {
                let slot = self.slots[index];
                if slot.hash != candidate.hash {
                    continue;
                }
                if slot.target == candidate.target {
                    if slot_order(candidate, slot).is_lt() {
                        self.slots[index] = candidate;
                        if index - primary_count == usize::from(self.backup_worst) {
                            self.recompute_backup_worst(primary_count);
                        }
                    }
                    return;
                }
                if same_hash_count == 0 || slot_is_worse(slot, self.slots[same_hash_worst]) {
                    same_hash_worst = index;
                }
                same_hash_count += 1;
                if same_hash_count == backup_hash_quota {
                    if slot_order(candidate, self.slots[same_hash_worst]).is_lt() {
                        self.slots[same_hash_worst] = candidate;
                        if same_hash_worst - primary_count == usize::from(self.backup_worst) {
                            self.recompute_backup_worst(primary_count);
                        }
                    }
                    return;
                }
            }

            let suffix_len = suffix_end - primary_count;
            if suffix_len < capacity {
                debug_assert!(self.slots.len() < self.slots.capacity());
                self.slots.push(candidate);
                if suffix_len == 0
                    || slot_is_worse(
                        candidate,
                        self.slots[primary_count + usize::from(self.backup_worst)],
                    )
                {
                    self.backup_worst = suffix_len as u16;
                }
                return;
            }

            debug_assert!(usize::from(self.backup_worst) < suffix_len);
            let worst = primary_count + usize::from(self.backup_worst);
            if slot_order(candidate, self.slots[worst]).is_lt() {
                self.slots[worst] = candidate;
                self.recompute_backup_worst(primary_count);
            }
            return;
        }

        if let Some(existing) = self.slots[primary_count..suffix_end]
            .iter()
            .position(|slot| slot.hash == candidate.hash)
            .map(|offset| primary_count + offset)
        {
            if !slot_order(candidate, self.slots[existing]).is_lt() {
                return;
            }
            self.slots[existing] = candidate;
            if existing - primary_count == usize::from(self.backup_worst) {
                self.recompute_backup_worst(primary_count);
            }
            return;
        }

        let suffix_len = suffix_end - primary_count;
        if suffix_len < capacity {
            debug_assert!(self.slots.len() < self.slots.capacity());
            self.slots.push(candidate);
            if suffix_len == 0
                || slot_is_worse(
                    candidate,
                    self.slots[primary_count + usize::from(self.backup_worst)],
                )
            {
                self.backup_worst = suffix_len as u16;
            }
            return;
        }

        debug_assert!(usize::from(self.backup_worst) < suffix_len);
        let worst = primary_count + usize::from(self.backup_worst);
        if !slot_order(candidate, self.slots[worst]).is_lt() {
            return;
        }
        self.slots[worst] = candidate;
        self.recompute_backup_worst(primary_count);
    }

    #[inline]
    fn rhalf_priority_order(
        policy: HashPruneBackfillPolicy,
        left: Slot,
        left_rank: usize,
        right: Slot,
        right_rank: usize,
    ) -> Ordering {
        match policy.mode() {
            super::params::PiPNNHashPruneBackfillMode::RHalfSoftTierQ3 => policy
                .priority_score(left.dist, left_rank)
                .total_cmp(&policy.priority_score(right.dist, right_rank))
                .then_with(|| left_rank.cmp(&right_rank))
                .then_with(|| left.target.cmp(&right.target))
                .then_with(|| left.hash.cmp(&right.hash)),
            super::params::PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => left_rank
                .cmp(&right_rank)
                .then_with(|| slot_order(left, right)),
            super::params::PiPNNHashPruneBackfillMode::RHalfHashSlot1
            | super::params::PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3 => left
                .dist
                .total_cmp(&right.dist)
                .then_with(|| left_rank.cmp(&right_rank))
                .then_with(|| left.target.cmp(&right.target))
                .then_with(|| left.hash.cmp(&right.hash)),
            _ if policy.is_tier_first() => left_rank
                .cmp(&right_rank)
                .then_with(|| slot_order(left, right)),
            _ => slot_order(left, right).then_with(|| left_rank.cmp(&right_rank)),
        }
    }

    #[inline]
    fn set_rhalf_primary_backup_count_at(
        &mut self,
        primary_pos: usize,
        count: usize,
    ) -> AnnResult<()> {
        debug_assert!(count <= 3);
        let meta = self.slots[primary_pos].meta.clear_backfill_count();
        self.slots[primary_pos].meta = if count == 0 {
            meta
        } else {
            meta.with_backfill_count(count)?
        };
        Ok(())
    }

    #[inline]
    fn decrement_rhalf_primary_backup_count(
        &mut self,
        primary_count: usize,
        hash: u16,
    ) -> AnnResult<()> {
        let primary_pos = self.slots[..primary_count]
            .binary_search_by_key(&hash, |slot| slot.hash)
            .map_err(|_| {
                AnnError::log_index_error(format!(
                    "HashPrune backup hash {hash} has no retained primary"
                ))
            })?;
        let count = self.slots[primary_pos].meta.backfill_count()?;
        if count == 0 {
            return Err(AnnError::log_index_error(format!(
                "HashPrune backup hash {hash} has a zero primary marker"
            )));
        }
        self.set_rhalf_primary_backup_count_at(primary_pos, count - 1)
    }

    #[inline(always)]
    fn recompute_rhalf_backup_worst(
        &mut self,
        primary_count: usize,
        policy: HashPruneBackfillPolicy,
    ) {
        let suffix_len = self.slots.len() - primary_count;
        if suffix_len == 0 {
            self.backup_worst = 0;
            return;
        }
        let mut worst = primary_count;
        for index in primary_count + 1..self.slots.len() {
            if Self::rhalf_priority_order(
                policy,
                self.slots[index],
                self.slots[index].meta.backfill_hit_layer(),
                self.slots[worst],
                self.slots[worst].meta.backfill_hit_layer(),
            )
            .is_gt()
            {
                worst = index;
            }
        }
        self.backup_worst = (worst - primary_count) as u16;
    }

    #[inline]
    fn remove_rhalf_backup_worst(
        &mut self,
        primary_count: usize,
        policy: HashPruneBackfillPolicy,
    ) -> AnnResult<()> {
        debug_assert!(self.slots.len() > primary_count);
        let worst = primary_count + usize::from(self.backup_worst);
        debug_assert!(worst < self.slots.len());
        let removed_hash = self.slots[worst].hash;
        self.slots.swap_remove(worst);
        self.decrement_rhalf_primary_backup_count(primary_count, removed_hash)?;
        self.recompute_rhalf_backup_worst(primary_count, policy);
        Ok(())
    }

    fn retain_rhalf_backup_candidate(
        &mut self,
        primary_count: usize,
        primary_pos: usize,
        candidate: Slot,
        remove_target: Option<u32>,
        policy: HashPruneBackfillPolicy,
    ) -> AnnResult<RHalfAdmissionEvent> {
        let capacity = policy.candidate_capacity_for(self.l_max, primary_count);
        if capacity == 0 {
            return Ok(RHalfAdmissionEvent::CapacityRejected);
        }

        let quota = policy.backup_hash_quota();
        debug_assert!((1..=3).contains(&quota));
        let suffix_len = self.slots.len() - primary_count;
        debug_assert!(suffix_len <= capacity);
        let marked_group_len = self.slots[primary_pos].meta.backfill_count()?;
        debug_assert!(marked_group_len <= quota);

        if marked_group_len == 0 {
            debug_assert_ne!(remove_target, Some(candidate.target));
            let mut candidate = candidate;
            candidate.meta = candidate.meta.with_backfill_hit_layer(1)?;
            if suffix_len < capacity {
                debug_assert!(self.slots.len() < self.slots.capacity());
                self.slots.push(candidate);
                self.set_rhalf_primary_backup_count_at(primary_pos, 1)?;
                if suffix_len == 0 {
                    self.backup_worst = 0;
                } else {
                    let worst = primary_count + usize::from(self.backup_worst);
                    let worst_slot = self.slots[worst];
                    if Self::rhalf_priority_order(
                        policy,
                        candidate,
                        1,
                        worst_slot,
                        worst_slot.meta.backfill_hit_layer(),
                    )
                    .is_gt()
                    {
                        self.backup_worst = suffix_len as u16;
                    }
                }
                return Ok(RHalfAdmissionEvent::Admitted);
            }

            let worst = primary_count + usize::from(self.backup_worst);
            let worst_slot = self.slots[worst];
            if !Self::rhalf_priority_order(
                policy,
                candidate,
                1,
                worst_slot,
                worst_slot.meta.backfill_hit_layer(),
            )
            .is_lt()
            {
                return Ok(RHalfAdmissionEvent::CapacityRejected);
            }
            debug_assert_ne!(worst_slot.hash, candidate.hash);
            self.slots[worst] = candidate;
            self.decrement_rhalf_primary_backup_count(primary_count, worst_slot.hash)?;
            self.set_rhalf_primary_backup_count_at(primary_pos, 1)?;
            self.recompute_rhalf_backup_worst(primary_count, policy);
            return Ok(RHalfAdmissionEvent::Replaced);
        }

        let mut old_indices = [usize::MAX; 3];
        let mut old_group = [candidate; 3];
        let mut old_len = 0usize;
        let mut outside_worst = None;
        for index in primary_count..self.slots.len() {
            let slot = self.slots[index];
            let layer = slot.meta.backfill_hit_layer();
            debug_assert!((1..=quota).contains(&layer));
            if slot.hash == candidate.hash {
                debug_assert!(old_len < quota);
                old_indices[old_len] = index;
                old_group[old_len] = slot;
                old_len += 1;
            } else if outside_worst.is_none_or(|worst| {
                let worst_slot = self.slots[worst];
                Self::rhalf_priority_order(
                    policy,
                    slot,
                    layer,
                    worst_slot,
                    worst_slot.meta.backfill_hit_layer(),
                )
                .is_gt()
            }) {
                outside_worst = Some(index);
            }
        }
        debug_assert_eq!(old_len, marked_group_len);
        old_group[..old_len].sort_unstable_by(|left, right| slot_order(*left, *right));

        let mut proposed = [candidate; 4];
        let mut proposed_len = 0usize;
        let mut candidate_merged = false;
        for &slot in &old_group[..old_len] {
            if remove_target == Some(slot.target) {
                continue;
            }
            if slot.target == candidate.target {
                proposed[proposed_len] = if slot_order(candidate, slot).is_lt() {
                    candidate
                } else {
                    Slot::new(slot.target, slot.hash, slot.dist)
                };
                candidate_merged = true;
            } else {
                proposed[proposed_len] = Slot::new(slot.target, slot.hash, slot.dist);
            }
            proposed_len += 1;
        }
        if !candidate_merged && remove_target != Some(candidate.target) {
            proposed[proposed_len] = candidate;
            proposed_len += 1;
        }
        proposed[..proposed_len].sort_unstable_by(|left, right| slot_order(*left, *right));
        proposed_len = proposed_len.min(quota);
        for (index, slot) in proposed[..proposed_len].iter_mut().enumerate() {
            slot.meta = slot.meta.with_backfill_hit_layer(index + 1)?;
        }

        let old_matches_proposed = |proposed_len: usize, proposed: &[Slot; 4]| {
            old_len == proposed_len
                && old_group[..old_len]
                    .iter()
                    .zip(&proposed[..proposed_len])
                    .all(|(left, right)| {
                        left.target == right.target
                            && left.hash == right.hash
                            && left.dist.to_bits() == right.dist.to_bits()
                    })
        };

        let pool_over_capacity = suffix_len - old_len + proposed_len > capacity;
        let mut outside_victim = None;
        let mut outside_victim_hash = None;
        if pool_over_capacity {
            debug_assert_eq!(suffix_len - old_len + proposed_len, capacity + 1);
            let mut group_victim = proposed_len.checked_sub(1);
            if let Some(outside_index) = outside_worst {
                let outside = self.slots[outside_index];
                if group_victim.is_none_or(|group_index| {
                    Self::rhalf_priority_order(
                        policy,
                        outside,
                        outside.meta.backfill_hit_layer(),
                        proposed[group_index],
                        group_index + 1,
                    )
                    .is_gt()
                }) {
                    outside_victim = Some(outside_index);
                    outside_victim_hash = Some(outside.hash);
                    group_victim = None;
                }
            }
            if group_victim.is_some() {
                proposed_len -= 1;
            }
        }

        if outside_victim.is_none() && old_matches_proposed(proposed_len, &proposed) {
            return Ok(if pool_over_capacity {
                RHalfAdmissionEvent::CapacityRejected
            } else {
                RHalfAdmissionEvent::Unchanged
            });
        }

        let shared = old_len.min(proposed_len);
        for offset in 0..shared {
            self.slots[old_indices[offset]] = proposed[offset];
        }
        if proposed_len > old_len {
            debug_assert_eq!(proposed_len, old_len + 1);
            if let Some(victim) = outside_victim.take() {
                self.slots[victim] = proposed[old_len];
            } else {
                debug_assert!(self.slots.len() < self.slots.capacity());
                self.slots.push(proposed[old_len]);
            }
        } else {
            for offset in (proposed_len..old_len).rev() {
                self.slots.swap_remove(old_indices[offset]);
            }
        }
        debug_assert!(outside_victim.is_none());
        if let Some(hash) = outside_victim_hash {
            self.decrement_rhalf_primary_backup_count(primary_count, hash)?;
        }
        self.set_rhalf_primary_backup_count_at(primary_pos, proposed_len)?;
        self.recompute_rhalf_backup_worst(primary_count, policy);
        let new_suffix_len = self.slots.len() - primary_count;
        Ok(if new_suffix_len > suffix_len {
            RHalfAdmissionEvent::Admitted
        } else {
            RHalfAdmissionEvent::Replaced
        })
    }

    #[inline]
    fn record_rhalf_admission_event(
        profile: &mut HashPruneBackfillProfile,
        event: RHalfAdmissionEvent,
    ) {
        match event {
            RHalfAdmissionEvent::Unchanged => profile.shortlist_unchanged += 1,
            RHalfAdmissionEvent::Admitted => profile.shortlist_admissions += 1,
            RHalfAdmissionEvent::Replaced => profile.shortlist_replacements += 1,
            RHalfAdmissionEvent::CapacityRejected => {
                profile.shortlist_capacity_rejections += 1;
            }
        }
    }

    #[inline]
    fn insert_primary_at(
        &mut self,
        primary_pos: usize,
        target: u32,
        hash: u16,
        dist: f32,
        new_primary_count: usize,
    ) {
        debug_assert!(self.slots.len() < self.slots.capacity());
        self.slots
            .insert(primary_pos, Slot::new(target, hash, dist));
        if new_primary_count == 1 {
            self.farthest_idx = 0;
            self.farthest_dist = dist;
            return;
        }
        if primary_pos <= self.farthest_idx {
            self.farthest_idx += 1;
        }
        if slot_is_worse(self.slots[primary_pos], self.slots[self.farthest_idx]) {
            self.farthest_idx = primary_pos;
            self.farthest_dist = dist;
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_backfill(
        &mut self,
        source: usize,
        target: u32,
        hash: u16,
        dist: f32,
    ) -> AnnResult<()> {
        let _ = source;
        self.insert_backfill_bounded(target, hash, dist, MAX_INTERNAL_BACKUPS);
        Ok(())
    }

    #[inline]
    pub(crate) fn insert_backfill_bounded(
        &mut self,
        target: u32,
        hash: u16,
        dist: f32,
        backup_limit: usize,
    ) {
        self.insert_backfill_bounded_with_hash_quota(target, hash, dist, backup_limit, 1);
    }

    #[inline]
    pub(crate) fn insert_backfill_bounded_with_hash_quota(
        &mut self,
        target: u32,
        hash: u16,
        dist: f32,
        backup_limit: usize,
        backup_hash_quota: usize,
    ) {
        debug_assert!((1..=MAX_INTERNAL_BACKUPS).contains(&backup_limit));
        debug_assert!((1..=3).contains(&backup_hash_quota));
        let marked_backups = usize::from(self.backup_count);
        debug_assert!(marked_backups <= MAX_INTERNAL_BACKUPS);
        if marked_backups == 0 {
            if self.slots.len() == self.l_max {
                let _ = self.insert_inner(target, hash, dist, None);
                return;
            }
            if let Err(primary_pos) = self.slots.binary_search_by_key(&hash, |slot| slot.hash) {
                let new_primary_count = self.slots.len() + 1;
                self.insert_primary_at(primary_pos, target, hash, dist, new_primary_count);
                return;
            }
        }

        let backup_count = marked_backups;
        debug_assert!(backup_count <= self.slots.len());
        let primary_count = self.slots.len() - backup_count;
        debug_assert!(primary_count <= self.l_max);
        debug_assert!(backup_count <= backup_limit.min(self.l_max - primary_count));

        if primary_count == self.l_max {
            debug_assert_eq!(backup_count, 0);
            let _ = self.insert_inner(target, hash, dist, None);
            return;
        }

        match self.slots[..primary_count].binary_search_by_key(&hash, |slot| slot.hash) {
            Ok(primary_pos) => {
                let primary = self.slots[primary_pos];
                if primary.target == target {
                    if candidate_is_better_than_slot(target, hash, dist, primary) {
                        self.slots[primary_pos] =
                            Slot::new_with_meta(target, hash, dist, primary.meta);
                        if primary_pos == self.farthest_idx {
                            self.recompute_farthest_prefix(primary_count);
                        }
                    }
                } else if candidate_is_better_than_slot(target, hash, dist, primary) {
                    self.slots[primary_pos] = Slot::new(target, hash, dist);
                    if primary_pos == self.farthest_idx {
                        self.recompute_farthest_prefix(primary_count);
                    }
                    self.retain_backup_candidate(
                        primary_count,
                        primary,
                        backup_limit,
                        backup_hash_quota,
                    );
                } else {
                    self.retain_backup_candidate(
                        primary_count,
                        Slot::new(target, hash, dist),
                        backup_limit,
                        backup_hash_quota,
                    );
                }
                let new_backup_count = self.slots.len() - primary_count;
                debug_assert!(new_backup_count <= backup_limit);
                self.backup_count = new_backup_count as u16;
            }
            Err(primary_pos) => {
                let new_primary_count = primary_count + 1;
                let new_backup_capacity = backup_limit.min(self.l_max - new_primary_count);
                if backup_count > new_backup_capacity {
                    debug_assert_eq!(backup_count, new_backup_capacity + 1);
                    let worst = primary_count + usize::from(self.backup_worst);
                    self.slots.swap_remove(worst);
                    self.recompute_backup_worst(primary_count);
                }
                self.insert_primary_at(primary_pos, target, hash, dist, new_primary_count);
                let new_backup_count = self.slots.len() - new_primary_count;
                debug_assert!(new_backup_count <= backup_limit);
                self.backup_count = new_backup_count as u16;
            }
        }
    }

    pub(crate) fn insert_backfill_with_policy(
        &mut self,
        source: usize,
        target: u32,
        hash: u16,
        dist: f32,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        if policy.is_rhalf() {
            self.validate_target_hash_consistency(source, target, hash)?;
        }
        self.insert_backfill_prevalidated_with_policy(source, target, hash, dist, policy, profile)
    }

    /// Inserts a candidate whose `(source, target, hash)` consistency was established at the
    /// candidate-generation or serialized-record boundary.
    #[inline]
    pub(crate) fn insert_backfill_prevalidated_with_policy(
        &mut self,
        source: usize,
        target: u32,
        hash: u16,
        dist: f32,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        if !policy.is_rhalf() {
            if policy.is_off() {
                let _ = self.insert_inner(target, hash, dist, None);
            } else {
                self.insert_backfill_bounded_with_hash_quota(
                    target,
                    hash,
                    dist,
                    policy.candidate_budget(),
                    policy.backup_hash_quota(),
                );
            }
            return Ok(());
        }

        let backup_count = usize::from(self.backup_count);
        let primary_count = self.slots.len().checked_sub(backup_count).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "HashPrune source {source} has invalid R-half backup boundary"
            ))
        })?;
        if primary_count == self.l_max {
            debug_assert_eq!(backup_count, 0);
            let _ = self.insert_inner(target, hash, dist, None);
            return Ok(());
        }

        match self.slots[..primary_count].binary_search_by_key(&hash, |slot| slot.hash) {
            Ok(primary_pos) => {
                let primary = self.slots[primary_pos];
                if primary.target == target {
                    if candidate_is_better_than_slot(target, hash, dist, primary) {
                        self.slots[primary_pos] =
                            Slot::new_with_meta(target, hash, dist, primary.meta);
                        if primary_pos == self.farthest_idx {
                            self.recompute_farthest_prefix(primary_count);
                        }
                    }
                } else if candidate_is_better_than_slot(target, hash, dist, primary) {
                    self.slots[primary_pos] = Slot::new_with_meta(target, hash, dist, primary.meta);
                    if primary_pos == self.farthest_idx {
                        self.recompute_farthest_prefix(primary_count);
                    }
                    let event = self.retain_rhalf_backup_candidate(
                        primary_count,
                        primary_pos,
                        primary,
                        Some(target),
                        policy,
                    )?;
                    Self::record_rhalf_admission_event(profile, event);
                } else {
                    let event = self.retain_rhalf_backup_candidate(
                        primary_count,
                        primary_pos,
                        Slot::new(target, hash, dist),
                        None,
                        policy,
                    )?;
                    Self::record_rhalf_admission_event(profile, event);
                }
                let new_backup_count = self.slots.len() - primary_count;
                debug_assert!(
                    new_backup_count <= policy.candidate_capacity_for(self.l_max, primary_count)
                );
                self.backup_count = new_backup_count as u16;
            }
            Err(primary_pos) => {
                let new_primary_count = primary_count + 1;
                let new_capacity = policy.candidate_capacity_for(self.l_max, new_primary_count);
                while self.slots.len() - primary_count > new_capacity {
                    self.remove_rhalf_backup_worst(primary_count, policy)?;
                }
                self.insert_primary_at(primary_pos, target, hash, dist, new_primary_count);
                let new_backup_count = self.slots.len() - new_primary_count;
                debug_assert!(new_backup_count <= new_capacity);
                self.backup_count = new_backup_count as u16;
                self.recompute_rhalf_backup_worst(new_primary_count, policy);
            }
        }
        Ok(())
    }

    /// 向水库插入一个候选 (target, hash, dist)。
    ///
    /// - 若 hash 冲突，则只保留更近的点；
    /// - 若无冲突且未满，则直接插入；
    /// - 若无冲突且已满，则若更近则淘汰当前最远点。
    pub fn insert(&mut self, target: u32, hash: u16, dist: f32) {
        let _ = self.insert_inner(target, hash, dist, None);
    }

    pub(crate) fn insert_profiled(
        &mut self,
        target: u32,
        hash: u16,
        dist: f32,
        profile: &mut HashPruneInsertProfile,
    ) {
        let _ = self.insert_inner(target, hash, dist, Some(profile));
    }

    pub(crate) fn insert_with_policy_profiled(
        &mut self,
        source: usize,
        target: u32,
        hash: u16,
        dist: f32,
        meta: HashPruneMeta,
        policy: HashPruneEvidencePolicy,
        profile: &mut HashPruneInsertProfile,
    ) -> AnnResult<()> {
        if policy.is_off() {
            self.insert_profiled(target, hash, dist, profile);
            return Ok(());
        }
        self.validate_target_hash_consistency(source, target, hash)?;
        if policy.enforces() {
            return Err(AnnError::log_index_error(
                "Phase 1 overflow policy is not installed".to_string(),
            ));
        }
        meta.to_spill_flags()?;

        if let Some(pos) = self.insert_inner(target, hash, dist, Some(profile)) {
            self.slots[pos].meta = self.slots[pos].meta.merge_prevalidated_candidate(meta);
        }
        Ok(())
    }

    /// Trusted report insertion for typed metadata that has already crossed an audited boundary.
    ///
    /// Call only after compact candidate preparation or a successful
    /// `HashPruneMeta::from_spill_flags` conversion. Slot-internal overflow metadata is never a
    /// valid input.
    #[inline]
    pub(crate) fn insert_report_prevalidated(
        &mut self,
        _source: usize,
        target: u32,
        hash: u16,
        dist: f32,
        meta: HashPruneMeta,
    ) {
        #[cfg(debug_assertions)]
        if let Err(error) = self.validate_target_hash_consistency(_source, target, hash) {
            debug_assert!(false, "{error}");
        }
        debug_assert!(meta.to_spill_flags().is_ok());
        if let Some(pos) = self.insert_inner(target, hash, dist, None) {
            let slot = &mut self.slots[pos];
            slot.meta = if slot.meta == HashPruneMeta::empty() {
                meta
            } else {
                slot.meta.merge_prevalidated_candidate(meta)
            };
        }
    }

    fn validate_target_hash_consistency(
        &self,
        source: usize,
        target: u32,
        new_hash: u16,
    ) -> AnnResult<()> {
        let Some(existing) = self
            .slots
            .iter()
            .find(|slot| slot.target == target && slot.hash != new_hash)
        else {
            return Ok(());
        };
        Err(AnnError::log_index_error(format!(
            "HashPrune reservoir duplicate hash mismatch for source {source} destination {target}: existing hash={} new hash={new_hash}",
            existing.hash
        )))
    }

    fn insert_inner(
        &mut self,
        target: u32,
        hash: u16,
        dist: f32,
        mut profile: Option<&mut HashPruneInsertProfile>,
    ) -> Option<usize> {
        if let Some(profile) = profile.as_deref_mut() {
            profile.candidates += 1;
        }
        // O(log n) 二分查找 hash 冲突
        match self.slots.binary_search_by_key(&hash, |s| s.hash) {
            Ok(pos) => {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.hash_hits += 1;
                }
                // hash 冲突：只保留更近的点
                if candidate_is_better_than_slot(target, hash, dist, self.slots[pos]) {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.hash_replacements += 1;
                    }
                    let replacement_meta = if self.slots[pos].target == target {
                        self.slots[pos].meta
                    } else {
                        HashPruneMeta::empty()
                    };
                    self.slots[pos] = Slot::new_with_meta(target, hash, dist, replacement_meta);
                    // 如果已满，更新 farthest 缓存
                    if pos == self.farthest_idx {
                        if let Some(profile) = profile.as_deref_mut() {
                            profile.farthest_recomputes += 1;
                        }
                        // 替换了最远点，需要重新扫描
                        self.recompute_farthest();
                    }
                } else if let Some(profile) = profile.as_deref_mut() {
                    profile.hash_rejects += 1;
                }
                (self.slots[pos].target == target).then_some(pos)
            }
            Err(insert_pos) => {
                if self.slots.len() < self.l_max {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.not_full_inserts += 1;
                    }
                    // 未满：有序插入，维护 slots 按 hash 排序
                    self.slots.insert(insert_pos, Slot::new(target, hash, dist));
                    // 更新 farthest 缓存（新插入的槽可能是最远的）
                    if self.slots.len() == 1 {
                        self.farthest_idx = 0;
                        self.farthest_dist = dist;
                    } else {
                        if insert_pos <= self.farthest_idx {
                            // 在最远点之前插入，最远点下标后移一位
                            self.farthest_idx += 1;
                        }
                        if slot_is_worse(self.slots[insert_pos], self.slots[self.farthest_idx]) {
                            self.farthest_idx = insert_pos;
                            self.farthest_dist = dist;
                        }
                    }
                    Some(insert_pos)
                } else {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.full_replacement_checks += 1;
                    }
                    // 已满：若新点比最远点更近，替换最远点
                    if candidate_is_better_than_slot(
                        target,
                        hash,
                        dist,
                        self.slots[self.farthest_idx],
                    ) {
                        if let Some(profile) = profile.as_deref_mut() {
                            profile.full_replacements += 1;
                        }
                        // Overwrite farthest slot in-place, then rotate to correct hash-sorted
                        // position. This replaces
                        // remove+partition_point+insert (3 memmoves) with a single rotate.
                        self.slots[self.farthest_idx] = Slot::new(target, hash, dist);
                        // Determine correct position and rotate into place
                        let final_pos = if self.farthest_idx > 0
                            && self.slots[self.farthest_idx - 1].hash > hash
                        {
                            // Need to move left: find the correct position
                            let target_pos =
                                self.slots[..self.farthest_idx].partition_point(|s| s.hash < hash);
                            self.slots[target_pos..=self.farthest_idx].rotate_right(1);
                            if let Some(profile) = profile.as_deref_mut() {
                                profile.rotate_rights += 1;
                            }
                            target_pos
                        } else if self.farthest_idx + 1 < self.slots.len()
                            && self.slots[self.farthest_idx + 1].hash < hash
                        {
                            // Need to move right: find the correct position
                            let target_pos = self.slots[self.farthest_idx + 1..]
                                .partition_point(|s| s.hash < hash)
                                + self.farthest_idx
                                + 1;
                            self.slots[self.farthest_idx..target_pos].rotate_left(1);
                            if let Some(profile) = profile.as_deref_mut() {
                                profile.rotate_lefts += 1;
                            }
                            target_pos - 1
                        } else {
                            self.farthest_idx
                        };
                        // else: already in correct position (hash order maintained)
                        if let Some(profile) = profile.as_deref_mut() {
                            profile.farthest_recomputes += 1;
                        }
                        self.recompute_farthest();
                        Some(final_pos)
                    } else {
                        if let Some(profile) = profile.as_deref_mut() {
                            profile.full_rejects += 1;
                        }
                        None
                    }
                }
            }
        }
    }

    fn finish_soft_tier_q3_fast(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        primary_count: usize,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        debug_assert_eq!(policy.mode(), PiPNNHashPruneBackfillMode::RHalfSoftTierQ3);
        debug_assert!(policy.is_rhalf());
        debug_assert!(!policy.uses_alpha_gate());

        let suffix_end = self.slots.len();
        let backup_count = suffix_end - primary_count;

        if !self.slots[0].dist.is_finite() || self.slots[0].dist < 0.0 {
            return Err(AnnError::log_index_error(format!(
                "HashPrune source {source} has invalid primary squared distance"
            )));
        }
        for pair in self.slots[..primary_count].windows(2) {
            if pair[0].hash >= pair[1].hash {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} primary hashes are not strictly ordered"
                )));
            }
            if !pair[1].dist.is_finite() || pair[1].dist < 0.0 {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} has invalid primary squared distance"
                )));
            }
        }

        profile.rows_with_nonempty_shortlist += 1;
        let mut marked_backup_total = 0_usize;
        for primary in &self.slots[..primary_count] {
            marked_backup_total += primary.meta.backfill_count()?;
        }
        if marked_backup_total != backup_count {
            return Err(AnnError::log_index_error(format!(
                "HashPrune source {source} primary markers retain {marked_backup_total} backups but suffix contains {backup_count}"
            )));
        }

        for index in primary_count..suffix_end {
            let slot = self.slots[index];
            if self.slots[..primary_count]
                .binary_search_by_key(&slot.hash, |primary| primary.hash)
                .is_err()
            {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} backup hash {} has no retained primary",
                    slot.hash
                )));
            }
            if !slot.dist.is_finite() || slot.dist < 0.0 {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} has invalid backup squared distance"
                )));
            }
            let hit_layer = slot.meta.backfill_hit_layer();
            if hit_layer == 0 || hit_layer > policy.backup_hash_quota() {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} backup hash {} has invalid hit layer {hit_layer} for quota {}",
                    slot.hash,
                    policy.backup_hash_quota()
                )));
            }
            profile.shortlisted_by_layer[hit_layer - 1] += 1;
        }

        for primary in &mut self.slots[..primary_count] {
            primary.meta = primary.meta.clear_backfill_count();
        }
        self.slots[primary_count..suffix_end].sort_unstable_by(|left, right| {
            Self::rhalf_priority_order(
                policy,
                *left,
                left.meta.backfill_hit_layer(),
                *right,
                right.meta.backfill_hit_layer(),
            )
        });

        let mut emitted_deeper = false;
        for slot in &mut self.slots[primary_count..suffix_end] {
            let hit_layer = slot.meta.backfill_hit_layer();
            slot.meta = slot.meta.clear_backfill_hit_layer();
            profile.emitted_by_layer[hit_layer - 1] += 1;
            emitted_deeper |= hit_layer > 1;
        }
        if emitted_deeper {
            profile.rows_with_deeper_backfill += 1;
        }
        profile.backfill_edges += backup_count as u64;
        profile.degree_after_sum += (primary_count + backup_count) as u64;
        profile.rows_with_backfill += 1;
        Ok(())
    }

    fn finish_backfill_impl(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
        use_soft_tier_q3_fast_path: bool,
    ) -> AnnResult<()> {
        if policy.is_off() {
            let count = self.backfill_count()?;
            if count != 0 {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune backfill policy mismatch for source {source}: off finalization received {count} backups"
                )));
            }
            debug_assert_eq!(self.backup_worst, 0);
            return Ok(());
        }

        let backup_count = usize::from(self.backup_count);
        let backup_worst = usize::from(self.backup_worst);
        self.backup_count = 0;
        self.backup_worst = 0;
        let primary_count = self.slots.len().checked_sub(backup_count).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "HashPrune source {source} has invalid final backup boundary"
            ))
        })?;
        if primary_count > self.l_max {
            return Err(AnnError::log_index_error(format!(
                "HashPrune source {} has invalid final layout: primary={} backup={} worst={} l_max={}",
                source, primary_count, backup_count, backup_worst, self.l_max
            )));
        }
        let max_backup_count = if policy.is_rhalf() {
            policy.candidate_capacity_for(self.l_max, primary_count)
        } else {
            MAX_INTERNAL_BACKUPS.min(self.l_max - primary_count)
        };
        if backup_count > max_backup_count
            || (backup_count == 0 && backup_worst != 0)
            || (backup_count != 0 && backup_worst >= backup_count)
        {
            return Err(AnnError::log_index_error(format!(
                "HashPrune source {} has invalid final layout: primary={} backup={} worst={} l_max={}",
                source, primary_count, backup_count, backup_worst, self.l_max
            )));
        }

        profile.rows_seen += 1;
        profile.primary_edges += primary_count as u64;
        profile.degree_before_sum += primary_count as u64;
        if primary_count < self.l_max {
            profile.rows_underfull_before += 1;
        }

        if backup_count == 0 {
            if policy.is_rhalf() && primary_count == self.l_max {
                profile.baseline_full_rows += 1;
            }
            profile.degree_after_sum += primary_count as u64;
            return Ok(());
        }

        if primary_count == 0 {
            return Err(AnnError::log_index_error(format!(
                "HashPrune source {source} has backups without primaries"
            )));
        }

        if use_soft_tier_q3_fast_path
            && matches!(policy.mode(), PiPNNHashPruneBackfillMode::RHalfSoftTierQ3)
        {
            return self.finish_soft_tier_q3_fast(source, policy, primary_count, profile);
        }

        let mut dmax2 = self.slots[0].dist;
        if !dmax2.is_finite() || dmax2 < 0.0 {
            return Err(AnnError::log_index_error(format!(
                "HashPrune source {source} has invalid primary squared distance"
            )));
        }
        for pair in self.slots[..primary_count].windows(2) {
            if pair[0].hash >= pair[1].hash {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} primary hashes are not strictly ordered"
                )));
            }
            if !pair[1].dist.is_finite() || pair[1].dist < 0.0 {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} has invalid primary squared distance"
                )));
            }
            if pair[1].dist.total_cmp(&dmax2).is_gt() {
                dmax2 = pair[1].dist;
            }
        }

        if policy.is_rhalf() {
            debug_assert!(!policy.post_veto_refill());
            profile.rows_with_nonempty_shortlist += 1;
            let suffix_end = self.slots.len();
            for primary_index in 0..primary_count {
                let primary = self.slots[primary_index];
                let marked = primary.meta.backfill_count()?;
                let actual = self.slots[primary_count..suffix_end]
                    .iter()
                    .filter(|backup| backup.hash == primary.hash)
                    .count();
                if marked != actual {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune source {source} primary hash {} marks {marked} backups but retains {actual}",
                        primary.hash
                    )));
                }
            }
            for index in primary_count..suffix_end {
                let slot = self.slots[index];
                if self.slots[..primary_count]
                    .binary_search_by_key(&slot.hash, |primary| primary.hash)
                    .is_err()
                {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune source {source} backup hash {} has no retained primary",
                        slot.hash
                    )));
                }
                if !slot.dist.is_finite() || slot.dist < 0.0 {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune source {source} has invalid backup squared distance"
                    )));
                }
                let hit_layer = slot.meta.backfill_hit_layer();
                if hit_layer == 0 || hit_layer > policy.backup_hash_quota() {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune source {source} backup hash {} has invalid hit layer {hit_layer} for quota {}",
                        slot.hash,
                        policy.backup_hash_quota()
                    )));
                }
                profile.shortlisted_by_layer[hit_layer - 1] += 1;
                if self.slots[..index]
                    .iter()
                    .any(|earlier| earlier.target == slot.target)
                {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune source {source} has duplicate backup target {}",
                        slot.target
                    )));
                }
                let expected_layer = 1 + self.slots[primary_count..suffix_end]
                    .iter()
                    .filter(|other| other.hash == slot.hash && slot_order(**other, slot).is_lt())
                    .count();
                if hit_layer != expected_layer {
                    return Err(AnnError::log_index_error(format!(
                        "HashPrune source {source} backup hash {} has invalid hit layer {hit_layer}, expected {expected_layer}",
                        slot.hash
                    )));
                }
            }
            for primary in &mut self.slots[..primary_count] {
                primary.meta = primary.meta.clear_backfill_count();
            }
            self.slots[primary_count..suffix_end].sort_unstable_by(|left, right| {
                Self::rhalf_priority_order(
                    policy,
                    *left,
                    left.meta.backfill_hit_layer(),
                    *right,
                    right.meta.backfill_hit_layer(),
                )
            });

            let threshold = policy.alpha_sq() * f64::from(dmax2);
            let mut write = primary_count;
            let mut vetoed = false;
            let mut emitted_deeper = false;
            for read in primary_count..suffix_end {
                let mut slot = self.slots[read];
                let hit_layer = slot.meta.backfill_hit_layer();
                debug_assert!((1..=3).contains(&hit_layer));
                let passes = !policy.uses_alpha_gate() || f64::from(slot.dist) <= threshold;
                slot.meta = slot.meta.clear_backfill_hit_layer();
                if passes {
                    self.slots[write] = slot;
                    write += 1;
                    profile.emitted_by_layer[hit_layer - 1] += 1;
                    emitted_deeper |= hit_layer > 1;
                } else {
                    profile.gate_rejected += 1;
                    profile.gate_rejected_by_layer[hit_layer - 1] += 1;
                    vetoed = true;
                }
            }
            for slot in &mut self.slots[write..suffix_end] {
                slot.meta = slot.meta.clear_backfill_hit_layer();
            }
            self.slots.truncate(write);
            if vetoed {
                profile.rows_with_veto_holes += 1;
            }
            if emitted_deeper {
                profile.rows_with_deeper_backfill += 1;
            }
            let emitted = write - primary_count;
            profile.backfill_edges += emitted as u64;
            profile.degree_after_sum += (primary_count + emitted) as u64;
            if emitted != 0 {
                profile.rows_with_backfill += 1;
            }
            return Ok(());
        }

        let original_len = self.slots.len();
        self.slots[primary_count..original_len]
            .sort_unstable_by(|left, right| slot_order(*left, *right));
        let mut unique_end = primary_count;
        for read in primary_count..original_len {
            let backup = self.slots[read];
            if self.slots[..primary_count]
                .iter()
                .all(|primary| primary.hash != backup.hash)
            {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} backup hash {} has no retained primary",
                    backup.hash
                )));
            }
            if self.slots[..primary_count]
                .iter()
                .any(|primary| primary.target == backup.target)
            {
                if policy.backup_hash_quota() > 1 {
                    continue;
                }
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} has duplicate backup target {}",
                    backup.target
                )));
            }
            if self.slots[primary_count..unique_end]
                .iter()
                .any(|earlier| earlier.target == backup.target)
            {
                if policy.backup_hash_quota() > 1 {
                    continue;
                }
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} has duplicate backup target {}",
                    backup.target
                )));
            }
            if self.slots[primary_count..unique_end]
                .iter()
                .filter(|earlier| earlier.hash == backup.hash)
                .count()
                >= policy.backup_hash_quota()
            {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} backup hash {} exceeds quota {}",
                    backup.hash,
                    policy.backup_hash_quota()
                )));
            }
            self.slots[unique_end] = backup;
            unique_end += 1;
        }
        self.slots.truncate(unique_end);
        let original_len = self.slots.len();
        for pair in self.slots[primary_count..original_len].windows(2) {
            if !slot_order(pair[0], pair[1]).is_lt() {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} backup suffix is not strictly ordered"
                )));
            }
        }

        let threshold = policy.alpha_sq() * f64::from(dmax2);
        let emit_budget = if policy.is_tiered() {
            policy
                .budget()
                .min(policy.degree_target().saturating_sub(primary_count))
                .min(self.l_max - primary_count)
        } else {
            policy.budget().min(self.l_max - primary_count)
        };
        let mut eligible_end = primary_count;
        for read in primary_count..original_len {
            let slot = self.slots[read];
            if !slot.dist.is_finite() || slot.dist < 0.0 {
                return Err(AnnError::log_index_error(format!(
                    "HashPrune source {source} has invalid backup squared distance"
                )));
            }
            if f64::from(slot.dist) > threshold {
                profile.gate_rejected += 1;
            } else {
                self.slots[eligible_end] = slot;
                eligible_end += 1;
            }
        }

        let eligible = eligible_end - primary_count;
        let emitted = eligible.min(emit_budget);
        profile.budget_rejected += (eligible - emitted) as u64;
        if policy.is_tiered() && emitted != 0 {
            let mut candidates = [self.slots[primary_count]; MAX_INTERNAL_BACKUPS];
            candidates[..eligible].copy_from_slice(&self.slots[primary_count..eligible_end]);
            let mut written = 0usize;
            for layer in 0..policy.backup_hash_quota() {
                for index in 0..eligible {
                    let candidate = candidates[index];
                    let hit_rank = candidates[..index]
                        .iter()
                        .filter(|earlier| earlier.hash == candidate.hash)
                        .count();
                    if hit_rank != layer {
                        continue;
                    }
                    self.slots[primary_count + written] = candidate;
                    written += 1;
                    if written == emitted {
                        break;
                    }
                }
                if written == emitted {
                    break;
                }
            }
            debug_assert_eq!(written, emitted);
        } else if policy.is_stratified() && eligible > emitted && emitted > 1 {
            let nearest = (3 * emitted).div_ceil(4);
            let farthest = emitted - nearest;
            let farthest_start = eligible_end - farthest;
            let mut write = primary_count + nearest;
            for read in farthest_start..eligible_end {
                self.slots[write] = self.slots[read];
                write += 1;
            }
        }
        self.slots.truncate(primary_count + emitted);
        profile.backfill_edges += emitted as u64;
        profile.degree_after_sum += (primary_count + emitted) as u64;
        if emitted != 0 {
            profile.rows_with_backfill += 1;
        }
        Ok(())
    }

    pub(crate) fn finish_backfill(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        self.finish_backfill_impl(source, policy, profile, true)
    }

    #[cfg(test)]
    fn finish_backfill_checked_for_test(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<()> {
        self.finish_backfill_impl(source, policy, profile, false)
    }

    pub(crate) fn into_neighbors_with_backfill(
        self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<Vec<u32>> {
        let row = self.into_finalized_slots_with_backfill(source, policy, profile)?;
        Ok(row.slots.into_iter().map(|slot| slot.target).collect())
    }

    pub(crate) fn into_slots_with_backfill(
        self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<Vec<Slot>> {
        Ok(self
            .into_finalized_slots_with_backfill(source, policy, profile)?
            .slots)
    }

    pub(crate) fn into_finalized_slots_with_backfill(
        mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<FinalizedHashPruneRow> {
        let primary_count = self
            .slots
            .len()
            .checked_sub(self.backfill_count()?)
            .ok_or_else(|| {
                AnnError::log_index_error(format!("invalid HashPrune boundary for source {source}"))
            })?;
        self.finish_backfill(source, policy, profile)?;
        Ok(FinalizedHashPruneRow {
            slots: self.slots,
            primary_count,
        })
    }

    pub(crate) fn drain_neighbors_with_backfill(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<Vec<u32>> {
        let row = self.drain_finalized_slots_with_backfill(source, policy, profile)?;
        Ok(row.slots.into_iter().map(|slot| slot.target).collect())
    }

    pub(crate) fn copy_neighbors_with_backfill_and_clear(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<Vec<u32>> {
        self.finish_backfill(source, policy, profile)?;
        let neighbors = self.slots.iter().map(|slot| slot.target).collect();
        self.clear();
        Ok(neighbors)
    }

    pub(crate) fn copy_finalized_slots_with_backfill_and_clear(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<FinalizedHashPruneRow> {
        let primary_count = self
            .slots
            .len()
            .checked_sub(self.backfill_count()?)
            .ok_or_else(|| {
                AnnError::log_index_error(format!("invalid HashPrune boundary for source {source}"))
            })?;
        self.finish_backfill(source, policy, profile)?;
        let slots = self.slots.clone();
        self.clear();
        Ok(FinalizedHashPruneRow {
            slots,
            primary_count,
        })
    }

    pub(crate) fn drain_slots_with_backfill(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<Vec<Slot>> {
        Ok(self
            .drain_finalized_slots_with_backfill(source, policy, profile)?
            .slots)
    }

    pub(crate) fn drain_finalized_slots_with_backfill(
        &mut self,
        source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<FinalizedHashPruneRow> {
        let primary_count = self
            .slots
            .len()
            .checked_sub(self.backfill_count()?)
            .ok_or_else(|| {
                AnnError::log_index_error(format!("invalid HashPrune boundary for source {source}"))
            })?;
        self.finish_backfill(source, policy, profile)?;
        let slots = std::mem::take(&mut self.slots);
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
        Ok(FinalizedHashPruneRow {
            slots,
            primary_count,
        })
    }

    fn assert_no_unfinalized_backfill(&self) {
        let count = self
            .backfill_count()
            .expect("valid internal HashPrune backup marker");
        assert_eq!(
            count, 0,
            "backfill reservoir must use policy-aware extraction"
        );
    }

    /// 取出当前水库中的邻居 ID。
    pub fn into_neighbors(self) -> Vec<u32> {
        self.assert_no_unfinalized_backfill();
        self.slots.into_iter().map(|s| s.target).collect()
    }

    pub fn into_slots(self) -> Vec<Slot> {
        self.assert_no_unfinalized_backfill();
        self.slots
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn drain_neighbors(&mut self) -> Vec<u32> {
        self.assert_no_unfinalized_backfill();
        let slots = std::mem::take(&mut self.slots);
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
        slots.into_iter().map(|s| s.target).collect()
    }

    pub(crate) fn copy_neighbors_and_clear(&mut self) -> Vec<u32> {
        self.assert_no_unfinalized_backfill();
        let neighbors = self.slots.iter().map(|slot| slot.target).collect();
        self.clear();
        neighbors
    }

    pub fn drain_slots(&mut self) -> Vec<Slot> {
        self.assert_no_unfinalized_backfill();
        let slots = std::mem::take(&mut self.slots);
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
        slots
    }
}

#[inline]
fn hashprune_candidate_total_order(
    left_target: u32,
    left_hash: u16,
    left_dist: f32,
    right_target: u32,
    right_hash: u16,
    right_dist: f32,
) -> Ordering {
    left_dist
        .total_cmp(&right_dist)
        .then_with(|| left_target.cmp(&right_target))
        .then_with(|| left_hash.cmp(&right_hash))
}

#[inline]
fn candidate_order(target: u32, hash: u16, dist: f32, other: Slot) -> Ordering {
    hashprune_candidate_total_order(target, hash, dist, other.target, other.hash, other.dist)
}

#[inline]
fn candidate_is_better_than_slot(target: u32, hash: u16, dist: f32, slot: Slot) -> bool {
    candidate_order(target, hash, dist, slot).is_lt()
}

#[inline]
fn slot_is_worse(left: Slot, right: Slot) -> bool {
    hashprune_candidate_total_order(
        left.target,
        left.hash,
        left.dist,
        right.target,
        right.hash,
        right.dist,
    )
    .is_gt()
}

#[inline]
fn slot_order(left: Slot, right: Slot) -> Ordering {
    hashprune_candidate_total_order(
        left.target,
        left.hash,
        left.dist,
        right.target,
        right.hash,
        right.dist,
    )
}

/// 为每个向量预计算 m 维 sketch：sketch[v][i] = v · H_i。
/// 其中 H_i 是随机超平面。
pub fn compute_sketches(
    dataset: &InmemDataset<f32>,
    num_points: usize,
    params: &PiPNNParams,
    num_threads: u32,
) -> AnnResult<SketchStore> {
    let store = super::point_store::InmemDatasetPointStore::new(dataset, num_points);
    compute_sketches_from_store(&store, num_points, params, num_threads)
}

pub fn compute_sketches_from_store(
    dataset: &dyn PointStore,
    num_points: usize,
    params: &PiPNNParams,
    num_threads: u32,
) -> AnnResult<SketchStore> {
    let dim = dataset.dim();
    let m = params.m_hash_bits;
    let hyperplanes = generate_hyperplanes(dim, m, params.random_seed);

    let mut sketch_data = vec![0.0f32; num_points.saturating_mul(m)];

    // 对每个点计算 sketch，并直接写入目标缓冲区。
    let build_row = |i: usize, out_row: &mut [f32]| -> AnnResult<()> {
        fill_sketch_row(dataset, i, &hyperplanes, out_row)
    };

    if num_threads == 1 {
        for (i, row) in sketch_data.chunks_exact_mut(m).enumerate() {
            build_row(i, row)?;
        }
    } else if num_threads == 0 {
        sketch_data
            .par_chunks_exact_mut(m)
            .enumerate()
            .try_for_each(|(i, row)| build_row(i, row))?;
    } else {
        with_rayon_thread_pool(num_threads, || {
            sketch_data
                .par_chunks_exact_mut(m)
                .enumerate()
                .try_for_each(|(i, row)| build_row(i, row))
        })??;
    }

    Ok(SketchStore {
        data: sketch_data,
        width: m,
    })
}

pub fn generate_hyperplanes(dim: usize, m: usize, random_seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(random_seed ^ 0x5a5a_5a5a_1234_5678);
    let dist = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let mut hyperplanes: Vec<f32> = Vec::with_capacity(m.saturating_mul(dim));
    for _ in 0..m {
        for _ in 0..dim {
            hyperplanes.push(dist.sample(&mut rng));
        }
    }
    hyperplanes
}

pub fn fill_sketch_row(
    dataset: &dyn PointStore,
    point_idx: usize,
    hyperplanes: &[f32],
    out_row: &mut [f32],
) -> AnnResult<()> {
    let dim = dataset.dim();
    if hyperplanes.len() != out_row.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Hyperplane shape mismatch: hyperplanes={} out_row={} dim={}",
            hyperplanes.len(),
            out_row.len(),
            dim
        )));
    }

    let mut point = vec![0.0f32; dim];
    dataset.read_point_into(point_idx as u32, &mut point)?;
    for (h_idx, h) in hyperplanes.chunks_exact(dim).enumerate() {
        let mut acc = 0.0f32;
        for (a, b) in point.iter().zip(h.iter()) {
            acc += *a * *b;
        }
        out_row[h_idx] = acc;
    }
    Ok(())
}

/// 根据个性化残差 sketch 计算哈希 h_p(c)。
///
/// 按照论文公式：使用 sign(sketch(c)[i] - sketch(p)[i]) 作为第 i 位。
pub fn compute_hash(sketch_p: &[f32], sketch_c: &[f32], m_bits: usize) -> u16 {
    assert!(m_bits <= 16, "m_bits must be <= 16");
    let mut h: u16 = 0;
    for i in 0..m_bits {
        let bit = if sketch_c[i] - sketch_p[i] >= 0.0 {
            1u16
        } else {
            0u16
        };
        h |= bit << i;
    }
    h
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::hint::black_box;
    #[cfg(debug_assertions)]
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::Instant;

    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    use super::{HashPruneInsertProfile, HashPruneReservoir, Slot, slot_order};
    use crate::algorithm::pipnn::hash_prune_backfill::{
        HashPruneBackfillPolicy, HashPruneBackfillProfile,
    };
    use crate::algorithm::pipnn::hash_prune_evidence::{
        HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneMeta,
    };
    use crate::algorithm::pipnn::params::{
        PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNParams,
    };

    fn sorted_neighbors(mut reservoir: HashPruneReservoir) -> Vec<u32> {
        let mut neighbors = reservoir.drain_neighbors();
        neighbors.sort_unstable();
        neighbors
    }

    #[test]
    fn runtime_row_round_trip_preserves_future_insertions() {
        let mut original = HashPruneReservoir::new(8);
        for (target, hash, dist) in [(11, 3, 4.0), (12, 1, 2.0), (13, 7, 3.0)] {
            original.insert(target, hash, dist);
        }
        let slots = original.slots().to_vec();
        let state = original.runtime_state();

        let mut restored = HashPruneReservoir::new(8);
        restored.load_runtime_row(&slots, state).unwrap();
        original.insert(21, 4, 1.0);
        restored.insert(21, 4, 1.0);

        let slot_keys = |reservoir: &HashPruneReservoir| {
            reservoir
                .slots()
                .iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits(), slot.meta))
                .collect::<Vec<_>>()
        };
        assert_eq!(slot_keys(&restored), slot_keys(&original));
        assert_eq!(restored.runtime_state(), original.runtime_state());
    }

    #[test]
    fn copied_finalization_keeps_scratch_capacity() {
        let mut reservoir = HashPruneReservoir::new(8);
        reservoir.insert(11, 3, 4.0);
        reservoir.insert(12, 1, 2.0);
        let capacity = reservoir.slots.capacity();
        let mut profile = HashPruneBackfillProfile::default();

        let row = reservoir
            .copy_finalized_slots_with_backfill_and_clear(
                0,
                HashPruneBackfillPolicy::off(),
                &mut profile,
            )
            .unwrap();

        assert_eq!(row.slots.len(), 2);
        assert!(reservoir.slots.is_empty());
        assert_eq!(reservoir.slots.capacity(), capacity);
    }

    fn insert_report(
        reservoir: &mut HashPruneReservoir,
        profile: &mut HashPruneInsertProfile,
        target: u32,
        hash: u16,
        dist: f32,
        class: HashPruneEvidenceClass,
    ) {
        reservoir
            .insert_with_policy_profiled(
                0,
                target,
                hash,
                dist,
                HashPruneMeta::with_evidence(class),
                HashPruneEvidencePolicy::report(),
                profile,
            )
            .unwrap();
    }

    fn insert_report_three_search_reference(
        reservoir: &mut HashPruneReservoir,
        profile: &mut HashPruneInsertProfile,
        target: u32,
        hash: u16,
        dist: f32,
        meta: HashPruneMeta,
    ) {
        meta.to_spill_flags().unwrap();
        let prior_same_target_meta = reservoir
            .slots
            .binary_search_by_key(&hash, |slot| slot.hash)
            .ok()
            .and_then(|pos| {
                (reservoir.slots[pos].target == target).then_some(reservoir.slots[pos].meta)
            });

        reservoir.insert_profiled(target, hash, dist, profile);

        let Ok(pos) = reservoir
            .slots
            .binary_search_by_key(&hash, |slot| slot.hash)
        else {
            return;
        };
        if reservoir.slots[pos].target != target {
            return;
        }
        let base_meta = prior_same_target_meta.unwrap_or(reservoir.slots[pos].meta);
        reservoir.slots[pos].meta = base_meta.merge_candidate(meta).unwrap();
    }

    fn slot_snapshot(reservoir: &HashPruneReservoir) -> Vec<(u32, u16, u32, HashPruneMeta)> {
        reservoir
            .slots()
            .iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits(), slot.meta))
            .collect()
    }

    fn backfill_snapshot(reservoir: &HashPruneReservoir) -> (usize, Vec<(u32, u16, u32)>) {
        let backup_count = reservoir.backfill_count().unwrap();
        let slots = reservoir
            .slots()
            .iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect();
        (backup_count, slots)
    }

    fn finalized_neighbors(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        budget: usize,
        alpha: f32,
    ) -> Vec<u32> {
        let mut reservoir = HashPruneReservoir::new(l_max);
        for &(target, hash, dist) in candidates {
            reservoir.insert_backfill_bounded(target, hash, dist, budget);
        }
        let mut profile = HashPruneBackfillProfile::default();
        reservoir
            .into_neighbors_with_backfill(
                0,
                HashPruneBackfillPolicy::distance_gated(budget, alpha),
                &mut profile,
            )
            .unwrap()
    }

    fn finalized_tiered_d52_neighbors(candidates: &[(u32, u16, f32)], l_max: usize) -> Vec<u32> {
        let policy = HashPruneBackfillPolicy::distance_gated_tiered_d52(8, 2.0);
        let mut reservoir = HashPruneReservoir::new(l_max);
        for &(target, hash, dist) in candidates {
            reservoir.insert_backfill_bounded_with_hash_quota(
                target,
                hash,
                dist,
                policy.candidate_budget(),
                policy.backup_hash_quota(),
            );
        }
        reservoir
            .into_neighbors_with_backfill(0, policy, &mut HashPruneBackfillProfile::default())
            .unwrap()
    }

    fn tiered_rhalf_policy(ablation: PiPNNHashPruneBackfillAblation) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf;
        params.hashprune_backfill_ablation = ablation;
        params.hashprune_backfill_budget = 0;
        params.hashprune_backfill_alpha = 1.1;
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    fn named_rhalf_policy_with_lmax(
        mode: PiPNNHashPruneBackfillMode,
        l_max: usize,
    ) -> HashPruneBackfillPolicy {
        named_rhalf_policy_with_lmax_and_cap(mode, l_max, None)
    }

    fn named_rhalf_policy_with_lmax_and_cap(
        mode: PiPNNHashPruneBackfillMode,
        l_max: usize,
        cap: Option<usize>,
    ) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.l_max = l_max;
        params.hashprune_backfill_mode = mode;
        params.hashprune_rhalf_max_backfill_per_row = cap;
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    fn named_rhalf_policy(mode: PiPNNHashPruneBackfillMode) -> HashPruneBackfillPolicy {
        named_rhalf_policy_with_lmax(mode, 4)
    }

    fn named_rhalf_cross_layer_backups(
        mode: PiPNNHashPruneBackfillMode,
        a2_dist_sq: f32,
    ) -> Vec<u32> {
        let candidates = [
            (10, 10, 0.25),
            (101, 10, 0.30),
            (102, 10, a2_dist_sq),
            (20, 20, 0.25),
            (201, 20, 1.00),
        ];
        let policy = named_rhalf_policy(mode);
        let mut targets = finalized_tiered_rhalf_snapshot(&candidates, 4, policy)
            .into_iter()
            .map(|(target, _, _)| target)
            .filter(|target| *target >= 100)
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets
    }

    fn tiered_rhalf_reservoir(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        policy: HashPruneBackfillPolicy,
    ) -> (HashPruneReservoir, HashPruneBackfillProfile) {
        let mut reservoir = HashPruneReservoir::new(l_max);
        let mut profile = HashPruneBackfillProfile::default();
        for &(target, hash, dist) in candidates {
            reservoir
                .insert_backfill_with_policy(0, target, hash, dist, policy, &mut profile)
                .unwrap();
        }
        (reservoir, profile)
    }

    #[test]
    fn finalized_row_reports_the_exact_primary_prefix() {
        let policy = named_rhalf_policy_with_lmax(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, 8);
        let candidates = [
            (10, 1, 1.0),
            (11, 1, 1.1),
            (12, 1, 1.2),
            (20, 2, 0.8),
            (21, 2, 1.0),
            (30, 3, 0.7),
        ];
        let (reservoir, mut profile) = tiered_rhalf_reservoir(&candidates, 8, policy);
        let row = reservoir
            .into_finalized_slots_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(row.primary_count, 3);
        assert!(
            row.slots[..row.primary_count]
                .windows(2)
                .all(|pair| pair[0].hash < pair[1].hash)
        );
        assert_eq!(row.slots.len() as u64, profile.degree_after_sum);
        assert_eq!(row.primary_count as u64, profile.primary_edges);
    }

    #[test]
    fn finalized_row_wrapper_preserves_existing_ordered_slots_and_profile() {
        let policy = named_rhalf_policy_with_lmax(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, 8);
        let mut rng = StdRng::seed_from_u64(0x0050_5249_4d41_5259);
        let candidates = (0..128_u32)
            .map(|target| {
                let hash = rng.random_range(0_u16..16);
                let dist = rng.random_range(1_u32..=100_000) as f32 / 1_000.0;
                (target + 1, hash, dist)
            })
            .collect::<Vec<_>>();
        let (left, mut left_profile) = tiered_rhalf_reservoir(&candidates, 8, policy);
        let (right, mut right_profile) = tiered_rhalf_reservoir(&candidates, 8, policy);
        let expected = left
            .into_slots_with_backfill(0, policy, &mut left_profile)
            .unwrap();
        let actual = right
            .into_finalized_slots_with_backfill(0, policy, &mut right_profile)
            .unwrap();
        let snapshot = |slots: Vec<Slot>| {
            slots
                .into_iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(snapshot(actual.slots), snapshot(expected));
        assert_eq!(right_profile, left_profile);
    }

    fn finalize_soft_tier_q3_path(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        checked: bool,
    ) -> (
        Vec<(u32, u16, u32, HashPruneMeta)>,
        HashPruneBackfillProfile,
    ) {
        finalize_soft_tier_q3_path_with_cap(candidates, l_max, checked, None)
    }

    fn finalize_soft_tier_q3_path_with_cap(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        checked: bool,
        cap: Option<usize>,
    ) -> (
        Vec<(u32, u16, u32, HashPruneMeta)>,
        HashPruneBackfillProfile,
    ) {
        let policy = named_rhalf_policy_with_lmax_and_cap(
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            l_max,
            cap,
        );
        let (mut reservoir, mut profile) = tiered_rhalf_reservoir(candidates, l_max, policy);
        if checked {
            reservoir
                .finish_backfill_checked_for_test(0, policy, &mut profile)
                .unwrap();
        } else {
            reservoir.finish_backfill(0, policy, &mut profile).unwrap();
        }
        assert_eq!(reservoir.backfill_count().unwrap(), 0);
        (slot_snapshot(&reservoir), profile)
    }

    fn assert_soft_tier_q3_paths_match(candidates: &[(u32, u16, f32)], l_max: usize) {
        let checked = finalize_soft_tier_q3_path(candidates, l_max, true);
        let fast = finalize_soft_tier_q3_path(candidates, l_max, false);
        assert_eq!(fast, checked);
    }

    fn reference_tiered_rhalf_slots(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        policy: HashPruneBackfillPolicy,
    ) -> Vec<(u32, u16, u32)> {
        let mut baseline = HashPruneReservoir::new(l_max);
        for &(target, hash, dist) in candidates {
            baseline.insert(target, hash, dist);
        }
        let mut primary = baseline.slots;
        if primary.is_empty() || primary.len() == l_max {
            return primary
                .into_iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                .collect();
        }

        let mut groups = BTreeMap::<u16, BTreeMap<u32, Slot>>::new();
        for primary_slot in &primary {
            groups.insert(primary_slot.hash, BTreeMap::new());
        }
        for &(target, hash, dist) in candidates {
            let Some(group) = groups.get_mut(&hash) else {
                continue;
            };
            let primary_slot = primary
                .iter()
                .find(|slot| slot.hash == hash)
                .copied()
                .unwrap();
            if primary_slot.target == target {
                continue;
            }
            let candidate = Slot::new(target, hash, dist);
            group
                .entry(target)
                .and_modify(|current| {
                    if slot_order(candidate, *current).is_lt() {
                        *current = candidate;
                    }
                })
                .or_insert(candidate);
        }

        let quota = policy.backup_hash_quota();
        let mut ordered_groups = Vec::new();
        for (_, group) in groups {
            let mut group = group.into_values().collect::<Vec<_>>();
            group.sort_unstable_by(|left, right| slot_order(*left, *right));
            group.truncate(quota);
            ordered_groups.push(group);
        }
        let mut shortlist = ordered_groups
            .into_iter()
            .flat_map(|group| {
                group
                    .into_iter()
                    .enumerate()
                    .map(|(layer, slot)| (slot, layer + 1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        shortlist.sort_unstable_by(|(left, left_rank), (right, right_rank)| {
            HashPruneReservoir::rhalf_priority_order(policy, *left, *left_rank, *right, *right_rank)
        });
        shortlist.truncate(policy.candidate_capacity_for(l_max, primary.len()));

        let dmax2 = primary
            .iter()
            .map(|slot| slot.dist)
            .max_by(f32::total_cmp)
            .unwrap();
        let threshold = policy.alpha_sq() * f64::from(dmax2);
        if policy.uses_alpha_gate() {
            shortlist.retain(|(slot, _)| f64::from(slot.dist) <= threshold);
        }
        primary.extend(shortlist.into_iter().map(|(slot, _)| slot));
        primary
            .into_iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect()
    }

    fn finalized_tiered_rhalf_snapshot(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        policy: HashPruneBackfillPolicy,
    ) -> Vec<(u32, u16, u32)> {
        let (reservoir, mut profile) = tiered_rhalf_reservoir(candidates, l_max, policy);
        reservoir
            .into_slots_with_backfill(0, policy, &mut profile)
            .unwrap()
            .into_iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect()
    }

    #[test]
    fn hashprune_soft_tier_q3_resolves_a2_b1_at_fixed_alpha_boundary() {
        assert_eq!(
            named_rhalf_cross_layer_backups(
                PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
                0.90f32.powi(2),
            ),
            vec![101, 102]
        );
        assert_eq!(
            named_rhalf_cross_layer_backups(
                PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
                0.95f32.powi(2),
            ),
            vec![101, 201]
        );
    }

    #[test]
    fn hashprune_soft_tier_q3_fast_finalizer_matches_checked_edge_cases() {
        let cases = [
            (1, vec![]),
            (1, vec![(1, 1, 1.0_f32)]),
            (
                4,
                vec![
                    (10, 10, 1.0),
                    (20, 20, 1.0),
                    (30, 30, 1.0),
                    (40, 40, 1.0),
                    (11, 10, 1.1),
                ],
            ),
            (
                8,
                vec![
                    (100, 10, 0.50),
                    (101, 10, 0.60),
                    (102, 10, 0.70),
                    (103, 10, 0.80),
                    (200, 20, 0.50),
                    (201, 20, 0.60),
                    (202, 20, 0.70),
                    (203, 20, 0.80),
                    (300, 30, 0.50),
                    (301, 30, 0.60),
                    (302, 30, 0.70),
                    (303, 30, 0.80),
                ],
            ),
            (
                7,
                vec![
                    (1000, 1, 1.0),
                    (1001, 1, 1.0),
                    (2000, 2, 1.0),
                    (2001, 2, 1.0),
                    (3000, 3, 1.0),
                    (3001, 3, 1.0),
                    (4000, 4, 1.0),
                    (4001, 4, 1.0),
                ],
            ),
        ];
        for (l_max, candidates) in cases {
            assert_soft_tier_q3_paths_match(&candidates, l_max);
        }
    }

    #[test]
    fn hashprune_soft_tier_q3_fast_finalizer_matches_checked_seeded_cases() {
        let mut rng = StdRng::seed_from_u64(0x534f_4654_5143_4641);
        for case in 0..256_u32 {
            let l_max = rng.random_range(1_usize..=32);
            let candidate_count = rng.random_range(0_usize..=l_max * 8);
            let hash_upper = u16::try_from(l_max * 2).unwrap();
            let mut candidates = Vec::with_capacity(candidate_count * 2);
            for ordinal in 0..candidate_count {
                let target = case * 10_000 + u32::try_from(ordinal).unwrap();
                let hash = rng.random_range(0_u16..hash_upper);
                let dist = rng.random_range(1_u32..=100_000) as f32 / 1_000.0;
                candidates.push((target, hash, dist));
                if rng.random_bool(0.20) {
                    let repeated = rng.random_range(1_u32..=100_000) as f32 / 1_000.0;
                    candidates.push((target, hash, repeated));
                }
            }
            candidates.shuffle(&mut rng);
            assert_soft_tier_q3_paths_match(&candidates, l_max);
        }
    }

    #[test]
    fn hashprune_rhalf_row_cap_fast_and_checked_finalizers_match() {
        let candidates = (0_u16..8)
            .flat_map(|hash| {
                [
                    (
                        1_000 + u32::from(hash),
                        hash,
                        1.00 + f32::from(hash) * 0.001,
                    ),
                    (
                        2_000 + u32::from(hash),
                        hash,
                        1.01 + f32::from(hash) * 0.001,
                    ),
                    (
                        3_000 + u32::from(hash),
                        hash,
                        1.02 + f32::from(hash) * 0.001,
                    ),
                    (
                        4_000 + u32::from(hash),
                        hash,
                        1.03 + f32::from(hash) * 0.001,
                    ),
                ]
            })
            .collect::<Vec<_>>();
        for cap in [2, 3, 4] {
            let checked = finalize_soft_tier_q3_path_with_cap(&candidates, 8, true, Some(cap));
            let fast = finalize_soft_tier_q3_path_with_cap(&candidates, 8, false, Some(cap));
            assert_eq!(fast, checked, "cap={cap}");
            assert!(
                fast.0.len().saturating_sub(8) <= cap,
                "cap={cap} row={:?}",
                fast.0
            );
        }
    }

    fn soft_tier_q3_reservoir_with_backups() -> (HashPruneReservoir, HashPruneBackfillPolicy) {
        let candidates = [
            (100, 10, 1.0),
            (101, 10, 1.1),
            (200, 20, 1.0),
            (201, 20, 1.1),
            (300, 30, 1.0),
            (301, 30, 1.1),
        ];
        let policy = named_rhalf_policy_with_lmax(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, 8);
        (tiered_rhalf_reservoir(&candidates, 8, policy).0, policy)
    }

    #[test]
    fn reservoir_state_restore_preserves_future_insertions() {
        let mut expected = HashPruneReservoir::new(4);
        for (target, hash, dist) in [(10, 1, 4.0), (20, 2, 3.0), (30, 3, 2.0)] {
            expected.insert(target, hash, dist);
        }
        let state = expected.state_snapshot();
        let mut restored =
            HashPruneReservoir::from_state(4, state, HashPruneBackfillPolicy::off()).unwrap();

        expected.insert(40, 4, 1.0);
        restored.insert(40, 4, 1.0);
        assert_eq!(slot_snapshot(&expected), slot_snapshot(&restored));
    }

    #[test]
    fn reservoir_state_restore_rebuilds_softtier_backup_cache() {
        let (reservoir, policy) = soft_tier_q3_reservoir_with_backups();
        let state = reservoir.state_snapshot();
        let restored = HashPruneReservoir::from_state(8, state, policy).unwrap();
        assert_eq!(slot_snapshot(&reservoir), slot_snapshot(&restored));
        assert_eq!(backfill_snapshot(&reservoir), backfill_snapshot(&restored));
    }

    #[test]
    fn reservoir_state_restore_rejects_backup_policy_mismatch() {
        let (reservoir, _) = soft_tier_q3_reservoir_with_backups();
        assert!(
            HashPruneReservoir::from_state(
                8,
                reservoir.state_snapshot(),
                HashPruneBackfillPolicy::off(),
            )
            .is_err()
        );
    }

    #[test]
    fn hashprune_soft_tier_q3_fast_finalizer_rejects_zero_hit_layer() {
        let (mut reservoir, policy) = soft_tier_q3_reservoir_with_backups();
        let primary_count = reservoir.slots.len() - reservoir.backfill_count().unwrap();
        reservoir.slots[primary_count].meta = reservoir.slots[primary_count]
            .meta
            .clear_backfill_hit_layer();
        let error = reservoir
            .finish_backfill(7, policy, &mut HashPruneBackfillProfile::default())
            .unwrap_err();
        assert!(error.to_string().contains("invalid hit layer"));
    }

    #[test]
    fn hashprune_soft_tier_q3_fast_finalizer_rejects_backup_without_primary() {
        let (mut reservoir, policy) = soft_tier_q3_reservoir_with_backups();
        let primary_count = reservoir.slots.len() - reservoir.backfill_count().unwrap();
        reservoir.slots[primary_count].hash = u16::MAX;
        let error = reservoir
            .finish_backfill(7, policy, &mut HashPruneBackfillProfile::default())
            .unwrap_err();
        assert!(error.to_string().contains("has no retained primary"));
    }

    #[test]
    fn hashprune_soft_tier_q3_fast_finalizer_rejects_marker_total_mismatch() {
        let (mut reservoir, policy) = soft_tier_q3_reservoir_with_backups();
        reservoir.slots[0].meta = reservoir.slots[0].meta.clear_backfill_count();
        let error = reservoir
            .finish_backfill(7, policy, &mut HashPruneBackfillProfile::default())
            .unwrap_err();
        assert!(error.to_string().contains("primary markers retain"));
    }

    #[test]
    #[ignore = "release-only SoftTier-Q3 finalizer microbenchmark"]
    fn benchmark_hashprune_soft_tier_q3_finalizers() {
        const ROWS: usize = 10_000;
        let l_max = 64;
        let policy =
            named_rhalf_policy_with_lmax(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, l_max);
        let candidates = (0_u16..24)
            .flat_map(|hash| {
                [
                    (1_000 + u32::from(hash), hash, 1.0 + f32::from(hash) * 0.001),
                    (2_000 + u32::from(hash), hash, 1.1 + f32::from(hash) * 0.001),
                    (3_000 + u32::from(hash), hash, 1.2 + f32::from(hash) * 0.001),
                    (4_000 + u32::from(hash), hash, 1.3 + f32::from(hash) * 0.001),
                ]
            })
            .collect::<Vec<_>>();
        let build_rows = || {
            (0..ROWS)
                .map(|_| tiered_rhalf_reservoir(&candidates, l_max, policy).0)
                .collect::<Vec<_>>()
        };

        let mut checked_rows = build_rows();
        let mut checked_profile = HashPruneBackfillProfile::default();
        let checked_start = Instant::now();
        for (source, reservoir) in checked_rows.iter_mut().enumerate() {
            reservoir
                .finish_backfill_checked_for_test(source, policy, &mut checked_profile)
                .unwrap();
        }
        let checked_wall = checked_start.elapsed();
        let checked_checksum = checked_rows
            .iter()
            .flat_map(|reservoir| reservoir.slots.iter())
            .fold(0_u64, |sum, slot| sum.wrapping_add(u64::from(slot.target)));
        std::hint::black_box(checked_checksum);
        drop(checked_rows);

        let mut fast_rows = build_rows();
        let mut fast_profile = HashPruneBackfillProfile::default();
        let fast_start = Instant::now();
        for (source, reservoir) in fast_rows.iter_mut().enumerate() {
            reservoir
                .finish_backfill(source, policy, &mut fast_profile)
                .unwrap();
        }
        let fast_wall = fast_start.elapsed();
        let fast_checksum = fast_rows
            .iter()
            .flat_map(|reservoir| reservoir.slots.iter())
            .fold(0_u64, |sum, slot| sum.wrapping_add(u64::from(slot.target)));
        std::hint::black_box(fast_checksum);

        assert_eq!(fast_profile, checked_profile);
        assert_eq!(fast_checksum, checked_checksum);
        eprintln!(
            "soft-tier-q3 finalizer rows={ROWS} checked_ms={} fast_ms={}",
            checked_wall.as_millis(),
            fast_wall.as_millis()
        );
    }

    #[test]
    fn hashprune_hard_tier_q3_always_prefers_b1_over_a2() {
        assert_eq!(
            named_rhalf_cross_layer_backups(
                PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
                0.90f32.powi(2),
            ),
            vec![101, 201]
        );
    }

    #[test]
    fn hashprune_distance_first_q3_prefers_nearer_a2() {
        assert_eq!(
            named_rhalf_cross_layer_backups(
                PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
                0.95f32.powi(2),
            ),
            vec![101, 102]
        );
    }

    #[test]
    fn hashprune_distance_first_q3_breaks_equal_distance_by_hit_layer() {
        assert_eq!(
            named_rhalf_cross_layer_backups(PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3, 1.0,),
            vec![101, 201]
        );
    }

    #[test]
    fn hashprune_hash_slot_1_never_retains_a2() {
        assert_eq!(
            named_rhalf_cross_layer_backups(
                PiPNNHashPruneBackfillMode::RHalfHashSlot1,
                0.90f32.powi(2),
            ),
            vec![101, 201]
        );
    }

    #[test]
    fn hashprune_named_rhalf_modes_are_insertion_order_invariant() {
        let candidates = vec![
            (10, 10, 0.25),
            (101, 10, 0.30),
            (102, 10, 0.81),
            (103, 10, 0.90),
            (20, 20, 0.25),
            (201, 20, 1.00),
            (202, 20, 1.10),
            (30, 30, 0.20),
            (301, 30, 0.70),
        ];
        for mode in [
            PiPNNHashPruneBackfillMode::RHalfHashSlot1,
            PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
        ] {
            let policy = named_rhalf_policy(mode);
            let expected = finalized_tiered_rhalf_snapshot(&candidates, 4, policy);
            let mut rng = StdRng::seed_from_u64(20260719);
            for _ in 0..64 {
                let mut shuffled = candidates.clone();
                shuffled.shuffle(&mut rng);
                assert_eq!(
                    finalized_tiered_rhalf_snapshot(&shuffled, 4, policy),
                    expected,
                    "mode={mode}"
                );
            }
        }
    }

    #[test]
    fn hashprune_named_rhalf_streaming_top_c_matches_full_q3_pool_oracle() {
        let candidates = (0..12u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 0.40 + f32::from(hash) * 0.01),
                    (200 + u32::from(hash), hash, 0.46 + f32::from(hash) * 0.02),
                    (300 + u32::from(hash), hash, 0.52 + f32::from(hash) * 0.015),
                    (400 + u32::from(hash), hash, 0.58 + f32::from(hash) * 0.012),
                    (500 + u32::from(hash), hash, 0.70 + f32::from(hash) * 0.01),
                ]
            })
            .collect::<Vec<_>>();
        for mode in [
            PiPNNHashPruneBackfillMode::RHalfHashSlot1,
            PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
        ] {
            let policy = named_rhalf_policy(mode);
            let expected = reference_tiered_rhalf_slots(&candidates, 16, policy);
            let mut rng = StdRng::seed_from_u64(0x5143_4655_4c4c);
            for _ in 0..64 {
                let mut shuffled = candidates.clone();
                shuffled.shuffle(&mut rng);
                assert_eq!(
                    finalized_tiered_rhalf_snapshot(&shuffled, 16, policy),
                    expected,
                    "mode={mode}"
                );
            }
        }
    }

    #[test]
    fn hashprune_tiered_rhalf_shortlist_membership_prefers_first_hit_layer() {
        let candidates = [
            (100, 10, 1.0),
            (200, 10, 1.1),
            (300, 10, 1.2),
            (101, 20, 1.0),
            (201, 20, 1.3),
            (301, 20, 1.4),
            (102, 30, 1.0),
            (202, 30, 2.0),
        ];
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let (reservoir, _) = tiered_rhalf_reservoir(&candidates, 8, policy);
        assert_eq!(reservoir.backfill_count().unwrap(), 4);
        let primary_count = reservoir.slots.len() - reservoir.backfill_count().unwrap();
        let mut targets = reservoir.slots[primary_count..]
            .iter()
            .map(|slot| slot.target)
            .collect::<Vec<_>>();
        targets.sort_unstable();
        assert_eq!(targets, vec![200, 201, 202, 300]);
    }

    #[test]
    fn hashprune_tiered_rhalf_supports_r96_budget48_without_layout_growth() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let mut candidates = Vec::new();
        for hash in 0..48u16 {
            candidates.push((1_000 + u32::from(hash), hash, 1.0));
            candidates.push((2_000 + u32::from(hash), hash, 1.1));
        }
        let (reservoir, _) = tiered_rhalf_reservoir(&candidates, 96, policy);
        assert_eq!(reservoir.backfill_count().unwrap(), 48);
        assert_eq!(reservoir.slots.len(), 96);
        assert_eq!(std::mem::size_of::<HashPruneReservoir>(), 48);
    }

    #[test]
    fn hashprune_tiered_rhalf_veto_does_not_refill_shortlist_hole() {
        let candidates = [
            (100, 10, 2.0),
            (200, 10, 2.01),
            (300, 10, 2.02),
            (101, 20, 2.0),
            (201, 20, 2.03),
            (301, 20, 2.04),
            (102, 30, 2.0),
            (202, 30, 3.0),
        ];
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::None);
        let (reservoir, mut profile) = tiered_rhalf_reservoir(&candidates, 8, policy);
        assert_eq!(profile.shortlist_admissions, 4);
        assert_eq!(profile.shortlist_replacements, 1);
        assert_eq!(profile.shortlist_capacity_rejections, 0);
        let slots = reservoir
            .into_slots_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert!(slots.iter().all(|slot| slot.meta.backfill_hit_layer() == 0));
        let neighbors = slots.iter().map(|slot| slot.target).collect::<Vec<_>>();
        assert_eq!(neighbors, vec![100, 101, 102, 200, 201, 300]);
        assert!(!neighbors.contains(&301));
        assert_eq!(profile.gate_rejected, 1);
        assert_eq!(profile.rows_with_nonempty_shortlist, 1);
        assert_eq!(profile.shortlisted_by_layer, [3, 1, 0]);
        assert_eq!(profile.emitted_by_layer, [2, 1, 0]);
        assert_eq!(profile.rows_with_deeper_backfill, 1);
        assert_eq!(profile.gate_rejected_by_layer, [1, 0, 0]);
        assert_eq!(profile.rows_with_veto_holes, 1);
    }

    #[test]
    fn hashprune_tiered_rhalf_profile_counts_global_capacity_rejection() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let candidates = [
            (100, 10, 1.0),
            (101, 20, 1.0),
            (102, 30, 1.0),
            (200, 10, 1.1),
            (201, 20, 1.2),
            (202, 30, 1.3),
            (300, 10, 9.0),
        ];
        let (reservoir, mut profile) = tiered_rhalf_reservoir(&candidates, 6, policy);
        assert_eq!(profile.shortlist_admissions, 3);
        assert_eq!(profile.shortlist_replacements, 0);
        assert_eq!(profile.shortlist_capacity_rejections, 1);
        reservoir
            .into_slots_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(profile.shortlisted_by_layer, [3, 0, 0]);
        assert_eq!(profile.emitted_by_layer, [3, 0, 0]);
    }

    #[test]
    fn hashprune_tiered_rhalf_profile_counts_baseline_full_rows() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let candidates = [(100, 10, 1.0), (101, 20, 1.0), (102, 30, 1.0)];
        let (reservoir, mut profile) = tiered_rhalf_reservoir(&candidates, 3, policy);
        reservoir
            .into_slots_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(profile.baseline_full_rows, 1);
        assert_eq!(profile.rows_with_nonempty_shortlist, 0);
    }

    #[test]
    fn hashprune_tiered_rhalf_primary_markers_match_backup_groups_and_clear_on_finish() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let candidates = (0..6u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, 1.1),
                    (300 + u32::from(hash), hash, 1.2),
                ]
            })
            .collect::<Vec<_>>();
        let (reservoir, mut profile) = tiered_rhalf_reservoir(&candidates, 12, policy);
        let backup_count = reservoir.backfill_count().unwrap();
        let primary_count = reservoir.slots.len() - backup_count;
        for primary in &reservoir.slots[..primary_count] {
            let actual = reservoir.slots[primary_count..]
                .iter()
                .filter(|backup| backup.hash == primary.hash)
                .count();
            assert_eq!(primary.meta.backfill_count().unwrap(), actual);
        }

        let slots = reservoir
            .into_slots_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert!(slots.iter().all(|slot| {
            slot.meta.backfill_count().unwrap() == 0 && slot.meta.backfill_hit_layer() == 0
        }));
    }

    #[test]
    fn hashprune_tiered_rhalf_rejects_target_hash_mismatch_before_mutation() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let mut profile = HashPruneBackfillProfile::default();
        let mut reservoir = HashPruneReservoir::new(8);
        reservoir
            .insert_backfill_with_policy(7, 100, 10, 1.0, policy, &mut profile)
            .unwrap();
        reservoir
            .insert_backfill_with_policy(7, 200, 10, 1.1, policy, &mut profile)
            .unwrap();
        let before = backfill_snapshot(&reservoir);
        let error = reservoir
            .insert_backfill_with_policy(7, 200, 20, 0.9, policy, &mut profile)
            .unwrap_err();
        assert!(error.to_string().contains("source 7 destination 200"));
        assert_eq!(backfill_snapshot(&reservoir), before);
    }

    #[test]
    fn hashprune_tiered_rhalf_matches_reference_for_permutations() {
        let candidates = (0..6u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.4 + f32::from(hash) * 0.01),
                    (200 + u32::from(hash), hash, 1.1 + f32::from(hash) * 0.01),
                    (300 + u32::from(hash), hash, 1.2 + f32::from(hash) * 0.01),
                    (400 + u32::from(hash), hash, 1.0 + f32::from(hash) * 0.01),
                    (500 + u32::from(hash), hash, 1.3 + f32::from(hash) * 0.01),
                ]
            })
            .collect::<Vec<_>>();
        for ablation in [
            PiPNNHashPruneBackfillAblation::NoVeto,
            PiPNNHashPruneBackfillAblation::DistanceFirst,
            PiPNNHashPruneBackfillAblation::Quota1,
            PiPNNHashPruneBackfillAblation::Quota2,
        ] {
            let policy = tiered_rhalf_policy(ablation);
            let expected = reference_tiered_rhalf_slots(&candidates, 12, policy);
            let mut orders = vec![
                (0..candidates.len()).collect::<Vec<_>>(),
                (0..candidates.len()).rev().collect::<Vec<_>>(),
                (0..candidates.len())
                    .step_by(2)
                    .chain((1..candidates.len()).step_by(2))
                    .collect::<Vec<_>>(),
            ];
            let mut rng = StdRng::seed_from_u64(0x5248_414c_46);
            for _ in 0..12 {
                let mut order = (0..candidates.len()).collect::<Vec<_>>();
                order.shuffle(&mut rng);
                orders.push(order);
            }
            for order in orders {
                let permuted = order
                    .into_iter()
                    .map(|index| candidates[index])
                    .collect::<Vec<_>>();
                assert_eq!(
                    finalized_tiered_rhalf_snapshot(&permuted, 12, policy),
                    expected,
                    "ablation={ablation:?}"
                );
            }
        }
    }

    #[test]
    fn hashprune_tiered_rhalf_capacity_shrinks_with_new_primaries() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let mut candidates = (0..3u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, 1.1),
                    (300 + u32::from(hash), hash, 1.2),
                ]
            })
            .collect::<Vec<_>>();
        let (reservoir, _) = tiered_rhalf_reservoir(&candidates, 8, policy);
        assert_eq!(reservoir.backfill_count().unwrap(), 4);

        candidates.extend([(103, 40, 1.0), (104, 50, 1.0)]);
        let (reservoir, _) = tiered_rhalf_reservoir(&candidates, 8, policy);
        assert_eq!(reservoir.backfill_count().unwrap(), 3);
        assert_eq!(
            finalized_tiered_rhalf_snapshot(&candidates, 8, policy),
            reference_tiered_rhalf_slots(&candidates, 8, policy)
        );
    }

    #[test]
    fn hashprune_tiered_rhalf_odd_r_and_full_rows_match_reference() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        let candidates = (0..7u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, 1.1),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(
            finalized_tiered_rhalf_snapshot(&candidates, 7, policy),
            reference_tiered_rhalf_slots(&candidates, 7, policy)
        );
        let one = [(100, 10, 1.0), (101, 10, 0.9)];
        assert_eq!(
            finalized_tiered_rhalf_snapshot(&one, 1, policy),
            reference_tiered_rhalf_slots(&one, 1, policy)
        );
    }

    #[test]
    fn hashprune_backfill_final_gate_uses_squared_euclidean_ratio() {
        let candidates = [
            (100, 10, 1.0),
            (101, 10, 2.41),
            (200, 20, 2.0),
            (201, 20, 2.43),
        ];
        assert_eq!(
            finalized_neighbors(&candidates, 4, 8, 1.1),
            vec![100, 200, 101]
        );
    }

    #[test]
    fn hashprune_backfill_budget_four_emits_only_nearest_four() {
        let mut candidates = Vec::new();
        for hash in 0..6u16 {
            candidates.push((100 + u32::from(hash), hash, 1.0 + hash as f32 * 0.01));
            candidates.push((200 + u32::from(hash), hash, 1.1 + hash as f32 * 0.01));
        }
        let mut reservoir = HashPruneReservoir::new(12);
        for &(target, hash, dist) in &candidates {
            reservoir.insert_backfill(0, target, hash, dist).unwrap();
        }
        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(
                0,
                HashPruneBackfillPolicy::distance_gated(4, 2.0),
                &mut profile,
            )
            .unwrap();
        assert_eq!(neighbors.len(), 10);
        assert_eq!(&neighbors[6..], &[200, 201, 202, 203]);
        assert_eq!(profile.rows_seen, 1);
        assert_eq!(profile.primary_edges, 6);
        assert_eq!(profile.backfill_edges, 4);
        assert_eq!(profile.gate_rejected, 0);
        assert_eq!(profile.budget_rejected, 2);
        assert_eq!(profile.degree_before_sum, 6);
        assert_eq!(profile.degree_after_sum, 10);
    }

    #[test]
    fn hashprune_stratified_k4_emits_three_nearest_and_one_farthest_backup() {
        let policy = HashPruneBackfillPolicy::distance_gated_stratified(4, 2.0);
        let mut reservoir = HashPruneReservoir::new(16);
        for hash in 0..8u16 {
            reservoir.insert_backfill_bounded(
                100 + u32::from(hash),
                hash,
                1.0,
                policy.candidate_budget(),
            );
            reservoir.insert_backfill_bounded(
                200 + u32::from(hash),
                hash,
                1.1 + hash as f32 * 0.1,
                policy.candidate_budget(),
            );
        }

        assert_eq!(reservoir.backfill_count().unwrap(), 8);
        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(0, policy, &mut profile)
            .unwrap();

        assert_eq!(&neighbors[..8], &[100, 101, 102, 103, 104, 105, 106, 107]);
        assert_eq!(&neighbors[8..], &[200, 201, 202, 207]);
        assert_eq!(profile.backfill_edges, 4);
        assert_eq!(profile.budget_rejected, 4);
    }

    #[test]
    fn hashprune_stratified_applies_gate_before_two_ended_selection() {
        let policy = HashPruneBackfillPolicy::distance_gated_stratified(4, 1.1);
        let mut reservoir = HashPruneReservoir::new(16);
        let backup_distances = [1.01, 1.02, 1.10, 1.15, 1.18, 1.20, 1.22, 1.30];
        for (hash, backup_dist) in backup_distances.into_iter().enumerate() {
            let hash = hash as u16;
            reservoir.insert_backfill_bounded(
                100 + u32::from(hash),
                hash,
                1.0,
                policy.candidate_budget(),
            );
            reservoir.insert_backfill_bounded(
                200 + u32::from(hash),
                hash,
                backup_dist,
                policy.candidate_budget(),
            );
        }

        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(0, policy, &mut profile)
            .unwrap();

        assert_eq!(&neighbors[8..], &[200, 201, 202, 205]);
        assert_eq!(profile.gate_rejected, 2);
        assert_eq!(profile.budget_rejected, 2);
    }

    #[test]
    fn hashprune_stratified_selection_is_permutation_independent() {
        let candidates = (0..8u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, 1.1 + hash as f32 * 0.1),
                ]
            })
            .collect::<Vec<_>>();
        let policy = HashPruneBackfillPolicy::distance_gated_stratified(4, 2.0);
        let run = |order: &[usize]| {
            let mut reservoir = HashPruneReservoir::new(16);
            for &index in order {
                let (target, hash, dist) = candidates[index];
                reservoir.insert_backfill_bounded(target, hash, dist, policy.candidate_budget());
            }
            reservoir
                .into_neighbors_with_backfill(0, policy, &mut HashPruneBackfillProfile::default())
                .unwrap()
        };

        let forward = (0..candidates.len()).collect::<Vec<_>>();
        let reverse = (0..candidates.len()).rev().collect::<Vec<_>>();
        let interleaved = [15, 0, 9, 2, 13, 4, 11, 6, 1, 14, 3, 12, 5, 10, 7, 8];
        let expected = run(&forward);
        assert_eq!(run(&reverse), expected);
        assert_eq!(run(&interleaved), expected);
    }

    #[test]
    fn hashprune_multihit_keeps_two_best_backups_per_hash() {
        let policy = HashPruneBackfillPolicy::distance_gated_multihit(6, 2.0);
        let mut reservoir = HashPruneReservoir::new(8);
        for (target, dist) in [(100, 1.0), (101, 1.2), (102, 1.3), (103, 1.1)] {
            reservoir.insert_backfill_bounded_with_hash_quota(
                target,
                10,
                dist,
                policy.candidate_budget(),
                policy.backup_hash_quota(),
            );
        }

        assert_eq!(reservoir.backfill_count().unwrap(), 2);
        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(neighbors, vec![100, 103, 101]);
        assert_eq!(profile.backfill_edges, 2);
    }

    #[test]
    fn hashprune_multihit_deduplicates_target_across_hashes() {
        let policy = HashPruneBackfillPolicy::distance_gated_multihit(6, 2.0);
        let mut reservoir = HashPruneReservoir::new(8);
        for (target, hash, dist) in [
            (100, 10, 1.0),
            (300, 10, 2.0),
            (200, 20, 1.0),
            (300, 20, 1.5),
        ] {
            reservoir.insert_backfill_bounded_with_hash_quota(
                target,
                hash,
                dist,
                policy.candidate_budget(),
                policy.backup_hash_quota(),
            );
        }

        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(neighbors, vec![100, 200, 300]);
        assert_eq!(profile.backfill_edges, 1);
    }

    #[test]
    fn hashprune_multihit_drops_backup_that_duplicates_primary_target() {
        let policy = HashPruneBackfillPolicy::distance_gated_multihit(6, 2.0);
        let mut reservoir = HashPruneReservoir::new(8);
        for (target, hash, dist) in [(100, 10, 1.0), (200, 20, 1.0), (100, 20, 2.0)] {
            reservoir.insert_backfill_bounded_with_hash_quota(
                target,
                hash,
                dist,
                policy.candidate_budget(),
                policy.backup_hash_quota(),
            );
        }

        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(neighbors, vec![100, 200]);
        assert_eq!(profile.backfill_edges, 0);
    }

    #[test]
    fn hashprune_multihit_pool_ten_emits_only_nearest_k6() {
        let policy = HashPruneBackfillPolicy::distance_gated_multihit(6, 2.0);
        let candidates = (0..8u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, 1.1 + hash as f32 * 0.01),
                    (300 + u32::from(hash), hash, 1.2 + hash as f32 * 0.01),
                ]
            })
            .collect::<Vec<_>>();
        let run = |order: &[usize]| {
            let mut reservoir = HashPruneReservoir::new(24);
            for &index in order {
                let (target, hash, dist) = candidates[index];
                reservoir.insert_backfill_bounded_with_hash_quota(
                    target,
                    hash,
                    dist,
                    policy.candidate_budget(),
                    policy.backup_hash_quota(),
                );
            }
            assert_eq!(reservoir.backfill_count().unwrap(), 10);
            let mut profile = HashPruneBackfillProfile::default();
            let neighbors = reservoir
                .into_neighbors_with_backfill(0, policy, &mut profile)
                .unwrap();
            (neighbors, profile)
        };

        let forward = (0..candidates.len()).collect::<Vec<_>>();
        let reverse = (0..candidates.len()).rev().collect::<Vec<_>>();
        let (neighbors, profile) = run(&forward);
        assert_eq!(&neighbors[8..], &[200, 201, 202, 203, 204, 205]);
        assert_eq!(profile.backfill_edges, 6);
        assert_eq!(profile.budget_rejected, 4);
        assert_eq!(run(&reverse).0, neighbors);
    }

    #[test]
    fn hashprune_tiered_d52_prefers_first_hit_layers() {
        let first_distances = [1.01, 1.04, 1.06, 1.08, 1.50];
        let second_distances = [1.02, 1.05, 1.07, 1.09, 1.51];
        let candidates = (0..5u16)
            .flat_map(|hash| {
                let index = usize::from(hash);
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, first_distances[index]),
                    (300 + u32::from(hash), hash, second_distances[index]),
                ]
            })
            .collect::<Vec<_>>();

        let neighbors = finalized_tiered_d52_neighbors(&candidates, 16);
        assert_eq!(&neighbors[..5], &[100, 101, 102, 103, 104]);
        assert_eq!(&neighbors[5..], &[200, 201, 202, 203, 204, 300, 301, 302]);
    }

    #[test]
    fn hashprune_tiered_d52_stops_at_degree_52() {
        let policy = HashPruneBackfillPolicy::distance_gated_tiered_d52(8, 2.0);
        let mut reservoir = HashPruneReservoir::new(64);
        for hash in 0..50u16 {
            reservoir.insert_backfill_bounded_with_hash_quota(
                1_000 + u32::from(hash),
                hash,
                1.0,
                policy.candidate_budget(),
                policy.backup_hash_quota(),
            );
        }
        for hash in 0..10u16 {
            reservoir.insert_backfill_bounded_with_hash_quota(
                2_000 + u32::from(hash),
                hash,
                1.01 + f32::from(hash) * 0.01,
                policy.candidate_budget(),
                policy.backup_hash_quota(),
            );
        }

        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(0, policy, &mut profile)
            .unwrap();
        assert_eq!(neighbors.len(), 52);
        assert_eq!(&neighbors[50..], &[2_000, 2_001]);
        assert_eq!(profile.primary_edges, 50);
        assert_eq!(profile.backfill_edges, 2);
        assert_eq!(profile.budget_rejected, 8);
        assert_eq!(profile.degree_after_sum, 52);
    }

    #[test]
    fn hashprune_tiered_d52_selection_is_permutation_independent() {
        let candidates = (0..5u16)
            .flat_map(|hash| {
                [
                    (100 + u32::from(hash), hash, 1.0),
                    (200 + u32::from(hash), hash, 1.1 + f32::from(hash) * 0.02),
                    (300 + u32::from(hash), hash, 1.2 + f32::from(hash) * 0.02),
                ]
            })
            .collect::<Vec<_>>();
        let forward = (0..candidates.len()).collect::<Vec<_>>();
        let reverse = (0..candidates.len()).rev().collect::<Vec<_>>();
        let interleaved = (0..candidates.len())
            .step_by(2)
            .chain((1..candidates.len()).step_by(2))
            .collect::<Vec<_>>();
        let run = |order: &[usize]| {
            let permuted = order
                .iter()
                .map(|&index| candidates[index])
                .collect::<Vec<_>>();
            finalized_tiered_d52_neighbors(&permuted, 16)
        };

        let expected = run(&forward);
        assert_eq!(run(&reverse), expected);
        assert_eq!(run(&interleaved), expected);
    }

    #[test]
    fn hashprune_backfill_budget_four_bounds_online_suffix_to_four() {
        let mut reservoir = HashPruneReservoir::new(16);
        for hash in 0..10u16 {
            reservoir.insert_backfill_bounded(
                100 + u32::from(hash),
                hash,
                1.0 + hash as f32 * 0.01,
                4,
            );
            reservoir.insert_backfill_bounded(
                200 + u32::from(hash),
                hash,
                1.1 + hash as f32 * 0.01,
                4,
            );
        }

        assert_eq!(reservoir.backfill_count().unwrap(), 4);
        let mut profile = HashPruneBackfillProfile::default();
        reservoir
            .finish_backfill(
                0,
                HashPruneBackfillPolicy::distance_gated(4, 2.0),
                &mut profile,
            )
            .unwrap();
        assert_eq!(profile.backfill_edges, 4);
        assert_eq!(profile.budget_rejected, 0);
    }

    #[test]
    fn hashprune_backfill_full_row_matches_baseline_exactly() {
        let candidates = [
            (10, 1, 4.0),
            (20, 2, 3.0),
            (30, 3, 2.0),
            (40, 4, 1.0),
            (11, 1, 0.5),
            (50, 5, 0.25),
        ];
        let mut baseline = HashPruneReservoir::new(4);
        for &(target, hash, dist) in &candidates {
            baseline.insert(target, hash, dist);
        }
        assert_eq!(
            finalized_neighbors(&candidates, 4, 8, 1.2),
            baseline.into_neighbors()
        );
    }

    #[test]
    fn hashprune_backfill_alpha_boundary_is_inclusive() {
        let boundary = 1.0f32;
        let just_above = f32::from_bits(boundary.to_bits() + 1);
        let candidates = [
            (100, 10, 0.5),
            (101, 10, boundary),
            (200, 20, 1.0),
            (201, 20, just_above),
        ];
        assert_eq!(
            finalized_neighbors(&candidates, 4, 8, 1.0),
            vec![100, 200, 101]
        );
    }

    #[test]
    fn hashprune_backfill_off_extraction_is_exact_and_unprofiled() {
        let candidates = [(10, 1, 4.0), (20, 2, 3.0), (11, 1, 0.5), (30, 3, 2.0)];
        let mut expected = HashPruneReservoir::new(4);
        let mut actual = HashPruneReservoir::new(4);
        for (target, hash, dist) in candidates {
            expected.insert(target, hash, dist);
            actual.insert(target, hash, dist);
        }
        let expected = expected
            .into_slots()
            .into_iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect::<Vec<_>>();
        let mut profile = HashPruneBackfillProfile::default();
        let actual = actual
            .into_slots_with_backfill(0, HashPruneBackfillPolicy::off(), &mut profile)
            .unwrap()
            .into_iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(profile, HashPruneBackfillProfile::default());
    }

    #[test]
    fn hashprune_backfill_policy_mismatch_is_an_error() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert_backfill(9, 100, 10, 1.0).unwrap();
        reservoir.insert_backfill(9, 101, 10, 1.1).unwrap();
        let mut profile = HashPruneBackfillProfile::default();
        let error = reservoir
            .finish_backfill(9, HashPruneBackfillPolicy::off(), &mut profile)
            .unwrap_err();
        assert!(error.to_string().contains("policy mismatch"));
    }

    fn reference_backfill_neighbors(
        candidates: &[(u32, u16, f32)],
        l_max: usize,
        budget: usize,
        alpha: f32,
    ) -> (usize, Vec<u32>) {
        let mut by_target = BTreeMap::<u32, (u16, f32)>::new();
        for &(target, hash, dist) in candidates {
            match by_target.get_mut(&target) {
                Some((known_hash, known_dist)) => {
                    assert_eq!(
                        *known_hash, hash,
                        "test corpus violates target/hash invariant"
                    );
                    if dist.total_cmp(known_dist).is_lt() {
                        *known_dist = dist;
                    }
                }
                None => {
                    by_target.insert(target, (hash, dist));
                }
            }
        }

        let mut groups = BTreeMap::<u16, Vec<Slot>>::new();
        for (target, (hash, dist)) in by_target {
            groups
                .entry(hash)
                .or_default()
                .push(Slot::new(target, hash, dist));
        }
        for group in groups.values_mut() {
            group.sort_unstable_by(|left, right| slot_order(*left, *right));
        }

        let mut primaries = groups.values().map(|group| group[0]).collect::<Vec<_>>();
        primaries.sort_unstable_by(|left, right| slot_order(*left, *right));
        primaries.truncate(l_max);
        primaries.sort_unstable_by_key(|slot| slot.hash);
        let primary_count = primaries.len();

        if primary_count == 0 || primary_count == l_max {
            return (
                primary_count,
                primaries.into_iter().map(|slot| slot.target).collect(),
            );
        }

        let retained_hashes = primaries
            .iter()
            .map(|slot| slot.hash)
            .collect::<BTreeSet<_>>();
        let dmax2 = primaries
            .iter()
            .map(|slot| slot.dist)
            .max_by(f32::total_cmp)
            .unwrap();
        let alpha = f64::from(alpha);
        let threshold = alpha * alpha * f64::from(dmax2);
        let mut backups = groups
            .iter()
            .filter(|(hash, group)| retained_hashes.contains(hash) && group.len() >= 2)
            .map(|(_, group)| group[1])
            .filter(|slot| f64::from(slot.dist) <= threshold)
            .collect::<Vec<_>>();
        backups.sort_unstable_by(|left, right| slot_order(*left, *right));
        backups.truncate(budget.min(l_max - primary_count));

        let mut neighbors = primaries
            .into_iter()
            .map(|slot| slot.target)
            .collect::<Vec<_>>();
        neighbors.extend(backups.into_iter().map(|slot| slot.target));
        (primary_count, neighbors)
    }

    #[test]
    fn hashprune_backfill_matches_reference_for_fixed_permutations() {
        let fixture = [
            (100, 10, 1.0),
            (101, 10, 1.3),
            (102, 10, 1.8),
            (200, 20, 2.0),
            (201, 20, 2.1),
            (300, 30, 1.5),
            (301, 30, 1.7),
            (400, 40, 3.0),
        ];
        let orders = [
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![7, 6, 5, 4, 3, 2, 1, 0],
            vec![3, 0, 6, 1, 7, 2, 4, 5],
            vec![5, 2, 7, 1, 4, 6, 0, 3],
        ];
        let expected = reference_backfill_neighbors(&fixture, 7, 3, 1.2).1;
        for order in orders {
            let permuted = order
                .into_iter()
                .map(|index| fixture[index])
                .collect::<Vec<_>>();
            assert_eq!(finalized_neighbors(&permuted, 7, 3, 1.2), expected);
        }
    }

    #[test]
    fn hashprune_backfill_matches_reference_for_seeded_random_cases() {
        let mut rng = StdRng::seed_from_u64(0x5eed_bacf_1111);
        for case in 0..200 {
            let l_max = rng.random_range(1usize..=16);
            let candidate_count = rng.random_range(0usize..=48);
            let mut candidates = Vec::<(u32, u16, f32)>::with_capacity(candidate_count);
            let mut next_target = 1u32;
            for _ in 0..candidate_count {
                let dist = rng.random_range(1u32..=20_000) as f32 / 1_000.0;
                if !candidates.is_empty() && rng.random_bool(0.2) {
                    let (target, hash, _) = candidates[rng.random_range(0..candidates.len())];
                    candidates.push((target, hash, dist));
                } else {
                    let hash = rng.random_range(0u16..32);
                    candidates.push((next_target, hash, dist));
                    next_target += 1;
                }
            }

            for budget in [1, 4, 8] {
                let mut alpha_suffixes = Vec::new();
                for alpha in [1.0, 1.1, 1.2] {
                    let (primary_count, expected) =
                        reference_backfill_neighbors(&candidates, l_max, budget, alpha);
                    for order in 0..3 {
                        let mut permuted = candidates.clone();
                        permuted.shuffle(&mut rng);
                        let actual = finalized_neighbors(&permuted, l_max, budget, alpha);
                        assert_eq!(
                            actual, expected,
                            "case={case} l_max={l_max} budget={budget} alpha={alpha} order={order}"
                        );
                        assert!(actual.len() <= l_max);
                        assert!(actual.len().saturating_sub(primary_count) <= budget);
                    }
                    alpha_suffixes.push(
                        expected[primary_count..]
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>(),
                    );
                }
                assert!(alpha_suffixes[0].is_subset(&alpha_suffixes[1]));
                assert!(alpha_suffixes[1].is_subset(&alpha_suffixes[2]));
            }
        }
    }

    #[test]
    fn hashprune_backfill_transition_to_full_row_discards_suffix() {
        let underfull = [(10, 1, 1.0), (11, 1, 1.1), (20, 2, 2.0)];
        let full = [(10, 1, 1.0), (11, 1, 1.1), (20, 2, 2.0), (30, 3, 3.0)];
        assert_eq!(
            finalized_neighbors(&underfull, 3, 8, 2.0),
            reference_backfill_neighbors(&underfull, 3, 8, 2.0).1
        );
        assert_eq!(
            finalized_neighbors(&full, 3, 8, 2.0),
            reference_backfill_neighbors(&full, 3, 8, 2.0).1
        );
    }

    #[test]
    fn hashprune_backfill_equal_distance_uses_target_tie_break() {
        let candidates = [
            (110, 10, 1.0),
            (100, 10, 1.0),
            (210, 20, 2.0),
            (200, 20, 2.0),
        ];
        assert_eq!(
            finalized_neighbors(&candidates, 4, 8, 1.0),
            reference_backfill_neighbors(&candidates, 4, 8, 1.0).1
        );
    }

    #[test]
    fn hashprune_backfill_preserves_fixed_layouts() {
        assert_eq!(std::mem::size_of::<Slot>(), 12);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<HashPruneReservoir>(), 48);

        let reservoir = HashPruneReservoir::new(64);
        assert_eq!(reservoir.slots.capacity(), 64);
    }

    #[test]
    fn reusable_neighbor_extraction_preserves_slot_capacity() {
        let mut reservoir = HashPruneReservoir::new(64);
        for point in 0..32_u32 {
            reservoir.insert(point, point as u16, point as f32);
        }
        let capacity = reservoir.slots.capacity();

        let neighbors = reservoir.copy_neighbors_and_clear();

        assert_eq!(neighbors.len(), 32);
        assert!(reservoir.slots.is_empty());
        assert_eq!(reservoir.slots.capacity(), capacity);
    }

    #[test]
    fn reusable_backfill_extraction_preserves_slot_capacity() {
        let mut reservoir = HashPruneReservoir::new(64);
        let policy = HashPruneBackfillPolicy::diagnostic_soft_tier_q3(1.02).unwrap();
        let mut profile = HashPruneBackfillProfile::default();
        for (target, hash, dist) in [
            (100, 10, 1.0),
            (101, 10, 1.1),
            (102, 10, 1.2),
            (200, 20, 2.0),
            (201, 20, 2.1),
        ] {
            reservoir
                .insert_backfill_with_policy(
                    target as usize,
                    target,
                    hash,
                    dist,
                    policy,
                    &mut profile,
                )
                .unwrap();
        }
        let capacity = reservoir.slots.capacity();

        let neighbors = reservoir
            .copy_neighbors_with_backfill_and_clear(0, policy, &mut profile)
            .unwrap();

        assert!(!neighbors.is_empty());
        assert!(reservoir.slots.is_empty());
        assert_eq!(reservoir.slots.capacity(), capacity);
    }

    #[test]
    fn hashprune_backfill_online_count_does_not_use_slot_metadata() {
        let mut reservoir = HashPruneReservoir::new(8);
        reservoir.insert_backfill_bounded(100, 10, 1.0, 4);
        reservoir.insert_backfill_bounded(101, 10, 1.1, 4);

        assert_eq!(reservoir.backfill_count().unwrap(), 1);
        assert!(
            reservoir
                .slots()
                .iter()
                .all(|slot| slot.meta == HashPruneMeta::empty())
        );
    }

    #[test]
    fn hashprune_backfill_unsorted_suffix_finalizes_in_strict_order() {
        let mut reservoir = HashPruneReservoir::new(8);
        for (target, hash, dist) in [
            (100, 10, 0.5),
            (200, 20, 0.6),
            (300, 30, 0.7),
            (101, 10, 3.0),
            (201, 20, 1.1),
            (301, 30, 2.0),
        ] {
            reservoir.insert_backfill_bounded(target, hash, dist, 3);
        }
        let primary_count = reservoir.slots().len() - reservoir.backfill_count().unwrap();
        assert_eq!(
            reservoir.slots()[primary_count..]
                .iter()
                .map(|slot| slot.target)
                .collect::<Vec<_>>(),
            vec![101, 201, 301]
        );

        reservoir.insert_backfill_bounded(400, 40, 0.8, 3);
        reservoir.insert_backfill_bounded(401, 40, 1.5, 3);
        let primary_count = reservoir.slots().len() - reservoir.backfill_count().unwrap();
        assert_eq!(
            reservoir.slots()[primary_count..]
                .iter()
                .map(|slot| slot.target)
                .collect::<Vec<_>>(),
            vec![401, 201, 301]
        );

        let mut profile = HashPruneBackfillProfile::default();
        let neighbors = reservoir
            .into_neighbors_with_backfill(
                0,
                HashPruneBackfillPolicy::distance_gated(3, 10.0),
                &mut profile,
            )
            .unwrap();
        assert_eq!(neighbors, vec![100, 200, 300, 400, 201, 401, 301]);
    }

    #[test]
    #[should_panic(expected = "policy-aware extraction")]
    fn hashprune_backfill_legacy_extraction_rejects_unfinalized_suffix() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert_backfill(0, 100, 10, 1.0).unwrap();
        reservoir.insert_backfill(0, 101, 10, 1.1).unwrap();
        let _ = reservoir.into_neighbors();
    }

    #[test]
    fn hashprune_backfill_keeps_primary_prefix_and_one_nearest_backup_per_hash() {
        let mut reservoir = HashPruneReservoir::new(4);
        for (target, hash, dist) in [
            (100, 10, 1.0),
            (102, 10, 1.8),
            (101, 10, 1.5),
            (200, 20, 2.0),
            (202, 20, 2.4),
            (201, 20, 2.2),
        ] {
            reservoir.insert_backfill(0, target, hash, dist).unwrap();
        }

        assert_eq!(
            backfill_snapshot(&reservoir),
            (
                2,
                vec![
                    (100, 10, 1.0f32.to_bits()),
                    (200, 20, 2.0f32.to_bits()),
                    (101, 10, 1.5f32.to_bits()),
                    (201, 20, 2.2f32.to_bits()),
                ],
            )
        );
    }

    #[test]
    fn hashprune_backfill_new_hash_evicts_backup_before_primary() {
        let mut reservoir = HashPruneReservoir::new(3);
        for (target, hash, dist) in [
            (100, 10, 1.0),
            (101, 10, 1.1),
            (200, 20, 2.0),
            (201, 20, 2.1),
            (300, 30, 9.0),
        ] {
            reservoir.insert_backfill(7, target, hash, dist).unwrap();
        }
        assert_eq!(reservoir.backfill_count().unwrap(), 0);
        assert_eq!(
            reservoir
                .slots()
                .iter()
                .map(|slot| (slot.target, slot.hash))
                .collect::<Vec<_>>(),
            vec![(100, 10), (200, 20), (300, 30)]
        );
    }

    #[test]
    fn hashprune_backfill_primary_improvement_displaces_old_primary_to_backup() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert_backfill(0, 100, 10, 3.0).unwrap();
        reservoir.insert_backfill(0, 101, 10, 4.0).unwrap();
        reservoir.insert_backfill(0, 102, 10, 2.0).unwrap();
        assert_eq!(
            reservoir
                .slots()
                .iter()
                .map(|slot| slot.target)
                .collect::<Vec<_>>(),
            vec![102, 100]
        );
        assert_eq!(reservoir.backfill_count().unwrap(), 1);
    }

    #[test]
    fn hashprune_backfill_rejected_collision_preserves_full_suffix() {
        let mut reservoir = HashPruneReservoir::new(4);
        for (target, hash, dist) in [
            (100, 10, 1.0),
            (200, 20, 2.0),
            (101, 10, 1.5),
            (201, 20, 2.5),
        ] {
            reservoir.insert_backfill(7, target, hash, dist).unwrap();
        }
        assert_eq!(reservoir.backfill_count().unwrap(), 2);
        let before = backfill_snapshot(&reservoir);

        reservoir.insert_backfill(7, 102, 10, 9.0).unwrap();

        assert_eq!(backfill_snapshot(&reservoir), before);
        reservoir
            .finish_backfill(
                7,
                HashPruneBackfillPolicy::distance_gated(2, 2.0),
                &mut HashPruneBackfillProfile::default(),
            )
            .unwrap();
    }

    fn assert_insert_profiles_equal(
        actual: &HashPruneInsertProfile,
        expected: &HashPruneInsertProfile,
    ) {
        assert_eq!(actual.candidates, expected.candidates);
        assert_eq!(actual.hash_hits, expected.hash_hits);
        assert_eq!(actual.hash_replacements, expected.hash_replacements);
        assert_eq!(actual.hash_rejects, expected.hash_rejects);
        assert_eq!(actual.not_full_inserts, expected.not_full_inserts);
        assert_eq!(
            actual.full_replacement_checks,
            expected.full_replacement_checks
        );
        assert_eq!(actual.full_replacements, expected.full_replacements);
        assert_eq!(actual.full_rejects, expected.full_rejects);
        assert_eq!(actual.farthest_recomputes, expected.farthest_recomputes);
        assert_eq!(actual.rotate_lefts, expected.rotate_lefts);
        assert_eq!(actual.rotate_rights, expected.rotate_rights);
        assert_eq!(
            actual.evidence.annotation_wall,
            expected.evidence.annotation_wall
        );
        assert_eq!(
            actual.evidence.coalesce_sort_wall,
            expected.evidence.coalesce_sort_wall
        );
        assert_eq!(
            actual.evidence.coalesce_merge_wall,
            expected.evidence.coalesce_merge_wall
        );
        assert_eq!(
            actual.evidence.coalesce_wall,
            expected.evidence.coalesce_wall
        );
        assert_eq!(
            actual.evidence.reciprocal_checks,
            expected.evidence.reciprocal_checks
        );
        assert_eq!(
            actual.evidence.reciprocal_hits,
            expected.evidence.reciprocal_hits
        );
        assert_eq!(
            actual.evidence.support_once_candidates,
            expected.evidence.support_once_candidates
        );
        assert_eq!(
            actual.evidence.support_twice_candidates,
            expected.evidence.support_twice_candidates
        );
        assert_eq!(actual.evidence.class_counts, expected.evidence.class_counts);
        assert_eq!(
            actual.evidence.candidates_before_coalesce,
            expected.evidence.candidates_before_coalesce
        );
        assert_eq!(
            actual.evidence.candidates_after_coalesce,
            expected.evidence.candidates_after_coalesce
        );
        assert_eq!(
            actual.evidence.duplicate_records_removed,
            expected.evidence.duplicate_records_removed
        );
        assert_eq!(
            actual.evidence.unannotated_candidates,
            expected.evidence.unannotated_candidates
        );
        assert_eq!(
            actual.evidence.qualified_collision_losers,
            expected.evidence.qualified_collision_losers
        );
        assert_eq!(
            actual.evidence.overflow_inserts,
            expected.evidence.overflow_inserts
        );
        assert_eq!(
            actual.evidence.overflow_replacements,
            expected.evidence.overflow_replacements
        );
        assert_eq!(
            actual.evidence.overflow_rejects,
            expected.evidence.overflow_rejects
        );
        assert_eq!(
            actual.evidence.overflow_evictions_by_new_hash,
            expected.evidence.overflow_evictions_by_new_hash
        );
        assert_eq!(
            actual.evidence.rows_with_overflow,
            expected.evidence.rows_with_overflow
        );
        assert_eq!(
            actual.evidence.spill_records_avoided,
            expected.evidence.spill_records_avoided
        );
        assert_eq!(
            actual.evidence.spill_bytes_avoided,
            expected.evidence.spill_bytes_avoided
        );
    }

    #[derive(Debug)]
    struct InsertBenchmarkResult {
        selection_checksum: u64,
        profile: HashPruneInsertProfile,
    }

    fn run_insert_microbenchmark_workload(
        reservoir: &mut HashPruneReservoir,
        records: &[(u32, u16, f32, HashPruneMeta)],
        mut insert: impl FnMut(
            &mut HashPruneReservoir,
            &mut HashPruneInsertProfile,
            usize,
            u32,
            u16,
            f32,
            HashPruneMeta,
        ),
    ) -> InsertBenchmarkResult {
        const RECORDS_PER_ROW: usize = 256;
        let mut profile = HashPruneInsertProfile::default();
        let mut checksum = 0_u64;

        for (source, row) in records.chunks_exact(RECORDS_PER_ROW).enumerate() {
            reservoir.clear();
            for &(target, hash, dist, meta) in row {
                insert(reservoir, &mut profile, source, target, hash, dist, meta);
            }
            for slot in reservoir.slots() {
                let evidence = slot.meta.evidence().map_or(7_u64, |class| class as u64);
                checksum = checksum
                    .wrapping_mul(0x9e37_79b1)
                    .wrapping_add(slot.target as u64)
                    .wrapping_add((slot.hash as u64) << 32)
                    .wrapping_add(slot.dist.to_bits() as u64)
                    .wrapping_add(evidence);
            }
        }
        InsertBenchmarkResult {
            selection_checksum: checksum,
            profile,
        }
    }

    fn best_ns_per_record(
        mut run: impl FnMut() -> InsertBenchmarkResult,
        record_count: usize,
    ) -> f64 {
        black_box(run());
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = Instant::now();
            let result = run();
            let elapsed = start.elapsed();
            black_box(result);
            best = best.min(elapsed.as_secs_f64() * 1e9 / record_count as f64);
        }
        best
    }

    type InsertBenchmarkRecord = (u32, u16, f32, HashPruneMeta);

    const PRODUCTION_SHAPE_RESERVOIRS: usize = 262_144;
    const PRODUCTION_SHAPE_L_MAX: usize = 64;
    const PRODUCTION_SHAPE_MIN_CANDIDATES: usize = 60;
    const PRODUCTION_SHAPE_MAX_CANDIDATES: usize = 96;
    const PRODUCTION_SHAPE_COLLISION_HASHES: usize = 48;
    const PRODUCTION_SHAPE_ACCESS_STRIDE: usize = 65_537;

    struct ProductionShapeWorkload {
        records: Vec<InsertBenchmarkRecord>,
        row_offsets: Vec<usize>,
    }

    #[derive(Debug)]
    struct ProductionShapeBenchmarkResult {
        insert: InsertBenchmarkResult,
        metadata_checksum: u64,
    }

    struct ProductionShapeBenchmarkMeasurement {
        best_ns_per_candidate: f64,
        result: ProductionShapeBenchmarkResult,
    }

    #[derive(Debug)]
    struct ProductionShapeBackfillBenchmarkResult {
        finalization_ns_per_row: f64,
        profile: HashPruneBackfillProfile,
        primary_checksum: u64,
        selection_checksum: u64,
    }

    struct ProductionShapeBackfillBenchmarkMeasurement {
        best_ns_per_candidate: f64,
        result: ProductionShapeBackfillBenchmarkResult,
    }

    fn production_shape_hash(source: usize, lane: usize) -> u16 {
        (source
            .wrapping_mul(131)
            .wrapping_add(lane.wrapping_mul(997))) as u16
    }

    fn build_production_shape_workload() -> ProductionShapeWorkload {
        let mut records = Vec::with_capacity(
            PRODUCTION_SHAPE_RESERVOIRS
                * (PRODUCTION_SHAPE_MIN_CANDIDATES + PRODUCTION_SHAPE_MAX_CANDIDATES)
                / 2,
        );
        let mut row_offsets = Vec::with_capacity(PRODUCTION_SHAPE_RESERVOIRS + 1);
        row_offsets.push(0);

        for source in 0..PRODUCTION_SHAPE_RESERVOIRS {
            let candidate_count = PRODUCTION_SHAPE_MIN_CANDIDATES
                + source % (PRODUCTION_SHAPE_MAX_CANDIDATES - PRODUCTION_SHAPE_MIN_CANDIDATES + 1);
            for lane in 0..candidate_count {
                let underfull_duplicate =
                    candidate_count <= PRODUCTION_SHAPE_L_MAX && lane + 1 == candidate_count;
                let full_duplicate =
                    candidate_count > PRODUCTION_SHAPE_L_MAX && lane == PRODUCTION_SHAPE_L_MAX;
                let hash = if candidate_count > PRODUCTION_SHAPE_L_MAX {
                    production_shape_hash(source, lane % PRODUCTION_SHAPE_COLLISION_HASHES)
                } else if underfull_duplicate || full_duplicate {
                    production_shape_hash(source, 7)
                } else {
                    production_shape_hash(source, lane)
                };
                let dist = if underfull_duplicate {
                    20_000.0 + (source % 257) as f32
                } else if full_duplicate {
                    if source % 4 == 0 {
                        0.01 + (source % 31) as f32 / 1_000.0
                    } else {
                        20_000.0 + (source % 257) as f32
                    }
                } else if lane < PRODUCTION_SHAPE_L_MAX {
                    100.0 + ((source * 17 + lane * 37) % 997) as f32
                } else if lane % 2 == 1 {
                    0.1 + ((source + lane) % 31) as f32 / 100.0
                } else {
                    20_000.0 + ((source + lane) % 257) as f32
                };
                let target = (source * 128 + lane) as u32;
                let class = HashPruneEvidenceClass::ALL[(source + lane) % 6];
                records.push((target, hash, dist, HashPruneMeta::with_evidence(class)));
            }
            row_offsets.push(records.len());
        }

        ProductionShapeWorkload {
            records,
            row_offsets,
        }
    }

    #[inline]
    fn fold_insert_benchmark_checksum(checksum: u64, value: u64) -> u64 {
        checksum
            .rotate_left(13)
            .wrapping_mul(0x9e37_79b1_85eb_ca87)
            .wrapping_add(value)
    }

    fn production_shape_checksums(reservoirs: &[HashPruneReservoir]) -> (u64, u64) {
        let mut selection_checksum = 0_u64;
        let mut metadata_checksum = 0_u64;

        for visit in 0..PRODUCTION_SHAPE_RESERVOIRS {
            let source =
                (visit * PRODUCTION_SHAPE_ACCESS_STRIDE) & (PRODUCTION_SHAPE_RESERVOIRS - 1);
            let slots = reservoirs[source].slots();
            selection_checksum = fold_insert_benchmark_checksum(selection_checksum, source as u64);
            selection_checksum =
                fold_insert_benchmark_checksum(selection_checksum, slots.len() as u64);
            metadata_checksum = fold_insert_benchmark_checksum(metadata_checksum, source as u64);
            metadata_checksum =
                fold_insert_benchmark_checksum(metadata_checksum, slots.len() as u64);

            for (slot_index, slot) in slots.iter().enumerate() {
                selection_checksum =
                    fold_insert_benchmark_checksum(selection_checksum, slot.target as u64);
                selection_checksum =
                    fold_insert_benchmark_checksum(selection_checksum, slot.hash as u64);
                selection_checksum =
                    fold_insert_benchmark_checksum(selection_checksum, slot.dist.to_bits() as u64);
                metadata_checksum =
                    fold_insert_benchmark_checksum(metadata_checksum, slot_index as u64);
                metadata_checksum =
                    fold_insert_benchmark_checksum(metadata_checksum, slot.target as u64);
                metadata_checksum = fold_insert_benchmark_checksum(
                    metadata_checksum,
                    slot.meta.to_spill_flags().unwrap() as u64,
                );
            }
        }

        (selection_checksum, metadata_checksum)
    }

    fn production_shape_primary_checksum(reservoirs: &[HashPruneReservoir]) -> u64 {
        let mut checksum = 0_u64;
        for visit in 0..PRODUCTION_SHAPE_RESERVOIRS {
            let source =
                (visit * PRODUCTION_SHAPE_ACCESS_STRIDE) & (PRODUCTION_SHAPE_RESERVOIRS - 1);
            let reservoir = &reservoirs[source];
            let backup_count = reservoir.backfill_count().unwrap();
            let primary_count = reservoir.slots.len() - backup_count;
            if primary_count == PRODUCTION_SHAPE_L_MAX {
                assert_eq!(backup_count, 0);
            }
            checksum = fold_insert_benchmark_checksum(checksum, source as u64);
            checksum = fold_insert_benchmark_checksum(checksum, primary_count as u64);
            for slot in &reservoir.slots[..primary_count] {
                checksum = fold_insert_benchmark_checksum(checksum, slot.target as u64);
                checksum = fold_insert_benchmark_checksum(checksum, slot.hash as u64);
                checksum = fold_insert_benchmark_checksum(checksum, slot.dist.to_bits() as u64);
            }
        }
        checksum
    }

    fn reset_production_shape_reservoirs(reservoirs: &mut [HashPruneReservoir]) {
        for reservoir in reservoirs {
            reservoir.clear();
        }
    }

    fn run_production_shape_insertions(
        workload: &ProductionShapeWorkload,
        reservoirs: &mut [HashPruneReservoir],
        insert: &mut impl FnMut(
            &mut HashPruneReservoir,
            &mut HashPruneInsertProfile,
            usize,
            u32,
            u16,
            f32,
            HashPruneMeta,
        ),
    ) -> (f64, ProductionShapeBenchmarkResult) {
        assert_eq!(reservoirs.len(), PRODUCTION_SHAPE_RESERVOIRS);
        assert_eq!(workload.row_offsets.len(), PRODUCTION_SHAPE_RESERVOIRS + 1);

        let mut profile = HashPruneInsertProfile::default();
        let start = Instant::now();
        for visit in 0..PRODUCTION_SHAPE_RESERVOIRS {
            let source =
                (visit * PRODUCTION_SHAPE_ACCESS_STRIDE) & (PRODUCTION_SHAPE_RESERVOIRS - 1);
            let row_start = workload.row_offsets[source];
            let row_end = workload.row_offsets[source + 1];
            let reservoir = &mut reservoirs[source];
            for &(target, hash, dist, meta) in &workload.records[row_start..row_end] {
                insert(reservoir, &mut profile, source, target, hash, dist, meta);
            }
        }
        let insertion_seconds = start.elapsed().as_secs_f64();

        let (selection_checksum, metadata_checksum) = production_shape_checksums(reservoirs);
        (
            insertion_seconds,
            ProductionShapeBenchmarkResult {
                insert: InsertBenchmarkResult {
                    selection_checksum,
                    profile,
                },
                metadata_checksum,
            },
        )
    }

    fn measure_production_shape_insertions(
        workload: &ProductionShapeWorkload,
        reservoirs: &mut [HashPruneReservoir],
        mut insert: impl FnMut(
            &mut HashPruneReservoir,
            &mut HashPruneInsertProfile,
            usize,
            u32,
            u16,
            f32,
            HashPruneMeta,
        ),
    ) -> ProductionShapeBenchmarkMeasurement {
        reset_production_shape_reservoirs(reservoirs);
        let (_, warmup_result) = run_production_shape_insertions(workload, reservoirs, &mut insert);
        black_box(warmup_result);

        let mut best_ns_per_candidate = f64::INFINITY;
        let mut best_result = None;
        for _ in 0..3 {
            reset_production_shape_reservoirs(reservoirs);
            let (insertion_seconds, result) =
                run_production_shape_insertions(workload, reservoirs, &mut insert);
            let ns_per_candidate = insertion_seconds * 1e9 / workload.records.len() as f64;
            let result = black_box(result);
            if ns_per_candidate < best_ns_per_candidate {
                best_ns_per_candidate = ns_per_candidate;
                best_result = Some(result);
            }
        }

        ProductionShapeBenchmarkMeasurement {
            best_ns_per_candidate,
            result: best_result.unwrap(),
        }
    }

    fn run_production_shape_backfill_insertions(
        workload: &ProductionShapeWorkload,
        reservoirs: &mut [HashPruneReservoir],
        policy: HashPruneBackfillPolicy,
    ) -> (f64, ProductionShapeBackfillBenchmarkResult) {
        assert_eq!(reservoirs.len(), PRODUCTION_SHAPE_RESERVOIRS);
        let mut profile = HashPruneBackfillProfile::default();
        let insertion_start = Instant::now();
        for visit in 0..PRODUCTION_SHAPE_RESERVOIRS {
            let source =
                (visit * PRODUCTION_SHAPE_ACCESS_STRIDE) & (PRODUCTION_SHAPE_RESERVOIRS - 1);
            let row_start = workload.row_offsets[source];
            let row_end = workload.row_offsets[source + 1];
            let reservoir = &mut reservoirs[source];
            for &(target, hash, dist, _meta) in &workload.records[row_start..row_end] {
                reservoir
                    .insert_backfill_prevalidated_with_policy(
                        source,
                        target,
                        hash,
                        dist,
                        policy,
                        &mut profile,
                    )
                    .expect("policy-aware benchmark insertion");
            }
        }
        let insertion_seconds = insertion_start.elapsed().as_secs_f64();
        let primary_checksum = production_shape_primary_checksum(reservoirs);

        let finalization_start = Instant::now();
        for visit in 0..PRODUCTION_SHAPE_RESERVOIRS {
            let source =
                (visit * PRODUCTION_SHAPE_ACCESS_STRIDE) & (PRODUCTION_SHAPE_RESERVOIRS - 1);
            let primary_count = reservoirs[source].slots.len()
                - reservoirs[source]
                    .backfill_count()
                    .expect("valid benchmark backup count");
            reservoirs[source]
                .finish_backfill(source, policy, &mut profile)
                .expect("distance-gated finalization");
            if policy.is_tiered() {
                assert!(
                    reservoirs[source].slots.len()
                        <= primary_count.max(policy.degree_target_for(PRODUCTION_SHAPE_L_MAX)),
                    "tiered backfill exceeded its per-row target at source {source}"
                );
            }
        }
        let finalization_ns_per_row =
            finalization_start.elapsed().as_secs_f64() * 1e9 / reservoirs.len() as f64;
        let (selection_checksum, _) = production_shape_checksums(reservoirs);
        assert!(
            reservoirs
                .iter()
                .all(|reservoir| reservoir.slots.capacity() == PRODUCTION_SHAPE_L_MAX)
        );

        (
            insertion_seconds,
            ProductionShapeBackfillBenchmarkResult {
                finalization_ns_per_row,
                profile,
                primary_checksum,
                selection_checksum,
            },
        )
    }

    fn measure_production_shape_backfill_insertions(
        workload: &ProductionShapeWorkload,
        reservoirs: &mut [HashPruneReservoir],
        policy: HashPruneBackfillPolicy,
    ) -> ProductionShapeBackfillBenchmarkMeasurement {
        reset_production_shape_reservoirs(reservoirs);
        let (_, warmup_result) =
            run_production_shape_backfill_insertions(workload, reservoirs, policy);
        black_box(warmup_result);

        let mut best_ns_per_candidate = f64::INFINITY;
        let mut best_result = None;
        for _ in 0..3 {
            reset_production_shape_reservoirs(reservoirs);
            let (insertion_seconds, result) =
                run_production_shape_backfill_insertions(workload, reservoirs, policy);
            let ns_per_candidate = insertion_seconds * 1e9 / workload.records.len() as f64;
            let result = black_box(result);
            if ns_per_candidate < best_ns_per_candidate {
                best_ns_per_candidate = ns_per_candidate;
                best_result = Some(result);
            }
        }

        ProductionShapeBackfillBenchmarkMeasurement {
            best_ns_per_candidate,
            result: best_result.unwrap(),
        }
    }

    #[test]
    fn hashprune_insert_benchmark_observes_all_profile_counters() {
        const RECORDS_PER_ROW: usize = 256;
        let mut records = Vec::with_capacity(RECORDS_PER_ROW);

        for hash in 100_u16..164 {
            let dist = match hash {
                110 => 900.0,
                150 => 1_000.0,
                _ => f32::from(hash - 99),
            };
            records.push((
                u32::from(hash),
                hash,
                dist,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            ));
        }
        records.extend([
            (
                12_000,
                120,
                0.25,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0),
            ),
            (
                12_100,
                121,
                10_000.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse),
            ),
            (
                5_000,
                50,
                0.1,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional),
            ),
            (
                30_000,
                300,
                0.2,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::ReverseRank0),
            ),
            (
                40_000,
                400,
                2_000.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0),
            ),
        ]);
        while records.len() < RECORDS_PER_ROW {
            let ordinal = records.len() as u32;
            records.push((
                50_000 + ordinal,
                120,
                20_000.0 + ordinal as f32,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            ));
        }

        let mut reservoir = HashPruneReservoir::new(64);
        let mut run = || {
            run_insert_microbenchmark_workload(
                &mut reservoir,
                &records,
                |reservoir, profile, _source, target, hash, dist, _meta| {
                    reservoir.insert_profiled(target, hash, dist, profile);
                },
            )
        };
        let first = run();
        let second = run();

        assert_eq!(first.selection_checksum, second.selection_checksum);
        assert_eq!(first.profile.candidates, RECORDS_PER_ROW);
        assert_eq!(first.profile.hash_hits, 189);
        assert_eq!(first.profile.hash_replacements, 1);
        assert_eq!(first.profile.hash_rejects, 188);
        assert_eq!(first.profile.not_full_inserts, 64);
        assert_eq!(first.profile.full_replacement_checks, 3);
        assert_eq!(first.profile.full_replacements, 2);
        assert_eq!(first.profile.full_rejects, 1);
        assert_eq!(first.profile.farthest_recomputes, 2);
        assert_eq!(first.profile.rotate_lefts, 1);
        assert_eq!(first.profile.rotate_rights, 1);
    }

    #[test]
    fn hashprune_metadata_layout_slot_is_12_bytes_and_carries_meta() {
        assert_eq!(std::mem::size_of::<super::Slot>(), 12);

        let meta = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        let slot = super::Slot::new_with_meta(7, 3, 1.25, meta);
        assert_eq!(slot.meta, meta);
    }

    #[test]
    fn reservoir_prefers_exact_distance_within_same_bfloat16_bucket() {
        let better = (10u32, 10u16, 66_703.0f32);
        let worse = (20u32, 20u16, 66_763.0f32);
        assert_eq!(
            super::BFloat16::from_f32(better.2).to_f32(),
            super::BFloat16::from_f32(worse.2).to_f32(),
            "test candidates must share a BF16 bucket to exercise exact-distance comparison"
        );

        let mut worse_first = HashPruneReservoir::new(1);
        worse_first.insert(worse.0, worse.1, worse.2);
        worse_first.insert(better.0, better.1, better.2);

        let mut better_first = HashPruneReservoir::new(1);
        better_first.insert(better.0, better.1, better.2);
        better_first.insert(worse.0, worse.1, worse.2);

        assert_eq!(sorted_neighbors(worse_first), vec![better.0]);
        assert_eq!(sorted_neighbors(better_first), vec![better.0]);
    }

    #[test]
    fn reservoir_full_path_uses_exact_distance_for_farthest_replacement() {
        let better = (10u32, 10u16, 66_703.0f32);
        let worse = (20u32, 20u16, 66_763.0f32);
        assert_eq!(
            super::BFloat16::from_f32(better.2).to_f32(),
            super::BFloat16::from_f32(worse.2).to_f32(),
            "test candidates must share a BF16 bucket to exercise exact-distance comparison"
        );

        let mut reservoir = HashPruneReservoir::new(2);
        reservoir.insert(1, 1, 10.0);
        reservoir.insert(worse.0, worse.1, worse.2);
        reservoir.insert(better.0, better.1, better.2);

        assert_eq!(sorted_neighbors(reservoir), vec![1, better.0]);
    }

    #[test]
    fn reservoir_same_hash_collision_uses_exact_distance() {
        let better = (10u32, 7u16, 66_703.0f32);
        let worse = (20u32, 7u16, 66_763.0f32);
        assert_eq!(
            super::BFloat16::from_f32(better.2).to_f32(),
            super::BFloat16::from_f32(worse.2).to_f32(),
            "test candidates must share a BF16 bucket to exercise exact-distance comparison"
        );

        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert(worse.0, worse.1, worse.2);
        reservoir.insert(better.0, better.1, better.2);

        assert_eq!(sorted_neighbors(reservoir), vec![better.0]);
    }

    #[test]
    fn reservoir_prefers_exact_nearer_distance_inside_legacy_epsilon_band() {
        let mut reservoir = HashPruneReservoir::new(1);
        reservoir.insert(30, 30, 1.0);
        reservoir.insert(20, 20, 1.0 + 1.5e-5);

        assert_eq!(sorted_neighbors(reservoir), vec![30]);
    }

    #[test]
    fn reservoir_same_hash_collision_prefers_exact_nearer_distance_inside_legacy_epsilon_band() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert(30, 7, 1.0);
        reservoir.insert(20, 7, 1.0 + 1.5e-5);

        assert_eq!(sorted_neighbors(reservoir), vec![30]);
    }

    #[test]
    fn hashprune_strict_order_is_permutation_independent_for_epsilon_cycle() {
        let candidates = [
            (2_u32, 7_u16, 0.0_f32),
            (1_u32, 7_u16, 1.5e-5_f32),
            (0_u32, 7_u16, 3.0e-5_f32),
        ];
        let orders = [[0_usize, 1, 2], [2_usize, 1, 0], [1_usize, 2, 0]];

        for order in orders {
            let mut reservoir = HashPruneReservoir::new(1);
            for idx in order {
                let (target, hash, dist) = candidates[idx];
                reservoir.insert(target, hash, dist);
            }
            assert_eq!(reservoir.slots()[0].target, 2);
            assert_eq!(reservoir.slots()[0].dist, 0.0);
        }
    }

    #[test]
    fn reservoir_insert_profile_counts_replacement_paths() {
        let mut reservoir = HashPruneReservoir::new(2);
        let mut profile = HashPruneInsertProfile::default();

        reservoir.insert_profiled(10, 1, 2.0, &mut profile);
        reservoir.insert_profiled(20, 2, 5.0, &mut profile);
        reservoir.insert_profiled(30, 2, 4.0, &mut profile);
        reservoir.insert_profiled(40, 2, 6.0, &mut profile);
        reservoir.insert_profiled(50, 3, 1.0, &mut profile);
        reservoir.insert_profiled(60, 4, 9.0, &mut profile);

        assert_eq!(sorted_neighbors(reservoir), vec![10, 50]);
        assert_eq!(profile.candidates, 6);
        assert_eq!(profile.hash_hits, 2);
        assert_eq!(profile.hash_replacements, 1);
        assert_eq!(profile.hash_rejects, 1);
        assert_eq!(profile.not_full_inserts, 2);
        assert_eq!(profile.full_replacement_checks, 2);
        assert_eq!(profile.full_replacements, 1);
        assert_eq!(profile.full_rejects, 1);
        assert_eq!(profile.farthest_recomputes, 2);
    }

    #[test]
    fn hashprune_report_insert_inner_returns_final_candidate_slot() {
        let mut same_hash = HashPruneReservoir::new(2);
        assert_eq!(same_hash.insert_inner(10, 20, 5.0, None), Some(0));
        assert_eq!(same_hash.insert_inner(10, 20, 6.0, None), Some(0));
        assert_eq!(same_hash.insert_inner(11, 20, 6.0, None), None);
        assert_eq!(same_hash.insert_inner(11, 20, 4.0, None), Some(0));
        assert_eq!(same_hash.insert_inner(12, 10, 3.0, None), Some(0));

        let mut rotate_right = HashPruneReservoir::new(3);
        rotate_right.insert(100, 10, 1.0);
        rotate_right.insert(300, 30, 2.0);
        rotate_right.insert(500, 50, 9.0);
        assert_eq!(rotate_right.insert_inner(200, 20, 0.5, None), Some(1));

        let mut rotate_left = HashPruneReservoir::new(3);
        rotate_left.insert(100, 10, 1.0);
        rotate_left.insert(300, 30, 9.0);
        rotate_left.insert(500, 50, 2.0);
        assert_eq!(rotate_left.insert_inner(600, 60, 0.5, None), Some(2));
        assert_eq!(rotate_left.insert_inner(700, 70, 20.0, None), None);
    }

    #[test]
    fn hashprune_report_prevalidated_matches_validated_for_permutations() {
        let candidates = [
            (
                100,
                10,
                1.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            ),
            (
                300,
                30,
                2.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse),
            ),
            (
                500,
                50,
                9.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0),
            ),
            (
                500,
                50,
                8.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
            ),
            (
                200,
                20,
                0.5,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0),
            ),
            (
                201,
                20,
                0.4,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse),
            ),
            (
                202,
                20,
                10.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0),
            ),
            (
                700,
                70,
                100.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional),
            ),
            (
                600,
                60,
                0.25,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::ReverseRank0),
            ),
            (
                400,
                40,
                0.1,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional),
            ),
        ];
        let orders = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [3, 0, 6, 1, 7, 2, 8, 4, 9, 5],
            [5, 2, 8, 1, 4, 7, 0, 3, 6, 9],
            [6, 4, 2, 0, 8, 7, 5, 3, 1, 9],
            [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        ];
        let mut coverage = HashPruneInsertProfile::default();

        for order in orders {
            let mut validated = HashPruneReservoir::new(3);
            let mut validated_profile = HashPruneInsertProfile::default();
            let mut prevalidated = HashPruneReservoir::new(3);

            for idx in order {
                let (target, hash, dist, meta) = candidates[idx];
                validated
                    .insert_with_policy_profiled(
                        0,
                        target,
                        hash,
                        dist,
                        meta,
                        HashPruneEvidencePolicy::report(),
                        &mut validated_profile,
                    )
                    .unwrap();
                prevalidated.insert_report_prevalidated(0, target, hash, dist, meta);
            }

            assert_eq!(slot_snapshot(&prevalidated), slot_snapshot(&validated));
            coverage.merge(validated_profile);
        }

        assert!(coverage.not_full_inserts > 0);
        assert!(coverage.hash_replacements > 0);
        assert!(coverage.hash_rejects > 0);
        assert!(coverage.full_replacements > 0);
        assert!(coverage.full_rejects > 0);
        assert!(coverage.rotate_lefts > 0);
        assert!(coverage.rotate_rights > 0);
    }

    #[test]
    fn hashprune_report_prevalidated_preserves_same_target_max_evidence() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert_report_prevalidated(
            0,
            10,
            30,
            5.0,
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0),
        );
        reservoir.insert_report_prevalidated(
            0,
            10,
            30,
            3.0,
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
        );
        reservoir.insert_report_prevalidated(0, 10, 30, 7.0, HashPruneMeta::mandatory());

        assert_eq!(reservoir.slots()[0].target, 10);
        assert_eq!(reservoir.slots()[0].dist, 3.0);
        assert!(reservoir.slots()[0].meta.is_mandatory());
        assert_eq!(
            reservoir.slots()[0].meta.evidence(),
            Some(HashPruneEvidenceClass::BidirectionalDirectRank0)
        );
    }

    #[test]
    fn hashprune_report_prevalidated_drops_displaced_collision_metadata() {
        let mut reservoir = HashPruneReservoir::new(4);
        let displaced_meta = HashPruneMeta::mandatory().merge_prevalidated_candidate(
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0),
        );
        reservoir.insert_report_prevalidated(0, 10, 30, 5.0, displaced_meta);
        reservoir.insert_report_prevalidated(
            0,
            20,
            30,
            3.0,
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse),
        );
        reservoir.insert_report_prevalidated(0, 10, 30, 6.0, displaced_meta);

        assert_eq!(reservoir.slots()[0].target, 20);
        assert_eq!(reservoir.slots()[0].dist, 3.0);
        assert!(!reservoir.slots()[0].meta.is_mandatory());
        assert_eq!(
            reservoir.slots()[0].meta.evidence(),
            Some(HashPruneEvidenceClass::Reverse)
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn hashprune_report_prevalidated_debug_rejects_overflow_metadata() {
        let mut reservoir = HashPruneReservoir::new(2);
        reservoir.insert(10, 10, 5.0);
        let before = slot_snapshot(&reservoir);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            reservoir.insert_report_prevalidated(
                0,
                20,
                20,
                1.0,
                HashPruneMeta::empty().mark_overflow(),
            );
        }));

        assert!(panic.is_err());
        assert_eq!(slot_snapshot(&reservoir), before);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn hashprune_report_prevalidated_debug_rejects_same_target_different_hash_before_mutation() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert_report_prevalidated(
            7,
            10,
            30,
            5.0,
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
        );
        let before = slot_snapshot(&reservoir);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            reservoir.insert_report_prevalidated(
                7,
                10,
                31,
                1.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0),
            );
        }));

        assert!(panic.is_err());
        assert_eq!(slot_snapshot(&reservoir), before);
        assert_eq!(reservoir.slots().len(), 1);
    }

    #[test]
    fn hashprune_report_prevalidated_keeps_slot_layout() {
        let mut reservoir = HashPruneReservoir::new(1);
        let meta = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        reservoir.insert_report_prevalidated(0, 7, 3, 1.25, meta);

        assert_eq!(std::mem::size_of::<super::Slot>(), 12);
        assert_eq!(
            slot_snapshot(&reservoir),
            vec![(7, 3, 1.25_f32.to_bits(), meta)]
        );
    }

    #[test]
    fn hashprune_report_same_target_better_replacement_retains_max_evidence() {
        let mut reservoir = HashPruneReservoir::new(4);
        let mut profile = HashPruneInsertProfile::default();

        insert_report(
            &mut reservoir,
            &mut profile,
            10,
            30,
            5.0,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        insert_report(
            &mut reservoir,
            &mut profile,
            10,
            30,
            3.0,
            HashPruneEvidenceClass::Direct,
        );

        assert_eq!(reservoir.slots()[0].target, 10);
        assert_eq!(reservoir.slots()[0].dist, 3.0);
        assert_eq!(
            reservoir.slots()[0].meta.evidence(),
            Some(HashPruneEvidenceClass::BidirectionalDirectRank0)
        );
    }

    #[test]
    fn hashprune_report_same_hash_different_target_replacement_drops_old_metadata() {
        let mut reservoir = HashPruneReservoir::new(4);
        let mut profile = HashPruneInsertProfile::default();

        insert_report(
            &mut reservoir,
            &mut profile,
            10,
            30,
            5.0,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        insert_report(
            &mut reservoir,
            &mut profile,
            20,
            30,
            3.0,
            HashPruneEvidenceClass::Direct,
        );

        assert_eq!(reservoir.slots()[0].target, 20);
        assert_eq!(
            reservoir.slots()[0].meta.evidence(),
            Some(HashPruneEvidenceClass::Direct)
        );
    }

    #[test]
    fn hashprune_report_underfull_insert_binds_metadata_to_insert_position() {
        let mut reservoir = HashPruneReservoir::new(4);
        let mut profile = HashPruneInsertProfile::default();
        insert_report(
            &mut reservoir,
            &mut profile,
            100,
            10,
            1.0,
            HashPruneEvidenceClass::Reverse,
        );
        insert_report(
            &mut reservoir,
            &mut profile,
            500,
            50,
            2.0,
            HashPruneEvidenceClass::Direct,
        );

        insert_report(
            &mut reservoir,
            &mut profile,
            300,
            30,
            1.5,
            HashPruneEvidenceClass::DirectRank0,
        );

        assert_eq!(reservoir.slots()[1].target, 300);
        assert_eq!(reservoir.slots()[1].hash, 30);
        assert_eq!(
            reservoir.slots()[1].meta.evidence(),
            Some(HashPruneEvidenceClass::DirectRank0)
        );
    }

    #[test]
    fn hashprune_report_full_replacement_binds_metadata_after_both_rotations() {
        let mut rotate_right = HashPruneReservoir::new(3);
        let mut rotate_right_profile = HashPruneInsertProfile::default();
        insert_report(
            &mut rotate_right,
            &mut rotate_right_profile,
            100,
            10,
            1.0,
            HashPruneEvidenceClass::Reverse,
        );
        insert_report(
            &mut rotate_right,
            &mut rotate_right_profile,
            300,
            30,
            2.0,
            HashPruneEvidenceClass::ReverseRank0,
        );
        insert_report(
            &mut rotate_right,
            &mut rotate_right_profile,
            500,
            50,
            9.0,
            HashPruneEvidenceClass::DirectRank0,
        );
        insert_report(
            &mut rotate_right,
            &mut rotate_right_profile,
            200,
            20,
            0.5,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        assert_eq!(rotate_right.slots()[1].target, 200);
        assert_eq!(
            rotate_right.slots()[1].meta.evidence(),
            Some(HashPruneEvidenceClass::BidirectionalDirectRank0)
        );

        let mut rotate_left = HashPruneReservoir::new(3);
        let mut rotate_left_profile = HashPruneInsertProfile::default();
        insert_report(
            &mut rotate_left,
            &mut rotate_left_profile,
            100,
            10,
            1.0,
            HashPruneEvidenceClass::Reverse,
        );
        insert_report(
            &mut rotate_left,
            &mut rotate_left_profile,
            300,
            30,
            9.0,
            HashPruneEvidenceClass::DirectRank0,
        );
        insert_report(
            &mut rotate_left,
            &mut rotate_left_profile,
            500,
            50,
            2.0,
            HashPruneEvidenceClass::Direct,
        );
        insert_report(
            &mut rotate_left,
            &mut rotate_left_profile,
            600,
            60,
            0.5,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        );
        assert_eq!(rotate_left.slots()[2].target, 600);
        assert_eq!(
            rotate_left.slots()[2].meta.evidence(),
            Some(HashPruneEvidenceClass::BidirectionalDirectRank0)
        );
        assert_eq!(rotate_left.slots()[1].target, 500);
        assert_eq!(
            rotate_left.slots()[1].meta.evidence(),
            Some(HashPruneEvidenceClass::Direct)
        );
    }

    #[test]
    fn hashprune_report_one_pass_matches_three_search_reference_for_permutations() {
        let candidates = [
            (200, 20, 8.0, HashPruneEvidenceClass::Direct),
            (400, 40, 7.0, HashPruneEvidenceClass::Reverse),
            (200, 20, 6.0, HashPruneEvidenceClass::DirectRank0),
            (201, 20, 5.0, HashPruneEvidenceClass::Bidirectional),
            (600, 60, 9.0, HashPruneEvidenceClass::ReverseRank0),
            (800, 80, 4.0, HashPruneEvidenceClass::Direct),
            (100, 10, 3.0, HashPruneEvidenceClass::DirectRank0),
            (900, 90, 2.0, HashPruneEvidenceClass::Bidirectional),
            (
                201,
                20,
                8.0,
                HashPruneEvidenceClass::BidirectionalDirectRank0,
            ),
        ];
        let orders = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1, 0],
            [3, 0, 6, 1, 7, 2, 8, 4, 5],
            [5, 2, 8, 1, 4, 7, 0, 3, 6],
            [6, 4, 2, 0, 8, 7, 5, 3, 1],
            [1, 3, 5, 7, 0, 2, 4, 6, 8],
        ];

        for order in orders {
            let mut one_pass = HashPruneReservoir::new(4);
            let mut one_pass_profile = HashPruneInsertProfile::default();
            let mut reference = HashPruneReservoir::new(4);
            let mut reference_profile = HashPruneInsertProfile::default();

            for idx in order {
                let (target, hash, dist, class) = candidates[idx];
                insert_report(
                    &mut one_pass,
                    &mut one_pass_profile,
                    target,
                    hash,
                    dist,
                    class,
                );
                insert_report_three_search_reference(
                    &mut reference,
                    &mut reference_profile,
                    target,
                    hash,
                    dist,
                    HashPruneMeta::with_evidence(class),
                );
            }

            assert_eq!(slot_snapshot(&one_pass), slot_snapshot(&reference));
            assert_insert_profiles_equal(&one_pass_profile, &reference_profile);
        }
    }

    #[test]
    #[ignore = "manual release microbenchmark"]
    fn hashprune_report_insert_one_pass_microbenchmark() {
        const ROWS: usize = 16_384;
        const RECORDS_PER_ROW: usize = 256;
        let mut records = Vec::with_capacity(ROWS * RECORDS_PER_ROW);
        for row in 0..ROWS {
            for lane in 0..RECORDS_PER_ROW {
                let ordinal = (lane * 73 + row * 17) & (RECORDS_PER_ROW - 1);
                let hash = ((ordinal * 37) % 96 + (row * 97) % 4096) as u16;
                let local_target = if ordinal >= 96 && ordinal % 8 == 0 {
                    ordinal - 96
                } else {
                    ordinal
                };
                let target = (row * 512 + local_target) as u32;
                let dist_seed = (ordinal * 7919 + row * 104_729) % 100_003;
                let dist = dist_seed as f32 / 32.0;
                let class = HashPruneEvidenceClass::ALL[(ordinal + row) % 6];
                records.push((target, hash, dist, HashPruneMeta::with_evidence(class)));
            }
        }
        black_box(&records);

        let mut plain_reservoir = HashPruneReservoir::new(64);
        let plain_ns = best_ns_per_record(
            || {
                run_insert_microbenchmark_workload(
                    &mut plain_reservoir,
                    &records,
                    |reservoir, _profile, _source, target, hash, dist, _meta| {
                        reservoir.insert(target, hash, dist);
                    },
                )
            },
            records.len(),
        );
        let mut profiled_reservoir = HashPruneReservoir::new(64);
        let profiled_ns = best_ns_per_record(
            || {
                run_insert_microbenchmark_workload(
                    &mut profiled_reservoir,
                    &records,
                    |reservoir, profile, _source, target, hash, dist, _meta| {
                        reservoir.insert_profiled(target, hash, dist, profile);
                    },
                )
            },
            records.len(),
        );
        let mut three_search_reservoir = HashPruneReservoir::new(64);
        let three_search_ns = best_ns_per_record(
            || {
                run_insert_microbenchmark_workload(
                    &mut three_search_reservoir,
                    &records,
                    |reservoir, profile, _source, target, hash, dist, meta| {
                        insert_report_three_search_reference(
                            reservoir, profile, target, hash, dist, meta,
                        );
                    },
                )
            },
            records.len(),
        );
        let mut one_pass_reservoir = HashPruneReservoir::new(64);
        let one_pass_ns = best_ns_per_record(
            || {
                run_insert_microbenchmark_workload(
                    &mut one_pass_reservoir,
                    &records,
                    |reservoir, profile, source, target, hash, dist, meta| {
                        reservoir
                            .insert_with_policy_profiled(
                                source,
                                target,
                                hash,
                                dist,
                                meta,
                                HashPruneEvidencePolicy::report(),
                                profile,
                            )
                            .unwrap();
                    },
                )
            },
            records.len(),
        );

        println!(
            "hashprune report insert microbenchmark: records={}, one warmup + best of 3; record generation and reservoir allocation/teardown excluded",
            records.len()
        );
        println!(
            "plain insert:              {plain_ns:8.2} ns/record, {ratio:.3}x plain",
            ratio = plain_ns / plain_ns
        );
        println!(
            "profiled insert:           {profiled_ns:8.2} ns/record, {ratio:.3}x plain",
            ratio = profiled_ns / plain_ns
        );
        println!(
            "old three-search report:   {three_search_ns:8.2} ns/record, {ratio:.3}x plain",
            ratio = three_search_ns / plain_ns
        );
        println!(
            "new one-pass report:       {one_pass_ns:8.2} ns/record, {ratio:.3}x plain",
            ratio = one_pass_ns / plain_ns
        );
        println!(
            "one-pass / three-search:   {ratio:.3}x",
            ratio = one_pass_ns / three_search_ns
        );
    }

    #[test]
    #[ignore = "manual release microbenchmark"]
    fn hashprune_report_production_shape_insert_microbenchmark() {
        let setup_start = Instant::now();
        let workload = build_production_shape_workload();
        let mut reservoirs = (0..PRODUCTION_SHAPE_RESERVOIRS)
            .map(|_| HashPruneReservoir::new(PRODUCTION_SHAPE_L_MAX))
            .collect::<Vec<_>>();
        let setup_seconds = setup_start.elapsed().as_secs_f64();
        black_box(&workload);
        black_box(&reservoirs);

        let mut min_candidates = usize::MAX;
        let mut max_candidates = 0;
        let mut underfull_rows = 0;
        let mut full_rows = 0;
        for offsets in workload.row_offsets.windows(2) {
            let candidate_count = offsets[1] - offsets[0];
            min_candidates = min_candidates.min(candidate_count);
            max_candidates = max_candidates.max(candidate_count);
            if candidate_count <= PRODUCTION_SHAPE_L_MAX {
                underfull_rows += 1;
            } else {
                full_rows += 1;
            }
        }
        assert_eq!(min_candidates, PRODUCTION_SHAPE_MIN_CANDIDATES);
        assert_eq!(max_candidates, PRODUCTION_SHAPE_MAX_CANDIDATES);
        assert!(underfull_rows > 0);
        assert!(full_rows > 0);

        let plain = measure_production_shape_insertions(
            &workload,
            &mut reservoirs,
            |reservoir, _profile, _source, target, hash, dist, _meta| {
                reservoir.insert(target, hash, dist);
            },
        );
        let profiled = measure_production_shape_insertions(
            &workload,
            &mut reservoirs,
            |reservoir, profile, _source, target, hash, dist, _meta| {
                reservoir.insert_profiled(target, hash, dist, profile);
            },
        );
        let validated_report = measure_production_shape_insertions(
            &workload,
            &mut reservoirs,
            |reservoir, profile, source, target, hash, dist, meta| {
                reservoir
                    .insert_with_policy_profiled(
                        source,
                        target,
                        hash,
                        dist,
                        meta,
                        HashPruneEvidencePolicy::report(),
                        profile,
                    )
                    .unwrap();
            },
        );
        let prevalidated_report = measure_production_shape_insertions(
            &workload,
            &mut reservoirs,
            |reservoir, _profile, source, target, hash, dist, meta| {
                reservoir.insert_report_prevalidated(source, target, hash, dist, meta);
            },
        );
        let backfill_k8 = measure_production_shape_backfill_insertions(
            &workload,
            &mut reservoirs,
            HashPruneBackfillPolicy::distance_gated(8, 1.1),
        );
        let backfill_k4 = measure_production_shape_backfill_insertions(
            &workload,
            &mut reservoirs,
            HashPruneBackfillPolicy::distance_gated(4, 1.1),
        );
        let backfill_stratified_k4 = measure_production_shape_backfill_insertions(
            &workload,
            &mut reservoirs,
            HashPruneBackfillPolicy::distance_gated_stratified(4, 1.1),
        );
        let backfill_multihit_k6 = measure_production_shape_backfill_insertions(
            &workload,
            &mut reservoirs,
            HashPruneBackfillPolicy::distance_gated_multihit(6, 1.1),
        );
        let backfill_tiered_d52_k8 = measure_production_shape_backfill_insertions(
            &workload,
            &mut reservoirs,
            HashPruneBackfillPolicy::distance_gated_tiered_d52(8, 1.1),
        );
        let backfill_tiered_rhalf = measure_production_shape_backfill_insertions(
            &workload,
            &mut reservoirs,
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::None),
        );

        assert_eq!(
            plain.result.insert.selection_checksum,
            profiled.result.insert.selection_checksum
        );
        assert_eq!(
            plain.result.insert.selection_checksum,
            validated_report.result.insert.selection_checksum
        );
        assert_eq!(
            validated_report.result.insert.selection_checksum,
            prevalidated_report.result.insert.selection_checksum
        );
        assert_eq!(
            plain.result.metadata_checksum,
            profiled.result.metadata_checksum
        );
        assert_ne!(
            plain.result.metadata_checksum,
            validated_report.result.metadata_checksum
        );
        assert_eq!(
            validated_report.result.metadata_checksum,
            prevalidated_report.result.metadata_checksum
        );
        assert_insert_profiles_equal(
            &profiled.result.insert.profile,
            &validated_report.result.insert.profile,
        );
        assert_insert_profiles_equal(
            &plain.result.insert.profile,
            &HashPruneInsertProfile::default(),
        );
        assert_insert_profiles_equal(
            &prevalidated_report.result.insert.profile,
            &HashPruneInsertProfile::default(),
        );
        assert_eq!(
            profiled.result.insert.profile.candidates,
            workload.records.len()
        );
        assert_eq!(
            backfill_k8.result.primary_checksum,
            plain.result.insert.selection_checksum
        );
        assert_eq!(
            backfill_k4.result.primary_checksum,
            plain.result.insert.selection_checksum
        );
        assert_eq!(
            backfill_stratified_k4.result.primary_checksum,
            plain.result.insert.selection_checksum
        );
        assert_eq!(
            backfill_multihit_k6.result.primary_checksum,
            plain.result.insert.selection_checksum
        );
        assert_eq!(
            backfill_tiered_d52_k8.result.primary_checksum,
            plain.result.insert.selection_checksum
        );
        assert_eq!(
            backfill_tiered_rhalf.result.primary_checksum,
            plain.result.insert.selection_checksum
        );
        for backfill in [
            &backfill_k8,
            &backfill_k4,
            &backfill_stratified_k4,
            &backfill_multihit_k6,
            &backfill_tiered_d52_k8,
            &backfill_tiered_rhalf,
        ] {
            assert_eq!(
                backfill.result.profile.rows_seen as usize,
                PRODUCTION_SHAPE_RESERVOIRS
            );
            assert_eq!(
                backfill.result.profile.primary_edges,
                backfill.result.profile.degree_before_sum
            );
            assert!(
                backfill.result.profile.degree_after_sum
                    >= backfill.result.profile.degree_before_sum
            );
        }
        let record_bytes = workload.records.len() * std::mem::size_of::<InsertBenchmarkRecord>();
        let row_offset_bytes = workload.row_offsets.len() * std::mem::size_of::<usize>();
        let reservoir_slot_bytes = PRODUCTION_SHAPE_RESERVOIRS
            * PRODUCTION_SHAPE_L_MAX
            * std::mem::size_of::<super::Slot>();
        let reservoir_header_bytes =
            PRODUCTION_SHAPE_RESERVOIRS * std::mem::size_of::<HashPruneReservoir>();
        let estimated_working_set_mib =
            (record_bytes + row_offset_bytes + reservoir_slot_bytes + reservoir_header_bytes)
                as f64
                / (1024.0 * 1024.0);

        println!(
            "hashprune production-shape insert microbenchmark: sources={}, l_max={}, candidates={}, candidates/source={}..{}, underfull_rows={}, full_rows={}, stride={}, estimated_working_set={estimated_working_set_mib:.1} MiB",
            PRODUCTION_SHAPE_RESERVOIRS,
            PRODUCTION_SHAPE_L_MAX,
            workload.records.len(),
            min_candidates,
            max_candidates,
            underfull_rows,
            full_rows,
            PRODUCTION_SHAPE_ACCESS_STRIDE,
        );
        println!(
            "statistics: one warmup + best of 3 measured iterations; setup={setup_seconds:.3}s excluded; workload generation, reservoir allocation/reset, ordered slot+metadata checksum, and teardown excluded; timed region is insertion only"
        );
        println!(
            "| variant | ns/candidate | x plain | x validated report | candidates | profile candidates | selection checksum | metadata checksum |"
        );
        for (name, measurement) in [
            ("plain", &plain),
            ("profiled", &profiled),
            ("validated report", &validated_report),
            ("prevalidated report", &prevalidated_report),
        ] {
            println!(
                "| {name} | {ns:.2} | {plain_ratio:.3} | {validated_ratio:.3} | {candidates} | {profile_candidates} | {selection:#018x} | {metadata:#018x} |",
                ns = measurement.best_ns_per_candidate,
                plain_ratio = measurement.best_ns_per_candidate / plain.best_ns_per_candidate,
                validated_ratio =
                    measurement.best_ns_per_candidate / validated_report.best_ns_per_candidate,
                candidates = workload.records.len(),
                profile_candidates = measurement.result.insert.profile.candidates,
                selection = measurement.result.insert.selection_checksum,
                metadata = measurement.result.metadata_checksum,
            );
        }
        println!(
            "| variant | ns/candidate | x plain | finalization ns/row | mean primary degree | mean final degree | selection checksum | reservoir capacity |"
        );
        for (name, backfill) in [
            ("distance-gated k=8 alpha=1.1", &backfill_k8),
            ("distance-gated k=4 alpha=1.1", &backfill_k4),
            (
                "distance-gated-stratified k=4 alpha=1.1",
                &backfill_stratified_k4,
            ),
            (
                "distance-gated-multihit k=6 alpha=1.1",
                &backfill_multihit_k6,
            ),
            (
                "distance-gated-tiered-d52 k=8 alpha=1.1",
                &backfill_tiered_d52_k8,
            ),
            (
                "distance-gated-tiered-rhalf B=R/2 q=3 alpha=1.1",
                &backfill_tiered_rhalf,
            ),
        ] {
            println!(
                "| {name} | {ns:.2} | {plain_ratio:.3} | {finalize:.2} | {degree_before:.3} | {degree_after:.3} | {selection:#018x} | {capacity} |",
                ns = backfill.best_ns_per_candidate,
                plain_ratio = backfill.best_ns_per_candidate / plain.best_ns_per_candidate,
                finalize = backfill.result.finalization_ns_per_row,
                degree_before = backfill.result.profile.mean_degree_before(),
                degree_after = backfill.result.profile.mean_degree_after(),
                selection = backfill.result.selection_checksum,
                capacity = PRODUCTION_SHAPE_L_MAX,
            );
        }
        println!(
            "| variant | hash hits | hash replacements | hash rejects | not-full inserts | full checks | full replacements | full rejects | farthest recomputes | rotate left | rotate right |"
        );
        for (name, measurement) in [
            ("plain", &plain),
            ("profiled", &profiled),
            ("validated report", &validated_report),
            ("prevalidated report", &prevalidated_report),
        ] {
            let profile = &measurement.result.insert.profile;
            println!(
                "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                profile.hash_hits,
                profile.hash_replacements,
                profile.hash_rejects,
                profile.not_full_inserts,
                profile.full_replacement_checks,
                profile.full_replacements,
                profile.full_rejects,
                profile.farthest_recomputes,
                profile.rotate_lefts,
                profile.rotate_rights,
            );
        }
        const MAX_TIERED_BACKFILL_OVERHEAD: f64 = 1.50;
        assert!(
            backfill_tiered_d52_k8.best_ns_per_candidate
                <= plain.best_ns_per_candidate * MAX_TIERED_BACKFILL_OVERHEAD,
            "tiered D52 insertion is {:.3}x plain",
            backfill_tiered_d52_k8.best_ns_per_candidate / plain.best_ns_per_candidate
        );
        assert!(
            backfill_tiered_rhalf.best_ns_per_candidate
                <= plain.best_ns_per_candidate * MAX_TIERED_BACKFILL_OVERHEAD,
            "tiered R-half insertion is {:.3}x plain",
            backfill_tiered_rhalf.best_ns_per_candidate / plain.best_ns_per_candidate
        );
    }

    #[test]
    fn hashprune_report_matches_off_and_merges_same_target_metadata() {
        let candidates = [
            (10, 1, 4.0, HashPruneEvidenceClass::Direct),
            (20, 2, 5.0, HashPruneEvidenceClass::Direct),
            (20, 2, 3.0, HashPruneEvidenceClass::DirectRank0),
            (30, 3, 2.0, HashPruneEvidenceClass::Bidirectional),
        ];
        let mut off = HashPruneReservoir::new(2);
        let mut report = HashPruneReservoir::new(2);
        let mut profile = HashPruneInsertProfile::default();

        for (target, hash, dist, class) in candidates {
            off.insert(target, hash, dist);
            report
                .insert_with_policy_profiled(
                    0,
                    target,
                    hash,
                    dist,
                    HashPruneMeta::with_evidence(class),
                    HashPruneEvidencePolicy::report(),
                    &mut profile,
                )
                .unwrap();
        }

        assert_eq!(
            off.slots()
                .iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                .collect::<Vec<_>>(),
            report
                .slots()
                .iter()
                .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
                .collect::<Vec<_>>()
        );
        let target_20 = report
            .slots()
            .iter()
            .find(|slot| slot.target == 20)
            .unwrap();
        assert_eq!(
            target_20.meta.evidence(),
            Some(HashPruneEvidenceClass::DirectRank0)
        );
    }

    #[test]
    fn hashprune_report_matches_off_keeps_enforce_disabled_until_phase1() {
        let mut reservoir = HashPruneReservoir::new(2);
        let mut profile = HashPruneInsertProfile::default();

        let error = reservoir
            .insert_with_policy_profiled(
                0,
                10,
                1,
                1.0,
                HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0),
                HashPruneEvidencePolicy::enforce_one(),
                &mut profile,
            )
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Phase 1 overflow policy is not installed")
        );
    }
}
