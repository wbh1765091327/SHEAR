use std::mem::{align_of, size_of};
use std::time::{Duration, Instant};
use std::{ptr, slice};

use memmap2::{MmapMut, MmapRaw};
use parking_lot::Mutex;

use super::hash_prune::{FinalizedHashPruneRow, HashPruneReservoir, HashPruneRuntimeState, Slot};
use super::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use crate::common::{AnnError, AnnResult};

pub(crate) const DEFAULT_FLAT_RESERVOIR_SEGMENT_ROWS: usize = 1 << 20;
pub(crate) const DEFAULT_FLAT_RESERVOIR_STRIPES: usize = 1 << 12;
const MMAP_PAGE_BYTES: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FlatReservoirMeta {
    len: u32,
    farthest_idx: u32,
    farthest_dist: f32,
    backup_count: u16,
    backup_worst: u16,
}

const _: () = assert!(size_of::<FlatReservoirMeta>() == 16);

impl FlatReservoirMeta {
    fn from_reservoir(reservoir: &HashPruneReservoir) -> AnnResult<Self> {
        let state = reservoir.runtime_state();
        Ok(Self {
            len: u32::try_from(reservoir.slots().len()).map_err(|_| {
                AnnError::log_index_error(
                    "flat HashPrune row length exceeds u32 metadata".to_string(),
                )
            })?,
            farthest_idx: u32::try_from(state.farthest_idx).map_err(|_| {
                AnnError::log_index_error(
                    "flat HashPrune farthest index exceeds u32 metadata".to_string(),
                )
            })?,
            farthest_dist: state.farthest_dist,
            backup_count: state.backup_count,
            backup_worst: state.backup_worst,
        })
    }

    fn decode(self, l_max: usize) -> AnnResult<(usize, HashPruneRuntimeState)> {
        let len = usize::try_from(self.len).map_err(|_| {
            AnnError::log_index_error("invalid flat HashPrune row length".to_string())
        })?;
        if len > l_max {
            return Err(AnnError::log_index_error(format!(
                "flat HashPrune row length {len} exceeds capacity {l_max}"
            )));
        }
        Ok((
            len,
            HashPruneRuntimeState {
                farthest_idx: usize::try_from(self.farthest_idx).map_err(|_| {
                    AnnError::log_index_error("invalid flat HashPrune farthest index".to_string())
                })?,
                farthest_dist: self.farthest_dist,
                backup_count: self.backup_count,
                backup_worst: self.backup_worst,
            },
        ))
    }
}

struct FlatReservoirSegment {
    first_source: usize,
    rows: usize,
    slots: MmapRaw,
    metadata: MmapRaw,
}

impl FlatReservoirSegment {
    fn new(first_source: usize, rows: usize, l_max: usize) -> AnnResult<Self> {
        let slot_count = checked_product(rows, l_max, "flat reservoir slot count")?;
        let slot_bytes = checked_product(slot_count, size_of::<Slot>(), "flat reservoir slots")?;
        let metadata_bytes = checked_product(
            rows,
            size_of::<FlatReservoirMeta>(),
            "flat reservoir metadata",
        )?;
        let slots = anonymous_raw(slot_bytes)?;
        let metadata = anonymous_raw(metadata_bytes)?;
        if (slots.as_ptr() as usize) % align_of::<Slot>() != 0
            || (metadata.as_ptr() as usize) % align_of::<FlatReservoirMeta>() != 0
        {
            return Err(AnnError::log_index_error(
                "flat resident reservoir mapping is insufficiently aligned".to_string(),
            ));
        }
        Ok(Self {
            first_source,
            rows,
            slots,
            metadata,
        })
    }

    fn exclusive_end(&self) -> usize {
        self.first_source + self.rows
    }

    fn local_index(&self, source: usize) -> AnnResult<usize> {
        let local = source.checked_sub(self.first_source).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "source {source} precedes flat reservoir segment {}",
                self.first_source
            ))
        })?;
        if local >= self.rows {
            return Err(AnnError::log_index_error(format!(
                "source {source} exceeds flat reservoir segment [{}, {})",
                self.first_source,
                self.exclusive_end()
            )));
        }
        Ok(local)
    }

    fn load_row(
        &self,
        source: usize,
        l_max: usize,
        scratch: &mut HashPruneReservoir,
    ) -> AnnResult<()> {
        let local = self.local_index(source)?;
        let meta_ptr = self.meta_ptr(local);
        // SAFETY: `local_index` proves the metadata element is in bounds. Mutable access to this
        // source is serialized by its deterministic stripe lock.
        let meta = unsafe { ptr::read(meta_ptr) };
        let (len, state) = meta.decode(l_max)?;
        let slots_ptr = self.slot_row_ptr(local, l_max);
        // SAFETY: the segment mapping contains `rows * l_max` aligned slots, and `len <= l_max`.
        // Only the initialized prefix described by metadata is exposed.
        let slots = unsafe { slice::from_raw_parts(slots_ptr.cast_const(), len) };
        scratch.load_runtime_row(slots, state)
    }

    fn store_row(
        &self,
        source: usize,
        l_max: usize,
        scratch: &HashPruneReservoir,
    ) -> AnnResult<()> {
        if scratch.slots().len() > l_max {
            return Err(AnnError::log_index_error(format!(
                "flat HashPrune row length {} exceeds capacity {l_max}",
                scratch.slots().len()
            )));
        }
        let local = self.local_index(source)?;
        let slots_ptr = self.slot_row_ptr(local, l_max);
        // SAFETY: the destination row has capacity `l_max`, the source prefix is no longer than
        // that capacity, and the stripe lock excludes access to this source while copying.
        unsafe {
            ptr::copy_nonoverlapping(scratch.slots().as_ptr(), slots_ptr, scratch.slots().len());
        }
        let meta = FlatReservoirMeta::from_reservoir(scratch)?;
        // SAFETY: `local_index` proves the metadata element is in bounds, and the source stripe
        // lock provides exclusive access to this row.
        unsafe {
            ptr::write(self.meta_ptr(local), meta);
        }
        Ok(())
    }

    fn slot_row_ptr(&self, local: usize, l_max: usize) -> *mut Slot {
        debug_assert!(local < self.rows);
        let slot_offset = local * l_max;
        // `MmapRaw` intentionally exposes interior mutability only through raw pointers.
        self.slots
            .as_mut_ptr()
            .cast::<Slot>()
            .wrapping_add(slot_offset)
    }

    fn meta_ptr(&self, local: usize) -> *mut FlatReservoirMeta {
        debug_assert!(local < self.rows);
        // `MmapRaw` intentionally exposes interior mutability only through raw pointers.
        self.metadata
            .as_mut_ptr()
            .cast::<FlatReservoirMeta>()
            .wrapping_add(local)
    }

    fn mapped_bytes(&self) -> usize {
        self.slots.len().saturating_add(self.metadata.len())
    }
}

pub(crate) struct FlatResidentReservoirStore {
    points: usize,
    l_max: usize,
    segment_rows: usize,
    segments: Vec<Option<FlatReservoirSegment>>,
    stripes: Vec<Mutex<HashPruneReservoir>>,
    mapped_bytes: usize,
    estimated_bytes: usize,
}

impl FlatResidentReservoirStore {
    pub(crate) fn new(points: usize, l_max: usize) -> AnnResult<Self> {
        Self::new_with_layout(
            points,
            l_max,
            DEFAULT_FLAT_RESERVOIR_SEGMENT_ROWS,
            DEFAULT_FLAT_RESERVOIR_STRIPES,
        )
    }

    fn new_with_layout(
        points: usize,
        l_max: usize,
        segment_rows: usize,
        stripe_count: usize,
    ) -> AnnResult<Self> {
        if l_max == 0 {
            return Err(AnnError::log_index_error(
                "flat resident reservoir requires l_max > 0".to_string(),
            ));
        }
        if segment_rows == 0 {
            return Err(AnnError::log_index_error(
                "flat resident reservoir requires segment_rows > 0".to_string(),
            ));
        }
        if stripe_count == 0 || !stripe_count.is_power_of_two() {
            return Err(AnnError::log_index_error(
                "flat resident reservoir stripe count must be a power of two".to_string(),
            ));
        }
        let estimated_bytes =
            Self::estimate_bytes_with_layout(points, l_max, segment_rows, stripe_count)?;
        let segment_count = points.div_ceil(segment_rows);
        let mut segments = Vec::with_capacity(segment_count);
        let mut mapped_bytes = 0usize;
        for segment_index in 0..segment_count {
            let first_source = segment_index.checked_mul(segment_rows).ok_or_else(|| {
                AnnError::log_index_error(
                    "flat resident reservoir segment offset overflow".to_string(),
                )
            })?;
            let rows = points.saturating_sub(first_source).min(segment_rows);
            let segment = FlatReservoirSegment::new(first_source, rows, l_max)?;
            mapped_bytes = mapped_bytes
                .checked_add(segment.mapped_bytes())
                .ok_or_else(|| {
                    AnnError::log_index_error(
                        "flat resident reservoir mapped-byte overflow".to_string(),
                    )
                })?;
            segments.push(Some(segment));
        }
        let stripes = (0..stripe_count)
            .map(|_| Mutex::new(HashPruneReservoir::new(l_max)))
            .collect();
        Ok(Self {
            points,
            l_max,
            segment_rows,
            segments,
            stripes,
            mapped_bytes,
            estimated_bytes,
        })
    }

    pub(crate) fn estimate_bytes(
        points: usize,
        l_max: usize,
        stripe_count: usize,
    ) -> AnnResult<usize> {
        Self::estimate_bytes_with_layout(
            points,
            l_max,
            DEFAULT_FLAT_RESERVOIR_SEGMENT_ROWS,
            stripe_count,
        )
    }

    fn estimate_bytes_with_layout(
        points: usize,
        l_max: usize,
        segment_rows: usize,
        stripe_count: usize,
    ) -> AnnResult<usize> {
        if segment_rows == 0 {
            return Err(AnnError::log_index_error(
                "flat resident reservoir requires segment_rows > 0".to_string(),
            ));
        }
        let total_slots =
            checked_product(points, l_max, "flat reservoir estimated total slot count")?;
        let _ = checked_product(
            total_slots,
            size_of::<Slot>(),
            "flat reservoir estimated total slots",
        )?;
        let _ = checked_product(
            points,
            size_of::<FlatReservoirMeta>(),
            "flat reservoir estimated total metadata",
        )?;
        let segment_count = points.div_ceil(segment_rows);
        let mut mapped_bytes = 0usize;
        for segment_index in 0..segment_count {
            let first_source = segment_index.checked_mul(segment_rows).ok_or_else(|| {
                AnnError::log_index_error(
                    "flat reservoir estimated segment offset overflow".to_string(),
                )
            })?;
            let rows = points.saturating_sub(first_source).min(segment_rows);
            let slot_count = checked_product(rows, l_max, "flat reservoir estimated slot count")?;
            let slot_bytes = checked_product(
                slot_count,
                size_of::<Slot>(),
                "flat reservoir estimated slots",
            )?;
            let metadata_bytes = checked_product(
                rows,
                size_of::<FlatReservoirMeta>(),
                "flat reservoir estimated metadata",
            )?;
            mapped_bytes = checked_sum(
                mapped_bytes,
                rounded_mapping_bytes(slot_bytes)?,
                "flat reservoir estimated mapped slots",
            )?;
            mapped_bytes = checked_sum(
                mapped_bytes,
                rounded_mapping_bytes(metadata_bytes)?,
                "flat reservoir estimated mapped metadata",
            )?;
        }
        let segment_descriptors = checked_product(
            segment_count,
            size_of::<Option<FlatReservoirSegment>>(),
            "flat reservoir estimated segment descriptors",
        )?;
        let scratch_slots = checked_product(
            stripe_count,
            l_max,
            "flat reservoir estimated scratch slots",
        )?;
        let scratch_slot_bytes = checked_product(
            scratch_slots,
            size_of::<Slot>(),
            "flat reservoir estimated scratch bytes",
        )?;
        let stripe_objects = checked_product(
            stripe_count,
            size_of::<Mutex<HashPruneReservoir>>(),
            "flat reservoir estimated stripe objects",
        )?;
        mapped_bytes
            .checked_add(segment_descriptors)
            .and_then(|bytes| bytes.checked_add(size_of::<Self>()))
            .and_then(|bytes| bytes.checked_add(scratch_slot_bytes))
            .and_then(|bytes| bytes.checked_add(stripe_objects))
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "flat resident reservoir byte estimate overflow".to_string(),
                )
            })
    }

    pub(crate) fn len(&self) -> usize {
        self.points
    }

    pub(crate) fn segment_rows(&self) -> usize {
        self.segment_rows
    }

    pub(crate) fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    pub(crate) fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    pub(crate) fn segment_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|segment| segment.is_some())
            .count()
    }

    pub(crate) fn stripe_count(&self) -> usize {
        self.stripes.len()
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        self.stripes.len().saturating_mul(
            size_of::<Mutex<HashPruneReservoir>>()
                .saturating_add(self.l_max.saturating_mul(size_of::<Slot>())),
        )
    }

    pub(crate) fn with_reservoir<T>(
        &self,
        source: usize,
        update: impl FnOnce(&mut HashPruneReservoir) -> AnnResult<T>,
    ) -> AnnResult<T> {
        self.with_reservoir_profiled(source, false, update)
            .map(|(value, _)| value)
    }

    pub(crate) fn with_reservoir_profiled<T>(
        &self,
        source: usize,
        measure_lock: bool,
        update: impl FnOnce(&mut HashPruneReservoir) -> AnnResult<T>,
    ) -> AnnResult<(T, Duration)> {
        let (segment, _) = self.segment_for_source(source)?;
        let stripe_index = source & (self.stripes.len() - 1);
        let lock_started = measure_lock.then(Instant::now);
        let mut scratch = self.stripes[stripe_index].lock();
        let lock_wall = lock_started.map_or(Duration::ZERO, |started| started.elapsed());
        if let Err(error) = segment.load_row(source, self.l_max, &mut scratch) {
            scratch.clear();
            return Err(error);
        }
        let update_result = update(&mut scratch);
        let store_result = segment.store_row(source, self.l_max, &scratch);
        scratch.clear();
        match (update_result, store_result) {
            (_, Err(error)) => Err(error),
            (Err(error), Ok(())) => Err(error),
            (Ok(value), Ok(())) => Ok((value, lock_wall)),
        }
    }

    pub(crate) fn finalize_row_with_logical_source(
        &self,
        source: usize,
        logical_source: usize,
        policy: HashPruneBackfillPolicy,
        profile: &mut HashPruneBackfillProfile,
    ) -> AnnResult<FinalizedHashPruneRow> {
        self.with_reservoir(source, |reservoir| {
            reservoir.copy_finalized_slots_with_backfill_and_clear(logical_source, policy, profile)
        })
    }

    pub(crate) fn release_segments_before(&mut self, next_source: usize) {
        for segment in &mut self.segments {
            if segment
                .as_ref()
                .is_some_and(|segment| segment.exclusive_end() <= next_source)
            {
                self.mapped_bytes = self.mapped_bytes.saturating_sub(
                    segment
                        .as_ref()
                        .map_or(0, FlatReservoirSegment::mapped_bytes),
                );
                *segment = None;
            }
        }
    }

    fn segment_for_source(&self, source: usize) -> AnnResult<(&FlatReservoirSegment, usize)> {
        if source >= self.points {
            return Err(AnnError::log_index_error(format!(
                "source {source} exceeds flat resident reservoir length {}",
                self.points
            )));
        }
        let segment_index = source / self.segment_rows;
        let segment = self
            .segments
            .get(segment_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "flat resident reservoir segment {segment_index} is unavailable"
                ))
            })?;
        Ok((segment, segment.local_index(source)?))
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        points: usize,
        l_max: usize,
        segment_rows: usize,
        stripe_count: usize,
    ) -> AnnResult<Self> {
        Self::new_with_layout(points, l_max, segment_rows, stripe_count)
    }

    #[cfg(test)]
    pub(crate) fn snapshot_slots(&self, source: usize) -> AnnResult<Vec<Slot>> {
        self.with_reservoir(source, |reservoir| Ok(reservoir.slots().to_vec()))
    }
}

fn anonymous_raw(bytes: usize) -> AnnResult<MmapRaw> {
    if bytes > isize::MAX as usize {
        return Err(AnnError::log_index_error(format!(
            "flat resident reservoir mapping exceeds isize::MAX: {bytes}"
        )));
    }
    let mmap = MmapMut::map_anon(bytes).map_err(|error| {
        AnnError::log_index_error(format!("failed to map flat resident reservoir: {error}"))
    })?;
    Ok(MmapRaw::from(mmap))
}

fn checked_product(left: usize, right: usize, label: &str) -> AnnResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| AnnError::log_index_error(format!("{label} overflow: {left} * {right}")))
}

fn checked_sum(left: usize, right: usize, label: &str) -> AnnResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| AnnError::log_index_error(format!("{label} overflow: {left} + {right}")))
}

fn rounded_mapping_bytes(bytes: usize) -> AnnResult<usize> {
    if bytes == 0 {
        return Ok(0);
    }
    bytes
        .checked_add(MMAP_PAGE_BYTES - 1)
        .map(|rounded| rounded / MMAP_PAGE_BYTES * MMAP_PAGE_BYTES)
        .ok_or_else(|| {
            AnnError::log_index_error(format!(
                "flat reservoir mapping-size rounding overflow: {bytes}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    fn slot_keys(slots: &[Slot]) -> Vec<(u32, u16, u32)> {
        slots
            .iter()
            .map(|slot| (slot.target, slot.hash, slot.dist.to_bits()))
            .collect()
    }

    #[test]
    fn flat_store_round_trips_rows_across_segment_boundary() {
        let store = FlatResidentReservoirStore::new_for_test(5, 4, 2, 2).unwrap();
        store
            .with_reservoir(1, |row| {
                row.insert(11, 1, 3.0);
                Ok(())
            })
            .unwrap();
        store
            .with_reservoir(2, |row| {
                row.insert(22, 2, 2.0);
                Ok(())
            })
            .unwrap();

        let left = store.snapshot_slots(1).unwrap();
        let right = store.snapshot_slots(2).unwrap();
        assert_eq!((left[0].target, left[0].hash), (11, 1));
        assert_eq!((right[0].target, right[0].hash), (22, 2));
        assert_eq!(store.segment_count(), 3);
        assert_eq!(store.stripe_count(), 2);
    }

    #[test]
    fn flat_store_rejects_invalid_layout_and_source() {
        assert!(FlatResidentReservoirStore::new_for_test(1, 0, 1, 1).is_err());
        assert!(FlatResidentReservoirStore::new_for_test(1, 1, 0, 1).is_err());
        assert!(FlatResidentReservoirStore::new_for_test(1, 1, 1, 3).is_err());

        let store = FlatResidentReservoirStore::new_for_test(1, 1, 1, 1).unwrap();
        assert!(store.snapshot_slots(1).is_err());
        assert!(FlatResidentReservoirStore::estimate_bytes(usize::MAX, 2, 1).is_err());
    }

    #[test]
    fn flat_store_sources_sharing_a_stripe_remain_isolated() {
        let store = FlatResidentReservoirStore::new_for_test(4, 8, 2, 1).unwrap();
        thread::scope(|scope| {
            scope.spawn(|| {
                for target in 100..132 {
                    store
                        .with_reservoir(1, |row| {
                            row.insert(target, (target % 17) as u16, target as f32);
                            Ok(())
                        })
                        .unwrap();
                }
            });
            scope.spawn(|| {
                for target in 200..232 {
                    store
                        .with_reservoir(3, |row| {
                            row.insert(target, (target % 19) as u16, target as f32);
                            Ok(())
                        })
                        .unwrap();
                }
            });
        });

        let left = store.snapshot_slots(1).unwrap();
        let right = store.snapshot_slots(3).unwrap();
        assert!(left.iter().all(|slot| slot.target < 200));
        assert!(right.iter().all(|slot| slot.target >= 200));
        assert_ne!(slot_keys(&left), slot_keys(&right));
    }
}
