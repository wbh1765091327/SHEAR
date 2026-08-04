use std::collections::BTreeMap;
use std::mem::size_of;

pub(crate) const PERSISTENT_ASYNC_D2H_SLOT_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PersistentAsyncD2hEligibility {
    pub(crate) requested: bool,
    pub(crate) sq_wmma_fused: bool,
    pub(crate) sidecar_preload: bool,
    pub(crate) ordered_scatter: bool,
    pub(crate) diagnostics_enabled: bool,
    pub(crate) max_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PersistentAsyncD2hFallbackReason {
    Disabled,
    UnsupportedBackend,
    SidecarPreloadUnavailable,
    OrderedScatterUnavailable,
    DiagnosticsEnabled,
    NoTasks,
}

impl PersistentAsyncD2hFallbackReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::UnsupportedBackend => "unsupported_backend",
            Self::SidecarPreloadUnavailable => "sidecar_preload_unavailable",
            Self::OrderedScatterUnavailable => "ordered_scatter_unavailable",
            Self::DiagnosticsEnabled => "diagnostics_enabled",
            Self::NoTasks => "no_tasks",
        }
    }
}

pub(crate) fn persistent_async_d2h_fallback_reason(
    eligibility: PersistentAsyncD2hEligibility,
) -> Option<PersistentAsyncD2hFallbackReason> {
    if !eligibility.requested {
        return Some(PersistentAsyncD2hFallbackReason::Disabled);
    }
    if !eligibility.sq_wmma_fused {
        return Some(PersistentAsyncD2hFallbackReason::UnsupportedBackend);
    }
    if !eligibility.sidecar_preload {
        return Some(PersistentAsyncD2hFallbackReason::SidecarPreloadUnavailable);
    }
    if !eligibility.ordered_scatter {
        return Some(PersistentAsyncD2hFallbackReason::OrderedScatterUnavailable);
    }
    if eligibility.diagnostics_enabled {
        return Some(PersistentAsyncD2hFallbackReason::DiagnosticsEnabled);
    }
    if eligibility.max_rows == 0 {
        return Some(PersistentAsyncD2hFallbackReason::NoTasks);
    }
    None
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PersistentOutputCapacity {
    pub(crate) max_leaf_rows: usize,
    pub(crate) max_leaf_count: usize,
    pub(crate) k: usize,
    pub(crate) witness_enabled: bool,
    pub(crate) counter_words: usize,
    pub(crate) wmma_diagnostic_words: usize,
}

impl PersistentOutputCapacity {
    pub(crate) fn device_slot_bytes(self) -> usize {
        let row_bytes = self.max_leaf_rows.saturating_mul(size_of::<u32>());
        let leaf_bytes = self
            .max_leaf_count
            .saturating_mul(size_of::<u64>())
            .saturating_mul(2);
        let topk_elements = self.max_leaf_rows.saturating_mul(self.k);
        let topk_bytes = topk_elements.saturating_mul(size_of::<u32>() + size_of::<f32>());
        let witness_bytes = usize::from(self.witness_enabled).saturating_mul(topk_bytes);
        let counter_bytes = self
            .counter_words
            .saturating_add(self.wmma_diagnostic_words)
            .saturating_mul(size_of::<u64>());
        row_bytes
            .saturating_add(leaf_bytes)
            .saturating_add(topk_bytes)
            .saturating_add(witness_bytes)
            .saturating_add(counter_bytes)
    }

    pub(crate) fn pinned_host_slot_bytes(self) -> usize {
        self.device_slot_bytes()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceSlotState {
    Free,
    Computing(usize),
    D2hCopying(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostSlotState {
    Free,
    D2hCopying(usize),
    HostReady(usize),
    CpuScatter(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PersistentSlotLease {
    pub(crate) sequence: usize,
    pub(crate) device_slot: usize,
    pub(crate) host_slot: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PersistentSlotError {
    StaleLease,
}

#[derive(Clone, Debug)]
pub(crate) struct PersistentAsyncD2hSlots {
    device: [DeviceSlotState; PERSISTENT_ASYNC_D2H_SLOT_COUNT],
    host: [HostSlotState; PERSISTENT_ASYNC_D2H_SLOT_COUNT],
}

impl Default for PersistentAsyncD2hSlots {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentAsyncD2hSlots {
    pub(crate) fn new() -> Self {
        Self {
            device: [DeviceSlotState::Free; PERSISTENT_ASYNC_D2H_SLOT_COUNT],
            host: [HostSlotState::Free; PERSISTENT_ASYNC_D2H_SLOT_COUNT],
        }
    }

    pub(crate) fn try_acquire(&mut self, sequence: usize) -> Option<PersistentSlotLease> {
        let device_slot = self
            .device
            .iter()
            .position(|state| *state == DeviceSlotState::Free)?;
        let host_slot = self
            .host
            .iter()
            .position(|state| *state == HostSlotState::Free)?;
        self.device[device_slot] = DeviceSlotState::Computing(sequence);
        self.host[host_slot] = HostSlotState::D2hCopying(sequence);
        Some(PersistentSlotLease {
            sequence,
            device_slot,
            host_slot,
        })
    }

    pub(crate) fn begin_d2h(
        &mut self,
        lease: PersistentSlotLease,
    ) -> Result<(), PersistentSlotError> {
        if self.device.get(lease.device_slot) != Some(&DeviceSlotState::Computing(lease.sequence))
            || self.host.get(lease.host_slot) != Some(&HostSlotState::D2hCopying(lease.sequence))
        {
            return Err(PersistentSlotError::StaleLease);
        }
        self.device[lease.device_slot] = DeviceSlotState::D2hCopying(lease.sequence);
        Ok(())
    }

    pub(crate) fn release_device_after_d2h(
        &mut self,
        lease: PersistentSlotLease,
    ) -> Result<(), PersistentSlotError> {
        if self.device.get(lease.device_slot) != Some(&DeviceSlotState::D2hCopying(lease.sequence))
            || self.host.get(lease.host_slot) != Some(&HostSlotState::D2hCopying(lease.sequence))
        {
            return Err(PersistentSlotError::StaleLease);
        }
        self.device[lease.device_slot] = DeviceSlotState::Free;
        self.host[lease.host_slot] = HostSlotState::HostReady(lease.sequence);
        Ok(())
    }

    pub(crate) fn begin_scatter(
        &mut self,
        lease: PersistentSlotLease,
    ) -> Result<(), PersistentSlotError> {
        if self.host.get(lease.host_slot) != Some(&HostSlotState::HostReady(lease.sequence)) {
            return Err(PersistentSlotError::StaleLease);
        }
        self.host[lease.host_slot] = HostSlotState::CpuScatter(lease.sequence);
        Ok(())
    }

    pub(crate) fn finish_scatter(
        &mut self,
        lease: PersistentSlotLease,
    ) -> Result<(), PersistentSlotError> {
        if self.host.get(lease.host_slot) != Some(&HostSlotState::CpuScatter(lease.sequence)) {
            return Err(PersistentSlotError::StaleLease);
        }
        self.host[lease.host_slot] = HostSlotState::Free;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SequencedCompletionError {
    AlreadyPublished(usize),
    Duplicate(usize),
    OutsideAcceptedRange(usize),
}

pub(crate) struct SequencedCompletionBuffer<T> {
    next_sequence: usize,
    accepted_sequences: usize,
    pending: BTreeMap<usize, T>,
}

impl<T> SequencedCompletionBuffer<T> {
    pub(crate) fn new(accepted_sequences: usize) -> Self {
        Self {
            next_sequence: 0,
            accepted_sequences,
            pending: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(
        &mut self,
        sequence: usize,
        value: T,
    ) -> Result<Vec<T>, SequencedCompletionError> {
        if sequence >= self.accepted_sequences {
            return Err(SequencedCompletionError::OutsideAcceptedRange(sequence));
        }
        if sequence < self.next_sequence {
            return Err(SequencedCompletionError::AlreadyPublished(sequence));
        }
        if self.pending.contains_key(&sequence) {
            return Err(SequencedCompletionError::Duplicate(sequence));
        }
        self.pending.insert(sequence, value);
        let mut ready = Vec::new();
        while let Some(value) = self.pending.remove(&self.next_sequence) {
            ready.push(value);
            self.next_sequence = self.next_sequence.saturating_add(1);
        }
        Ok(ready)
    }

    pub(crate) fn is_drained(&self) -> bool {
        self.next_sequence == self.accepted_sequences && self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PersistentAsyncD2hEligibility, PersistentAsyncD2hFallbackReason, PersistentAsyncD2hSlots,
        PersistentOutputCapacity, PersistentSlotError, SequencedCompletionBuffer,
        persistent_async_d2h_fallback_reason,
    };

    #[test]
    fn persistent_async_d2h_capacity_accounts_for_every_output_array() {
        let capacity = PersistentOutputCapacity {
            max_leaf_rows: 10,
            max_leaf_count: 3,
            k: 4,
            witness_enabled: true,
            counter_words: 5,
            wmma_diagnostic_words: 7,
        };

        let row_bytes = 10 * size_of::<u32>();
        let leaf_bytes = 3 * size_of::<u64>() * 2;
        let topk_bytes = 10 * 4 * (size_of::<u32>() + size_of::<f32>());
        let witness_bytes = 10 * 4 * (size_of::<u32>() + size_of::<f32>());
        let counter_bytes = (5 + 7) * size_of::<u64>();
        let expected = row_bytes + leaf_bytes + topk_bytes + witness_bytes + counter_bytes;

        assert_eq!(capacity.device_slot_bytes(), expected);
        assert_eq!(capacity.pinned_host_slot_bytes(), expected);
    }

    #[test]
    fn persistent_async_d2h_device_slot_waits_for_copy_completion() {
        let mut slots = PersistentAsyncD2hSlots::new();
        let lease = slots.try_acquire(0).expect("first lease");

        slots.begin_d2h(lease).expect("compute-to-copy transition");
        assert_eq!(
            slots.release_device_after_d2h(lease),
            Ok(()),
            "D2H completion releases the device slot"
        );

        assert_eq!(
            slots.begin_d2h(lease),
            Err(PersistentSlotError::StaleLease),
            "a released device lease cannot be reused"
        );
    }

    #[test]
    fn persistent_async_d2h_host_slot_waits_for_scatter_completion() {
        let mut slots = PersistentAsyncD2hSlots::new();
        let first = slots.try_acquire(0).expect("first lease");
        slots.begin_d2h(first).expect("begin D2H");
        slots.release_device_after_d2h(first).expect("complete D2H");
        slots.begin_scatter(first).expect("begin scatter");

        let second = slots.try_acquire(1).expect("second lease");
        slots.begin_d2h(second).expect("second begin D2H");
        slots
            .release_device_after_d2h(second)
            .expect("second complete D2H");

        assert!(
            slots.try_acquire(2).is_none(),
            "both host slots remain leased until CPU scatter finishes"
        );
        slots.finish_scatter(first).expect("finish first scatter");
        assert!(slots.try_acquire(2).is_some());
    }

    #[test]
    fn persistent_async_d2h_two_occupied_device_slots_apply_backpressure() {
        let mut slots = PersistentAsyncD2hSlots::new();
        let first = slots.try_acquire(0).expect("first lease");
        let second = slots.try_acquire(1).expect("second lease");

        assert!(slots.try_acquire(2).is_none());
        slots.begin_d2h(first).expect("first begin D2H");
        slots
            .release_device_after_d2h(first)
            .expect("first complete D2H");
        slots.begin_scatter(first).expect("first begin scatter");
        slots.finish_scatter(first).expect("first finish scatter");
        assert!(slots.try_acquire(2).is_some());

        slots.begin_d2h(second).expect("second begin D2H");
    }

    #[test]
    fn persistent_async_d2h_completions_publish_in_sequence() {
        let mut completions = SequencedCompletionBuffer::new(4);

        assert!(completions.insert(1, "one").expect("insert one").is_empty());
        assert_eq!(
            completions.insert(0, "zero").expect("insert zero"),
            vec!["zero", "one"]
        );
        assert!(
            completions
                .insert(3, "three")
                .expect("insert three")
                .is_empty()
        );
        assert_eq!(
            completions.insert(2, "two").expect("insert two"),
            vec!["two", "three"]
        );
        assert!(completions.is_drained());
    }

    #[test]
    fn persistent_async_d2h_duplicate_completion_keeps_original_value() {
        let mut completions = SequencedCompletionBuffer::new(2);

        assert!(
            completions
                .insert(1, "original")
                .expect("insert original")
                .is_empty()
        );
        assert_eq!(
            completions.insert(1, "replacement"),
            Err(super::SequencedCompletionError::Duplicate(1))
        );
        assert_eq!(
            completions.insert(0, "zero").expect("insert zero"),
            vec!["zero", "original"]
        );
        assert!(completions.is_drained());
    }

    #[test]
    fn persistent_async_d2h_admits_only_the_production_fused_path() {
        let eligible = PersistentAsyncD2hEligibility {
            requested: true,
            sq_wmma_fused: true,
            sidecar_preload: true,
            ordered_scatter: true,
            diagnostics_enabled: false,
            max_rows: 1024,
        };
        assert_eq!(persistent_async_d2h_fallback_reason(eligible), None);

        for (rejected, expected) in [
            (
                PersistentAsyncD2hEligibility {
                    requested: false,
                    ..eligible
                },
                PersistentAsyncD2hFallbackReason::Disabled,
            ),
            (
                PersistentAsyncD2hEligibility {
                    sq_wmma_fused: false,
                    ..eligible
                },
                PersistentAsyncD2hFallbackReason::UnsupportedBackend,
            ),
            (
                PersistentAsyncD2hEligibility {
                    sidecar_preload: false,
                    ..eligible
                },
                PersistentAsyncD2hFallbackReason::SidecarPreloadUnavailable,
            ),
            (
                PersistentAsyncD2hEligibility {
                    ordered_scatter: false,
                    ..eligible
                },
                PersistentAsyncD2hFallbackReason::OrderedScatterUnavailable,
            ),
            (
                PersistentAsyncD2hEligibility {
                    diagnostics_enabled: true,
                    ..eligible
                },
                PersistentAsyncD2hFallbackReason::DiagnosticsEnabled,
            ),
            (
                PersistentAsyncD2hEligibility {
                    max_rows: 0,
                    ..eligible
                },
                PersistentAsyncD2hFallbackReason::NoTasks,
            ),
        ] {
            assert_eq!(
                persistent_async_d2h_fallback_reason(rejected),
                Some(expected)
            );
        }
    }
}
