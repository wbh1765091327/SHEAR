use std::time::Duration;

use super::params::{PiPNNHashPruneEvidenceMode, PiPNNParams};
use crate::common::{AnnError, AnnResult};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum HashPruneEvidenceClass {
    Reverse = 0,
    ReverseRank0 = 1,
    Direct = 2,
    Bidirectional = 3,
    DirectRank0 = 4,
    BidirectionalDirectRank0 = 5,
}

impl HashPruneEvidenceClass {
    pub(crate) const ALL: [Self; 6] = [
        Self::Reverse,
        Self::ReverseRank0,
        Self::Direct,
        Self::Bidirectional,
        Self::DirectRank0,
        Self::BidirectionalDirectRank0,
    ];

    #[inline]
    pub(crate) fn from_ranks(
        direct_rank: Option<usize>,
        reverse_rank: Option<usize>,
    ) -> Option<Self> {
        match (direct_rank, reverse_rank) {
            (None, None) => None,
            (None, Some(0)) => Some(Self::ReverseRank0),
            (None, Some(_)) => Some(Self::Reverse),
            (Some(0), None) => Some(Self::DirectRank0),
            (Some(0), Some(_)) => Some(Self::BidirectionalDirectRank0),
            (Some(_), None) => Some(Self::Direct),
            (Some(_), Some(_)) => Some(Self::Bidirectional),
        }
    }

    #[inline]
    pub(crate) fn qualifies_phase1(self) -> bool {
        matches!(
            self,
            Self::Bidirectional | Self::DirectRank0 | Self::BidirectionalDirectRank0
        )
    }

    #[inline]
    pub(crate) fn index(self) -> usize {
        self as usize
    }

    #[inline]
    fn from_u8(value: u8) -> Option<Self> {
        Self::ALL.get(value as usize).copied()
    }
}

const META_MANDATORY: u16 = 1 << 0;
const META_CLASS_SHIFT: u32 = 1;
const META_CLASS_MASK: u16 = 0b111 << META_CLASS_SHIFT;
const META_EVIDENCE_VALID: u16 = 1 << 4;
const META_OVERFLOW: u16 = 1 << 5;
const META_BACKFILL_COUNT_SHIFT: u32 = 6;
const META_BACKFILL_COUNT_MASK: u16 = 0b1111 << META_BACKFILL_COUNT_SHIFT;
const META_BACKFILL_HIT_LAYER_SHIFT: u32 = 10;
const META_BACKFILL_HIT_LAYER_MASK: u16 = 0b11 << META_BACKFILL_HIT_LAYER_SHIFT;
const META_RESERVED_MASK: u16 = !0b11_1111;
const META_STATE_ALLOWED_MASK: u16 = META_MANDATORY
    | META_CLASS_MASK
    | META_EVIDENCE_VALID
    | META_OVERFLOW
    | META_BACKFILL_COUNT_MASK
    | META_BACKFILL_HIT_LAYER_MASK;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HashPruneMeta(u16);

impl HashPruneMeta {
    #[inline]
    pub(crate) const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub(crate) const fn mandatory() -> Self {
        Self(META_MANDATORY)
    }

    #[inline]
    pub(crate) const fn with_evidence(class: HashPruneEvidenceClass) -> Self {
        Self(META_EVIDENCE_VALID | ((class as u16) << META_CLASS_SHIFT))
    }

    #[inline]
    pub(crate) fn evidence(self) -> Option<HashPruneEvidenceClass> {
        if self.0 & META_EVIDENCE_VALID == 0 {
            return None;
        }
        let class = ((self.0 & META_CLASS_MASK) >> META_CLASS_SHIFT) as u8;
        HashPruneEvidenceClass::from_u8(class)
    }

    #[inline]
    pub(crate) const fn is_mandatory(self) -> bool {
        self.0 & META_MANDATORY != 0
    }

    pub(crate) fn merge_candidate(self, other: Self) -> AnnResult<Self> {
        self.validate_candidate("left candidate metadata")?;
        other.validate_candidate("right candidate metadata")?;

        Ok(self.merge_prevalidated_candidate(other))
    }

    #[inline]
    pub(crate) fn merge_prevalidated_candidate(self, other: Self) -> Self {
        let mut merged = if self.is_mandatory() || other.is_mandatory() {
            Self::mandatory()
        } else {
            Self::empty()
        };
        if let Some(class) = match (self.evidence(), other.evidence()) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(class), None) | (None, Some(class)) => Some(class),
            (None, None) => None,
        } {
            merged.0 |= Self::with_evidence(class).0;
        }
        merged
    }

    #[inline]
    pub(crate) const fn mark_overflow(mut self) -> Self {
        self.0 |= META_OVERFLOW;
        self
    }

    #[inline]
    pub(crate) const fn clear_overflow(mut self) -> Self {
        self.0 &= !META_OVERFLOW;
        self
    }

    #[inline]
    pub(crate) const fn is_overflow(self) -> bool {
        self.0 & META_OVERFLOW != 0
    }

    pub(crate) fn with_backfill_count(mut self, count: usize) -> AnnResult<Self> {
        if !(1..=8).contains(&count) {
            return Err(AnnError::log_index_error(format!(
                "HashPrune internal backup count must be in 1..=8, got {count}"
            )));
        }
        self.0 &= !META_BACKFILL_COUNT_MASK;
        self.0 |= (count as u16) << META_BACKFILL_COUNT_SHIFT;
        Ok(self)
    }

    pub(crate) fn backfill_count(self) -> AnnResult<usize> {
        let count = ((self.0 & META_BACKFILL_COUNT_MASK) >> META_BACKFILL_COUNT_SHIFT) as usize;
        if count > 8 {
            return Err(AnnError::log_index_error(format!(
                "HashPrune internal backup count exceeds 8: {count}"
            )));
        }
        Ok(count)
    }

    #[inline]
    pub(crate) const fn clear_backfill_count(mut self) -> Self {
        self.0 &= !META_BACKFILL_COUNT_MASK;
        self
    }

    pub(crate) fn with_backfill_hit_layer(mut self, layer: usize) -> AnnResult<Self> {
        if !(1..=3).contains(&layer) {
            return Err(AnnError::log_index_error(format!(
                "HashPrune internal backup hit layer must be in 1..=3, got {layer}"
            )));
        }
        self.0 &= !META_BACKFILL_HIT_LAYER_MASK;
        self.0 |= (layer as u16) << META_BACKFILL_HIT_LAYER_SHIFT;
        Ok(self)
    }

    #[inline]
    pub(crate) const fn backfill_hit_layer(self) -> usize {
        ((self.0 & META_BACKFILL_HIT_LAYER_MASK) >> META_BACKFILL_HIT_LAYER_SHIFT) as usize
    }

    #[inline]
    pub(crate) const fn clear_backfill_hit_layer(mut self) -> Self {
        self.0 &= !META_BACKFILL_HIT_LAYER_MASK;
        self
    }

    pub(crate) fn to_spill_flags(self) -> AnnResult<u16> {
        self.validate_candidate("spill candidate metadata")?;
        Ok(self.0)
    }

    pub(crate) fn from_spill_flags(flags: u16) -> AnnResult<Self> {
        let meta = Self(flags);
        meta.validate_candidate("serialized spill candidate metadata")?;
        Ok(meta)
    }

    pub(crate) fn to_state_flags(self) -> AnnResult<u16> {
        self.validate_state("reservoir state metadata")?;
        Ok(self.0)
    }

    pub(crate) fn from_state_flags(flags: u16) -> AnnResult<Self> {
        let meta = Self(flags);
        meta.validate_state("serialized reservoir state metadata")?;
        Ok(meta)
    }

    fn validate_state(self, context: &str) -> AnnResult<()> {
        if self.0 & !META_STATE_ALLOWED_MASK != 0 {
            return Err(AnnError::log_index_error(format!(
                "{context} contains reserved bits: flags={:#06x}",
                self.0
            )));
        }

        let class_bits = self.0 & META_CLASS_MASK;
        let evidence_valid = self.0 & META_EVIDENCE_VALID != 0;
        if (!evidence_valid && class_bits != 0) || (evidence_valid && self.evidence().is_none()) {
            return Err(AnnError::log_index_error(format!(
                "{context} contains invalid evidence bits: flags={:#06x}",
                self.0
            )));
        }
        self.backfill_count()?;
        Ok(())
    }

    fn validate_candidate(self, context: &str) -> AnnResult<()> {
        if self.0 & META_RESERVED_MASK != 0 {
            return Err(AnnError::log_index_error(format!(
                "{context} contains reserved bits: flags={:#06x}",
                self.0
            )));
        }
        if self.is_overflow() {
            return Err(AnnError::log_index_error(format!(
                "{context} contains the internal reservoir overflow bit"
            )));
        }

        let class_bits = self.0 & META_CLASS_MASK;
        let evidence_valid = self.0 & META_EVIDENCE_VALID != 0;
        if !evidence_valid && class_bits != 0 {
            return Err(AnnError::log_index_error(format!(
                "{context} contains evidence class bits without the evidence-valid bit"
            )));
        }
        if evidence_valid && self.evidence().is_none() {
            return Err(AnnError::log_index_error(format!(
                "{context} contains an unknown evidence class: flags={:#06x}",
                self.0
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HashPruneEvidencePolicy {
    mode: PiPNNHashPruneEvidenceMode,
    overflow_budget: usize,
}

impl HashPruneEvidencePolicy {
    #[inline]
    pub(crate) const fn off() -> Self {
        Self {
            mode: PiPNNHashPruneEvidenceMode::Off,
            overflow_budget: 0,
        }
    }

    #[inline]
    pub(crate) const fn report() -> Self {
        Self {
            mode: PiPNNHashPruneEvidenceMode::Report,
            overflow_budget: 0,
        }
    }

    #[inline]
    pub(crate) const fn enforce_one() -> Self {
        Self {
            mode: PiPNNHashPruneEvidenceMode::Enforce,
            overflow_budget: 1,
        }
    }

    pub(crate) fn from_params(params: &PiPNNParams) -> AnnResult<Self> {
        params.validate_hashprune_evidence_config()?;
        Ok(Self {
            mode: params.hashprune_evidence_mode,
            overflow_budget: params.hashprune_evidence_overflow_budget,
        })
    }

    #[inline]
    pub(crate) const fn mode(self) -> PiPNNHashPruneEvidenceMode {
        self.mode
    }

    #[inline]
    pub(crate) const fn overflow_budget(self) -> usize {
        self.overflow_budget
    }

    #[inline]
    pub(crate) const fn is_off(self) -> bool {
        matches!(self.mode, PiPNNHashPruneEvidenceMode::Off)
    }

    #[inline]
    pub(crate) const fn annotates(self) -> bool {
        !self.is_off()
    }

    #[inline]
    pub(crate) const fn enforces(self) -> bool {
        matches!(self.mode, PiPNNHashPruneEvidenceMode::Enforce)
    }
}

impl Default for HashPruneEvidencePolicy {
    fn default() -> Self {
        Self::off()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HashPruneEvidenceProfile {
    pub(crate) annotation_wall: Duration,
    pub(crate) coalesce_sort_wall: Duration,
    pub(crate) coalesce_merge_wall: Duration,
    pub(crate) coalesce_wall: Duration,
    pub(crate) reciprocal_checks: usize,
    pub(crate) reciprocal_hits: usize,
    pub(crate) support_once_candidates: usize,
    pub(crate) support_twice_candidates: usize,
    pub(crate) class_counts: [usize; 6],
    pub(crate) candidates_before_coalesce: usize,
    pub(crate) candidates_after_coalesce: usize,
    pub(crate) duplicate_records_removed: usize,
    pub(crate) unannotated_candidates: usize,
    pub(crate) qualified_collision_losers: usize,
    pub(crate) overflow_inserts: usize,
    pub(crate) overflow_replacements: usize,
    pub(crate) overflow_rejects: usize,
    pub(crate) overflow_evictions_by_new_hash: usize,
    pub(crate) rows_with_overflow: usize,
    pub(crate) spill_records_avoided: usize,
    pub(crate) spill_bytes_avoided: usize,
}

impl HashPruneEvidenceProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        self.annotation_wall += other.annotation_wall;
        self.coalesce_sort_wall += other.coalesce_sort_wall;
        self.coalesce_merge_wall += other.coalesce_merge_wall;
        self.coalesce_wall += other.coalesce_wall;
        self.reciprocal_checks += other.reciprocal_checks;
        self.reciprocal_hits += other.reciprocal_hits;
        self.support_once_candidates += other.support_once_candidates;
        self.support_twice_candidates += other.support_twice_candidates;
        for (left, right) in self.class_counts.iter_mut().zip(other.class_counts) {
            *left += right;
        }
        self.candidates_before_coalesce += other.candidates_before_coalesce;
        self.candidates_after_coalesce += other.candidates_after_coalesce;
        self.duplicate_records_removed += other.duplicate_records_removed;
        self.unannotated_candidates += other.unannotated_candidates;
        self.qualified_collision_losers += other.qualified_collision_losers;
        self.overflow_inserts += other.overflow_inserts;
        self.overflow_replacements += other.overflow_replacements;
        self.overflow_rejects += other.overflow_rejects;
        self.overflow_evictions_by_new_hash += other.overflow_evictions_by_new_hash;
        self.rows_with_overflow += other.rows_with_overflow;
        self.spill_records_avoided += other.spill_records_avoided;
        self.spill_bytes_avoided += other.spill_bytes_avoided;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn hashprune_evidence_class_from_ranks_covers_all_six_rows() {
        let cases = [
            (None, Some(1), HashPruneEvidenceClass::Reverse),
            (None, Some(0), HashPruneEvidenceClass::ReverseRank0),
            (Some(1), None, HashPruneEvidenceClass::Direct),
            (Some(1), Some(0), HashPruneEvidenceClass::Bidirectional),
            (Some(0), None, HashPruneEvidenceClass::DirectRank0),
            (
                Some(0),
                Some(1),
                HashPruneEvidenceClass::BidirectionalDirectRank0,
            ),
        ];

        for (direct_rank, reverse_rank, expected) in cases {
            assert_eq!(
                HashPruneEvidenceClass::from_ranks(direct_rank, reverse_rank),
                Some(expected)
            );
        }
        assert_eq!(HashPruneEvidenceClass::from_ranks(None, None), None);
    }

    #[test]
    fn hashprune_evidence_class_phase1_qualification_matches_policy() {
        for class in [
            HashPruneEvidenceClass::Reverse,
            HashPruneEvidenceClass::ReverseRank0,
            HashPruneEvidenceClass::Direct,
        ] {
            assert!(!class.qualifies_phase1());
        }
        for class in [
            HashPruneEvidenceClass::Bidirectional,
            HashPruneEvidenceClass::DirectRank0,
            HashPruneEvidenceClass::BidirectionalDirectRank0,
        ] {
            assert!(class.qualifies_phase1());
        }
    }

    #[test]
    fn hashprune_evidence_meta_distinguishes_reverse_from_missing() {
        let missing = HashPruneMeta::empty();
        let reverse = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Reverse);

        assert_eq!(missing.evidence(), None);
        assert_eq!(reverse.evidence(), Some(HashPruneEvidenceClass::Reverse));
        assert_ne!(missing, reverse);
    }

    #[test]
    fn hashprune_evidence_meta_merges_mandatory_with_numeric_class_max() {
        let direct = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let reverse_rank0 = HashPruneMeta::with_evidence(HashPruneEvidenceClass::ReverseRank0);
        let merged = direct
            .merge_candidate(reverse_rank0)
            .unwrap()
            .merge_candidate(HashPruneMeta::mandatory())
            .unwrap();
        assert_eq!(merged.evidence(), Some(HashPruneEvidenceClass::Direct));
        assert!(merged.is_mandatory());

        let bidirectional = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional);
        let direct_rank0 = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        assert_eq!(
            bidirectional
                .merge_candidate(direct_rank0)
                .unwrap()
                .evidence(),
            Some(HashPruneEvidenceClass::DirectRank0)
        );
    }

    #[test]
    fn hashprune_evidence_meta_spill_round_trip_and_rejections() {
        for class in HashPruneEvidenceClass::ALL {
            let meta = HashPruneMeta::mandatory()
                .merge_candidate(HashPruneMeta::with_evidence(class))
                .unwrap();
            let flags = meta.to_spill_flags().unwrap();
            assert_eq!(HashPruneMeta::from_spill_flags(flags).unwrap(), meta);
        }

        assert!(HashPruneMeta::from_spill_flags(META_RESERVED_MASK & (1 << 6)).is_err());
        assert!(HashPruneMeta::from_spill_flags(META_OVERFLOW).is_err());
        assert!(
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct)
                .mark_overflow()
                .to_spill_flags()
                .is_err()
        );
    }

    #[test]
    fn reservoir_state_flags_round_trip_internal_bits() {
        let meta = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0)
            .mark_overflow()
            .with_backfill_count(3)
            .unwrap()
            .with_backfill_hit_layer(2)
            .unwrap();
        let flags = meta.to_state_flags().unwrap();
        assert_eq!(HashPruneMeta::from_state_flags(flags).unwrap(), meta);
        assert!(meta.to_spill_flags().is_err());
    }

    #[test]
    fn reservoir_state_flags_reject_unknown_high_bits() {
        assert!(HashPruneMeta::from_state_flags(1 << 15).is_err());
    }

    #[test]
    fn hashprune_evidence_meta_overflow_marker_is_internal_and_reversible() {
        let meta = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        let overflow = meta.mark_overflow();
        assert!(overflow.is_overflow());
        assert_eq!(overflow.clear_overflow(), meta);
    }

    #[test]
    fn hashprune_backfill_count_marker_is_internal_reversible_and_bounded() {
        let base = HashPruneMeta::empty();
        for count in 1..=8 {
            let marked = base.with_backfill_count(count).unwrap();
            assert_eq!(marked.backfill_count().unwrap(), count);
            assert_eq!(marked.clear_backfill_count(), base);
            assert!(marked.to_spill_flags().is_err());
        }
        assert!(base.with_backfill_count(0).is_err());
        assert!(base.with_backfill_count(9).is_err());

        let malformed = HashPruneMeta(9 << META_BACKFILL_COUNT_SHIFT);
        assert!(malformed.backfill_count().is_err());
    }

    #[test]
    fn hashprune_backfill_hit_layer_marker_is_internal_and_reversible() {
        let base = HashPruneMeta::empty();
        for layer in 1..=3 {
            let marked = base.with_backfill_hit_layer(layer).unwrap();
            assert_eq!(marked.backfill_hit_layer(), layer);
            assert_eq!(marked.clear_backfill_hit_layer(), base);
            assert!(marked.to_spill_flags().is_err());
        }
        assert!(base.with_backfill_hit_layer(0).is_err());
        assert!(base.with_backfill_hit_layer(4).is_err());
    }

    #[test]
    fn hashprune_evidence_profile_merges_all_counters_additively() {
        let mut left = HashPruneEvidenceProfile {
            annotation_wall: Duration::from_millis(3),
            coalesce_sort_wall: Duration::from_millis(2),
            coalesce_merge_wall: Duration::from_millis(3),
            coalesce_wall: Duration::from_millis(5),
            reciprocal_checks: 7,
            reciprocal_hits: 2,
            support_once_candidates: 11,
            support_twice_candidates: 13,
            class_counts: [1, 2, 3, 4, 5, 6],
            candidates_before_coalesce: 17,
            candidates_after_coalesce: 19,
            duplicate_records_removed: 23,
            unannotated_candidates: 29,
            qualified_collision_losers: 31,
            overflow_inserts: 37,
            overflow_replacements: 41,
            overflow_rejects: 43,
            overflow_evictions_by_new_hash: 47,
            rows_with_overflow: 53,
            spill_records_avoided: 59,
            spill_bytes_avoided: 61,
        };
        let right = left.clone();

        left.merge(right);

        assert_eq!(left.annotation_wall, Duration::from_millis(6));
        assert_eq!(left.coalesce_sort_wall, Duration::from_millis(4));
        assert_eq!(left.coalesce_merge_wall, Duration::from_millis(6));
        assert_eq!(left.coalesce_wall, Duration::from_millis(10));
        assert_eq!(left.reciprocal_checks, 14);
        assert_eq!(left.reciprocal_hits, 4);
        assert_eq!(left.support_once_candidates, 22);
        assert_eq!(left.support_twice_candidates, 26);
        assert_eq!(left.class_counts, [2, 4, 6, 8, 10, 12]);
        assert_eq!(left.candidates_before_coalesce, 34);
        assert_eq!(left.candidates_after_coalesce, 38);
        assert_eq!(left.duplicate_records_removed, 46);
        assert_eq!(left.unannotated_candidates, 58);
        assert_eq!(left.qualified_collision_losers, 62);
        assert_eq!(left.overflow_inserts, 74);
        assert_eq!(left.overflow_replacements, 82);
        assert_eq!(left.overflow_rejects, 86);
        assert_eq!(left.overflow_evictions_by_new_hash, 94);
        assert_eq!(left.rows_with_overflow, 106);
        assert_eq!(left.spill_records_avoided, 118);
        assert_eq!(left.spill_bytes_avoided, 122);
    }

    #[test]
    fn hashprune_evidence_coalesce_timings_merge_exactly_once() {
        let mut left = HashPruneEvidenceProfile {
            coalesce_sort_wall: Duration::from_millis(3),
            coalesce_merge_wall: Duration::from_millis(5),
            coalesce_wall: Duration::from_millis(8),
            ..HashPruneEvidenceProfile::default()
        };

        left.merge(HashPruneEvidenceProfile {
            coalesce_sort_wall: Duration::from_millis(7),
            coalesce_merge_wall: Duration::from_millis(11),
            coalesce_wall: Duration::from_millis(18),
            ..HashPruneEvidenceProfile::default()
        });

        assert_eq!(left.coalesce_sort_wall, Duration::from_millis(10));
        assert_eq!(left.coalesce_merge_wall, Duration::from_millis(16));
        assert_eq!(left.coalesce_wall, Duration::from_millis(26));
    }
}
