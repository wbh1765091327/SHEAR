use super::params::{PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNParams};
use crate::common::{AnnError, AnnResult};

pub(crate) const MAX_INTERNAL_BACKUPS: usize = 16;
const MULTIHIT_CANDIDATE_BUDGET: usize = 10;
const SOFT_TIER_ALPHA: f32 = 1.1;
const SOFT_TIER_ALPHA_SQ: f64 = 1.21;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HashPruneBackfillPolicy {
    mode: PiPNNHashPruneBackfillMode,
    ablation: PiPNNHashPruneBackfillAblation,
    budget: usize,
    alpha: f32,
    alpha_sq: f64,
    requested_hash_quota: Option<usize>,
    hash_quota: usize,
    requested_priority_alpha: Option<f32>,
    priority_alpha: f32,
    priority_alpha_sq: f64,
    requested_max_backfill_per_row: Option<usize>,
}

// All constructors reject non-finite alpha values, so equality is reflexive.
impl Eq for HashPruneBackfillPolicy {}

impl HashPruneBackfillPolicy {
    pub(crate) const fn off() -> Self {
        Self {
            mode: PiPNNHashPruneBackfillMode::Off,
            ablation: PiPNNHashPruneBackfillAblation::None,
            budget: 0,
            alpha: 1.0,
            alpha_sq: 1.0,
            requested_hash_quota: None,
            hash_quota: 1,
            requested_priority_alpha: None,
            priority_alpha: 1.0,
            priority_alpha_sq: 1.0,
            requested_max_backfill_per_row: None,
        }
    }

    pub(crate) fn from_params(params: &PiPNNParams) -> AnnResult<Self> {
        params.validate_hashprune_backfill_config()?;
        let soft_tier =
            params.hashprune_backfill_mode == PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        let hash_quota =
            params
                .hashprune_rhalf_hash_quota
                .unwrap_or(match params.hashprune_backfill_mode {
                    PiPNNHashPruneBackfillMode::RHalfHashSlot1 => 1,
                    PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
                    | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
                    | PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => 3,
                    _ => 0,
                });
        let priority_alpha = if soft_tier {
            params
                .hashprune_soft_tier_priority_alpha
                .unwrap_or(SOFT_TIER_ALPHA)
        } else {
            1.0
        };
        let priority_alpha_sq = if priority_alpha.to_bits() == SOFT_TIER_ALPHA.to_bits() {
            SOFT_TIER_ALPHA_SQ
        } else {
            f64::from(priority_alpha) * f64::from(priority_alpha)
        };
        Ok(Self {
            mode: params.hashprune_backfill_mode,
            ablation: params.hashprune_backfill_ablation,
            budget: params.hashprune_backfill_budget,
            alpha: params.hashprune_backfill_alpha,
            alpha_sq: f64::from(params.hashprune_backfill_alpha)
                * f64::from(params.hashprune_backfill_alpha),
            requested_hash_quota: params.hashprune_rhalf_hash_quota,
            hash_quota,
            requested_priority_alpha: params.hashprune_soft_tier_priority_alpha,
            priority_alpha,
            priority_alpha_sq,
            requested_max_backfill_per_row: params.hashprune_rhalf_max_backfill_per_row,
        })
    }

    pub(crate) fn diagnostic_soft_tier_q3(priority_alpha: f32) -> AnnResult<Self> {
        if !priority_alpha.is_finite() || priority_alpha < 1.0 {
            return Err(AnnError::log_index_error(
                "diagnostic SoftTier priority alpha must be finite and >= 1.0".to_string(),
            ));
        }
        let priority_alpha_sq = if priority_alpha.to_bits() == SOFT_TIER_ALPHA.to_bits() {
            SOFT_TIER_ALPHA_SQ
        } else {
            f64::from(priority_alpha) * f64::from(priority_alpha)
        };
        Ok(Self {
            mode: PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            ablation: PiPNNHashPruneBackfillAblation::None,
            budget: 0,
            alpha: 1.0,
            alpha_sq: 1.0,
            requested_hash_quota: None,
            hash_quota: 3,
            requested_priority_alpha: None,
            priority_alpha,
            priority_alpha_sq,
            requested_max_backfill_per_row: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn distance_gated(budget: usize, alpha: f32) -> Self {
        assert!((1..=MAX_INTERNAL_BACKUPS).contains(&budget));
        assert!(alpha.is_finite() && alpha >= 1.0);
        Self {
            mode: PiPNNHashPruneBackfillMode::DistanceGated,
            ablation: PiPNNHashPruneBackfillAblation::None,
            budget,
            alpha,
            alpha_sq: f64::from(alpha) * f64::from(alpha),
            requested_hash_quota: None,
            hash_quota: 1,
            requested_priority_alpha: None,
            priority_alpha: 1.0,
            priority_alpha_sq: 1.0,
            requested_max_backfill_per_row: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn distance_gated_stratified(budget: usize, alpha: f32) -> Self {
        assert!((1..=MAX_INTERNAL_BACKUPS).contains(&budget));
        assert!(alpha.is_finite() && alpha >= 1.0);
        Self {
            mode: PiPNNHashPruneBackfillMode::DistanceGatedStratified,
            ablation: PiPNNHashPruneBackfillAblation::None,
            budget,
            alpha,
            alpha_sq: f64::from(alpha) * f64::from(alpha),
            requested_hash_quota: None,
            hash_quota: 1,
            requested_priority_alpha: None,
            priority_alpha: 1.0,
            priority_alpha_sq: 1.0,
            requested_max_backfill_per_row: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn distance_gated_multihit(budget: usize, alpha: f32) -> Self {
        assert!((1..=8).contains(&budget));
        assert!(alpha.is_finite() && alpha >= 1.0);
        Self {
            mode: PiPNNHashPruneBackfillMode::DistanceGatedMultihit,
            ablation: PiPNNHashPruneBackfillAblation::None,
            budget,
            alpha,
            alpha_sq: f64::from(alpha) * f64::from(alpha),
            requested_hash_quota: None,
            hash_quota: 2,
            requested_priority_alpha: None,
            priority_alpha: 1.0,
            priority_alpha_sq: 1.0,
            requested_max_backfill_per_row: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn distance_gated_tiered_d52(budget: usize, alpha: f32) -> Self {
        assert_eq!(budget, 8);
        assert!(alpha.is_finite() && alpha >= 1.0);
        Self {
            mode: PiPNNHashPruneBackfillMode::DistanceGatedTieredD52,
            ablation: PiPNNHashPruneBackfillAblation::None,
            budget,
            alpha,
            alpha_sq: f64::from(alpha) * f64::from(alpha),
            requested_hash_quota: None,
            hash_quota: 3,
            requested_priority_alpha: None,
            priority_alpha: 1.0,
            priority_alpha_sq: 1.0,
            requested_max_backfill_per_row: None,
        }
    }

    pub(crate) const fn mode(self) -> PiPNNHashPruneBackfillMode {
        self.mode
    }

    pub(crate) const fn ablation(self) -> PiPNNHashPruneBackfillAblation {
        self.ablation
    }

    pub(crate) const fn is_off(self) -> bool {
        matches!(self.mode, PiPNNHashPruneBackfillMode::Off)
    }

    pub(crate) const fn is_distance_gated(self) -> bool {
        matches!(
            self.mode,
            PiPNNHashPruneBackfillMode::DistanceGated
                | PiPNNHashPruneBackfillMode::DistanceGatedStratified
                | PiPNNHashPruneBackfillMode::DistanceGatedMultihit
                | PiPNNHashPruneBackfillMode::DistanceGatedTieredD52
                | PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
        )
    }

    pub(crate) const fn is_stratified(self) -> bool {
        matches!(
            self.mode,
            PiPNNHashPruneBackfillMode::DistanceGatedStratified
        )
    }

    pub(crate) const fn is_multihit(self) -> bool {
        matches!(self.mode, PiPNNHashPruneBackfillMode::DistanceGatedMultihit)
    }

    pub(crate) const fn is_tiered(self) -> bool {
        matches!(
            self.mode,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52
                | PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
                | PiPNNHashPruneBackfillMode::RHalfHardTierQ3
        )
    }

    pub(crate) const fn is_rhalf(self) -> bool {
        matches!(
            self.mode,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
                | PiPNNHashPruneBackfillMode::RHalfHashSlot1
                | PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
                | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
                | PiPNNHashPruneBackfillMode::RHalfHardTierQ3
        )
    }

    pub(crate) const fn budget(self) -> usize {
        self.budget
    }

    pub(crate) const fn candidate_budget(self) -> usize {
        match self.mode {
            PiPNNHashPruneBackfillMode::DistanceGatedStratified => 8,
            PiPNNHashPruneBackfillMode::DistanceGatedMultihit
            | PiPNNHashPruneBackfillMode::DistanceGatedTieredD52 => MULTIHIT_CANDIDATE_BUDGET,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
            | PiPNNHashPruneBackfillMode::RHalfHashSlot1
            | PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
            | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
            | PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => self.budget,
            PiPNNHashPruneBackfillMode::Off | PiPNNHashPruneBackfillMode::DistanceGated => {
                self.budget
            }
        }
    }

    pub(crate) const fn backup_hash_quota(self) -> usize {
        match self.mode {
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52 => 3,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf => match self.ablation {
                PiPNNHashPruneBackfillAblation::Quota1 => 1,
                PiPNNHashPruneBackfillAblation::Quota2 => 2,
                _ => 3,
            },
            PiPNNHashPruneBackfillMode::RHalfHashSlot1
            | PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
            | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
            | PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => self.hash_quota,
            PiPNNHashPruneBackfillMode::DistanceGatedMultihit => 2,
            _ => 1,
        }
    }

    pub(crate) const fn requested_hash_quota(self) -> Option<usize> {
        self.requested_hash_quota
    }

    pub(crate) const fn degree_target(self) -> usize {
        if matches!(
            self.mode,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52
        ) {
            52
        } else {
            0
        }
    }

    pub(crate) const fn budget_rule(self) -> &'static str {
        if !self.is_rhalf() {
            return "fixed";
        }
        match self.ablation {
            PiPNNHashPruneBackfillAblation::BudgetQuarter => "floor-r-quarter",
            PiPNNHashPruneBackfillAblation::NoHalfCap => "all-spare",
            _ => "floor-r-half",
        }
    }

    pub(crate) const fn budget_for(self, r: usize) -> usize {
        if !self.is_rhalf() {
            return self.budget;
        }
        match self.ablation {
            PiPNNHashPruneBackfillAblation::BudgetQuarter => r / 4,
            PiPNNHashPruneBackfillAblation::NoHalfCap => r,
            _ => r / 2,
        }
    }

    pub(crate) const fn candidate_capacity_for(self, r: usize, primary_count: usize) -> usize {
        let budget = self.budget_for(r);
        let row_cap = match self.requested_max_backfill_per_row {
            Some(requested) if requested < budget => requested,
            _ => budget,
        };
        let spare = r.saturating_sub(primary_count);
        if row_cap < spare { row_cap } else { spare }
    }

    pub(crate) const fn requested_max_backfill_per_row(self) -> Option<usize> {
        self.requested_max_backfill_per_row
    }

    pub(crate) const fn effective_max_backfill_per_row(self, r: usize) -> usize {
        let budget = self.budget_for(r);
        match self.requested_max_backfill_per_row {
            Some(requested) if requested < budget => requested,
            _ => budget,
        }
    }

    pub(crate) const fn capacity_rule(self) -> &'static str {
        if self.requested_max_backfill_per_row.is_some() {
            "min-derived-budget-row-cap-spare"
        } else {
            "min-derived-budget-spare"
        }
    }

    pub(crate) const fn is_tier_first(self) -> bool {
        self.is_tiered() && !matches!(self.ablation, PiPNNHashPruneBackfillAblation::DistanceFirst)
    }

    pub(crate) const fn priority_rule(self) -> &'static str {
        match self.mode {
            PiPNNHashPruneBackfillMode::RHalfHashSlot1 => "distance-first-q1",
            PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3 => "distance-first-q3",
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3 => "soft-tier-q3",
            PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => "hard-tier-q3",
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf => {
                if matches!(self.ablation, PiPNNHashPruneBackfillAblation::DistanceFirst) {
                    "legacy-distance-first"
                } else {
                    "legacy-hard-tier"
                }
            }
            _ => "legacy-distance",
        }
    }

    pub(crate) const fn priority_alpha(self) -> f32 {
        self.priority_alpha
    }

    pub(crate) const fn requested_priority_alpha(self) -> Option<f32> {
        self.requested_priority_alpha
    }

    pub(crate) const fn priority_score(self, squared_distance: f32, hit_layer: usize) -> f64 {
        let distance = squared_distance as f64;
        if !matches!(self.mode, PiPNNHashPruneBackfillMode::RHalfSoftTierQ3) {
            return distance;
        }
        match hit_layer {
            1 => distance,
            2 => distance * self.priority_alpha_sq,
            3 => distance * self.priority_alpha_sq * self.priority_alpha_sq,
            _ => distance,
        }
    }

    pub(crate) const fn uses_alpha_gate(self) -> bool {
        self.is_distance_gated()
            && !(self.is_rhalf() && matches!(self.ablation, PiPNNHashPruneBackfillAblation::NoVeto))
    }

    pub(crate) const fn post_veto_refill(self) -> bool {
        false
    }

    pub(crate) const fn degree_target_for(self, r: usize) -> usize {
        if self.is_rhalf() {
            r
        } else {
            self.degree_target()
        }
    }

    pub(crate) const fn alpha(self) -> f32 {
        self.alpha
    }

    pub(crate) const fn alpha_sq(self) -> f64 {
        self.alpha_sq
    }
}

impl Default for HashPruneBackfillPolicy {
    fn default() -> Self {
        Self::off()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HashPruneBackfillProfile {
    pub(crate) rows_seen: u64,
    pub(crate) rows_underfull_before: u64,
    pub(crate) rows_with_backfill: u64,
    pub(crate) primary_edges: u64,
    pub(crate) backfill_edges: u64,
    pub(crate) gate_rejected: u64,
    pub(crate) budget_rejected: u64,
    pub(crate) degree_before_sum: u64,
    pub(crate) degree_after_sum: u64,
    pub(crate) baseline_full_rows: u64,
    pub(crate) rows_with_nonempty_shortlist: u64,
    pub(crate) shortlist_admissions: u64,
    pub(crate) shortlist_replacements: u64,
    pub(crate) shortlist_unchanged: u64,
    pub(crate) shortlist_capacity_rejections: u64,
    pub(crate) shortlisted_by_layer: [u64; 3],
    pub(crate) emitted_by_layer: [u64; 3],
    pub(crate) rows_with_deeper_backfill: u64,
    pub(crate) gate_rejected_by_layer: [u64; 3],
    pub(crate) rows_with_veto_holes: u64,
}

impl HashPruneBackfillProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        self.rows_seen += other.rows_seen;
        self.rows_underfull_before += other.rows_underfull_before;
        self.rows_with_backfill += other.rows_with_backfill;
        self.primary_edges += other.primary_edges;
        self.backfill_edges += other.backfill_edges;
        self.gate_rejected += other.gate_rejected;
        self.budget_rejected += other.budget_rejected;
        self.degree_before_sum += other.degree_before_sum;
        self.degree_after_sum += other.degree_after_sum;
        self.baseline_full_rows += other.baseline_full_rows;
        self.rows_with_nonempty_shortlist += other.rows_with_nonempty_shortlist;
        self.shortlist_admissions += other.shortlist_admissions;
        self.shortlist_replacements += other.shortlist_replacements;
        self.shortlist_unchanged += other.shortlist_unchanged;
        self.shortlist_capacity_rejections += other.shortlist_capacity_rejections;
        for layer in 0..3 {
            self.shortlisted_by_layer[layer] += other.shortlisted_by_layer[layer];
            self.emitted_by_layer[layer] += other.emitted_by_layer[layer];
            self.gate_rejected_by_layer[layer] += other.gate_rejected_by_layer[layer];
        }
        self.rows_with_deeper_backfill += other.rows_with_deeper_backfill;
        self.rows_with_veto_holes += other.rows_with_veto_holes;
    }

    pub(crate) fn mean_degree_before(&self) -> f64 {
        if self.rows_seen == 0 {
            0.0
        } else {
            self.degree_before_sum as f64 / self.rows_seen as f64
        }
    }

    pub(crate) fn mean_degree_after(&self) -> f64 {
        if self.rows_seen == 0 {
            0.0
        } else {
            self.degree_after_sum as f64 / self.rows_seen as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::pipnn::params::{
        PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNParams,
    };

    fn tiered_rhalf_policy(ablation: PiPNNHashPruneBackfillAblation) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf;
        params.hashprune_backfill_ablation = ablation;
        params.hashprune_backfill_budget = 0;
        params.hashprune_backfill_alpha = 1.1;
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    #[test]
    fn hashprune_backfill_policy_precomputes_squared_alpha() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGated;
        params.hashprune_backfill_budget = 8;
        params.hashprune_backfill_alpha = 1.1;
        let policy = HashPruneBackfillPolicy::from_params(&params).unwrap();
        assert!(policy.is_distance_gated());
        assert_eq!(policy.budget(), 8);
        assert!((policy.alpha_sq() - 1.21).abs() < 1e-6);
    }

    #[test]
    fn diagnostic_soft_alpha_110_is_identical_to_named_softtier() {
        let mut params = PiPNNParams::default();
        params.l_max = 8;
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        let named = HashPruneBackfillPolicy::from_params(&params).unwrap();
        let diagnostic = HashPruneBackfillPolicy::diagnostic_soft_tier_q3(1.10).unwrap();
        for layer in 1..=3 {
            assert_eq!(
                named.priority_score(7.0, layer),
                diagnostic.priority_score(7.0, layer)
            );
        }
        assert_eq!(named.priority_alpha(), 1.1);
    }

    #[test]
    fn diagnostic_soft_alpha_100_matches_distance_priority_scores() {
        let diagnostic = HashPruneBackfillPolicy::diagnostic_soft_tier_q3(1.0).unwrap();
        for layer in 1..=3 {
            assert_eq!(diagnostic.priority_score(7.0, layer), 7.0);
        }
    }

    #[test]
    fn named_rhalf_policy_resolves_quota_and_soft_alpha_overrides() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_rhalf_hash_quota = Some(2);
        params.hashprune_soft_tier_priority_alpha = Some(1.4);
        let policy = HashPruneBackfillPolicy::from_params(&params).unwrap();
        assert_eq!(policy.backup_hash_quota(), 2);
        assert_eq!(policy.priority_alpha(), 1.4);
        assert!((policy.priority_score(10.0, 2) - 19.6).abs() < 1e-6);
    }

    #[test]
    fn named_rhalf_default_policy_contract_is_unchanged() {
        let cases = [
            (PiPNNHashPruneBackfillMode::RHalfHashSlot1, 1, 1.0),
            (PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3, 3, 1.0),
            (PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, 3, 1.1),
            (PiPNNHashPruneBackfillMode::RHalfHardTierQ3, 3, 1.0),
        ];
        for (mode, quota, alpha) in cases {
            let mut params = PiPNNParams::default();
            params.hashprune_backfill_mode = mode;
            let policy = HashPruneBackfillPolicy::from_params(&params).unwrap();
            assert_eq!(policy.backup_hash_quota(), quota);
            assert_eq!(policy.priority_alpha(), alpha);
            assert_eq!(policy.budget_for(64), 32);
        }
    }

    fn named_rhalf_policy_with_row_cap(
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

    #[test]
    fn hashprune_rhalf_row_cap_applies_at_the_shared_capacity_boundary() {
        for cap in [2, 3, 4] {
            let policy = named_rhalf_policy_with_row_cap(
                PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
                64,
                Some(cap),
            );
            assert_eq!(policy.requested_max_backfill_per_row(), Some(cap));
            assert_eq!(policy.effective_max_backfill_per_row(64), cap);
            assert_eq!(policy.candidate_capacity_for(64, 0), cap);
            assert_eq!(policy.candidate_capacity_for(64, 63), 1);
            assert_eq!(policy.candidate_capacity_for(64, 64), 0);
            assert_eq!(policy.capacity_rule(), "min-derived-budget-row-cap-spare");
        }
    }

    #[test]
    fn hashprune_rhalf_row_cap_omission_preserves_the_floor_r_half_capacity() {
        let policy =
            named_rhalf_policy_with_row_cap(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, 64, None);
        assert_eq!(policy.requested_max_backfill_per_row(), None);
        assert_eq!(policy.effective_max_backfill_per_row(64), 32);
        assert_eq!(policy.candidate_capacity_for(64, 0), 32);
        assert_eq!(policy.capacity_rule(), "min-derived-budget-spare");
    }

    #[test]
    fn hashprune_tiered_rhalf_policy_derives_full_contract_from_r() {
        let policy = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::None);
        assert!(policy.is_rhalf());
        assert!(policy.is_tiered());
        assert_eq!(policy.ablation(), PiPNNHashPruneBackfillAblation::None);
        assert_eq!(policy.budget_rule(), "floor-r-half");
        assert_eq!(policy.backup_hash_quota(), 3);
        assert!(policy.is_tier_first());
        assert!(policy.uses_alpha_gate());
        assert!(!policy.post_veto_refill());
        for (r, budget) in [(32, 16), (64, 32), (96, 48)] {
            assert_eq!(policy.budget_for(r), budget);
            assert_eq!(policy.candidate_capacity_for(r, 0), budget);
            assert_eq!(policy.candidate_capacity_for(r, r - 3), 3);
            assert_eq!(policy.degree_target_for(r), r);
        }
    }

    #[test]
    fn hashprune_tiered_rhalf_policy_ablations_change_only_named_rule() {
        let quarter = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::BudgetQuarter);
        assert_eq!(quarter.budget_rule(), "floor-r-quarter");
        assert_eq!(quarter.budget_for(64), 16);
        assert_eq!(quarter.backup_hash_quota(), 3);
        assert!(quarter.is_tier_first());
        assert!(quarter.uses_alpha_gate());

        let uncapped = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoHalfCap);
        assert_eq!(uncapped.budget_rule(), "all-spare");
        assert_eq!(uncapped.budget_for(64), 64);
        assert_eq!(uncapped.candidate_capacity_for(64, 17), 47);

        let quota1 = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::Quota1);
        let quota2 = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::Quota2);
        assert_eq!(quota1.backup_hash_quota(), 1);
        assert_eq!(quota2.backup_hash_quota(), 2);
        assert_eq!(quota1.budget_for(64), 32);
        assert_eq!(quota2.budget_for(64), 32);

        let distance_first = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::DistanceFirst);
        assert!(!distance_first.is_tier_first());
        assert_eq!(distance_first.backup_hash_quota(), 3);
        assert_eq!(distance_first.budget_for(64), 32);

        let no_veto = tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto);
        assert!(!no_veto.uses_alpha_gate());
        assert!(!no_veto.post_veto_refill());
        assert!(no_veto.is_tier_first());
        assert_eq!(no_veto.budget_for(64), 32);
    }

    #[test]
    fn hashprune_stratified_policy_uses_pool_eight_for_k4_output() {
        let policy = HashPruneBackfillPolicy::distance_gated_stratified(4, 1.1);
        assert!(policy.is_distance_gated());
        assert!(policy.is_stratified());
        assert_eq!(policy.budget(), 4);
        assert_eq!(policy.candidate_budget(), 8);
    }

    #[test]
    fn hashprune_multihit_policy_uses_pool_ten_and_two_backups_per_hash() {
        let policy = HashPruneBackfillPolicy::distance_gated_multihit(6, 1.1);
        assert!(policy.is_distance_gated());
        assert!(policy.is_multihit());
        assert_eq!(policy.budget(), 6);
        assert_eq!(policy.candidate_budget(), 10);
        assert_eq!(policy.backup_hash_quota(), 2);
    }

    #[test]
    fn hashprune_tiered_d52_policy_uses_pool_ten_quota_three_and_target_52() {
        let policy = HashPruneBackfillPolicy::distance_gated_tiered_d52(8, 1.1);
        assert!(policy.is_distance_gated());
        assert!(policy.is_tiered());
        assert_eq!(policy.budget(), 8);
        assert_eq!(policy.candidate_budget(), 10);
        assert_eq!(policy.backup_hash_quota(), 3);
        assert_eq!(policy.degree_target(), 52);
    }

    #[test]
    fn hashprune_backfill_profile_merges_row_totals() {
        let mut left = HashPruneBackfillProfile {
            rows_seen: 2,
            rows_underfull_before: 1,
            rows_with_backfill: 1,
            primary_edges: 6,
            backfill_edges: 2,
            gate_rejected: 3,
            budget_rejected: 4,
            degree_before_sum: 6,
            degree_after_sum: 8,
            ..Default::default()
        };
        left.merge(HashPruneBackfillProfile {
            rows_seen: 1,
            primary_edges: 3,
            degree_before_sum: 3,
            degree_after_sum: 3,
            ..Default::default()
        });
        assert_eq!(left.rows_seen, 3);
        assert_eq!(left.primary_edges, 9);
        assert_eq!(left.mean_degree_before(), 3.0);
        assert_eq!(left.mean_degree_after(), 11.0 / 3.0);
    }

    #[test]
    fn hashprune_tiered_rhalf_profile_merges_online_and_layer_totals() {
        let mut left = HashPruneBackfillProfile {
            baseline_full_rows: 2,
            rows_with_nonempty_shortlist: 3,
            shortlist_admissions: 5,
            shortlist_replacements: 7,
            shortlist_unchanged: 9,
            shortlist_capacity_rejections: 11,
            shortlisted_by_layer: [13, 17, 19],
            emitted_by_layer: [23, 29, 31],
            gate_rejected_by_layer: [37, 41, 43],
            rows_with_veto_holes: 47,
            ..Default::default()
        };
        left.merge(HashPruneBackfillProfile {
            baseline_full_rows: 1,
            rows_with_nonempty_shortlist: 2,
            shortlist_admissions: 3,
            shortlist_replacements: 4,
            shortlist_unchanged: 6,
            shortlist_capacity_rejections: 5,
            shortlisted_by_layer: [7, 11, 13],
            emitted_by_layer: [17, 19, 23],
            gate_rejected_by_layer: [29, 31, 37],
            rows_with_veto_holes: 41,
            ..Default::default()
        });

        assert_eq!(left.baseline_full_rows, 3);
        assert_eq!(left.rows_with_nonempty_shortlist, 5);
        assert_eq!(left.shortlist_admissions, 8);
        assert_eq!(left.shortlist_replacements, 11);
        assert_eq!(left.shortlist_unchanged, 15);
        assert_eq!(left.shortlist_capacity_rejections, 16);
        assert_eq!(left.shortlisted_by_layer, [20, 28, 32]);
        assert_eq!(left.emitted_by_layer, [40, 48, 54]);
        assert_eq!(left.gate_rejected_by_layer, [66, 72, 80]);
        assert_eq!(left.rows_with_veto_holes, 88);
    }
}
