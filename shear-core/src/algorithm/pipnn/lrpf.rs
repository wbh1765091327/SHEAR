use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct LrpfCandidate {
    pub(crate) dst: u32,
    pub(crate) dist: f32,
}

impl LrpfCandidate {
    pub(crate) fn new(dst: u32, dist: f32) -> Self {
        Self { dst, dist }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct LrpfWitness {
    pub(crate) pivot_idx: u16,
    pub(crate) victim_idx: u16,
    pub(crate) margin: f32,
}

impl LrpfWitness {
    pub(crate) fn new(pivot_idx: u16, victim_idx: u16, margin: f32) -> Self {
        Self {
            pivot_idx,
            victim_idx,
            margin,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LrpfConfig {
    pub(crate) degree: usize,
    pub(crate) protected_core: usize,
    pub(crate) pre_spill_width: usize,
    pub(crate) long_edge_quota: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LrpfCompressedRow {
    pub(crate) candidates: Vec<LrpfCandidate>,
    pub(crate) witnesses: Vec<LrpfWitness>,
}

pub(crate) fn reduce_lrpf_row(
    candidates: &[LrpfCandidate],
    witnesses: &[LrpfWitness],
    config: LrpfConfig,
) -> Vec<u32> {
    if config.degree == 0 {
        return Vec::new();
    }

    let sorted = sorted_unique_candidates(candidates);
    let selected = select_lrpf_candidate_indices(
        &sorted,
        witnesses,
        config.degree,
        config.protected_core,
        config.long_edge_quota,
    );

    sorted
        .iter()
        .zip(selected)
        .filter_map(|(candidate, selected)| selected.then_some(candidate.dst))
        .collect()
}

pub(crate) fn compress_lrpf_row_before_spill(
    candidates: &[LrpfCandidate],
    witnesses: &[LrpfWitness],
    config: LrpfConfig,
) -> LrpfCompressedRow {
    if config.pre_spill_width == 0 {
        return LrpfCompressedRow {
            candidates: candidates.to_vec(),
            witnesses: witnesses.to_vec(),
        };
    }

    let sorted = sorted_unique_candidates(candidates);
    if sorted.is_empty() {
        return LrpfCompressedRow::default();
    }

    let protected = config.protected_core.min(sorted.len());
    let target_width = config.pre_spill_width.max(protected).min(sorted.len());
    let keep = if sorted.len() <= target_width {
        vec![true; sorted.len()]
    } else {
        select_lrpf_candidate_indices(
            &sorted,
            witnesses,
            target_width,
            config.protected_core,
            config.long_edge_quota,
        )
    };
    let kept = keep.iter().filter(|&&selected| selected).count();

    let mut old_to_new = vec![None; sorted.len()];
    let mut compressed_candidates = Vec::with_capacity(kept);
    for (old_idx, candidate) in sorted.iter().copied().enumerate() {
        if keep[old_idx] {
            let new_idx = compressed_candidates.len();
            old_to_new[old_idx] = u16::try_from(new_idx).ok();
            compressed_candidates.push(candidate);
        }
    }

    let compressed_witnesses = witnesses
        .iter()
        .filter_map(|witness| {
            if witness.margin <= 0.0 {
                return None;
            }
            let pivot_idx = old_to_new
                .get(usize::from(witness.pivot_idx))
                .copied()
                .flatten()?;
            let victim_idx = old_to_new
                .get(usize::from(witness.victim_idx))
                .copied()
                .flatten()?;
            Some(LrpfWitness::new(pivot_idx, victim_idx, witness.margin))
        })
        .collect();

    LrpfCompressedRow {
        candidates: compressed_candidates,
        witnesses: compressed_witnesses,
    }
}

fn select_lrpf_candidate_indices(
    sorted: &[LrpfCandidate],
    witnesses: &[LrpfWitness],
    target_width: usize,
    protected_core: usize,
    long_edge_quota: usize,
) -> Vec<bool> {
    let mut selected = vec![false; sorted.len()];
    if target_width == 0 || sorted.is_empty() {
        return selected;
    }

    let target = target_width.min(sorted.len());
    let protected = protected_core.min(target);
    let mut selected_count = 0usize;
    for slot in selected.iter_mut().take(protected) {
        *slot = true;
        selected_count += 1;
    }

    let long_quota = long_edge_quota.min(target.saturating_sub(selected_count));
    let primary_budget = target.saturating_sub(long_quota);
    for idx in protected..sorted.len() {
        if selected_count >= primary_budget {
            break;
        }
        if lrpf_candidate_is_witnessed(idx, witnesses, &selected) {
            continue;
        }
        selected[idx] = true;
        selected_count += 1;
    }

    let mut long_kept = 0usize;
    for idx in (protected..sorted.len()).rev() {
        if selected_count >= target || long_kept >= long_quota {
            break;
        }
        if selected[idx] {
            continue;
        }
        if lrpf_candidate_is_witnessed(idx, witnesses, &selected) {
            continue;
        }
        selected[idx] = true;
        selected_count += 1;
        long_kept += 1;
    }

    for idx in protected..sorted.len() {
        if selected_count >= target {
            break;
        }
        if selected[idx] {
            continue;
        }
        if lrpf_candidate_is_witnessed(idx, witnesses, &selected) {
            continue;
        }
        selected[idx] = true;
        selected_count += 1;
    }

    selected
}

fn sorted_unique_candidates(candidates: &[LrpfCandidate]) -> Vec<LrpfCandidate> {
    let mut best_by_dst: HashMap<u32, f32> = HashMap::new();
    for candidate in candidates {
        if candidate.dst == u32::MAX || !candidate.dist.is_finite() {
            continue;
        }
        best_by_dst
            .entry(candidate.dst)
            .and_modify(|dist| {
                if candidate.dist < *dist {
                    *dist = candidate.dist;
                }
            })
            .or_insert(candidate.dist);
    }

    let mut sorted: Vec<_> = best_by_dst
        .into_iter()
        .map(|(dst, dist)| LrpfCandidate { dst, dist })
        .collect();
    sorted.sort_by(|left, right| {
        left.dist
            .total_cmp(&right.dist)
            .then_with(|| left.dst.cmp(&right.dst))
    });
    sorted
}

fn lrpf_candidate_is_witnessed(
    victim_idx: usize,
    witnesses: &[LrpfWitness],
    selected: &[bool],
) -> bool {
    witnesses.iter().any(|witness| {
        witness.margin > 0.0
            && usize::from(witness.victim_idx) == victim_idx
            && selected
                .get(usize::from(witness.pivot_idx))
                .copied()
                .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lrpf_keeps_nearest_core_before_using_witnesses() {
        let candidates = vec![
            LrpfCandidate::new(10, 1.0),
            LrpfCandidate::new(20, 2.0),
            LrpfCandidate::new(30, 3.0),
        ];
        let witnesses = vec![LrpfWitness::new(0, 1, 0.5)];
        let config = LrpfConfig {
            degree: 2,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 0,
        };

        let out = reduce_lrpf_row(&candidates, &witnesses, config);

        assert_eq!(out, vec![10, 30]);
    }

    #[test]
    fn lrpf_deduplicates_dst_by_shortest_distance() {
        let candidates = vec![
            LrpfCandidate::new(7, 4.0),
            LrpfCandidate::new(7, 1.5),
            LrpfCandidate::new(8, 2.0),
        ];
        let config = LrpfConfig {
            degree: 2,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 0,
        };

        let out = reduce_lrpf_row(&candidates, &[], config);

        assert_eq!(out, vec![7, 8]);
    }

    #[test]
    fn lrpf_strictly_drops_witnessed_short_candidates_without_fillback() {
        let candidates = vec![
            LrpfCandidate::new(10, 1.0),
            LrpfCandidate::new(20, 2.0),
            LrpfCandidate::new(30, 3.0),
            LrpfCandidate::new(40, 4.0),
        ];
        let witnesses = vec![LrpfWitness::new(0, 1, 0.5), LrpfWitness::new(0, 2, 0.5)];
        let config = LrpfConfig {
            degree: 4,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 0,
        };

        let out = reduce_lrpf_row(&candidates, &witnesses, config);

        assert_eq!(out, vec![10, 40]);
    }

    #[test]
    fn lrpf_navigation_quota_skips_witnessed_far_candidates() {
        let candidates = vec![
            LrpfCandidate::new(10, 1.0),
            LrpfCandidate::new(20, 2.0),
            LrpfCandidate::new(30, 3.0),
            LrpfCandidate::new(40, 4.0),
            LrpfCandidate::new(50, 5.0),
        ];
        let witnesses = vec![
            LrpfWitness::new(0, 1, 0.5),
            LrpfWitness::new(0, 2, 0.5),
            LrpfWitness::new(0, 4, 0.5),
        ];
        let config = LrpfConfig {
            degree: 3,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 2,
        };

        let out = reduce_lrpf_row(&candidates, &witnesses, config);

        assert_eq!(out, vec![10, 40]);
    }

    #[test]
    fn lrpf_returns_unused_long_quota_to_unwitnessed_primary_candidates() {
        let candidates = vec![
            LrpfCandidate::new(10, 1.0),
            LrpfCandidate::new(20, 2.0),
            LrpfCandidate::new(30, 3.0),
            LrpfCandidate::new(40, 4.0),
            LrpfCandidate::new(50, 5.0),
        ];
        let witnesses = vec![LrpfWitness::new(0, 4, 0.5)];
        let config = LrpfConfig {
            degree: 4,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 2,
        };

        let out = reduce_lrpf_row(&candidates, &witnesses, config);

        assert_eq!(out, vec![10, 20, 30, 40]);
    }

    #[test]
    fn lrpf_reserves_space_for_long_edges_inside_target_degree() {
        let candidates = vec![
            LrpfCandidate::new(10, 1.0),
            LrpfCandidate::new(20, 2.0),
            LrpfCandidate::new(30, 3.0),
            LrpfCandidate::new(40, 4.0),
            LrpfCandidate::new(50, 5.0),
        ];
        let config = LrpfConfig {
            degree: 3,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 1,
        };

        let out = reduce_lrpf_row(&candidates, &[], config);

        assert_eq!(out, vec![10, 20, 50]);
    }

    #[test]
    fn lrpf_candidate_record_does_not_carry_sketch_hash() {
        assert_eq!(std::mem::size_of::<LrpfCandidate>(), 8);
    }

    #[test]
    fn lrpf_pre_spill_compression_bounds_width_but_preserves_core() {
        let candidates = vec![
            LrpfCandidate::new(1, 1.0),
            LrpfCandidate::new(2, 2.0),
            LrpfCandidate::new(3, 3.0),
            LrpfCandidate::new(4, 4.0),
        ];
        let witnesses = vec![LrpfWitness::new(0, 2, 0.5), LrpfWitness::new(1, 3, 0.5)];
        let config = LrpfConfig {
            degree: 2,
            protected_core: 1,
            pre_spill_width: 3,
            long_edge_quota: 0,
        };

        let compressed = compress_lrpf_row_before_spill(&candidates, &witnesses, config);

        assert!(compressed.candidates.len() <= 3);
        assert!(compressed.candidates.iter().any(|c| c.dst == 1));
        assert!(compressed.witnesses.len() <= witnesses.len());
    }

    #[test]
    fn lrpf_pre_spill_off_preserves_all_candidates() {
        let candidates = vec![
            LrpfCandidate::new(1, 1.0),
            LrpfCandidate::new(2, 2.0),
            LrpfCandidate::new(3, 3.0),
        ];
        let config = LrpfConfig {
            degree: 2,
            protected_core: 1,
            pre_spill_width: 0,
            long_edge_quota: 0,
        };

        let compressed = compress_lrpf_row_before_spill(&candidates, &[], config);

        assert_eq!(compressed.candidates, candidates);
    }
}
