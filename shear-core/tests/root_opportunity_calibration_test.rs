use shear_core::algorithm::pipnn::root_opportunity_calibration::{
    child_work_metrics, core_drop_kept_len, exact_l2_knn, membership_intersection_count,
    opportunity_observation, summarize_opportunity_observations,
};

#[test]
fn root_opportunity_core_drop_uses_sqrt_distance_gap_not_squared_gap() {
    let distances_sq = [1.0, 4.0, 9.0, 16.0, 100.0];

    let kept = core_drop_kept_len(&distances_sq, 4, 20.0, 5);

    assert_eq!(kept, 5);
}

#[test]
fn root_opportunity_core_drop_keeps_core_r_on_high_gap_and_fixed_top10_on_low_gap() {
    let high_gap = [
        0.0, 1.0, 4.0, 9.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0, 256.0,
    ];
    assert_eq!(core_drop_kept_len(&high_gap, 4, 5.0, 10), 4);

    let low_gap = [
        0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0,
    ];
    assert_eq!(core_drop_kept_len(&low_gap, 4, 5.0, 10), 10);
}

#[test]
fn root_opportunity_membership_intersection_counts_fixed_and_candidate_opportunities() {
    let fixed = membership_intersection_count(&[0, 1, 2, 3], &[2, 3, 4, 5]);
    let candidate = membership_intersection_count(&[0, 1], &[1, 2]);

    assert_eq!(fixed, 2);
    assert_eq!(candidate, 1);
}

#[test]
fn root_opportunity_critical_miss_requires_fixed_overlap_and_candidate_zero_overlap() {
    let no_fixed_overlap = opportunity_observation(&[0], &[1], &[], &[], 0);
    assert!(!no_fixed_overlap.critical_miss_vs_fixed);

    let candidate_overlap = opportunity_observation(&[0, 1], &[1, 2], &[1], &[1], 0);
    assert!(!candidate_overlap.critical_miss_vs_fixed);

    let miss = opportunity_observation(&[0, 1], &[1, 2], &[0], &[2], 0);
    assert!(miss.critical_miss_vs_fixed);
}

#[test]
fn root_opportunity_weighted_opportunity_loss_uses_inverse_rank_weights() {
    let rank0_loss = opportunity_observation(&[0], &[0], &[1], &[2], 0);
    let rank1_no_loss = opportunity_observation(&[0], &[0], &[0], &[0], 1);

    let summary = summarize_opportunity_observations(&[rank0_loss, rank1_no_loss], 0.05);

    assert!((summary.mean_opportunity_loss_vs_fixed - 0.5).abs() < 1e-12);
    assert!((summary.weighted_opportunity_loss_vs_fixed - (2.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn root_opportunity_child_work_metrics_are_deterministic_and_normalized() {
    let metrics = child_work_metrics(&[4, 2, 0], 40.0);

    assert!((metrics.sum_child_size_squared_ratio - 0.5).abs() < 1e-12);
    assert!((metrics.predicted_work_score - metrics.sum_child_size_squared_ratio).abs() < 1e-12);
    assert!((metrics.largest_child_ratio - (4.0 / 6.0)).abs() < 1e-12);
    assert!((metrics.child_size_cv - (2.0_f64 / 3.0_f64).sqrt()).abs() < 1e-12);
}

#[test]
fn root_opportunity_exact_knn_skips_self_and_tie_breaks_by_point_id() {
    let data = [0.0, 1.0, -1.0, 2.0];

    let neighbors = exact_l2_knn(&data, 4, 1, 0, 3).unwrap();

    let ids: Vec<_> = neighbors.iter().map(|neighbor| neighbor.point_id).collect();
    assert_eq!(ids, vec![1, 2, 3]);
    assert_eq!(neighbors[0].distance_sq, 1.0);
    assert_eq!(neighbors[1].distance_sq, 1.0);
}
