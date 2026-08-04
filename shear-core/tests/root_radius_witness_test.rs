use shear_core::algorithm::pipnn::root_radius_witness::{
    RadiusWitnessConfig, RadiusWitnessSnapshot, RootRadiusWitnessStore, SmallKMin,
    select_radius_certified_fanout,
};

#[test]
fn root_radius_witness_k_smallest_radius_insertion_is_order_independent() {
    let samples = [(3.0, 30), (1.0, 20), (1.0, 10), (2.0, 40), (0.5, 50)];
    let mut forward = SmallKMin::new(3);
    let mut reverse = SmallKMin::new(3);

    for &(distance, point_id) in &samples {
        forward.insert(distance, point_id);
    }
    for &(distance, point_id) in samples.iter().rev() {
        reverse.insert(distance, point_id);
    }

    assert_eq!(forward.entries(), reverse.entries());
    assert_eq!(forward.entries(), vec![(0.5, 50), (1.0, 10), (1.0, 20)]);
    assert_eq!(forward.kth_radius(), Some(1.0));
}

#[test]
fn root_radius_witness_k_smallest_capacity_never_exceeds_k() {
    let mut min = SmallKMin::new(2);

    for point_id in 0..10_u32 {
        min.insert(point_id as f32, point_id);
    }

    assert_eq!(min.len(), 2);
    assert_eq!(min.entries(), vec![(0.0, 0), (1.0, 1)]);
}

#[test]
fn root_radius_witness_kth_radius_is_absent_until_cell_has_k_entries() {
    let mut min = SmallKMin::new(3);

    min.insert(0.5, 10);
    min.insert(1.5, 11);

    assert_eq!(min.kth_radius(), None);
    min.insert(2.5, 12);
    assert_eq!(min.kth_radius(), Some(2.5));
}

#[test]
fn root_radius_witness_snapshot_reports_ready_cell_count() {
    let mut store = RootRadiusWitnessStore::new(3, 2);
    store.insert(0, 10, 1.0);
    store.insert(1, 20, 2.0);
    store.insert(1, 21, 3.0);

    let snapshot = store.snapshot();

    assert_eq!(snapshot.ready_cells(), 1);
    assert_eq!(snapshot.kth_radius(0), None);
    assert_eq!(snapshot.kth_radius(1), Some(3.0));
    assert_eq!(snapshot.kth_radius(2), None);
}

#[test]
fn root_radius_witness_selector_certifies_smallest_valid_radius_fanout() {
    let config = RadiusWitnessConfig {
        k: 2,
        fixed_fanout: 4,
        candidate_rs: vec![1, 2, 3],
        warmup_points: 0,
        dry_run: false,
    };
    let snapshot =
        RadiusWitnessSnapshot::from_radii(vec![Some(10.0), Some(0.4), Some(2.0), Some(2.0)]);

    let decision = select_radius_certified_fanout(
        &config,
        &snapshot,
        &[0, 1, 2, 3],
        &[0.0, 4.0, 36.0, 49.0],
        10,
    );

    assert_eq!(decision.kept, 2);
    assert!(decision.certified);
    assert!(!decision.insufficient_radius);
}

#[test]
fn root_radius_witness_selector_falls_back_when_no_top_r_radius_is_ready() {
    let config = RadiusWitnessConfig {
        k: 2,
        fixed_fanout: 4,
        candidate_rs: vec![1, 2],
        warmup_points: 0,
        dry_run: false,
    };
    let snapshot = RadiusWitnessSnapshot::from_radii(vec![None, None, Some(1.0), Some(1.0)]);

    let decision = select_radius_certified_fanout(
        &config,
        &snapshot,
        &[0, 1, 2, 3],
        &[0.0, 4.0, 36.0, 49.0],
        10,
    );

    assert_eq!(decision.kept, 4);
    assert!(decision.fallback);
    assert!(decision.insufficient_radius);
}

#[test]
fn root_radius_witness_selector_uses_sqrt_distances_not_squared_gaps() {
    let config = RadiusWitnessConfig {
        k: 1,
        fixed_fanout: 3,
        candidate_rs: vec![1],
        warmup_points: 0,
        dry_run: false,
    };
    let snapshot = RadiusWitnessSnapshot::from_radii(vec![Some(1.1), Some(1.0), Some(1.0)]);

    let decision =
        select_radius_certified_fanout(&config, &snapshot, &[0, 1, 2], &[1.0, 9.0, 16.0], 10);

    assert_eq!(decision.kept, 3);
    assert!(!decision.certified);
    assert!(decision.fallback);
}
