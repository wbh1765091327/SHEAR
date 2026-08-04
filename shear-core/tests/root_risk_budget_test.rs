use shear_core::algorithm::pipnn::root_risk_budget::{
    RootRiskBudgetConfig, RootRiskBudgetSnapshot, RootRiskBudgetStore, RootRiskSigmaMode,
    RootRiskStatsSource, RootRiskUtilityRecord, calibrate_budget_lambda,
    membership_utility_from_margin, select_budgeted_utility_fanout, select_risk_budgeted_fanout,
};

fn budget_config(target_avg_fanout: f32) -> RootRiskBudgetConfig {
    RootRiskBudgetConfig {
        fixed_fanout: 10,
        min_fanout: 4,
        candidate_rs: vec![4, 6, 8, 10],
        target_avg_fanout,
        warmup_points: 0,
        sigma_lambda: 1.0,
        budget_lambda_override: None,
        sigma_mode: RootRiskSigmaMode::CellQ90Q50,
    }
}

#[test]
fn utility_selector_keeps_high_utility_optional_memberships_under_budget() {
    let config = budget_config(8.0);
    let high_boundary = RootRiskUtilityRecord::from_utilities(vec![
        1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.01, 0.01, 0.01, 0.01,
    ]);
    let core_point = RootRiskUtilityRecord::from_utilities(vec![
        1.0, 0.90, 0.85, 0.80, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
    ]);

    let high_decision = select_budgeted_utility_fanout(&config, &high_boundary, 0.30);
    let core_decision = select_budgeted_utility_fanout(&config, &core_point, 0.30);

    assert_eq!(high_decision.kept, 6);
    assert_eq!(core_decision.kept, 4);
    assert!(high_decision.retained_utility > core_decision.retained_utility);
}

#[test]
fn calibrated_budget_lambda_tracks_target_average_fanout() {
    let records: Vec<_> = (0..=100)
        .map(|idx| {
            let tail_utility = idx as f64 / 100.0;
            let mut utilities = vec![1.0; 4];
            utilities.extend(std::iter::repeat(tail_utility).take(6));
            RootRiskUtilityRecord::from_utilities(utilities)
        })
        .collect();

    for target in [8.0, 7.0, 6.0] {
        let config = budget_config(target);
        let calibration = calibrate_budget_lambda(&config, &records);
        assert!((calibration.avg_fanout - f64::from(target)).abs() <= 0.25);
    }
}

#[test]
fn candidate_prefixes_are_fixed_and_ties_choose_smaller_fanout() {
    let config = budget_config(8.0);
    let zero_utility = RootRiskUtilityRecord::from_utilities(vec![0.0; 10]);
    let decision = select_budgeted_utility_fanout(&config, &zero_utility, 0.0);

    assert_eq!(decision.kept, 4);
    assert!(config.candidate_rs.contains(&decision.kept));

    let positive_utility = RootRiskUtilityRecord::from_utilities(vec![1.0; 10]);
    let decision = select_budgeted_utility_fanout(&config, &positive_utility, 0.0);
    assert_eq!(decision.kept, 10);
}

#[test]
fn cell_stats_fall_back_to_global_stats_before_fixed_fanout() {
    let config = budget_config(8.0);
    let mut store = RootRiskBudgetStore::new(2);
    store.insert(0, 1.0);
    for sample in 0..20 {
        store.insert(1, 1.0 + sample as f32 * 0.02);
    }
    let snapshot = store.snapshot_for_config(&config);
    let leader_pair_distances = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let decision = select_risk_budgeted_fanout(
        &config,
        &snapshot,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        &[1.0, 1.21, 1.44, 1.69, 4.0, 4.41, 4.84, 5.29, 5.76, 6.25],
        &leader_pair_distances,
        0.8,
    );

    assert_ne!(decision.kept, config.fixed_fanout);
    assert_eq!(decision.stats_source, RootRiskStatsSource::Global);

    let empty_snapshot =
        RootRiskBudgetSnapshot::from_cell_samples_with_config(vec![vec![]], &config);
    let fixed = select_risk_budgeted_fanout(
        &config,
        &empty_snapshot,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        &[1.0, 1.21, 1.44, 1.69, 4.0, 4.41, 4.84, 5.29, 5.76, 6.25],
        &leader_pair_distances,
        0.8,
    );

    assert_eq!(fixed.kept, config.fixed_fanout);
    assert!(fixed.fallback);
    assert_eq!(fixed.stats_source, RootRiskStatsSource::Unavailable);
}

#[test]
fn exact_leader_pair_margin_assigns_more_utility_to_boundary_points_than_core_points() {
    let sigma = 0.5;
    let boundary_utility = membership_utility_from_margin(1.0, 1.21, Some(10.0), sigma);
    let core_utility = membership_utility_from_margin(1.0, 81.0, Some(10.0), sigma);
    let conservative_boundary = membership_utility_from_margin(1.0, 1.21, None, sigma);

    assert!(boundary_utility > conservative_boundary);
    assert!(boundary_utility > core_utility);
    assert!(core_utility < 1e-6);
}
