use shear_core::model::rabitq::graph_search::{
    Rabitq1Codec, Rabitq1Estimate, bin_data_bytes, generate_rotator, padded_dim, rotate_vectors,
    rotator_bytes,
};

#[test]
fn graph_rabitq1_layout_and_rotator_are_deterministic() {
    assert_eq!(padded_dim(64).unwrap(), 64);
    assert_eq!(padded_dim(67).unwrap(), 128);
    assert!(padded_dim(63).is_err());
    assert!(padded_dim(4096).is_err());
    assert_eq!(rotator_bytes(128).unwrap(), 64);
    assert_eq!(bin_data_bytes(128).unwrap(), 28);

    let first = generate_rotator(67, 42).unwrap();
    let second = generate_rotator(67, 42).unwrap();
    let different = generate_rotator(67, 43).unwrap();
    assert_eq!(first, second);
    assert_ne!(first, different);
    assert_eq!(first.len(), 64);
}

fn exercise_codec(dim: usize) {
    let points: Vec<f32> = (0..3 * dim)
        .map(|index| ((index * 17 + 3) % 101) as f32 / 31.0)
        .collect();
    let centers: Vec<f32> = points[0..2 * dim].to_vec();
    let query: Vec<f32> = (0..dim)
        .map(|index| ((index * 11 + 5) % 97) as f32 / 29.0)
        .collect();
    let state = generate_rotator(dim, 42).unwrap();
    let rotated_centers = rotate_vectors(dim, &state, &centers, 2).unwrap();
    let codec = Rabitq1Codec::new(dim, state, rotated_centers, 2).unwrap();
    let quantized = codec.quantize(&points, 3, 1).unwrap();
    let mut prepared = codec.prepare_query(&query).unwrap();
    let ids = [0_u32, 1, 2];
    let estimates = prepared
        .estimate_ids(&quantized.center_ids, &quantized.bin_data, &ids)
        .unwrap();
    let reference = prepared
        .reference_estimate(
            quantized.center_ids[1],
            &quantized.bin_data[quantized.bin_stride..2 * quantized.bin_stride],
        )
        .unwrap();

    assert_eq!(quantized.center_ids.len(), 3);
    assert_eq!(quantized.bin_data.len(), 3 * quantized.bin_stride);
    assert_eq!(estimates.len(), 3);
    assert_eq!(estimates[1], reference);
    assert!(estimates.iter().all(Rabitq1Estimate::is_finite));
}

#[test]
fn graph_rabitq1_codec_matches_hnsw_get_bin_est_flow() {
    exercise_codec(64);
    exercise_codec(67);
}
