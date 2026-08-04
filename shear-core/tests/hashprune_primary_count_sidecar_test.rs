use shear_core::model::hashprune_provenance::{
    HashPruneGraphEncoding, HashPrunePrimaryCountSidecar, HashPrunePrimaryCountWriter,
};

#[test]
fn primary_count_sidecar_round_trips_and_binds_graph_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("primary_count.sidecar");
    let mut writer = HashPrunePrimaryCountWriter::create(&path, 4, 64).unwrap();
    writer.write_row(0, 2, 3).unwrap();
    writer.write_row(1, 0, 0).unwrap();
    writer.write_row(2, 4, 4).unwrap();
    writer.write_row(3, 1, 2).unwrap();
    writer
        .finish(17, HashPruneGraphEncoding::SequentialDegreeOnly)
        .unwrap();

    let sidecar = HashPrunePrimaryCountSidecar::load(&path, 4).unwrap();
    assert_eq!(sidecar.header().rows, 4);
    assert_eq!(sidecar.header().degree_cap, 64);
    assert_eq!(sidecar.header().graph_start, 17);
    assert_eq!(sidecar.counts(), &[2, 0, 4, 1]);
    sidecar.validate_graph_degrees(17, [3, 0, 4, 2]).unwrap();
    assert!(sidecar.validate_graph_degrees(18, [3, 0, 4, 2]).is_err());
    assert!(sidecar.validate_graph_degrees(17, [1, 0, 4, 2]).is_err());
}

#[test]
fn primary_count_writer_rejects_missing_duplicate_and_invalid_rows() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.sidecar");
    let mut writer = HashPrunePrimaryCountWriter::create(&missing, 2, 64).unwrap();
    writer.write_row(0, 1, 1).unwrap();
    assert!(
        writer
            .finish(0, HashPruneGraphEncoding::SequentialDegreeOnly)
            .is_err()
    );

    let invalid = dir.path().join("invalid.sidecar");
    let mut writer = HashPrunePrimaryCountWriter::create(&invalid, 1, 64).unwrap();
    assert!(writer.write_row(1, 0, 0).is_err());
    assert!(writer.write_row(0, 2, 1).is_err());
}

#[test]
fn primary_count_loader_rejects_truncation_and_checksum_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid.sidecar");
    let mut writer = HashPrunePrimaryCountWriter::create(&path, 3, 8).unwrap();
    for source in 0..3_u64 {
        writer
            .write_row(source, source as usize, source as usize)
            .unwrap();
    }
    writer
        .finish(1, HashPruneGraphEncoding::SequentialDegreeOnly)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();

    let truncated = dir.path().join("truncated.sidecar");
    std::fs::write(&truncated, &bytes[..bytes.len() - 1]).unwrap();
    assert!(HashPrunePrimaryCountSidecar::load(&truncated, 3).is_err());

    let corrupt = dir.path().join("corrupt.sidecar");
    let mut corrupt_bytes = bytes;
    *corrupt_bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&corrupt, corrupt_bytes).unwrap();
    assert!(HashPrunePrimaryCountSidecar::load(&corrupt, 3).is_err());
}
