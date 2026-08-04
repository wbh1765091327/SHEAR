use std::io::Cursor;
use std::path::{Path, PathBuf};

use shear_core::model::rabitq::graph_search::{
    BuildRabitq1Options, CRC64_ECMA_CHECK_123456789, Rabitq1Codec, Rabitq1Header, Rabitq1Sidecar,
    Section, base_window_fingerprint, build_rabitq1_sidecar, crc64, deterministic_sample_ids,
    generate_rotator, rotate_vectors, train_centers,
};

struct TestSidecar {
    _dir: tempfile::TempDir,
    base_path: PathBuf,
    sidecar_path: PathBuf,
    bytes: Vec<u8>,
}

impl TestSidecar {
    fn new(rows: usize, dim: usize, center_count: usize) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.fbin");
        let sidecar_path = dir.path().join("base.rabitq1");
        let points = (0..rows * dim)
            .map(|index| ((index * 29 + 7) % 251) as f32 / 37.0)
            .collect::<Vec<_>>();
        let mut base_bytes = Vec::with_capacity(8 + points.len() * 4);
        base_bytes.extend_from_slice(&(rows as u32).to_le_bytes());
        base_bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        for value in &points {
            base_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&base_path, &base_bytes).unwrap();

        let centers = points[..center_count * dim].to_vec();
        let state = generate_rotator(dim, 42).unwrap();
        let rotated_centers = rotate_vectors(dim, &state, &centers, center_count).unwrap();
        let codec =
            Rabitq1Codec::new(dim, state.clone(), rotated_centers.clone(), center_count).unwrap();
        let quantized = codec.quantize(&points, rows, 2).unwrap();
        let fingerprint =
            base_window_fingerprint(&mut Cursor::new(&base_bytes), base_bytes.len() as u64)
                .unwrap();
        let mut header = Rabitq1Header::new_layout(
            rows as u64,
            dim as u32,
            center_count as u32,
            42,
            base_bytes.len() as u64,
            crc64(&base_bytes),
            fingerprint,
        )
        .unwrap();
        let mut bytes = vec![0_u8; header.file_bytes as usize];

        let section_range =
            |section: Section| section.offset as usize..(section.offset + section.length) as usize;
        bytes[section_range(header.rotator)].copy_from_slice(&state);
        let center_bytes = rotated_centers
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        bytes[section_range(header.centers)].copy_from_slice(&center_bytes);
        bytes[section_range(header.center_ids)].copy_from_slice(&quantized.center_ids);
        bytes[section_range(header.bin_data)].copy_from_slice(&quantized.bin_data);
        header.rotator.crc64 = crc64(&bytes[section_range(header.rotator)]);
        header.centers.crc64 = crc64(&bytes[section_range(header.centers)]);
        header.center_ids.crc64 = crc64(&bytes[section_range(header.center_ids)]);
        header.bin_data.crc64 = crc64(&bytes[section_range(header.bin_data)]);
        header.payload_crc64 = crc64(&bytes[256..]);
        header.seal_header_crc().unwrap();
        bytes[..256].copy_from_slice(&header.encode().unwrap());
        std::fs::write(&sidecar_path, &bytes).unwrap();

        Self {
            _dir: dir,
            base_path,
            sidecar_path,
            bytes,
        }
    }

    fn assert_rejected(&self, expected: &str, mutate: impl FnOnce(&mut Vec<u8>)) {
        let mut bytes = self.bytes.clone();
        mutate(&mut bytes);
        if bytes.len() >= 256 {
            bytes[200..208].fill(0);
            let header_crc = crc64(&bytes[..256]);
            bytes[200..208].copy_from_slice(&header_crc.to_le_bytes());
        }
        let path = sibling(&self.sidecar_path, expected);
        std::fs::write(&path, bytes).unwrap();
        let error = Rabitq1Sidecar::load(&path, &self.base_path)
            .err()
            .expect("mutated sidecar must be rejected")
            .to_string();
        assert!(
            error.to_ascii_lowercase().contains(expected),
            "expected {expected:?} in {error:?}"
        );
    }

    fn assert_truncation_rejected(&self) {
        let path = sibling(&self.sidecar_path, "truncated");
        std::fs::write(&path, &self.bytes[..self.bytes.len() - 1]).unwrap();
        let error = Rabitq1Sidecar::load(&path, &self.base_path)
            .err()
            .expect("truncated sidecar must be rejected")
            .to_string();
        assert!(error.to_ascii_lowercase().contains("file"), "{error}");
    }

    fn assert_fingerprint_window_mutation_rejected(&self) {
        let mut base = std::fs::read(&self.base_path).unwrap();
        base[18] ^= 1;
        let path = sibling(&self.base_path, "mutated-base");
        std::fs::write(&path, base).unwrap();
        let error = Rabitq1Sidecar::load(&self.sidecar_path, &path)
            .err()
            .expect("mismatched base must be rejected")
            .to_string();
        assert!(
            error.to_ascii_lowercase().contains("fingerprint"),
            "{error}"
        );
    }
}

fn sibling(path: &Path, label: &str) -> PathBuf {
    path.with_file_name(format!(
        "{label}-{}",
        path.file_name().unwrap().to_string_lossy()
    ))
}

#[test]
fn crc64_ecma_matches_the_standard_check_value() {
    assert_eq!(crc64(b"123456789"), CRC64_ECMA_CHECK_123456789);
}

#[test]
fn rabitq1_header_round_trips_without_struct_padding() {
    let mut header = Rabitq1Header::new_layout(32, 67, 16, 42, 8 + 32 * 67 * 4, 7, 9).unwrap();
    header.rotator.crc64 = 11;
    header.seal_header_crc().unwrap();
    let encoded = header.encode().unwrap();
    assert_eq!(encoded.len(), 256);
    let decoded = Rabitq1Header::decode(&encoded).unwrap();
    assert_eq!(decoded, header);
    assert_eq!(decoded.flags, 0);
    assert_eq!(decoded.bin_stride, 28);
    assert_eq!(decoded.rotator, Section::new(256, 64, 11));
}

#[test]
fn rabitq1_loader_rejects_corruption_and_base_mismatch() {
    let fixture = TestSidecar::new(32, 67, 16);
    fixture.assert_rejected("magic", |bytes| bytes[0] ^= 0x80);
    fixture.assert_rejected("version", |bytes| {
        bytes[8..12].copy_from_slice(&2_u32.to_le_bytes())
    });
    fixture.assert_rejected("flags", |bytes| {
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes())
    });
    fixture.assert_rejected("rows", |bytes| {
        bytes[32..40].copy_from_slice(&31_u64.to_le_bytes())
    });
    fixture.assert_rejected("dimension", |bytes| {
        bytes[40..44].copy_from_slice(&68_u32.to_le_bytes())
    });
    fixture.assert_rejected("section", |bytes| {
        bytes[96..104].copy_from_slice(&257_u64.to_le_bytes())
    });
    fixture.assert_rejected("checksum", |bytes| {
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
    });
    fixture.assert_truncation_rejected();
    fixture.assert_fingerprint_window_mutation_rejected();
}

#[test]
fn sample_ids_are_sorted_unique_and_seeded() {
    let first = deterministic_sample_ids(1_000, 128, 42).unwrap();
    let second = deterministic_sample_ids(1_000, 128, 42).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 128);
    assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(first.iter().all(|&id| id < 1_000));
}

#[test]
fn kmeans_is_byte_deterministic_and_uses_smaller_id_on_ties() {
    let points = [
        0.0_f32, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1,
    ];
    let first = train_centers(&points, 8, 2, 2, 20, 42).unwrap();
    let second = train_centers(&points, 8, 2, 2, 20, 42).unwrap();
    assert_eq!(first, second);
    let mut centers = first.chunks_exact(2).collect::<Vec<_>>();
    centers.sort_by(|left, right| left[0].total_cmp(&right[0]));
    assert_eq!(centers[0], &[0.05, 0.05]);
    assert_eq!(centers[1], &[10.05, 10.05]);
}

#[test]
fn builder_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("tiny.fbin");
    let mut base = Vec::new();
    base.extend_from_slice(&32_u32.to_le_bytes());
    base.extend_from_slice(&67_u32.to_le_bytes());
    for index in 0..32 * 67 {
        let value = ((index * 31 + 11) % 257) as f32 / 43.0;
        base.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&base_path, base).unwrap();

    let first_path = dir.path().join("first.rabitq1");
    let second_path = dir.path().join("second.rabitq1");
    let options = BuildRabitq1Options {
        center_count: 16,
        sample_rows: 32,
        kmeans_iterations: 20,
        seed: 42,
        batch_rows: 7,
        threads: 2,
    };
    let first_report = build_rabitq1_sidecar(&base_path, &first_path, &options).unwrap();
    let second_report = build_rabitq1_sidecar(&base_path, &second_path, &options).unwrap();
    assert_eq!(
        std::fs::read(&first_path).unwrap(),
        std::fs::read(&second_path).unwrap()
    );

    let first = Rabitq1Sidecar::load(&first_path, &base_path).unwrap();
    let second = Rabitq1Sidecar::load(&second_path, &base_path).unwrap();
    assert_eq!(first.header().rows, 32);
    assert_eq!(first.header().dim, 67);
    assert_eq!(first.header().padded_dim, 128);
    assert_eq!(first.header().centroid_count, 16);
    assert!(first.center_ids().iter().all(|&center| center < 16));
    assert_eq!(first.bin_data().len(), 32 * 28);
    assert_eq!(first.header(), second.header());
    assert_eq!(first_report.output_crc64, second_report.output_crc64);

    assert!(!dir.path().join("first.rabitq1.partial").exists());
    assert!(!dir.path().join("second.rabitq1.partial").exists());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("first.rabitq1.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["schema"], "graph_rabitq1_manifest_v1");
    assert_eq!(manifest["ex_data_bytes"], 0);
}
