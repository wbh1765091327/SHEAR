mod bridge;
mod builder;
mod checksum;
mod format;
mod kmeans;

pub use bridge::{
    QuantizedRows, Rabitq1Codec, Rabitq1Estimate, Rabitq1Query, bin_data_bytes, generate_rotator,
    padded_dim, rotate_vectors, rotator_bytes,
};
pub use builder::{BuildRabitq1Options, BuildRabitq1Report, build_rabitq1_sidecar};
pub use checksum::{
    CRC64_ECMA_CHECK_123456789, Crc64, base_window_fingerprint, crc64, crc64_reader,
};
pub use format::{
    RABITQ1_CHECKSUM_CRC64_ECMA, RABITQ1_ENDIAN_TAG, RABITQ1_HEADER_BYTES, RABITQ1_MAGIC,
    RABITQ1_METRIC_SQUARED_L2, RABITQ1_VERSION, RABITQ1_WINDOW_BYTES, RABITQ1_WINDOW_COUNT,
    Rabitq1Header, Rabitq1Sidecar, Rabitq1ValidationReport, Section,
};
pub use kmeans::{deterministic_sample_ids, train_centers};
