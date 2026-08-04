use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use super::bridge::{Rabitq1Codec, Rabitq1Query, bin_data_bytes, padded_dim, rotator_bytes};
use super::checksum::{base_window_fingerprint, crc64};
use crate::common::{AnnError, AnnResult};

pub const RABITQ1_MAGIC: [u8; 8] = *b"RBTQ1\0\0\0";
pub const RABITQ1_VERSION: u32 = 1;
pub const RABITQ1_HEADER_BYTES: usize = 256;
pub const RABITQ1_ENDIAN_TAG: u32 = 0x0102_0304;
pub const RABITQ1_METRIC_SQUARED_L2: u32 = 1;
pub const RABITQ1_CHECKSUM_CRC64_ECMA: u32 = 1;
pub const RABITQ1_WINDOW_BYTES: u32 = 1_048_576;
pub const RABITQ1_WINDOW_COUNT: u32 = 3;
const RABITQ1_QUANTIZER_VERSION: u32 = 1;
const RABITQ1_FINGERPRINT_ALGORITHM: u32 = 1;
const SECTION_ALIGNMENT: u64 = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Section {
    pub offset: u64,
    pub length: u64,
    pub crc64: u64,
}

impl Section {
    pub const fn new(offset: u64, length: u64, crc64: u64) -> Self {
        Self {
            offset,
            length,
            crc64,
        }
    }

    fn end(self) -> AnnResult<u64> {
        self.offset.checked_add(self.length).ok_or_else(|| {
            AnnError::log_index_error("graph RaBitQ1 section end overflow".to_string())
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rabitq1Header {
    pub magic: [u8; 8],
    pub version: u32,
    pub header_bytes: u32,
    pub endian_tag: u32,
    pub flags: u32,
    pub metric: u32,
    pub quantizer_version: u32,
    pub rows: u64,
    pub dim: u32,
    pub padded_dim: u32,
    pub centroid_count: u32,
    pub center_id_bytes: u32,
    pub code_bytes: u32,
    pub bin_stride: u32,
    pub seed: u64,
    pub base_file_bytes: u64,
    pub base_full_crc64: u64,
    pub base_window_fingerprint: u64,
    pub rotator: Section,
    pub centers: Section,
    pub center_ids: Section,
    pub bin_data: Section,
    pub payload_crc64: u64,
    pub header_crc64: u64,
    pub file_bytes: u64,
    pub checksum_algorithm: u32,
    pub fingerprint_algorithm: u32,
    pub window_bytes: u32,
    pub window_count: u32,
    pub reserved: [u8; 24],
}

fn format_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(format!("graph RaBitQ1 header: {}", message.into()))
}

fn checked_mul(left: u64, right: u64, label: &str) -> AnnResult<u64> {
    left.checked_mul(right)
        .ok_or_else(|| format_error(format!("{label} length overflow")))
}

fn align64(value: u64) -> AnnResult<u64> {
    value
        .checked_add(SECTION_ALIGNMENT - 1)
        .map(|value| value / SECTION_ALIGNMENT * SECTION_ALIGNMENT)
        .ok_or_else(|| format_error("section alignment overflow"))
}

fn put_u32(bytes: &mut [u8; RABITQ1_HEADER_BYTES], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8; RABITQ1_HEADER_BYTES], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed header field"),
    )
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed header field"),
    )
}

fn put_section(bytes: &mut [u8; RABITQ1_HEADER_BYTES], offset: usize, section: Section) {
    put_u64(bytes, offset, section.offset);
    put_u64(bytes, offset + 8, section.length);
    put_u64(bytes, offset + 16, section.crc64);
}

fn get_section(bytes: &[u8], offset: usize) -> Section {
    Section::new(
        get_u64(bytes, offset),
        get_u64(bytes, offset + 8),
        get_u64(bytes, offset + 16),
    )
}

impl Rabitq1Header {
    pub fn new_layout(
        rows: u64,
        dim: u32,
        centroid_count: u32,
        seed: u64,
        base_file_bytes: u64,
        base_full_crc64: u64,
        base_window_fingerprint: u64,
    ) -> AnnResult<Self> {
        if !cfg!(target_endian = "little") {
            return Err(format_error("little-endian host required"));
        }
        let padded = u32::try_from(padded_dim(dim as usize)?)
            .map_err(|_| format_error("padded dimension does not fit the version-1 header"))?;
        let rotator_length = u64::try_from(rotator_bytes(padded as usize)?)
            .map_err(|_| format_error("rotator length does not fit the version-1 header"))?;
        let bin_stride = u32::try_from(bin_data_bytes(padded as usize)?)
            .map_err(|_| format_error("BinData stride does not fit the version-1 header"))?;
        let code_bytes = padded / 8;
        let centers_length = checked_mul(
            checked_mul(u64::from(centroid_count), u64::from(padded), "center")?,
            4,
            "center",
        )?;
        let bin_data_length = checked_mul(rows, u64::from(bin_stride), "BinData")?;

        let rotator = Section::new(RABITQ1_HEADER_BYTES as u64, rotator_length, 0);
        let centers = Section::new(align64(rotator.end()?)?, centers_length, 0);
        let center_ids = Section::new(align64(centers.end()?)?, rows, 0);
        let bin_data = Section::new(align64(center_ids.end()?)?, bin_data_length, 0);
        let file_bytes = bin_data.end()?;

        let header = Self {
            magic: RABITQ1_MAGIC,
            version: RABITQ1_VERSION,
            header_bytes: RABITQ1_HEADER_BYTES as u32,
            endian_tag: RABITQ1_ENDIAN_TAG,
            flags: 0,
            metric: RABITQ1_METRIC_SQUARED_L2,
            quantizer_version: RABITQ1_QUANTIZER_VERSION,
            rows,
            dim,
            padded_dim: padded,
            centroid_count,
            center_id_bytes: 1,
            code_bytes,
            bin_stride,
            seed,
            base_file_bytes,
            base_full_crc64,
            base_window_fingerprint,
            rotator,
            centers,
            center_ids,
            bin_data,
            payload_crc64: 0,
            header_crc64: 0,
            file_bytes,
            checksum_algorithm: RABITQ1_CHECKSUM_CRC64_ECMA,
            fingerprint_algorithm: RABITQ1_FINGERPRINT_ALGORITHM,
            window_bytes: RABITQ1_WINDOW_BYTES,
            window_count: RABITQ1_WINDOW_COUNT,
            reserved: [0; 24],
        };
        header.validate_structure()?;
        Ok(header)
    }

    fn encode_fields(&self) -> [u8; RABITQ1_HEADER_BYTES] {
        let mut bytes = [0_u8; RABITQ1_HEADER_BYTES];
        bytes[0..8].copy_from_slice(&self.magic);
        put_u32(&mut bytes, 8, self.version);
        put_u32(&mut bytes, 12, self.header_bytes);
        put_u32(&mut bytes, 16, self.endian_tag);
        put_u32(&mut bytes, 20, self.flags);
        put_u32(&mut bytes, 24, self.metric);
        put_u32(&mut bytes, 28, self.quantizer_version);
        put_u64(&mut bytes, 32, self.rows);
        put_u32(&mut bytes, 40, self.dim);
        put_u32(&mut bytes, 44, self.padded_dim);
        put_u32(&mut bytes, 48, self.centroid_count);
        put_u32(&mut bytes, 52, self.center_id_bytes);
        put_u32(&mut bytes, 56, self.code_bytes);
        put_u32(&mut bytes, 60, self.bin_stride);
        put_u64(&mut bytes, 64, self.seed);
        put_u64(&mut bytes, 72, self.base_file_bytes);
        put_u64(&mut bytes, 80, self.base_full_crc64);
        put_u64(&mut bytes, 88, self.base_window_fingerprint);
        put_section(&mut bytes, 96, self.rotator);
        put_section(&mut bytes, 120, self.centers);
        put_section(&mut bytes, 144, self.center_ids);
        put_section(&mut bytes, 168, self.bin_data);
        put_u64(&mut bytes, 192, self.payload_crc64);
        put_u64(&mut bytes, 200, self.header_crc64);
        put_u64(&mut bytes, 208, self.file_bytes);
        put_u32(&mut bytes, 216, self.checksum_algorithm);
        put_u32(&mut bytes, 220, self.fingerprint_algorithm);
        put_u32(&mut bytes, 224, self.window_bytes);
        put_u32(&mut bytes, 228, self.window_count);
        bytes[232..256].copy_from_slice(&self.reserved);
        bytes
    }

    fn calculated_header_crc(&self) -> u64 {
        let mut bytes = self.encode_fields();
        bytes[200..208].fill(0);
        crc64(&bytes)
    }

    pub fn seal_header_crc(&mut self) -> AnnResult<()> {
        self.validate_structure()?;
        self.header_crc64 = self.calculated_header_crc();
        Ok(())
    }

    pub fn encode(&self) -> AnnResult<[u8; RABITQ1_HEADER_BYTES]> {
        self.validate_structure()?;
        if self.header_crc64 != self.calculated_header_crc() {
            return Err(format_error("header checksum mismatch"));
        }
        Ok(self.encode_fields())
    }

    pub fn decode(bytes: &[u8]) -> AnnResult<Self> {
        if bytes.len() != RABITQ1_HEADER_BYTES {
            return Err(format_error(format!(
                "header byte length {} is not {RABITQ1_HEADER_BYTES}",
                bytes.len()
            )));
        }
        let header = Self {
            magic: bytes[0..8].try_into().expect("fixed header field"),
            version: get_u32(bytes, 8),
            header_bytes: get_u32(bytes, 12),
            endian_tag: get_u32(bytes, 16),
            flags: get_u32(bytes, 20),
            metric: get_u32(bytes, 24),
            quantizer_version: get_u32(bytes, 28),
            rows: get_u64(bytes, 32),
            dim: get_u32(bytes, 40),
            padded_dim: get_u32(bytes, 44),
            centroid_count: get_u32(bytes, 48),
            center_id_bytes: get_u32(bytes, 52),
            code_bytes: get_u32(bytes, 56),
            bin_stride: get_u32(bytes, 60),
            seed: get_u64(bytes, 64),
            base_file_bytes: get_u64(bytes, 72),
            base_full_crc64: get_u64(bytes, 80),
            base_window_fingerprint: get_u64(bytes, 88),
            rotator: get_section(bytes, 96),
            centers: get_section(bytes, 120),
            center_ids: get_section(bytes, 144),
            bin_data: get_section(bytes, 168),
            payload_crc64: get_u64(bytes, 192),
            header_crc64: get_u64(bytes, 200),
            file_bytes: get_u64(bytes, 208),
            checksum_algorithm: get_u32(bytes, 216),
            fingerprint_algorithm: get_u32(bytes, 220),
            window_bytes: get_u32(bytes, 224),
            window_count: get_u32(bytes, 228),
            reserved: bytes[232..256].try_into().expect("fixed header field"),
        };
        header.validate_structure()?;
        if header.header_crc64 != header.calculated_header_crc() {
            return Err(format_error("header checksum mismatch"));
        }
        Ok(header)
    }

    pub fn validate_file_bytes(&self, actual_file_bytes: u64) -> AnnResult<()> {
        if self.file_bytes != actual_file_bytes {
            return Err(format_error(format!(
                "file byte length {actual_file_bytes} does not match {}",
                self.file_bytes
            )));
        }
        Ok(())
    }

    fn validate_structure(&self) -> AnnResult<()> {
        if !cfg!(target_endian = "little") {
            return Err(format_error("little-endian host required"));
        }
        if self.magic != RABITQ1_MAGIC {
            return Err(format_error("magic mismatch"));
        }
        if self.version != RABITQ1_VERSION {
            return Err(format_error("version mismatch"));
        }
        if self.header_bytes != RABITQ1_HEADER_BYTES as u32 {
            return Err(format_error("header size mismatch"));
        }
        if self.endian_tag != RABITQ1_ENDIAN_TAG {
            return Err(format_error("endian tag mismatch"));
        }
        if self.flags != 0 {
            return Err(format_error("flags must be zero"));
        }
        if self.metric != RABITQ1_METRIC_SQUARED_L2 {
            return Err(format_error("metric mismatch"));
        }
        if self.quantizer_version != RABITQ1_QUANTIZER_VERSION {
            return Err(format_error("quantizer version mismatch"));
        }
        if self.checksum_algorithm != RABITQ1_CHECKSUM_CRC64_ECMA {
            return Err(format_error("checksum algorithm mismatch"));
        }
        if self.fingerprint_algorithm != RABITQ1_FINGERPRINT_ALGORITHM {
            return Err(format_error("fingerprint algorithm mismatch"));
        }
        if self.window_bytes != RABITQ1_WINDOW_BYTES || self.window_count != RABITQ1_WINDOW_COUNT {
            return Err(format_error("fingerprint window configuration mismatch"));
        }
        if self.reserved != [0; 24] {
            return Err(format_error("reserved bytes must be zero"));
        }
        if self.rows == 0 {
            return Err(format_error("rows must be nonzero"));
        }
        if self.dim == 0 {
            return Err(format_error("dimension must be nonzero"));
        }
        if self.centroid_count == 0 || self.centroid_count > 256 {
            return Err(format_error("centroid count must be in 1..=256"));
        }
        if self.center_id_bytes != 1 {
            return Err(format_error("center ID width must be one byte"));
        }

        let expected_padded = u32::try_from(padded_dim(self.dim as usize)?)
            .map_err(|_| format_error("padded dimension does not fit the version-1 header"))?;
        if self.padded_dim != expected_padded {
            return Err(format_error("padded dimension mismatch"));
        }
        if self.code_bytes != self.padded_dim / 8 {
            return Err(format_error("code byte length mismatch"));
        }
        if self.bin_stride != self.code_bytes + 12 {
            return Err(format_error("BinData stride mismatch"));
        }

        let expected_rotator_length = u64::try_from(rotator_bytes(self.padded_dim as usize)?)
            .map_err(|_| format_error("rotator length does not fit the version-1 header"))?;
        let expected_centers_length = checked_mul(
            checked_mul(
                u64::from(self.centroid_count),
                u64::from(self.padded_dim),
                "center",
            )?,
            4,
            "center",
        )?;
        let expected_bin_length = checked_mul(self.rows, u64::from(self.bin_stride), "BinData")?;
        let expected_rotator = Section::new(
            RABITQ1_HEADER_BYTES as u64,
            expected_rotator_length,
            self.rotator.crc64,
        );
        let expected_centers = Section::new(
            align64(expected_rotator.end()?)?,
            expected_centers_length,
            self.centers.crc64,
        );
        let expected_center_ids = Section::new(
            align64(expected_centers.end()?)?,
            self.rows,
            self.center_ids.crc64,
        );
        let expected_bin_data = Section::new(
            align64(expected_center_ids.end()?)?,
            expected_bin_length,
            self.bin_data.crc64,
        );

        if self.rotator != expected_rotator
            || self.centers != expected_centers
            || self.center_ids != expected_center_ids
            || self.bin_data != expected_bin_data
        {
            return Err(format_error(
                "section layout or rows/dimension-derived length mismatch",
            ));
        }
        for section in [self.rotator, self.centers, self.center_ids, self.bin_data] {
            if section.offset % SECTION_ALIGNMENT != 0 {
                return Err(format_error("section offset is not 64-byte aligned"));
            }
        }
        if self.file_bytes != expected_bin_data.end()? {
            return Err(format_error(
                "file byte length does not match section layout",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Rabitq1ValidationReport {
    pub schema: &'static str,
    pub rows: u64,
    pub dim: u32,
    pub padded_dim: u32,
    pub center_count: u32,
    pub bin_stride: u32,
    pub resident_bytes: usize,
    pub load_seconds: f64,
    pub payload_crc64: u64,
    pub base_full_crc64: u64,
    pub base_window_fingerprint: u64,
    pub ex_data_bytes: u64,
}

impl Rabitq1ValidationReport {
    pub fn to_pretty_json(&self) -> AnnResult<String> {
        serde_json::to_string_pretty(self).map_err(|error| {
            AnnError::log_index_error(format!(
                "serialize graph RaBitQ1 validation report: {error}"
            ))
        })
    }
}

pub struct Rabitq1Sidecar {
    bytes: Box<[u8]>,
    header: Rabitq1Header,
    codec: Rabitq1Codec,
    load_seconds: f64,
}

fn section_slice<'a>(bytes: &'a [u8], section: Section, label: &str) -> AnnResult<&'a [u8]> {
    let start = usize::try_from(section.offset)
        .map_err(|_| format_error(format!("{label} section offset does not fit memory")))?;
    let end_u64 = section.end()?;
    let end = usize::try_from(end_u64)
        .map_err(|_| format_error(format!("{label} section end does not fit memory")))?;
    bytes
        .get(start..end)
        .ok_or_else(|| format_error(format!("{label} section is outside the file")))
}

fn verify_section_crc(bytes: &[u8], section: Section, label: &str) -> AnnResult<()> {
    if crc64(section_slice(bytes, section, label)?) != section.crc64 {
        return Err(format_error(format!("{label} checksum mismatch")));
    }
    Ok(())
}

fn read_fbin_identity(base_path: &Path, header: &Rabitq1Header) -> AnnResult<()> {
    let mut base = File::open(base_path)?;
    let actual_bytes = base.metadata()?.len();
    if actual_bytes != header.base_file_bytes {
        return Err(format_error(format!(
            "base file byte length {actual_bytes} does not match {}",
            header.base_file_bytes
        )));
    }
    let expected_bytes = 8_u64
        .checked_add(
            header
                .rows
                .checked_mul(u64::from(header.dim))
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| format_error("base file length overflow"))?,
        )
        .ok_or_else(|| format_error("base file length overflow"))?;
    if actual_bytes != expected_bytes {
        return Err(format_error(
            "base file byte length does not match rows and dimension",
        ));
    }
    let mut fbin_header = [0_u8; 8];
    base.read_exact(&mut fbin_header)?;
    let rows = u32::from_le_bytes(fbin_header[0..4].try_into().expect("fixed fbin field"));
    let dim = u32::from_le_bytes(fbin_header[4..8].try_into().expect("fixed fbin field"));
    if u64::from(rows) != header.rows {
        return Err(format_error(format!(
            "base rows {rows} do not match {}",
            header.rows
        )));
    }
    if dim != header.dim {
        return Err(format_error(format!(
            "base dimension {dim} does not match {}",
            header.dim
        )));
    }
    let fingerprint = base_window_fingerprint(&mut base, actual_bytes)?;
    if fingerprint != header.base_window_fingerprint {
        return Err(format_error("base fingerprint mismatch"));
    }
    Ok(())
}

fn decode_centers(bytes: &[u8], header: &Rabitq1Header) -> AnnResult<Vec<f32>> {
    let section = section_slice(bytes, header.centers, "centers")?;
    if section.len() % 4 != 0 {
        return Err(format_error(
            "center section is not a sequence of f32 values",
        ));
    }
    section
        .chunks_exact(4)
        .enumerate()
        .map(|(index, chunk)| {
            let value = f32::from_le_bytes(chunk.try_into().expect("four-byte center value"));
            if !value.is_finite() {
                return Err(format_error(format!("center value {index} is non-finite")));
            }
            Ok(value)
        })
        .collect()
}

fn validate_point_records(bytes: &[u8], header: &Rabitq1Header) -> AnnResult<()> {
    let center_ids = section_slice(bytes, header.center_ids, "center IDs")?;
    for (row, &center_id) in center_ids.iter().enumerate() {
        if u32::from(center_id) >= header.centroid_count {
            return Err(format_error(format!(
                "center ID {center_id} for row {row} is outside {} centers",
                header.centroid_count
            )));
        }
    }

    let bin_data = section_slice(bytes, header.bin_data, "BinData")?;
    let stride = header.bin_stride as usize;
    for (row, record) in bin_data.chunks_exact(stride).enumerate() {
        let factors = &record[stride - 12..];
        for (index, name) in ["f_add", "f_rescale", "f_error"].into_iter().enumerate() {
            let start = index * 4;
            let value = f32::from_le_bytes(
                factors[start..start + 4]
                    .try_into()
                    .expect("four-byte RaBitQ factor"),
            );
            if !value.is_finite() {
                return Err(format_error(format!(
                    "BinData row {row} has non-finite {name}"
                )));
            }
        }
    }
    Ok(())
}

impl Rabitq1Sidecar {
    pub fn load(sidecar_path: &Path, base_path: &Path) -> AnnResult<Self> {
        let started = Instant::now();
        let mut file = File::open(sidecar_path)?;
        let file_bytes_u64 = file.metadata()?.len();
        if file_bytes_u64 < RABITQ1_HEADER_BYTES as u64 {
            return Err(format_error("file is shorter than the version-1 header"));
        }
        let file_bytes = usize::try_from(file_bytes_u64)
            .map_err(|_| format_error("file is too large for resident loading"))?;
        let mut bytes = vec![0_u8; file_bytes];
        file.read_exact(&mut bytes)?;
        let header = Rabitq1Header::decode(&bytes[..RABITQ1_HEADER_BYTES])?;
        header.validate_file_bytes(file_bytes_u64)?;

        verify_section_crc(&bytes, header.rotator, "rotator")?;
        verify_section_crc(&bytes, header.centers, "centers")?;
        verify_section_crc(&bytes, header.center_ids, "center IDs")?;
        verify_section_crc(&bytes, header.bin_data, "BinData")?;
        if crc64(&bytes[RABITQ1_HEADER_BYTES..]) != header.payload_crc64 {
            return Err(format_error("payload checksum mismatch"));
        }
        read_fbin_identity(base_path, &header)?;
        validate_point_records(&bytes, &header)?;

        let state = section_slice(&bytes, header.rotator, "rotator")?.to_vec();
        let centers = decode_centers(&bytes, &header)?;
        let codec = Rabitq1Codec::new(
            header.dim as usize,
            state,
            centers,
            header.centroid_count as usize,
        )?;
        Ok(Self {
            bytes: bytes.into_boxed_slice(),
            header,
            codec,
            load_seconds: started.elapsed().as_secs_f64(),
        })
    }

    pub fn header(&self) -> &Rabitq1Header {
        &self.header
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn load_seconds(&self) -> f64 {
        self.load_seconds
    }

    pub fn center_ids(&self) -> &[u8] {
        section_slice(&self.bytes, self.header.center_ids, "center IDs")
            .expect("validated resident sidecar section")
    }

    pub fn bin_data(&self) -> &[u8] {
        section_slice(&self.bytes, self.header.bin_data, "BinData")
            .expect("validated resident sidecar section")
    }

    pub fn prepare_query(&self, query: &[f32]) -> AnnResult<Rabitq1Query<'_>> {
        self.codec.prepare_query(query)
    }

    pub fn validation_report(&self) -> Rabitq1ValidationReport {
        Rabitq1ValidationReport {
            schema: "graph_rabitq1_validation_v1",
            rows: self.header.rows,
            dim: self.header.dim,
            padded_dim: self.header.padded_dim,
            center_count: self.header.centroid_count,
            bin_stride: self.header.bin_stride,
            resident_bytes: self.resident_bytes(),
            load_seconds: self.load_seconds,
            payload_crc64: self.header.payload_crc64,
            base_full_crc64: self.header.base_full_crc64,
            base_window_fingerprint: self.header.base_window_fingerprint,
            ex_data_bytes: 0,
        }
    }
}
