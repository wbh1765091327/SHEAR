use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use crate::common::{AnnError, AnnResult};

const MAGIC: [u8; 8] = *b"HPPCNT01";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 128;
const ELEMENT_WIDTH: u32 = 1;
const BUILD_POLICY_SOFT_TIER_Q3: u32 = 1;
const SAMPLE_SEED: u32 = 42;
const SAMPLE_DENOMINATOR: u32 = 4096;
const UNWRITTEN: u16 = u16::MAX;
const CHUNK_ROWS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum HashPruneGraphEncoding {
    SequentialDegreeOnly = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashPrunePrimaryCountHeader {
    pub rows: u64,
    pub degree_cap: u32,
    pub graph_start: u32,
    pub graph_encoding: HashPruneGraphEncoding,
    pub element_width: u32,
    pub payload_bytes: u64,
    pub payload_sha256: [u8; 32],
    pub build_policy: u32,
    pub sample_seed: u32,
    pub sample_denominator: u32,
}

pub struct HashPrunePrimaryCountWriter {
    final_path: PathBuf,
    partial_path: PathBuf,
    writer: BufWriter<File>,
    rows: u64,
    degree_cap: u32,
    next_row: u64,
    chunk_start: u64,
    chunk_used: usize,
    chunk: Vec<u16>,
    payload_hasher: Sha256,
}

impl HashPrunePrimaryCountWriter {
    pub fn create(path: &Path, rows: usize, degree_cap: usize) -> AnnResult<Self> {
        if degree_cap == 0 || degree_cap > usize::from(u8::MAX) {
            return Err(sidecar_error(format!(
                "primary-count degree cap must be in 1..=255, got {degree_cap}"
            )));
        }
        let rows = u64::try_from(rows)?;
        let degree_cap = u32::try_from(degree_cap)?;
        let partial_path = append_suffix(path, ".partial");
        if path.exists() || partial_path.exists() {
            return Err(sidecar_error(format!(
                "refusing to overwrite primary-count sidecar {} or {}",
                path.display(),
                partial_path.display()
            )));
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&[0_u8; HEADER_BYTES])?;
        Ok(Self {
            final_path: path.to_path_buf(),
            partial_path,
            writer,
            rows,
            degree_cap,
            next_row: 0,
            chunk_start: 0,
            chunk_used: 0,
            chunk: vec![UNWRITTEN; CHUNK_ROWS],
            payload_hasher: Sha256::new(),
        })
    }

    pub fn write_row(
        &mut self,
        source: u64,
        primary_count: usize,
        emitted_degree: usize,
    ) -> AnnResult<()> {
        if source != self.next_row
            || source >= self.rows
            || primary_count > emitted_degree
            || emitted_degree > self.degree_cap as usize
            || primary_count > usize::from(u8::MAX)
        {
            return Err(sidecar_error(format!(
                "invalid primary-count row source={source} expected={} primary={primary_count} degree={emitted_degree} cap={}",
                self.next_row, self.degree_cap
            )));
        }
        self.chunk[self.chunk_used] = primary_count as u16;
        self.chunk_used += 1;
        self.next_row += 1;
        if self.chunk_used == self.chunk.len() {
            self.flush_chunk()?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> AnnResult<()> {
        if self.chunk_used == 0 {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(self.chunk_used);
        for &value in &self.chunk[..self.chunk_used] {
            if value == UNWRITTEN || value > u16::from(u8::MAX) {
                return Err(sidecar_error(format!(
                    "unwritten or invalid primary-count value in chunk starting at {}",
                    self.chunk_start
                )));
            }
            bytes.push(value as u8);
        }
        self.payload_hasher.update(&bytes);
        self.writer.write_all(&bytes)?;
        self.chunk.fill(UNWRITTEN);
        self.chunk_start = self.next_row;
        self.chunk_used = 0;
        Ok(())
    }

    pub fn finish(
        mut self,
        graph_start: u32,
        graph_encoding: HashPruneGraphEncoding,
    ) -> AnnResult<()> {
        if self.next_row != self.rows {
            return Err(sidecar_error(format!(
                "primary-count sidecar is incomplete: wrote {} of {} rows",
                self.next_row, self.rows
            )));
        }
        self.flush_chunk()?;
        let payload_sha256: [u8; 32] = self.payload_hasher.clone().finalize().into();
        let header = HashPrunePrimaryCountHeader {
            rows: self.rows,
            degree_cap: self.degree_cap,
            graph_start,
            graph_encoding,
            element_width: ELEMENT_WIDTH,
            payload_bytes: self.rows,
            payload_sha256,
            build_policy: BUILD_POLICY_SOFT_TIER_Q3,
            sample_seed: SAMPLE_SEED,
            sample_denominator: SAMPLE_DENOMINATOR,
        };
        self.writer.flush()?;
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&encode_header(&header))?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        drop(self.writer);
        fs::rename(&self.partial_path, &self.final_path)?;
        sync_parent(&self.final_path)
    }
}

pub struct HashPrunePrimaryCountSidecar {
    _file: File,
    mmap: Mmap,
    header: HashPrunePrimaryCountHeader,
}

impl HashPrunePrimaryCountSidecar {
    pub fn load(path: &Path, expected_rows: usize) -> AnnResult<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < HEADER_BYTES as u64 {
            return Err(sidecar_error(
                "primary-count sidecar is shorter than its header",
            ));
        }
        let mut header_bytes = [0_u8; HEADER_BYTES];
        file.read_exact(&mut header_bytes)?;
        let header = decode_header(&header_bytes)?;
        if header.rows != u64::try_from(expected_rows)? {
            return Err(sidecar_error(format!(
                "primary-count row mismatch: header={} expected={expected_rows}",
                header.rows
            )));
        }
        let expected_len = (HEADER_BYTES as u64)
            .checked_add(header.payload_bytes)
            .ok_or_else(|| sidecar_error("primary-count sidecar length overflow"))?;
        if file_len != expected_len || header.payload_bytes != header.rows {
            return Err(sidecar_error(format!(
                "primary-count sidecar length mismatch: file={file_len} expected={expected_len} payload={} rows={}",
                header.payload_bytes, header.rows
            )));
        }
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let payload = &mmap[HEADER_BYTES..];
        let actual_sha256: [u8; 32] = Sha256::digest(payload).into();
        if actual_sha256 != header.payload_sha256 {
            return Err(sidecar_error("primary-count payload SHA-256 mismatch"));
        }
        Ok(Self {
            _file: file,
            mmap,
            header,
        })
    }

    pub fn header(&self) -> &HashPrunePrimaryCountHeader {
        &self.header
    }

    pub fn counts(&self) -> &[u8] {
        &self.mmap[HEADER_BYTES..]
    }

    pub fn primary_count(&self, source: usize) -> Option<usize> {
        self.counts().get(source).copied().map(usize::from)
    }

    pub fn validate_graph_degrees<I>(&self, graph_start: u32, degrees: I) -> AnnResult<()>
    where
        I: IntoIterator<Item = u32>,
    {
        if self.header.graph_encoding != HashPruneGraphEncoding::SequentialDegreeOnly {
            return Err(sidecar_error(
                "primary-count sidecar graph encoding mismatch",
            ));
        }
        if graph_start != self.header.graph_start {
            return Err(sidecar_error(format!(
                "primary-count graph start mismatch: sidecar={} graph={graph_start}",
                self.header.graph_start
            )));
        }
        let mut degrees = degrees.into_iter();
        for (source, &primary_count) in self.counts().iter().enumerate() {
            let degree = degrees.next().ok_or_else(|| {
                sidecar_error(format!("graph degree stream ended before source {source}"))
            })?;
            if u32::from(primary_count) > degree || degree > self.header.degree_cap {
                return Err(sidecar_error(format!(
                    "invalid graph boundary at source {source}: primary={primary_count} degree={degree} cap={}",
                    self.header.degree_cap
                )));
            }
        }
        if degrees.next().is_some() {
            return Err(sidecar_error(
                "graph degree stream contains more rows than the primary-count sidecar",
            ));
        }
        Ok(())
    }

    pub fn resident_bytes(&self) -> usize {
        self.mmap.len()
    }
}

fn encode_header(header: &HashPrunePrimaryCountHeader) -> [u8; HEADER_BYTES] {
    let mut bytes = [0_u8; HEADER_BYTES];
    bytes[0..8].copy_from_slice(&MAGIC);
    put_u32(&mut bytes, 8, VERSION);
    put_u32(&mut bytes, 12, HEADER_BYTES as u32);
    put_u64(&mut bytes, 16, header.rows);
    put_u32(&mut bytes, 24, header.degree_cap);
    put_u32(&mut bytes, 28, header.graph_start);
    put_u32(&mut bytes, 32, header.graph_encoding as u32);
    put_u32(&mut bytes, 36, header.element_width);
    put_u64(&mut bytes, 40, header.payload_bytes);
    bytes[48..80].copy_from_slice(&header.payload_sha256);
    put_u32(&mut bytes, 80, header.build_policy);
    put_u32(&mut bytes, 84, header.sample_seed);
    put_u32(&mut bytes, 88, header.sample_denominator);
    bytes
}

fn decode_header(bytes: &[u8; HEADER_BYTES]) -> AnnResult<HashPrunePrimaryCountHeader> {
    if bytes[0..8] != MAGIC {
        return Err(sidecar_error("invalid primary-count sidecar magic"));
    }
    if get_u32(bytes, 8) != VERSION || get_u32(bytes, 12) != HEADER_BYTES as u32 {
        return Err(sidecar_error(
            "unsupported primary-count sidecar version or header size",
        ));
    }
    let graph_encoding = match get_u32(bytes, 32) {
        value if value == HashPruneGraphEncoding::SequentialDegreeOnly as u32 => {
            HashPruneGraphEncoding::SequentialDegreeOnly
        }
        value => {
            return Err(sidecar_error(format!(
                "unsupported primary-count graph encoding {value}"
            )));
        }
    };
    let element_width = get_u32(bytes, 36);
    let build_policy = get_u32(bytes, 80);
    let sample_seed = get_u32(bytes, 84);
    let sample_denominator = get_u32(bytes, 88);
    if element_width != ELEMENT_WIDTH
        || build_policy != BUILD_POLICY_SOFT_TIER_Q3
        || sample_seed != SAMPLE_SEED
        || sample_denominator != SAMPLE_DENOMINATOR
        || bytes[92..].iter().any(|&value| value != 0)
    {
        return Err(sidecar_error(
            "primary-count sidecar has unsupported policy, sampling, width, or reserved fields",
        ));
    }
    let degree_cap = get_u32(bytes, 24);
    if degree_cap == 0 || degree_cap > u32::from(u8::MAX) {
        return Err(sidecar_error(format!(
            "invalid primary-count degree cap {degree_cap}"
        )));
    }
    let mut payload_sha256 = [0_u8; 32];
    payload_sha256.copy_from_slice(&bytes[48..80]);
    Ok(HashPrunePrimaryCountHeader {
        rows: get_u64(bytes, 16),
        degree_cap,
        graph_start: get_u32(bytes, 28),
        graph_encoding,
        element_width,
        payload_bytes: get_u64(bytes, 40),
        payload_sha256,
        build_policy,
        sample_seed,
        sample_denominator,
    })
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn sync_parent(path: &Path) -> AnnResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| sidecar_error("primary-count output path has no parent"))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
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

fn sidecar_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(message.into())
}
