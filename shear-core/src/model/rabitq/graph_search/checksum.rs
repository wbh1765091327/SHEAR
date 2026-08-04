use std::io::{Read, Seek, SeekFrom};

use crate::common::{AnnError, AnnResult};

pub const CRC64_ECMA_CHECK_123456789: u64 = 0x6C40_DF5F_0B49_7347;
const CRC64_ECMA_POLYNOMIAL: u64 = 0x42F0_E1EB_A9EA_3693;
const CRC_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const FINGERPRINT_WINDOW_BYTES: u64 = 1_048_576;

#[derive(Clone, Copy, Debug, Default)]
pub struct Crc64(u64);

impl Crc64 {
    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte) << 56;
            for _ in 0..8 {
                self.0 = if self.0 & (1_u64 << 63) != 0 {
                    (self.0 << 1) ^ CRC64_ECMA_POLYNOMIAL
                } else {
                    self.0 << 1
                };
            }
        }
    }

    pub fn finish(self) -> u64 {
        self.0
    }
}

pub fn crc64(bytes: &[u8]) -> u64 {
    let mut digest = Crc64::default();
    digest.update(bytes);
    digest.finish()
}

pub fn crc64_reader(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut digest = Crc64::default();
    let mut buffer = vec![0_u8; CRC_BUFFER_BYTES];
    loop {
        let bytes = reader.read(&mut buffer)?;
        if bytes == 0 {
            return Ok(digest.finish());
        }
        digest.update(&buffer[..bytes]);
    }
}

pub fn base_window_fingerprint(reader: &mut (impl Read + Seek), file_bytes: u64) -> AnnResult<u64> {
    if file_bytes < 8 {
        return Err(AnnError::log_index_error(
            "graph RaBitQ1 base file is shorter than its fbin header".to_string(),
        ));
    }

    let payload_bytes = file_bytes - 8;
    let window_bytes = payload_bytes.min(FINGERPRINT_WINDOW_BYTES);
    let windows = [
        (0_u64, 8_u64),
        (8_u64, window_bytes),
        (8_u64 + (payload_bytes - window_bytes) / 2, window_bytes),
        (file_bytes - window_bytes, window_bytes),
    ];
    let mut digest = Crc64::default();
    let mut buffer = vec![0_u8; FINGERPRINT_WINDOW_BYTES as usize];
    for (offset, length) in windows {
        digest.update(&offset.to_le_bytes());
        digest.update(&length.to_le_bytes());
        reader.seek(SeekFrom::Start(offset))?;
        let length = usize::try_from(length).map_err(|_| {
            AnnError::log_index_error("graph RaBitQ1 fingerprint window is too large".to_string())
        })?;
        reader.read_exact(&mut buffer[..length])?;
        digest.update(&buffer[..length]);
    }
    Ok(digest.finish())
}
