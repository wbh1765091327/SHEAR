use std::ffi::CStr;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::common::{AnnError, AnnResult};
use crate::model::rabitq::ffi;

fn ffi_error(operation: &str) -> AnnError {
    let detail = unsafe {
        let pointer = ffi::graph_rabitq1_last_error();
        if pointer.is_null() {
            "unknown C++ error".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    AnnError::log_index_error(format!("{operation}: {detail}"))
}

pub fn padded_dim(dim: usize) -> AnnResult<usize> {
    let value = unsafe { ffi::graph_rabitq1_padded_dim(dim) };
    if value == 0 {
        return Err(AnnError::log_index_error(format!(
            "invalid graph RaBitQ1 dimension {dim}"
        )));
    }
    Ok(value)
}

pub fn rotator_bytes(padded: usize) -> AnnResult<usize> {
    let value = unsafe { ffi::graph_rabitq1_rotator_bytes(padded) };
    if value == 0 {
        return Err(AnnError::log_index_error(format!(
            "invalid graph RaBitQ1 padded dimension {padded}"
        )));
    }
    Ok(value)
}

pub fn bin_data_bytes(padded: usize) -> AnnResult<usize> {
    let value = unsafe { ffi::graph_rabitq1_bin_data_bytes(padded) };
    if value == 0 {
        return Err(AnnError::log_index_error(format!(
            "invalid graph RaBitQ1 padded dimension {padded}"
        )));
    }
    Ok(value)
}

pub fn generate_rotator(dim: usize, seed: u64) -> AnnResult<Vec<u8>> {
    let padded = padded_dim(dim)?;
    let mut output = vec![0; rotator_bytes(padded)?];
    let status = unsafe {
        ffi::graph_rabitq1_generate_rotator(dim, seed, output.as_mut_ptr(), output.len())
    };
    if status != 0 {
        return Err(ffi_error("generate graph RaBitQ1 rotator"));
    }
    Ok(output)
}

fn checked_len(rows: usize, stride: usize, label: &str) -> AnnResult<usize> {
    rows.checked_mul(stride)
        .ok_or_else(|| AnnError::log_index_error(format!("graph RaBitQ1 {label} length overflow")))
}

pub fn rotate_vectors(dim: usize, state: &[u8], input: &[f32], rows: usize) -> AnnResult<Vec<f32>> {
    if rows == 0 || input.len() != checked_len(rows, dim, "input")? {
        return Err(AnnError::log_index_error(format!(
            "graph RaBitQ1 input length {} does not match {rows}x{dim}",
            input.len()
        )));
    }
    let padded = padded_dim(dim)?;
    if state.len() != rotator_bytes(padded)? {
        return Err(AnnError::log_index_error(format!(
            "invalid graph RaBitQ1 rotator length {}",
            state.len()
        )));
    }
    let mut output = vec![0.0; checked_len(rows, padded, "rotated output")?];
    let status = unsafe {
        ffi::graph_rabitq1_rotate_vectors(
            dim,
            state.as_ptr(),
            state.len(),
            input.as_ptr(),
            rows,
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error("rotate graph RaBitQ1 vectors"));
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(AnnError::log_index_error(
            "graph RaBitQ1 rotation produced a non-finite value".to_string(),
        ));
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rabitq1Estimate {
    pub est_dist: f32,
    pub low_dist: f32,
}

impl Rabitq1Estimate {
    pub fn is_finite(&self) -> bool {
        self.est_dist.is_finite() && self.low_dist.is_finite()
    }
}

pub struct QuantizedRows {
    pub center_ids: Vec<u8>,
    pub bin_data: Vec<u8>,
    pub bin_stride: usize,
}

pub struct Rabitq1Codec {
    raw: NonNull<std::ffi::c_void>,
    dim: usize,
    padded_dim: usize,
    center_count: usize,
    bin_stride: usize,
}

unsafe impl Send for Rabitq1Codec {}
unsafe impl Sync for Rabitq1Codec {}

impl Rabitq1Codec {
    pub fn new(
        dim: usize,
        state: Vec<u8>,
        rotated_centers: Vec<f32>,
        center_count: usize,
    ) -> AnnResult<Self> {
        let padded = padded_dim(dim)?;
        if center_count == 0 || center_count > 256 {
            return Err(AnnError::log_index_error(
                "graph RaBitQ1 center_count must be in 1..=256".to_string(),
            ));
        }
        if state.len() != rotator_bytes(padded)? {
            return Err(AnnError::log_index_error(format!(
                "invalid graph RaBitQ1 rotator length {}",
                state.len()
            )));
        }
        let expected_centers = checked_len(center_count, padded, "center")?;
        if rotated_centers.len() != expected_centers {
            return Err(AnnError::log_index_error(format!(
                "graph RaBitQ1 center length {} does not match {center_count}x{padded}",
                rotated_centers.len()
            )));
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::graph_rabitq1_codec_create(
                dim,
                state.as_ptr(),
                state.len(),
                rotated_centers.as_ptr(),
                center_count,
                &mut raw,
            )
        };
        if status != 0 {
            return Err(ffi_error("create graph RaBitQ1 codec"));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            AnnError::log_index_error("graph RaBitQ1 codec returned a null handle".to_string())
        })?;
        Ok(Self {
            raw,
            dim,
            padded_dim: padded,
            center_count,
            bin_stride: bin_data_bytes(padded)?,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    pub fn center_count(&self) -> usize {
        self.center_count
    }

    pub fn bin_stride(&self) -> usize {
        self.bin_stride
    }

    pub fn quantize(&self, input: &[f32], rows: usize, threads: usize) -> AnnResult<QuantizedRows> {
        if rows == 0 || threads == 0 || input.len() != checked_len(rows, self.dim, "input")? {
            return Err(AnnError::log_index_error(
                "invalid graph RaBitQ1 quantization dimensions or thread count".to_string(),
            ));
        }
        let mut center_ids = vec![0; rows];
        let mut bin_data = vec![0; checked_len(rows, self.bin_stride, "BinData")?];
        let status = unsafe {
            ffi::graph_rabitq1_quantize(
                self.raw.as_ptr(),
                input.as_ptr(),
                rows,
                threads,
                center_ids.as_mut_ptr(),
                bin_data.as_mut_ptr(),
                bin_data.len(),
            )
        };
        if status != 0 {
            return Err(ffi_error("quantize graph RaBitQ1 vectors"));
        }
        Ok(QuantizedRows {
            center_ids,
            bin_data,
            bin_stride: self.bin_stride,
        })
    }

    pub fn prepare_query(&self, query: &[f32]) -> AnnResult<Rabitq1Query<'_>> {
        if query.len() != self.dim {
            return Err(AnnError::log_index_error(format!(
                "graph RaBitQ1 query length {} does not match {}",
                query.len(),
                self.dim
            )));
        }
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::graph_rabitq1_query_create(
                self.raw.as_ptr(),
                query.as_ptr(),
                query.len(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(ffi_error("create graph RaBitQ1 query"));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            AnnError::log_index_error("graph RaBitQ1 query returned a null handle".to_string())
        })?;
        Ok(Rabitq1Query {
            raw,
            marker: PhantomData,
        })
    }
}

impl Drop for Rabitq1Codec {
    fn drop(&mut self) {
        unsafe { ffi::graph_rabitq1_codec_destroy(self.raw.as_ptr()) };
    }
}

pub struct Rabitq1Query<'codec> {
    raw: NonNull<std::ffi::c_void>,
    marker: PhantomData<&'codec Rabitq1Codec>,
}

impl Rabitq1Query<'_> {
    pub fn estimate_ids(
        &mut self,
        center_ids: &[u8],
        bin_data: &[u8],
        ids: &[u32],
    ) -> AnnResult<Vec<Rabitq1Estimate>> {
        let rows = center_ids.len();
        if rows == 0 || ids.is_empty() || bin_data.len() % rows != 0 {
            return Err(AnnError::log_index_error(
                "invalid graph RaBitQ1 estimate buffers".to_string(),
            ));
        }
        let bin_stride = bin_data.len() / rows;
        let mut output = vec![ffi::GraphRabitq1Estimate::default(); ids.len()];
        let status = unsafe {
            ffi::graph_rabitq1_estimate_ids(
                self.raw.as_ptr(),
                center_ids.as_ptr(),
                bin_data.as_ptr(),
                rows,
                bin_stride,
                ids.as_ptr(),
                ids.len(),
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(ffi_error("estimate graph RaBitQ1 IDs"));
        }
        output
            .into_iter()
            .map(|estimate| {
                let estimate = Rabitq1Estimate {
                    est_dist: estimate.est_dist,
                    low_dist: estimate.low_dist,
                };
                if !estimate.is_finite() {
                    return Err(AnnError::log_index_error(
                        "graph RaBitQ1 estimate is non-finite".to_string(),
                    ));
                }
                Ok(estimate)
            })
            .collect()
    }

    pub fn reference_estimate(
        &mut self,
        center_id: u8,
        bin_data: &[u8],
    ) -> AnnResult<Rabitq1Estimate> {
        let mut output = ffi::GraphRabitq1Estimate::default();
        let status = unsafe {
            ffi::graph_rabitq1_reference_estimate(
                self.raw.as_ptr(),
                center_id,
                bin_data.as_ptr(),
                bin_data.len(),
                &mut output,
            )
        };
        if status != 0 {
            return Err(ffi_error("reference graph RaBitQ1 estimate"));
        }
        let estimate = Rabitq1Estimate {
            est_dist: output.est_dist,
            low_dist: output.low_dist,
        };
        if !estimate.is_finite() {
            return Err(AnnError::log_index_error(
                "graph RaBitQ1 reference estimate is non-finite".to_string(),
            ));
        }
        Ok(estimate)
    }
}

impl Drop for Rabitq1Query<'_> {
    fn drop(&mut self) {
        unsafe { ffi::graph_rabitq1_query_destroy(self.raw.as_ptr()) };
    }
}
