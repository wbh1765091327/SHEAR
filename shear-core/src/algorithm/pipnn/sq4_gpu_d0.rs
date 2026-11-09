//! One raw upload, original-space D0, and one packed residual code per point.
//! The CUDA FHT/assignment organization is adapted from the existing SQ8+
//! staging implementation in the main checkout; its partition-dependent
//! encoding, reconstructed norm and rotated-space D0 are deliberately replaced.
use std::sync::Arc;
use std::time::Instant;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PinnedHostSlice, PushKernelArg, sys};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use super::dco_rabitq::FhtKacRotationPlan;
use super::sq4_primary::PrimaryChunk;
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug, Default)]
pub(super) struct Profile {
    pub rows: usize,
    pub encoded_memberships: usize,
    pub batches: usize,
    pub host_read_ms: f64,
    pub reader_wait_ms: f64,
    pub assigned_gpu_rows: usize,
    pub exact_gpu_rows: usize,
    pub primary_assignment_gpu_rows: usize,
    pub primary_assignment_d2h_bytes: usize,
    pub uncertain_rows: usize,
    pub checked_cpu_rows: usize,
    pub checked_batches: usize,
    pub corrected_rows: usize,
    pub raw_h2d_bytes: usize,
    pub code_d2h_bytes: usize,
    pub assignment_d2h_bytes: usize,
    pub corrected_ids_h2d_bytes: usize,
    pub workspace_bytes: usize,
    pub prepare_ms: f64,
    pub h2d_ms: f64,
    pub assign_ms: f64,
    pub encode_ms: f64,
    pub d2h_ms: f64,
    pub compatibility_ms: f64,
}

fn error(e: impl std::fmt::Display) -> AnnError {
    AnnError::log_index_error(format!("SQ4+ GPU D0 failed (CUDA errors do not fall back): {e}"))
}

pub(super) struct GpuPrimaryD0 {
    stream: Arc<CudaStream>,
    assign: CudaFunction,
    assign_exact: CudaFunction,
    assign_budget: CudaFunction,
    primary: Option<PrimaryEncoder>,
    exact: bool,
    encode_repeats: usize,
    centers_t: CudaSlice<f32>,
    dot_scores: CudaSlice<f32>,
    encode: CudaFunction,
    input: CudaSlice<f32>,
    pinned: PinnedHostSlice<f32>,
    centers: CudaSlice<f32>,
    center_norms: CudaSlice<f32>,
    signs: CudaSlice<u32>,
    selected: CudaSlice<u32>,
    uncertain: CudaSlice<u32>,
    codes: CudaSlice<i8>,
    scales: CudaSlice<f32>,
    norms: CudaSlice<f32>,
    invalid: CudaSlice<u32>,
    intermediate: CudaSlice<f32>,
    capture: u32,
    dim: usize,
    padded: usize,
    leaders: usize,
    fanout: usize,
    max_rows: usize,
    rotation_scale: f32,
    error_factor: f32,
    underflow_bound: f32,
    centers_subnormal: u32,
    pub profile: Profile,
}

struct PrimaryEncoder {
    centers: CudaSlice<f32>, centers_t: CudaSlice<f32>, norms: CudaSlice<f32>, selected: CudaSlice<u32>,
}

impl GpuPrimaryD0 {
    pub(super) fn new(device: usize, dim: usize, padded: usize, max_rows: usize,
        centers: &[f32], center_norms: &[f32], fanout: usize,
        rotation: &FhtKacRotationPlan, budget: usize) -> AnnResult<Self> {
        let start = Instant::now();
        let leaders = center_norms.len();
        let (signs, rotation_scale) = rotation.gpu_parameters();
        if !dim.is_power_of_two() || dim > 4096 || padded < dim || padded % 2 != 0
            || max_rows == 0 || max_rows > 4096 || leaders == 0 || leaders > u32::MAX as usize
            || centers.len() != leaders * dim || signs.len() != 4 * dim
            || fanout == 0 || fanout > leaders || fanout > 32
            || centers.iter().chain(center_norms).any(|v| !v.is_finite()) {
            return Err(error("invalid shape or nonfinite centers"));
        }
        let workspace_bytes = max_rows * (dim * 4 + fanout * 4 + 4 + padded / 2 + 8)
            + centers.len() * 4 + leaders * 4 + signs.len() * 4 + 16;
        if workspace_bytes > budget { return Err(error("workspace budget exceeded")); }
        let context = CudaContext::new(device).map_err(error)?;
        let stream = context.new_stream().map_err(error)?;
        let ptx = compile_ptx_with_opts(include_str!("sq4_gpu_d0.cu"), CompileOptions {
            ftz: Some(false), fmad: Some(false), options: vec!["--std=c++17".into()],
            ..CompileOptions::default()
        }).map_err(error)?;
        let module = context.load_module(ptx).map_err(error)?;
        let assign = module.load_function("sq4_primary_assign").map_err(error)?;
        let assign_exact = module.load_function("sq4_primary_assign_exact").map_err(error)?;
        let assign_budget = module.load_function("sq4_budget_assign_exact").map_err(error)?;
        let centers_t = stream.alloc_zeros(1).map_err(error)?;
        let dot_scores = stream.alloc_zeros(1).map_err(error)?;
        let encode = module.load_function("sq4_primary_encode").map_err(error)?;
        let input = stream.alloc_zeros(max_rows * dim).map_err(error)?;
        let mut pinned = unsafe { context.alloc_pinned::<f32>(max_rows * dim) }.map_err(error)?;
        pinned.as_mut_slice().map_err(error)?.fill(0.0);
        let centers_subnormal = u32::from(centers.iter().any(|v| v.is_subnormal()));
        let centers = stream.clone_htod(centers).map_err(error)?;
        let center_norms = stream.clone_htod(center_norms).map_err(error)?;
        let signs = stream.clone_htod(signs).map_err(error)?;
        let selected = stream.alloc_zeros(max_rows * fanout).map_err(error)?;
        let uncertain = stream.alloc_zeros(max_rows).map_err(error)?;
        let codes = stream.alloc_zeros(max_rows * (padded / 2)).map_err(error)?;
        let scales = stream.alloc_zeros(max_rows).map_err(error)?;
        let norms = stream.alloc_zeros(max_rows).map_err(error)?;
        let invalid = stream.alloc_zeros(1).map_err(error)?;
        let intermediate = stream.alloc_zeros(1).map_err(error)?;
        // Standard dot-product forward error: each CPU/GPU FP32 evaluation
        // differs from the exact dot by <= gamma_m * sum |x_i*c_i|.
        // m=2*d+32 covers separate multiplies, any summation tree/FMA and
        // BLAS alpha/beta epilogues. Bound their difference by 2*gamma_m.
        // Round coefficients UP and use directed-rounded magnitude sums in
        // CUDA. An additive normal-minimum term also covers underflow/FTZ.
        // This is a conservative certificate, not a data-dependent epsilon.
        let m = (2 * dim + 32) as f64;
        let mu = m * 2.0f64.powi(-24);
        let error_factor = ((2.0 * mu / (1.0 - mu)) as f32).next_up();
        let underflow_bound = ((4.0 * m * f32::MIN_POSITIVE as f64) as f32).next_up();
        stream.synchronize().map_err(error)?;
        let profile = Profile {workspace_bytes, prepare_ms: start.elapsed().as_secs_f64() * 1000.0,
            ..Profile::default()};
        Ok(Self {stream, assign, assign_exact, assign_budget, primary: None, exact: false, encode_repeats: 1, centers_t, dot_scores,
            encode, input, pinned, centers, center_norms, signs,
            selected, uncertain, codes, scales, norms, invalid, intermediate, capture: 0,
            dim, padded, leaders, fanout, max_rows, rotation_scale, error_factor, underflow_bound,
            centers_subnormal, profile})
    }

    /// One residual code for every ordered D0 membership; raw input stays resident.
    pub(super) fn enable_partition_encoding(&mut self, budget: usize) -> AnnResult<()> {
        if self.primary.is_some() || !matches!(self.leaders, 64 | 512) || self.capture != 0 {
            return Err(error("partition encoding requires 64 or 512 shared routing centers"));
        }
        let rows = self.max_rows * self.fanout;
        let extra = self.max_rows * (self.fanout - 1) * (self.padded / 2 + 8);
        if self.profile.workspace_bytes + extra > budget { return Err(error("partition workspace budget exceeded")); }
        self.codes = self.stream.alloc_zeros(rows * (self.padded / 2)).map_err(error)?;
        self.scales = self.stream.alloc_zeros(rows).map_err(error)?;
        self.norms = self.stream.alloc_zeros(rows).map_err(error)?;
        self.profile.workspace_bytes += extra;
        self.encode_repeats = self.fanout;
        Ok(())
    }

    /// Exact compatibility with the pinned ndarray 0.14 / matrixmultiply 0.2.4
    /// FMA path used by LAION. Other shapes/backends are not silently accepted.
    pub(super) fn enable_exact_assignment(&mut self, centers: &[f32], budget: usize) -> AnnResult<()> {
        if self.dim != 512 || centers.len() != 512 * self.leaders {
            return Err(error("gpu-exact requires dim=512 and consistent routing centers"));
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut mxcsr = 0u32;
            unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags)); }
            if !std::is_x86_feature_detected!("fma") || mxcsr & 0xe040 != 0 {
                return Err(error("gpu-exact requires CPU FMA, round-to-nearest and no FTZ/DAZ"));
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        return Err(error("gpu-exact is validated against the x86_64 FMA reference"));
        let bytes = centers.len() * 4;
        if self.profile.workspace_bytes + bytes > budget { return Err(error("exact workspace budget exceeded")); }
        let leaders = self.leaders;
        let transposed = (0..512).flat_map(|axis| (0..leaders).map(move |c| centers[c * 512 + axis]))
            .collect::<Vec<_>>();
        self.centers_t = self.stream.clone_htod(&transposed).map_err(error)?;
        self.stream.synchronize().map_err(error)?;
        self.profile.workspace_bytes += bytes;
        self.exact = true;
        Ok(())
    }

    /// Keep the validated 64-center quantizer independent of budget routing.
    pub(super) fn set_primary_centers(&mut self, centers: &[f32], budget: usize) -> AnnResult<()> {
        if self.dim != 512 || centers.len() != 512 * 64 || !self.exact {
            return Err(error("separate primary codebook requires 512D gpu-exact and 64 centers"));
        }
        let bytes = centers.len()*8 + 64*4 + self.max_rows*4;
        if self.profile.workspace_bytes + bytes > budget { return Err(error("primary codebook exceeds workspace budget")); }
        let norms: Vec<f32> = centers.chunks_exact(512).map(|r|r.iter().map(|v|v*v).sum()).collect();
        let transposed: Vec<f32> = (0..512).flat_map(|axis|(0..64).map(move |c|centers[c*512+axis])).collect();
        self.primary = Some(PrimaryEncoder {
            centers:self.stream.clone_htod(centers).map_err(error)?,
            centers_t:self.stream.clone_htod(&transposed).map_err(error)?,
            norms:self.stream.clone_htod(&norms).map_err(error)?,
            selected:self.stream.alloc_zeros(self.max_rows).map_err(error)?,
        });
        self.profile.workspace_bytes += bytes;
        Ok(())
    }

    /// `reference` MUST invoke the original CPU GEMM on this entire unchanged
    /// batch (same dimensions, order and stride), not a compacted suspect subset.
    pub(super) fn stage(&mut self, points: &[f32], assign_on_gpu: bool,
        reference: impl FnOnce() -> AnnResult<Vec<u32>>) -> AnnResult<(Vec<u32>, PrimaryChunk)> {
        if points.len() % self.dim != 0 || points.is_empty()
            || points.len() / self.dim > self.max_rows || points.iter().any(|v| !v.is_finite()) {
            return Err(error("invalid or nonfinite batch"));
        }
        if self.leaders > 256 && self.primary.is_none() && self.encode_repeats == 1 {
            return Err(error("routing above 256 centers requires partition encoding or a separate primary codebook"));
        }
        if self.primary.is_some() && !assign_on_gpu { return Err(error("separate primary codebook requires GPU routing")); }
        let rows = points.len() / self.dim;
        let timing_stream = self.stream.clone();
        let record = || timing_stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(error);
        self.pinned.as_mut_slice().map_err(error)?[..points.len()].copy_from_slice(points);
        let upload = record()?;
        self.stream.memcpy_htod(&self.pinned.as_slice().map_err(error)?[..points.len()],
            &mut self.input.slice_mut(..points.len())).map_err(error)?;
        let uploaded = record()?;
        let (n, d, padded, leaders, k) = (rows as u32, self.dim as u32,
            self.padded as u32, self.leaders as u32, self.fanout as u32);
        let mut selected;
        let mut flags = Vec::new();
        if assign_on_gpu {
            let begin = record()?;
            if self.exact {
                let function = if self.leaders == 64 { &self.assign_exact } else { &self.assign_budget };
                let mut launch = self.stream.launch_builder(function);
                launch.arg(&self.input).arg(&self.centers_t).arg(&self.center_norms).arg(&n);
                if self.leaders != 64 { launch.arg(&leaders); }
                launch.arg(&k).arg(&mut self.selected).arg(&self.capture).arg(&mut self.dot_scores);
                unsafe { launch.launch(LaunchConfig {grid_dim: (n.div_ceil(4),1,1),
                    block_dim: (256,1,1), shared_mem_bytes: 0}) }.map_err(error)?;
                self.profile.exact_gpu_rows += rows;
            } else {
            let mut launch = self.stream.launch_builder(&self.assign);
            launch.arg(&self.input).arg(&self.centers).arg(&self.center_norms)
                .arg(&n).arg(&d).arg(&leaders).arg(&k).arg(&self.error_factor)
                .arg(&self.underflow_bound).arg(&self.centers_subnormal)
                .arg(&mut self.selected).arg(&mut self.uncertain);
            unsafe { launch.launch(LaunchConfig {grid_dim: (n,1,1), block_dim: (256,1,1),
                shared_mem_bytes: (d + 3 * leaders) * 4}) }.map_err(error)?;
            }
            let assigned = record()?;
            selected = self.stream.clone_dtoh(&self.selected.slice(..rows * self.fanout)).map_err(error)?;
            if !self.exact { flags = self.stream.clone_dtoh(&self.uncertain.slice(..rows)).map_err(error)?; }
            let downloaded = record()?;
            downloaded.synchronize().map_err(error)?;
            self.profile.assign_ms += begin.elapsed_ms(&assigned).map_err(error)? as f64;
            self.profile.d2h_ms += assigned.elapsed_ms(&downloaded).map_err(error)? as f64;
            self.profile.assigned_gpu_rows += rows;
            self.profile.assignment_d2h_bytes += (rows * self.fanout + if self.exact {0} else {rows}) * 4;
        } else {
            selected = Vec::new();
        }
        if !assign_on_gpu || flags.iter().any(|&v| v != 0) {
            let check = Instant::now();
            let expected = reference()?;
            if expected.len() != rows * self.fanout || expected.iter().any(|&id| id as usize >= self.leaders) {
                return Err(error("invalid CPU compatibility assignments"));
            }
            self.profile.compatibility_ms += check.elapsed().as_secs_f64() * 1000.0;
            self.profile.checked_cpu_rows += rows;
            self.profile.checked_batches += 1;
            if assign_on_gpu {
                self.profile.uncertain_rows += flags.iter().filter(|&&v| v != 0).count();
                for row in 0..rows {
                    let a = &selected[row*self.fanout..(row+1)*self.fanout];
                    let b = &expected[row*self.fanout..(row+1)*self.fanout];
                    if a != b {
                        if flags[row] == 0 { return Err(error("D0 interval certificate violated CPU ordering")); }
                        self.profile.corrected_rows += 1;
                    }
                }
            }
            selected = expected;
            self.stream.memcpy_htod(&selected, &mut self.selected.slice_mut(..selected.len())).map_err(error)?;
            self.profile.corrected_ids_h2d_bytes += selected.len() * 4;
        }
        let primary_ids = if let Some(primary) = self.primary.as_mut() {
            let begin = record()?;
            let mut launch = self.stream.launch_builder(&self.assign_exact);
            let one = 1u32; let capture = 0u32;
            launch.arg(&self.input).arg(&primary.centers_t).arg(&primary.norms)
                .arg(&n).arg(&one).arg(&mut primary.selected).arg(&capture).arg(&mut self.dot_scores);
            unsafe {launch.launch(LaunchConfig {grid_dim:(n.div_ceil(4),1,1),block_dim:(256,1,1),shared_mem_bytes:0})}.map_err(error)?;
            let assigned = record()?;
            let ids = self.stream.clone_dtoh(&primary.selected.slice(..rows)).map_err(error)?;
            let downloaded = record()?; downloaded.synchronize().map_err(error)?;
            self.profile.assign_ms += begin.elapsed_ms(&assigned).map_err(error)? as f64;
            self.profile.d2h_ms += assigned.elapsed_ms(&downloaded).map_err(error)? as f64;
            self.profile.primary_assignment_gpu_rows += rows;
            self.profile.primary_assignment_d2h_bytes += rows*4;
            Some(ids)
        } else { None };
        // CUDA/validation errors are fatal. The only CPU work above is the
        // explicit compatibility check; residual encoding always stays on GPU.
        self.stream.memcpy_htod(&[0u32], &mut self.invalid).map_err(error)?;
        let begin = record()?;
        let output_rows = rows * self.encode_repeats;
        let repeats = self.encode_repeats as u32;
        let mut launch = self.stream.launch_builder(&self.encode);
        let (encode_centers, encode_selected, encode_k) = if let Some(primary) = self.primary.as_ref() {
            (&primary.centers, &primary.selected, 1u32)
        } else { (&self.centers, &self.selected, k) };
        launch.arg(&self.input).arg(encode_centers).arg(encode_selected).arg(&self.signs)
            .arg(&n).arg(&d).arg(&padded).arg(&encode_k).arg(&repeats).arg(&self.rotation_scale)
            .arg(&mut self.codes).arg(&mut self.scales).arg(&mut self.norms)
            .arg(&mut self.invalid).arg(&self.capture).arg(&mut self.intermediate);
        unsafe { launch.launch(LaunchConfig {grid_dim: (output_rows as u32,1,1), block_dim: (256,1,1),
            shared_mem_bytes: d * 4}) }.map_err(error)?;
        let encoded = record()?;
        let codes = self.stream.clone_dtoh(&self.codes.slice(..output_rows * (self.padded / 2))).map_err(error)?;
        let scales = self.stream.clone_dtoh(&self.scales.slice(..output_rows)).map_err(error)?;
        let norms = self.stream.clone_dtoh(&self.norms.slice(..output_rows)).map_err(error)?;
        let invalid = self.stream.clone_dtoh(&self.invalid).map_err(error)?;
        let downloaded = record()?;
        downloaded.synchronize().map_err(error)?;
        if invalid[0] != 0 { return Err(error("unrepresentable rotated residual")); }
        let centers = if self.encode_repeats > 1 { selected.iter().map(|&id| id as u16).collect() }
            else if let Some(ids) = primary_ids { ids.into_iter().map(|id|id as u16).collect() }
            else { selected.chunks_exact(self.fanout).map(|r| r[0] as u16).collect() };
        let codes = PrimaryChunk::from_gpu(codes, scales, norms, centers, self.padded)?;
        self.profile.h2d_ms += upload.elapsed_ms(&uploaded).map_err(error)? as f64;
        self.profile.encode_ms += begin.elapsed_ms(&encoded).map_err(error)? as f64;
        self.profile.d2h_ms += encoded.elapsed_ms(&downloaded).map_err(error)? as f64;
        self.profile.rows += rows;
        self.profile.encoded_memberships += output_rows;
        self.profile.batches += 1;
        self.profile.raw_h2d_bytes += points.len() * 4;
        self.profile.code_d2h_bytes += output_rows * (self.padded / 2 + 8);
        Ok((selected, codes))
    }

    #[cfg(test)]
    pub(super) fn capture_intermediates(&mut self) -> AnnResult<()> {
        self.intermediate = self.stream.alloc_zeros(2 * self.max_rows * self.dim).map_err(error)?;
        self.dot_scores = self.stream.alloc_zeros(2 * self.max_rows * self.leaders).map_err(error)?;
        self.capture = 1;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn captured_dots_scores(&self, rows: usize) -> AnnResult<Vec<f32>> {
        self.stream.clone_dtoh(&self.dot_scores.slice(..2 * rows * self.leaders)).map_err(error)
    }

    #[cfg(test)]
    pub(super) fn intermediates(&self, rows: usize) -> AnnResult<Vec<f32>> {
        self.stream.clone_dtoh(&self.intermediate.slice(..2 * rows * self.dim)).map_err(error)
    }
}

/// Full GPU Lloyd iterations with the original sample initialization, ordered
/// accumulation and empty-cluster rule. Only the final centers leave the GPU.
pub(super) fn train_partition_kmeans(samples: &[f32], leaders: usize, iterations: usize, device: usize,
    budget: usize, capture: bool) -> AnnResult<(Vec<f32>, Vec<Vec<f32>>)> {
    let rows = samples.len()/512;
    if !matches!(leaders, 64 | 512) || samples.len()%512 != 0 || rows < leaders || rows > u32::MAX as usize/leaders
        || samples.iter().any(|v|!v.is_finite()) {
        return Err(error("GPU partition training requires finite 512D samples, 64/512 centers and at least one sample per center"));
    }
    let bytes = samples.len()*4 + rows*(leaders+1)*4 + leaders*512*8 + leaders*4 + 4;
    if bytes > budget { return Err(error("GPU k-means workspace budget exceeded")); }
    let started = Instant::now();
    let context = CudaContext::new(device).map_err(error)?;
    let stream = context.new_stream().map_err(error)?;
    let ptx = compile_ptx_with_opts(include_str!("sq4_gpu_d0.cu"),CompileOptions {
        ftz:Some(false),fmad:Some(false),options:vec!["--std=c++17".into()],..CompileOptions::default()
    }).map_err(error)?;
    let module=context.load_module(ptx).map_err(error)?;
    let init=module.load_function("sq4_kmeans_init").map_err(error)?;
    let prepare=module.load_function("sq4_kmeans_prepare").map_err(error)?;
    let assign=module.load_function(if leaders == 64 {"sq4_primary_assign_exact"} else {"sq4_budget_assign_exact"}).map_err(error)?;
    let update=module.load_function("sq4_kmeans_update").map_err(error)?;
    let data=stream.clone_htod(samples).map_err(error)?;
    let mut centers=stream.alloc_zeros::<f32>(leaders*512).map_err(error)?;
    let mut transposed=stream.alloc_zeros::<f32>(leaders*512).map_err(error)?;
    let mut norms=stream.alloc_zeros::<f32>(leaders).map_err(error)?;
    let mut selected=stream.alloc_zeros::<u32>(rows).map_err(error)?;
    let mut members=stream.alloc_zeros::<u32>(leaders*rows).map_err(error)?;
    let mut unused=stream.alloc_zeros::<f32>(1).map_err(error)?;
    let n=rows as u32; let k=leaders as u32; let one=1u32; let zero=0u32;
    let cfg=LaunchConfig {grid_dim:(k,1,1),block_dim:(256,1,1),shared_mem_bytes:0};
    let mut launch=stream.launch_builder(&init); launch.arg(&data).arg(&n).arg(&k).arg(&mut centers);
    unsafe {launch.launch(cfg)}.map_err(error)?;
    stream.synchronize().map_err(error)?;
    let prepare_ms=started.elapsed().as_secs_f64()*1000.0;
    let mut assign_ms=0.0; let mut update_ms=0.0; let mut history=Vec::new();
    for _ in 0..iterations.max(1) {
        let begin=stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(error)?;
        let mut launch=stream.launch_builder(&prepare);
        launch.arg(&centers).arg(&k).arg(&mut transposed).arg(&mut norms);
        unsafe {launch.launch(cfg)}.map_err(error)?;
        let mut launch=stream.launch_builder(&assign);
        launch.arg(&data).arg(&transposed).arg(&norms).arg(&n);
        if leaders != 64 { launch.arg(&k); }
        launch.arg(&one)
            .arg(&mut selected).arg(&zero).arg(&mut unused);
        unsafe {launch.launch(LaunchConfig {grid_dim:(n.div_ceil(4),1,1),block_dim:(256,1,1),shared_mem_bytes:0})}.map_err(error)?;
        let assigned=stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(error)?;
        let mut launch=stream.launch_builder(&update);
        launch.arg(&data).arg(&selected).arg(&n).arg(&k).arg(&mut members).arg(&mut centers);
        unsafe {launch.launch(cfg)}.map_err(error)?;
        let updated=stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(error)?;
        updated.synchronize().map_err(error)?;
        assign_ms += begin.elapsed_ms(&assigned).map_err(error)? as f64;
        update_ms += assigned.elapsed_ms(&updated).map_err(error)? as f64;
        if capture { history.push(stream.clone_dtoh(&centers).map_err(error)?); }
    }
    let centers=stream.clone_dtoh(&centers).map_err(error)?;
    if centers.iter().any(|v|!v.is_finite()) { return Err(error("GPU k-means produced nonfinite centers")); }
    tracing::info!("SQ4 partition kmeans: backend=gpu samples={} centers={} iterations={} workspace_bytes={} raw_h2d_bytes={} centers_d2h_bytes={} prepare_ms={:.3} assign_ms={:.3} update_ms={:.3} wall_ms={:.3} cpu_assignment_rows=0 cpu_update_rows=0",
        rows,leaders,iterations.max(1),bytes,samples.len()*4,centers.len()*4,prepare_ms,assign_ms,update_ms,started.elapsed().as_secs_f64()*1000.0);
    Ok((centers,history))
}
