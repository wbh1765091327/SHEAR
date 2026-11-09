//! Bounded CUDA sketch service shared by the CPU D0 batch workers.
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Debug, Default)]
pub(crate) struct GpuSketchProfile {
    pub rows: usize,
    pub launches: usize,
    pub h2d_bytes: usize,
    pub d2h_bytes: usize,
    pub workspace_bytes: usize,
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
    pub service_ms: f64,
    pub wait_ms: f64,
    pub prepare_ms: f64,
}

fn error(err: impl std::fmt::Display) -> AnnError {
    AnnError::log_index_error(format!("GPU sketch failed (CPU fallback disabled): {err}"))
}

#[cfg(feature = "pipnn_cuda")]
mod cuda {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;
    use parking_lot::Mutex;
    use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream,
        LaunchConfig, PinnedHostSlice, PushKernelArg, sys};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    struct Engine {
        stream: Arc<CudaStream>,
        function: CudaFunction,
        planes: CudaSlice<f32>,
        input: CudaSlice<f32>,
        output: CudaSlice<f32>,
        pinned_input: PinnedHostSlice<f32>,
        profile: GpuSketchProfile,
    }

    pub(crate) struct GpuSketch {
        dim: usize,
        width: usize,
        plane_stride: usize,
        max_rows: usize,
        engine: Mutex<Engine>,
    }

    impl GpuSketch {
        pub(crate) fn new(device: usize, dim: usize, width: usize, max_rows: usize,
            hyperplanes: &[f32], budget_bytes: usize) -> AnnResult<Self> {
            let start = Instant::now();
            if dim == 0 || width == 0 || max_rows == 0 || hyperplanes.len() != dim.saturating_mul(width)
                || dim > u32::MAX as usize || width > u32::MAX as usize || max_rows > u32::MAX as usize / 16 {
                return Err(error("invalid sketch shape"));
            }
            let plane_stride = width.div_ceil(16) * 16;
            let workspace_bytes = max_rows.saturating_mul(dim.saturating_add(width))
                .saturating_add(dim.saturating_mul(plane_stride)).saturating_mul(4);
            if workspace_bytes > budget_bytes {
                return Err(error(format!("workspace {workspace_bytes} exceeds budget {budget_bytes}")));
            }
            let context = CudaContext::new(device).map_err(error)?;
            let stream = context.new_stream().map_err(error)?;
            let ptx = compile_ptx_with_opts(include_str!("gpu_sketch.cu"), CompileOptions {
                ftz: Some(false), fmad: Some(false),
                options: vec!["--std=c++17".into()], ..CompileOptions::default()
            }).map_err(error)?;
            let module = context.load_module(ptx).map_err(error)?;
            let function = module.load_function("hashprune_sketch_fp32").map_err(error)?;
            let mut transposed = vec![0.0; dim * plane_stride];
            for projection in 0..width {
                for axis in 0..dim {
                    transposed[axis * plane_stride + projection] = hyperplanes[projection * dim + axis];
                }
            }
            let planes = stream.clone_htod(&transposed).map_err(error)?;
            let input = stream.alloc_zeros(max_rows * dim).map_err(error)?;
            let output = stream.alloc_zeros(max_rows * width).map_err(error)?;
            // Write-combined pinned memory is only read by the GPU, not used for D2H.
            let mut pinned_input = unsafe { context.alloc_pinned::<f32>(max_rows * dim) }.map_err(error)?;
            pinned_input.as_mut_slice().map_err(error)?.fill(0.0);
            stream.synchronize().map_err(error)?;
            let profile = GpuSketchProfile {workspace_bytes, prepare_ms: start.elapsed().as_secs_f64()*1000.0,
                ..GpuSketchProfile::default()};
            Ok(Self {dim, width, plane_stride, max_rows,
                engine: Mutex::new(Engine {stream, function, planes, input, output, pinned_input, profile})})
        }

        pub(crate) fn compute(&self, points: &[f32], output_host: &mut [f32]) -> AnnResult<()> {
            if points.len() % self.dim != 0 || points.len()/self.dim > self.max_rows
                || output_host.len() != points.len()/self.dim * self.width {
                return Err(error("invalid batch shape"));
            }
            let rows = points.len()/self.dim;
            if rows == 0 { return Ok(()); }
            let wait = Instant::now();
            let mut engine = self.engine.lock();
            let wait_ms = wait.elapsed().as_secs_f64()*1000.0;
            let service = Instant::now();
            let Engine {stream, function, planes, input, output, pinned_input, profile} = &mut *engine;
            pinned_input.as_mut_slice().map_err(error)?[..points.len()].copy_from_slice(points);
            let record = || stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(error);
            let start = record()?;
            stream.memcpy_htod(&pinned_input.as_slice().map_err(error)?[..points.len()],
                &mut input.slice_mut(..points.len())).map_err(error)?;
            let uploaded = record()?;
            let (rows_u32, dim, width, stride) = (rows as u32, self.dim as u32, self.width as u32, self.plane_stride as u32);
            let mut launch = stream.launch_builder(function);
            launch.arg(&*input).arg(&*planes).arg(&mut *output)
                .arg(&rows_u32).arg(&dim).arg(&width).arg(&stride);
            unsafe { launch.launch(LaunchConfig {grid_dim: ((rows_u32 * 16).div_ceil(256),1,1),
                block_dim: (256,1,1), shared_mem_bytes: 0}) }.map_err(error)?;
            let computed = record()?;
            stream.memcpy_dtoh(&output.slice(..output_host.len()), output_host).map_err(error)?;
            let downloaded = record()?;
            downloaded.synchronize().map_err(error)?;
            profile.h2d_ms += start.elapsed_ms(&uploaded).map_err(error)? as f64;
            profile.kernel_ms += uploaded.elapsed_ms(&computed).map_err(error)? as f64;
            profile.d2h_ms += computed.elapsed_ms(&downloaded).map_err(error)? as f64;
            profile.rows += rows;
            profile.launches += 1;
            profile.h2d_bytes += points.len()*4;
            profile.d2h_bytes += output_host.len()*4;
            profile.wait_ms += wait_ms;
            profile.service_ms += service.elapsed().as_secs_f64()*1000.0;
            Ok(())
        }

        pub(crate) fn profile(&self) -> GpuSketchProfile { self.engine.lock().profile.clone() }
    }
}

#[cfg(feature = "pipnn_cuda")]
pub(crate) use cuda::GpuSketch;

#[cfg(not(feature = "pipnn_cuda"))]
pub(crate) struct GpuSketch;
#[cfg(not(feature = "pipnn_cuda"))]
impl GpuSketch {
    pub(crate) fn new(_: usize, _: usize, _: usize, _: usize, _: &[f32], _: usize) -> AnnResult<Self> {
        Err(error("build requires pipnn_cuda"))
    }
    pub(crate) fn compute(&self, _: &[f32], _: &mut [f32]) -> AnnResult<()> { Err(error("build requires pipnn_cuda")) }
    pub(crate) fn profile(&self) -> GpuSketchProfile { GpuSketchProfile::default() }
}

#[cfg(all(test, feature = "pipnn_cuda"))]
mod tests {
    use super::*;
    use crate::algorithm::pipnn::{external_store::fill_sketch_row_from_slice, hash_prune::generate_hyperplanes};

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn gpu_sketch_preserves_fp32_bits_and_reuses_bounded_buffers() {
        for (dim, width) in [(1, 1), (5, 17), (32, 12), (96, 12), (512, 12), (1024, 12)] {
            let planes = generate_hyperplanes(dim, width, 42);
            assert!(GpuSketch::new(0, dim, width, 4096, &planes, 1).is_err());
            let gpu = GpuSketch::new(0, dim, width, 4096, &planes, 32 << 20).unwrap();
            let mut total_rows = 0;
            for rows in [0, 1, 15, 16, 17, 4095, 4096, 1] {
                let points = (0..rows*dim).map(|i| match i/dim % 7 {
                    0 => -0.0,
                    1 => f32::from_bits(1 + (i % 1023) as u32),
                    2 => if i % 2 == 0 { 1.0e10 } else { -1.0e10 },
                    _ => ((i * 37 % 1013) as f32 - 506.0) / 511.0,
                }).collect::<Vec<_>>();
                let mut output = vec![f32::NAN; rows*width];
                gpu.compute(&points, &mut output).unwrap();
                let mut expected = vec![0.0; width];
                for (row, (point, actual)) in points.chunks_exact(dim).zip(output.chunks_exact(width)).enumerate() {
                    fill_sketch_row_from_slice(point, &planes, &mut expected).unwrap();
                    assert_eq!(actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                               expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                               "dim={dim} width={width} rows={rows} row={row}");
                }
                total_rows += rows;
            }
            assert!(gpu.compute(&vec![0.0; 4097*dim], &mut vec![0.0; 4097*width]).is_err());
            let profile = gpu.profile();
            assert_eq!(profile.rows, total_rows);
            assert_eq!(profile.launches, 7);
            assert_eq!(profile.h2d_bytes, total_rows*dim*4);
            assert_eq!(profile.d2h_bytes, total_rows*width*4);
            eprintln!("GPU sketch dim={dim} width={width} {profile:?}");
        }
    }
}
