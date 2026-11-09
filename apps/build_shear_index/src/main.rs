use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::{fs, io};

use clap::Parser;
use shear_core::algorithm::PiPNNParams;
use shear_core::algorithm::pipnn::params::PiPNNGpuSubtreePolicy;
use shear_core::algorithm::pipnn::{
    PiPNNComputeBackendKind, PiPNNGpuSubtreeAssignmentBackend, PiPNNGpuSubtreeChildScheduleMode,
    PiPNNGpuSubtreeD0PartitionMode, PiPNNGpuSubtreeD0ThroughputModelMode,
    PiPNNGpuSubtreeD1PlusLazySplitMode, PiPNNGpuSubtreeD1RootPartitionMode, PiPNNGpuSubtreeDcoMode,
    PiPNNGpuSubtreeDcoRerankMode, PiPNNGpuSubtreeExecutionMode, PiPNNGpuSubtreeLeafTopKBackend,
    PiPNNGpuSubtreePersistentLargeTicketBatch, PiPNNGpuSubtreePersistentSchedulerMode,
    PiPNNGpuSubtreeQcodeD1PayloadMode, PiPNNGpuSubtreeQcodeStagingMode,
    PiPNNGpuSubtreeRabitqD0AssignMode, PiPNNGpuSubtreeRabitqD1DistanceMode,
    PiPNNGpuSubtreeRabitqD1TopKMode, PiPNNGpuSubtreeRabitqPlusMode,
    PiPNNGpuSubtreeRabitqPlusRotator, PiPNNGpuSubtreeRabitqPlusSidecarMode,
    PiPNNGpuSubtreeSq8AssignmentKernel, PiPNNGpuSubtreeSq8D0AssignmentKernel,
    PiPNNGpuSubtreeSq8LeafExecutor, PiPNNGpuSubtreeSq8LeafMode, PiPNNGpuSubtreeSqStagingMode,
    PiPNNGpuSubtreeSqWmmaMode, PiPNNGpuSubtreeSqWmmaSidecarMode, PiPNNGpuSubtreeTurboQuantMode,
    PiPNNGpuSubtreeTurboQuantSidecarMode, PiPNNGpuSubtreeVectorStagingMode,
    PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNHashPruneEvidenceMode,
    PiPNNOnlineSuffixReduceMode, PiPNNOomIoMode, PiPNNPartitionerKind, PiPNNPipelineComputeMode,
    PiPNNPruneBackend, RootFanoutPolicy, RootRiskSigmaMode, SqInt8CpuKernel,
    create_rabitq_plus_sidecar_from_fbin, create_rabitq_split_sidecar_from_fbin,
    create_sq_wmma_sidecar_from_fbin_with_bits_native_int4,
    create_sq_wmma_sidecar_from_fbin_with_bits_native_int4_and_resident_prefix,
    create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4,
    create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4_and_resident_prefix,
    create_turboquant_mse4_sidecar_from_fbin, create_turboquant_mse4_sidecar_from_fbin_cuda,
    estimate_sq_wmma_sidecar_resident_rows,
};
use shear_core::common::{AnnError, AnnResult, Metric};
use shear_core::index::{
    AnnInMemIndex, InmemIndex, build_pipnn_graph_oom_to_file, build_pipnn_graph_oom_to_file_u8,
};
use shear_core::model::default_param_vals::ALPHA;
use shear_core::model::{IndexConfiguration, IndexWriteParametersBuilder};
use shear_core::utils::{Timer, file_exists, load_metadata_from_file, round_up};

const TURBOQUANT_CASCADE_MAX_SURVIVORS: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TurboQuantSidecarBuildBackend {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl TurboQuantSidecarBuildBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for TurboQuantSidecarBuildBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TurboQuantSidecarBuildBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(format!(
                "Unsupported TurboQuant-MSE4 sidecar build backend '{other}', expected one of: auto, cpu, cuda"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ProcSample {
    thread_count: Option<usize>,
    rss_kib: Option<usize>,
    running_ns: Option<u64>,
}

#[derive(Debug)]
struct PhaseTracker {
    enabled: bool,
    stop_flag: Arc<AtomicBool>,
    current_phase: Arc<Mutex<String>>,
    sampler: Option<JoinHandle<Vec<ProcPhaseSample>>>,
    phase_name: String,
    phase_start: Instant,
    phase_cpu_start_ns: Option<u64>,
    completed_phases: Vec<CompletedPhase>,
}

#[derive(Clone, Debug)]
struct ProcPhaseSample {
    phase_name: String,
    wall_offset: Duration,
    running_ns: Option<u64>,
    thread_count: Option<usize>,
    rss_kib: Option<usize>,
}

#[derive(Clone, Debug)]
struct CompletedPhase {
    name: String,
    wall_time: Duration,
    cpu_time: Duration,
    peak_threads: usize,
    peak_rss_kib: Option<usize>,
    max_cpu_cores: f64,
    sample_count: usize,
}

#[derive(Clone, Debug, Default)]
struct BuildPipnnTiming {
    phases: Vec<CompletedPhase>,
}

#[derive(Clone, Debug)]
struct DriverTimingReport {
    total_wall: Duration,
    pre_build_wall: Duration,
    build_call_wall: Duration,
    post_build_wall: Duration,
    build_success: bool,
    profile_json_path: Option<PathBuf>,
    phases: Vec<CompletedPhase>,
}

impl PhaseTracker {
    fn from_env() -> Self {
        let enabled = phase_monitoring_enabled();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let current_phase = Arc::new(Mutex::new("startup".to_string()));
        let sampler = enabled
            .then(|| spawn_phase_sampler(Arc::clone(&stop_flag), Arc::clone(&current_phase)));

        Self {
            enabled,
            stop_flag,
            current_phase,
            sampler,
            phase_name: "startup".to_string(),
            phase_start: Instant::now(),
            phase_cpu_start_ns: current_running_ns(),
            completed_phases: Vec::new(),
        }
    }

    fn start_phase(&mut self, name: &str) {
        self.finish_current_phase();
        self.phase_name = name.to_string();
        *self.current_phase.lock().unwrap() = self.phase_name.clone();
        self.phase_start = Instant::now();
        self.phase_cpu_start_ns = current_running_ns();
    }

    fn finish(mut self) -> Vec<CompletedPhase> {
        self.finish_current_phase();
        if self.enabled {
            self.stop_flag.store(true, Ordering::Relaxed);
            let mut samples = self
                .sampler
                .take()
                .and_then(|handle| handle.join().ok())
                .unwrap_or_default();
            if let Some(sample) = read_proc_sample() {
                samples.push(ProcPhaseSample {
                    phase_name: self.phase_name.clone(),
                    wall_offset: self.phase_start.elapsed(),
                    running_ns: sample.running_ns,
                    thread_count: sample.thread_count,
                    rss_kib: sample.rss_kib,
                });
            }

            annotate_completed_phases(&mut self.completed_phases, &samples);
            print_phase_monitoring_report(&self.completed_phases, &samples);
        }

        self.completed_phases
    }

    fn finish_current_phase(&mut self) {
        let wall_time = self.phase_start.elapsed();
        let cpu_end_ns = current_running_ns();
        let cpu_time = match (self.phase_cpu_start_ns, cpu_end_ns) {
            (Some(start), Some(end)) if end >= start => Duration::from_nanos(end - start),
            _ => Duration::ZERO,
        };

        self.completed_phases.push(CompletedPhase {
            name: self.phase_name.clone(),
            wall_time,
            cpu_time,
            peak_threads: 0,
            peak_rss_kib: None,
            max_cpu_cores: 0.0,
            sample_count: 0,
        });
    }
}

fn phase_monitoring_enabled() -> bool {
    std::env::var("SHEAR_PIPNN_PHASE_MONITOR")
        .or_else(|_| std::env::var("VELOANN_PIPNN_PHASE_MONITOR"))
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn spawn_phase_sampler(
    stop_flag: Arc<AtomicBool>,
    current_phase: Arc<Mutex<String>>,
) -> JoinHandle<Vec<ProcPhaseSample>> {
    let start = Instant::now();
    const SAMPLE_INTERVAL: Duration = Duration::from_millis(200);

    thread::spawn(move || {
        let mut samples = Vec::new();
        while !stop_flag.load(Ordering::Relaxed) {
            let phase_name = current_phase.lock().unwrap().clone();
            if let Some(sample) = read_proc_sample() {
                samples.push(ProcPhaseSample {
                    phase_name,
                    wall_offset: start.elapsed(),
                    running_ns: sample.running_ns,
                    thread_count: sample.thread_count,
                    rss_kib: sample.rss_kib,
                });
            }
            thread::sleep(SAMPLE_INTERVAL);
        }
        samples
    })
}

fn read_proc_sample() -> Option<ProcSample> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let mut snapshot = parse_status_snapshot(&status)?;
    snapshot.running_ns = current_running_ns();
    Some(snapshot)
}

fn current_running_ns() -> Option<u64> {
    let schedstat = fs::read_to_string("/proc/self/schedstat").ok()?;
    parse_schedstat_running_ns(&schedstat)
}

fn parse_status_snapshot(status: &str) -> Option<ProcSample> {
    let mut thread_count = None;
    let mut rss_kib = None;

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Threads:") {
            thread_count = rest.trim().parse::<usize>().ok();
        } else if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kib = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<usize>().ok());
        }
    }

    if thread_count.is_none() && rss_kib.is_none() {
        None
    } else {
        Some(ProcSample {
            thread_count,
            rss_kib,
            running_ns: None,
        })
    }
}

fn parse_schedstat_running_ns(schedstat: &str) -> Option<u64> {
    schedstat
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
}

fn effective_gpu_subtree_d0_capacity_enabled(args: &Args) -> bool {
    if args.gpu_subtree_d0_capacity_disable {
        return false;
    }
    if args.gpu_subtree_d0_capacity_enable {
        return true;
    }
    args.partitioner == PiPNNPartitionerKind::GpuSubtree && args.oom_enable
}

fn effective_gpu_subtree_d0_capacity_ratios(args: &Args) -> (f64, f64, f64) {
    // D0 planning targets 8%; upload admission has a separate 50% high-water limit.
    let defaults = if args.gpu_subtree_sq4_plus { (0.50, 0.08, 0.08) } else { (0.60, 0.50, 0.15) };
    (args.gpu_subtree_d0_capacity_high_ratio.unwrap_or(defaults.0),
     args.gpu_subtree_d0_capacity_target_ratio.unwrap_or(defaults.1),
     args.gpu_subtree_d0_capacity_low_ratio.unwrap_or(defaults.2))
}

fn validate_args(args: &Args) -> AnnResult<()> {
    let gpu_subtree_d0_capacity_enabled = effective_gpu_subtree_d0_capacity_enabled(args);
    if (args.gpu_subtree_d0_partition_mode == PiPNNGpuSubtreeD0PartitionMode::BudgetKMeansTopK
        || (args.gpu_subtree_sq4_plus && args.gpu_subtree_d0_partition_mode == PiPNNGpuSubtreeD0PartitionMode::RbcFanout))
        && (!args.gpu_subtree_sq4_plus || !gpu_subtree_d0_capacity_enabled) {
        return Err(AnnError::log_index_config_error("gpu_subtree_d0_partition_mode".into(),
            "budget-kmeans-topk requires SQ4+ and D0 capacity planning".into()));
    }


    if args.gpu_subtree_persistent_launch_policy == PiPNNGpuSubtreePolicy::Enforce {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_persistent_launch_policy".to_string(),
            "validated launch cache support is not installed".to_string(),
        ));
    }
    if args.gpu_subtree_hbm_policy == PiPNNGpuSubtreePolicy::Enforce {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_hbm_policy".to_string(),
            "automatic HBM enforcement is not installed".to_string(),
        ));
    }
    if args.m_hash_bits == 0 || args.m_hash_bits > 16 {
        return Err(AnnError::log_index_config_error(
            "m_hash_bits".to_string(),
            "m_hash_bits must be in [1, 16]".to_string(),
        ));
    }
    if !args.lrpf_pre_spill_width_multiplier.is_finite()
        || args.lrpf_pre_spill_width_multiplier < 1.0
    {
        return Err(AnnError::log_index_config_error(
            "lrpf_pre_spill_width_multiplier".to_string(),
            "lrpf_pre_spill_width_multiplier must be finite and >= 1.0".to_string(),
        ));
    }
    if args.lrpf_protected_core > args.leaf_knn {
        return Err(AnnError::log_index_config_error(
            "lrpf_protected_core".to_string(),
            "lrpf_protected_core must be <= leaf_knn".to_string(),
        ));
    }
    if args.lrpf_target_degree == 0 {
        return Err(AnnError::log_index_config_error(
            "lrpf_target_degree".to_string(),
            "lrpf_target_degree must be >= 1".to_string(),
        ));
    }
    if !(4..=8).contains(&args.gpu_subtree_sq_bits) {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_bits".to_string(),
            "gpu_subtree_sq_bits must be in [4, 8]".to_string(),
        ));
    }
    let gpu_subtree_partitioner = matches!(args.partitioner, PiPNNPartitionerKind::GpuSubtree);
    if let Some(capture_dir) = args.gpu_subtree_tessa_capture_dir.as_deref() {
        let frozen_a3 = gpu_subtree_partitioner
            && args.pipnn_backend.eq_ignore_ascii_case("cuda")
            && args.gpu_subtree_execution_mode == PiPNNGpuSubtreeExecutionMode::FusedPersistent
            && args.gpu_subtree_d0_partition_mode
                == PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
            && args.gpu_subtree_sq_staging_mode == PiPNNGpuSubtreeSqStagingMode::GpuD0
            && args.gpu_subtree_sq_wmma_mode == PiPNNGpuSubtreeSqWmmaMode::Build
            && args.gpu_subtree_sq8_assignment_kernel == PiPNNGpuSubtreeSq8AssignmentKernel::Wmma
            && args.gpu_subtree_sq8_leaf_executor == PiPNNGpuSubtreeSq8LeafExecutor::Wmma
            && args.gpu_subtree_persistent_large_ticket_batch
                == PiPNNGpuSubtreePersistentLargeTicketBatch::Eight;
        if !frozen_a3 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_tessa_capture_dir".to_string(),
                "TESSA capture requires the frozen A3 contract: gpu-subtree/CUDA, gpu-kmeans-coarse D0, gpu-d0 SQ staging, fused-persistent execution, SQ-WMMA build, forced WMMA assignment and leaf executors, and ticket batch 8"
                    .to_string(),
            ));
        }

        let capture_dir = normalize_cli_path(capture_dir)?;
        let index_dir = normalize_cli_path(Path::new(&args.index_path_prefix))?;
        if paths_overlap(&capture_dir, &index_dir) {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_tessa_capture_dir".to_string(),
                "TESSA capture directory must not overlap the index output directory".to_string(),
            ));
        }
        if !args.oom_temp_dir.is_empty() {
            let oom_dir = normalize_cli_path(Path::new(&args.oom_temp_dir))?;
            if paths_overlap(&capture_dir, &oom_dir) {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_tessa_capture_dir".to_string(),
                    "TESSA capture directory must not overlap the OOM temporary directory"
                        .to_string(),
                ));
            }
        }
    }
    if args.gpu_subtree_sq_staging_raw_d1_diagnostic
        && args.gpu_subtree_sq_staging_mode != PiPNNGpuSubtreeSqStagingMode::GpuD0
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_staging_raw_d1_diagnostic".to_string(),
            "raw-D1 diagnostic requires --gpu-subtree-sq-staging-mode gpu-d0".to_string(),
        ));
    }
    if args.gpu_subtree_sq8_rotated_residual_d0 {
        if args.gpu_subtree_sq_staging_mode != PiPNNGpuSubtreeSqStagingMode::GpuD0
            || !(args.gpu_subtree_sq_staging_raw_d1_diagnostic
                || args.gpu_subtree_sq8_rotated_residual_d1)
            || args.gpu_subtree_d0_partition_mode != PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq8_rotated_residual_d0".to_string(),
                "rotated residual SQ8 D0 requires gpu-d0 SQ staging, gpu-kmeans-coarse, and either raw D1+ or rotated residual SQ8 D1+".to_string(),
            ));
        }
    }
    if args.gpu_subtree_sq8_rotated_residual_d1
        && (!args.gpu_subtree_sq8_rotated_residual_d0
            || args.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
            || args.gpu_subtree_sq_staging_raw_d1_diagnostic)
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq8_rotated_residual_d1".to_string(),
            "rotated residual SQ8 D1+ requires rotated residual D0 and SQ-WMMA build, without raw-D1 diagnostic".to_string(),
        ));
    }
    if args.gpu_subtree_sq8_rotated_residual_d1_int8 && !args.gpu_subtree_sq8_rotated_residual_d1 {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq8_rotated_residual_d1_int8".to_string(),
            "direct INT8 D1+ requires --gpu-subtree-sq8-rotated-residual-d1".to_string(),
        ));
    }
    if args.gpu_subtree_sq_staging_mode.is_enabled() {
        let sq_direct_d1 = matches!(
            args.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
                | PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
        ) || args.gpu_subtree_sq8_rotated_residual_d0;
        let turboquant_requested = args.gpu_subtree_turboquant_mode.is_enabled()
            || args.gpu_subtree_turboquant_sidecar_mode.is_enabled()
            || args.gpu_subtree_turboquant_sidecar_prefix.is_some();
        let turboquant_d1_build =
            args.gpu_subtree_turboquant_mode == PiPNNGpuSubtreeTurboQuantMode::Build;
        if args.gpu_subtree_sq_staging_raw_d1_diagnostic {
            if args.gpu_subtree_sq_wmma_mode.is_enabled() {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_sq_staging_raw_d1_diagnostic".to_string(),
                    "raw-D1 diagnostic cannot be combined with SQ-WMMA D1+ distance".to_string(),
                ));
            }
            if turboquant_requested {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_sq_staging_raw_d1_diagnostic".to_string(),
                    "raw-D1 diagnostic cannot be combined with TurboQuant D1+ distance".to_string(),
                ));
            }
        }
        if args.gpu_subtree_sq_native_int4 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_native_int4".to_string(),
                "SQ GPU-D0 staging does not support native INT4".to_string(),
            ));
        }
        if args.gpu_subtree_sq_bits != 8 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_bits".to_string(),
                "SQ GPU-D0 staging requires --gpu-subtree-sq-bits 8".to_string(),
            ));
        }
        if args.gpu_subtree_qcode_staging_mode.is_enabled()
            || args.gpu_subtree_rabitq_plus_mode.is_enabled()
            || args.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled()
            || args.gpu_subtree_rabitq_plus_sidecar_prefix.is_some()
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_staging_mode".to_string(),
                "SQ GPU-D0 staging cannot be combined with RabitQ".to_string(),
            ));
        }
        if sq_direct_d1 && turboquant_requested {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_staging_mode".to_string(),
                "SQ GPU-D0 direct D1 staging cannot be combined with TurboQuant".to_string(),
            ));
        }
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_staging_mode".to_string(),
                "SQ GPU-D0 staging requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !args.pipnn_backend.eq_ignore_ascii_case("cuda") {
            return Err(AnnError::log_index_config_error(
                "pipnn_backend".to_string(),
                "SQ GPU-D0 staging requires --pipnn-backend=cuda".to_string(),
            ));
        }
        if !turboquant_d1_build
            && args.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
            && !args.gpu_subtree_sq_staging_raw_d1_diagnostic
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_wmma_mode".to_string(),
                "SQ GPU-D0 staging requires --gpu-subtree-sq-wmma-mode build unless TurboQuant-MSE4 build owns D1+ distance"
                    .to_string(),
            ));
        }
        if sq_direct_d1 {
            if args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
                || args.gpu_subtree_sq_wmma_sidecar_prefix.is_some()
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_sq_wmma_sidecar_mode".to_string(),
                    "SQ GPU-D0 direct D1 staging runs without SQ+WMMA sidecar files".to_string(),
                ));
            }
        } else if args.gpu_subtree_sq_wmma_sidecar_mode != PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
            || args.gpu_subtree_sq_wmma_sidecar_prefix.is_none()
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_wmma_sidecar_mode".to_string(),
                "SQ GPU-D0 staging supports only sidecar rebuild with --gpu-subtree-sq-wmma-sidecar-prefix"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_sq_staging_upload_shard_rows == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_staging_upload_shard_rows".to_string(),
                "SQ GPU-D0 staging requires upload shard rows > 0".to_string(),
            ));
        }
    }
    if args.gpu_subtree_sq4_plus
        && (!args.gpu_subtree_sq_native_int4
            || args.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
            || !matches!(args.gpu_subtree_d0_partition_mode,
                PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK | PiPNNGpuSubtreeD0PartitionMode::BudgetKMeansTopK | PiPNNGpuSubtreeD0PartitionMode::RbcFanout)
            || !args.data_type.eq_ignore_ascii_case("float")
            || !args.dist_fn.eq_ignore_ascii_case("l2")
            || !args.oom_enable
            || args.root_stop_after_d00
            || args.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly)
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq4_plus".to_string(),
            "SQ4+ requires a float L2 OOM graph build with budget-kmeans-topk (default) or explicit fixed tagore-kmeans-topk D0 and native packed SQ4 WMMA build".to_string(),
        ));
    }
    if args.gpu_subtree_sq_native_int4 && args.gpu_subtree_sq_bits != 4 {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_native_int4".to_string(),
            "native INT4 SQ requires --gpu-subtree-sq-bits 4".to_string(),
        ));
    }
    let native_int4_sidecar_create = matches!(
        args.gpu_subtree_sq_wmma_sidecar_mode,
        PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly | PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
    ) && args.gpu_subtree_sq_wmma_sidecar_prefix.is_some();
    if args.gpu_subtree_sq_native_int4
        && args.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
        && !native_int4_sidecar_create
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_native_int4".to_string(),
            "native INT4 SQ requires --gpu-subtree-sq-wmma-mode build or SQ+WMMA sidecar create/rebuild".to_string(),
        ));
    }

    if !args.auto_params && (args.leaf_size_min == 0 || args.leaf_size_min > args.leaf_size_max) {
        return Err(AnnError::log_index_config_error(
            "leaf_size_min".to_string(),
            "leaf_size_min must be > 0 and <= leaf_size_max".to_string(),
        ));
    }
    if gpu_subtree_partitioner {
        if !args.oom_enable {
            return Err(AnnError::log_index_config_error(
                "partitioner".to_string(),
                "gpu-subtree requires --oom-enable".to_string(),
            ));
        }
        if !args.pipnn_backend.eq_ignore_ascii_case("cuda") {
            return Err(AnnError::log_index_config_error(
                "pipnn_backend".to_string(),
                "gpu-subtree requires --pipnn-backend=cuda".to_string(),
            ));
        }
        if !args.dist_fn.eq_ignore_ascii_case("l2") {
            return Err(AnnError::log_index_config_error(
                "dist_fn".to_string(),
                "gpu-subtree currently supports only l2".to_string(),
            ));
        }
    }
    if args.gpu_subtree_execution_mode == PiPNNGpuSubtreeExecutionMode::TwoStage
        && args.gpu_subtree_leaf_topk_backend == PiPNNGpuSubtreeLeafTopKBackend::CublasGemm
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_execution_mode".to_string(),
            "two-stage GPU-subtree execution cannot be combined with --gpu-subtree-leaf-topk-backend cublas-gemm"
                .to_string(),
        ));
    }
    if matches!(
        args.gpu_subtree_rabitq_d0_assign_mode,
        PiPNNGpuSubtreeRabitqD0AssignMode::Tiled
            | PiPNNGpuSubtreeRabitqD0AssignMode::Batch
            | PiPNNGpuSubtreeRabitqD0AssignMode::WmmaBatch
    ) && args.gpu_subtree_qcode_staging_mode != PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_rabitq_d0_assign_mode".to_string(),
            "tiled/batch/wmma-batch residual RaBitQ D0 assignment requires --gpu-subtree-qcode-staging-mode rabitq-residual"
                .to_string(),
        ));
    }
    if args.gpu_subtree_rabitq_d1_topk_mode == PiPNNGpuSubtreeRabitqD1TopKMode::WmmaBatch {
        if args.gpu_subtree_qcode_staging_mode != PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_d1_topk_mode".to_string(),
                "WMMA-batch residual RaBitQ D1+ top-k requires --gpu-subtree-qcode-staging-mode rabitq-residual"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_qcode_d1_payload_mode != PiPNNGpuSubtreeQcodeD1PayloadMode::ChildLocal {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_d1_topk_mode".to_string(),
                "WMMA-batch residual RaBitQ D1+ top-k requires child-local qcode payload"
                    .to_string(),
            ));
        }
    }
    if args.gpu_subtree_rabitq_d1_distance_mode == PiPNNGpuSubtreeRabitqD1DistanceMode::SplitShortEx
    {
        if args.gpu_subtree_qcode_staging_mode != PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_d1_distance_mode".to_string(),
                "split-short-ex residual RaBitQ D1 distance requires --gpu-subtree-qcode-staging-mode rabitq-residual"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_qcode_d1_payload_mode != PiPNNGpuSubtreeQcodeD1PayloadMode::ChildLocal {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_d1_distance_mode".to_string(),
                "split-short-ex residual RaBitQ D1 distance supports only child-local qcode payload"
                    .to_string(),
            ));
        }
        if let Some(survivors) = args.gpu_subtree_rabitq_short_survivors {
            if survivors == 0 || survivors > 128 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_rabitq_short_survivors".to_string(),
                    format!("split-short-ex survivors must be in 1..=128, got {survivors}"),
                ));
            }
        }
    }
    if args.gpu_subtree_qcode_staging_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_mode".to_string(),
                "RabitQ residual qcode staging requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !args.oom_enable {
            return Err(AnnError::log_index_config_error(
                "oom_enable".to_string(),
                "RabitQ residual qcode staging requires --oom-enable".to_string(),
            ));
        }
        if !args.pipnn_backend.eq_ignore_ascii_case("cuda") {
            return Err(AnnError::log_index_config_error(
                "pipnn_backend".to_string(),
                "RabitQ residual qcode staging requires --pipnn-backend=cuda".to_string(),
            ));
        }
        if !args.dist_fn.eq_ignore_ascii_case("l2") || !args.data_type.eq_ignore_ascii_case("float")
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_mode".to_string(),
                "RabitQ residual qcode staging requires float L2 input".to_string(),
            ));
        }
        if args.gpu_subtree_rabitq_residual_centroids == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_residual_centroids".to_string(),
                "RabitQ residual qcode staging requires at least one residual centroid".to_string(),
            ));
        }
        if args.gpu_subtree_qcode_staging_upload_shard_rows == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_upload_shard_rows".to_string(),
                "RabitQ residual qcode staging requires upload shard rows > 0".to_string(),
            ));
        }
        if args.gpu_subtree_rabitq_d0_row_tile == 0
            || args.gpu_subtree_rabitq_d0_row_chunk == 0
            || args.gpu_subtree_rabitq_d0_leader_tile == 0
            || args.gpu_subtree_rabitq_d0_partial_r == 0
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_d0_assign_mode".to_string(),
                "RabitQ residual qcode staging requires positive D0 row tile, row chunk, leader tile, and partial R"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_sq_wmma_mode.is_enabled()
            || args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
            || args.gpu_subtree_sq_wmma_sidecar_prefix.is_some()
            || args.gpu_subtree_sq_native_int4
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_mode".to_string(),
                "RabitQ residual qcode staging cannot be combined with SQ".to_string(),
            ));
        }
        if args.gpu_subtree_rabitq_plus_mode.is_enabled()
            || args.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled()
            || args.gpu_subtree_rabitq_plus_sidecar_prefix.is_some()
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_mode".to_string(),
                "RabitQ residual qcode staging cannot be combined with dense-i8 RabitQ+ sidecar modes"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_turboquant_mode.is_enabled()
            || args.gpu_subtree_turboquant_sidecar_mode.is_enabled()
            || args.gpu_subtree_turboquant_sidecar_prefix.is_some()
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_mode".to_string(),
                "RabitQ residual qcode staging cannot be combined with TurboQuant".to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_mode.is_enabled()
            || args.gpu_subtree_d0_dco_mode.is_enabled()
            || args.gpu_subtree_leaf_dco_mode.is_enabled()
            || args.gpu_subtree_assignment_dco_audit
            || args.gpu_subtree_d0_dco_audit
            || args.gpu_subtree_leaf_dco_audit
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_qcode_staging_mode".to_string(),
                "RabitQ residual qcode staging owns D0/D1+ approximate distance and cannot be combined with DCO/audit switches"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_vector_staging_mode != PiPNNGpuSubtreeVectorStagingMode::Off {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_vector_staging_mode".to_string(),
                "RabitQ residual qcode staging uses resident qcodes and cannot be combined with D1+ raw-vector staging"
                    .to_string(),
            ));
        }
    }

    if matches!(
        args.gpu_subtree_d0_partition_mode,
        PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK
    ) {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_partition_mode".to_string(),
                "tagore-kmeans-topk D0 mode requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !args.root_stop_after_d00 && !gpu_subtree_d0_capacity_enabled {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_enable".to_string(),
                "tagore-kmeans-topk full build requires --gpu-subtree-d0-capacity-enable for lazy D1+ admission split"
                    .to_string(),
            ));
        }
    }
    if matches!(
        args.gpu_subtree_d0_partition_mode,
        PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
    ) {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_partition_mode".to_string(),
                "gpu-kmeans-coarse D0 mode requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !matches!(
            args.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0
                | PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
                | PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
        ) {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_staging_mode".to_string(),
                "gpu-kmeans-coarse D0 mode requires --gpu-subtree-sq-staging-mode gpu-d0, gpu-d0-shard-local-d1, or gpu-d0-cluster-batch-d1"
                    .to_string(),
            ));
        }
        if !args.root_stop_after_d00 && !gpu_subtree_d0_capacity_enabled {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_enable".to_string(),
                "gpu-kmeans-coarse full build requires --gpu-subtree-d0-capacity-enable for lazy D1+ admission split"
                    .to_string(),
            ));
        }
    }

    if args.gpu_subtree_d1_root_partition_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d1_root_partition_mode".to_string(),
                "D1 root k-means partition mode requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if args.gpu_subtree_sq_staging_mode != PiPNNGpuSubtreeSqStagingMode::GpuD0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_staging_mode".to_string(),
                "D1 root k-means partition mode currently requires --gpu-subtree-sq-staging-mode gpu-d0"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_execution_mode != PiPNNGpuSubtreeExecutionMode::FusedPersistent {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_execution_mode".to_string(),
                "D1 root k-means partition mode currently requires --gpu-subtree-execution-mode fused-persistent"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_vector_staging_mode != PiPNNGpuSubtreeVectorStagingMode::Off {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_vector_staging_mode".to_string(),
                "D1 root k-means partition mode currently requires --gpu-subtree-vector-staging-mode off"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_child_schedule != PiPNNGpuSubtreeChildScheduleMode::Natural {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_child_schedule".to_string(),
                "D1 root k-means partition mode currently requires natural D1+ child order"
                    .to_string(),
            ));
        }
    }

    if args.gpu_subtree_assignment_dco_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_mode".to_string(),
                "assignment DCO requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !matches!(
            args.gpu_subtree_assignment_dco_mode,
            PiPNNGpuSubtreeDcoMode::PrefixProjection | PiPNNGpuSubtreeDcoMode::Quantized
        ) {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_mode".to_string(),
                "only assignment DCO prefix-projection and quantized are wired; keep other modes disabled"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_mode == PiPNNGpuSubtreeDcoMode::Quantized
            && args.gpu_subtree_assignment_backend
                != PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_backend".to_string(),
                "assignment DCO quantized currently requires --gpu-subtree-assignment-backend persistent-wmma-tf32"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_proj_dim == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_proj_dim".to_string(),
                "assignment DCO requires --gpu-subtree-assignment-dco-proj-dim > 0".to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_survivors == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_survivors".to_string(),
                "assignment DCO requires --gpu-subtree-assignment-dco-survivors > 0".to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_rerank_mode == PiPNNGpuSubtreeDcoRerankMode::ExactStamp {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_rerank_mode".to_string(),
                "assignment DCO does not support exact-stamp; use exact or none".to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_mode == PiPNNGpuSubtreeDcoMode::Quantized
            && args.gpu_subtree_assignment_dco_survivors > 4
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_survivors".to_string(),
                "assignment DCO quantized currently supports at most 4 WMMA survivors".to_string(),
            ));
        }
        if args.gpu_subtree_assignment_dco_survivors > 32 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_assignment_dco_survivors".to_string(),
                "assignment DCO currently supports at most 32 survivors".to_string(),
            ));
        }
    }
    if args.gpu_subtree_d0_dco_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_mode".to_string(),
                "D0 DCO requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !args.oom_enable {
            return Err(AnnError::log_index_config_error(
                "oom_enable".to_string(),
                "D0 DCO requires --oom-enable".to_string(),
            ));
        }
        if args.root_fanout_policy != RootFanoutPolicy::Fixed {
            return Err(AnnError::log_index_config_error(
                "root_fanout_policy".to_string(),
                "D0 DCO currently supports only fixed root fanout".to_string(),
            ));
        }
        if args.gpu_subtree_d0_partition_mode != PiPNNGpuSubtreeD0PartitionMode::RbcFanout {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_partition_mode".to_string(),
                "D0 DCO currently supports only rbc-fanout D0 partition mode".to_string(),
            ));
        }
        let d0_rabitq_dco = matches!(
            args.gpu_subtree_d0_dco_mode,
            PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        );
        let d0_sq_dco = args.gpu_subtree_d0_dco_mode == PiPNNGpuSubtreeDcoMode::SqInt8Filter;
        if !d0_rabitq_dco && !d0_sq_dco {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_mode".to_string(),
                "D0 DCO currently supports rabitq-split-filter/rabitq-wmma-filter or sq-int8-filter"
                    .to_string(),
            ));
        }
        if d0_rabitq_dco
            && args.gpu_subtree_d0_dco_rerank_mode != PiPNNGpuSubtreeDcoRerankMode::None
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_rerank_mode".to_string(),
                "D0 RabitQ-Library 1-bit/ex-bit boosting requires --gpu-subtree-d0-dco-rerank-mode none"
                    .to_string(),
            ));
        }
        if d0_sq_dco && args.gpu_subtree_d0_dco_rerank_mode != PiPNNGpuSubtreeDcoRerankMode::None {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_rerank_mode".to_string(),
                "D0 SQ-int8 approximate-only requires --gpu-subtree-d0-dco-rerank-mode none"
                    .to_string(),
            ));
        }
        if d0_rabitq_dco && args.gpu_subtree_d0_dco_audit {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_audit".to_string(),
                "D0 RabitQ-Library 1-bit/ex-bit boosting does not support raw audit because it would reread raw points"
                    .to_string(),
            ));
        }
        if d0_sq_dco && args.gpu_subtree_d0_dco_audit {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_audit".to_string(),
                "D0 SQ-int8 approximate-only does not support raw audit because it would reread raw points"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_d0_dco_rerank_mode == PiPNNGpuSubtreeDcoRerankMode::ExactStamp {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_rerank_mode".to_string(),
                "D0 DCO does not support exact-stamp; use exact or none".to_string(),
            ));
        }
        if args.gpu_subtree_d0_dco_survivors == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_survivors".to_string(),
                "D0 DCO requires --gpu-subtree-d0-dco-survivors > 0".to_string(),
            ));
        }
        if args.gpu_subtree_d0_dco_survivors < args.fanout_top {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_dco_survivors".to_string(),
                "D0 DCO survivors must be at least --fanout-top".to_string(),
            ));
        }
        if d0_rabitq_dco
            && (!matches!(
                args.gpu_subtree_rabitq_plus_sidecar_mode,
                PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly
                    | PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
                    | PiPNNGpuSubtreeRabitqPlusSidecarMode::Rebuild
            ) || args.gpu_subtree_rabitq_plus_sidecar_prefix.is_none())
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_plus_sidecar_mode".to_string(),
                "D0 RabitQ-Library split DCO requires --gpu-subtree-rabitq-plus-sidecar-mode create-only|require|rebuild and --gpu-subtree-rabitq-plus-sidecar-prefix"
                    .to_string(),
            ));
        }
        if d0_sq_dco
            && (!matches!(
                args.gpu_subtree_sq_wmma_sidecar_mode,
                PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly
                    | PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
            ) || args.gpu_subtree_sq_wmma_sidecar_prefix.is_none())
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_wmma_sidecar_mode".to_string(),
                "D0 SQ-int8 approximate-only must regenerate its SQ+WMMA sidecar; use --gpu-subtree-sq-wmma-sidecar-mode create-only|rebuild and --gpu-subtree-sq-wmma-sidecar-prefix"
                    .to_string(),
            ));
        }
    }
    if args.gpu_subtree_leaf_dco_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_mode".to_string(),
                "leaf top-k DCO requires --partitioner gpu-subtree".to_string(),
            ));
        }
        let leaf_dco_backend_supported = match args.gpu_subtree_leaf_dco_mode {
            PiPNNGpuSubtreeDcoMode::WmmaFilter | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter => {
                args.gpu_subtree_leaf_topk_backend
                    == PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
            }
            _ => {
                args.gpu_subtree_leaf_topk_backend
                    == PiPNNGpuSubtreeLeafTopKBackend::PersistentTiledGemmFp32
            }
        };
        if !leaf_dco_backend_supported {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_mode".to_string(),
                "leaf top-k DCO prefix-projection/ddc-res require --gpu-subtree-leaf-topk-backend persistent-tiled-gemm-fp32; wmma-filter/rabitq-split-filter require persistent-wmma-tf32"
                    .to_string(),
            ));
        }
        if !matches!(
            args.gpu_subtree_leaf_dco_mode,
            PiPNNGpuSubtreeDcoMode::PrefixProjection
                | PiPNNGpuSubtreeDcoMode::DdcRes
                | PiPNNGpuSubtreeDcoMode::WmmaFilter
                | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        ) {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_mode".to_string(),
                "only leaf top-k DCO prefix-projection, ddc-res, wmma-filter, and rabitq-split-filter are wired; dense-i8 RabitQ+ WMMA uses --gpu-subtree-rabitq-plus-mode build instead of --gpu-subtree-leaf-dco-mode rabitq-wmma-filter"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::DdcRes
            && args.gpu_subtree_leaf_dco_rerank_mode != PiPNNGpuSubtreeDcoRerankMode::Exact
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_rerank_mode".to_string(),
                "leaf top-k DCO ddc-res requires exact rerank mode because pruned pairs are skipped and surviving pairs use exact distances"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::DdcRes {
            if args.gpu_subtree_leaf_dco_ddc_res_pca_matrix.is_none() {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_ddc_res_pca_matrix".to_string(),
                    "leaf top-k DCO ddc-res requires a ResInfer PCA matrix sidecar; raw-space DDC_res is invalid"
                        .to_string(),
                ));
            }
            if !args.gpu_subtree_leaf_dco_ddc_res_projected_input {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_ddc_res_projected_input".to_string(),
                    "leaf top-k DCO ddc-res currently requires the input data file to already be in the PCA coordinate system generated by ResInfer"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_dco_proj_dim != 0 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_proj_dim".to_string(),
                    "leaf top-k DCO ddc-res uses --gpu-subtree-leaf-dco-ddc-res-base-dim, not --gpu-subtree-leaf-dco-proj-dim"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_dco_ddc_res_base_dim == 0 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_ddc_res_base_dim".to_string(),
                    "leaf top-k DCO ddc-res requires --gpu-subtree-leaf-dco-ddc-res-base-dim > 0"
                        .to_string(),
                ));
            }
            if !args.gpu_subtree_leaf_dco_ddc_res_sigma_count.is_finite()
                || args.gpu_subtree_leaf_dco_ddc_res_sigma_count <= 0.0
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_ddc_res_sigma_count".to_string(),
                    "leaf top-k DCO ddc-res requires finite sigma_count > 0".to_string(),
                ));
            }
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::PrefixProjection
            && args.gpu_subtree_leaf_dco_proj_dim == 0
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_proj_dim".to_string(),
                "leaf top-k DCO prefix-projection requires --gpu-subtree-leaf-dco-proj-dim > 0"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::PrefixProjection
            && args.gpu_subtree_leaf_dco_survivors == 0
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_survivors".to_string(),
                "leaf top-k DCO prefix-projection requires --gpu-subtree-leaf-dco-survivors > 0"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::PrefixProjection
            && args.gpu_subtree_leaf_dco_survivors > 32
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_leaf_dco_survivors".to_string(),
                "leaf top-k DCO prefix-projection currently supports at most 32 survivors"
                    .to_string(),
            ));
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::WmmaFilter {
            if args.gpu_subtree_leaf_dco_survivors == 0 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_survivors".to_string(),
                    "leaf top-k DCO wmma-filter requires --gpu-subtree-leaf-dco-survivors > 0"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_dco_survivors > 32 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_survivors".to_string(),
                    "leaf top-k DCO wmma-filter currently supports at most 32 survivors"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_dco_proj_dim != 0 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_proj_dim".to_string(),
                    "leaf top-k DCO wmma-filter does not use --gpu-subtree-leaf-dco-proj-dim"
                        .to_string(),
                ));
            }
        }
        if args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::RabitqSplitFilter {
            if args.gpu_subtree_leaf_dco_survivors == 0 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_survivors".to_string(),
                    "leaf top-k DCO rabitq-split-filter requires --gpu-subtree-leaf-dco-survivors > 0"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_dco_survivors > 32 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_survivors".to_string(),
                    "leaf top-k DCO rabitq-split-filter currently supports at most 32 survivors"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_dco_proj_dim != 0 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_dco_proj_dim".to_string(),
                    "leaf top-k DCO rabitq-split-filter uses fixed 1bit+7bit split_distance_boosting and does not use --gpu-subtree-leaf-dco-proj-dim"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_rabitq_plus_bits != 8 {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_rabitq_plus_bits".to_string(),
                    format!(
                        "leaf top-k DCO rabitq-split-filter currently supports --gpu-subtree-rabitq-plus-bits=8 (1bit+7bit), got {}",
                        args.gpu_subtree_rabitq_plus_bits
                    ),
                ));
            }
            if args.gpu_subtree_rabitq_plus_rotator != PiPNNGpuSubtreeRabitqPlusRotator::FhtKac {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_rabitq_plus_rotator".to_string(),
                    format!(
                        "leaf top-k DCO rabitq-split-filter requires --gpu-subtree-rabitq-plus-rotator fht-kac, got {}",
                        args.gpu_subtree_rabitq_plus_rotator.as_str()
                    ),
                ));
            }
            if !matches!(
                args.gpu_subtree_rabitq_plus_sidecar_mode,
                PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
                    | PiPNNGpuSubtreeRabitqPlusSidecarMode::Rebuild
            ) || args.gpu_subtree_rabitq_plus_sidecar_prefix.is_none()
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_rabitq_plus_sidecar_mode".to_string(),
                    "leaf top-k DCO rabitq-split-filter requires --gpu-subtree-rabitq-plus-sidecar-mode require|rebuild and --gpu-subtree-rabitq-plus-sidecar-prefix"
                        .to_string(),
                ));
            }
        }
    }

    if args.gpu_subtree_rabitq_plus_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_plus_mode".to_string(),
                "Dense-i8 RabitQ+ GPU-subtree mode requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !matches!(args.gpu_subtree_rabitq_plus_bits, 4 | 5 | 7 | 8) {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_rabitq_plus_bits".to_string(),
                "Dense-i8 RabitQ+ total bits must be one of 4, 5, 7, or 8".to_string(),
            ));
        }
        if args.gpu_subtree_rabitq_plus_mode == PiPNNGpuSubtreeRabitqPlusMode::Build {
            if !matches!(
                args.gpu_subtree_rabitq_plus_sidecar_mode,
                PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
                    | PiPNNGpuSubtreeRabitqPlusSidecarMode::Rebuild
            ) || args.gpu_subtree_rabitq_plus_sidecar_prefix.is_none()
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_rabitq_plus_sidecar_mode".to_string(),
                    "Dense-i8 RabitQ+ build mode requires --gpu-subtree-rabitq-plus-sidecar-mode require|rebuild and --gpu-subtree-rabitq-plus-sidecar-prefix"
                        .to_string(),
                ));
            }
            if !args.pipnn_backend.eq_ignore_ascii_case("cuda") {
                return Err(AnnError::log_index_config_error(
                    "pipnn_backend".to_string(),
                    "Dense-i8 RabitQ+ build mode requires --pipnn-backend=cuda".to_string(),
                ));
            }
            if args.gpu_subtree_assignment_backend
                != PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_assignment_backend".to_string(),
                    "Dense-i8 RabitQ+ build mode currently requires --gpu-subtree-assignment-backend persistent-wmma-tf32"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_topk_backend
                != PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_topk_backend".to_string(),
                    "Dense-i8 RabitQ+ build mode currently requires --gpu-subtree-leaf-topk-backend persistent-wmma-tf32"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_assignment_dco_mode.is_enabled()
                || args.gpu_subtree_leaf_dco_mode.is_enabled()
                || args.gpu_subtree_assignment_dco_audit
                || args.gpu_subtree_leaf_dco_audit
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_rabitq_plus_mode".to_string(),
                    "Dense-i8 RabitQ+ build mode owns assignment/leaf quantized distance and cannot be combined with existing DCO/audit switches"
                        .to_string(),
                ));
            }
        }
    } else if !matches!(args.gpu_subtree_rabitq_plus_bits, 4 | 5 | 7 | 8) {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_rabitq_plus_bits".to_string(),
            "Dense-i8 RabitQ+ total bits must be one of 4, 5, 7, or 8".to_string(),
        ));
    }
    if args.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled()
        && args.gpu_subtree_rabitq_plus_sidecar_prefix.is_none()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_rabitq_plus_sidecar_prefix".to_string(),
            "RabitQ sidecar mode requires --gpu-subtree-rabitq-plus-sidecar-prefix".to_string(),
        ));
    }
    if args.gpu_subtree_rabitq_plus_sidecar_prefix.is_some()
        && !args.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_rabitq_plus_sidecar_mode".to_string(),
            "RabitQ sidecar prefix requires --gpu-subtree-rabitq-plus-sidecar-mode".to_string(),
        ));
    }
    if args.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled()
        && !args.data_type.eq_ignore_ascii_case("float")
    {
        return Err(AnnError::log_index_config_error(
            "data_type".to_string(),
            "RabitQ sidecar creation/loading currently requires --data-type float".to_string(),
        ));
    }
    if args.estimate_only && args.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled() {
        return Err(AnnError::log_index_config_error(
            "estimate_only".to_string(),
            "RabitQ sidecar mode conflicts with --estimate-only".to_string(),
        ));
    }
    if matches!(
        args.gpu_subtree_rabitq_plus_sidecar_mode,
        PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
            | PiPNNGpuSubtreeRabitqPlusSidecarMode::Rebuild
    ) && !args.gpu_subtree_rabitq_plus_mode.is_enabled()
        && !matches!(
            args.gpu_subtree_d0_dco_mode,
            PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        )
        && args.gpu_subtree_leaf_dco_mode != PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_rabitq_plus_mode".to_string(),
            "Dense-i8 RabitQ+ sidecar require/rebuild is meaningful with --gpu-subtree-rabitq-plus-mode shadow|build; RabitQ-Library split sidecar require/rebuild is meaningful with --gpu-subtree-d0-dco-mode rabitq-split-filter or --gpu-subtree-leaf-dco-mode rabitq-split-filter"
                .to_string(),
        ));
    }

    if args.gpu_subtree_rabitq_plus_mode.is_enabled() && args.gpu_subtree_sq_wmma_mode.is_enabled()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_wmma_mode".to_string(),
            "SQ+WMMA mode and dense-i8 RabitQ+ mode both own D1+ quantized distance and cannot be enabled together"
                .to_string(),
        ));
    }
    if args.gpu_subtree_turboquant_bits != 4 {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_bits".to_string(),
            "TurboQuant-MSE4 v1 supports only --gpu-subtree-turboquant-bits 4".to_string(),
        ));
    }
    if args.gpu_subtree_turboquant_cascade_low_dim == 0 {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_cascade_low_dim".to_string(),
            "TurboQuant-MSE4 cascade low_dim must be > 0".to_string(),
        ));
    }
    if args.gpu_subtree_turboquant_cascade_survivor_cap > TURBOQUANT_CASCADE_MAX_SURVIVORS {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_cascade_survivor_cap".to_string(),
            format!(
                "TurboQuant-MSE4 cascade survivor_cap currently supports at most {TURBOQUANT_CASCADE_MAX_SURVIVORS}"
            ),
        ));
    }
    if args.gpu_subtree_turboquant_adaptive_survivor_low_cap > TURBOQUANT_CASCADE_MAX_SURVIVORS {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_adaptive_survivor_low_cap".to_string(),
            format!(
                "TurboQuant-MSE4 adaptive survivor low cap currently supports at most {TURBOQUANT_CASCADE_MAX_SURVIVORS}"
            ),
        ));
    }
    let adaptive_threshold_set = args.gpu_subtree_turboquant_adaptive_survivor_threshold > 0;
    let adaptive_low_cap_set = args.gpu_subtree_turboquant_adaptive_survivor_low_cap > 0;
    if adaptive_threshold_set != adaptive_low_cap_set {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_adaptive_survivor_threshold".to_string(),
            "TurboQuant-MSE4 adaptive survivor threshold and low cap must be set together"
                .to_string(),
        ));
    }
    let enabled_d1_quantized_modes = usize::from(args.gpu_subtree_rabitq_plus_mode.is_enabled())
        + usize::from(args.gpu_subtree_sq_wmma_mode.is_enabled())
        + usize::from(args.gpu_subtree_turboquant_mode.is_enabled());
    if enabled_d1_quantized_modes > 1 {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_mode".to_string(),
            "TurboQuant-MSE4, SQ+WMMA, and dense-i8 RabitQ+ modes all own D1+ quantized distance and cannot be enabled together"
                .to_string(),
        ));
    }

    if args.gpu_subtree_sq_wmma_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_wmma_mode".to_string(),
                "SQ+WMMA GPU-subtree mode requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if args.gpu_subtree_sq_wmma_mode == PiPNNGpuSubtreeSqWmmaMode::Build {
            let sq_direct_d1 = matches!(
                args.gpu_subtree_sq_staging_mode,
                PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
                    | PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
            ) || args.gpu_subtree_sq8_rotated_residual_d1 || args.gpu_subtree_sq4_plus;
            if !sq_direct_d1
                && (!matches!(
                    args.gpu_subtree_sq_wmma_sidecar_mode,
                    PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
                ) || args.gpu_subtree_sq_wmma_sidecar_prefix.is_none())
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_sq_wmma_sidecar_mode".to_string(),
                    "SQ+WMMA build mode must regenerate its SQ+WMMA sidecar; use --gpu-subtree-sq-wmma-sidecar-mode rebuild and --gpu-subtree-sq-wmma-sidecar-prefix"
                        .to_string(),
                ));
            }
            if !args.pipnn_backend.eq_ignore_ascii_case("cuda") {
                return Err(AnnError::log_index_config_error(
                    "pipnn_backend".to_string(),
                    "SQ+WMMA build mode requires --pipnn-backend=cuda".to_string(),
                ));
            }
            if args.gpu_subtree_assignment_backend
                != PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_assignment_backend".to_string(),
                    "SQ+WMMA build mode currently requires --gpu-subtree-assignment-backend persistent-wmma-tf32"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_topk_backend
                != PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_topk_backend".to_string(),
                    "SQ+WMMA build mode currently requires --gpu-subtree-leaf-topk-backend persistent-wmma-tf32"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_assignment_dco_mode.is_enabled()
                || args.gpu_subtree_leaf_dco_mode.is_enabled()
                || args.gpu_subtree_assignment_dco_audit
                || args.gpu_subtree_leaf_dco_audit
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_sq_wmma_mode".to_string(),
                    "SQ+WMMA build mode owns assignment/leaf quantized distance and cannot be combined with existing DCO/audit switches"
                        .to_string(),
                ));
            }
        }
    }
    if args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
        && args.gpu_subtree_sq_wmma_sidecar_prefix.is_none()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_wmma_sidecar_prefix".to_string(),
            "SQ+WMMA sidecar mode requires --gpu-subtree-sq-wmma-sidecar-prefix".to_string(),
        ));
    }
    if args.gpu_subtree_sq_wmma_sidecar_prefix.is_some()
        && !args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_wmma_sidecar_mode".to_string(),
            "SQ+WMMA sidecar prefix requires --gpu-subtree-sq-wmma-sidecar-mode".to_string(),
        ));
    }
    if args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
        && !args.data_type.eq_ignore_ascii_case("float")
        && !args.data_type.eq_ignore_ascii_case("uint8")
    {
        return Err(AnnError::log_index_config_error(
            "data_type".to_string(),
            "SQ+WMMA sidecar creation/loading currently requires --data-type float or --data-type uint8".to_string(),
        ));
    }
    if args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
        && args.data_type.eq_ignore_ascii_case("uint8")
    {
        let native_sq8 = args.gpu_subtree_sq_bits == 8 && !args.gpu_subtree_sq_native_int4;
        let native_sq4 = args.gpu_subtree_sq_bits == 4 && args.gpu_subtree_sq_native_int4;
        if !native_sq8 && !native_sq4 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_sq_bits".to_string(),
                "uint8 SQ+WMMA sidecar supports native SQ8 dense-i8 with --gpu-subtree-sq-bits 8, or native SQ4 packed-s4 with --gpu-subtree-sq-bits 4 --gpu-subtree-sq-native-int4".to_string(),
            ));
        }
    }
    let uint8_skips_index_build = args.estimate_only
        || args.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly;
    if args.data_type.eq_ignore_ascii_case("uint8") && !uint8_skips_index_build {
        if !args.oom_enable {
            return Err(AnnError::log_index_config_error(
                "oom_enable".to_string(),
                "uint8 PiPNN build requires --oom-enable".to_string(),
            ));
        }
        if args.final_prune {
            return Err(AnnError::log_index_config_error(
                "final_prune".to_string(),
                "uint8 PiPNN build currently requires --final-prune=false".to_string(),
            ));
        }
    }
    if args.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::Require {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_wmma_sidecar_mode".to_string(),
            "SQ+WMMA sidecar require would reuse an existing sidecar; use create-only for diagnostics or rebuild for a graph build"
                .to_string(),
        ));
    }
    if args.estimate_only && args.gpu_subtree_sq_wmma_sidecar_mode.is_enabled() {
        return Err(AnnError::log_index_config_error(
            "estimate_only".to_string(),
            "SQ+WMMA sidecar mode conflicts with --estimate-only".to_string(),
        ));
    }
    if matches!(
        args.gpu_subtree_sq_wmma_sidecar_mode,
        PiPNNGpuSubtreeSqWmmaSidecarMode::Require | PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
    ) && !args.gpu_subtree_sq_wmma_mode.is_enabled()
        && args.gpu_subtree_d0_dco_mode != PiPNNGpuSubtreeDcoMode::SqInt8Filter
        && args.gpu_subtree_sq_staging_mode != PiPNNGpuSubtreeSqStagingMode::GpuD0
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq_wmma_mode".to_string(),
            "SQ+WMMA sidecar require/rebuild is only meaningful with --gpu-subtree-sq-wmma-mode shadow|build, --gpu-subtree-d0-dco-mode sq-int8-filter, or --gpu-subtree-sq-staging-mode gpu-d0"
                .to_string(),
        ));
    }

    if args.gpu_subtree_turboquant_mode.is_enabled() {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_turboquant_mode".to_string(),
                "TurboQuant-MSE4 GPU-subtree mode requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if args.gpu_subtree_turboquant_mode == PiPNNGpuSubtreeTurboQuantMode::Build {
            if !matches!(
                args.gpu_subtree_turboquant_sidecar_mode,
                PiPNNGpuSubtreeTurboQuantSidecarMode::Require
                    | PiPNNGpuSubtreeTurboQuantSidecarMode::Rebuild
            ) || args.gpu_subtree_turboquant_sidecar_prefix.is_none()
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_turboquant_sidecar_mode".to_string(),
                    "TurboQuant-MSE4 build mode requires --gpu-subtree-turboquant-sidecar-mode require|rebuild and --gpu-subtree-turboquant-sidecar-prefix"
                        .to_string(),
                ));
            }
            if !args.pipnn_backend.eq_ignore_ascii_case("cuda") {
                return Err(AnnError::log_index_config_error(
                    "pipnn_backend".to_string(),
                    "TurboQuant-MSE4 build mode requires --pipnn-backend=cuda".to_string(),
                ));
            }
            if args.gpu_subtree_assignment_backend
                != PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_assignment_backend".to_string(),
                    "TurboQuant-MSE4 build mode currently requires --gpu-subtree-assignment-backend persistent-wmma-tf32"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_leaf_topk_backend
                != PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_leaf_topk_backend".to_string(),
                    "TurboQuant-MSE4 build mode currently requires --gpu-subtree-leaf-topk-backend persistent-wmma-tf32"
                        .to_string(),
                ));
            }
            if args.gpu_subtree_assignment_dco_mode.is_enabled()
                || args.gpu_subtree_leaf_dco_mode.is_enabled()
                || args.gpu_subtree_assignment_dco_audit
                || args.gpu_subtree_leaf_dco_audit
            {
                return Err(AnnError::log_index_config_error(
                    "gpu_subtree_turboquant_mode".to_string(),
                    "TurboQuant-MSE4 build mode owns assignment/leaf quantized distance and cannot be combined with existing DCO/audit switches"
                        .to_string(),
                ));
            }
        }
    }
    if args.gpu_subtree_turboquant_sidecar_mode.is_enabled()
        && args.gpu_subtree_turboquant_sidecar_prefix.is_none()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_sidecar_prefix".to_string(),
            "TurboQuant-MSE4 sidecar mode requires --gpu-subtree-turboquant-sidecar-prefix"
                .to_string(),
        ));
    }
    if args.gpu_subtree_turboquant_sidecar_prefix.is_some()
        && !args.gpu_subtree_turboquant_sidecar_mode.is_enabled()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_sidecar_mode".to_string(),
            "TurboQuant-MSE4 sidecar prefix requires --gpu-subtree-turboquant-sidecar-mode"
                .to_string(),
        ));
    }
    if args.gpu_subtree_turboquant_sidecar_mode.is_enabled()
        && !args.data_type.eq_ignore_ascii_case("float")
    {
        return Err(AnnError::log_index_config_error(
            "data_type".to_string(),
            "TurboQuant-MSE4 sidecar creation/loading currently requires --data-type float"
                .to_string(),
        ));
    }
    if args.estimate_only && args.gpu_subtree_turboquant_sidecar_mode.is_enabled() {
        return Err(AnnError::log_index_config_error(
            "estimate_only".to_string(),
            "TurboQuant-MSE4 sidecar mode conflicts with --estimate-only".to_string(),
        ));
    }
    if matches!(
        args.gpu_subtree_turboquant_sidecar_mode,
        PiPNNGpuSubtreeTurboQuantSidecarMode::Require
            | PiPNNGpuSubtreeTurboQuantSidecarMode::Rebuild
    ) && !args.gpu_subtree_turboquant_mode.is_enabled()
    {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_turboquant_mode".to_string(),
            "TurboQuant-MSE4 sidecar require/rebuild is only meaningful with --gpu-subtree-turboquant-mode shadow|build"
                .to_string(),
        ));
    }

    if gpu_subtree_d0_capacity_enabled {
        if !gpu_subtree_partitioner {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_enable".to_string(),
                "D0 capacity safety valve requires --partitioner gpu-subtree".to_string(),
            ));
        }
        if !args.oom_enable {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_enable".to_string(),
                "D0 capacity safety valve requires --oom-enable".to_string(),
            ));
        }
        if args.gpu_subtree_d0_capacity_min_efficient_rows == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_min_efficient_rows".to_string(),
                "D0 capacity safety min efficient rows must be > 0".to_string(),
            ));
        }
        if args.gpu_subtree_d0_capacity_quality_min_root_leaders == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_quality_min_root_leaders".to_string(),
                "D0 configured root leaders must be > 0".to_string(),
            ));
        }
        if args.gpu_subtree_d0_capacity_root_leader_cap == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_root_leader_cap".to_string(),
                "D0 capacity safety root leader cap must be > 0".to_string(),
            ));
        }
        if args.gpu_subtree_d0_capacity_split_recursion_limit == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_capacity_split_recursion_limit".to_string(),
                "D0 capacity-aware split recursion limit must be > 0".to_string(),
            ));
        }
    }
    if args.gpu_subtree_d0_throughput_model != PiPNNGpuSubtreeD0ThroughputModelMode::Off {
        if !gpu_subtree_d0_capacity_enabled {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_throughput_model".to_string(),
                "D0 throughput model requires --gpu-subtree-d0-capacity-enable".to_string(),
            ));
        }
        if args.gpu_subtree_d0_throughput_model == PiPNNGpuSubtreeD0ThroughputModelMode::Enforce
            && args.gpu_subtree_d0_throughput_calibration_json.is_none()
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_throughput_calibration_json".to_string(),
                "D0 throughput enforce mode requires --gpu-subtree-d0-throughput-calibration-json"
                    .to_string(),
            ));
        }
        if !args
            .gpu_subtree_d0_throughput_persistent_growth_limit
            .is_finite()
            || args.gpu_subtree_d0_throughput_persistent_growth_limit < 1.0
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_throughput_persistent_growth_limit".to_string(),
                "D0 throughput persistent growth limit must be finite and >= 1.0".to_string(),
            ));
        }
        if args.gpu_subtree_d0_throughput_preload_visible_limit_ms == 0 {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_throughput_preload_visible_limit_ms".to_string(),
                "D0 throughput preload visible limit must be > 0".to_string(),
            ));
        }
        if !args.gpu_subtree_d0_throughput_min_ratio.is_finite()
            || args.gpu_subtree_d0_throughput_min_ratio <= 0.0
            || args.gpu_subtree_d0_throughput_min_ratio > 1.0
        {
            return Err(AnnError::log_index_config_error(
                "gpu_subtree_d0_throughput_min_ratio".to_string(),
                "D0 throughput min ratio must be finite and in (0, 1]".to_string(),
            ));
        }
    }
    let (capacity_high, capacity_target, capacity_low) = effective_gpu_subtree_d0_capacity_ratios(args);
    for (name, ratio) in [
        (
            "gpu_subtree_d0_capacity_high_ratio",
            capacity_high,
        ),
        (
            "gpu_subtree_d0_capacity_target_ratio",
            capacity_target,
        ),
        (
            "gpu_subtree_d0_capacity_low_ratio",
            capacity_low,
        ),
    ] {
        if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
            return Err(AnnError::log_index_config_error(
                name.to_string(),
                "D0 capacity ratio must be finite and in (0, 1]".to_string(),
            ));
        }
    }
    if capacity_target > capacity_high {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_d0_capacity_target_ratio".to_string(),
            "D0 capacity target ratio must be <= high ratio".to_string(),
        ));
    }
    if !args.oom_reservoir_share.is_finite()
        || args.oom_reservoir_share < 0.0
        || args.oom_reservoir_share > 1.0
    {
        return Err(AnnError::log_index_config_error(
            "oom_reservoir_share".to_string(),
            "OOM reservoir share must be finite and in [0, 1]".to_string(),
        ));
    }
    if let Some(cap_gb) = args.oom_sidecar_resident_budget_cap_gb {
        if !cap_gb.is_finite() || cap_gb < 0.0 {
            return Err(AnnError::log_index_config_error(
                "oom_sidecar_resident_budget_cap_gb".to_string(),
                "OOM sidecar resident budget cap must be finite and non-negative".to_string(),
            ));
        }
    }

    if args.oom_enable {
        if args.oom_strict_io && matches!(args.oom_io_mode, PiPNNOomIoMode::Buffered) {
            return Err(AnnError::log_index_config_error(
                "oom_io_mode".to_string(),
                "--oom-strict-io conflicts with --oom-io-mode=buffered".to_string(),
            ));
        }
        let explicit_cuda_pipeline = args.pipeline_runtime_enable
            && matches!(args.pipeline_compute_mode, PiPNNPipelineComputeMode::Cuda)
            && args.pipnn_backend.eq_ignore_ascii_case("cuda");
        if !args.pipnn_backend.eq_ignore_ascii_case("cpu")
            && !explicit_cuda_pipeline
            && !gpu_subtree_partitioner
        {
            return Err(AnnError::log_index_config_error(
                "pipnn_backend".to_string(),
                concat!(
                    "OOM PiPNN mode supports --pipnn-backend=cpu, or explicit ",
                    "--pipnn-backend=cuda with --pipeline-runtime-enable --pipeline-compute-mode=cuda, ",
                    "or --partitioner=gpu-subtree"
                )
                .to_string(),
            ));
        }
    }

    if args.root_radius_k == 0 || args.root_radius_k > 32 {
        return Err(AnnError::log_index_config_error(
            "root_radius_k".to_string(),
            "root radius k must be in [1, 32]".to_string(),
        ));
    }
    if parse_root_radius_candidate_rs(&args.root_radius_candidate_rs)?.is_empty() {
        return Err(AnnError::log_index_config_error(
            "root_radius_candidate_rs".to_string(),
            "root radius candidate rs must contain at least one positive integer".to_string(),
        ));
    }
    if matches!(
        args.root_fanout_policy,
        RootFanoutPolicy::CellRadiusCertified
    ) && !args.root_radius_dry_run
        && !args.root_stop_after_d00
    {
        return Err(AnnError::log_index_config_error(
            "root_fanout_policy".to_string(),
            "cell-radius-certified is diagnostic-only; use --root-radius-dry-run or --root-stop-after-d00"
                .to_string(),
        ));
    }
    let uses_root_risk =
        matches!(args.root_fanout_policy, RootFanoutPolicy::RiskBudgeted) || args.root_risk_dry_run;
    if uses_root_risk {
        let root_risk_candidate_rs =
            parse_root_risk_candidate_rs(&args.root_risk_candidate_rs, args.fanout_top)?;
        let min_candidate = root_risk_candidate_rs.iter().copied().min().unwrap_or(1);
        if !args.root_risk_target_avg_fanout.is_finite()
            || args.root_risk_target_avg_fanout < min_candidate as f32
            || args.root_risk_target_avg_fanout > args.fanout_top as f32
        {
            return Err(AnnError::log_index_config_error(
                "root_risk_target_avg_fanout".to_string(),
                format!(
                    "root risk target avg fanout must be in [{min_candidate}, {}]",
                    args.fanout_top
                ),
            ));
        }
        if !args.root_risk_sigma_lambda.is_finite() || args.root_risk_sigma_lambda <= 0.0 {
            return Err(AnnError::log_index_config_error(
                "root_risk_sigma_lambda".to_string(),
                "root risk sigma lambda must be > 0".to_string(),
            ));
        }
        if args
            .root_risk_budget_lambda
            .is_some_and(|lambda| !lambda.is_finite() || lambda < 0.0)
        {
            return Err(AnnError::log_index_config_error(
                "root_risk_budget_lambda".to_string(),
                "root risk budget lambda must be finite and >= 0".to_string(),
            ));
        }
        if !args.root_risk_dry_run && !args.root_stop_after_d00 && !args.root_risk_allow_full_build
        {
            return Err(AnnError::log_index_config_error(
                "root_fanout_policy".to_string(),
                "risk-budgeted is root-only; use --root-risk-dry-run, --root-stop-after-d00, or --root-risk-allow-full-build"
                    .to_string(),
            ));
        }
    }
    if args.root_stop_after_d00 && !args.oom_enable {
        return Err(AnnError::log_index_config_error(
            "root_stop_after_d00".to_string(),
            "root-stop-after-d00 requires --oom-enable".to_string(),
        ));
    }
    if !args.pipnn_gpu_workspace_budget_gb.is_finite() || args.pipnn_gpu_workspace_budget_gb < 0.0 {
        return Err(AnnError::log_index_config_error(
            "pipnn_gpu_workspace_budget_gb".to_string(),
            "GPU workspace budget must be finite and >= 0".to_string(),
        ));
    }
    if matches!(args.partitioner, PiPNNPartitionerKind::KMeans) && args.kmeans_iterations == 0 {
        return Err(AnnError::log_index_config_error(
            "kmeans_iterations".to_string(),
            "kmeans iterations must be > 0".to_string(),
        ));
    }
    if matches!(args.partitioner, PiPNNPartitionerKind::KMeans) && args.kmeans_oversplit_factor == 0
    {
        return Err(AnnError::log_index_config_error(
            "kmeans_oversplit_factor".to_string(),
            "kmeans oversplit factor must be > 0".to_string(),
        ));
    }

    Ok(())
}

fn normalize_cli_path(path: &Path) -> AnnResult<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(Path::new("/")),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    Ok(normalized)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn apply_pipeline_args(params: &mut PiPNNParams, args: &Args) {
    params.pipeline_runtime_enable = args.pipeline_runtime_enable;
    params.pipeline_compute_mode = args.pipeline_compute_mode;
    params.pipeline_leaf_descriptor_queue_capacity = args.pipeline_leaf_descriptor_queue_capacity;
    params.pipeline_staged_queue_capacity = args.pipeline_staged_queue_capacity;
    params.pipeline_topk_queue_capacity = args.pipeline_topk_queue_capacity;
    params.pipeline_edge_queue_capacity = args.pipeline_edge_queue_capacity;
    params.pipeline_max_inflight_compute_tasks_active =
        args.pipeline_max_inflight_compute_tasks_active;
    params.pipeline_max_inflight_compute_tasks_drain =
        args.pipeline_max_inflight_compute_tasks_drain;
}

fn parse_root_radius_candidate_rs(value: &str) -> AnnResult<Vec<usize>> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = trimmed.parse::<usize>().map_err(|_| {
            AnnError::log_index_config_error(
                "root_radius_candidate_rs".to_string(),
                format!("invalid root radius candidate r '{trimmed}'"),
            )
        })?;
        if parsed > 0 {
            out.push(parsed);
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn parse_root_risk_candidate_rs(value: &str, fanout_top: usize) -> AnnResult<Vec<usize>> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = trimmed.parse::<usize>().map_err(|_| {
            AnnError::log_index_config_error(
                "root_risk_candidate_rs".to_string(),
                format!("invalid root risk candidate r '{trimmed}'"),
            )
        })?;
        if parsed == 0 || parsed > fanout_top {
            return Err(AnnError::log_index_config_error(
                "root_risk_candidate_rs".to_string(),
                "root risk candidate rs must be in [1, fanout_top]".to_string(),
            ));
        }
        out.push(parsed);
    }
    if fanout_top == 0 {
        return Err(AnnError::log_index_config_error(
            "fanout_top".to_string(),
            "fanout_top must be >= 1".to_string(),
        ));
    }
    out.push(fanout_top);
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err(AnnError::log_index_config_error(
            "root_risk_candidate_rs".to_string(),
            "root risk candidate rs must contain at least one positive integer".to_string(),
        ));
    }
    Ok(out)
}

fn annotate_completed_phases(phases: &mut [CompletedPhase], samples: &[ProcPhaseSample]) {
    for phase in phases {
        let mut peak_threads = 0usize;
        let mut peak_rss_kib: Option<usize> = None;
        let mut sample_count = 0usize;
        let mut max_cpu_cores = 0.0f64;
        let mut previous: Option<&ProcPhaseSample> = None;

        for sample in samples
            .iter()
            .filter(|sample| sample.phase_name == phase.name)
        {
            sample_count += 1;
            if let Some(thread_count) = sample.thread_count {
                peak_threads = peak_threads.max(thread_count);
            }
            if let Some(rss_kib) = sample.rss_kib {
                peak_rss_kib = Some(match peak_rss_kib {
                    Some(peak) => peak.max(rss_kib),
                    None => rss_kib,
                });
            }

            if let Some(prev) = previous {
                if let Some(cpu_cores) = sample_cpu_cores(prev, sample) {
                    max_cpu_cores = max_cpu_cores.max(cpu_cores);
                }
            }
            previous = Some(sample);
        }

        phase.peak_threads = peak_threads;
        phase.peak_rss_kib = peak_rss_kib;
        phase.max_cpu_cores = max_cpu_cores.max(phase_cpu_cores_used(phase));
        phase.sample_count = sample_count;
    }
}

fn sample_cpu_cores(previous: &ProcPhaseSample, current: &ProcPhaseSample) -> Option<f64> {
    let prev_running_ns = previous.running_ns?;
    let current_running_ns = current.running_ns?;
    if current_running_ns < prev_running_ns {
        return None;
    }

    let wall_delta = current.wall_offset.checked_sub(previous.wall_offset)?;
    if wall_delta.is_zero() {
        return None;
    }

    Some((current_running_ns - prev_running_ns) as f64 / wall_delta.as_secs_f64() / 1_000_000_000.0)
}

fn print_phase_monitoring_report(phases: &[CompletedPhase], samples: &[ProcPhaseSample]) {
    println!("=== PiPNN Phase Monitor ===");
    for phase in phases {
        println!(
            "  phase={} wall={:.3}s cpu={:.3}s avg_cpu_cores={:.2} peak_cpu_cores={:.2} peak_threads={} peak_rss_gib={} samples={}",
            phase.name,
            phase.wall_time.as_secs_f64(),
            phase.cpu_time.as_secs_f64(),
            phase_cpu_cores_used(phase),
            phase.max_cpu_cores,
            phase.peak_threads,
            format_optional_gib(phase.peak_rss_kib),
            phase.sample_count,
        );
    }

    if !samples.is_empty() {
        println!("  sampled timeline:");
        for sample in samples {
            println!(
                "    t={:.1}s phase={} threads={} rss_gib={} cpu_running_s={}",
                sample.wall_offset.as_secs_f64(),
                sample.phase_name,
                sample
                    .thread_count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                format_optional_gib(sample.rss_kib),
                sample
                    .running_ns
                    .map(|value| format!("{:.3}", value as f64 / 1_000_000_000.0))
                    .unwrap_or_else(|| "-".to_string()),
            );
        }
    }

    println!("===========================");
}

fn phase_cpu_cores_used(phase: &CompletedPhase) -> f64 {
    if phase.wall_time.is_zero() {
        0.0
    } else {
        phase.cpu_time.as_secs_f64() / phase.wall_time.as_secs_f64()
    }
}

fn format_optional_gib(rss_kib: Option<usize>) -> String {
    rss_kib
        .map(|value| format!("{:.2}", value as f64 / 1024.0 / 1024.0))
        .unwrap_or_else(|| "-".to_string())
}

fn write_driver_timing_json(path: &Path, report: &DriverTimingReport) -> AnnResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let profile_stage_total_ms = report
        .profile_json_path
        .as_deref()
        .and_then(read_profile_stage_total_ms);
    let total_wall_ms = duration_ms(report.total_wall);
    let build_call_wall_ms = duration_ms(report.build_call_wall);
    let total_minus_profile_stage_ms = signed_ms_delta(total_wall_ms, profile_stage_total_ms);
    let build_call_minus_profile_stage_ms =
        signed_ms_delta(build_call_wall_ms, profile_stage_total_ms);

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"schema\": \"pipnn_driver_timing_v1\",\n");
    json.push_str("  \"unit\": \"ms\",\n");
    json.push_str(&format!("  \"build_success\": {},\n", report.build_success));
    json.push_str(&format!("  \"total_wall_ms\": {},\n", total_wall_ms));
    json.push_str(&format!(
        "  \"pre_build_wall_ms\": {},\n",
        duration_ms(report.pre_build_wall)
    ));
    json.push_str(&format!(
        "  \"build_call_wall_ms\": {},\n",
        build_call_wall_ms
    ));
    json.push_str(&format!(
        "  \"post_build_wall_ms\": {},\n",
        duration_ms(report.post_build_wall)
    ));
    match profile_stage_total_ms {
        Some(stage_ms) => {
            json.push_str(&format!("  \"profile_stage_total_ms\": {},\n", stage_ms));
        }
        None => json.push_str("  \"profile_stage_total_ms\": null,\n"),
    }
    push_optional_i128_field(
        &mut json,
        "total_minus_profile_stage_ms",
        total_minus_profile_stage_ms,
    );
    json.push_str(",\n");
    push_optional_i128_field(
        &mut json,
        "build_call_minus_profile_stage_ms",
        build_call_minus_profile_stage_ms,
    );
    json.push_str(",\n");
    if let Some(profile_path) = &report.profile_json_path {
        json.push_str(&format!(
            "  \"profile_json_path\": \"{}\",\n",
            json_escape(&profile_path.display().to_string())
        ));
    } else {
        json.push_str("  \"profile_json_path\": null,\n");
    }
    json.push_str("  \"phases\": [\n");
    for (idx, phase) in report.phases.iter().enumerate() {
        let comma = if idx + 1 == report.phases.len() {
            ""
        } else {
            ","
        };
        json.push_str(&format!(
            "    {{ \"name\": \"{}\", \"wall_ms\": {}, \"cpu_ms\": {}, \"avg_cpu_cores\": {:.6}, \"peak_cpu_cores\": {:.6}, \"peak_threads\": {}, \"peak_rss_kib\": {}, \"samples\": {} }}{}\n",
            json_escape(&phase.name),
            duration_ms(phase.wall_time),
            duration_ms(phase.cpu_time),
            phase_cpu_cores_used(phase),
            phase.max_cpu_cores,
            phase.peak_threads,
            optional_usize_json(phase.peak_rss_kib),
            phase.sample_count,
            comma,
        ));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    std::fs::write(path, json)?;
    Ok(())
}

fn read_profile_stage_total_ms(path: &Path) -> Option<u128> {
    let text = std::fs::read_to_string(path).ok()?;
    extract_json_u128_after_section(&text, "serial_timing", "stage_total_ms")
}

fn extract_json_u128_after_section(text: &str, section: &str, field: &str) -> Option<u128> {
    let section_pos = text.find(&format!("\"{section}\""))?;
    let rest = &text[section_pos..];
    let field_pos = rest.find(&format!("\"{field}\""))?;
    let after_field = &rest[field_pos + field.len() + 2..];
    let colon_pos = after_field.find(':')?;
    let mut digits = after_field[colon_pos + 1..].trim_start().chars();
    let mut value = String::new();
    while let Some(ch) = digits.next() {
        if ch.is_ascii_digit() {
            value.push(ch);
        } else {
            break;
        }
    }
    value.parse::<u128>().ok()
}

fn duration_ms(duration: Duration) -> u128 {
    duration.as_millis()
}

fn signed_ms_delta(left_ms: u128, right_ms: Option<u128>) -> Option<i128> {
    right_ms.map(|right| left_ms as i128 - right as i128)
}

fn push_optional_i128_field(json: &mut String, name: &str, value: Option<i128>) {
    match value {
        Some(value) => json.push_str(&format!("  \"{name}\": {value}")),
        None => json.push_str(&format!("  \"{name}\": null")),
    }
}

fn optional_usize_json(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn gib_to_bytes(value_gib: f64) -> AnnResult<usize> {
    if !value_gib.is_finite() || value_gib < 0.0 {
        return Err(AnnError::log_index_config_error(
            "oom_budget".to_string(),
            "OOM memory values must be finite and >= 0".to_string(),
        ));
    }

    let bytes = value_gib * 1024.0 * 1024.0 * 1024.0;
    if bytes > usize::MAX as f64 {
        return Err(AnnError::log_index_config_error(
            "oom_budget".to_string(),
            "OOM memory value is too large".to_string(),
        ));
    }

    Ok(bytes.round() as usize)
}

fn apply_online_suffix_args(params: &mut PiPNNParams, args: &Args) -> AnnResult<()> {
    params.online_suffix_reduce_mode = args.online_suffix_reduce;
    params.online_suffix_direct_queue_bytes = gib_to_bytes(args.online_suffix_direct_queue_gb)?;
    params.online_suffix_state_cache_bytes = gib_to_bytes(args.online_suffix_state_cache_gb)?;
    params.online_suffix_state_page_cache_bytes =
        gib_to_bytes(args.online_suffix_state_page_cache_gb)?;
    params.online_suffix_state_writeback_bytes =
        gib_to_bytes(args.online_suffix_state_writeback_gb)?;
    params.online_suffix_spill_buffer_bytes = gib_to_bytes(args.online_suffix_spill_buffer_gb)?;
    params.online_suffix_reorder_window_bytes = gib_to_bytes(args.online_suffix_reorder_window_gb)?;
    params.online_suffix_read_window_bytes = gib_to_bytes(args.online_suffix_read_window_gb)?;
    params.online_suffix_pinned_prefix_points = args.online_suffix_pinned_prefix_points;
    params.online_suffix_reduce_workers = args.online_suffix_reduce_workers;
    params.online_suffix_delta_budget_bytes = gib_to_bytes(args.online_suffix_delta_budget_gb)?;
    params.online_suffix_delta_buffer_bytes = gib_to_bytes(args.online_suffix_delta_buffer_gb)?;
    params.online_suffix_range_points = args.online_suffix_range_points;
    params.online_suffix_range_trigger_bytes = gib_to_bytes(args.online_suffix_range_trigger_gb)?;
    params.online_suffix_max_runs_per_range = args.online_suffix_max_runs_per_range;
    params.online_suffix_compaction_workers = args.online_suffix_compaction_workers;
    params.online_suffix_cache_hot_updates = args.online_suffix_cache_hot_updates;
    params.online_suffix_state_overprovision_ratio = args.online_suffix_state_overprovision_ratio;
    Ok(())
}

fn apply_gpu_subtree_args(params: &mut PiPNNParams, args: &Args) {
    params.prune_backend = args.prune_backend;
    params.hashprune_evidence_mode = args.hashprune_evidence_mode;
    params.hashprune_evidence_overflow_budget = args.hashprune_evidence_overflow_budget;
    params.hashprune_backfill_mode = args.hashprune_backfill_mode;
    params.hashprune_backfill_ablation = args.hashprune_backfill_ablation;
    params.hashprune_backfill_budget = args.hashprune_backfill_budget;
    params.hashprune_backfill_alpha = args.hashprune_backfill_alpha;
    params.hashprune_rhalf_hash_quota = args.hashprune_rhalf_hash_quota;
    params.hashprune_soft_tier_priority_alpha = args.hashprune_soft_tier_priority_alpha;
    params.hashprune_rhalf_max_backfill_per_row = args.hashprune_rhalf_max_backfill_per_row;
    params.hashprune_backfill_diagnostics_prefix =
        args.hashprune_backfill_diagnostics_prefix.clone();
    params.lrpf_pre_spill_compress_enable = args.lrpf_pre_spill_compress;
    params.lrpf_pre_spill_width_multiplier = args.lrpf_pre_spill_width_multiplier;
    params.lrpf_protected_core = args.lrpf_protected_core;
    params.lrpf_target_degree = args.lrpf_target_degree;
    params.lrpf_long_edge_quota = args.lrpf_long_edge_quota;
    params.partitioner = args.partitioner;
    params.gpu_subtree_root_assignment_backend = args.gpu_subtree_root_assignment_backend;
    params.gpu_subtree_assignment_backend = args.gpu_subtree_assignment_backend;
    params.gpu_subtree_assignment_dco_mode = args.gpu_subtree_assignment_dco_mode;
    params.gpu_subtree_assignment_dco_rerank_mode = args.gpu_subtree_assignment_dco_rerank_mode;
    params.gpu_subtree_assignment_dco_proj_dim = args.gpu_subtree_assignment_dco_proj_dim;
    params.gpu_subtree_assignment_dco_survivors = args.gpu_subtree_assignment_dco_survivors;
    params.gpu_subtree_assignment_dco_audit = args.gpu_subtree_assignment_dco_audit;
    params.gpu_subtree_d0_dco_mode = args.gpu_subtree_d0_dco_mode;
    params.gpu_subtree_d0_dco_rerank_mode = args.gpu_subtree_d0_dco_rerank_mode;
    params.gpu_subtree_d0_dco_survivors = args.gpu_subtree_d0_dco_survivors;
    params.gpu_subtree_d0_dco_audit = args.gpu_subtree_d0_dco_audit;
    params.gpu_subtree_d0_sq_cpu_kernel = args.gpu_subtree_d0_sq_cpu_kernel;
    params.gpu_subtree_execution_mode = args.gpu_subtree_execution_mode;
    params.gpu_subtree_leaf_topk_backend = args.gpu_subtree_leaf_topk_backend;
    params.gpu_subtree_leaf_dco_mode = args.gpu_subtree_leaf_dco_mode;
    params.gpu_subtree_leaf_dco_rerank_mode = args.gpu_subtree_leaf_dco_rerank_mode;
    params.gpu_subtree_leaf_dco_proj_dim = args.gpu_subtree_leaf_dco_proj_dim;
    params.gpu_subtree_leaf_dco_survivors = args.gpu_subtree_leaf_dco_survivors;
    params.gpu_subtree_leaf_dco_audit = args.gpu_subtree_leaf_dco_audit;
    params.gpu_subtree_leaf_dco_ddc_res_pca_matrix =
        args.gpu_subtree_leaf_dco_ddc_res_pca_matrix.clone();
    params.gpu_subtree_leaf_dco_ddc_res_projected_input =
        args.gpu_subtree_leaf_dco_ddc_res_projected_input;
    params.gpu_subtree_leaf_dco_ddc_res_base_dim = args.gpu_subtree_leaf_dco_ddc_res_base_dim;
    params.gpu_subtree_leaf_dco_ddc_res_sigma_count = args.gpu_subtree_leaf_dco_ddc_res_sigma_count;
    params.gpu_subtree_rabitq_plus_mode = args.gpu_subtree_rabitq_plus_mode;
    params.gpu_subtree_rabitq_plus_bits = args.gpu_subtree_rabitq_plus_bits;
    params.gpu_subtree_rabitq_plus_rotator = args.gpu_subtree_rabitq_plus_rotator;
    params.gpu_subtree_rabitq_plus_seed = args.gpu_subtree_rabitq_plus_seed;
    params.gpu_subtree_rabitq_plus_sidecar_prefix =
        args.gpu_subtree_rabitq_plus_sidecar_prefix.clone();
    params.gpu_subtree_rabitq_plus_sidecar_mode = args.gpu_subtree_rabitq_plus_sidecar_mode;
    params.gpu_subtree_qcode_staging_mode = args.gpu_subtree_qcode_staging_mode;
    params.gpu_subtree_qcode_d1_payload_mode = args.gpu_subtree_qcode_d1_payload_mode;
    params.gpu_subtree_rabitq_d1_distance_mode = args.gpu_subtree_rabitq_d1_distance_mode;
    params.gpu_subtree_rabitq_short_survivors = args.gpu_subtree_rabitq_short_survivors;
    params.gpu_subtree_rabitq_residual_centroids =
        args.gpu_subtree_rabitq_residual_centroids.max(1);
    params.gpu_subtree_qcode_staging_upload_shard_rows =
        args.gpu_subtree_qcode_staging_upload_shard_rows.max(1);
    params.gpu_subtree_qcode_global_d0_leaders = args.gpu_subtree_qcode_global_d0_leaders;
    params.gpu_subtree_rabitq_d0_assign_mode = args.gpu_subtree_rabitq_d0_assign_mode;
    params.gpu_subtree_rabitq_d1_topk_mode = args.gpu_subtree_rabitq_d1_topk_mode;
    params.gpu_subtree_sq_staging_mode = args.gpu_subtree_sq_staging_mode;
    params.gpu_subtree_sq8_rotated_residual_d0 = args.gpu_subtree_sq8_rotated_residual_d0;
    params.gpu_subtree_sq8_rotated_residual_d1 = args.gpu_subtree_sq8_rotated_residual_d1;
    params.gpu_subtree_sq8_rotated_residual_d1_int8 = args.gpu_subtree_sq8_rotated_residual_d1_int8;
    params.gpu_subtree_sq8_rotated_residual_seed = args.gpu_subtree_sq8_rotated_residual_seed;
    params.gpu_subtree_sq_staging_upload_shard_rows =
        args.gpu_subtree_sq_staging_upload_shard_rows.max(1);
    params.gpu_subtree_sq_staging_raw_d1_diagnostic = args.gpu_subtree_sq_staging_raw_d1_diagnostic;
    params.gpu_subtree_rabitq_d0_row_tile = args.gpu_subtree_rabitq_d0_row_tile.max(1);
    params.gpu_subtree_rabitq_d0_row_chunk = args.gpu_subtree_rabitq_d0_row_chunk.max(1);
    params.gpu_subtree_rabitq_d0_leader_tile = args.gpu_subtree_rabitq_d0_leader_tile.max(1);
    params.gpu_subtree_rabitq_d0_partial_r = args.gpu_subtree_rabitq_d0_partial_r.max(1);
    params.gpu_subtree_sq_bits = args.gpu_subtree_sq_bits;
    params.gpu_subtree_sq4_plus = args.gpu_subtree_sq4_plus;
    params.gpu_subtree_sq4_primary_center = args
        .gpu_subtree_sq4_primary_center
        .unwrap_or(args.gpu_subtree_sq4_plus);
    params.gpu_subtree_sq_native_int4 = args.gpu_subtree_sq_native_int4;
    params.gpu_subtree_sq_wmma_mode = args.gpu_subtree_sq_wmma_mode;
    params.gpu_subtree_sq_wmma_sidecar_prefix = if args.gpu_subtree_sq4_plus {
        None
    } else if args.gpu_subtree_sq8_rotated_residual_d1 {
        Some(PathBuf::from(&args.index_path_prefix).join("rotated_residual_sq8"))
    } else {
        args.gpu_subtree_sq_wmma_sidecar_prefix.clone()
    };
    params.gpu_subtree_sq_wmma_sidecar_mode = if args.gpu_subtree_sq4_plus {
        PiPNNGpuSubtreeSqWmmaSidecarMode::Off
    } else {
        args.gpu_subtree_sq_wmma_sidecar_mode
    };
    params.gpu_subtree_turboquant_mode = args.gpu_subtree_turboquant_mode;
    params.gpu_subtree_turboquant_bits = args.gpu_subtree_turboquant_bits;
    params.gpu_subtree_turboquant_seed = args.gpu_subtree_turboquant_seed;
    params.gpu_subtree_turboquant_sidecar_prefix =
        args.gpu_subtree_turboquant_sidecar_prefix.clone();
    params.gpu_subtree_turboquant_sidecar_mode = args.gpu_subtree_turboquant_sidecar_mode;
    params.gpu_subtree_turboquant_cascade_low_dim =
        args.gpu_subtree_turboquant_cascade_low_dim.max(1);
    params.gpu_subtree_turboquant_cascade_survivor_cap =
        args.gpu_subtree_turboquant_cascade_survivor_cap;
    params.gpu_subtree_turboquant_adaptive_survivor_threshold =
        args.gpu_subtree_turboquant_adaptive_survivor_threshold;
    params.gpu_subtree_turboquant_adaptive_survivor_low_cap =
        args.gpu_subtree_turboquant_adaptive_survivor_low_cap;
    params.gpu_subtree_persistent_leaf_tile_rows =
        args.gpu_subtree_persistent_leaf_tile_rows.max(1);
    params.gpu_subtree_persistent_large_ticket_batch =
        args.gpu_subtree_persistent_large_ticket_batch;
    params.gpu_subtree_persistent_scheduler_mode = args.gpu_subtree_persistent_scheduler_mode;
    params.gpu_subtree_persistent_idle_backoff_ns = args.gpu_subtree_persistent_idle_backoff_ns;
    params.gpu_subtree_sq8_d0_assignment_kernel = args.gpu_subtree_sq8_d0_assignment_kernel;
    params.gpu_subtree_sq8_assignment_kernel = args.gpu_subtree_sq8_assignment_kernel;
    params.gpu_subtree_sq8_leaf_executor = args.gpu_subtree_sq8_leaf_executor;
    params.gpu_subtree_tessa_capture_dir = args.gpu_subtree_tessa_capture_dir.clone();
    params.gpu_subtree_sq8_leaf_mode = args.gpu_subtree_sq8_leaf_mode;
    params.gpu_subtree_d0_partition_mode = if args.gpu_subtree_sq4_plus
        && args.gpu_subtree_d0_partition_mode == PiPNNGpuSubtreeD0PartitionMode::RbcFanout {
        PiPNNGpuSubtreeD0PartitionMode::BudgetKMeansTopK
    } else { args.gpu_subtree_d0_partition_mode };
    params.gpu_subtree_d1_root_partition_mode = args.gpu_subtree_d1_root_partition_mode;
    params.gpu_subtree_d0_tagore_partitions = args.gpu_subtree_d0_tagore_partitions;
    params.gpu_subtree_d0_tagore_k = args.gpu_subtree_d0_tagore_k;
    params.gpu_subtree_d0_tagore_sample_fraction = args.gpu_subtree_d0_tagore_sample_fraction;
    params.gpu_subtree_d0_capacity_enable = effective_gpu_subtree_d0_capacity_enabled(args);
    params.gpu_subtree_d1plus_lazy_split_mode = args.gpu_subtree_d1plus_lazy_split_mode;
    let (capacity_high, capacity_target, capacity_low) = effective_gpu_subtree_d0_capacity_ratios(args);
    params.gpu_subtree_d0_capacity_high_ratio = capacity_high;
    params.gpu_subtree_d0_capacity_target_ratio = capacity_target;
    params.gpu_subtree_d0_capacity_low_ratio = capacity_low;
    params.gpu_subtree_d0_capacity_min_efficient_rows =
        args.gpu_subtree_d0_capacity_min_efficient_rows;
    params.gpu_subtree_d0_capacity_quality_min_root_leaders =
        args.gpu_subtree_d0_capacity_quality_min_root_leaders;
    params.gpu_subtree_d0_capacity_root_leader_cap = args.gpu_subtree_d0_capacity_root_leader_cap;
    params.gpu_subtree_d0_capacity_split_recursion_limit =
        args.gpu_subtree_d0_capacity_split_recursion_limit;
    params.gpu_subtree_d0_throughput_model = args.gpu_subtree_d0_throughput_model;
    params.gpu_subtree_d0_throughput_persistent_growth_limit =
        args.gpu_subtree_d0_throughput_persistent_growth_limit;
    params.gpu_subtree_d0_throughput_preload_visible_limit_ms =
        args.gpu_subtree_d0_throughput_preload_visible_limit_ms;
    params.gpu_subtree_d0_throughput_min_ratio = args.gpu_subtree_d0_throughput_min_ratio;
    params.gpu_subtree_d0_throughput_calibration_path =
        args.gpu_subtree_d0_throughput_calibration_json.clone();
    params.gpu_subtree_d0_throughput_child_p95_guardrail_rows =
        args.gpu_subtree_d0_throughput_child_p95_guardrail_rows;
    params.gpu_subtree_persistent_launch_policy = args.gpu_subtree_persistent_launch_policy;
    params.gpu_subtree_hbm_policy = args.gpu_subtree_hbm_policy;
    params.gpu_subtree_launch_calibration_cache = args.gpu_subtree_launch_calibration_cache.clone();
    params.gpu_subtree_launch_calibration_refresh = args.gpu_subtree_launch_calibration_refresh;
    params.gpu_subtree_vector_staging_mode = args.gpu_subtree_vector_staging_mode;
    params.gpu_subtree_vector_staging_budget_bytes = args.gpu_subtree_vector_staging_budget_bytes;
    params.gpu_subtree_vector_staging_window_children =
        args.gpu_subtree_vector_staging_window_children.max(1);
    params.gpu_subtree_child_schedule_mode = args.gpu_subtree_child_schedule;
}

fn apply_prune_backend_env_overrides(params: &mut PiPNNParams) -> AnnResult<()> {
    apply_prune_backend_env_overrides_with(
        params,
        |name| std::env::var(name).ok(),
        cli_flag_present,
    )
}

const PRODUCTION_HASHPRUNE_BACKFILL_MODE: PiPNNHashPruneBackfillMode =
    PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
const PRODUCTION_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA: f32 = 1.02;
const PRODUCTION_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW: usize = 3;

fn apply_production_hashprune_defaults(params: &mut PiPNNParams) {
    if params.hashprune_backfill_mode == PiPNNHashPruneBackfillMode::RHalfSoftTierQ3 {
        params
            .hashprune_soft_tier_priority_alpha
            .get_or_insert(PRODUCTION_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA);
        params
            .hashprune_rhalf_max_backfill_per_row
            .get_or_insert(PRODUCTION_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW);
    }
}

fn apply_prune_backend_env_overrides_with(
    params: &mut PiPNNParams,
    get_env: impl Fn(&str) -> Option<String>,
    cli_flag_is_present: impl Fn(&str) -> bool,
) -> AnnResult<()> {
    if !cli_flag_is_present("--prune-backend") {
        if let Some(raw) = get_env("SHEAR_PIPNN_PRUNE_BACKEND") {
            params.prune_backend = raw.parse::<PiPNNPruneBackend>().map_err(|err| {
                AnnError::log_index_config_error("prune_backend".to_string(), err)
            })?;
        }
    }
    if !cli_flag_is_present("--hashprune-evidence-mode") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_EVIDENCE_MODE") {
            params.hashprune_evidence_mode =
                raw.parse::<PiPNNHashPruneEvidenceMode>().map_err(|err| {
                    AnnError::log_index_config_error("hashprune_evidence_mode".to_string(), err)
                })?;
        }
    }
    if !cli_flag_is_present("--hashprune-evidence-overflow-budget") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_EVIDENCE_OVERFLOW_BUDGET") {
            params.hashprune_evidence_overflow_budget = raw.parse::<usize>().map_err(|_| {
                AnnError::log_index_config_error(
                    "hashprune_evidence_overflow_budget".to_string(),
                    format!("invalid HashPrune evidence overflow budget '{raw}'"),
                )
            })?;
        }
    }
    if !cli_flag_is_present("--hashprune-backfill-mode") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE") {
            params.hashprune_backfill_mode =
                raw.parse::<PiPNNHashPruneBackfillMode>().map_err(|err| {
                    AnnError::log_index_config_error("hashprune_backfill_mode".to_string(), err)
                })?;
        }
    }
    if !cli_flag_is_present("--hashprune-backfill-ablation") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_BACKFILL_ABLATION") {
            params.hashprune_backfill_ablation = raw
                .parse::<PiPNNHashPruneBackfillAblation>()
                .map_err(|err| {
                    AnnError::log_index_config_error("hashprune_backfill_ablation".to_string(), err)
                })?;
        }
    }
    if !cli_flag_is_present("--hashprune-backfill-budget") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_BACKFILL_BUDGET") {
            params.hashprune_backfill_budget = raw.parse::<usize>().map_err(|_| {
                AnnError::log_index_config_error(
                    "hashprune_backfill_budget".to_string(),
                    format!("invalid HashPrune backfill budget '{raw}'"),
                )
            })?;
        }
    }
    if !cli_flag_is_present("--hashprune-backfill-alpha") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_BACKFILL_ALPHA") {
            params.hashprune_backfill_alpha = raw.parse::<f32>().map_err(|_| {
                AnnError::log_index_config_error(
                    "hashprune_backfill_alpha".to_string(),
                    format!("invalid HashPrune backfill alpha '{raw}'"),
                )
            })?;
        }
    }
    if !cli_flag_is_present("--hashprune-rhalf-hash-quota") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_RHALF_HASH_QUOTA") {
            params.hashprune_rhalf_hash_quota = Some(raw.parse::<usize>().map_err(|_| {
                AnnError::log_index_config_error(
                    "hashprune_rhalf_hash_quota".to_string(),
                    format!("invalid named R-half HashPrune quota '{raw}'"),
                )
            })?);
        }
    }
    if !cli_flag_is_present("--hashprune-soft-tier-priority-alpha") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA") {
            params.hashprune_soft_tier_priority_alpha = Some(raw.parse::<f32>().map_err(|_| {
                AnnError::log_index_config_error(
                    "hashprune_soft_tier_priority_alpha".to_string(),
                    format!("invalid named SoftTier priority alpha '{raw}'"),
                )
            })?);
        }
    }
    if !cli_flag_is_present("--hashprune-rhalf-max-backfill-per-row") {
        if let Some(raw) = get_env("SHEAR_PIPNN_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW") {
            params.hashprune_rhalf_max_backfill_per_row =
                Some(raw.parse::<usize>().map_err(|_| {
                    AnnError::log_index_config_error(
                        "hashprune_rhalf_max_backfill_per_row".to_string(),
                        format!("invalid named R-half maximum backfill per row '{raw}'"),
                    )
                })?);
        }
    }
    if !cli_flag_is_present("--lrpf-pre-spill-compress") {
        if let Some(raw) = get_env("SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS") {
            params.lrpf_pre_spill_compress_enable =
                parse_env_bool_value("SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS", &raw)?;
        }
    }
    if !cli_flag_is_present("--lrpf-pre-spill-width-multiplier") {
        if let Some(raw) = get_env("SHEAR_PIPNN_LRPF_PRE_SPILL_WIDTH_MULTIPLIER") {
            params.lrpf_pre_spill_width_multiplier = raw.parse::<f32>().map_err(|_| {
                AnnError::log_index_config_error(
                    "lrpf_pre_spill_width_multiplier".to_string(),
                    format!("invalid LRPF pre-spill width multiplier '{raw}'"),
                )
            })?;
        }
    }
    if !cli_flag_is_present("--lrpf-protected-core") {
        if let Some(raw) = get_env("SHEAR_PIPNN_LRPF_PROTECTED_CORE") {
            params.lrpf_protected_core = raw.parse::<usize>().map_err(|_| {
                AnnError::log_index_config_error(
                    "lrpf_protected_core".to_string(),
                    format!("invalid LRPF protected core '{raw}'"),
                )
            })?;
        }
    }
    if !cli_flag_is_present("--lrpf-target-degree") {
        if let Some(raw) = get_env("SHEAR_PIPNN_LRPF_TARGET_DEGREE") {
            params.lrpf_target_degree = raw.parse::<usize>().map_err(|_| {
                AnnError::log_index_config_error(
                    "lrpf_target_degree".to_string(),
                    format!("invalid LRPF target degree '{raw}'"),
                )
            })?;
        }
    }
    if !cli_flag_is_present("--lrpf-long-edge-quota") {
        if let Some(raw) = get_env("SHEAR_PIPNN_LRPF_LONG_EDGE_QUOTA") {
            params.lrpf_long_edge_quota = raw.parse::<usize>().map_err(|_| {
                AnnError::log_index_config_error(
                    "lrpf_long_edge_quota".to_string(),
                    format!("invalid LRPF long edge quota '{raw}'"),
                )
            })?;
        }
    }

    if !params.lrpf_pre_spill_width_multiplier.is_finite()
        || params.lrpf_pre_spill_width_multiplier < 1.0
    {
        return Err(AnnError::log_index_config_error(
            "lrpf_pre_spill_width_multiplier".to_string(),
            "lrpf_pre_spill_width_multiplier must be finite and >= 1.0".to_string(),
        ));
    }
    if params.lrpf_protected_core > params.leaf_knn {
        return Err(AnnError::log_index_config_error(
            "lrpf_protected_core".to_string(),
            "lrpf_protected_core must be <= leaf_knn".to_string(),
        ));
    }
    if params.lrpf_target_degree == 0 {
        return Err(AnnError::log_index_config_error(
            "lrpf_target_degree".to_string(),
            "lrpf_target_degree must be >= 1".to_string(),
        ));
    }
    apply_production_hashprune_defaults(params);
    params.validate_hashprune_policies()
}

fn parse_env_bool_value(name: &str, raw: &str) -> AnnResult<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(AnnError::log_index_config_error(
            name.to_string(),
            format!("expected boolean env value for {name}, got '{raw}'"),
        )),
    }
}

fn cli_flag_present(flag: &str) -> bool {
    let flag_os = std::ffi::OsStr::new(flag);
    std::env::args_os().any(|arg| {
        if arg == flag_os {
            return true;
        }
        let Some(text) = arg.to_str() else {
            return false;
        };
        text.strip_prefix(flag)
            .is_some_and(|suffix| suffix.starts_with('='))
    })
}

fn maybe_create_rabitq_plus_sidecar(args: &Args, params: &PiPNNParams) -> AnnResult<bool> {
    if !matches!(
        args.gpu_subtree_rabitq_plus_sidecar_mode,
        PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly
            | PiPNNGpuSubtreeRabitqPlusSidecarMode::Rebuild
    ) {
        return Ok(false);
    }
    let prefix = args
        .gpu_subtree_rabitq_plus_sidecar_prefix
        .as_ref()
        .ok_or_else(|| {
            AnnError::log_index_config_error(
                "gpu_subtree_rabitq_plus_sidecar_prefix".to_string(),
                "RabitQ sidecar creation requires --gpu-subtree-rabitq-plus-sidecar-prefix"
                    .to_string(),
            )
        })?;

    let split_sidecar_requested = matches!(
        args.gpu_subtree_d0_dco_mode,
        PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
    ) || args.gpu_subtree_leaf_dco_mode
        == PiPNNGpuSubtreeDcoMode::RabitqSplitFilter;
    if split_sidecar_requested {
        println!(
            "PiPNN: creating RabitQ-Library split sidecar prefix={} bits={} rotator={} seed={} io_mode={}",
            prefix.display(),
            args.gpu_subtree_rabitq_plus_bits,
            args.gpu_subtree_rabitq_plus_rotator,
            args.gpu_subtree_rabitq_plus_seed,
            params.effective_oom_io_mode()
        );
        let report = create_rabitq_split_sidecar_from_fbin(
            &PathBuf::from(&args.data_path),
            prefix,
            args.gpu_subtree_rabitq_plus_bits,
            args.gpu_subtree_rabitq_plus_rotator,
            args.gpu_subtree_rabitq_plus_seed,
            params.effective_oom_io_mode(),
        )?;
        println!(
            concat!(
                "PiPNN: RabitQ-Library split sidecar ready rows={} dim={} padded_dim={} bits={} ex_bits={} ",
                "batch_count={} batch_bytes={} ex_row_bytes={} batch_total_bytes={} ex_total_bytes={} ",
                "fingerprint={} total_s={:.3} chunk_alloc_s={:.3} raw_read_s={:.3} ",
                "source_hash_s={:.3} quantize_s={:.3} batch_output_s={:.3} ",
                "ex_output_s={:.3} header_write_s={:.3} write_s={:.3}"
            ),
            report.rows,
            report.dim,
            report.padded_dim,
            report.bits,
            report.ex_bits,
            report.batch_count,
            report.batch_bytes,
            report.ex_row_bytes,
            report.batch_total_bytes,
            report.ex_total_bytes,
            report.source_fingerprint,
            report.total_wall.as_secs_f64(),
            report.chunk_alloc_wall.as_secs_f64(),
            report.raw_read_wall.as_secs_f64(),
            report.source_hash_wall.as_secs_f64(),
            report.quantize_wall.as_secs_f64(),
            report.batch_output_wall.as_secs_f64(),
            report.ex_output_wall.as_secs_f64(),
            report.header_write_wall.as_secs_f64(),
            report.write_wall.as_secs_f64()
        );
        return Ok(args.gpu_subtree_rabitq_plus_sidecar_mode
            == PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly);
    }

    println!(
        "PiPNN: creating dense-i8 RabitQ+ sidecar prefix={} bits={} rotator={} seed={} io_mode={}",
        prefix.display(),
        args.gpu_subtree_rabitq_plus_bits,
        args.gpu_subtree_rabitq_plus_rotator,
        args.gpu_subtree_rabitq_plus_seed,
        params.effective_oom_io_mode()
    );
    let report = create_rabitq_plus_sidecar_from_fbin(
        &PathBuf::from(&args.data_path),
        prefix,
        args.gpu_subtree_rabitq_plus_bits,
        args.gpu_subtree_rabitq_plus_rotator,
        args.gpu_subtree_rabitq_plus_seed,
        params.effective_oom_io_mode(),
    )?;
    println!(
        concat!(
            "PiPNN: dense-i8 RabitQ+ sidecar ready rows={} dim={} padded_dim={} bits={} ",
            "codes_bytes={} meta_rows_bytes={} fingerprint={} total_s={:.3} ",
            "chunk_alloc_s={:.3} raw_read_s={:.3} source_hash_s={:.3} ",
            "quantize_s={:.3} code_output_s={:.3} norm_s={:.3} ",
            "meta_output_s={:.3} header_write_s={:.3} write_s={:.3}"
        ),
        report.rows,
        report.dim,
        report.padded_dim,
        report.bits,
        report.codes_bytes,
        report.meta_rows_bytes,
        report.source_fingerprint,
        report.total_wall.as_secs_f64(),
        report.chunk_alloc_wall.as_secs_f64(),
        report.raw_read_wall.as_secs_f64(),
        report.source_hash_wall.as_secs_f64(),
        report.quantize_wall.as_secs_f64(),
        report.code_output_wall.as_secs_f64(),
        report.norm_wall.as_secs_f64(),
        report.meta_output_wall.as_secs_f64(),
        report.header_write_wall.as_secs_f64(),
        report.write_wall.as_secs_f64()
    );

    Ok(args.gpu_subtree_rabitq_plus_sidecar_mode
        == PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly)
}

fn maybe_create_sq_wmma_sidecar(
    args: &Args,
    params: &PiPNNParams,
    data_num: usize,
    data_dim: usize,
) -> AnnResult<bool> {
    if args.gpu_subtree_sq4_plus {
        if params.gpu_subtree_sq4_primary_center {
            println!("PiPNN: SQ4+ encodes each point once against its nearest D0 center");
        } else {
            println!("PiPNN: SQ4+ uses per-partition rotated residual codes");
        }
        return Ok(false);
    }
    if args.gpu_subtree_sq_staging_mode.is_enabled() {
        println!(
            "PiPNN: deferring SQ+WMMA sidecar creation to SQ GPU-D0 staging prefix={} bits={} upload_shard_rows={}",
            args.gpu_subtree_sq_wmma_sidecar_prefix
                .as_ref()
                .map(|prefix| prefix.display().to_string())
                .unwrap_or_else(|| "<missing>".to_string()),
            args.gpu_subtree_sq_bits,
            args.gpu_subtree_sq_staging_upload_shard_rows
        );
        return Ok(false);
    }
    if !matches!(
        args.gpu_subtree_sq_wmma_sidecar_mode,
        PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly | PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
    ) {
        return Ok(false);
    }
    let prefix = args
        .gpu_subtree_sq_wmma_sidecar_prefix
        .as_ref()
        .ok_or_else(|| {
            AnnError::log_index_config_error(
                "gpu_subtree_sq_wmma_sidecar_prefix".to_string(),
                "SQ+WMMA sidecar creation requires --gpu-subtree-sq-wmma-sidecar-prefix"
                    .to_string(),
            )
        })?;

    println!(
        "PiPNN: creating SQ+WMMA sidecar prefix={} bits={} native_int4={} io_mode={}",
        prefix.display(),
        args.gpu_subtree_sq_bits,
        args.gpu_subtree_sq_native_int4,
        params.effective_oom_io_mode()
    );
    let resident_rows =
        if args.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild {
            estimate_sq_wmma_sidecar_resident_rows(params, data_num, data_dim)
        } else {
            0
        };
    let data_path = PathBuf::from(&args.data_path);
    let is_uint8 = args.data_type.eq_ignore_ascii_case("uint8");
    let report = if resident_rows > 0 {
        if is_uint8 {
            create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4_and_resident_prefix(
                &data_path,
                prefix,
                args.gpu_subtree_sq_bits,
                params.effective_oom_io_mode(),
                resident_rows,
                args.gpu_subtree_sq_native_int4,
            )?
        } else {
            create_sq_wmma_sidecar_from_fbin_with_bits_native_int4_and_resident_prefix(
                &data_path,
                prefix,
                args.gpu_subtree_sq_bits,
                params.effective_oom_io_mode(),
                resident_rows,
                args.gpu_subtree_sq_native_int4,
            )?
        }
    } else if is_uint8 {
        create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4(
            &data_path,
            prefix,
            args.gpu_subtree_sq_bits,
            params.effective_oom_io_mode(),
            args.gpu_subtree_sq_native_int4,
        )?
    } else {
        create_sq_wmma_sidecar_from_fbin_with_bits_native_int4(
            &data_path,
            prefix,
            args.gpu_subtree_sq_bits,
            params.effective_oom_io_mode(),
            args.gpu_subtree_sq_native_int4,
        )?
    };
    println!(
        concat!(
            "PiPNN: SQ+WMMA sidecar ready rows={} dim={} padded_dim={} ",
            "bits={} code_layout={} code_row_bytes={} codes_bytes={} meta_rows_bytes={} fingerprint={} total_s={:.3} ",
            "setup_s={:.3} chunk_alloc_s={:.3} raw_read_s={:.3} source_hash_s={:.3} ",
            "quantize_s={:.3} code_output_s={:.3} meta_output_s={:.3} ",
            "flush_s={:.3} header_write_s={:.3} write_s={:.3} runtime_resident_rows={}"
        ),
        report.rows,
        report.dim,
        report.padded_dim,
        report.bits,
        report.code_layout,
        report.code_row_bytes,
        report.codes_bytes,
        report.meta_rows_bytes,
        report.source_fingerprint,
        report.total_wall.as_secs_f64(),
        report.setup_wall.as_secs_f64(),
        report.chunk_alloc_wall.as_secs_f64(),
        report.raw_read_wall.as_secs_f64(),
        report.source_hash_wall.as_secs_f64(),
        report.quantize_wall.as_secs_f64(),
        report.code_output_wall.as_secs_f64(),
        report.meta_output_wall.as_secs_f64(),
        report.flush_wall.as_secs_f64(),
        report.header_write_wall.as_secs_f64(),
        report.write_wall.as_secs_f64(),
        resident_rows
    );

    Ok(args.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly)
}

fn maybe_create_turboquant_mse4_sidecar(args: &Args, params: &PiPNNParams) -> AnnResult<bool> {
    if !matches!(
        args.gpu_subtree_turboquant_sidecar_mode,
        PiPNNGpuSubtreeTurboQuantSidecarMode::CreateOnly
            | PiPNNGpuSubtreeTurboQuantSidecarMode::Rebuild
    ) {
        return Ok(false);
    }
    let prefix = args
        .gpu_subtree_turboquant_sidecar_prefix
        .as_ref()
        .ok_or_else(|| {
            AnnError::log_index_config_error(
                "gpu_subtree_turboquant_sidecar_prefix".to_string(),
                "TurboQuant-MSE4 sidecar creation requires --gpu-subtree-turboquant-sidecar-prefix"
                    .to_string(),
            )
        })?;
    let build_backend = match args.gpu_subtree_turboquant_sidecar_build_backend {
        TurboQuantSidecarBuildBackend::Auto if args.pipnn_backend.eq_ignore_ascii_case("cuda") => {
            TurboQuantSidecarBuildBackend::Cuda
        }
        TurboQuantSidecarBuildBackend::Auto => TurboQuantSidecarBuildBackend::Cpu,
        other => other,
    };
    println!(
        "PiPNN: creating TurboQuant-MSE4 sidecar prefix={} bits={} seed={} rotator=dense-qr build_backend={} io_mode={}",
        prefix.display(),
        args.gpu_subtree_turboquant_bits,
        args.gpu_subtree_turboquant_seed,
        build_backend,
        params.effective_oom_io_mode(),
    );
    let report = match build_backend {
        TurboQuantSidecarBuildBackend::Cpu => create_turboquant_mse4_sidecar_from_fbin(
            &PathBuf::from(&args.data_path),
            prefix,
            args.gpu_subtree_turboquant_bits,
            args.gpu_subtree_turboquant_seed,
            params.effective_oom_io_mode(),
        )?,
        TurboQuantSidecarBuildBackend::Cuda => create_turboquant_mse4_sidecar_from_fbin_cuda(
            &PathBuf::from(&args.data_path),
            prefix,
            args.gpu_subtree_turboquant_bits,
            args.gpu_subtree_turboquant_seed,
            params.effective_oom_io_mode(),
            args.pipnn_gpu_device,
        )?,
        TurboQuantSidecarBuildBackend::Auto => unreachable!("auto backend resolved above"),
    };
    println!(
        concat!(
            "PiPNN: TurboQuant-MSE4 sidecar ready rows={} dim={} code_bytes_per_row={} bits={} seed={} ",
            "codes_bytes={} meta_rows_bytes={} fingerprint={} total_s={:.3} ",
            "setup_s={:.3} chunk_alloc_s={:.3} raw_read_s={:.3} source_hash_s={:.3} ",
            "quantize_s={:.3} code_output_s={:.3} meta_output_s={:.3} ",
            "flush_s={:.3} header_write_s={:.3} write_s={:.3}"
        ),
        report.rows,
        report.dim,
        report.code_bytes_per_row,
        report.bits,
        report.seed,
        report.codes_bytes,
        report.meta_rows_bytes,
        report.source_fingerprint,
        report.total_wall.as_secs_f64(),
        report.setup_wall.as_secs_f64(),
        report.chunk_alloc_wall.as_secs_f64(),
        report.raw_read_wall.as_secs_f64(),
        report.source_hash_wall.as_secs_f64(),
        report.quantize_wall.as_secs_f64(),
        report.code_output_wall.as_secs_f64(),
        report.meta_output_wall.as_secs_f64(),
        report.flush_wall.as_secs_f64(),
        report.header_write_wall.as_secs_f64(),
        report.write_wall.as_secs_f64(),
    );

    Ok(
        args.gpu_subtree_turboquant_sidecar_mode
            == PiPNNGpuSubtreeTurboQuantSidecarMode::CreateOnly,
    )
}

struct SqWmmaSidecarCleanupGuard {
    prefix: PathBuf,
}

impl SqWmmaSidecarCleanupGuard {
    fn for_rebuild(args: &Args) -> Option<Self> {
        if !args.gpu_subtree_sq4_plus
            && args.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild {
            args.gpu_subtree_sq_wmma_sidecar_prefix
                .clone()
                .map(|prefix| Self { prefix })
        } else {
            None
        }
    }
}

impl Drop for SqWmmaSidecarCleanupGuard {
    fn drop(&mut self) {
        match remove_sq_wmma_sidecar_files(&self.prefix) {
            Ok(removed) => {
                println!(
                    "PiPNN: removed temporary SQ+WMMA sidecar prefix={} files={}",
                    self.prefix.display(),
                    removed
                );
            }
            Err(err) => {
                eprintln!(
                    "PiPNN: warning: failed to remove temporary SQ+WMMA sidecar prefix={}: {}",
                    self.prefix.display(),
                    err
                );
            }
        }
    }
}

fn remove_sq_wmma_sidecar_files(prefix: &Path) -> io::Result<usize> {
    let mut removed = 0usize;
    for suffix in ["codes", "meta"] {
        let path = PathBuf::from(format!("{}.{}", prefix.display(), suffix));
        match fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }
    Ok(removed)
}

fn apply_user_leader_precedence(params: &mut PiPNNParams, args: &Args, max_leaders_explicit: bool) {
    if !max_leaders_explicit {
        return;
    }
    params.max_leaders = args.max_leaders;
    if effective_gpu_subtree_d0_capacity_enabled(args) {
        params.gpu_subtree_d0_capacity_quality_min_root_leaders = args.max_leaders.max(1);
        params.gpu_subtree_d0_capacity_root_leader_cap = params
            .gpu_subtree_d0_capacity_root_leader_cap
            .max(args.max_leaders.max(1));
    }
}

/// 使用 PiPNN 构图并生成 `_mem.index` 的简易驱动程序。
#[allow(clippy::too_many_arguments)]
fn build_shear_index(
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    index_path_prefix: &str,
    num_threads: u32,
    overwrite: bool,
    subgraph_threshold: f32,
    pipnn_params: &PiPNNParams,
) -> AnnResult<BuildPipnnTiming> {
    build_shear_index_with_data_type(
        "float",
        metric,
        data_path,
        r,
        l,
        index_path_prefix,
        num_threads,
        overwrite,
        subgraph_threshold,
        pipnn_params,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_shear_index_with_data_type(
    data_type: &str,
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    index_path_prefix: &str,
    num_threads: u32,
    overwrite: bool,
    subgraph_threshold: f32,
    pipnn_params: &PiPNNParams,
) -> AnnResult<BuildPipnnTiming> {
    let data_type = data_type.to_ascii_lowercase();
    let mut phase_tracker = PhaseTracker::from_env();

    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_saturate_graph(true)
        .with_num_threads(num_threads)
        .build();

    let (data_num, data_dim) = load_metadata_from_file(data_path).unwrap();

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 32_u64) as usize,
        data_num,
        false,
        0,
        false,
        0,
        1.3f32,
        index_write_parameters,
        subgraph_threshold,
    );

    let data_path_buf = PathBuf::from(data_path);
    phase_tracker.start_phase("load_data");
    if !pipnn_params.oom_enable {
        if data_type != "float" {
            return Err(AnnError::log_index_config_error(
                "oom_enable".to_string(),
                "uint8 PiPNN build requires --oom-enable".to_string(),
            ));
        }
        let mut index = InmemIndex::<f32>::new(config.clone())?;
        index.load_data_only(&data_path_buf, data_num)?;
        let index_path_prefix_buf = PathBuf::from(index_path_prefix);
        std::fs::create_dir_all(&index_path_prefix_buf)?;
        let inmem_index_path = index_path_prefix_buf.join("_mem.index");

        if !overwrite && file_exists(&inmem_index_path) {
            println!(
                "PiPNN: in-memory graph {:?} already exists, use --overwrite to rebuild.",
                inmem_index_path
            );
            let phases = phase_tracker.finish();
            return Ok(BuildPipnnTiming { phases });
        }

        let timer = Timer::new();
        println!(
            "PiPNN: building graph for {} points (dim={}) with R={} Lbuild={} m_hash_bits={}...",
            data_num, data_dim, r, l, pipnn_params.m_hash_bits
        );

        phase_tracker.start_phase("build_graph");
        let result = (|| -> AnnResult<()> {
            index.build_graph_pipnn(pipnn_params)?;
            if pipnn_params.root_radius_dry_run || pipnn_params.root_risk_dry_run {
                return Ok(());
            }

            let diff = timer.elapsed();
            println!("PiPNN: graph build done in {} seconds", diff.as_secs_f64());
            println!("PiPNN: saving in-memory graph to {:?}", inmem_index_path);
            phase_tracker.start_phase("save_graph");
            index.save(&inmem_index_path)?;
            Ok(())
        })();
        let phases = phase_tracker.finish();
        return result.map(|_| BuildPipnnTiming { phases });
    }

    let timer = Timer::new();
    let index_path_prefix_buf = PathBuf::from(index_path_prefix);
    std::fs::create_dir_all(&index_path_prefix_buf)?;
    let inmem_index_path = index_path_prefix_buf.join("_mem.index");
    if !overwrite && file_exists(&inmem_index_path) {
        println!(
            "PiPNN: in-memory graph {:?} already exists, use --overwrite to rebuild.",
            inmem_index_path
        );
        let phases = phase_tracker.finish();
        return Ok(BuildPipnnTiming { phases });
    }
    println!(
        "PiPNN: building graph for {} points (dim={}) with R={} Lbuild={} m_hash_bits={}...",
        data_num, data_dim, r, l, pipnn_params.m_hash_bits
    );

    phase_tracker.start_phase("build_graph");
    let result = (|| -> AnnResult<()> {
        match data_type.as_str() {
            "float" => build_pipnn_graph_oom_to_file(
                &data_path_buf,
                data_num,
                &config,
                pipnn_params,
                &inmem_index_path,
            )?,
            "uint8" => build_pipnn_graph_oom_to_file_u8(
                &data_path_buf,
                data_num,
                &config,
                pipnn_params,
                &inmem_index_path,
            )?,
            _ => {
                return Err(AnnError::log_index_config_error(
                    "data_type".to_string(),
                    "Invalid data type".to_string(),
                ));
            }
        }
        if pipnn_params.root_radius_dry_run
            || pipnn_params.root_risk_dry_run
            || pipnn_params.root_stop_after_d00
        {
            return Ok(());
        }

        let diff = timer.elapsed();
        println!("PiPNN: graph build done in {} seconds", diff.as_secs_f64());
        println!(
            "PiPNN: OOM graph written directly to {:?}",
            inmem_index_path
        );
        Ok(())
    })();
    let phases = phase_tracker.finish();

    result.map(|_| BuildPipnnTiming { phases })
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Data type <int8/uint8/float>
    #[clap(long)]
    data_type: String,

    /// Distance function <l2/cosine>
    #[clap(long)]
    dist_fn: String,

    /// Input data file in bin format
    #[clap(long)]
    data_path: String,

    /// Path prefix for saving index file components
    #[clap(long)]
    index_path_prefix: String,

    /// Maximum graph degree (R), 也是 PiPNN 的 l_max
    #[clap(long, short = 'R', default_value_t = 64)]
    max_degree: u32,

    /// Build complexity (Lbuild)，与默认构图保持一致
    #[clap(long, short = 'L', default_value_t = 100)]
    lbuild: u32,

    /// Number of threads used for building index (defaults to num of CPU logic cores)
    #[clap(long, short = 'T', default_value_t = 0)]
    num_threads: u32,

    /// Bound on the memory footprint of the index at search time in GB (保留接口，对 PiPNN
    /// 无直接影响)
    #[clap(long, short = 'B', default_value_t = 0.0)]
    search_dram_budget: f64,

    /// Limit on the memory allowed for building the index in GB (保留接口)
    #[clap(long, short = 'M', default_value_t = 0.0)]
    build_dram_budget: f64,

    /// Number of PQ bytes to build the index; 0 for full precision build (当前仅支持 full
    /// precision)
    #[clap(long, default_value_t = 0)]
    build_pq_bytes: u32,

    /// Set true for OPQ compression while using PQ distance comparisons (当前未支持)
    #[clap(long)]
    use_opq: bool,

    /// Overwrite existing index files
    #[clap(long)]
    overwrite: bool,

    #[clap(long, default_value_t = 0.0)]
    subgraph_threshold: f32,

    /// HashPrune 残差哈希位数 m，必须 <= 16
    #[clap(long, default_value_t = 12)]
    m_hash_bits: usize,

    /// Candidate pruning backend <hashprune/lrpf>
    #[clap(long, default_value_t = PiPNNPruneBackend::HashPrune)]
    prune_backend: PiPNNPruneBackend,

    /// Evidence-Aware HashPrune mode <off/report/enforce>
    #[clap(long, default_value_t = PiPNNHashPruneEvidenceMode::Off)]
    hashprune_evidence_mode: PiPNNHashPruneEvidenceMode,

    /// Evidence-qualified overflow slots; valid values are mode-dependent
    #[clap(long, default_value_t = 0)]
    hashprune_evidence_overflow_budget: usize,

    /// Bounded HashPrune spare-slot fill mode
    /// <off/rhalf-hash-slot-1/rhalf-distance-first-q3/rhalf-soft-tier-q3/rhalf-hard-tier-q3/
    /// distance-gated/distance-gated-stratified/distance-gated-multihit/
    /// distance-gated-tiered-d52/distance-gated-tiered-rhalf>
    #[clap(long, default_value_t = PRODUCTION_HASHPRUNE_BACKFILL_MODE)]
    hashprune_backfill_mode: PiPNNHashPruneBackfillMode,

    /// Named R-half research ablation
    #[clap(long, default_value_t = PiPNNHashPruneBackfillAblation::None)]
    hashprune_backfill_ablation: PiPNNHashPruneBackfillAblation,

    /// Maximum emitted distance-gated backups per row
    #[clap(long, default_value_t = 0)]
    hashprune_backfill_budget: usize,

    /// Euclidean distance ratio relative to the farthest primary
    #[clap(long, default_value_t = 1.0)]
    hashprune_backfill_alpha: f32,

    /// Optional per-hash quota for named R-half policies (1..=3)
    #[clap(long)]
    hashprune_rhalf_hash_quota: Option<usize>,

    /// Optional SoftTier priority alpha (production SoftTier default: 1.02)
    #[clap(long)]
    hashprune_soft_tier_priority_alpha: Option<f32>,

    /// Optional maximum retained backfill edges per row (production SoftTier default: 3)
    #[clap(long)]
    hashprune_rhalf_max_backfill_per_row: Option<usize>,

    /// Enable optional LRPF pre-spill compression
    #[clap(long)]
    lrpf_pre_spill_compress: bool,

    /// LRPF pre-spill width multiplier, clamped downstream to [l_max, 4*l_max]
    #[clap(long, default_value_t = 2.0)]
    lrpf_pre_spill_width_multiplier: f32,

    /// Number of nearest compact candidates protected from LRPF witness pruning
    #[clap(long, default_value_t = 1)]
    lrpf_protected_core: usize,

    /// LRPF output degree target before clamping to l_max
    #[clap(long, default_value_t = 64)]
    lrpf_target_degree: usize,

    /// LRPF farthest-candidate quota reserved for navigation edges
    #[clap(long, default_value_t = 0)]
    lrpf_long_edge_quota: usize,

    /// 叶子最小大小 C_min
    #[clap(long, default_value_t = 256)]
    leaf_size_min: usize,

    /// 叶子最大大小 C_max
    #[clap(long, default_value_t = 2048)]
    leaf_size_max: usize,

    /// RBC 采样 leader 比例 P_samp
    #[clap(long, default_value_t = 0.001)]
    psamp_fraction: f64,

    /// 顶层 fanout（例如 10）
    #[clap(long, default_value_t = 10)]
    fanout_top: usize,

    /// 第二层及以后 fanout（例如 3）
    #[clap(long, default_value_t = 3)]
    fanout_second: usize,

    /// Root fanout policy <fixed/cell-radius-certified/risk-budgeted>
    #[clap(long, default_value_t = RootFanoutPolicy::Fixed)]
    root_fanout_policy: RootFanoutPolicy,

    /// Number of owner-distance samples retained for each root cell radius certificate
    #[clap(long, default_value_t = 10)]
    root_radius_k: usize,

    /// Candidate root fanouts tried by cell-radius certification, comma-separated
    #[clap(long, default_value = "4,6,8")]
    root_radius_candidate_rs: String,

    /// Number of root assignments emitted with fixed fanout before radius certificates are used
    #[clap(long, default_value_t = 500_000)]
    root_radius_warmup_points: usize,

    /// Record cell-radius telemetry while still emitting fixed root fanout assignments
    #[clap(long)]
    root_radius_dry_run: bool,

    /// Stop OOM builds after depth-0 root materialization and preserve partition_d00_runs.bin
    #[clap(long)]
    root_stop_after_d00: bool,

    /// Target average root fanout for risk-budgeted utility selection
    #[clap(long, default_value_t = 8.0)]
    root_risk_target_avg_fanout: f32,

    /// Sigma multiplier for risk-budgeted utility selection
    #[clap(long, default_value_t = 1.0)]
    root_risk_sigma_lambda: f32,

    /// Optional explicit budget lambda for risk-budgeted utility selection
    #[clap(long)]
    root_risk_budget_lambda: Option<f32>,

    /// Sigma estimator for risk-budgeted utility selection
    #[clap(long, default_value_t = RootRiskSigmaMode::CellQ90Q50)]
    root_risk_sigma_mode: RootRiskSigmaMode,

    /// Number of pilot root assignments used for risk-budgeted calibration
    #[clap(long, default_value_t = 500_000)]
    root_risk_warmup_points: usize,

    /// Candidate root fanouts tried by risk budgeting, comma-separated
    #[clap(long, default_value = "4,6,8,10")]
    root_risk_candidate_rs: String,

    /// Record risk-budgeted telemetry while still emitting fixed root fanout assignments
    #[clap(long)]
    root_risk_dry_run: bool,

    /// Experimental: allow a full build using risk-budgeted root assignments
    #[clap(long, hide = true)]
    root_risk_allow_full_build: bool,

    /// PiPNN streaming partition channel <rbc/kmeans>
    #[clap(long, default_value_t = PiPNNPartitionerKind::Rbc)]
    partitioner: PiPNNPartitionerKind,

    /// Persistent GPU leaf top-k query-row tile size
    #[clap(long, default_value_t = 128)]
    gpu_subtree_persistent_leaf_tile_rows: usize,

    /// Consecutive 16-row tickets claimed per persistent large-task CTA
    #[clap(long, default_value_t = PiPNNGpuSubtreePersistentLargeTicketBatch::Auto, hide = true)]
    gpu_subtree_persistent_large_ticket_batch: PiPNNGpuSubtreePersistentLargeTicketBatch,

    /// Persistent top-k scheduler specialization
    #[clap(long, default_value_t = PiPNNGpuSubtreePersistentSchedulerMode::Auto, hide = true)]
    gpu_subtree_persistent_scheduler_mode: PiPNNGpuSubtreePersistentSchedulerMode,

    /// Maximum adaptive-idle sleep request in nanoseconds
    #[clap(long, default_value_t = 350, hide = true)]
    gpu_subtree_persistent_idle_backoff_ns: u32,

    /// Dense SQ8 large-node assignment executor
    #[clap(long, default_value_t = PiPNNGpuSubtreeSq8AssignmentKernel::Auto, hide = true)]
    gpu_subtree_sq8_assignment_kernel: PiPNNGpuSubtreeSq8AssignmentKernel,

    /// SQ8 global D0 assignment executor
    #[clap(
        long,
        default_value_t = PiPNNGpuSubtreeSq8D0AssignmentKernel::Auto,
        hide = true
    )]
    gpu_subtree_sq8_d0_assignment_kernel: PiPNNGpuSubtreeSq8D0AssignmentKernel,

    /// Dense SQ8 persistent leaf distance executor
    #[clap(long, default_value_t = PiPNNGpuSubtreeSq8LeafExecutor::Auto, hide = true)]
    gpu_subtree_sq8_leaf_executor: PiPNNGpuSubtreeSq8LeafExecutor,

    /// Experiment-only deterministic TESSA production-task capture directory
    #[clap(long, hide = true)]
    gpu_subtree_tessa_capture_dir: Option<PathBuf>,

    /// Dense SQ8 persistent leaf ownership policy
    #[clap(long, default_value_t = PiPNNGpuSubtreeSq8LeafMode::Auto, hide = true)]
    gpu_subtree_sq8_leaf_mode: PiPNNGpuSubtreeSq8LeafMode,

    /// D0 partition mode; SQ4+ defaults to budget-kmeans-topk, other modes to rbc-fanout
    #[clap(long, default_value_t = PiPNNGpuSubtreeD0PartitionMode::RbcFanout, hide = true)]
    gpu_subtree_d0_partition_mode: PiPNNGpuSubtreeD0PartitionMode,

    /// Explicit fixed Tagore-style diagnostic partition count (unused by budget-kmeans-topk)
    #[clap(long, default_value_t = 32, hide = true)]
    gpu_subtree_d0_tagore_partitions: usize,

    /// Diagnostic Tagore-style D0 top-k assignment count
    #[clap(long, default_value_t = 2, hide = true)]
    gpu_subtree_d0_tagore_k: usize,

    /// Diagnostic Tagore-style D0 k-means training sample fraction
    #[clap(long, default_value_t = 0.01, hide = true)]
    gpu_subtree_d0_tagore_sample_fraction: f64,

    /// Experimental D1 root partition mode inside each D0 child
    #[clap(long, default_value_t = PiPNNGpuSubtreeD1RootPartitionMode::Off, hide = true)]
    gpu_subtree_d1_root_partition_mode: PiPNNGpuSubtreeD1RootPartitionMode,

    /// Experimental GPU subtree D1+ vector staging mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeVectorStagingMode::Off, hide = true)]
    gpu_subtree_vector_staging_mode: PiPNNGpuSubtreeVectorStagingMode,

    /// Experimental GPU subtree fanout-aware vector staging scratch budget in bytes
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_vector_staging_budget_bytes: usize,

    /// Experimental GPU subtree rolling vector reuse recent-child window
    #[clap(long, default_value_t = 8, hide = true)]
    gpu_subtree_vector_staging_window_children: usize,

    /// Experimental GPU subtree D1+ child schedule
    #[clap(long, default_value_t = PiPNNGpuSubtreeChildScheduleMode::Natural, hide = true)]
    gpu_subtree_child_schedule: PiPNNGpuSubtreeChildScheduleMode,

    /// Diagnostic GPU subtree D0/root assignment backend; canonical runs use cpu
    #[clap(long, default_value_t = PiPNNComputeBackendKind::Cpu, hide = true)]
    gpu_subtree_root_assignment_backend: PiPNNComputeBackendKind,

    /// GPU subtree D1 assignment backend; canonical runs use persistent-wmma-tf32
    #[clap(long, default_value_t = PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32)]
    gpu_subtree_assignment_backend: PiPNNGpuSubtreeAssignmentBackend,

    /// Experimental GPU subtree assignment DCO mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeDcoMode::None, hide = true)]
    gpu_subtree_assignment_dco_mode: PiPNNGpuSubtreeDcoMode,

    /// Experimental assignment DCO rerank mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeDcoRerankMode::Exact, hide = true)]
    gpu_subtree_assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,

    /// Experimental assignment DCO projection dimension, code width, or block size
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_assignment_dco_proj_dim: usize,

    /// Experimental assignment DCO survivors retained before rerank or approximate output
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_assignment_dco_survivors: usize,

    /// Enable experimental assignment DCO false-prune audit telemetry
    #[clap(long, hide = true)]
    gpu_subtree_assignment_dco_audit: bool,

    /// Experimental D0/root assignment DCO mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeDcoMode::None, hide = true)]
    gpu_subtree_d0_dco_mode: PiPNNGpuSubtreeDcoMode,

    /// Experimental D0/root assignment DCO rerank mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeDcoRerankMode::Exact, hide = true)]
    gpu_subtree_d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,

    /// Experimental D0/root assignment DCO survivors retained before rerank or approximate output
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_d0_dco_survivors: usize,

    /// Enable experimental D0/root DCO false-prune audit telemetry
    #[clap(long, hide = true)]
    gpu_subtree_d0_dco_audit: bool,

    /// CPU kernel used by experimental D0/root SQ-int8 approximate assignment
    #[clap(long, default_value_t = SqInt8CpuKernel::Auto, hide = true)]
    gpu_subtree_d0_sq_cpu_kernel: SqInt8CpuKernel,

    /// GPU subtree execution schedule; canonical runs use fused-persistent
    #[clap(long, default_value_t = PiPNNGpuSubtreeExecutionMode::FusedPersistent)]
    gpu_subtree_execution_mode: PiPNNGpuSubtreeExecutionMode,

    /// GPU subtree leaf top-k backend; canonical runs use persistent-wmma-tf32
    #[clap(long, default_value_t = PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32)]
    gpu_subtree_leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend,

    /// Experimental GPU subtree leaf top-k DCO mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeDcoMode::None, hide = true)]
    gpu_subtree_leaf_dco_mode: PiPNNGpuSubtreeDcoMode,

    /// Experimental leaf top-k DCO rerank mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeDcoRerankMode::Exact, hide = true)]
    gpu_subtree_leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,

    /// Experimental leaf top-k DCO projection dimension, code width, or DDC block size
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_leaf_dco_proj_dim: usize,

    /// Experimental leaf top-k DCO survivors retained before rerank or approximate output
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_leaf_dco_survivors: usize,

    /// Enable experimental leaf top-k DCO false-prune audit telemetry
    #[clap(long, hide = true)]
    gpu_subtree_leaf_dco_audit: bool,

    /// ResInfer DDC_res PCA matrix sidecar (.fvecs)
    #[clap(long, hide = true)]
    gpu_subtree_leaf_dco_ddc_res_pca_matrix: Option<PathBuf>,

    /// Confirm input data is already projected by the ResInfer PCA matrix
    #[clap(long, hide = true)]
    gpu_subtree_leaf_dco_ddc_res_projected_input: bool,

    /// ResInfer DDC_res incremental block size (`base_dim`)
    #[clap(long, default_value_t = 32, hide = true)]
    gpu_subtree_leaf_dco_ddc_res_base_dim: usize,

    /// ResInfer DDC_res bound width (`sigma_count`)
    #[clap(long, default_value_t = 3.0, hide = true)]
    gpu_subtree_leaf_dco_ddc_res_sigma_count: f32,

    /// Experimental GPU subtree dense-i8 RabitQ+ quantized-only resident mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeRabitqPlusMode::Off, hide = true)]
    gpu_subtree_rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode,

    /// Experimental GPU subtree dense-i8 RabitQ+ total bits
    #[clap(long, default_value_t = 8, hide = true)]
    gpu_subtree_rabitq_plus_bits: usize,

    /// Experimental GPU subtree dense-i8 RabitQ+ deterministic rotator
    #[clap(long, default_value_t = PiPNNGpuSubtreeRabitqPlusRotator::FhtKac, hide = true)]
    gpu_subtree_rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator,

    /// Experimental GPU subtree dense-i8 RabitQ+ quantization/rotator seed
    #[clap(long, default_value_t = 42, hide = true)]
    gpu_subtree_rabitq_plus_seed: u64,

    /// Experimental RabitQ sidecar prefix: dense-i8 RabitQ+ uses .codes/.meta; RabitQ-Library
    /// split uses .split.*
    #[clap(long, hide = true)]
    gpu_subtree_rabitq_plus_sidecar_prefix: Option<PathBuf>,

    /// Experimental RabitQ sidecar mode <off/require/create-only/rebuild>
    #[clap(long, default_value_t = PiPNNGpuSubtreeRabitqPlusSidecarMode::Off, hide = true)]
    gpu_subtree_rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode,

    /// Experimental global-D0 qcode staging mode <off/rabitq-residual>
    #[clap(long, default_value_t = PiPNNGpuSubtreeQcodeStagingMode::Off, hide = true)]
    gpu_subtree_qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode,

    /// Experimental qcode D1+ payload mode <full-resident/child-local>
    #[clap(long, default_value_t = PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident, hide = true)]
    gpu_subtree_qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode,

    /// Experimental residual RaBitQ D1 distance mode <dense-i8-wmma/split-short-ex>
    #[clap(long, default_value_t = PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma, hide = true)]
    gpu_subtree_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode,

    /// Survivor count kept by the residual RaBitQ split short-code filter before ex refine
    #[clap(long, hide = true)]
    gpu_subtree_rabitq_short_survivors: Option<usize>,

    /// Residual k-means centroid count for RabitQ residual qcode staging
    #[clap(long, default_value_t = 16, hide = true)]
    gpu_subtree_rabitq_residual_centroids: usize,

    /// Raw rows uploaded per RabitQ residual staging shard
    #[clap(long, default_value_t = 524_288, hide = true)]
    gpu_subtree_qcode_staging_upload_shard_rows: usize,

    /// Optional global D0 leader count for RabitQ residual qcode staging; 0 uses planner default
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_qcode_global_d0_leaders: usize,

    /// Experimental residual RaBitQ global-D0 assignment mode <row-serial/tiled/batch/wmma-batch>
    #[clap(long, default_value_t = PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial, hide = true)]
    gpu_subtree_rabitq_d0_assign_mode: PiPNNGpuSubtreeRabitqD0AssignMode,

    /// Experimental residual RaBitQ D1+ top-k mode <row-serial/wmma-batch>
    #[clap(long, default_value_t = PiPNNGpuSubtreeRabitqD1TopKMode::RowSerial, hide = true)]
    gpu_subtree_rabitq_d1_topk_mode: PiPNNGpuSubtreeRabitqD1TopKMode,

    /// Experimental SQ8 staging mode <off/gpu-d0>
    #[clap(long, default_value_t = PiPNNGpuSubtreeSqStagingMode::Off, hide = true)]
    gpu_subtree_sq_staging_mode: PiPNNGpuSubtreeSqStagingMode,

    /// Experimental rotated residual SQ8 for D0; D1+ uses raw vectors
    #[clap(long, hide = true)]
    gpu_subtree_sq8_rotated_residual_d0: bool,

    /// Use the shared rotated residual SQ8 sidecar for D1+
    #[clap(long, hide = true)]
    gpu_subtree_sq8_rotated_residual_d1: bool,

    /// Compute D1+ distances directly from residual INT8 codes and center IDs on GPU
    #[clap(long, hide = true)]
    gpu_subtree_sq8_rotated_residual_d1_int8: bool,

    /// FHT-Kac rotation seed for rotated residual SQ8 D0
    #[clap(long, default_value_t = 42, hide = true)]
    gpu_subtree_sq8_rotated_residual_seed: u64,

    /// Raw rows uploaded per SQ8 GPU-D0 staging shard
    #[clap(long, default_value_t = 524_288, hide = true)]
    gpu_subtree_sq_staging_upload_shard_rows: usize,

    /// Keep SQ8 GPU-D0 staging but use raw FP32 for D1+ diagnostic comparisons
    #[clap(long, hide = true)]
    gpu_subtree_sq_staging_raw_d1_diagnostic: bool,

    /// Row tile for tiled residual RaBitQ D0 assignment
    #[clap(long, default_value_t = 4, hide = true)]
    gpu_subtree_rabitq_d0_row_tile: usize,

    /// Row chunk for tiled residual RaBitQ D0 assignment partial buffers
    #[clap(long, default_value_t = 4096, hide = true)]
    gpu_subtree_rabitq_d0_row_chunk: usize,

    /// Leader tile for tiled residual RaBitQ D0 assignment
    #[clap(long, default_value_t = 64, hide = true)]
    gpu_subtree_rabitq_d0_leader_tile: usize,

    /// Partial top-R candidates kept per row and leader tile
    #[clap(long, default_value_t = 8, hide = true)]
    gpu_subtree_rabitq_d0_partial_r: usize,

    /// Effective SQ code bits for SQ-int8 paths; storage remains i8
    #[clap(long, default_value_t = 8, hide = true)]
    gpu_subtree_sq_bits: usize,

    /// Encode each D1+ child with a shared random rotation and its D0 k-means center
    #[clap(long)]
    gpu_subtree_sq4_plus: bool,
    /// Encode each point once against its nearest D0 center (default for SQ4+); use =false for per-partition codes.
    #[arg(long, requires = "gpu_subtree_sq4_plus", num_args = 0..=1,
        require_equals = true, default_missing_value = "true")]
    gpu_subtree_sq4_primary_center: Option<bool>,

    /// Experimental packed SQ4 + native INT4 WMMA path
    #[clap(long, hide = true)]
    gpu_subtree_sq_native_int4: bool,

    /// Experimental GPU subtree SQ-int8 WMMA quantized-only resident mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeSqWmmaMode::Off, hide = true)]
    gpu_subtree_sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode,

    /// Experimental SQ-int8 WMMA sidecar prefix (without .codes/.meta suffix)
    #[clap(long, hide = true)]
    gpu_subtree_sq_wmma_sidecar_prefix: Option<PathBuf>,

    /// Experimental SQ-int8 WMMA sidecar mode <off/require/create-only/rebuild>
    #[clap(long, default_value_t = PiPNNGpuSubtreeSqWmmaSidecarMode::Off, hide = true)]
    gpu_subtree_sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode,

    /// Experimental GPU subtree TurboQuant-MSE4 quantized-only resident mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeTurboQuantMode::Off, hide = true)]
    gpu_subtree_turboquant_mode: PiPNNGpuSubtreeTurboQuantMode,

    /// Experimental TurboQuant-MSE4 scalar quantizer bits; v1 accepts only 4
    #[clap(long, default_value_t = 4, hide = true)]
    gpu_subtree_turboquant_bits: usize,

    /// Experimental TurboQuant-MSE4 dense-QR rotation seed
    #[clap(long, default_value_t = 42, hide = true)]
    gpu_subtree_turboquant_seed: u64,

    /// Experimental TurboQuant-MSE4 sidecar prefix (without .tq4.codes/.tq4.meta suffix)
    #[clap(long, hide = true)]
    gpu_subtree_turboquant_sidecar_prefix: Option<PathBuf>,

    /// Experimental TurboQuant-MSE4 sidecar mode <off/require/create-only/rebuild>
    #[clap(long, default_value_t = PiPNNGpuSubtreeTurboQuantSidecarMode::Off, hide = true)]
    gpu_subtree_turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode,

    /// Experimental TurboQuant-MSE4 sidecar construction backend <auto/cpu/cuda>
    #[clap(long, default_value_t = TurboQuantSidecarBuildBackend::Auto, hide = true)]
    gpu_subtree_turboquant_sidecar_build_backend: TurboQuantSidecarBuildBackend,

    /// Experimental TurboQuant-MSE4 cascade low-dimensional prefix
    #[clap(long, default_value_t = 128, hide = true)]
    gpu_subtree_turboquant_cascade_low_dim: usize,

    /// Experimental TurboQuant-MSE4 cascade survivor cap; 0 disables rerank
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_turboquant_cascade_survivor_cap: usize,

    /// Experimental TurboQuant-MSE4 adaptive rerank threshold; 0 disables adaptive cap
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_turboquant_adaptive_survivor_threshold: usize,

    /// Experimental TurboQuant-MSE4 survivor cap used below the adaptive threshold
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_turboquant_adaptive_survivor_low_cap: usize,

    /// Enable experimental GPU-subtree D0 capacity telemetry and oversized split-only safety
    /// repair
    #[clap(long, hide = true)]
    gpu_subtree_d0_capacity_enable: bool,

    /// Disable GPU-subtree D0 capacity admission and split safety
    #[clap(long, hide = true, conflicts_with = "gpu_subtree_d0_capacity_enable")]
    gpu_subtree_d0_capacity_disable: bool,

    /// Experimental D1+ task splitting policy
    #[clap(
        long,
        default_value_t = PiPNNGpuSubtreeD1PlusLazySplitMode::On,
        hide = true
    )]
    gpu_subtree_d1plus_lazy_split_mode: PiPNNGpuSubtreeD1PlusLazySplitMode,

    /// Subtree upload admission high-water ratio (SQ4+: 0.50; independent of D0 planning target)
    #[clap(long, hide = true)]
    gpu_subtree_d0_capacity_high_ratio: Option<f64>,

    /// D0 leader-planning budget and lazy split target ratio (SQ4+: 0.08)
    #[clap(long, hide = true)]
    gpu_subtree_d0_capacity_target_ratio: Option<f64>,

    /// Experimental D0 small-child telemetry ratio of resident GPU admission capacity
    #[clap(long, hide = true)]
    gpu_subtree_d0_capacity_low_ratio: Option<f64>,

    /// Experimental minimum efficient rows for standalone GPU subtree telemetry
    #[clap(long, default_value_t = 32 * 1024, hide = true)]
    gpu_subtree_d0_capacity_min_efficient_rows: usize,

    /// Experimental configured D0 root leader count while capacity telemetry acts as a safety
    /// valve
    #[clap(long, default_value_t = 64, hide = true)]
    gpu_subtree_d0_capacity_quality_min_root_leaders: usize,

    /// Experimental hard cap for configured D0 root leaders
    #[clap(long, default_value_t = 10_000, hide = true)]
    gpu_subtree_d0_capacity_root_leader_cap: usize,

    /// Experimental recursive split-only repair limit for one oversized D0 child
    #[clap(long, default_value_t = 8, hide = true)]
    gpu_subtree_d0_capacity_split_recursion_limit: usize,

    /// Experimental D0 capacity throughput model mode
    #[clap(long, default_value_t = PiPNNGpuSubtreeD0ThroughputModelMode::Off, hide = true)]
    gpu_subtree_d0_throughput_model: PiPNNGpuSubtreeD0ThroughputModelMode,

    /// Calibrated persistent-kernel growth limit for D0 throughput selection
    #[clap(long, default_value_t = 1.10, hide = true)]
    gpu_subtree_d0_throughput_persistent_growth_limit: f64,

    /// Calibrated visible sidecar preload limit for D0 throughput selection
    #[clap(long, default_value_t = 5000, hide = true)]
    gpu_subtree_d0_throughput_preload_visible_limit_ms: usize,

    /// Lowest D0 row-cap ratio considered by the throughput model
    #[clap(long, default_value_t = 0.50, hide = true)]
    gpu_subtree_d0_throughput_min_ratio: f64,

    /// Explicit D0 throughput calibration JSON file. Required for enforce mode.
    #[clap(long, hide = true)]
    gpu_subtree_d0_throughput_calibration_json: Option<PathBuf>,

    /// Optional p95 child row guardrail used by calibrated D0 throughput enforce mode
    #[clap(long, default_value_t = 0, hide = true)]
    gpu_subtree_d0_throughput_child_p95_guardrail_rows: usize,

    #[clap(long, default_value_t = PiPNNGpuSubtreePolicy::Report)]
    gpu_subtree_persistent_launch_policy: PiPNNGpuSubtreePolicy,

    #[clap(long, default_value_t = PiPNNGpuSubtreePolicy::Report)]
    gpu_subtree_hbm_policy: PiPNNGpuSubtreePolicy,

    #[clap(long)]
    gpu_subtree_launch_calibration_cache: Option<PathBuf>,

    #[clap(long, default_value_t = false)]
    gpu_subtree_launch_calibration_refresh: bool,

    /// K-means assignment/update iterations per recursive split
    #[clap(long, default_value_t = 4)]
    kmeans_iterations: usize,

    /// K-means child-count oversplit multiplier
    #[clap(long, default_value_t = 2)]
    kmeans_oversplit_factor: usize,

    /// 叶内 k-NN 候选个数
    #[clap(long, default_value_t = 2)]
    leaf_knn: usize,

    /// 单层最多 leader 数
    #[clap(long, default_value_t = 1000)]
    max_leaders: usize,

    /// 是否启用最终 RobustPrune 精剪
    #[clap(long)]
    final_prune: bool,

    /// 最终精剪 alpha 参数
    #[clap(long, default_value_t = 1.2)]
    final_prune_alpha: f32,

    /// 随机种子
    #[clap(long, default_value_t = 42)]
    seed: u64,

    /// PiPNN 重计算核使用的后端 <cpu/auto/cuda>
    #[clap(long, default_value = "cpu")]
    pipnn_backend: String,

    /// PiPNN GPU 后端使用的设备编号
    #[clap(long, default_value_t = 0)]
    pipnn_gpu_device: usize,

    /// PiPNN GPU compact top-k workspace budget（GiB）；0 表示使用库内保守默认值
    #[clap(long, default_value_t = 0.0)]
    pipnn_gpu_workspace_budget_gb: f64,

    /// Dump real CUDA compact top-k batches to this directory for isolated replay
    #[clap(long)]
    pipnn_cuda_compact_dump_dir: Option<PathBuf>,

    /// Maximum number of real CUDA compact top-k batches to dump; 0 disables dumping
    #[clap(long, default_value_t = 0)]
    pipnn_cuda_compact_dump_limit: usize,

    /// 启用默认关闭的 PiPNN pipeline runtime
    #[clap(long)]
    pipeline_runtime_enable: bool,

    /// Pipeline compute lane 模式 <passthrough/cpu-shadow/cuda/mock>
    #[clap(long, default_value_t = PiPNNPipelineComputeMode::Passthrough)]
    pipeline_compute_mode: PiPNNPipelineComputeMode,

    /// 覆盖 LeafDescriptorQueue 容量
    #[clap(long)]
    pipeline_leaf_descriptor_queue_capacity: Option<usize>,

    /// 覆盖 StagedLeafQueue 容量
    #[clap(long)]
    pipeline_staged_queue_capacity: Option<usize>,

    /// 覆盖 CompactTopKQueue 容量
    #[clap(long)]
    pipeline_topk_queue_capacity: Option<usize>,

    /// 覆盖 EdgeFlushQueue 容量
    #[clap(long)]
    pipeline_edge_queue_capacity: Option<usize>,

    /// 覆盖 producer active 时 compute lane 最大 inflight
    #[clap(long)]
    pipeline_max_inflight_compute_tasks_active: Option<usize>,

    /// 覆盖 producer finished 后 compute lane 最大 inflight
    #[clap(long)]
    pipeline_max_inflight_compute_tasks_drain: Option<usize>,

    /// 自动推荐参数模式：根据数据集规模和线程数自动计算最优参数
    /// 启用后会覆盖手动指定的 c_max / max_leaders / psamp_fraction / fanout / leaf_knn
    #[clap(long)]
    auto_params: bool,

    /// 仅估算构图时间，不实际构图（可与 --auto-params 配合使用）
    #[clap(long)]
    estimate_only: bool,

    /// 启用 PiPNN OOM research 模式
    #[clap(long)]
    oom_enable: bool,

    /// OOM research 模式总内存预算（GiB）；0 表示按宿主机/cgroup 内存自动计算
    #[clap(long, default_value_t = 0.0)]
    oom_memory_budget_gb: f64,

    /// Fraction of the legacy OOM resident reservoir budget to keep
    #[clap(long, default_value_t = 1.0)]
    oom_reservoir_share: f64,

    /// Diagnostic cap for OOM sidecar resident budget in GiB
    #[clap(long)]
    oom_sidecar_resident_budget_cap_gb: Option<f64>,

    /// Legacy diagnostic alias for --oom-io-mode direct
    #[clap(long, hide = true)]
    oom_strict_io: bool,

    /// OOM bulk I/O mode; canonical GPU-subtree validation uses buffered
    #[clap(long, default_value_t = PiPNNOomIoMode::Auto)]
    oom_io_mode: PiPNNOomIoMode,

    /// OOM research 模式 sketch cache 预算（GiB）
    #[clap(long, default_value_t = 0.0)]
    oom_sketch_cache_gb: f64,

    /// Retired diagnostic raw vector cache budget in GiB
    #[clap(long, default_value_t = 0.0, hide = true)]
    oom_raw_vector_cache_gb: f64,

    /// OOM research 模式 spill cache 预算（GiB）
    #[clap(long, default_value_t = 0.0)]
    oom_spill_cache_gb: f64,

    /// Online disk-suffix HashPrune mode <off/direct-first/writeback-cache>
    #[clap(long, default_value_t = PiPNNOnlineSuffixReduceMode::Off)]
    online_suffix_reduce: PiPNNOnlineSuffixReduceMode,

    /// Direct candidate queue budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_direct_queue_gb: f64,

    /// Source-reservoir cache budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_state_cache_gb: f64,

    /// Disk-state page-cache budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_state_page_cache_gb: f64,

    /// Disk-state asynchronous writeback budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_state_writeback_gb: f64,

    /// Fallback spill writer budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_spill_buffer_gb: f64,

    /// Per-shard reorder window budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_reorder_window_gb: f64,

    /// Grouped state-read window budget in GiB; 0 uses the library default
    #[clap(long, default_value_t = 0.0)]
    online_suffix_read_window_gb: f64,

    /// Optional pinned source prefix for writeback-cache mode
    #[clap(long)]
    online_suffix_pinned_prefix_points: Option<usize>,

    /// Online suffix reducer workers; 0 derives a bounded worker count
    #[clap(long, default_value_t = 0)]
    online_suffix_reduce_workers: usize,

    /// Maximum uncompacted log-assisted delta/checkpoint bytes in GiB
    #[clap(long, default_value_t = 0.0)]
    online_suffix_delta_budget_gb: f64,

    /// DRAM buffer used to form sorted log-assisted runs in GiB
    #[clap(long, default_value_t = 0.0)]
    online_suffix_delta_buffer_gb: f64,

    /// Source IDs per log-assisted compaction range; 0 selects automatic sizing
    #[clap(long, default_value_t = 0)]
    online_suffix_range_points: usize,

    /// Uncompacted bytes that schedule range compaction in GiB
    #[clap(long, default_value_t = 0.0)]
    online_suffix_range_trigger_gb: f64,

    /// Soft active-run reference for slot-bounded early compaction eligibility
    #[clap(long, default_value_t = 0)]
    online_suffix_max_runs_per_range: usize,

    /// Background log-assisted range compaction workers
    #[clap(long, default_value_t = 0)]
    online_suffix_compaction_workers: usize,

    /// Per-epoch source updates required for compaction-only cache promotion
    #[clap(long, default_value_t = 0)]
    online_suffix_cache_hot_updates: usize,

    /// Physical disk-state slots per logical suffix source
    #[clap(long, default_value_t = 2.0)]
    online_suffix_state_overprovision_ratio: f64,

    /// OOM research 模式临时目录
    #[clap(long, default_value = "")]
    oom_temp_dir: String,

    /// 保留 OOM research 中间产物
    #[clap(long)]
    oom_keep_artifacts: bool,

    /// 将 OOM research profiling 额外写到 JSON 文件
    #[clap(long)]
    oom_profile_json: Option<PathBuf>,

    /// Write default-off SoftTier construction diagnostics into this directory.
    #[clap(long)]
    hashprune_backfill_diagnostics_prefix: Option<PathBuf>,

    /// Write driver-level wall-clock phase timing JSON for profile/external reconciliation
    #[clap(long, hide = true)]
    driver_timing_json: Option<PathBuf>,
}

fn main() -> AnnResult<()> {
    let main_start = Instant::now();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
        )
        .init();

    let args = Args::parse();
    let max_leaders_explicit = cli_flag_present("--max-leaders");

    validate_args(&args)?;

    let metric = args
        .dist_fn
        .parse::<Metric>()
        .map_err(|err| AnnError::log_index_config_error("dist_fn".to_string(), err.to_string()))?;
    let compute_backend = args
        .pipnn_backend
        .parse::<PiPNNComputeBackendKind>()
        .map_err(|err| AnnError::log_index_config_error("pipnn_backend".to_string(), err))?;

    let (data_num, data_dim) = load_metadata_from_file(&args.data_path).unwrap();
    if args.gpu_subtree_sq4_plus && !data_dim.is_power_of_two() {
        return Err(AnnError::log_index_config_error(
            "gpu_subtree_sq4_plus".to_string(),
            "SQ4+ FHT rotation requires a power-of-two dimension".to_string(),
        ));
    }

    let num_threads = if args.num_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(1)
    } else {
        args.num_threads
    };

    println!(
        "Starting PiPNN index build with R: {}  Lbuild: {}  alpha: {}  #threads: {} search_DRAM_budget: {} build_DRAM_budget: {}",
        args.max_degree,
        args.lbuild,
        ALPHA,
        num_threads,
        args.search_dram_budget,
        args.build_dram_budget
    );

    let mut pipnn_params = if args.auto_params {
        let mut params = PiPNNParams::recommend(
            data_num,
            data_dim,
            args.max_degree as usize,
            num_threads as usize,
        );
        // 保留用户显式指定的非推荐参数
        params.m_hash_bits = args.m_hash_bits;
        params.final_prune = args.final_prune;
        params.final_prune_alpha = args.final_prune_alpha;
        params.random_seed = args.seed;
        params.metric = metric;
        params.compute_backend = compute_backend;
        params.gpu_device_id = args.pipnn_gpu_device;
        params.gpu_workspace_budget_bytes = gib_to_bytes(args.pipnn_gpu_workspace_budget_gb)?;
        params.cuda_compact_dump_dir = args.pipnn_cuda_compact_dump_dir.clone();
        params.cuda_compact_dump_limit = args.pipnn_cuda_compact_dump_limit;
        apply_pipeline_args(&mut params, &args);
        params.oom_enable = args.oom_enable;
        params.oom_memory_budget_bytes = gib_to_bytes(args.oom_memory_budget_gb)?;
        params.oom_strict_io = args.oom_strict_io;
        params.oom_io_mode = args.oom_io_mode;
        params.oom_reservoir_budget_share = args.oom_reservoir_share;
        params.oom_sidecar_resident_budget_cap_bytes = args
            .oom_sidecar_resident_budget_cap_gb
            .map(gib_to_bytes)
            .transpose()?;
        params.oom_sketch_cache_bytes = gib_to_bytes(args.oom_sketch_cache_gb)?;
        params.oom_raw_vector_cache_bytes = gib_to_bytes(args.oom_raw_vector_cache_gb)?;
        params.oom_spill_cache_bytes = gib_to_bytes(args.oom_spill_cache_gb)?;
        apply_online_suffix_args(&mut params, &args)?;
        params.oom_temp_dir = PathBuf::from(&args.oom_temp_dir);
        params.oom_keep_artifacts = args.oom_keep_artifacts || args.root_stop_after_d00;
        params.oom_profile_json = args.oom_profile_json.clone();
        params.root_fanout_policy = args.root_fanout_policy;
        params.root_radius_k = args.root_radius_k;
        params.root_radius_candidate_rs =
            parse_root_radius_candidate_rs(&args.root_radius_candidate_rs)?;
        params.root_radius_warmup_points = args.root_radius_warmup_points;
        params.root_radius_dry_run = args.root_radius_dry_run;
        params.root_stop_after_d00 = args.root_stop_after_d00;
        params.root_risk_target_avg_fanout = args.root_risk_target_avg_fanout;
        params.root_risk_sigma_lambda = args.root_risk_sigma_lambda;
        params.root_risk_budget_lambda = args.root_risk_budget_lambda;
        params.root_risk_sigma_mode = args.root_risk_sigma_mode;
        params.root_risk_warmup_points = args.root_risk_warmup_points;
        params.root_risk_candidate_rs =
            parse_root_risk_candidate_rs(&args.root_risk_candidate_rs, args.fanout_top)?;
        params.root_risk_dry_run = args.root_risk_dry_run;
        params.root_risk_allow_full_build = args.root_risk_allow_full_build;
        apply_gpu_subtree_args(&mut params, &args);
        apply_user_leader_precedence(&mut params, &args, max_leaders_explicit);
        params.kmeans_iterations = args.kmeans_iterations;
        params.kmeans_oversplit_factor = args.kmeans_oversplit_factor;
        params
    } else {
        let mut params = PiPNNParams::default();
        params.m_hash_bits = args.m_hash_bits;
        params.l_max = args.max_degree as usize;
        params.c_min = args.leaf_size_min;
        params.c_max = args.leaf_size_max;
        params.psamp_fraction = args.psamp_fraction;
        params.max_leaders = args.max_leaders;
        params.fanout_top = args.fanout_top;
        params.fanout_second = args.fanout_second;
        params.leaf_knn = args.leaf_knn;
        params.final_prune = args.final_prune;
        params.final_prune_alpha = args.final_prune_alpha;
        params.random_seed = args.seed;
        params.metric = metric;
        params.compute_backend = compute_backend;
        params.gpu_device_id = args.pipnn_gpu_device;
        params.gpu_workspace_budget_bytes = gib_to_bytes(args.pipnn_gpu_workspace_budget_gb)?;
        params.cuda_compact_dump_dir = args.pipnn_cuda_compact_dump_dir.clone();
        params.cuda_compact_dump_limit = args.pipnn_cuda_compact_dump_limit;
        apply_pipeline_args(&mut params, &args);
        params.oom_enable = args.oom_enable;
        params.oom_memory_budget_bytes = gib_to_bytes(args.oom_memory_budget_gb)?;
        params.oom_strict_io = args.oom_strict_io;
        params.oom_io_mode = args.oom_io_mode;
        params.oom_reservoir_budget_share = args.oom_reservoir_share;
        params.oom_sidecar_resident_budget_cap_bytes = args
            .oom_sidecar_resident_budget_cap_gb
            .map(gib_to_bytes)
            .transpose()?;
        params.oom_sketch_cache_bytes = gib_to_bytes(args.oom_sketch_cache_gb)?;
        params.oom_raw_vector_cache_bytes = gib_to_bytes(args.oom_raw_vector_cache_gb)?;
        params.oom_spill_cache_bytes = gib_to_bytes(args.oom_spill_cache_gb)?;
        apply_online_suffix_args(&mut params, &args)?;
        params.oom_temp_dir = PathBuf::from(&args.oom_temp_dir);
        params.oom_keep_artifacts = args.oom_keep_artifacts || args.root_stop_after_d00;
        params.oom_profile_json = args.oom_profile_json.clone();
        params.root_fanout_policy = args.root_fanout_policy;
        params.root_radius_k = args.root_radius_k;
        params.root_radius_candidate_rs =
            parse_root_radius_candidate_rs(&args.root_radius_candidate_rs)?;
        params.root_radius_warmup_points = args.root_radius_warmup_points;
        params.root_radius_dry_run = args.root_radius_dry_run;
        params.root_stop_after_d00 = args.root_stop_after_d00;
        params.root_risk_target_avg_fanout = args.root_risk_target_avg_fanout;
        params.root_risk_sigma_lambda = args.root_risk_sigma_lambda;
        params.root_risk_budget_lambda = args.root_risk_budget_lambda;
        params.root_risk_sigma_mode = args.root_risk_sigma_mode;
        params.root_risk_warmup_points = args.root_risk_warmup_points;
        params.root_risk_candidate_rs =
            parse_root_risk_candidate_rs(&args.root_risk_candidate_rs, args.fanout_top)?;
        params.root_risk_dry_run = args.root_risk_dry_run;
        params.root_risk_allow_full_build = args.root_risk_allow_full_build;
        apply_gpu_subtree_args(&mut params, &args);
        apply_user_leader_precedence(&mut params, &args, max_leaders_explicit);
        params.kmeans_iterations = args.kmeans_iterations;
        params.kmeans_oversplit_factor = args.kmeans_oversplit_factor;
        params
    };
    apply_prune_backend_env_overrides(&mut pipnn_params)?;

    if maybe_create_rabitq_plus_sidecar(&args, &pipnn_params)? {
        if matches!(
            args.gpu_subtree_d0_dco_mode,
            PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter | PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        ) || args.gpu_subtree_leaf_dco_mode == PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        {
            println!(
                "PiPNN: RabitQ-Library split sidecar create-only completed; skipping index build"
            );
        } else {
            println!("PiPNN: dense-i8 RabitQ+ sidecar create-only completed; skipping index build");
        }
        return Ok(());
    }
    let sq_wmma_sidecar_cleanup = SqWmmaSidecarCleanupGuard::for_rebuild(&args);
    if maybe_create_sq_wmma_sidecar(&args, &pipnn_params, data_num, data_dim)? {
        println!("PiPNN: SQ+WMMA sidecar create-only completed; skipping index build");
        return Ok(());
    }
    if maybe_create_turboquant_mse4_sidecar(&args, &pipnn_params)? {
        println!("PiPNN: TurboQuant-MSE4 sidecar create-only completed; skipping index build");
        return Ok(());
    }

    if args.estimate_only {
        let est = pipnn_params.estimate_build_time(data_num, data_dim, num_threads as usize);
        println!("=== Estimate Only ===");
        println!("  Dataset: n={}, d={}", data_num, data_dim);
        println!(
            "  Estimated build time: {:.0}s ({:.1} min)",
            est,
            est / 60.0
        );
        println!("=====================");
        return Ok(());
    }

    let pre_build_wall = main_start.elapsed();
    let build_call_start = Instant::now();
    let build_result = match args.data_type.to_ascii_lowercase().as_str() {
        "float" | "uint8" => build_shear_index_with_data_type(
            args.data_type.as_str(),
            metric,
            &args.data_path,
            args.max_degree,
            args.lbuild,
            &args.index_path_prefix,
            num_threads,
            args.overwrite,
            args.subgraph_threshold,
            &pipnn_params,
        ),
        _ => {
            println!("Unsupported type. Use one of int8, uint8, float or f16.");
            return Err(AnnError::log_index_config_error(
                "data_type".to_string(),
                "Invalid data type".to_string(),
            ));
        }
    };
    let build_call_wall = build_call_start.elapsed();
    let post_build_start = Instant::now();
    drop(sq_wmma_sidecar_cleanup);
    let post_build_wall = post_build_start.elapsed();

    if let Some(driver_timing_json) = &args.driver_timing_json {
        let report = DriverTimingReport {
            total_wall: main_start.elapsed(),
            pre_build_wall,
            build_call_wall,
            post_build_wall,
            build_success: build_result.is_ok(),
            profile_json_path: args.oom_profile_json.clone(),
            phases: build_result
                .as_ref()
                .map(|timing| timing.phases.clone())
                .unwrap_or_default(),
        };
        if let Err(timing_err) = write_driver_timing_json(driver_timing_json, &report) {
            if build_result.is_ok() {
                return Err(timing_err);
            }
            eprintln!("PiPNN: failed to write driver timing JSON: {timing_err}");
        }
    }

    match build_result {
        Ok(_) => {
            println!("PiPNN: index build completed successfully");
            Ok(())
        }
        Err(err) => {
            println!("PiPNN: error: {:?}", err);
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Duration;

    use clap::{CommandFactory, Parser};
    use shear_core::algorithm::pipnn::params::PiPNNGpuSubtreePolicy;
    use shear_core::algorithm::pipnn::{
        PiPNNGpuSubtreeAssignmentBackend, PiPNNGpuSubtreeChildScheduleMode,
        PiPNNGpuSubtreeD0PartitionMode, PiPNNGpuSubtreeD0ThroughputModelMode,
        PiPNNGpuSubtreeD1PlusLazySplitMode, PiPNNGpuSubtreeD1RootPartitionMode,
        PiPNNGpuSubtreeDcoMode, PiPNNGpuSubtreeDcoRerankMode, PiPNNGpuSubtreeExecutionMode,
        PiPNNGpuSubtreeLeafTopKBackend, PiPNNGpuSubtreePersistentLargeTicketBatch,
        PiPNNGpuSubtreePersistentSchedulerMode, PiPNNGpuSubtreeQcodeD1PayloadMode,
        PiPNNGpuSubtreeQcodeStagingMode, PiPNNGpuSubtreeRabitqD0AssignMode,
        PiPNNGpuSubtreeRabitqD1DistanceMode, PiPNNGpuSubtreeRabitqD1TopKMode,
        PiPNNGpuSubtreeRabitqPlusMode, PiPNNGpuSubtreeRabitqPlusRotator,
        PiPNNGpuSubtreeRabitqPlusSidecarMode, PiPNNGpuSubtreeSq8AssignmentKernel,
        PiPNNGpuSubtreeSq8D0AssignmentKernel, PiPNNGpuSubtreeSq8LeafExecutor,
        PiPNNGpuSubtreeSq8LeafMode, PiPNNGpuSubtreeSqStagingMode, PiPNNGpuSubtreeSqWmmaMode,
        PiPNNGpuSubtreeSqWmmaSidecarMode, PiPNNGpuSubtreeTurboQuantMode,
        PiPNNGpuSubtreeTurboQuantSidecarMode, PiPNNGpuSubtreeVectorStagingMode,
        PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNHashPruneEvidenceMode,
        PiPNNOnlineSuffixReduceMode, PiPNNOomIoMode, PiPNNPartitionerKind,
        PiPNNPipelineComputeMode, PiPNNPruneBackend, SqInt8CpuKernel,
    };
    use shear_core::algorithm::{PiPNNComputeBackendKind, PiPNNParams, RootFanoutPolicy};
    use shear_core::common::Metric;
    use shear_core::index::InmemIndex;
    use shear_core::model::{IndexConfiguration, IndexWriteParametersBuilder};
    use shear_core::utils::round_up;

    use super::{
        Args, CompletedPhase, DriverTimingReport, TurboQuantSidecarBuildBackend,
        apply_gpu_subtree_args, apply_online_suffix_args, apply_prune_backend_env_overrides_with,
        build_shear_index, build_shear_index_with_data_type,
        effective_gpu_subtree_d0_capacity_enabled, parse_schedstat_running_ns,
        parse_status_snapshot, phase_cpu_cores_used, remove_sq_wmma_sidecar_files, validate_args,
        write_driver_timing_json,
    };

    fn unique_temp_dir(label: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!(
            "shear_core-{label}-{}-{}",
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_test_fbin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(dim as u32).to_le_bytes()).unwrap();
        for row in 0..rows {
            for axis in 0..dim {
                let value = row as f32 * 0.25 + axis as f32 * 0.5;
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        file.flush().unwrap();
    }

    fn write_test_u8bin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(dim as u32).to_le_bytes()).unwrap();
        for row in 0..rows {
            for axis in 0..dim {
                file.write_all(&[((row * 17 + axis * 29) % 256) as u8])
                    .unwrap();
            }
        }
        file.flush().unwrap();
    }

    #[test]
    fn cli_maps_online_suffix_reduce_controls() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--online-suffix-reduce",
            "writeback-cache",
            "--online-suffix-direct-queue-gb",
            "0.25",
            "--online-suffix-state-cache-gb",
            "0.375",
            "--online-suffix-state-page-cache-gb",
            "0.5",
            "--online-suffix-read-window-gb",
            "0.03125",
            "--online-suffix-pinned-prefix-points",
            "12345",
            "--online-suffix-state-writeback-gb",
            "0.125",
            "--online-suffix-spill-buffer-gb",
            "0.0625",
            "--online-suffix-reorder-window-gb",
            "0.015625",
            "--online-suffix-reduce-workers",
            "3",
            "--online-suffix-state-overprovision-ratio",
            "2.25",
        ]);
        let mut params = PiPNNParams::default();
        apply_online_suffix_args(&mut params, &args).unwrap();

        assert_eq!(
            params.online_suffix_reduce_mode,
            PiPNNOnlineSuffixReduceMode::WritebackCache
        );
        assert_eq!(params.online_suffix_direct_queue_bytes, 256 * 1024 * 1024);
        assert_eq!(params.online_suffix_state_cache_bytes, 384 * 1024 * 1024);
        assert_eq!(
            params.online_suffix_state_page_cache_bytes,
            512 * 1024 * 1024
        );
        assert_eq!(
            params.online_suffix_state_writeback_bytes,
            128 * 1024 * 1024
        );
        assert_eq!(params.online_suffix_spill_buffer_bytes, 64 * 1024 * 1024);
        assert_eq!(params.online_suffix_reorder_window_bytes, 16 * 1024 * 1024);
        assert_eq!(params.online_suffix_read_window_bytes, 32 * 1024 * 1024);
        assert_eq!(params.online_suffix_pinned_prefix_points, Some(12_345));
        assert_eq!(params.online_suffix_reduce_workers, 3);
        assert_eq!(params.online_suffix_state_overprovision_ratio, 2.25);
    }

    #[test]
    fn cli_maps_log_assisted_controls() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--online-suffix-reduce",
            "log-assisted",
            "--online-suffix-delta-budget-gb",
            "4",
            "--online-suffix-delta-buffer-gb",
            "0.0625",
            "--online-suffix-range-points",
            "65536",
            "--online-suffix-range-trigger-gb",
            "0.125",
            "--online-suffix-max-runs-per-range",
            "6",
            "--online-suffix-compaction-workers",
            "2",
            "--online-suffix-cache-hot-updates",
            "3",
        ]);
        let mut params = PiPNNParams::default();
        apply_online_suffix_args(&mut params, &args).unwrap();

        assert_eq!(
            params.online_suffix_reduce_mode,
            PiPNNOnlineSuffixReduceMode::LogAssisted
        );
        assert_eq!(
            params.online_suffix_delta_budget_bytes,
            4 * 1024 * 1024 * 1024
        );
        assert_eq!(params.online_suffix_delta_buffer_bytes, 64 * 1024 * 1024);
        assert_eq!(params.online_suffix_range_points, 65_536);
        assert_eq!(params.online_suffix_range_trigger_bytes, 128 * 1024 * 1024);
        assert_eq!(params.online_suffix_max_runs_per_range, 6);
        assert_eq!(params.online_suffix_compaction_workers, 2);
        assert_eq!(params.online_suffix_cache_hot_updates, 3);
    }

    #[test]
    fn parse_status_snapshot_reads_threads_and_rss_kib() {
        let status = "\
Name:\tbuild_shear_index\n\
State:\tR (running)\n\
Threads:\t42\n\
VmRSS:\t  123456 kB\n";

        let snapshot = parse_status_snapshot(status).expect("status snapshot");
        assert_eq!(snapshot.thread_count, Some(42));
        assert_eq!(snapshot.rss_kib, Some(123_456));
    }

    #[test]
    fn parse_schedstat_running_ns_reads_first_field() {
        let schedstat = "123456789 987654321 42\n";
        assert_eq!(parse_schedstat_running_ns(schedstat), Some(123_456_789));
    }

    #[test]
    fn completed_phase_cpu_cores_used_uses_wall_and_cpu_deltas() {
        let phase = CompletedPhase {
            name: "build_graph".to_string(),
            wall_time: Duration::from_secs(2),
            cpu_time: Duration::from_secs(6),
            peak_threads: 14,
            peak_rss_kib: Some(2_048_000),
            max_cpu_cores: 3.5,
            sample_count: 4,
        };

        assert!((phase_cpu_cores_used(&phase) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn driver_timing_json_includes_profile_gap_and_phases() {
        let dir = unique_temp_dir("driver-timing-json");
        let timing_path = dir.join("driver_timing.json");
        let profile_path = dir.join("profile.json");
        std::fs::write(
            &profile_path,
            r#"{"serial_timing":{"stage_total_ms":761660},"gpu_subtree_timing":{"stage_total_ms":761660}}"#,
        )
        .unwrap();

        let report = DriverTimingReport {
            total_wall: Duration::from_millis(855_450),
            pre_build_wall: Duration::from_millis(100),
            build_call_wall: Duration::from_millis(855_000),
            post_build_wall: Duration::from_millis(350),
            build_success: true,
            profile_json_path: Some(profile_path.clone()),
            phases: vec![
                CompletedPhase {
                    name: "startup".to_string(),
                    wall_time: Duration::from_millis(10),
                    cpu_time: Duration::from_millis(5),
                    peak_threads: 1,
                    peak_rss_kib: Some(1024),
                    max_cpu_cores: 0.5,
                    sample_count: 1,
                },
                CompletedPhase {
                    name: "build_graph".to_string(),
                    wall_time: Duration::from_millis(761_660),
                    cpu_time: Duration::from_millis(100_000),
                    peak_threads: 19,
                    peak_rss_kib: Some(73_207_280),
                    max_cpu_cores: 2.0,
                    sample_count: 2,
                },
            ],
        };

        write_driver_timing_json(&timing_path, &report).unwrap();
        let json = std::fs::read_to_string(&timing_path).unwrap();

        assert!(json.contains(r#""schema": "pipnn_driver_timing_v1""#));
        assert!(json.contains(r#""total_wall_ms": 855450"#));
        assert!(json.contains(r#""profile_stage_total_ms": 761660"#));
        assert!(json.contains(r#""total_minus_profile_stage_ms": 93790"#));
        assert!(json.contains(r#""build_call_minus_profile_stage_ms": 93340"#));
        assert!(json.contains(r#""name": "build_graph""#));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn build_shear_index_parses_lrpf_prune_backend_flags() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--prune-backend",
            "lrpf",
            "--lrpf-pre-spill-compress",
            "--lrpf-pre-spill-width-multiplier",
            "2.5",
            "--lrpf-protected-core",
            "2",
            "--lrpf-target-degree",
            "64",
            "--lrpf-long-edge-quota",
            "0",
        ])
        .unwrap();

        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(args.prune_backend, PiPNNPruneBackend::LRPF);
        assert_eq!(params.prune_backend, PiPNNPruneBackend::LRPF);
        assert!(params.lrpf_pre_spill_compress_enable);
        assert_eq!(params.lrpf_pre_spill_width_multiplier, 2.5);
        assert_eq!(params.lrpf_protected_core, 2);
        assert_eq!(params.lrpf_target_degree, 64);
        assert_eq!(params.lrpf_long_edge_quota, 0);
    }

    #[test]
    fn hashprune_evidence_cli_parses_all_modes() {
        for (raw_mode, expected_mode, budget) in [
            ("off", PiPNNHashPruneEvidenceMode::Off, "0"),
            ("report", PiPNNHashPruneEvidenceMode::Report, "0"),
            ("enforce", PiPNNHashPruneEvidenceMode::Enforce, "1"),
        ] {
            let args = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--hashprune-evidence-mode",
                raw_mode,
                "--hashprune-evidence-overflow-budget",
                budget,
            ])
            .unwrap();

            let mut params = PiPNNParams::default();
            apply_gpu_subtree_args(&mut params, &args);
            assert_eq!(params.hashprune_evidence_mode, expected_mode);
            assert_eq!(
                params.hashprune_evidence_overflow_budget,
                budget.parse::<usize>().unwrap()
            );
            assert!(params.validate_hashprune_evidence_config().is_ok());
        }
    }

    #[test]
    fn hashprune_evidence_cli_rejects_invalid_mode_budget_pair() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-evidence-mode",
            "report",
            "--hashprune-evidence-overflow-budget",
            "1",
        ])
        .unwrap();

        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert!(params.validate_hashprune_evidence_config().is_err());
    }

    #[test]
    fn hashprune_backfill_cli_parses_approved_tuple() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "distance-gated",
            "--hashprune-backfill-budget",
            "8",
            "--hashprune-backfill-alpha",
            "1.1",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::DistanceGated
        );
        assert_eq!(params.hashprune_backfill_budget, 8);
        assert_eq!(params.hashprune_backfill_alpha, 1.1);
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_stratified_backfill_cli_parses_k4_tuple() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "distance-gated-stratified",
            "--hashprune-backfill-budget",
            "4",
            "--hashprune-backfill-alpha",
            "1.1",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::DistanceGatedStratified
        );
        assert_eq!(params.hashprune_backfill_budget, 4);
        assert_eq!(params.hashprune_backfill_alpha, 1.1);
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_multihit_backfill_cli_parses_k6_tuple() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "distance-gated-multihit",
            "--hashprune-backfill-budget",
            "6",
            "--hashprune-backfill-alpha",
            "1.1",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::DistanceGatedMultihit
        );
        assert_eq!(params.hashprune_backfill_budget, 6);
        assert_eq!(params.hashprune_backfill_alpha, 1.1);
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_tiered_d52_backfill_cli_parses_k8_tuple() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "distance-gated-tiered-d52",
            "--hashprune-backfill-budget",
            "8",
            "--hashprune-backfill-alpha",
            "1.1",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52
        );
        assert_eq!(params.hashprune_backfill_budget, 8);
        assert_eq!(params.hashprune_backfill_alpha, 1.1);
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_tiered_rhalf_cli_parses_full_mode_and_named_ablations() {
        for (raw, expected) in [
            ("none", PiPNNHashPruneBackfillAblation::None),
            (
                "budget-quarter",
                PiPNNHashPruneBackfillAblation::BudgetQuarter,
            ),
            ("no-half-cap", PiPNNHashPruneBackfillAblation::NoHalfCap),
            ("quota-1", PiPNNHashPruneBackfillAblation::Quota1),
            ("quota-2", PiPNNHashPruneBackfillAblation::Quota2),
            (
                "distance-first",
                PiPNNHashPruneBackfillAblation::DistanceFirst,
            ),
            ("no-veto", PiPNNHashPruneBackfillAblation::NoVeto),
        ] {
            let args = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--hashprune-backfill-mode",
                "distance-gated-tiered-rhalf",
                "--hashprune-backfill-ablation",
                raw,
                "--hashprune-backfill-budget",
                "0",
                "--hashprune-backfill-alpha",
                "1.1",
            ])
            .unwrap();
            let mut params = PiPNNParams::default();
            apply_gpu_subtree_args(&mut params, &args);
            assert_eq!(
                params.hashprune_backfill_mode,
                PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
            );
            assert_eq!(params.hashprune_backfill_ablation, expected);
            assert!(params.validate_hashprune_backfill_config().is_ok());
        }
    }

    #[test]
    fn hashprune_named_rhalf_cli_modes_parse_with_frozen_defaults() {
        for (raw, expected) in [
            (
                "rhalf-hash-slot-1",
                PiPNNHashPruneBackfillMode::RHalfHashSlot1,
            ),
            (
                "rhalf-distance-first-q3",
                PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
            ),
            (
                "rhalf-soft-tier-q3",
                PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            ),
            (
                "rhalf-hard-tier-q3",
                PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
            ),
        ] {
            let args = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--hashprune-backfill-mode",
                raw,
            ])
            .unwrap();
            let mut params = PiPNNParams::default();
            apply_gpu_subtree_args(&mut params, &args);
            assert_eq!(params.hashprune_backfill_mode, expected);
            assert_eq!(
                params.hashprune_backfill_ablation,
                PiPNNHashPruneBackfillAblation::None
            );
            assert_eq!(params.hashprune_backfill_budget, 0);
            assert_eq!(params.hashprune_backfill_alpha, 1.0);
            assert!(params.validate_hashprune_backfill_config().is_ok());
        }
    }

    #[test]
    fn hashprune_named_rhalf_cli_accepts_quota_and_soft_tier_alpha() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "rhalf-soft-tier-q3",
            "--hashprune-rhalf-hash-quota",
            "2",
            "--hashprune-soft-tier-priority-alpha",
            "1.4",
            "--hashprune-rhalf-max-backfill-per-row",
            "3",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(params.hashprune_rhalf_hash_quota, Some(2));
        assert_eq!(params.hashprune_soft_tier_priority_alpha, Some(1.4));
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, Some(3));
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_named_rhalf_cli_tuning_defaults_to_none() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "rhalf-soft-tier-q3",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(params.hashprune_rhalf_hash_quota, None);
        assert_eq!(params.hashprune_soft_tier_priority_alpha, None);
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, None);
    }

    #[test]
    fn hashprune_production_cli_defaults_to_soft_tier_cap3_alpha1p02() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        apply_prune_backend_env_overrides_with(&mut params, |_| None, |_| false).unwrap();

        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
        );
        assert_eq!(params.hashprune_soft_tier_priority_alpha, Some(1.02));
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, Some(3));
    }

    #[test]
    fn hashprune_explicit_non_soft_mode_does_not_inherit_soft_tier_tuning() {
        for (raw, expected) in [
            ("off", PiPNNHashPruneBackfillMode::Off),
            (
                "rhalf-hard-tier-q3",
                PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
            ),
        ] {
            let args = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--hashprune-backfill-mode",
                raw,
            ])
            .unwrap();
            let mut params = PiPNNParams::default();
            apply_gpu_subtree_args(&mut params, &args);
            apply_prune_backend_env_overrides_with(
                &mut params,
                |_| None,
                |flag| flag == "--hashprune-backfill-mode",
            )
            .unwrap();

            assert_eq!(params.hashprune_backfill_mode, expected);
            assert_eq!(params.hashprune_soft_tier_priority_alpha, None);
            assert_eq!(params.hashprune_rhalf_max_backfill_per_row, None);
        }
    }

    #[test]
    fn hashprune_env_off_overrides_the_production_cli_default() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| (name == "SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE").then(|| "off".to_string()),
            |_| false,
        )
        .unwrap();

        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::Off
        );
        assert_eq!(params.hashprune_soft_tier_priority_alpha, None);
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, None);
    }

    #[test]
    fn hashprune_explicit_soft_tier_tuning_overrides_env_and_production_defaults() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "rhalf-soft-tier-q3",
            "--hashprune-soft-tier-priority-alpha",
            "1.4",
            "--hashprune-rhalf-max-backfill-per-row",
            "6",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE" => Some("off".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA" => Some("1.8".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW" => Some("2".to_string()),
                _ => None,
            },
            |flag| {
                matches!(
                    flag,
                    "--hashprune-backfill-mode"
                        | "--hashprune-soft-tier-priority-alpha"
                        | "--hashprune-rhalf-max-backfill-per-row"
                )
            },
        )
        .unwrap();

        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
        );
        assert_eq!(params.hashprune_soft_tier_priority_alpha, Some(1.4));
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, Some(6));
    }

    #[test]
    fn hashprune_library_params_default_remains_off() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::Off
        );
        assert_eq!(params.hashprune_soft_tier_priority_alpha, None);
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, None);
    }

    #[test]
    fn hashprune_named_rhalf_cli_rejects_invalid_row_caps_at_validation() {
        for (mode, cap) in [
            ("rhalf-soft-tier-q3", "0"),
            ("rhalf-soft-tier-q3", "33"),
            ("off", "2"),
        ] {
            let args = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--hashprune-backfill-mode",
                mode,
                "--hashprune-rhalf-max-backfill-per-row",
                cap,
            ])
            .unwrap();
            let mut params = PiPNNParams::default();
            apply_gpu_subtree_args(&mut params, &args);
            assert!(params.validate_hashprune_backfill_config().is_err());
        }
    }

    #[test]
    fn hashprune_named_rhalf_tuning_env_respects_cli_precedence() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_RHALF_HASH_QUOTA" => Some("2".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA" => Some("1.4".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW" => Some("3".to_string()),
                _ => None,
            },
            |_| false,
        )
        .unwrap();
        assert_eq!(params.hashprune_rhalf_hash_quota, Some(2));
        assert_eq!(params.hashprune_soft_tier_priority_alpha, Some(1.4));
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, Some(3));

        params.hashprune_rhalf_hash_quota = Some(3);
        params.hashprune_soft_tier_priority_alpha = Some(1.2);
        params.hashprune_rhalf_max_backfill_per_row = Some(4);
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_RHALF_HASH_QUOTA" => Some("1".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA" => Some("2.0".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW" => Some("2".to_string()),
                _ => None,
            },
            |flag| {
                matches!(
                    flag,
                    "--hashprune-rhalf-hash-quota"
                        | "--hashprune-soft-tier-priority-alpha"
                        | "--hashprune-rhalf-max-backfill-per-row"
                )
            },
        )
        .unwrap();
        assert_eq!(params.hashprune_rhalf_hash_quota, Some(3));
        assert_eq!(params.hashprune_soft_tier_priority_alpha, Some(1.2));
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, Some(4));
    }

    #[test]
    fn hashprune_diagnostics_cli_propagates_the_output_directory() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--hashprune-backfill-mode",
            "rhalf-soft-tier-q3",
            "--hashprune-backfill-diagnostics-prefix",
            "/tmp/hashprune-diag",
        ])
        .unwrap();
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.hashprune_backfill_diagnostics_prefix,
            Some(PathBuf::from("/tmp/hashprune-diag"))
        );
        assert!(params.validate_hashprune_policies().is_ok());
    }

    #[test]
    fn hashprune_backfill_env_applies_only_when_cli_is_absent() {
        let mut params = PiPNNParams::default();
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE" => Some("distance-gated".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_BUDGET" => Some("4".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ALPHA" => Some("1.2".to_string()),
                _ => None,
            },
            |_| false,
        )
        .unwrap();
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::DistanceGated
        );
        assert_eq!(params.hashprune_backfill_budget, 4);
        assert_eq!(params.hashprune_backfill_alpha, 1.2);

        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGated;
        params.hashprune_backfill_budget = 8;
        params.hashprune_backfill_alpha = 1.1;
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE" => Some("off".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_BUDGET" => Some("0".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ALPHA" => Some("1.0".to_string()),
                _ => None,
            },
            |flag| {
                matches!(
                    flag,
                    "--hashprune-backfill-mode"
                        | "--hashprune-backfill-budget"
                        | "--hashprune-backfill-alpha"
                )
            },
        )
        .unwrap();
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::DistanceGated
        );
        assert_eq!(params.hashprune_backfill_budget, 8);
        assert_eq!(params.hashprune_backfill_alpha, 1.1);
    }

    #[test]
    fn hashprune_tiered_rhalf_ablation_env_respects_cli_precedence() {
        let mut params = PiPNNParams::default();
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE" => {
                    Some("distance-gated-tiered-rhalf".to_string())
                }
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ABLATION" => Some("quota-2".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_BUDGET" => Some("0".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ALPHA" => Some("1.1".to_string()),
                _ => None,
            },
            |_| false,
        )
        .unwrap();
        assert_eq!(
            params.hashprune_backfill_ablation,
            PiPNNHashPruneBackfillAblation::Quota2
        );
        assert!(params.validate_hashprune_backfill_config().is_ok());

        params.hashprune_backfill_ablation = PiPNNHashPruneBackfillAblation::NoVeto;
        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| (name == "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ABLATION").then(|| "none".to_string()),
            |flag| flag == "--hashprune-backfill-ablation",
        )
        .unwrap();
        assert_eq!(
            params.hashprune_backfill_ablation,
            PiPNNHashPruneBackfillAblation::NoVeto
        );
    }

    #[test]
    fn hashprune_backfill_cli_rejects_noncanonical_numeric_values_at_validation() {
        for raw_alpha in ["NaN", "0.99", "inf"] {
            let args = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--hashprune-backfill-mode",
                "distance-gated",
                "--hashprune-backfill-budget",
                "8",
                "--hashprune-backfill-alpha",
                raw_alpha,
            ])
            .unwrap();
            let mut params = PiPNNParams::default();
            apply_gpu_subtree_args(&mut params, &args);
            assert!(params.validate_hashprune_backfill_config().is_err());
        }
    }

    #[test]
    fn hashprune_evidence_env_overrides_params_when_cli_flags_absent() {
        let mut params = PiPNNParams::default();

        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_EVIDENCE_MODE" => Some("report".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_EVIDENCE_OVERFLOW_BUDGET" => Some("0".to_string()),
                _ => None,
            },
            |_| false,
        )
        .unwrap();

        assert_eq!(
            params.hashprune_evidence_mode,
            PiPNNHashPruneEvidenceMode::Report
        );
        assert_eq!(params.hashprune_evidence_overflow_budget, 0);
    }

    #[test]
    fn hashprune_evidence_env_keeps_explicit_cli_values() {
        let mut params = PiPNNParams::default();
        params.hashprune_evidence_mode = PiPNNHashPruneEvidenceMode::Enforce;
        params.hashprune_evidence_overflow_budget = 1;

        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_HASHPRUNE_EVIDENCE_MODE" => Some("off".to_string()),
                "SHEAR_PIPNN_HASHPRUNE_EVIDENCE_OVERFLOW_BUDGET" => Some("0".to_string()),
                _ => None,
            },
            |flag| {
                matches!(
                    flag,
                    "--hashprune-evidence-mode" | "--hashprune-evidence-overflow-budget"
                )
            },
        )
        .unwrap();

        assert_eq!(
            params.hashprune_evidence_mode,
            PiPNNHashPruneEvidenceMode::Enforce
        );
        assert_eq!(params.hashprune_evidence_overflow_budget, 1);
    }

    #[test]
    fn prune_backend_env_overrides_params_when_cli_flag_absent() {
        let mut params = PiPNNParams::default();
        params.lrpf_target_degree = 32;
        params.lrpf_long_edge_quota = 9;

        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| match name {
                "SHEAR_PIPNN_PRUNE_BACKEND" => Some("lrpf".to_string()),
                "SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS" => Some("1".to_string()),
                "SHEAR_PIPNN_LRPF_PRE_SPILL_WIDTH_MULTIPLIER" => Some("3.0".to_string()),
                "SHEAR_PIPNN_LRPF_PROTECTED_CORE" => Some("2".to_string()),
                "SHEAR_PIPNN_LRPF_TARGET_DEGREE" => Some("64".to_string()),
                "SHEAR_PIPNN_LRPF_LONG_EDGE_QUOTA" => Some("0".to_string()),
                _ => None,
            },
            |_| false,
        )
        .unwrap();

        assert_eq!(params.prune_backend, PiPNNPruneBackend::LRPF);
        assert!(params.lrpf_pre_spill_compress_enable);
        assert_eq!(params.lrpf_pre_spill_width_multiplier, 3.0);
        assert_eq!(params.lrpf_protected_core, 2);
        assert_eq!(params.lrpf_target_degree, 64);
        assert_eq!(params.lrpf_long_edge_quota, 0);
    }

    #[test]
    fn prune_backend_env_keeps_explicit_cli_backend() {
        let mut params = PiPNNParams::default();

        apply_prune_backend_env_overrides_with(
            &mut params,
            |name| (name == "SHEAR_PIPNN_PRUNE_BACKEND").then(|| "lrpf".to_string()),
            |flag| flag == "--prune-backend",
        )
        .unwrap();

        assert_eq!(params.prune_backend, PiPNNPruneBackend::HashPrune);
    }

    #[test]
    fn args_default_to_cpu_backend_and_device_zero() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
        ]);

        assert_eq!(args.pipnn_backend, "cpu");
        assert_eq!(args.pipnn_gpu_device, 0);
        assert_eq!(args.partitioner, PiPNNPartitionerKind::Rbc);
        assert_eq!(args.kmeans_iterations, 4);
        assert_eq!(args.kmeans_oversplit_factor, 2);
        assert_eq!(args.lrpf_target_degree, 64);
        assert_eq!(args.lrpf_long_edge_quota, 0);
    }

    #[test]
    fn args_accept_kmeans_partitioner_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "kmeans",
            "--kmeans-iterations",
            "6",
            "--kmeans-oversplit-factor",
            "3",
        ]);

        assert_eq!(args.partitioner, PiPNNPartitionerKind::KMeans);
        assert_eq!(args.kmeans_iterations, 6);
        assert_eq!(args.kmeans_oversplit_factor, 3);
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_gpu_subtree_partitioner_with_cuda_oom_l2() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
        ]);

        assert_eq!(args.partitioner, PiPNNPartitionerKind::GpuSubtree);
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_default_gpu_subtree_to_cpu_d0_and_persistent_wmma_workers() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_root_assignment_backend,
            PiPNNComputeBackendKind::Cpu
        );
        assert_eq!(
            args.gpu_subtree_assignment_backend,
            PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
        );
        assert_eq!(
            args.gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn help_hides_retired_and_diagnostic_gpu_subtree_flags() {
        let help = Args::command().render_long_help().to_string();
        let hidden_flags = [
            "--oom-strict-io",
            "--oom-raw-vector-cache-gb",
            "--gpu-subtree-d0-partition-mode",
            "--gpu-subtree-d0-tagore-partitions",
            "--gpu-subtree-vector-staging-mode",
            "--gpu-subtree-child-schedule",
            "--gpu-subtree-root-assignment-backend",
            "--gpu-subtree-assignment-dco-mode",
            "--gpu-subtree-assignment-dco-proj-dim",
            "--gpu-subtree-leaf-dco-mode",
            "--gpu-subtree-leaf-dco-proj-dim",
            "--gpu-subtree-leaf-dco-ddc-res-pca-matrix",
            "--gpu-subtree-leaf-dco-ddc-res-projected-input",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-d0-capacity-high-ratio",
        ];

        for flag in hidden_flags {
            assert!(
                !help.contains(flag),
                "retired or diagnostic flag leaked into help: {flag}\n{help}"
            );
        }

        assert!(help.contains("--gpu-subtree-assignment-backend"));
        assert!(help.contains("--gpu-subtree-leaf-topk-backend"));
        assert!(help.contains("--oom-io-mode"));
        assert!(help.contains("persistent-wmma-tf32"));
        assert!(!help.contains("persistent-cuvs-bruteforce-fp32"));
        assert!(!help.contains("cublas-gemm"));
    }

    #[test]
    fn args_reject_removed_gpu_subtree_legacy_flags() {
        let base = [
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ];

        for removed_flag in [
            "--gpu-subtree-output",
            "--gpu-subtree-overlap-probe",
            "--gpu-subtree-persistent-device-queue",
            "--no-gpu-subtree-persistent-device-queue",
        ] {
            let mut argv = base.to_vec();
            argv.push(removed_flag);
            if removed_flag == "--gpu-subtree-output" {
                argv.push("legacy-leaves");
            }
            assert!(
                Args::try_parse_from(argv).is_err(),
                "{removed_flag} should be removed from the persistent-only GPU subtree CLI"
            );
        }
    }

    #[test]
    fn args_accept_gpu_subtree_persistent_leaf_tile_rows() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-persistent-leaf-tile-rows",
            "256",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(args.gpu_subtree_persistent_leaf_tile_rows, 256);
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_and_propagate_low_dim_sq8_execution_modes() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-persistent-large-ticket-batch",
            "8",
            "--gpu-subtree-sq8-assignment-kernel",
            "wmma",
            "--gpu-subtree-sq8-d0-assignment-kernel",
            "wmma-batch",
            "--gpu-subtree-sq8-leaf-executor",
            "wmma",
            "--gpu-subtree-sq8-leaf-mode",
            "d96-k2",
            "--gpu-subtree-persistent-scheduler-mode",
            "adaptive-idle",
            "--gpu-subtree-persistent-idle-backoff-ns",
            "700",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_persistent_large_ticket_batch,
            PiPNNGpuSubtreePersistentLargeTicketBatch::Eight
        );
        assert_eq!(
            params.gpu_subtree_sq8_assignment_kernel,
            PiPNNGpuSubtreeSq8AssignmentKernel::Wmma
        );
        assert_eq!(
            params.gpu_subtree_sq8_d0_assignment_kernel,
            PiPNNGpuSubtreeSq8D0AssignmentKernel::WmmaBatch
        );
        assert_eq!(
            params.gpu_subtree_sq8_leaf_executor,
            PiPNNGpuSubtreeSq8LeafExecutor::Wmma
        );
        assert_eq!(
            params.gpu_subtree_sq8_leaf_mode,
            PiPNNGpuSubtreeSq8LeafMode::D96K2
        );
        assert_eq!(
            params.gpu_subtree_persistent_scheduler_mode,
            PiPNNGpuSubtreePersistentSchedulerMode::AdaptiveIdle
        );
        assert_eq!(params.gpu_subtree_persistent_idle_backoff_ns, 700);
    }

    #[test]
    fn tessa_capture_cli_requires_and_propagates_the_frozen_a3_contract() {
        fn a3_args() -> Args {
            Args::parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/tessa-index",
                "--partitioner",
                "gpu-subtree",
                "--gpu-subtree-d0-partition-mode",
                "gpu-kmeans-coarse",
                "--gpu-subtree-d0-capacity-enable",
                "--gpu-subtree-sq-staging-mode",
                "gpu-d0",
                "--gpu-subtree-sq-wmma-mode",
                "build",
                "--gpu-subtree-sq-wmma-sidecar-prefix",
                "/tmp/tessa-sidecar",
                "--gpu-subtree-sq-wmma-sidecar-mode",
                "rebuild",
                "--gpu-subtree-execution-mode",
                "fused-persistent",
                "--gpu-subtree-sq8-assignment-kernel",
                "wmma",
                "--gpu-subtree-sq8-leaf-executor",
                "wmma",
                "--gpu-subtree-persistent-large-ticket-batch",
                "8",
                "--gpu-subtree-tessa-capture-dir",
                "/tmp/tessa-capture",
                "--pipnn-backend",
                "cuda",
                "--oom-enable",
                "--oom-temp-dir",
                "/tmp/tessa-oom",
            ])
        }

        let args = a3_args();
        validate_args(&args).expect("the frozen A3 capture contract should validate");
        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.gpu_subtree_tessa_capture_dir,
            Some(PathBuf::from("/tmp/tessa-capture"))
        );

        let mut invalid = a3_args();
        invalid.gpu_subtree_sq8_assignment_kernel = PiPNNGpuSubtreeSq8AssignmentKernel::WarpDp4a;
        assert!(validate_args(&invalid).is_err());

        let mut invalid = a3_args();
        invalid.gpu_subtree_sq8_leaf_executor = PiPNNGpuSubtreeSq8LeafExecutor::WarpDp4a;
        assert!(validate_args(&invalid).is_err());

        let mut invalid = a3_args();
        invalid.gpu_subtree_persistent_large_ticket_batch =
            PiPNNGpuSubtreePersistentLargeTicketBatch::One;
        assert!(validate_args(&invalid).is_err());

        let mut invalid = a3_args();
        invalid.gpu_subtree_tessa_capture_dir = Some(PathBuf::from("/tmp/tessa-index/capture"));
        assert!(validate_args(&invalid).is_err());

        let mut invalid = a3_args();
        invalid.gpu_subtree_tessa_capture_dir = Some(PathBuf::from("/tmp/tessa-oom/capture"));
        assert!(validate_args(&invalid).is_err());
    }

    #[test]
    fn validate_args_accepts_gpu_subtree_d0_tagore_partition_mode_for_root_stop_after_d00() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "tagore-kmeans-topk",
            "--gpu-subtree-d0-tagore-partitions",
            "32",
            "--gpu-subtree-d0-tagore-k",
            "3",
            "--gpu-subtree-d0-tagore-sample-fraction",
            "0.01",
            "--root-stop-after-d00",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_d0_partition_mode,
            PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_accepts_gpu_subtree_d0_tagore_topk_flag_for_root_stop_after_d00() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "tagore-kmeans-topk",
            "--gpu-subtree-d0-tagore-partitions",
            "32",
            "--gpu-subtree-d0-tagore-k",
            "6",
            "--root-stop-after-d00",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert!(
            args.is_ok(),
            "topk D0 diagnostic args should parse: {args:?}"
        );
        validate_args(&args.unwrap()).unwrap();
    }

    #[test]
    fn validate_args_accepts_gpu_subtree_d0_tagore_partition_mode_for_full_build_with_capacity() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "tagore-kmeans-topk",
            "--gpu-subtree-d0-tagore-partitions",
            "32",
            "--gpu-subtree-d0-tagore-k",
            "3",
            "--gpu-subtree-d0-tagore-sample-fraction",
            "0.01",
            "--gpu-subtree-d0-capacity-enable",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn sq4_budget_default_routes_through_capacity_planner() {
        let args=Args::parse_from(["build_shear_index","--data-type","float","--dist-fn","l2",
            "--data-path","/tmp/data.bin","--index-path-prefix","/tmp/index","--partitioner","gpu-subtree",
            "--oom-enable","--pipnn-backend","cuda","--gpu-subtree-sq-wmma-mode","build",
            "--gpu-subtree-sq-bits","4","--gpu-subtree-sq-native-int4","--gpu-subtree-sq4-plus"]);
        validate_args(&args).unwrap();
        let mut p=PiPNNParams::default();apply_gpu_subtree_args(&mut p,&args);
        assert_eq!(p.gpu_subtree_d0_partition_mode,PiPNNGpuSubtreeD0PartitionMode::BudgetKMeansTopK);
        assert!(p.gpu_subtree_d0_capacity_enable);assert!(p.gpu_subtree_d0_capacity_root_leader_cap>64);
        assert_eq!(super::effective_gpu_subtree_d0_capacity_ratios(&args), (0.50,0.08,0.08));
        assert_eq!((p.gpu_subtree_d0_capacity_high_ratio,p.gpu_subtree_d0_capacity_target_ratio,p.gpu_subtree_d0_capacity_low_ratio),(0.50,0.08,0.08));
        let mut explicit=args;
        explicit.gpu_subtree_d0_capacity_high_ratio=Some(0.60);
        explicit.gpu_subtree_d0_capacity_target_ratio=Some(0.50);
        explicit.gpu_subtree_d0_capacity_low_ratio=Some(0.15);
        validate_args(&explicit).unwrap();
        apply_gpu_subtree_args(&mut p,&explicit);
        assert_eq!((p.gpu_subtree_d0_capacity_high_ratio,p.gpu_subtree_d0_capacity_target_ratio,p.gpu_subtree_d0_capacity_low_ratio),(0.60,0.50,0.15));
    }

    #[test]
    fn validate_args_accepts_gpu_kmeans_coarse_with_sq_gpu_d0_staging() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "gpu-kmeans-coarse",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-staging-upload-shard-rows",
            "262144",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-execution-mode",
            "fused-persistent",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d0_partition_mode,
            PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
        );
        assert_eq!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0
        );
        assert_eq!(
            params.gpu_subtree_execution_mode,
            PiPNNGpuSubtreeExecutionMode::FusedPersistent
        );
    }

    #[test]
    fn validate_args_accepts_gpu_kmeans_coarse_staging_with_raw_fp32_d1_diagnostic() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "gpu-kmeans-coarse",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-staging-upload-shard-rows",
            "262144",
            "--gpu-subtree-sq-staging-raw-d1-diagnostic",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-assignment-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-execution-mode",
            "fused-persistent",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ])
        .expect("raw-D1 diagnostic flag should parse");

        validate_args(&args)
            .expect("GPU-D0 staging should allow an explicit raw-FP32 D1 diagnostic");
        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert!(params.gpu_subtree_sq_staging_raw_d1_diagnostic);
    }

    #[test]
    fn validate_args_accepts_d1_root_kmeans_coarse_with_sq_gpu_d0_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "gpu-kmeans-coarse",
            "--gpu-subtree-d1-root-partition-mode",
            "kmeans-coarse",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-execution-mode",
            "fused-persistent",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d1_root_partition_mode,
            PiPNNGpuSubtreeD1RootPartitionMode::KMeansCoarse
        );
    }

    #[test]
    fn validate_args_accepts_gpu_kmeans_coarse_with_sq_gpu_d0_shard_local_d1() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "gpu-kmeans-coarse",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0-shard-local-d1",
            "--gpu-subtree-sq-staging-upload-shard-rows",
            "262144",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "off",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-execution-mode",
            "fused-persistent",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d0_partition_mode,
            PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
        );
        assert_eq!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::Off
        );
    }

    #[test]
    fn validate_args_accepts_gpu_kmeans_coarse_with_sq_gpu_d0_cluster_batch_d1() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "gpu-kmeans-coarse",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0-cluster-batch-d1",
            "--gpu-subtree-sq-staging-upload-shard-rows",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "off",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-execution-mode",
            "fused-persistent",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d0_partition_mode,
            PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
        );
        assert_eq!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::Off
        );
    }

    #[test]
    fn args_accept_gpu_subtree_vector_staging_mode_and_budget() {
        let default_args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        assert_eq!(
            default_args.gpu_subtree_vector_staging_mode,
            PiPNNGpuSubtreeVectorStagingMode::Off
        );
        assert_eq!(default_args.gpu_subtree_vector_staging_budget_bytes, 0);
        assert_eq!(default_args.gpu_subtree_vector_staging_window_children, 8);
        assert_eq!(
            default_args.gpu_subtree_child_schedule,
            PiPNNGpuSubtreeChildScheduleMode::Natural
        );

        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-vector-staging-mode",
            "rolling-cache",
            "--gpu-subtree-vector-staging-budget-bytes",
            "1048576",
            "--gpu-subtree-vector-staging-window-children",
            "3",
            "--gpu-subtree-child-schedule",
            "overlap-greedy",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_vector_staging_mode,
            PiPNNGpuSubtreeVectorStagingMode::RollingCache
        );
        assert_eq!(params.gpu_subtree_vector_staging_budget_bytes, 1_048_576);
        assert_eq!(params.gpu_subtree_vector_staging_window_children, 3);
        assert_eq!(
            params.gpu_subtree_child_schedule_mode,
            PiPNNGpuSubtreeChildScheduleMode::OverlapGreedy
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_tagore_mode_even_without_gpu_subtree_partitioner() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "rbc",
            "--gpu-subtree-d0-partition-mode",
            "tagore-kmeans-topk",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("requires --partitioner gpu-subtree"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_tagore_full_build_with_capacity_disabled() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-d0-partition-mode",
            "tagore-kmeans-topk",
            "--gpu-subtree-d0-tagore-k",
            "5",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
            "--gpu-subtree-d0-capacity-disable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("gpu-subtree-d0-capacity-enable"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn args_accept_gpu_subtree_d0_capacity_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--pipeline-compute-mode",
            "cuda",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-d0-capacity-high-ratio",
            "0.82",
            "--gpu-subtree-d0-capacity-target-ratio",
            "0.70",
            "--gpu-subtree-d0-capacity-low-ratio",
            "0.15",
            "--gpu-subtree-d0-capacity-min-efficient-rows",
            "65536",
            "--gpu-subtree-d0-capacity-quality-min-root-leaders",
            "128",
            "--gpu-subtree-d0-capacity-root-leader-cap",
            "4096",
            "--gpu-subtree-d0-capacity-split-recursion-limit",
            "6",
            "--gpu-subtree-d1plus-lazy-split-mode",
            "off",
        ]);
        assert!(args.gpu_subtree_d0_capacity_enable);
        assert_eq!(args.gpu_subtree_d0_capacity_high_ratio, Some(0.82));
        assert_eq!(args.gpu_subtree_d0_capacity_target_ratio, Some(0.70));
        assert_eq!(args.gpu_subtree_d0_capacity_low_ratio, Some(0.15));
        assert_eq!(args.gpu_subtree_d0_capacity_min_efficient_rows, 65_536);
        assert_eq!(args.gpu_subtree_d0_capacity_quality_min_root_leaders, 128);
        assert_eq!(args.gpu_subtree_d0_capacity_root_leader_cap, 4096);
        assert_eq!(args.gpu_subtree_d0_capacity_split_recursion_limit, 6);
        assert_eq!(
            args.gpu_subtree_d1plus_lazy_split_mode,
            PiPNNGpuSubtreeD1PlusLazySplitMode::Off
        );

        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert!(params.gpu_subtree_d0_capacity_enable);
        assert_eq!(params.gpu_subtree_d0_capacity_high_ratio, 0.82);
        assert_eq!(params.gpu_subtree_d0_capacity_target_ratio, 0.70);
        assert_eq!(params.gpu_subtree_d0_capacity_low_ratio, 0.15);
        assert_eq!(params.gpu_subtree_d0_capacity_min_efficient_rows, 65_536);
        assert_eq!(params.gpu_subtree_d0_capacity_quality_min_root_leaders, 128);
        assert_eq!(params.gpu_subtree_d0_capacity_root_leader_cap, 4096);
        assert_eq!(params.gpu_subtree_d0_capacity_split_recursion_limit, 6);
        assert_eq!(
            params.gpu_subtree_d1plus_lazy_split_mode,
            PiPNNGpuSubtreeD1PlusLazySplitMode::Off
        );
    }

    #[test]
    fn args_accept_gpu_subtree_d0_throughput_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--pipeline-compute-mode",
            "cuda",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-d0-throughput-model",
            "report",
            "--gpu-subtree-d0-throughput-persistent-growth-limit",
            "1.25",
            "--gpu-subtree-d0-throughput-preload-visible-limit-ms",
            "7000",
            "--gpu-subtree-d0-throughput-min-ratio",
            "0.60",
            "--gpu-subtree-d0-throughput-calibration-json",
            "/tmp/wiki35m-d0-calibration.json",
            "--gpu-subtree-d0-throughput-child-p95-guardrail-rows",
            "4000000",
        ]);
        assert_eq!(
            args.gpu_subtree_d0_throughput_model,
            PiPNNGpuSubtreeD0ThroughputModelMode::Report
        );
        assert_eq!(args.gpu_subtree_d0_throughput_persistent_growth_limit, 1.25);
        assert_eq!(
            args.gpu_subtree_d0_throughput_preload_visible_limit_ms,
            7000
        );
        assert_eq!(args.gpu_subtree_d0_throughput_min_ratio, 0.60);
        assert_eq!(
            args.gpu_subtree_d0_throughput_calibration_json,
            Some(PathBuf::from("/tmp/wiki35m-d0-calibration.json"))
        );
        assert_eq!(
            args.gpu_subtree_d0_throughput_child_p95_guardrail_rows,
            4_000_000
        );

        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.gpu_subtree_d0_throughput_model,
            PiPNNGpuSubtreeD0ThroughputModelMode::Report
        );
        assert_eq!(
            params.gpu_subtree_d0_throughput_persistent_growth_limit,
            1.25
        );
        assert_eq!(
            params.gpu_subtree_d0_throughput_preload_visible_limit_ms,
            7000
        );
        assert_eq!(params.gpu_subtree_d0_throughput_min_ratio, 0.60);
        assert_eq!(
            params.gpu_subtree_d0_throughput_calibration_path,
            Some(PathBuf::from("/tmp/wiki35m-d0-calibration.json"))
        );
        assert_eq!(
            params.gpu_subtree_d0_throughput_child_p95_guardrail_rows,
            4_000_000
        );
    }

    #[test]
    fn gpu_subtree_hardware_policy_args() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--gpu-subtree-persistent-launch-policy",
            "enforce",
            "--gpu-subtree-hbm-policy",
            "off",
            "--gpu-subtree-launch-calibration-cache",
            "/tmp/shear-cache",
            "--gpu-subtree-launch-calibration-refresh",
        ]);

        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_persistent_launch_policy,
            PiPNNGpuSubtreePolicy::Enforce
        );
        assert_eq!(params.gpu_subtree_hbm_policy, PiPNNGpuSubtreePolicy::Off);
        assert_eq!(
            params.gpu_subtree_launch_calibration_cache,
            Some(PathBuf::from("/tmp/shear-cache"))
        );
        assert!(params.gpu_subtree_launch_calibration_refresh);
    }

    #[test]
    fn gpu_subtree_hardware_policy_args_defaults_are_applied() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
        ]);

        assert_eq!(
            args.gpu_subtree_persistent_launch_policy,
            PiPNNGpuSubtreePolicy::Report
        );
        assert_eq!(args.gpu_subtree_hbm_policy, PiPNNGpuSubtreePolicy::Report);
        assert_eq!(args.gpu_subtree_launch_calibration_cache, None);
        assert!(!args.gpu_subtree_launch_calibration_refresh);

        let mut params = PiPNNParams::default();
        apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_persistent_launch_policy,
            PiPNNGpuSubtreePolicy::Report
        );
        assert_eq!(params.gpu_subtree_hbm_policy, PiPNNGpuSubtreePolicy::Report);
        assert_eq!(params.gpu_subtree_launch_calibration_cache, None);
        assert!(!params.gpu_subtree_launch_calibration_refresh);
    }

    #[test]
    fn gpu_subtree_hardware_policy_args_reject_uninstalled_persistent_launch_enforcement() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--gpu-subtree-persistent-launch-policy",
            "enforce",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("validated launch cache support is not installed"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn gpu_subtree_hardware_policy_args_reject_uninstalled_hbm_enforcement() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--gpu-subtree-hbm-policy",
            "enforce",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("automatic HBM enforcement is not installed"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn args_default_gpu_subtree_d0_capacity_ratios_are_strict() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
        ]);

        assert!(!args.gpu_subtree_d0_capacity_enable);
        assert!(!args.gpu_subtree_d0_capacity_disable);
        assert!(effective_gpu_subtree_d0_capacity_enabled(&args));
        assert_eq!(super::effective_gpu_subtree_d0_capacity_ratios(&args), (0.60, 0.50, 0.15));

        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert!(params.gpu_subtree_d0_capacity_enable);
        assert_eq!(params.gpu_subtree_d0_capacity_high_ratio, 0.60);
        assert_eq!(params.gpu_subtree_d0_capacity_target_ratio, 0.50);
        assert_eq!(params.gpu_subtree_d0_capacity_low_ratio, 0.15);
    }

    #[test]
    fn args_scope_default_d0_capacity_to_gpu_subtree_oom() {
        let gpu_subtree_without_oom = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
        ]);
        assert!(!effective_gpu_subtree_d0_capacity_enabled(
            &gpu_subtree_without_oom
        ));

        let rbc_with_oom = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "rbc",
            "--oom-enable",
        ]);
        assert!(!effective_gpu_subtree_d0_capacity_enabled(&rbc_with_oom));
    }

    #[test]
    fn args_explicit_d0_capacity_disable_overrides_production_default() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-capacity-disable",
        ]);

        assert!(!effective_gpu_subtree_d0_capacity_enabled(&args));
        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert!(!params.gpu_subtree_d0_capacity_enable);
    }

    #[test]
    fn args_reject_conflicting_d0_capacity_overrides() {
        let err = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-d0-capacity-disable",
        ])
        .unwrap_err();

        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn explicit_max_leaders_takes_precedence_for_d0_capacity_root_leaders() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-d0-capacity-quality-min-root-leaders",
            "64",
            "--max-leaders",
            "5000",
        ]);
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);
        super::apply_user_leader_precedence(&mut params, &args, true);

        assert_eq!(params.max_leaders, 5000);
        assert_eq!(
            params.gpu_subtree_d0_capacity_quality_min_root_leaders,
            5000
        );
        assert!(params.gpu_subtree_d0_capacity_root_leader_cap >= 5000);
    }

    #[test]
    fn default_max_leaders_does_not_override_d0_capacity_root_leaders() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-d0-capacity-enable",
        ]);
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);
        super::apply_user_leader_precedence(&mut params, &args, false);

        assert_eq!(
            params.gpu_subtree_d0_capacity_quality_min_root_leaders,
            args.gpu_subtree_d0_capacity_quality_min_root_leaders
        );
    }

    fn gpu_subtree_d0_capacity_args(partitioner: &str, oom_enable: bool, extra: &[&str]) -> Args {
        let mut argv = vec![
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--pipnn-backend",
            "cuda",
            "--partitioner",
            partitioner,
            "--gpu-subtree-d0-capacity-enable",
        ];
        if oom_enable {
            argv.push("--oom-enable");
        }
        argv.extend(extra.iter().copied());
        Args::parse_from(argv)
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_capacity_without_gpu_subtree_partitioner() {
        let args = gpu_subtree_d0_capacity_args("rbc", true, &[]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("requires --partitioner gpu-subtree"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_capacity_without_oom_enable() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-d0-capacity-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("requires --oom-enable"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_capacity_target_above_high_ratio() {
        let args = gpu_subtree_d0_capacity_args(
            "gpu-subtree",
            true,
            &[
                "--gpu-subtree-d0-capacity-high-ratio",
                "0.70",
                "--gpu-subtree-d0-capacity-target-ratio",
                "0.80",
            ],
        );

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("target ratio must be <= high ratio"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_capacity_zero_split_recursion_limit() {
        let args = gpu_subtree_d0_capacity_args(
            "gpu-subtree",
            true,
            &["--gpu-subtree-d0-capacity-split-recursion-limit", "0"],
        );

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("split recursion limit must be > 0"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_d0_throughput_without_capacity_model() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-d0-capacity-disable",
            "--gpu-subtree-d0-throughput-model",
            "report",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(
            format!("{err:?}").contains("gpu_subtree_d0_throughput_model"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_d0_throughput_enforce_without_calibration() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "data.fbin",
            "--index-path-prefix",
            "out/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-d0-throughput-model",
            "enforce",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(
            format!("{err:?}").contains("calibration"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_invalid_d0_throughput_ranges() {
        for extra in [
            ["--gpu-subtree-d0-throughput-persistent-growth-limit", "0.9"],
            ["--gpu-subtree-d0-throughput-preload-visible-limit-ms", "0"],
            ["--gpu-subtree-d0-throughput-min-ratio", "1.5"],
        ] {
            let mut argv = vec![
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "data.fbin",
                "--index-path-prefix",
                "out/index",
                "--partitioner",
                "gpu-subtree",
                "--oom-enable",
                "--pipnn-backend",
                "cuda",
                "--gpu-subtree-d0-capacity-enable",
                "--gpu-subtree-d0-throughput-model",
                "report",
            ];
            argv.extend(extra);
            let args = Args::parse_from(argv);
            assert!(validate_args(&args).is_err());
        }
    }

    #[test]
    fn apply_gpu_subtree_args_propagates_persistent_leaf_tile_rows_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-persistent-leaf-tile-rows",
            "256",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(params.partitioner, PiPNNPartitionerKind::GpuSubtree);
        assert_eq!(params.gpu_subtree_persistent_leaf_tile_rows, 256);
    }

    #[test]
    fn args_accept_gpu_subtree_leaf_topk_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "cublas-gemm",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::CublasGemm
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_gpu_subtree_execution_mode_two_stage() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-execution-mode",
            "two-stage",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_execution_mode,
            PiPNNGpuSubtreeExecutionMode::TwoStage
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_rejects_two_stage_with_cublas_gemm_leaf_topk_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-execution-mode",
            "two-stage",
            "--gpu-subtree-leaf-topk-backend",
            "cublas-gemm",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string().contains("two-stage GPU-subtree execution"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn args_accept_persistent_tiled_gemm_leaf_topk_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::PersistentTiledGemmFp32
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_persistent_cuvs_bruteforce_leaf_topk_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "cuvs-bruteforce",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::CuvsBruteforce
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_persistent_wmma_tf32_leaf_topk_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_gpu_subtree_assignment_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_assignment_backend,
            PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn args_accept_gpu_subtree_root_assignment_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-root-assignment-backend",
            "cuda",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        assert_eq!(
            args.gpu_subtree_root_assignment_backend,
            PiPNNComputeBackendKind::Cuda
        );
        validate_args(&args).unwrap();
    }

    #[test]
    fn apply_gpu_subtree_args_propagates_backends_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-root-assignment-backend",
            "cuda",
            "--gpu-subtree-assignment-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-topk-backend",
            "cublas-gemm",
            "--gpu-subtree-execution-mode",
            "two-stage",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_root_assignment_backend,
            PiPNNComputeBackendKind::Cuda
        );
        assert_eq!(
            params.gpu_subtree_assignment_backend,
            PiPNNGpuSubtreeAssignmentBackend::PersistentTiledGemmFp32
        );
        assert_eq!(
            params.gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::CublasGemm
        );
        assert_eq!(
            params.gpu_subtree_execution_mode,
            PiPNNGpuSubtreeExecutionMode::TwoStage
        );
    }

    #[test]
    fn args_accept_gpu_subtree_dco_flags_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-dco-mode",
            "prefix-projection",
            "--gpu-subtree-assignment-dco-rerank-mode",
            "none",
            "--gpu-subtree-assignment-dco-proj-dim",
            "32",
            "--gpu-subtree-assignment-dco-survivors",
            "12",
            "--gpu-subtree-assignment-dco-audit",
            "--gpu-subtree-leaf-dco-mode",
            "wmma-filter",
            "--gpu-subtree-leaf-dco-rerank-mode",
            "exact",
            "--gpu-subtree-leaf-dco-proj-dim",
            "64",
            "--gpu-subtree-leaf-dco-survivors",
            "96",
            "--gpu-subtree-leaf-dco-audit",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_assignment_dco_mode,
            PiPNNGpuSubtreeDcoMode::PrefixProjection
        );
        assert_eq!(
            params.gpu_subtree_assignment_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::None
        );
        assert_eq!(params.gpu_subtree_assignment_dco_proj_dim, 32);
        assert_eq!(params.gpu_subtree_assignment_dco_survivors, 12);
        assert!(params.gpu_subtree_assignment_dco_audit);
        assert_eq!(params.gpu_subtree_d0_dco_mode, PiPNNGpuSubtreeDcoMode::None);
        assert_eq!(
            params.gpu_subtree_leaf_dco_mode,
            PiPNNGpuSubtreeDcoMode::WmmaFilter
        );
        assert_eq!(
            params.gpu_subtree_leaf_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(params.gpu_subtree_leaf_dco_proj_dim, 64);
        assert_eq!(params.gpu_subtree_leaf_dco_survivors, 96);
        assert!(params.gpu_subtree_leaf_dco_audit);
    }

    #[test]
    fn args_default_gpu_subtree_dco_modes_are_none() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
        ]);

        assert_eq!(
            args.gpu_subtree_assignment_dco_mode,
            PiPNNGpuSubtreeDcoMode::None
        );
        assert_eq!(
            args.gpu_subtree_assignment_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(args.gpu_subtree_assignment_dco_proj_dim, 0);
        assert_eq!(args.gpu_subtree_assignment_dco_survivors, 0);
        assert!(!args.gpu_subtree_assignment_dco_audit);
        assert_eq!(args.gpu_subtree_d0_dco_mode, PiPNNGpuSubtreeDcoMode::None);
        assert_eq!(
            args.gpu_subtree_d0_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(args.gpu_subtree_d0_dco_survivors, 0);
        assert!(!args.gpu_subtree_d0_dco_audit);
        assert_eq!(args.gpu_subtree_d0_sq_cpu_kernel, SqInt8CpuKernel::Auto);
        assert_eq!(args.gpu_subtree_leaf_dco_mode, PiPNNGpuSubtreeDcoMode::None);
        assert_eq!(
            args.gpu_subtree_leaf_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(
            args.gpu_subtree_execution_mode,
            PiPNNGpuSubtreeExecutionMode::FusedPersistent
        );
        assert_eq!(args.gpu_subtree_leaf_dco_proj_dim, 0);
        assert_eq!(args.gpu_subtree_leaf_dco_survivors, 0);
        assert!(!args.gpu_subtree_leaf_dco_audit);
        assert!(args.gpu_subtree_leaf_dco_ddc_res_pca_matrix.is_none());
        assert!(!args.gpu_subtree_leaf_dco_ddc_res_projected_input);
        assert_eq!(args.gpu_subtree_leaf_dco_ddc_res_base_dim, 32);
        assert_eq!(args.gpu_subtree_leaf_dco_ddc_res_sigma_count, 3.0);
        assert_eq!(
            args.gpu_subtree_rabitq_plus_mode,
            PiPNNGpuSubtreeRabitqPlusMode::Off
        );
        assert_eq!(args.gpu_subtree_rabitq_plus_bits, 8);
        assert_eq!(
            args.gpu_subtree_rabitq_plus_rotator,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac
        );
        assert_eq!(args.gpu_subtree_rabitq_plus_seed, 42);
        assert!(args.gpu_subtree_rabitq_plus_sidecar_prefix.is_none());
        assert_eq!(
            args.gpu_subtree_rabitq_plus_sidecar_mode,
            PiPNNGpuSubtreeRabitqPlusSidecarMode::Off
        );
        assert_eq!(
            args.gpu_subtree_qcode_staging_mode,
            PiPNNGpuSubtreeQcodeStagingMode::Off
        );
        assert_eq!(
            args.gpu_subtree_qcode_d1_payload_mode,
            PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident
        );
        assert_eq!(args.gpu_subtree_rabitq_residual_centroids, 16);
        assert_eq!(args.gpu_subtree_qcode_staging_upload_shard_rows, 524_288);
        assert_eq!(args.gpu_subtree_qcode_global_d0_leaders, 0);
        assert_eq!(
            args.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::Off
        );
        assert_eq!(args.gpu_subtree_sq_staging_upload_shard_rows, 524_288);
        assert_eq!(
            args.gpu_subtree_rabitq_d0_assign_mode,
            PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial
        );
        assert_eq!(args.gpu_subtree_rabitq_d0_row_tile, 4);
        assert_eq!(args.gpu_subtree_rabitq_d0_row_chunk, 4096);
        assert_eq!(args.gpu_subtree_rabitq_d0_leader_tile, 64);
        assert_eq!(args.gpu_subtree_rabitq_d0_partial_r, 8);
        assert_eq!(
            args.gpu_subtree_sq_wmma_mode,
            PiPNNGpuSubtreeSqWmmaMode::Off
        );
        assert_eq!(args.gpu_subtree_sq_bits, 8);
        assert!(!args.gpu_subtree_sq_native_int4);
        assert!(args.gpu_subtree_sq_wmma_sidecar_prefix.is_none());
        assert_eq!(
            args.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::Off
        );
        assert_eq!(
            args.gpu_subtree_turboquant_mode,
            PiPNNGpuSubtreeTurboQuantMode::Off
        );
        assert_eq!(args.gpu_subtree_turboquant_bits, 4);
        assert_eq!(args.gpu_subtree_turboquant_seed, 42);
        assert!(args.gpu_subtree_turboquant_sidecar_prefix.is_none());
        assert_eq!(
            args.gpu_subtree_turboquant_sidecar_mode,
            PiPNNGpuSubtreeTurboQuantSidecarMode::Off
        );
        assert_eq!(
            args.gpu_subtree_turboquant_sidecar_build_backend,
            TurboQuantSidecarBuildBackend::Auto
        );
        assert_eq!(args.gpu_subtree_turboquant_cascade_low_dim, 128);
        assert_eq!(args.gpu_subtree_turboquant_cascade_survivor_cap, 0);
        assert_eq!(args.gpu_subtree_turboquant_adaptive_survivor_threshold, 0);
        assert_eq!(args.gpu_subtree_turboquant_adaptive_survivor_low_cap, 0);
    }

    #[test]
    fn args_accept_gpu_subtree_turboquant_build_flags_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-turboquant-mode",
            "build",
            "--gpu-subtree-turboquant-sidecar-mode",
            "require",
            "--gpu-subtree-turboquant-sidecar-prefix",
            "/tmp/tq-sidecar",
            "--gpu-subtree-turboquant-sidecar-build-backend",
            "cuda",
            "--gpu-subtree-turboquant-cascade-low-dim",
            "256",
            "--gpu-subtree-turboquant-cascade-survivor-cap",
            "32",
            "--gpu-subtree-turboquant-adaptive-survivor-threshold",
            "1024",
            "--gpu-subtree-turboquant-adaptive-survivor-low-cap",
            "16",
        ]);

        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.gpu_subtree_turboquant_mode,
            PiPNNGpuSubtreeTurboQuantMode::Build
        );
        assert_eq!(params.gpu_subtree_turboquant_bits, 4);
        assert_eq!(params.gpu_subtree_turboquant_seed, 42);
        assert_eq!(
            params.gpu_subtree_turboquant_sidecar_mode,
            PiPNNGpuSubtreeTurboQuantSidecarMode::Require
        );
        assert_eq!(
            params.gpu_subtree_turboquant_sidecar_prefix,
            Some(PathBuf::from("/tmp/tq-sidecar"))
        );
        assert_eq!(
            args.gpu_subtree_turboquant_sidecar_build_backend,
            TurboQuantSidecarBuildBackend::Cuda
        );
        assert_eq!(params.gpu_subtree_turboquant_cascade_low_dim, 256);
        assert_eq!(params.gpu_subtree_turboquant_cascade_survivor_cap, 32);
        assert_eq!(
            params.gpu_subtree_turboquant_adaptive_survivor_threshold,
            1024
        );
        assert_eq!(params.gpu_subtree_turboquant_adaptive_survivor_low_cap, 16);
    }

    #[test]
    fn args_accept_sq_gpu_d0_staging_with_turboquant_build() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-d0-partition-mode",
            "gpu-kmeans-coarse",
            "--gpu-subtree-d0-capacity-enable",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/sq-gpu-d0-sidecar",
            "--gpu-subtree-turboquant-mode",
            "build",
            "--gpu-subtree-turboquant-sidecar-mode",
            "require",
            "--gpu-subtree-turboquant-sidecar-prefix",
            "/tmp/tq-sidecar",
        ]);

        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();
        super::apply_gpu_subtree_args(&mut params, &args);
        assert_eq!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_mode,
            PiPNNGpuSubtreeSqWmmaMode::Off
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_prefix,
            Some(PathBuf::from("/tmp/sq-gpu-d0-sidecar"))
        );
        assert_eq!(
            params.gpu_subtree_turboquant_mode,
            PiPNNGpuSubtreeTurboQuantMode::Build
        );
    }

    #[test]
    fn args_reject_sq_gpu_d0_direct_d1_staging_with_turboquant_build() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0-shard-local-d1",
            "--gpu-subtree-turboquant-mode",
            "build",
            "--gpu-subtree-turboquant-sidecar-mode",
            "require",
            "--gpu-subtree-turboquant-sidecar-prefix",
            "/tmp/tq-sidecar",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            format!("{err:?}")
                .contains("SQ GPU-D0 direct D1 staging cannot be combined with TurboQuant"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn args_reject_gpu_subtree_turboquant_zero_cascade_low_dim() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
            "--gpu-subtree-turboquant-mode",
            "shadow",
            "--gpu-subtree-turboquant-cascade-low-dim",
            "0",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("TurboQuant-MSE4 cascade low_dim must be > 0")
        );
    }

    #[test]
    fn args_reject_gpu_subtree_turboquant_non_four_bit() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
            "--gpu-subtree-turboquant-mode",
            "shadow",
            "--gpu-subtree-turboquant-bits",
            "8",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("TurboQuant-MSE4 v1 supports only"));
    }

    #[test]
    fn args_accept_gpu_subtree_d0_rabitq_dco_sidecar_flags_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "rabitq-wmma-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "12",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki.rabitq8",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d0_dco_mode,
            PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter
        );
        assert_eq!(
            params.gpu_subtree_d0_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::None
        );
        assert_eq!(params.gpu_subtree_d0_dco_survivors, 12);
        assert!(!params.gpu_subtree_d0_dco_audit);
        assert_eq!(
            params.gpu_subtree_rabitq_plus_sidecar_mode,
            PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
        );
    }

    #[test]
    fn args_accept_gpu_subtree_d0_rabitq_split_sidecar_flags_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "rabitq-split-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki.rabitq8split",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d0_dco_mode,
            PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        );
        assert_eq!(
            params.gpu_subtree_d0_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::None
        );
        assert_eq!(params.gpu_subtree_d0_dco_survivors, 32);
        assert!(!params.gpu_subtree_d0_dco_audit);
        assert_eq!(
            params.gpu_subtree_rabitq_plus_sidecar_mode,
            PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
        );
    }

    #[test]
    fn args_accept_gpu_subtree_d0_sq_int8_sidecar_flags_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "sq-int8",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-sq-bits",
            "5",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_d0_dco_mode,
            PiPNNGpuSubtreeDcoMode::SqInt8Filter
        );
        assert_eq!(
            params.gpu_subtree_d0_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::None
        );
        assert_eq!(params.gpu_subtree_d0_dco_survivors, 32);
        assert!(!params.gpu_subtree_d0_dco_audit);
        assert_eq!(params.gpu_subtree_sq_bits, 5);
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_sq_bits_outside_supported_range() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-bits",
            "3",
        ]);

        let err = validate_args(&args).expect_err("SQ bits below 4 should be rejected");
        assert!(
            format!("{err:?}").contains("gpu_subtree_sq_bits must be in [4, 8]"),
            "{err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_sq_native_int4_without_four_bit_sq() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/index.sq",
            "--gpu-subtree-sq-bits",
            "5",
            "--gpu-subtree-sq-native-int4",
        ]);

        let err = validate_args(&args).expect_err("native INT4 should require SQ bits=4");
        assert!(
            format!("{err:?}").contains("native INT4 SQ requires --gpu-subtree-sq-bits 4"),
            "{err:?}"
        );
    }

    #[test]
    fn args_accept_gpu_subtree_sq_native_int4_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/index.sq",
            "--gpu-subtree-sq-bits",
            "4",
            "--gpu-subtree-sq-native-int4",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(params.gpu_subtree_sq_bits, 4);
        assert!(params.gpu_subtree_sq_native_int4);
        assert_eq!(
            params.gpu_subtree_sq_wmma_mode,
            PiPNNGpuSubtreeSqWmmaMode::Build
        );
    }

    #[test]
    fn validate_args_accepts_uint8_sq_wmma_native_sq4_create_only() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.u8bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/index.u8.sq4",
            "--gpu-subtree-sq-bits",
            "4",
            "--gpu-subtree-sq-native-int4",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_accepts_uint8_sq_wmma_native_sq8_create_only() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.u8bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/index.u8.sq8",
            "--gpu-subtree-sq-bits",
            "8",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_rejects_uint8_full_build_without_oom() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.u8bin",
            "--index-path-prefix",
            "/tmp/index",
        ]);

        let err =
            validate_args(&args).expect_err("uint8 full build should require OOM point store");
        assert!(
            format!("{err:?}").contains("uint8 PiPNN build requires --oom-enable"),
            "{err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_uint8_oom_final_prune() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.u8bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--final-prune",
        ]);

        let err = validate_args(&args)
            .expect_err("uint8 OOM build should not fall back to f32 final prune");
        assert!(
            format!("{err:?}").contains("uint8 PiPNN build currently requires --final-prune=false"),
            "{err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_uint8_sq_wmma_invalid_native_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.u8bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/index.u8.sq5",
            "--gpu-subtree-sq-bits",
            "5",
        ]);

        let err = validate_args(&args).expect_err("uint8 SQ sidecar should reject non-native mode");
        assert!(
            format!("{err:?}").contains("uint8 SQ+WMMA sidecar supports native SQ8 dense-i8"),
            "{err:?}"
        );
    }

    #[test]
    fn maybe_create_sq_wmma_sidecar_creates_uint8_native_sq4_sidecar() {
        let temp = unique_temp_dir("uint8-sq4-sidecar");
        let data_path = temp.join("tiny.u8bin");
        let prefix = temp.join("tiny.u8.sq4");
        write_test_u8bin(&data_path, 5, 7);
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            data_path.to_str().unwrap(),
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            prefix.to_str().unwrap(),
            "--gpu-subtree-sq-bits",
            "4",
            "--gpu-subtree-sq-native-int4",
        ]);
        validate_args(&args).unwrap();

        let create_only =
            super::maybe_create_sq_wmma_sidecar(&args, &PiPNNParams::default(), 5, 7).unwrap();

        assert!(create_only);
        assert!(PathBuf::from(format!("{}.codes", prefix.display())).exists());
        assert!(PathBuf::from(format!("{}.meta", prefix.display())).exists());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn maybe_create_sq_wmma_sidecar_creates_uint8_native_sq8_sidecar() {
        let temp = unique_temp_dir("uint8-sq8-sidecar");
        let data_path = temp.join("tiny.u8bin");
        let prefix = temp.join("tiny.u8.sq8");
        write_test_u8bin(&data_path, 5, 7);
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "uint8",
            "--dist-fn",
            "l2",
            "--data-path",
            data_path.to_str().unwrap(),
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            prefix.to_str().unwrap(),
            "--gpu-subtree-sq-bits",
            "8",
        ]);
        validate_args(&args).unwrap();

        let create_only =
            super::maybe_create_sq_wmma_sidecar(&args, &PiPNNParams::default(), 5, 7).unwrap();

        assert!(create_only);
        assert!(PathBuf::from(format!("{}.codes", prefix.display())).exists());
        assert!(PathBuf::from(format!("{}.meta", prefix.display())).exists());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn args_accept_gpu_subtree_d0_sq_cpu_kernel_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "sq-int8-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-d0-sq-cpu-kernel",
            "avx512-vnni",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();

        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            args.gpu_subtree_d0_sq_cpu_kernel,
            SqInt8CpuKernel::Avx512Vnni
        );
        assert_eq!(
            params.gpu_subtree_d0_sq_cpu_kernel,
            SqInt8CpuKernel::Avx512Vnni
        );
    }

    #[test]
    fn validate_args_rejects_d0_sq_int8_exact_rerank() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "sq-int8-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "exact",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("approximate-only"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_d0_sq_int8_without_sq_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "sq-int8-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--pipnn-backend",
            "cuda",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("must regenerate its SQ+WMMA sidecar"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_d0_rabitq_wmma_exact_rerank() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "rabitq-wmma-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "exact",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki.rabitq8split",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("1-bit/ex-bit boosting"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_d0_rabitq_wmma_raw_audit() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "rabitq-wmma-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-d0-dco-audit",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki.rabitq8split",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("raw audit"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn args_accept_gpu_subtree_rabitq_plus_shadow_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-rabitq-plus-mode",
            "shadow",
            "--gpu-subtree-rabitq-plus-bits",
            "5",
            "--gpu-subtree-rabitq-plus-rotator",
            "none",
            "--gpu-subtree-rabitq-plus-seed",
            "777",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_rabitq_plus_mode,
            PiPNNGpuSubtreeRabitqPlusMode::Shadow
        );
        assert_eq!(params.gpu_subtree_rabitq_plus_bits, 5);
        assert_eq!(
            params.gpu_subtree_rabitq_plus_rotator,
            PiPNNGpuSubtreeRabitqPlusRotator::None
        );
        assert_eq!(params.gpu_subtree_rabitq_plus_seed, 777);
        assert!(params.gpu_subtree_rabitq_plus_sidecar_prefix.is_none());
        assert_eq!(
            params.gpu_subtree_rabitq_plus_sidecar_mode,
            PiPNNGpuSubtreeRabitqPlusSidecarMode::Off
        );
    }

    #[test]
    fn args_accept_rabitq_residual_qcode_staging_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-qcode-staging-mode",
            "rabitq-residual",
            "--gpu-subtree-qcode-d1-payload-mode",
            "child-local",
            "--gpu-subtree-rabitq-d1-distance-mode",
            "split-short-ex",
            "--gpu-subtree-rabitq-short-survivors",
            "64",
            "--gpu-subtree-rabitq-residual-centroids",
            "32",
            "--gpu-subtree-qcode-staging-upload-shard-rows",
            "262144",
            "--gpu-subtree-qcode-global-d0-leaders",
            "256",
            "--gpu-subtree-rabitq-d0-assign-mode",
            "batch",
            "--gpu-subtree-rabitq-d1-topk-mode",
            "wmma-batch",
            "--gpu-subtree-rabitq-d0-row-tile",
            "8",
            "--gpu-subtree-rabitq-d0-row-chunk",
            "2048",
            "--gpu-subtree-rabitq-d0-leader-tile",
            "32",
            "--gpu-subtree-rabitq-d0-partial-r",
            "16",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_qcode_staging_mode,
            PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual
        );
        assert_eq!(
            params.gpu_subtree_qcode_d1_payload_mode,
            PiPNNGpuSubtreeQcodeD1PayloadMode::ChildLocal
        );
        assert_eq!(
            params.gpu_subtree_rabitq_d1_distance_mode,
            PiPNNGpuSubtreeRabitqD1DistanceMode::SplitShortEx
        );
        assert_eq!(params.gpu_subtree_rabitq_short_survivors, Some(64));
        assert_eq!(params.gpu_subtree_rabitq_residual_centroids, 32);
        assert_eq!(params.gpu_subtree_qcode_staging_upload_shard_rows, 262_144);
        assert_eq!(params.gpu_subtree_qcode_global_d0_leaders, 256);
        assert_eq!(
            params.gpu_subtree_rabitq_d0_assign_mode,
            PiPNNGpuSubtreeRabitqD0AssignMode::Batch
        );
        assert_eq!(
            params.gpu_subtree_rabitq_d1_topk_mode,
            PiPNNGpuSubtreeRabitqD1TopKMode::WmmaBatch
        );
        assert_eq!(params.gpu_subtree_rabitq_d0_row_tile, 8);
        assert_eq!(params.gpu_subtree_rabitq_d0_row_chunk, 2048);
        assert_eq!(params.gpu_subtree_rabitq_d0_leader_tile, 32);
        assert_eq!(params.gpu_subtree_rabitq_d0_partial_r, 16);
    }

    #[test]
    fn args_accept_sq_gpu_d0_staging_and_propagate_to_params() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-staging-upload-shard-rows",
            "262144",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0
        );
        assert_eq!(params.gpu_subtree_sq_staging_upload_shard_rows, 262_144);
        assert_eq!(params.gpu_subtree_sq_bits, 8);
        assert_eq!(
            params.gpu_subtree_sq_wmma_mode,
            PiPNNGpuSubtreeSqWmmaMode::Build
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild
        );
        assert_eq!(
            params.gpu_subtree_sq_wmma_sidecar_prefix,
            Some(PathBuf::from("/tmp/wiki10m.sq8"))
        );
    }

    #[test]
    fn validate_args_rejects_sq_gpu_d0_staging_with_non_eight_bit_sq() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-bits",
            "7",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("SQ GPU-D0 staging requires --gpu-subtree-sq-bits 8"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_sq_gpu_d0_staging_with_create_only_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("SQ GPU-D0 staging supports only sidecar rebuild"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_sq_gpu_d0_staging_with_native_int4() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-sq-native-int4",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("SQ GPU-D0 staging does not support native INT4"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_sq_gpu_d0_staging_with_rabitq_qcode_staging() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--gpu-subtree-qcode-staging-mode",
            "rabitq-residual",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("SQ GPU-D0 staging cannot be combined with RabitQ"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_tiled_rabitq_d0_without_qcode_staging() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-rabitq-d0-assign-mode",
            "tiled",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            format!("{err:?}").contains("tiled/batch/wmma-batch residual RaBitQ D0 assignment requires --gpu-subtree-qcode-staging-mode rabitq-residual"),
            "{err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_wmma_batch_rabitq_d0_without_qcode_staging() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-rabitq-d0-assign-mode",
            "wmma-batch",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            format!("{err:?}").contains("tiled/batch/wmma-batch residual RaBitQ D0 assignment requires --gpu-subtree-qcode-staging-mode rabitq-residual"),
            "{err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_wmma_batch_rabitq_d1_without_child_local_payload() {
        let args = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-qcode-staging-mode",
            "rabitq-residual",
            "--gpu-subtree-rabitq-d1-topk-mode",
            "wmma-batch",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ])
        .expect("wmma-batch D1+ CLI should parse");

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains(
                "WMMA-batch residual RaBitQ D1+ top-k requires child-local qcode payload"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_rabitq_residual_qcode_staging_with_sq_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-qcode-staging-mode",
            "rabitq-residual",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("RabitQ residual qcode staging cannot be combined with SQ"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn global_seed_does_not_change_rabitq_plus_seed() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-rabitq-plus-mode",
            "shadow",
            "--seed",
            "1234",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);
        let mut params = PiPNNParams::default();

        validate_args(&args).unwrap();
        params.random_seed = args.seed;
        super::apply_gpu_subtree_args(&mut params, &args);

        assert_eq!(params.random_seed, 1234);
        assert_eq!(params.gpu_subtree_rabitq_plus_seed, 42);
    }

    #[test]
    fn validate_args_accepts_gpu_subtree_rabitq_plus_build_quantized_only_path() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-rabitq-plus-mode",
            "build",
            "--gpu-subtree-rabitq-plus-bits",
            "8",
            "--gpu-subtree-rabitq-plus-rotator",
            "fht-kac",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki10m.rabitq8",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args)
            .expect("dense-i8 RabitQ+ build mode should validate on the WMMA CUDA path");
    }

    #[test]
    fn args_accept_gpu_subtree_rabitq_plus_sidecar_create_only() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki10m.rabitq8",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "create-only",
        ]);

        validate_args(&args).unwrap();
        assert_eq!(
            args.gpu_subtree_rabitq_plus_sidecar_prefix,
            Some(PathBuf::from("/tmp/wiki10m.rabitq8"))
        );
        assert_eq!(
            args.gpu_subtree_rabitq_plus_sidecar_mode,
            PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly
        );
    }

    #[test]
    fn rabitq_plus_rebuild_creates_sidecar_before_index_build() {
        let dir = unique_temp_dir("rabitq-plus-rebuild-sidecar");
        let data_path = dir.join("tiny.fbin");
        let prefix = dir.join("tiny.rabitq8");
        write_test_fbin(&data_path, 4, 16);
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            data_path.to_str().unwrap(),
            "--index-path-prefix",
            dir.join("index").to_str().unwrap(),
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-rabitq-plus-mode",
            "build",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            prefix.to_str().unwrap(),
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "rebuild",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();
        params.oom_io_mode = args.oom_io_mode;

        let create_only = super::maybe_create_rabitq_plus_sidecar(&args, &params).unwrap();

        assert!(!create_only, "rebuild should continue into index build");
        assert!(PathBuf::from(format!("{}.codes", prefix.display())).exists());
        assert!(PathBuf::from(format!("{}.meta", prefix.display())).exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn d0_rabitq_split_create_only_creates_split_sidecar() {
        let dir = unique_temp_dir("d0-rabitq-split-sidecar");
        let data_path = dir.join("tiny.fbin");
        let prefix = dir.join("tiny.rabitq8");
        write_test_fbin(&data_path, 64, 64);
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            data_path.to_str().unwrap(),
            "--index-path-prefix",
            dir.join("index").to_str().unwrap(),
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--pipnn-backend",
            "cuda",
            "--gpu-subtree-d0-dco-mode",
            "rabitq-split-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            prefix.to_str().unwrap(),
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "create-only",
        ]);
        validate_args(&args).unwrap();
        let mut params = PiPNNParams::default();
        params.oom_io_mode = args.oom_io_mode;

        let create_only = super::maybe_create_rabitq_plus_sidecar(&args, &params).unwrap();

        assert!(create_only, "create-only should skip index build");
        assert!(PathBuf::from(format!("{}.split.batch", prefix.display())).exists());
        assert!(PathBuf::from(format!("{}.split.ex", prefix.display())).exists());
        assert!(PathBuf::from(format!("{}.split.meta", prefix.display())).exists());
        assert!(!PathBuf::from(format!("{}.codes", prefix.display())).exists());
        assert!(!PathBuf::from(format!("{}.meta", prefix.display())).exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_rabitq_plus_build_without_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-rabitq-plus-mode",
            "build",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("requires --gpu-subtree-rabitq-plus-sidecar-mode"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_accepts_gpu_subtree_sq_wmma_build_quantized_only_path() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("SQ+WMMA build mode should validate on the WMMA CUDA path");
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_sq_wmma_build_with_reused_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("must regenerate its SQ+WMMA sidecar"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_d0_sq_with_reused_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--oom-enable",
            "--gpu-subtree-d0-dco-mode",
            "sq-int8-filter",
            "--gpu-subtree-d0-dco-rerank-mode",
            "none",
            "--gpu-subtree-d0-dco-survivors",
            "32",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("must regenerate its SQ+WMMA sidecar"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn remove_sq_wmma_sidecar_files_deletes_codes_and_meta() {
        let temp = unique_temp_dir("sq-sidecar-cleanup");
        let prefix = temp.join("wiki10m.sq8");
        let codes = PathBuf::from(format!("{}.codes", prefix.display()));
        let meta = PathBuf::from(format!("{}.meta", prefix.display()));
        std::fs::write(&codes, b"codes").unwrap();
        std::fs::write(&meta, b"meta").unwrap();

        let removed = remove_sq_wmma_sidecar_files(&prefix).unwrap();

        assert_eq!(removed, 2);
        assert!(!codes.exists());
        assert!(!meta.exists());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn maybe_create_sq_wmma_sidecar_skips_cpu_creation_for_sq_gpu_d0_staging() {
        let temp = unique_temp_dir("sq-gpu-d0-sidecar-skip");
        let prefix = temp.join("wiki10m.sq8");
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/missing-gpu-d0-staging-input.fbin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-sq-staging-mode",
            "gpu-d0",
            "--gpu-subtree-sq-bits",
            "8",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            prefix.to_str().unwrap(),
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "rebuild",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let skipped_create_only =
            super::maybe_create_sq_wmma_sidecar(&args, &PiPNNParams::default(), 8, 4)
                .expect("SQ GPU-D0 staging should defer sidecar creation to root partitioning");

        assert!(!skipped_create_only);
        assert!(!PathBuf::from(format!("{}.codes", prefix.display())).exists());
        assert!(!PathBuf::from(format!("{}.meta", prefix.display())).exists());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn args_accept_gpu_subtree_sq_wmma_sidecar_create_only() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-sq-wmma-sidecar-prefix",
            "/tmp/wiki10m.sq8",
            "--gpu-subtree-sq-wmma-sidecar-mode",
            "create-only",
        ]);

        validate_args(&args).unwrap();
        assert_eq!(
            args.gpu_subtree_sq_wmma_sidecar_prefix,
            Some(PathBuf::from("/tmp/wiki10m.sq8"))
        );
        assert_eq!(
            args.gpu_subtree_sq_wmma_sidecar_mode,
            PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_sq_wmma_build_without_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-sq-wmma-mode",
            "build",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("must regenerate its SQ+WMMA sidecar"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_rabitq_plus_build_with_existing_dco() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-dco-mode",
            "wmma-filter",
            "--gpu-subtree-leaf-dco-survivors",
            "16",
            "--gpu-subtree-rabitq-plus-mode",
            "build",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki10m.rabitq8",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("cannot be combined with existing DCO/audit switches"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_rabitq_plus_invalid_bits_even_when_off() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--gpu-subtree-rabitq-plus-bits",
            "6",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("Dense-i8 RabitQ+ total bits must be one of 4, 5, 7, or 8"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_accepts_assignment_projection_dco_when_kernel_is_wired() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-dco-mode",
            "prefix-projection",
            "--gpu-subtree-assignment-dco-proj-dim",
            "32",
            "--gpu-subtree-assignment-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("assignment prefix-projection DCO should validate");
    }

    #[test]
    fn validate_args_rejects_assignment_exact_stamp_rerank_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-dco-mode",
            "prefix-projection",
            "--gpu-subtree-assignment-dco-rerank-mode",
            "exact-stamp",
            "--gpu-subtree-assignment-dco-proj-dim",
            "32",
            "--gpu-subtree-assignment-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("assignment DCO does not support exact-stamp"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_accepts_assignment_quantized_dco_when_wmma_kernel_is_wired() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-assignment-dco-mode",
            "quantized",
            "--gpu-subtree-assignment-dco-proj-dim",
            "64",
            "--gpu-subtree-assignment-dco-survivors",
            "4",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("assignment quantized DCO should validate");
    }

    #[test]
    fn validate_args_rejects_unwired_assignment_dco_modes() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-dco-mode",
            "shape-aware",
            "--gpu-subtree-assignment-dco-proj-dim",
            "32",
            "--gpu-subtree-assignment-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}")
                .contains("only assignment DCO prefix-projection and quantized are wired"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_assignment_quantized_dco_for_non_wmma_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-assignment-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-assignment-dco-mode",
            "quantized",
            "--gpu-subtree-assignment-dco-proj-dim",
            "64",
            "--gpu-subtree-assignment-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("assignment DCO quantized currently requires"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_accepts_wired_leaf_projection_dco() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-dco-mode",
            "prefix-projection",
            "--gpu-subtree-leaf-dco-proj-dim",
            "32",
            "--gpu-subtree-leaf-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("wired leaf prefix-projection DCO should validate");
    }

    #[test]
    fn validate_args_accepts_leaf_ddc_res_with_pca_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-dco-mode",
            "ddc-res",
            "--gpu-subtree-leaf-dco-ddc-res-pca-matrix",
            "/tmp/wiki_pca_matrix_10.fvecs",
            "--gpu-subtree-leaf-dco-ddc-res-projected-input",
            "--gpu-subtree-leaf-dco-ddc-res-base-dim",
            "32",
            "--gpu-subtree-leaf-dco-ddc-res-sigma-count",
            "3.5",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("faithful leaf ddc-res DCO should validate");
        assert_eq!(
            args.gpu_subtree_leaf_dco_ddc_res_pca_matrix,
            Some(PathBuf::from("/tmp/wiki_pca_matrix_10.fvecs"))
        );
        assert!(args.gpu_subtree_leaf_dco_ddc_res_projected_input);
        assert_eq!(args.gpu_subtree_leaf_dco_ddc_res_base_dim, 32);
        assert_eq!(args.gpu_subtree_leaf_dco_ddc_res_sigma_count, 3.5);
    }

    #[test]
    fn validate_args_rejects_leaf_ddc_res_without_pca_sidecar() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-dco-mode",
            "ddc-res",
            "--gpu-subtree-leaf-dco-ddc-res-projected-input",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("requires a ResInfer PCA matrix sidecar"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_leaf_ddc_res_without_exact_rerank() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-dco-mode",
            "ddc-res",
            "--gpu-subtree-leaf-dco-rerank-mode",
            "none",
            "--gpu-subtree-leaf-dco-ddc-res-pca-matrix",
            "/tmp/wiki_pca_matrix_10.fvecs",
            "--gpu-subtree-leaf-dco-ddc-res-projected-input",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("ddc-res requires exact rerank"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_leaf_dco_for_non_tiled_backend() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-dco-mode",
            "prefix-projection",
            "--gpu-subtree-leaf-dco-proj-dim",
            "32",
            "--gpu-subtree-leaf-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("prefix-projection/ddc-res require"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_accepts_leaf_wmma_filter_dco_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-dco-mode",
            "wmma-filter",
            "--gpu-subtree-leaf-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("wmma-filter leaf DCO should be a wired mode");
    }

    #[test]
    fn validate_args_accepts_leaf_wmma_filter_exact_stamp_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-dco-mode",
            "wmma-filter",
            "--gpu-subtree-leaf-dco-rerank-mode",
            "exact-stamp",
            "--gpu-subtree-leaf-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("wmma-filter exact-stamp leaf DCO should validate");
    }

    #[test]
    fn validate_args_accepts_leaf_rabitq_split_filter_exact_stamp_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-dco-mode",
            "rabitq-split-filter",
            "--gpu-subtree-leaf-dco-rerank-mode",
            "exact-stamp",
            "--gpu-subtree-leaf-dco-survivors",
            "16",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki.rabitq8split",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        validate_args(&args).expect("RabitQ-Library split exact-stamp leaf DCO should validate");
    }

    #[test]
    fn validate_args_rejects_leaf_rabitq_split_filter_proj_dim() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-wmma-tf32",
            "--gpu-subtree-leaf-dco-mode",
            "rabitq-split-filter",
            "--gpu-subtree-leaf-dco-survivors",
            "16",
            "--gpu-subtree-leaf-dco-proj-dim",
            "8",
            "--gpu-subtree-rabitq-plus-sidecar-prefix",
            "/tmp/wiki.rabitq8split",
            "--gpu-subtree-rabitq-plus-sidecar-mode",
            "require",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains("1bit+7bit split_distance_boosting"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_unwired_leaf_dco_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--gpu-subtree-leaf-topk-backend",
            "persistent-tiled-gemm-fp32",
            "--gpu-subtree-leaf-dco-mode",
            "shape-aware",
            "--gpu-subtree-leaf-dco-survivors",
            "8",
            "--pipnn-backend",
            "cuda",
            "--oom-enable",
        ]);

        let err = validate_args(&args).unwrap_err();

        assert!(
            format!("{err:?}").contains(
                "only leaf top-k DCO prefix-projection, ddc-res, wmma-filter, and rabitq-split-filter are wired"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn validate_args_rejects_gpu_subtree_without_required_cuda_oom_l2_boundary() {
        let mut args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "gpu-subtree",
            "--pipnn-backend",
            "cuda",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("gpu-subtree requires --oom-enable")
        );

        args.oom_enable = true;
        args.pipnn_backend = "cpu".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("gpu-subtree requires --pipnn-backend=cuda")
        );

        args.pipnn_backend = "cuda".to_string();
        args.dist_fn = "cosine".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("gpu-subtree currently supports only l2")
        );
    }

    #[test]
    fn validate_args_rejects_invalid_kmeans_params() {
        let mut args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--partitioner",
            "kmeans",
            "--kmeans-iterations",
            "0",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("kmeans_iterations"));

        args.kmeans_iterations = 4;
        args.kmeans_oversplit_factor = 0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("kmeans_oversplit_factor"));
    }

    #[test]
    fn args_accept_explicit_backend_and_device() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--pipnn-backend",
            "auto",
            "--pipnn-gpu-device",
            "3",
        ]);

        assert_eq!(args.pipnn_backend, "auto");
        assert_eq!(args.pipnn_gpu_device, 3);
    }

    #[test]
    fn args_accept_gpu_workspace_budget_without_enabling_gpu_by_default() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--pipnn-gpu-workspace-budget-gb",
            "0.5",
        ]);

        assert_eq!(args.pipnn_backend, "cpu");
        assert_eq!(
            args.pipeline_compute_mode,
            PiPNNPipelineComputeMode::Passthrough
        );
        assert_eq!(args.pipnn_gpu_workspace_budget_gb, 0.5);
    }

    #[test]
    fn args_accept_cuda_pipeline_compute_mode_without_running_gpu() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--pipeline-runtime-enable",
            "--pipeline-compute-mode",
            "gpu",
        ]);

        assert!(args.pipeline_runtime_enable);
        assert_eq!(args.pipeline_compute_mode, PiPNNPipelineComputeMode::Cuda);
    }

    #[test]
    fn args_accept_oom_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--oom-io-mode",
            "buffered",
            "--oom-memory-budget-gb",
            "64",
            "--oom-reservoir-share",
            "0.5",
            "--oom-sidecar-resident-budget-cap-gb",
            "0",
            "--oom-sketch-cache-gb",
            "8",
            "--oom-raw-vector-cache-gb",
            "6",
            "--oom-spill-cache-gb",
            "4",
            "--oom-temp-dir",
            "/tmp/pipnn-oom",
            "--oom-keep-artifacts",
            "--oom-profile-json",
            "/tmp/pipnn-profile.json",
        ]);

        assert!(args.oom_enable);
        assert!(!args.oom_strict_io);
        assert_eq!(args.oom_io_mode, PiPNNOomIoMode::Buffered);
        assert_eq!(args.oom_memory_budget_gb, 64.0);
        assert_eq!(args.oom_reservoir_share, 0.5);
        assert_eq!(args.oom_sidecar_resident_budget_cap_gb, Some(0.0));
        assert_eq!(args.oom_sketch_cache_gb, 8.0);
        assert_eq!(args.oom_raw_vector_cache_gb, 6.0);
        assert_eq!(args.oom_spill_cache_gb, 4.0);
        assert_eq!(args.oom_temp_dir, "/tmp/pipnn-oom");
        assert!(args.oom_keep_artifacts);
        assert_eq!(
            args.oom_profile_json,
            Some(PathBuf::from("/tmp/pipnn-profile.json"))
        );
    }

    #[test]
    fn args_keep_legacy_oom_strict_io_as_direct_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--oom-strict-io",
            "--oom-io-mode",
            "direct",
        ]);

        assert!(args.oom_enable);
        assert!(args.oom_strict_io);
        assert_eq!(args.oom_io_mode, PiPNNOomIoMode::Direct);
        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_rejects_invalid_oom_reservoir_share() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-reservoir-share",
            "1.5",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string().contains("oom_reservoir_share"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_args_rejects_invalid_oom_sidecar_resident_cap() {
        for cap in ["-1", "NaN"] {
            let arg = format!("--oom-sidecar-resident-budget-cap-gb={cap}");
            let args = Args::parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                arg.as_str(),
            ]);

            let err = validate_args(&args).unwrap_err();
            assert!(
                err.to_string()
                    .contains("oom_sidecar_resident_budget_cap_gb"),
                "unexpected error for cap {cap}: {err}"
            );
        }
    }

    #[test]
    fn args_reject_conflicting_oom_strict_io_and_buffered_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--oom-strict-io",
            "--oom-io-mode",
            "buffered",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("conflicts"));
    }

    #[test]
    fn args_accept_cuda_compact_dump_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--pipnn-cuda-compact-dump-dir",
            "/tmp/pipnn-compact-dump",
            "--pipnn-cuda-compact-dump-limit",
            "128",
        ]);

        assert_eq!(
            args.pipnn_cuda_compact_dump_dir,
            Some(PathBuf::from("/tmp/pipnn-compact-dump"))
        );
        assert_eq!(args.pipnn_cuda_compact_dump_limit, 128);
    }

    #[test]
    fn build_shear_index_accepts_root_radius_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "cell-radius-certified",
            "--root-radius-k",
            "7",
            "--root-radius-candidate-rs",
            "3,5",
            "--root-radius-warmup-points",
            "123",
            "--root-radius-dry-run",
        ]);

        assert_eq!(
            args.root_fanout_policy,
            RootFanoutPolicy::CellRadiusCertified
        );
        assert_eq!(args.root_radius_k, 7);
        assert_eq!(args.root_radius_candidate_rs, "3,5");
        assert_eq!(
            super::parse_root_radius_candidate_rs(&args.root_radius_candidate_rs).unwrap(),
            vec![3, 5]
        );
        assert_eq!(args.root_radius_warmup_points, 123);
        assert!(args.root_radius_dry_run);
    }

    #[test]
    fn root_risk_build_shear_index_accepts_flags() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "risk-budgeted-root-assign",
            "--root-risk-target-avg-fanout",
            "7.0",
            "--root-risk-sigma-lambda",
            "1.5",
            "--root-risk-budget-lambda",
            "0.125",
            "--root-risk-sigma-mode",
            "cell-q90-q50",
            "--root-risk-candidate-rs",
            "4,6,8",
            "--root-risk-warmup-points",
            "321",
            "--root-risk-dry-run",
        ]);

        assert_eq!(args.root_fanout_policy, RootFanoutPolicy::RiskBudgeted);
        assert_eq!(args.root_risk_target_avg_fanout, 7.0);
        assert_eq!(args.root_risk_sigma_lambda, 1.5);
        assert_eq!(args.root_risk_budget_lambda, Some(0.125));
        assert_eq!(args.root_risk_sigma_mode.to_string(), "cell-q90-q50");
        assert_eq!(
            super::parse_root_risk_candidate_rs(&args.root_risk_candidate_rs, args.fanout_top)
                .unwrap(),
            vec![4, 6, 8, 10]
        );
        assert_eq!(args.root_risk_warmup_points, 321);
        assert!(args.root_risk_dry_run);
        validate_args(&args).unwrap();
    }

    #[test]
    fn build_shear_index_rejects_old_root_risk_epsilon_flag() {
        let err = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-risk-epsilon",
            "0.01",
        ])
        .unwrap_err();

        assert!(err.to_string().contains("unexpected argument"));
    }

    #[test]
    fn build_shear_index_rejects_old_root_witness_flags_and_policies() {
        for policy in ["witness", "witness-certified", "wc-ira"] {
            let policy_err = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                "--root-fanout-policy",
                policy,
            ])
            .unwrap_err();
            assert!(
                policy_err
                    .to_string()
                    .contains("Unsupported root fanout policy"),
                "{policy} should be rejected"
            );
        }

        for flag in [
            "--root-witness-k",
            "--root-witness-candidate-rs",
            "--root-witnesses-per-cell",
            "--root-witness-warmup-points",
            "--root-witness-min-witnesses",
            "--root-witness-dry-run",
        ] {
            let flag_err = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                flag,
            ])
            .unwrap_err();
            assert!(
                flag_err.to_string().contains("unexpected argument"),
                "{flag} should be rejected"
            );
        }
    }

    #[test]
    fn validate_args_rejects_invalid_root_radius_inputs() {
        let mut args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-radius-k",
            "0",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("root radius k must be in [1, 32]"));

        args.root_radius_k = 33;
        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains("root radius k must be in [1, 32]"));

        args.root_radius_k = 10;
        args.root_radius_candidate_rs = "0,0".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("root radius candidate rs must contain at least one positive integer")
        );
    }

    #[test]
    fn validate_args_rejects_stop_after_d00_without_oom() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-stop-after-d00",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("root-stop-after-d00 requires --oom-enable")
        );
    }

    #[test]
    fn validate_args_rejects_full_cell_radius_certified_root_radius_builds() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "cell-radius-certified",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string().contains(
                "cell-radius-certified is diagnostic-only; use --root-radius-dry-run or --root-stop-after-d00"
            )
        );
    }

    #[test]
    fn validate_args_accepts_cell_radius_certified_root_radius_dry_run() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "cell-radius-certified",
            "--root-radius-dry-run",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_accepts_cell_radius_certified_root_radius_stop_after_d00_oom() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "cell-radius-certified",
            "--root-stop-after-d00",
            "--oom-enable",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn root_risk_validate_args_rejects_invalid_inputs() {
        let mut args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "risk-budgeted",
            "--root-risk-dry-run",
            "--root-risk-target-avg-fanout",
            "3.0",
        ]);
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("root risk target avg fanout must be in")
        );

        args.root_risk_target_avg_fanout = 7.0;
        args.root_risk_sigma_lambda = 0.0;
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("root risk sigma lambda must be > 0")
        );

        args.root_risk_sigma_lambda = 1.0;
        args.root_risk_budget_lambda = Some(-1.0);
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("root risk budget lambda must be finite and >= 0")
        );

        args.root_risk_budget_lambda = None;
        args.root_risk_candidate_rs = "0,11".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("root risk candidate rs must be in [1, fanout_top]")
        );
    }

    #[test]
    fn root_risk_validate_args_rejects_full_builds() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "risk-budgeted",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(err.to_string().contains(
            "risk-budgeted is root-only; use --root-risk-dry-run, --root-stop-after-d00, or --root-risk-allow-full-build"
        ));
    }

    #[test]
    fn root_risk_validate_args_accepts_explicit_full_build_override() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "risk-budgeted",
            "--root-risk-allow-full-build",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn root_risk_validate_args_accepts_stop_after_d00_oom() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "risk-budgeted",
            "--root-stop-after-d00",
            "--oom-enable",
        ]);

        validate_args(&args).unwrap();
    }

    #[test]
    fn build_shear_index_rejects_coverage_policy_and_flags() {
        let policy_err = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "coverage",
        ])
        .unwrap_err();
        assert!(
            policy_err
                .to_string()
                .contains("Unsupported root fanout policy")
        );

        for flag in [
            "--root-coverage-pair-file",
            "--root-coverage-auto-calibrate",
            "--root-coverage-probe-fanout",
            "--root-coverage-validation-split",
            "--root-coverage-target-miss",
            "--root-coverage-cross-target-miss",
            "--root-coverage-cell-target-miss",
            "--root-coverage-boundary-target-miss",
            "--root-coverage-min-cell-sources",
            "--root-coverage-delta",
            "--root-coverage-sample-points",
            "--root-coverage-dry-run",
            "--root-coverage-calibration-initial-sources",
            "--root-coverage-calibration-max-sources",
            "--root-coverage-calibration-boundary-pool",
            "--root-coverage-calibration-knn",
            "--root-coverage-calibration-threads",
            "--root-coverage-calibration-query-batch",
            "--root-coverage-calibration-base-block",
        ] {
            let flag_err = Args::try_parse_from([
                "build_shear_index",
                "--data-type",
                "float",
                "--dist-fn",
                "l2",
                "--data-path",
                "/tmp/data.bin",
                "--index-path-prefix",
                "/tmp/index",
                flag,
            ])
            .unwrap_err();
            assert!(
                flag_err.to_string().contains("unexpected argument"),
                "{flag} should be rejected"
            );
        }
    }

    #[test]
    fn build_shear_index_rejects_legacy_margin_policy() {
        let err = Args::try_parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--root-fanout-policy",
            "margin",
        ])
        .unwrap_err();

        assert!(err.to_string().contains("Unsupported root fanout policy"));
    }

    #[test]
    fn validate_args_rejects_non_cpu_backend_in_oom_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--pipnn-backend",
            "auto",
        ]);

        let err = validate_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("OOM PiPNN mode supports --pipnn-backend=cpu")
        );
    }

    #[test]
    fn validate_args_accepts_explicit_cuda_pipeline_in_oom_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--pipeline-runtime-enable",
            "--pipeline-compute-mode",
            "cuda",
            "--pipnn-backend",
            "cuda",
        ]);

        validate_args(&args).expect("explicit OOM CUDA pipeline should be accepted");
    }

    #[test]
    fn validate_args_accepts_final_prune_in_oom_mode() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--final-prune",
        ]);

        validate_args(&args).expect("OOM PiPNN final prune should be accepted");
    }

    #[test]
    fn oom_mode_accepts_final_prune_for_driver_validation() {
        let args = Args::parse_from([
            "build_shear_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.bin",
            "--index-path-prefix",
            "/tmp/index",
            "--oom-enable",
            "--final-prune",
            "--pipnn-backend",
            "cpu",
        ]);

        validate_args(&args).expect("OOM PiPNN final prune should now be supported");
    }

    #[test]
    fn oom_build_writes_loadable_mem_index() {
        let dir = unique_temp_dir("pipnn-oom-build");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 32, 8);

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.final_prune = false;
        params.oom_temp_dir = dir.join("oom-artifacts");

        build_shear_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            16,
            index_dir.to_str().unwrap(),
            1,
            true,
            0.0,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());

        let index_write_parameters = IndexWriteParametersBuilder::new(16, 8)
            .with_num_threads(1)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            8,
            round_up(8, 32),
            32,
            false,
            0,
            false,
            0,
            1.3,
            index_write_parameters,
            0.0,
        );
        let mut index = InmemIndex::<f32>::new(config).unwrap();
        index
            .load_graph(&graph_path, 32)
            .expect("OOM build should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oom_uint8_build_writes_loadable_mem_index() {
        let dir = unique_temp_dir("pipnn-oom-uint8-build");
        let data_path = dir.join("tiny.u8bin");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_u8bin(&data_path, 32, 8);

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.final_prune = false;
        params.oom_temp_dir = dir.join("oom-artifacts");

        build_shear_index_with_data_type(
            "uint8",
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            16,
            index_dir.to_str().unwrap(),
            1,
            true,
            0.0,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());

        let index_write_parameters = IndexWriteParametersBuilder::new(16, 8)
            .with_num_threads(1)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            8,
            round_up(8, 32),
            32,
            false,
            0,
            false,
            0,
            1.3,
            index_write_parameters,
            0.0,
        );
        let mut index = InmemIndex::<f32>::new(config).unwrap();
        index
            .load_graph(&graph_path, 32)
            .expect("uint8 OOM build should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oom_build_creates_missing_output_directory() {
        let dir = unique_temp_dir("pipnn-oom-build-missing-output-dir");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("nested").join("index");
        write_test_fbin(&data_path, 32, 8);

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.final_prune = false;
        params.oom_temp_dir = dir.join("oom-artifacts");

        build_shear_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            16,
            index_dir.to_str().unwrap(),
            1,
            true,
            0.0,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oom_root_stop_after_d00_writes_partition_runs_without_mem_index() {
        let dir = unique_temp_dir("pipnn-oom-root-stop-d00");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 64, 8);

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.final_prune = false;
        params.oom_temp_dir = dir.join("oom-artifacts");
        params.root_stop_after_d00 = true;
        params.c_min = 2;
        params.c_max = 4;
        params.max_leaders = 8;
        params.fanout_top = 2;

        build_shear_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            16,
            index_dir.to_str().unwrap(),
            1,
            true,
            0.0,
            &params,
        )
        .unwrap();

        assert!(
            params.oom_temp_dir.join("partition_d00_runs.bin").exists(),
            "root stop should preserve depth-0 partition runs"
        );
        assert!(
            !index_dir.join("_mem.index").exists(),
            "root stop should exit before graph output"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oom_build_supports_final_prune_and_writes_loadable_mem_index() {
        let dir = unique_temp_dir("pipnn-oom-build-final-prune");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 48, 8);

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.final_prune = true;
        params.oom_temp_dir = dir.join("oom-artifacts");

        build_shear_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            16,
            index_dir.to_str().unwrap(),
            1,
            true,
            0.0,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());

        let index_write_parameters = IndexWriteParametersBuilder::new(16, 8)
            .with_num_threads(1)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            8,
            round_up(8, 32),
            48,
            false,
            0,
            false,
            0,
            1.3,
            index_write_parameters,
            0.0,
        );
        let mut index = InmemIndex::<f32>::new(config).unwrap();
        index
            .load_graph(&graph_path, 48)
            .expect("OOM build with final prune should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
