pub mod compact_replay;
pub(crate) mod compute_backend;
pub mod tessa_capture;
pub mod tessa_replay;
pub use compact_replay::{
    CudaCompactReplayBatch, CudaCompactReplayBatchMetadata, CudaCompactReplayBatchView,
    load_cuda_compact_replay_batch, load_cuda_compact_replay_batches,
    write_cuda_compact_replay_batch,
};
pub mod params;
pub use params::{
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
    PiPNNOnlineSuffixBudgets, PiPNNOnlineSuffixReduceMode, PiPNNOomIoMode, PiPNNParams,
    PiPNNPartitionerKind, PiPNNPipelineComputeMode, PiPNNPruneBackend, RootFanoutPolicy,
};
pub use root_risk_budget::RootRiskSigmaMode;

mod d0_capacity;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dco_assignment;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dco_leaf;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dco_rabitq;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dco_sq_int8;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dco_turboquant;
pub(crate) mod online_suffix_hashprune;
pub(crate) mod persistent_async_d2h;
pub(crate) mod range_reservoir_store;
pub(crate) mod reservoir_delta_log;
pub(crate) mod reservoir_writeback_cache;
mod sq4_primary;
mod gpu_sketch;
#[cfg(feature = "pipnn_cuda")]
mod sq4_gpu_d0;
pub(crate) mod sq8_rotated_residual_sidecar;
pub(crate) mod throughput_model;
pub use dco_sq_int8::SqInt8CpuKernel;
pub(crate) mod ddc_res;
pub mod direct_io;
pub(crate) mod disk_reservoir_store;
pub mod external_store;
pub(crate) mod flat_resident_reservoir;
pub(crate) mod hardware_adaptation;
pub mod hash_prune;
pub(crate) mod hash_prune_backfill;
pub(crate) mod hash_prune_diagnostics;
pub(crate) mod hash_prune_evidence;
pub mod kmeans_partition;
pub mod leaf_build;
pub(crate) mod log_assisted_reservoir;
pub mod lrpf;
pub mod lrpf_spill;
pub mod point_store;
pub(crate) mod qcode_staging;
pub use point_store::PointBatchStats;
pub mod rbc_partition;
pub mod root_opportunity_calibration;
pub mod root_radius_witness;
pub mod root_risk_budget;
mod scheduler;
pub mod spill_hashprune;

pub(crate) const STABLE_DISTANCE_TIE_EPS: f32 = 2.0e-5;
const PIPNN_RAYON_STACK_BYTES: usize = 32 * 1024 * 1024;
const GPU_SUBTREE_DEFAULT_ASYNC_FLUSH_WORKERS: usize = 8;
const RESERVOIR_INIT_WORKERS_ENV: &str = "SHEAR_PIPNN_RESERVOIR_INIT_WORKERS";
const RESERVOIR_LAYOUT_ENV: &str = "SHEAR_PIPNN_RESERVOIR_LAYOUT";
const DEFAULT_RESERVOIR_INIT_WORKERS: usize = 8;
const RESERVOIR_INIT_CHUNK_POINTS: usize = 256 * 1024;
const FINAL_EMIT_CHUNK_SOURCES: usize = 32 * 1024;
const FINAL_EMIT_FIFO_CHUNKS: usize = 2;
const FINAL_EMIT_INITIAL_WORKERS: usize = 8;
const FINAL_EMIT_SCALE_WINDOW_CHUNKS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum D0ReadTerminal {
    Running,
    Complete,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct D0ReadProgressState {
    ready_prefix: usize,
    initialized_prefix: usize,
    terminal: D0ReadTerminal,
    failure: Option<String>,
    wait_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct D0ReadProgressSnapshot {
    ready_prefix: usize,
    initialized_prefix: usize,
    terminal: D0ReadTerminal,
    wait_count: usize,
}

pub(crate) struct D0ReadProgress {
    target_prefix: usize,
    state: Mutex<D0ReadProgressState>,
    changed: Condvar,
}

impl D0ReadProgress {
    fn new(target_prefix: usize) -> Self {
        Self {
            target_prefix,
            state: Mutex::new(D0ReadProgressState {
                ready_prefix: 0,
                initialized_prefix: 0,
                terminal: D0ReadTerminal::Running,
                failure: None,
                wait_count: 0,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn advance(&self, ready_prefix: usize) {
        let mut state = self.state.lock();
        state.ready_prefix = state.ready_prefix.max(ready_prefix.min(self.target_prefix));
        self.changed.notify_all();
    }

    pub(crate) fn record_copied_shard(&self, row_start: usize, rows: usize) -> Result<(), String> {
        let mut state = self.state.lock();
        if state.ready_prefix >= self.target_prefix {
            return Ok(());
        }
        if state.terminal == D0ReadTerminal::Cancelled {
            return Err(state
                .failure
                .clone()
                .unwrap_or_else(|| "D0 raw read cancelled".to_string()));
        }
        if state.terminal == D0ReadTerminal::Complete {
            return Err("D0 raw read was already marked complete".to_string());
        }
        if row_start != state.ready_prefix {
            return Err(format!(
                "D0 raw read expected row_start {}, got {}",
                state.ready_prefix, row_start
            ));
        }
        state.ready_prefix = row_start.saturating_add(rows).min(self.target_prefix);
        self.changed.notify_all();
        Ok(())
    }

    pub(crate) fn complete(&self, ready_prefix: usize) {
        let mut state = self.state.lock();
        state.ready_prefix = state.ready_prefix.max(ready_prefix.min(self.target_prefix));
        state.terminal = D0ReadTerminal::Complete;
        self.changed.notify_all();
    }

    pub(crate) fn cancel(&self, reason: impl Into<String>) {
        let mut state = self.state.lock();
        if state.terminal != D0ReadTerminal::Complete {
            state.terminal = D0ReadTerminal::Cancelled;
            state.failure = Some(reason.into());
        }
        self.changed.notify_all();
    }

    fn wait_for_ready_prefix(&self, built_prefix: usize) -> Result<usize, String> {
        let mut state = self.state.lock();
        while state.ready_prefix <= built_prefix
            && state.terminal == D0ReadTerminal::Running
            && built_prefix < self.target_prefix
        {
            state.wait_count += 1;
            self.changed.notify_all();
            self.changed.wait(&mut state);
        }
        if state.terminal == D0ReadTerminal::Cancelled {
            return Err(state
                .failure
                .clone()
                .unwrap_or_else(|| "D0 raw read cancelled".to_string()));
        }
        if state.terminal == D0ReadTerminal::Complete && state.ready_prefix < self.target_prefix {
            return Err(format!(
                "D0 raw read completed at prefix {} before reservoir target {}",
                state.ready_prefix, self.target_prefix
            ));
        }
        Ok(state.ready_prefix)
    }

    fn record_initialized(&self, initialized_prefix: usize) {
        let mut state = self.state.lock();
        assert!(initialized_prefix <= state.ready_prefix);
        state.initialized_prefix = state.initialized_prefix.max(initialized_prefix);
        self.changed.notify_all();
    }

    fn snapshot(&self) -> D0ReadProgressSnapshot {
        let state = self.state.lock();
        D0ReadProgressSnapshot {
            ready_prefix: state.ready_prefix,
            initialized_prefix: state.initialized_prefix,
            terminal: state.terminal,
            wait_count: state.wait_count,
        }
    }

    #[cfg(test)]
    fn wait_until_waiting_for_test(&self) {
        let mut state = self.state.lock();
        while state.wait_count == 0 {
            self.changed.wait(&mut state);
        }
    }

    #[cfg(test)]
    fn wait_until_initialized_for_test(&self, prefix: usize) {
        let mut state = self.state.lock();
        while state.initialized_prefix < prefix {
            self.changed.wait(&mut state);
        }
    }
}

fn initial_final_emit_workers(host_parallelism: usize) -> usize {
    FINAL_EMIT_INITIAL_WORKERS.min(host_parallelism.max(1))
}

fn next_final_emit_workers(
    current: usize,
    host_parallelism: usize,
    finalize_wall: Duration,
    callback_wall: Duration,
) -> usize {
    let host_parallelism = host_parallelism.max(1);
    let current = current.max(1).min(host_parallelism);
    if finalize_wall.as_nanos() >= callback_wall.as_nanos().saturating_mul(5) / 4 {
        current.saturating_mul(2).min(host_parallelism)
    } else {
        current
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentReservoirLayout {
    Flat,
    Legacy,
}

impl ResidentReservoirLayout {
    fn from_override(configured: Option<&std::ffi::OsStr>) -> AnnResult<Self> {
        match configured.and_then(std::ffi::OsStr::to_str) {
            None if configured.is_none() => Ok(Self::Flat),
            Some("flat") => Ok(Self::Flat),
            Some("legacy") => Ok(Self::Legacy),
            Some(value) => Err(AnnError::log_index_error(format!(
                "unsupported {RESERVOIR_LAYOUT_ENV} value '{value}', expected flat or legacy"
            ))),
            None => Err(AnnError::log_index_error(format!(
                "{RESERVOIR_LAYOUT_ENV} must contain valid UTF-8"
            ))),
        }
    }

    fn from_env() -> AnnResult<Self> {
        let configured = std::env::var_os(RESERVOIR_LAYOUT_ENV);
        Self::from_override(configured.as_deref())
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Legacy => "legacy",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReservoirInitializationConfig {
    requested: String,
    overlap_enabled: bool,
    workers: usize,
    invalid_override: bool,
}

impl ReservoirInitializationConfig {
    fn d1_blocking_internal() -> Self {
        Self {
            requested: "<internal>".to_string(),
            overlap_enabled: false,
            workers: 1,
            invalid_override: false,
        }
    }

    fn initialization_mode(&self) -> &'static str {
        if self.overlap_enabled {
            "d0-overlap"
        } else if self.invalid_override {
            "d1-invalid-override"
        } else {
            "d1-blocking"
        }
    }

    fn allocation_phase(&self) -> &'static str {
        if self.overlap_enabled {
            "D0-overlap"
        } else {
            "D1"
        }
    }
}

fn parse_reservoir_initialization_config(
    configured: Option<&std::ffi::OsStr>,
    host_parallelism: usize,
) -> ReservoirInitializationConfig {
    let host_parallelism = host_parallelism.max(1);
    let default_workers = DEFAULT_RESERVOIR_INIT_WORKERS.min(host_parallelism);
    match configured {
        None => ReservoirInitializationConfig {
            requested: "<unset>".to_string(),
            overlap_enabled: true,
            workers: default_workers,
            invalid_override: false,
        },
        Some(value) => match value.to_str() {
            Some("0") => ReservoirInitializationConfig {
                requested: "0".to_string(),
                overlap_enabled: false,
                workers: 1,
                invalid_override: false,
            },
            Some(value) => match value.parse::<usize>() {
                Ok(workers) if workers > 0 => ReservoirInitializationConfig {
                    requested: value.to_string(),
                    overlap_enabled: true,
                    workers: workers.min(host_parallelism),
                    invalid_override: false,
                },
                _ => ReservoirInitializationConfig {
                    requested: "<invalid>".to_string(),
                    overlap_enabled: false,
                    workers: 1,
                    invalid_override: true,
                },
            },
            None => ReservoirInitializationConfig {
                requested: "<non-utf8>".to_string(),
                overlap_enabled: false,
                workers: 1,
                invalid_override: true,
            },
        },
    }
}

fn reservoir_initialization_config_from_env(pool_threads: usize) -> ReservoirInitializationConfig {
    let host_parallelism = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or_else(|_| pool_threads.max(1));
    let configured = std::env::var_os(RESERVOIR_INIT_WORKERS_ENV);
    parse_reservoir_initialization_config(configured.as_deref(), host_parallelism)
}

fn run_with_reservoir_initialization_overlap<T, I, P>(
    overlap_enabled: bool,
    d0_read_progress: Option<&D0ReadProgress>,
    initialize: I,
    pipeline: P,
) -> AnnResult<T>
where
    I: FnOnce() -> AnnResult<()> + Send,
    P: FnOnce() -> AnnResult<T>,
{
    if !overlap_enabled {
        return pipeline();
    }

    std::thread::scope(|scope| {
        let initializer = scope.spawn(initialize);
        let pipeline_result = pipeline();
        if pipeline_result.is_err() {
            if let Some(progress) = d0_read_progress {
                progress.cancel("D0/pipeline failed before reservoir initialization completed");
            }
        }
        let initializer_result = initializer.join();

        match pipeline_result {
            Err(error) => Err(error),
            Ok(value) => match initializer_result {
                Ok(Ok(())) => Ok(value),
                Ok(Err(error)) => Err(error),
                Err(_) => Err(AnnError::log_index_error(
                    "resident reservoir initialization worker panicked".to_string(),
                )),
            },
        }
    })
}

#[inline]
pub(crate) fn stable_distance_order(
    left_dist: f32,
    left_target: u32,
    right_dist: f32,
    right_target: u32,
) -> std::cmp::Ordering {
    if (left_dist - right_dist).abs() <= STABLE_DISTANCE_TIE_EPS {
        left_target.cmp(&right_target)
    } else {
        left_dist.total_cmp(&right_dist)
    }
}

#[inline]
pub(crate) fn uses_hashprune_candidate_backend(params: &PiPNNParams) -> bool {
    params.prune_backend == PiPNNPruneBackend::HashPrune
}

#[inline]
pub(crate) fn uses_lrpf_candidate_backend(params: &PiPNNParams) -> bool {
    params.prune_backend == PiPNNPruneBackend::LRPF
}

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::{Condvar, Mutex};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use self::compute_backend::{
    cpu_backend, resolve_compute_backend, validate_compute_backend_before_d0,
};
use self::dco_rabitq::{
    RABITQ_DEFAULT_K_ALIGN, RABITQ_PLUS_SIDECAR_META_ROW_BYTES, RabitqPlusSidecar,
    create_rabitq_plus_sidecar_from_point_store, create_rabitq_split_sidecar_from_point_store,
};
use self::dco_sq_int8::{
    SQ_INT8_DEFAULT_BITS, SQ_INT8_DEFAULT_K_ALIGN, SQ_WMMA_SIDECAR_META_ROW_BYTES,
    SqWmmaCodeLayout, SqWmmaSidecar, SqWmmaSidecarCreateReport, cache_sq_wmma_created_sidecar,
    create_sq_wmma_sidecar_from_point_store_with_bits_layout,
    create_sq_wmma_sidecar_from_u8_point_store_native_i8_with_resident_prefix,
    take_cached_sq_wmma_created_sidecar_with_bits,
};
use self::dco_turboquant::{
    TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES, TurboQuantMse4Sidecar,
    TurboQuantMse4SidecarCreateReport, create_turboquant_mse4_sidecar_from_point_store,
    create_turboquant_mse4_sidecar_from_point_store_cuda,
};
use self::direct_io::DirectIoConfig;
use self::disk_reservoir_store::{DiskReservoirLayout, DiskReservoirStoreConfig};
use self::external_store::{DiskSketchStore, ResidentPrefixSketchAccessor};
use self::flat_resident_reservoir::FlatResidentReservoirStore;
use self::hash_prune::{HashPruneReservoir, SketchAccessor, Slot};
use self::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use self::hash_prune_diagnostics::{
    HashPruneDiagnosticFinalizer, HashPruneDiagnosticsSession, HashPruneTraceCollector,
};
use self::hash_prune_evidence::{
    HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneEvidenceProfile,
};
use self::kmeans_partition::{KMeansPartitionTelemetry, kmeans_partition_streaming};
use self::leaf_build::{
    CompactTopKSink, FlatReservoirEdgeSink, LeafProfile, OomCompactTopKSink,
    OomLrpfCompactTopKSink, PendingEdgeSink, ReservoirEdgeSink, effective_lrpf_pre_spill_width,
    effective_lrpf_target_degree,
};
use self::lrpf::LrpfConfig;
use self::lrpf_spill::{
    ExternalLrpfReducer, LrpfFragmentSink, LrpfFragmentSpillSink, LrpfReduceStats, LrpfSpillStats,
};
use self::online_suffix_hashprune::{
    OnlineSuffixPipeline, OnlineSuffixReduceConfig, OnlineSuffixReduceStats,
};
use self::point_store::{
    DirectPointStore, InmemDatasetPointStore, PointStore, RawVectorCachePointStore, U8PointStore,
    raw_vector_cache_row_capacity, uring_gather_counters,
};
use self::rbc_partition::{
    AsyncCompactFlushWorkerPolicy, CompactFlushMode, GpuSubtreeRuntimeContext,
    GpuSubtreeStreamingProfile, RbcPartitionTelemetry, rbc_partition_streaming_with_backend,
    root_fanout_profile_json, root_radius_dry_run_json, run_gpu_subtree_d0_preflight_with_context,
    run_gpu_subtree_from_d0_plan_with_context, run_gpu_subtree_qcode_staging_preflight,
    run_gpu_subtree_sq_staging_preflight, run_gpu_subtree_sq_staging_preflight_with_sketch,
};
use self::scheduler::{
    LeafTaskContext, SchedulerBudget, SchedulerTelemetry, run_pipeline_scheduler_scope,
    run_scheduler_scope,
};
use self::spill_hashprune::{
    ExternalHashPruneReducer, ExternalReduceConfig, SpillEdgeSink, SpillReduceStats, SpillRunStats,
};
use self::sq8_rotated_residual_sidecar::Sq8RotatedResidualSidecar;
use crate::common::{AnnError, AnnResult};
use crate::model::InmemDataset;

#[derive(Clone, Debug, PartialEq)]
pub struct RabitqPlusSidecarBuildReport {
    pub rows: usize,
    pub dim: usize,
    pub padded_dim: usize,
    pub bits: usize,
    pub codes_bytes: usize,
    pub meta_rows_bytes: usize,
    pub source_fingerprint: u64,
    pub total_wall: Duration,
    pub chunk_alloc_wall: Duration,
    pub raw_read_wall: Duration,
    pub source_hash_wall: Duration,
    pub quantize_wall: Duration,
    pub code_output_wall: Duration,
    pub norm_wall: Duration,
    pub meta_output_wall: Duration,
    pub header_write_wall: Duration,
    pub write_wall: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RabitqSplitSidecarBuildReport {
    pub rows: usize,
    pub dim: usize,
    pub padded_dim: usize,
    pub bits: usize,
    pub ex_bits: usize,
    pub batch_count: usize,
    pub batch_bytes: usize,
    pub ex_row_bytes: usize,
    pub batch_total_bytes: usize,
    pub ex_total_bytes: usize,
    pub source_fingerprint: u64,
    pub total_wall: Duration,
    pub chunk_alloc_wall: Duration,
    pub raw_read_wall: Duration,
    pub source_hash_wall: Duration,
    pub quantize_wall: Duration,
    pub batch_output_wall: Duration,
    pub ex_output_wall: Duration,
    pub header_write_wall: Duration,
    pub write_wall: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqWmmaSidecarBuildReport {
    pub rows: usize,
    pub dim: usize,
    pub padded_dim: usize,
    pub bits: usize,
    pub code_layout: String,
    pub code_row_bytes: usize,
    pub codes_bytes: usize,
    pub meta_rows_bytes: usize,
    pub source_fingerprint: u64,
    pub total_wall: Duration,
    pub setup_wall: Duration,
    pub chunk_alloc_wall: Duration,
    pub raw_read_wall: Duration,
    pub source_hash_wall: Duration,
    pub quantize_wall: Duration,
    pub code_output_wall: Duration,
    pub meta_output_wall: Duration,
    pub flush_wall: Duration,
    pub header_write_wall: Duration,
    pub write_wall: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantMse4SidecarBuildReport {
    pub rows: usize,
    pub dim: usize,
    pub code_bytes_per_row: usize,
    pub bits: usize,
    pub seed: u64,
    pub codes_bytes: usize,
    pub meta_rows_bytes: usize,
    pub source_fingerprint: u64,
    pub total_wall: Duration,
    pub setup_wall: Duration,
    pub chunk_alloc_wall: Duration,
    pub raw_read_wall: Duration,
    pub source_hash_wall: Duration,
    pub quantize_wall: Duration,
    pub code_output_wall: Duration,
    pub meta_output_wall: Duration,
    pub flush_wall: Duration,
    pub header_write_wall: Duration,
    pub write_wall: Duration,
}

pub fn create_rabitq_plus_sidecar_from_fbin(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
    oom_io_mode: PiPNNOomIoMode,
) -> AnnResult<RabitqPlusSidecarBuildReport> {
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    let source_fbin_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat dense-i8 RabitQ+ sidecar source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    let report = create_rabitq_plus_sidecar_from_point_store(
        &store,
        prefix,
        bits,
        rotator,
        seed,
        source_fbin_bytes,
    )?;
    Ok(RabitqPlusSidecarBuildReport {
        rows: report.header.rows,
        dim: report.header.dim,
        padded_dim: report.header.padded_dim,
        bits: report.header.bits,
        codes_bytes: report.header.codes_bytes,
        meta_rows_bytes: report.header.meta_rows_bytes,
        source_fingerprint: report.header.source_fingerprint,
        total_wall: report.total_wall,
        chunk_alloc_wall: report.chunk_alloc_wall,
        raw_read_wall: report.raw_read_wall,
        source_hash_wall: report.source_hash_wall,
        quantize_wall: report.quantize_wall,
        code_output_wall: report.code_output_wall,
        norm_wall: report.norm_wall,
        meta_output_wall: report.meta_output_wall,
        header_write_wall: report.header_write_wall,
        write_wall: report.write_wall,
    })
}

pub fn create_rabitq_split_sidecar_from_fbin(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
    oom_io_mode: PiPNNOomIoMode,
) -> AnnResult<RabitqSplitSidecarBuildReport> {
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    let source_fbin_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat RabitQ-Library split sidecar source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    let report = create_rabitq_split_sidecar_from_point_store(
        &store,
        prefix,
        bits,
        rotator,
        seed,
        source_fbin_bytes,
    )?;
    Ok(RabitqSplitSidecarBuildReport {
        rows: report.header.rows,
        dim: report.header.dim,
        padded_dim: report.header.padded_dim,
        bits: report.header.bits,
        ex_bits: report.header.ex_bits,
        batch_count: report.header.batch_count,
        batch_bytes: report.header.batch_bytes,
        ex_row_bytes: report.header.ex_row_bytes,
        batch_total_bytes: report.header.batch_total_bytes,
        ex_total_bytes: report.header.ex_total_bytes,
        source_fingerprint: report.header.source_fingerprint,
        total_wall: report.total_wall,
        chunk_alloc_wall: report.chunk_alloc_wall,
        raw_read_wall: report.raw_read_wall,
        source_hash_wall: report.source_hash_wall,
        quantize_wall: report.quantize_wall,
        batch_output_wall: report.batch_output_wall,
        ex_output_wall: report.ex_output_wall,
        header_write_wall: report.header_write_wall,
        write_wall: report.write_wall,
    })
}

pub fn create_sq_wmma_sidecar_from_fbin(
    dataset_path: &Path,
    prefix: &Path,
    oom_io_mode: PiPNNOomIoMode,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    create_sq_wmma_sidecar_from_fbin_with_bits(
        dataset_path,
        prefix,
        SQ_INT8_DEFAULT_BITS,
        oom_io_mode,
    )
}

pub fn create_sq_wmma_sidecar_from_fbin_with_bits(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    oom_io_mode: PiPNNOomIoMode,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    create_sq_wmma_sidecar_from_fbin_with_bits_native_int4(
        dataset_path,
        prefix,
        bits,
        oom_io_mode,
        false,
    )
}

pub fn create_sq_wmma_sidecar_from_fbin_with_bits_native_int4(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    oom_io_mode: PiPNNOomIoMode,
    native_int4: bool,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    let source_fbin_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat SQ+WMMA sidecar source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    let layout = if native_int4 {
        SqWmmaCodeLayout::PackedS4
    } else {
        SqWmmaCodeLayout::DenseI8
    };
    let report = create_sq_wmma_sidecar_from_point_store_with_bits_layout(
        &store,
        prefix,
        source_fbin_bytes,
        bits,
        layout,
        0,
    )?
    .report;
    Ok(sq_wmma_build_report_from_create(report))
}

pub fn create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    oom_io_mode: PiPNNOomIoMode,
    native_int4: bool,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    validate_u8_sq_wmma_sidecar_mode(bits, native_int4)?;
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let source_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat SQ+WMMA sidecar uint8 source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    if bits == SQ_INT8_DEFAULT_BITS && !native_int4 {
        let store = U8PointStore::open_with_config(dataset_path, io_cfg)?;
        let report = create_sq_wmma_sidecar_from_u8_point_store_native_i8_with_resident_prefix(
            &store,
            prefix,
            source_bytes,
            0,
        )?
        .report;
        return Ok(sq_wmma_build_report_from_create(report));
    }

    let store = U8PointStore::open_centered_with_config(dataset_path, io_cfg)?;
    let layout = if native_int4 {
        SqWmmaCodeLayout::PackedS4
    } else {
        SqWmmaCodeLayout::DenseI8
    };
    let report = create_sq_wmma_sidecar_from_point_store_with_bits_layout(
        &store,
        prefix,
        source_bytes,
        bits,
        layout,
        0,
    )?
    .report;
    Ok(sq_wmma_build_report_from_create(report))
}

pub fn create_turboquant_mse4_sidecar_from_fbin(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    seed: u64,
    oom_io_mode: PiPNNOomIoMode,
) -> AnnResult<TurboQuantMse4SidecarBuildReport> {
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    let source_fbin_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat TurboQuant-MSE4 sidecar source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    let report = create_turboquant_mse4_sidecar_from_point_store(
        &store,
        prefix,
        bits,
        seed,
        source_fbin_bytes,
    )?;
    Ok(turboquant_mse4_build_report_from_create(report))
}

pub fn create_turboquant_mse4_sidecar_from_fbin_cuda(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    seed: u64,
    oom_io_mode: PiPNNOomIoMode,
    device_id: usize,
) -> AnnResult<TurboQuantMse4SidecarBuildReport> {
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    let source_fbin_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat TurboQuant-MSE4 sidecar source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    let report = create_turboquant_mse4_sidecar_from_point_store_cuda(
        &store,
        prefix,
        bits,
        seed,
        source_fbin_bytes,
        device_id,
    )?
    .report;
    Ok(turboquant_mse4_build_report_from_create(report))
}

pub fn create_sq_wmma_sidecar_from_fbin_with_resident_prefix(
    dataset_path: &Path,
    prefix: &Path,
    oom_io_mode: PiPNNOomIoMode,
    resident_rows: usize,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    create_sq_wmma_sidecar_from_fbin_with_bits_and_resident_prefix(
        dataset_path,
        prefix,
        SQ_INT8_DEFAULT_BITS,
        oom_io_mode,
        resident_rows,
    )
}

pub fn create_sq_wmma_sidecar_from_fbin_with_bits_and_resident_prefix(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    oom_io_mode: PiPNNOomIoMode,
    resident_rows: usize,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    create_sq_wmma_sidecar_from_fbin_with_bits_native_int4_and_resident_prefix(
        dataset_path,
        prefix,
        bits,
        oom_io_mode,
        resident_rows,
        false,
    )
}

pub fn create_sq_wmma_sidecar_from_fbin_with_bits_native_int4_and_resident_prefix(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    oom_io_mode: PiPNNOomIoMode,
    resident_rows: usize,
    native_int4: bool,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    let source_fbin_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat SQ+WMMA sidecar source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    let layout = if native_int4 {
        SqWmmaCodeLayout::PackedS4
    } else {
        SqWmmaCodeLayout::DenseI8
    };
    let artifact = create_sq_wmma_sidecar_from_point_store_with_bits_layout(
        &store,
        prefix,
        source_fbin_bytes,
        bits,
        layout,
        resident_rows,
    )?;
    if let Some(sidecar) = artifact.runtime_sidecar {
        cache_sq_wmma_created_sidecar(sidecar);
    }
    Ok(sq_wmma_build_report_from_create(artifact.report))
}

pub fn create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4_and_resident_prefix(
    dataset_path: &Path,
    prefix: &Path,
    bits: usize,
    oom_io_mode: PiPNNOomIoMode,
    resident_rows: usize,
    native_int4: bool,
) -> AnnResult<SqWmmaSidecarBuildReport> {
    validate_u8_sq_wmma_sidecar_mode(bits, native_int4)?;
    let io_cfg = match oom_io_mode {
        PiPNNOomIoMode::Direct => DirectIoConfig::enabled_with_alignment(4096),
        PiPNNOomIoMode::Auto | PiPNNOomIoMode::Buffered => DirectIoConfig::disabled(),
    };
    let source_bytes = fs::metadata(dataset_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to stat SQ+WMMA sidecar uint8 source '{}': {err}",
                dataset_path.display()
            ))
        })?
        .len();
    if bits == SQ_INT8_DEFAULT_BITS && !native_int4 {
        let store = U8PointStore::open_with_config(dataset_path, io_cfg)?;
        let artifact = create_sq_wmma_sidecar_from_u8_point_store_native_i8_with_resident_prefix(
            &store,
            prefix,
            source_bytes,
            resident_rows,
        )?;
        if let Some(sidecar) = artifact.runtime_sidecar {
            cache_sq_wmma_created_sidecar(sidecar);
        }
        return Ok(sq_wmma_build_report_from_create(artifact.report));
    }

    let store = U8PointStore::open_centered_with_config(dataset_path, io_cfg)?;
    let layout = if native_int4 {
        SqWmmaCodeLayout::PackedS4
    } else {
        SqWmmaCodeLayout::DenseI8
    };
    let artifact = create_sq_wmma_sidecar_from_point_store_with_bits_layout(
        &store,
        prefix,
        source_bytes,
        bits,
        layout,
        resident_rows,
    )?;
    if let Some(sidecar) = artifact.runtime_sidecar {
        cache_sq_wmma_created_sidecar(sidecar);
    }
    Ok(sq_wmma_build_report_from_create(artifact.report))
}

fn validate_u8_sq_wmma_sidecar_mode(bits: usize, native_int4: bool) -> AnnResult<()> {
    let native_sq8 = bits == SQ_INT8_DEFAULT_BITS && !native_int4;
    let native_sq4 = bits == 4 && native_int4;
    if native_sq8 || native_sq4 {
        return Ok(());
    }
    Err(AnnError::log_index_error(format!(
        "uint8 SQ+WMMA sidecar supports native SQ8 dense-i8 (bits=8 without native-int4) or native SQ4 packed-s4 (bits=4 with native-int4), got bits={bits} native_int4={native_int4}"
    )))
}

pub fn estimate_sq_wmma_sidecar_resident_rows(
    params: &PiPNNParams,
    rows: usize,
    dim: usize,
) -> usize {
    let memory_plan = OomMemoryPlan::from_params_for_dataset(params, rows, dim);
    if memory_plan.sidecar_resident_kind == OomSidecarResidentKind::SqWmma
        && memory_plan.sidecar_resident_admissible
    {
        memory_plan.sidecar_resident_admitted_rows
    } else {
        0
    }
}

fn sq_wmma_build_report_from_create(report: SqWmmaSidecarCreateReport) -> SqWmmaSidecarBuildReport {
    SqWmmaSidecarBuildReport {
        rows: report.header.rows,
        dim: report.header.dim,
        padded_dim: report.header.padded_dim,
        bits: report.header.bits,
        code_layout: report.header.code_layout.as_str().to_string(),
        code_row_bytes: report.header.code_row_bytes,
        codes_bytes: report.header.codes_bytes,
        meta_rows_bytes: report.header.meta_rows_bytes,
        source_fingerprint: report.header.source_fingerprint,
        total_wall: report.total_wall,
        setup_wall: report.setup_wall,
        chunk_alloc_wall: report.chunk_alloc_wall,
        raw_read_wall: report.raw_read_wall,
        source_hash_wall: report.source_hash_wall,
        quantize_wall: report.quantize_wall,
        code_output_wall: report.code_output_wall,
        meta_output_wall: report.meta_output_wall,
        flush_wall: report.flush_wall,
        header_write_wall: report.header_write_wall,
        write_wall: report.write_wall,
    }
}

fn turboquant_mse4_build_report_from_create(
    report: TurboQuantMse4SidecarCreateReport,
) -> TurboQuantMse4SidecarBuildReport {
    TurboQuantMse4SidecarBuildReport {
        rows: report.header.rows,
        dim: report.header.dim,
        code_bytes_per_row: report.header.code_bytes_per_row,
        bits: report.header.bits,
        seed: report.header.seed,
        codes_bytes: report.header.codes_bytes,
        meta_rows_bytes: report.header.meta_rows_bytes,
        source_fingerprint: report.header.source_fingerprint,
        total_wall: report.total_wall,
        setup_wall: report.setup_wall,
        chunk_alloc_wall: report.chunk_alloc_wall,
        raw_read_wall: report.raw_read_wall,
        source_hash_wall: report.source_hash_wall,
        quantize_wall: report.quantize_wall,
        code_output_wall: report.code_output_wall,
        meta_output_wall: report.meta_output_wall,
        flush_wall: report.flush_wall,
        header_write_wall: report.header_write_wall,
        write_wall: report.write_wall,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResidentReservoirPlan {
    resident_points: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OnlineSuffixResidencyPlan {
    pipeline_source_start: usize,
    pinned_prefix_points: usize,
    dense_resident_points: usize,
}

fn plan_online_suffix_residency(
    params: &PiPNNParams,
    num_points: usize,
    resident_budget_points: usize,
    online_suffix_enabled: bool,
) -> AnnResult<OnlineSuffixResidencyPlan> {
    if !online_suffix_enabled || !params.online_suffix_reduce_mode.uses_writeback_cache() {
        return Ok(OnlineSuffixResidencyPlan {
            pipeline_source_start: resident_budget_points,
            pinned_prefix_points: 0,
            dense_resident_points: resident_budget_points,
        });
    }
    let pinned_prefix_points = params
        .online_suffix_pinned_prefix_points
        .unwrap_or(resident_budget_points);
    if pinned_prefix_points > num_points || pinned_prefix_points > resident_budget_points {
        return Err(AnnError::log_index_config_error(
            "online_suffix_pinned_prefix_points".to_string(),
            format!(
                "requested {pinned_prefix_points} pinned sources exceeds the resident reservoir budget of {resident_budget_points} sources"
            ),
        ));
    }
    Ok(OnlineSuffixResidencyPlan {
        pipeline_source_start: 0,
        pinned_prefix_points,
        dense_resident_points: 0,
    })
}

fn should_use_sq_staging_fused_sketch(
    hashprune_candidate_backend: bool,
    lrpf_candidate_backend: bool,
    params: &PiPNNParams,
) -> bool {
    hashprune_candidate_backend
        && !lrpf_candidate_backend
        && matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree)
        && (params.gpu_subtree_sq_staging_mode.is_enabled()
            || (params.gpu_subtree_sq4_plus
                && (params.gpu_subtree_sq4_primary_center || rbc_partition::partition_gpu_d0_requested())
                && (params.sq4_budget_d0() || params.gpu_subtree_d0_partition_mode
                    == PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK)
                && params.m_hash_bits > 0
                && std::env::var("SHEAR_PIPNN_SQ4_D0_FUSED_SKETCH").as_deref() != Ok("0")))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OomSidecarResidentKind {
    #[default]
    None,
    RabitqPlus,
    SqWmma,
    TurboQuantMse4,
}

impl OomSidecarResidentKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::RabitqPlus => "rabitq_plus",
            Self::SqWmma => "sq_wmma",
            Self::TurboQuantMse4 => "turboquant_mse4",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OomSidecarResidentEstimate {
    kind: OomSidecarResidentKind,
    required_bytes: usize,
    rows: usize,
    row_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OomMemoryPlan {
    total_budget_bytes: usize,
    reservoir_budget_bytes: usize,
    online_suffix_enabled: bool,
    online_suffix_queue_budget_bytes: usize,
    online_suffix_state_cache_budget_bytes: usize,
    online_suffix_page_cache_budget_bytes: usize,
    online_suffix_writeback_budget_bytes: usize,
    online_suffix_spill_budget_bytes: usize,
    online_suffix_reorder_budget_bytes: usize,
    online_suffix_decode_budget_bytes: usize,
    online_suffix_read_window_budget_bytes: usize,
    online_suffix_delta_buffer_budget_bytes: usize,
    online_suffix_compaction_budget_bytes: usize,
    online_suffix_metadata_budget_bytes: usize,
    online_suffix_budget_bytes: usize,
    resident_subtree_budget_bytes: usize,
    required_resident_bytes: usize,
    optional_resident_budget_bytes: usize,
    optional_resident_admitted_bytes: usize,
    optional_resident_unassigned_bytes: usize,
    transient_pipeline_budget_bytes: usize,
    d1plus_working_set_budget_bytes: usize,
    sidecar_resident_kind: OomSidecarResidentKind,
    sidecar_resident_budget_bytes: usize,
    sidecar_resident_required_bytes: usize,
    sidecar_resident_admitted_rows: usize,
    sidecar_resident_admissible: bool,
    scratch_budget_bytes: usize,
    io_headroom_bytes: usize,
    raw_vector_cache_budget_bytes: usize,
}

const OOM_RESERVOIR_FLOOR_PERCENT: usize = 45;
const OOM_TRANSIENT_MIN_PERCENT: usize = 15;
const OOM_IO_HEADROOM_PERCENT: usize = 5;

impl OomMemoryPlan {
    fn from_budget_with_raw_cache(
        total_budget_bytes: usize,
        requested_raw_cache_bytes: usize,
        online_suffix_budgets: Option<PiPNNOnlineSuffixBudgets>,
        online_suffix_metadata_budget_bytes: usize,
        reservoir_required_bytes: usize,
        required_resident_bytes: usize,
        reservoir_budget_share: f64,
        sidecar_resident_budget_cap_bytes: Option<usize>,
        sidecar_estimate: OomSidecarResidentEstimate,
    ) -> Self {
        let reservoir_budget_share = reservoir_budget_share.clamp(0.0, 1.0);
        let scratch_budget_bytes =
            total_budget_bytes.saturating_mul(OOM_TRANSIENT_MIN_PERCENT) / 100;
        let io_headroom_bytes = total_budget_bytes.saturating_mul(OOM_IO_HEADROOM_PERCENT) / 100;
        let online_suffix_enabled = online_suffix_budgets.is_some();
        let online_suffix_queue_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.direct_queue_bytes)
            .unwrap_or(0);
        let online_suffix_state_cache_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.state_cache_bytes)
            .unwrap_or(0);
        let online_suffix_page_cache_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.state_page_cache_bytes)
            .unwrap_or(0);
        let online_suffix_writeback_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.state_writeback_bytes)
            .unwrap_or(0);
        let online_suffix_spill_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.spill_buffer_bytes)
            .unwrap_or(0);
        let online_suffix_reorder_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.reorder_window_bytes)
            .unwrap_or(0);
        let online_suffix_decode_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.decode_bytes)
            .unwrap_or(0);
        let online_suffix_read_window_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.read_window_bytes)
            .unwrap_or(0);
        let online_suffix_delta_buffer_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.delta_buffer_bytes)
            .unwrap_or(0);
        let online_suffix_compaction_budget_bytes = online_suffix_budgets
            .map(|budgets| budgets.compaction_bytes)
            .unwrap_or(0);
        let online_suffix_metadata_budget_bytes = if online_suffix_enabled {
            online_suffix_metadata_budget_bytes
        } else {
            0
        };
        let online_suffix_budget_bytes = online_suffix_budgets
            .map(PiPNNOnlineSuffixBudgets::total_bytes)
            .unwrap_or(0)
            .saturating_add(online_suffix_metadata_budget_bytes);
        let available_after_fixed = total_budget_bytes.saturating_sub(
            scratch_budget_bytes
                .saturating_add(io_headroom_bytes)
                .saturating_add(online_suffix_budget_bytes)
                .saturating_add(required_resident_bytes),
        );
        let raw_vector_cache_budget_bytes = requested_raw_cache_bytes.min(available_after_fixed);
        let d1plus_admission_bytes =
            available_after_fixed.saturating_sub(raw_vector_cache_budget_bytes);

        let uncapped_sidecar_resident_budget_bytes =
            sidecar_estimate.required_bytes.min(d1plus_admission_bytes);
        let sidecar_resident_budget_bytes = sidecar_resident_budget_cap_bytes
            .map(|cap| cap.min(uncapped_sidecar_resident_budget_bytes))
            .unwrap_or(uncapped_sidecar_resident_budget_bytes);
        let sidecar_resident_admitted_rows = if sidecar_estimate.required_bytes > 0 {
            let row_bytes = sidecar_estimate.row_bytes.max(1);
            (sidecar_resident_budget_bytes / row_bytes).min(sidecar_estimate.rows)
        } else {
            0
        };
        let optional_resident_admitted_bytes =
            sidecar_resident_admitted_rows.saturating_mul(sidecar_estimate.row_bytes);
        let available_after_sidecar =
            d1plus_admission_bytes.saturating_sub(optional_resident_admitted_bytes);
        let reservoir_target_bytes = if reservoir_required_bytes == 0 {
            scale_budget_by_share(
                total_budget_bytes.saturating_mul(OOM_RESERVOIR_FLOOR_PERCENT) / 100,
                reservoir_budget_share,
            )
        } else {
            scale_budget_by_share(reservoir_required_bytes, reservoir_budget_share)
        };
        let reservoir_budget_bytes = reservoir_target_bytes.min(available_after_sidecar);
        let sidecar_resident_admissible = optional_resident_admitted_bytes > 0;
        let optional_resident_budget_bytes = sidecar_resident_budget_bytes;
        let optional_resident_unassigned_bytes = 0;
        let resident_subtree_budget_bytes = sidecar_resident_budget_bytes;
        let transient_pipeline_budget_bytes = scratch_budget_bytes;
        let d1plus_working_set_budget_bytes = required_resident_bytes
            .saturating_add(reservoir_budget_bytes)
            .saturating_add(optional_resident_admitted_bytes);
        let allocated = reservoir_budget_bytes
            .saturating_add(required_resident_bytes)
            .saturating_add(optional_resident_admitted_bytes)
            .saturating_add(optional_resident_unassigned_bytes)
            .saturating_add(scratch_budget_bytes)
            .saturating_add(io_headroom_bytes)
            .saturating_add(online_suffix_budget_bytes)
            .saturating_add(raw_vector_cache_budget_bytes);
        let io_headroom_bytes =
            io_headroom_bytes.saturating_add(total_budget_bytes.saturating_sub(allocated));

        Self {
            total_budget_bytes,
            reservoir_budget_bytes,
            online_suffix_enabled,
            online_suffix_queue_budget_bytes,
            online_suffix_state_cache_budget_bytes,
            online_suffix_page_cache_budget_bytes,
            online_suffix_writeback_budget_bytes,
            online_suffix_spill_budget_bytes,
            online_suffix_reorder_budget_bytes,
            online_suffix_decode_budget_bytes,
            online_suffix_read_window_budget_bytes,
            online_suffix_delta_buffer_budget_bytes,
            online_suffix_compaction_budget_bytes,
            online_suffix_metadata_budget_bytes,
            online_suffix_budget_bytes,
            resident_subtree_budget_bytes,
            required_resident_bytes,
            optional_resident_budget_bytes,
            optional_resident_admitted_bytes,
            optional_resident_unassigned_bytes,
            transient_pipeline_budget_bytes,
            d1plus_working_set_budget_bytes,
            sidecar_resident_kind: sidecar_estimate.kind,
            sidecar_resident_budget_bytes,
            sidecar_resident_required_bytes: sidecar_estimate.required_bytes,
            sidecar_resident_admitted_rows,
            sidecar_resident_admissible,
            scratch_budget_bytes,
            io_headroom_bytes,
            raw_vector_cache_budget_bytes,
        }
    }

    #[cfg(test)]
    fn total_tracked_bytes(&self) -> usize {
        self.reservoir_budget_bytes
            .saturating_add(self.required_resident_bytes)
            .saturating_add(self.optional_resident_admitted_bytes)
            .saturating_add(self.optional_resident_unassigned_bytes)
            .saturating_add(self.scratch_budget_bytes)
            .saturating_add(self.io_headroom_bytes)
            .saturating_add(self.online_suffix_budget_bytes)
            .saturating_add(self.raw_vector_cache_budget_bytes)
    }

    #[cfg(test)]
    fn from_params(params: &PiPNNParams) -> Self {
        Self::from_params_for_dataset(params, 0, 0)
    }

    fn from_params_for_dataset(params: &PiPNNParams, rows: usize, dim: usize) -> Self {
        Self::from_params_for_dataset_with_reservoir_layout(
            params,
            rows,
            dim,
            ResidentReservoirLayout::Legacy,
        )
    }

    fn from_params_for_dataset_with_reservoir_layout(
        params: &PiPNNParams,
        rows: usize,
        dim: usize,
        reservoir_layout: ResidentReservoirLayout,
    ) -> Self {
        let online_suffix_budgets = matches!(
            online_suffix_eligibility_for_dataset(params, rows),
            OnlineSuffixEligibility::Enabled
        )
        .then(|| params.effective_online_suffix_budgets());
        let online_suffix_metadata_bytes = online_suffix_budgets
            .map(|budgets| {
                if params.online_suffix_reduce_mode.uses_log_assisted() {
                    let range_points = effective_log_assisted_range_points(
                        params.online_suffix_range_points,
                        rows.max(1),
                        params.l_max,
                    )
                    .unwrap_or(1);
                    log_assisted_metadata_budget_bytes(
                        rows,
                        range_points,
                        budgets.delta_budget_bytes,
                    )
                } else {
                    online_suffix_metadata_budget_bytes(
                        rows,
                        params.online_suffix_state_overprovision_ratio,
                    )
                }
            })
            .unwrap_or(0);
        Self::from_budget_with_raw_cache(
            params.effective_oom_memory_budget_bytes(),
            params.oom_raw_vector_cache_bytes,
            online_suffix_budgets,
            online_suffix_metadata_bytes,
            estimate_reservoir_bytes_for_layout(rows, params.l_max, reservoir_layout),
            if uses_hashprune_candidate_backend(params) {
                rows.saturating_mul(params.m_hash_bits)
                    .saturating_mul(size_of::<f32>())
            } else {
                0
            },
            params.effective_oom_reservoir_budget_share(),
            params.oom_sidecar_resident_budget_cap_bytes,
            estimate_oom_sidecar_resident(params, rows, dim),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OnlineSuffixEligibility {
    Disabled,
    Enabled,
    ExternalFallback(&'static str),
}

fn online_suffix_eligibility(params: &PiPNNParams) -> OnlineSuffixEligibility {
    if !params.online_suffix_reduce_mode.is_enabled() {
        return OnlineSuffixEligibility::Disabled;
    }
    if params.prune_backend != PiPNNPruneBackend::HashPrune {
        return OnlineSuffixEligibility::ExternalFallback("prune_backend_not_hashprune");
    }
    if params.partitioner != PiPNNPartitionerKind::GpuSubtree {
        return OnlineSuffixEligibility::ExternalFallback("partitioner_not_gpu_subtree");
    }
    if params.hashprune_evidence_mode == PiPNNHashPruneEvidenceMode::Enforce {
        return OnlineSuffixEligibility::ExternalFallback("evidence_enforce_not_supported");
    }
    if params.compact_spill_digest_enable {
        return OnlineSuffixEligibility::ExternalFallback("compact_spill_digest_changes_pool");
    }
    if params.root_stop_after_d00 {
        return OnlineSuffixEligibility::ExternalFallback("root_stop_has_no_suffix_reduce");
    }
    OnlineSuffixEligibility::Enabled
}

fn online_suffix_eligibility_for_dataset(
    params: &PiPNNParams,
    rows: usize,
) -> OnlineSuffixEligibility {
    let eligibility = online_suffix_eligibility(params);
    if eligibility != OnlineSuffixEligibility::Enabled
        || !params.online_suffix_reduce_mode.uses_log_assisted()
        || rows == 0
    {
        return eligibility;
    }
    let Ok(layout) = DiskReservoirLayout::new(params.l_max, 4096) else {
        return eligibility;
    };
    let base_working_set_bytes = rows.saturating_mul(layout.record_bytes);
    let delta_budget_bytes = params.effective_online_suffix_budgets().delta_budget_bytes;
    if base_working_set_bytes > delta_budget_bytes {
        return OnlineSuffixEligibility::ExternalFallback(
            "log_assisted_base_working_set_exceeds_delta_budget",
        );
    }
    eligibility
}

fn online_suffix_metadata_budget_bytes(points: usize, overprovision_ratio: f64) -> usize {
    if !overprovision_ratio.is_finite() || overprovision_ratio <= 1.0 {
        return usize::MAX;
    }
    let physical_slots_f64 = points as f64 * overprovision_ratio;
    if physical_slots_f64 > usize::MAX as f64 {
        return usize::MAX;
    }
    let physical_slots = physical_slots_f64.ceil() as usize;
    points
        .checked_mul(size_of::<u64>())
        .and_then(|bytes| bytes.checked_add(physical_slots.div_ceil(8)))
        .unwrap_or(usize::MAX)
}

const LOG_ASSISTED_TARGET_BASE_BYTES: usize = 128 * 1024 * 1024;
const LOG_ASSISTED_RANGE_METADATA_BYTES: usize = 4096;
const LOG_ASSISTED_ACTIVE_SOURCE_METADATA_BYTES: usize = 48;

fn effective_log_assisted_range_points(
    configured_points: usize,
    source_points: usize,
    l_max: usize,
) -> AnnResult<usize> {
    if configured_points != 0 {
        return Ok(configured_points);
    }
    let layout = DiskReservoirLayout::new(l_max, 4096)?;
    Ok((LOG_ASSISTED_TARGET_BASE_BYTES / layout.record_bytes)
        .max(1)
        .min(source_points.max(1)))
}

fn log_assisted_metadata_budget_bytes(
    points: usize,
    range_points: usize,
    delta_budget_bytes: usize,
) -> usize {
    let active_source_bound =
        points.min(delta_budget_bytes / reservoir_delta_log::DELTA_RECORD_BYTES);
    let range_count = points.div_ceil(range_points.max(1));
    active_source_bound
        .saturating_mul(LOG_ASSISTED_ACTIVE_SOURCE_METADATA_BYTES)
        .saturating_add(range_count.saturating_mul(LOG_ASSISTED_RANGE_METADATA_BYTES))
}

fn scale_budget_by_share(bytes: usize, share: f64) -> usize {
    if share <= 0.0 {
        0
    } else if share >= 1.0 {
        bytes
    } else {
        (bytes as f64 * share).round() as usize
    }
}

fn estimate_oom_sidecar_resident(
    params: &PiPNNParams,
    rows: usize,
    dim: usize,
) -> OomSidecarResidentEstimate {
    if rows == 0 || dim == 0 {
        return OomSidecarResidentEstimate::default();
    }
    if params.gpu_subtree_rabitq_plus_mode == PiPNNGpuSubtreeRabitqPlusMode::Build
        && params.gpu_subtree_rabitq_plus_sidecar_mode.is_enabled()
        && params.gpu_subtree_rabitq_plus_sidecar_mode
            != PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly
        && params.gpu_subtree_rabitq_plus_sidecar_prefix.is_some()
    {
        return OomSidecarResidentEstimate {
            kind: OomSidecarResidentKind::RabitqPlus,
            required_bytes: estimate_dense_i8_sidecar_resident_bytes(
                rows,
                dim,
                RABITQ_DEFAULT_K_ALIGN,
                RABITQ_PLUS_SIDECAR_META_ROW_BYTES,
            ),
            rows,
            row_bytes: estimate_dense_i8_sidecar_row_bytes(
                dim,
                RABITQ_DEFAULT_K_ALIGN,
                RABITQ_PLUS_SIDECAR_META_ROW_BYTES,
            ),
        };
    }
    if params.gpu_subtree_sq_wmma_mode == PiPNNGpuSubtreeSqWmmaMode::Build
        && !params.gpu_subtree_sq4_plus
        && params.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
        && params.gpu_subtree_sq_wmma_sidecar_mode != PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly
        && params.gpu_subtree_sq_wmma_sidecar_prefix.is_some()
    {
        return OomSidecarResidentEstimate {
            kind: OomSidecarResidentKind::SqWmma,
            required_bytes: estimate_dense_i8_sidecar_resident_bytes(
                rows,
                dim,
                SQ_INT8_DEFAULT_K_ALIGN,
                SQ_WMMA_SIDECAR_META_ROW_BYTES,
            ),
            rows,
            row_bytes: estimate_dense_i8_sidecar_row_bytes(
                dim,
                SQ_INT8_DEFAULT_K_ALIGN,
                SQ_WMMA_SIDECAR_META_ROW_BYTES,
            ),
        };
    }
    if params.gpu_subtree_turboquant_mode == PiPNNGpuSubtreeTurboQuantMode::Build
        && params.gpu_subtree_turboquant_sidecar_mode.is_enabled()
        && params.gpu_subtree_turboquant_sidecar_mode
            != PiPNNGpuSubtreeTurboQuantSidecarMode::CreateOnly
        && params.gpu_subtree_turboquant_sidecar_prefix.is_some()
    {
        let row_bytes = dim
            .div_ceil(2)
            .saturating_add(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES);
        return OomSidecarResidentEstimate {
            kind: OomSidecarResidentKind::TurboQuantMse4,
            required_bytes: rows.saturating_mul(row_bytes),
            rows,
            row_bytes,
        };
    }
    OomSidecarResidentEstimate::default()
}

fn maybe_preload_rabitq_plus_resident_sidecar(
    params: &PiPNNParams,
    dataset: &dyn PointStore,
    rows: usize,
    memory_plan: &OomMemoryPlan,
) -> AnnResult<Option<RabitqPlusSidecar>> {
    if !matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree)
        || !memory_plan.sidecar_resident_admissible
        || memory_plan.sidecar_resident_kind != OomSidecarResidentKind::RabitqPlus
    {
        return Ok(None);
    }
    let Some(prefix) = params.gpu_subtree_rabitq_plus_sidecar_prefix.as_ref() else {
        return Ok(None);
    };
    let sidecar = RabitqPlusSidecar::preload_prefix_validated(
        prefix,
        rows,
        dataset.dim(),
        params.gpu_subtree_rabitq_plus_bits,
        params.gpu_subtree_rabitq_plus_rotator,
        params.gpu_subtree_rabitq_plus_seed,
        memory_plan.sidecar_resident_admitted_rows,
    )?;
    info!(
        "PiPNN dense-i8 RabitQ+ resident sidecar preloaded: prefix={} resident_rows={} rows={} bytes={} elapsed={:?}",
        sidecar.prefix().display(),
        sidecar.resident_rows(),
        rows,
        sidecar.resident_bytes(),
        sidecar.preload_wall()
    );
    Ok(Some(sidecar))
}

fn maybe_open_sq_wmma_runtime_sidecar(
    params: &PiPNNParams,
    dataset: &dyn PointStore,
    rows: usize,
    memory_plan: &OomMemoryPlan,
) -> AnnResult<Option<SqWmmaSidecar>> {
    if !matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree)
        || params.gpu_subtree_sq4_plus
        || params.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
        || !params.gpu_subtree_sq_wmma_sidecar_mode.is_enabled()
        || params.gpu_subtree_sq_wmma_sidecar_mode == PiPNNGpuSubtreeSqWmmaSidecarMode::CreateOnly
    {
        return Ok(None);
    }
    let Some(prefix) = params.gpu_subtree_sq_wmma_sidecar_prefix.as_ref() else {
        return Ok(None);
    };
    let suffix_io_cfg = if params.oom_enable {
        DirectIoConfig::enabled_with_alignment(4096)
    } else {
        DirectIoConfig::disabled()
    };
    if let Some(sidecar) = take_cached_sq_wmma_created_sidecar_with_bits(
        prefix,
        rows,
        dataset.dim(),
        params.gpu_subtree_sq_bits,
    )? {
        let admitted_rows = memory_plan.sidecar_resident_admitted_rows;
        if memory_plan.sidecar_resident_admissible
            && memory_plan.sidecar_resident_kind == OomSidecarResidentKind::SqWmma
            && sidecar.resident_rows() <= admitted_rows
        {
            info!(
                "PiPNN SQ+WMMA resident sidecar reused from just-created runtime cache: prefix={} resident_rows={} rows={} bytes={}",
                sidecar.prefix().display(),
                sidecar.resident_rows(),
                rows,
                sidecar.resident_bytes(),
            );
            return Ok(Some(sidecar));
        }
        info!(
            "PiPNN SQ+WMMA just-created resident sidecar discarded because current memory plan admits {} rows: prefix={} cached_resident_rows={}",
            admitted_rows,
            sidecar.prefix().display(),
            sidecar.resident_rows(),
        );
    }
    let sidecar = if memory_plan.sidecar_resident_admissible
        && memory_plan.sidecar_resident_kind == OomSidecarResidentKind::SqWmma
    {
        let sidecar = SqWmmaSidecar::preload_prefix_validated_with_bits_and_suffix_io(
            prefix,
            rows,
            dataset.dim(),
            params.gpu_subtree_sq_bits,
            memory_plan.sidecar_resident_admitted_rows,
            suffix_io_cfg,
        )?;
        info!(
            "PiPNN SQ+WMMA resident sidecar preloaded: prefix={} resident_rows={} rows={} bytes={} suffix_direct_io={} elapsed={:?}",
            sidecar.prefix().display(),
            sidecar.resident_rows(),
            rows,
            sidecar.resident_bytes(),
            sidecar.uses_direct_io(),
            sidecar.preload_wall()
        );
        sidecar
    } else {
        let sidecar = SqWmmaSidecar::open_validated_with_bits_and_io(
            prefix,
            rows,
            dataset.dim(),
            params.gpu_subtree_sq_bits,
            suffix_io_cfg,
        )?;
        info!(
            "PiPNN SQ+WMMA sidecar opened for this fused D1+ build: prefix={} resident_admissible={} required_bytes={} budget_bytes={} direct_io={}",
            sidecar.prefix().display(),
            memory_plan.sidecar_resident_admissible,
            memory_plan.sidecar_resident_required_bytes,
            memory_plan.sidecar_resident_budget_bytes,
            sidecar.uses_direct_io(),
        );
        sidecar
    };
    Ok(Some(sidecar))
}

fn maybe_open_turboquant_runtime_sidecar(
    params: &PiPNNParams,
    dataset: &dyn PointStore,
    rows: usize,
    memory_plan: &OomMemoryPlan,
) -> AnnResult<Option<TurboQuantMse4Sidecar>> {
    if !matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree)
        || params.gpu_subtree_turboquant_mode != PiPNNGpuSubtreeTurboQuantMode::Build
        || !params.gpu_subtree_turboquant_sidecar_mode.is_enabled()
        || params.gpu_subtree_turboquant_sidecar_mode
            == PiPNNGpuSubtreeTurboQuantSidecarMode::CreateOnly
    {
        return Ok(None);
    }
    let Some(prefix) = params.gpu_subtree_turboquant_sidecar_prefix.as_ref() else {
        return Ok(None);
    };
    let sidecar = if memory_plan.sidecar_resident_admissible
        && memory_plan.sidecar_resident_kind == OomSidecarResidentKind::TurboQuantMse4
    {
        let sidecar = TurboQuantMse4Sidecar::preload_prefix_validated(
            prefix,
            rows,
            dataset.dim(),
            params.gpu_subtree_turboquant_bits,
            params.gpu_subtree_turboquant_seed,
            memory_plan.sidecar_resident_admitted_rows,
        )?;
        info!(
            "PiPNN TurboQuant-MSE4 resident sidecar preloaded: prefix={} resident_rows={} rows={} bytes={} elapsed={:?}",
            sidecar.prefix().display(),
            sidecar.resident_rows(),
            rows,
            sidecar.resident_bytes(),
            sidecar.preload_wall()
        );
        sidecar
    } else {
        let sidecar = TurboQuantMse4Sidecar::open_validated(
            prefix,
            rows,
            dataset.dim(),
            params.gpu_subtree_turboquant_bits,
            params.gpu_subtree_turboquant_seed,
        )?;
        info!(
            "PiPNN TurboQuant-MSE4 sidecar opened for this fused D1+ build: prefix={} resident_admissible={} required_bytes={} budget_bytes={}",
            sidecar.prefix().display(),
            memory_plan.sidecar_resident_admissible,
            memory_plan.sidecar_resident_required_bytes,
            memory_plan.sidecar_resident_budget_bytes,
        );
        sidecar
    };
    Ok(Some(sidecar))
}

fn estimate_dense_i8_sidecar_row_bytes(dim: usize, k_align: usize, meta_row_bytes: usize) -> usize {
    let align = k_align.max(1);
    let padded_dim = dim.saturating_add(align - 1) / align * align;
    padded_dim.saturating_add(meta_row_bytes)
}

fn estimate_dense_i8_sidecar_resident_bytes(
    rows: usize,
    dim: usize,
    k_align: usize,
    meta_row_bytes: usize,
) -> usize {
    rows.saturating_mul(estimate_dense_i8_sidecar_row_bytes(
        dim,
        k_align,
        meta_row_bytes,
    ))
}

struct HybridEdgeSink<'a> {
    resident_points: usize,
    resident_sink: &'a dyn PendingEdgeSink,
    suffix_sink: &'a dyn PendingEdgeSink,
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
}

enum ResidentReservoirStorage {
    Legacy(Vec<Mutex<HashPruneReservoir>>),
    Flat(FlatResidentReservoirStore),
}

impl ResidentReservoirStorage {
    fn len(&self) -> usize {
        match self {
            Self::Legacy(reservoirs) => reservoirs.len(),
            Self::Flat(store) => store.len(),
        }
    }

    fn layout(&self) -> ResidentReservoirLayout {
        match self {
            Self::Legacy(_) => ResidentReservoirLayout::Legacy,
            Self::Flat(_) => ResidentReservoirLayout::Flat,
        }
    }

    #[cfg(test)]
    fn iter(&self) -> std::slice::Iter<'_, Mutex<HashPruneReservoir>> {
        match self {
            Self::Legacy(reservoirs) => reservoirs.iter(),
            Self::Flat(_) => panic!("flat resident storage has no per-source reservoir iterator"),
        }
    }

    #[cfg(test)]
    fn into_legacy_for_test(self) -> Vec<Mutex<HashPruneReservoir>> {
        match self {
            Self::Legacy(reservoirs) => reservoirs,
            Self::Flat(_) => panic!("expected legacy resident storage in test"),
        }
    }
}

struct ResidentReservoirAllocation {
    reservoirs: ResidentReservoirStorage,
    estimated_bytes: usize,
    allocation_wall: Duration,
    initialization: ReservoirInitializationProfile,
}

#[derive(Clone, Debug)]
struct ReservoirInitializationProfile {
    requested: String,
    layout: &'static str,
    mode: &'static str,
    allocation_phase: &'static str,
    workers: usize,
    overlap_enabled: bool,
    invalid_override: bool,
    parallel_fallback: bool,
    d0_paced: bool,
    ready_prefix: usize,
    initialized_prefix: usize,
    frontier_wait_count: usize,
    frontier_wait_wall: Duration,
    allocation_wall: Duration,
    d1_wait_wall: Duration,
    mapped_bytes: usize,
    segment_count: usize,
    stripe_count: usize,
    scratch_bytes: usize,
}

impl Default for ReservoirInitializationProfile {
    fn default() -> Self {
        Self {
            requested: "<not-applicable>".to_string(),
            layout: "not-applicable",
            mode: "disabled",
            allocation_phase: "D1",
            workers: 0,
            overlap_enabled: false,
            invalid_override: false,
            parallel_fallback: false,
            d0_paced: false,
            ready_prefix: 0,
            initialized_prefix: 0,
            frontier_wait_count: 0,
            frontier_wait_wall: Duration::ZERO,
            allocation_wall: Duration::ZERO,
            d1_wait_wall: Duration::ZERO,
            mapped_bytes: 0,
            segment_count: 0,
            stripe_count: 0,
            scratch_bytes: 0,
        }
    }
}

impl ReservoirInitializationProfile {
    fn hidden_wall(&self) -> Duration {
        self.allocation_wall
            .saturating_sub(self.d1_wait_wall.min(self.allocation_wall))
    }
}

fn allocate_resident_reservoirs_with_pool_builder<B>(
    points: usize,
    l_max: usize,
    config: &ReservoirInitializationConfig,
    build_pool: B,
) -> ResidentReservoirAllocation
where
    B: FnOnce(usize) -> Result<rayon::ThreadPool, String>,
{
    let started = Instant::now();
    let mut parallel_fallback = false;
    let mut actual_workers = config.workers.max(1);
    let reservoirs = if actual_workers == 1 {
        (0..points)
            .map(|_| Mutex::new(HashPruneReservoir::new(l_max)))
            .collect()
    } else {
        match build_pool(actual_workers) {
            Ok(pool) => pool.install(|| {
                (0..points)
                    .into_par_iter()
                    .map(|_| Mutex::new(HashPruneReservoir::new(l_max)))
                    .collect()
            }),
            Err(error) => {
                warn!(
                    "PiPNN reservoir initialization pool failed; using sequential fallback: {}",
                    error
                );
                actual_workers = 1;
                parallel_fallback = true;
                (0..points)
                    .map(|_| Mutex::new(HashPruneReservoir::new(l_max)))
                    .collect()
            }
        }
    };
    let allocation_wall = started.elapsed();
    ResidentReservoirAllocation {
        reservoirs: ResidentReservoirStorage::Legacy(reservoirs),
        estimated_bytes: estimate_reservoir_bytes(points, l_max),
        allocation_wall,
        initialization: ReservoirInitializationProfile {
            requested: config.requested.clone(),
            layout: ResidentReservoirLayout::Legacy.as_str(),
            mode: config.initialization_mode(),
            allocation_phase: config.allocation_phase(),
            workers: actual_workers,
            overlap_enabled: config.overlap_enabled,
            invalid_override: config.invalid_override,
            parallel_fallback,
            d0_paced: false,
            ready_prefix: 0,
            initialized_prefix: 0,
            frontier_wait_count: 0,
            frontier_wait_wall: Duration::ZERO,
            allocation_wall,
            d1_wait_wall: Duration::ZERO,
            mapped_bytes: 0,
            segment_count: 0,
            stripe_count: 0,
            scratch_bytes: 0,
        },
    }
}

fn allocate_resident_reservoirs(
    points: usize,
    l_max: usize,
    config: &ReservoirInitializationConfig,
) -> ResidentReservoirAllocation {
    allocate_resident_reservoirs_with_pool_builder(points, l_max, config, |workers| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("reservoir-init-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
}

fn allocate_flat_resident_reservoirs(
    points: usize,
    l_max: usize,
    config: &ReservoirInitializationConfig,
) -> AnnResult<ResidentReservoirAllocation> {
    let started = Instant::now();
    let store = FlatResidentReservoirStore::new(points, l_max)?;
    let estimated_bytes = store.estimated_bytes();
    let mapped_bytes = store.mapped_bytes();
    let segment_count = store.segment_count();
    let stripe_count = store.stripe_count();
    let scratch_bytes = store.scratch_bytes();
    let allocation_wall = started.elapsed();
    Ok(ResidentReservoirAllocation {
        reservoirs: ResidentReservoirStorage::Flat(store),
        estimated_bytes,
        allocation_wall,
        initialization: ReservoirInitializationProfile {
            requested: config.requested.clone(),
            layout: ResidentReservoirLayout::Flat.as_str(),
            mode: "flat-mmap",
            allocation_phase: config.allocation_phase(),
            workers: 1,
            overlap_enabled: config.overlap_enabled,
            invalid_override: config.invalid_override,
            parallel_fallback: false,
            d0_paced: false,
            ready_prefix: 0,
            initialized_prefix: points,
            frontier_wait_count: 0,
            frontier_wait_wall: Duration::ZERO,
            allocation_wall,
            d1_wait_wall: Duration::ZERO,
            mapped_bytes,
            segment_count,
            stripe_count,
            scratch_bytes,
        },
    })
}

fn build_reservoir_interval(
    start: usize,
    end: usize,
    l_max: usize,
    pool: Option<&rayon::ThreadPool>,
) -> Vec<Mutex<HashPruneReservoir>> {
    match pool {
        Some(pool) => pool.install(|| {
            (start..end)
                .into_par_iter()
                .map(|_| Mutex::new(HashPruneReservoir::new(l_max)))
                .collect()
        }),
        None => (start..end)
            .map(|_| Mutex::new(HashPruneReservoir::new(l_max)))
            .collect(),
    }
}

fn allocate_resident_reservoirs_paced_with_pool_builder<B>(
    points: usize,
    l_max: usize,
    config: &ReservoirInitializationConfig,
    progress: &D0ReadProgress,
    build_pool: B,
) -> Result<ResidentReservoirAllocation, String>
where
    B: FnOnce(usize) -> Result<rayon::ThreadPool, String>,
{
    let started = Instant::now();
    let mut parallel_fallback = false;
    let mut actual_workers = config.workers.max(1);
    let pool = if actual_workers == 1 {
        None
    } else {
        match build_pool(actual_workers) {
            Ok(pool) => Some(pool),
            Err(error) => {
                warn!(
                    "PiPNN paced reservoir initialization pool failed; using sequential fallback: {}",
                    error
                );
                actual_workers = 1;
                parallel_fallback = true;
                None
            }
        }
    };

    let mut reservoirs = Vec::with_capacity(points);
    let mut frontier_wait_wall = Duration::ZERO;
    while reservoirs.len() < points {
        let wait_started = Instant::now();
        let ready_prefix = progress.wait_for_ready_prefix(reservoirs.len())?;
        frontier_wait_wall += wait_started.elapsed();
        let end = ready_prefix
            .min(reservoirs.len().saturating_add(RESERVOIR_INIT_CHUNK_POINTS))
            .min(points);
        let interval = build_reservoir_interval(reservoirs.len(), end, l_max, pool.as_ref());
        reservoirs.extend(interval);
        progress.record_initialized(reservoirs.len());
    }

    let snapshot = progress.snapshot();
    let allocation_wall = started.elapsed();
    Ok(ResidentReservoirAllocation {
        reservoirs: ResidentReservoirStorage::Legacy(reservoirs),
        estimated_bytes: estimate_reservoir_bytes(points, l_max),
        allocation_wall,
        initialization: ReservoirInitializationProfile {
            requested: config.requested.clone(),
            layout: ResidentReservoirLayout::Legacy.as_str(),
            mode: config.initialization_mode(),
            allocation_phase: config.allocation_phase(),
            workers: actual_workers,
            overlap_enabled: config.overlap_enabled,
            invalid_override: config.invalid_override,
            parallel_fallback,
            d0_paced: true,
            ready_prefix: snapshot.ready_prefix,
            initialized_prefix: snapshot.initialized_prefix,
            frontier_wait_count: snapshot.wait_count,
            frontier_wait_wall,
            allocation_wall,
            d1_wait_wall: Duration::ZERO,
            mapped_bytes: 0,
            segment_count: 0,
            stripe_count: 0,
            scratch_bytes: 0,
        },
    })
}

fn allocate_resident_reservoirs_paced(
    points: usize,
    l_max: usize,
    config: &ReservoirInitializationConfig,
    progress: &D0ReadProgress,
) -> Result<ResidentReservoirAllocation, String> {
    allocate_resident_reservoirs_paced_with_pool_builder(
        points,
        l_max,
        config,
        progress,
        |workers| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|index| format!("reservoir-init-{index}"))
                .build()
                .map_err(|error| error.to_string())
        },
    )
}

enum ResidentReservoirInitializationOutcome {
    Ready(ResidentReservoirAllocation),
    Failed(String),
}

struct DeferredReservoirs {
    points: usize,
    l_max: usize,
    layout: ResidentReservoirLayout,
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostics: Option<Arc<HashPruneTraceCollector>>,
    initialization_config: ReservoirInitializationConfig,
    d0_read_progress: Option<Arc<D0ReadProgress>>,
    d1_wait_wall: Mutex<Duration>,
    #[cfg(test)]
    allocation_count: AtomicUsize,
    allocation: OnceLock<ResidentReservoirInitializationOutcome>,
}

impl DeferredReservoirs {
    #[cfg(test)]
    fn new(points: usize, l_max: usize) -> Self {
        Self::new_with_policies(
            points,
            l_max,
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
    }

    fn new_with_policy(points: usize, l_max: usize, policy: HashPruneEvidencePolicy) -> Self {
        Self::new_with_policies(points, l_max, policy, HashPruneBackfillPolicy::off())
    }

    fn new_with_policies(
        points: usize,
        l_max: usize,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> Self {
        Self::new_with_policies_and_diagnostics(points, l_max, policy, backfill_policy, None)
    }

    fn new_with_policies_and_diagnostics(
        points: usize,
        l_max: usize,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostics: Option<Arc<HashPruneTraceCollector>>,
    ) -> Self {
        debug_assert!(policy.is_off() || backfill_policy.is_off());
        Self {
            points,
            l_max,
            layout: ResidentReservoirLayout::Legacy,
            policy,
            backfill_policy,
            diagnostics,
            initialization_config: ReservoirInitializationConfig::d1_blocking_internal(),
            d0_read_progress: None,
            d1_wait_wall: Mutex::new(Duration::ZERO),
            #[cfg(test)]
            allocation_count: AtomicUsize::new(0),
            allocation: OnceLock::new(),
        }
    }

    fn with_initialization_config(
        mut self,
        initialization_config: ReservoirInitializationConfig,
    ) -> Self {
        debug_assert!(self.allocation.get().is_none());
        self.initialization_config = initialization_config;
        self
    }

    fn with_layout(mut self, layout: ResidentReservoirLayout) -> Self {
        debug_assert!(self.allocation.get().is_none());
        self.layout = layout;
        self
    }

    fn with_d0_read_progress(mut self, progress: Arc<D0ReadProgress>) -> Self {
        debug_assert!(self.allocation.get().is_none());
        self.d0_read_progress = Some(progress);
        self
    }

    fn initialize(&self) -> AnnResult<()> {
        self.allocation().map(|_| ())
    }

    fn allocation(&self) -> AnnResult<&ResidentReservoirAllocation> {
        let outcome = self.allocation.get_or_init(|| {
            #[cfg(test)]
            self.allocation_count.fetch_add(1, Ordering::SeqCst);
            let allocation = match self.layout {
                ResidentReservoirLayout::Flat => allocate_flat_resident_reservoirs(
                    self.points,
                    self.l_max,
                    &self.initialization_config,
                )
                .map_err(|error| error.to_string()),
                ResidentReservoirLayout::Legacy => match self.d0_read_progress.as_deref() {
                    Some(progress) => allocate_resident_reservoirs_paced(
                        self.points,
                        self.l_max,
                        &self.initialization_config,
                        progress,
                    ),
                    None => Ok(allocate_resident_reservoirs(
                        self.points,
                        self.l_max,
                        &self.initialization_config,
                    )),
                },
            };
            match allocation {
                Ok(allocation) => {
                    info!(
                        "PiPNN reservoir initialization done: points={} estimated_bytes={} layout={} mapped_bytes={} segments={} stripes={} scratch_bytes={} mode={} workers={} parallel_fallback={} d0_paced={} ready_prefix={} initialized_prefix={} frontier_wait_count={} frontier_wait={:?} elapsed={:?}",
                        self.points,
                        allocation.estimated_bytes,
                        allocation.initialization.layout,
                        allocation.initialization.mapped_bytes,
                        allocation.initialization.segment_count,
                        allocation.initialization.stripe_count,
                        allocation.initialization.scratch_bytes,
                        allocation.initialization.mode,
                        allocation.initialization.workers,
                        allocation.initialization.parallel_fallback,
                        allocation.initialization.d0_paced,
                        allocation.initialization.ready_prefix,
                        allocation.initialization.initialized_prefix,
                        allocation.initialization.frontier_wait_count,
                        allocation.initialization.frontier_wait_wall,
                        allocation.allocation_wall,
                    );
                    ResidentReservoirInitializationOutcome::Ready(allocation)
                }
                Err(error) => {
                    warn!("PiPNN paced reservoir initialization failed: {}", error);
                    ResidentReservoirInitializationOutcome::Failed(error)
                }
            }
        });
        match outcome {
            ResidentReservoirInitializationOutcome::Ready(allocation) => Ok(allocation),
            ResidentReservoirInitializationOutcome::Failed(error) => {
                Err(AnnError::log_index_error(error.clone()))
            }
        }
    }

    fn into_allocation(self) -> AnnResult<ResidentReservoirAllocation> {
        let d1_wait_wall = self.d1_wait_wall.into_inner();
        let outcome = self.allocation.into_inner().ok_or_else(|| {
            AnnError::log_index_error(
                "D1 reservoir storage was not initialized before graph emission".to_string(),
            )
        })?;
        let mut allocation = match outcome {
            ResidentReservoirInitializationOutcome::Ready(allocation) => allocation,
            ResidentReservoirInitializationOutcome::Failed(error) => {
                return Err(AnnError::log_index_error(error));
            }
        };
        allocation.initialization.d1_wait_wall = d1_wait_wall;
        Ok(allocation)
    }

    #[cfg(test)]
    fn is_initialized(&self) -> bool {
        matches!(
            self.allocation.get(),
            Some(ResidentReservoirInitializationOutcome::Ready(_))
        )
    }

    #[cfg(test)]
    fn has_ready_allocation_for_test(&self) -> bool {
        self.is_initialized()
    }

    #[cfg(test)]
    fn allocation_count_for_test(&self) -> usize {
        self.allocation_count.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentEmitConfig {
    chunk_sources: usize,
    fifo_chunks: usize,
    initial_workers: usize,
    host_parallelism: usize,
    scale_window_chunks: usize,
}

impl ResidentEmitConfig {
    fn production() -> Self {
        let host_parallelism = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(FINAL_EMIT_INITIAL_WORKERS);
        Self {
            chunk_sources: FINAL_EMIT_CHUNK_SOURCES,
            fifo_chunks: FINAL_EMIT_FIFO_CHUNKS,
            initial_workers: initial_final_emit_workers(host_parallelism),
            host_parallelism,
            scale_window_chunks: FINAL_EMIT_SCALE_WINDOW_CHUNKS,
        }
    }

    #[cfg(test)]
    fn for_test(
        chunk_sources: usize,
        fifo_chunks: usize,
        initial_workers: usize,
        host_parallelism: usize,
    ) -> Self {
        let host_parallelism = host_parallelism.max(1);
        Self {
            chunk_sources: chunk_sources.max(1),
            fifo_chunks: fifo_chunks.max(1),
            initial_workers: initial_workers.max(1).min(host_parallelism),
            host_parallelism,
            scale_window_chunks: FINAL_EMIT_SCALE_WINDOW_CHUNKS,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ResidentEmitProfile {
    parallel: bool,
    chunk_sources: usize,
    fifo_chunks: usize,
    initial_workers: usize,
    peak_workers: usize,
    chunks: usize,
    rows: usize,
    max_completed_inflight: usize,
    finalize_service_wall: Duration,
    callback_service_wall: Duration,
    consumer_wait_wall: Duration,
    producer_backpressure_wall: Duration,
    writer_buffer_bytes: usize,
}

#[derive(Debug)]
struct ResidentEmitResult {
    backfill: HashPruneBackfillProfile,
    telemetry: ResidentEmitProfile,
}

struct ResidentEmitRow {
    source: u32,
    neighbors: Vec<u32>,
}

struct ResidentEmitPartition {
    rows: Vec<ResidentEmitRow>,
    backfill: HashPruneBackfillProfile,
    error: Option<AnnError>,
}

struct ResidentEmitChunk {
    first_source: usize,
    rows: Vec<ResidentEmitRow>,
    backfill: HashPruneBackfillProfile,
    error: Option<AnnError>,
    finalize_wall: Duration,
}

fn emit_resident_reservoirs_serial<F>(
    point_start: usize,
    reservoirs: Vec<Mutex<HashPruneReservoir>>,
    policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_node_done: &mut F,
) -> AnnResult<ResidentEmitResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let mut profile = HashPruneBackfillProfile::default();
    let mut telemetry = ResidentEmitProfile {
        writer_buffer_bytes: crate::index::DirectMemGraphWriter::BUFFER_BYTES,
        ..ResidentEmitProfile::default()
    };
    for (offset, reservoir) in reservoirs.into_iter().enumerate() {
        let point = point_start + offset;
        let finalize_start = Instant::now();
        let row = reservoir.into_inner().into_finalized_slots_with_backfill(
            point,
            policy,
            &mut profile,
        )?;
        if let (Some(collector), Some(finalizer)) =
            (diagnostic_collector, diagnostic_finalizer.as_deref_mut())
        {
            finalizer.observe_row(point, collector, &row)?;
        }
        let neighbors = row.slots.into_iter().map(|slot| slot.target).collect();
        telemetry.finalize_service_wall += finalize_start.elapsed();
        let callback_start = Instant::now();
        on_node_done(point as u32, neighbors)?;
        telemetry.callback_service_wall += callback_start.elapsed();
        telemetry.rows += 1;
    }
    telemetry.chunks = usize::from(telemetry.rows > 0);
    Ok(ResidentEmitResult {
        backfill: profile,
        telemetry,
    })
}

fn emit_flat_resident_reservoirs_serial<F>(
    point_start: usize,
    mut store: FlatResidentReservoirStore,
    policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_node_done: &mut F,
) -> AnnResult<ResidentEmitResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let mut profile = HashPruneBackfillProfile::default();
    let mut telemetry = ResidentEmitProfile {
        writer_buffer_bytes: crate::index::DirectMemGraphWriter::BUFFER_BYTES,
        ..ResidentEmitProfile::default()
    };
    let points = store.len();
    let segment_rows = store.segment_rows();
    for offset in 0..points {
        let point = point_start + offset;
        let finalize_start = Instant::now();
        let row = store.finalize_row_with_logical_source(offset, point, policy, &mut profile)?;
        if let (Some(collector), Some(finalizer)) =
            (diagnostic_collector, diagnostic_finalizer.as_deref_mut())
        {
            finalizer.observe_row(point, collector, &row)?;
        }
        let neighbors = row.slots.into_iter().map(|slot| slot.target).collect();
        telemetry.finalize_service_wall += finalize_start.elapsed();
        let callback_start = Instant::now();
        on_node_done(point as u32, neighbors)?;
        telemetry.callback_service_wall += callback_start.elapsed();
        telemetry.rows += 1;
        let next_offset = offset + 1;
        if next_offset % segment_rows == 0 || next_offset == points {
            store.release_segments_before(next_offset);
        }
    }
    telemetry.chunks = usize::from(telemetry.rows > 0);
    Ok(ResidentEmitResult {
        backfill: profile,
        telemetry,
    })
}

fn build_final_emit_pool(workers: usize) -> AnnResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .thread_name(|index| format!("resident-finalize-{index}"))
        .build()
        .map_err(|error| {
            AnnError::log_index_error(format!(
                "failed to create resident finalization pool: {error}"
            ))
        })
}

fn finalize_resident_partition(
    point_start: usize,
    reservoirs: Vec<Mutex<HashPruneReservoir>>,
    policy: HashPruneBackfillPolicy,
) -> ResidentEmitPartition {
    let mut rows = Vec::with_capacity(reservoirs.len());
    let mut backfill = HashPruneBackfillProfile::default();
    let mut error = None;
    for (offset, reservoir) in reservoirs.into_iter().enumerate() {
        let source = point_start + offset;
        match reservoir.into_inner().into_finalized_slots_with_backfill(
            source,
            policy,
            &mut backfill,
        ) {
            Ok(row) => rows.push(ResidentEmitRow {
                source: source as u32,
                neighbors: row.slots.into_iter().map(|slot| slot.target).collect(),
            }),
            Err(row_error) => {
                error = Some(row_error);
                break;
            }
        }
    }
    ResidentEmitPartition {
        rows,
        backfill,
        error,
    }
}

fn finalize_flat_resident_partition(
    point_start: usize,
    store: &FlatResidentReservoirStore,
    source_start: usize,
    source_end: usize,
    policy: HashPruneBackfillPolicy,
) -> ResidentEmitPartition {
    let mut rows = Vec::with_capacity(source_end.saturating_sub(source_start));
    let mut backfill = HashPruneBackfillProfile::default();
    let mut error = None;
    for source in source_start..source_end {
        let logical_source = point_start + source;
        match store.finalize_row_with_logical_source(source, logical_source, policy, &mut backfill)
        {
            Ok(row) => rows.push(ResidentEmitRow {
                source: logical_source as u32,
                neighbors: row.slots.into_iter().map(|slot| slot.target).collect(),
            }),
            Err(row_error) => {
                error = Some(row_error);
                break;
            }
        }
    }
    ResidentEmitPartition {
        rows,
        backfill,
        error,
    }
}

fn emit_resident_reservoirs_parallel_with_config<F>(
    point_start: usize,
    reservoirs: Vec<Mutex<HashPruneReservoir>>,
    policy: HashPruneBackfillPolicy,
    config: ResidentEmitConfig,
    on_node_done: &mut F,
) -> AnnResult<ResidentEmitResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let config = ResidentEmitConfig {
        chunk_sources: config.chunk_sources.max(1),
        fifo_chunks: config.fifo_chunks.max(1),
        initial_workers: config
            .initial_workers
            .max(1)
            .min(config.host_parallelism.max(1)),
        host_parallelism: config.host_parallelism.max(1),
        scale_window_chunks: config.scale_window_chunks.max(1),
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let received_chunks = Arc::new(AtomicUsize::new(0));
    let (completed_tx, completed_rx) = mpsc::sync_channel(config.fifo_chunks);
    let (scale_tx, scale_rx) = mpsc::channel::<usize>();

    std::thread::scope(|scope| {
        let producer_cancelled = Arc::clone(&cancelled);
        let producer_received_chunks = Arc::clone(&received_chunks);
        let producer = scope.spawn(move || -> AnnResult<ResidentEmitProfile> {
            let mut workers = config.initial_workers;
            let mut pool = build_final_emit_pool(workers)?;
            let mut telemetry = ResidentEmitProfile {
                parallel: true,
                chunk_sources: config.chunk_sources,
                fifo_chunks: config.fifo_chunks,
                initial_workers: workers,
                peak_workers: workers,
                writer_buffer_bytes: crate::index::DirectMemGraphWriter::BUFFER_BYTES,
                ..ResidentEmitProfile::default()
            };
            let mut reservoir_iter = reservoirs.into_iter();
            let mut chunk_offset = 0usize;
            let mut sent_chunks = 0usize;

            loop {
                if producer_cancelled.load(Ordering::Acquire) {
                    break;
                }
                let mut requested_workers = workers;
                while let Ok(requested) = scale_rx.try_recv() {
                    requested_workers =
                        requested_workers.max(requested.min(config.host_parallelism));
                }
                if requested_workers > workers {
                    pool = build_final_emit_pool(requested_workers)?;
                    workers = requested_workers;
                    telemetry.peak_workers = telemetry.peak_workers.max(workers);
                }

                let chunk = reservoir_iter
                    .by_ref()
                    .take(config.chunk_sources)
                    .collect::<Vec<_>>();
                if chunk.is_empty() {
                    break;
                }
                let first_source = point_start + chunk_offset;
                chunk_offset += chunk.len();
                let partition_count = workers.saturating_mul(4).min(chunk.len()).max(1);
                let partition_sources = chunk.len().div_ceil(partition_count);
                let mut partitions = Vec::with_capacity(partition_count);
                let mut chunk_iter = chunk.into_iter();
                let mut partition_offset = 0usize;
                loop {
                    let partition = chunk_iter
                        .by_ref()
                        .take(partition_sources)
                        .collect::<Vec<_>>();
                    if partition.is_empty() {
                        break;
                    }
                    partitions.push((first_source + partition_offset, partition));
                    partition_offset += partitions
                        .last()
                        .map(|(_, reservoirs)| reservoirs.len())
                        .unwrap_or(0);
                }

                let finalize_start = Instant::now();
                let partition_results = pool.install(|| {
                    partitions
                        .into_par_iter()
                        .map(|(partition_start, partition)| {
                            finalize_resident_partition(partition_start, partition, policy)
                        })
                        .collect::<Vec<_>>()
                });
                let finalize_wall = finalize_start.elapsed();
                telemetry.finalize_service_wall += finalize_wall;

                let mut rows = Vec::new();
                let mut backfill = HashPruneBackfillProfile::default();
                let mut error = None;
                for partition in partition_results {
                    backfill.merge(partition.backfill);
                    rows.extend(partition.rows);
                    if partition.error.is_some() {
                        error = partition.error;
                        break;
                    }
                }
                let has_error = error.is_some();
                let completed = ResidentEmitChunk {
                    first_source,
                    rows,
                    backfill,
                    error,
                    finalize_wall,
                };
                let backpressure_start = Instant::now();
                if completed_tx.send(completed).is_err() {
                    break;
                }
                telemetry.producer_backpressure_wall += backpressure_start.elapsed();
                sent_chunks += 1;
                telemetry.chunks = sent_chunks;
                telemetry.max_completed_inflight = telemetry.max_completed_inflight.max(
                    sent_chunks
                        .saturating_sub(producer_received_chunks.load(Ordering::Acquire))
                        .min(config.fifo_chunks),
                );
                if has_error {
                    break;
                }
            }
            Ok(telemetry)
        });

        let mut backfill = HashPruneBackfillProfile::default();
        let mut rows = 0usize;
        let mut chunks = 0usize;
        let mut consumer_wait_wall = Duration::ZERO;
        let mut callback_service_wall = Duration::ZERO;
        let mut window_finalize_wall = Duration::ZERO;
        let mut window_callback_wall = Duration::ZERO;
        let mut current_workers = config.initial_workers;
        let mut expected_source = point_start;
        let mut consumer_error = None;

        loop {
            let wait_start = Instant::now();
            let completed = match completed_rx.recv() {
                Ok(completed) => completed,
                Err(_) => {
                    consumer_wait_wall += wait_start.elapsed();
                    break;
                }
            };
            consumer_wait_wall += wait_start.elapsed();
            received_chunks.fetch_add(1, Ordering::Release);
            chunks += 1;

            if completed.first_source != expected_source {
                consumer_error = Some(AnnError::log_index_error(format!(
                    "resident finalization emitted source {} while {} was expected",
                    completed.first_source, expected_source
                )));
                break;
            }

            window_finalize_wall += completed.finalize_wall;
            let callback_start = Instant::now();
            for row in completed.rows {
                if row.source as usize != expected_source {
                    consumer_error = Some(AnnError::log_index_error(format!(
                        "resident finalization emitted source {} while {} was expected",
                        row.source, expected_source
                    )));
                    break;
                }
                if let Err(error) = on_node_done(row.source, row.neighbors) {
                    consumer_error = Some(error);
                    break;
                }
                expected_source += 1;
                rows += 1;
            }
            let callback_wall = callback_start.elapsed();
            callback_service_wall += callback_wall;
            window_callback_wall += callback_wall;
            if consumer_error.is_some() {
                break;
            }
            backfill.merge(completed.backfill);
            if let Some(error) = completed.error {
                consumer_error = Some(error);
                break;
            }

            if chunks % config.scale_window_chunks == 0 {
                let next_workers = next_final_emit_workers(
                    current_workers,
                    config.host_parallelism,
                    window_finalize_wall,
                    window_callback_wall,
                );
                if next_workers > current_workers {
                    current_workers = next_workers;
                    let _ = scale_tx.send(current_workers);
                }
                window_finalize_wall = Duration::ZERO;
                window_callback_wall = Duration::ZERO;
            }
        }

        cancelled.store(true, Ordering::Release);
        drop(completed_rx);
        drop(scale_tx);
        let producer_result = producer.join().map_err(|_| {
            AnnError::log_index_error("resident finalization producer panicked".to_string())
        })?;
        if let Some(error) = consumer_error {
            return Err(error);
        }
        let mut telemetry = producer_result?;
        telemetry.rows = rows;
        telemetry.chunks = chunks;
        telemetry.callback_service_wall = callback_service_wall;
        telemetry.consumer_wait_wall = consumer_wait_wall;
        Ok(ResidentEmitResult {
            backfill,
            telemetry,
        })
    })
}

fn emit_flat_resident_reservoirs_parallel_with_config<F>(
    point_start: usize,
    store: FlatResidentReservoirStore,
    policy: HashPruneBackfillPolicy,
    config: ResidentEmitConfig,
    on_node_done: &mut F,
) -> AnnResult<ResidentEmitResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let config = ResidentEmitConfig {
        chunk_sources: config.chunk_sources.max(1),
        fifo_chunks: config.fifo_chunks.max(1),
        initial_workers: config
            .initial_workers
            .max(1)
            .min(config.host_parallelism.max(1)),
        host_parallelism: config.host_parallelism.max(1),
        scale_window_chunks: config.scale_window_chunks.max(1),
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let received_chunks = Arc::new(AtomicUsize::new(0));
    let (completed_tx, completed_rx) = mpsc::sync_channel(config.fifo_chunks);
    let (scale_tx, scale_rx) = mpsc::channel::<usize>();

    std::thread::scope(|scope| {
        let producer_cancelled = Arc::clone(&cancelled);
        let producer_received_chunks = Arc::clone(&received_chunks);
        let producer = scope.spawn(move || -> AnnResult<ResidentEmitProfile> {
            let mut store = store;
            let points = store.len();
            let mut workers = config.initial_workers;
            let mut pool = build_final_emit_pool(workers)?;
            let mut telemetry = ResidentEmitProfile {
                parallel: true,
                chunk_sources: config.chunk_sources,
                fifo_chunks: config.fifo_chunks,
                initial_workers: workers,
                peak_workers: workers,
                writer_buffer_bytes: crate::index::DirectMemGraphWriter::BUFFER_BYTES,
                ..ResidentEmitProfile::default()
            };
            let mut chunk_offset = 0usize;
            let mut sent_chunks = 0usize;

            while chunk_offset < points {
                if producer_cancelled.load(Ordering::Acquire) {
                    break;
                }
                let mut requested_workers = workers;
                while let Ok(requested) = scale_rx.try_recv() {
                    requested_workers =
                        requested_workers.max(requested.min(config.host_parallelism));
                }
                if requested_workers > workers {
                    pool = build_final_emit_pool(requested_workers)?;
                    workers = requested_workers;
                    telemetry.peak_workers = telemetry.peak_workers.max(workers);
                }

                let chunk_end = chunk_offset
                    .saturating_add(config.chunk_sources)
                    .min(points);
                let chunk_len = chunk_end - chunk_offset;
                let partition_count = workers.saturating_mul(4).min(chunk_len).max(1);
                let partition_sources = chunk_len.div_ceil(partition_count);
                let mut partitions = Vec::with_capacity(partition_count);
                let mut partition_start = chunk_offset;
                while partition_start < chunk_end {
                    let partition_end = partition_start
                        .saturating_add(partition_sources)
                        .min(chunk_end);
                    partitions.push((partition_start, partition_end));
                    partition_start = partition_end;
                }

                let finalize_start = Instant::now();
                let partition_results = pool.install(|| {
                    partitions
                        .into_par_iter()
                        .map(|(partition_start, partition_end)| {
                            finalize_flat_resident_partition(
                                point_start,
                                &store,
                                partition_start,
                                partition_end,
                                policy,
                            )
                        })
                        .collect::<Vec<_>>()
                });
                let finalize_wall = finalize_start.elapsed();
                telemetry.finalize_service_wall += finalize_wall;
                store.release_segments_before(chunk_end);

                let mut rows = Vec::with_capacity(chunk_len);
                let mut backfill = HashPruneBackfillProfile::default();
                let mut error = None;
                for partition in partition_results {
                    backfill.merge(partition.backfill);
                    rows.extend(partition.rows);
                    if partition.error.is_some() {
                        error = partition.error;
                        break;
                    }
                }
                let has_error = error.is_some();
                let completed = ResidentEmitChunk {
                    first_source: point_start + chunk_offset,
                    rows,
                    backfill,
                    error,
                    finalize_wall,
                };
                let backpressure_start = Instant::now();
                if completed_tx.send(completed).is_err() {
                    break;
                }
                telemetry.producer_backpressure_wall += backpressure_start.elapsed();
                chunk_offset = chunk_end;
                sent_chunks += 1;
                telemetry.chunks = sent_chunks;
                telemetry.max_completed_inflight = telemetry.max_completed_inflight.max(
                    sent_chunks
                        .saturating_sub(producer_received_chunks.load(Ordering::Acquire))
                        .min(config.fifo_chunks),
                );
                if has_error {
                    break;
                }
            }
            Ok(telemetry)
        });

        let mut backfill = HashPruneBackfillProfile::default();
        let mut rows = 0usize;
        let mut chunks = 0usize;
        let mut consumer_wait_wall = Duration::ZERO;
        let mut callback_service_wall = Duration::ZERO;
        let mut window_finalize_wall = Duration::ZERO;
        let mut window_callback_wall = Duration::ZERO;
        let mut current_workers = config.initial_workers;
        let mut expected_source = point_start;
        let mut consumer_error = None;

        loop {
            let wait_start = Instant::now();
            let completed = match completed_rx.recv() {
                Ok(completed) => completed,
                Err(_) => {
                    consumer_wait_wall += wait_start.elapsed();
                    break;
                }
            };
            consumer_wait_wall += wait_start.elapsed();
            received_chunks.fetch_add(1, Ordering::Release);
            chunks += 1;

            if completed.first_source != expected_source {
                consumer_error = Some(AnnError::log_index_error(format!(
                    "flat resident finalization emitted source {} while {} was expected",
                    completed.first_source, expected_source
                )));
                break;
            }

            window_finalize_wall += completed.finalize_wall;
            let callback_start = Instant::now();
            for row in completed.rows {
                if row.source as usize != expected_source {
                    consumer_error = Some(AnnError::log_index_error(format!(
                        "flat resident finalization emitted source {} while {} was expected",
                        row.source, expected_source
                    )));
                    break;
                }
                if let Err(error) = on_node_done(row.source, row.neighbors) {
                    consumer_error = Some(error);
                    break;
                }
                expected_source += 1;
                rows += 1;
            }
            let callback_wall = callback_start.elapsed();
            callback_service_wall += callback_wall;
            window_callback_wall += callback_wall;
            if consumer_error.is_some() {
                break;
            }
            backfill.merge(completed.backfill);
            if let Some(error) = completed.error {
                consumer_error = Some(error);
                break;
            }

            if chunks % config.scale_window_chunks == 0 {
                let next_workers = next_final_emit_workers(
                    current_workers,
                    config.host_parallelism,
                    window_finalize_wall,
                    window_callback_wall,
                );
                if next_workers > current_workers {
                    current_workers = next_workers;
                    let _ = scale_tx.send(current_workers);
                }
                window_finalize_wall = Duration::ZERO;
                window_callback_wall = Duration::ZERO;
            }
        }

        cancelled.store(true, Ordering::Release);
        drop(completed_rx);
        drop(scale_tx);
        let producer_result = producer.join().map_err(|_| {
            AnnError::log_index_error("flat resident finalization producer panicked".to_string())
        })?;
        if let Some(error) = consumer_error {
            return Err(error);
        }
        let mut telemetry = producer_result?;
        telemetry.rows = rows;
        telemetry.chunks = chunks;
        telemetry.callback_service_wall = callback_service_wall;
        telemetry.consumer_wait_wall = consumer_wait_wall;
        Ok(ResidentEmitResult {
            backfill,
            telemetry,
        })
    })
}

fn emit_resident_reservoirs<F>(
    point_start: usize,
    reservoirs: Vec<Mutex<HashPruneReservoir>>,
    policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_node_done: &mut F,
) -> AnnResult<ResidentEmitResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if diagnostic_collector.is_some() || diagnostic_finalizer.is_some() {
        emit_resident_reservoirs_serial(
            point_start,
            reservoirs,
            policy,
            diagnostic_collector,
            diagnostic_finalizer,
            on_node_done,
        )
    } else {
        emit_resident_reservoirs_parallel_with_config(
            point_start,
            reservoirs,
            policy,
            ResidentEmitConfig::production(),
            on_node_done,
        )
    }
}

fn emit_resident_storage<F>(
    point_start: usize,
    storage: ResidentReservoirStorage,
    policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_node_done: &mut F,
) -> AnnResult<ResidentEmitResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    match storage {
        ResidentReservoirStorage::Legacy(reservoirs) => emit_resident_reservoirs(
            point_start,
            reservoirs,
            policy,
            diagnostic_collector,
            diagnostic_finalizer,
            on_node_done,
        ),
        ResidentReservoirStorage::Flat(store) => {
            if diagnostic_collector.is_some() || diagnostic_finalizer.is_some() {
                emit_flat_resident_reservoirs_serial(
                    point_start,
                    store,
                    policy,
                    diagnostic_collector,
                    diagnostic_finalizer,
                    on_node_done,
                )
            } else {
                emit_flat_resident_reservoirs_parallel_with_config(
                    point_start,
                    store,
                    policy,
                    ResidentEmitConfig::production(),
                    on_node_done,
                )
            }
        }
    }
}

impl<'a> HybridEdgeSink<'a> {
    fn new_with_policy(
        resident_points: usize,
        resident_sink: &'a dyn PendingEdgeSink,
        suffix_sink: &'a dyn PendingEdgeSink,
        policy: HashPruneEvidencePolicy,
    ) -> Self {
        Self::new_with_policies(
            resident_points,
            resident_sink,
            suffix_sink,
            policy,
            HashPruneBackfillPolicy::off(),
        )
    }

    fn new_with_policies(
        resident_points: usize,
        resident_sink: &'a dyn PendingEdgeSink,
        suffix_sink: &'a dyn PendingEdgeSink,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> Self {
        debug_assert_eq!(resident_sink.hashprune_evidence_policy(), policy);
        debug_assert_eq!(resident_sink.hashprune_backfill_policy(), backfill_policy);
        debug_assert_eq!(suffix_sink.hashprune_evidence_policy(), policy);
        debug_assert_eq!(suffix_sink.hashprune_backfill_policy(), backfill_policy);
        Self {
            resident_points,
            resident_sink,
            suffix_sink,
            policy,
            backfill_policy,
        }
    }
}

impl PendingEdgeSink for DeferredReservoirs {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        self.policy
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        self.backfill_policy
    }

    fn prepare_for_d1(&self) -> AnnResult<()> {
        let wait_started = Instant::now();
        let _ = self.allocation()?;
        *self.d1_wait_wall.lock() += wait_started.elapsed();
        Ok(())
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<self::leaf_build::PendingEdge>,
    ) -> AnnResult<self::leaf_build::PendingEdgeFlushProfile> {
        let allocation = self.allocation()?;
        match &allocation.reservoirs {
            ResidentReservoirStorage::Legacy(reservoirs) => {
                ReservoirEdgeSink::new_with_policies_and_diagnostics(
                    reservoirs,
                    self.policy,
                    self.backfill_policy,
                    self.diagnostics.as_deref(),
                )
                .flush_pending_edges(edges)
            }
            ResidentReservoirStorage::Flat(store) => {
                FlatReservoirEdgeSink::new_with_policies_and_diagnostics(
                    store,
                    self.policy,
                    self.backfill_policy,
                    self.diagnostics.as_deref(),
                )
                .flush_pending_edges(edges)
            }
        }
    }
}

impl PendingEdgeSink for HybridEdgeSink<'_> {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        self.policy
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        self.backfill_policy
    }

    fn prepare_for_d1(&self) -> AnnResult<()> {
        self.resident_sink.prepare_for_d1()?;
        self.suffix_sink.prepare_for_d1()
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<self::leaf_build::PendingEdge>,
    ) -> AnnResult<self::leaf_build::PendingEdgeFlushProfile> {
        if edges.is_empty() {
            return Ok(self::leaf_build::PendingEdgeFlushProfile::default());
        }

        let mut resident_edges = Vec::new();
        let mut spill_edges = Vec::new();
        for edge in edges.drain(..) {
            if edge.p < self.resident_points {
                resident_edges.push(edge);
            } else {
                spill_edges.push(edge);
            }
        }

        let mut profile = self::leaf_build::PendingEdgeFlushProfile::default();
        if !spill_edges.is_empty() {
            profile.merge(self.suffix_sink.flush_pending_edges(&mut spill_edges)?);
        }
        if !resident_edges.is_empty() {
            profile.merge(
                self.resident_sink
                    .flush_pending_edges(&mut resident_edges)?,
            );
        }
        Ok(profile)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LrpfProfileCounters {
    candidate_records: usize,
    witness_records: usize,
    fragments: usize,
    pre_spill_pruned_candidates: usize,
    reducer_pruned_candidates: usize,
}

#[derive(Clone, Copy, Debug)]
struct EmptySketchAccessor {
    rows: usize,
    width: usize,
}

impl EmptySketchAccessor {
    fn new(rows: usize, width: usize) -> Self {
        Self { rows, width }
    }
}

impl SketchAccessor for EmptySketchAccessor {
    fn width(&self) -> usize {
        self.width
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn row_copy_into(&self, idx: usize, _dst: &mut [f32]) -> AnnResult<()> {
        Err(AnnError::log_index_error(format!(
            "LRPF compact path attempted to read sketch row {idx}"
        )))
    }

    fn resident_bytes(&self) -> usize {
        0
    }

    fn read_rows_into(&self, ids: &[u32], _out: &mut [f32]) -> AnnResult<()> {
        if ids.is_empty() {
            Ok(())
        } else {
            Err(AnnError::log_index_error(format!(
                "LRPF compact path attempted to read {} sketch rows",
                ids.len()
            )))
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NullPendingEdgeSink;

impl PendingEdgeSink for NullPendingEdgeSink {
    fn flush_pending_edges(
        &self,
        edges: &mut Vec<self::leaf_build::PendingEdge>,
    ) -> AnnResult<self::leaf_build::PendingEdgeFlushProfile> {
        if edges.is_empty() {
            Ok(self::leaf_build::PendingEdgeFlushProfile::default())
        } else {
            Err(AnnError::log_index_error(format!(
                "LRPF compact path attempted to emit {} HashPrune edges",
                edges.len()
            )))
        }
    }
}

/// 使用 PiPNN 在内存中构建邻接表形式的图。
///
/// - `dataset`：已经通过 InmemDataset 加载好的向量数据；
/// - `num_points`：参与构图的点数（通常等于 InmemIndex.num_active_pts）；
/// - `num_threads`：用于叶子并行处理的线程数（0 表示 Rayon 默认线程数）。
/// - `on_node_done`：回调函数，当某个节点的构图完成时调用，传入 `(u32, Vec<u32>)`。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PiPNNGraphBuildResult {
    pub entry_point: Option<u32>,
}

pub fn build_pipnn_graph<F>(
    dataset: &InmemDataset<f32>,
    num_points: usize,
    num_threads: u32,
    params: &PiPNNParams,
    on_node_done: F,
) -> AnnResult<PiPNNGraphBuildResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let store = InmemDatasetPointStore::new(dataset, num_points);
    build_pipnn_graph_with_store(&store, num_points, num_threads, params, on_node_done)
}

pub fn build_pipnn_graph_with_store<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    num_threads: u32,
    params: &PiPNNParams,
    mut on_node_done: F,
) -> AnnResult<PiPNNGraphBuildResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    params.validate_hashprune_policies()?;
    let hashprune_backfill_policy = HashPruneBackfillPolicy::from_params(params)?;
    if num_points == 0 {
        if let Some(prefix) = params.hashprune_backfill_diagnostics_prefix.as_deref() {
            HashPruneDiagnosticsSession::create(
                prefix,
                0,
                params.l_max,
                hashprune_backfill_policy,
            )?
            .finish(0, &HashPruneBackfillProfile::default())?;
        }
        return Ok(PiPNNGraphBuildResult::default());
    }

    assert!(params.m_hash_bits <= 16, "m_hash_bits must be <= 16");

    // 当前 Vertex::compare 仅实现 L2，度量通过 Metric::L2 传入。
    let metric = params.metric();
    let effective_threads = if num_threads == 0 {
        rayon::current_num_threads()
    } else {
        num_threads as usize
    };
    let detailed_profiling = detailed_pipnn_profiling_enabled();
    info!(
        "PiPNN build start: num_points={}, num_threads={}",
        num_points, num_threads
    );
    if detailed_profiling {
        info!("PiPNN detailed profiling enabled via SHEAR_PIPNN_PROFILE");
    }
    log_memory_telemetry(
        "start",
        num_points,
        params,
        dataset.dim(),
        0,
        0,
        effective_threads,
    );

    let blas_env_backup = BlasThreadEnv::capture();

    let result = (|| {
        let leaf_compute_backend = resolve_compute_backend(params)?;
        validate_compute_backend_before_d0(leaf_compute_backend.as_ref(), params)?;
        let assignment_backend = cpu_backend();

        if params.root_radius_dry_run || params.root_risk_dry_run {
            let all_indices: Vec<u32> = (0..num_points as u32).collect();
            let mut rng = StdRng::seed_from_u64(params.random_seed);
            let json = root_radius_dry_run_json(dataset, &all_indices, metric, params, &mut rng)?;
            println!("{json}");
            return Ok(PiPNNGraphBuildResult::default());
        }

        if uses_lrpf_candidate_backend(params)
            && (!params.oom_enable
                || !matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree))
        {
            return Err(AnnError::log_index_error(
                "prune_backend=lrpf currently requires OOM gpu-subtree compact top-k mode"
                    .to_string(),
            ));
        }

        if params.oom_enable {
            return build_pipnn_graph_oom(
                dataset,
                num_points,
                effective_threads,
                params,
                detailed_profiling,
                leaf_compute_backend.as_ref(),
                &mut on_node_done,
            );
        }

        let mut diagnostics = params
            .hashprune_backfill_diagnostics_prefix
            .as_deref()
            .map(|prefix| {
                HashPruneDiagnosticsSession::create(
                    prefix,
                    num_points,
                    params.l_max,
                    hashprune_backfill_policy,
                )
            })
            .transpose()?;

        // 1) 预计算所有点的 m 维 sketch（一次性完成）。 RBC 之前必须完成，因为 process_leaf
        //    依赖它。
        let t_sketch = Instant::now();
        let sketches =
            hash_prune::compute_sketches_from_store(dataset, num_points, params, num_threads)?;
        let sketch_bytes = estimate_sketch_bytes(&sketches);
        info!(
            "PiPNN compute_sketches done: elapsed={:?}",
            t_sketch.elapsed()
        );

        log_memory_telemetry(
            "after sketches",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            0,
            effective_threads,
        );

        // 2) 为每个点分配 HashPrune 水库（带互斥锁以支持多线程叶处理）。 也必须预先分配。
        let t_res = Instant::now();
        let mut reservoirs: Vec<Mutex<HashPruneReservoir>> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            reservoirs.push(Mutex::new(HashPruneReservoir::new(params.l_max)));
        }
        let reservoir_bytes = estimate_reservoir_bytes(num_points, params.l_max);
        info!(
            "PiPNN init reservoirs done: num_points={}, elapsed={:?}",
            num_points,
            t_res.elapsed()
        );
        log_memory_telemetry(
            "after reservoirs",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            reservoir_bytes,
            effective_threads,
        );

        // 3) 启动统一调度的 RBC + Leaf 处理。 RBC 递归与叶子构建共用同一个受限 Rayon
        //    池，num_threads 是整个 PiPNN 构建阶段的硬上限。叶子任务采用 scope 内 FIFO
        //    调度，并在饱和时回退为 inline， 避免额外 OS consumer 线程和双倍线程占用。
        let t_rbc_leaf = Instant::now();
        let all_indices: Vec<u32> = (0..num_points as u32).collect();

        // 进度条
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message("PiPNN streaming processing...");

        let leaf_scratch_bytes = estimate_leaf_scratch_bytes(
            params.kernel_safe_leaf_size(),
            dataset.dim(),
            params.leaf_knn,
        );
        let scheduler_budget =
            SchedulerBudget::for_worker_count(effective_threads, params.kernel_safe_leaf_size());

        // 在线程并行处理叶子之前，将底层 BLAS/OpenMP 线程数钉死为 1，避免 oversubscription。
        BlasThreadEnv::set_single_threaded();

        // 单个 Rayon 池承载 RBC 递归与叶子处理，线程数受 --num-threads 硬限制。
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(effective_threads)
            .stack_size(PIPNN_RAYON_STACK_BYTES)
            .build()
            .map_err(|e| {
                AnnError::log_index_error(format!("Failed to create Rayon pool: {}", e))
            })?;

        let edge_sink = ReservoirEdgeSink::new_with_policies_and_diagnostics(
            reservoirs.as_slice(),
            HashPruneEvidencePolicy::from_params(params)?,
            hashprune_backfill_policy,
            diagnostics
                .as_ref()
                .map(HashPruneDiagnosticsSession::collector),
        );
        let leaf_context = LeafTaskContext {
            compute_backend: leaf_compute_backend.as_ref(),
            dataset,
            metric,
            sketches: &sketches,
            params,
            edge_sink: &edge_sink,
            detailed_profiling,
            #[cfg(test)]
            observer: None,
        };
        let producer_start = Instant::now();
        let mut rbc_telemetry = RbcPartitionTelemetry::default();
        let mut kmeans_telemetry = KMeansPartitionTelemetry::default();
        let (scheduler_profile, scheduler_telemetry) =
            run_scheduler_scope(&pool, scheduler_budget, leaf_context, &pb, |emitter| {
                emit_streaming_partition_leaves(
                    assignment_backend,
                    dataset,
                    &all_indices,
                    params,
                    effective_threads,
                    None,
                    emitter,
                    &mut rbc_telemetry,
                    &mut kmeans_telemetry,
                )?;
                Ok(())
            })?;
        let producer_wall = producer_start.elapsed();

        let streaming_profile = StreamingProfile {
            producer_wall,
            total_wall: t_rbc_leaf.elapsed(),
            leaf: scheduler_profile,
            telemetry: scheduler_telemetry,
            rbc: rbc_telemetry,
            kmeans: kmeans_telemetry,
            entry_point: None,
        };

        pb.finish_with_message("PiPNN streaming processing done");
        info!(
            "PiPNN streaming (RBC+Leaf) done: elapsed={:?}",
            t_rbc_leaf.elapsed()
        );
        if detailed_profiling {
            let (uring_batches, uring_windows, uring_bytes, uring_fallbacks) =
                uring_gather_counters();
            info!(
                "PiPNN streaming profile: producer_wall={:?} total_wall={:?} leaf_parallelism={:.2} leaf_total_wall={:?} leaf_idle_estimate={:?} leaves={} points={} blockwise_leaves={} peak_inflight_leaf_tasks={} peak_leaf_backlog={} producer_help_drains={} producer_help_yields={} inline_leaf_fallbacks={} large_leaf_count={} large_leaf_block_tasks={}",
                streaming_profile.producer_wall,
                streaming_profile.total_wall,
                streaming_profile.leaf_parallelism(),
                streaming_profile.leaf.total_wall,
                streaming_profile.leaf_idle_time(),
                streaming_profile.leaf.leaves,
                streaming_profile.leaf.points,
                streaming_profile.leaf.blockwise_leaves,
                streaming_profile.telemetry.peak_inflight_leaf_tasks,
                streaming_profile.telemetry.peak_leaf_backlog,
                streaming_profile.telemetry.producer_help_drains,
                streaming_profile.telemetry.producer_help_yields,
                streaming_profile.telemetry.inline_leaf_fallbacks,
                streaming_profile.telemetry.large_leaf_count,
                streaming_profile.telemetry.large_leaf_block_tasks,
            );
            info!(
                "PiPNN leaf profile: load={:?} distance={:?} topk={:?} hash={:?} flush={:?}",
                streaming_profile.leaf.load,
                streaming_profile.leaf.distance,
                streaming_profile.leaf.topk,
                streaming_profile.leaf.hash,
                streaming_profile.leaf.flush,
            );
            info!(
                "PiPNN point io_uring gather: batches={} windows={} bytes={} fallbacks={}",
                uring_batches, uring_windows, uring_bytes, uring_fallbacks,
            );
        }
        log_memory_telemetry(
            "after RBC+leaf",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            reservoir_bytes,
            effective_threads,
        );
        info!(
            "PiPNN leaf scratch estimate: max_leaf_size={} per_worker_bytes={} active_scratch_bytes_estimate={}",
            params.kernel_safe_leaf_size(),
            leaf_scratch_bytes,
            leaf_scratch_bytes.saturating_mul(effective_threads),
        );

        // 4) 把 HashPrune 水库转换成邻接表 Vec<Vec<u32>>。 逐个转换并调用回调。
        let t_graph = Instant::now();
        let resident_emit = if let Some(session) = diagnostics.as_mut() {
            let (collector, finalizer) = session.parts();
            emit_resident_reservoirs(
                0,
                reservoirs,
                hashprune_backfill_policy,
                Some(collector),
                Some(finalizer),
                &mut on_node_done,
            )?
        } else {
            emit_resident_reservoirs(
                0,
                reservoirs,
                hashprune_backfill_policy,
                None,
                None,
                &mut on_node_done,
            )?
        };
        let mut hashprune_backfill_profile = resident_emit.backfill;
        hashprune_backfill_profile.merge(streaming_profile.leaf.hashprune_backfill.clone());
        if let Some(session) = diagnostics.take() {
            session.finish(0, &hashprune_backfill_profile)?;
        }
        info!("PiPNN build graph done: elapsed={:?}", t_graph.elapsed());
        log_memory_telemetry(
            "after graph",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            0,
            effective_threads,
        );

        Ok(PiPNNGraphBuildResult::default())
    })();

    blas_env_backup.restore();

    result
}

fn build_pipnn_graph_oom<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    requested_threads: usize,
    params: &PiPNNParams,
    detailed_profiling: bool,
    compute_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    on_node_done: &mut F,
) -> AnnResult<PiPNNGraphBuildResult>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    params.validate_online_suffix_reduce_config()?;
    let hashprune_evidence_policy = HashPruneEvidencePolicy::from_params(params)?;
    let hashprune_backfill_policy = HashPruneBackfillPolicy::from_params(params)?;
    let mut diagnostics = params
        .hashprune_backfill_diagnostics_prefix
        .as_deref()
        .map(|prefix| {
            HashPruneDiagnosticsSession::create(
                prefix,
                num_points,
                params.l_max,
                hashprune_backfill_policy,
            )
        })
        .transpose()?;
    let artifact_dir = prepare_oom_artifact_dir(params)?;
    let sketch_path = artifact_dir.join("sketches.bin");
    let candidate_path = artifact_dir.join("candidate.run");
    let lrpf_candidate_path = artifact_dir.join("lrpf_fragments.bin");
    let profile_path = params.oom_profile_json.clone();
    let overall_start = Instant::now();
    let online_suffix_eligibility = online_suffix_eligibility_for_dataset(params, num_points);
    let hashprune_candidate_backend = uses_hashprune_candidate_backend(params);
    let lrpf_candidate_backend = uses_lrpf_candidate_backend(params);
    let resident_reservoir_layout = if hashprune_candidate_backend {
        ResidentReservoirLayout::from_env()?
    } else {
        ResidentReservoirLayout::Legacy
    };
    let memory_plan = OomMemoryPlan::from_params_for_dataset_with_reservoir_layout(
        params,
        num_points,
        dataset.dim(),
        resident_reservoir_layout,
    );
    let memory_budget_bytes = memory_plan.total_budget_bytes;
    let raw_vector_cache_store = if memory_plan.raw_vector_cache_budget_bytes > 0 {
        let store =
            RawVectorCachePointStore::new(dataset, memory_plan.raw_vector_cache_budget_bytes)?;
        info!(
            "PiPNN OOM raw vector cache enabled: budget_bytes={} row_capacity={} row_bytes={}",
            store.budget_bytes(),
            store.row_capacity(),
            dataset.dim().saturating_mul(size_of::<f32>()),
        );
        Some(store)
    } else {
        None
    };
    let dataset: &dyn PointStore = match raw_vector_cache_store.as_ref() {
        Some(store) => store,
        None => dataset,
    };
    let execution_threads = derive_oom_execution_threads(requested_threads, params, dataset.dim());
    let pool_threads = execution_threads.pool_threads;
    let leaf_budget_threads = execution_threads.leaf_budget_threads;
    info!(
        "PiPNN OOM thread model: requested_threads={} pool_threads={} leaf_budget_threads={} memory_budget_bytes={} memory_budget_source={}",
        requested_threads,
        pool_threads,
        leaf_budget_threads,
        memory_budget_bytes,
        if params.oom_memory_budget_bytes == 0 {
            "auto-host-cgroup"
        } else {
            "explicit"
        }
    );
    if let OnlineSuffixEligibility::ExternalFallback(reason) = online_suffix_eligibility {
        info!(
            "PiPNN online suffix requested mode={} selected external reduction: reason={reason}",
            params.online_suffix_reduce_mode
        );
    }

    if lrpf_candidate_backend && !matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree) {
        return Err(AnnError::log_index_error(
            "prune_backend=lrpf currently requires OOM gpu-subtree compact top-k mode".to_string(),
        ));
    }
    let empty_sketches = EmptySketchAccessor::new(num_points, params.m_hash_bits);
    let shard_points = derive_spill_shard_points(num_points, params, memory_budget_bytes);
    let lrpf_shard_points = if lrpf_candidate_backend && params.lrpf_pre_spill_compress_enable {
        derive_lrpf_digest_shard_points(num_points, shard_points)
    } else {
        shard_points
    };
    let compact_spill_digest_width = effective_compact_spill_digest_width(params);
    let estimated_sketch_bytes = if hashprune_candidate_backend {
        num_points
            .saturating_mul(params.m_hash_bits)
            .saturating_mul(size_of::<f32>())
    } else {
        0
    };
    let resident_plan = if hashprune_candidate_backend {
        plan_resident_reservoir_prefix_with_layout(
            num_points,
            leaf_budget_threads,
            params,
            dataset.dim(),
            estimated_sketch_bytes,
            shard_points,
            memory_plan.reservoir_budget_bytes,
            resident_reservoir_layout,
        )
    } else {
        ResidentReservoirPlan::default()
    };
    let online_suffix_residency = plan_online_suffix_residency(
        params,
        num_points,
        resident_plan.resident_points,
        memory_plan.online_suffix_enabled,
    )?;
    let sq_staging_fused_sketch_candidate = should_use_sq_staging_fused_sketch(
        hashprune_candidate_backend,
        lrpf_candidate_backend,
        params,
    );
    let sketch_persist_start = Instant::now();
    let disk_sketches = if hashprune_candidate_backend && !sq_staging_fused_sketch_candidate {
        Some(DiskSketchStore::build_resident_from_point_store(
            &sketch_path,
            dataset,
            num_points,
            params,
            pool_threads as u32,
        )?)
    } else {
        None
    };
    let fused_rabitq_plus_sidecar_sketch: Option<
        self::dco_rabitq::RabitqPlusSidecarSketchBuildReport,
    > = None;
    let mut sq_staging_fused_sketch_enabled = false;
    let mut sq_staging_fused_sketch_compute_wall = Duration::ZERO;
    let mut sq_staging_fused_sketch_join_wait_wall = Duration::ZERO;
    let sketch_persist_wall = if hashprune_candidate_backend && !sq_staging_fused_sketch_candidate {
        sketch_persist_start.elapsed()
    } else {
        Duration::ZERO
    };
    let sketch_bytes = disk_sketches
        .as_ref()
        .map(|sketches| sketches.resident_bytes())
        .unwrap_or(if sq_staging_fused_sketch_candidate {
            estimated_sketch_bytes
        } else {
            0
        });
    let spill_bytes = disk_sketches
        .as_ref()
        .map(|sketches| sketches.file_bytes())
        .unwrap_or(0);
    let streaming_sketches: &dyn SketchAccessor = disk_sketches
        .as_ref()
        .map(|sketches| sketches as &dyn SketchAccessor)
        .unwrap_or(&empty_sketches);

    if !sq_staging_fused_sketch_candidate {
        log_memory_telemetry(
            "oom after sketches",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            0,
            leaf_budget_threads,
        );
    }
    let mut effective_sketch_bytes = sketch_bytes;
    let rabitq_plus_resident_sidecar =
        maybe_preload_rabitq_plus_resident_sidecar(params, dataset, num_points, &memory_plan)?;
    let sq_wmma_sidecar = if params.gpu_subtree_sq_staging_mode.is_enabled() {
        None
    } else {
        maybe_open_sq_wmma_runtime_sidecar(params, dataset, num_points, &memory_plan)?
    };
    let turboquant_sidecar =
        maybe_open_turboquant_runtime_sidecar(params, dataset, num_points, &memory_plan)?;
    let gpu_subtree_runtime_context = GpuSubtreeRuntimeContext {
        rabitq_plus_resident_sidecar: rabitq_plus_resident_sidecar.as_ref(),
        sq_wmma_sidecar: sq_wmma_sidecar.as_ref(),
        rotated_residual_sq8_sidecar: None,
        turboquant_sidecar: turboquant_sidecar.as_ref(),
        rabitq_residual_payload: None,
        rabitq_residual_qcode_store: None,
        transient_pipeline_budget_bytes: Some(memory_plan.transient_pipeline_budget_bytes),
    };

    let mut reservoir_initialization = ReservoirInitializationProfile::default();
    let (
        streaming_profile,
        streaming_wall,
        spill_stats,
        reduce_stats,
        reduce_wall,
        emit_wall,
        resident_reservoir_bytes,
        reservoir_allocation_wall,
        lrpf_profile_counters,
        resident_emit_profile,
        mut hashprune_backfill_profile,
        online_suffix_stats,
    ) = if lrpf_candidate_backend {
        let lrpf_sink = if params.lrpf_pre_spill_compress_enable {
            LrpfFragmentSpillSink::create_digest(
                &lrpf_candidate_path,
                lrpf_shard_points,
                LrpfConfig {
                    degree: effective_lrpf_target_degree(params),
                    protected_core: params.lrpf_protected_core,
                    pre_spill_width: effective_lrpf_pre_spill_width(params),
                    long_edge_quota: params.lrpf_long_edge_quota,
                },
                params.oom_spill_cache_bytes,
            )?
        } else {
            LrpfFragmentSpillSink::create(&lrpf_candidate_path, lrpf_shard_points)?
        };
        let null_edge_sink = NullPendingEdgeSink;
        let streaming_start = Instant::now();
        let streaming_profile = run_streaming_partition_pipeline_with_gpu_subtree_context(
            dataset,
            num_points,
            pool_threads,
            leaf_budget_threads,
            params,
            &artifact_dir,
            streaming_sketches,
            detailed_profiling,
            compute_backend,
            &null_edge_sink,
            gpu_subtree_runtime_context,
            Some(&lrpf_sink),
        )?;
        let streaming_wall = streaming_start.elapsed();
        if params.root_stop_after_d00 {
            println!(
                "{}",
                root_fanout_profile_json(&streaming_profile.rbc.root_fanout)
            );
            info!(
                "PiPNN OOM root-stop-after-d00 completed: artifact_dir={:?}",
                artifact_dir
            );
            return Ok(PiPNNGraphBuildResult::default());
        }
        let lrpf_artifacts = lrpf_sink.finish()?;
        let lrpf_spill_stats = ExternalLrpfReducer::summarize_spill_artifacts(&lrpf_artifacts);
        let spill_stats = spill_run_stats_from_lrpf(&lrpf_spill_stats);

        let reduce_start = Instant::now();
        let lrpf_reduce_stats =
            ExternalLrpfReducer::reduce_spill_artifacts_sharded_in_order_profiled(
                num_points,
                effective_lrpf_target_degree(params),
                params.lrpf_protected_core,
                params.lrpf_long_edge_quota,
                lrpf_shard_points,
                &lrpf_artifacts,
                |point, neighbors| on_node_done(point, neighbors),
            )?;
        let reduce_wall = reduce_start.elapsed();
        let reduce_stats = spill_reduce_stats_from_lrpf(&lrpf_reduce_stats);
        info!(
            "PiPNN OOM LRPF external-reduce mode done: streaming_elapsed={:?} reduce_elapsed={:?} fragments={} candidates={} witnesses={}",
            streaming_wall,
            reduce_wall,
            lrpf_spill_stats.fragment_count,
            lrpf_spill_stats.candidate_records,
            lrpf_spill_stats.witness_records
        );
        let lrpf_pre_spill_pruned_candidates =
            streaming_profile.leaf.lrpf_pre_spill_pruned_candidates;

        (
            streaming_profile,
            streaming_wall,
            spill_stats,
            reduce_stats,
            reduce_wall,
            Duration::ZERO,
            0,
            Duration::ZERO,
            LrpfProfileCounters {
                candidate_records: lrpf_spill_stats.candidate_records,
                witness_records: lrpf_spill_stats.witness_records,
                fragments: lrpf_spill_stats.fragment_count,
                pre_spill_pruned_candidates: lrpf_pre_spill_pruned_candidates,
                reducer_pruned_candidates: lrpf_reduce_stats.reducer_pruned_candidates,
            },
            ResidentEmitProfile::default(),
            HashPruneBackfillProfile::default(),
            OnlineSuffixReduceStats::default(),
        )
    } else if resident_plan.resident_points == num_points {
        let reservoir_bytes = estimate_reservoir_bytes_for_layout(
            num_points,
            params.l_max,
            resident_reservoir_layout,
        );
        let reservoir_init_config = reservoir_initialization_config_from_env(pool_threads);
        let overlap_enabled = reservoir_init_config.overlap_enabled && !params.root_stop_after_d00;
        let d0_read_progress = (overlap_enabled && sq_staging_fused_sketch_candidate
            && params.gpu_subtree_sq_staging_mode.is_enabled())
            .then(|| Arc::new(D0ReadProgress::new(num_points)));
        info!(
            "PiPNN OOM resident-reservoir mode planned: num_points={} reservoir_bytes_estimate={} layout={} allocation_phase={} workers={}",
            num_points,
            reservoir_bytes,
            resident_reservoir_layout.as_str(),
            if overlap_enabled { "D0-overlap" } else { "D1" },
            reservoir_init_config.workers,
        );
        let deferred_reservoirs = DeferredReservoirs::new_with_policies_and_diagnostics(
            num_points,
            params.l_max,
            hashprune_evidence_policy,
            hashprune_backfill_policy,
            diagnostics
                .as_ref()
                .map(HashPruneDiagnosticsSession::collector_arc),
        )
        .with_layout(resident_reservoir_layout)
        .with_initialization_config(ReservoirInitializationConfig {
            overlap_enabled,
            ..reservoir_init_config
        });
        let deferred_reservoirs = match &d0_read_progress {
            Some(progress) => deferred_reservoirs.with_d0_read_progress(Arc::clone(progress)),
            None => deferred_reservoirs,
        };
        let pipeline_d0_read_progress = d0_read_progress.clone();
        let streaming_start = Instant::now();
        let streaming_profile = run_with_reservoir_initialization_overlap(
            overlap_enabled,
            d0_read_progress.as_deref(),
            || deferred_reservoirs.initialize(),
            || {
                if sq_staging_fused_sketch_candidate {
                    let (
                        profile,
                        fused_sketches,
                        fused_sketch_compute_wall,
                        fused_sketch_join_wait_wall,
                    ) = run_streaming_partition_pipeline_with_sq_staging_fused_sketch(
                        dataset,
                        num_points,
                        pool_threads,
                        params,
                        &artifact_dir,
                        &sketch_path,
                        detailed_profiling,
                        compute_backend,
                        &deferred_reservoirs,
                        gpu_subtree_runtime_context,
                        pipeline_d0_read_progress,
                    )?;
                    sq_staging_fused_sketch_enabled = true;
                    sq_staging_fused_sketch_compute_wall = fused_sketch_compute_wall;
                    sq_staging_fused_sketch_join_wait_wall = fused_sketch_join_wait_wall;
                    effective_sketch_bytes = fused_sketches.resident_bytes();
                    log_memory_telemetry(
                        "oom after SQ staging fused sketches",
                        num_points,
                        params,
                        dataset.dim(),
                        effective_sketch_bytes,
                        reservoir_bytes,
                        leaf_budget_threads,
                    );
                    Ok(profile)
                } else {
                    run_streaming_partition_pipeline_with_gpu_subtree_context(
                        dataset,
                        num_points,
                        pool_threads,
                        leaf_budget_threads,
                        params,
                        &artifact_dir,
                        streaming_sketches,
                        detailed_profiling,
                        compute_backend,
                        &deferred_reservoirs,
                        gpu_subtree_runtime_context,
                        None,
                    )
                }
            },
        )?;
        let streaming_wall = streaming_start.elapsed();
        info!(
            "PiPNN OOM resident-reservoir streaming done: elapsed={:?}",
            streaming_wall
        );
        if params.root_stop_after_d00 {
            println!(
                "{}",
                root_fanout_profile_json(&streaming_profile.rbc.root_fanout)
            );
            info!(
                "PiPNN OOM root-stop-after-d00 completed: artifact_dir={:?}",
                artifact_dir
            );
            return Ok(PiPNNGraphBuildResult::default());
        }

        let emit_start = Instant::now();
        let allocation = deferred_reservoirs.into_allocation()?;
        debug_assert_eq!(allocation.estimated_bytes, reservoir_bytes);
        let reservoir_allocation_wall = allocation.allocation_wall;
        reservoir_initialization = allocation.initialization.clone();
        let resident_emit = if let Some(session) = diagnostics.as_mut() {
            let (collector, finalizer) = session.parts();
            emit_resident_storage(
                0,
                allocation.reservoirs,
                hashprune_backfill_policy,
                Some(collector),
                Some(finalizer),
                on_node_done,
            )?
        } else {
            emit_resident_storage(
                0,
                allocation.reservoirs,
                hashprune_backfill_policy,
                None,
                None,
                on_node_done,
            )?
        };
        let emit_wall = emit_start.elapsed();
        log_memory_telemetry(
            "oom after resident emit",
            num_points,
            params,
            dataset.dim(),
            effective_sketch_bytes,
            0,
            leaf_budget_threads,
        );

        (
            streaming_profile,
            streaming_wall,
            SpillRunStats::default(),
            SpillReduceStats::default(),
            Duration::ZERO,
            emit_wall,
            reservoir_bytes,
            reservoir_allocation_wall,
            LrpfProfileCounters::default(),
            resident_emit.telemetry,
            resident_emit.backfill,
            OnlineSuffixReduceStats::default(),
        )
    } else if resident_plan.resident_points > 0 {
        let resident_points = resident_plan.resident_points;
        let dense_resident_points = online_suffix_residency.dense_resident_points;
        let reservoir_bytes = estimate_reservoir_bytes_for_layout(
            dense_resident_points,
            params.l_max,
            resident_reservoir_layout,
        );
        let reservoir_init_config = reservoir_initialization_config_from_env(pool_threads);
        let overlap_enabled = reservoir_init_config.overlap_enabled && !params.root_stop_after_d00;
        let d0_read_progress =
            (overlap_enabled && sq_staging_fused_sketch_candidate && dense_resident_points > 0
                && params.gpu_subtree_sq_staging_mode.is_enabled())
                .then(|| Arc::new(D0ReadProgress::new(dense_resident_points)));
        info!(
            "PiPNN OOM hybrid reservoir plan: budget_points={} dense_points={} pinned_points={} total_points={} dense_reservoir_bytes_estimate={} layout={} allocation_phase={} workers={}",
            resident_points,
            dense_resident_points,
            online_suffix_residency.pinned_prefix_points,
            num_points,
            reservoir_bytes,
            resident_reservoir_layout.as_str(),
            if overlap_enabled { "D0-overlap" } else { "D1" },
            reservoir_init_config.workers,
        );
        let deferred_reservoirs = DeferredReservoirs::new_with_policies_and_diagnostics(
            dense_resident_points,
            params.l_max,
            hashprune_evidence_policy,
            hashprune_backfill_policy,
            diagnostics
                .as_ref()
                .map(HashPruneDiagnosticsSession::collector_arc),
        )
        .with_layout(resident_reservoir_layout)
        .with_initialization_config(ReservoirInitializationConfig {
            overlap_enabled,
            ..reservoir_init_config
        });
        let deferred_reservoirs = match &d0_read_progress {
            Some(progress) => deferred_reservoirs.with_d0_read_progress(Arc::clone(progress)),
            None => deferred_reservoirs,
        };
        let resident_sketches = if sq_staging_fused_sketch_candidate {
            None
        } else {
            let resident_sketches = ResidentPrefixSketchAccessor::from_disk_prefix(
                disk_sketches
                    .as_ref()
                    .expect("HashPrune OOM requires disk sketches"),
                resident_points,
            )?;
            let total_sketch_bytes =
                sketch_bytes.saturating_add(resident_sketches.resident_bytes());
            effective_sketch_bytes = total_sketch_bytes;
            Some(resident_sketches)
        };
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let online_pipeline = if memory_plan.online_suffix_enabled {
            Some(OnlineSuffixPipeline::start(online_suffix_reduce_config(
                params,
                &artifact_dir,
                online_suffix_residency.pipeline_source_start,
                num_points,
                online_suffix_residency.pinned_prefix_points,
                shard_points,
                pool_threads,
                &memory_plan,
                io_cfg,
                hashprune_evidence_policy,
                hashprune_backfill_policy,
                diagnostics
                    .as_ref()
                    .map(HashPruneDiagnosticsSession::collector_arc),
            ))?)
        } else {
            None
        };
        let spill_sink = if online_pipeline.is_none() {
            Some(create_candidate_spill_sink(
                &candidate_path,
                params,
                shard_points,
                io_cfg,
                hashprune_evidence_policy,
                hashprune_backfill_policy,
            )?)
        } else {
            None
        };
        let suffix_sink: &dyn PendingEdgeSink = match (&online_pipeline, &spill_sink) {
            (Some(pipeline), None) => pipeline,
            (None, Some(spill_sink)) => spill_sink,
            _ => {
                return Err(AnnError::log_index_error(
                    "hybrid suffix reduction selected an invalid sink combination".to_string(),
                ));
            }
        };
        let hybrid_sink = HybridEdgeSink::new_with_policies(
            dense_resident_points,
            &deferred_reservoirs,
            suffix_sink,
            hashprune_evidence_policy,
            hashprune_backfill_policy,
        );
        let pipeline_d0_read_progress = d0_read_progress.clone();
        let streaming_start = Instant::now();
        let streaming_profile = run_with_reservoir_initialization_overlap(
            overlap_enabled,
            d0_read_progress.as_deref(),
            || deferred_reservoirs.initialize(),
            || {
                if sq_staging_fused_sketch_candidate {
                    let (
                        profile,
                        fused_sketches,
                        fused_sketch_compute_wall,
                        fused_sketch_join_wait_wall,
                    ) = run_streaming_partition_pipeline_with_sq_staging_fused_sketch(
                        dataset,
                        num_points,
                        pool_threads,
                        params,
                        &artifact_dir,
                        &sketch_path,
                        detailed_profiling,
                        compute_backend,
                        &hybrid_sink,
                        gpu_subtree_runtime_context,
                        pipeline_d0_read_progress,
                    )?;
                    sq_staging_fused_sketch_enabled = true;
                    sq_staging_fused_sketch_compute_wall = fused_sketch_compute_wall;
                    sq_staging_fused_sketch_join_wait_wall = fused_sketch_join_wait_wall;
                    effective_sketch_bytes = fused_sketches.resident_bytes();
                    log_memory_telemetry(
                        "oom after SQ staging fused sketches",
                        num_points,
                        params,
                        dataset.dim(),
                        effective_sketch_bytes,
                        reservoir_bytes,
                        leaf_budget_threads,
                    );
                    Ok(profile)
                } else {
                    run_streaming_partition_pipeline_with_gpu_subtree_context(
                        dataset,
                        num_points,
                        pool_threads,
                        leaf_budget_threads,
                        params,
                        &artifact_dir,
                        resident_sketches
                            .as_ref()
                            .expect("HashPrune hybrid OOM requires resident prefix sketches"),
                        detailed_profiling,
                        compute_backend,
                        &hybrid_sink,
                        gpu_subtree_runtime_context,
                        None,
                    )
                }
            },
        )?;
        let streaming_wall = streaming_start.elapsed();
        if let Some(pipeline) = &online_pipeline {
            pipeline.mark_gpu_complete();
        }
        if params.root_stop_after_d00 {
            println!(
                "{}",
                root_fanout_profile_json(&streaming_profile.rbc.root_fanout)
            );
            info!(
                "PiPNN OOM root-stop-after-d00 completed: artifact_dir={:?}",
                artifact_dir
            );
            return Ok(PiPNNGraphBuildResult::default());
        }
        drop(hybrid_sink);

        match (online_pipeline, spill_sink) {
            (Some(pipeline), None) => {
                let online_state = pipeline.finish()?;
                let online_stats = online_state.stats().clone();

                let emit_start = Instant::now();
                let allocation = deferred_reservoirs.into_allocation()?;
                debug_assert_eq!(allocation.estimated_bytes, reservoir_bytes);
                let reservoir_allocation_wall = allocation.allocation_wall;
                reservoir_initialization = allocation.initialization.clone();
                let ResidentEmitResult {
                    backfill: mut hybrid_backfill_profile,
                    telemetry: resident_emit_profile,
                } = if let Some(session) = diagnostics.as_mut() {
                    let (collector, finalizer) = session.parts();
                    emit_resident_storage(
                        0,
                        allocation.reservoirs,
                        hashprune_backfill_policy,
                        Some(collector),
                        Some(finalizer),
                        on_node_done,
                    )?
                } else {
                    emit_resident_storage(
                        0,
                        allocation.reservoirs,
                        hashprune_backfill_policy,
                        None,
                        None,
                        on_node_done,
                    )?
                };
                let window_sources = memory_plan
                    .online_suffix_decode_budget_bytes
                    .checked_div(online_state.record_bytes())
                    .unwrap_or(0)
                    .max(1);
                let suffix_backfill_profile = if let Some(session) = diagnostics.as_mut() {
                    let (collector, finalizer) = session.parts();
                    online_state.emit_rows(
                        window_sources,
                        Some(collector),
                        Some(finalizer),
                        on_node_done,
                    )?
                } else {
                    online_state.emit_rows(window_sources, None, None, on_node_done)?
                };
                let emit_wall = emit_start.elapsed();

                let mut reduce_stats = SpillReduceStats::default();
                reduce_stats.evidence = online_stats.evidence.clone();
                reduce_stats.backfill = online_stats.backfill.clone();
                reduce_stats.backfill.merge(suffix_backfill_profile);
                hybrid_backfill_profile.merge(reduce_stats.backfill.clone());
                if !params.oom_keep_artifacts {
                    online_state.cleanup()?;
                }

                (
                    streaming_profile,
                    streaming_wall,
                    SpillRunStats::default(),
                    reduce_stats,
                    Duration::ZERO,
                    emit_wall,
                    reservoir_bytes,
                    reservoir_allocation_wall,
                    LrpfProfileCounters::default(),
                    resident_emit_profile,
                    hybrid_backfill_profile,
                    online_stats,
                )
            }
            (None, Some(spill_sink)) => {
                let spill_artifacts = spill_sink.finish()?;
                let spill_stats =
                    ExternalHashPruneReducer::summarize_spill_artifacts(&spill_artifacts);

                let emit_start = Instant::now();
                let allocation = deferred_reservoirs.into_allocation()?;
                debug_assert_eq!(allocation.estimated_bytes, reservoir_bytes);
                let reservoir_allocation_wall = allocation.allocation_wall;
                reservoir_initialization = allocation.initialization.clone();
                let ResidentEmitResult {
                    backfill: mut hybrid_backfill_profile,
                    telemetry: resident_emit_profile,
                } = if let Some(session) = diagnostics.as_mut() {
                    let (collector, finalizer) = session.parts();
                    emit_resident_storage(
                        0,
                        allocation.reservoirs,
                        hashprune_backfill_policy,
                        Some(collector),
                        Some(finalizer),
                        on_node_done,
                    )?
                } else {
                    emit_resident_storage(
                        0,
                        allocation.reservoirs,
                        hashprune_backfill_policy,
                        None,
                        None,
                        on_node_done,
                    )?
                };
                let emit_wall = emit_start.elapsed();

                let reduce_start = Instant::now();
                let reduce_config = external_reduce_config_for_eligibility(
                    io_cfg,
                    pool_threads,
                    online_suffix_eligibility,
                );
                if reduce_config.workers > pool_threads {
                    info!(
                        "PiPNN OOM external HashPrune adaptive concurrency enabled: build_threads={} initial_workers={} max_workers={} batch_records={}",
                        pool_threads,
                        reduce_config.initial_workers,
                        reduce_config.workers,
                        reduce_config.batch_records
                    );
                }
                let reduce_stats = if let Some(session) = diagnostics.as_mut() {
                    let (collector, finalizer) = session.parts();
                    ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_from_with_execution_config(
                        resident_points,
                        num_points,
                        params.l_max,
                        shard_points,
                        &spill_artifacts,
                        hashprune_evidence_policy,
                        hashprune_backfill_policy,
                        Some(collector),
                        Some(finalizer),
                        reduce_config,
                        |point, neighbors| on_node_done(point, neighbors),
                    )?
                } else {
                    ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_from_with_execution_config(
                        resident_points,
                        num_points,
                        params.l_max,
                        shard_points,
                        &spill_artifacts,
                        hashprune_evidence_policy,
                        hashprune_backfill_policy,
                        None,
                        None,
                        reduce_config,
                        |point, neighbors| on_node_done(point, neighbors),
                    )?
                };
                let reduce_wall = reduce_start.elapsed();
                hybrid_backfill_profile.merge(reduce_stats.backfill.clone());

                (
                    streaming_profile,
                    streaming_wall,
                    spill_stats,
                    reduce_stats,
                    reduce_wall,
                    emit_wall,
                    reservoir_bytes,
                    reservoir_allocation_wall,
                    LrpfProfileCounters::default(),
                    resident_emit_profile,
                    hybrid_backfill_profile,
                    OnlineSuffixReduceStats::default(),
                )
            }
            _ => {
                return Err(AnnError::log_index_error(
                    "hybrid suffix reduction finished with an invalid sink combination".to_string(),
                ));
            }
        }
    } else {
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let online_pipeline = if memory_plan.online_suffix_enabled {
            Some(OnlineSuffixPipeline::start(online_suffix_reduce_config(
                params,
                &artifact_dir,
                0,
                num_points,
                0,
                shard_points,
                pool_threads,
                &memory_plan,
                io_cfg,
                hashprune_evidence_policy,
                hashprune_backfill_policy,
                diagnostics
                    .as_ref()
                    .map(HashPruneDiagnosticsSession::collector_arc),
            ))?)
        } else {
            None
        };
        let spill_sink = if online_pipeline.is_none() {
            Some(create_candidate_spill_sink(
                &candidate_path,
                params,
                shard_points,
                io_cfg,
                hashprune_evidence_policy,
                hashprune_backfill_policy,
            )?)
        } else {
            None
        };
        let suffix_sink: &dyn PendingEdgeSink = match (&online_pipeline, &spill_sink) {
            (Some(pipeline), None) => pipeline,
            (None, Some(spill_sink)) => spill_sink,
            _ => {
                return Err(AnnError::log_index_error(
                    "full suffix reduction selected an invalid sink combination".to_string(),
                ));
            }
        };
        let streaming_start = Instant::now();
        let streaming_profile = if sq_staging_fused_sketch_candidate {
            let (profile, fused_sketches, fused_sketch_compute_wall, fused_sketch_join_wait_wall) =
                run_streaming_partition_pipeline_with_sq_staging_fused_sketch(
                    dataset,
                    num_points,
                    pool_threads,
                    params,
                    &artifact_dir,
                    &sketch_path,
                    detailed_profiling,
                    compute_backend,
                    suffix_sink,
                    gpu_subtree_runtime_context,
                    None,
                )?;
            sq_staging_fused_sketch_enabled = true;
            sq_staging_fused_sketch_compute_wall = fused_sketch_compute_wall;
            sq_staging_fused_sketch_join_wait_wall = fused_sketch_join_wait_wall;
            effective_sketch_bytes = fused_sketches.resident_bytes();
            log_memory_telemetry(
                "oom after SQ staging fused sketches",
                num_points,
                params,
                dataset.dim(),
                effective_sketch_bytes,
                0,
                leaf_budget_threads,
            );
            profile
        } else {
            run_streaming_partition_pipeline_with_gpu_subtree_context(
                dataset,
                num_points,
                pool_threads,
                leaf_budget_threads,
                params,
                &artifact_dir,
                streaming_sketches,
                detailed_profiling,
                compute_backend,
                suffix_sink,
                gpu_subtree_runtime_context,
                None,
            )?
        };
        let streaming_wall = streaming_start.elapsed();
        if let Some(pipeline) = &online_pipeline {
            pipeline.mark_gpu_complete();
        }
        if params.root_stop_after_d00 {
            println!(
                "{}",
                root_fanout_profile_json(&streaming_profile.rbc.root_fanout)
            );
            info!(
                "PiPNN OOM root-stop-after-d00 completed: artifact_dir={:?}",
                artifact_dir
            );
            return Ok(PiPNNGraphBuildResult::default());
        }
        match (online_pipeline, spill_sink) {
            (Some(pipeline), None) => {
                let online_state = pipeline.finish()?;
                let online_stats = online_state.stats().clone();
                let window_sources = memory_plan
                    .online_suffix_decode_budget_bytes
                    .checked_div(online_state.record_bytes())
                    .unwrap_or(0)
                    .max(1);
                let emit_start = Instant::now();
                let emitted_backfill = if let Some(session) = diagnostics.as_mut() {
                    let (collector, finalizer) = session.parts();
                    online_state.emit_rows(
                        window_sources,
                        Some(collector),
                        Some(finalizer),
                        on_node_done,
                    )?
                } else {
                    online_state.emit_rows(window_sources, None, None, on_node_done)?
                };
                let emit_wall = emit_start.elapsed();
                let mut reduce_stats = SpillReduceStats::default();
                reduce_stats.evidence = online_stats.evidence.clone();
                reduce_stats.backfill = online_stats.backfill.clone();
                reduce_stats.backfill.merge(emitted_backfill);
                let online_backfill_profile = reduce_stats.backfill.clone();
                if !params.oom_keep_artifacts {
                    online_state.cleanup()?;
                }
                info!(
                    "PiPNN OOM online suffix mode done: streaming_elapsed={:?} drain_elapsed={:?} emit_elapsed={:?}",
                    streaming_wall, online_stats.post_gpu_drain_wall, emit_wall
                );

                (
                    streaming_profile,
                    streaming_wall,
                    SpillRunStats::default(),
                    reduce_stats,
                    Duration::ZERO,
                    emit_wall,
                    0,
                    Duration::ZERO,
                    LrpfProfileCounters::default(),
                    ResidentEmitProfile::default(),
                    online_backfill_profile,
                    online_stats,
                )
            }
            (None, Some(spill_sink)) => {
                let spill_artifacts = spill_sink.finish()?;
                let spill_stats =
                    ExternalHashPruneReducer::summarize_spill_artifacts(&spill_artifacts);

                let reduce_start = Instant::now();
                let reduce_config = external_reduce_config_for_eligibility(
                    io_cfg,
                    pool_threads,
                    online_suffix_eligibility,
                );
                if reduce_config.workers > pool_threads {
                    info!(
                        "PiPNN OOM external HashPrune adaptive concurrency enabled: build_threads={} initial_workers={} max_workers={} batch_records={}",
                        pool_threads,
                        reduce_config.initial_workers,
                        reduce_config.workers,
                        reduce_config.batch_records
                    );
                }
                let reduce_stats = if let Some(session) = diagnostics.as_mut() {
                    let (collector, finalizer) = session.parts();
                    ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config(
                        num_points,
                        params.l_max,
                        shard_points,
                        &spill_artifacts,
                        hashprune_evidence_policy,
                        hashprune_backfill_policy,
                        Some(collector),
                        Some(finalizer),
                        reduce_config,
                        |point, neighbors| on_node_done(point, neighbors),
                    )?
                } else {
                    ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config(
                        num_points,
                        params.l_max,
                        shard_points,
                        &spill_artifacts,
                        hashprune_evidence_policy,
                        hashprune_backfill_policy,
                        None,
                        None,
                        reduce_config,
                        |point, neighbors| on_node_done(point, neighbors),
                    )?
                };
                let reduce_wall = reduce_start.elapsed();
                let external_backfill_profile = reduce_stats.backfill.clone();
                info!(
                    "PiPNN OOM external-reduce mode done: streaming_elapsed={:?} reduce_elapsed={:?}",
                    streaming_wall, reduce_wall
                );

                (
                    streaming_profile,
                    streaming_wall,
                    spill_stats,
                    reduce_stats,
                    reduce_wall,
                    Duration::ZERO,
                    0,
                    Duration::ZERO,
                    LrpfProfileCounters::default(),
                    ResidentEmitProfile::default(),
                    external_backfill_profile,
                    OnlineSuffixReduceStats::default(),
                )
            }
            _ => {
                return Err(AnnError::log_index_error(
                    "full suffix reduction finished with an invalid sink combination".to_string(),
                ));
            }
        }
    };

    hashprune_backfill_profile.merge(streaming_profile.leaf.hashprune_backfill.clone());
    if let Some(session) = diagnostics.take() {
        session.finish(
            streaming_profile.entry_point.unwrap_or(0),
            &hashprune_backfill_profile,
        )?;
    }

    let (uring_batches, uring_windows, uring_bytes, uring_fallbacks) = uring_gather_counters();
    if detailed_profiling {
        info!(
            "PiPNN OOM point io_uring gather: batches={} windows={} bytes={} fallbacks={}",
            uring_batches, uring_windows, uring_bytes, uring_fallbacks,
        );
    }

    if let Some(path) = profile_path {
        let hashprune_evidence = final_hashprune_evidence_profile(
            &streaming_profile.leaf.hashprune_evidence,
            &reduce_stats.evidence,
        );
        let persistent_scheduler_mode_actual = persistent_scheduler_mode_actual(
            params.gpu_subtree_execution_mode,
            params.gpu_subtree_persistent_scheduler_mode,
        );
        write_oom_profile(
            &path,
            &OomProfile {
                num_points,
                dim: dataset.dim(),
                leaf_knn: params.leaf_knn,
                partitioner: params.partitioner,
                root_assignment_backend: params.gpu_subtree_root_assignment_backend,
                assignment_backend: params.gpu_subtree_assignment_backend,
                d0_dco_mode: params.gpu_subtree_d0_dco_mode,
                d0_dco_rerank_mode: params.gpu_subtree_d0_dco_rerank_mode,
                d0_dco_survivors: params.gpu_subtree_d0_dco_survivors,
                d0_dco_audit: params.gpu_subtree_d0_dco_audit,
                d0_sq_cpu_kernel: params.gpu_subtree_d0_sq_cpu_kernel,
                d1plus_lazy_split_mode: params.gpu_subtree_d1plus_lazy_split_mode,
                assignment_dco_mode: params.gpu_subtree_assignment_dco_mode,
                assignment_dco_rerank_mode: params.gpu_subtree_assignment_dco_rerank_mode,
                assignment_dco_proj_dim: params.gpu_subtree_assignment_dco_proj_dim,
                assignment_dco_survivors: params.gpu_subtree_assignment_dco_survivors,
                assignment_dco_audit: params.gpu_subtree_assignment_dco_audit,
                leaf_topk_backend: params.gpu_subtree_leaf_topk_backend,
                leaf_dco_mode: params.gpu_subtree_leaf_dco_mode,
                leaf_dco_rerank_mode: params.gpu_subtree_leaf_dco_rerank_mode,
                leaf_dco_proj_dim: params.gpu_subtree_leaf_dco_proj_dim,
                leaf_dco_survivors: params.gpu_subtree_leaf_dco_survivors,
                leaf_dco_audit: params.gpu_subtree_leaf_dco_audit,
                leaf_dco_ddc_res_pca_matrix: params.gpu_subtree_leaf_dco_ddc_res_pca_matrix.clone(),
                leaf_dco_ddc_res_projected_input: params
                    .gpu_subtree_leaf_dco_ddc_res_projected_input,
                leaf_dco_ddc_res_base_dim: params.gpu_subtree_leaf_dco_ddc_res_base_dim,
                leaf_dco_ddc_res_sigma_count: params.gpu_subtree_leaf_dco_ddc_res_sigma_count,
                rabitq_plus_mode: params.gpu_subtree_rabitq_plus_mode,
                rabitq_plus_bits: params.gpu_subtree_rabitq_plus_bits,
                rabitq_plus_rotator: params.gpu_subtree_rabitq_plus_rotator,
                rabitq_plus_seed: params.gpu_subtree_rabitq_plus_seed,
                rabitq_plus_sidecar_mode: params.gpu_subtree_rabitq_plus_sidecar_mode,
                rabitq_plus_sidecar_prefix: params.gpu_subtree_rabitq_plus_sidecar_prefix.clone(),
                rabitq_plus_fused_sidecar_sketch_enabled: fused_rabitq_plus_sidecar_sketch
                    .is_some(),
                rabitq_plus_fused_prestage_total_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.total_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_chunk_alloc_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.chunk_alloc_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_raw_read_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.raw_read_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_source_hash_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.source_hash_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_quantize_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.quantize_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_code_output_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.code_output_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_norm_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.norm_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_meta_output_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.meta_output_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_header_write_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.header_write_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sidecar_write_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sidecar.write_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sketch_alloc_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sketch_alloc_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sketch_compute_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sketch_compute_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sketch_extend_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sketch_extend_wall)
                    .unwrap_or_default(),
                rabitq_plus_fused_sketch_write_wall: fused_rabitq_plus_sidecar_sketch
                    .as_ref()
                    .map(|report| report.sketch_write_wall)
                    .unwrap_or_default(),
                qcode_staging_mode: params.gpu_subtree_qcode_staging_mode,
                qcode_d1_payload_mode: params.gpu_subtree_qcode_d1_payload_mode,
                qcode_rabitq_d1_distance_mode: params.gpu_subtree_rabitq_d1_distance_mode,
                qcode_rabitq_short_survivors: params.gpu_subtree_rabitq_short_survivors,
                rabitq_residual_centroids: params.gpu_subtree_rabitq_residual_centroids,
                qcode_staging_upload_shard_rows: params.gpu_subtree_qcode_staging_upload_shard_rows,
                qcode_global_d0_leaders: params.gpu_subtree_qcode_global_d0_leaders,
                sq_staging_mode: params.gpu_subtree_sq_staging_mode,
                sq_staging_upload_shard_rows: params.gpu_subtree_sq_staging_upload_shard_rows,
                sq_bits: params.gpu_subtree_sq_bits,
                sq_native_int4: params.gpu_subtree_sq_native_int4,
                sq_wmma_mode: params.gpu_subtree_sq_wmma_mode,
                persistent_large_ticket_batch_requested: params
                    .gpu_subtree_persistent_large_ticket_batch,
                persistent_sq8_assignment_kernel_requested: params
                    .gpu_subtree_sq8_assignment_kernel,
                persistent_sq8_leaf_executor_requested: params.gpu_subtree_sq8_leaf_executor,
                persistent_sq8_leaf_mode_requested: params.gpu_subtree_sq8_leaf_mode,
                persistent_scheduler_mode_requested: params.gpu_subtree_persistent_scheduler_mode,
                persistent_scheduler_mode_actual,
                persistent_idle_backoff_ns_requested: params.gpu_subtree_persistent_idle_backoff_ns,
                persistent_idle_backoff_ns_actual: if persistent_scheduler_mode_actual
                    == PiPNNGpuSubtreePersistentSchedulerMode::AdaptiveIdle
                {
                    params.gpu_subtree_persistent_idle_backoff_ns
                } else {
                    0
                },
                sq_wmma_sidecar_mode: params.gpu_subtree_sq_wmma_sidecar_mode,
                sq_wmma_sidecar_prefix: params.gpu_subtree_sq_wmma_sidecar_prefix.clone(),
                sq_staging_fused_sketch_enabled,
                sq_staging_fused_sketch_compute_wall,
                sq_staging_fused_sketch_join_wait_wall,
                turboquant_mode: params.gpu_subtree_turboquant_mode,
                turboquant_bits: params.gpu_subtree_turboquant_bits,
                turboquant_seed: params.gpu_subtree_turboquant_seed,
                turboquant_sidecar_mode: params.gpu_subtree_turboquant_sidecar_mode,
                turboquant_sidecar_prefix: params.gpu_subtree_turboquant_sidecar_prefix.clone(),
                turboquant_cascade_low_dim: params.gpu_subtree_turboquant_cascade_low_dim,
                turboquant_cascade_survivor_cap: params.gpu_subtree_turboquant_cascade_survivor_cap,
                turboquant_adaptive_survivor_threshold: params
                    .gpu_subtree_turboquant_adaptive_survivor_threshold,
                turboquant_adaptive_survivor_low_cap: params
                    .gpu_subtree_turboquant_adaptive_survivor_low_cap,
                memory_budget_bytes,
                memory_budget_source: if params.oom_memory_budget_bytes == 0 {
                    "auto-host-cgroup"
                } else {
                    "explicit"
                },
                oom_io_mode: params.effective_oom_io_mode(),
                oom_reservoir_budget_share: params.effective_oom_reservoir_budget_share(),
                oom_sidecar_resident_budget_cap_bytes: params.oom_sidecar_resident_budget_cap_bytes,
                effective_threads: pool_threads,
                pool_threads,
                leaf_budget_threads,
                resident_reservoir_points: resident_plan.resident_points,
                resident_reservoir_bytes,
                reservoir_allocation_wall,
                reservoir_initialization,
                reduce_shard_points: lrpf_shard_points,
                compact_spill_digest_enabled: compact_spill_digest_width.is_some(),
                compact_spill_digest_width: compact_spill_digest_width.unwrap_or(0),
                prune_backend: params.prune_backend,
                hashprune_evidence_policy,
                hashprune_backfill_policy,
                hashprune_backfill_degree_cap: params.l_max,
                lrpf_target_degree: effective_lrpf_target_degree(params),
                lrpf_long_edge_quota: params.lrpf_long_edge_quota,
                lrpf_pre_spill_compress_enabled: params.lrpf_pre_spill_compress_enable,
                lrpf_pre_spill_width: if params.lrpf_pre_spill_compress_enable {
                    effective_lrpf_pre_spill_width(params)
                } else {
                    0
                },
                lrpf_candidate_records: lrpf_profile_counters.candidate_records,
                lrpf_witness_records: lrpf_profile_counters.witness_records,
                lrpf_fragments: lrpf_profile_counters.fragments,
                lrpf_pre_spill_pruned_candidates: lrpf_profile_counters.pre_spill_pruned_candidates,
                lrpf_reducer_pruned_candidates: lrpf_profile_counters.reducer_pruned_candidates,
                reservoir_budget_bytes: memory_plan.reservoir_budget_bytes,
                resident_subtree_budget_bytes: memory_plan.resident_subtree_budget_bytes,
                required_resident_bytes: memory_plan.required_resident_bytes,
                optional_resident_budget_bytes: memory_plan.optional_resident_budget_bytes,
                optional_resident_admitted_bytes: memory_plan.optional_resident_admitted_bytes,
                optional_resident_unassigned_bytes: memory_plan.optional_resident_unassigned_bytes,
                transient_pipeline_budget_bytes: memory_plan.transient_pipeline_budget_bytes,
                d1plus_working_set_budget_bytes: memory_plan.d1plus_working_set_budget_bytes,
                sidecar_resident_kind: memory_plan.sidecar_resident_kind,
                sidecar_resident_budget_bytes: memory_plan.sidecar_resident_budget_bytes,
                sidecar_resident_required_bytes: memory_plan.sidecar_resident_required_bytes,
                sidecar_resident_admitted_rows: memory_plan.sidecar_resident_admitted_rows,
                sidecar_resident_admissible: memory_plan.sidecar_resident_admissible,
                scratch_budget_bytes: memory_plan.scratch_budget_bytes,
                io_headroom_bytes: memory_plan.io_headroom_bytes,
                raw_vector_cache_budget_bytes: memory_plan.raw_vector_cache_budget_bytes,
                raw_vector_cache_rows: raw_vector_cache_row_capacity(
                    memory_plan.raw_vector_cache_budget_bytes,
                    dataset.dim(),
                ),
                online_suffix_mode: params.online_suffix_reduce_mode,
                online_suffix_fallback_reason: match online_suffix_eligibility {
                    OnlineSuffixEligibility::ExternalFallback(reason) => Some(reason),
                    OnlineSuffixEligibility::Disabled | OnlineSuffixEligibility::Enabled => None,
                },
                online_suffix_config: online_suffix_profile_config(
                    params,
                    &memory_plan,
                    pool_threads,
                    num_points.saturating_sub(online_suffix_residency.pipeline_source_start),
                    shard_points,
                ),
                online_suffix: online_suffix_stats,
                sketch_bytes: effective_sketch_bytes,
                sketch_file_bytes: spill_bytes,
                spill: spill_stats,
                reduce: reduce_stats,
                sketch_persist_wall,
                streaming_wall,
                reduce_wall,
                emit_wall,
                resident_emit: resident_emit_profile,
                total_wall: overall_start.elapsed(),
                scheduler: streaming_profile.telemetry,
                rbc: streaming_profile.rbc,
                kmeans: streaming_profile.kmeans,
                hashprune_evidence,
                hashprune_backfill: hashprune_backfill_profile,
                leaf: streaming_profile.leaf,
                memory: collect_oom_memory_report(),
                temp_files: collect_temp_file_report(&artifact_dir)?,
                uring_batches,
                uring_windows,
                uring_bytes,
                uring_fallbacks,
            },
        )?;
    }

    if resident_reservoir_bytes > 0 {
        info!(
            "PiPNN OOM completed with resident reservoirs: estimated_bytes={resident_reservoir_bytes} budget_bytes={memory_budget_bytes}"
        );
    }

    if !params.oom_keep_artifacts {
        cleanup_oom_artifacts(&artifact_dir)?;
    }

    Ok(PiPNNGraphBuildResult {
        entry_point: streaming_profile.entry_point,
    })
}

fn final_hashprune_evidence_profile(
    leaf_evidence: &HashPruneEvidenceProfile,
    reduce_evidence: &HashPruneEvidenceProfile,
) -> HashPruneEvidenceProfile {
    let mut evidence = leaf_evidence.clone();
    evidence.merge(reduce_evidence.clone());
    evidence
}

#[cfg(test)]
fn run_streaming_partition_pipeline(
    dataset: &dyn PointStore,
    num_points: usize,
    pool_threads: usize,
    leaf_budget_threads: usize,
    params: &PiPNNParams,
    artifact_dir: &Path,
    sketches: &dyn SketchAccessor,
    detailed_profiling: bool,
    leaf_compute_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    edge_sink: &dyn PendingEdgeSink,
) -> AnnResult<StreamingProfile> {
    run_streaming_partition_pipeline_with_gpu_subtree_context(
        dataset,
        num_points,
        pool_threads,
        leaf_budget_threads,
        params,
        artifact_dir,
        sketches,
        detailed_profiling,
        leaf_compute_backend,
        edge_sink,
        GpuSubtreeRuntimeContext::default(),
        None,
    )
}

fn run_streaming_partition_pipeline_with_gpu_subtree_context(
    dataset: &dyn PointStore,
    num_points: usize,
    pool_threads: usize,
    leaf_budget_threads: usize,
    params: &PiPNNParams,
    artifact_dir: &Path,
    sketches: &dyn SketchAccessor,
    detailed_profiling: bool,
    leaf_compute_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    edge_sink: &dyn PendingEdgeSink,
    gpu_subtree_runtime_context: GpuSubtreeRuntimeContext<'_>,
    lrpf_fragment_sink: Option<&dyn LrpfFragmentSink>,
) -> AnnResult<StreamingProfile> {
    let t_rbc_leaf = Instant::now();
    let all_indices: Vec<u32> = (0..num_points as u32).collect();
    if matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree)
        && !matches!(leaf_compute_backend.kind(), PiPNNComputeBackendKind::Cuda)
    {
        return Err(AnnError::log_index_error(
            "gpu-subtree requires CUDA compute backend".to_string(),
        ));
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    pb.set_message("PiPNN streaming processing...");

    let leaf_scratch_bytes = estimate_leaf_scratch_bytes(
        params.kernel_safe_leaf_size(),
        dataset.dim(),
        params.leaf_knn,
    );
    let scheduler_budget = SchedulerBudget::for_memory_budget(
        leaf_budget_threads,
        params.kernel_safe_leaf_size(),
        params.effective_oom_memory_budget_bytes(),
    );

    BlasThreadEnv::set_single_threaded();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .stack_size(PIPNN_RAYON_STACK_BYTES)
        .build()
        .map_err(|e| AnnError::log_index_error(format!("Failed to create Rayon pool: {}", e)))?;

    let leaf_context = LeafTaskContext {
        compute_backend: leaf_compute_backend,
        dataset,
        metric: params.metric(),
        sketches,
        params,
        edge_sink,
        detailed_profiling,
        #[cfg(test)]
        observer: None,
    };
    let producer_start = Instant::now();
    let mut rbc_telemetry = RbcPartitionTelemetry::default();
    let mut kmeans_telemetry = KMeansPartitionTelemetry::default();
    let subtree_backend = if matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree) {
        leaf_compute_backend
    } else {
        cpu_backend()
    };
    if matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree) {
        let root_assignment_backend = match params.gpu_subtree_root_assignment_backend {
            PiPNNComputeBackendKind::Cpu | PiPNNComputeBackendKind::Auto => cpu_backend(),
            PiPNNComputeBackendKind::Cuda => leaf_compute_backend,
        };
        if uses_lrpf_candidate_backend(params) {
            let Some(fragment_sink) = lrpf_fragment_sink else {
                return Err(AnnError::log_index_error(
                    "prune_backend=lrpf requires an LRPF fragment sink".to_string(),
                ));
            };
            let compact_sink = OomLrpfCompactTopKSink::new(params, fragment_sink);
            let compact_profile = pool.install(|| {
                emit_streaming_partition_compact_topk(
                    root_assignment_backend,
                    subtree_backend,
                    dataset,
                    &all_indices,
                    params,
                    pool_threads,
                    Some(artifact_dir),
                    &compact_sink,
                    gpu_subtree_runtime_context,
                )
            })?;
            let producer_wall = producer_start.elapsed();
            let entry_point = compact_profile.entry_point;
            pb.finish_with_message("PiPNN gpu-subtree LRPF compact top-k done");
            return Ok(StreamingProfile {
                producer_wall,
                total_wall: t_rbc_leaf.elapsed(),
                leaf: compact_profile.leaf,
                telemetry: SchedulerTelemetry::default(),
                rbc: compact_profile.rbc,
                kmeans: KMeansPartitionTelemetry::default(),
                entry_point,
            });
        }
        let compact_sink = OomCompactTopKSink::new(sketches, params, edge_sink);
        let compact_profile = pool.install(|| {
            emit_streaming_partition_compact_topk(
                root_assignment_backend,
                subtree_backend,
                dataset,
                &all_indices,
                params,
                pool_threads,
                Some(artifact_dir),
                &compact_sink,
                gpu_subtree_runtime_context,
            )
        })?;
        let producer_wall = producer_start.elapsed();
        let entry_point = compact_profile.entry_point;
        pb.finish_with_message("PiPNN gpu-subtree compact top-k done");
        return Ok(StreamingProfile {
            producer_wall,
            total_wall: t_rbc_leaf.elapsed(),
            leaf: compact_profile.leaf,
            telemetry: SchedulerTelemetry::default(),
            rbc: compact_profile.rbc,
            kmeans: KMeansPartitionTelemetry::default(),
            entry_point,
        });
    }
    let (scheduler_profile, scheduler_telemetry) = if params.pipeline_runtime_enable {
        run_pipeline_scheduler_scope(&pool, scheduler_budget, leaf_context, &pb, |emitter| {
            emit_streaming_partition_leaves(
                subtree_backend,
                dataset,
                &all_indices,
                params,
                pool_threads,
                Some(artifact_dir),
                emitter,
                &mut rbc_telemetry,
                &mut kmeans_telemetry,
            )?;
            Ok(())
        })?
    } else {
        run_scheduler_scope(&pool, scheduler_budget, leaf_context, &pb, |emitter| {
            emit_streaming_partition_leaves(
                subtree_backend,
                dataset,
                &all_indices,
                params,
                pool_threads,
                Some(artifact_dir),
                emitter,
                &mut rbc_telemetry,
                &mut kmeans_telemetry,
            )?;
            Ok(())
        })?
    };
    let producer_wall = producer_start.elapsed();

    pb.finish_with_message("PiPNN streaming processing done");
    info!(
        "PiPNN streaming (RBC+Leaf) done: elapsed={:?}",
        t_rbc_leaf.elapsed()
    );
    info!(
        "PiPNN leaf scratch estimate: max_leaf_size={} per_worker_bytes={} active_scratch_bytes_estimate={}",
        params.kernel_safe_leaf_size(),
        leaf_scratch_bytes,
        leaf_scratch_bytes.saturating_mul(leaf_budget_threads),
    );
    info!(
        "PiPNN streaming thread model: pool_threads={} leaf_budget_threads={}",
        pool_threads, leaf_budget_threads
    );
    info!(
        "PiPNN pipeline runtime: enabled={} mode={}",
        params.pipeline_runtime_enable, params.pipeline_compute_mode
    );

    Ok(StreamingProfile {
        producer_wall,
        total_wall: t_rbc_leaf.elapsed(),
        leaf: scheduler_profile,
        telemetry: scheduler_telemetry,
        rbc: rbc_telemetry,
        kmeans: kmeans_telemetry,
        entry_point: None,
    })
}

fn emit_streaming_partition_compact_topk(
    root_assignment_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    subtree_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    dataset: &dyn PointStore,
    all_indices: &[u32],
    params: &PiPNNParams,
    pool_threads: usize,
    artifact_dir: Option<&Path>,
    compact_sink: &dyn CompactTopKSink,
    gpu_subtree_runtime_context: GpuSubtreeRuntimeContext<'_>,
) -> AnnResult<GpuSubtreeStreamingProfile> {
    let mut rng = StdRng::seed_from_u64(params.random_seed);
    if params.gpu_subtree_qcode_staging_mode.is_enabled() {
        let (d0_plan, rabitq_residual_payload) = run_gpu_subtree_qcode_staging_preflight(
            subtree_backend,
            dataset,
            all_indices,
            params.metric(),
            params,
            pool_threads as u32,
            &mut rng,
            artifact_dir,
        )?;
        if !params.root_stop_after_d00 && !d0_plan.child_runs.is_empty() {
            compact_sink.prepare_for_d1()?;
        }
        let qcode_runtime_context = GpuSubtreeRuntimeContext {
            rabitq_residual_qcode_store: Some(&rabitq_residual_payload),
            ..gpu_subtree_runtime_context
        };
        let queue_budget_bytes = params
            .effective_oom_memory_budget_bytes()
            .saturating_mul(5)
            .saturating_div(100)
            .max(64 * 1024 * 1024);
        return run_gpu_subtree_from_d0_plan_with_context(
            subtree_backend,
            dataset,
            params.metric(),
            params,
            d0_plan,
            compact_sink,
            gpu_subtree_compact_flush_mode(queue_budget_bytes, pool_threads),
            qcode_runtime_context,
        );
    }
    if params.gpu_subtree_sq_staging_mode.is_enabled() {
        let d0_plan = run_gpu_subtree_sq_staging_preflight(
            subtree_backend,
            dataset,
            all_indices,
            params.metric(),
            params,
            pool_threads as u32,
            &mut rng,
            artifact_dir,
        )?;
        if !params.root_stop_after_d00 && !d0_plan.child_runs.is_empty() {
            compact_sink.prepare_for_d1()?;
        }
        let sq_staging_sidecar = if matches!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
                | PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
        ) || params.gpu_subtree_sq8_rotated_residual_d1
            || params.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
        {
            None
        } else {
            let sq_staging_memory_plan =
                OomMemoryPlan::from_params_for_dataset(params, all_indices.len(), dataset.dim());
            let Some(sidecar) = maybe_open_sq_wmma_runtime_sidecar(
                params,
                dataset,
                all_indices.len(),
                &sq_staging_memory_plan,
            )?
            else {
                return Err(AnnError::log_index_error(
                    "SQ GPU-D0 staging wrote a sidecar but could not reopen it for D1+".to_string(),
                ));
            };
            Some(sidecar)
        };
        let rotated_sidecar =
            open_rotated_residual_sq8_runtime_sidecar(params, all_indices.len(), dataset.dim())?;
        let sq_staging_runtime_context = GpuSubtreeRuntimeContext {
            sq_wmma_sidecar: sq_staging_sidecar.as_ref(),
            rotated_residual_sq8_sidecar: rotated_sidecar.as_ref(),
            ..gpu_subtree_runtime_context
        };
        let queue_budget_bytes = params
            .effective_oom_memory_budget_bytes()
            .saturating_mul(5)
            .saturating_div(100)
            .max(64 * 1024 * 1024);
        return run_gpu_subtree_from_d0_plan_with_context(
            subtree_backend,
            dataset,
            params.metric(),
            params,
            d0_plan,
            compact_sink,
            gpu_subtree_compact_flush_mode(queue_budget_bytes, pool_threads),
            sq_staging_runtime_context,
        );
    }
    let d0_plan = run_gpu_subtree_d0_preflight_with_context(
        root_assignment_backend,
        dataset,
        all_indices,
        params.metric(),
        params,
        pool_threads as u32,
        &mut rng,
        artifact_dir,
        gpu_subtree_runtime_context,
    )?;
    if !params.root_stop_after_d00 && !d0_plan.child_runs.is_empty() {
        compact_sink.prepare_for_d1()?;
    }
    let queue_budget_bytes = params
        .effective_oom_memory_budget_bytes()
        .saturating_mul(5)
        .saturating_div(100)
        .max(64 * 1024 * 1024);
    run_gpu_subtree_from_d0_plan_with_context(
        subtree_backend,
        dataset,
        params.metric(),
        params,
        d0_plan,
        compact_sink,
        gpu_subtree_compact_flush_mode(queue_budget_bytes, pool_threads),
        gpu_subtree_runtime_context,
    )
}

fn open_rotated_residual_sq8_runtime_sidecar(
    params: &PiPNNParams,
    rows: usize,
    dim: usize,
) -> AnnResult<Option<Sq8RotatedResidualSidecar>> {
    if !params.gpu_subtree_sq8_rotated_residual_d1 {
        return Ok(None);
    }
    let prefix = params
        .gpu_subtree_sq_wmma_sidecar_prefix
        .as_ref()
        .ok_or_else(|| {
            AnnError::log_index_error("Rotated residual SQ8 sidecar prefix is missing".to_string())
        })?;
    Ok(Some(Sq8RotatedResidualSidecar::open(
        prefix,
        rows,
        dim,
        params.gpu_subtree_sq8_rotated_residual_seed,
    )?))
}

fn emit_streaming_partition_compact_topk_sq_staging_fused_sketch(
    subtree_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    dataset: &dyn PointStore,
    all_indices: &[u32],
    params: &PiPNNParams,
    pool_threads: usize,
    artifact_dir: &Path,
    sketch_path: &Path,
    edge_sink: &dyn PendingEdgeSink,
    gpu_subtree_runtime_context: GpuSubtreeRuntimeContext<'_>,
    d0_read_progress: Option<Arc<D0ReadProgress>>,
) -> AnnResult<(
    GpuSubtreeStreamingProfile,
    DiskSketchStore,
    Duration,
    Duration,
)> {
    let mut rng = StdRng::seed_from_u64(params.random_seed);
    let preflight = if params.gpu_subtree_sq4_plus
        && !params.gpu_subtree_sq_staging_mode.is_enabled()
    {
        rbc_partition::run_gpu_subtree_sq4_d0_preflight_with_sketch(
            subtree_backend, dataset, all_indices, params.metric(), params,
            pool_threads as u32, &mut rng, Some(artifact_dir), sketch_path,
            gpu_subtree_runtime_context,
            match std::env::var("SHEAR_PIPNN_SQ4_D0_SKETCH_BACKEND").as_deref() {
                Ok("cpu") | Err(std::env::VarError::NotPresent) => false,
                Ok("gpu") => true,
                other => return Err(AnnError::log_index_error(format!(
                    "Invalid SQ4+ D0 sketch backend {other:?}; expected cpu or explicit gpu experiment"))),
            },
        )?
    } else {
        run_gpu_subtree_sq_staging_preflight_with_sketch(
            subtree_backend,
            dataset,
            all_indices,
            params.metric(),
            params,
            pool_threads as u32,
            &mut rng,
            Some(artifact_dir),
            sketch_path,
            d0_read_progress,
        )?
    };
    let fused_sketch_compute_wall = preflight.fused_sketch_compute_wall;
    let fused_sketch_join_wait_wall = preflight.fused_sketch_join_wait_wall;
    let fused_sketches = preflight.fused_sketches.ok_or_else(|| {
        AnnError::log_index_error(
            "D0 fused sketch preflight returned no sketches".to_string(),
        )
    })?;
    if !params.root_stop_after_d00 && !preflight.d0_plan.child_runs.is_empty() {
        edge_sink.prepare_for_d1()?;
    }
    let sq_staging_sidecar = if matches!(
        params.gpu_subtree_sq_staging_mode,
        PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
            | PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
    ) || params.gpu_subtree_sq8_rotated_residual_d1
        || params.gpu_subtree_sq4_plus
        || params.gpu_subtree_sq_wmma_mode != PiPNNGpuSubtreeSqWmmaMode::Build
    {
        None
    } else {
        let sq_staging_memory_plan =
            OomMemoryPlan::from_params_for_dataset(params, all_indices.len(), dataset.dim());
        let Some(sidecar) = maybe_open_sq_wmma_runtime_sidecar(
            params,
            dataset,
            all_indices.len(),
            &sq_staging_memory_plan,
        )?
        else {
            return Err(AnnError::log_index_error(
                "SQ GPU-D0 staging wrote a sidecar but could not reopen it for D1+".to_string(),
            ));
        };
        Some(sidecar)
    };
    let rotated_sidecar =
        open_rotated_residual_sq8_runtime_sidecar(params, all_indices.len(), dataset.dim())?;
    let sq_staging_runtime_context = GpuSubtreeRuntimeContext {
        sq_wmma_sidecar: sq_staging_sidecar.as_ref(),
        rotated_residual_sq8_sidecar: rotated_sidecar.as_ref(),
        ..gpu_subtree_runtime_context
    };
    let queue_budget_bytes = params
        .effective_oom_memory_budget_bytes()
        .saturating_mul(5)
        .saturating_div(100)
        .max(64 * 1024 * 1024);
    let compact_profile = {
        let compact_sink = OomCompactTopKSink::new(&fused_sketches, params, edge_sink);
        run_gpu_subtree_from_d0_plan_with_context(
            subtree_backend,
            dataset,
            params.metric(),
            params,
            preflight.d0_plan,
            &compact_sink,
            gpu_subtree_compact_flush_mode(queue_budget_bytes, pool_threads),
            sq_staging_runtime_context,
        )?
    };
    Ok((
        compact_profile,
        fused_sketches,
        fused_sketch_compute_wall,
        fused_sketch_join_wait_wall,
    ))
}

fn run_streaming_partition_pipeline_with_sq_staging_fused_sketch(
    dataset: &dyn PointStore,
    num_points: usize,
    pool_threads: usize,
    params: &PiPNNParams,
    artifact_dir: &Path,
    sketch_path: &Path,
    detailed_profiling: bool,
    leaf_compute_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    edge_sink: &dyn PendingEdgeSink,
    gpu_subtree_runtime_context: GpuSubtreeRuntimeContext<'_>,
    d0_read_progress: Option<Arc<D0ReadProgress>>,
) -> AnnResult<(StreamingProfile, DiskSketchStore, Duration, Duration)> {
    let t_rbc_leaf = Instant::now();
    let all_indices: Vec<u32> = (0..num_points as u32).collect();
    if !matches!(params.partitioner, PiPNNPartitionerKind::GpuSubtree) {
        return Err(AnnError::log_index_error(
            "SQ staging fused sketch path requires gpu-subtree partitioner".to_string(),
        ));
    }
    if !matches!(leaf_compute_backend.kind(), PiPNNComputeBackendKind::Cuda) {
        return Err(AnnError::log_index_error(
            "gpu-subtree requires CUDA compute backend".to_string(),
        ));
    }
    if detailed_profiling {
        info!("PiPNN SQ staging fused sketch path enabled");
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    pb.set_message("PiPNN gpu-subtree SQ staging fused sketch processing...");

    BlasThreadEnv::set_single_threaded();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .stack_size(PIPNN_RAYON_STACK_BYTES)
        .build()
        .map_err(|e| AnnError::log_index_error(format!("Failed to create Rayon pool: {}", e)))?;

    let producer_start = Instant::now();
    let (compact_profile, fused_sketches, fused_sketch_compute_wall, fused_sketch_join_wait_wall) =
        pool.install(|| {
            emit_streaming_partition_compact_topk_sq_staging_fused_sketch(
                leaf_compute_backend,
                dataset,
                &all_indices,
                params,
                pool_threads,
                artifact_dir,
                sketch_path,
                edge_sink,
                gpu_subtree_runtime_context,
                d0_read_progress,
            )
        })?;
    let producer_wall = producer_start.elapsed();
    let entry_point = compact_profile.entry_point;
    pb.finish_with_message("PiPNN gpu-subtree SQ staging fused sketch compact top-k done");

    Ok((
        StreamingProfile {
            producer_wall,
            total_wall: t_rbc_leaf.elapsed(),
            leaf: compact_profile.leaf,
            telemetry: SchedulerTelemetry::default(),
            rbc: compact_profile.rbc,
            kmeans: KMeansPartitionTelemetry::default(),
            entry_point,
        },
        fused_sketches,
        fused_sketch_compute_wall,
        fused_sketch_join_wait_wall,
    ))
}

fn parse_gpu_subtree_async_flush_worker_policy(
    configured: Option<&std::ffi::OsStr>,
    host_parallelism: usize,
) -> AsyncCompactFlushWorkerPolicy {
    let max_workers = host_parallelism.max(1);
    let adaptive = |invalid_override| AsyncCompactFlushWorkerPolicy::Adaptive {
        initial_workers: GPU_SUBTREE_DEFAULT_ASYNC_FLUSH_WORKERS.min(max_workers),
        max_workers,
        invalid_override,
    };

    match configured.and_then(std::ffi::OsStr::to_str) {
        None if configured.is_none() => adaptive(false),
        Some(value) => match value.parse::<usize>() {
            Ok(workers) if workers > 0 => AsyncCompactFlushWorkerPolicy::Fixed { workers },
            _ => adaptive(true),
        },
        None => adaptive(true),
    }
}

fn gpu_subtree_compact_flush_mode(
    queue_budget_bytes: usize,
    pool_threads: usize,
) -> CompactFlushMode {
    if std::env::var("SHEAR_PIPNN_COMPACT_FLUSH_MODE")
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("sync"))
    {
        return CompactFlushMode::Sync;
    }

    let host_parallelism = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or_else(|_| pool_threads.max(1));
    let configured = std::env::var_os("SHEAR_PIPNN_ASYNC_FLUSH_WORKERS");
    CompactFlushMode::AsyncOrdered {
        queue_budget_bytes,
        worker_policy: parse_gpu_subtree_async_flush_worker_policy(
            configured.as_deref(),
            host_parallelism,
        ),
    }
}

fn emit_streaming_partition_leaves(
    assignment_backend: &dyn self::compute_backend::PiPNNComputeBackend,
    dataset: &dyn PointStore,
    all_indices: &[u32],
    params: &PiPNNParams,
    pool_threads: usize,
    artifact_dir: Option<&Path>,
    emitter: &dyn self::scheduler::LeafEmitter,
    rbc_telemetry: &mut RbcPartitionTelemetry,
    kmeans_telemetry: &mut KMeansPartitionTelemetry,
) -> AnnResult<()> {
    match params.partitioner {
        PiPNNPartitionerKind::Rbc => {
            let mut rng = StdRng::seed_from_u64(params.random_seed);
            *rbc_telemetry = rbc_partition_streaming_with_backend(
                assignment_backend,
                dataset,
                all_indices,
                params.metric(),
                params,
                pool_threads as u32,
                &mut rng,
                artifact_dir,
                emitter,
            )?;
        }
        PiPNNPartitionerKind::KMeans => {
            *kmeans_telemetry =
                kmeans_partition_streaming(dataset, all_indices, params.metric(), params, emitter)?;
        }
        PiPNNPartitionerKind::GpuSubtree => {
            let _ = emitter;
            return Err(AnnError::log_index_error(
                "gpu-subtree must use persistent compact top-k output".to_string(),
            ));
        }
    }
    Ok(())
}

fn prepare_oom_artifact_dir(params: &PiPNNParams) -> AnnResult<PathBuf> {
    let base_dir = if params.oom_temp_dir.as_os_str().is_empty() {
        std::env::temp_dir().join("shear_core-pipnn-oom")
    } else {
        params.oom_temp_dir.clone()
    };
    fs::create_dir_all(&base_dir)?;
    Ok(base_dir)
}

fn cleanup_oom_artifacts(dir: &Path) -> AnnResult<()> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct OomProfile {
    num_points: usize,
    dim: usize,
    leaf_knn: usize,
    partitioner: PiPNNPartitionerKind,
    root_assignment_backend: PiPNNComputeBackendKind,
    assignment_backend: PiPNNGpuSubtreeAssignmentBackend,
    d0_dco_mode: PiPNNGpuSubtreeDcoMode,
    d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    d0_dco_survivors: usize,
    d0_dco_audit: bool,
    d0_sq_cpu_kernel: SqInt8CpuKernel,
    d1plus_lazy_split_mode: PiPNNGpuSubtreeD1PlusLazySplitMode,
    assignment_dco_mode: PiPNNGpuSubtreeDcoMode,
    assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    assignment_dco_proj_dim: usize,
    assignment_dco_survivors: usize,
    assignment_dco_audit: bool,
    leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend,
    leaf_dco_mode: PiPNNGpuSubtreeDcoMode,
    leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    leaf_dco_proj_dim: usize,
    leaf_dco_survivors: usize,
    leaf_dco_audit: bool,
    leaf_dco_ddc_res_pca_matrix: Option<PathBuf>,
    leaf_dco_ddc_res_projected_input: bool,
    leaf_dco_ddc_res_base_dim: usize,
    leaf_dco_ddc_res_sigma_count: f32,
    rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode,
    rabitq_plus_bits: usize,
    rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    rabitq_plus_seed: u64,
    rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode,
    rabitq_plus_sidecar_prefix: Option<PathBuf>,
    rabitq_plus_fused_sidecar_sketch_enabled: bool,
    rabitq_plus_fused_prestage_total_wall: Duration,
    rabitq_plus_fused_chunk_alloc_wall: Duration,
    rabitq_plus_fused_sidecar_raw_read_wall: Duration,
    rabitq_plus_fused_source_hash_wall: Duration,
    rabitq_plus_fused_sidecar_quantize_wall: Duration,
    rabitq_plus_fused_sidecar_code_output_wall: Duration,
    rabitq_plus_fused_sidecar_norm_wall: Duration,
    rabitq_plus_fused_sidecar_meta_output_wall: Duration,
    rabitq_plus_fused_sidecar_header_write_wall: Duration,
    rabitq_plus_fused_sidecar_write_wall: Duration,
    rabitq_plus_fused_sketch_alloc_wall: Duration,
    rabitq_plus_fused_sketch_compute_wall: Duration,
    rabitq_plus_fused_sketch_extend_wall: Duration,
    rabitq_plus_fused_sketch_write_wall: Duration,
    qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode,
    qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode,
    qcode_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode,
    qcode_rabitq_short_survivors: Option<usize>,
    rabitq_residual_centroids: usize,
    qcode_staging_upload_shard_rows: usize,
    qcode_global_d0_leaders: usize,
    sq_staging_mode: PiPNNGpuSubtreeSqStagingMode,
    sq_staging_upload_shard_rows: usize,
    sq_bits: usize,
    sq_native_int4: bool,
    sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode,
    persistent_large_ticket_batch_requested: PiPNNGpuSubtreePersistentLargeTicketBatch,
    persistent_sq8_assignment_kernel_requested: PiPNNGpuSubtreeSq8AssignmentKernel,
    persistent_sq8_leaf_executor_requested: PiPNNGpuSubtreeSq8LeafExecutor,
    persistent_sq8_leaf_mode_requested: PiPNNGpuSubtreeSq8LeafMode,
    persistent_scheduler_mode_requested: PiPNNGpuSubtreePersistentSchedulerMode,
    persistent_scheduler_mode_actual: PiPNNGpuSubtreePersistentSchedulerMode,
    persistent_idle_backoff_ns_requested: u32,
    persistent_idle_backoff_ns_actual: u32,
    sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode,
    sq_wmma_sidecar_prefix: Option<PathBuf>,
    sq_staging_fused_sketch_enabled: bool,
    sq_staging_fused_sketch_compute_wall: Duration,
    sq_staging_fused_sketch_join_wait_wall: Duration,
    turboquant_mode: PiPNNGpuSubtreeTurboQuantMode,
    turboquant_bits: usize,
    turboquant_seed: u64,
    turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode,
    turboquant_sidecar_prefix: Option<PathBuf>,
    turboquant_cascade_low_dim: usize,
    turboquant_cascade_survivor_cap: usize,
    turboquant_adaptive_survivor_threshold: usize,
    turboquant_adaptive_survivor_low_cap: usize,
    memory_budget_bytes: usize,
    memory_budget_source: &'static str,
    oom_io_mode: PiPNNOomIoMode,
    oom_reservoir_budget_share: f64,
    oom_sidecar_resident_budget_cap_bytes: Option<usize>,
    effective_threads: usize,
    pool_threads: usize,
    leaf_budget_threads: usize,
    resident_reservoir_points: usize,
    resident_reservoir_bytes: usize,
    reservoir_allocation_wall: Duration,
    reservoir_initialization: ReservoirInitializationProfile,
    reduce_shard_points: usize,
    compact_spill_digest_enabled: bool,
    compact_spill_digest_width: usize,
    prune_backend: PiPNNPruneBackend,
    hashprune_evidence_policy: HashPruneEvidencePolicy,
    hashprune_backfill_policy: HashPruneBackfillPolicy,
    hashprune_backfill_degree_cap: usize,
    lrpf_target_degree: usize,
    lrpf_long_edge_quota: usize,
    lrpf_pre_spill_compress_enabled: bool,
    lrpf_pre_spill_width: usize,
    lrpf_candidate_records: usize,
    lrpf_witness_records: usize,
    lrpf_fragments: usize,
    lrpf_pre_spill_pruned_candidates: usize,
    lrpf_reducer_pruned_candidates: usize,
    reservoir_budget_bytes: usize,
    resident_subtree_budget_bytes: usize,
    required_resident_bytes: usize,
    optional_resident_budget_bytes: usize,
    optional_resident_admitted_bytes: usize,
    optional_resident_unassigned_bytes: usize,
    transient_pipeline_budget_bytes: usize,
    d1plus_working_set_budget_bytes: usize,
    sidecar_resident_kind: OomSidecarResidentKind,
    sidecar_resident_budget_bytes: usize,
    sidecar_resident_required_bytes: usize,
    sidecar_resident_admitted_rows: usize,
    sidecar_resident_admissible: bool,
    scratch_budget_bytes: usize,
    io_headroom_bytes: usize,
    raw_vector_cache_budget_bytes: usize,
    raw_vector_cache_rows: usize,
    online_suffix_mode: PiPNNOnlineSuffixReduceMode,
    online_suffix_fallback_reason: Option<&'static str>,
    online_suffix_config: OnlineSuffixProfileConfig,
    online_suffix: OnlineSuffixReduceStats,
    sketch_bytes: usize,
    sketch_file_bytes: usize,
    spill: SpillRunStats,
    reduce: SpillReduceStats,
    sketch_persist_wall: Duration,
    streaming_wall: Duration,
    reduce_wall: Duration,
    emit_wall: Duration,
    resident_emit: ResidentEmitProfile,
    total_wall: Duration,
    scheduler: SchedulerTelemetry,
    rbc: RbcPartitionTelemetry,
    kmeans: KMeansPartitionTelemetry,
    hashprune_evidence: HashPruneEvidenceProfile,
    hashprune_backfill: HashPruneBackfillProfile,
    leaf: LeafProfile,
    memory: OomMemoryReport,
    temp_files: TempFileReport,
    uring_batches: u64,
    uring_windows: u64,
    uring_bytes: u64,
    uring_fallbacks: u64,
}

#[derive(Clone, Debug, Default)]
struct OnlineSuffixProfileConfig {
    active: bool,
    configured_direct_queue_bytes: usize,
    configured_state_cache_bytes: usize,
    configured_state_page_cache_bytes: usize,
    configured_state_writeback_bytes: usize,
    configured_spill_buffer_bytes: usize,
    configured_reorder_window_bytes: usize,
    configured_read_window_bytes: usize,
    configured_delta_budget_bytes: usize,
    configured_delta_buffer_bytes: usize,
    configured_range_points: usize,
    configured_range_trigger_bytes: usize,
    configured_max_runs_per_range: usize,
    configured_compaction_workers: usize,
    configured_cache_hot_updates: usize,
    effective_direct_queue_bytes: usize,
    effective_state_cache_bytes: usize,
    effective_state_page_cache_bytes: usize,
    effective_state_writeback_bytes: usize,
    effective_spill_buffer_bytes: usize,
    effective_reorder_window_bytes: usize,
    effective_decode_bytes: usize,
    effective_read_window_bytes: usize,
    effective_delta_budget_bytes: usize,
    effective_delta_buffer_bytes: usize,
    effective_compaction_bytes: usize,
    effective_shard_points: usize,
    effective_range_points: usize,
    effective_range_trigger_bytes: usize,
    effective_max_runs_per_range: usize,
    effective_compaction_workers: usize,
    effective_drain_compaction_workers: usize,
    effective_cache_hot_updates: usize,
    reserved_metadata_bytes: usize,
    reserved_total_bytes: usize,
    configured_reduce_workers: usize,
    effective_reduce_workers: usize,
    state_overprovision_ratio: f64,
}

#[derive(Clone, Debug, Default)]
struct OomMemoryReport {
    rss_kb_samples: Vec<usize>,
    rss_hwm_kb: Option<usize>,
    vm_peak_kb: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct FileCategoryStats {
    files: usize,
    bytes: u64,
}

#[derive(Clone, Debug, Default)]
struct TempFileReport {
    total: FileCategoryStats,
    candidate: FileCategoryStats,
    partition_depth0: FileCategoryStats,
    partition_depth1: FileCategoryStats,
    partition_depth2: FileCategoryStats,
    partition_depth3: FileCategoryStats,
    other: FileCategoryStats,
}

#[derive(Clone, Copy, Debug)]
struct OomSerialTimingBreakdown {
    sketch_persist: Duration,
    d0_root_assignment: Duration,
    d0_root_assignment_total: Duration,
    d0_kmeans_train: Duration,
    d0_kmeans_assign: Duration,
    d0_sq_raw_read: Duration,
    d0_sq_raw_h2d: Duration,
    d0_sq_quantize: Duration,
    d0_sq_d0_assign: Duration,
    d0_sq_qcode_write: Duration,
    d0_residual: Duration,
    d0_nested_requested_total: Duration,
    d0_sq_qcode_write_total: Duration,
    d0_nested_overlap_hidden: Duration,
    d1_materialize: Duration,
    d2_replay: Duration,
    persistent_session_init: Duration,
    d1plus_sidecar_wait: Duration,
    d1plus_sidecar_gather_total: Duration,
    d1plus_payload_h2d: Duration,
    d1plus_persistent_kernel: Duration,
    d1plus_cpu_result_scatter: Duration,
    d1plus_backend_residual: Duration,
    d1plus_child_replay: Duration,
    d1plus_lazy_split: Duration,
    d1plus_push_wait: Duration,
    d1plus_validate: Duration,
    d1plus_merge_profile: Duration,
    d1plus_push_batch_residual: Duration,
    d1plus_push_batch_total: Duration,
    persistent_session_drain: Duration,
    d1plus_producer_residual: Duration,
    streaming_residual: Duration,
    external_reduce: Duration,
    emit: Duration,
    stage_outer_residual: Duration,
    accounted_sum: Duration,
    stage_total: Duration,
}

impl OomSerialTimingBreakdown {
    fn from_profile(profile: &OomProfile) -> Self {
        let d1plus_sidecar_gather_total = profile.rbc.gpu_subtree_rabitq_plus_sidecar_gather_wall
            + profile.rbc.gpu_subtree_sq_wmma_sidecar_gather_wall
            + profile.rbc.gpu_subtree_turboquant_sidecar_gather_wall;
        let d1plus_sidecar_wait = if profile.rbc.gpu_subtree_gpu_wait_for_sidecar_wall
            > Duration::ZERO
            || profile.rbc.gpu_subtree_sidecar_gather_hidden_wall > Duration::ZERO
        {
            profile.rbc.gpu_subtree_gpu_wait_for_sidecar_wall
        } else {
            d1plus_sidecar_gather_total
        };
        let d0_root_assignment = if profile.rbc.gpu_subtree_d0_wall > Duration::ZERO {
            profile.rbc.gpu_subtree_d0_wall
        } else {
            profile.rbc.d0_ads_assignment_wall
        };
        let d0_sq_qcode_write_total = oom_profile_duration_sum(&[
            profile.rbc.gpu_subtree_sq_staging_qcode_d2h_wall,
            profile
                .rbc
                .gpu_subtree_sq_staging_assignment_fragment_d2h_wall,
            profile
                .rbc
                .gpu_subtree_sq_staging_assignment_fragment_reduce_wall,
            profile.rbc.gpu_subtree_sq_staging_sidecar_write_wall,
        ]);
        let d0_has_split_details = profile.rbc.gpu_subtree_d0_wall > Duration::ZERO
            && oom_profile_duration_sum(&[
                profile.rbc.gpu_subtree_d0_tagore_train_wall,
                profile.rbc.gpu_subtree_d0_tagore_assign_wall,
                profile.rbc.gpu_subtree_sq_staging_raw_read_wall,
                profile.rbc.gpu_subtree_sq_staging_raw_h2d_wall,
                profile.rbc.gpu_subtree_sq_staging_gpu_quantize_wall,
                profile
                    .rbc
                    .gpu_subtree_sq_staging_host_rotated_quantize_wall,
                profile.rbc.gpu_subtree_sq_staging_gpu_d0_assign_wall,
                d0_sq_qcode_write_total,
            ]) > Duration::ZERO;
        let d0_kmeans_assign_requested =
            if profile.rbc.gpu_subtree_sq_staging_gpu_d0_assign_wall > Duration::ZERO {
                Duration::ZERO
            } else {
                profile.rbc.gpu_subtree_d0_tagore_assign_wall
            };
        let d0_nested_requested_total = oom_profile_duration_sum(&[
            profile.rbc.gpu_subtree_d0_tagore_train_wall,
            d0_kmeans_assign_requested,
            profile.rbc.gpu_subtree_sq_staging_raw_read_wall,
            profile.rbc.gpu_subtree_sq_staging_raw_h2d_wall,
            profile.rbc.gpu_subtree_sq_staging_gpu_quantize_wall,
            profile
                .rbc
                .gpu_subtree_sq_staging_host_rotated_quantize_wall,
            profile.rbc.gpu_subtree_sq_staging_gpu_d0_assign_wall,
            d0_sq_qcode_write_total,
        ]);
        let (
            d0_root_assignment_row,
            d0_kmeans_train,
            d0_kmeans_assign,
            d0_sq_raw_read,
            d0_sq_raw_h2d,
            d0_sq_quantize,
            d0_sq_d0_assign,
            d0_sq_qcode_write,
            d0_residual,
            d0_nested_overlap_hidden,
        ) = if d0_has_split_details {
            let mut d0_remaining = d0_root_assignment;
            let d0_kmeans_train = oom_profile_take_serial_part(
                &mut d0_remaining,
                profile.rbc.gpu_subtree_d0_tagore_train_wall,
            );
            let d0_kmeans_assign =
                oom_profile_take_serial_part(&mut d0_remaining, d0_kmeans_assign_requested);
            let d0_sq_raw_read = oom_profile_take_serial_part(
                &mut d0_remaining,
                profile.rbc.gpu_subtree_sq_staging_raw_read_wall,
            );
            let d0_sq_raw_h2d = oom_profile_take_serial_part(
                &mut d0_remaining,
                profile.rbc.gpu_subtree_sq_staging_raw_h2d_wall,
            );
            let d0_sq_quantize = oom_profile_take_serial_part(
                &mut d0_remaining,
                profile.rbc.gpu_subtree_sq_staging_gpu_quantize_wall
                    + profile
                        .rbc
                        .gpu_subtree_sq_staging_host_rotated_quantize_wall,
            );
            let d0_sq_d0_assign = oom_profile_take_serial_part(
                &mut d0_remaining,
                profile.rbc.gpu_subtree_sq_staging_gpu_d0_assign_wall,
            );
            let d0_sq_qcode_write =
                oom_profile_take_serial_part(&mut d0_remaining, d0_sq_qcode_write_total);
            (
                Duration::ZERO,
                d0_kmeans_train,
                d0_kmeans_assign,
                d0_sq_raw_read,
                d0_sq_raw_h2d,
                d0_sq_quantize,
                d0_sq_d0_assign,
                d0_sq_qcode_write,
                d0_remaining,
                d0_nested_requested_total.saturating_sub(d0_root_assignment),
            )
        } else {
            (
                d0_root_assignment,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            )
        };
        let d1plus_payload_h2d_total = profile.rbc.gpu_subtree_vector_h2d_wall
            + profile
                .rbc
                .gpu_subtree_rabitq_residual_leader_payload_h2d_wall
            + profile
                .rbc
                .gpu_subtree_rabitq_residual_global_payload_upload_wall
            + profile
                .rbc
                .gpu_subtree_rabitq_residual_d1plus_child_id_h2d_wall
            + profile.rbc.gpu_subtree_rabitq_plus_h2d_wall
            + profile.rbc.gpu_subtree_sq_wmma_h2d_wall
            + profile.rbc.gpu_subtree_turboquant_h2d_wall;
        let d2_replay = profile
            .rbc
            .d2_replay_wall
            .saturating_sub(profile.rbc.gpu_subtree_d1plus_child_replay_wall);
        let persistent_kernel_requested = if profile.rbc.gpu_subtree_persistent_async_d2h_enabled
            && profile.rbc.gpu_subtree_persistent_kernel_event_wall > Duration::ZERO
        {
            profile.rbc.gpu_subtree_persistent_kernel_event_wall
        } else {
            profile.rbc.gpu_subtree_persistent_total_wall
        };
        let mut d1plus_producer_remaining = profile.rbc.gpu_subtree_d1plus_producer_wall;
        let persistent_session_init = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_persistent_session_init_wall,
        );
        let persistent_session_drain = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_persistent_d2h_final_drain_wall,
        );
        let d1plus_sidecar_wait =
            oom_profile_take_serial_part(&mut d1plus_producer_remaining, d1plus_sidecar_wait);
        let d1plus_backend_budget = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_d1plus_backend_wall,
        );
        let mut d1plus_backend_remaining = d1plus_backend_budget;
        let d1plus_payload_h2d =
            oom_profile_take_serial_part(&mut d1plus_backend_remaining, d1plus_payload_h2d_total);
        let d1plus_persistent_kernel = oom_profile_take_serial_part(
            &mut d1plus_backend_remaining,
            persistent_kernel_requested,
        );
        let synchronous_scatter = if profile.rbc.gpu_subtree_async_result_scatter_enabled {
            Duration::ZERO
        } else {
            oom_profile_take_serial_part(
                &mut d1plus_backend_remaining,
                profile.rbc.gpu_subtree_cpu_result_scatter_wall,
            )
        };
        let d1plus_backend_residual = d1plus_backend_remaining;
        let d1plus_cpu_result_scatter = if profile.rbc.gpu_subtree_async_result_scatter_enabled {
            oom_profile_take_serial_part(
                &mut d1plus_producer_remaining,
                profile.rbc.gpu_subtree_async_result_scatter_wait_wall,
            )
        } else {
            synchronous_scatter
        };
        let d1plus_child_replay = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_d1plus_child_replay_wall,
        );
        let d1plus_lazy_split = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_d1plus_lazy_split_wall,
        );
        let d1plus_push_wait = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_d1plus_push_wait_wall,
        );
        let d1plus_validate = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_d1plus_validate_wall,
        );
        let d1plus_merge_profile = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            profile.rbc.gpu_subtree_d1plus_merge_profile_wall,
        );
        let d1plus_push_batch_requested = oom_profile_residual_duration(
            profile.rbc.gpu_subtree_d1plus_push_batch_wall,
            &[profile.rbc.gpu_subtree_d1plus_push_wait_wall],
        );
        let d1plus_push_batch_residual = oom_profile_take_serial_part(
            &mut d1plus_producer_remaining,
            d1plus_push_batch_requested,
        );
        let d1plus_producer_residual = d1plus_producer_remaining;
        let streaming_residual = oom_profile_residual_duration(
            profile.streaming_wall,
            &[
                d0_root_assignment,
                profile.rbc.d1_materialize_wall,
                d2_replay,
                profile.rbc.gpu_subtree_d1plus_producer_wall,
            ],
        );
        let stage_outer_residual = oom_profile_residual_duration(
            profile.total_wall,
            &[
                profile.sketch_persist_wall,
                profile.streaming_wall,
                profile.reduce_wall,
                profile.emit_wall,
            ],
        );
        let rows = [
            profile.sketch_persist_wall,
            d0_root_assignment_row,
            d0_kmeans_train,
            d0_kmeans_assign,
            d0_sq_raw_read,
            d0_sq_raw_h2d,
            d0_sq_quantize,
            d0_sq_d0_assign,
            d0_sq_qcode_write,
            d0_residual,
            profile.rbc.d1_materialize_wall,
            d2_replay,
            persistent_session_init,
            d1plus_sidecar_wait,
            d1plus_payload_h2d,
            d1plus_persistent_kernel,
            d1plus_cpu_result_scatter,
            d1plus_backend_residual,
            d1plus_child_replay,
            d1plus_lazy_split,
            d1plus_push_wait,
            d1plus_validate,
            d1plus_merge_profile,
            d1plus_push_batch_residual,
            persistent_session_drain,
            d1plus_producer_residual,
            streaming_residual,
            profile.reduce_wall,
            profile.emit_wall,
            stage_outer_residual,
        ];
        Self {
            sketch_persist: profile.sketch_persist_wall,
            d0_root_assignment: d0_root_assignment_row,
            d0_root_assignment_total: d0_root_assignment,
            d0_kmeans_train,
            d0_kmeans_assign,
            d0_sq_raw_read,
            d0_sq_raw_h2d,
            d0_sq_quantize,
            d0_sq_d0_assign,
            d0_sq_qcode_write,
            d0_residual,
            d0_nested_requested_total,
            d0_sq_qcode_write_total,
            d0_nested_overlap_hidden,
            d1_materialize: profile.rbc.d1_materialize_wall,
            d2_replay,
            persistent_session_init,
            d1plus_sidecar_wait,
            d1plus_sidecar_gather_total,
            d1plus_payload_h2d,
            d1plus_persistent_kernel,
            d1plus_cpu_result_scatter,
            d1plus_backend_residual,
            d1plus_child_replay,
            d1plus_lazy_split,
            d1plus_push_wait,
            d1plus_validate,
            d1plus_merge_profile,
            d1plus_push_batch_residual,
            d1plus_push_batch_total: profile.rbc.gpu_subtree_d1plus_push_batch_wall,
            persistent_session_drain,
            d1plus_producer_residual,
            streaming_residual,
            external_reduce: profile.reduce_wall,
            emit: profile.emit_wall,
            stage_outer_residual,
            accounted_sum: oom_profile_duration_sum(&rows),
            stage_total: profile.total_wall,
        }
    }
}

fn oom_profile_duration_sum(parts: &[Duration]) -> Duration {
    parts
        .iter()
        .copied()
        .fold(Duration::ZERO, |sum, part| sum + part)
}

fn oom_profile_residual_duration(total: Duration, parts: &[Duration]) -> Duration {
    parts
        .iter()
        .fold(total, |remaining, part| remaining.saturating_sub(*part))
}

fn oom_profile_take_serial_part(remaining: &mut Duration, requested: Duration) -> Duration {
    let taken = requested.min(*remaining);
    *remaining = remaining.saturating_sub(taken);
    taken
}

fn write_oom_profile(path: &Path, profile: &OomProfile) -> AnnResult<()> {
    let resident_subtree_avg_points = if profile.rbc.resident_subtree_count == 0 {
        0.0
    } else {
        profile.rbc.resident_subtree_points as f64 / profile.rbc.resident_subtree_count as f64
    };
    let serial_timing = OomSerialTimingBreakdown::from_profile(profile);
    let mut raw_vector_cache_stats = profile.leaf.io_stats.clone();
    raw_vector_cache_stats.merge(&profile.rbc.gpu_subtree_host_io_stats);
    let mut file = fs::File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"num_points\": {},", profile.num_points)?;
    writeln!(file, "  \"dim\": {},", profile.dim)?;
    writeln!(file, "  \"leaf_knn\": {},", profile.leaf_knn)?;
    writeln!(
        file,
        "  \"memory_budget_bytes\": {},",
        profile.memory_budget_bytes
    )?;
    writeln!(
        file,
        "  \"memory_budget_source\": \"{}\",",
        profile.memory_budget_source
    )?;
    writeln!(
        file,
        "  \"oom_io_mode\": \"{}\",",
        profile.oom_io_mode.as_str()
    )?;
    writeln!(
        file,
        "  \"oom_reservoir_budget_share\": {:.6},",
        profile.oom_reservoir_budget_share
    )?;
    let sidecar_cap = profile
        .oom_sidecar_resident_budget_cap_bytes
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "null".to_string());
    writeln!(
        file,
        "  \"oom_sidecar_resident_budget_cap_bytes\": {},",
        sidecar_cap
    )?;
    writeln!(
        file,
        "  \"effective_threads\": {},",
        profile.effective_threads
    )?;
    writeln!(file, "  \"pool_threads\": {},", profile.pool_threads)?;
    writeln!(
        file,
        "  \"leaf_budget_threads\": {},",
        profile.leaf_budget_threads
    )?;
    writeln!(
        file,
        "  \"reduce_shard_points\": {},",
        profile.reduce_shard_points
    )?;
    writeln!(
        file,
        "  \"compact_spill_digest\": {{ \"enabled\": {}, \"width\": {} }},",
        profile.compact_spill_digest_enabled, profile.compact_spill_digest_width,
    )?;
    writeln!(
        file,
        "  \"prune_backend\": \"{}\",",
        profile.prune_backend.as_str()
    )?;
    writeln!(
        file,
        "  \"lrpf\": {{ \"target_degree\": {}, \"long_edge_quota\": {}, \"pre_spill_compress_enabled\": {}, \"pre_spill_width\": {}, \"candidate_records\": {}, \"witness_records\": {}, \"fragments\": {}, \"pre_spill_pruned_candidates\": {}, \"reducer_pruned_candidates\": {} }},",
        profile.lrpf_target_degree,
        profile.lrpf_long_edge_quota,
        profile.lrpf_pre_spill_compress_enabled,
        profile.lrpf_pre_spill_width,
        profile.lrpf_candidate_records,
        profile.lrpf_witness_records,
        profile.lrpf_fragments,
        profile.lrpf_pre_spill_pruned_candidates,
        profile.lrpf_reducer_pruned_candidates,
    )?;
    writeln!(
        file,
        "  \"resident_reservoir\": {{ \"points\": {}, \"bytes\": {}, \"layout\": \"{}\", \"mapped_bytes\": {}, \"segment_count\": {}, \"stripe_count\": {}, \"scratch_bytes\": {}, \"allocation_phase\": \"{}\", \"initialization_mode\": \"{}\", \"initialization_requested\": \"{}\", \"initialization_workers\": {}, \"initialization_overlap_enabled\": {}, \"initialization_invalid_override\": {}, \"initialization_parallel_fallback\": {}, \"initialization_d0_paced\": {}, \"initialization_ready_prefix\": {}, \"initialization_initialized_prefix\": {}, \"initialization_frontier_wait_count\": {}, \"initialization_frontier_wait_ms\": {}, \"allocation_ms\": {}, \"d1_wait_ms\": {}, \"hidden_ms\": {} }},",
        profile.resident_reservoir_points,
        profile.resident_reservoir_bytes,
        profile.reservoir_initialization.layout,
        profile.reservoir_initialization.mapped_bytes,
        profile.reservoir_initialization.segment_count,
        profile.reservoir_initialization.stripe_count,
        profile.reservoir_initialization.scratch_bytes,
        profile.reservoir_initialization.allocation_phase,
        profile.reservoir_initialization.mode,
        profile.reservoir_initialization.requested,
        profile.reservoir_initialization.workers,
        profile.reservoir_initialization.overlap_enabled,
        profile.reservoir_initialization.invalid_override,
        profile.reservoir_initialization.parallel_fallback,
        profile.reservoir_initialization.d0_paced,
        profile.reservoir_initialization.ready_prefix,
        profile.reservoir_initialization.initialized_prefix,
        profile.reservoir_initialization.frontier_wait_count,
        profile
            .reservoir_initialization
            .frontier_wait_wall
            .as_millis(),
        profile.reservoir_allocation_wall.as_millis(),
        profile.reservoir_initialization.d1_wait_wall.as_millis(),
        profile.reservoir_initialization.hidden_wall().as_millis(),
    )?;
    writeln!(
        file,
        "  \"resident_emit\": {{ \"parallel\": {}, \"chunk_sources\": {}, \"fifo_chunks\": {}, \"initial_workers\": {}, \"peak_workers\": {}, \"chunks\": {}, \"rows\": {}, \"max_completed_inflight\": {}, \"finalize_service_ms\": {}, \"callback_service_ms\": {}, \"consumer_wait_ms\": {}, \"producer_backpressure_ms\": {}, \"writer_buffer_bytes\": {} }},",
        profile.resident_emit.parallel,
        profile.resident_emit.chunk_sources,
        profile.resident_emit.fifo_chunks,
        profile.resident_emit.initial_workers,
        profile.resident_emit.peak_workers,
        profile.resident_emit.chunks,
        profile.resident_emit.rows,
        profile.resident_emit.max_completed_inflight,
        profile.resident_emit.finalize_service_wall.as_millis(),
        profile.resident_emit.callback_service_wall.as_millis(),
        profile.resident_emit.consumer_wait_wall.as_millis(),
        profile.resident_emit.producer_backpressure_wall.as_millis(),
        profile.resident_emit.writer_buffer_bytes,
    )?;
    writeln!(
        file,
        "  \"memory_plan\": {{ \"total_budget_bytes\": {}, \"reservoir_budget_bytes\": {}, \"resident_subtree_budget_bytes\": {}, \"required_resident_bytes\": {}, \"optional_resident_budget_bytes\": {}, \"optional_resident_admitted_bytes\": {}, \"optional_resident_unassigned_bytes\": {}, \"transient_pipeline_budget_bytes\": {}, \"d1plus_working_set_budget_bytes\": {}, \"sidecar_resident_kind\": \"{}\", \"sidecar_resident_budget_bytes\": {}, \"sidecar_resident_required_bytes\": {}, \"sidecar_resident_admitted_rows\": {}, \"sidecar_resident_admissible\": {}, \"scratch_budget_bytes\": {}, \"io_headroom_bytes\": {}, \"raw_vector_cache_budget_bytes\": {} }},",
        profile.memory_budget_bytes,
        profile.reservoir_budget_bytes,
        profile.resident_subtree_budget_bytes,
        profile.required_resident_bytes,
        profile.optional_resident_budget_bytes,
        profile.optional_resident_admitted_bytes,
        profile.optional_resident_unassigned_bytes,
        profile.transient_pipeline_budget_bytes,
        profile.d1plus_working_set_budget_bytes,
        profile.sidecar_resident_kind.as_str(),
        profile.sidecar_resident_budget_bytes,
        profile.sidecar_resident_required_bytes,
        profile.sidecar_resident_admitted_rows,
        profile.sidecar_resident_admissible,
        profile.scratch_budget_bytes,
        profile.io_headroom_bytes,
        profile.raw_vector_cache_budget_bytes,
    )?;
    writeln!(
        file,
        "  \"raw_vector_cache\": {{ \"enabled\": {}, \"budget_bytes\": {}, \"row_capacity\": {}, \"hits\": {}, \"misses\": {}, \"hit_bytes\": {}, \"miss_bytes\": {}, \"inserts\": {}, \"evictions\": {}, \"hit_ratio\": {:.6} }},",
        profile.raw_vector_cache_budget_bytes > 0,
        profile.raw_vector_cache_budget_bytes,
        profile.raw_vector_cache_rows,
        raw_vector_cache_stats.raw_cache_hits,
        raw_vector_cache_stats.raw_cache_misses,
        raw_vector_cache_stats.raw_cache_hit_bytes,
        raw_vector_cache_stats.raw_cache_miss_bytes,
        raw_vector_cache_stats.raw_cache_inserts,
        raw_vector_cache_stats.raw_cache_evictions,
        raw_vector_cache_stats.raw_cache_hit_ratio(),
    )?;
    let state_writeback_records_per_page = if profile.online_suffix.state_pages_written == 0 {
        0.0
    } else {
        profile.online_suffix.state_writeback_records as f64
            / profile.online_suffix.state_pages_written as f64
    };
    let mut online_suffix_reduce = serde_json::json!({
        "mode": profile.online_suffix_mode.as_str(),
        "active": profile.online_suffix_config.active,
        "fallback_reason": profile.online_suffix_fallback_reason,
        "configured_direct_queue_bytes": profile.online_suffix_config.configured_direct_queue_bytes,
        "configured_state_cache_bytes": profile.online_suffix_config.configured_state_cache_bytes,
        "configured_state_page_cache_bytes": profile.online_suffix_config.configured_state_page_cache_bytes,
        "configured_state_writeback_bytes": profile.online_suffix_config.configured_state_writeback_bytes,
        "configured_spill_buffer_bytes": profile.online_suffix_config.configured_spill_buffer_bytes,
        "configured_reorder_window_bytes": profile.online_suffix_config.configured_reorder_window_bytes,
        "configured_read_window_bytes": profile.online_suffix_config.configured_read_window_bytes,
        "configured_delta_budget_bytes": profile.online_suffix_config.configured_delta_budget_bytes,
        "configured_delta_buffer_bytes": profile.online_suffix_config.configured_delta_buffer_bytes,
        "configured_range_points": profile.online_suffix_config.configured_range_points,
        "configured_range_trigger_bytes": profile.online_suffix_config.configured_range_trigger_bytes,
        "configured_max_runs_per_range": profile.online_suffix_config.configured_max_runs_per_range,
        "configured_compaction_workers": profile.online_suffix_config.configured_compaction_workers,
        "configured_cache_hot_updates": profile.online_suffix_config.configured_cache_hot_updates,
        "effective_direct_queue_bytes": profile.online_suffix_config.effective_direct_queue_bytes,
        "effective_state_cache_bytes": profile.online_suffix_config.effective_state_cache_bytes,
        "effective_state_page_cache_bytes": profile.online_suffix_config.effective_state_page_cache_bytes,
        "effective_state_writeback_bytes": profile.online_suffix_config.effective_state_writeback_bytes,
        "effective_spill_buffer_bytes": profile.online_suffix_config.effective_spill_buffer_bytes,
        "effective_reorder_window_bytes": profile.online_suffix_config.effective_reorder_window_bytes,
        "effective_decode_bytes": profile.online_suffix_config.effective_decode_bytes,
        "effective_read_window_bytes": profile.online_suffix_config.effective_read_window_bytes,
        "effective_delta_budget_bytes": profile.online_suffix_config.effective_delta_budget_bytes,
        "effective_delta_buffer_bytes": profile.online_suffix_config.effective_delta_buffer_bytes,
        "effective_compaction_bytes": profile.online_suffix_config.effective_compaction_bytes,
        "effective_shard_points": profile.online_suffix_config.effective_shard_points,
        "effective_range_points": profile.online_suffix_config.effective_range_points,
        "effective_range_trigger_bytes": profile.online_suffix_config.effective_range_trigger_bytes,
        "effective_max_runs_per_range": profile.online_suffix_config.effective_max_runs_per_range,
        "effective_compaction_workers": profile.online_suffix_config.effective_compaction_workers,
        "effective_drain_compaction_workers": profile.online_suffix_config.effective_drain_compaction_workers,
        "effective_cache_hot_updates": profile.online_suffix_config.effective_cache_hot_updates,
        "reserved_metadata_bytes": profile.online_suffix_config.reserved_metadata_bytes,
        "reserved_total_bytes": profile.online_suffix_config.reserved_total_bytes,
        "configured_reduce_workers": profile.online_suffix_config.configured_reduce_workers,
        "effective_reduce_workers": profile.online_suffix_config.effective_reduce_workers,
        "state_overprovision_ratio": profile.online_suffix_config.state_overprovision_ratio,
    });
    let online_suffix_io = serde_json::json!({
        "direct_candidates": profile.online_suffix.direct_candidates,
        "direct_batches": profile.online_suffix.direct_batches,
        "direct_queue_peak_bytes": profile.online_suffix.direct_queue_peak_bytes,
        "spill_fallback_candidates": profile.online_suffix.spill_fallback_candidates,
        "spill_fallback_bytes": profile.online_suffix.spill_fallback_bytes,
        "spill_segments_consumed": profile.online_suffix.spill_segments_consumed,
        "state_records_loaded": profile.online_suffix.state_records_loaded,
        "state_records_updated": profile.online_suffix.state_records_updated,
        "state_page_cache_hits": profile.online_suffix.state_page_cache_hits,
        "state_page_cache_misses": profile.online_suffix.state_page_cache_misses,
        "state_pages_read": profile.online_suffix.state_pages_read,
        "state_pages_written": profile.online_suffix.state_pages_written,
        "state_slots_reused": profile.online_suffix.state_slots_reused,
        "state_write_amplification": profile.online_suffix.state_write_amplification,
        "backlog_bytes_at_last_gpu": profile.online_suffix.backlog_bytes_at_last_gpu,
        "post_gpu_drain_ms": profile.online_suffix.post_gpu_drain_wall.as_millis(),
        "online_reduce_wall_ms": profile.online_suffix.online_reduce_wall.as_millis(),
        "online_reduce_hidden_ms": profile.online_suffix.online_reduce_hidden_wall.as_millis(),
    });
    let online_suffix_log = serde_json::json!({
        "cold_batch_commits": profile.online_suffix.cold_batch_commits,
        "cold_batch_records": profile.online_suffix.cold_batch_records,
        "delta_records": profile.online_suffix.delta_records,
        "delta_bytes": profile.online_suffix.delta_bytes,
        "delta_runs": profile.online_suffix.delta_runs,
        "delta_buffer_flushes": profile.online_suffix.delta_buffer_flushes,
        "delta_uncompacted_bytes": profile.online_suffix.delta_uncompacted_bytes,
        "delta_uncompacted_peak_bytes": profile.online_suffix.delta_uncompacted_peak_bytes,
        "delta_backpressure_count": profile.online_suffix.delta_backpressure_count,
        "delta_backpressure_ms": profile.online_suffix.delta_backpressure_wall.as_millis(),
        "checkpoint_records": profile.online_suffix.checkpoint_records,
        "checkpoint_runs": profile.online_suffix.checkpoint_runs,
        "checkpoint_bytes": profile.online_suffix.checkpoint_bytes,
        "range_compactions": profile.online_suffix.range_compactions,
        "delta_run_pressure_crossings": profile.online_suffix.delta_run_pressure_crossings,
        "delta_run_pressure_compactions": profile.online_suffix.delta_run_pressure_compactions,
        "range_pre_gpu_materializations": profile.online_suffix.range_pre_gpu_materializations,
        "log_shards_finalized": profile.online_suffix.log_shards_finalized,
        "range_ingress_drained_compactions": profile.online_suffix.range_ingress_drained_compactions,
        "delta_peak_active_runs_per_range": profile.online_suffix.delta_peak_active_runs_per_range,
        "range_compactions_inflight": profile.online_suffix.range_compactions_inflight,
        "range_compactions_peak_inflight": profile.online_suffix.range_compactions_peak_inflight,
        "post_gpu_range_compactions_peak_inflight": profile.online_suffix.post_gpu_range_compactions_peak_inflight,
        "range_rows_compacted": profile.online_suffix.range_rows_compacted,
        "range_rows_changed": profile.online_suffix.range_rows_changed,
        "range_compaction_input_bytes": profile.online_suffix.range_compaction_input_bytes,
        "range_compaction_output_bytes": profile.online_suffix.range_compaction_output_bytes,
        "range_compaction_read_ms": profile.online_suffix.range_compaction_read_wall.as_millis(),
        "range_compaction_merge_ms": profile.online_suffix.range_compaction_merge_wall.as_millis(),
        "range_compaction_write_ms": profile.online_suffix.range_compaction_write_wall.as_millis(),
        "range_compaction_hidden_ms": profile.online_suffix.range_compaction_hidden_wall.as_millis(),
        "range_manifest_switches": profile.online_suffix.range_manifest_switches,
        "cold_miss_base_reads": profile.online_suffix.cold_miss_base_reads,
        "log_base_reads": profile.online_suffix.log_base_reads,
        "log_base_writes": profile.online_suffix.log_base_writes,
        "post_gpu_uncompacted_bytes": profile.online_suffix.post_gpu_uncompacted_bytes,
        "post_gpu_compaction_ms": profile.online_suffix.post_gpu_compaction_wall.as_millis(),
        "log_cache_promotions": profile.online_suffix.log_cache_promotions,
        "log_cache_promotion_skipped_post_cut": profile.online_suffix.log_cache_promotion_skipped_post_cut,
        "log_cache_hits": profile.online_suffix.log_cache_hits,
        "log_cache_dirty_checkpoints": profile.online_suffix.log_cache_dirty_checkpoints,
    });
    let online_suffix_borrowed = serde_json::json!({
        "borrowed_spill_records": profile.online_suffix.borrowed_spill_records,
        "borrowed_spill_runs": profile.online_suffix.borrowed_spill_runs,
        "borrowed_spill_bytes": profile.online_suffix.borrowed_spill_bytes,
    });
    let online_suffix_cache = serde_json::json!({
        "state_cache_partitions": profile.online_suffix.state_cache_partitions,
        "state_cache_capacity_entries": profile.online_suffix.state_cache_capacity_entries,
        "state_cache_peak_entries": profile.online_suffix.state_cache_peak_entries,
        "state_cache_hits": profile.online_suffix.state_cache_hits,
        "state_cache_misses": profile.online_suffix.state_cache_misses,
        "state_cache_clean_evictions": profile.online_suffix.state_cache_clean_evictions,
        "state_cache_dirty_evictions": profile.online_suffix.state_cache_dirty_evictions,
        "state_cache_coalesced_updates": profile.online_suffix.state_cache_coalesced_updates,
        "state_cache_updates_per_commit": profile.online_suffix.state_cache_updates_per_commit,
        "state_read_windows": profile.online_suffix.state_read_windows,
        "state_read_pages_requested": profile.online_suffix.state_read_pages_requested,
        "state_read_pages_coalesced": profile.online_suffix.state_read_pages_coalesced,
        "state_writeback_cohorts": profile.online_suffix.state_writeback_cohorts,
        "state_writeback_records": profile.online_suffix.state_writeback_records,
        "state_writeback_records_per_page": state_writeback_records_per_page,
        "state_alloc_empty_page": profile.online_suffix.state_alloc_empty_page,
        "state_alloc_resident_partial_page": profile.online_suffix.state_alloc_resident_partial_page,
        "state_alloc_cold_partial_page": profile.online_suffix.state_alloc_cold_partial_page,
        "pinned_prefix_points": profile.online_suffix.pinned_prefix_points,
    });
    {
        let online_suffix_reduce_object = online_suffix_reduce
            .as_object_mut()
            .expect("online suffix profile must be a JSON object");
        online_suffix_reduce_object.extend(
            online_suffix_io
                .as_object()
                .expect("online suffix I/O profile must be a JSON object")
                .clone(),
        );
        online_suffix_reduce_object.extend(
            online_suffix_cache
                .as_object()
                .expect("online suffix cache profile must be a JSON object")
                .clone(),
        );
        online_suffix_reduce_object.extend(
            online_suffix_log
                .as_object()
                .expect("online suffix log profile must be a JSON object")
                .clone(),
        );
        online_suffix_reduce_object.extend(
            online_suffix_borrowed
                .as_object()
                .expect("online suffix borrowed-spill profile must be a JSON object")
                .clone(),
        );
    }
    writeln!(file, "  \"online_suffix_reduce\": {online_suffix_reduce},")?;
    writeln!(file, "  \"sketch_bytes\": {},", profile.sketch_bytes)?;
    writeln!(
        file,
        "  \"sketch_file_bytes\": {},",
        profile.sketch_file_bytes
    )?;
    writeln!(
        file,
        "  \"spill\": {{ \"part_files\": {}, \"segment_count\": {}, \"shard_groups\": {}, \"max_parts_per_shard\": {}, \"max_segments_per_shard\": {}, \"record_bytes\": {}, \"manifest_bytes\": {} }},",
        profile.spill.part_files,
        profile.spill.segment_count,
        profile.spill.shard_groups,
        profile.spill.max_parts_per_shard,
        profile.spill.max_segments_per_shard,
        profile.spill.record_bytes,
        profile.spill.manifest_bytes,
    )?;
    let reduce_read_decode_residual = profile.reduce.read_decode_wall.saturating_sub(
        profile.reduce.record_read_wall
            + profile.reduce.record_decode_wall
            + profile.reduce.record_append_wall,
    );
    writeln!(
        file,
        "  \"reduce\": {{ \"shard_groups\": {}, \"segments_scanned\": {}, \"part_files_opened\": {}, \"max_segments_in_group\": {}, \"parallel_workers\": {}, \"decoder_workers\": {}, \"parallel_batches\": {}, \"parallel_records\": {}, \"batch_peak_records\": {}, \"batch_peak_bytes\": {}, \"read_decode_ms\": {}, \"record_read_ms\": {}, \"record_decode_ms\": {}, \"record_append_ms\": {}, \"read_decode_residual_ms\": {}, \"insert_ms\": {}, \"pipeline_ms\": {}, \"finalize_emit_ms\": {} }},",
        profile.reduce.shard_groups,
        profile.reduce.segments_scanned,
        profile.reduce.part_files_opened,
        profile.reduce.max_segments_in_group,
        profile.reduce.parallel_workers,
        profile.reduce.decoder_workers,
        profile.reduce.parallel_batches,
        profile.reduce.parallel_records,
        profile.reduce.batch_peak_records,
        profile.reduce.batch_peak_bytes,
        profile.reduce.read_decode_wall.as_millis(),
        profile.reduce.record_read_wall.as_millis(),
        profile.reduce.record_decode_wall.as_millis(),
        profile.reduce.record_append_wall.as_millis(),
        reduce_read_decode_residual.as_millis(),
        profile.reduce.insert_wall.as_millis(),
        profile.reduce.pipeline_wall.as_millis(),
        profile.reduce.finalize_emit_wall.as_millis(),
    )?;
    writeln!(
        file,
        "  \"stage_ms\": {{ \"sketch_persist\": {}, \"streaming\": {}, \"d0_ads_assignment\": {}, \"d1_ads_assignment\": {}, \"d1_materialize\": {}, \"d2_replay\": {}, \"reduce\": {}, \"external_reduce\": {}, \"emit\": {}, \"total\": {} }},",
        profile.sketch_persist_wall.as_millis(),
        profile.streaming_wall.as_millis(),
        profile.rbc.d0_ads_assignment_wall.as_millis(),
        profile.rbc.d1_ads_assignment_wall.as_millis(),
        profile.rbc.d1_materialize_wall.as_millis(),
        profile.rbc.d2_replay_wall.as_millis(),
        profile.reduce_wall.as_millis(),
        profile.reduce_wall.as_millis(),
        profile.emit_wall.as_millis(),
        profile.total_wall.as_millis(),
    )?;
    writeln!(
        file,
        "  \"serial_timing\": {{ \"schema\": \"gpu_subtree_serial_nonoverlap_v1\", \"unit\": \"ms\", \"rows\": {{ \"sketch_persist_ms\": {}, \"d0_root_assignment_ms\": {}, \"d0_kmeans_train_ms\": {}, \"d0_kmeans_assign_ms\": {}, \"d0_sq_raw_read_ms\": {}, \"d0_sq_raw_h2d_ms\": {}, \"d0_sq_quantize_ms\": {}, \"d0_sq_d0_assign_ms\": {}, \"d0_sq_qcode_write_ms\": {}, \"d0_residual_ms\": {}, \"d1_materialize_ms\": {}, \"d2_replay_ms\": {}, \"persistent_session_init_ms\": {}, \"d1plus_sidecar_wait_ms\": {}, \"d1plus_payload_h2d_ms\": {}, \"d1plus_persistent_kernel_ms\": {}, \"d1plus_cpu_result_scatter_ms\": {}, \"d1plus_backend_residual_ms\": {}, \"d1plus_child_replay_ms\": {}, \"d1plus_lazy_split_ms\": {}, \"d1plus_push_wait_ms\": {}, \"d1plus_validate_ms\": {}, \"d1plus_merge_profile_ms\": {}, \"d1plus_push_batch_residual_ms\": {}, \"persistent_session_drain_ms\": {}, \"d1plus_producer_residual_ms\": {}, \"streaming_residual_ms\": {}, \"external_reduce_ms\": {}, \"emit_ms\": {}, \"stage_outer_residual_ms\": {} }}, \"accounted_sum_ms\": {}, \"stage_total_ms\": {}, \"sum_matches_stage_total\": {}, \"diagnostics\": {{ \"d1plus_producer_ms\": {}, \"d1plus_backend_ms\": {}, \"streaming_ms\": {}, \"d1plus_sidecar_gather_total_ms\": {}, \"d0_root_assignment_total_ms\": {}, \"d0_nested_requested_total_ms\": {}, \"d0_sq_qcode_write_total_ms\": {}, \"d0_nested_overlap_hidden_ms\": {}, \"cpu_compact_flush_ms\": {}, \"post_last_gpu_cpu_tail_ms\": {}, \"post_last_gpu_cpu_tail_observed\": {}, \"edge_sink_ms\": {}, \"edge_sink_sort_ms\": {}, \"edge_sink_insert_ms\": {}, \"sidecar_gather_active_ms\": {}, \"sidecar_gather_hidden_ms\": {}, \"gpu_wait_for_sidecar_ms\": {}, \"async_result_scatter_worker_ms\": {}, \"async_result_scatter_wait_ms\": {}, \"async_result_scatter_join_wait_ms\": {}, \"async_result_scatter_max_inflight\": {}, \"async_result_scatter_peak_raw_bytes\": {}, \"async_result_scatter_peak_context_bytes\": {}, \"async_result_scatter_peak_host_bytes\": {}, \"d1plus_child_replay_ms\": {}, \"d1plus_lazy_split_ms\": {}, \"d1plus_push_wait_ms\": {}, \"d1plus_validate_ms\": {}, \"d1plus_merge_profile_ms\": {}, \"d1plus_push_batch_total_ms\": {}, \"host_vector_dedup_ms\": {}, \"host_vector_child_gather_ms\": {}, \"sidecar_preload_workers\": {}, \"sidecar_preload_children\": {}, \"sidecar_queue_max_bytes\": {}, \"sidecar_queue_max_children\": {}, \"d1plus_admission_split_count\": {}, \"d1plus_admission_split_rows\": {}, \"d1plus_preflight_split_count\": {}, \"d1plus_preflight_split_rows\": {} }} }},",
        serial_timing.sketch_persist.as_millis(),
        serial_timing.d0_root_assignment.as_millis(),
        serial_timing.d0_kmeans_train.as_millis(),
        serial_timing.d0_kmeans_assign.as_millis(),
        serial_timing.d0_sq_raw_read.as_millis(),
        serial_timing.d0_sq_raw_h2d.as_millis(),
        serial_timing.d0_sq_quantize.as_millis(),
        serial_timing.d0_sq_d0_assign.as_millis(),
        serial_timing.d0_sq_qcode_write.as_millis(),
        serial_timing.d0_residual.as_millis(),
        serial_timing.d1_materialize.as_millis(),
        serial_timing.d2_replay.as_millis(),
        serial_timing.persistent_session_init.as_millis(),
        serial_timing.d1plus_sidecar_wait.as_millis(),
        serial_timing.d1plus_payload_h2d.as_millis(),
        serial_timing.d1plus_persistent_kernel.as_millis(),
        serial_timing.d1plus_cpu_result_scatter.as_millis(),
        serial_timing.d1plus_backend_residual.as_millis(),
        serial_timing.d1plus_child_replay.as_millis(),
        serial_timing.d1plus_lazy_split.as_millis(),
        serial_timing.d1plus_push_wait.as_millis(),
        serial_timing.d1plus_validate.as_millis(),
        serial_timing.d1plus_merge_profile.as_millis(),
        serial_timing.d1plus_push_batch_residual.as_millis(),
        serial_timing.persistent_session_drain.as_millis(),
        serial_timing.d1plus_producer_residual.as_millis(),
        serial_timing.streaming_residual.as_millis(),
        serial_timing.external_reduce.as_millis(),
        serial_timing.emit.as_millis(),
        serial_timing.stage_outer_residual.as_millis(),
        serial_timing.accounted_sum.as_millis(),
        serial_timing.stage_total.as_millis(),
        serial_timing.accounted_sum == serial_timing.stage_total,
        profile.rbc.gpu_subtree_d1plus_producer_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_backend_wall.as_millis(),
        profile.streaming_wall.as_millis(),
        serial_timing.d1plus_sidecar_gather_total.as_millis(),
        serial_timing.d0_root_assignment_total.as_millis(),
        serial_timing.d0_nested_requested_total.as_millis(),
        serial_timing.d0_sq_qcode_write_total.as_millis(),
        serial_timing.d0_nested_overlap_hidden.as_millis(),
        profile.rbc.gpu_subtree_cpu_compact_flush_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_post_last_gpu_cpu_tail_wall
            .as_millis(),
        profile.rbc.gpu_subtree_post_last_gpu_cpu_tail_observed,
        profile.leaf.edge_sink_wall.as_millis(),
        profile.leaf.edge_sink_sort_wall.as_millis(),
        profile.leaf.edge_sink_insert_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_sidecar_gather_active_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_sidecar_gather_hidden_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_gpu_wait_for_sidecar_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_async_result_scatter_worker_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_async_result_scatter_wait_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_async_result_scatter_join_wait_wall
            .as_millis(),
        profile.rbc.gpu_subtree_async_result_scatter_max_inflight,
        profile.rbc.gpu_subtree_async_result_scatter_peak_raw_bytes,
        profile
            .rbc
            .gpu_subtree_async_result_scatter_peak_context_bytes,
        profile.rbc.gpu_subtree_async_result_scatter_peak_host_bytes,
        profile.rbc.gpu_subtree_d1plus_child_replay_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_lazy_split_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_push_wait_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_validate_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_d1plus_merge_profile_wall
            .as_millis(),
        serial_timing.d1plus_push_batch_total.as_millis(),
        profile.rbc.gpu_subtree_host_vector_dedup_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_host_vector_child_gather_wall
            .as_millis(),
        profile.rbc.gpu_subtree_sidecar_preload_workers,
        profile.rbc.gpu_subtree_sidecar_preload_children,
        profile.rbc.gpu_subtree_sidecar_queue_max_bytes,
        profile.rbc.gpu_subtree_sidecar_queue_max_children,
        profile.rbc.gpu_subtree_d1plus_admission_split_count,
        profile.rbc.gpu_subtree_d1plus_admission_split_rows,
        profile.rbc.gpu_subtree_d1plus_preflight_split_count,
        profile.rbc.gpu_subtree_d1plus_preflight_split_rows,
    )?;
    writeln!(
        file,
        "  \"memory\": {{ \"rss_kb_samples\": [{}], \"rss_hwm_kb\": {}, \"vm_peak_kb\": {} }},",
        profile
            .memory
            .rss_kb_samples
            .iter()
            .map(|sample| sample.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        format_optional_number(profile.memory.rss_hwm_kb),
        format_optional_number(profile.memory.vm_peak_kb),
    )?;
    writeln!(
        file,
        "  \"temp_files\": {{ \"total\": {{ \"files\": {}, \"bytes\": {} }}, \"candidate\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth0\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth1\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth2\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth3\": {{ \"files\": {}, \"bytes\": {} }}, \"other\": {{ \"files\": {}, \"bytes\": {} }} }},",
        profile.temp_files.total.files,
        profile.temp_files.total.bytes,
        profile.temp_files.candidate.files,
        profile.temp_files.candidate.bytes,
        profile.temp_files.partition_depth0.files,
        profile.temp_files.partition_depth0.bytes,
        profile.temp_files.partition_depth1.files,
        profile.temp_files.partition_depth1.bytes,
        profile.temp_files.partition_depth2.files,
        profile.temp_files.partition_depth2.bytes,
        profile.temp_files.partition_depth3.files,
        profile.temp_files.partition_depth3.bytes,
        profile.temp_files.other.files,
        profile.temp_files.other.bytes,
    )?;
    writeln!(
        file,
        "  \"io_uring\": {{ \"point_gather_batches\": {}, \"point_gather_windows\": {}, \"point_gather_bytes\": {}, \"point_gather_fallbacks\": {} }},",
        profile.uring_batches, profile.uring_windows, profile.uring_bytes, profile.uring_fallbacks,
    )?;
    writeln!(
        file,
        "  \"scheduler\": {{ \"peak_inflight_leaf_tasks\": {}, \"peak_leaf_backlog\": {}, \"inline_leaf_fallbacks\": {}, \"producer_help_drains\": {}, \"producer_help_yields\": {}, \"large_leaf_count\": {}, \"large_leaf_block_tasks\": {}, \"producer_wait_ms\": {}, \"staging_wait_ms\": {}, \"compute_wait_ms\": {}, \"edge_wait_ms\": {}, \"producer_blocked_ms\": {}, \"compute_blocked_ms\": {}, \"edge_blocked_ms\": {}, \"peak_leaf_descriptor_queue\": {}, \"peak_staged_queue\": {}, \"peak_topk_queue\": {}, \"peak_edge_queue\": {}, \"cpu_shadow_unsupported_fallbacks\": {}, \"gpu_workspace_pressure_events\": {}, \"gpu_cpu_fallbacks\": {}, \"post_producer_leaf_drain_ms\": {} }},",
        profile.scheduler.peak_inflight_leaf_tasks,
        profile.scheduler.peak_leaf_backlog,
        profile.scheduler.inline_leaf_fallbacks,
        profile.scheduler.producer_help_drains,
        profile.scheduler.producer_help_yields,
        profile.scheduler.large_leaf_count,
        profile.scheduler.large_leaf_block_tasks,
        profile.scheduler.producer_wait_ms,
        profile.scheduler.staging_wait_ms,
        profile.scheduler.compute_wait_ms,
        profile.scheduler.edge_wait_ms,
        profile.scheduler.producer_blocked_ms,
        profile.scheduler.compute_blocked_ms,
        profile.scheduler.edge_blocked_ms,
        profile.scheduler.peak_leaf_descriptor_queue,
        profile.scheduler.peak_staged_queue,
        profile.scheduler.peak_topk_queue,
        profile.scheduler.peak_edge_queue,
        profile.scheduler.cpu_shadow_unsupported_fallbacks,
        profile.scheduler.gpu_workspace_pressure_events,
        profile.scheduler.gpu_cpu_fallbacks,
        profile.scheduler.post_producer_leaf_drain_ms,
    )?;
    writeln!(
        file,
        "  \"pipeline\": {{ \"descriptor_receive_wall_ms\": {}, \"descriptor_batch_extend_wall_ms\": {}, \"stage_wall_ms\": {}, \"compute_flush_wall_ms\": {}, \"topk_queue_roundtrip_wall_ms\": {}, \"edge_queue_roundtrip_wall_ms\": {}, \"edge_flush_wall_ms\": {}, \"descriptor_queue_empty_count\": {}, \"cuda_batch_count\": {}, \"cuda_batch_leaves\": {}, \"cuda_batch_rows\": {}, \"cuda_batch_target_bytes\": {}, \"cuda_batch_actual_bytes\": {}, \"cuda_batch_underfilled_count\": {} }},",
        profile.scheduler.descriptor_receive_wall_ms,
        profile.scheduler.descriptor_batch_extend_wall_ms,
        profile.scheduler.stage_wall_ms,
        profile.scheduler.compute_flush_wall_ms,
        profile.scheduler.topk_queue_roundtrip_wall_ms,
        profile.scheduler.edge_queue_roundtrip_wall_ms,
        profile.scheduler.edge_flush_wall_ms,
        profile.scheduler.descriptor_queue_empty_count,
        profile.scheduler.cuda_batch_count,
        profile.scheduler.cuda_batch_leaves,
        profile.scheduler.cuda_batch_rows,
        profile.scheduler.cuda_batch_target_bytes,
        profile.scheduler.cuda_batch_actual_bytes,
        profile.scheduler.cuda_batch_underfilled_count,
    )?;
    writeln!(
        file,
        "  \"leaf_flow\": {{ \"host_load_wall_ms\": {}, \"backend_compute_wall_ms\": {}, \"compact_materialize_wall_ms\": {}, \"hash_wall_ms\": {}, \"edge_sink_wall_ms\": {}, \"edge_sink_sort_wall_ms\": {}, \"edge_sink_insert_wall_ms\": {}, \"edge_sink_lock_wall_ms\": {}, \"edge_sink_insert_body_wall_ms\": {}, \"edge_sink_edges\": {}, \"edge_sink_flushes\": {}, \"edge_sink_groups\": {}, \"edge_sink_max_group_edges\": {}, \"edge_sink_insert_candidates\": {}, \"edge_sink_hash_hits\": {}, \"edge_sink_hash_replacements\": {}, \"edge_sink_hash_rejects\": {}, \"edge_sink_not_full_inserts\": {}, \"edge_sink_full_replacement_checks\": {}, \"edge_sink_full_replacements\": {}, \"edge_sink_full_rejects\": {}, \"edge_sink_farthest_recomputes\": {}, \"edge_sink_rotate_lefts\": {}, \"edge_sink_rotate_rights\": {}, \"flush_wall_ms\": {}, \"compact_lune_witness_enabled\": {}, \"compact_lune_witness_slots\": {}, \"compact_lune_witnesses\": {}, \"compact_lune_prune_enabled\": {}, \"compact_lune_prune_protected_core\": {}, \"compact_lune_pruned_slots\": {}, \"compact_lune_pruned_rows\": {} }},",
        profile.leaf.host_load_wall.as_millis(),
        profile.leaf.backend_compute_wall.as_millis(),
        profile.leaf.compact_materialize_wall.as_millis(),
        profile.leaf.hash_wall.as_millis(),
        profile.leaf.edge_sink_wall.as_millis(),
        profile.leaf.edge_sink_sort_wall.as_millis(),
        profile.leaf.edge_sink_insert_wall.as_millis(),
        profile.leaf.edge_sink_lock_wall.as_millis(),
        profile.leaf.edge_sink_insert_body_wall.as_millis(),
        profile.leaf.edge_sink_edges,
        profile.leaf.edge_sink_flushes,
        profile.leaf.edge_sink_groups,
        profile.leaf.edge_sink_max_group_edges,
        profile.leaf.edge_sink_insert_candidates,
        profile.leaf.edge_sink_hash_hits,
        profile.leaf.edge_sink_hash_replacements,
        profile.leaf.edge_sink_hash_rejects,
        profile.leaf.edge_sink_not_full_inserts,
        profile.leaf.edge_sink_full_replacement_checks,
        profile.leaf.edge_sink_full_replacements,
        profile.leaf.edge_sink_full_rejects,
        profile.leaf.edge_sink_farthest_recomputes,
        profile.leaf.edge_sink_rotate_lefts,
        profile.leaf.edge_sink_rotate_rights,
        profile.leaf.flush_wall.as_millis(),
        profile.leaf.compact_lune_witness_enabled,
        profile.leaf.compact_lune_witness_slots,
        profile.leaf.compact_lune_witnesses,
        profile.leaf.compact_lune_prune_enabled,
        profile.leaf.compact_lune_prune_protected_core,
        profile.leaf.compact_lune_pruned_slots,
        profile.leaf.compact_lune_pruned_rows,
    )?;
    writeln!(
        file,
        "  \"hashprune_evidence\": {{ \"mode\": \"{}\", \"overflow_budget\": {}, \"annotation_ms\": {}, \"coalesce_sort_ms\": {}, \"coalesce_merge_ms\": {}, \"coalesce_ms\": {}, \"reciprocal_checks\": {}, \"reciprocal_hits\": {}, \"support_once_candidates\": {}, \"support_twice_candidates\": {}, \"class_reverse\": {}, \"class_reverse_rank0\": {}, \"class_direct\": {}, \"class_bidirectional\": {}, \"class_direct_rank0\": {}, \"class_bidirectional_direct_rank0\": {}, \"candidates_before_coalesce\": {}, \"candidates_after_coalesce\": {}, \"duplicate_records_removed\": {}, \"unannotated_candidates\": {}, \"qualified_collision_losers\": {}, \"overflow_inserts\": {}, \"overflow_replacements\": {}, \"overflow_rejects\": {}, \"overflow_evictions_by_new_hash\": {}, \"rows_with_overflow\": {}, \"spill_records_avoided\": {}, \"spill_bytes_avoided\": {} }},",
        profile.hashprune_evidence_policy.mode().as_str(),
        profile.hashprune_evidence_policy.overflow_budget(),
        profile.hashprune_evidence.annotation_wall.as_millis(),
        profile.hashprune_evidence.coalesce_sort_wall.as_millis(),
        profile.hashprune_evidence.coalesce_merge_wall.as_millis(),
        profile.hashprune_evidence.coalesce_wall.as_millis(),
        profile.hashprune_evidence.reciprocal_checks,
        profile.hashprune_evidence.reciprocal_hits,
        profile.hashprune_evidence.support_once_candidates,
        profile.hashprune_evidence.support_twice_candidates,
        profile.hashprune_evidence.class_counts[HashPruneEvidenceClass::Reverse.index()],
        profile.hashprune_evidence.class_counts[HashPruneEvidenceClass::ReverseRank0.index()],
        profile.hashprune_evidence.class_counts[HashPruneEvidenceClass::Direct.index()],
        profile.hashprune_evidence.class_counts[HashPruneEvidenceClass::Bidirectional.index()],
        profile.hashprune_evidence.class_counts[HashPruneEvidenceClass::DirectRank0.index()],
        profile.hashprune_evidence.class_counts
            [HashPruneEvidenceClass::BidirectionalDirectRank0.index()],
        profile.hashprune_evidence.candidates_before_coalesce,
        profile.hashprune_evidence.candidates_after_coalesce,
        profile.hashprune_evidence.duplicate_records_removed,
        profile.hashprune_evidence.unannotated_candidates,
        profile.hashprune_evidence.qualified_collision_losers,
        profile.hashprune_evidence.overflow_inserts,
        profile.hashprune_evidence.overflow_replacements,
        profile.hashprune_evidence.overflow_rejects,
        profile.hashprune_evidence.overflow_evictions_by_new_hash,
        profile.hashprune_evidence.rows_with_overflow,
        profile.hashprune_evidence.spill_records_avoided,
        profile.hashprune_evidence.spill_bytes_avoided,
    )?;
    // Preserve the canonical f32 display (for example, 1.1) instead of
    // exposing the widened binary representation as 1.100000023841858.
    let backfill_alpha_json = profile
        .hashprune_backfill_policy
        .alpha()
        .to_string()
        .parse::<serde_json::Number>()
        .expect("validated HashPrune backfill alpha must be a finite JSON number");
    let backfill_priority_alpha_json = profile
        .hashprune_backfill_policy
        .priority_alpha()
        .to_string()
        .parse::<serde_json::Number>()
        .expect("HashPrune backfill priority alpha must be a finite JSON number");
    let backfill_requested_priority_alpha_json = profile
        .hashprune_backfill_policy
        .requested_priority_alpha()
        .map(|value| {
            value
                .to_string()
                .parse::<serde_json::Number>()
                .expect("requested HashPrune priority alpha must be a finite JSON number")
        });
    let mut backfill_json = serde_json::json!({
        "mode": profile.hashprune_backfill_policy.mode().as_str(),
        "ablation": profile.hashprune_backfill_policy.ablation().as_str(),
        "budget_rule": profile.hashprune_backfill_policy.budget_rule(),
        "budget": profile.hashprune_backfill_policy.budget(),
        "candidate_budget": profile.hashprune_backfill_policy.candidate_budget(),
        "derived_budget": profile.hashprune_backfill_policy.budget_for(profile.hashprune_backfill_degree_cap),
        "backup_hash_quota": profile.hashprune_backfill_policy.backup_hash_quota(),
        "hash_quota": profile.hashprune_backfill_policy.backup_hash_quota(),
        "degree_target": profile.hashprune_backfill_policy.degree_target(),
        "degree_cap": profile.hashprune_backfill_degree_cap,
        "alpha": backfill_alpha_json,
        "priority_rule": profile.hashprune_backfill_policy.priority_rule(),
        "priority_alpha": backfill_priority_alpha_json,
        "uses_alpha_gate": profile.hashprune_backfill_policy.uses_alpha_gate(),
        "post_veto_refill": profile.hashprune_backfill_policy.post_veto_refill(),
        "rows_seen": profile.hashprune_backfill.rows_seen,
        "rows_underfull_before": profile.hashprune_backfill.rows_underfull_before,
        "rows_with_backfill": profile.hashprune_backfill.rows_with_backfill,
        "baseline_full_rows": profile.hashprune_backfill.baseline_full_rows,
        "rows_with_nonempty_shortlist": profile.hashprune_backfill.rows_with_nonempty_shortlist,
        "shortlist_admissions": profile.hashprune_backfill.shortlist_admissions,
        "shortlist_replacements": profile.hashprune_backfill.shortlist_replacements,
        "shortlist_unchanged": profile.hashprune_backfill.shortlist_unchanged,
        "shortlist_capacity_rejections": profile.hashprune_backfill.shortlist_capacity_rejections,
        "shortlisted_layer_1": profile.hashprune_backfill.shortlisted_by_layer[0],
        "shortlisted_layer_2": profile.hashprune_backfill.shortlisted_by_layer[1],
        "shortlisted_layer_3": profile.hashprune_backfill.shortlisted_by_layer[2],
        "emitted_layer_1": profile.hashprune_backfill.emitted_by_layer[0],
        "emitted_layer_2": profile.hashprune_backfill.emitted_by_layer[1],
        "emitted_layer_3": profile.hashprune_backfill.emitted_by_layer[2],
        "rows_with_deeper_backfill": profile.hashprune_backfill.rows_with_deeper_backfill,
        "gate_rejected_layer_1": profile.hashprune_backfill.gate_rejected_by_layer[0],
        "gate_rejected_layer_2": profile.hashprune_backfill.gate_rejected_by_layer[1],
        "gate_rejected_layer_3": profile.hashprune_backfill.gate_rejected_by_layer[2],
        "rows_with_veto_holes": profile.hashprune_backfill.rows_with_veto_holes,
        "primary_edges": profile.hashprune_backfill.primary_edges,
        "backfill_edges": profile.hashprune_backfill.backfill_edges,
        "gate_rejected": profile.hashprune_backfill.gate_rejected,
        "budget_rejected": profile.hashprune_backfill.budget_rejected,
        "mean_degree_before": profile.hashprune_backfill.mean_degree_before(),
        "mean_degree_after": profile.hashprune_backfill.mean_degree_after(),
    });
    let backfill_object = backfill_json
        .as_object_mut()
        .expect("HashPrune backfill profile must be a JSON object");
    backfill_object.insert(
        "capacity_rule".to_string(),
        serde_json::json!(profile.hashprune_backfill_policy.capacity_rule()),
    );
    backfill_object.insert(
        "requested_max_backfill_per_row".to_string(),
        serde_json::json!(
            profile
                .hashprune_backfill_policy
                .requested_max_backfill_per_row()
        ),
    );
    backfill_object.insert(
        "effective_max_backfill_per_row".to_string(),
        serde_json::json!(
            profile
                .hashprune_backfill_policy
                .effective_max_backfill_per_row(profile.hashprune_backfill_degree_cap)
        ),
    );
    backfill_object.insert(
        "requested_hash_quota".to_string(),
        serde_json::json!(profile.hashprune_backfill_policy.requested_hash_quota()),
    );
    backfill_object.insert(
        "effective_hash_quota".to_string(),
        serde_json::json!(profile.hashprune_backfill_policy.backup_hash_quota()),
    );
    backfill_object.insert(
        "requested_priority_alpha".to_string(),
        serde_json::json!(backfill_requested_priority_alpha_json),
    );
    backfill_object.insert(
        "effective_priority_alpha".to_string(),
        serde_json::json!(backfill_priority_alpha_json),
    );
    writeln!(file, "  \"hashprune_backfill\": {backfill_json},")?;
    writeln!(
        file,
        "  \"gpu\": {{ \"gpu_h2d_bytes\": {}, \"gpu_d2h_bytes\": {}, \"gpu_h2d_ms\": {}, \"gpu_d2h_ms\": {}, \"gpu_kernel_ms\": {}, \"gpu_sync_ms\": {}, \"gpu_h2d_submit_wall_ms\": {}, \"gpu_kernel_launch_wall_ms\": {}, \"gpu_kernel_elapsed_ms\": {}, \"gpu_d2h_submit_wall_ms\": {}, \"gpu_post_d2h_sync_wall_ms\": {}, \"gpu_stream_total_elapsed_ms\": {}, \"gpu_workspace_peak_bytes\": {}, \"gpu_workspace_budget_bytes\": {}, \"gpu_workspace_pressure_events\": {}, \"gpu_cpu_fallbacks\": {} }},",
        profile.leaf.gpu_h2d_bytes,
        profile.leaf.gpu_d2h_bytes,
        profile.leaf.gpu_h2d.as_millis(),
        profile.leaf.gpu_d2h.as_millis(),
        profile.leaf.gpu_kernel.as_millis(),
        profile.leaf.gpu_sync.as_millis(),
        profile.leaf.gpu_h2d_submit_wall.as_millis(),
        profile.leaf.gpu_kernel_launch.as_millis(),
        profile.leaf.gpu_kernel_elapsed.as_millis(),
        profile.leaf.gpu_d2h_submit_wall.as_millis(),
        profile.leaf.gpu_post_d2h_sync_wall.as_millis(),
        profile.leaf.gpu_stream_total_elapsed.as_millis(),
        profile.leaf.gpu_workspace_peak_bytes,
        profile.leaf.gpu_workspace_budget_bytes,
        profile.leaf.gpu_workspace_pressure_events,
        profile.leaf.gpu_cpu_fallbacks,
    )?;
    writeln!(
        file,
        "  \"gpu_subtree_timing\": {{ \"d0_root_assignment_ms\": {}, \"d1_materialize_ms\": {}, \"d2_replay_ms\": {}, \"d1plus_producer_ms\": {}, \"d1plus_lazy_split_ms\": {}, \"d1plus_backend_ms\": {}, \"host_load_vectors_ms\": {}, \"host_point_read_ms\": {}, \"host_vector_dedup_ms\": {}, \"host_vector_child_gather_ms\": {}, \"child_schedule_plan_ms\": {}, \"rabitq_residual_centroid_train_ms\": {}, \"rabitq_residual_centroid_assign_ms\": {}, \"rabitq_residual_gpu_quantize_ms\": {}, \"rabitq_residual_gpu_d0_assign_ms\": {}, \"rabitq_residual_d0_assign_distance_kernel_ms\": {}, \"rabitq_residual_d0_assign_merge_kernel_ms\": {}, \"rabitq_residual_leader_prepare_ms\": {}, \"rabitq_residual_leader_payload_h2d_ms\": {}, \"rabitq_residual_assignment_fragment_reduce_ms\": {}, \"rabitq_residual_qcode_shard_write_ms\": {}, \"rabitq_residual_qcode_child_gather_ms\": {}, \"rabitq_residual_global_payload_upload_ms\": {}, \"rabitq_residual_d1plus_child_id_h2d_ms\": {}, \"rabitq_plus_quantize_ms\": {}, \"rabitq_plus_read_ms\": {}, \"rabitq_plus_h2d_ms\": {}, \"rabitq_plus_kernel_event_ms\": {}, \"sq_wmma_quantize_ms\": {}, \"sq_wmma_h2d_ms\": {}, \"sq_wmma_kernel_event_ms\": {}, \"turboquant_quantize_ms\": {}, \"turboquant_read_ms\": {}, \"turboquant_h2d_ms\": {}, \"turboquant_kernel_event_ms\": {}, \"vector_h2d_ms\": {}, \"persistent_kernel_prepare_ms\": {}, \"persistent_kernel_event_ms\": {}, \"persistent_total_ms\": {}, \"cpu_result_scatter_ms\": {}, \"cpu_compact_flush_ms\": {}, \"edge_sink_ms\": {}, \"external_reduce_ms\": {}, \"emit_ms\": {}, \"stage_total_ms\": {}, \"build_wall_ms\": {} }},",
        profile.rbc.d0_ads_assignment_wall.as_millis(),
        profile.rbc.d1_materialize_wall.as_millis(),
        profile.rbc.d2_replay_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_producer_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_lazy_split_wall.as_millis(),
        profile.rbc.gpu_subtree_d1plus_backend_wall.as_millis(),
        profile.rbc.gpu_subtree_host_load_vectors_wall.as_millis(),
        profile.rbc.gpu_subtree_host_point_read_wall.as_millis(),
        profile.rbc.gpu_subtree_host_vector_dedup_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_host_vector_child_gather_wall
            .as_millis(),
        profile.rbc.gpu_subtree_child_schedule_plan_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_centroid_train_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_centroid_assign_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_gpu_quantize_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_gpu_d0_assign_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_d0_assign_distance_kernel_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_d0_assign_merge_kernel_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_leader_prepare_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_leader_payload_h2d_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_assignment_fragment_reduce_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_qcode_shard_write_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_qcode_child_gather_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_global_payload_upload_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_residual_d1plus_child_id_h2d_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_plus_quantize_wall
            .as_millis(),
        profile.rbc.gpu_subtree_rabitq_plus_read_wall.as_millis(),
        profile.rbc.gpu_subtree_rabitq_plus_h2d_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_rabitq_plus_kernel_event_wall
            .as_millis(),
        profile.rbc.gpu_subtree_sq_wmma_quantize_wall.as_millis(),
        profile.rbc.gpu_subtree_sq_wmma_h2d_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_sq_wmma_kernel_event_wall
            .as_millis(),
        profile.rbc.gpu_subtree_turboquant_quantize_wall.as_millis(),
        profile.rbc.gpu_subtree_turboquant_read_wall.as_millis(),
        profile.rbc.gpu_subtree_turboquant_h2d_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_turboquant_kernel_event_wall
            .as_millis(),
        profile.rbc.gpu_subtree_vector_h2d_wall.as_millis(),
        profile
            .rbc
            .gpu_subtree_persistent_kernel_prepare_wall
            .as_millis(),
        profile
            .rbc
            .gpu_subtree_persistent_kernel_event_wall
            .as_millis(),
        profile.rbc.gpu_subtree_persistent_total_wall.as_millis(),
        profile.rbc.gpu_subtree_cpu_result_scatter_wall.as_millis(),
        profile.rbc.gpu_subtree_cpu_compact_flush_wall.as_millis(),
        profile.leaf.edge_sink_wall.as_millis(),
        profile.reduce_wall.as_millis(),
        profile.emit_wall.as_millis(),
        profile.total_wall.as_millis(),
        profile.total_wall.as_millis(),
    )?;
    writeln!(
        file,
        "  \"gpu_subtree_io\": {{ \"raw_vector_cache_enabled\": {}, \"raw_vector_cache_budget_bytes\": {}, \"host_point_read_bytes\": {}, \"host_point_read_range_calls\": {}, \"host_point_read_range_rows\": {}, \"host_point_read_point_calls\": {}, \"raw_cache_hits\": {}, \"raw_cache_misses\": {}, \"raw_cache_hit_bytes\": {}, \"raw_cache_miss_bytes\": {}, \"raw_cache_hit_ratio\": {:.6}, \"vector_reuse_hits\": {}, \"vector_reuse_misses\": {}, \"vector_reuse_hit_bytes\": {}, \"vector_reuse_miss_bytes\": {}, \"vector_reuse_hit_ratio\": {:.6}, \"gpu_h2d_bytes\": {}, \"gpu_d2h_bytes\": {}, \"candidate_spill_bytes\": {}, \"candidate_spill_files\": {}, \"reduce_segments_scanned\": {}, \"reduce_part_files_opened\": {}, \"reduce_logical_read_bytes\": {}, \"reduce_reader_open_calls\": {}, \"reduce_direct_read_calls\": {}, \"reduce_direct_read_bytes\": {}, \"reduce_direct_io\": {}, \"reduce_workspace_points\": {}, \"reduce_workspace_bytes\": {}, \"reduce_workspace_alloc_ms\": {}, \"reduce_read_decode_ms\": {}, \"reduce_record_read_ms\": {}, \"reduce_record_decode_ms\": {}, \"reduce_record_append_ms\": {}, \"reduce_read_decode_residual_ms\": {}, \"reduce_insert_ms\": {}, \"reduce_pipeline_wall_ms\": {}, \"reduce_pipeline_overlap_ms\": {}, \"reduce_finalize_emit_ms\": {}, \"reduce_decoder_workers\": {}, \"reduce_parallel_workers\": {}, \"reduce_parallel_batches\": {}, \"reduce_parallel_records\": {}, \"reduce_batch_peak_records\": {}, \"reduce_batch_peak_bytes\": {} }},",
        profile.raw_vector_cache_budget_bytes > 0,
        profile.raw_vector_cache_budget_bytes,
        profile.rbc.gpu_subtree_host_io_stats.bytes_read,
        profile.rbc.gpu_subtree_host_io_stats.range_calls,
        profile.rbc.gpu_subtree_host_io_stats.range_rows_read,
        profile.rbc.gpu_subtree_host_io_stats.point_calls,
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_hits,
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_misses,
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_hit_bytes,
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_miss_bytes,
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_hit_ratio(),
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_hits,
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_misses,
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_hit_bytes,
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_miss_bytes,
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_hit_ratio(),
        profile.rbc.gpu_subtree_compact_h2d_bytes,
        profile.rbc.gpu_subtree_compact_d2h_bytes,
        profile.temp_files.candidate.bytes,
        profile.temp_files.candidate.files,
        profile.reduce.segments_scanned,
        profile.reduce.part_files_opened,
        profile.reduce.logical_read_bytes,
        profile.reduce.reader_open_calls,
        profile.reduce.direct_read_calls,
        profile.reduce.direct_read_bytes,
        profile.reduce.direct_io,
        profile.reduce.workspace_points,
        profile.reduce.workspace_bytes,
        profile.reduce.workspace_alloc_wall.as_millis(),
        profile.reduce.read_decode_wall.as_millis(),
        profile.reduce.record_read_wall.as_millis(),
        profile.reduce.record_decode_wall.as_millis(),
        profile.reduce.record_append_wall.as_millis(),
        reduce_read_decode_residual.as_millis(),
        profile.reduce.insert_wall.as_millis(),
        profile.reduce.pipeline_wall.as_millis(),
        profile.reduce.pipeline_overlap_wall.as_millis(),
        profile.reduce.finalize_emit_wall.as_millis(),
        profile.reduce.decoder_workers,
        profile.reduce.parallel_workers,
        profile.reduce.parallel_batches,
        profile.reduce.parallel_records,
        profile.reduce.batch_peak_records,
        profile.reduce.batch_peak_bytes,
    )?;
    writeln!(
        file,
        "  \"gpu_subtree_counters\": {{ \"d0_child_runs\": {}, \"d0_child_points\": {}, \"d0_child_max_len\": {}, \"d1plus_admission_split_count\": {}, \"d1plus_preflight_split_count\": {}, \"d1plus_prevented_cpu_fallback_count\": {}, \"compact_attempt_count\": {}, \"compact_success_count\": {}, \"compact_batches\": {}, \"compact_rows\": {}, \"persistent_attempt_count\": {}, \"persistent_success_count\": {}, \"persistent_fallback_count\": {}, \"persistent_leaf_count\": {}, \"persistent_topk_rows\": {}, \"persistent_partition_frontier_end\": {}, \"persistent_partition_level_rounds\": {}, \"persistent_queue_overflow_flags\": {}, \"cpu_fallbacks\": {} }},",
        profile.rbc.gpu_subtree_d0_child_run_count,
        profile.rbc.gpu_subtree_d0_child_points,
        profile.rbc.gpu_subtree_d0_child_max_len,
        profile.rbc.gpu_subtree_d1plus_admission_split_count,
        profile.rbc.gpu_subtree_d1plus_preflight_split_count,
        profile.rbc.gpu_subtree_d1plus_prevented_cpu_fallback_count,
        profile.rbc.gpu_subtree_compact_attempt_count,
        profile.rbc.gpu_subtree_compact_success_count,
        profile.rbc.gpu_subtree_compact_batches,
        profile.rbc.gpu_subtree_compact_rows,
        profile.rbc.gpu_subtree_persistent_attempt_count,
        profile.rbc.gpu_subtree_persistent_success_count,
        profile.rbc.gpu_subtree_persistent_fallback_count,
        profile.rbc.gpu_subtree_persistent_leaf_count,
        profile.rbc.gpu_subtree_persistent_topk_rows,
        profile.rbc.gpu_subtree_persistent_partition_frontier_end,
        profile.rbc.gpu_subtree_persistent_partition_level_rounds,
        profile.rbc.gpu_subtree_persistent_queue_overflow_flags,
        profile.rbc.gpu_subtree_cpu_fallbacks,
    )?;
    let gpu_subtree_enabled = matches!(profile.partitioner, PiPNNPartitionerKind::GpuSubtree);
    let root_assignment_backend = if gpu_subtree_enabled {
        profile.root_assignment_backend.as_str()
    } else {
        PiPNNComputeBackendKind::Cpu.as_str()
    };
    let subtree_assignment_backend = if gpu_subtree_enabled {
        profile.assignment_backend.as_str()
    } else {
        PiPNNComputeBackendKind::Cpu.as_str()
    };
    let mut gpu_subtree_map = serde_json::Map::new();
    fn insert_json_value<T: serde::Serialize>(
        map: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        value: T,
    ) {
        map.insert(
            key.to_string(),
            serde_json::to_value(value).expect("OOM profile value should serialize"),
        );
    }
    fn serde_json_u128_value(value: u128) -> serde_json::Value {
        if value <= u64::MAX as u128 {
            serde_json::Value::Number(serde_json::Number::from(value as u64))
        } else {
            serde_json::Value::String(value.to_string())
        }
    }
    fn insert_json_u128(
        map: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        value: u128,
    ) {
        map.insert(key.to_string(), serde_json_u128_value(value));
    }
    fn insert_json_option_u128(
        map: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        value: Option<u128>,
    ) {
        map.insert(
            key.to_string(),
            value.map_or(serde_json::Value::Null, serde_json_u128_value),
        );
    }
    fn ratio_or_zero(numerator: usize, denominator: usize) -> f64 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f64 / denominator as f64
        }
    }
    fn residual_duration(total: Duration, parts: &[Duration]) -> Duration {
        parts
            .iter()
            .fold(total, |remaining, part| remaining.saturating_sub(*part))
    }
    insert_json_value(&mut gpu_subtree_map, "enabled", gpu_subtree_enabled);
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_lazy_split_mode",
        profile.d1plus_lazy_split_mode.as_str(),
    );
    let d0_partition_mode = if profile.rbc.gpu_subtree_d0_partition_mode.is_empty() {
        "rbc-fanout"
    } else {
        profile.rbc.gpu_subtree_d0_partition_mode
    };
    insert_json_value(&mut gpu_subtree_map, "d0_partition_mode", d0_partition_mode);
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_tagore_partitions",
        profile.rbc.gpu_subtree_d0_tagore_partitions,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_tagore_k",
        profile.rbc.gpu_subtree_d0_tagore_k,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_tagore_sample_points",
        profile.rbc.gpu_subtree_d0_tagore_sample_points,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_tagore_train_ms",
        profile.rbc.gpu_subtree_d0_tagore_train_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_tagore_assign_ms",
        profile.rbc.gpu_subtree_d0_tagore_assign_wall.as_millis(),
    );
    let d1_root_partition_mode = if profile.rbc.gpu_subtree_d1_root_partition_mode.is_empty() {
        "off"
    } else {
        profile.rbc.gpu_subtree_d1_root_partition_mode
    };
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_partition_mode",
        d1_root_partition_mode,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_enabled",
        d1_root_partition_mode == "kmeans-coarse",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_child_count",
        profile.rbc.gpu_subtree_d1_root_kmeans_child_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_centroids_total",
        profile.rbc.gpu_subtree_d1_root_kmeans_centroids_total,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_train_ms",
        profile
            .rbc
            .gpu_subtree_d1_root_kmeans_train_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_assign_ms",
        profile
            .rbc
            .gpu_subtree_d1_root_kmeans_assign_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_input_rows",
        profile.rbc.gpu_subtree_d1_root_kmeans_input_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_output_rows",
        profile.rbc.gpu_subtree_d1_root_kmeans_output_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1_root_kmeans_max_child_rows",
        profile.rbc.gpu_subtree_d1_root_kmeans_max_child_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "root_assignment_backend",
        root_assignment_backend,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_backend",
        subtree_assignment_backend,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "subtree_assignment_backend",
        subtree_assignment_backend,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_topk_backend",
        profile.leaf_topk_backend.as_str(),
    );
    let d0_dco = &profile.rbc.gpu_subtree_d0_dco;
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_mode",
        profile.d0_dco_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_rerank_mode",
        profile.d0_dco_rerank_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_survivors",
        profile.d0_dco_survivors,
    );
    insert_json_value(&mut gpu_subtree_map, "d0_dco_audit", profile.d0_dco_audit);
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_sq_cpu_kernel",
        profile.d0_sq_cpu_kernel.as_str(),
    );
    insert_json_value(&mut gpu_subtree_map, "d0_dco_nodes", d0_dco.nodes);
    insert_json_value(&mut gpu_subtree_map, "d0_dco_rows", d0_dco.rows);
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_leader_pairs",
        d0_dco.leader_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_prefilter_pairs",
        d0_dco.prefilter_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_pruned_pairs",
        d0_dco.pruned_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_survivor_pairs",
        d0_dco.survivor_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_exact_pairs",
        d0_dco.exact_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_approx_output_pairs",
        d0_dco.approx_output_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_false_prune_audit_pairs",
        d0_dco.false_prune_audit_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_false_prune_audit_misses",
        d0_dco.false_prune_audit_misses,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_sidecar_gather_ms",
        profile
            .rbc
            .gpu_subtree_d0_dco_sidecar_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_approx_ms",
        profile.rbc.gpu_subtree_d0_dco_approx_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_dco_exact_ms",
        profile.rbc.gpu_subtree_d0_dco_exact_wall.as_millis(),
    );
    let assignment_dco = &profile.rbc.gpu_subtree_assignment_dco;
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_mode",
        profile.assignment_dco_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_rerank_mode",
        profile.assignment_dco_rerank_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_proj_dim",
        profile.assignment_dco_proj_dim,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_survivors",
        profile.assignment_dco_survivors,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_audit",
        profile.assignment_dco_audit,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_nodes",
        assignment_dco.nodes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_rows",
        assignment_dco.rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_leader_pairs",
        assignment_dco.leader_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_prefilter_pairs",
        assignment_dco.prefilter_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_pruned_pairs",
        assignment_dco.pruned_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_survivor_pairs",
        assignment_dco.survivor_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_exact_pairs",
        assignment_dco.exact_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_approx_output_pairs",
        assignment_dco.approx_output_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_fallback_nodes",
        assignment_dco.fallback_nodes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_overflow_nodes",
        assignment_dco.overflow_nodes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_false_prune_audit_pairs",
        assignment_dco.false_prune_audit_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_false_prune_audit_misses",
        assignment_dco.false_prune_audit_misses,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_metadata_bytes",
        assignment_dco.metadata_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_dco_metadata_h2d_ms",
        assignment_dco.metadata_h2d_wall.as_millis(),
    );
    let leaf_dco = &profile.rbc.gpu_subtree_leaf_dco;
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_mode",
        profile.leaf_dco_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_rerank_mode",
        profile.leaf_dco_rerank_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_proj_dim",
        profile.leaf_dco_proj_dim,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_survivors",
        profile.leaf_dco_survivors,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_audit",
        profile.leaf_dco_audit,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_ddc_res_pca_matrix",
        profile
            .leaf_dco_ddc_res_pca_matrix
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_ddc_res_projected_input",
        profile.leaf_dco_ddc_res_projected_input,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_ddc_res_base_dim",
        profile.leaf_dco_ddc_res_base_dim,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_ddc_res_sigma_count",
        profile.leaf_dco_ddc_res_sigma_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "qcode_staging_mode",
        profile.qcode_staging_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_staging_mode",
        profile.sq_staging_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "qcode_d1_payload_mode",
        profile.qcode_d1_payload_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_d1_distance_mode",
        profile.qcode_rabitq_d1_distance_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_short_survivors",
        profile.qcode_rabitq_short_survivors,
    );
    let mut rabitq_residual_map = serde_json::Map::new();
    insert_json_value(
        &mut rabitq_residual_map,
        "enabled",
        profile.qcode_staging_mode == PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual
            || profile.rbc.gpu_subtree_qcode_staging_enabled,
    );
    insert_json_value(&mut rabitq_residual_map, "experimental", true);
    insert_json_value(
        &mut rabitq_residual_map,
        "mode",
        profile.qcode_staging_mode.as_str(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d1_payload_mode",
        if profile.rbc.gpu_subtree_qcode_d1_payload_mode.is_empty() {
            profile.qcode_d1_payload_mode.as_str()
        } else {
            profile.rbc.gpu_subtree_qcode_d1_payload_mode
        },
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d1_distance_mode",
        profile.qcode_rabitq_d1_distance_mode.as_str(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "short_survivors",
        profile.qcode_rabitq_short_survivors,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "centroids",
        profile
            .rabitq_residual_centroids
            .max(profile.rbc.gpu_subtree_qcode_rabitq_residual_centroids),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "upload_shard_rows",
        profile
            .qcode_staging_upload_shard_rows
            .max(profile.rbc.gpu_subtree_qcode_staging_upload_shard_rows),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "global_d0_leaders",
        profile
            .qcode_global_d0_leaders
            .max(profile.rbc.gpu_subtree_qcode_global_d0_leaders),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "distance_formula",
        match profile.qcode_rabitq_d1_distance_mode {
            PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma => {
                "experimental_reconstructed_query_to_resident_code_residual_rabitq"
            }
            PiPNNGpuSubtreeRabitqD1DistanceMode::SplitShortEx => {
                "experimental_split_short_ex_code_code_residual_rabitq"
            }
        },
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "rotator",
        profile.rabitq_plus_rotator.as_str(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "rotator_seed",
        profile.rabitq_plus_seed,
    );
    insert_json_value(&mut rabitq_residual_map, "rotated_dim", profile.dim);
    insert_json_value(
        &mut rabitq_residual_map,
        "centroid_train_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_centroid_train_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "centroid_assign_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_centroid_assign_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "gpu_rabitq_quantize_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_gpu_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "gpu_rabitq_d0_assign_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_gpu_d0_assign_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "leader_prepare_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_leader_prepare_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "leader_payload_h2d_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_leader_payload_h2d_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "assignment_fragment_reduce_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_assignment_fragment_reduce_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "qcode_shard_write_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_qcode_shard_write_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "qcode_child_gather_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_qcode_child_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "global_rabitq_payload_upload_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_global_payload_upload_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d1plus_child_id_h2d_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_d1plus_child_id_h2d_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "host_lazy_split_count",
        profile
            .rbc
            .gpu_subtree_rabitq_residual_host_lazy_split_count,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_kernel_mode",
        profile.rbc.gpu_subtree_rabitq_d0_assign_kernel_mode,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_rows",
        profile.rbc.gpu_subtree_rabitq_d0_assign_rows,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_leaders",
        profile.rbc.gpu_subtree_rabitq_d0_assign_leaders,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_pair_count",
        profile.rbc.gpu_subtree_rabitq_d0_assign_pair_count,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_fanout",
        profile.rbc.gpu_subtree_rabitq_d0_assign_fanout,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_row_tile",
        profile.rbc.gpu_subtree_rabitq_d0_assign_row_tile,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_row_chunk",
        profile.rbc.gpu_subtree_rabitq_d0_assign_row_chunk,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_leader_tile",
        profile.rbc.gpu_subtree_rabitq_d0_assign_leader_tile,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_partial_r",
        profile.rbc.gpu_subtree_rabitq_d0_assign_partial_r,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_leader_tile_count",
        profile.rbc.gpu_subtree_rabitq_d0_assign_leader_tile_count,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_partial_bytes",
        profile.rbc.gpu_subtree_rabitq_d0_assign_partial_bytes,
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_distance_kernel_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_d0_assign_distance_kernel_wall
            .as_millis(),
    );
    insert_json_value(
        &mut rabitq_residual_map,
        "d0_assign_merge_kernel_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_d0_assign_merge_kernel_wall
            .as_millis(),
    );
    insert_json_value(&mut gpu_subtree_map, "rabitq_residual", rabitq_residual_map);
    let sq_gpu_staging_total_ms = (profile.rbc.gpu_subtree_sq_staging_raw_read_wall
        + profile.rbc.gpu_subtree_sq_staging_raw_h2d_wall
        + profile.rbc.gpu_subtree_sq_staging_gpu_quantize_wall
        + profile
            .rbc
            .gpu_subtree_sq_staging_host_rotated_quantize_wall
        + profile.rbc.gpu_subtree_sq_staging_qcode_d2h_wall
        + profile.rbc.gpu_subtree_sq_staging_gpu_d0_assign_wall
        + profile
            .rbc
            .gpu_subtree_sq_staging_assignment_fragment_d2h_wall
        + profile.rbc.gpu_subtree_sq_staging_leader_prepare_wall
        + profile.rbc.gpu_subtree_sq_staging_leader_h2d_wall
        + profile.rbc.gpu_subtree_sq_staging_leader_quantize_wall
        + profile
            .rbc
            .gpu_subtree_sq_staging_assignment_fragment_reduce_wall
        + profile.rbc.gpu_subtree_sq_staging_sidecar_write_wall)
        .as_millis();
    let mut sq_gpu_staging_map = serde_json::Map::new();
    insert_json_value(
        &mut sq_gpu_staging_map,
        "enabled",
        profile.sq_staging_mode == PiPNNGpuSubtreeSqStagingMode::GpuD0
            || profile.rbc.gpu_subtree_sq_staging_enabled,
    );
    insert_json_value(&mut sq_gpu_staging_map, "experimental", true);
    insert_json_value(
        &mut sq_gpu_staging_map,
        "mode",
        if profile.rbc.gpu_subtree_sq_staging_mode.is_empty() {
            profile.sq_staging_mode.as_str()
        } else {
            profile.rbc.gpu_subtree_sq_staging_mode
        },
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "pipeline_mode",
        profile.rbc.gpu_subtree_sq_staging_pipeline_mode,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "wall_ms",
        profile.rbc.gpu_subtree_d0_wall.as_millis(),
    );
    insert_json_value(&mut sq_gpu_staging_map, "bits", SQ_INT8_DEFAULT_BITS);
    insert_json_value(
        &mut sq_gpu_staging_map,
        "upload_shard_rows",
        profile
            .sq_staging_upload_shard_rows
            .max(profile.rbc.gpu_subtree_sq_staging_upload_shard_rows),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "rows",
        profile.rbc.gpu_subtree_sq_staging_rows,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "dim",
        profile.rbc.gpu_subtree_sq_staging_dim,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "padded_dim",
        profile.rbc.gpu_subtree_sq_staging_padded_dim,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "shards",
        profile.rbc.gpu_subtree_sq_staging_shard_count,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "shard_count",
        profile.rbc.gpu_subtree_sq_staging_shard_count,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "global_d0_leaders",
        profile.rbc.gpu_subtree_sq_staging_global_d0_leaders,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "leader_count",
        profile.rbc.gpu_subtree_sq_staging_global_d0_leaders,
    );
    let rotated_residual_d0 = profile
        .rbc
        .gpu_subtree_sq_d0_assign_kernel_mode
        .contains("rotated-residual");
    insert_json_value(
        &mut sq_gpu_staging_map,
        "distance_formula",
        if rotated_residual_d0 {
            "sq8_rotated_residual_reconstructed_l2"
        } else {
            "sq8_approx_l2_norms_scales_dot"
        },
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_distance",
        if rotated_residual_d0 {
            "sq8-residual-plus-center"
        } else {
            "sq8-code-code"
        },
    );
    insert_json_value(&mut sq_gpu_staging_map, "d0_leader_scope", "global");
    insert_json_value(
        &mut sq_gpu_staging_map,
        "sidecar_source",
        profile.rbc.gpu_subtree_sq_staging_sidecar_source,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d1_distance",
        match profile.rbc.gpu_subtree_sq_staging_sidecar_source {
            "rotated-residual-v2-int8" => "sq8-residual-center-int8",
            "rotated-residual-v2" => "reconstructed-fp32",
            "none-rotated-residual" => "raw-fp32",
            _ => "sq8-int8",
        },
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "sidecar_compatible",
        profile.rbc.gpu_subtree_sq_staging_sidecar_compatible,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "raw_read_ms",
        profile.rbc.gpu_subtree_sq_staging_raw_read_wall.as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "raw_h2d_ms",
        profile.rbc.gpu_subtree_sq_staging_raw_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "gpu_sq_quantize_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_gpu_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "host_rotated_quantize_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_host_rotated_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "quantize_wall_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_gpu_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "quantize_kernel_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_gpu_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "gpu_sq_d0_assign_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_gpu_d0_assign_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_wall_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_gpu_d0_assign_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "leader_prepare_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_leader_prepare_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "leader_quantize_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_leader_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "leader_h2d_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_leader_h2d_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "leader_code_bytes",
        profile.rbc.gpu_subtree_sq_staging_leader_code_bytes,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "qcode_d2h_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_qcode_d2h_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "assignment_fragment_count",
        profile
            .rbc
            .gpu_subtree_sq_d0_assign_rows
            .saturating_mul(profile.rbc.gpu_subtree_sq_d0_assign_fanout),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "assignment_fragment_d2h_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_assignment_fragment_d2h_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "assignment_fragment_reduce_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_assignment_fragment_reduce_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "sidecar_write_ms",
        profile
            .rbc
            .gpu_subtree_sq_staging_sidecar_write_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_child_count",
        profile.rbc.gpu_subtree_d0_child_run_count,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_child_max_rows",
        profile.rbc.gpu_subtree_d0_child_max_len,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_child_p50_rows",
        profile.rbc.gpu_subtree_d0_child_p50_len,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_child_p95_rows",
        profile.rbc.gpu_subtree_d0_child_p95_len,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_child_p99_rows",
        profile.rbc.gpu_subtree_d0_child_p99_len,
    );
    insert_json_value(&mut sq_gpu_staging_map, "total_ms", sq_gpu_staging_total_ms);
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_kernel_requested",
        profile.rbc.gpu_subtree_sq_d0_assign_kernel_requested,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_kernel_mode",
        profile.rbc.gpu_subtree_sq_d0_assign_kernel_mode,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_rows",
        profile.rbc.gpu_subtree_sq_d0_assign_rows,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_leaders",
        profile.rbc.gpu_subtree_sq_d0_assign_leaders,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_pair_count",
        profile.rbc.gpu_subtree_sq_d0_assign_pair_count,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_fanout",
        profile.rbc.gpu_subtree_sq_d0_assign_fanout,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_row_tile",
        profile.rbc.gpu_subtree_sq_d0_assign_row_tile,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_row_chunk",
        profile.rbc.gpu_subtree_sq_d0_assign_row_chunk,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_leader_tile",
        profile.rbc.gpu_subtree_sq_d0_assign_leader_tile,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_partial_r",
        profile.rbc.gpu_subtree_sq_d0_assign_partial_r,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_leader_tile_count",
        profile.rbc.gpu_subtree_sq_d0_assign_leader_tile_count,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_partial_bytes",
        profile.rbc.gpu_subtree_sq_d0_assign_partial_bytes,
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_distance_kernel_ms",
        profile
            .rbc
            .gpu_subtree_sq_d0_assign_distance_kernel_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_kernel_ms",
        profile
            .rbc
            .gpu_subtree_sq_d0_assign_distance_kernel_wall
            .as_millis(),
    );
    insert_json_value(
        &mut sq_gpu_staging_map,
        "d0_assign_merge_kernel_ms",
        profile
            .rbc
            .gpu_subtree_sq_d0_assign_merge_kernel_wall
            .as_millis(),
    );
    insert_json_value(&mut gpu_subtree_map, "sq_gpu_staging", sq_gpu_staging_map);
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_mode",
        profile.rabitq_plus_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_bits",
        profile.rabitq_plus_bits,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_rotator",
        profile.rabitq_plus_rotator.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_seed",
        profile.rabitq_plus_seed,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_mode",
        profile.rabitq_plus_sidecar_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_prefix",
        profile
            .rabitq_plus_sidecar_prefix
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_sketch_enabled",
        profile.rabitq_plus_fused_sidecar_sketch_enabled,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_prestage_total_ms",
        profile.rabitq_plus_fused_prestage_total_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_chunk_alloc_ms",
        profile.rabitq_plus_fused_chunk_alloc_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_raw_read_ms",
        profile.rabitq_plus_fused_sidecar_raw_read_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_source_hash_ms",
        profile.rabitq_plus_fused_source_hash_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_quantize_ms",
        profile.rabitq_plus_fused_sidecar_quantize_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_code_output_ms",
        profile
            .rabitq_plus_fused_sidecar_code_output_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_norm_ms",
        profile.rabitq_plus_fused_sidecar_norm_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_meta_output_ms",
        profile
            .rabitq_plus_fused_sidecar_meta_output_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_header_write_ms",
        profile
            .rabitq_plus_fused_sidecar_header_write_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sidecar_write_ms",
        profile.rabitq_plus_fused_sidecar_write_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sketch_alloc_ms",
        profile.rabitq_plus_fused_sketch_alloc_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sketch_compute_ms",
        profile.rabitq_plus_fused_sketch_compute_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sketch_extend_ms",
        profile.rabitq_plus_fused_sketch_extend_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_sketch_write_ms",
        profile.rabitq_plus_fused_sketch_write_wall.as_millis(),
    );
    let fused_unattributed_wall = residual_duration(
        profile.rabitq_plus_fused_prestage_total_wall,
        &[
            profile.rabitq_plus_fused_chunk_alloc_wall,
            profile.rabitq_plus_fused_sidecar_raw_read_wall,
            profile.rabitq_plus_fused_source_hash_wall,
            profile.rabitq_plus_fused_sidecar_quantize_wall,
            profile.rabitq_plus_fused_sidecar_code_output_wall,
            profile.rabitq_plus_fused_sidecar_norm_wall,
            profile.rabitq_plus_fused_sidecar_meta_output_wall,
            profile.rabitq_plus_fused_sidecar_header_write_wall,
            profile.rabitq_plus_fused_sketch_alloc_wall,
            profile.rabitq_plus_fused_sketch_compute_wall,
            profile.rabitq_plus_fused_sketch_extend_wall,
            profile.rabitq_plus_fused_sketch_write_wall,
        ],
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_fused_unattributed_ms",
        fused_unattributed_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_format",
        "dense_i8_codes_meta",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_distance_formula",
        "dense_i8_delta_code_sum_norm",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_split_sidecar_format",
        "library_split_batch_ex_meta",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_split_distance_formula",
        "rabitq_library_split_distance_boosting",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_rows",
        profile.rbc.gpu_subtree_rabitq_plus_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_padded_dim",
        profile.rbc.gpu_subtree_rabitq_plus_padded_dim,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_code_bytes",
        profile.rbc.gpu_subtree_rabitq_plus_code_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_quantize_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_quantize_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_read_ms",
        profile.rbc.gpu_subtree_rabitq_plus_read_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_h2d_ms",
        profile.rbc.gpu_subtree_rabitq_plus_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_rows",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_codes_bytes",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_codes_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_meta_bytes",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_meta_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_gather_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_sidecar_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_read_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_sidecar_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_resident",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_resident,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_preload_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_sidecar_preload_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_resident_bytes",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_resident_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_resident_rows",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_resident_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_file_rows",
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_file_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_resident_gather_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_sidecar_resident_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_sidecar_file_gather_ms",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_sidecar_file_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_mode",
        profile.sq_wmma_mode.as_str(),
    );
    let persistent_large_ticket_batch_actual =
        profile.rbc.gpu_subtree_persistent_large_ticket_batch_actual;
    let persistent_sq8_assignment_kernel_actual = match (
        profile.rbc.gpu_subtree_persistent_large_sq8_dp4a_tiles > 0,
        profile.rbc.gpu_subtree_persistent_large_sq8_wmma_tiles > 0,
    ) {
        (true, true) => "mixed",
        (true, false) => "warp-dp4a",
        (false, true) if profile.sq_native_int4 => "int4-wmma",
        (false, true) => "wmma",
        (false, false) => "none",
    };
    let persistent_sq8_leaf_executor_actual = match (
        profile.rbc.gpu_subtree_persistent_leaf_sq8_dp4a_tiles > 0,
        profile.rbc.gpu_subtree_persistent_leaf_sq8_wmma_tiles > 0,
    ) {
        (true, true) => "mixed",
        (true, false) => "warp-dp4a",
        (false, true) => "wmma",
        (false, false) => "none",
    };
    let persistent_sq8_leaf_mode_actual = if profile
        .rbc
        .gpu_subtree_persistent_leaf_row_owned_tasks
        .saturating_add(profile.rbc.gpu_subtree_persistent_leaf_staged_tasks)
        == 0
    {
        "none"
    } else {
        match profile.persistent_sq8_leaf_mode_requested {
            PiPNNGpuSubtreeSq8LeafMode::Staged => "staged",
            PiPNNGpuSubtreeSq8LeafMode::RowOwned => "row-owned",
            PiPNNGpuSubtreeSq8LeafMode::D96K2 => "d96-k2",
            PiPNNGpuSubtreeSq8LeafMode::Auto
                if profile.sq_wmma_mode == PiPNNGpuSubtreeSqWmmaMode::Build
                    && profile.sq_bits == 8
                    && !profile.sq_native_int4
                    && profile.dim <= 256
                    && profile.leaf_knn <= 4 =>
            {
                "row-owned"
            }
            PiPNNGpuSubtreeSq8LeafMode::Auto => "staged",
        }
    };
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_ticket_batch_requested",
        profile.persistent_large_ticket_batch_requested.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_ticket_batch_actual",
        persistent_large_ticket_batch_actual,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_sq8_assignment_kernel_requested",
        profile.persistent_sq8_assignment_kernel_requested.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_sq8_assignment_kernel_actual",
        persistent_sq8_assignment_kernel_actual,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_sq8_leaf_executor_requested",
        profile.persistent_sq8_leaf_executor_requested.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_sq8_leaf_executor_actual",
        persistent_sq8_leaf_executor_actual,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_sq8_leaf_mode_requested",
        profile.persistent_sq8_leaf_mode_requested.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_sq8_leaf_mode_actual",
        persistent_sq8_leaf_mode_actual,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_scheduler_mode_requested",
        profile.persistent_scheduler_mode_requested.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_scheduler_mode_actual",
        profile.persistent_scheduler_mode_actual.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_idle_backoff_ns_requested",
        profile.persistent_idle_backoff_ns_requested,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_idle_backoff_ns_actual",
        profile.persistent_idle_backoff_ns_actual,
    );
    insert_json_value(&mut gpu_subtree_map, "sq_bits", profile.sq_bits);
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_native_int4",
        profile.sq_native_int4,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_mode",
        profile.sq_wmma_sidecar_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_prefix",
        profile
            .sq_wmma_sidecar_prefix
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_staging_fused_sketch_enabled",
        profile.sq_staging_fused_sketch_enabled,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_staging_fused_sketch_compute_ms",
        profile.sq_staging_fused_sketch_compute_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_staging_fused_sketch_join_wait_ms",
        profile.sq_staging_fused_sketch_join_wait_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_mode",
        profile.turboquant_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_bits",
        profile.turboquant_bits,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_seed",
        profile.turboquant_seed,
    );
    insert_json_value(&mut gpu_subtree_map, "turboquant_rotator", "dense-qr");
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_mode",
        profile.turboquant_sidecar_mode.as_str(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_prefix",
        profile
            .turboquant_sidecar_prefix
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_format",
        "tq4_packed_codes_norm_meta",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_distance_formula",
        "norm_i2_norm_j2_minus_2_norms_lut_dot",
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_rows",
        profile.rbc.gpu_subtree_sq_wmma_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_padded_dim",
        profile.rbc.gpu_subtree_sq_wmma_padded_dim,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_code_bytes",
        profile.rbc.gpu_subtree_sq_wmma_code_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_quantize_ms",
        profile.rbc.gpu_subtree_sq_wmma_quantize_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_h2d_ms",
        profile.rbc.gpu_subtree_sq_wmma_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_sq_wmma_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_rows",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_codes_bytes",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_codes_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_meta_bytes",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_meta_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_gather_ms",
        profile
            .rbc
            .gpu_subtree_sq_wmma_sidecar_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_resident",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_preload_ms",
        profile
            .rbc
            .gpu_subtree_sq_wmma_sidecar_preload_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_resident_bytes",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_resident_rows",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_file_rows",
        profile.rbc.gpu_subtree_sq_wmma_sidecar_file_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_resident_gather_ms",
        profile
            .rbc
            .gpu_subtree_sq_wmma_sidecar_resident_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sq_wmma_sidecar_file_gather_ms",
        profile
            .rbc
            .gpu_subtree_sq_wmma_sidecar_file_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_rows",
        profile.rbc.gpu_subtree_turboquant_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_padded_dim",
        profile.rbc.gpu_subtree_turboquant_padded_dim,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_code_bytes",
        profile.rbc.gpu_subtree_turboquant_code_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_quantize_ms",
        profile.rbc.gpu_subtree_turboquant_quantize_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_read_ms",
        profile.rbc.gpu_subtree_turboquant_read_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_h2d_ms",
        profile.rbc.gpu_subtree_turboquant_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_rows",
        profile.rbc.gpu_subtree_turboquant_sidecar_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_codes_bytes",
        profile.rbc.gpu_subtree_turboquant_sidecar_codes_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_meta_bytes",
        profile.rbc.gpu_subtree_turboquant_sidecar_meta_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_gather_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_sidecar_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_read_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_sidecar_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_resident",
        profile.rbc.gpu_subtree_turboquant_sidecar_resident,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_preload_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_sidecar_preload_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_resident_bytes",
        profile.rbc.gpu_subtree_turboquant_sidecar_resident_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_resident_rows",
        profile.rbc.gpu_subtree_turboquant_sidecar_resident_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_file_rows",
        profile.rbc.gpu_subtree_turboquant_sidecar_file_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_resident_gather_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_sidecar_resident_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_sidecar_file_gather_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_sidecar_file_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_shadow_mismatch_rows",
        profile.rbc.gpu_subtree_turboquant_shadow_mismatch_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_shadow_audit_rows",
        profile.rbc.gpu_subtree_turboquant_shadow_audit_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_shadow_audit_pairs",
        profile.rbc.gpu_subtree_turboquant_shadow_audit_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_shadow_overlap_pairs",
        profile.rbc.gpu_subtree_turboquant_shadow_overlap_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_shadow_topk_overlap_x1000",
        profile.rbc.gpu_subtree_turboquant_shadow_topk_overlap_x1000,
    );
    let mut turboquant_cascade_map = serde_json::Map::new();
    let turboquant_cascade_low_dim = if profile.rbc.gpu_subtree_turboquant_cascade_low_dim == 0 {
        profile.turboquant_cascade_low_dim
    } else {
        profile.rbc.gpu_subtree_turboquant_cascade_low_dim
    };
    let turboquant_cascade_survivor_cap =
        if profile.rbc.gpu_subtree_turboquant_cascade_survivor_cap == 0 {
            profile.turboquant_cascade_survivor_cap
        } else {
            profile.rbc.gpu_subtree_turboquant_cascade_survivor_cap
        };
    let turboquant_adaptive_survivor_threshold = if profile
        .rbc
        .gpu_subtree_turboquant_adaptive_survivor_threshold
        == 0
    {
        profile.turboquant_adaptive_survivor_threshold
    } else {
        profile
            .rbc
            .gpu_subtree_turboquant_adaptive_survivor_threshold
    };
    let turboquant_adaptive_survivor_low_cap =
        if profile.rbc.gpu_subtree_turboquant_adaptive_survivor_low_cap == 0 {
            profile.turboquant_adaptive_survivor_low_cap
        } else {
            profile.rbc.gpu_subtree_turboquant_adaptive_survivor_low_cap
        };
    insert_json_value(
        &mut turboquant_cascade_map,
        "shadow_rows_sampled",
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_shadow_rows_sampled,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "low_dim",
        turboquant_cascade_low_dim,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "survivor_cap",
        turboquant_cascade_survivor_cap,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "adaptive_survivor_threshold",
        turboquant_adaptive_survivor_threshold,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "adaptive_survivor_low_cap",
        turboquant_adaptive_survivor_low_cap,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "no_rerank_overlap_at_k",
        ratio_or_zero(
            profile
                .rbc
                .gpu_subtree_turboquant_cascade_no_rerank_overlap_pairs,
            profile.rbc.gpu_subtree_turboquant_cascade_audit_pairs_at_k,
        ),
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "overlap_at_k",
        ratio_or_zero(
            profile
                .rbc
                .gpu_subtree_turboquant_cascade_overlap_pairs_at_k,
            profile.rbc.gpu_subtree_turboquant_cascade_audit_pairs_at_k,
        ),
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "overlap_at_survivor_cap",
        ratio_or_zero(
            profile
                .rbc
                .gpu_subtree_turboquant_cascade_survivor_overlap_pairs,
            profile
                .rbc
                .gpu_subtree_turboquant_cascade_survivor_audit_pairs,
        ),
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "mismatch_rows",
        profile.rbc.gpu_subtree_turboquant_cascade_mismatch_rows,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "cpu_shadow_ms",
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_cpu_shadow_wall
            .as_millis(),
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "low_dim_pairs",
        profile.rbc.gpu_subtree_turboquant_cascade_low_dim_pairs,
    );
    insert_json_value(
        &mut turboquant_cascade_map,
        "estimated_full_pairs_saved",
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_estimated_full_pairs_saved,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "turboquant_cascade",
        turboquant_cascade_map,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "raw_resident_h2d_bytes_saved_estimate",
        profile
            .rbc
            .gpu_subtree_raw_resident_h2d_bytes_saved_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_shadow_mismatch_rows",
        profile.rbc.gpu_subtree_rabitq_plus_shadow_mismatch_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_shadow_audit_rows",
        profile.rbc.gpu_subtree_rabitq_plus_shadow_audit_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_shadow_audit_pairs",
        profile.rbc.gpu_subtree_rabitq_plus_shadow_audit_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_shadow_overlap_pairs",
        profile.rbc.gpu_subtree_rabitq_plus_shadow_overlap_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "rabitq_plus_shadow_topk_overlap_x1000",
        profile
            .rbc
            .gpu_subtree_rabitq_plus_shadow_topk_overlap_x1000,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_leaf_count",
        leaf_dco.leaf_count,
    );
    insert_json_value(&mut gpu_subtree_map, "leaf_dco_rows", leaf_dco.rows);
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_pair_count",
        leaf_dco.pair_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_prefilter_pairs",
        leaf_dco.prefilter_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_pruned_pairs",
        leaf_dco.pruned_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_survivor_pairs",
        leaf_dco.survivor_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_exact_pairs",
        leaf_dco.exact_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_exact_stamp_pairs",
        leaf_dco.exact_stamp_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_approx_output_pairs",
        leaf_dco.approx_output_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tile_prefilter_tiles",
        leaf_dco.tile_prefilter_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tile_pruned_tiles",
        leaf_dco.tile_pruned_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tile_partial_tiles",
        leaf_dco.tile_partial_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tile_full_exact_tiles",
        leaf_dco.tile_full_exact_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_exact_fallback_rows",
        leaf_dco.exact_fallback_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_survivor_overflow_rows",
        leaf_dco.survivor_overflow_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_audit_only_pairs",
        leaf_dco.audit_only_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_false_prune_audit_pairs",
        leaf_dco.false_prune_audit_pairs,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_false_prune_audit_misses",
        leaf_dco.false_prune_audit_misses,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_metadata_bytes",
        leaf_dco.metadata_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_metadata_prepare_ms",
        leaf_dco.metadata_prepare_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_metadata_h2d_ms",
        leaf_dco.metadata_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_prefilter_clock_ticks",
        leaf_dco.prefilter_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_exact_rerank_clock_ticks",
        leaf_dco.exact_rerank_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_audit_clock_ticks",
        leaf_dco.audit_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tq_prefilter_clock_ticks",
        leaf_dco.tq_prefilter_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tq_survivor_insert_clock_ticks",
        leaf_dco.tq_survivor_insert_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tq_exact_rerank_clock_ticks",
        leaf_dco.tq_exact_rerank_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tq_output_clock_ticks",
        leaf_dco.tq_output_clock_ticks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tile_started",
        leaf_dco.tile_started,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_tile_done",
        leaf_dco.tile_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_prefilter_row_started",
        leaf_dco.prefilter_row_started,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_prefilter_row_done",
        leaf_dco.prefilter_row_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_exact_rerank_row_started",
        leaf_dco.exact_rerank_row_started,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_exact_rerank_row_done",
        leaf_dco.exact_rerank_row_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_audit_row_started",
        leaf_dco.audit_row_started,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_audit_row_done",
        leaf_dco.audit_row_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_progress_heartbeat",
        leaf_dco.progress_heartbeat,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_staged_tiles",
        leaf_dco.staged_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_col_tasks_started",
        leaf_dco.col_tasks_started,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_col_tasks_done",
        leaf_dco.col_tasks_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_finalize_started",
        leaf_dco.finalize_started,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_finalize_done",
        leaf_dco.finalize_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_active_slots_max",
        leaf_dco.active_slots_max,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_slot_wait_polls",
        leaf_dco.slot_wait_polls,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_row_lock_spins",
        leaf_dco.row_lock_spins,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_dco_staged_rows",
        leaf_dco.staged_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_count",
        profile.rbc.gpu_subtree_compact_attempt_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_points",
        profile.rbc.gpu_subtree_compact_attempt_points,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_wall_ms",
        profile.rbc.gpu_subtree_d0_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_child_run_count",
        profile.rbc.gpu_subtree_d0_child_run_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_child_points",
        profile.rbc.gpu_subtree_d0_child_points,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_child_max_len",
        profile.rbc.gpu_subtree_d0_child_max_len,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_child_p50_len",
        profile.rbc.gpu_subtree_d0_child_p50_len,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_child_p95_len",
        profile.rbc.gpu_subtree_d0_child_p95_len,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_child_p99_len",
        profile.rbc.gpu_subtree_d0_child_p99_len,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_partition_hash",
        profile.rbc.gpu_subtree_d0_partition_hash,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d0_task_list_hash",
        profile.rbc.gpu_subtree_d0_task_list_hash,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_producer_ms",
        profile.rbc.gpu_subtree_d1plus_producer_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_child_replay_ms",
        profile.rbc.gpu_subtree_d1plus_child_replay_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_lazy_split_ms",
        profile.rbc.gpu_subtree_d1plus_lazy_split_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_backend_ms",
        profile.rbc.gpu_subtree_d1plus_backend_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_push_wait_ms",
        profile.rbc.gpu_subtree_d1plus_push_wait_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_queue_max_bytes",
        profile.rbc.gpu_subtree_d1plus_queue_max_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_queue_max_batches",
        profile.rbc.gpu_subtree_d1plus_queue_max_batches,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sidecar_gather_active_ms",
        profile
            .rbc
            .gpu_subtree_sidecar_gather_active_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sidecar_gather_hidden_ms",
        profile
            .rbc
            .gpu_subtree_sidecar_gather_hidden_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "gpu_wait_for_sidecar_ms",
        profile
            .rbc
            .gpu_subtree_gpu_wait_for_sidecar_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sidecar_queue_max_bytes",
        profile.rbc.gpu_subtree_sidecar_queue_max_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sidecar_preload_children",
        profile.rbc.gpu_subtree_sidecar_preload_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sidecar_preload_workers",
        profile.rbc.gpu_subtree_sidecar_preload_workers,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "sidecar_queue_max_children",
        profile.rbc.gpu_subtree_sidecar_queue_max_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_admission_split_count",
        profile.rbc.gpu_subtree_d1plus_admission_split_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_admission_split_rows",
        profile.rbc.gpu_subtree_d1plus_admission_split_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_admission_split_max_before",
        profile.rbc.gpu_subtree_d1plus_admission_split_max_before,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_admission_split_max_after",
        profile.rbc.gpu_subtree_d1plus_admission_split_max_after,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_preflight_split_count",
        profile.rbc.gpu_subtree_d1plus_preflight_split_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_lazy_split_source",
        profile.rbc.gpu_subtree_d1plus_lazy_split_source,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_preflight_split_rows",
        profile.rbc.gpu_subtree_d1plus_preflight_split_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_preflight_split_max_before",
        profile.rbc.gpu_subtree_d1plus_preflight_split_max_before,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_preflight_split_max_after",
        profile.rbc.gpu_subtree_d1plus_preflight_split_max_after,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_prevented_cpu_fallback_count",
        profile.rbc.gpu_subtree_d1plus_prevented_cpu_fallback_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_point_read_ms",
        profile.rbc.gpu_subtree_host_point_read_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_vector_dedup_ms",
        profile.rbc.gpu_subtree_host_vector_dedup_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_vector_child_gather_ms",
        profile
            .rbc
            .gpu_subtree_host_vector_child_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_child_rows_total",
        profile.rbc.gpu_subtree_d1plus_child_rows_total,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_unique_rows_estimate",
        profile.rbc.gpu_subtree_d1plus_unique_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_repeat_rows_estimate",
        profile.rbc.gpu_subtree_d1plus_repeat_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_repeat_factor_x1000",
        profile.rbc.gpu_subtree_d1plus_repeat_factor_x1000,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_rolling_prev1_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_prev1_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_rolling_window4_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_window4_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_rolling_window8_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_window8_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_rolling_window16_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_window16_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_natural_rolling_prev1_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_prev1_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_natural_rolling_window4_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_window4_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_natural_rolling_window8_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_window8_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_natural_rolling_window16_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_window16_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_scheduled_rolling_prev1_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_prev1_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_scheduled_rolling_window4_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_window4_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_scheduled_rolling_window8_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_window8_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d1plus_scheduled_rolling_window16_hit_rows_estimate",
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_window16_hit_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_schedule_mode",
        profile.rbc.gpu_subtree_child_schedule_mode,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_schedule_plan_ms",
        profile.rbc.gpu_subtree_child_schedule_plan_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_schedule_plan_bytes",
        profile.rbc.gpu_subtree_child_schedule_plan_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_schedule_reordered_children",
        profile.rbc.gpu_subtree_child_schedule_reordered_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "child_schedule_overlap_edges",
        profile.rbc.gpu_subtree_child_schedule_overlap_edges,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_enabled",
        profile.rbc.gpu_subtree_vector_reuse_enabled,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_budget_bytes",
        profile.rbc.gpu_subtree_vector_reuse_budget_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_window_children",
        profile.rbc.gpu_subtree_vector_reuse_window_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_hit_rows",
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_hits,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_miss_rows",
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_misses,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_hit_bytes",
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_hit_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_miss_bytes",
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_miss_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_insert_rows",
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_inserts,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_evicted_rows",
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_evictions,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_bypass_rows",
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_bypass_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_lookup_ms",
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_lookup_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_reuse_insert_ms",
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_insert_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_groups",
        profile.rbc.gpu_subtree_vector_staging_groups,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_materialized_children",
        profile.rbc.gpu_subtree_vector_staging_materialized_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_fallback_groups",
        profile.rbc.gpu_subtree_vector_staging_fallback_groups,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_sum_child_rows",
        profile.rbc.gpu_subtree_vector_staging_sum_child_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_union_rows",
        profile.rbc.gpu_subtree_vector_staging_union_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_duplicate_rows",
        profile.rbc.gpu_subtree_vector_staging_duplicate_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_saved_read_rows_estimate",
        profile
            .rbc
            .gpu_subtree_vector_staging_saved_read_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_staging_peak_union_bytes",
        profile.rbc.gpu_subtree_vector_staging_peak_union_bytes,
    );
    let host_materialization_reuse_factor_x1000 =
        if profile.rbc.gpu_subtree_host_materialization_union_rows == 0 {
            0
        } else {
            profile
                .rbc
                .gpu_subtree_host_materialization_sum_child_rows
                .saturating_mul(1000)
                / profile.rbc.gpu_subtree_host_materialization_union_rows
        };
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_reuse_enabled",
        profile.rbc.gpu_subtree_host_materialization_reuse_enabled,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_groups",
        profile.rbc.gpu_subtree_host_materialization_groups,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_grouped_children",
        profile
            .rbc
            .gpu_subtree_host_materialization_grouped_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_materialized_children",
        profile
            .rbc
            .gpu_subtree_host_materialization_materialized_children,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_fallback_groups",
        profile.rbc.gpu_subtree_host_materialization_fallback_groups,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_sum_child_rows",
        profile.rbc.gpu_subtree_host_materialization_sum_child_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_union_rows",
        profile.rbc.gpu_subtree_host_materialization_union_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_duplicate_rows",
        profile.rbc.gpu_subtree_host_materialization_duplicate_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_reuse_factor_x1000",
        host_materialization_reuse_factor_x1000,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_saved_read_rows_estimate",
        profile
            .rbc
            .gpu_subtree_host_materialization_saved_read_rows_estimate,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_peak_union_bytes",
        profile
            .rbc
            .gpu_subtree_host_materialization_peak_union_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_plan_ms",
        profile
            .rbc
            .gpu_subtree_host_materialization_plan_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_union_load_ms",
        profile
            .rbc
            .gpu_subtree_host_materialization_union_load_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_child_gather_ms",
        profile
            .rbc
            .gpu_subtree_host_materialization_child_gather_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_materialization_backend_ms",
        profile
            .rbc
            .gpu_subtree_host_materialization_backend_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_ms",
        profile.rbc.gpu_subtree_async_flush_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_produce_ms",
        profile.rbc.gpu_subtree_async_flush_produce_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_join_wait_ms",
        profile
            .rbc
            .gpu_subtree_async_flush_join_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "post_last_gpu_cpu_tail_ms",
        profile
            .rbc
            .gpu_subtree_post_last_gpu_cpu_tail_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "post_last_gpu_cpu_tail_observed",
        profile.rbc.gpu_subtree_post_last_gpu_cpu_tail_observed,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_worker_ms_sum",
        profile
            .rbc
            .gpu_subtree_async_flush_worker_wall_sum
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_worker_count",
        profile.rbc.gpu_subtree_async_flush_worker_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_worker_mode",
        profile.rbc.gpu_subtree_async_flush_worker_mode.clone(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_initial_workers",
        profile.rbc.gpu_subtree_async_flush_initial_workers,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_max_workers",
        profile.rbc.gpu_subtree_async_flush_max_workers,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_peak_workers",
        profile.rbc.gpu_subtree_async_flush_peak_workers,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_scale_up_events",
        profile.rbc.gpu_subtree_async_flush_scale_up_events,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_queue_budget_initial_bytes",
        profile
            .rbc
            .gpu_subtree_async_flush_queue_budget_initial_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_queue_budget_peak_bytes",
        profile.rbc.gpu_subtree_async_flush_queue_budget_peak_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_controller_windows",
        profile.rbc.gpu_subtree_async_flush_controller_windows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_controller_required_workers_peak",
        profile
            .rbc
            .gpu_subtree_async_flush_controller_required_workers_peak,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_chunks",
        profile.rbc.gpu_subtree_async_flush_chunks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_batches",
        profile.rbc.gpu_subtree_async_flush_batches,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_rows",
        profile.rbc.gpu_subtree_async_flush_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_flush_errors",
        profile.rbc.gpu_subtree_async_flush_errors,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_requested",
        profile
            .rbc
            .gpu_subtree_async_result_scatter_requested
            .clone(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_enabled",
        profile.rbc.gpu_subtree_async_result_scatter_enabled,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_submissions",
        profile.rbc.gpu_subtree_async_result_scatter_submissions,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_completions",
        profile.rbc.gpu_subtree_async_result_scatter_completions,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_failures",
        profile.rbc.gpu_subtree_async_result_scatter_failures,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_worker_ms",
        profile
            .rbc
            .gpu_subtree_async_result_scatter_worker_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_wait_ms",
        profile
            .rbc
            .gpu_subtree_async_result_scatter_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_join_wait_ms",
        profile
            .rbc
            .gpu_subtree_async_result_scatter_join_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_max_inflight",
        profile.rbc.gpu_subtree_async_result_scatter_max_inflight,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_peak_raw_bytes",
        profile.rbc.gpu_subtree_async_result_scatter_peak_raw_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_peak_context_bytes",
        profile
            .rbc
            .gpu_subtree_async_result_scatter_peak_context_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "async_result_scatter_peak_host_bytes",
        profile.rbc.gpu_subtree_async_result_scatter_peak_host_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_async_d2h_requested",
        profile.rbc.gpu_subtree_persistent_async_d2h_requested,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_async_d2h_enabled",
        profile.rbc.gpu_subtree_persistent_async_d2h_enabled,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_async_d2h_fallback_reason",
        (!profile
            .rbc
            .gpu_subtree_persistent_async_d2h_fallback_reason
            .is_empty())
        .then_some(
            profile
                .rbc
                .gpu_subtree_persistent_async_d2h_fallback_reason
                .clone(),
        ),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_device_output_slots",
        profile.rbc.gpu_subtree_persistent_device_output_slots,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_device_output_slot_bytes",
        profile.rbc.gpu_subtree_persistent_device_output_slot_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_pinned_host_slots",
        profile.rbc.gpu_subtree_persistent_pinned_host_slots,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_pinned_host_slot_bytes",
        profile.rbc.gpu_subtree_persistent_pinned_host_slot_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_session_init_ms",
        profile
            .rbc
            .gpu_subtree_persistent_session_init_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_pinned_allocation_ms",
        profile
            .rbc
            .gpu_subtree_persistent_pinned_allocation_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_d2h_header_copy_ms",
        profile
            .rbc
            .gpu_subtree_persistent_d2h_header_copy_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_d2h_payload_event_ms",
        profile
            .rbc
            .gpu_subtree_persistent_d2h_payload_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_d2h_bytes",
        profile.rbc.gpu_subtree_persistent_d2h_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_device_slot_wait_ms",
        profile
            .rbc
            .gpu_subtree_persistent_device_slot_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_host_slot_wait_ms",
        profile
            .rbc
            .gpu_subtree_persistent_host_slot_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_d2h_final_drain_ms",
        profile
            .rbc
            .gpu_subtree_persistent_d2h_final_drain_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_d2h_hidden_estimate_ms",
        profile
            .rbc
            .gpu_subtree_persistent_d2h_hidden_estimate_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_passes",
        profile.rbc.gpu_subtree_compact_assignment_passes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "leaf_count",
        profile.rbc.gpu_subtree_compact_leaf_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "compact_topk_batches",
        profile.rbc.gpu_subtree_compact_batches,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "compact_topk_rows",
        profile.rbc.gpu_subtree_compact_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "h2d_bytes",
        profile.rbc.gpu_subtree_compact_h2d_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "d2h_bytes",
        profile.rbc.gpu_subtree_compact_d2h_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "resident_h2d_bytes",
        profile.rbc.gpu_subtree_resident_h2d_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "resident_topk_metadata_h2d_bytes",
        profile.rbc.gpu_subtree_resident_topk_metadata_h2d_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_normalize_ms",
        profile.rbc.gpu_subtree_host_normalize_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_load_vectors_ms",
        profile.rbc.gpu_subtree_host_load_vectors_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_norms_ms",
        profile.rbc.gpu_subtree_host_norms_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "subtree_vector_h2d_ms",
        profile.rbc.gpu_subtree_vector_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "subtree_global_ids_h2d_ms",
        profile.rbc.gpu_subtree_global_ids_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "subtree_norms_h2d_ms",
        profile.rbc.gpu_subtree_norms_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_active_ids_h2d_ms",
        profile
            .rbc
            .gpu_subtree_assignment_active_ids_h2d_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_leader_ids_h2d_ms",
        profile
            .rbc
            .gpu_subtree_assignment_leader_ids_h2d_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_assignment_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_d2h_submit_ms",
        profile
            .rbc
            .gpu_subtree_assignment_d2h_submit_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_sync_wait_ms",
        profile
            .rbc
            .gpu_subtree_assignment_sync_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "cpu_cluster_merge_ms",
        profile.rbc.gpu_subtree_cpu_cluster_merge_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "cpu_leaf_global_to_local_ms",
        profile
            .rbc
            .gpu_subtree_cpu_leaf_global_to_local_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "resident_topk_metadata_h2d_ms",
        profile
            .rbc
            .gpu_subtree_resident_topk_metadata_h2d_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "resident_topk_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_resident_topk_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "resident_topk_d2h_submit_ms",
        profile
            .rbc
            .gpu_subtree_resident_topk_d2h_submit_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "resident_topk_sync_wait_ms",
        profile
            .rbc
            .gpu_subtree_resident_topk_sync_wait_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "cpu_result_scatter_ms",
        profile.rbc.gpu_subtree_cpu_result_scatter_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "partition_only_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_partition_only_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "partition_only_total_ms",
        profile
            .rbc
            .gpu_subtree_partition_only_total_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_total_ms",
        profile.rbc.gpu_subtree_library_topk_total_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_gemm_ms",
        profile.rbc.gpu_subtree_library_topk_gemm_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_select_ms",
        profile.rbc.gpu_subtree_library_topk_select_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_h2d_ms",
        profile.rbc.gpu_subtree_library_topk_h2d_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_d2h_ms",
        profile.rbc.gpu_subtree_library_topk_d2h_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_workspace_peak_bytes",
        profile.rbc.gpu_subtree_library_topk_workspace_peak_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_workspace_budget_bytes",
        profile.rbc.gpu_subtree_library_topk_workspace_budget_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_leaf_count",
        profile.rbc.gpu_subtree_library_topk_leaf_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_rows",
        profile.rbc.gpu_subtree_library_topk_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_batches",
        profile.rbc.gpu_subtree_library_topk_batches,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "library_topk_cpu_scatter_ms",
        profile
            .rbc
            .gpu_subtree_library_topk_cpu_scatter_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "vector_upload_ms",
        profile.rbc.gpu_subtree_vector_upload_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_kernel_ms",
        profile.rbc.gpu_subtree_assignment_kernel_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "assignment_d2h_ms",
        profile.rbc.gpu_subtree_assignment_d2h_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "topk_kernel_ms",
        profile.rbc.gpu_subtree_topk_kernel_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "topk_d2h_ms",
        profile.rbc.gpu_subtree_topk_d2h_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "host_control_ms",
        profile.rbc.gpu_subtree_host_control_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "total_ms",
        profile.rbc.gpu_subtree_compact_total_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "cpu_compact_flush_ms",
        profile.rbc.gpu_subtree_cpu_compact_flush_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "workspace_peak_bytes",
        profile.rbc.gpu_subtree_workspace_peak_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "workspace_budget_bytes",
        profile.rbc.gpu_subtree_workspace_budget_bytes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "workspace_peak_over_budget",
        ratio_or_zero(
            profile.rbc.gpu_subtree_workspace_peak_bytes,
            profile.rbc.gpu_subtree_workspace_budget_bytes,
        ),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "cpu_fallbacks",
        profile.rbc.gpu_subtree_cpu_fallbacks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "compact_unsupported_count",
        profile.rbc.gpu_subtree_compact_unsupported_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "compact_unsupported_points",
        profile.rbc.gpu_subtree_compact_unsupported_points,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_attempt_count",
        profile.rbc.gpu_subtree_persistent_attempt_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_success_count",
        profile.rbc.gpu_subtree_persistent_success_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_fallback_count",
        profile.rbc.gpu_subtree_persistent_fallback_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_worker_blocks",
        profile.rbc.gpu_subtree_persistent_partition_worker_blocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_worker_blocks",
        profile.rbc.gpu_subtree_persistent_topk_worker_blocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_pops",
        profile.rbc.gpu_subtree_persistent_partition_pops,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_pushes",
        profile.rbc.gpu_subtree_persistent_leaf_pushes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_pops",
        profile.rbc.gpu_subtree_persistent_topk_pops,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_claims",
        profile.rbc.gpu_subtree_persistent_partition_claims,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_empty_polls",
        profile.rbc.gpu_subtree_persistent_partition_empty_polls,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_frontier_end",
        profile.rbc.gpu_subtree_persistent_partition_frontier_end,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_level_rounds",
        profile.rbc.gpu_subtree_persistent_partition_level_rounds,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_claims",
        profile.rbc.gpu_subtree_persistent_topk_claims,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_empty_polls",
        profile.rbc.gpu_subtree_persistent_topk_empty_polls,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_done_cas_wins",
        profile.rbc.gpu_subtree_persistent_done_cas_wins,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_active_wait_cycles",
        profile.rbc.gpu_subtree_persistent_active_wait_cycles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_busy_clocks",
        profile.rbc.gpu_subtree_persistent_partition_busy_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_wait_clocks",
        profile.rbc.gpu_subtree_persistent_partition_wait_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_busy_clocks",
        profile.rbc.gpu_subtree_persistent_topk_busy_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_wait_clocks",
        profile.rbc.gpu_subtree_persistent_topk_wait_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_help_clocks",
        profile.rbc.gpu_subtree_persistent_large_help_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_help_clocks",
        profile.rbc.gpu_subtree_persistent_topk_help_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_partition_helped_topk_tiles",
        profile
            .rbc
            .gpu_subtree_persistent_partition_helped_topk_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_helped_large_tiles",
        profile.rbc.gpu_subtree_persistent_topk_helped_large_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_elastic_role_switches",
        profile.rbc.gpu_subtree_persistent_elastic_role_switches,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_local_counter_flushes",
        profile
            .rbc
            .gpu_subtree_persistent_topk_local_counter_flushes,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_idle_backoff_calls",
        profile.rbc.gpu_subtree_persistent_topk_idle_backoff_calls,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_idle_backoff_requested_ns",
        profile
            .rbc
            .gpu_subtree_persistent_topk_idle_backoff_requested_ns,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_idle_leaf_resets",
        profile.rbc.gpu_subtree_persistent_topk_idle_leaf_resets,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_idle_large_resets",
        profile.rbc.gpu_subtree_persistent_topk_idle_large_resets,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_queue_overflow_flags",
        profile.rbc.gpu_subtree_persistent_queue_overflow_flags,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_kernel_prepare_ms",
        profile
            .rbc
            .gpu_subtree_persistent_kernel_prepare_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_kernel_event_ms",
        profile
            .rbc
            .gpu_subtree_persistent_kernel_event_wall
            .as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_total_ms",
        profile.rbc.gpu_subtree_persistent_total_wall.as_millis(),
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_topk_rows",
        profile.rbc.gpu_subtree_persistent_topk_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_count",
        profile.rbc.gpu_subtree_persistent_leaf_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_effective_root_fanout",
        profile.rbc.gpu_subtree_persistent_effective_root_fanout,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_effective_deeper_fanout",
        profile.rbc.gpu_subtree_persistent_effective_deeper_fanout,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_fanout_assignment_slots",
        profile.rbc.gpu_subtree_persistent_fanout_assignment_slots,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_replicated_rows",
        profile.rbc.gpu_subtree_persistent_replicated_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_unique_resident_rows",
        profile.rbc.gpu_subtree_persistent_unique_resident_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_row_amplification_x1000",
        profile
            .rbc
            .gpu_subtree_persistent_leaf_row_amplification_x1000,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_active_large",
        profile.rbc.gpu_subtree_persistent_snapshot_active_large,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_generation",
        profile.rbc.gpu_subtree_persistent_snapshot_large_generation,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_phase",
        profile.rbc.gpu_subtree_persistent_snapshot_large_phase,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_total_tiles",
        profile
            .rbc
            .gpu_subtree_persistent_snapshot_large_total_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_count_next",
        profile.rbc.gpu_subtree_persistent_snapshot_large_count_next,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_count_done",
        profile.rbc.gpu_subtree_persistent_snapshot_large_count_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_scatter_next",
        profile
            .rbc
            .gpu_subtree_persistent_snapshot_large_scatter_next,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_snapshot_large_scatter_done",
        profile
            .rbc
            .gpu_subtree_persistent_snapshot_large_scatter_done,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_node_claims",
        profile.rbc.gpu_subtree_persistent_large_node_claims,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_node_splits",
        profile.rbc.gpu_subtree_persistent_large_node_splits,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_split_retries",
        profile.rbc.gpu_subtree_persistent_large_split_retries,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_balanced_fallbacks",
        profile.rbc.gpu_subtree_persistent_large_balanced_fallbacks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_node_rows",
        profile.rbc.gpu_subtree_persistent_large_node_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_node_max_rows",
        profile.rbc.gpu_subtree_persistent_large_node_max_rows,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_node_empty_polls",
        profile.rbc.gpu_subtree_persistent_large_node_empty_polls,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_node_serialized_wait_cycles",
        profile
            .rbc
            .gpu_subtree_persistent_large_node_serialized_wait_cycles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_count_distance_clocks",
        profile
            .rbc
            .gpu_subtree_persistent_large_count_distance_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_count_commit_clocks",
        profile.rbc.gpu_subtree_persistent_large_count_commit_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_scatter_clocks",
        profile.rbc.gpu_subtree_persistent_large_scatter_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_publish_clocks",
        profile.rbc.gpu_subtree_persistent_large_publish_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_ticket_range_claims",
        profile.rbc.gpu_subtree_persistent_large_ticket_range_claims,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_physical_tiles",
        profile.rbc.gpu_subtree_persistent_large_physical_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_sq8_dp4a_tiles",
        profile.rbc.gpu_subtree_persistent_large_sq8_dp4a_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_large_sq8_wmma_tiles",
        profile.rbc.gpu_subtree_persistent_large_sq8_wmma_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_sq8_dp4a_tiles",
        profile.rbc.gpu_subtree_persistent_leaf_sq8_dp4a_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_sq8_wmma_tiles",
        profile.rbc.gpu_subtree_persistent_leaf_sq8_wmma_tiles,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_row_owned_tasks",
        profile.rbc.gpu_subtree_persistent_leaf_row_owned_tasks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_staged_tasks",
        profile.rbc.gpu_subtree_persistent_leaf_staged_tasks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_score_clocks",
        profile.rbc.gpu_subtree_persistent_leaf_score_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_row_lock_merge_clocks",
        profile
            .rbc
            .gpu_subtree_persistent_leaf_row_lock_merge_clocks,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_size_min",
        profile.rbc.gpu_subtree_persistent_leaf_size_min,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_size_max",
        profile.rbc.gpu_subtree_persistent_leaf_size_max,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_size_sum",
        profile.rbc.gpu_subtree_persistent_leaf_size_sum,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_size_count",
        profile.rbc.gpu_subtree_persistent_leaf_size_count,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_target",
        profile.rbc.gpu_subtree_persistent_leaf_target,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_min",
        profile.rbc.gpu_subtree_persistent_leaf_min,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_max",
        profile.rbc.gpu_subtree_persistent_leaf_max,
    );
    insert_json_value(
        &mut gpu_subtree_map,
        "persistent_leaf_tile_rows",
        profile.rbc.gpu_subtree_persistent_leaf_tile_rows,
    );
    if let Some(decision) = &profile.rbc.d0_throughput_model {
        let mut throughput_map = serde_json::Map::new();
        insert_json_value(&mut throughput_map, "mode", decision.mode.as_str());
        insert_json_value(
            &mut throughput_map,
            "memory_safe_rows",
            decision.memory_safe_rows,
        );
        insert_json_value(
            &mut throughput_map,
            "memory_safe_root_leaders",
            decision.memory_safe_root_leaders,
        );
        insert_json_value(&mut throughput_map, "selected_rows", decision.selected_rows);
        insert_json_value(
            &mut throughput_map,
            "selected_root_leaders",
            decision.selected_root_leaders,
        );
        insert_json_value(
            &mut throughput_map,
            "score_preferred_rows",
            decision.score_preferred_rows,
        );
        insert_json_value(
            &mut throughput_map,
            "score_preferred_root_leaders",
            decision.score_preferred_root_leaders,
        );
        insert_json_value(
            &mut throughput_map,
            "selected_ratio_x1000",
            decision.selected_ratio_x1000,
        );
        insert_json_value(
            &mut throughput_map,
            "candidate_count",
            decision.candidate_count,
        );
        insert_json_value(
            &mut throughput_map,
            "candidate_generation_version",
            decision.candidate_generation_version,
        );
        throughput_map.insert(
            "candidate_sources".to_string(),
            serde_json::Value::Array(
                decision
                    .candidate_sources
                    .iter()
                    .map(|source| serde_json::Value::String((*source).to_string()))
                    .collect(),
            ),
        );
        insert_json_value(
            &mut throughput_map,
            "child_skew_factor_x1000",
            decision.child_skew_factor_x1000,
        );
        insert_json_u128(&mut throughput_map, "row_weight", decision.row_weight);
        insert_json_u128(&mut throughput_map, "leaf_weight", decision.leaf_weight);
        insert_json_u128(&mut throughput_map, "topk_weight", decision.topk_weight);
        insert_json_u128(&mut throughput_map, "split_weight", decision.split_weight);
        insert_json_u128(
            &mut throughput_map,
            "d0_leader_weight",
            decision.d0_leader_weight,
        );
        insert_json_option_u128(
            &mut throughput_map,
            "predicted_persistent_ms",
            decision.predicted_persistent_ms,
        );
        insert_json_option_u128(
            &mut throughput_map,
            "predicted_preload_visible_ms",
            decision.predicted_preload_visible_ms,
        );
        insert_json_option_u128(
            &mut throughput_map,
            "predicted_d0_penalty_ms",
            decision.predicted_d0_penalty_ms,
        );
        insert_json_option_u128(
            &mut throughput_map,
            "predicted_total_ms",
            decision.predicted_total_ms,
        );
        insert_json_u128(
            &mut throughput_map,
            "predicted_persistent_score",
            decision.predicted_persistent_score,
        );
        insert_json_u128(
            &mut throughput_map,
            "predicted_preload_visible_score",
            decision.predicted_preload_visible_score,
        );
        insert_json_u128(
            &mut throughput_map,
            "predicted_d0_penalty_score",
            decision.predicted_d0_penalty_score,
        );
        insert_json_u128(
            &mut throughput_map,
            "structural_total_score",
            decision.structural_total_score,
        );
        insert_json_value(
            &mut throughput_map,
            "predicted_p95_child_rows",
            decision.predicted_p95_child_rows,
        );
        insert_json_value(
            &mut throughput_map,
            "predicted_leaf_count",
            decision.predicted_leaf_count,
        );
        insert_json_value(
            &mut throughput_map,
            "predicted_topk_pops",
            decision.predicted_topk_pops,
        );
        insert_json_value(
            &mut throughput_map,
            "predicted_lazy_split_count",
            decision.predicted_lazy_split_count,
        );
        insert_json_value(
            &mut throughput_map,
            "calibration_state",
            decision.calibration_state,
        );
        insert_json_value(
            &mut throughput_map,
            "calibration_source",
            &decision.calibration_source,
        );
        insert_json_value(&mut throughput_map, "reason", decision.reason);

        let candidates = decision
            .candidates
            .iter()
            .map(|candidate| {
                let mut candidate_map = serde_json::Map::new();
                insert_json_value(&mut candidate_map, "source", candidate.source.as_str());
                insert_json_value(&mut candidate_map, "rows", candidate.rows);
                insert_json_value(&mut candidate_map, "ratio_x1000", candidate.ratio_x1000);
                insert_json_value(
                    &mut candidate_map,
                    "target_root_leaders",
                    candidate.target_root_leaders,
                );
                insert_json_value(&mut candidate_map, "root_leaders", candidate.root_leaders);
                insert_json_value(
                    &mut candidate_map,
                    "predicted_avg_child_rows",
                    candidate.predicted_avg_child_rows,
                );
                insert_json_value(
                    &mut candidate_map,
                    "predicted_p95_child_rows",
                    candidate.predicted_p95_child_rows,
                );
                insert_json_value(
                    &mut candidate_map,
                    "predicted_leaf_count",
                    candidate.predicted_leaf_count,
                );
                insert_json_value(
                    &mut candidate_map,
                    "predicted_topk_pops",
                    candidate.predicted_topk_pops,
                );
                insert_json_value(
                    &mut candidate_map,
                    "predicted_lazy_split_count",
                    candidate.predicted_lazy_split_count,
                );
                insert_json_u128(
                    &mut candidate_map,
                    "predicted_persistent_score",
                    candidate.predicted_persistent_score,
                );
                insert_json_u128(
                    &mut candidate_map,
                    "predicted_preload_visible_score",
                    candidate.predicted_preload_visible_score,
                );
                insert_json_u128(
                    &mut candidate_map,
                    "predicted_d0_penalty_score",
                    candidate.predicted_d0_penalty_score,
                );
                insert_json_u128(
                    &mut candidate_map,
                    "structural_total_score",
                    candidate.structural_total_score,
                );
                insert_json_option_u128(
                    &mut candidate_map,
                    "predicted_persistent_ms",
                    candidate.predicted_persistent_ms,
                );
                insert_json_option_u128(
                    &mut candidate_map,
                    "predicted_preload_visible_ms",
                    candidate.predicted_preload_visible_ms,
                );
                insert_json_option_u128(
                    &mut candidate_map,
                    "predicted_d0_penalty_ms",
                    candidate.predicted_d0_penalty_ms,
                );
                insert_json_option_u128(
                    &mut candidate_map,
                    "predicted_child_ms",
                    candidate.predicted_child_ms,
                );
                insert_json_option_u128(
                    &mut candidate_map,
                    "predicted_total_ms",
                    candidate.predicted_total_ms,
                );
                insert_json_value(&mut candidate_map, "admitted", candidate.admitted);
                insert_json_value(&mut candidate_map, "reject_reason", candidate.reject_reason);
                serde_json::Value::Object(candidate_map)
            })
            .collect::<Vec<_>>();
        throughput_map.insert(
            "candidates".to_string(),
            serde_json::Value::Array(candidates),
        );
        gpu_subtree_map.insert(
            "d0_throughput_model".to_string(),
            serde_json::Value::Object(throughput_map),
        );
    }
    let d0_capacity = &profile.rbc.d0_capacity;
    let mut d0_capacity_map = serde_json::Map::new();
    insert_json_value(&mut d0_capacity_map, "enabled", d0_capacity.enabled);
    insert_json_value(&mut d0_capacity_map, "c_gpu", d0_capacity.c_gpu);
    insert_json_value(&mut d0_capacity_map, "c_high", d0_capacity.c_high);
    insert_json_value(&mut d0_capacity_map, "c_target", d0_capacity.c_target);
    insert_json_value(&mut d0_capacity_map, "c_low", d0_capacity.c_low);
    insert_json_value(&mut d0_capacity_map, "threshold_policy", d0_capacity.threshold_policy);
    insert_json_value(&mut d0_capacity_map, "high_budget_bytes", d0_capacity.high_budget_bytes);
    insert_json_value(&mut d0_capacity_map, "high_required_bytes", d0_capacity.high_required_bytes);
    insert_json_value(&mut d0_capacity_map, "target_budget_bytes", d0_capacity.target_budget_bytes);
    insert_json_value(&mut d0_capacity_map, "target_required_bytes", d0_capacity.target_required_bytes);
    insert_json_value(&mut d0_capacity_map, "k_capacity", d0_capacity.k_capacity);
    insert_json_value(
        &mut d0_capacity_map,
        "quality_min_root_leaders",
        d0_capacity.quality_min_root_leaders,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "root_leader_cap",
        d0_capacity.root_leader_cap,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "root_leader_source",
        d0_capacity.root_leader_source,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "root_leaders_initial",
        d0_capacity.root_leaders_initial,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "root_leaders_final",
        d0_capacity.root_leaders_final,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "oversized_child_count",
        d0_capacity.oversized_child_count,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "oversized_split_count",
        d0_capacity.oversized_split_count,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "oversized_split_rows",
        d0_capacity.oversized_split_rows,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "small_child_count",
        d0_capacity.small_child_count,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "small_child_rows",
        d0_capacity.small_child_rows,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "actual_child_max_unique_rows",
        d0_capacity.actual_child_max_unique_rows,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "actual_child_max_over_c_high",
        ratio_or_zero(d0_capacity.actual_child_max_unique_rows, d0_capacity.c_high),
    );
    insert_json_value(
        &mut d0_capacity_map,
        "actual_child_p50_unique_rows",
        d0_capacity.actual_child_p50_unique_rows,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "actual_child_p95_unique_rows",
        d0_capacity.actual_child_p95_unique_rows,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "actual_child_p99_unique_rows",
        d0_capacity.actual_child_p99_unique_rows,
    );
    insert_json_value(
        &mut d0_capacity_map,
        "actual_child_p99_over_c_target",
        ratio_or_zero(
            d0_capacity.actual_child_p99_unique_rows,
            d0_capacity.c_target,
        ),
    );
    insert_json_value(
        &mut d0_capacity_map,
        "admission_failures",
        d0_capacity.admission_failures,
    );
    gpu_subtree_map.insert(
        "d0_capacity".to_string(),
        serde_json::Value::Object(d0_capacity_map),
    );
    let gpu_subtree_json = serde_json::Value::Object(gpu_subtree_map);
    writeln!(file, "  \"gpu_subtree\": {},", gpu_subtree_json)?;
    writeln!(
        file,
        "  \"resident_subtree\": {{ \"resident_subtree_count\": {}, \"resident_subtree_points\": {}, \"resident_subtree_bytes\": {}, \"resident_subtree_read_wall\": {}, \"resident_subtree_compute_wall\": {}, \"resident_subtree_emit_wall_ms\": {}, \"resident_subtree_leaf_count\": {}, \"resident_subtree_fallback_count\": {}, \"resident_subtree_fallback_points\": {}, \"resident_subtree_max_points\": {}, \"resident_subtree_avg_points\": {:.6}, \"resident_subtree_permit_wait\": {} }},",
        profile.rbc.resident_subtree_count,
        profile.rbc.resident_subtree_points,
        profile.rbc.resident_subtree_bytes,
        profile.rbc.resident_subtree_read_wall.as_millis(),
        profile.rbc.resident_subtree_compute_wall.as_millis(),
        profile.rbc.resident_subtree_emit_wall.as_millis(),
        profile.rbc.resident_subtree_leaf_count,
        profile.rbc.resident_subtree_fallback_count,
        profile.rbc.resident_subtree_fallback_points,
        profile.rbc.resident_subtree_max_points,
        resident_subtree_avg_points,
        profile.rbc.resident_subtree_permit_wait.as_millis(),
    )?;
    let root_json = root_fanout_profile_json(&profile.rbc.root_fanout);
    let root_body = root_json
        .trim()
        .strip_prefix("{")
        .and_then(|value| value.strip_suffix("}"))
        .map(str::trim)
        .unwrap_or("\"root_fanout\": null");
    writeln!(file, "  {root_body},")?;
    writeln!(
        file,
        "  \"partitioner\": {{ \"kind\": \"{}\", \"kmeans\": {{ \"points\": {}, \"emitted_points\": {}, \"leaves\": {}, \"split_attempts\": {}, \"assignment_passes\": {}, \"empty_clusters\": {}, \"fallback_splits\": {}, \"max_leaf_size\": {}, \"wall_ms\": {} }} }},",
        profile.partitioner_kind(),
        profile.kmeans.points,
        profile.kmeans.emitted_points,
        profile.kmeans.leaves,
        profile.kmeans.split_attempts,
        profile.kmeans.assignment_passes,
        profile.kmeans.empty_clusters,
        profile.kmeans.fallback_splits,
        profile.kmeans.max_leaf_size,
        profile.kmeans.wall.as_millis(),
    )?;
    writeln!(
        file,
        "  \"leaf\": {{ \"leaves\": {}, \"points\": {}, \"blockwise_leaves\": {}, \"total_wall_ms\": {}, \"load_ms\": {}, \"distance_ms\": {}, \"topk_ms\": {}, \"hash_ms\": {}, \"flush_ms\": {} }}",
        profile.leaf.leaves,
        profile.leaf.points,
        profile.leaf.blockwise_leaves,
        profile.leaf.total_wall.as_millis(),
        profile.leaf.load.as_millis(),
        profile.leaf.distance.as_millis(),
        profile.leaf.topk.as_millis(),
        profile.leaf.hash.as_millis(),
        profile.leaf.flush.as_millis(),
    )?;
    writeln!(file, "}}")?;
    Ok(())
}

impl OomProfile {
    fn partitioner_kind(&self) -> &'static str {
        self.partitioner.as_str()
    }
}

fn detailed_pipnn_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SHEAR_PIPNN_PROFILE")
            .or_else(|_| std::env::var("VELOANN_PIPNN_PROFILE"))
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                matches!(
                    value.as_str(),
                    "1" | "true" | "yes" | "on" | "detail" | "detailed"
                )
            })
            .unwrap_or(false)
    })
}

#[derive(Clone, Debug, Default)]
struct StreamingProfile {
    producer_wall: Duration,
    total_wall: Duration,
    leaf: LeafProfile,
    telemetry: SchedulerTelemetry,
    rbc: RbcPartitionTelemetry,
    kmeans: KMeansPartitionTelemetry,
    entry_point: Option<u32>,
}

impl StreamingProfile {
    fn leaf_idle_time(&self) -> Duration {
        self.total_wall.saturating_sub(self.leaf.total_wall)
    }

    fn leaf_parallelism(&self) -> f64 {
        if self.total_wall.is_zero() {
            0.0
        } else {
            self.leaf.total_wall.as_secs_f64() / self.total_wall.as_secs_f64()
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlasThreadEnv {
    values: HashMap<&'static str, Option<String>>,
}

impl BlasThreadEnv {
    const KEYS: [&'static str; 6] = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "MKL_DYNAMIC",
    ];

    fn capture() -> Self {
        let mut values = HashMap::with_capacity(Self::KEYS.len());
        for key in Self::KEYS {
            values.insert(key, std::env::var(key).ok());
        }
        Self { values }
    }

    fn set_single_threaded() {
        for key in [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "GOTO_NUM_THREADS",
        ] {
            unsafe {
                std::env::set_var(key, "1");
            }
        }
        unsafe {
            std::env::set_var("MKL_DYNAMIC", "0");
        }
    }

    fn restore(&self) {
        for key in Self::KEYS {
            match self.values.get(key).and_then(|value| value.as_ref()) {
                Some(value) => unsafe {
                    std::env::set_var(key, value);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProcessMemorySnapshot {
    rss_bytes: Option<usize>,
    hwm_bytes: Option<usize>,
    vm_peak_bytes: Option<usize>,
}

fn log_memory_telemetry(
    phase: &str,
    num_points: usize,
    params: &PiPNNParams,
    dim: usize,
    sketch_bytes: usize,
    reservoir_bytes: usize,
    worker_count: usize,
) {
    let snapshot = read_process_memory_snapshot();
    let per_worker_scratch =
        estimate_leaf_scratch_bytes(params.kernel_safe_leaf_size(), dim, params.leaf_knn);
    let active_scratch_bytes = per_worker_scratch.saturating_mul(worker_count.max(1));

    info!(
        "PiPNN memory telemetry [{}]: points={} rss_bytes={} vm_hwm_bytes={} sketch_bytes={} reservoir_bytes={} active_scratch_bytes_estimate={} max_full_matrix_leaf_size={}",
        phase,
        num_points,
        format_optional_bytes(snapshot.rss_bytes),
        format_optional_bytes(snapshot.hwm_bytes),
        sketch_bytes,
        reservoir_bytes,
        active_scratch_bytes,
        params.kernel_safe_leaf_size(),
    );
}

fn read_process_memory_snapshot() -> ProcessMemorySnapshot {
    #[cfg(target_os = "linux")]
    {
        return parse_proc_self_status()
            .map(
                |(rss_bytes, hwm_bytes, vm_peak_bytes)| ProcessMemorySnapshot {
                    rss_bytes,
                    hwm_bytes,
                    vm_peak_bytes,
                },
            )
            .unwrap_or_default();
    }

    #[cfg(not(target_os = "linux"))]
    {
        ProcessMemorySnapshot::default()
    }
}

#[cfg(target_os = "linux")]
fn parse_proc_self_status() -> Option<(Option<usize>, Option<usize>, Option<usize>)> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_bytes = None;
    let mut hwm_bytes = None;
    let mut vm_peak_bytes = None;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            rss_bytes = parse_status_kib_field(line);
        } else if line.starts_with("VmHWM:") {
            hwm_bytes = parse_status_kib_field(line);
        } else if line.starts_with("VmPeak:") {
            vm_peak_bytes = parse_status_kib_field(line);
        }
    }

    Some((rss_bytes, hwm_bytes, vm_peak_bytes))
}

#[cfg(target_os = "linux")]
fn parse_status_kib_field(line: &str) -> Option<usize> {
    let value_kib = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    value_kib.checked_mul(1024)
}

fn format_optional_bytes(value: Option<usize>) -> String {
    value
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_number(value: Option<usize>) -> String {
    value
        .map(|number| number.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn collect_oom_memory_report() -> OomMemoryReport {
    let snapshot = read_process_memory_snapshot();
    let rss_kb = snapshot
        .rss_bytes
        .map(|bytes| bytes / 1024)
        .into_iter()
        .collect();
    OomMemoryReport {
        rss_kb_samples: rss_kb,
        rss_hwm_kb: snapshot.hwm_bytes.map(|bytes| bytes / 1024),
        vm_peak_kb: snapshot.vm_peak_bytes.map(|bytes| bytes / 1024),
    }
}

fn collect_temp_file_report(dir: &Path) -> AnnResult<TempFileReport> {
    let mut report = TempFileReport::default();
    if !dir.exists() {
        return Ok(report);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let bytes = entry.metadata()?.len();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        record_file_category(&mut report.total, bytes);
        if (name.starts_with("candidate_shard") && name.ends_with(".log"))
            || name == "candidate_segments.manifest"
        {
            record_file_category(&mut report.candidate, bytes);
        } else if let Some(depth) = parse_partition_depth(name) {
            match depth {
                0 => record_file_category(&mut report.partition_depth0, bytes),
                1 => record_file_category(&mut report.partition_depth1, bytes),
                2 => record_file_category(&mut report.partition_depth2, bytes),
                3 => record_file_category(&mut report.partition_depth3, bytes),
                _ => record_file_category(&mut report.other, bytes),
            }
        } else {
            record_file_category(&mut report.other, bytes);
        }
    }

    Ok(report)
}

fn record_file_category(stats: &mut FileCategoryStats, bytes: u64) {
    stats.files += 1;
    stats.bytes += bytes;
}

fn parse_partition_depth(name: &str) -> Option<usize> {
    let suffix = name.strip_prefix("partition_d")?;
    let depth = suffix.get(0..2)?.parse::<usize>().ok()?;
    Some(depth)
}

fn estimate_sketch_bytes(sketches: &dyn SketchAccessor) -> usize {
    sketches.resident_bytes()
}

fn derive_spill_shard_points(
    num_points: usize,
    params: &PiPNNParams,
    memory_budget_bytes: usize,
) -> usize {
    let spill_budget = if params.oom_spill_cache_bytes == 0 {
        (memory_budget_bytes / 8).max(1)
    } else {
        params.oom_spill_cache_bytes.max(1)
    };
    let reservoir_bytes_per_point =
        std::mem::size_of::<HashPruneReservoir>() + params.l_max.saturating_mul(size_of::<Slot>());
    let bytes_per_point = reservoir_bytes_per_point.max(1);
    (spill_budget / bytes_per_point).clamp(1, num_points.max(1))
}

fn derive_lrpf_digest_shard_points(num_points: usize, base_shard_points: usize) -> usize {
    base_shard_points.min(1_000_000).clamp(1, num_points.max(1))
}

fn external_reduce_config(io_cfg: DirectIoConfig, pool_threads: usize) -> ExternalReduceConfig {
    ExternalReduceConfig::new(io_cfg, pool_threads.max(1), 64 * 1024)
}

fn external_reduce_worker_bounds(
    pool_threads: usize,
    eligibility: OnlineSuffixEligibility,
    host_parallelism: usize,
) -> (usize, usize) {
    let pool_threads = pool_threads.max(1);
    let max_workers = match eligibility {
        OnlineSuffixEligibility::Disabled
        | OnlineSuffixEligibility::ExternalFallback(
            "log_assisted_base_working_set_exceeds_delta_budget",
        ) => pool_threads.max(host_parallelism.max(1)),
        OnlineSuffixEligibility::Enabled | OnlineSuffixEligibility::ExternalFallback(_) => {
            pool_threads
        }
    };
    (pool_threads, max_workers)
}

fn external_reduce_config_for_eligibility(
    io_cfg: DirectIoConfig,
    pool_threads: usize,
    eligibility: OnlineSuffixEligibility,
) -> ExternalReduceConfig {
    let host_parallelism = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(pool_threads.max(1));
    let (initial_workers, max_workers) =
        external_reduce_worker_bounds(pool_threads, eligibility, host_parallelism);
    let batch_records = initial_workers
        .saturating_mul(8 * 1024)
        .clamp(64 * 1024, 1024 * 1024);
    ExternalReduceConfig::with_worker_bounds(io_cfg, initial_workers, max_workers, batch_records)
}

fn create_candidate_spill_sink(
    candidate_path: &Path,
    params: &PiPNNParams,
    shard_points: usize,
    io_cfg: DirectIoConfig,
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
) -> AnnResult<SpillEdgeSink> {
    if let Some(digest_width) = effective_compact_spill_digest_width(params) {
        info!(
            "PiPNN OOM compact spill digest enabled: width={} l_max={} multiplier={:.3}",
            digest_width, params.l_max, params.compact_spill_digest_width_multiplier
        );
        SpillEdgeSink::create_digest_sharded_with_config_and_policies(
            candidate_path,
            params.oom_spill_cache_bytes,
            shard_points,
            digest_width,
            io_cfg,
            policy,
            backfill_policy,
        )
    } else {
        SpillEdgeSink::create_sharded_with_config_and_policies(
            candidate_path,
            params.oom_spill_cache_bytes,
            shard_points,
            io_cfg,
            policy,
            backfill_policy,
        )
    }
}

fn online_suffix_reduce_config(
    params: &PiPNNParams,
    artifact_dir: &Path,
    source_start: usize,
    source_end: usize,
    pinned_prefix_points: usize,
    shard_points: usize,
    pool_threads: usize,
    memory_plan: &OomMemoryPlan,
    io_cfg: DirectIoConfig,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<Arc<HashPruneTraceCollector>>,
) -> OnlineSuffixReduceConfig {
    let reducer_workers = effective_online_suffix_reduce_workers(params, pool_threads);
    let online_budgets = params.effective_online_suffix_budgets();
    OnlineSuffixReduceConfig {
        source_start,
        source_end,
        shard_points,
        l_max: params.l_max,
        direct_queue_bytes: memory_plan.online_suffix_queue_budget_bytes,
        spill_buffer_bytes: memory_plan.online_suffix_spill_budget_bytes,
        reorder_window_bytes: memory_plan.online_suffix_reorder_budget_bytes,
        reducer_workers,
        mode: params.online_suffix_reduce_mode,
        state_cache_bytes: memory_plan.online_suffix_state_cache_budget_bytes,
        read_window_bytes: memory_plan.online_suffix_read_window_budget_bytes,
        pinned_prefix_points,
        delta_budget_bytes: online_budgets.delta_budget_bytes,
        delta_buffer_bytes: online_budgets.delta_buffer_bytes,
        compaction_bytes: memory_plan.online_suffix_compaction_budget_bytes,
        range_points: effective_log_assisted_range_points(
            params.online_suffix_range_points,
            source_end.saturating_sub(source_start),
            params.l_max,
        )
        .unwrap_or(shard_points),
        range_trigger_bytes: params.effective_online_suffix_range_trigger_bytes(),
        max_runs_per_range: params.effective_online_suffix_max_runs_per_range(),
        compaction_workers: params.effective_online_suffix_compaction_workers(pool_threads),
        drain_compaction_workers: params
            .effective_online_suffix_drain_compaction_workers(pool_threads),
        cache_hot_updates: params.effective_online_suffix_cache_hot_updates(),
        store: DiskReservoirStoreConfig {
            root: artifact_dir.join(format!(
                "online_suffix_state_{source_start:010}_{source_end:010}"
            )),
            source_start,
            source_end,
            shard_points,
            l_max: params.l_max,
            page_cache_bytes: memory_plan.online_suffix_page_cache_budget_bytes,
            writeback_bytes: memory_plan.online_suffix_writeback_budget_bytes,
            decode_bytes: memory_plan.online_suffix_decode_budget_bytes,
            writeback_workers: reducer_workers,
            overprovision_ratio: params.online_suffix_state_overprovision_ratio,
            io: io_cfg,
            policy_key: online_suffix_policy_key(params),
            backfill_policy,
        },
        evidence_policy,
        backfill_policy,
        diagnostic_collector,
    }
}

fn effective_online_suffix_reduce_workers(params: &PiPNNParams, pool_threads: usize) -> usize {
    let pool_threads = pool_threads.max(1);
    if params.online_suffix_reduce_workers == 0 {
        pool_threads.min(4)
    } else {
        params.online_suffix_reduce_workers.clamp(1, pool_threads)
    }
}

fn online_suffix_profile_config(
    params: &PiPNNParams,
    memory_plan: &OomMemoryPlan,
    pool_threads: usize,
    source_points: usize,
    base_shard_points: usize,
) -> OnlineSuffixProfileConfig {
    let effective = params.effective_online_suffix_budgets();
    let effective_reduce_workers = effective_online_suffix_reduce_workers(params, pool_threads);
    let effective_range_points = if params.online_suffix_reduce_mode.uses_log_assisted() {
        effective_log_assisted_range_points(
            params.online_suffix_range_points,
            source_points,
            params.l_max,
        )
        .unwrap_or(0)
    } else {
        0
    };
    OnlineSuffixProfileConfig {
        active: memory_plan.online_suffix_enabled,
        configured_direct_queue_bytes: params.online_suffix_direct_queue_bytes,
        configured_state_cache_bytes: params.online_suffix_state_cache_bytes,
        configured_state_page_cache_bytes: params.online_suffix_state_page_cache_bytes,
        configured_state_writeback_bytes: params.online_suffix_state_writeback_bytes,
        configured_spill_buffer_bytes: params.online_suffix_spill_buffer_bytes,
        configured_reorder_window_bytes: params.online_suffix_reorder_window_bytes,
        configured_read_window_bytes: params.online_suffix_read_window_bytes,
        configured_delta_budget_bytes: params.online_suffix_delta_budget_bytes,
        configured_delta_buffer_bytes: params.online_suffix_delta_buffer_bytes,
        configured_range_points: params.online_suffix_range_points,
        configured_range_trigger_bytes: params.online_suffix_range_trigger_bytes,
        configured_max_runs_per_range: params.online_suffix_max_runs_per_range,
        configured_compaction_workers: params.online_suffix_compaction_workers,
        configured_cache_hot_updates: params.online_suffix_cache_hot_updates,
        effective_direct_queue_bytes: effective.direct_queue_bytes,
        effective_state_cache_bytes: effective.state_cache_bytes,
        effective_state_page_cache_bytes: effective.state_page_cache_bytes,
        effective_state_writeback_bytes: effective.state_writeback_bytes,
        effective_spill_buffer_bytes: effective.spill_buffer_bytes,
        effective_reorder_window_bytes: effective.reorder_window_bytes,
        effective_decode_bytes: effective.decode_bytes,
        effective_read_window_bytes: effective.read_window_bytes,
        effective_delta_budget_bytes: effective.delta_budget_bytes,
        effective_delta_buffer_bytes: memory_plan.online_suffix_delta_buffer_budget_bytes,
        effective_compaction_bytes: memory_plan.online_suffix_compaction_budget_bytes,
        effective_shard_points: if params.online_suffix_reduce_mode.is_enabled() {
            base_shard_points
        } else {
            0
        },
        effective_range_points,
        effective_range_trigger_bytes: if params.online_suffix_reduce_mode.uses_log_assisted() {
            params.effective_online_suffix_range_trigger_bytes()
        } else {
            0
        },
        effective_max_runs_per_range: if params.online_suffix_reduce_mode.uses_log_assisted() {
            params.effective_online_suffix_max_runs_per_range()
        } else {
            0
        },
        effective_compaction_workers: if params.online_suffix_reduce_mode.uses_log_assisted() {
            params.effective_online_suffix_compaction_workers(pool_threads)
        } else {
            0
        },
        effective_drain_compaction_workers: if params.online_suffix_reduce_mode.uses_log_assisted()
        {
            params.effective_online_suffix_drain_compaction_workers(pool_threads)
        } else {
            0
        },
        effective_cache_hot_updates: if params.online_suffix_reduce_mode.uses_log_assisted() {
            params.effective_online_suffix_cache_hot_updates()
        } else {
            0
        },
        reserved_metadata_bytes: memory_plan.online_suffix_metadata_budget_bytes,
        reserved_total_bytes: memory_plan.online_suffix_budget_bytes,
        configured_reduce_workers: params.online_suffix_reduce_workers,
        effective_reduce_workers,
        state_overprovision_ratio: params.online_suffix_state_overprovision_ratio,
    }
}

fn online_suffix_policy_key(params: &PiPNNParams) -> [u8; 32] {
    let mut canonical = Vec::with_capacity(64);
    canonical.extend_from_slice(&(params.l_max as u64).to_le_bytes());
    canonical.push(hashprune_evidence_mode_tag(params.hashprune_evidence_mode));
    canonical.push(hashprune_backfill_mode_tag(params.hashprune_backfill_mode));
    canonical.extend_from_slice(&(params.hashprune_backfill_budget as u64).to_le_bytes());
    canonical.extend_from_slice(&params.hashprune_backfill_alpha.to_bits().to_le_bytes());
    append_optional_usize(&mut canonical, params.hashprune_rhalf_hash_quota);
    append_optional_f32(&mut canonical, params.hashprune_soft_tier_priority_alpha);
    append_optional_usize(&mut canonical, params.hashprune_rhalf_max_backfill_per_row);
    Sha256::digest(&canonical).into()
}

fn append_optional_usize(output: &mut Vec<u8>, value: Option<usize>) {
    output.push(u8::from(value.is_some()));
    output.extend_from_slice(&(value.unwrap_or(0) as u64).to_le_bytes());
}

fn append_optional_f32(output: &mut Vec<u8>, value: Option<f32>) {
    output.push(u8::from(value.is_some()));
    output.extend_from_slice(&value.unwrap_or(0.0).to_bits().to_le_bytes());
}

fn hashprune_evidence_mode_tag(mode: PiPNNHashPruneEvidenceMode) -> u8 {
    match mode {
        PiPNNHashPruneEvidenceMode::Off => 0,
        PiPNNHashPruneEvidenceMode::Report => 1,
        PiPNNHashPruneEvidenceMode::Enforce => 2,
    }
}

fn hashprune_backfill_mode_tag(mode: PiPNNHashPruneBackfillMode) -> u8 {
    match mode {
        PiPNNHashPruneBackfillMode::Off => 0,
        PiPNNHashPruneBackfillMode::DistanceGated => 1,
        PiPNNHashPruneBackfillMode::DistanceGatedStratified => 2,
        PiPNNHashPruneBackfillMode::DistanceGatedMultihit => 3,
        PiPNNHashPruneBackfillMode::DistanceGatedTieredD52 => 4,
        PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf => 5,
        PiPNNHashPruneBackfillMode::RHalfHashSlot1 => 6,
        PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3 => 7,
        PiPNNHashPruneBackfillMode::RHalfSoftTierQ3 => 8,
        PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => 9,
    }
}

fn spill_run_stats_from_lrpf(stats: &LrpfSpillStats) -> SpillRunStats {
    SpillRunStats {
        part_files: stats.part_files,
        segment_count: stats.fragment_count,
        shard_groups: stats.part_files,
        max_parts_per_shard: usize::from(stats.part_files > 0),
        max_segments_per_shard: stats.fragment_count,
        record_bytes: stats.record_bytes,
        manifest_bytes: 0,
    }
}

fn spill_reduce_stats_from_lrpf(stats: &LrpfReduceStats) -> SpillReduceStats {
    SpillReduceStats {
        shard_groups: stats.shard_groups,
        segments_scanned: stats.fragments_scanned,
        part_files_opened: stats.part_files_opened,
        max_segments_in_group: stats.max_fragments_in_group,
        evidence: Default::default(),
        backfill: Default::default(),
        ..SpillReduceStats::default()
    }
}

fn effective_compact_spill_digest_width(params: &PiPNNParams) -> Option<usize> {
    if !compact_spill_digest_enabled(params) {
        return None;
    }

    let base = params.l_max.max(1);
    let multiplier = std::env::var("SHEAR_PIPNN_COMPACT_SPILL_DIGEST_WIDTH_MULTIPLIER")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(params.compact_spill_digest_width_multiplier)
        .max(f32::EPSILON);
    let raw_width = ((base as f32) * multiplier).ceil() as usize;
    Some(raw_width.clamp(base, base.saturating_mul(4).max(base)))
}

fn compact_spill_digest_enabled(params: &PiPNNParams) -> bool {
    params.compact_spill_digest_enable
        || std::env::var("SHEAR_PIPNN_COMPACT_SPILL_DIGEST")
            .map(|value| {
                matches!(
                    value.as_str(),
                    "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
                )
            })
            .unwrap_or(false)
}

fn plan_resident_reservoir_prefix(
    num_points: usize,
    effective_threads: usize,
    params: &PiPNNParams,
    dim: usize,
    sketch_bytes: usize,
    shard_points: usize,
    reservoir_budget_bytes: usize,
) -> ResidentReservoirPlan {
    plan_resident_reservoir_prefix_with_layout(
        num_points,
        effective_threads,
        params,
        dim,
        sketch_bytes,
        shard_points,
        reservoir_budget_bytes,
        ResidentReservoirLayout::Legacy,
    )
}

fn plan_resident_reservoir_prefix_with_layout(
    num_points: usize,
    _effective_threads: usize,
    params: &PiPNNParams,
    _dim: usize,
    _sketch_bytes: usize,
    shard_points: usize,
    reservoir_budget_bytes: usize,
    layout: ResidentReservoirLayout,
) -> ResidentReservoirPlan {
    let resident_points_raw = resident_reservoir_points_for_budget(
        num_points,
        params.l_max,
        reservoir_budget_bytes,
        layout,
    );
    if resident_points_raw >= num_points {
        return ResidentReservoirPlan {
            resident_points: num_points,
        };
    }
    if matches!(
        online_suffix_eligibility_for_dataset(params, num_points),
        OnlineSuffixEligibility::Enabled
    ) {
        return ResidentReservoirPlan {
            resident_points: resident_points_raw,
        };
    }

    let shard_points = shard_points.max(1);
    let resident_points = resident_points_raw
        .checked_div(shard_points)
        .unwrap_or(0)
        .saturating_mul(shard_points)
        .min(num_points);
    ResidentReservoirPlan { resident_points }
}

fn derive_oom_worker_count(requested_threads: usize, params: &PiPNNParams, dim: usize) -> usize {
    let requested_threads = requested_threads.max(1);
    let memory_budget_bytes = params.effective_oom_memory_budget_bytes();
    let per_worker_bytes =
        estimate_leaf_scratch_bytes(params.kernel_safe_leaf_size(), dim, params.leaf_knn)
            .saturating_add(64 * 1024 * 1024)
            .max(1);
    let shared_reserve_bytes = memory_budget_bytes
        .saturating_mul(2)
        .checked_div(3)
        .unwrap_or(memory_budget_bytes);
    let worker_budget = memory_budget_bytes
        .saturating_sub(shared_reserve_bytes)
        .max(per_worker_bytes);
    let budget_threads = (worker_budget / per_worker_bytes).max(1);
    requested_threads.min(budget_threads.max(1))
}

fn estimate_reservoir_bytes(num_points: usize, l_max: usize) -> usize {
    let slot_bytes = size_of::<Slot>();
    let reservoir_struct_bytes =
        size_of::<HashPruneReservoir>() + size_of::<Mutex<HashPruneReservoir>>();
    num_points.saturating_mul(reservoir_struct_bytes + l_max.saturating_mul(slot_bytes))
}

fn estimate_flat_reservoir_bytes(num_points: usize, l_max: usize) -> usize {
    FlatResidentReservoirStore::estimate_bytes(
        num_points,
        l_max,
        self::flat_resident_reservoir::DEFAULT_FLAT_RESERVOIR_STRIPES,
    )
    .unwrap_or(usize::MAX)
}

fn estimate_reservoir_bytes_for_layout(
    num_points: usize,
    l_max: usize,
    layout: ResidentReservoirLayout,
) -> usize {
    match layout {
        ResidentReservoirLayout::Flat => estimate_flat_reservoir_bytes(num_points, l_max),
        ResidentReservoirLayout::Legacy => estimate_reservoir_bytes(num_points, l_max),
    }
}

fn resident_reservoir_points_for_budget(
    num_points: usize,
    l_max: usize,
    budget_bytes: usize,
    layout: ResidentReservoirLayout,
) -> usize {
    match layout {
        ResidentReservoirLayout::Legacy => budget_bytes / estimate_reservoir_bytes(1, l_max).max(1),
        ResidentReservoirLayout::Flat => {
            let mut low = 0usize;
            let mut high = num_points;
            while low < high {
                let midpoint = low + (high - low).div_ceil(2);
                if estimate_flat_reservoir_bytes(midpoint, l_max) <= budget_bytes {
                    low = midpoint;
                } else {
                    high = midpoint - 1;
                }
            }
            low
        }
    }
    .min(num_points)
}

fn estimate_leaf_scratch_bytes(max_leaf_size: usize, dim: usize, leaf_knn: usize) -> usize {
    let leaf_size = max_leaf_size.max(1);
    let dmat = leaf_size
        .saturating_mul(leaf_size)
        .saturating_mul(size_of::<f32>());
    let cand = leaf_size.saturating_mul(size_of::<(usize, f32)>());
    let x = leaf_size
        .saturating_mul(dim)
        .saturating_mul(size_of::<f32>());
    let pending_edge_size =
        size_of::<usize>() + size_of::<u32>() + size_of::<u16>() + size_of::<f32>();
    let edges = leaf_size
        .saturating_mul(leaf_knn.max(1).saturating_mul(2))
        .saturating_mul(pending_edge_size);
    dmat.saturating_add(cand)
        .saturating_add(x)
        .saturating_add(edges)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OomExecutionThreads {
    pool_threads: usize,
    leaf_budget_threads: usize,
}

fn derive_oom_execution_threads(
    requested_threads: usize,
    params: &PiPNNParams,
    dim: usize,
) -> OomExecutionThreads {
    let pool_threads = requested_threads.max(1);
    let leaf_budget_threads = derive_oom_worker_count(pool_threads, params, dim);
    OomExecutionThreads {
        pool_threads,
        leaf_budget_threads,
    }
}

fn persistent_scheduler_mode_actual(
    execution_mode: PiPNNGpuSubtreeExecutionMode,
    requested: PiPNNGpuSubtreePersistentSchedulerMode,
) -> PiPNNGpuSubtreePersistentSchedulerMode {
    if execution_mode == PiPNNGpuSubtreeExecutionMode::TwoStage {
        PiPNNGpuSubtreePersistentSchedulerMode::Baseline
    } else {
        requested.resolved()
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::PathBuf;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;
    use std::{env, fs};

    use parking_lot::Mutex;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use tempfile::tempdir;

    use super::dco_sq_int8::SQ_INT8_DEFAULT_BITS;
    use super::{
        PiPNNComputeBackendKind, PiPNNGpuSubtreeAssignmentBackend, PiPNNGpuSubtreeDcoMode,
        PiPNNGpuSubtreeDcoRerankMode, PiPNNGpuSubtreeExecutionMode, PiPNNGpuSubtreeLeafTopKBackend,
        PiPNNGpuSubtreePersistentLargeTicketBatch, PiPNNGpuSubtreePersistentSchedulerMode,
        PiPNNGpuSubtreeQcodeD1PayloadMode, PiPNNGpuSubtreeQcodeStagingMode,
        PiPNNGpuSubtreeRabitqD1DistanceMode, PiPNNGpuSubtreeRabitqPlusMode,
        PiPNNGpuSubtreeRabitqPlusRotator, PiPNNGpuSubtreeRabitqPlusSidecarMode,
        PiPNNGpuSubtreeSq8AssignmentKernel, PiPNNGpuSubtreeSq8LeafExecutor,
        PiPNNGpuSubtreeSq8LeafMode, PiPNNGpuSubtreeSqStagingMode, PiPNNGpuSubtreeSqWmmaMode,
        PiPNNGpuSubtreeSqWmmaSidecarMode, PiPNNGpuSubtreeTurboQuantMode,
        PiPNNGpuSubtreeTurboQuantSidecarMode, PiPNNOnlineSuffixReduceMode, PiPNNOomIoMode,
        PiPNNParams, PiPNNPartitionerKind, PiPNNPruneBackend, SqInt8CpuKernel,
        resolve_compute_backend,
    };
    use crate::algorithm::pipnn::compute_backend::PiPNNComputeBackend;
    use crate::algorithm::pipnn::direct_io::DirectIoConfig;
    use crate::algorithm::pipnn::disk_reservoir_store::DiskReservoirStoreConfig;
    use crate::algorithm::pipnn::hash_prune::{HashPruneReservoir, SketchAccessor, SketchStore};
    use crate::algorithm::pipnn::hash_prune_evidence::{
        HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneMeta,
    };
    use crate::algorithm::pipnn::leaf_build::{
        CompactTopKBatch, LeafProfile, PendingEdge, PendingEdgeSink, flush_compact_topk_batch,
    };
    use crate::algorithm::pipnn::online_suffix_hashprune::{
        OnlineSuffixPipeline, OnlineSuffixReduceConfig,
    };
    use crate::algorithm::pipnn::point_store::{InmemDatasetPointStore, PointStore};
    use crate::algorithm::pipnn::rbc_partition::GemmProfile;
    use crate::algorithm::pipnn::spill_hashprune::{SpillReduceStats, SpillRunStats};
    use crate::common::{AnnError, AnnResult, Metric};
    use crate::model::InmemDataset;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    #[test]
    fn final_emit_workers_start_at_eight_and_scale_only_when_finalize_dominates() {
        assert_eq!(super::initial_final_emit_workers(64), 8);
        assert_eq!(super::initial_final_emit_workers(4), 4);
        assert_eq!(
            super::next_final_emit_workers(
                8,
                64,
                Duration::from_millis(125),
                Duration::from_millis(100),
            ),
            16
        );
        assert_eq!(
            super::next_final_emit_workers(
                8,
                64,
                Duration::from_millis(124),
                Duration::from_millis(100),
            ),
            8
        );
        assert_eq!(
            super::next_final_emit_workers(
                64,
                64,
                Duration::from_secs(2),
                Duration::from_millis(1),
            ),
            64
        );
    }

    fn resident_emit_test_reservoirs() -> Vec<Mutex<HashPruneReservoir>> {
        (0..8)
            .map(|row| {
                let mut reservoir = HashPruneReservoir::new(4);
                for offset in 0..row.min(4) {
                    reservoir.insert(
                        1_000 + row as u32 * 10 + offset as u32,
                        (row * 7 + offset) as u16,
                        row as f32 + offset as f32 * 0.25,
                    );
                }
                Mutex::new(reservoir)
            })
            .collect()
    }

    fn resident_emit_test_flat_store()
    -> crate::algorithm::pipnn::flat_resident_reservoir::FlatResidentReservoirStore {
        let store =
            crate::algorithm::pipnn::flat_resident_reservoir::FlatResidentReservoirStore::new_for_test(
                8, 4, 3, 4,
            )
            .unwrap();
        for row in 0..8 {
            store
                .with_reservoir(row, |reservoir| {
                    for offset in 0..row.min(4) {
                        reservoir.insert(
                            1_000 + row as u32 * 10 + offset as u32,
                            (row * 7 + offset) as u16,
                            row as f32 + offset as f32 * 0.25,
                        );
                    }
                    Ok(())
                })
                .unwrap();
        }
        store
    }

    #[test]
    fn parallel_resident_emit_matches_serial_rows_and_profile() {
        let serial_reservoirs = resident_emit_test_reservoirs();
        let parallel_reservoirs = resident_emit_test_reservoirs();
        let mut serial_rows = Vec::new();
        let mut parallel_rows = Vec::new();

        let serial = super::emit_resident_reservoirs_serial(
            17,
            serial_reservoirs,
            super::HashPruneBackfillPolicy::off(),
            None,
            None,
            &mut |source, row| {
                serial_rows.push((source, row));
                Ok(())
            },
        )
        .unwrap();
        let parallel = super::emit_resident_reservoirs_parallel_with_config(
            17,
            parallel_reservoirs,
            super::HashPruneBackfillPolicy::off(),
            super::ResidentEmitConfig::for_test(3, 2, 4, 4),
            &mut |source, row| {
                parallel_rows.push((source, row));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(parallel_rows, serial_rows);
        assert_eq!(parallel.backfill, serial.backfill);
        assert_eq!(parallel.telemetry.rows, parallel_rows.len());
        assert!(parallel.telemetry.max_completed_inflight <= 2);
    }

    #[test]
    fn flat_final_emit_matches_legacy_rows_and_profile() {
        let mut legacy_rows = Vec::new();
        let mut flat_rows = Vec::new();
        let config = super::ResidentEmitConfig::for_test(3, 2, 4, 4);

        let legacy = super::emit_resident_reservoirs_parallel_with_config(
            17,
            resident_emit_test_reservoirs(),
            super::HashPruneBackfillPolicy::off(),
            config,
            &mut |source, row| {
                legacy_rows.push((source, row));
                Ok(())
            },
        )
        .unwrap();
        let flat = super::emit_flat_resident_reservoirs_parallel_with_config(
            17,
            resident_emit_test_flat_store(),
            super::HashPruneBackfillPolicy::off(),
            config,
            &mut |source, row| {
                flat_rows.push((source, row));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(flat_rows, legacy_rows);
        assert_eq!(flat.backfill, legacy.backfill);
        assert_eq!(flat.telemetry.rows, legacy.telemetry.rows);
        assert!(flat.telemetry.max_completed_inflight <= 2);
    }

    #[test]
    fn parallel_resident_emit_stops_at_callback_error_and_joins_producer() {
        let mut emitted = Vec::new();
        let error = super::emit_resident_reservoirs_parallel_with_config(
            100,
            resident_emit_test_reservoirs(),
            super::HashPruneBackfillPolicy::off(),
            super::ResidentEmitConfig::for_test(2, 2, 4, 4),
            &mut |source, row| {
                if source == 103 {
                    return Err(AnnError::log_index_error(
                        "injected emit failure".to_string(),
                    ));
                }
                emitted.push((source, row));
                Ok(())
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("injected emit failure"));
        assert_eq!(
            emitted
                .iter()
                .map(|(source, _)| *source)
                .collect::<Vec<_>>(),
            vec![100, 101, 102]
        );
    }

    #[test]
    fn reservoir_init_config_defaults_to_eight_and_clamps_to_host() {
        assert_eq!(
            super::parse_reservoir_initialization_config(None, 64),
            super::ReservoirInitializationConfig {
                requested: "<unset>".to_string(),
                overlap_enabled: true,
                workers: 8,
                invalid_override: false,
            }
        );
        assert_eq!(
            super::parse_reservoir_initialization_config(None, 4),
            super::ReservoirInitializationConfig {
                requested: "<unset>".to_string(),
                overlap_enabled: true,
                workers: 4,
                invalid_override: false,
            }
        );
    }

    #[test]
    fn reservoir_init_config_accepts_fixed_override_and_clamps_to_host() {
        assert_eq!(
            super::parse_reservoir_initialization_config(Some(OsStr::new("3")), 64),
            super::ReservoirInitializationConfig {
                requested: "3".to_string(),
                overlap_enabled: true,
                workers: 3,
                invalid_override: false,
            }
        );
        assert_eq!(
            super::parse_reservoir_initialization_config(Some(OsStr::new("128")), 64),
            super::ReservoirInitializationConfig {
                requested: "128".to_string(),
                overlap_enabled: true,
                workers: 64,
                invalid_override: false,
            }
        );
    }

    #[test]
    fn reservoir_init_config_zero_restores_d1_blocking_mode() {
        assert_eq!(
            super::parse_reservoir_initialization_config(Some(OsStr::new("0")), 64),
            super::ReservoirInitializationConfig {
                requested: "0".to_string(),
                overlap_enabled: false,
                workers: 1,
                invalid_override: false,
            }
        );
    }

    #[test]
    fn reservoir_init_config_invalid_value_uses_safe_d1_fallback() {
        assert_eq!(
            super::parse_reservoir_initialization_config(Some(OsStr::new("bad")), 64),
            super::ReservoirInitializationConfig {
                requested: "<invalid>".to_string(),
                overlap_enabled: false,
                workers: 1,
                invalid_override: true,
            }
        );
    }

    #[test]
    fn resident_reservoir_layout_parses_flat_and_legacy() {
        assert_eq!(
            super::ResidentReservoirLayout::from_override(None).unwrap(),
            super::ResidentReservoirLayout::Flat
        );
        assert_eq!(
            super::ResidentReservoirLayout::from_override(Some(OsStr::new("flat"))).unwrap(),
            super::ResidentReservoirLayout::Flat
        );
        assert_eq!(
            super::ResidentReservoirLayout::from_override(Some(OsStr::new("legacy"))).unwrap(),
            super::ResidentReservoirLayout::Legacy
        );
        assert!(super::ResidentReservoirLayout::from_override(Some(OsStr::new("bad"))).is_err());
    }

    #[test]
    fn resident_reservoir_layout_allocates_flat_and_preserves_legacy_override() {
        let flat =
            super::DeferredReservoirs::new(4, 8).with_layout(super::ResidentReservoirLayout::Flat);
        flat.initialize().unwrap();
        let flat = flat.into_allocation().unwrap();
        assert_eq!(
            flat.reservoirs.layout(),
            super::ResidentReservoirLayout::Flat
        );
        assert_eq!(
            flat.estimated_bytes,
            super::estimate_flat_reservoir_bytes(4, 8)
        );
        assert_eq!(flat.initialization.layout, "flat");
        assert_eq!(flat.initialization.segment_count, 1);
        assert_eq!(
            flat.initialization.stripe_count,
            super::flat_resident_reservoir::DEFAULT_FLAT_RESERVOIR_STRIPES
        );
        assert!(flat.initialization.mapped_bytes > 0);
        assert!(flat.initialization.scratch_bytes > 0);

        let legacy = super::DeferredReservoirs::new(4, 8)
            .with_layout(super::ResidentReservoirLayout::Legacy);
        legacy.initialize().unwrap();
        let legacy = legacy.into_allocation().unwrap();
        assert_eq!(
            legacy.reservoirs.layout(),
            super::ResidentReservoirLayout::Legacy
        );
        assert_eq!(
            legacy.estimated_bytes,
            super::estimate_reservoir_bytes(4, 8)
        );
        assert_eq!(legacy.initialization.layout, "legacy");
        assert_eq!(legacy.initialization.segment_count, 0);
        assert_eq!(legacy.initialization.stripe_count, 0);
    }

    #[test]
    fn flat_reservoir_budget_admission_accounts_for_fixed_scratch_once() {
        let l_max = 8;
        let minimum_admitted = 37;
        let budget = super::estimate_flat_reservoir_bytes(minimum_admitted, l_max);

        let admitted = super::resident_reservoir_points_for_budget(
            100,
            l_max,
            budget,
            super::ResidentReservoirLayout::Flat,
        );
        assert!(admitted >= minimum_admitted);
        assert!(super::estimate_flat_reservoir_bytes(admitted, l_max) <= budget);
        if admitted < 100 {
            assert!(super::estimate_flat_reservoir_bytes(admitted + 1, l_max) > budget);
        }
        assert_eq!(
            super::resident_reservoir_points_for_budget(
                100,
                l_max,
                super::estimate_flat_reservoir_bytes(0, l_max).saturating_sub(1),
                super::ResidentReservoirLayout::Flat,
            ),
            0
        );
    }

    #[cfg(unix)]
    #[test]
    fn reservoir_init_config_non_utf8_value_uses_safe_d1_fallback() {
        use std::os::unix::ffi::OsStringExt;

        let raw = std::ffi::OsString::from_vec(vec![0xff]);
        assert_eq!(
            super::parse_reservoir_initialization_config(Some(raw.as_os_str()), 64),
            super::ReservoirInitializationConfig {
                requested: "<non-utf8>".to_string(),
                overlap_enabled: false,
                workers: 1,
                invalid_override: true,
            }
        );
    }

    struct RecordingPendingEdgeSink {
        label: &'static str,
        calls: std::sync::Arc<Mutex<Vec<&'static str>>>,
    }

    impl PendingEdgeSink for RecordingPendingEdgeSink {
        fn flush_pending_edges(
            &self,
            edges: &mut Vec<PendingEdge>,
        ) -> AnnResult<crate::algorithm::pipnn::leaf_build::PendingEdgeFlushProfile> {
            self.calls.lock().push(self.label);
            edges.clear();
            Ok(Default::default())
        }
    }

    #[test]
    fn hybrid_sink_submits_suffix_before_resident_prefix() {
        let calls = std::sync::Arc::new(Mutex::new(Vec::new()));
        let resident = RecordingPendingEdgeSink {
            label: "resident",
            calls: std::sync::Arc::clone(&calls),
        };
        let suffix = RecordingPendingEdgeSink {
            label: "suffix",
            calls: std::sync::Arc::clone(&calls),
        };
        let sink = super::HybridEdgeSink::new_with_policies(
            2,
            &resident,
            &suffix,
            HashPruneEvidencePolicy::off(),
            super::HashPruneBackfillPolicy::off(),
        );
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 3,
                c: 11,
                hash: 2,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];
        sink.flush_pending_edges(&mut edges).unwrap();
        assert_eq!(&*calls.lock(), &["suffix", "resident"]);
    }

    fn online_suffix_test_config(
        root: &std::path::Path,
        source_start: usize,
        source_end: usize,
    ) -> OnlineSuffixReduceConfig {
        let shard_points = 2;
        OnlineSuffixReduceConfig {
            source_start,
            source_end,
            shard_points,
            l_max: 4,
            direct_queue_bytes: 1024 * 1024,
            spill_buffer_bytes: 4096,
            reorder_window_bytes: 1024 * 1024,
            reducer_workers: 2,
            mode: PiPNNOnlineSuffixReduceMode::DirectFirst,
            state_cache_bytes: 0,
            read_window_bytes: 0,
            pinned_prefix_points: 0,
            delta_budget_bytes: 1024 * 1024,
            delta_buffer_bytes: 4096,
            compaction_bytes: 64 * 1024 * 1024,
            range_points: (source_end - source_start).max(1),
            range_trigger_bytes: 4096,
            max_runs_per_range: 4,
            compaction_workers: 1,
            drain_compaction_workers: 1,
            cache_hot_updates: 2,
            store: DiskReservoirStoreConfig {
                root: root.join("online-state"),
                source_start,
                source_end,
                shard_points,
                l_max: 4,
                page_cache_bytes: 8192,
                writeback_bytes: 4096,
                decode_bytes: 4096,
                writeback_workers: 2,
                overprovision_ratio: 2.0,
                io: DirectIoConfig::disabled(),
                policy_key: [0; 32],
                backfill_policy: super::HashPruneBackfillPolicy::off(),
            },
            evidence_policy: HashPruneEvidencePolicy::off(),
            backfill_policy: super::HashPruneBackfillPolicy::off(),
            diagnostic_collector: None,
        }
    }

    fn online_suffix_test_edges(num_points: usize) -> Vec<PendingEdge> {
        (0..num_points)
            .map(|source| PendingEdge {
                p: source,
                c: 100 + source as u32,
                hash: source as u16,
                dist: source as f32 + 1.0,
                meta: HashPruneMeta::empty(),
            })
            .collect()
    }

    #[test]
    fn writeback_cache_maps_resident_budget_to_a_pinned_full_range_pipeline() {
        let mut params = PiPNNParams::default();
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::WritebackCache;

        let plan = super::plan_online_suffix_residency(&params, 100, 40, true).unwrap();

        assert_eq!(plan.pipeline_source_start, 0);
        assert_eq!(plan.pinned_prefix_points, 40);
        assert_eq!(plan.dense_resident_points, 0);
    }

    #[test]
    fn log_assisted_auto_reducer_workers_do_not_fragment_source_ownership() {
        let mut params = PiPNNParams::default();
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::LogAssisted;
        assert_eq!(super::effective_online_suffix_reduce_workers(&params, 8), 4);

        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::DirectFirst;
        assert_eq!(super::effective_online_suffix_reduce_workers(&params, 8), 4);

        params.online_suffix_reduce_workers = 6;
        assert_eq!(super::effective_online_suffix_reduce_workers(&params, 8), 6);
    }

    #[test]
    fn log_assisted_wide_base_falls_back_before_reserving_online_memory() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 72 * 1024 * 1024 * 1024;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::LogAssisted;
        params.online_suffix_delta_budget_bytes = 8 * 1024 * 1024 * 1024;

        assert_eq!(
            super::online_suffix_eligibility_for_dataset(&params, 10_000_000),
            super::OnlineSuffixEligibility::Enabled
        );
        assert_eq!(
            super::online_suffix_eligibility_for_dataset(&params, 35_000_000),
            super::OnlineSuffixEligibility::ExternalFallback(
                "log_assisted_base_working_set_exceeds_delta_budget"
            )
        );

        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, 35_000_000, 1024);
        assert!(!plan.online_suffix_enabled);
        assert_eq!(plan.online_suffix_budget_bytes, 0);

        params.online_suffix_delta_budget_bytes = 32 * 1024 * 1024 * 1024;
        assert_eq!(
            super::online_suffix_eligibility_for_dataset(&params, 35_000_000),
            super::OnlineSuffixEligibility::Enabled
        );
    }

    #[test]
    fn online_suffix_hybrid_emits_every_source_once_in_order() {
        let dir = tempdir().unwrap();
        let num_points = 8;
        let resident_points = 3;
        let resident = super::DeferredReservoirs::new(resident_points, 4);
        let pipeline = OnlineSuffixPipeline::start(online_suffix_test_config(
            dir.path(),
            resident_points,
            num_points,
        ))
        .unwrap();
        let sink = super::HybridEdgeSink::new_with_policies(
            resident_points,
            &resident,
            &pipeline,
            HashPruneEvidencePolicy::off(),
            super::HashPruneBackfillPolicy::off(),
        );
        sink.prepare_for_d1().unwrap();
        let mut edges = online_suffix_test_edges(num_points);
        sink.flush_pending_edges(&mut edges).unwrap();
        pipeline.mark_gpu_complete();
        drop(sink);

        let state = pipeline.finish().unwrap();
        let allocation = resident.into_allocation().unwrap();
        let mut emitted_sources = Vec::new();
        super::emit_resident_reservoirs(
            0,
            allocation.reservoirs.into_legacy_for_test(),
            super::HashPruneBackfillPolicy::off(),
            None,
            None,
            &mut |source, _neighbors| {
                emitted_sources.push(source);
                Ok(())
            },
        )
        .unwrap();
        state
            .emit_rows(1, None, None, &mut |source, _neighbors| {
                emitted_sources.push(source);
                Ok(())
            })
            .unwrap();

        assert_eq!(emitted_sources, (0..num_points as u32).collect::<Vec<_>>());
        assert_eq!(
            emitted_sources
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            num_points
        );
        state.cleanup().unwrap();
    }

    #[test]
    fn online_suffix_output_error_preserves_state_artifacts() {
        let dir = tempdir().unwrap();
        let num_points = 8;
        let resident_points = 3;
        let state_dir = dir.path().join("online-state");
        let resident = super::DeferredReservoirs::new(resident_points, 4);
        let pipeline = OnlineSuffixPipeline::start(online_suffix_test_config(
            dir.path(),
            resident_points,
            num_points,
        ))
        .unwrap();
        let sink = super::HybridEdgeSink::new_with_policies(
            resident_points,
            &resident,
            &pipeline,
            HashPruneEvidencePolicy::off(),
            super::HashPruneBackfillPolicy::off(),
        );
        sink.prepare_for_d1().unwrap();
        let mut edges = online_suffix_test_edges(num_points);
        sink.flush_pending_edges(&mut edges).unwrap();
        pipeline.mark_gpu_complete();
        drop(sink);

        let state = pipeline.finish().unwrap();
        let allocation = resident.into_allocation().unwrap();
        let mut output = |source, _neighbors| {
            if source == 5 {
                return Err(AnnError::log_index_error(
                    "injected output failure".to_string(),
                ));
            }
            Ok(())
        };
        super::emit_resident_reservoirs(
            0,
            allocation.reservoirs.into_legacy_for_test(),
            super::HashPruneBackfillPolicy::off(),
            None,
            None,
            &mut output,
        )
        .unwrap();
        let error = state.emit_rows(1, None, None, &mut output).unwrap_err();

        assert!(error.to_string().contains("injected output failure"));
        assert!(state_dir.exists());
        assert!(fs::read_dir(&state_dir).unwrap().next().is_some());
    }

    #[test]
    fn prune_backend_lrpf_skips_sketch_candidate_spill_path() {
        let mut params = PiPNNParams::default();
        params.prune_backend = PiPNNPruneBackend::LRPF;

        assert!(super::uses_lrpf_candidate_backend(&params));
        assert!(!super::uses_hashprune_candidate_backend(&params));
    }

    #[test]
    fn external_reduce_config_uses_effective_build_threads() {
        let config = super::external_reduce_config(DirectIoConfig::enabled_with_alignment(4096), 8);

        assert!(config.io.enabled);
        assert_eq!(config.workers, 8);
        assert_eq!(config.batch_records, 64 * 1024);
    }

    #[test]
    fn default_external_and_wide_fallback_offer_idle_host_cpus_without_forcing_them() {
        let wide_fallback = super::OnlineSuffixEligibility::ExternalFallback(
            "log_assisted_base_working_set_exceeds_delta_budget",
        );
        let unrelated_fallback =
            super::OnlineSuffixEligibility::ExternalFallback("partitioner_not_gpu_subtree");

        assert_eq!(
            super::external_reduce_worker_bounds(8, super::OnlineSuffixEligibility::Disabled, 64),
            (8, 64)
        );
        assert_eq!(
            super::external_reduce_worker_bounds(8, super::OnlineSuffixEligibility::Enabled, 64),
            (8, 8)
        );
        assert_eq!(
            super::external_reduce_worker_bounds(8, wide_fallback, 64),
            (8, 64)
        );
        assert_eq!(
            super::external_reduce_worker_bounds(8, unrelated_fallback, 64),
            (8, 8)
        );
        assert_eq!(
            super::external_reduce_worker_bounds(8, wide_fallback, 4),
            (8, 8)
        );
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn unset(key: &'static str) -> Self {
            let original = env::var(key).ok();
            unsafe {
                env::remove_var(key);
            }
            Self { key, original }
        }

        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var(key).ok();
            unsafe {
                env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => env::set_var(self.key, value),
                    None => env::remove_var(self.key),
                }
            }
        }
    }

    fn assert_json_u64(value: &Value, path: &[&str], expected: u64) {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .unwrap_or_else(|| panic!("missing JSON path {path:?} at key {key}"));
        }
        assert_eq!(
            current.as_u64(),
            Some(expected),
            "unexpected JSON value at {path:?}: {current}"
        );
    }

    #[cfg(feature = "pipnn_cuda")]
    fn json_u64(value: &Value, path: &[&str]) -> u64 {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .unwrap_or_else(|| panic!("missing JSON path {path:?} at key {key}"));
        }
        current
            .as_u64()
            .unwrap_or_else(|| panic!("expected u64 JSON value at {path:?}: {current}"))
    }

    fn assert_json_f64(value: &Value, path: &[&str], expected: f64) {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .unwrap_or_else(|| panic!("missing JSON path {path:?} at key {key}"));
        }
        let actual = current
            .as_f64()
            .unwrap_or_else(|| panic!("expected numeric JSON value at {path:?}: {current}"));
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "unexpected JSON value at {path:?}: expected {expected}, got {actual}"
        );
    }

    fn assert_json_str(value: &Value, path: &[&str], expected: &str) {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .unwrap_or_else(|| panic!("missing JSON path {path:?} at key {key}"));
        }
        assert_eq!(
            current
                .as_str()
                .unwrap_or_else(|| panic!("expected string JSON value at {path:?}: {current}")),
            expected
        );
    }

    fn assert_json_bool(value: &Value, path: &[&str], expected: bool) {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .unwrap_or_else(|| panic!("missing JSON path {path:?} at key {key}"));
        }
        assert_eq!(
            current
                .as_bool()
                .unwrap_or_else(|| panic!("expected bool JSON value at {path:?}: {current}")),
            expected
        );
    }

    fn minimal_oom_profile_for_json(temp_files: super::TempFileReport) -> super::OomProfile {
        super::OomProfile {
            num_points: 1,
            dim: 1,
            leaf_knn: 2,
            partitioner: PiPNNPartitionerKind::Rbc,
            root_assignment_backend: PiPNNComputeBackendKind::Cpu,
            assignment_backend: PiPNNGpuSubtreeAssignmentBackend::Current,
            d0_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            d0_dco_survivors: 0,
            d0_dco_audit: false,
            d0_sq_cpu_kernel: SqInt8CpuKernel::Auto,
            d1plus_lazy_split_mode: super::PiPNNGpuSubtreeD1PlusLazySplitMode::On,
            assignment_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            assignment_dco_proj_dim: 0,
            assignment_dco_survivors: 0,
            assignment_dco_audit: false,
            leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend::PersistentCustom,
            leaf_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            leaf_dco_proj_dim: 0,
            leaf_dco_survivors: 0,
            leaf_dco_audit: false,
            leaf_dco_ddc_res_pca_matrix: None,
            leaf_dco_ddc_res_projected_input: false,
            leaf_dco_ddc_res_base_dim: 32,
            leaf_dco_ddc_res_sigma_count: 3.0,
            rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode::Off,
            rabitq_plus_bits: 8,
            rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            rabitq_plus_seed: 42,
            rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode::Off,
            rabitq_plus_sidecar_prefix: None,
            rabitq_plus_fused_sidecar_sketch_enabled: false,
            rabitq_plus_fused_prestage_total_wall: Duration::ZERO,
            rabitq_plus_fused_chunk_alloc_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_raw_read_wall: Duration::ZERO,
            rabitq_plus_fused_source_hash_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_quantize_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_code_output_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_norm_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_meta_output_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_header_write_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_write_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_alloc_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_compute_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_extend_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_write_wall: Duration::ZERO,
            qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode::Off,
            qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident,
            qcode_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma,
            qcode_rabitq_short_survivors: None,
            rabitq_residual_centroids: 16,
            qcode_staging_upload_shard_rows: 524_288,
            qcode_global_d0_leaders: 0,
            sq_staging_mode: PiPNNGpuSubtreeSqStagingMode::Off,
            sq_staging_upload_shard_rows: 524_288,
            sq_bits: SQ_INT8_DEFAULT_BITS,
            sq_native_int4: false,
            sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode::Off,
            persistent_large_ticket_batch_requested: PiPNNGpuSubtreePersistentLargeTicketBatch::One,
            persistent_sq8_assignment_kernel_requested:
                PiPNNGpuSubtreeSq8AssignmentKernel::WarpDp4a,
            persistent_sq8_leaf_executor_requested: PiPNNGpuSubtreeSq8LeafExecutor::Auto,
            persistent_sq8_leaf_mode_requested: PiPNNGpuSubtreeSq8LeafMode::Staged,
            persistent_scheduler_mode_requested: PiPNNGpuSubtreePersistentSchedulerMode::Baseline,
            persistent_scheduler_mode_actual: PiPNNGpuSubtreePersistentSchedulerMode::Baseline,
            persistent_idle_backoff_ns_requested: 350,
            persistent_idle_backoff_ns_actual: 0,
            sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode::Off,
            sq_wmma_sidecar_prefix: None,
            sq_staging_fused_sketch_enabled: false,
            sq_staging_fused_sketch_compute_wall: Duration::ZERO,
            sq_staging_fused_sketch_join_wait_wall: Duration::ZERO,
            turboquant_mode: PiPNNGpuSubtreeTurboQuantMode::Off,
            turboquant_bits: 4,
            turboquant_seed: 42,
            turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode::Off,
            turboquant_sidecar_prefix: None,
            turboquant_cascade_low_dim: 128,
            turboquant_cascade_survivor_cap: 0,
            turboquant_adaptive_survivor_threshold: 0,
            turboquant_adaptive_survivor_low_cap: 0,
            memory_budget_bytes: 1,
            memory_budget_source: "explicit",
            oom_io_mode: PiPNNOomIoMode::Buffered,
            oom_reservoir_budget_share: 1.0,
            oom_sidecar_resident_budget_cap_bytes: None,
            effective_threads: 1,
            pool_threads: 1,
            leaf_budget_threads: 1,
            resident_reservoir_points: 0,
            resident_reservoir_bytes: 0,
            reservoir_allocation_wall: Duration::ZERO,
            reservoir_initialization: super::ReservoirInitializationProfile::default(),
            reduce_shard_points: 1,
            compact_spill_digest_enabled: false,
            compact_spill_digest_width: 0,
            prune_backend: PiPNNPruneBackend::HashPrune,
            hashprune_evidence_policy: super::HashPruneEvidencePolicy::off(),
            hashprune_backfill_policy: super::HashPruneBackfillPolicy::off(),
            hashprune_backfill_degree_cap: 64,
            lrpf_target_degree: 64,
            lrpf_long_edge_quota: 0,
            lrpf_pre_spill_compress_enabled: false,
            lrpf_pre_spill_width: 0,
            lrpf_candidate_records: 0,
            lrpf_witness_records: 0,
            lrpf_fragments: 0,
            lrpf_pre_spill_pruned_candidates: 0,
            lrpf_reducer_pruned_candidates: 0,
            reservoir_budget_bytes: 0,
            resident_subtree_budget_bytes: 0,
            required_resident_bytes: 0,
            optional_resident_budget_bytes: 0,
            optional_resident_admitted_bytes: 0,
            optional_resident_unassigned_bytes: 0,
            transient_pipeline_budget_bytes: 0,
            d1plus_working_set_budget_bytes: 0,
            sidecar_resident_kind: super::OomSidecarResidentKind::None,
            sidecar_resident_budget_bytes: 0,
            sidecar_resident_required_bytes: 0,
            sidecar_resident_admitted_rows: 0,
            sidecar_resident_admissible: false,
            scratch_budget_bytes: 0,
            io_headroom_bytes: 0,
            raw_vector_cache_budget_bytes: 0,
            raw_vector_cache_rows: 0,
            online_suffix_mode: super::PiPNNOnlineSuffixReduceMode::Off,
            online_suffix_fallback_reason: None,
            online_suffix_config: super::OnlineSuffixProfileConfig::default(),
            online_suffix: super::OnlineSuffixReduceStats::default(),
            sketch_bytes: 0,
            sketch_file_bytes: 0,
            spill: super::SpillRunStats::default(),
            reduce: super::SpillReduceStats::default(),
            sketch_persist_wall: Duration::ZERO,
            streaming_wall: Duration::ZERO,
            reduce_wall: Duration::ZERO,
            emit_wall: Duration::ZERO,
            resident_emit: super::ResidentEmitProfile::default(),
            total_wall: Duration::ZERO,
            scheduler: super::SchedulerTelemetry::default(),
            rbc: super::RbcPartitionTelemetry::default(),
            kmeans: super::KMeansPartitionTelemetry::default(),
            hashprune_evidence: super::HashPruneEvidenceProfile::default(),
            hashprune_backfill: super::HashPruneBackfillProfile::default(),
            leaf: super::LeafProfile::default(),
            memory: super::OomMemoryReport::default(),
            temp_files,
            uring_batches: 0,
            uring_windows: 0,
            uring_bytes: 0,
            uring_fallbacks: 0,
        }
    }

    fn build_test_dataset(num_points: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        for point in 0..num_points {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = point as f32 * 0.25 + axis as f32 * 0.5;
            }
        }
        dataset
    }

    fn write_test_fbin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut bytes = Vec::with_capacity(8 + rows * dim * std::mem::size_of::<f32>());
        bytes.extend_from_slice(&(rows as u32).to_le_bytes());
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        for row in 0..rows {
            for axis in 0..dim {
                let value = row as f32 * 0.25 + axis as f32 * 0.5;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, bytes).unwrap();
    }

    fn write_test_u8bin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut bytes = Vec::with_capacity(8 + rows * dim);
        bytes.extend_from_slice(&(rows as u32).to_le_bytes());
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        for row in 0..rows {
            for axis in 0..dim {
                bytes.push(((row * 17 + axis * 29) % 256) as u8);
            }
        }
        fs::write(path, bytes).unwrap();
    }

    fn collect_graph(
        dataset: &InmemDataset<f32>,
        num_points: usize,
        params: &PiPNNParams,
    ) -> Vec<Vec<u32>> {
        let mut graph = vec![Vec::new(); num_points];
        super::build_pipnn_graph(dataset, num_points, 1, params, |point, neighbors| {
            graph[point as usize] = neighbors;
            Ok(())
        })
        .unwrap();

        for neighbors in &mut graph {
            neighbors.sort_unstable();
        }
        graph
    }

    #[cfg(feature = "pipnn_cuda")]
    fn graph_edge_recall(candidate: &[Vec<u32>], baseline: &[Vec<u32>]) -> f64 {
        let mut expected_edges = 0_usize;
        let mut matched_edges = 0_usize;
        for (candidate_neighbors, baseline_neighbors) in candidate.iter().zip(baseline) {
            expected_edges += baseline_neighbors.len();
            for &neighbor in baseline_neighbors {
                if candidate_neighbors.binary_search(&neighbor).is_ok() {
                    matched_edges += 1;
                }
            }
        }
        if expected_edges == 0 {
            return 1.0;
        }
        matched_edges as f64 / expected_edges as f64
    }

    struct FakeCudaLeafOnlyBackend;

    impl PiPNNComputeBackend for FakeCudaLeafOnlyBackend {
        fn kind(&self) -> PiPNNComputeBackendKind {
            PiPNNComputeBackendKind::Cuda
        }

        fn assign_clusters_l2(
            &self,
            _dataset: &dyn PointStore,
            _cur: &[u32],
            _leaders: &[u32],
            _local_fanout: usize,
            _memory_budget_bytes: Option<usize>,
        ) -> AnnResult<(Vec<Vec<u32>>, GemmProfile)> {
            Err(AnnError::log_index_error(
                "leaf CUDA backend must not be used for RBC assignment".to_string(),
            ))
        }

        fn process_leaf_parallel_large_l2(
            &self,
            _dataset: &dyn PointStore,
            _sketches: &dyn SketchAccessor,
            _leaf: &[u32],
            _params: &PiPNNParams,
            _edge_sink: &dyn PendingEdgeSink,
            _block_rows: usize,
        ) -> AnnResult<LeafProfile> {
            Err(AnnError::log_index_error(
                "leaf CUDA compact path should not use large-leaf full processing".to_string(),
            ))
        }

        fn compute_leaf_topk_l2(
            &self,
            dataset: &dyn PointStore,
            leaf: &[u32],
            params: &PiPNNParams,
            block_rows: usize,
        ) -> AnnResult<(CompactTopKBatch, LeafProfile)> {
            let _block_rows = block_rows;
            super::leaf_build::compute_leaf_topk_l2_cpu_shadow(dataset, leaf, params)
        }
    }

    struct CountingCudaBackend {
        total_points: usize,
        root_sized_assignment_calls: AtomicUsize,
        subtree_assignment_calls: AtomicUsize,
    }

    impl CountingCudaBackend {
        fn new(total_points: usize) -> Self {
            Self {
                total_points,
                root_sized_assignment_calls: AtomicUsize::new(0),
                subtree_assignment_calls: AtomicUsize::new(0),
            }
        }

        fn root_sized_assignment_calls(&self) -> usize {
            self.root_sized_assignment_calls.load(Ordering::Relaxed)
        }

        fn subtree_assignment_calls(&self) -> usize {
            self.subtree_assignment_calls.load(Ordering::Relaxed)
        }
    }

    impl PiPNNComputeBackend for CountingCudaBackend {
        fn kind(&self) -> PiPNNComputeBackendKind {
            PiPNNComputeBackendKind::Cuda
        }

        fn assign_clusters_l2(
            &self,
            dataset: &dyn PointStore,
            cur: &[u32],
            leaders: &[u32],
            local_fanout: usize,
            memory_budget_bytes: Option<usize>,
        ) -> AnnResult<(Vec<Vec<u32>>, GemmProfile)> {
            if cur.len() == self.total_points {
                self.root_sized_assignment_calls
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.subtree_assignment_calls
                    .fetch_add(1, Ordering::Relaxed);
            }
            super::compute_backend::cpu_backend().assign_clusters_l2(
                dataset,
                cur,
                leaders,
                local_fanout,
                memory_budget_bytes,
            )
        }

        fn process_leaf_parallel_large_l2(
            &self,
            dataset: &dyn PointStore,
            sketches: &dyn SketchAccessor,
            leaf: &[u32],
            params: &PiPNNParams,
            edge_sink: &dyn PendingEdgeSink,
            block_rows: usize,
        ) -> AnnResult<LeafProfile> {
            super::compute_backend::cpu_backend().process_leaf_parallel_large_l2(
                dataset, sketches, leaf, params, edge_sink, block_rows,
            )
        }

        fn compute_leaf_topk_l2(
            &self,
            dataset: &dyn PointStore,
            leaf: &[u32],
            params: &PiPNNParams,
            block_rows: usize,
        ) -> AnnResult<(CompactTopKBatch, LeafProfile)> {
            let _block_rows = block_rows;
            super::leaf_build::compute_leaf_topk_l2_cpu_shadow(dataset, leaf, params)
        }
    }

    struct FakeCompactTopKSubtreeBackend {
        compact_subtree_calls: AtomicUsize,
        leaf_topk_calls: AtomicUsize,
        large_leaf_calls: AtomicUsize,
        drop_last_point: bool,
        row_repetitions: usize,
    }

    impl FakeCompactTopKSubtreeBackend {
        fn new() -> Self {
            Self {
                compact_subtree_calls: AtomicUsize::new(0),
                leaf_topk_calls: AtomicUsize::new(0),
                large_leaf_calls: AtomicUsize::new(0),
                drop_last_point: false,
                row_repetitions: 1,
            }
        }

        fn incomplete_output() -> Self {
            Self {
                drop_last_point: true,
                ..Self::new()
            }
        }

        fn overlapping_rows(row_repetitions: usize) -> Self {
            Self {
                row_repetitions: row_repetitions.max(1),
                ..Self::new()
            }
        }

        fn compact_subtree_calls(&self) -> usize {
            self.compact_subtree_calls.load(Ordering::Relaxed)
        }

        fn leaf_topk_calls(&self) -> usize {
            self.leaf_topk_calls.load(Ordering::Relaxed)
        }

        fn large_leaf_calls(&self) -> usize {
            self.large_leaf_calls.load(Ordering::Relaxed)
        }

        fn compact_result_for_request(
            &self,
            request: &super::compute_backend::GpuSubtreePartitionRequest<'_>,
            params: &PiPNNParams,
        ) -> AnnResult<super::compute_backend::GpuSubtreeTopKResult> {
            self.compact_subtree_calls.fetch_add(1, Ordering::Relaxed);
            let mut point_ids = request.point_ids.to_vec();
            if self.drop_last_point {
                point_ids.pop();
            }
            if self.row_repetitions > 1 {
                let unique_points = point_ids;
                point_ids = Vec::with_capacity(unique_points.len() * self.row_repetitions);
                for _ in 0..self.row_repetitions {
                    point_ids.extend_from_slice(&unique_points);
                }
            }
            let k = params.leaf_knn.min(point_ids.len().saturating_sub(1));
            let rows = point_ids.len();
            let mut batch = CompactTopKBatch {
                leaf_id: self.compact_subtree_calls() as u64,
                rows,
                point_ids,
                k,
                topk_ids: vec![u32::MAX; rows.saturating_mul(k)],
                topk_dists: vec![f32::INFINITY; rows.saturating_mul(k)],
                estimated_dtoh_bytes: request
                    .point_ids
                    .len()
                    .saturating_mul(k)
                    .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                ..CompactTopKBatch::default()
            };
            if k > 0 {
                for row in 0..batch.rows {
                    let neighbor = if row + 1 < batch.rows { row + 1 } else { 0 };
                    batch.topk_ids[row * k] = batch.point_ids[neighbor];
                    batch.topk_dists[row * k] = 1.0;
                }
            }
            Ok(super::compute_backend::GpuSubtreeTopKResult {
                compact_batches: vec![batch],
                assignment_passes: 1,
                leaf_count: 1,
                max_active_points: request.point_ids.len(),
                profile: super::compute_backend::GpuSubtreeTopKProfile {
                    child_points: request.point_ids.len(),
                    d2h_bytes: request
                        .point_ids
                        .len()
                        .saturating_mul(k)
                        .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    persistent_attempt_count: 1,
                    persistent_success_count: 1,
                    persistent_partition_worker_blocks: 16,
                    persistent_topk_worker_blocks: 16,
                    persistent_partition_pops: 1,
                    persistent_leaf_pushes: 1,
                    persistent_topk_pops: 1,
                    persistent_partition_claims: 1,
                    persistent_topk_claims: 1,
                    persistent_done_cas_wins: 1,
                    persistent_leaf_size_min: rows,
                    persistent_leaf_size_max: rows,
                    persistent_leaf_size_sum: rows,
                    persistent_leaf_size_count: 1,
                    persistent_leaf_target: params.c_max,
                    persistent_leaf_max: params.c_max,
                    persistent_queue_overflow_flags: 0,
                    persistent_kernel_event_wall: Duration::from_millis(21),
                    persistent_total_wall: Duration::from_millis(22),
                    persistent_topk_rows: rows,
                    persistent_leaf_count: 1,
                    ..super::compute_backend::GpuSubtreeTopKProfile::default()
                },
            })
        }
    }

    impl PiPNNComputeBackend for FakeCompactTopKSubtreeBackend {
        fn kind(&self) -> PiPNNComputeBackendKind {
            PiPNNComputeBackendKind::Cuda
        }

        fn assign_clusters_l2(
            &self,
            _dataset: &dyn PointStore,
            _cur: &[u32],
            _leaders: &[u32],
            _local_fanout: usize,
            _memory_budget_bytes: Option<usize>,
        ) -> AnnResult<(Vec<Vec<u32>>, GemmProfile)> {
            Err(AnnError::log_index_error(
                "compact gpu-subtree backend must not run D1+ assign fallback".to_string(),
            ))
        }

        fn process_leaf_parallel_large_l2(
            &self,
            _dataset: &dyn PointStore,
            _sketches: &dyn SketchAccessor,
            _leaf: &[u32],
            _params: &PiPNNParams,
            _edge_sink: &dyn PendingEdgeSink,
            _block_rows: usize,
        ) -> AnnResult<LeafProfile> {
            self.large_leaf_calls.fetch_add(1, Ordering::Relaxed);
            Err(AnnError::log_index_error(
                "compact gpu-subtree must bypass large-leaf processing".to_string(),
            ))
        }

        fn compute_leaf_topk_l2(
            &self,
            _dataset: &dyn PointStore,
            _leaf: &[u32],
            _params: &PiPNNParams,
            _block_rows: usize,
        ) -> AnnResult<(CompactTopKBatch, LeafProfile)> {
            self.leaf_topk_calls.fetch_add(1, Ordering::Relaxed);
            Err(AnnError::log_index_error(
                "compact gpu-subtree must bypass scheduler leaf top-k".to_string(),
            ))
        }

        fn supports_subtree_l2_compact_topk(&self) -> bool {
            true
        }

        fn supports_quantized_subtree_l2_compact_topk(&self) -> bool {
            true
        }

        fn quantize_sq8_and_assign_global_d0(
            &self,
            request: super::dco_sq_int8::SqGpuD0QuantizeAssignRequest<'_>,
            params: &PiPNNParams,
        ) -> AnnResult<super::dco_sq_int8::SqGpuD0QuantizeAssignResult> {
            super::compute_backend::cpu_backend().quantize_sq8_and_assign_global_d0(request, params)
        }

        fn process_subtree_l2_compact_topk(
            &self,
            _dataset: &dyn PointStore,
            request: &super::compute_backend::GpuSubtreePartitionRequest<'_>,
            params: &PiPNNParams,
        ) -> AnnResult<super::compute_backend::GpuSubtreeTopKResult> {
            self.compact_result_for_request(request, params)
        }

        fn process_quantized_subtree_l2_compact_topk(
            &self,
            request: &super::compute_backend::GpuSubtreePartitionRequest<'_>,
            _quantized: super::compute_backend::GpuSubtreeQuantizedInput<'_>,
            params: &PiPNNParams,
        ) -> AnnResult<super::compute_backend::GpuSubtreeTopKResult> {
            self.compact_result_for_request(request, params)
        }
    }

    #[test]
    fn streaming_profile_idle_time_saturates_at_zero() {
        let mut profile = super::StreamingProfile::default();
        profile.total_wall = Duration::from_millis(25);
        profile.leaf.total_wall = Duration::from_millis(40);

        assert_eq!(profile.leaf_idle_time(), Duration::ZERO);
    }

    #[test]
    fn scheduler_budget_uses_bounded_leaf_backlog() {
        let budget = super::scheduler::SchedulerBudget::for_worker_count(42, 3500);

        assert_eq!(budget.worker_count, 42);
        assert_eq!(budget.leaf_backlog_capacity, 336);
        assert_eq!(budget.large_leaf_min_size, 7000);
        assert_eq!(budget.leaf_block_rows, 1024);
    }

    #[test]
    fn resident_async_flush_policy_defaults_to_eight_and_host_ceiling() {
        assert_eq!(
            super::parse_gpu_subtree_async_flush_worker_policy(None, 64),
            super::AsyncCompactFlushWorkerPolicy::Adaptive {
                initial_workers: 8,
                max_workers: 64,
                invalid_override: false,
            }
        );
        assert_eq!(
            super::parse_gpu_subtree_async_flush_worker_policy(None, 4),
            super::AsyncCompactFlushWorkerPolicy::Adaptive {
                initial_workers: 4,
                max_workers: 4,
                invalid_override: false,
            }
        );
    }

    #[test]
    fn resident_async_flush_policy_preserves_fixed_override() {
        assert_eq!(
            super::parse_gpu_subtree_async_flush_worker_policy(Some(OsStr::new("3")), 64),
            super::AsyncCompactFlushWorkerPolicy::Fixed { workers: 3 }
        );
    }

    #[test]
    fn resident_async_flush_policy_marks_invalid_override_and_uses_adaptive_bounds() {
        for value in [OsStr::new("0"), OsStr::new("invalid")] {
            assert_eq!(
                super::parse_gpu_subtree_async_flush_worker_policy(Some(value), 32),
                super::AsyncCompactFlushWorkerPolicy::Adaptive {
                    initial_workers: 8,
                    max_workers: 32,
                    invalid_override: true,
                }
            );
        }
    }

    #[test]
    fn gpu_subtree_compact_flush_mode_defaults_to_adaptive_workers() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _mode_guard = EnvVarGuard::unset("SHEAR_PIPNN_COMPACT_FLUSH_MODE");
        let _workers_guard = EnvVarGuard::unset("SHEAR_PIPNN_ASYNC_FLUSH_WORKERS");

        let mode = super::gpu_subtree_compact_flush_mode(4096, 64);

        match mode {
            super::CompactFlushMode::AsyncOrdered {
                queue_budget_bytes,
                worker_policy,
            } => {
                assert_eq!(queue_budget_bytes, 4096);
                assert_eq!(
                    worker_policy,
                    super::AsyncCompactFlushWorkerPolicy::Adaptive {
                        initial_workers: 8,
                        max_workers: 64,
                        invalid_override: false,
                    }
                );
            }
            super::CompactFlushMode::Sync => panic!("default compact flush mode should be async"),
        }
    }

    #[test]
    fn gpu_subtree_compact_flush_mode_keeps_async_worker_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _mode_guard = EnvVarGuard::unset("SHEAR_PIPNN_COMPACT_FLUSH_MODE");
        let _workers_guard = EnvVarGuard::set("SHEAR_PIPNN_ASYNC_FLUSH_WORKERS", "3");

        let mode = super::gpu_subtree_compact_flush_mode(8192, 64);

        match mode {
            super::CompactFlushMode::AsyncOrdered {
                queue_budget_bytes,
                worker_policy,
            } => {
                assert_eq!(queue_budget_bytes, 8192);
                assert_eq!(
                    worker_policy,
                    super::AsyncCompactFlushWorkerPolicy::Fixed { workers: 3 }
                );
            }
            super::CompactFlushMode::Sync => panic!("default compact flush mode should be async"),
        }
    }

    #[test]
    fn gpu_subtree_compact_flush_mode_sync_override_remains_authoritative() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _mode_guard = EnvVarGuard::set("SHEAR_PIPNN_COMPACT_FLUSH_MODE", "sync");

        assert_eq!(
            super::gpu_subtree_compact_flush_mode(4096, 64),
            super::CompactFlushMode::Sync
        );
    }

    #[test]
    fn blas_thread_env_round_trips_extended_thread_vars() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = super::BlasThreadEnv::capture();

        unsafe {
            env::set_var("OPENBLAS_NUM_THREADS", "8");
            env::set_var("OMP_NUM_THREADS", "9");
            env::set_var("MKL_NUM_THREADS", "10");
            env::set_var("BLIS_NUM_THREADS", "11");
            env::set_var("GOTO_NUM_THREADS", "12");
            env::set_var("MKL_DYNAMIC", "1");
        }
        let snapshot = super::BlasThreadEnv::capture();

        super::BlasThreadEnv::set_single_threaded();

        assert_eq!(env::var("OPENBLAS_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("OMP_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("MKL_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("BLIS_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("GOTO_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("MKL_DYNAMIC").unwrap(), "0");

        snapshot.restore();

        assert_eq!(env::var("OPENBLAS_NUM_THREADS").unwrap(), "8");
        assert_eq!(env::var("OMP_NUM_THREADS").unwrap(), "9");
        assert_eq!(env::var("MKL_NUM_THREADS").unwrap(), "10");
        assert_eq!(env::var("BLIS_NUM_THREADS").unwrap(), "11");
        assert_eq!(env::var("GOTO_NUM_THREADS").unwrap(), "12");
        assert_eq!(env::var("MKL_DYNAMIC").unwrap(), "1");

        original.restore();
    }

    #[test]
    fn resolve_cpu_backend_returns_cpu() {
        let params = PiPNNParams::default();
        let backend = resolve_compute_backend(&params).unwrap();
        assert_eq!(backend.kind(), PiPNNComputeBackendKind::Cpu);
    }

    #[test]
    fn resolve_auto_backend_falls_back_to_cpu_without_cuda() {
        let mut params = PiPNNParams::default();
        params.compute_backend = PiPNNComputeBackendKind::Auto;

        let backend = resolve_compute_backend(&params).unwrap();
        assert_eq!(backend.kind(), PiPNNComputeBackendKind::Cpu);
    }

    #[test]
    fn resolve_cuda_backend_rejects_non_l2_metric() {
        let mut params = PiPNNParams::default();
        params.compute_backend = PiPNNComputeBackendKind::Cuda;
        params.metric = Metric::Cosine;

        let err = resolve_compute_backend(&params).err().unwrap();
        assert!(
            err.to_string()
                .contains("CUDA backend currently supports only L2")
        );
    }

    #[cfg(not(feature = "pipnn_cuda"))]
    #[test]
    fn resolve_cuda_backend_errors_when_feature_is_unavailable() {
        let mut params = PiPNNParams::default();
        params.compute_backend = PiPNNComputeBackendKind::Cuda;

        let err = resolve_compute_backend(&params).err().unwrap();
        assert!(err.to_string().contains("built without CUDA support"));
    }

    #[test]
    fn oom_build_keeps_external_artifacts_when_requested() {
        let dataset = build_test_dataset(1024, 8);
        let temp = tempdir().unwrap();
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = true;
        params.final_prune = false;
        params.c_min = 2;
        params.c_max = 4;
        params.max_depth = 6;
        params.max_leaders = 32;
        params.fanout_top = 4;
        params.fanout_second = 2;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let mut visited = 0usize;
        super::build_pipnn_graph(&dataset, 1024, 1, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 1024);
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().any(|name| name.contains("sketch")),
            "expected OOM build to persist sketch artifacts, found {entries:?}"
        );
        assert!(
            entries
                .iter()
                .any(|name| name.starts_with("partition_d") && name.ends_with("_runs.bin")),
            "expected OOM build to persist append-only partition run artifacts, found {entries:?}"
        );
        assert!(
            entries.iter().any(|name| name == "partition_d03_runs.bin"),
            "expected OOM build to persist depth-3 partition run artifacts, found {entries:?}"
        );
        assert!(
            entries.iter().all(|name| !name.contains("_child_")),
            "expected OOM build to avoid per-child partition run files, found {entries:?}"
        );
        let has_candidate_artifacts = entries.iter().any(|name| name.contains("candidate"));
        let memory_plan = super::OomMemoryPlan::from_params(&params);
        assert!(
            has_candidate_artifacts
                || super::plan_resident_reservoir_prefix(
                    1024,
                    1,
                    &params,
                    dataset.dim,
                    0,
                    1024,
                    memory_plan.reservoir_budget_bytes,
                )
                .resident_points
                    > 0,
            "expected candidate artifacts unless resident reservoir mode is active, found {entries:?}"
        );
    }

    #[test]
    fn oom_build_matches_in_memory_neighbors_on_small_dataset() {
        let dataset = build_test_dataset(48, 8);

        let mut in_memory_params = PiPNNParams::default();
        in_memory_params.final_prune = false;

        let mut oom_params = in_memory_params.clone();
        oom_params.oom_enable = true;
        oom_params.oom_keep_artifacts = false;
        oom_params.oom_temp_dir = tempdir().unwrap().keep();

        let in_memory_graph = collect_graph(&dataset, 48, &in_memory_params);
        let oom_graph = collect_graph(&dataset, 48, &oom_params);

        assert_eq!(oom_graph, in_memory_graph);
    }

    #[test]
    fn pipeline_passthrough_oom_build_matches_default_oom_on_small_dataset() {
        let dataset = build_test_dataset(48, 8);

        let mut default_params = PiPNNParams::default();
        default_params.final_prune = false;
        default_params.oom_enable = true;
        default_params.oom_keep_artifacts = false;
        default_params.oom_temp_dir = tempdir().unwrap().keep();

        let mut pipeline_params = default_params.clone();
        pipeline_params.oom_temp_dir = tempdir().unwrap().keep();
        pipeline_params.pipeline_runtime_enable = true;
        pipeline_params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::Passthrough;
        pipeline_params.pipeline_leaf_descriptor_queue_capacity = Some(1);

        let default_graph = collect_graph(&dataset, 48, &default_params);
        let pipeline_graph = collect_graph(&dataset, 48, &pipeline_params);

        assert_eq!(pipeline_graph, default_graph);
    }

    #[test]
    fn pipeline_cpu_shadow_oom_build_matches_default_oom_on_small_l2_dataset() {
        let dataset = build_test_dataset(48, 8);

        let mut default_params = PiPNNParams::default();
        default_params.final_prune = false;
        default_params.oom_enable = true;
        default_params.oom_keep_artifacts = false;
        default_params.oom_temp_dir = tempdir().unwrap().keep();

        let mut pipeline_params = default_params.clone();
        pipeline_params.oom_temp_dir = tempdir().unwrap().keep();
        pipeline_params.pipeline_runtime_enable = true;
        pipeline_params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::CpuShadow;
        pipeline_params.pipeline_leaf_descriptor_queue_capacity = Some(1);

        let default_graph = collect_graph(&dataset, 48, &default_params);
        let pipeline_graph = collect_graph(&dataset, 48, &pipeline_params);

        assert_eq!(pipeline_graph, default_graph);
    }

    #[test]
    fn oom_streaming_uses_cpu_assignment_backend_with_cuda_leaf_backend() {
        let dataset = build_test_dataset(48, 8);
        let store = super::InmemDatasetPointStore::new(&dataset, 48);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.pipeline_runtime_enable = true;
        params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::Cuda;
        params.pipeline_leaf_descriptor_queue_capacity = Some(1);
        params.c_min = 2;
        params.c_max = 4;
        params.max_depth = 6;
        params.max_leaders = 32;
        params.fanout_top = 4;
        params.fanout_second = 2;
        let sketches =
            super::DiskSketchStore::build_from_point_store(&sketch_path, &store, 48, &params, 1)
                .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..48)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let leaf_backend = FakeCudaLeafOnlyBackend;

        let profile = super::run_streaming_partition_pipeline(
            &store,
            48,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &leaf_backend,
            &edge_sink,
        )
        .expect("streaming should use CPU assignment and fake CUDA leaf compute");

        assert!(profile.leaf.leaves > 0);
        assert!(profile.leaf.points > 0);
        assert_eq!(profile.telemetry.cpu_shadow_unsupported_fallbacks, 0);
    }

    #[test]
    fn gpu_subtree_compact_output_bypasses_leaf_scheduler() {
        let num_points = 256;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 64;
        params.max_depth = 4;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 2;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::new();

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("gpu-subtree compact output should flush compact batches directly");

        assert_eq!(backend.large_leaf_calls(), 0);
        assert_eq!(backend.leaf_topk_calls(), 0);
        assert!(backend.compact_subtree_calls() > 0);
        assert!(profile.leaf.points > 0);
        assert_eq!(
            profile.rbc.gpu_subtree_compact_success_count,
            backend.compact_subtree_calls()
        );
        assert_eq!(profile.rbc.gpu_subtree_compact_rows, profile.leaf.points);
        assert_eq!(
            profile.rbc.gpu_subtree_async_flush_batches,
            profile.rbc.gpu_subtree_compact_batches
        );
        assert_eq!(
            profile.rbc.gpu_subtree_async_flush_rows,
            profile.rbc.gpu_subtree_compact_rows
        );
        assert!(profile.rbc.gpu_subtree_d1plus_queue_max_bytes > 0);
    }

    #[test]
    fn sq_gpu_d0_staging_reopens_sq_wmma_sidecar_for_resident_d1plus() {
        let num_points = 32;
        let dim = 8;
        let dataset = build_test_dataset(num_points, dim);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let sidecar_prefix = artifact_dir.path().join("staged.sq8");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 64;
        params.max_depth = 4;
        params.max_leaders = 4;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 2;
        params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0;
        params.gpu_subtree_sq_staging_upload_shard_rows = 8;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(sidecar_prefix);

        let memory_plan = super::OomMemoryPlan::from_params_for_dataset(&params, num_points, dim);
        assert_eq!(
            memory_plan.sidecar_resident_kind,
            super::OomSidecarResidentKind::SqWmma
        );
        assert!(memory_plan.sidecar_resident_admissible);
        assert_eq!(memory_plan.sidecar_resident_admitted_rows, num_points);

        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::new();

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("SQ GPU-D0 staging should feed resident SQ+WMMA D1+");

        assert!(profile.rbc.gpu_subtree_sq_staging_enabled);
        assert!(profile.rbc.gpu_subtree_sq_staging_global_d0_leaders > 0);
        assert!(backend.compact_subtree_calls() > 0);
        assert!(profile.rbc.gpu_subtree_sq_wmma_sidecar_resident);
        assert!(
            profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_rows
                >= profile.rbc.gpu_subtree_sq_wmma_sidecar_rows
        );
        assert_eq!(profile.rbc.gpu_subtree_sq_wmma_sidecar_file_rows, 0);
    }

    #[test]
    fn sq_gpu_d0_shard_local_d1_skips_sq_wmma_sidecar_reopen() {
        let num_points = 32;
        let dim = 8;
        let dataset = build_test_dataset(num_points, dim);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 64;
        params.max_depth = 4;
        params.max_leaders = 4;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 2;
        params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1;
        params.gpu_subtree_sq_staging_upload_shard_rows = 8;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Off;

        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::new();

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("SQ GPU-D0 shard-local D1 should not require a SQ+WMMA sidecar");

        assert!(profile.rbc.gpu_subtree_sq_staging_enabled);
        assert_eq!(
            profile.rbc.gpu_subtree_sq_staging_sidecar_source,
            "gpu-d0-shard-local-d1"
        );
        assert!(backend.compact_subtree_calls() > 0);
        assert!(!profile.rbc.gpu_subtree_sq_wmma_sidecar_resident);
        assert_eq!(profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_rows, 0);
        assert_eq!(profile.rbc.gpu_subtree_sq_wmma_sidecar_file_rows, 0);
    }

    #[test]
    fn gpu_subtree_compact_output_writes_valid_reservoir_edges() {
        let num_points = 128;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 32;
        params.max_depth = 3;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 1;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::new();

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("compact gpu-subtree pipeline");

        assert_eq!(profile.rbc.gpu_subtree_compact_rows, num_points);
        assert_eq!(profile.leaf.points, num_points);
        let non_empty_reservoirs = reservoirs
            .iter()
            .filter(|reservoir| !reservoir.lock().slots().is_empty())
            .count();
        assert_eq!(non_empty_reservoirs, num_points);
        for (point, reservoir) in reservoirs.iter().enumerate() {
            for candidate in reservoir.lock().slots() {
                assert_ne!(candidate.target as usize, point);
                assert!((candidate.target as usize) < num_points);
            }
        }
    }

    #[test]
    fn gpu_subtree_d0_d1_split_sync_matches_wrapper() {
        let num_points = 128;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 32;
        params.max_depth = 3;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 1;
        let indices: Vec<u32> = (0..num_points as u32).collect();

        let wrapper_dir = tempdir().unwrap();
        let wrapper_sketch_path = wrapper_dir.path().join("wrapper_sketches.bin");
        let wrapper_sketches = super::DiskSketchStore::build_from_point_store(
            &wrapper_sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("wrapper sketches");
        let wrapper_reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let wrapper_edge_sink = super::ReservoirEdgeSink::new(wrapper_reservoirs.as_slice());
        let wrapper_sink =
            super::OomCompactTopKSink::new(&wrapper_sketches, &params, &wrapper_edge_sink);
        let wrapper_backend = FakeCompactTopKSubtreeBackend::new();
        let mut wrapper_rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(params.random_seed);
        let wrapper_profile = super::rbc_partition::rbc_partition_gpu_subtree_streaming(
            &wrapper_backend,
            &store,
            &indices,
            params.metric(),
            &params,
            1,
            &mut wrapper_rng,
            Some(wrapper_dir.path()),
            None,
            Some(&wrapper_sink),
        )
        .expect("wrapper sync gpu-subtree");

        let split_dir = tempdir().unwrap();
        let mut split_rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(params.random_seed);
        let d0_plan = super::rbc_partition::run_gpu_subtree_d0_preflight(
            super::compute_backend::cpu_backend(),
            &store,
            &indices,
            params.metric(),
            &params,
            1,
            &mut split_rng,
            Some(split_dir.path()),
        )
        .expect("split D0 preflight");
        let split_sketch_path = split_dir.path().join("split_sketches.bin");
        let split_sketches = super::DiskSketchStore::build_from_point_store(
            &split_sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("split sketches");
        let split_reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let split_edge_sink = super::ReservoirEdgeSink::new(split_reservoirs.as_slice());
        let split_sink = super::OomCompactTopKSink::new(&split_sketches, &params, &split_edge_sink);
        let split_backend = FakeCompactTopKSubtreeBackend::new();
        let split_profile = super::rbc_partition::run_gpu_subtree_from_d0_plan(
            &split_backend,
            &store,
            params.metric(),
            &params,
            d0_plan,
            &split_sink,
            super::rbc_partition::CompactFlushMode::Sync,
        )
        .expect("split sync gpu-subtree");

        assert_eq!(
            split_profile.rbc.gpu_subtree_compact_rows,
            wrapper_profile.rbc.gpu_subtree_compact_rows
        );
        assert_eq!(split_profile.leaf.points, wrapper_profile.leaf.points);
        assert_eq!(
            split_profile.rbc.gpu_subtree_compact_success_count,
            wrapper_profile.rbc.gpu_subtree_compact_success_count
        );
        let wrapper_non_empty = wrapper_reservoirs
            .iter()
            .filter(|reservoir| !reservoir.lock().slots().is_empty())
            .count();
        let split_non_empty = split_reservoirs
            .iter()
            .filter(|reservoir| !reservoir.lock().slots().is_empty())
            .count();
        assert_eq!(split_non_empty, wrapper_non_empty);
    }

    #[test]
    fn gpu_subtree_d0_preflight_routes_root_assignment_through_supplied_backend() {
        let num_points = 320usize;
        let dataset = build_test_dataset(num_points, 8);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let indices: Vec<u32> = (0..num_points as u32).collect();
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.c_min = 4;
        params.c_max = 16;
        params.max_depth = 5;
        params.max_leaders = 8;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.random_seed = 777;
        let dir = tempdir().unwrap();
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(params.random_seed);
        let root_backend = CountingCudaBackend::new(num_points);

        let d0_plan = super::rbc_partition::run_gpu_subtree_d0_preflight(
            &root_backend,
            &store,
            &indices,
            params.metric(),
            &params,
            1,
            &mut rng,
            Some(dir.path()),
        )
        .expect("D0 preflight with supplied root backend");

        assert!(
            root_backend.root_sized_assignment_calls() > 0,
            "D0 root assignment should call the supplied root backend"
        );
        assert_eq!(root_backend.subtree_assignment_calls(), 0);
        assert_eq!(d0_plan.child_stats.total_points, num_points);
    }

    #[test]
    fn gpu_subtree_compact_output_allows_overlapping_leaf_rows() {
        let num_points = 64;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 32;
        params.max_depth = 4;
        params.max_leaders = 1;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 1;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::overlapping_rows(2);

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("overlapping compact rows should be accepted");

        assert_eq!(backend.compact_subtree_calls(), 1);
        assert_eq!(profile.rbc.gpu_subtree_compact_rows, num_points * 2);
        assert_eq!(profile.leaf.points, num_points * 2);
        let non_empty_reservoirs = reservoirs
            .iter()
            .filter(|reservoir| !reservoir.lock().slots().is_empty())
            .count();
        assert_eq!(non_empty_reservoirs, num_points);
    }

    #[test]
    fn gpu_subtree_compact_root_no_split_routes_full_root_to_compact_backend() {
        let num_points = 96;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 32;
        params.max_depth = 4;
        params.max_leaders = 1;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 2;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::new();

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("root no-split compact output should route through compact backend");

        assert_eq!(backend.large_leaf_calls(), 0);
        assert_eq!(backend.leaf_topk_calls(), 0);
        assert_eq!(backend.compact_subtree_calls(), 1);
        assert_eq!(profile.leaf.points, num_points);
        assert_eq!(profile.rbc.gpu_subtree_compact_rows, num_points);
    }

    #[test]
    fn gpu_subtree_compact_root_natural_size_routes_full_root_to_compact_backend() {
        let num_points = 32;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 64;
        params.max_depth = 4;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 2;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::new();

        let profile = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect("root natural-size compact output should route through compact backend");

        assert_eq!(backend.large_leaf_calls(), 0);
        assert_eq!(backend.leaf_topk_calls(), 0);
        assert_eq!(backend.compact_subtree_calls(), 1);
        assert_eq!(profile.leaf.points, num_points);
        assert_eq!(profile.rbc.gpu_subtree_compact_rows, num_points);
    }

    #[test]
    fn gpu_subtree_compact_output_rejects_incomplete_child_coverage() {
        let num_points = 256;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 64;
        params.max_depth = 4;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        params.leaf_knn = 2;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCompactTopKSubtreeBackend::incomplete_output();

        let err = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .unwrap_err();

        assert_eq!(backend.large_leaf_calls(), 0);
        assert_eq!(backend.leaf_topk_calls(), 0);
        assert!(backend.compact_subtree_calls() > 0);
        assert!(
            err.to_string()
                .contains("gpu-subtree compact top-k output does not cover child run"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gpu_subtree_streaming_rejects_cpu_backend_even_without_cli_validation() {
        let num_points = 128;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 32;
        params.max_depth = 3;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());

        let err = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            super::compute_backend::cpu_backend(),
            &edge_sink,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("gpu-subtree requires CUDA compute backend")
        );
    }

    #[test]
    fn gpu_subtree_compact_output_rejects_cuda_backend_without_compact_support() {
        let num_points = 128;
        let dataset = build_test_dataset(num_points, 4);
        let store = super::InmemDatasetPointStore::new(&dataset, num_points);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.c_min = 8;
        params.c_max = 32;
        params.max_depth = 3;
        params.max_leaders = 2;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.psamp_fraction = 1.0;
        let sketches = super::DiskSketchStore::build_from_point_store(
            &sketch_path,
            &store,
            num_points,
            &params,
            1,
        )
        .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());
        let backend = FakeCudaLeafOnlyBackend;

        let err = super::run_streaming_partition_pipeline(
            &store,
            num_points,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            &backend,
            &edge_sink,
        )
        .expect_err("compact gpu-subtree must fail loudly without compact support");

        assert!(
            err.to_string()
                .contains("gpu-subtree compact top-k requires CUDA compact subtree support"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn oom_streaming_kmeans_partitioner_emits_hard_partition_leaves() {
        let dataset = build_test_dataset(64, 8);
        let store = super::InmemDatasetPointStore::new(&dataset, 64);
        let artifact_dir = tempdir().unwrap();
        let sketch_path = artifact_dir.path().join("sketches.bin");
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = true;
        params.oom_temp_dir = PathBuf::from(artifact_dir.path());
        params.partitioner = PiPNNPartitionerKind::KMeans;
        params.c_min = 2;
        params.c_max = 8;
        params.max_leaders = 16;
        params.kmeans_iterations = 2;
        params.kmeans_oversplit_factor = 2;
        params.leaf_knn = 2;
        params.final_prune = false;
        let sketches =
            super::DiskSketchStore::build_from_point_store(&sketch_path, &store, 64, &params, 1)
                .expect("sketches");
        let reservoirs: Vec<Mutex<HashPruneReservoir>> = (0..64)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = super::ReservoirEdgeSink::new(reservoirs.as_slice());

        let profile = super::run_streaming_partition_pipeline(
            &store,
            64,
            1,
            1,
            &params,
            artifact_dir.path(),
            &sketches,
            false,
            super::compute_backend::cpu_backend(),
            &edge_sink,
        )
        .expect("k-means streaming partition should feed the leaf pipeline");

        assert_eq!(profile.leaf.points, 64);
        assert_eq!(profile.kmeans.points, 64);
        assert_eq!(profile.kmeans.emitted_points, 64);
        assert!(profile.kmeans.leaves > 0);
        assert_eq!(profile.rbc.root_fanout.fixed_fanout, 0);
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn pipeline_cuda_oom_build_matches_cpu_shadow_and_writes_gpu_profile_on_small_l2_dataset() {
        let dataset = build_test_dataset(48, 8);

        let mut cpu_shadow_params = PiPNNParams::default();
        cpu_shadow_params.final_prune = false;
        cpu_shadow_params.oom_enable = true;
        cpu_shadow_params.oom_keep_artifacts = false;
        cpu_shadow_params.oom_temp_dir = tempdir().unwrap().keep();
        cpu_shadow_params.pipeline_runtime_enable = true;
        cpu_shadow_params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::CpuShadow;
        cpu_shadow_params.pipeline_leaf_descriptor_queue_capacity = Some(1);

        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("cuda_pipeline_profile.json");
        let mut cuda_params = cpu_shadow_params.clone();
        cuda_params.oom_temp_dir = dir.path().join("artifacts");
        cuda_params.oom_profile_json = Some(profile_path.clone());
        cuda_params.compute_backend = PiPNNComputeBackendKind::Cuda;
        cuda_params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::Cuda;
        cuda_params.gpu_workspace_budget_bytes = 64 * 1024 * 1024;

        let cpu_shadow_graph = collect_graph(&dataset, 48, &cpu_shadow_params);
        let cuda_graph = collect_graph(&dataset, 48, &cuda_params);

        assert_eq!(cuda_graph, cpu_shadow_graph);
        let text = fs::read_to_string(&profile_path).unwrap();
        assert!(text.contains("\"gpu\""));
        assert!(
            !text.contains("\"gpu_h2d_bytes\": 0"),
            "expected CUDA pipeline profile to record nonzero H2D bytes: {text}"
        );
        assert!(
            !text.contains("\"gpu_d2h_bytes\": 0"),
            "expected CUDA pipeline profile to record nonzero compact DtoH bytes: {text}"
        );
        assert!(text.contains("\"gpu_cpu_fallbacks\": 0"));
        assert!(text.contains("\"gpu_workspace_pressure_events\": 0"));
        assert!(text.contains("\"gpu_h2d_submit_wall_ms\""));
        assert!(text.contains("\"gpu_kernel_launch_wall_ms\""));
        assert!(text.contains("\"gpu_kernel_elapsed_ms\""));
        assert!(text.contains("\"gpu_d2h_submit_wall_ms\""));
        assert!(text.contains("\"gpu_post_d2h_sync_wall_ms\""));
        assert!(text.contains("\"gpu_stream_total_elapsed_ms\""));
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn gpu_subtree_cuda_oom_build_matches_cpu_shadow_and_profiles_persistent_subtree() {
        let dataset = build_test_dataset(160, 8);

        let mut cpu_shadow_params = PiPNNParams::default();
        cpu_shadow_params.final_prune = false;
        cpu_shadow_params.oom_enable = true;
        cpu_shadow_params.oom_keep_artifacts = false;
        cpu_shadow_params.oom_temp_dir = tempdir().unwrap().keep();
        cpu_shadow_params.pipeline_runtime_enable = true;
        cpu_shadow_params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::CpuShadow;
        cpu_shadow_params.c_min = 4;
        cpu_shadow_params.c_max = 16;
        cpu_shadow_params.max_depth = 5;
        cpu_shadow_params.max_leaders = 8;
        cpu_shadow_params.fanout_top = 1;
        cpu_shadow_params.fanout_second = 1;
        cpu_shadow_params.psamp_fraction = 1.0;
        cpu_shadow_params.random_seed = 321;

        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("gpu_subtree_profile.json");
        let mut gpu_subtree_params = cpu_shadow_params.clone();
        gpu_subtree_params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        gpu_subtree_params.compute_backend = PiPNNComputeBackendKind::Cuda;
        gpu_subtree_params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::Cuda;
        gpu_subtree_params.oom_temp_dir = dir.path().join("artifacts");
        gpu_subtree_params.oom_profile_json = Some(profile_path.clone());
        gpu_subtree_params.gpu_workspace_budget_bytes = 64 * 1024 * 1024;

        let cpu_shadow_graph = collect_graph(&dataset, 160, &cpu_shadow_params);
        let gpu_subtree_graph = collect_graph(&dataset, 160, &gpu_subtree_params);

        assert!(gpu_subtree_graph.iter().all(|neighbors| {
            neighbors
                .iter()
                .all(|&neighbor| (neighbor as usize) < gpu_subtree_graph.len())
        }));
        let recall_against_cpu_shadow = graph_edge_recall(&gpu_subtree_graph, &cpu_shadow_graph);
        assert!(
            recall_against_cpu_shadow >= 0.75,
            "gpu-subtree compact top-k recall against CPU shadow too low: {recall_against_cpu_shadow}"
        );

        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            value
                .get("partitioner")
                .and_then(|partitioner| partitioner.get("kind"))
                .and_then(Value::as_str),
            Some("gpu-subtree")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("enabled"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("qcode_staging_mode"))
                .and_then(Value::as_str),
            Some("off")
        );
        assert_json_bool(
            &value,
            &["gpu_subtree", "rabitq_residual", "enabled"],
            false,
        );
        assert_json_u64(&value, &["gpu_subtree", "rabitq_residual", "centroids"], 16);
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_residual", "upload_shard_rows"],
            524_288,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_residual", "global_d0_leaders"],
            0,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_residual", "host_lazy_split_count"],
            0,
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("root_assignment_backend"))
                .and_then(Value::as_str),
            Some("cpu")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("assignment_backend"))
                .and_then(Value::as_str),
            Some("current")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("subtree_assignment_backend"))
                .and_then(Value::as_str),
            Some("current")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("leaf_topk_backend"))
                .and_then(Value::as_str),
            Some("cuda")
        );
        assert!(json_u64(&value, &["gpu_subtree", "child_count"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "child_points"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "compact_topk_batches"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "compact_topk_rows"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "leaf_count"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "persistent_attempt_count"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "persistent_success_count"]) > 0);
        assert!(json_u64(&value, &["gpu_subtree", "persistent_leaf_count"]) > 0);
        assert_json_u64(&value, &["gpu_subtree", "compact_unsupported_count"], 0);
    }

    #[test]
    fn pipeline_oom_profile_json_includes_queue_and_backpressure_fields() {
        let dataset = build_test_dataset(48, 8);
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("pipeline_profile.json");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.oom_keep_artifacts = false;
        params.oom_temp_dir = dir.path().join("artifacts");
        params.oom_profile_json = Some(profile_path.clone());
        params.pipeline_runtime_enable = true;
        params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::Passthrough;
        params.pipeline_leaf_descriptor_queue_capacity = Some(1);

        let mut visited = 0usize;
        super::build_pipnn_graph(&dataset, 48, 1, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 48);
        let text = fs::read_to_string(&profile_path).unwrap();
        assert!(text.contains("\"producer_wait_ms\""));
        assert!(text.contains("\"peak_leaf_descriptor_queue\""));
        assert!(text.contains("\"producer_blocked_ms\""));
        assert!(text.contains("\"post_producer_leaf_drain_ms\""));
        assert!(text.contains("\"pipeline\""));
        assert!(text.contains("\"stage_wall_ms\""));
        assert!(text.contains("\"compute_flush_wall_ms\""));
        assert!(text.contains("\"edge_flush_wall_ms\""));
        assert!(text.contains("\"leaf_flow\""));
        assert!(text.contains("\"host_load_wall_ms\""));
        assert!(text.contains("\"backend_compute_wall_ms\""));
    }

    #[test]
    fn pipeline_oom_profile_json_includes_semantic_stage_boundaries() {
        let dataset = build_test_dataset(48, 8);
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("pipeline_profile.json");
        let mut params = PiPNNParams::default();
        params.final_prune = false;
        params.oom_enable = true;
        params.oom_keep_artifacts = false;
        params.oom_temp_dir = dir.path().join("artifacts");
        params.oom_profile_json = Some(profile_path.clone());
        params.pipeline_runtime_enable = true;
        params.pipeline_compute_mode = super::PiPNNPipelineComputeMode::Passthrough;
        params.pipeline_leaf_descriptor_queue_capacity = Some(1);

        super::build_pipnn_graph(&dataset, 48, 1, &params, |_, neighbors| {
            let _ = neighbors;
            Ok(())
        })
        .unwrap();

        let text = fs::read_to_string(&profile_path).unwrap();
        assert!(text.contains("\"d0_ads_assignment\""));
        assert!(text.contains("\"d1_ads_assignment\""));
        assert!(text.contains("\"d1_materialize\""));
        assert!(text.contains("\"d2_replay\""));
        assert!(text.contains("\"external_reduce\""));
    }

    #[test]
    fn oom_build_skips_candidate_spill_when_budget_can_hold_full_reservoirs() {
        let dataset = build_test_dataset(128, 8);
        let temp = tempdir().unwrap();
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = true;
        params.final_prune = false;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let mut visited = 0usize;
        super::build_pipnn_graph(&dataset, 128, 4, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 128);
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().all(|name| !name.contains("candidate")),
            "expected OOM build to skip candidate spill artifacts when the full reservoir fits in memory, found {entries:?}"
        );
    }

    #[test]
    fn oom_build_cleans_up_external_artifacts_by_default() {
        let dataset = build_test_dataset(32, 8);
        let temp = tempdir().unwrap();
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = false;
        params.final_prune = false;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let mut visited = 0usize;
        super::build_pipnn_graph(&dataset, 32, 1, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 32);
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.is_empty(),
            "expected OOM build to clean artifacts by default, found {entries:?}"
        );
    }

    #[test]
    fn oom_profile_json_includes_memory_temp_file_and_stage_sections() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let artifact_dir = dir.path().join("artifacts");
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::write(artifact_dir.join("sketches.bin"), vec![0u8; 16]).unwrap();
        fs::write(
            artifact_dir.join("candidate_shard00000_part000.log"),
            vec![0u8; 16],
        )
        .unwrap();
        fs::write(
            artifact_dir.join("candidate_segments.manifest"),
            vec![0u8; 8],
        )
        .unwrap();
        fs::write(artifact_dir.join("partition_d01_runs.bin"), vec![0u8; 8]).unwrap();

        let profile = super::OomProfile {
            num_points: 32,
            dim: 8,
            leaf_knn: 2,
            partitioner: PiPNNPartitionerKind::Rbc,
            root_assignment_backend: PiPNNComputeBackendKind::Cpu,
            assignment_backend: PiPNNGpuSubtreeAssignmentBackend::Current,
            d0_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            d0_dco_survivors: 0,
            d0_dco_audit: false,
            d0_sq_cpu_kernel: SqInt8CpuKernel::Auto,
            d1plus_lazy_split_mode: super::PiPNNGpuSubtreeD1PlusLazySplitMode::On,
            assignment_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            assignment_dco_proj_dim: 0,
            assignment_dco_survivors: 0,
            assignment_dco_audit: false,
            leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend::PersistentCustom,
            leaf_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            leaf_dco_proj_dim: 0,
            leaf_dco_survivors: 0,
            leaf_dco_audit: false,
            leaf_dco_ddc_res_pca_matrix: None,
            leaf_dco_ddc_res_projected_input: false,
            leaf_dco_ddc_res_base_dim: 32,
            leaf_dco_ddc_res_sigma_count: 3.0,
            rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode::Off,
            rabitq_plus_bits: 8,
            rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            rabitq_plus_seed: 42,
            rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode::Off,
            rabitq_plus_sidecar_prefix: None,
            rabitq_plus_fused_sidecar_sketch_enabled: false,
            rabitq_plus_fused_prestage_total_wall: Duration::ZERO,
            rabitq_plus_fused_chunk_alloc_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_raw_read_wall: Duration::ZERO,
            rabitq_plus_fused_source_hash_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_quantize_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_code_output_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_norm_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_meta_output_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_header_write_wall: Duration::ZERO,
            rabitq_plus_fused_sidecar_write_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_alloc_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_compute_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_extend_wall: Duration::ZERO,
            rabitq_plus_fused_sketch_write_wall: Duration::ZERO,
            qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode::Off,
            qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident,
            qcode_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma,
            qcode_rabitq_short_survivors: None,
            rabitq_residual_centroids: 16,
            qcode_staging_upload_shard_rows: 524_288,
            qcode_global_d0_leaders: 0,
            sq_staging_mode: PiPNNGpuSubtreeSqStagingMode::Off,
            sq_staging_upload_shard_rows: 524_288,
            sq_bits: SQ_INT8_DEFAULT_BITS,
            sq_native_int4: false,
            sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode::Off,
            persistent_large_ticket_batch_requested: PiPNNGpuSubtreePersistentLargeTicketBatch::One,
            persistent_sq8_assignment_kernel_requested:
                PiPNNGpuSubtreeSq8AssignmentKernel::WarpDp4a,
            persistent_sq8_leaf_executor_requested: PiPNNGpuSubtreeSq8LeafExecutor::Auto,
            persistent_sq8_leaf_mode_requested: PiPNNGpuSubtreeSq8LeafMode::Staged,
            persistent_scheduler_mode_requested: PiPNNGpuSubtreePersistentSchedulerMode::Baseline,
            persistent_scheduler_mode_actual: PiPNNGpuSubtreePersistentSchedulerMode::Baseline,
            persistent_idle_backoff_ns_requested: 350,
            persistent_idle_backoff_ns_actual: 0,
            sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode::Off,
            sq_wmma_sidecar_prefix: None,
            sq_staging_fused_sketch_enabled: false,
            sq_staging_fused_sketch_compute_wall: Duration::ZERO,
            sq_staging_fused_sketch_join_wait_wall: Duration::ZERO,
            turboquant_mode: PiPNNGpuSubtreeTurboQuantMode::Off,
            turboquant_bits: 4,
            turboquant_seed: 42,
            turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode::Off,
            turboquant_sidecar_prefix: None,
            turboquant_cascade_low_dim: 128,
            turboquant_cascade_survivor_cap: 0,
            turboquant_adaptive_survivor_threshold: 0,
            turboquant_adaptive_survivor_low_cap: 0,
            memory_budget_bytes: 12 * 1024 * 1024 * 1024,
            memory_budget_source: "auto-host-cgroup",
            oom_io_mode: PiPNNOomIoMode::Buffered,
            oom_reservoir_budget_share: 0.5,
            oom_sidecar_resident_budget_cap_bytes: Some(0),
            effective_threads: 4,
            pool_threads: 4,
            leaf_budget_threads: 3,
            resident_reservoir_points: 4096,
            resident_reservoir_bytes: 1_048_576,
            reservoir_allocation_wall: Duration::from_millis(27),
            reservoir_initialization: super::ReservoirInitializationProfile {
                requested: "<unset>".to_string(),
                layout: "legacy",
                mode: "d0-overlap",
                allocation_phase: "D0-overlap",
                workers: 8,
                overlap_enabled: true,
                invalid_override: false,
                parallel_fallback: false,
                d0_paced: true,
                ready_prefix: 4096,
                initialized_prefix: 4096,
                frontier_wait_count: 7,
                frontier_wait_wall: Duration::from_millis(19),
                allocation_wall: Duration::from_millis(27),
                d1_wait_wall: Duration::from_millis(3),
                mapped_bytes: 0,
                segment_count: 0,
                stripe_count: 0,
                scratch_bytes: 0,
            },
            reduce_shard_points: 128,
            compact_spill_digest_enabled: true,
            compact_spill_digest_width: 128,
            prune_backend: PiPNNPruneBackend::LRPF,
            hashprune_evidence_policy: super::HashPruneEvidencePolicy::off(),
            hashprune_backfill_policy: super::HashPruneBackfillPolicy::off(),
            hashprune_backfill_degree_cap: 64,
            lrpf_target_degree: 64,
            lrpf_long_edge_quota: 0,
            lrpf_pre_spill_compress_enabled: true,
            lrpf_pre_spill_width: 64,
            lrpf_candidate_records: 7,
            lrpf_witness_records: 3,
            lrpf_fragments: 2,
            lrpf_pre_spill_pruned_candidates: 5,
            lrpf_reducer_pruned_candidates: 4,
            reservoir_budget_bytes: 4 * 1024 * 1024 * 1024,
            resident_subtree_budget_bytes: 4 * 1024 * 1024 * 1024,
            required_resident_bytes: 0,
            optional_resident_budget_bytes: 4 * 1024 * 1024 * 1024,
            optional_resident_admitted_bytes: 0,
            optional_resident_unassigned_bytes: 4 * 1024 * 1024 * 1024,
            transient_pipeline_budget_bytes: 2 * 1024 * 1024 * 1024,
            d1plus_working_set_budget_bytes: 0,
            sidecar_resident_kind: super::OomSidecarResidentKind::None,
            sidecar_resident_budget_bytes: 0,
            sidecar_resident_required_bytes: 0,
            sidecar_resident_admitted_rows: 0,
            sidecar_resident_admissible: false,
            scratch_budget_bytes: 2 * 1024 * 1024 * 1024,
            io_headroom_bytes: 2 * 1024 * 1024 * 1024,
            raw_vector_cache_budget_bytes: 0,
            raw_vector_cache_rows: 0,
            online_suffix_mode: super::PiPNNOnlineSuffixReduceMode::Off,
            online_suffix_fallback_reason: None,
            online_suffix_config: super::OnlineSuffixProfileConfig::default(),
            online_suffix: super::OnlineSuffixReduceStats::default(),
            sketch_bytes: 128,
            sketch_file_bytes: 256,
            spill: super::SpillRunStats {
                part_files: 1,
                segment_count: 2,
                shard_groups: 1,
                max_parts_per_shard: 1,
                max_segments_per_shard: 2,
                record_bytes: 16,
                manifest_bytes: 8,
            },
            reduce: super::SpillReduceStats {
                shard_groups: 1,
                segments_scanned: 2,
                part_files_opened: 1,
                max_segments_in_group: 2,
                evidence: Default::default(),
                backfill: Default::default(),
                ..super::SpillReduceStats::default()
            },
            sketch_persist_wall: Duration::from_millis(11),
            streaming_wall: Duration::from_millis(22),
            reduce_wall: Duration::from_millis(33),
            emit_wall: Duration::from_millis(44),
            resident_emit: super::ResidentEmitProfile::default(),
            total_wall: Duration::from_millis(55),
            scheduler: super::SchedulerTelemetry {
                descriptor_receive_wall_ms: 99,
                descriptor_batch_extend_wall_ms: 100,
                stage_wall_ms: 101,
                compute_flush_wall_ms: 102,
                topk_queue_roundtrip_wall_ms: 103,
                edge_queue_roundtrip_wall_ms: 104,
                edge_flush_wall_ms: 105,
                descriptor_queue_empty_count: 4,
                cuda_batch_count: 2,
                cuda_batch_leaves: 3,
                cuda_batch_rows: 7,
                cuda_batch_target_bytes: 4096,
                cuda_batch_actual_bytes: 2048,
                cuda_batch_underfilled_count: 1,
                ..super::SchedulerTelemetry::default()
            },
            rbc: super::RbcPartitionTelemetry {
                resident_subtree_count: 2,
                resident_subtree_points: 128,
                resident_subtree_bytes: 4096,
                resident_subtree_read_wall: Duration::from_millis(7),
                resident_subtree_compute_wall: Duration::from_millis(9),
                resident_subtree_emit_wall: Duration::from_millis(67),
                resident_subtree_leaf_count: 5,
                resident_subtree_fallback_count: 1,
                resident_subtree_fallback_points: 89,
                resident_subtree_max_points: 55,
                resident_subtree_permit_wait: Duration::from_millis(3),
                d0_ads_assignment_wall: Duration::from_millis(13),
                d1_ads_assignment_wall: Duration::from_millis(17),
                d1_materialize_wall: Duration::from_millis(19),
                d2_replay_wall: Duration::from_millis(23),
                root_fanout: super::rbc_partition::RootFanoutProfile {
                    policy: "cell-radius-certified".to_string(),
                    selection_semantics: "cell_radius_certified_root_assignment_v1".to_string(),
                    enabled: true,
                    dry_run: false,
                    fallback_reason: None,
                    k: 10,
                    fixed_fanout: 8,
                    candidate_rs: vec![4, 6],
                    warmup_points: 500_000,
                    total_points: 32,
                    warmup_points_seen: 0,
                    certified_points: 24,
                    fallback_points: 8,
                    insufficient_radius_points: 2,
                    kept_hist: vec![0, 10, 8, 0, 6, 0, 0, 0, 8],
                    ready_cells: 4,
                    avg_fanout: 2.0,
                    selector_ms: 12,
                    radius_update_ms: 3,
                    projected_assignment_bytes: 256,
                    projected_partition_d00_bytes_pre_dedup: 256,
                    root_leader_hash: 12345,
                    ..super::rbc_partition::RootFanoutProfile::default()
                },
                ..super::RbcPartitionTelemetry::default()
            },
            kmeans: super::KMeansPartitionTelemetry::default(),
            hashprune_evidence: super::HashPruneEvidenceProfile::default(),
            hashprune_backfill: super::HashPruneBackfillProfile::default(),
            leaf: super::LeafProfile {
                host_load_wall: Duration::from_millis(61),
                backend_compute_wall: Duration::from_millis(62),
                compact_materialize_wall: Duration::from_millis(63),
                hash_wall: Duration::from_millis(64),
                edge_sink_wall: Duration::from_millis(65),
                edge_sink_sort_wall: Duration::from_millis(66),
                edge_sink_insert_wall: Duration::from_millis(67),
                edge_sink_lock_wall: Duration::from_millis(71),
                edge_sink_insert_body_wall: Duration::from_millis(72),
                edge_sink_edges: 68,
                edge_sink_flushes: 69,
                edge_sink_groups: 73,
                edge_sink_max_group_edges: 74,
                edge_sink_insert_candidates: 75,
                edge_sink_hash_hits: 76,
                edge_sink_hash_replacements: 77,
                edge_sink_hash_rejects: 78,
                edge_sink_not_full_inserts: 79,
                edge_sink_full_replacement_checks: 80,
                edge_sink_full_replacements: 81,
                edge_sink_full_rejects: 82,
                edge_sink_farthest_recomputes: 83,
                edge_sink_rotate_lefts: 84,
                edge_sink_rotate_rights: 85,
                flush_wall: Duration::from_millis(70),
                gpu_h2d_bytes: 11,
                gpu_d2h_bytes: 22,
                gpu_h2d: Duration::from_millis(31),
                gpu_d2h: Duration::from_millis(32),
                gpu_kernel: Duration::from_millis(33),
                gpu_sync: Duration::from_millis(34),
                gpu_h2d_submit_wall: Duration::from_millis(35),
                gpu_kernel_launch: Duration::from_millis(36),
                gpu_kernel_elapsed: Duration::from_millis(37),
                gpu_d2h_submit_wall: Duration::from_millis(38),
                gpu_post_d2h_sync_wall: Duration::from_millis(39),
                gpu_stream_total_elapsed: Duration::from_millis(40),
                compact_lune_witness_enabled: true,
                compact_lune_witness_slots: 41,
                compact_lune_witnesses: 17,
                compact_lune_prune_enabled: true,
                compact_lune_prune_protected_core: 1,
                compact_lune_pruned_slots: 9,
                compact_lune_pruned_rows: 5,
                ..super::LeafProfile::default()
            },
            memory: super::OomMemoryReport {
                rss_kb_samples: vec![128, 256],
                rss_hwm_kb: Some(512),
                vm_peak_kb: Some(1024),
            },
            temp_files: super::collect_temp_file_report(&artifact_dir).unwrap(),
            uring_batches: 1,
            uring_windows: 2,
            uring_bytes: 4096,
            uring_fallbacks: 0,
        };

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_json_str(&value, &["oom_io_mode"], "buffered");
        assert_json_f64(&value, &["oom_reservoir_budget_share"], 0.5);
        assert_json_u64(&value, &["oom_sidecar_resident_budget_cap_bytes"], 0);
        assert_json_bool(&value, &["compact_spill_digest", "enabled"], true);
        assert_json_u64(&value, &["compact_spill_digest", "width"], 128);
        assert_json_str(&value, &["prune_backend"], "lrpf");
        assert_json_u64(&value, &["lrpf", "target_degree"], 64);
        assert_json_u64(&value, &["lrpf", "long_edge_quota"], 0);
        assert_json_bool(&value, &["lrpf", "pre_spill_compress_enabled"], true);
        assert_json_u64(&value, &["lrpf", "pre_spill_width"], 64);
        assert_json_u64(&value, &["lrpf", "candidate_records"], 7);
        assert_json_u64(&value, &["lrpf", "witness_records"], 3);
        assert_json_u64(&value, &["lrpf", "fragments"], 2);
        assert_json_u64(&value, &["lrpf", "pre_spill_pruned_candidates"], 5);
        assert_json_u64(&value, &["lrpf", "reducer_pruned_candidates"], 4);
        assert_json_u64(
            &value,
            &["memory_plan", "reservoir_budget_bytes"],
            4 * 1024 * 1024 * 1024,
        );
        assert_json_u64(&value, &["resident_reservoir", "points"], 4096);
        assert_json_u64(&value, &["resident_reservoir", "bytes"], 1_048_576);
        assert_json_str(&value, &["resident_reservoir", "layout"], "legacy");
        assert_json_u64(&value, &["resident_reservoir", "mapped_bytes"], 0);
        assert_json_u64(&value, &["resident_reservoir", "segment_count"], 0);
        assert_json_u64(&value, &["resident_reservoir", "stripe_count"], 0);
        assert_json_u64(&value, &["resident_reservoir", "scratch_bytes"], 0);
        assert_json_str(
            &value,
            &["resident_reservoir", "allocation_phase"],
            "D0-overlap",
        );
        assert_json_str(
            &value,
            &["resident_reservoir", "initialization_mode"],
            "d0-overlap",
        );
        assert_json_str(
            &value,
            &["resident_reservoir", "initialization_requested"],
            "<unset>",
        );
        assert_json_u64(&value, &["resident_reservoir", "initialization_workers"], 8);
        assert_json_bool(
            &value,
            &["resident_reservoir", "initialization_overlap_enabled"],
            true,
        );
        assert_json_bool(
            &value,
            &["resident_reservoir", "initialization_invalid_override"],
            false,
        );
        assert_json_bool(
            &value,
            &["resident_reservoir", "initialization_parallel_fallback"],
            false,
        );
        assert_json_bool(
            &value,
            &["resident_reservoir", "initialization_d0_paced"],
            true,
        );
        assert_json_u64(
            &value,
            &["resident_reservoir", "initialization_ready_prefix"],
            4096,
        );
        assert_json_u64(
            &value,
            &["resident_reservoir", "initialization_initialized_prefix"],
            4096,
        );
        assert_json_u64(
            &value,
            &["resident_reservoir", "initialization_frontier_wait_count"],
            7,
        );
        assert_json_u64(
            &value,
            &["resident_reservoir", "initialization_frontier_wait_ms"],
            19,
        );
        assert_json_u64(&value, &["resident_reservoir", "allocation_ms"], 27);
        assert_json_u64(&value, &["resident_reservoir", "d1_wait_ms"], 3);
        assert_json_u64(&value, &["resident_reservoir", "hidden_ms"], 24);
        assert_json_u64(
            &value,
            &["memory_plan", "resident_subtree_budget_bytes"],
            4 * 1024 * 1024 * 1024,
        );
        assert_json_u64(&value, &["memory_plan", "required_resident_bytes"], 0);
        assert_json_u64(
            &value,
            &["memory_plan", "optional_resident_budget_bytes"],
            4 * 1024 * 1024 * 1024,
        );
        assert_json_u64(
            &value,
            &["memory_plan", "optional_resident_admitted_bytes"],
            0,
        );
        assert_json_u64(
            &value,
            &["memory_plan", "optional_resident_unassigned_bytes"],
            4 * 1024 * 1024 * 1024,
        );
        assert_json_u64(
            &value,
            &["memory_plan", "transient_pipeline_budget_bytes"],
            2 * 1024 * 1024 * 1024,
        );
        assert_json_u64(
            &value,
            &["memory_plan", "d1plus_working_set_budget_bytes"],
            0,
        );
        assert_json_str(&value, &["memory_plan", "sidecar_resident_kind"], "none");
        assert_json_u64(&value, &["memory_plan", "sidecar_resident_budget_bytes"], 0);
        assert_json_u64(
            &value,
            &["memory_plan", "sidecar_resident_required_bytes"],
            0,
        );
        assert_json_u64(
            &value,
            &["memory_plan", "sidecar_resident_admitted_rows"],
            0,
        );
        assert_json_bool(
            &value,
            &["memory_plan", "sidecar_resident_admissible"],
            false,
        );
        assert_json_u64(&value, &["memory_plan", "raw_vector_cache_budget_bytes"], 0);
        assert_json_bool(&value, &["raw_vector_cache", "enabled"], false);
        assert_json_u64(&value, &["raw_vector_cache", "budget_bytes"], 0);
        assert_json_u64(&value, &["raw_vector_cache", "hits"], 0);
        assert_json_u64(&value, &["scheduler", "gpu_workspace_pressure_events"], 0);
        assert_json_u64(&value, &["scheduler", "gpu_cpu_fallbacks"], 0);
        assert_json_u64(&value, &["pipeline", "descriptor_receive_wall_ms"], 99);
        assert_json_u64(
            &value,
            &["pipeline", "descriptor_batch_extend_wall_ms"],
            100,
        );
        assert_json_u64(&value, &["pipeline", "stage_wall_ms"], 101);
        assert_json_u64(&value, &["pipeline", "compute_flush_wall_ms"], 102);
        assert_json_u64(&value, &["pipeline", "topk_queue_roundtrip_wall_ms"], 103);
        assert_json_u64(&value, &["pipeline", "edge_queue_roundtrip_wall_ms"], 104);
        assert_json_u64(&value, &["pipeline", "edge_flush_wall_ms"], 105);
        assert_json_u64(&value, &["pipeline", "descriptor_queue_empty_count"], 4);
        assert_json_u64(&value, &["pipeline", "cuda_batch_count"], 2);
        assert_json_u64(&value, &["pipeline", "cuda_batch_leaves"], 3);
        assert_json_u64(&value, &["pipeline", "cuda_batch_rows"], 7);
        assert_json_u64(&value, &["pipeline", "cuda_batch_target_bytes"], 4096);
        assert_json_u64(&value, &["pipeline", "cuda_batch_actual_bytes"], 2048);
        assert_json_u64(&value, &["pipeline", "cuda_batch_underfilled_count"], 1);
        assert_json_u64(&value, &["leaf_flow", "host_load_wall_ms"], 61);
        assert_json_u64(&value, &["leaf_flow", "backend_compute_wall_ms"], 62);
        assert_json_u64(&value, &["leaf_flow", "compact_materialize_wall_ms"], 63);
        assert_json_u64(&value, &["leaf_flow", "hash_wall_ms"], 64);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_wall_ms"], 65);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_sort_wall_ms"], 66);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_insert_wall_ms"], 67);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_lock_wall_ms"], 71);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_insert_body_wall_ms"], 72);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_edges"], 68);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_flushes"], 69);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_groups"], 73);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_max_group_edges"], 74);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_insert_candidates"], 75);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_hash_hits"], 76);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_hash_replacements"], 77);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_hash_rejects"], 78);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_not_full_inserts"], 79);
        assert_json_u64(
            &value,
            &["leaf_flow", "edge_sink_full_replacement_checks"],
            80,
        );
        assert_json_u64(&value, &["leaf_flow", "edge_sink_full_replacements"], 81);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_full_rejects"], 82);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_farthest_recomputes"], 83);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_rotate_lefts"], 84);
        assert_json_u64(&value, &["leaf_flow", "edge_sink_rotate_rights"], 85);
        assert_json_u64(&value, &["leaf_flow", "flush_wall_ms"], 70);
        assert_json_bool(&value, &["leaf_flow", "compact_lune_witness_enabled"], true);
        assert_json_u64(&value, &["leaf_flow", "compact_lune_witness_slots"], 41);
        assert_json_u64(&value, &["leaf_flow", "compact_lune_witnesses"], 17);
        assert_json_bool(&value, &["leaf_flow", "compact_lune_prune_enabled"], true);
        assert_json_u64(
            &value,
            &["leaf_flow", "compact_lune_prune_protected_core"],
            1,
        );
        assert_json_u64(&value, &["leaf_flow", "compact_lune_pruned_slots"], 9);
        assert_json_u64(&value, &["leaf_flow", "compact_lune_pruned_rows"], 5);
        assert_json_u64(&value, &["gpu", "gpu_h2d_bytes"], 11);
        assert_json_u64(&value, &["gpu", "gpu_d2h_bytes"], 22);
        assert_json_u64(&value, &["gpu", "gpu_h2d_ms"], 31);
        assert_json_u64(&value, &["gpu", "gpu_d2h_ms"], 32);
        assert_json_u64(&value, &["gpu", "gpu_kernel_ms"], 33);
        assert_json_u64(&value, &["gpu", "gpu_sync_ms"], 34);
        assert_json_u64(&value, &["gpu", "gpu_h2d_submit_wall_ms"], 35);
        assert_json_u64(&value, &["gpu", "gpu_kernel_launch_wall_ms"], 36);
        assert_json_u64(&value, &["gpu", "gpu_kernel_elapsed_ms"], 37);
        assert_json_u64(&value, &["gpu", "gpu_d2h_submit_wall_ms"], 38);
        assert_json_u64(&value, &["gpu", "gpu_post_d2h_sync_wall_ms"], 39);
        assert_json_u64(&value, &["gpu", "gpu_stream_total_elapsed_ms"], 40);
        assert_json_u64(&value, &["gpu", "gpu_workspace_pressure_events"], 0);
        assert_json_u64(&value, &["gpu", "gpu_cpu_fallbacks"], 0);
        assert_json_u64(
            &value,
            &["resident_subtree", "resident_subtree_emit_wall_ms"],
            67,
        );
        assert_json_u64(
            &value,
            &["resident_subtree", "resident_subtree_leaf_count"],
            5,
        );
        assert_json_u64(
            &value,
            &["resident_subtree", "resident_subtree_fallback_points"],
            89,
        );
        assert_json_u64(
            &value,
            &["resident_subtree", "resident_subtree_max_points"],
            55,
        );
        assert_json_f64(
            &value,
            &["resident_subtree", "resident_subtree_avg_points"],
            64.0,
        );
        assert!(text.contains("\"memory\""));
        assert!(text.contains("\"rss_kb_samples\""));
        assert!(text.contains("\"vm_peak_kb\""));
        assert!(text.contains("\"memory_budget_bytes\""));
        assert!(text.contains("\"effective_threads\""));
        assert!(text.contains("\"pool_threads\""));
        assert!(text.contains("\"leaf_budget_threads\""));
        assert!(text.contains("\"reduce_shard_points\""));
        assert!(text.contains("\"spill\""));
        assert!(text.contains("\"part_files\""));
        assert!(text.contains("\"segment_count\""));
        assert!(text.contains("\"record_bytes\""));
        assert!(text.contains("\"manifest_bytes\""));
        assert!(text.contains("\"reduce\""));
        assert!(text.contains("\"segments_scanned\""));
        assert!(text.contains("\"part_files_opened\""));
        assert!(text.contains("\"temp_files\""));
        assert!(text.contains("\"candidate\""));
        assert!(text.contains("\"partition_depth1\""));
        assert!(text.contains("\"stage_ms\""));
        assert!(text.contains("\"streaming\""));
        assert!(text.contains("\"d0_ads_assignment\": 13"));
        assert!(text.contains("\"d1_ads_assignment\": 17"));
        assert!(text.contains("\"d1_materialize\": 19"));
        assert!(text.contains("\"d2_replay\": 23"));
        assert!(text.contains("\"external_reduce\": 33"));
        assert!(text.contains("\"total\""));
        assert!(text.contains("\"io_uring\""));
        assert!(text.contains("\"point_gather_batches\""));
        assert!(text.contains("\"point_gather_fallbacks\""));
        assert!(text.contains("\"resident_subtree\""));
        assert!(text.contains("\"resident_subtree_count\": 2"));
        assert!(text.contains("\"resident_subtree_points\": 128"));
        assert!(text.contains("\"resident_subtree_bytes\": 4096"));
        assert!(text.contains("\"resident_subtree_fallback_count\": 1"));
        assert!(text.contains("\"producer_wait_ms\""));
        assert!(text.contains("\"peak_leaf_descriptor_queue\""));
        assert!(text.contains("\"producer_blocked_ms\""));
        assert!(text.contains("\"cpu_shadow_unsupported_fallbacks\""));
        assert!(text.contains("\"gpu_workspace_pressure_events\""));
        assert!(text.contains("\"gpu_cpu_fallbacks\""));
        assert!(text.contains("\"post_producer_leaf_drain_ms\""));
        assert!(text.contains("\"pipeline\""));
        assert!(text.contains("\"descriptor_receive_wall_ms\": 99"));
        assert!(text.contains("\"descriptor_batch_extend_wall_ms\": 100"));
        assert!(text.contains("\"stage_wall_ms\": 101"));
        assert!(text.contains("\"compute_flush_wall_ms\": 102"));
        assert!(text.contains("\"topk_queue_roundtrip_wall_ms\": 103"));
        assert!(text.contains("\"edge_queue_roundtrip_wall_ms\": 104"));
        assert!(text.contains("\"edge_flush_wall_ms\": 105"));
        assert!(text.contains("\"descriptor_queue_empty_count\": 4"));
        assert!(text.contains("\"cuda_batch_count\": 2"));
        assert!(text.contains("\"cuda_batch_leaves\": 3"));
        assert!(text.contains("\"cuda_batch_rows\": 7"));
        assert!(text.contains("\"cuda_batch_target_bytes\": 4096"));
        assert!(text.contains("\"cuda_batch_actual_bytes\": 2048"));
        assert!(text.contains("\"cuda_batch_underfilled_count\": 1"));
        assert!(text.contains("\"leaf_flow\""));
        assert!(text.contains("\"host_load_wall_ms\": 61"));
        assert!(text.contains("\"backend_compute_wall_ms\": 62"));
        assert!(text.contains("\"compact_materialize_wall_ms\": 63"));
        assert!(text.contains("\"hash_wall_ms\": 64"));
        assert!(text.contains("\"edge_sink_wall_ms\": 65"));
        assert!(text.contains("\"edge_sink_sort_wall_ms\": 66"));
        assert!(text.contains("\"edge_sink_insert_wall_ms\": 67"));
        assert!(text.contains("\"edge_sink_edges\": 68"));
        assert!(text.contains("\"edge_sink_flushes\": 69"));
        assert!(text.contains("\"flush_wall_ms\": 70"));
        assert!(text.contains("\"gpu\""));
        assert!(text.contains("\"gpu_h2d_bytes\": 11"));
        assert!(text.contains("\"gpu_d2h_bytes\": 22"));
        assert!(text.contains("\"gpu_h2d_ms\": 31"));
        assert!(text.contains("\"gpu_d2h_ms\": 32"));
        assert!(text.contains("\"gpu_kernel_ms\": 33"));
        assert!(text.contains("\"gpu_sync_ms\": 34"));
        assert!(text.contains("\"gpu_h2d_submit_wall_ms\": 35"));
        assert!(text.contains("\"gpu_kernel_launch_wall_ms\": 36"));
        assert!(text.contains("\"gpu_kernel_elapsed_ms\": 37"));
        assert!(text.contains("\"gpu_d2h_submit_wall_ms\": 38"));
        assert!(text.contains("\"gpu_post_d2h_sync_wall_ms\": 39"));
        assert!(text.contains("\"gpu_stream_total_elapsed_ms\": 40"));
        assert!(text.contains("\"gpu_workspace_peak_bytes\": 0"));
        assert!(text.contains("\"gpu_workspace_budget_bytes\": 0"));
        assert!(text.contains("\"gpu_workspace_pressure_events\": 0"));
        assert!(text.contains("\"gpu_cpu_fallbacks\": 0"));
        assert!(text.contains("\"resident_subtree_emit_wall_ms\": 67"));
        assert!(text.contains("\"resident_subtree_leaf_count\": 5"));
        assert!(text.contains("\"resident_subtree_fallback_points\": 89"));
        assert!(text.contains("\"resident_subtree_max_points\": 55"));
        assert!(text.contains("\"resident_subtree_avg_points\": 64.000000"));
        assert!(text.contains("\"partitioner\""));
        assert!(text.contains("\"kind\": \"rbc\""));
        assert!(text.contains("\"kmeans\""));
        assert!(text.contains("\"assignment_passes\""));
        assert!(text.contains("\"root_fanout\""));
        assert!(text.contains("\"policy\": \"cell-radius-certified\""));
        assert!(
            text.contains("\"selection_semantics\": \"cell_radius_certified_root_assignment_v1\"")
        );
        assert!(text.contains("\"enabled\": true"));
        assert!(text.contains("\"fixed_fanout\": 8"));
        assert!(text.contains("\"certified_points\": 24"));
        assert!(text.contains("\"insufficient_radius_points\": 2"));
        assert!(text.contains("\"selector_ms\": 12"));
        assert!(text.contains("\"radius_update_ms\": 3"));
        assert!(text.contains("\"root_leader_hash\": 12345"));
    }

    #[test]
    fn oom_profile_json_zero_resident_subtree_average_is_numeric_zero() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let profile = minimal_oom_profile_for_json(super::TempFileReport::default());

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(&value, &["resident_subtree", "resident_subtree_count"], 0);
        assert_json_u64(&value, &["resident_subtree", "resident_subtree_points"], 0);
        assert_json_f64(
            &value,
            &["resident_subtree", "resident_subtree_avg_points"],
            0.0,
        );
    }

    #[test]
    fn oom_profile_json_reports_resident_emit_pipeline() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.resident_emit = super::ResidentEmitProfile {
            parallel: true,
            chunk_sources: 32_768,
            fifo_chunks: 2,
            initial_workers: 8,
            peak_workers: 16,
            chunks: 4,
            rows: 100_000,
            max_completed_inflight: 2,
            finalize_service_wall: Duration::from_millis(11),
            callback_service_wall: Duration::from_millis(7),
            consumer_wait_wall: Duration::from_millis(3),
            producer_backpressure_wall: Duration::from_millis(5),
            writer_buffer_bytes: 8 * 1024 * 1024,
        };

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(&profile_path).unwrap()).unwrap();

        assert_json_bool(&value, &["resident_emit", "parallel"], true);
        assert_json_u64(&value, &["resident_emit", "chunk_sources"], 32_768);
        assert_json_u64(&value, &["resident_emit", "fifo_chunks"], 2);
        assert_json_u64(&value, &["resident_emit", "initial_workers"], 8);
        assert_json_u64(&value, &["resident_emit", "peak_workers"], 16);
        assert_json_u64(&value, &["resident_emit", "chunks"], 4);
        assert_json_u64(&value, &["resident_emit", "rows"], 100_000);
        assert_json_u64(&value, &["resident_emit", "max_completed_inflight"], 2);
        assert_json_u64(&value, &["resident_emit", "finalize_service_ms"], 11);
        assert_json_u64(&value, &["resident_emit", "callback_service_ms"], 7);
        assert_json_u64(&value, &["resident_emit", "consumer_wait_ms"], 3);
        assert_json_u64(&value, &["resident_emit", "producer_backpressure_ms"], 5);
        assert_json_u64(
            &value,
            &["resident_emit", "writer_buffer_bytes"],
            8 * 1024 * 1024,
        );
    }

    #[test]
    fn oom_profile_json_reports_d1plus_lazy_split_mode() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.d1plus_lazy_split_mode = super::PiPNNGpuSubtreeD1PlusLazySplitMode::Off;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(&profile_path).unwrap()).unwrap();

        assert_json_str(&value, &["gpu_subtree", "d1plus_lazy_split_mode"], "off");
    }

    #[test]
    fn hashprune_evidence_profile_merges_leaf_and_reduce_exactly_once() {
        let leaf = super::HashPruneEvidenceProfile {
            annotation_wall: Duration::from_millis(11),
            coalesce_sort_wall: Duration::from_millis(3),
            coalesce_merge_wall: Duration::from_millis(5),
            coalesce_wall: Duration::from_millis(8),
            reciprocal_checks: 17,
            class_counts: [1, 2, 3, 4, 5, 6],
            candidates_before_coalesce: 19,
            overflow_inserts: 23,
            spill_bytes_avoided: 29,
            ..super::HashPruneEvidenceProfile::default()
        };
        let reduce = super::HashPruneEvidenceProfile {
            annotation_wall: Duration::from_millis(31),
            coalesce_sort_wall: Duration::from_millis(7),
            coalesce_merge_wall: Duration::from_millis(11),
            coalesce_wall: Duration::from_millis(18),
            reciprocal_checks: 41,
            class_counts: [6, 5, 4, 3, 2, 1],
            candidates_before_coalesce: 43,
            overflow_inserts: 47,
            spill_bytes_avoided: 53,
            ..super::HashPruneEvidenceProfile::default()
        };

        let merged = super::final_hashprune_evidence_profile(&leaf, &reduce);

        assert_eq!(merged.annotation_wall, Duration::from_millis(42));
        assert_eq!(merged.coalesce_sort_wall, Duration::from_millis(10));
        assert_eq!(merged.coalesce_merge_wall, Duration::from_millis(16));
        assert_eq!(merged.coalesce_wall, Duration::from_millis(26));
        assert_eq!(merged.reciprocal_checks, 58);
        assert_eq!(merged.class_counts, [7; 6]);
        assert_eq!(merged.candidates_before_coalesce, 62);
        assert_eq!(merged.overflow_inserts, 70);
        assert_eq!(merged.spill_bytes_avoided, 82);
        assert_eq!(leaf.annotation_wall, Duration::from_millis(11));
        assert_eq!(reduce.annotation_wall, Duration::from_millis(31));
    }

    #[test]
    fn oom_profile_json_reports_hashprune_coalesce_sort_and_merge_ms() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        let leaf = super::HashPruneEvidenceProfile {
            coalesce_sort_wall: Duration::from_millis(3),
            coalesce_merge_wall: Duration::from_millis(5),
            coalesce_wall: Duration::from_millis(8),
            ..super::HashPruneEvidenceProfile::default()
        };
        let reduce = super::HashPruneEvidenceProfile {
            coalesce_sort_wall: Duration::from_millis(7),
            coalesce_merge_wall: Duration::from_millis(11),
            coalesce_wall: Duration::from_millis(18),
            ..super::HashPruneEvidenceProfile::default()
        };
        profile.hashprune_evidence = super::final_hashprune_evidence_profile(&leaf, &reduce);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();

        assert_json_u64(&value, &["hashprune_evidence", "coalesce_sort_ms"], 10);
        assert_json_u64(&value, &["hashprune_evidence", "coalesce_merge_ms"], 16);
        assert_json_u64(&value, &["hashprune_evidence", "coalesce_ms"], 26);
    }

    #[test]
    fn oom_profile_json_reports_hashprune_evidence_exact_keys() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let profile = minimal_oom_profile_for_json(super::TempFileReport::default());

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        let evidence = value
            .get("hashprune_evidence")
            .and_then(Value::as_object)
            .expect("missing hashprune_evidence JSON object");
        let expected_keys = [
            "mode",
            "overflow_budget",
            "annotation_ms",
            "coalesce_sort_ms",
            "coalesce_merge_ms",
            "coalesce_ms",
            "reciprocal_checks",
            "reciprocal_hits",
            "support_once_candidates",
            "support_twice_candidates",
            "class_reverse",
            "class_reverse_rank0",
            "class_direct",
            "class_bidirectional",
            "class_direct_rank0",
            "class_bidirectional_direct_rank0",
            "candidates_before_coalesce",
            "candidates_after_coalesce",
            "duplicate_records_removed",
            "unannotated_candidates",
            "qualified_collision_losers",
            "overflow_inserts",
            "overflow_replacements",
            "overflow_rejects",
            "overflow_evictions_by_new_hash",
            "rows_with_overflow",
            "spill_records_avoided",
            "spill_bytes_avoided",
        ];

        assert_eq!(evidence.len(), expected_keys.len());
        for &key in &expected_keys {
            assert!(
                evidence.contains_key(key),
                "missing exact JSON key hashprune_evidence.{key}"
            );
        }
        assert_json_str(&value, &["hashprune_evidence", "mode"], "off");
        for &key in expected_keys.iter().skip(1) {
            assert_json_u64(&value, &["hashprune_evidence", key], 0);
        }
    }

    #[test]
    fn oom_profile_json_reports_online_suffix_reduce() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.online_suffix_mode = super::PiPNNOnlineSuffixReduceMode::WritebackCache;
        profile.online_suffix_fallback_reason = None;
        profile.online_suffix_config.configured_state_cache_bytes = 1024;
        profile.online_suffix_config.configured_read_window_bytes = 2048;
        profile.online_suffix_config.effective_state_cache_bytes = 4096;
        profile.online_suffix_config.effective_read_window_bytes = 8192;
        profile.online_suffix.direct_candidates = 100;
        profile.online_suffix.direct_batches = 5;
        profile.online_suffix.direct_queue_peak_bytes = 4096;
        profile.online_suffix.spill_fallback_candidates = 7;
        profile.online_suffix.spill_fallback_bytes = 112;
        profile.online_suffix.spill_segments_consumed = 2;
        profile.online_suffix.state_pages_read = 3;
        profile.online_suffix.state_pages_written = 4;
        profile.online_suffix.state_cache_partitions = 4;
        profile.online_suffix.state_cache_hits = 10;
        profile.online_suffix.state_cache_misses = 2;
        profile.online_suffix.state_cache_coalesced_updates = 7;
        profile.online_suffix.state_writeback_records = 12;
        profile.online_suffix.state_writeback_cohorts = 3;
        profile.online_suffix.state_alloc_empty_page = 8;
        profile.online_suffix.state_alloc_resident_partial_page = 3;
        profile.online_suffix.state_alloc_cold_partial_page = 1;
        profile.online_suffix.pinned_prefix_points = 100;
        profile.online_suffix.backlog_bytes_at_last_gpu = 2048;
        profile.online_suffix.post_gpu_drain_wall = Duration::from_millis(11);
        profile.online_suffix.online_reduce_wall = Duration::from_millis(21);
        profile.online_suffix.online_reduce_hidden_wall = Duration::from_millis(10);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(&value, &["online_suffix_reduce", "mode"], "writeback-cache");
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "effective_state_cache_bytes"],
            4096,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "effective_read_window_bytes"],
            8192,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "direct_candidates"], 100);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "spill_fallback_candidates"],
            7,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "state_pages_written"], 4);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "state_cache_partitions"],
            4,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "state_cache_hits"], 10);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "state_cache_coalesced_updates"],
            7,
        );
        assert_json_f64(
            &value,
            &["online_suffix_reduce", "state_writeback_records_per_page"],
            3.0,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "state_alloc_empty_page"],
            8,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "pinned_prefix_points"],
            100,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "post_gpu_drain_ms"], 11);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "online_reduce_hidden_ms"],
            10,
        );
    }

    #[test]
    fn log_assisted_profile_json_reports_log_and_compaction_counters() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.online_suffix_mode = super::PiPNNOnlineSuffixReduceMode::LogAssisted;
        profile.online_suffix_config.effective_delta_budget_bytes = 8192;
        profile.online_suffix_config.effective_delta_buffer_bytes = 4096;
        profile.online_suffix_config.effective_compaction_bytes = 2048;
        profile.online_suffix_config.effective_range_points = 1024;
        profile.online_suffix_config.effective_compaction_workers = 2;
        profile
            .online_suffix_config
            .effective_drain_compaction_workers = 8;
        profile.online_suffix.delta_records = 17;
        profile.online_suffix.borrowed_spill_records = 11;
        profile.online_suffix.borrowed_spill_runs = 3;
        profile.online_suffix.borrowed_spill_bytes = 176;
        profile.online_suffix.cold_batch_commits = 4;
        profile.online_suffix.cold_batch_records = 13;
        profile.online_suffix.delta_buffer_flushes = 3;
        profile.online_suffix.checkpoint_runs = 2;
        profile.online_suffix.delta_uncompacted_peak_bytes = 4096;
        profile.online_suffix.range_compactions = 2;
        profile.online_suffix.delta_run_pressure_crossings = 3;
        profile.online_suffix.delta_run_pressure_compactions = 2;
        profile.online_suffix.range_pre_gpu_materializations = 1;
        profile.online_suffix.log_shards_finalized = 4;
        profile.online_suffix.range_ingress_drained_compactions = 48;
        profile.online_suffix.delta_peak_active_runs_per_range = 384;
        profile.online_suffix.range_compactions_inflight = 0;
        profile.online_suffix.range_compactions_peak_inflight = 8;
        profile
            .online_suffix
            .post_gpu_range_compactions_peak_inflight = 6;
        profile.online_suffix.range_rows_changed = 1536;
        profile.online_suffix.range_compaction_input_bytes = 16384;
        profile.online_suffix.range_compaction_output_bytes = 32768;
        profile.online_suffix.range_manifest_switches = 2;
        profile.online_suffix.cold_miss_base_reads = 0;
        profile.online_suffix.post_gpu_uncompacted_bytes = 512;
        profile.online_suffix.range_compaction_read_wall = Duration::from_millis(4);
        profile.online_suffix.range_compaction_merge_wall = Duration::from_millis(5);
        profile.online_suffix.range_compaction_write_wall = Duration::from_millis(6);
        profile.online_suffix.range_compaction_hidden_wall = Duration::from_millis(9);
        profile.online_suffix.post_gpu_compaction_wall = Duration::from_millis(6);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(&value, &["online_suffix_reduce", "mode"], "log-assisted");
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "effective_delta_budget_bytes"],
            8192,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "effective_compaction_bytes"],
            2048,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "delta_records"], 17);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "borrowed_spill_records"],
            11,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "borrowed_spill_runs"], 3);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "borrowed_spill_bytes"],
            176,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "cold_batch_commits"], 4);
        assert_json_u64(&value, &["online_suffix_reduce", "cold_batch_records"], 13);
        assert_json_u64(&value, &["online_suffix_reduce", "delta_buffer_flushes"], 3);
        assert_json_u64(&value, &["online_suffix_reduce", "checkpoint_runs"], 2);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "delta_uncompacted_peak_bytes"],
            4096,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "range_compactions"], 2);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "delta_run_pressure_crossings"],
            3,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "delta_run_pressure_compactions"],
            2,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_pre_gpu_materializations"],
            1,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "log_shards_finalized"], 4);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_ingress_drained_compactions"],
            48,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "delta_peak_active_runs_per_range"],
            384,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "effective_compaction_workers"],
            2,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "effective_drain_compaction_workers"],
            8,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compactions_inflight"],
            0,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compactions_peak_inflight"],
            8,
        );
        assert_json_u64(
            &value,
            &[
                "online_suffix_reduce",
                "post_gpu_range_compactions_peak_inflight",
            ],
            6,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_rows_changed"],
            1536,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compaction_input_bytes"],
            16384,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compaction_output_bytes"],
            32768,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_manifest_switches"],
            2,
        );
        assert_json_u64(&value, &["online_suffix_reduce", "cold_miss_base_reads"], 0);
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "post_gpu_uncompacted_bytes"],
            512,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compaction_read_ms"],
            4,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compaction_merge_ms"],
            5,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compaction_write_ms"],
            6,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "range_compaction_hidden_ms"],
            9,
        );
        assert_json_u64(
            &value,
            &["online_suffix_reduce", "post_gpu_compaction_ms"],
            6,
        );
    }

    #[test]
    fn oom_profile_json_reports_hashprune_backfill_exact_keys() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.hashprune_backfill_policy =
            super::HashPruneBackfillPolicy::distance_gated_stratified(4, 1.1);
        profile.hashprune_backfill = super::HashPruneBackfillProfile {
            rows_seen: 10,
            rows_underfull_before: 8,
            rows_with_backfill: 6,
            primary_edges: 400,
            backfill_edges: 40,
            gate_rejected: 3,
            budget_rejected: 2,
            degree_before_sum: 400,
            degree_after_sum: 440,
            ..Default::default()
        };

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        let object = value
            .get("hashprune_backfill")
            .and_then(Value::as_object)
            .expect("missing hashprune_backfill JSON object");
        let expected_keys = [
            "mode",
            "ablation",
            "budget_rule",
            "budget",
            "candidate_budget",
            "derived_budget",
            "capacity_rule",
            "requested_max_backfill_per_row",
            "effective_max_backfill_per_row",
            "backup_hash_quota",
            "hash_quota",
            "requested_hash_quota",
            "effective_hash_quota",
            "degree_target",
            "degree_cap",
            "alpha",
            "priority_rule",
            "priority_alpha",
            "requested_priority_alpha",
            "effective_priority_alpha",
            "uses_alpha_gate",
            "post_veto_refill",
            "rows_seen",
            "rows_underfull_before",
            "rows_with_backfill",
            "baseline_full_rows",
            "rows_with_nonempty_shortlist",
            "shortlist_admissions",
            "shortlist_replacements",
            "shortlist_unchanged",
            "shortlist_capacity_rejections",
            "shortlisted_layer_1",
            "shortlisted_layer_2",
            "shortlisted_layer_3",
            "emitted_layer_1",
            "emitted_layer_2",
            "emitted_layer_3",
            "rows_with_deeper_backfill",
            "gate_rejected_layer_1",
            "gate_rejected_layer_2",
            "gate_rejected_layer_3",
            "rows_with_veto_holes",
            "primary_edges",
            "backfill_edges",
            "gate_rejected",
            "budget_rejected",
            "mean_degree_before",
            "mean_degree_after",
        ];
        assert_eq!(object.len(), expected_keys.len());
        for key in expected_keys {
            assert!(object.contains_key(key), "missing hashprune_backfill.{key}");
        }
        assert_json_str(
            &value,
            &["hashprune_backfill", "mode"],
            "distance-gated-stratified",
        );
        assert_json_u64(&value, &["hashprune_backfill", "budget"], 4);
        assert_json_u64(&value, &["hashprune_backfill", "candidate_budget"], 8);
        assert_json_u64(&value, &["hashprune_backfill", "backup_hash_quota"], 1);
        assert_json_u64(&value, &["hashprune_backfill", "degree_target"], 0);
        assert_json_f64(&value, &["hashprune_backfill", "alpha"], 1.1);
        assert_eq!(
            value["hashprune_backfill"]["uses_alpha_gate"],
            Value::Bool(true)
        );
        assert_json_u64(&value, &["hashprune_backfill", "rows_seen"], 10);
        assert_json_u64(&value, &["hashprune_backfill", "backfill_edges"], 40);
        assert_json_f64(&value, &["hashprune_backfill", "mean_degree_before"], 40.0);
        assert_json_f64(&value, &["hashprune_backfill", "mean_degree_after"], 44.0);
    }

    #[test]
    fn oom_profile_json_reports_hashprune_tiered_rhalf_contract() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut params = super::PiPNNParams::default();
        params.hashprune_backfill_mode =
            super::PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf;
        params.hashprune_backfill_ablation = super::PiPNNHashPruneBackfillAblation::None;
        params.hashprune_backfill_alpha = 1.1;

        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.hashprune_backfill_policy =
            super::HashPruneBackfillPolicy::from_params(&params).unwrap();
        profile.hashprune_backfill_degree_cap = 64;
        profile.hashprune_backfill.shortlisted_by_layer = [11, 7, 3];
        profile.hashprune_backfill.emitted_by_layer = [10, 5, 2];
        profile.hashprune_backfill.gate_rejected_by_layer = [1, 2, 1];

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(
            &value,
            &["hashprune_backfill", "mode"],
            "distance-gated-tiered-rhalf",
        );
        assert_json_str(&value, &["hashprune_backfill", "ablation"], "none");
        assert_json_str(
            &value,
            &["hashprune_backfill", "budget_rule"],
            "floor-r-half",
        );
        assert_json_u64(&value, &["hashprune_backfill", "derived_budget"], 32);
        assert_json_str(
            &value,
            &["hashprune_backfill", "capacity_rule"],
            "min-derived-budget-spare",
        );
        assert_eq!(
            value["hashprune_backfill"]["requested_max_backfill_per_row"],
            Value::Null
        );
        assert_json_u64(
            &value,
            &["hashprune_backfill", "effective_max_backfill_per_row"],
            32,
        );
        assert_json_u64(&value, &["hashprune_backfill", "hash_quota"], 3);
        assert_json_u64(&value, &["hashprune_backfill", "degree_cap"], 64);
        assert_json_f64(&value, &["hashprune_backfill", "alpha"], 1.1);
        assert_eq!(
            value["hashprune_backfill"]["uses_alpha_gate"],
            Value::Bool(true)
        );
        assert_eq!(
            value["hashprune_backfill"]["post_veto_refill"],
            Value::Bool(false)
        );
        assert_json_u64(&value, &["hashprune_backfill", "shortlisted_layer_1"], 11);
        assert_json_u64(&value, &["hashprune_backfill", "shortlisted_layer_2"], 7);
        assert_json_u64(&value, &["hashprune_backfill", "shortlisted_layer_3"], 3);
        assert_json_u64(&value, &["hashprune_backfill", "emitted_layer_1"], 10);
        assert_json_u64(&value, &["hashprune_backfill", "emitted_layer_2"], 5);
        assert_json_u64(&value, &["hashprune_backfill", "emitted_layer_3"], 2);
        assert_json_u64(&value, &["hashprune_backfill", "gate_rejected_layer_1"], 1);
        assert_json_u64(&value, &["hashprune_backfill", "gate_rejected_layer_2"], 2);
        assert_json_u64(&value, &["hashprune_backfill", "gate_rejected_layer_3"], 1);
    }

    #[test]
    fn oom_profile_json_reports_named_soft_tier_contract() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut params = super::PiPNNParams::default();
        params.hashprune_backfill_mode = super::PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_rhalf_hash_quota = Some(2);
        params.hashprune_soft_tier_priority_alpha = Some(1.4);
        params.hashprune_rhalf_max_backfill_per_row = Some(3);

        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.hashprune_backfill_policy =
            super::HashPruneBackfillPolicy::from_params(&params).unwrap();
        profile.hashprune_backfill_degree_cap = 64;
        profile.hashprune_backfill.rows_with_deeper_backfill = 7;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(
            &value,
            &["hashprune_backfill", "mode"],
            "rhalf-soft-tier-q3",
        );
        assert_json_str(
            &value,
            &["hashprune_backfill", "priority_rule"],
            "soft-tier-q3",
        );
        assert_json_f64(&value, &["hashprune_backfill", "priority_alpha"], 1.4);
        assert_json_u64(&value, &["hashprune_backfill", "requested_hash_quota"], 2);
        assert_json_u64(&value, &["hashprune_backfill", "effective_hash_quota"], 2);
        assert_json_f64(
            &value,
            &["hashprune_backfill", "requested_priority_alpha"],
            1.4,
        );
        assert_json_f64(
            &value,
            &["hashprune_backfill", "effective_priority_alpha"],
            1.4,
        );
        assert_json_u64(&value, &["hashprune_backfill", "derived_budget"], 32);
        assert_json_str(
            &value,
            &["hashprune_backfill", "capacity_rule"],
            "min-derived-budget-row-cap-spare",
        );
        assert_json_u64(
            &value,
            &["hashprune_backfill", "requested_max_backfill_per_row"],
            3,
        );
        assert_json_u64(
            &value,
            &["hashprune_backfill", "effective_max_backfill_per_row"],
            3,
        );
        assert_json_u64(&value, &["hashprune_backfill", "backup_hash_quota"], 2);
        assert_json_u64(
            &value,
            &["hashprune_backfill", "rows_with_deeper_backfill"],
            7,
        );
        assert_eq!(
            value["hashprune_backfill"]["uses_alpha_gate"],
            Value::Bool(false)
        );
    }

    #[test]
    fn oom_profile_json_reports_hashprune_backfill_off_as_zero_work() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(&value, &["hashprune_backfill", "mode"], "off");
        assert_json_u64(&value, &["hashprune_backfill", "budget"], 0);
        assert_json_u64(&value, &["hashprune_backfill", "candidate_budget"], 0);
        assert_json_u64(&value, &["hashprune_backfill", "backup_hash_quota"], 1);
        assert_json_u64(&value, &["hashprune_backfill", "degree_target"], 0);
        assert_json_f64(&value, &["hashprune_backfill", "alpha"], 1.0);
        for key in [
            "rows_seen",
            "rows_underfull_before",
            "rows_with_backfill",
            "primary_edges",
            "backfill_edges",
            "gate_rejected",
            "budget_rejected",
        ] {
            assert_json_u64(&value, &["hashprune_backfill", key], 0);
        }
        assert_json_f64(&value, &["hashprune_backfill", "mean_degree_before"], 0.0);
        assert_json_f64(&value, &["hashprune_backfill", "mean_degree_after"], 0.0);
    }

    #[test]
    fn oom_profile_json_reports_hashprune_tiered_d52_contract() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.hashprune_backfill_policy =
            super::HashPruneBackfillPolicy::distance_gated_tiered_d52(8, 1.1);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(
            &value,
            &["hashprune_backfill", "mode"],
            "distance-gated-tiered-d52",
        );
        assert_json_u64(&value, &["hashprune_backfill", "budget"], 8);
        assert_json_u64(&value, &["hashprune_backfill", "candidate_budget"], 10);
        assert_json_u64(&value, &["hashprune_backfill", "backup_hash_quota"], 3);
        assert_json_u64(&value, &["hashprune_backfill", "degree_target"], 52);
    }

    #[test]
    fn oom_profile_json_includes_serial_nonoverlap_timing() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.partitioner = PiPNNPartitionerKind::GpuSubtree;
        profile.sketch_persist_wall = Duration::from_millis(10);
        profile.streaming_wall = Duration::from_millis(110);
        profile.reduce_wall = Duration::from_millis(20);
        profile.emit_wall = Duration::from_millis(30);
        profile.total_wall = Duration::from_millis(200);
        profile.rbc.d0_ads_assignment_wall = Duration::from_millis(12);
        profile.rbc.gpu_subtree_d0_wall = Duration::from_millis(25);
        profile.rbc.gpu_subtree_d0_tagore_train_wall = Duration::from_millis(2);
        profile.rbc.gpu_subtree_sq_staging_raw_read_wall = Duration::from_millis(3);
        profile.rbc.gpu_subtree_sq_staging_raw_h2d_wall = Duration::from_millis(4);
        profile.rbc.gpu_subtree_sq_staging_gpu_quantize_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_sq_staging_gpu_d0_assign_wall = Duration::from_millis(6);
        profile.rbc.gpu_subtree_sq_staging_qcode_d2h_wall = Duration::from_millis(2);
        profile
            .rbc
            .gpu_subtree_sq_staging_assignment_fragment_d2h_wall = Duration::from_millis(2);
        profile
            .rbc
            .gpu_subtree_sq_staging_assignment_fragment_reduce_wall = Duration::from_millis(2);
        profile.rbc.gpu_subtree_sq_staging_sidecar_write_wall = Duration::from_millis(2);
        profile.rbc.d1_materialize_wall = Duration::from_millis(8);
        profile.rbc.d2_replay_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_d1plus_producer_wall = Duration::from_millis(70);
        profile.rbc.gpu_subtree_d1plus_backend_wall = Duration::from_millis(40);
        profile.rbc.gpu_subtree_sq_wmma_sidecar_gather_wall = Duration::from_millis(7);
        profile.rbc.gpu_subtree_sidecar_gather_active_wall = Duration::from_millis(7);
        profile.rbc.gpu_subtree_sidecar_gather_hidden_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_gpu_wait_for_sidecar_wall = Duration::from_millis(2);
        profile.rbc.gpu_subtree_d1plus_child_replay_wall = Duration::from_millis(6);
        profile.rbc.gpu_subtree_d1plus_lazy_split_wall = Duration::from_millis(3);
        profile.rbc.gpu_subtree_d1plus_push_wait_wall = Duration::from_millis(1);
        profile.rbc.gpu_subtree_d1plus_validate_wall = Duration::from_millis(4);
        profile.rbc.gpu_subtree_d1plus_merge_profile_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_d1plus_push_batch_wall = Duration::from_millis(8);
        profile.rbc.gpu_subtree_host_vector_dedup_wall = Duration::from_millis(9);
        profile.rbc.gpu_subtree_host_vector_child_gather_wall = Duration::from_millis(11);
        profile.rbc.gpu_subtree_sidecar_preload_workers = 3;
        profile.rbc.gpu_subtree_sidecar_preload_children = 5;
        profile.rbc.gpu_subtree_sidecar_queue_max_bytes = 1024;
        profile.rbc.gpu_subtree_sidecar_queue_max_children = 2;
        profile.rbc.gpu_subtree_d1plus_preflight_split_count = 4;
        profile.rbc.gpu_subtree_d1plus_preflight_split_rows = 512;
        profile.rbc.gpu_subtree_sq_wmma_h2d_wall = Duration::from_millis(3);
        profile.rbc.gpu_subtree_persistent_total_wall = Duration::from_millis(20);
        profile.rbc.gpu_subtree_cpu_result_scatter_wall = Duration::from_millis(4);
        profile.rbc.gpu_subtree_cpu_compact_flush_wall = Duration::from_millis(99);
        profile.rbc.gpu_subtree_post_last_gpu_cpu_tail_wall = Duration::from_millis(47);
        profile.rbc.gpu_subtree_post_last_gpu_cpu_tail_observed = true;
        profile.leaf.edge_sink_wall = Duration::from_millis(88);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_str(
            &value,
            &["serial_timing", "schema"],
            "gpu_subtree_serial_nonoverlap_v1",
        );
        assert_json_str(&value, &["serial_timing", "unit"], "ms");
        assert_json_u64(&value, &["serial_timing", "rows", "sketch_persist_ms"], 10);
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d0_root_assignment_ms"],
            0,
        );
        assert_json_u64(&value, &["serial_timing", "rows", "d0_kmeans_train_ms"], 2);
        assert_json_u64(&value, &["serial_timing", "rows", "d0_sq_raw_read_ms"], 3);
        assert_json_u64(&value, &["serial_timing", "rows", "d0_sq_raw_h2d_ms"], 4);
        assert_json_u64(&value, &["serial_timing", "rows", "d0_sq_quantize_ms"], 5);
        assert_json_u64(&value, &["serial_timing", "rows", "d0_sq_d0_assign_ms"], 6);
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d0_sq_qcode_write_ms"],
            5,
        );
        assert_json_u64(&value, &["serial_timing", "rows", "d0_residual_ms"], 0);
        assert_json_u64(&value, &["serial_timing", "rows", "d1_materialize_ms"], 8);
        assert_json_u64(&value, &["serial_timing", "rows", "d2_replay_ms"], 0);
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_sidecar_wait_ms"],
            2,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_payload_h2d_ms"],
            3,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_persistent_kernel_ms"],
            20,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_cpu_result_scatter_ms"],
            4,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_backend_residual_ms"],
            13,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_child_replay_ms"],
            6,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_lazy_split_ms"],
            3,
        );
        assert_json_u64(&value, &["serial_timing", "rows", "d1plus_push_wait_ms"], 1);
        assert_json_u64(&value, &["serial_timing", "rows", "d1plus_validate_ms"], 4);
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_merge_profile_ms"],
            5,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_push_batch_residual_ms"],
            7,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_producer_residual_ms"],
            2,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "streaming_residual_ms"],
            7,
        );
        assert_json_u64(&value, &["serial_timing", "rows", "external_reduce_ms"], 20);
        assert_json_u64(&value, &["serial_timing", "rows", "emit_ms"], 30);
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "stage_outer_residual_ms"],
            30,
        );
        assert_json_u64(&value, &["serial_timing", "accounted_sum_ms"], 200);
        assert_json_u64(&value, &["serial_timing", "stage_total_ms"], 200);
        assert_json_bool(&value, &["serial_timing", "sum_matches_stage_total"], true);
        assert!(
            value
                .pointer("/serial_timing/rows/post_last_gpu_cpu_tail_ms")
                .is_none()
        );
        assert!(
            value
                .pointer("/serial_timing/rows/post_last_gpu_cpu_tail_observed")
                .is_none()
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "d1plus_sidecar_gather_total_ms",
            ],
            7,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "d0_root_assignment_total_ms",
            ],
            25,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d0_sq_qcode_write_total_ms"],
            8,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "d0_nested_overlap_hidden_ms",
            ],
            3,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "cpu_compact_flush_ms"],
            99,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "post_last_gpu_cpu_tail_ms"],
            47,
        );
        assert_json_bool(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "post_last_gpu_cpu_tail_observed",
            ],
            true,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "edge_sink_ms"],
            88,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d1plus_child_replay_ms"],
            6,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d1plus_lazy_split_ms"],
            3,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d1plus_push_wait_ms"],
            1,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d1plus_validate_ms"],
            4,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d1plus_merge_profile_ms"],
            5,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "d1plus_push_batch_total_ms"],
            8,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "host_vector_dedup_ms"],
            9,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "host_vector_child_gather_ms",
            ],
            11,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "sidecar_preload_workers"],
            3,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "sidecar_preload_children"],
            5,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "sidecar_queue_max_bytes"],
            1024,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "diagnostics", "sidecar_queue_max_children"],
            2,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "d1plus_preflight_split_count",
            ],
            4,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "d1plus_preflight_split_rows",
            ],
            512,
        );
    }

    #[test]
    fn oom_profile_json_charges_async_scatter_wait_to_producer() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.partitioner = PiPNNPartitionerKind::GpuSubtree;
        profile.streaming_wall = Duration::from_millis(80);
        profile.total_wall = Duration::from_millis(80);
        profile.rbc.gpu_subtree_d1plus_producer_wall = Duration::from_millis(80);
        profile.rbc.gpu_subtree_d1plus_backend_wall = Duration::from_millis(50);
        profile.rbc.gpu_subtree_sq_wmma_h2d_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_persistent_total_wall = Duration::from_millis(30);
        profile.rbc.gpu_subtree_cpu_result_scatter_wall = Duration::from_millis(20);
        profile.rbc.gpu_subtree_async_result_scatter_requested = "1".to_string();
        profile.rbc.gpu_subtree_async_result_scatter_enabled = true;
        profile.rbc.gpu_subtree_async_result_scatter_wait_wall = Duration::from_millis(4);
        profile.rbc.gpu_subtree_persistent_async_d2h_requested = true;
        profile.rbc.gpu_subtree_persistent_async_d2h_enabled = true;
        profile.rbc.gpu_subtree_persistent_device_output_slots = 2;
        profile.rbc.gpu_subtree_persistent_device_output_slot_bytes = 1024;
        profile.rbc.gpu_subtree_persistent_pinned_host_slots = 2;
        profile.rbc.gpu_subtree_persistent_pinned_host_slot_bytes = 2048;
        profile.rbc.gpu_subtree_persistent_session_init_wall = Duration::from_millis(2);
        profile.rbc.gpu_subtree_persistent_pinned_allocation_wall = Duration::from_millis(1);
        profile.rbc.gpu_subtree_persistent_d2h_header_copy_wall = Duration::from_millis(3);
        profile.rbc.gpu_subtree_persistent_d2h_payload_event_wall = Duration::from_millis(12);
        profile.rbc.gpu_subtree_persistent_d2h_bytes = 8192;
        profile.rbc.gpu_subtree_persistent_device_slot_wait_wall = Duration::from_millis(2);
        profile.rbc.gpu_subtree_persistent_host_slot_wait_wall = Duration::from_millis(1);
        profile.rbc.gpu_subtree_persistent_d2h_final_drain_wall = Duration::from_millis(1);
        profile.rbc.gpu_subtree_persistent_d2h_hidden_estimate_wall = Duration::from_millis(12);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(&profile_path).unwrap()).unwrap();

        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_cpu_result_scatter_ms"],
            4,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_backend_residual_ms"],
            15,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "persistent_session_init_ms"],
            2,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "persistent_session_drain_ms"],
            1,
        );
        assert_json_u64(&value, &["serial_timing", "accounted_sum_ms"], 80);
        assert_json_u64(&value, &["serial_timing", "stage_total_ms"], 80);
        assert_json_bool(&value, &["serial_timing", "sum_matches_stage_total"], true);
        assert_json_u64(&value, &["gpu_subtree", "cpu_result_scatter_ms"], 20);
        assert_json_u64(&value, &["gpu_subtree", "async_result_scatter_wait_ms"], 4);
        assert_json_bool(
            &value,
            &["gpu_subtree", "persistent_async_d2h_enabled"],
            true,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_device_output_slots"],
            2,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_pinned_host_slot_bytes"],
            2048,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_session_init_ms"], 2);
        assert_json_u64(&value, &["gpu_subtree", "persistent_d2h_header_copy_ms"], 3);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_d2h_payload_event_ms"],
            12,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_d2h_bytes"], 8192);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_d2h_hidden_estimate_ms"],
            12,
        );
    }

    #[test]
    fn oom_serial_timing_clips_d1plus_rows_to_producer_wall() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.partitioner = PiPNNPartitionerKind::GpuSubtree;
        profile.streaming_wall = Duration::from_millis(100);
        profile.total_wall = Duration::from_millis(100);
        profile.rbc.d0_ads_assignment_wall = Duration::from_millis(10);
        profile.rbc.gpu_subtree_d1plus_producer_wall = Duration::from_millis(90);
        profile.rbc.gpu_subtree_d1plus_backend_wall = Duration::from_millis(70);
        profile.rbc.gpu_subtree_gpu_wait_for_sidecar_wall = Duration::from_millis(10);
        profile.rbc.gpu_subtree_sidecar_gather_hidden_wall = Duration::from_millis(1);
        profile.rbc.gpu_subtree_sq_wmma_h2d_wall = Duration::from_millis(10);
        profile.rbc.gpu_subtree_persistent_total_wall = Duration::from_millis(40);
        profile.rbc.gpu_subtree_cpu_result_scatter_wall = Duration::from_millis(10);
        profile.rbc.gpu_subtree_d1plus_child_replay_wall = Duration::from_millis(20);
        profile.rbc.gpu_subtree_d1plus_lazy_split_wall = Duration::from_millis(20);
        profile.rbc.gpu_subtree_d1plus_push_wait_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_d1plus_validate_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_d1plus_push_batch_wall = Duration::from_millis(5);

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_child_replay_ms"],
            10,
        );
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_lazy_split_ms"],
            0,
        );
        assert_json_u64(&value, &["serial_timing", "rows", "d1plus_push_wait_ms"], 0);
        assert_json_u64(&value, &["serial_timing", "rows", "d1plus_validate_ms"], 0);
        assert_json_u64(
            &value,
            &["serial_timing", "rows", "d1plus_push_batch_residual_ms"],
            0,
        );
        assert_json_u64(&value, &["serial_timing", "accounted_sum_ms"], 100);
        assert_json_u64(&value, &["serial_timing", "stage_total_ms"], 100);
        assert_json_bool(&value, &["serial_timing", "sum_matches_stage_total"], true);
    }

    #[test]
    fn oom_profile_json_preserves_gpu_subtree_partitioner_kind() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.partitioner = PiPNNPartitionerKind::GpuSubtree;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_eq!(
            value
                .get("partitioner")
                .and_then(|partitioner| partitioner.get("kind"))
                .and_then(Value::as_str),
            Some("gpu-subtree")
        );
    }

    #[test]
    fn profile_json_reports_auto_d96_k2_leaf_as_row_owned_mode() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.dim = 96;
        profile.leaf_knn = 2;
        profile.sq_bits = 8;
        profile.sq_native_int4 = false;
        profile.sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        profile.persistent_sq8_leaf_mode_requested = PiPNNGpuSubtreeSq8LeafMode::Auto;
        profile.rbc.gpu_subtree_persistent_leaf_row_owned_tasks = 1;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(&value, &["leaf_knn"], 2);
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_mode_requested"],
            "auto",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_mode_actual"],
            "row-owned",
        );
    }

    #[test]
    fn profile_json_reports_256d_auto_policy_as_batch8_wmma_row_owned() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.dim = 256;
        profile.leaf_knn = 2;
        profile.sq_bits = 8;
        profile.sq_native_int4 = false;
        profile.sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        profile.persistent_large_ticket_batch_requested =
            PiPNNGpuSubtreePersistentLargeTicketBatch::Auto;
        profile.persistent_sq8_assignment_kernel_requested =
            PiPNNGpuSubtreeSq8AssignmentKernel::Auto;
        profile.persistent_sq8_leaf_mode_requested = PiPNNGpuSubtreeSq8LeafMode::Auto;
        profile.rbc.gpu_subtree_persistent_large_ticket_batch_actual = 8;
        profile.rbc.gpu_subtree_persistent_large_sq8_wmma_tiles = 1;
        profile.rbc.gpu_subtree_persistent_leaf_row_owned_tasks = 1;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_ticket_batch_actual"],
            8,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_assignment_kernel_actual"],
            "wmma",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_mode_actual"],
            "row-owned",
        );
    }

    #[test]
    fn profile_json_reports_sq8_leaf_executor_and_d0_hashes() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.persistent_sq8_leaf_executor_requested = PiPNNGpuSubtreeSq8LeafExecutor::Wmma;
        profile.rbc.gpu_subtree_persistent_leaf_sq8_dp4a_tiles = 3;
        profile.rbc.gpu_subtree_persistent_leaf_sq8_wmma_tiles = 5;
        profile.rbc.gpu_subtree_d0_partition_hash = 0x0123_4567_89ab_cdef;
        profile.rbc.gpu_subtree_d0_task_list_hash = 0xfedc_ba98_7654_3210;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(&profile_path).unwrap()).unwrap();

        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_executor_requested"],
            "wmma",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_executor_actual"],
            "mixed",
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_leaf_sq8_dp4a_tiles"],
            3,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_leaf_sq8_wmma_tiles"],
            5,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_partition_hash"],
            0x0123_4567_89ab_cdef,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_task_list_hash"],
            0xfedc_ba98_7654_3210,
        );
    }

    #[test]
    fn profile_json_uses_launch_reported_large_ticket_batch() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.dim = 96;
        profile.sq_bits = 8;
        profile.sq_native_int4 = false;
        profile.sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        profile.persistent_large_ticket_batch_requested =
            PiPNNGpuSubtreePersistentLargeTicketBatch::Auto;
        profile.persistent_sq8_assignment_kernel_requested =
            PiPNNGpuSubtreeSq8AssignmentKernel::Auto;
        profile.rbc.gpu_subtree_persistent_large_ticket_batch_actual = 1;
        profile.rbc.gpu_subtree_persistent_large_sq8_dp4a_tiles = 1;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_ticket_batch_actual"],
            1,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_assignment_kernel_actual"],
            "warp-dp4a",
        );
    }

    #[test]
    fn two_stage_profile_reports_baseline_scheduler_actual() {
        assert_eq!(
            super::persistent_scheduler_mode_actual(
                PiPNNGpuSubtreeExecutionMode::TwoStage,
                PiPNNGpuSubtreePersistentSchedulerMode::AdaptiveIdle,
            ),
            PiPNNGpuSubtreePersistentSchedulerMode::Baseline,
        );
        assert_eq!(
            super::persistent_scheduler_mode_actual(
                PiPNNGpuSubtreeExecutionMode::FusedPersistent,
                PiPNNGpuSubtreePersistentSchedulerMode::LocalCounters,
            ),
            PiPNNGpuSubtreePersistentSchedulerMode::LocalCounters,
        );
    }

    #[test]
    fn profile_json_reports_high_dim_auto_leaf_as_staged_strategy() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.dim = 1024;
        profile.leaf_knn = 2;
        profile.sq_bits = 8;
        profile.sq_native_int4 = false;
        profile.sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        profile.persistent_large_ticket_batch_requested =
            PiPNNGpuSubtreePersistentLargeTicketBatch::Auto;
        profile.persistent_sq8_assignment_kernel_requested =
            PiPNNGpuSubtreeSq8AssignmentKernel::Auto;
        profile.persistent_sq8_leaf_mode_requested = PiPNNGpuSubtreeSq8LeafMode::Auto;
        profile.rbc.gpu_subtree_persistent_large_ticket_batch_actual = 8;
        profile.rbc.gpu_subtree_persistent_large_sq8_wmma_tiles = 1;
        profile.rbc.gpu_subtree_persistent_leaf_row_owned_tasks = 1;
        profile.rbc.gpu_subtree_persistent_leaf_staged_tasks = 1;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_mode_actual"],
            "staged",
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_ticket_batch_actual"],
            8,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_assignment_kernel_actual"],
            "wmma",
        );
    }

    #[test]
    fn profile_json_reports_sq8_d0_assignment_requested_and_actual() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.rbc.gpu_subtree_sq_d0_assign_kernel_requested = "auto";
        profile.rbc.gpu_subtree_sq_d0_assign_kernel_mode = "cuda-wmma-batch";
        profile.rbc.gpu_subtree_sq_d0_assign_row_tile = 8;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let value: Value =
            serde_json::from_str(&fs::read_to_string(profile_path).unwrap()).unwrap();
        assert_json_str(
            &value,
            &[
                "gpu_subtree",
                "sq_gpu_staging",
                "d0_assign_kernel_requested",
            ],
            "auto",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "sq_gpu_staging", "d0_assign_kernel_mode"],
            "cuda-wmma-batch",
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_gpu_staging", "d0_assign_row_tile"],
            8,
        );
    }

    #[test]
    fn gpu_subtree_profile_json_includes_gpu_subtree_profile_fields() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.partitioner = PiPNNPartitionerKind::GpuSubtree;
        profile.root_assignment_backend = PiPNNComputeBackendKind::Cuda;
        profile.assignment_backend = PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32;
        profile.leaf_topk_backend = PiPNNGpuSubtreeLeafTopKBackend::CublasGemm;
        profile.d0_sq_cpu_kernel = SqInt8CpuKernel::Avx512Vnni;
        profile.assignment_dco_mode = PiPNNGpuSubtreeDcoMode::PrefixProjection;
        profile.assignment_dco_rerank_mode = PiPNNGpuSubtreeDcoRerankMode::None;
        profile.assignment_dco_proj_dim = 32;
        profile.assignment_dco_survivors = 12;
        profile.assignment_dco_audit = true;
        profile.leaf_dco_mode = PiPNNGpuSubtreeDcoMode::WmmaFilter;
        profile.leaf_dco_rerank_mode = PiPNNGpuSubtreeDcoRerankMode::Exact;
        profile.leaf_dco_proj_dim = 64;
        profile.leaf_dco_survivors = 96;
        profile.leaf_dco_audit = true;
        profile.rabitq_plus_mode = PiPNNGpuSubtreeRabitqPlusMode::Shadow;
        profile.rabitq_plus_bits = 7;
        profile.rabitq_plus_rotator = PiPNNGpuSubtreeRabitqPlusRotator::None;
        profile.rabitq_plus_seed = 314;
        profile.rabitq_plus_sidecar_mode = PiPNNGpuSubtreeRabitqPlusSidecarMode::Require;
        profile.rabitq_plus_sidecar_prefix = Some(PathBuf::from("/tmp/wiki10m.rabitq7"));
        profile.rabitq_plus_fused_sidecar_sketch_enabled = true;
        profile.rabitq_plus_fused_prestage_total_wall = Duration::from_millis(168);
        profile.rabitq_plus_fused_chunk_alloc_wall = Duration::from_millis(68);
        profile.rabitq_plus_fused_sidecar_raw_read_wall = Duration::from_millis(69);
        profile.rabitq_plus_fused_source_hash_wall = Duration::from_millis(74);
        profile.rabitq_plus_fused_sidecar_quantize_wall = Duration::from_millis(70);
        profile.rabitq_plus_fused_sidecar_code_output_wall = Duration::from_millis(75);
        profile.rabitq_plus_fused_sidecar_norm_wall = Duration::from_millis(76);
        profile.rabitq_plus_fused_sidecar_meta_output_wall = Duration::from_millis(77);
        profile.rabitq_plus_fused_sidecar_header_write_wall = Duration::from_millis(78);
        profile.rabitq_plus_fused_sidecar_write_wall = Duration::from_millis(71);
        profile.rabitq_plus_fused_sketch_alloc_wall = Duration::from_millis(79);
        profile.rabitq_plus_fused_sketch_compute_wall = Duration::from_millis(72);
        profile.rabitq_plus_fused_sketch_extend_wall = Duration::from_millis(80);
        profile.rabitq_plus_fused_sketch_write_wall = Duration::from_millis(73);
        profile.sq_bits = 5;
        profile.sq_native_int4 = true;
        profile.sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        profile.persistent_large_ticket_batch_requested =
            PiPNNGpuSubtreePersistentLargeTicketBatch::Eight;
        profile.persistent_sq8_assignment_kernel_requested =
            PiPNNGpuSubtreeSq8AssignmentKernel::Wmma;
        profile.persistent_sq8_leaf_mode_requested = PiPNNGpuSubtreeSq8LeafMode::D96K2;
        profile.persistent_scheduler_mode_requested =
            PiPNNGpuSubtreePersistentSchedulerMode::AdaptiveIdle;
        profile.persistent_scheduler_mode_actual =
            PiPNNGpuSubtreePersistentSchedulerMode::AdaptiveIdle;
        profile.persistent_idle_backoff_ns_requested = 700;
        profile.persistent_idle_backoff_ns_actual = 700;
        profile.sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        profile.sq_wmma_sidecar_prefix = Some(PathBuf::from("/tmp/wiki10m.sq8"));
        profile.sq_staging_fused_sketch_enabled = true;
        profile.sq_staging_fused_sketch_compute_wall = Duration::from_millis(86);
        profile.sq_staging_fused_sketch_join_wait_wall = Duration::from_millis(87);
        profile.turboquant_mode = PiPNNGpuSubtreeTurboQuantMode::Shadow;
        profile.turboquant_cascade_low_dim = 256;
        profile.turboquant_cascade_survivor_cap = 64;
        profile.rbc.gpu_subtree_compact_attempt_count = 5;
        profile.rbc.gpu_subtree_compact_attempt_points = 50;
        profile.rbc.gpu_subtree_compact_success_count = 3;
        profile.rbc.gpu_subtree_compact_success_points = 30;
        profile.rbc.gpu_subtree_compact_unsupported_count = 2;
        profile.rbc.gpu_subtree_compact_unsupported_points = 20;
        profile.rbc.gpu_subtree_compact_leaf_count = 9;
        profile.rbc.gpu_subtree_compact_assignment_passes = 12;
        profile.rbc.gpu_subtree_compact_max_points = 11;
        profile.rbc.gpu_subtree_compact_batches = 9;
        profile.rbc.gpu_subtree_compact_rows = 30;
        profile.rbc.gpu_subtree_compact_h2d_bytes = 1024;
        profile.rbc.gpu_subtree_compact_d2h_bytes = 256;
        profile.rbc.gpu_subtree_host_io_stats.bytes_read = 2048;
        profile.rbc.gpu_subtree_host_io_stats.range_calls = 2;
        profile.rbc.gpu_subtree_host_io_stats.range_rows_read = 40;
        profile.rbc.gpu_subtree_host_io_stats.point_calls = 1;
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_hits = 5;
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_misses = 15;
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_hit_bytes = 512;
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_miss_bytes = 1536;
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_inserts = 15;
        profile.rbc.gpu_subtree_host_io_stats.raw_cache_evictions = 3;
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_hits = 9;
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_misses = 11;
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_hit_bytes = 9216;
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_miss_bytes = 11264;
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_inserts = 20;
        profile.rbc.gpu_subtree_host_io_stats.vector_reuse_evictions = 4;
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_bypass_rows = 2;
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_lookup_wall = Duration::from_millis(24);
        profile
            .rbc
            .gpu_subtree_host_io_stats
            .vector_reuse_insert_wall = Duration::from_millis(25);
        profile.raw_vector_cache_budget_bytes = 4096;
        profile.raw_vector_cache_rows = 2;
        profile.rbc.gpu_subtree_resident_h2d_bytes = 768;
        profile.rbc.gpu_subtree_resident_topk_metadata_h2d_bytes = 256;
        profile.rbc.gpu_subtree_host_normalize_wall = Duration::from_millis(1);
        profile.rbc.gpu_subtree_host_load_vectors_wall = Duration::from_millis(2);
        profile.rbc.gpu_subtree_host_norms_wall = Duration::from_millis(3);
        profile.rbc.gpu_subtree_vector_h2d_wall = Duration::from_millis(4);
        profile.rbc.gpu_subtree_global_ids_h2d_wall = Duration::from_millis(5);
        profile.rbc.gpu_subtree_norms_h2d_wall = Duration::from_millis(6);
        profile.rbc.gpu_subtree_assignment_active_ids_h2d_wall = Duration::from_millis(7);
        profile.rbc.gpu_subtree_assignment_leader_ids_h2d_wall = Duration::from_millis(8);
        profile.rbc.gpu_subtree_assignment_kernel_event_wall = Duration::from_millis(9);
        profile.rbc.gpu_subtree_assignment_d2h_submit_wall = Duration::from_millis(10);
        profile.rbc.gpu_subtree_assignment_sync_wait_wall = Duration::from_millis(11);
        profile.rbc.gpu_subtree_cpu_cluster_merge_wall = Duration::from_millis(12);
        profile.rbc.gpu_subtree_cpu_leaf_global_to_local_wall = Duration::from_millis(13);
        profile.rbc.gpu_subtree_resident_topk_metadata_h2d_wall = Duration::from_millis(14);
        profile.rbc.gpu_subtree_resident_topk_kernel_event_wall = Duration::from_millis(15);
        profile.rbc.gpu_subtree_resident_topk_d2h_submit_wall = Duration::from_millis(16);
        profile.rbc.gpu_subtree_resident_topk_sync_wait_wall = Duration::from_millis(17);
        profile.rbc.gpu_subtree_cpu_result_scatter_wall = Duration::from_millis(18);
        profile.rbc.gpu_subtree_partition_only_kernel_event_wall = Duration::from_millis(61);
        profile.rbc.gpu_subtree_partition_only_total_wall = Duration::from_millis(62);
        profile.rbc.gpu_subtree_library_topk_total_wall = Duration::from_millis(63);
        profile.rbc.gpu_subtree_library_topk_gemm_wall = Duration::from_millis(64);
        profile.rbc.gpu_subtree_library_topk_select_wall = Duration::from_millis(65);
        profile.rbc.gpu_subtree_library_topk_h2d_wall = Duration::from_millis(66);
        profile.rbc.gpu_subtree_library_topk_d2h_wall = Duration::from_millis(67);
        profile.rbc.gpu_subtree_library_topk_workspace_peak_bytes = 16_384;
        profile.rbc.gpu_subtree_library_topk_workspace_budget_bytes = 32_768;
        profile.rbc.gpu_subtree_library_topk_leaf_count = 7;
        profile.rbc.gpu_subtree_library_topk_rows = 30;
        profile.rbc.gpu_subtree_library_topk_batches = 7;
        profile.rbc.gpu_subtree_library_topk_cpu_scatter_wall = Duration::from_millis(68);
        profile.rbc.gpu_subtree_vector_upload_wall = Duration::from_millis(15);
        profile.rbc.gpu_subtree_assignment_kernel_wall = Duration::from_millis(9);
        profile.rbc.gpu_subtree_assignment_d2h_wall = Duration::from_millis(21);
        profile.rbc.gpu_subtree_topk_kernel_wall = Duration::from_millis(15);
        profile.rbc.gpu_subtree_topk_d2h_wall = Duration::from_millis(33);
        profile.rbc.gpu_subtree_host_control_wall = Duration::from_millis(49);
        profile.rbc.gpu_subtree_compact_total_wall = Duration::from_millis(19);
        profile.rbc.gpu_subtree_cpu_compact_flush_wall = Duration::from_millis(20);
        profile.rbc.gpu_subtree_async_flush_wall = Duration::from_millis(40);
        profile.rbc.gpu_subtree_async_flush_produce_wall = Duration::from_millis(41);
        profile.rbc.gpu_subtree_async_flush_join_wait_wall = Duration::from_millis(42);
        profile.rbc.gpu_subtree_async_flush_worker_wall_sum = Duration::from_millis(43);
        profile.rbc.gpu_subtree_async_flush_worker_count = 12;
        profile.rbc.gpu_subtree_async_flush_worker_mode = "adaptive".to_string();
        profile.rbc.gpu_subtree_async_flush_initial_workers = 8;
        profile.rbc.gpu_subtree_async_flush_max_workers = 64;
        profile.rbc.gpu_subtree_async_flush_peak_workers = 12;
        profile.rbc.gpu_subtree_async_flush_scale_up_events = 2;
        profile
            .rbc
            .gpu_subtree_async_flush_queue_budget_initial_bytes = 268_435_456;
        profile.rbc.gpu_subtree_async_flush_queue_budget_peak_bytes = 402_653_184;
        profile.rbc.gpu_subtree_async_flush_controller_windows = 7;
        profile
            .rbc
            .gpu_subtree_async_flush_controller_required_workers_peak = 11;
        profile.rbc.gpu_subtree_async_result_scatter_requested = "1".to_string();
        profile.rbc.gpu_subtree_async_result_scatter_enabled = true;
        profile.rbc.gpu_subtree_async_result_scatter_submissions = 6;
        profile.rbc.gpu_subtree_async_result_scatter_completions = 6;
        profile.rbc.gpu_subtree_async_result_scatter_failures = 1;
        profile.rbc.gpu_subtree_async_result_scatter_worker_wall = Duration::from_millis(44);
        profile.rbc.gpu_subtree_async_result_scatter_wait_wall = Duration::from_millis(45);
        profile.rbc.gpu_subtree_async_result_scatter_join_wait_wall = Duration::from_millis(46);
        profile.rbc.gpu_subtree_async_result_scatter_max_inflight = 1;
        profile.rbc.gpu_subtree_async_result_scatter_peak_raw_bytes = 4096;
        profile
            .rbc
            .gpu_subtree_async_result_scatter_peak_context_bytes = 1024;
        profile.rbc.gpu_subtree_async_result_scatter_peak_host_bytes = 5120;
        profile.rbc.gpu_subtree_post_last_gpu_cpu_tail_wall = Duration::from_millis(47);
        profile.rbc.gpu_subtree_post_last_gpu_cpu_tail_observed = true;
        profile.rbc.gpu_subtree_async_flush_chunks = 5;
        profile.rbc.gpu_subtree_workspace_peak_bytes = 4096;
        profile.rbc.gpu_subtree_workspace_budget_bytes = 8192;
        profile.rbc.gpu_subtree_cpu_fallbacks = 1;
        profile.rbc.gpu_subtree_sidecar_gather_active_wall = Duration::from_millis(88);
        profile.rbc.gpu_subtree_sidecar_gather_hidden_wall = Duration::from_millis(66);
        profile.rbc.gpu_subtree_gpu_wait_for_sidecar_wall = Duration::from_millis(22);
        profile.rbc.gpu_subtree_sidecar_queue_max_bytes = 123_456;
        profile.rbc.gpu_subtree_sidecar_preload_children = 3;
        profile.rbc.gpu_subtree_sidecar_preload_workers = 2;
        profile.rbc.gpu_subtree_sidecar_queue_max_children = 4;
        profile.rbc.gpu_subtree_d1plus_admission_split_count = 2;
        profile.rbc.gpu_subtree_d1plus_admission_split_rows = 1_000;
        profile.rbc.gpu_subtree_d1plus_admission_split_max_before = 700;
        profile.rbc.gpu_subtree_d1plus_admission_split_max_after = 350;
        profile.rbc.gpu_subtree_d1plus_prevented_cpu_fallback_count = 2;
        profile.rbc.gpu_subtree_rabitq_plus_rows = 123;
        profile.rbc.gpu_subtree_rabitq_plus_padded_dim = 1024;
        profile.rbc.gpu_subtree_rabitq_plus_code_bytes = 125_952;
        profile.rbc.gpu_subtree_rabitq_plus_quantize_wall = Duration::from_millis(73);
        profile.rbc.gpu_subtree_rabitq_plus_read_wall = Duration::from_millis(74);
        profile.rbc.gpu_subtree_rabitq_plus_h2d_wall = Duration::from_millis(75);
        profile.rbc.gpu_subtree_rabitq_plus_kernel_event_wall = Duration::from_millis(76);
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_rows = 123;
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_codes_bytes = 125_952;
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_meta_bytes = 1_476;
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_gather_wall = Duration::from_millis(77);
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_resident = true;
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_preload_wall = Duration::from_millis(78);
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_resident_bytes = 127_428;
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_resident_rows = 91;
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_file_rows = 32;
        profile
            .rbc
            .gpu_subtree_rabitq_plus_sidecar_resident_gather_wall = Duration::from_millis(79);
        profile.rbc.gpu_subtree_rabitq_plus_sidecar_file_gather_wall = Duration::from_millis(80);
        profile.rbc.gpu_subtree_sq_wmma_rows = 321;
        profile.rbc.gpu_subtree_sq_wmma_padded_dim = 1024;
        profile.rbc.gpu_subtree_sq_wmma_code_bytes = 328_704;
        profile.rbc.gpu_subtree_sq_wmma_quantize_wall = Duration::from_millis(81);
        profile.rbc.gpu_subtree_sq_wmma_h2d_wall = Duration::from_millis(82);
        profile.rbc.gpu_subtree_sq_wmma_kernel_event_wall = Duration::from_millis(83);
        profile.rbc.gpu_subtree_sq_wmma_sidecar_rows = 321;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_codes_bytes = 328_704;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_meta_bytes = 2_568;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_gather_wall = Duration::from_millis(84);
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident = true;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_preload_wall = Duration::from_millis(85);
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_bytes = 10_567_680;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_rows = 300;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_file_rows = 21;
        profile.rbc.gpu_subtree_sq_wmma_sidecar_resident_gather_wall = Duration::from_millis(86);
        profile.rbc.gpu_subtree_sq_wmma_sidecar_file_gather_wall = Duration::from_millis(87);
        profile
            .rbc
            .gpu_subtree_raw_resident_h2d_bytes_saved_estimate = 503_808;
        profile.rbc.gpu_subtree_rabitq_plus_shadow_mismatch_rows = 6;
        profile.rbc.gpu_subtree_rabitq_plus_shadow_audit_rows = 16;
        profile.rbc.gpu_subtree_rabitq_plus_shadow_audit_pairs = 64;
        profile.rbc.gpu_subtree_rabitq_plus_shadow_overlap_pairs = 56;
        profile
            .rbc
            .gpu_subtree_rabitq_plus_shadow_topk_overlap_x1000 = 875;
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_shadow_rows_sampled = 16;
        profile.rbc.gpu_subtree_turboquant_cascade_low_dim = 256;
        profile.rbc.gpu_subtree_turboquant_cascade_survivor_cap = 64;
        profile
            .rbc
            .gpu_subtree_turboquant_adaptive_survivor_threshold = 1024;
        profile.rbc.gpu_subtree_turboquant_adaptive_survivor_low_cap = 16;
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_no_rerank_overlap_pairs = 48;
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_overlap_pairs_at_k = 48;
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_survivor_overlap_pairs = 60;
        profile.rbc.gpu_subtree_turboquant_cascade_audit_pairs_at_k = 64;
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_survivor_audit_pairs = 64;
        profile.rbc.gpu_subtree_turboquant_cascade_mismatch_rows = 4;
        profile.rbc.gpu_subtree_turboquant_cascade_cpu_shadow_wall = Duration::from_millis(89);
        profile.rbc.gpu_subtree_turboquant_cascade_low_dim_pairs = 1024;
        profile
            .rbc
            .gpu_subtree_turboquant_cascade_estimated_full_pairs_saved = 768;
        profile.rbc.gpu_subtree_host_point_read_wall = Duration::from_millis(21);
        profile.rbc.gpu_subtree_host_vector_dedup_wall = Duration::from_millis(22);
        profile.rbc.gpu_subtree_host_vector_child_gather_wall = Duration::from_millis(23);
        profile.rbc.gpu_subtree_d1plus_child_rows_total = 120;
        profile.rbc.gpu_subtree_d1plus_unique_rows_estimate = 80;
        profile.rbc.gpu_subtree_d1plus_repeat_rows_estimate = 40;
        profile.rbc.gpu_subtree_d1plus_repeat_factor_x1000 = 1500;
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_prev1_hit_rows_estimate = 10;
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_window4_hit_rows_estimate = 20;
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_window8_hit_rows_estimate = 30;
        profile
            .rbc
            .gpu_subtree_d1plus_rolling_window16_hit_rows_estimate = 40;
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_prev1_hit_rows_estimate = 10;
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_window4_hit_rows_estimate = 20;
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_window8_hit_rows_estimate = 30;
        profile
            .rbc
            .gpu_subtree_d1plus_natural_rolling_window16_hit_rows_estimate = 40;
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_prev1_hit_rows_estimate = 11;
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_window4_hit_rows_estimate = 21;
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_window8_hit_rows_estimate = 31;
        profile
            .rbc
            .gpu_subtree_d1plus_scheduled_rolling_window16_hit_rows_estimate = 41;
        profile.rbc.gpu_subtree_child_schedule_mode = "overlap-greedy";
        profile.rbc.gpu_subtree_child_schedule_plan_wall = Duration::from_millis(26);
        profile.rbc.gpu_subtree_child_schedule_plan_bytes = 65_536;
        profile.rbc.gpu_subtree_child_schedule_reordered_children = 12;
        profile.rbc.gpu_subtree_child_schedule_overlap_edges = 7;
        profile.rbc.gpu_subtree_vector_reuse_enabled = true;
        profile.rbc.gpu_subtree_vector_reuse_budget_bytes = 4096;
        profile.rbc.gpu_subtree_vector_reuse_window_children = 3;
        profile.rbc.gpu_subtree_vector_staging_groups = 4;
        profile.rbc.gpu_subtree_vector_staging_materialized_children = 8;
        profile.rbc.gpu_subtree_vector_staging_fallback_groups = 1;
        profile.rbc.gpu_subtree_vector_staging_sum_child_rows = 120;
        profile.rbc.gpu_subtree_vector_staging_union_rows = 80;
        profile.rbc.gpu_subtree_vector_staging_duplicate_rows = 40;
        profile
            .rbc
            .gpu_subtree_vector_staging_saved_read_rows_estimate = 40;
        profile.rbc.gpu_subtree_vector_staging_peak_union_bytes = 4096;
        profile.rbc.gpu_subtree_persistent_attempt_count = 3;
        profile.rbc.gpu_subtree_persistent_success_count = 2;
        profile.rbc.gpu_subtree_persistent_fallback_count = 1;
        profile.rbc.gpu_subtree_persistent_partition_worker_blocks = 5;
        profile.rbc.gpu_subtree_persistent_topk_worker_blocks = 4;
        profile.rbc.gpu_subtree_persistent_partition_pops = 34;
        profile.rbc.gpu_subtree_persistent_leaf_pushes = 38;
        profile.rbc.gpu_subtree_persistent_topk_pops = 38;
        profile.rbc.gpu_subtree_persistent_partition_claims = 35;
        profile.rbc.gpu_subtree_persistent_partition_empty_polls = 36;
        profile.rbc.gpu_subtree_persistent_partition_frontier_end = 37;
        profile.rbc.gpu_subtree_persistent_partition_level_rounds = 4;
        profile.rbc.gpu_subtree_persistent_topk_claims = 39;
        profile.rbc.gpu_subtree_persistent_topk_empty_polls = 40;
        profile.rbc.gpu_subtree_persistent_done_cas_wins = 1;
        profile.rbc.gpu_subtree_persistent_active_wait_cycles = 48;
        profile.rbc.gpu_subtree_persistent_partition_busy_clocks = 101;
        profile.rbc.gpu_subtree_persistent_partition_wait_clocks = 102;
        profile.rbc.gpu_subtree_persistent_topk_busy_clocks = 103;
        profile.rbc.gpu_subtree_persistent_topk_wait_clocks = 104;
        profile.rbc.gpu_subtree_persistent_large_help_clocks = 105;
        profile.rbc.gpu_subtree_persistent_topk_help_clocks = 106;
        profile
            .rbc
            .gpu_subtree_persistent_partition_helped_topk_tiles = 9;
        profile.rbc.gpu_subtree_persistent_topk_helped_large_tiles = 14;
        profile.rbc.gpu_subtree_persistent_elastic_role_switches = 18;
        profile.rbc.gpu_subtree_persistent_queue_overflow_flags = 0b1111;
        profile.rbc.gpu_subtree_persistent_kernel_prepare_wall = Duration::from_millis(57);
        profile.rbc.gpu_subtree_persistent_kernel_event_wall = Duration::from_millis(54);
        profile.rbc.gpu_subtree_persistent_total_wall = Duration::from_millis(60);
        profile.rbc.gpu_subtree_persistent_topk_rows = 66;
        profile.rbc.gpu_subtree_persistent_leaf_count = 76;
        profile.rbc.gpu_subtree_persistent_effective_root_fanout = 3;
        profile.rbc.gpu_subtree_persistent_effective_deeper_fanout = 1;
        profile.rbc.gpu_subtree_persistent_fanout_assignment_slots = 90;
        profile.rbc.gpu_subtree_persistent_replicated_rows = 66;
        profile.rbc.gpu_subtree_persistent_unique_resident_rows = 44;
        profile
            .rbc
            .gpu_subtree_persistent_leaf_row_amplification_x1000 = 1500;
        profile.rbc.gpu_subtree_persistent_snapshot_active_large = 1;
        profile.rbc.gpu_subtree_persistent_snapshot_large_generation = 9;
        profile.rbc.gpu_subtree_persistent_snapshot_large_phase = 2;
        profile
            .rbc
            .gpu_subtree_persistent_snapshot_large_total_tiles = 1_123_089;
        profile.rbc.gpu_subtree_persistent_snapshot_large_count_next = 1_123_089;
        profile.rbc.gpu_subtree_persistent_snapshot_large_count_done = 1_123_089;
        profile
            .rbc
            .gpu_subtree_persistent_snapshot_large_scatter_next = 1_123_089;
        profile
            .rbc
            .gpu_subtree_persistent_snapshot_large_scatter_done = 1_123_089;
        profile.rbc.gpu_subtree_persistent_large_node_claims = 41;
        profile.rbc.gpu_subtree_persistent_large_node_splits = 42;
        profile.rbc.gpu_subtree_persistent_large_split_retries = 43;
        profile.rbc.gpu_subtree_persistent_large_balanced_fallbacks = 44;
        profile.rbc.gpu_subtree_persistent_large_node_rows = 43;
        profile.rbc.gpu_subtree_persistent_large_node_max_rows = 8192;
        profile.rbc.gpu_subtree_persistent_large_node_empty_polls = 44;
        profile
            .rbc
            .gpu_subtree_persistent_large_node_serialized_wait_cycles = 45;
        profile
            .rbc
            .gpu_subtree_persistent_large_count_distance_clocks = 201;
        profile.rbc.gpu_subtree_persistent_large_count_commit_clocks = 202;
        profile.rbc.gpu_subtree_persistent_large_scatter_clocks = 203;
        profile.rbc.gpu_subtree_persistent_large_publish_clocks = 204;
        profile.rbc.gpu_subtree_persistent_large_ticket_batch_actual = 8;
        profile.rbc.gpu_subtree_persistent_large_ticket_range_claims = 205;
        profile.rbc.gpu_subtree_persistent_large_physical_tiles = 206;
        profile.rbc.gpu_subtree_persistent_large_sq8_dp4a_tiles = 207;
        profile.rbc.gpu_subtree_persistent_large_sq8_wmma_tiles = 0;
        profile.rbc.gpu_subtree_persistent_leaf_row_owned_tasks = 209;
        profile.rbc.gpu_subtree_persistent_leaf_staged_tasks = 0;
        profile.rbc.gpu_subtree_persistent_leaf_score_clocks = 211;
        profile
            .rbc
            .gpu_subtree_persistent_leaf_row_lock_merge_clocks = 212;
        profile
            .rbc
            .gpu_subtree_persistent_topk_local_counter_flushes = 213;
        profile.rbc.gpu_subtree_persistent_topk_idle_backoff_calls = 214;
        profile
            .rbc
            .gpu_subtree_persistent_topk_idle_backoff_requested_ns = 74_900;
        profile.rbc.gpu_subtree_persistent_topk_idle_leaf_resets = 216;
        profile.rbc.gpu_subtree_persistent_topk_idle_large_resets = 217;
        profile.rbc.gpu_subtree_persistent_leaf_size_min = 128;
        profile.rbc.gpu_subtree_persistent_leaf_size_max = 1024;
        profile.rbc.gpu_subtree_persistent_leaf_size_sum = 4096;
        profile.rbc.gpu_subtree_persistent_leaf_size_count = 6;
        profile.rbc.gpu_subtree_persistent_leaf_target = 512;
        profile.rbc.gpu_subtree_persistent_leaf_min = 128;
        profile.rbc.gpu_subtree_persistent_leaf_max = 1024;
        profile.rbc.gpu_subtree_persistent_leaf_tile_rows = 256;
        profile.rbc.gpu_subtree_assignment_dco.nodes = 7;
        profile.rbc.gpu_subtree_assignment_dco.rows = 700;
        profile.rbc.gpu_subtree_assignment_dco.leader_pairs = 7000;
        profile.rbc.gpu_subtree_assignment_dco.prefilter_pairs = 3000;
        profile.rbc.gpu_subtree_assignment_dco.pruned_pairs = 2000;
        profile.rbc.gpu_subtree_assignment_dco.survivor_pairs = 1000;
        profile.rbc.gpu_subtree_assignment_dco.exact_pairs = 0;
        profile.rbc.gpu_subtree_assignment_dco.approx_output_pairs = 1000;
        profile.rbc.gpu_subtree_assignment_dco.fallback_nodes = 1;
        profile.rbc.gpu_subtree_assignment_dco.overflow_nodes = 2;
        profile
            .rbc
            .gpu_subtree_assignment_dco
            .false_prune_audit_pairs = 128;
        profile
            .rbc
            .gpu_subtree_assignment_dco
            .false_prune_audit_misses = 3;
        profile.rbc.gpu_subtree_assignment_dco.metadata_bytes = 4096;
        profile.rbc.gpu_subtree_assignment_dco.metadata_h2d_wall = Duration::from_millis(70);
        profile.rbc.gpu_subtree_leaf_dco.leaf_count = 5;
        profile.rbc.gpu_subtree_leaf_dco.rows = 500;
        profile.rbc.gpu_subtree_leaf_dco.pair_count = 50_000;
        profile.rbc.gpu_subtree_leaf_dco.prefilter_pairs = 20_000;
        profile.rbc.gpu_subtree_leaf_dco.pruned_pairs = 15_000;
        profile.rbc.gpu_subtree_leaf_dco.survivor_pairs = 5_000;
        profile.rbc.gpu_subtree_leaf_dco.exact_pairs = 5_000;
        profile.rbc.gpu_subtree_leaf_dco.exact_stamp_pairs = 1_000;
        profile.rbc.gpu_subtree_leaf_dco.approx_output_pairs = 0;
        profile.rbc.gpu_subtree_leaf_dco.tile_prefilter_tiles = 11;
        profile.rbc.gpu_subtree_leaf_dco.tile_pruned_tiles = 12;
        profile.rbc.gpu_subtree_leaf_dco.tile_partial_tiles = 13;
        profile.rbc.gpu_subtree_leaf_dco.tile_full_exact_tiles = 14;
        profile.rbc.gpu_subtree_leaf_dco.exact_fallback_rows = 15;
        profile.rbc.gpu_subtree_leaf_dco.survivor_overflow_rows = 16;
        profile.rbc.gpu_subtree_leaf_dco.audit_only_pairs = 17;
        profile.rbc.gpu_subtree_leaf_dco.false_prune_audit_pairs = 256;
        profile.rbc.gpu_subtree_leaf_dco.false_prune_audit_misses = 4;
        profile.rbc.gpu_subtree_leaf_dco.metadata_bytes = 8192;
        profile.rbc.gpu_subtree_leaf_dco.metadata_prepare_wall = Duration::from_millis(72);
        profile.rbc.gpu_subtree_leaf_dco.metadata_h2d_wall = Duration::from_millis(71);
        profile.rbc.gpu_subtree_leaf_dco.prefilter_clock_ticks = 123_456;
        profile.rbc.gpu_subtree_leaf_dco.exact_rerank_clock_ticks = 23_456;
        profile.rbc.gpu_subtree_leaf_dco.audit_clock_ticks = 3_456;
        profile.rbc.gpu_subtree_leaf_dco.tile_started = 18;
        profile.rbc.gpu_subtree_leaf_dco.tile_done = 17;
        profile.rbc.gpu_subtree_leaf_dco.prefilter_row_started = 500;
        profile.rbc.gpu_subtree_leaf_dco.prefilter_row_done = 499;
        profile.rbc.gpu_subtree_leaf_dco.exact_rerank_row_started = 498;
        profile.rbc.gpu_subtree_leaf_dco.exact_rerank_row_done = 497;
        profile.rbc.gpu_subtree_leaf_dco.audit_row_started = 496;
        profile.rbc.gpu_subtree_leaf_dco.audit_row_done = 495;
        profile.rbc.gpu_subtree_leaf_dco.progress_heartbeat = 2_000;
        profile.rbc.gpu_subtree_leaf_dco.staged_tiles = 19;
        profile.rbc.gpu_subtree_leaf_dco.col_tasks_started = 20;
        profile.rbc.gpu_subtree_leaf_dco.col_tasks_done = 20;
        profile.rbc.gpu_subtree_leaf_dco.finalize_started = 21;
        profile.rbc.gpu_subtree_leaf_dco.finalize_done = 21;
        profile.rbc.gpu_subtree_leaf_dco.active_slots_max = 4;
        profile.rbc.gpu_subtree_leaf_dco.slot_wait_polls = 3;
        profile.rbc.gpu_subtree_leaf_dco.row_lock_spins = 2;
        profile.rbc.gpu_subtree_leaf_dco.staged_rows = 256;
        profile.rbc.d0_capacity.enabled = true;
        profile.rbc.d0_capacity.c_gpu = 400_000;
        profile.rbc.d0_capacity.c_high = 340_000;
        profile.rbc.d0_capacity.c_target = 300_000;
        profile.rbc.d0_capacity.c_low = 64_000;
        profile.rbc.d0_capacity.k_capacity = 200;
        profile.rbc.d0_capacity.quality_min_root_leaders = 64;
        profile.rbc.d0_capacity.root_leader_cap = 10_000;
        profile.rbc.d0_capacity.root_leader_source = "configured";
        profile.rbc.d0_capacity.root_leaders_initial = 64;
        profile.rbc.d0_capacity.root_leaders_final = 64;
        profile.rbc.d0_capacity.actual_child_max_unique_rows = 320_000;
        profile.rbc.d0_capacity.actual_child_p50_unique_rows = 180_000;
        profile.rbc.d0_capacity.actual_child_p95_unique_rows = 290_000;
        profile.rbc.d0_capacity.actual_child_p99_unique_rows = 320_000;
        profile.rbc.d0_capacity.oversized_child_count = 1;
        profile.rbc.d0_capacity.small_child_count = 2;
        profile.reduce.logical_read_bytes = 32;
        profile.reduce.reader_open_calls = 1;
        profile.reduce.direct_read_calls = 1;
        profile.reduce.direct_read_bytes = 4096;
        profile.reduce.direct_io = true;
        profile.reduce.workspace_points = 128;
        profile.reduce.workspace_bytes = 16_384;
        profile.reduce.workspace_alloc_wall = Duration::from_millis(11);
        profile.reduce.read_decode_wall = Duration::from_millis(12);
        profile.reduce.record_read_wall = Duration::from_millis(3);
        profile.reduce.record_decode_wall = Duration::from_millis(4);
        profile.reduce.record_append_wall = Duration::from_millis(2);
        profile.reduce.insert_wall = Duration::from_millis(13);
        profile.reduce.pipeline_wall = Duration::from_millis(15);
        profile.reduce.pipeline_overlap_wall = Duration::from_millis(6);
        profile.reduce.finalize_emit_wall = Duration::from_millis(14);
        profile.reduce.decoder_workers = 2;
        profile.reduce.parallel_workers = 4;
        profile.reduce.parallel_batches = 7;
        profile.reduce.parallel_records = 1234;
        profile.reduce.batch_peak_records = 256;
        profile.reduce.batch_peak_bytes = 4096;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(&value, &["raw_vector_cache", "budget_bytes"], 4096);
        assert_json_u64(&value, &["raw_vector_cache", "row_capacity"], 2);
        assert_json_u64(&value, &["raw_vector_cache", "hits"], 5);
        assert_json_u64(&value, &["raw_vector_cache", "misses"], 15);
        assert_json_u64(&value, &["raw_vector_cache", "hit_bytes"], 512);
        assert_json_u64(&value, &["raw_vector_cache", "miss_bytes"], 1536);
        assert_json_f64(&value, &["raw_vector_cache", "hit_ratio"], 0.25);
        assert_json_u64(&value, &["gpu_subtree_timing", "host_load_vectors_ms"], 2);
        assert_json_u64(&value, &["gpu_subtree_timing", "host_point_read_ms"], 21);
        assert_json_u64(&value, &["gpu_subtree_timing", "host_vector_dedup_ms"], 22);
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "host_vector_child_gather_ms"],
            23,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "child_schedule_plan_ms"],
            26,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "rabitq_plus_quantize_ms"],
            73,
        );
        assert_json_u64(&value, &["gpu_subtree_timing", "rabitq_plus_read_ms"], 74);
        assert_json_u64(&value, &["gpu_subtree_timing", "rabitq_plus_h2d_ms"], 75);
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "rabitq_plus_kernel_event_ms"],
            76,
        );
        assert_json_u64(&value, &["gpu_subtree_timing", "sq_wmma_quantize_ms"], 81);
        assert_json_u64(&value, &["gpu_subtree_timing", "sq_wmma_h2d_ms"], 82);
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "sq_wmma_kernel_event_ms"],
            83,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "persistent_kernel_prepare_ms"],
            57,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree_timing", "persistent_kernel_event_ms"],
            54,
        );
        assert_json_u64(&value, &["gpu_subtree_timing", "cpu_compact_flush_ms"], 20);
        assert_json_u64(&value, &["gpu_subtree_io", "host_point_read_bytes"], 2048);
        assert_json_u64(&value, &["gpu_subtree_io", "raw_cache_hits"], 5);
        assert_json_u64(&value, &["gpu_subtree_io", "raw_cache_misses"], 15);
        assert_json_u64(&value, &["gpu_subtree_io", "vector_reuse_hits"], 9);
        assert_json_u64(&value, &["gpu_subtree_io", "vector_reuse_misses"], 11);
        assert_json_u64(&value, &["gpu_subtree_io", "gpu_h2d_bytes"], 1024);
        assert_json_u64(&value, &["gpu_subtree_io", "gpu_d2h_bytes"], 256);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_logical_read_bytes"], 32);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_reader_open_calls"], 1);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_direct_read_calls"], 1);
        assert_json_u64(
            &value,
            &["gpu_subtree_io", "reduce_direct_read_bytes"],
            4096,
        );
        assert_eq!(
            value["gpu_subtree_io"]["reduce_direct_io"],
            serde_json::Value::Bool(true)
        );
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_workspace_points"], 128);
        assert_json_u64(
            &value,
            &["gpu_subtree_io", "reduce_workspace_bytes"],
            16_384,
        );
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_workspace_alloc_ms"], 11);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_read_decode_ms"], 12);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_record_read_ms"], 3);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_record_decode_ms"], 4);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_record_append_ms"], 2);
        assert_json_u64(
            &value,
            &["gpu_subtree_io", "reduce_read_decode_residual_ms"],
            3,
        );
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_insert_ms"], 13);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_pipeline_wall_ms"], 15);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_pipeline_overlap_ms"], 6);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_finalize_emit_ms"], 14);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_decoder_workers"], 2);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_parallel_workers"], 4);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_parallel_batches"], 7);
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_parallel_records"], 1234);
        assert_json_u64(
            &value,
            &["gpu_subtree_io", "reduce_batch_peak_records"],
            256,
        );
        assert_json_u64(&value, &["gpu_subtree_io", "reduce_batch_peak_bytes"], 4096);
        assert_json_u64(
            &value,
            &["gpu_subtree_counters", "persistent_attempt_count"],
            3,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree_counters", "persistent_success_count"],
            2,
        );
        assert_json_u64(&value, &["gpu_subtree_counters", "compact_rows"], 30);
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("root_assignment_backend"))
                .and_then(Value::as_str),
            Some("cuda")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("assignment_backend"))
                .and_then(Value::as_str),
            Some("persistent-wmma-tf32")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("subtree_assignment_backend"))
                .and_then(Value::as_str),
            Some("persistent-wmma-tf32")
        );
        assert_json_str(&value, &["gpu_subtree", "d0_sq_cpu_kernel"], "avx512-vnni");
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("leaf_topk_backend"))
                .and_then(Value::as_str),
            Some("cublas-gemm")
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "assignment_dco_mode"],
            "prefix-projection",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "assignment_dco_rerank_mode"],
            "none",
        );
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_proj_dim"], 32);
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_survivors"], 12);
        assert_json_bool(&value, &["gpu_subtree", "assignment_dco_audit"], true);
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_nodes"], 7);
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_rows"], 700);
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_leader_pairs"],
            7000,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_prefilter_pairs"],
            3000,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_pruned_pairs"],
            2000,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_survivor_pairs"],
            1000,
        );
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_exact_pairs"], 0);
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_approx_output_pairs"],
            1000,
        );
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_fallback_nodes"], 1);
        assert_json_u64(&value, &["gpu_subtree", "assignment_dco_overflow_nodes"], 2);
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_false_prune_audit_pairs"],
            128,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_false_prune_audit_misses"],
            3,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_metadata_bytes"],
            4096,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "assignment_dco_metadata_h2d_ms"],
            70,
        );
        assert_json_str(&value, &["gpu_subtree", "leaf_dco_mode"], "wmma-filter");
        assert_json_str(&value, &["gpu_subtree", "leaf_dco_rerank_mode"], "exact");
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_proj_dim"], 64);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_survivors"], 96);
        assert_json_bool(&value, &["gpu_subtree", "leaf_dco_audit"], true);
        assert_json_str(&value, &["gpu_subtree", "rabitq_plus_mode"], "shadow");
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_bits"], 7);
        assert_json_str(&value, &["gpu_subtree", "rabitq_plus_rotator"], "none");
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_seed"], 314);
        assert_json_str(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_mode"],
            "require",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_prefix"],
            "/tmp/wiki10m.rabitq7",
        );
        assert_json_bool(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_sketch_enabled"],
            true,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_prestage_total_ms"],
            168,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_chunk_alloc_ms"],
            68,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_raw_read_ms"],
            69,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_source_hash_ms"],
            74,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_quantize_ms"],
            70,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_code_output_ms"],
            75,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_norm_ms"],
            76,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_meta_output_ms"],
            77,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_header_write_ms"],
            78,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sidecar_write_ms"],
            71,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sketch_alloc_ms"],
            79,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sketch_compute_ms"],
            72,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sketch_extend_ms"],
            80,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_sketch_write_ms"],
            73,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_fused_unattributed_ms"],
            0,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_format"],
            "dense_i8_codes_meta",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "rabitq_plus_distance_formula"],
            "dense_i8_delta_code_sum_norm",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "rabitq_split_sidecar_format"],
            "library_split_batch_ex_meta",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "rabitq_split_distance_formula"],
            "rabitq_library_split_distance_boosting",
        );
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_rows"], 123);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_padded_dim"], 1024);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_code_bytes"], 125_952);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_quantize_ms"], 73);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_read_ms"], 74);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_h2d_ms"], 75);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_kernel_event_ms"], 76);
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_sidecar_rows"], 123);
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_codes_bytes"],
            125_952,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_meta_bytes"],
            1_476,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_gather_ms"],
            77,
        );
        assert_json_u64(&value, &["gpu_subtree", "rabitq_plus_sidecar_read_ms"], 77);
        assert_json_bool(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_resident"],
            true,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_preload_ms"],
            78,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_resident_bytes"],
            127_428,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_resident_rows"],
            91,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_file_rows"],
            32,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_resident_gather_ms"],
            79,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_sidecar_file_gather_ms"],
            80,
        );
        assert_json_str(&value, &["gpu_subtree", "sq_wmma_mode"], "build");
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_large_ticket_batch_requested"],
            "8",
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_ticket_batch_actual"],
            8,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_assignment_kernel_requested"],
            "wmma",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_assignment_kernel_actual"],
            "warp-dp4a",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_mode_requested"],
            "d96-k2",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_sq8_leaf_mode_actual"],
            "d96-k2",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_scheduler_mode_requested"],
            "adaptive-idle",
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "persistent_scheduler_mode_actual"],
            "adaptive-idle",
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_idle_backoff_ns_requested"],
            700,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_idle_backoff_ns_actual"],
            700,
        );
        assert_json_u64(&value, &["gpu_subtree", "sq_bits"], 5);
        assert_json_bool(&value, &["gpu_subtree", "sq_native_int4"], true);
        assert_json_str(&value, &["gpu_subtree", "sq_wmma_sidecar_mode"], "require");
        assert_json_str(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_prefix"],
            "/tmp/wiki10m.sq8",
        );
        assert_json_bool(
            &value,
            &["gpu_subtree", "sq_staging_fused_sketch_enabled"],
            true,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_staging_fused_sketch_compute_ms"],
            86,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_staging_fused_sketch_join_wait_ms"],
            87,
        );
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_rows"], 321);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_padded_dim"], 1024);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_code_bytes"], 328_704);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_quantize_ms"], 81);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_h2d_ms"], 82);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_kernel_event_ms"], 83);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_sidecar_rows"], 321);
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_codes_bytes"],
            328_704,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_meta_bytes"],
            2_568,
        );
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_sidecar_gather_ms"], 84);
        assert_json_bool(&value, &["gpu_subtree", "sq_wmma_sidecar_resident"], true);
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_sidecar_preload_ms"], 85);
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_resident_bytes"],
            10_567_680,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_resident_rows"],
            300,
        );
        assert_json_u64(&value, &["gpu_subtree", "sq_wmma_sidecar_file_rows"], 21);
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_resident_gather_ms"],
            86,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "sq_wmma_sidecar_file_gather_ms"],
            87,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "raw_resident_h2d_bytes_saved_estimate"],
            503_808,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_shadow_mismatch_rows"],
            6,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_shadow_audit_rows"],
            16,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_shadow_audit_pairs"],
            64,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_shadow_overlap_pairs"],
            56,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "rabitq_plus_shadow_topk_overlap_x1000"],
            875,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "turboquant_cascade", "low_dim"],
            256,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "turboquant_cascade", "survivor_cap"],
            64,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "turboquant_cascade",
                "adaptive_survivor_threshold",
            ],
            1024,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "turboquant_cascade",
                "adaptive_survivor_low_cap",
            ],
            16,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "turboquant_cascade", "shadow_rows_sampled"],
            16,
        );
        assert_json_f64(
            &value,
            &[
                "gpu_subtree",
                "turboquant_cascade",
                "no_rerank_overlap_at_k",
            ],
            0.75,
        );
        assert_json_f64(
            &value,
            &["gpu_subtree", "turboquant_cascade", "overlap_at_k"],
            0.75,
        );
        assert_json_f64(
            &value,
            &[
                "gpu_subtree",
                "turboquant_cascade",
                "overlap_at_survivor_cap",
            ],
            0.9375,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "turboquant_cascade", "mismatch_rows"],
            4,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "turboquant_cascade", "cpu_shadow_ms"],
            89,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "turboquant_cascade",
                "estimated_full_pairs_saved",
            ],
            768,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_leaf_count"], 5);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_rows"], 500);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_pair_count"], 50_000);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_prefilter_pairs"], 20_000);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_pruned_pairs"], 15_000);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_survivor_pairs"], 5_000);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_exact_pairs"], 5_000);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_exact_stamp_pairs"],
            1_000,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_approx_output_pairs"], 0);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_tile_prefilter_tiles"],
            11,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_tile_pruned_tiles"], 12);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_tile_partial_tiles"], 13);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_tile_full_exact_tiles"],
            14,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_exact_fallback_rows"], 15);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_survivor_overflow_rows"],
            16,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_audit_only_pairs"], 17);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_false_prune_audit_pairs"],
            256,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_false_prune_audit_misses"],
            4,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_metadata_bytes"], 8192);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_metadata_prepare_ms"], 72);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_metadata_h2d_ms"], 71);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_prefilter_clock_ticks"],
            123_456,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_exact_rerank_clock_ticks"],
            23_456,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_audit_clock_ticks"],
            3_456,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_tile_started"], 18);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_tile_done"], 17);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_prefilter_row_started"],
            500,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_prefilter_row_done"], 499);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_exact_rerank_row_started"],
            498,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_exact_rerank_row_done"],
            497,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_audit_row_started"], 496);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_audit_row_done"], 495);
        assert_json_u64(
            &value,
            &["gpu_subtree", "leaf_dco_progress_heartbeat"],
            2_000,
        );
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_staged_tiles"], 19);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_col_tasks_started"], 20);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_col_tasks_done"], 20);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_finalize_started"], 21);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_finalize_done"], 21);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_active_slots_max"], 4);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_slot_wait_polls"], 3);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_row_lock_spins"], 2);
        assert_json_u64(&value, &["gpu_subtree", "leaf_dco_staged_rows"], 256);
        assert_json_u64(&value, &["gpu_subtree", "child_count"], 5);
        assert_json_u64(&value, &["gpu_subtree", "child_points"], 50);
        assert_json_u64(&value, &["gpu_subtree", "assignment_passes"], 12);
        assert_json_u64(&value, &["gpu_subtree", "leaf_count"], 9);
        assert_json_u64(&value, &["gpu_subtree", "compact_topk_batches"], 9);
        assert_json_u64(&value, &["gpu_subtree", "compact_topk_rows"], 30);
        assert_json_u64(&value, &["gpu_subtree", "h2d_bytes"], 1024);
        assert_json_u64(&value, &["gpu_subtree", "d2h_bytes"], 256);
        assert_json_u64(&value, &["gpu_subtree", "resident_h2d_bytes"], 768);
        assert_json_u64(
            &value,
            &["gpu_subtree", "resident_topk_metadata_h2d_bytes"],
            256,
        );
        assert_json_u64(&value, &["gpu_subtree", "host_normalize_ms"], 1);
        assert_json_u64(&value, &["gpu_subtree", "host_load_vectors_ms"], 2);
        assert_json_u64(&value, &["gpu_subtree", "host_point_read_ms"], 21);
        assert_json_u64(&value, &["gpu_subtree", "host_vector_dedup_ms"], 22);
        assert_json_u64(&value, &["gpu_subtree", "host_vector_child_gather_ms"], 23);
        assert_json_u64(&value, &["gpu_subtree", "d1plus_child_rows_total"], 120);
        assert_json_u64(&value, &["gpu_subtree", "d1plus_unique_rows_estimate"], 80);
        assert_json_u64(&value, &["gpu_subtree", "d1plus_repeat_rows_estimate"], 40);
        assert_json_u64(&value, &["gpu_subtree", "d1plus_repeat_factor_x1000"], 1500);
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_rolling_prev1_hit_rows_estimate"],
            10,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_rolling_window4_hit_rows_estimate"],
            20,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_rolling_window8_hit_rows_estimate"],
            30,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_rolling_window16_hit_rows_estimate"],
            40,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_natural_rolling_prev1_hit_rows_estimate",
            ],
            10,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_natural_rolling_window4_hit_rows_estimate",
            ],
            20,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_natural_rolling_window8_hit_rows_estimate",
            ],
            30,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_natural_rolling_window16_hit_rows_estimate",
            ],
            40,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_scheduled_rolling_prev1_hit_rows_estimate",
            ],
            11,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_scheduled_rolling_window4_hit_rows_estimate",
            ],
            21,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_scheduled_rolling_window8_hit_rows_estimate",
            ],
            31,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d1plus_scheduled_rolling_window16_hit_rows_estimate",
            ],
            41,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "child_schedule_mode"],
            "overlap-greedy",
        );
        assert_json_u64(&value, &["gpu_subtree", "child_schedule_plan_ms"], 26);
        assert_json_u64(
            &value,
            &["gpu_subtree", "child_schedule_plan_bytes"],
            65_536,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "child_schedule_reordered_children"],
            12,
        );
        assert_json_u64(&value, &["gpu_subtree", "child_schedule_overlap_edges"], 7);
        assert_json_bool(&value, &["gpu_subtree", "vector_reuse_enabled"], true);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_budget_bytes"], 4096);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_window_children"], 3);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_hit_rows"], 9);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_miss_rows"], 11);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_hit_bytes"], 9216);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_miss_bytes"], 11264);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_insert_rows"], 20);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_evicted_rows"], 4);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_bypass_rows"], 2);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_lookup_ms"], 24);
        assert_json_u64(&value, &["gpu_subtree", "vector_reuse_insert_ms"], 25);
        assert_json_u64(&value, &["gpu_subtree", "vector_staging_groups"], 4);
        assert_json_u64(
            &value,
            &["gpu_subtree", "vector_staging_materialized_children"],
            8,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "vector_staging_fallback_groups"],
            1,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "vector_staging_sum_child_rows"],
            120,
        );
        assert_json_u64(&value, &["gpu_subtree", "vector_staging_union_rows"], 80);
        assert_json_u64(
            &value,
            &["gpu_subtree", "vector_staging_duplicate_rows"],
            40,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "vector_staging_saved_read_rows_estimate"],
            40,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "vector_staging_peak_union_bytes"],
            4096,
        );
        assert_json_u64(&value, &["gpu_subtree", "host_norms_ms"], 3);
        assert_json_u64(&value, &["gpu_subtree", "subtree_vector_h2d_ms"], 4);
        assert_json_u64(&value, &["gpu_subtree", "subtree_global_ids_h2d_ms"], 5);
        assert_json_u64(&value, &["gpu_subtree", "subtree_norms_h2d_ms"], 6);
        assert_json_u64(&value, &["gpu_subtree", "assignment_active_ids_h2d_ms"], 7);
        assert_json_u64(&value, &["gpu_subtree", "assignment_leader_ids_h2d_ms"], 8);
        assert_json_u64(&value, &["gpu_subtree", "assignment_kernel_event_ms"], 9);
        assert_json_u64(&value, &["gpu_subtree", "assignment_d2h_submit_ms"], 10);
        assert_json_u64(&value, &["gpu_subtree", "assignment_sync_wait_ms"], 11);
        assert_json_u64(&value, &["gpu_subtree", "cpu_cluster_merge_ms"], 12);
        assert_json_u64(&value, &["gpu_subtree", "cpu_leaf_global_to_local_ms"], 13);
        assert_json_u64(
            &value,
            &["gpu_subtree", "resident_topk_metadata_h2d_ms"],
            14,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "resident_topk_kernel_event_ms"],
            15,
        );
        assert_json_u64(&value, &["gpu_subtree", "resident_topk_d2h_submit_ms"], 16);
        assert_json_u64(&value, &["gpu_subtree", "resident_topk_sync_wait_ms"], 17);
        assert_json_u64(&value, &["gpu_subtree", "cpu_result_scatter_ms"], 18);
        assert_json_u64(
            &value,
            &["gpu_subtree", "partition_only_kernel_event_ms"],
            61,
        );
        assert_json_u64(&value, &["gpu_subtree", "partition_only_total_ms"], 62);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_total_ms"], 63);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_gemm_ms"], 64);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_select_ms"], 65);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_h2d_ms"], 66);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_d2h_ms"], 67);
        assert_json_u64(
            &value,
            &["gpu_subtree", "library_topk_workspace_peak_bytes"],
            16_384,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "library_topk_workspace_budget_bytes"],
            32_768,
        );
        assert_json_u64(&value, &["gpu_subtree", "library_topk_leaf_count"], 7);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_rows"], 30);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_batches"], 7);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_cpu_scatter_ms"], 68);
        assert_json_u64(&value, &["gpu_subtree", "vector_upload_ms"], 15);
        assert_json_u64(&value, &["gpu_subtree", "assignment_kernel_ms"], 9);
        assert_json_u64(&value, &["gpu_subtree", "assignment_d2h_ms"], 21);
        assert_json_u64(&value, &["gpu_subtree", "topk_kernel_ms"], 15);
        assert_json_u64(&value, &["gpu_subtree", "topk_d2h_ms"], 33);
        assert_json_u64(&value, &["gpu_subtree", "host_control_ms"], 49);
        assert_json_u64(&value, &["gpu_subtree", "total_ms"], 19);
        assert_json_u64(&value, &["gpu_subtree", "cpu_compact_flush_ms"], 20);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_ms"], 40);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_produce_ms"], 41);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_join_wait_ms"], 42);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_worker_ms_sum"], 43);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_worker_count"], 12);
        assert_json_str(
            &value,
            &["gpu_subtree", "async_flush_worker_mode"],
            "adaptive",
        );
        assert_json_u64(&value, &["gpu_subtree", "async_flush_initial_workers"], 8);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_max_workers"], 64);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_peak_workers"], 12);
        assert_json_u64(&value, &["gpu_subtree", "async_flush_scale_up_events"], 2);
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_flush_queue_budget_initial_bytes"],
            268_435_456,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_flush_queue_budget_peak_bytes"],
            402_653_184,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_flush_controller_windows"],
            7,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "async_flush_controller_required_workers_peak",
            ],
            11,
        );
        assert_json_str(
            &value,
            &["gpu_subtree", "async_result_scatter_requested"],
            "1",
        );
        assert_json_bool(
            &value,
            &["gpu_subtree", "async_result_scatter_enabled"],
            true,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_submissions"],
            6,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_completions"],
            6,
        );
        assert_json_u64(&value, &["gpu_subtree", "async_result_scatter_failures"], 1);
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_worker_ms"],
            44,
        );
        assert_json_u64(&value, &["gpu_subtree", "async_result_scatter_wait_ms"], 45);
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_join_wait_ms"],
            46,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_max_inflight"],
            1,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_peak_raw_bytes"],
            4096,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_peak_context_bytes"],
            1024,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "async_result_scatter_peak_host_bytes"],
            5120,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "async_result_scatter_worker_ms",
            ],
            44,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "async_result_scatter_wait_ms",
            ],
            45,
        );
        assert_json_u64(
            &value,
            &[
                "serial_timing",
                "diagnostics",
                "async_result_scatter_join_wait_ms",
            ],
            46,
        );
        assert_json_u64(&value, &["gpu_subtree", "post_last_gpu_cpu_tail_ms"], 47);
        assert_json_bool(
            &value,
            &["gpu_subtree", "post_last_gpu_cpu_tail_observed"],
            true,
        );
        assert_json_u64(&value, &["gpu_subtree", "async_flush_chunks"], 5);
        assert_json_u64(&value, &["gpu_subtree", "workspace_peak_bytes"], 4096);
        assert_json_u64(&value, &["gpu_subtree", "workspace_budget_bytes"], 8192);
        assert_json_f64(&value, &["gpu_subtree", "workspace_peak_over_budget"], 0.5);
        assert_json_u64(&value, &["gpu_subtree", "cpu_fallbacks"], 1);
        assert_json_u64(&value, &["gpu_subtree", "sidecar_gather_active_ms"], 88);
        assert_json_u64(&value, &["gpu_subtree", "sidecar_gather_hidden_ms"], 66);
        assert_json_u64(&value, &["gpu_subtree", "gpu_wait_for_sidecar_ms"], 22);
        assert_json_u64(&value, &["gpu_subtree", "sidecar_queue_max_bytes"], 123_456);
        assert_json_u64(&value, &["gpu_subtree", "sidecar_preload_children"], 3);
        assert_json_u64(&value, &["gpu_subtree", "sidecar_preload_workers"], 2);
        assert_json_u64(&value, &["gpu_subtree", "sidecar_queue_max_children"], 4);
        assert_json_u64(&value, &["gpu_subtree", "d1plus_admission_split_count"], 2);
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_admission_split_rows"],
            1_000,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_admission_split_max_before"],
            700,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_admission_split_max_after"],
            350,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d1plus_prevented_cpu_fallback_count"],
            2,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_attempt_count"], 3);
        assert_json_u64(&value, &["gpu_subtree", "persistent_success_count"], 2);
        assert_json_u64(&value, &["gpu_subtree", "persistent_fallback_count"], 1);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_worker_blocks"],
            5,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_worker_blocks"], 4);
        assert_json_u64(&value, &["gpu_subtree", "persistent_partition_pops"], 34);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_pushes"], 38);
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_pops"], 38);
        assert_json_u64(&value, &["gpu_subtree", "persistent_partition_claims"], 35);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_empty_polls"],
            36,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_frontier_end"],
            37,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_level_rounds"],
            4,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_claims"], 39);
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_empty_polls"], 40);
        assert_json_u64(&value, &["gpu_subtree", "persistent_done_cas_wins"], 1);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_active_wait_cycles"],
            48,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_busy_clocks"],
            101,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_wait_clocks"],
            102,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_busy_clocks"], 103);
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_wait_clocks"], 104);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_help_clocks"],
            105,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_help_clocks"], 106);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_partition_helped_topk_tiles"],
            9,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_topk_helped_large_tiles"],
            14,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_elastic_role_switches"],
            18,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_queue_overflow_flags"],
            0b1111,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_kernel_prepare_ms"], 57);
        assert_json_u64(&value, &["gpu_subtree", "persistent_kernel_event_ms"], 54);
        assert_json_u64(&value, &["gpu_subtree", "persistent_total_ms"], 60);
        assert_json_u64(&value, &["gpu_subtree", "persistent_topk_rows"], 66);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_count"], 76);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_effective_root_fanout"],
            3,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_effective_deeper_fanout"],
            1,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_fanout_assignment_slots"],
            90,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_replicated_rows"], 66);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_unique_resident_rows"],
            44,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_leaf_row_amplification_x1000"],
            1500,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_active_large"],
            1,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_generation"],
            9,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_phase"],
            2,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_total_tiles"],
            1_123_089,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_count_next"],
            1_123_089,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_count_done"],
            1_123_089,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_scatter_next"],
            1_123_089,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_snapshot_large_scatter_done"],
            1_123_089,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_large_node_claims"], 41);
        assert_json_u64(&value, &["gpu_subtree", "persistent_large_node_splits"], 42);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_split_retries"],
            43,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_balanced_fallbacks"],
            44,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_large_node_rows"], 43);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_node_max_rows"],
            8192,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_node_empty_polls"],
            44,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "persistent_large_node_serialized_wait_cycles",
            ],
            45,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_count_distance_clocks"],
            201,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_count_commit_clocks"],
            202,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_scatter_clocks"],
            203,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_publish_clocks"],
            204,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_ticket_range_claims"],
            205,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_physical_tiles"],
            206,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_sq8_dp4a_tiles"],
            207,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_large_sq8_wmma_tiles"],
            0,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_leaf_row_owned_tasks"],
            209,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_staged_tasks"], 0);
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_leaf_score_clocks"],
            211,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_leaf_row_lock_merge_clocks"],
            212,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_topk_local_counter_flushes"],
            213,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_topk_idle_backoff_calls"],
            214,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_topk_idle_backoff_requested_ns"],
            74_900,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_topk_idle_leaf_resets"],
            216,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "persistent_topk_idle_large_resets"],
            217,
        );
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_size_min"], 128);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_size_max"], 1024);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_size_sum"], 4096);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_size_count"], 6);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_target"], 512);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_min"], 128);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_max"], 1024);
        assert_json_u64(&value, &["gpu_subtree", "persistent_leaf_tile_rows"], 256);
        assert_json_u64(&value, &["gpu_subtree", "compact_unsupported_count"], 2);
        assert_json_u64(&value, &["gpu_subtree", "compact_unsupported_points"], 20);
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("d0_capacity"))
                .and_then(|section| section.get("enabled"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_json_u64(&value, &["gpu_subtree", "d0_capacity", "c_gpu"], 400_000);
        assert_json_u64(&value, &["gpu_subtree", "d0_capacity", "c_target"], 300_000);
        assert_json_u64(&value, &["gpu_subtree", "d0_capacity", "k_capacity"], 200);
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_capacity", "quality_min_root_leaders"],
            64,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_capacity", "root_leader_cap"],
            10_000,
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("d0_capacity"))
                .and_then(|section| section.get("root_leader_source"))
                .and_then(Value::as_str),
            Some("configured")
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_capacity", "actual_child_max_unique_rows"],
            320_000,
        );
        assert_json_f64(
            &value,
            &["gpu_subtree", "d0_capacity", "actual_child_max_over_c_high"],
            320_000_f64 / 340_000_f64,
        );
        assert_json_f64(
            &value,
            &[
                "gpu_subtree",
                "d0_capacity",
                "actual_child_p99_over_c_target",
            ],
            320_000_f64 / 300_000_f64,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_capacity", "root_leaders_final"],
            64,
        );
    }

    #[test]
    fn profile_json_includes_d0_throughput_model_when_present() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.rbc.d0_throughput_model =
            Some(super::throughput_model::D0ThroughputModelDecision {
                mode: super::PiPNNGpuSubtreeD0ThroughputModelMode::Report,
                memory_safe_rows: 400_000,
                memory_safe_root_leaders: 64,
                selected_rows: 400_000,
                selected_root_leaders: 64,
                score_preferred_rows: 240_000,
                score_preferred_root_leaders: 64,
                selected_ratio_x1000: 1000,
                candidate_count: 1,
                candidate_generation_version: "leader-aware-v1",
                candidate_sources: vec!["row_ratio", "leader_target"],
                child_skew_factor_x1000: 1500,
                row_weight: 1,
                leaf_weight: 1024,
                topk_weight: 256,
                split_weight: 4096,
                d0_leader_weight: 64,
                predicted_persistent_ms: None,
                predicted_preload_visible_ms: None,
                predicted_d0_penalty_ms: None,
                predicted_total_ms: Some(u64::MAX as u128 + 1),
                predicted_persistent_score: 123,
                predicted_preload_visible_score: 45,
                predicted_d0_penalty_score: 4096,
                structural_total_score: 4219,
                predicted_p95_child_rows: 300_000,
                predicted_leaf_count: 1200,
                predicted_topk_pops: 900,
                predicted_lazy_split_count: 1,
                calibration_state: "missing",
                calibration_source: Some("unit".to_string()),
                reason: "report_only_missing_calibration_score_prefers_smaller",
                candidates: vec![super::throughput_model::D0ThroughputCandidate {
                    source: super::throughput_model::D0ThroughputCandidateSource::RowRatio,
                    rows: 400_000,
                    ratio_x1000: 1000,
                    target_root_leaders: None,
                    root_leaders: 64,
                    predicted_avg_child_rows: 200_000,
                    predicted_p95_child_rows: 300_000,
                    predicted_leaf_count: 1200,
                    predicted_topk_pops: 900,
                    predicted_lazy_split_count: 1,
                    predicted_persistent_score: 123,
                    predicted_preload_visible_score: 45,
                    predicted_d0_penalty_score: 4096,
                    structural_total_score: 4219,
                    predicted_persistent_ms: None,
                    predicted_preload_visible_ms: None,
                    predicted_d0_penalty_ms: None,
                    predicted_child_ms: None,
                    predicted_total_ms: None,
                    admitted: true,
                    reject_reason: None,
                }],
            });

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_throughput_model", "memory_safe_rows"],
            400_000,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d0_throughput_model",
                "memory_safe_root_leaders",
            ],
            64,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d0_throughput_model",
                "selected_root_leaders",
            ],
            64,
        );
        assert_json_u64(
            &value,
            &["gpu_subtree", "d0_throughput_model", "score_preferred_rows"],
            240_000,
        );
        assert_json_u64(
            &value,
            &[
                "gpu_subtree",
                "d0_throughput_model",
                "score_preferred_root_leaders",
            ],
            64,
        );
        assert_eq!(
            value["gpu_subtree"]["d0_throughput_model"]["mode"].as_str(),
            Some("report")
        );
        assert_eq!(
            value["gpu_subtree"]["d0_throughput_model"]["candidate_generation_version"].as_str(),
            Some("leader-aware-v1")
        );
        assert_eq!(
            value["gpu_subtree"]["d0_throughput_model"]["calibration_source"].as_str(),
            Some("unit")
        );
        let candidate_sources = value["gpu_subtree"]["d0_throughput_model"]["candidate_sources"]
            .as_array()
            .unwrap();
        assert!(
            candidate_sources
                .iter()
                .any(|source| source.as_str() == Some("row_ratio"))
        );
        assert!(
            candidate_sources
                .iter()
                .any(|source| source.as_str() == Some("leader_target"))
        );
        assert!(value["gpu_subtree"]["d0_throughput_model"]["predicted_persistent_ms"].is_null());
        assert_eq!(
            value["gpu_subtree"]["d0_throughput_model"]["predicted_total_ms"].as_str(),
            Some("18446744073709551616")
        );
        let candidates = value["gpu_subtree"]["d0_throughput_model"]["candidates"]
            .as_array()
            .unwrap();
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate["source"].as_str(), Some("row_ratio"));
        assert_eq!(
            candidate["predicted_avg_child_rows"].as_u64(),
            Some(200_000)
        );
        assert_eq!(candidate["predicted_d0_penalty_score"].as_u64(), Some(4096));
        assert_eq!(candidate["structural_total_score"].as_u64(), Some(4219));
        assert_eq!(candidate["admitted"].as_bool(), Some(true));
        assert!(candidate["reject_reason"].is_null());
    }

    #[test]
    fn oom_profile_json_includes_gpu_subtree_boundary_section() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let mut profile = minimal_oom_profile_for_json(super::TempFileReport::default());
        profile.partitioner = PiPNNPartitionerKind::GpuSubtree;

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("enabled"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("root_assignment_backend"))
                .and_then(Value::as_str),
            Some("cpu")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("assignment_backend"))
                .and_then(Value::as_str),
            Some("current")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("subtree_assignment_backend"))
                .and_then(Value::as_str),
            Some("current")
        );
        assert_eq!(
            value
                .get("gpu_subtree")
                .and_then(|section| section.get("leaf_topk_backend"))
                .and_then(Value::as_str),
            Some("persistent-custom")
        );
        assert_json_u64(&value, &["gpu_subtree", "child_count"], 0);
        assert_json_u64(&value, &["gpu_subtree", "child_points"], 0);
        assert_json_u64(&value, &["gpu_subtree", "assignment_passes"], 0);
        assert_json_u64(&value, &["gpu_subtree", "leaf_count"], 0);
        assert_json_u64(&value, &["gpu_subtree", "compact_topk_batches"], 0);
        assert_json_u64(&value, &["gpu_subtree", "compact_topk_rows"], 0);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_total_ms"], 0);
        assert_json_u64(&value, &["gpu_subtree", "library_topk_rows"], 0);
    }

    #[test]
    fn collect_temp_file_report_classifies_partition_candidate_and_other_bytes() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("sketches.bin"), vec![0u8; 10]).unwrap();
        fs::write(
            dir.path().join("candidate_shard00000_part000.log"),
            vec![0u8; 11],
        )
        .unwrap();
        fs::write(dir.path().join("candidate_segments.manifest"), vec![0u8; 7]).unwrap();
        fs::write(dir.path().join("partition_d00_runs.bin"), vec![0u8; 12]).unwrap();
        fs::write(dir.path().join("partition_d02_runs.bin"), vec![0u8; 13]).unwrap();
        fs::write(dir.path().join("misc.tmp"), vec![0u8; 14]).unwrap();

        let report = super::collect_temp_file_report(dir.path()).unwrap();
        assert_eq!(report.total.files, 6);
        assert_eq!(report.total.bytes, 67);
        assert_eq!(report.candidate.files, 2);
        assert_eq!(report.candidate.bytes, 18);
        assert_eq!(report.partition_depth0.files, 1);
        assert_eq!(report.partition_depth0.bytes, 12);
        assert_eq!(report.partition_depth2.files, 1);
        assert_eq!(report.partition_depth2.bytes, 13);
        assert_eq!(report.other.files, 2);
        assert_eq!(report.other.bytes, 24);
    }

    #[test]
    fn derive_oom_worker_count_obeys_memory_budget() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 8192;

        let tight = super::derive_oom_worker_count(64, &params, 768);
        assert!(tight < 64);
        assert!(tight >= 1);
    }

    #[test]
    fn derive_oom_worker_count_reserves_budget_for_non_leaf_work() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 12 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let workers = super::derive_oom_worker_count(42, &params, 768);
        assert!(
            workers < 42,
            "worker count should be clamped under a 12 GiB OOM budget"
        );
        assert!(workers >= 1);
    }

    #[test]
    fn oom_memory_plan_reserves_reservoir_floor_without_clamping_leaf_threads() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 28 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let plan = super::OomMemoryPlan::from_params(&params);

        assert!(plan.reservoir_budget_bytes < plan.total_budget_bytes);
        assert!(plan.reservoir_budget_bytes > 0);
        assert_eq!(plan.resident_subtree_budget_bytes, 0);
        assert_eq!(super::derive_oom_worker_count(42, &params, 768), 42);
    }

    #[test]
    fn oom_memory_plan_keeps_reservoir_target_when_raw_cache_and_target_both_fit() {
        let mut base = PiPNNParams::default();
        base.oom_memory_budget_bytes = 40 * 1024 * 1024 * 1024;
        let baseline = super::OomMemoryPlan::from_params(&base);

        let mut with_cache = base.clone();
        with_cache.oom_raw_vector_cache_bytes = 6 * 1024 * 1024 * 1024;
        let plan = super::OomMemoryPlan::from_params(&with_cache);

        assert_eq!(plan.total_budget_bytes, baseline.total_budget_bytes);
        assert_eq!(
            plan.raw_vector_cache_budget_bytes,
            with_cache.oom_raw_vector_cache_bytes
        );
        assert_eq!(plan.reservoir_budget_bytes, baseline.reservoir_budget_bytes);
        assert_eq!(
            plan.resident_subtree_budget_bytes,
            baseline.resident_subtree_budget_bytes
        );
        assert_eq!(plan.scratch_budget_bytes, baseline.scratch_budget_bytes);
    }

    #[test]
    fn online_suffix_memory_is_reserved_before_resident_prefix() {
        let rows = 10_000_000;
        let mut baseline = PiPNNParams::default();
        baseline.oom_enable = true;
        baseline.oom_memory_budget_bytes = 10 * 1024 * 1024 * 1024;
        baseline.partitioner = PiPNNPartitionerKind::GpuSubtree;

        let mut online = baseline.clone();
        online.online_suffix_reduce_mode = super::PiPNNOnlineSuffixReduceMode::DirectFirst;

        let baseline_plan = super::OomMemoryPlan::from_params_for_dataset(&baseline, rows, 1024);
        let online_plan = super::OomMemoryPlan::from_params_for_dataset(&online, rows, 1024);
        assert!(online_plan.online_suffix_budget_bytes > 0);
        assert!(online_plan.reservoir_budget_bytes < baseline_plan.reservoir_budget_bytes);
    }

    #[test]
    fn log_assisted_memory_plan_tracks_delta_and_compaction_dram() {
        let rows = 10_000_000;
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 16 * 1024 * 1024 * 1024;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.online_suffix_reduce_mode = super::PiPNNOnlineSuffixReduceMode::LogAssisted;
        params.online_suffix_delta_budget_bytes = 8 * 1024 * 1024 * 1024;

        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, 1024);
        assert!(plan.online_suffix_delta_buffer_budget_bytes > 0);
        assert!(plan.online_suffix_compaction_budget_bytes > 0);
        assert_eq!(plan.online_suffix_page_cache_budget_bytes, 0);
        assert_eq!(plan.online_suffix_decode_budget_bytes, 0);
        assert!(plan.total_tracked_bytes() <= plan.total_budget_bytes);
    }

    #[test]
    fn log_assisted_memory_auto_range_targets_a_bounded_base_segment() {
        let points = super::effective_log_assisted_range_points(0, 10_000_000, 64).unwrap();
        let layout = super::DiskReservoirLayout::new(64, 4096).unwrap();
        let bytes = points * layout.record_bytes;
        assert!((64 * 1024 * 1024..=256 * 1024 * 1024).contains(&bytes));
        assert_eq!(
            super::effective_log_assisted_range_points(1234, 10_000_000, 64).unwrap(),
            1234
        );
    }

    #[test]
    fn online_suffix_metadata_budget_covers_location_and_bitmap() {
        let points = 35_000_000;
        let bytes = super::online_suffix_metadata_budget_bytes(points, 2.0);
        let location_bytes = points * size_of::<u64>();
        let allocator_bitmap_bytes = (points * 2).div_ceil(8);
        assert!(bytes >= location_bytes + allocator_bitmap_bytes);
    }

    #[test]
    fn online_suffix_memory_is_not_reserved_for_ineligible_build() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.online_suffix_reduce_mode = super::PiPNNOnlineSuffixReduceMode::DirectFirst;
        assert_eq!(
            super::online_suffix_eligibility(&params),
            super::OnlineSuffixEligibility::ExternalFallback("partitioner_not_gpu_subtree")
        );
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, 1_000_000, 128);
        assert!(!plan.online_suffix_enabled);
        assert_eq!(plan.online_suffix_budget_bytes, 0);
    }

    #[test]
    fn online_suffix_supported_policy_fallback_reasons_are_exact() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.online_suffix_reduce_mode = super::PiPNNOnlineSuffixReduceMode::DirectFirst;
        params.hashprune_evidence_mode = super::PiPNNHashPruneEvidenceMode::Enforce;
        assert_eq!(
            super::online_suffix_eligibility(&params),
            super::OnlineSuffixEligibility::ExternalFallback("evidence_enforce_not_supported")
        );

        params.hashprune_evidence_mode = super::PiPNNHashPruneEvidenceMode::Off;
        params.compact_spill_digest_enable = true;
        assert_eq!(
            super::online_suffix_eligibility(&params),
            super::OnlineSuffixEligibility::ExternalFallback("compact_spill_digest_changes_pool")
        );
    }

    #[test]
    fn online_suffix_policy_key_changes_with_every_serialized_policy_field() {
        let base = PiPNNParams::default();
        let base_key = super::online_suffix_policy_key(&base);
        let mut variants = Vec::new();

        let mut changed = base.clone();
        changed.l_max += 1;
        variants.push(changed);
        let mut changed = base.clone();
        changed.hashprune_evidence_mode = super::PiPNNHashPruneEvidenceMode::Report;
        variants.push(changed);
        let mut changed = base.clone();
        changed.hashprune_backfill_mode = super::PiPNNHashPruneBackfillMode::DistanceGated;
        variants.push(changed);
        let mut changed = base.clone();
        changed.hashprune_backfill_budget = 1;
        variants.push(changed);
        let mut changed = base.clone();
        changed.hashprune_backfill_alpha = 1.1;
        variants.push(changed);
        let mut changed = base.clone();
        changed.hashprune_rhalf_hash_quota = Some(2);
        variants.push(changed);
        let mut changed = base.clone();
        changed.hashprune_soft_tier_priority_alpha = Some(1.2);
        variants.push(changed);
        let mut changed = base;
        changed.hashprune_rhalf_max_backfill_per_row = Some(3);
        variants.push(changed);

        for changed in variants {
            assert_ne!(super::online_suffix_policy_key(&changed), base_key);
        }
    }

    #[test]
    fn d0_read_progress_is_monotonic_and_clamped_to_target() {
        let progress = super::D0ReadProgress::new(8);

        progress.advance(3);
        progress.advance(2);
        progress.advance(99);

        let snapshot = progress.snapshot();
        assert_eq!(snapshot.ready_prefix, 8);
        assert_eq!(snapshot.initialized_prefix, 0);
        assert_eq!(snapshot.terminal, super::D0ReadTerminal::Running);
    }

    #[test]
    fn d0_read_progress_cancellation_wakes_a_waiter() {
        let progress = std::sync::Arc::new(super::D0ReadProgress::new(8));
        let waiter = std::sync::Arc::clone(&progress);

        let result = std::thread::scope(|scope| {
            let worker = scope.spawn(move || waiter.wait_for_ready_prefix(0));
            progress.wait_until_waiting_for_test();
            progress.cancel("injected D0 failure");
            worker.join().unwrap()
        });

        let error = result.unwrap_err();
        assert!(error.contains("injected D0 failure"));
    }

    #[test]
    fn d0_read_progress_rejects_a_noncontiguous_copied_shard() {
        let progress = super::D0ReadProgress::new(8);

        progress.record_copied_shard(0, 2).unwrap();
        let error = progress.record_copied_shard(3, 2).unwrap_err();

        assert!(error.contains("expected row_start 2, got 3"));
        assert_eq!(progress.snapshot().ready_prefix, 2);
    }

    #[test]
    fn deferred_reservoirs_never_initialize_beyond_d0_ready_prefix() {
        let progress = std::sync::Arc::new(super::D0ReadProgress::new(4));
        let reservoirs = super::DeferredReservoirs::new(4, 8)
            .with_initialization_config(super::ReservoirInitializationConfig {
                requested: "2".to_string(),
                overlap_enabled: true,
                workers: 2,
                invalid_override: false,
            })
            .with_d0_read_progress(std::sync::Arc::clone(&progress));

        std::thread::scope(|scope| {
            let initializer = scope.spawn(|| reservoirs.initialize());
            progress.wait_until_waiting_for_test();
            assert_eq!(progress.snapshot().initialized_prefix, 0);

            progress.advance(2);
            progress.wait_until_initialized_for_test(2);
            let snapshot = progress.snapshot();
            assert_eq!(snapshot.ready_prefix, 2);
            assert_eq!(snapshot.initialized_prefix, 2);

            progress.complete(4);
            initializer.join().unwrap().unwrap();
        });

        let allocation = reservoirs.into_allocation().unwrap();
        assert_eq!(allocation.reservoirs.len(), 4);
        assert!(allocation.initialization.d0_paced);
        assert_eq!(allocation.initialization.ready_prefix, 4);
        assert_eq!(allocation.initialization.initialized_prefix, 4);
    }

    #[test]
    fn deferred_reservoirs_allocate_only_when_d1_is_prepared() {
        let reservoirs = super::DeferredReservoirs::new(4, 8);
        assert!(!reservoirs.is_initialized());

        reservoirs.prepare_for_d1().unwrap();

        assert!(reservoirs.is_initialized());
        let allocation = reservoirs.into_allocation().unwrap();
        assert_eq!(allocation.reservoirs.len(), 4);
        assert_eq!(
            allocation.estimated_bytes,
            super::estimate_reservoir_bytes(4, 8)
        );
        assert!(allocation.allocation_wall >= Duration::ZERO);
    }

    #[test]
    fn deferred_reservoirs_parallel_initialization_preserves_source_indexing() {
        let reservoirs = super::DeferredReservoirs::new(128, 8).with_initialization_config(
            super::ReservoirInitializationConfig {
                requested: "4".to_string(),
                overlap_enabled: true,
                workers: 4,
                invalid_override: false,
            },
        );

        reservoirs.initialize().unwrap();
        reservoirs.prepare_for_d1().unwrap();

        let allocation = reservoirs.into_allocation().unwrap();
        assert_eq!(allocation.reservoirs.len(), 128);
        assert_eq!(allocation.initialization.workers, 4);
        assert!(!allocation.initialization.parallel_fallback);
        assert_eq!(
            allocation.estimated_bytes,
            super::estimate_reservoir_bytes(128, 8)
        );
        for (source, reservoir) in allocation.reservoirs.iter().enumerate() {
            assert!(reservoir.lock().slots().is_empty(), "source={source}");
        }
    }

    #[test]
    fn deferred_reservoirs_concurrent_prepare_publishes_one_allocation() {
        let reservoirs = super::DeferredReservoirs::new(256, 8).with_initialization_config(
            super::ReservoirInitializationConfig {
                requested: "4".to_string(),
                overlap_enabled: true,
                workers: 4,
                invalid_override: false,
            },
        );

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| reservoirs.prepare_for_d1().unwrap());
            }
        });

        assert_eq!(reservoirs.allocation_count_for_test(), 1);
        assert_eq!(reservoirs.into_allocation().unwrap().reservoirs.len(), 256);
    }

    #[test]
    fn reservoir_parallel_pool_failure_falls_back_to_sequential_allocation() {
        let config = super::ReservoirInitializationConfig {
            requested: "4".to_string(),
            overlap_enabled: true,
            workers: 4,
            invalid_override: false,
        };
        let allocation =
            super::allocate_resident_reservoirs_with_pool_builder(32, 8, &config, |_| {
                Err("injected thread-pool failure".to_string())
            });

        assert_eq!(allocation.reservoirs.len(), 32);
        assert_eq!(allocation.initialization.workers, 1);
        assert!(allocation.initialization.parallel_fallback);
    }

    #[test]
    fn paced_reservoir_pipeline_error_cancels_blocked_initializer() {
        let progress = std::sync::Arc::new(super::D0ReadProgress::new(4));
        let reservoirs = super::DeferredReservoirs::new(4, 8)
            .with_d0_read_progress(std::sync::Arc::clone(&progress));

        let error = super::run_with_reservoir_initialization_overlap(
            true,
            Some(progress.as_ref()),
            || reservoirs.initialize(),
            || Err::<usize, _>(AnnError::log_index_error("injected D0 failure".to_string())),
        )
        .unwrap_err();

        assert!(error.to_string().contains("injected D0 failure"));
        assert!(!reservoirs.has_ready_allocation_for_test());
        assert_eq!(reservoirs.allocation_count_for_test(), 1);
    }

    #[test]
    fn paced_reservoir_d1_waiter_observes_the_same_allocation() {
        let progress = std::sync::Arc::new(super::D0ReadProgress::new(8));
        let reservoirs = super::DeferredReservoirs::new(8, 8)
            .with_d0_read_progress(std::sync::Arc::clone(&progress));

        std::thread::scope(|scope| {
            let initializer = scope.spawn(|| reservoirs.initialize());
            progress.wait_until_waiting_for_test();
            let d1 = scope.spawn(|| reservoirs.prepare_for_d1());
            progress.complete(8);
            initializer.join().unwrap().unwrap();
            d1.join().unwrap().unwrap();
        });

        assert_eq!(reservoirs.allocation_count_for_test(), 1);
        assert_eq!(reservoirs.into_allocation().unwrap().reservoirs.len(), 8);
    }

    #[test]
    fn reservoir_initialization_runs_concurrently_with_d0_work() {
        let init_started = std::sync::Barrier::new(2);
        let release_init = std::sync::Barrier::new(2);
        let d0_observed_init = AtomicBool::new(false);

        let result = super::run_with_reservoir_initialization_overlap(
            true,
            None,
            || {
                init_started.wait();
                release_init.wait();
                Ok(())
            },
            || {
                init_started.wait();
                d0_observed_init.store(true, Ordering::SeqCst);
                release_init.wait();
                Ok(17usize)
            },
        )
        .unwrap();

        assert_eq!(result, 17);
        assert!(d0_observed_init.load(Ordering::SeqCst));
    }

    #[test]
    fn reservoir_initialization_disabled_runs_only_the_pipeline() {
        let initialize_called = AtomicBool::new(false);

        let result = super::run_with_reservoir_initialization_overlap(
            false,
            None,
            || {
                initialize_called.store(true, Ordering::SeqCst);
                Ok(())
            },
            || Ok(23usize),
        )
        .unwrap();

        assert_eq!(result, 23);
        assert!(!initialize_called.load(Ordering::SeqCst));
    }

    #[test]
    fn reservoir_initialization_pipeline_error_wins_after_join() {
        let init_finished = AtomicBool::new(false);
        let error = super::run_with_reservoir_initialization_overlap(
            true,
            None,
            || {
                init_finished.store(true, Ordering::SeqCst);
                Ok(())
            },
            || Err::<usize, _>(AnnError::log_index_error("injected D0 failure".to_string())),
        )
        .unwrap_err();

        assert!(init_finished.load(Ordering::SeqCst));
        assert!(error.to_string().contains("injected D0 failure"));
    }

    const PHASE0_IDENTITY_NUM_POINTS: usize = 3;
    const PHASE0_IDENTITY_L_MAX: usize = 2;
    const PHASE0_IDENTITY_SHA256: &str =
        "f290cbb815b9a995825ff069620fdb785b8a354918753092c7b83e8de12eefa6";

    #[derive(Clone, Copy, Debug)]
    enum Phase0IdentityBackend {
        Resident,
        Spill,
        Hybrid,
    }

    impl Phase0IdentityBackend {
        fn label(self) -> &'static str {
            match self {
                Self::Resident => "resident",
                Self::Spill => "spill",
                Self::Hybrid => "hybrid",
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Phase0IdentityTraceShape {
        batch_count: usize,
        rows: usize,
        row_k_slots: usize,
        point_ids: usize,
        topk_ids: usize,
        topk_dists: usize,
        estimated_dtoh_bytes: usize,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Phase0IdentityProfileShape {
        leaves: usize,
        points: usize,
        sketch_rows_prefetched: usize,
        gpu_h2d_bytes: usize,
        gpu_d2h_bytes: usize,
        edge_sink_edges: usize,
        edge_sink_flushes: usize,
        compact_lune_witness_slots: usize,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Phase0IdentitySelection {
        target: u32,
        hash: u16,
        dist_bits: u32,
        mandatory: bool,
        overflow: bool,
        evidence: Option<HashPruneEvidenceClass>,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Phase0IdentityEvidenceCounters {
        reciprocal_checks: usize,
        reciprocal_hits: usize,
        support_once_candidates: usize,
        support_twice_candidates: usize,
        class_counts: [usize; 6],
        candidates_before_coalesce: usize,
        candidates_after_coalesce: usize,
        duplicate_records_removed: usize,
        unannotated_candidates: usize,
        qualified_collision_losers: usize,
        overflow_inserts: usize,
        overflow_replacements: usize,
        overflow_rejects: usize,
        overflow_evictions_by_new_hash: usize,
        rows_with_overflow: usize,
        spill_records_avoided: usize,
        spill_bytes_avoided: usize,
    }

    #[derive(Debug)]
    struct Phase0IdentityRun {
        policy: HashPruneEvidencePolicy,
        adjacency: Vec<Vec<u32>>,
        digest: String,
        resident_selection: Option<Vec<Vec<Phase0IdentitySelection>>>,
        trace_shape: Phase0IdentityTraceShape,
        leaf: LeafProfile,
        spill: SpillRunStats,
        reduce: SpillReduceStats,
    }

    type Phase0IdentityRunner = fn(
        &[CompactTopKBatch],
        &SketchStore,
        &PiPNNParams,
        HashPruneEvidencePolicy,
    ) -> AnnResult<Phase0IdentityRun>;

    fn phase0_identity_compact_trace() -> Vec<CompactTopKBatch> {
        let first = CompactTopKBatch {
            leaf_id: 17,
            point_ids: vec![0, 1, 2],
            rows: 3,
            k: 2,
            topk_ids: vec![1, 2, 2, 0, 1, u32::MAX],
            topk_dists: vec![1.0, 2.0, 1.5, 1.0, 1.5, f32::INFINITY],
            estimated_dtoh_bytes: 3 * 2 * (size_of::<u32>() + size_of::<f32>()),
            ..CompactTopKBatch::default()
        };
        let mut replay = first.clone();
        replay.leaf_id = 29;
        vec![first, replay]
    }

    fn phase0_identity_sketches() -> SketchStore {
        SketchStore::from_test_data(vec![2.0, 3.0, 1.0, 2.0, 0.0, 1.0], 2)
    }

    fn phase0_identity_trace_shape(trace: &[CompactTopKBatch]) -> Phase0IdentityTraceShape {
        Phase0IdentityTraceShape {
            batch_count: trace.len(),
            rows: trace.iter().map(|batch| batch.rows).sum(),
            row_k_slots: trace
                .iter()
                .map(|batch| batch.rows.saturating_mul(batch.k))
                .sum(),
            point_ids: trace.iter().map(|batch| batch.point_ids.len()).sum(),
            topk_ids: trace.iter().map(|batch| batch.topk_ids.len()).sum(),
            topk_dists: trace.iter().map(|batch| batch.topk_dists.len()).sum(),
            estimated_dtoh_bytes: trace
                .iter()
                .map(CompactTopKBatch::estimated_total_dtoh_bytes)
                .sum(),
        }
    }

    fn phase0_identity_profile_shape(profile: &LeafProfile) -> Phase0IdentityProfileShape {
        Phase0IdentityProfileShape {
            leaves: profile.leaves,
            points: profile.points,
            sketch_rows_prefetched: profile.sketch_rows_prefetched,
            gpu_h2d_bytes: profile.gpu_h2d_bytes,
            gpu_d2h_bytes: profile.gpu_d2h_bytes,
            edge_sink_edges: profile.edge_sink_edges,
            edge_sink_flushes: profile.edge_sink_flushes,
            compact_lune_witness_slots: profile.compact_lune_witness_slots,
        }
    }

    fn phase0_identity_evidence_counters(
        run: &Phase0IdentityRun,
    ) -> Phase0IdentityEvidenceCounters {
        let evidence = super::final_hashprune_evidence_profile(
            &run.leaf.hashprune_evidence,
            &run.reduce.evidence,
        );
        Phase0IdentityEvidenceCounters {
            reciprocal_checks: evidence.reciprocal_checks,
            reciprocal_hits: evidence.reciprocal_hits,
            support_once_candidates: evidence.support_once_candidates,
            support_twice_candidates: evidence.support_twice_candidates,
            class_counts: evidence.class_counts,
            candidates_before_coalesce: evidence.candidates_before_coalesce,
            candidates_after_coalesce: evidence.candidates_after_coalesce,
            duplicate_records_removed: evidence.duplicate_records_removed,
            unannotated_candidates: evidence.unannotated_candidates,
            qualified_collision_losers: evidence.qualified_collision_losers,
            overflow_inserts: evidence.overflow_inserts,
            overflow_replacements: evidence.overflow_replacements,
            overflow_rejects: evidence.overflow_rejects,
            overflow_evictions_by_new_hash: evidence.overflow_evictions_by_new_hash,
            rows_with_overflow: evidence.rows_with_overflow,
            spill_records_avoided: evidence.spill_records_avoided,
            spill_bytes_avoided: evidence.spill_bytes_avoided,
        }
    }

    fn phase0_identity_flush_trace(
        trace: &[CompactTopKBatch],
        sketches: &SketchStore,
        params: &PiPNNParams,
        sink: &dyn PendingEdgeSink,
    ) -> AnnResult<LeafProfile> {
        let mut merged = LeafProfile::default();
        for batch in trace {
            merged.merge(flush_compact_topk_batch(batch, sketches, params, sink)?);
        }
        Ok(merged)
    }

    fn phase0_identity_take_resident(
        reservoirs: super::DeferredReservoirs,
    ) -> AnnResult<(Vec<Vec<u32>>, Vec<Vec<Phase0IdentitySelection>>)> {
        let allocation = reservoirs.into_allocation()?;
        let mut adjacency = Vec::with_capacity(allocation.reservoirs.len());
        let mut selection = Vec::with_capacity(allocation.reservoirs.len());
        for reservoir in allocation.reservoirs.into_legacy_for_test() {
            let slots = reservoir.into_inner().into_slots();
            adjacency.push(slots.iter().map(|slot| slot.target).collect());
            selection.push(
                slots
                    .iter()
                    .map(|slot| Phase0IdentitySelection {
                        target: slot.target,
                        hash: slot.hash,
                        dist_bits: slot.dist.to_bits(),
                        mandatory: slot.meta.is_mandatory(),
                        overflow: slot.meta.is_overflow(),
                        evidence: slot.meta.evidence(),
                    })
                    .collect(),
            );
        }
        Ok((adjacency, selection))
    }

    fn phase0_identity_canonical_ordered_adjacency_sha256(
        adjacency: &[Vec<u32>],
        l_max: usize,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update((adjacency.len() as u64).to_le_bytes());
        hasher.update((l_max as u64).to_le_bytes());
        for (src, neighbors) in adjacency.iter().enumerate() {
            hasher.update((src as u32).to_le_bytes());
            hasher.update((neighbors.len() as u32).to_le_bytes());
            for &dst in neighbors {
                hasher.update(dst.to_le_bytes());
            }
        }
        hex::encode(hasher.finalize())
    }

    fn phase0_identity_result(
        trace: &[CompactTopKBatch],
        policy: HashPruneEvidencePolicy,
        adjacency: Vec<Vec<u32>>,
        resident_selection: Option<Vec<Vec<Phase0IdentitySelection>>>,
        leaf: LeafProfile,
        spill: SpillRunStats,
        reduce: SpillReduceStats,
    ) -> Phase0IdentityRun {
        let digest =
            phase0_identity_canonical_ordered_adjacency_sha256(&adjacency, PHASE0_IDENTITY_L_MAX);
        Phase0IdentityRun {
            policy,
            adjacency,
            digest,
            resident_selection,
            trace_shape: phase0_identity_trace_shape(trace),
            leaf,
            spill,
            reduce,
        }
    }

    fn phase0_identity_run_resident(
        trace: &[CompactTopKBatch],
        sketches: &SketchStore,
        params: &PiPNNParams,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<Phase0IdentityRun> {
        let reservoirs = super::DeferredReservoirs::new_with_policy(
            PHASE0_IDENTITY_NUM_POINTS,
            PHASE0_IDENTITY_L_MAX,
            policy,
        );
        reservoirs.prepare_for_d1()?;
        let leaf = phase0_identity_flush_trace(trace, sketches, params, &reservoirs)?;
        let (adjacency, selection) = phase0_identity_take_resident(reservoirs)?;
        Ok(phase0_identity_result(
            trace,
            policy,
            adjacency,
            Some(selection),
            leaf,
            SpillRunStats::default(),
            SpillReduceStats::default(),
        ))
    }

    fn phase0_identity_run_spill(
        trace: &[CompactTopKBatch],
        sketches: &SketchStore,
        params: &PiPNNParams,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<Phase0IdentityRun> {
        let dir = tempdir().unwrap();
        let sink = super::create_candidate_spill_sink(
            &dir.path().join("candidate.run"),
            params,
            1,
            DirectIoConfig::disabled(),
            policy,
            super::HashPruneBackfillPolicy::off(),
        )?;
        let leaf = phase0_identity_flush_trace(trace, sketches, params, &sink)?;
        let artifacts = sink.finish()?;
        let spill = super::ExternalHashPruneReducer::summarize_spill_artifacts(&artifacts);
        let mut adjacency = vec![Vec::new(); PHASE0_IDENTITY_NUM_POINTS];
        let reduce = super::ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_with_policy(
            PHASE0_IDENTITY_NUM_POINTS,
            PHASE0_IDENTITY_L_MAX,
            1,
            &artifacts,
            policy,
            |point, neighbors| {
                adjacency[point as usize] = neighbors;
                Ok(())
            },
        )?;
        Ok(phase0_identity_result(
            trace, policy, adjacency, None, leaf, spill, reduce,
        ))
    }

    fn phase0_identity_run_hybrid(
        trace: &[CompactTopKBatch],
        sketches: &SketchStore,
        params: &PiPNNParams,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<Phase0IdentityRun> {
        let dir = tempdir().unwrap();
        let resident_points = 2;
        let reservoirs = super::DeferredReservoirs::new_with_policy(
            resident_points,
            PHASE0_IDENTITY_L_MAX,
            policy,
        );
        let spill_sink = super::create_candidate_spill_sink(
            &dir.path().join("candidate.run"),
            params,
            1,
            DirectIoConfig::disabled(),
            policy,
            super::HashPruneBackfillPolicy::off(),
        )?;
        let hybrid_sink = super::HybridEdgeSink::new_with_policy(
            resident_points,
            &reservoirs,
            &spill_sink,
            policy,
        );
        hybrid_sink.prepare_for_d1()?;
        let leaf = phase0_identity_flush_trace(trace, sketches, params, &hybrid_sink)?;
        drop(hybrid_sink);

        let artifacts = spill_sink.finish()?;
        let spill = super::ExternalHashPruneReducer::summarize_spill_artifacts(&artifacts);
        let (mut adjacency, selection) = phase0_identity_take_resident(reservoirs)?;
        adjacency.resize_with(PHASE0_IDENTITY_NUM_POINTS, Vec::new);
        let reduce = super::ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_from_with_policy(
            resident_points,
            PHASE0_IDENTITY_NUM_POINTS,
            PHASE0_IDENTITY_L_MAX,
            1,
            &artifacts,
            policy,
            |point, neighbors| {
                adjacency[point as usize] = neighbors;
                Ok(())
            },
        )?;
        Ok(phase0_identity_result(
            trace,
            policy,
            adjacency,
            Some(selection),
            leaf,
            spill,
            reduce,
        ))
    }

    fn phase0_identity_expected_off_selection() -> Vec<Vec<Phase0IdentitySelection>> {
        vec![
            vec![Phase0IdentitySelection {
                target: 1,
                hash: 0,
                dist_bits: 1.0f32.to_bits(),
                mandatory: false,
                overflow: false,
                evidence: None,
            }],
            vec![
                Phase0IdentitySelection {
                    target: 2,
                    hash: 0,
                    dist_bits: 1.5f32.to_bits(),
                    mandatory: false,
                    overflow: false,
                    evidence: None,
                },
                Phase0IdentitySelection {
                    target: 0,
                    hash: 3,
                    dist_bits: 1.0f32.to_bits(),
                    mandatory: false,
                    overflow: false,
                    evidence: None,
                },
            ],
            vec![Phase0IdentitySelection {
                target: 1,
                hash: 3,
                dist_bits: 1.5f32.to_bits(),
                mandatory: false,
                overflow: false,
                evidence: None,
            }],
        ]
    }

    fn phase0_identity_expected_report_selection() -> Vec<Vec<Phase0IdentitySelection>> {
        vec![
            vec![Phase0IdentitySelection {
                target: 1,
                hash: 0,
                dist_bits: 1.0f32.to_bits(),
                mandatory: false,
                overflow: false,
                evidence: Some(HashPruneEvidenceClass::BidirectionalDirectRank0),
            }],
            vec![
                Phase0IdentitySelection {
                    target: 2,
                    hash: 0,
                    dist_bits: 1.5f32.to_bits(),
                    mandatory: false,
                    overflow: false,
                    evidence: Some(HashPruneEvidenceClass::BidirectionalDirectRank0),
                },
                Phase0IdentitySelection {
                    target: 0,
                    hash: 3,
                    dist_bits: 1.0f32.to_bits(),
                    mandatory: false,
                    overflow: false,
                    evidence: Some(HashPruneEvidenceClass::Bidirectional),
                },
            ],
            vec![Phase0IdentitySelection {
                target: 1,
                hash: 3,
                dist_bits: 1.5f32.to_bits(),
                mandatory: false,
                overflow: false,
                evidence: Some(HashPruneEvidenceClass::BidirectionalDirectRank0),
            }],
        ]
    }

    fn phase0_identity_expected_leaf_shape(
        backend: Phase0IdentityBackend,
    ) -> Phase0IdentityProfileShape {
        Phase0IdentityProfileShape {
            leaves: 2,
            points: 6,
            sketch_rows_prefetched: 6,
            gpu_h2d_bytes: 0,
            gpu_d2h_bytes: 0,
            edge_sink_edges: 20,
            edge_sink_flushes: match backend {
                Phase0IdentityBackend::Resident | Phase0IdentityBackend::Spill => 2,
                Phase0IdentityBackend::Hybrid => 4,
            },
            compact_lune_witness_slots: 0,
        }
    }

    fn phase0_identity_expected_report_counters(
        backend: Phase0IdentityBackend,
    ) -> Phase0IdentityEvidenceCounters {
        let (spill_records_avoided, spill_bytes_avoided) = match backend {
            Phase0IdentityBackend::Resident => (0, 0),
            Phase0IdentityBackend::Spill => (8, 128),
            Phase0IdentityBackend::Hybrid => (2, 32),
        };

        // Per frozen batch, the five valid top-k entries yield reciprocal checks/hits 5/4,
        // support once/twice 2/4, class buckets [1, 0, 1, 2, 0, 6], and coalesce counts
        // before/after/removed 10/6/4. The identical replay doubles those trace-derived counts;
        // only spill avoidance changes with backend ownership.
        Phase0IdentityEvidenceCounters {
            reciprocal_checks: 10,
            reciprocal_hits: 8,
            support_once_candidates: 4,
            support_twice_candidates: 8,
            class_counts: [2, 0, 2, 4, 0, 12],
            candidates_before_coalesce: 20,
            candidates_after_coalesce: 12,
            duplicate_records_removed: 8,
            unannotated_candidates: 0,
            qualified_collision_losers: 0,
            overflow_inserts: 0,
            overflow_replacements: 0,
            overflow_rejects: 0,
            overflow_evictions_by_new_hash: 0,
            rows_with_overflow: 0,
            spill_records_avoided,
            spill_bytes_avoided,
        }
    }

    fn assert_phase0_identity_mode_oracle(
        label: &str,
        run: &Phase0IdentityRun,
        expected_policy: HashPruneEvidencePolicy,
        expected_counters: Phase0IdentityEvidenceCounters,
    ) {
        assert_eq!(run.policy, expected_policy, "{label}");
        assert_eq!(run.policy.mode(), expected_policy.mode(), "{label}");
        assert_eq!(run.policy.overflow_budget(), 0, "{label}");

        let actual_counters = phase0_identity_evidence_counters(run);
        assert_eq!(actual_counters, expected_counters, "{label}");
        if expected_policy.annotates() {
            assert_eq!(
                actual_counters.candidates_before_coalesce,
                actual_counters
                    .candidates_after_coalesce
                    .saturating_add(actual_counters.duplicate_records_removed),
                "{label} violated before = after + duplicate_removed"
            );
        }
    }

    fn assert_phase0_identity_backend_shape(
        backend: Phase0IdentityBackend,
        label: &str,
        run: &Phase0IdentityRun,
    ) {
        match backend {
            Phase0IdentityBackend::Resident => {
                assert_eq!(run.spill.segment_count, 0, "{label}");
                assert_eq!(run.reduce.segments_scanned, 0, "{label}");
            }
            Phase0IdentityBackend::Spill => {
                assert_eq!(run.spill.record_bytes, 16, "{label}");
                assert!(run.spill.shard_groups > 1, "{label}");
                assert!(
                    run.spill.segment_count > run.spill.shard_groups
                        || run.spill.max_segments_per_shard >= 2,
                    "{label} must exercise multi-segment replay"
                );
                assert_eq!(run.reduce.shard_groups, run.spill.shard_groups, "{label}");
                assert_eq!(
                    run.reduce.segments_scanned, run.spill.segment_count,
                    "{label}"
                );
                assert_eq!(
                    run.reduce.max_segments_in_group, run.spill.max_segments_per_shard,
                    "{label}"
                );
            }
            Phase0IdentityBackend::Hybrid => {
                assert_eq!(run.spill.record_bytes, 16, "{label}");
                assert_eq!(run.spill.shard_groups, 1, "{label}");
                assert!(
                    run.spill.segment_count > run.spill.shard_groups
                        || run.spill.max_segments_per_shard >= 2,
                    "{label} must exercise multi-segment tail replay"
                );
                assert_eq!(run.reduce.shard_groups, run.spill.shard_groups, "{label}");
                assert_eq!(
                    run.reduce.segments_scanned, run.spill.segment_count,
                    "{label}"
                );
                assert_eq!(
                    run.reduce.max_segments_in_group, run.spill.max_segments_per_shard,
                    "{label}"
                );
            }
        }
    }

    fn run_hashprune_phase0_same_compact_trace_identity_matrix() {
        let trace = phase0_identity_compact_trace();
        let sketches = phase0_identity_sketches();
        let mut params = PiPNNParams {
            leaf_knn: 2,
            l_max: PHASE0_IDENTITY_L_MAX,
            ..PiPNNParams::default()
        };
        params.m_hash_bits = 2;
        params.oom_spill_cache_bytes = size_of::<PendingEdge>();

        let expected_adjacency = vec![vec![1], vec![2, 0], vec![1]];
        let mut id_sorted_adjacency = expected_adjacency.clone();
        for neighbors in &mut id_sorted_adjacency {
            neighbors.sort_unstable();
        }
        assert_ne!(
            id_sorted_adjacency, expected_adjacency,
            "fixture must expose neighbor-ID sorting as an ordered-adjacency regression"
        );
        assert_ne!(
            phase0_identity_canonical_ordered_adjacency_sha256(
                &id_sorted_adjacency,
                PHASE0_IDENTITY_L_MAX,
            ),
            PHASE0_IDENTITY_SHA256,
            "neighbor-ID sorting must change the canonical ordered-adjacency digest"
        );
        let expected_trace_shape = Phase0IdentityTraceShape {
            batch_count: 2,
            rows: 6,
            row_k_slots: 12,
            point_ids: 6,
            topk_ids: 12,
            topk_dists: 12,
            estimated_dtoh_bytes: 2 * 3 * 2 * (size_of::<u32>() + size_of::<f32>()),
        };
        let expected_off_selection = phase0_identity_expected_off_selection();
        let expected_report_selection = phase0_identity_expected_report_selection();
        let cases: [(Phase0IdentityBackend, Phase0IdentityRunner); 3] = [
            (
                Phase0IdentityBackend::Resident,
                phase0_identity_run_resident,
            ),
            (Phase0IdentityBackend::Spill, phase0_identity_run_spill),
            (Phase0IdentityBackend::Hybrid, phase0_identity_run_hybrid),
        ];
        let mut cross_backend_off = None;
        let mut cross_backend_report = None;

        for (backend, runner) in cases {
            let off_label = format!("{}/off", backend.label());
            let report_label = format!("{}/report", backend.label());
            let off = runner(&trace, &sketches, &params, HashPruneEvidencePolicy::off())
                .unwrap_or_else(|error| panic!("{off_label} failed: {error}"));
            let report = runner(
                &trace,
                &sketches,
                &params,
                HashPruneEvidencePolicy::report(),
            )
            .unwrap_or_else(|error| panic!("{report_label} failed: {error}"));

            let expected_leaf_shape = phase0_identity_expected_leaf_shape(backend);
            for (label, run) in [(&off_label, &off), (&report_label, &report)] {
                assert_eq!(run.adjacency, expected_adjacency, "{label}");
                assert_eq!(run.digest, PHASE0_IDENTITY_SHA256, "{label}");
                assert_eq!(run.trace_shape, expected_trace_shape, "{label}");
                assert_eq!(
                    phase0_identity_profile_shape(&run.leaf),
                    expected_leaf_shape,
                    "{label}"
                );
                assert_phase0_identity_backend_shape(backend, label, run);
            }

            assert_eq!(
                off.adjacency,
                report.adjacency,
                "{} off/report",
                backend.label()
            );
            assert_eq!(off.digest, report.digest, "{} off/report", backend.label());
            assert_eq!(
                off.trace_shape,
                report.trace_shape,
                "{} payload",
                backend.label()
            );
            assert_phase0_identity_mode_oracle(
                &off_label,
                &off,
                HashPruneEvidencePolicy::off(),
                Phase0IdentityEvidenceCounters::default(),
            );
            assert_phase0_identity_mode_oracle(
                &report_label,
                &report,
                HashPruneEvidencePolicy::report(),
                phase0_identity_expected_report_counters(backend),
            );

            let expected_off_backend_selection = match backend {
                Phase0IdentityBackend::Resident => Some(expected_off_selection.clone()),
                Phase0IdentityBackend::Spill => None,
                Phase0IdentityBackend::Hybrid => Some(expected_off_selection[..2].to_vec()),
            };
            let expected_report_backend_selection = match backend {
                Phase0IdentityBackend::Resident => Some(expected_report_selection.clone()),
                Phase0IdentityBackend::Spill => None,
                Phase0IdentityBackend::Hybrid => Some(expected_report_selection[..2].to_vec()),
            };
            assert_eq!(
                off.resident_selection.as_ref(),
                expected_off_backend_selection.as_ref(),
                "{off_label}"
            );
            assert_eq!(
                report.resident_selection.as_ref(),
                expected_report_backend_selection.as_ref(),
                "{report_label}"
            );

            if let Some((adjacency, digest)) = &cross_backend_off {
                assert_eq!(
                    &off.adjacency, adjacency,
                    "{off_label} cross-backend adjacency"
                );
                assert_eq!(&off.digest, digest, "{off_label} cross-backend digest");
            } else {
                cross_backend_off = Some((off.adjacency.clone(), off.digest.clone()));
            }
            if let Some((adjacency, digest)) = &cross_backend_report {
                assert_eq!(
                    &report.adjacency, adjacency,
                    "{report_label} cross-backend adjacency"
                );
                assert_eq!(
                    &report.digest, digest,
                    "{report_label} cross-backend digest"
                );
            } else {
                cross_backend_report = Some((report.adjacency.clone(), report.digest.clone()));
            }
        }
    }

    #[test]
    fn hashprune_phase0_same_compact_trace_off_report_identity_across_resident_spill_hybrid() {
        run_hashprune_phase0_same_compact_trace_identity_matrix();
    }

    #[test]
    fn oom_memory_plan_accounts_rabitq_sidecar_as_optional_resident_admission() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 200 * 1024 * 1024 * 1024;
        params.gpu_subtree_rabitq_plus_mode = PiPNNGpuSubtreeRabitqPlusMode::Build;
        params.gpu_subtree_rabitq_plus_sidecar_mode = PiPNNGpuSubtreeRabitqPlusSidecarMode::Require;
        params.gpu_subtree_rabitq_plus_sidecar_prefix = Some(PathBuf::from("/tmp/wiki35m.rabitq8"));

        let rows = 35_000_000;
        let dim = 1024;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        let expected_sidecar_bytes = super::estimate_dense_i8_sidecar_resident_bytes(
            rows,
            dim,
            super::RABITQ_DEFAULT_K_ALIGN,
            super::RABITQ_PLUS_SIDECAR_META_ROW_BYTES,
        );

        assert_eq!(
            plan.sidecar_resident_kind,
            super::OomSidecarResidentKind::RabitqPlus
        );
        assert_eq!(plan.sidecar_resident_required_bytes, expected_sidecar_bytes);
        assert_eq!(
            plan.sidecar_resident_budget_bytes,
            plan.resident_subtree_budget_bytes
        );
        assert!(plan.sidecar_resident_admissible);
        assert_eq!(
            plan.required_resident_bytes,
            rows * params.m_hash_bits * size_of::<f32>()
        );
        assert_eq!(
            plan.optional_resident_budget_bytes,
            plan.resident_subtree_budget_bytes
        );
        assert_eq!(
            plan.optional_resident_admitted_bytes,
            expected_sidecar_bytes
        );
        assert_eq!(plan.sidecar_resident_admitted_rows, rows);
        assert_eq!(plan.optional_resident_unassigned_bytes, 0);
        assert_eq!(
            plan.transient_pipeline_budget_bytes,
            plan.scratch_budget_bytes
        );
        assert_eq!(
            plan.d1plus_working_set_budget_bytes,
            plan.required_resident_bytes
                + plan.reservoir_budget_bytes
                + plan.optional_resident_admitted_bytes
        );
    }

    #[test]
    fn oom_memory_plan_stops_reservoir_admission_at_full_payload() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 200 * 1024 * 1024 * 1024;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(PathBuf::from("/tmp/wiki10m.sq8"));

        let rows = 10_000_000;
        let dim = 1024;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        let full_reservoir_bytes = super::estimate_reservoir_bytes(rows, params.l_max);

        assert_eq!(
            plan.sidecar_resident_kind,
            super::OomSidecarResidentKind::SqWmma
        );
        assert_eq!(plan.sidecar_resident_admitted_rows, rows);
        assert_eq!(
            plan.optional_resident_admitted_bytes,
            plan.sidecar_resident_required_bytes
        );
        assert_eq!(plan.optional_resident_unassigned_bytes, 0);
        assert_eq!(plan.reservoir_budget_bytes, full_reservoir_bytes);
        assert!(plan.io_headroom_bytes > params.effective_oom_memory_budget_bytes() * 5 / 100);
    }

    #[test]
    fn oom_memory_plan_admits_full_sq_sidecar_before_reservoir() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 150 * 1024 * 1024 * 1024;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(PathBuf::from("/tmp/laion100m.sq8"));

        let rows = 100_000_000;
        let dim = 512;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        let full_sidecar_bytes = super::estimate_dense_i8_sidecar_resident_bytes(
            rows,
            dim,
            super::SQ_INT8_DEFAULT_K_ALIGN,
            super::SQ_WMMA_SIDECAR_META_ROW_BYTES,
        );
        let total = params.effective_oom_memory_budget_bytes();
        let expected_available = total
            - total * 15 / 100
            - total * 5 / 100
            - rows * params.m_hash_bits * size_of::<f32>();
        let expected_reservoir_budget = expected_available
            .saturating_sub(full_sidecar_bytes)
            .min(super::estimate_reservoir_bytes(rows, params.l_max));

        assert_eq!(plan.sidecar_resident_admitted_rows, rows);
        assert_eq!(plan.sidecar_resident_budget_bytes, full_sidecar_bytes);
        assert_eq!(plan.optional_resident_admitted_bytes, full_sidecar_bytes);
        assert_eq!(plan.reservoir_budget_bytes, expected_reservoir_budget);
    }

    #[test]
    fn oom_memory_plan_applies_reservoir_share_after_sidecar_admission() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 50 * 1024 * 1024 * 1024;
        params.oom_reservoir_budget_share = 0.5;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(PathBuf::from("/tmp/wiki35m.sq8"));

        let rows = 35_000_000;
        let dim = 1024;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        let total = params.effective_oom_memory_budget_bytes();
        let expected_sketch_bytes = rows * params.m_hash_bits * size_of::<f32>();
        let expected_sidecar_budget = super::estimate_dense_i8_sidecar_resident_bytes(
            rows,
            dim,
            super::SQ_INT8_DEFAULT_K_ALIGN,
            super::SQ_WMMA_SIDECAR_META_ROW_BYTES,
        );
        let expected_available_after_sidecar = total
            - total * 15 / 100
            - total * 5 / 100
            - expected_sketch_bytes
            - expected_sidecar_budget;
        let expected_reservoir_budget = (super::estimate_reservoir_bytes(rows, params.l_max) / 2)
            .min(expected_available_after_sidecar);

        assert_eq!(plan.sidecar_resident_budget_bytes, expected_sidecar_budget);
        assert_eq!(plan.sidecar_resident_admitted_rows, rows);
        assert_eq!(plan.reservoir_budget_bytes, expected_reservoir_budget);
        assert_eq!(
            plan.transient_pipeline_budget_bytes,
            plan.scratch_budget_bytes
        );
    }

    #[test]
    fn oom_memory_plan_zero_reservoir_share_keeps_reservoir_at_zero() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 50 * 1024 * 1024 * 1024;
        params.oom_reservoir_budget_share = 0.0;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(PathBuf::from("/tmp/wiki35m.sq8"));

        let rows = 35_000_000;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, 1024);

        assert_eq!(plan.reservoir_budget_bytes, 0);
        assert_eq!(plan.sidecar_resident_admitted_rows, rows);
        assert!(
            plan.io_headroom_bytes > params.effective_oom_memory_budget_bytes() * 5 / 100,
            "unused sidecar budget should become headroom instead of flowing back to reservoirs"
        );
    }

    #[test]
    fn oom_memory_plan_sidecar_zero_cap_leaves_budget_for_reservoir() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 32 * 1024 * 1024 * 1024;
        params.oom_reservoir_budget_share = 1.0;
        params.oom_sidecar_resident_budget_cap_bytes = Some(0);
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(PathBuf::from("/tmp/wiki35m.sq8"));

        let rows = 35_000_000;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, 1024);
        let total = params.effective_oom_memory_budget_bytes();
        let expected_scratch = total * 15 / 100;
        let expected_io_headroom = total * 5 / 100;
        let expected_sketch_bytes = rows * params.m_hash_bits * size_of::<f32>();
        let expected_reservoir_budget =
            (total - expected_scratch - expected_io_headroom - expected_sketch_bytes)
                .min(super::estimate_reservoir_bytes(rows, params.l_max));

        assert_eq!(plan.sidecar_resident_budget_bytes, 0);
        assert_eq!(plan.sidecar_resident_admitted_rows, 0);
        assert!(!plan.sidecar_resident_admissible);
        assert_eq!(plan.optional_resident_admitted_bytes, 0);
        assert_eq!(plan.reservoir_budget_bytes, expected_reservoir_budget);
        assert!(plan.io_headroom_bytes >= expected_io_headroom);
    }

    #[test]
    fn oom_memory_plan_sidecar_cap_above_auto_budget_keeps_existing_plan() {
        let mut baseline_params = PiPNNParams::default();
        baseline_params.oom_memory_budget_bytes = 32 * 1024 * 1024 * 1024;
        baseline_params.oom_reservoir_budget_share = 1.0;
        baseline_params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        baseline_params.gpu_subtree_sq_wmma_sidecar_mode =
            PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        baseline_params.gpu_subtree_sq_wmma_sidecar_prefix =
            Some(PathBuf::from("/tmp/wiki35m.sq8"));

        let rows = 35_000_000;
        let baseline = super::OomMemoryPlan::from_params_for_dataset(&baseline_params, rows, 1024);
        let mut capped_params = baseline_params.clone();
        capped_params.oom_sidecar_resident_budget_cap_bytes =
            Some(baseline.sidecar_resident_budget_bytes + 1024);
        let capped = super::OomMemoryPlan::from_params_for_dataset(&capped_params, rows, 1024);

        assert_eq!(
            capped.sidecar_resident_budget_bytes,
            baseline.sidecar_resident_budget_bytes
        );
        assert_eq!(
            capped.sidecar_resident_admitted_rows,
            baseline.sidecar_resident_admitted_rows
        );
        assert_eq!(
            capped.reservoir_budget_bytes,
            baseline.reservoir_budget_bytes
        );
    }

    #[test]
    fn oom_memory_plan_partially_admits_rabitq_sidecar_when_resident_budget_is_tight() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 40 * 1024 * 1024 * 1024;
        params.gpu_subtree_rabitq_plus_mode = PiPNNGpuSubtreeRabitqPlusMode::Build;
        params.gpu_subtree_rabitq_plus_sidecar_mode = PiPNNGpuSubtreeRabitqPlusSidecarMode::Require;
        params.gpu_subtree_rabitq_plus_sidecar_prefix = Some(PathBuf::from("/tmp/wiki35m.rabitq8"));

        let rows = 35_000_000;
        let plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, 1024);

        assert_eq!(
            plan.sidecar_resident_kind,
            super::OomSidecarResidentKind::RabitqPlus
        );
        assert!(plan.sidecar_resident_required_bytes > plan.sidecar_resident_budget_bytes);
        assert!(plan.sidecar_resident_admissible);
        assert!(plan.sidecar_resident_admitted_rows > 0);
        assert!(plan.sidecar_resident_admitted_rows < rows);
        assert!(plan.optional_resident_admitted_bytes > 0);
        assert!(plan.optional_resident_admitted_bytes < plan.sidecar_resident_required_bytes);
        assert_eq!(plan.optional_resident_unassigned_bytes, 0);
        assert_eq!(
            plan.transient_pipeline_budget_bytes,
            plan.scratch_budget_bytes
        );
        assert_eq!(
            plan.d1plus_working_set_budget_bytes,
            plan.required_resident_bytes
                + plan.reservoir_budget_bytes
                + plan.optional_resident_admitted_bytes
        );
    }

    #[test]
    fn oom_sq_wmma_runtime_sidecar_uses_direct_io_when_residency_is_unavailable() {
        let rows = 5;
        let dim = 6;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("tiny.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();
        super::dco_sq_int8::create_sq_wmma_sidecar_from_point_store(
            &store,
            &prefix,
            source_bytes as u64,
        )
        .unwrap();

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(prefix);
        let mut memory_plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        memory_plan.sidecar_resident_admissible = false;

        let sidecar =
            super::maybe_open_sq_wmma_runtime_sidecar(&params, &store, rows, &memory_plan)
                .unwrap()
                .expect("SQ+WMMA runtime sidecar should be opened");

        assert!(!sidecar.is_resident());
        assert!(sidecar.uses_direct_io());
        let gathered = sidecar.gather(&[3_u32, 1, 0]).unwrap();
        assert!(!gathered.resident_source);
        assert_eq!(gathered.global_ids, [3_u32, 1, 0]);
    }

    #[test]
    fn sq_gpu_d0_runtime_sidecar_uses_disk_fallback_without_resident_admission() {
        let rows = 5;
        let dim = 6;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("disk-fallback.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();
        super::dco_sq_int8::create_sq_wmma_sidecar_from_point_store(
            &store,
            &prefix,
            source_bytes as u64,
        )
        .unwrap();

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(prefix);
        let mut memory_plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        memory_plan.sidecar_resident_admissible = false;
        memory_plan.sidecar_resident_admitted_rows = 0;

        let sidecar =
            super::maybe_open_sq_wmma_runtime_sidecar(&params, &store, rows, &memory_plan)
                .unwrap()
                .expect("disk-backed SQ sidecar");
        assert!(!sidecar.is_resident());
        assert!(sidecar.uses_direct_io());
        assert_eq!(sidecar.gather(&[3, 1, 0]).unwrap().file_rows, 3);
    }

    #[test]
    fn sq_gpu_d0_runtime_sidecar_preloads_admitted_prefix_from_disk() {
        let rows = 6;
        let dim = 6;
        let resident_rows = 3;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("post-d0-preload.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();
        super::dco_sq_int8::create_sq_wmma_sidecar_from_point_store(
            &store,
            &prefix,
            source_bytes as u64,
        )
        .unwrap();

        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(prefix);
        let mut memory_plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        memory_plan.sidecar_resident_kind = super::OomSidecarResidentKind::SqWmma;
        memory_plan.sidecar_resident_admissible = true;
        memory_plan.sidecar_resident_admitted_rows = resident_rows;

        let sidecar =
            super::maybe_open_sq_wmma_runtime_sidecar(&params, &store, rows, &memory_plan)
                .unwrap()
                .expect("post-D0 preloaded SQ sidecar");
        assert!(sidecar.is_resident());
        assert!(!sidecar.is_fully_resident());
        assert_eq!(sidecar.resident_rows(), resident_rows);
        assert_eq!(sidecar.gather(&[0, 4]).unwrap().file_rows, 1);
    }

    #[test]
    fn sq_wmma_runtime_sidecar_consumes_created_resident_prefix_cache() {
        let rows = 6;
        let dim = 6;
        let resident_rows = 3;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("created-cache.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();
        let artifact =
            super::dco_sq_int8::create_sq_wmma_sidecar_from_point_store_with_resident_prefix(
                &store,
                &prefix,
                source_bytes as u64,
                resident_rows,
            )
            .unwrap();
        super::dco_sq_int8::cache_sq_wmma_created_sidecar(
            artifact
                .runtime_sidecar
                .expect("creation should provide a resident prefix sidecar"),
        );

        let mut params = PiPNNParams::default();
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
        params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Require;
        params.gpu_subtree_sq_wmma_sidecar_prefix = Some(prefix.clone());
        let mut memory_plan = super::OomMemoryPlan::from_params_for_dataset(&params, rows, dim);
        memory_plan.sidecar_resident_kind = super::OomSidecarResidentKind::SqWmma;
        memory_plan.sidecar_resident_admissible = true;
        memory_plan.sidecar_resident_admitted_rows = resident_rows;

        let sidecar =
            super::maybe_open_sq_wmma_runtime_sidecar(&params, &store, rows, &memory_plan)
                .unwrap()
                .expect("SQ+WMMA runtime sidecar should be opened");

        assert!(sidecar.is_resident());
        assert!(!sidecar.is_fully_resident());
        assert_eq!(sidecar.resident_rows(), resident_rows);
        assert_eq!(sidecar.preload_wall(), Duration::ZERO);
        assert!(
            super::dco_sq_int8::take_cached_sq_wmma_created_sidecar(&prefix, rows, dim)
                .unwrap()
                .is_none(),
            "runtime open should consume the just-created sidecar exactly once"
        );
    }

    #[test]
    fn sq_wmma_fbin_resident_prefix_wrapper_populates_runtime_cache() {
        let rows = 6;
        let dim = 6;
        let resident_rows = 3;
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("tiny.fbin");
        let prefix = dir.path().join("tiny.sq8");
        write_test_fbin(&data_path, rows, dim);

        let report = super::create_sq_wmma_sidecar_from_fbin_with_resident_prefix(
            &data_path,
            &prefix,
            PiPNNOomIoMode::Buffered,
            resident_rows,
        )
        .unwrap();

        assert_eq!(report.rows, rows);
        let sidecar = super::dco_sq_int8::take_cached_sq_wmma_created_sidecar(&prefix, rows, dim)
            .unwrap()
            .expect("fbin wrapper should populate the one-shot runtime sidecar cache");
        assert_eq!(sidecar.resident_rows(), resident_rows);
        assert_eq!(sidecar.preload_wall(), Duration::ZERO);
    }

    #[test]
    fn sq_wmma_u8bin_native_sq4_wrapper_creates_packed_sidecar() {
        let rows = 5;
        let dim = 7;
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("tiny.u8bin");
        let prefix = dir.path().join("tiny.u8.sq4");
        write_test_u8bin(&data_path, rows, dim);

        let report = super::create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4(
            &data_path,
            &prefix,
            4,
            PiPNNOomIoMode::Buffered,
            true,
        )
        .unwrap();

        assert_eq!(report.rows, rows);
        assert_eq!(report.dim, dim);
        assert_eq!(report.bits, 4);
        assert_eq!(report.code_layout, "packed-s4");
        assert_eq!(report.code_row_bytes, report.padded_dim / 2);
        assert_eq!(report.codes_bytes, rows * report.code_row_bytes);

        let sidecar =
            super::dco_sq_int8::SqWmmaSidecar::open_validated_with_bits(&prefix, rows, dim, 4)
                .unwrap();
        assert_eq!(sidecar.header().code_layout.as_str(), "packed-s4");
        assert_eq!(sidecar.header().code_row_bytes, report.code_row_bytes);
    }

    #[test]
    fn sq_wmma_u8bin_sq8_wrapper_uses_native_centered_int8_without_rescaling() {
        let rows = 5;
        let dim = 7;
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("tiny.u8bin");
        let prefix = dir.path().join("tiny.u8.sq8");
        write_test_u8bin(&data_path, rows, dim);

        let report = super::create_sq_wmma_sidecar_from_u8bin_with_bits_native_int4(
            &data_path,
            &prefix,
            8,
            PiPNNOomIoMode::Buffered,
            false,
        )
        .unwrap();

        assert_eq!(report.rows, rows);
        assert_eq!(report.dim, dim);
        assert_eq!(report.bits, 8);
        assert_eq!(report.code_layout, "dense-i8");
        assert_eq!(report.code_row_bytes, report.padded_dim);
        assert_eq!(report.codes_bytes, rows * report.code_row_bytes);

        let sidecar =
            super::dco_sq_int8::SqWmmaSidecar::open_validated_with_bits(&prefix, rows, dim, 8)
                .unwrap();
        let ids = [0_u32, 4];
        let gathered = sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.code_layout.as_str(), "dense-i8");
        assert_eq!(gathered.scales, vec![1.0, 1.0]);

        for (slot, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            let mut expected_codes = vec![0_i8; gathered.padded_dim];
            let mut expected_norm = 0.0_f32;
            for axis in 0..dim {
                let raw = ((row * 17 + axis * 29) % 256) as u8;
                let code = raw as i16 - 128;
                expected_codes[axis] = code as i8;
                expected_norm += (code * code) as f32;
            }
            let got_codes = &gathered.codes[slot * gathered.code_row_bytes
                ..slot * gathered.code_row_bytes + gathered.padded_dim];
            assert_eq!(got_codes, expected_codes);
            assert_eq!(gathered.norms[slot], expected_norm);
        }

        let expected_l2: f32 = (0..dim)
            .map(|axis| {
                let lhs = ((ids[0] as usize * 17 + axis * 29) % 256) as f32;
                let rhs = ((ids[1] as usize * 17 + axis * 29) % 256) as f32;
                let delta = lhs - rhs;
                delta * delta
            })
            .sum();
        assert_eq!(gathered.approx_l2(0, 1), expected_l2);
    }

    #[test]
    fn one_gib_oom_budget_currently_clamps_leaf_budget_threads_to_two() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let workers = super::derive_oom_worker_count(12, &params, 768);
        assert_eq!(
            workers, 2,
            "this captures the current leaf-budget behavior that also explains the observed RBC workers=2 regression"
        );
    }

    #[test]
    fn oom_execution_threads_keep_full_pool_width_while_clamping_leaf_budget_threads() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let threads = super::derive_oom_execution_threads(12, &params, 768);
        assert_eq!(threads.pool_threads, 12);
        assert_eq!(threads.leaf_budget_threads, 2);
    }

    #[test]
    fn resident_prefix_planning_uses_leaf_budget_threads_not_pool_width() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let threads = super::derive_oom_execution_threads(12, &params, 768);
        let memory_plan = super::OomMemoryPlan::from_params(&params);
        let plan_with_leaf_budget = super::plan_resident_reservoir_prefix(
            5_000_000,
            threads.leaf_budget_threads,
            &params,
            768,
            0,
            100_000,
            memory_plan.reservoir_budget_bytes,
        );
        let plan_with_pool_width = super::plan_resident_reservoir_prefix(
            5_000_000,
            threads.pool_threads,
            &params,
            768,
            0,
            100_000,
            memory_plan.reservoir_budget_bytes,
        );

        assert!(
            plan_with_leaf_budget.resident_points >= plan_with_pool_width.resident_points,
            "resident planning should be based on leaf-budget scratch, not the wider RBC/root pool width"
        );
    }

    #[test]
    fn sq_staging_fused_sketch_candidate_allows_full_resident_reservoir_plan() {
        let num_points = 10_000usize;
        let mut params = PiPNNParams::default();
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0;
        let full_resident_plan = super::ResidentReservoirPlan {
            resident_points: num_points,
        };

        assert_eq!(full_resident_plan.resident_points, num_points);
        assert!(
            super::should_use_sq_staging_fused_sketch(true, false, &params),
            "full-resident reservoir builds should fuse HashPrune sketch computation into SQ GPU-D0 staging"
        );
    }

    #[test]
    fn derive_spill_shard_points_uses_memory_budget_when_spill_budget_is_unset() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.oom_spill_cache_bytes = 0;

        let shard_points = super::derive_spill_shard_points(
            1_000_000,
            &params,
            params.effective_oom_memory_budget_bytes(),
        );
        assert!(shard_points > 1);
        assert!(shard_points <= 1_000_000);
    }

    #[test]
    fn derive_spill_shard_points_keeps_actual_reservoir_slab_within_scratch_budget() {
        let mut params = PiPNNParams::default();
        params.l_max = 100;
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.oom_spill_cache_bytes = 0;
        let scratch_budget = params.oom_memory_budget_bytes / 8;

        let shard_points =
            super::derive_spill_shard_points(1_000_000, &params, params.oom_memory_budget_bytes);
        let actual_bytes_per_point = std::mem::size_of::<HashPruneReservoir>()
            + params.l_max * std::mem::size_of::<super::Slot>();

        assert!(
            shard_points.saturating_mul(actual_bytes_per_point) <= scratch_budget,
            "external reducer slab exceeds scratch budget: points={shard_points} bytes_per_point={actual_bytes_per_point} scratch_budget={scratch_budget}"
        );
    }

    #[test]
    fn derive_lrpf_digest_shard_points_caps_large_reduce_groups() {
        assert_eq!(
            super::derive_lrpf_digest_shard_points(10_000_000, 10_000_000),
            1_000_000
        );
        assert_eq!(
            super::derive_lrpf_digest_shard_points(100_000, 10_000_000),
            100_000
        );
        assert_eq!(
            super::derive_lrpf_digest_shard_points(10_000_000, 100_000),
            100_000
        );
    }

    #[test]
    fn compact_spill_digest_width_uses_multiplier_and_clamps_to_degree_window() {
        let mut params = PiPNNParams::default();
        params.l_max = 64;
        params.compact_spill_digest_enable = true;
        params.compact_spill_digest_width_multiplier = 2.0;
        assert_eq!(
            super::effective_compact_spill_digest_width(&params),
            Some(128)
        );

        params.compact_spill_digest_width_multiplier = 0.25;
        assert_eq!(
            super::effective_compact_spill_digest_width(&params),
            Some(64)
        );

        params.compact_spill_digest_width_multiplier = 10.0;
        assert_eq!(
            super::effective_compact_spill_digest_width(&params),
            Some(256)
        );

        params.compact_spill_digest_enable = false;
        params.compact_spill_digest_width_multiplier = 2.0;
        assert_eq!(super::effective_compact_spill_digest_width(&params), None);
    }

    #[test]
    fn resident_reservoir_plan_can_select_partial_prefix_when_budget_is_between_full_and_zero() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 4 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;
        let shard_points = 100_000;

        let plan = super::plan_resident_reservoir_prefix(
            10_000_000,
            8,
            &params,
            768,
            0,
            shard_points,
            super::OomMemoryPlan::from_params(&params).reservoir_budget_bytes,
        );

        assert!(
            plan.resident_points > 0,
            "expected some prefix to remain resident"
        );
        assert!(
            plan.resident_points < 10_000_000,
            "expected budget to be insufficient for full residency"
        );
        assert_eq!(
            plan.resident_points % shard_points,
            0,
            "resident prefix should be shard aligned"
        );
    }

    #[test]
    fn online_suffix_resident_prefix_is_not_rounded_away_by_external_shard_size() {
        let mut params = PiPNNParams::default();
        params.partitioner = PiPNNPartitionerKind::GpuSubtree;
        params.online_suffix_reduce_mode = super::PiPNNOnlineSuffixReduceMode::DirectFirst;
        let expected_resident_points = 12_345;
        let reservoir_budget_bytes =
            super::estimate_reservoir_bytes(expected_resident_points, params.l_max);

        let plan = super::plan_resident_reservoir_prefix(
            1_000_000,
            8,
            &params,
            768,
            0,
            100_000,
            reservoir_budget_bytes,
        );

        assert_eq!(plan.resident_points, expected_resident_points);
    }

    #[test]
    fn resident_reservoir_plan_keeps_all_rows_when_full_payload_fits_before_shard_alignment() {
        let params = PiPNNParams::default();
        let num_points = 100_000_000;
        let plan = super::plan_resident_reservoir_prefix(
            num_points,
            8,
            &params,
            512,
            4_800_000_000,
            47_934_902,
            super::estimate_reservoir_bytes(num_points, params.l_max),
        );

        assert_eq!(plan.resident_points, num_points);
    }

    #[test]
    fn oom_build_hybrid_resident_prefix_emits_each_point_once_and_keeps_spill_artifacts() {
        std::thread::Builder::new()
            .name("oom-hybrid-spill-test".to_string())
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let num_points = 5_000usize;
                let dim = 8usize;
                let dataset = build_test_dataset(num_points, dim);
                let temp = tempdir().unwrap();
                let mut params = PiPNNParams::default();
                params.oom_enable = true;
                params.oom_keep_artifacts = true;
                params.final_prune = false;
                params.l_max = 4_096;
                params.max_full_matrix_leaf_size = 256;
                params.c_min = 16;
                params.c_max = 32;
                params.max_depth = 6;
                params.max_leaders = 32;
                params.fanout_top = 4;
                params.fanout_second = 2;
                params.oom_memory_budget_bytes = 1280 * 1024 * 1024;
                params.oom_spill_cache_bytes = 16 * 1024 * 1024;
                params.oom_temp_dir = PathBuf::from(temp.path());

                let budget = params.effective_oom_memory_budget_bytes();
                let shard_points = super::derive_spill_shard_points(num_points, &params, budget);
                let effective_threads = super::derive_oom_worker_count(1, &params, dim);
                let resident_plan = super::plan_resident_reservoir_prefix(
                    num_points,
                    effective_threads,
                    &params,
                    dim,
                    0,
                    shard_points,
                    super::OomMemoryPlan::from_params(&params).reservoir_budget_bytes,
                );
                assert!(
                    resident_plan.resident_points > 0 && resident_plan.resident_points < num_points,
                    "test configuration should force hybrid resident+spill mode, got resident_points={} num_points={} shard_points={}",
                    resident_plan.resident_points,
                    num_points,
                    shard_points
                );

                let mut seen = vec![0usize; num_points];
                super::build_pipnn_graph(&dataset, num_points, 1, &params, |point, _neighbors| {
                    seen[point as usize] += 1;
                    Ok(())
                })
                .unwrap();

                assert!(
                    seen.iter().all(|count| *count == 1),
                    "expected every point to be emitted exactly once"
                );
                let entries: Vec<_> = fs::read_dir(temp.path())
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .collect();
                assert!(
                    entries.iter().any(|name| name.contains("candidate")),
                    "expected hybrid mode to keep spill artifacts for the tail, found {entries:?}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
