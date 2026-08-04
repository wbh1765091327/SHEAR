use std::cell::RefCell;
use std::cmp;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_queue::ArrayQueue;
use indicatif::ProgressBar;
use parking_lot::Mutex;
use rayon::ScopeFifo;

use super::compute_backend::PiPNNComputeBackend;
use super::hash_prune::SketchAccessor;
use super::leaf_build::{
    CompactTopKBatch, LeafProfile, LeafScratch, PendingEdgeSink, compute_leaf_topk_l2_cpu_shadow,
    flush_compact_topk_batch, process_leaf_parallel_large_profiled_with_backend_and_sink,
    process_leaf_profiled_with_sink,
};
use super::params::{PiPNNComputeBackendKind, PiPNNParams, PiPNNPipelineComputeMode};
use super::point_store::PointStore;
use crate::common::{AnnError, AnnResult, Metric};

const TARGET_CUDA_BATCH_VECTOR_BYTES: usize = 64 * 1024 * 1024;
const MAX_CUDA_BATCH_LEAVES: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerBudget {
    pub worker_count: usize,
    pub leaf_backlog_capacity: usize,
    pub large_leaf_min_size: usize,
    pub leaf_block_rows: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LeafDescriptor {
    pub(crate) leaf_id: u64,
    pub(crate) point_ids: Vec<u32>,
    pub(crate) leaf_size: usize,
    pub(crate) depth: usize,
    pub(crate) run_id: usize,
    pub(crate) window_start: usize,
    pub(crate) estimated_vector_bytes: usize,
    pub(crate) estimated_topk_bytes: usize,
}

impl LeafDescriptor {
    fn from_leaf(leaf_id: u64, leaf: Vec<u32>, dim: usize, k: usize) -> Self {
        let leaf_size = leaf.len();
        let estimated_topk_bytes = leaf_size
            .saturating_mul(k)
            .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>());
        Self {
            leaf_id,
            point_ids: leaf,
            leaf_size,
            depth: 0,
            run_id: 0,
            window_start: 0,
            estimated_vector_bytes: leaf_size
                .saturating_mul(dim)
                .saturating_mul(std::mem::size_of::<f32>()),
            estimated_topk_bytes: estimated_topk_bytes.saturating_mul(2),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StagedLeafBatch {
    pub(crate) leaf_ids: Vec<u64>,
    pub(crate) descriptors: Vec<LeafDescriptor>,
    pub(crate) host_vector_buffer: Option<Vec<f32>>,
    pub(crate) sketch_handle: Option<u64>,
    pub(crate) estimated_htod_bytes: usize,
    pub(crate) estimated_dtoh_bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EdgeFlushBatch {
    pub(crate) leaf_id: u64,
    pub(crate) estimated_dtoh_bytes: usize,
    pub(crate) topk: Option<CompactTopKBatch>,
}

impl SchedulerBudget {
    pub const DEFAULT_LEAF_BLOCK_ROWS: usize = 1024;
    const MAX_LEAF_BACKLOG_CAPACITY: usize = 512;
    const MIN_LEAF_BACKLOG_CAPACITY: usize = 1;
    const TARGET_LEAF_BACKLOG_BYTES: usize = 64 * 1024 * 1024;

    #[inline]
    pub fn for_worker_count(worker_count: usize, kernel_safe_leaf_size: usize) -> Self {
        Self::for_memory_budget(worker_count, kernel_safe_leaf_size, usize::MAX)
    }

    #[inline]
    pub fn for_memory_budget(
        worker_count: usize,
        kernel_safe_leaf_size: usize,
        memory_budget_bytes: usize,
    ) -> Self {
        let worker_count = worker_count.max(1);
        let estimated_leaf_bytes = kernel_safe_leaf_size
            .max(1)
            .saturating_mul(std::mem::size_of::<u32>())
            .max(1);
        let budget_backlog_capacity = if memory_budget_bytes == usize::MAX {
            Self::MAX_LEAF_BACKLOG_CAPACITY
        } else {
            (memory_budget_bytes.min(Self::TARGET_LEAF_BACKLOG_BYTES) / estimated_leaf_bytes).clamp(
                Self::MIN_LEAF_BACKLOG_CAPACITY,
                Self::MAX_LEAF_BACKLOG_CAPACITY,
            )
        };
        let leaf_backlog_capacity = worker_count
            .saturating_mul(8)
            .clamp(Self::MIN_LEAF_BACKLOG_CAPACITY, budget_backlog_capacity);
        let large_leaf_min_size = cmp::max(4096, kernel_safe_leaf_size.saturating_mul(2));

        Self {
            worker_count,
            leaf_backlog_capacity,
            large_leaf_min_size,
            leaf_block_rows: Self::DEFAULT_LEAF_BLOCK_ROWS,
        }
    }
}

pub trait LeafEmitter: Sync {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()>;
}

impl LeafEmitter for crossbeam::channel::Sender<Vec<u32>> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.send(leaf)
            .map_err(|err| AnnError::log_index_error(format!("Failed to emit PiPNN leaf: {err}")))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerTelemetry {
    pub peak_inflight_leaf_tasks: usize,
    pub peak_leaf_backlog: usize,
    pub inline_leaf_fallbacks: usize,
    pub producer_help_drains: usize,
    pub producer_help_yields: usize,
    pub large_leaf_count: usize,
    pub large_leaf_block_tasks: usize,
    pub producer_wait_ms: u64,
    pub staging_wait_ms: u64,
    pub compute_wait_ms: u64,
    pub edge_wait_ms: u64,
    pub descriptor_receive_wall_ms: u64,
    pub descriptor_batch_extend_wall_ms: u64,
    pub stage_wall_ms: u64,
    pub compute_flush_wall_ms: u64,
    pub topk_queue_roundtrip_wall_ms: u64,
    pub edge_queue_roundtrip_wall_ms: u64,
    pub edge_flush_wall_ms: u64,
    pub producer_blocked_ms: u64,
    pub compute_blocked_ms: u64,
    pub edge_blocked_ms: u64,
    pub peak_leaf_descriptor_queue: usize,
    pub peak_staged_queue: usize,
    pub peak_topk_queue: usize,
    pub peak_edge_queue: usize,
    pub descriptor_queue_empty_count: usize,
    pub cuda_batch_count: usize,
    pub cuda_batch_leaves: usize,
    pub cuda_batch_rows: usize,
    pub cuda_batch_target_bytes: usize,
    pub cuda_batch_actual_bytes: usize,
    pub cuda_batch_underfilled_count: usize,
    pub cpu_shadow_unsupported_fallbacks: usize,
    pub gpu_workspace_pressure_events: usize,
    pub gpu_cpu_fallbacks: usize,
    pub post_producer_leaf_drain_ms: u64,
}

#[cfg(test)]
pub trait SchedulerTestObserver: Sync {
    fn on_leaf_start(&self, leaf: &[u32]);
}

#[derive(Clone, Copy)]
pub struct LeafTaskContext<'a> {
    pub compute_backend: &'a dyn PiPNNComputeBackend,
    pub dataset: &'a dyn PointStore,
    pub metric: Metric,
    pub sketches: &'a dyn SketchAccessor,
    pub params: &'a PiPNNParams,
    pub edge_sink: &'a dyn PendingEdgeSink,
    pub detailed_profiling: bool,
    #[cfg(test)]
    pub observer: Option<&'a dyn SchedulerTestObserver>,
}

pub struct PipnnScheduler<'a> {
    budget: SchedulerBudget,
    context: LeafTaskContext<'a>,
    progress: &'a ProgressBar,
    leaf_backlog: ArrayQueue<Vec<u32>>,
    outstanding_leaf_tasks: AtomicUsize,
    inflight_leaf_tasks: AtomicUsize,
    peak_inflight_leaf_tasks: AtomicUsize,
    peak_leaf_backlog: AtomicUsize,
    active_leaf_drainers: AtomicUsize,
    producer_active: AtomicBool,
    inline_leaf_fallbacks: AtomicUsize,
    producer_help_drains: AtomicUsize,
    producer_help_yields: AtomicUsize,
    large_leaf_count: AtomicUsize,
    large_leaf_block_tasks: AtomicUsize,
    first_error: Mutex<Option<String>>,
    leaf_profile: Mutex<LeafProfile>,
}

pub struct PipelineRuntimeConfig {
    pub leaf_descriptor_queue_capacity: usize,
    pub staged_queue_capacity: usize,
    pub topk_queue_capacity: usize,
    pub edge_queue_capacity: usize,
    pub max_inflight_compute_tasks_active: usize,
    pub max_inflight_compute_tasks_drain: usize,
}

impl PipelineRuntimeConfig {
    pub fn from_params(params: &PiPNNParams, budget: SchedulerBudget) -> Self {
        let worker_count = budget.worker_count.max(1);
        Self {
            leaf_descriptor_queue_capacity: params
                .pipeline_leaf_descriptor_queue_capacity
                .unwrap_or(budget.leaf_backlog_capacity)
                .max(1),
            staged_queue_capacity: params
                .pipeline_staged_queue_capacity
                .unwrap_or(worker_count)
                .max(1),
            topk_queue_capacity: params
                .pipeline_topk_queue_capacity
                .unwrap_or(worker_count.saturating_mul(2))
                .max(1),
            edge_queue_capacity: params
                .pipeline_edge_queue_capacity
                .unwrap_or(worker_count.saturating_mul(2))
                .max(1),
            max_inflight_compute_tasks_active: params
                .pipeline_max_inflight_compute_tasks_active
                .unwrap_or(worker_count / 3)
                .max(1),
            max_inflight_compute_tasks_drain: params
                .pipeline_max_inflight_compute_tasks_drain
                .unwrap_or(worker_count.saturating_sub(2))
                .max(1),
        }
    }
}

pub struct ScopedLeafEmitter<'borrow, 'scope, 'a> {
    scheduler: Arc<PipnnScheduler<'a>>,
    scope: &'borrow ScopeFifo<'scope>,
}

thread_local! {
    static TLS_LEAF_SCRATCH: RefCell<Option<LeafScratch>> = const { RefCell::new(None) };
}

impl<'a> PipnnScheduler<'a> {
    pub fn new(
        budget: SchedulerBudget,
        context: LeafTaskContext<'a>,
        progress: &'a ProgressBar,
    ) -> Arc<Self> {
        Arc::new(Self {
            budget,
            context,
            progress,
            leaf_backlog: ArrayQueue::new(budget.leaf_backlog_capacity.max(1)),
            outstanding_leaf_tasks: AtomicUsize::new(0),
            inflight_leaf_tasks: AtomicUsize::new(0),
            peak_inflight_leaf_tasks: AtomicUsize::new(0),
            peak_leaf_backlog: AtomicUsize::new(0),
            active_leaf_drainers: AtomicUsize::new(0),
            producer_active: AtomicBool::new(true),
            inline_leaf_fallbacks: AtomicUsize::new(0),
            producer_help_drains: AtomicUsize::new(0),
            producer_help_yields: AtomicUsize::new(0),
            large_leaf_count: AtomicUsize::new(0),
            large_leaf_block_tasks: AtomicUsize::new(0),
            first_error: Mutex::new(None),
            leaf_profile: Mutex::new(LeafProfile::default()),
        })
    }

    pub fn leaf_profile(&self) -> LeafProfile {
        self.leaf_profile.lock().clone()
    }

    pub fn telemetry(&self) -> SchedulerTelemetry {
        SchedulerTelemetry {
            peak_inflight_leaf_tasks: self.peak_inflight_leaf_tasks.load(Ordering::Relaxed),
            peak_leaf_backlog: self.peak_leaf_backlog.load(Ordering::Relaxed),
            inline_leaf_fallbacks: self.inline_leaf_fallbacks.load(Ordering::Relaxed),
            producer_help_drains: self.producer_help_drains.load(Ordering::Relaxed),
            producer_help_yields: self.producer_help_yields.load(Ordering::Relaxed),
            large_leaf_count: self.large_leaf_count.load(Ordering::Relaxed),
            large_leaf_block_tasks: self.large_leaf_block_tasks.load(Ordering::Relaxed),
            ..SchedulerTelemetry::default()
        }
    }

    pub fn check_for_error(&self) -> AnnResult<()> {
        if let Some(message) = self.first_error.lock().clone() {
            Err(AnnError::log_index_error(message))
        } else {
            Ok(())
        }
    }

    pub fn wait_until_idle(&self) -> AnnResult<()> {
        while self.outstanding_leaf_tasks.load(Ordering::Acquire) != 0 {
            self.check_for_error()?;
            if !self.try_help_from_backlog(false)? && rayon::yield_now().is_none() {
                self.producer_help_yields.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
            }
        }
        self.check_for_error()
    }

    fn mark_producer_finished(&self) {
        self.producer_active.store(false, Ordering::Release);
    }

    fn producer_is_active(&self) -> bool {
        self.producer_active.load(Ordering::Acquire)
    }

    fn max_leaf_worker_capacity(&self) -> usize {
        self.budget.worker_count.saturating_sub(1).max(1)
    }

    fn active_producer_leaf_worker_cap(&self) -> usize {
        if self.budget.worker_count <= 1 {
            return 1;
        }

        // Keep a modest reserve for the RBC producer/recursion path while it is still generating
        // work, but let the leaf backlog consume most workers once the producer cannot keep the
        // pool saturated on its own. Reserving only about one third for RBC avoids the observed
        // late-phase underutilization where producer-active mode capped leaf work at ~1/3 of the
        // pool even though many workers would otherwise sit idle.
        self.budget
            .worker_count
            .saturating_sub(self.budget.worker_count.div_ceil(3))
            .max(1)
    }

    fn spawned_leaf_worker_limit_for_backlog(&self, backlog: usize) -> usize {
        if backlog == 0 {
            return 0;
        }

        let cap = if self.producer_is_active() {
            self.active_producer_leaf_worker_cap()
        } else {
            self.max_leaf_worker_capacity()
        };

        backlog.min(cap)
    }

    fn try_enqueue_leaf(&self, leaf: Vec<u32>) -> Result<(), Vec<u32>> {
        self.outstanding_leaf_tasks.fetch_add(1, Ordering::AcqRel);
        match self.leaf_backlog.push(leaf) {
            Ok(()) => {
                self.update_peak_backlog(self.leaf_backlog.len());
                Ok(())
            }
            Err(leaf) => {
                self.outstanding_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
                Err(leaf)
            }
        }
    }

    fn update_peak_inflight(&self, inflight: usize) {
        let mut peak = self.peak_inflight_leaf_tasks.load(Ordering::Acquire);
        while inflight > peak {
            match self.peak_inflight_leaf_tasks.compare_exchange_weak(
                peak,
                inflight,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn update_peak_backlog(&self, backlog: usize) {
        let mut peak = self.peak_leaf_backlog.load(Ordering::Acquire);
        while backlog > peak {
            match self.peak_leaf_backlog.compare_exchange_weak(
                peak,
                backlog,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn record_error(&self, err: AnnError) {
        let mut first_error = self.first_error.lock();
        if first_error.is_none() {
            *first_error = Some(err.to_string());
        }
    }

    fn start_leaf_execution(&self) {
        let inflight = self.inflight_leaf_tasks.fetch_add(1, Ordering::AcqRel) + 1;
        self.update_peak_inflight(inflight);
    }

    fn finish_leaf_execution(&self) {
        self.inflight_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
        self.outstanding_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
    }

    fn try_take_queued_leaf(&self) -> Option<Vec<u32>> {
        let leaf = self.leaf_backlog.pop()?;
        self.start_leaf_execution();
        Some(leaf)
    }

    fn process_taken_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        let result = self.process_leaf_task(leaf);
        self.finish_leaf_execution();
        result
    }

    fn try_help_from_backlog(&self, count_drain: bool) -> AnnResult<bool> {
        let Some(leaf) = self.try_take_queued_leaf() else {
            return Ok(false);
        };
        if count_drain {
            self.producer_help_drains.fetch_add(1, Ordering::Relaxed);
        }
        self.process_taken_leaf(leaf)?;
        Ok(true)
    }

    fn finish_leaf_drainer<'scope>(self: &Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        self.active_leaf_drainers.fetch_sub(1, Ordering::AcqRel);
        self.maybe_spawn_leaf_drainer(scope);
    }

    fn run_leaf_drainer<'scope>(self: Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        loop {
            if self.check_for_error().is_err() {
                break;
            }

            let Some(leaf) = self.try_take_queued_leaf() else {
                break;
            };

            if let Err(err) = self.process_taken_leaf(leaf) {
                self.record_error(err);
                break;
            }
        }

        self.finish_leaf_drainer(scope);
    }

    fn maybe_spawn_leaf_drainer<'scope>(self: &Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        loop {
            let backlog = self.leaf_backlog.len();
            if backlog == 0 {
                return;
            }

            let active = self.active_leaf_drainers.load(Ordering::Acquire);
            let target = self.spawned_leaf_worker_limit_for_backlog(backlog);
            if active >= target {
                return;
            }

            match self.active_leaf_drainers.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let scheduler = Arc::clone(self);
                    scope.spawn_fifo(move |scope| scheduler.run_leaf_drainer(scope));
                }
                Err(_) => continue,
            }
        }
    }

    fn process_leaf_task(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.check_for_error()?;
        #[cfg(test)]
        if let Some(observer) = self.context.observer {
            observer.on_leaf_start(&leaf);
        }

        let profile = process_leaf_passthrough(
            self.budget,
            self.context,
            &leaf,
            &self.large_leaf_count,
            &self.large_leaf_block_tasks,
        )?;
        self.finish_profile(profile);
        Ok(())
    }

    fn finish_profile(&self, mut profile: LeafProfile) {
        if !self.context.detailed_profiling {
            profile.total_wall = Duration::ZERO;
            profile.load = Duration::ZERO;
            profile.distance = Duration::ZERO;
            profile.topk = Duration::ZERO;
            profile.hash = Duration::ZERO;
            profile.flush = Duration::ZERO;
        }

        self.leaf_profile.lock().merge(profile);
        self.progress.inc(1);
        if self.progress.position() % 1000 == 0 {
            self.progress
                .set_message(format!("Processed {}", self.progress.position()));
        }
    }
}

pub fn run_scheduler_scope<'a, F>(
    pool: &rayon::ThreadPool,
    budget: SchedulerBudget,
    context: LeafTaskContext<'a>,
    progress: &'a ProgressBar,
    producer: F,
) -> AnnResult<(LeafProfile, SchedulerTelemetry)>
where
    F: FnOnce(&dyn LeafEmitter) -> AnnResult<()> + Send,
{
    let scheduler = PipnnScheduler::new(budget, context, progress);
    let scheduler_for_scope = Arc::clone(&scheduler);
    pool.scope_fifo(move |scope| {
        let emitter = ScopedLeafEmitter {
            scheduler: Arc::clone(&scheduler_for_scope),
            scope,
        };
        let result = producer(&emitter);
        scheduler_for_scope.mark_producer_finished();
        scheduler_for_scope.maybe_spawn_leaf_drainer(scope);
        let idle_result = scheduler_for_scope.wait_until_idle();
        result.and(idle_result)
    })?;

    Ok((scheduler.leaf_profile(), scheduler.telemetry()))
}

pub fn run_pipeline_scheduler_scope<'a, F>(
    pool: &rayon::ThreadPool,
    budget: SchedulerBudget,
    context: LeafTaskContext<'a>,
    progress: &'a ProgressBar,
    producer: F,
) -> AnnResult<(LeafProfile, SchedulerTelemetry)>
where
    F: FnOnce(&dyn LeafEmitter) -> AnnResult<()> + Send,
{
    let scheduler = PipelineScheduler::new(budget, context, progress);
    let scheduler_for_scope = Arc::clone(&scheduler);
    pool.scope_fifo(move |scope| {
        let emitter = ScopedPipelineLeafEmitter {
            scheduler: Arc::clone(&scheduler_for_scope),
            scope,
        };
        let producer_result = producer(&emitter);
        scheduler_for_scope.mark_producer_finished();
        let flush_result = scheduler_for_scope.flush_cuda_pending_descriptors();
        let result = producer_result.and(flush_result);
        scheduler_for_scope.maybe_spawn_descriptor_drainer(scope);
        let idle_result = scheduler_for_scope.wait_until_idle();
        result.and(idle_result)
    })?;

    Ok((scheduler.leaf_profile(), scheduler.telemetry()))
}

struct PipelineScheduler<'a> {
    budget: SchedulerBudget,
    config: PipelineRuntimeConfig,
    context: LeafTaskContext<'a>,
    progress: &'a ProgressBar,
    leaf_descriptor_queue: ArrayQueue<LeafDescriptor>,
    staged_queue: ArrayQueue<StagedLeafBatch>,
    topk_queue: ArrayQueue<CompactTopKBatch>,
    edge_queue: ArrayQueue<EdgeFlushBatch>,
    cuda_pending_descriptors: Mutex<Vec<LeafDescriptor>>,
    outstanding_leaf_tasks: AtomicUsize,
    inflight_compute_tasks: AtomicUsize,
    active_descriptor_drainers: AtomicUsize,
    producer_active: AtomicBool,
    next_leaf_id: AtomicUsize,
    peak_inflight_leaf_tasks: AtomicUsize,
    peak_leaf_descriptor_queue: AtomicUsize,
    peak_staged_queue: AtomicUsize,
    peak_topk_queue: AtomicUsize,
    peak_edge_queue: AtomicUsize,
    producer_wait_ms: AtomicUsize,
    staging_wait_ms: AtomicUsize,
    compute_wait_ms: AtomicUsize,
    edge_wait_ms: AtomicUsize,
    descriptor_receive_wall_ms: AtomicUsize,
    descriptor_batch_extend_wall_ms: AtomicUsize,
    stage_wall_ms: AtomicUsize,
    compute_flush_wall_ms: AtomicUsize,
    topk_queue_roundtrip_wall_ms: AtomicUsize,
    edge_queue_roundtrip_wall_ms: AtomicUsize,
    edge_flush_wall_ms: AtomicUsize,
    producer_blocked_ms: AtomicUsize,
    compute_blocked_ms: AtomicUsize,
    edge_blocked_ms: AtomicUsize,
    producer_help_drains: AtomicUsize,
    producer_help_yields: AtomicUsize,
    large_leaf_count: AtomicUsize,
    large_leaf_block_tasks: AtomicUsize,
    cpu_shadow_unsupported_fallbacks: AtomicUsize,
    gpu_workspace_pressure_events: AtomicUsize,
    gpu_cpu_fallbacks: AtomicUsize,
    descriptor_queue_empty_count: AtomicUsize,
    cuda_batch_count: AtomicUsize,
    cuda_batch_leaves: AtomicUsize,
    cuda_batch_rows: AtomicUsize,
    cuda_batch_target_bytes: AtomicUsize,
    cuda_batch_actual_bytes: AtomicUsize,
    cuda_batch_underfilled_count: AtomicUsize,
    post_producer_leaf_drain_ms: AtomicUsize,
    drain_start: Mutex<Option<Instant>>,
    first_error: Mutex<Option<String>>,
    leaf_profile: Mutex<LeafProfile>,
}

struct ScopedPipelineLeafEmitter<'borrow, 'scope, 'a> {
    scheduler: Arc<PipelineScheduler<'a>>,
    scope: &'borrow ScopeFifo<'scope>,
}

impl<'a> PipelineScheduler<'a> {
    fn new(
        budget: SchedulerBudget,
        context: LeafTaskContext<'a>,
        progress: &'a ProgressBar,
    ) -> Arc<Self> {
        let config = PipelineRuntimeConfig::from_params(context.params, budget);
        Arc::new(Self {
            budget,
            leaf_descriptor_queue: ArrayQueue::new(config.leaf_descriptor_queue_capacity),
            staged_queue: ArrayQueue::new(config.staged_queue_capacity),
            topk_queue: ArrayQueue::new(config.topk_queue_capacity),
            edge_queue: ArrayQueue::new(config.edge_queue_capacity),
            cuda_pending_descriptors: Mutex::new(Vec::new()),
            config,
            context,
            progress,
            outstanding_leaf_tasks: AtomicUsize::new(0),
            inflight_compute_tasks: AtomicUsize::new(0),
            active_descriptor_drainers: AtomicUsize::new(0),
            producer_active: AtomicBool::new(true),
            next_leaf_id: AtomicUsize::new(0),
            peak_inflight_leaf_tasks: AtomicUsize::new(0),
            peak_leaf_descriptor_queue: AtomicUsize::new(0),
            peak_staged_queue: AtomicUsize::new(0),
            peak_topk_queue: AtomicUsize::new(0),
            peak_edge_queue: AtomicUsize::new(0),
            producer_wait_ms: AtomicUsize::new(0),
            staging_wait_ms: AtomicUsize::new(0),
            compute_wait_ms: AtomicUsize::new(0),
            edge_wait_ms: AtomicUsize::new(0),
            descriptor_receive_wall_ms: AtomicUsize::new(0),
            descriptor_batch_extend_wall_ms: AtomicUsize::new(0),
            stage_wall_ms: AtomicUsize::new(0),
            compute_flush_wall_ms: AtomicUsize::new(0),
            topk_queue_roundtrip_wall_ms: AtomicUsize::new(0),
            edge_queue_roundtrip_wall_ms: AtomicUsize::new(0),
            edge_flush_wall_ms: AtomicUsize::new(0),
            producer_blocked_ms: AtomicUsize::new(0),
            compute_blocked_ms: AtomicUsize::new(0),
            edge_blocked_ms: AtomicUsize::new(0),
            producer_help_drains: AtomicUsize::new(0),
            producer_help_yields: AtomicUsize::new(0),
            large_leaf_count: AtomicUsize::new(0),
            large_leaf_block_tasks: AtomicUsize::new(0),
            cpu_shadow_unsupported_fallbacks: AtomicUsize::new(0),
            gpu_workspace_pressure_events: AtomicUsize::new(0),
            gpu_cpu_fallbacks: AtomicUsize::new(0),
            descriptor_queue_empty_count: AtomicUsize::new(0),
            cuda_batch_count: AtomicUsize::new(0),
            cuda_batch_leaves: AtomicUsize::new(0),
            cuda_batch_rows: AtomicUsize::new(0),
            cuda_batch_target_bytes: AtomicUsize::new(0),
            cuda_batch_actual_bytes: AtomicUsize::new(0),
            cuda_batch_underfilled_count: AtomicUsize::new(0),
            post_producer_leaf_drain_ms: AtomicUsize::new(0),
            drain_start: Mutex::new(None),
            first_error: Mutex::new(None),
            leaf_profile: Mutex::new(LeafProfile::default()),
        })
    }

    fn leaf_profile(&self) -> LeafProfile {
        self.leaf_profile.lock().clone()
    }

    fn telemetry(&self) -> SchedulerTelemetry {
        let stage_wall_ms = self.stage_wall_ms.load(Ordering::Relaxed) as u64;
        let compute_flush_wall_ms = self.compute_flush_wall_ms.load(Ordering::Relaxed) as u64;
        let edge_flush_wall_ms = self.edge_flush_wall_ms.load(Ordering::Relaxed) as u64;
        SchedulerTelemetry {
            peak_inflight_leaf_tasks: self.peak_inflight_leaf_tasks.load(Ordering::Relaxed),
            peak_leaf_backlog: self.peak_leaf_descriptor_queue.load(Ordering::Relaxed),
            producer_help_drains: self.producer_help_drains.load(Ordering::Relaxed),
            producer_help_yields: self.producer_help_yields.load(Ordering::Relaxed),
            large_leaf_count: self.large_leaf_count.load(Ordering::Relaxed),
            large_leaf_block_tasks: self.large_leaf_block_tasks.load(Ordering::Relaxed),
            producer_wait_ms: self.producer_wait_ms.load(Ordering::Relaxed) as u64,
            staging_wait_ms: stage_wall_ms,
            compute_wait_ms: compute_flush_wall_ms,
            edge_wait_ms: edge_flush_wall_ms,
            descriptor_receive_wall_ms: self.descriptor_receive_wall_ms.load(Ordering::Relaxed)
                as u64,
            descriptor_batch_extend_wall_ms: self
                .descriptor_batch_extend_wall_ms
                .load(Ordering::Relaxed) as u64,
            stage_wall_ms,
            compute_flush_wall_ms,
            topk_queue_roundtrip_wall_ms: self.topk_queue_roundtrip_wall_ms.load(Ordering::Relaxed)
                as u64,
            edge_queue_roundtrip_wall_ms: self.edge_queue_roundtrip_wall_ms.load(Ordering::Relaxed)
                as u64,
            edge_flush_wall_ms,
            producer_blocked_ms: self.producer_blocked_ms.load(Ordering::Relaxed) as u64,
            compute_blocked_ms: self.compute_blocked_ms.load(Ordering::Relaxed) as u64,
            edge_blocked_ms: self.edge_blocked_ms.load(Ordering::Relaxed) as u64,
            peak_leaf_descriptor_queue: self.peak_leaf_descriptor_queue.load(Ordering::Relaxed),
            peak_staged_queue: self.peak_staged_queue.load(Ordering::Relaxed),
            peak_topk_queue: self.peak_topk_queue.load(Ordering::Relaxed),
            peak_edge_queue: self.peak_edge_queue.load(Ordering::Relaxed),
            descriptor_queue_empty_count: self.descriptor_queue_empty_count.load(Ordering::Relaxed),
            cuda_batch_count: self.cuda_batch_count.load(Ordering::Relaxed),
            cuda_batch_leaves: self.cuda_batch_leaves.load(Ordering::Relaxed),
            cuda_batch_rows: self.cuda_batch_rows.load(Ordering::Relaxed),
            cuda_batch_target_bytes: self.cuda_batch_target_bytes.load(Ordering::Relaxed),
            cuda_batch_actual_bytes: self.cuda_batch_actual_bytes.load(Ordering::Relaxed),
            cuda_batch_underfilled_count: self.cuda_batch_underfilled_count.load(Ordering::Relaxed),
            cpu_shadow_unsupported_fallbacks: self
                .cpu_shadow_unsupported_fallbacks
                .load(Ordering::Relaxed),
            gpu_workspace_pressure_events: self
                .gpu_workspace_pressure_events
                .load(Ordering::Relaxed),
            gpu_cpu_fallbacks: self.gpu_cpu_fallbacks.load(Ordering::Relaxed),
            post_producer_leaf_drain_ms: self.post_producer_leaf_drain_ms.load(Ordering::Relaxed)
                as u64,
            ..SchedulerTelemetry::default()
        }
    }

    fn check_for_error(&self) -> AnnResult<()> {
        if let Some(message) = self.first_error.lock().clone() {
            Err(AnnError::log_index_error(message))
        } else {
            Ok(())
        }
    }

    fn record_error(&self, err: AnnError) {
        let mut first_error = self.first_error.lock();
        if first_error.is_none() {
            *first_error = Some(err.to_string());
        }
    }

    fn mark_producer_finished(&self) {
        self.producer_active.store(false, Ordering::Release);
        if self.outstanding_leaf_tasks.load(Ordering::Acquire) > 0 {
            *self.drain_start.lock() = Some(Instant::now());
        }
    }

    fn producer_is_active(&self) -> bool {
        self.producer_active.load(Ordering::Acquire)
    }

    fn update_peak(counter: &AtomicUsize, value: usize) {
        let mut peak = counter.load(Ordering::Acquire);
        while value > peak {
            match counter.compare_exchange_weak(peak, value, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn max_inflight_compute_tasks(&self) -> usize {
        if self.producer_is_active() {
            self.config.max_inflight_compute_tasks_active
        } else {
            self.config.max_inflight_compute_tasks_drain
        }
    }

    fn descriptor_worker_target(&self) -> usize {
        let backlog = self.leaf_descriptor_queue.len();
        if backlog == 0 {
            return 0;
        }
        backlog.min(self.max_inflight_compute_tasks())
    }

    fn try_enqueue_descriptor(&self, descriptor: LeafDescriptor) -> Result<(), LeafDescriptor> {
        self.outstanding_leaf_tasks.fetch_add(1, Ordering::AcqRel);
        match self.leaf_descriptor_queue.push(descriptor) {
            Ok(()) => {
                Self::update_peak(
                    &self.peak_leaf_descriptor_queue,
                    self.leaf_descriptor_queue.len(),
                );
                Ok(())
            }
            Err(descriptor) => {
                self.outstanding_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
                Err(descriptor)
            }
        }
    }

    fn start_descriptor_batch_execution(&self) {
        let inflight = self.inflight_compute_tasks.fetch_add(1, Ordering::AcqRel) + 1;
        Self::update_peak(&self.peak_inflight_leaf_tasks, inflight);
    }

    fn try_take_descriptor(&self) -> Option<LeafDescriptor> {
        let descriptor = self.leaf_descriptor_queue.pop()?;
        self.start_descriptor_batch_execution();
        Some(descriptor)
    }

    fn try_take_descriptor_without_inflight(&self) -> Option<LeafDescriptor> {
        self.leaf_descriptor_queue.pop()
    }

    fn finish_leaf_execution_count(&self, count: usize) {
        self.inflight_compute_tasks.fetch_sub(1, Ordering::AcqRel);
        if self
            .outstanding_leaf_tasks
            .fetch_sub(count, Ordering::AcqRel)
            == count
        {
            if let Some(start) = self.drain_start.lock().take() {
                self.post_producer_leaf_drain_ms
                    .fetch_add(elapsed_ms_usize(start), Ordering::Relaxed);
            }
        }
    }

    fn cuda_batch_target_bytes(&self, first_vector_bytes: usize) -> usize {
        self.context
            .params
            .effective_gpu_workspace_budget_bytes()
            .min(TARGET_CUDA_BATCH_VECTOR_BYTES)
            .max(first_vector_bytes.max(1))
    }

    fn cuda_descriptor_batch_ready(&self, descriptors: &[LeafDescriptor]) -> bool {
        if descriptors.len() >= MAX_CUDA_BATCH_LEAVES {
            return true;
        }
        let vector_bytes: usize = descriptors
            .iter()
            .map(|descriptor| descriptor.estimated_vector_bytes)
            .sum();
        vector_bytes
            >= self.cuda_batch_target_bytes(
                descriptors
                    .first()
                    .map_or(1, |descriptor| descriptor.estimated_vector_bytes),
            )
    }

    fn push_cuda_pending_descriptor(
        &self,
        descriptor: LeafDescriptor,
    ) -> AnnResult<Option<Vec<LeafDescriptor>>> {
        self.outstanding_leaf_tasks.fetch_add(1, Ordering::AcqRel);
        let mut pending = self.cuda_pending_descriptors.lock();
        pending.push(descriptor);
        if self.cuda_descriptor_batch_ready(&pending) {
            Ok(Some(std::mem::take(&mut *pending)))
        } else {
            Ok(None)
        }
    }

    fn process_cuda_pending_batch(
        &self,
        descriptors: Vec<LeafDescriptor>,
        descriptor_queue_empty: bool,
    ) -> AnnResult<()> {
        if descriptors.is_empty() {
            return Ok(());
        }
        let count = descriptors.len();
        let first_vector_bytes = descriptors
            .first()
            .map_or(1, |descriptor| descriptor.estimated_vector_bytes);
        let target_bytes = self.cuda_batch_target_bytes(first_vector_bytes);
        let actual_bytes: usize = descriptors
            .iter()
            .map(|descriptor| descriptor.estimated_vector_bytes)
            .sum();
        self.cuda_batch_target_bytes
            .fetch_add(target_bytes, Ordering::Relaxed);
        self.cuda_batch_actual_bytes
            .fetch_add(actual_bytes, Ordering::Relaxed);
        if descriptors.len() < MAX_CUDA_BATCH_LEAVES && actual_bytes < target_bytes {
            self.cuda_batch_underfilled_count
                .fetch_add(1, Ordering::Relaxed);
        }
        if descriptor_queue_empty {
            self.descriptor_queue_empty_count
                .fetch_add(1, Ordering::Relaxed);
        }
        self.start_descriptor_batch_execution();
        let result = self.process_descriptors(descriptors);
        self.finish_leaf_execution_count(count);
        result
    }

    fn emit_cuda_descriptor(&self, descriptor: LeafDescriptor) -> AnnResult<()> {
        if let Some(batch) = self.push_cuda_pending_descriptor(descriptor)? {
            self.process_cuda_pending_batch(batch, false)?;
        }
        Ok(())
    }

    fn flush_cuda_pending_descriptors(&self) -> AnnResult<()> {
        let batch = {
            let mut pending = self.cuda_pending_descriptors.lock();
            std::mem::take(&mut *pending)
        };
        self.process_cuda_pending_batch(batch, true)
    }

    fn wait_until_idle(&self) -> AnnResult<()> {
        while self.outstanding_leaf_tasks.load(Ordering::Acquire) != 0 {
            self.check_for_error()?;
            if !self.try_help_from_descriptor_queue(false)? && rayon::yield_now().is_none() {
                self.producer_help_yields.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
            }
        }
        self.check_for_error()
    }

    fn maybe_spawn_descriptor_drainer<'scope>(self: &Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        loop {
            let target = self.descriptor_worker_target();
            if target == 0 {
                return;
            }
            let active = self.active_descriptor_drainers.load(Ordering::Acquire);
            if active >= target {
                return;
            }
            match self.active_descriptor_drainers.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let scheduler = Arc::clone(self);
                    scope.spawn_fifo(move |scope| scheduler.run_descriptor_drainer(scope));
                }
                Err(_) => continue,
            }
        }
    }

    fn run_descriptor_drainer<'scope>(self: Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        loop {
            if self.check_for_error().is_err() {
                break;
            }
            let Some(descriptor) = self.try_take_descriptor() else {
                break;
            };
            if let Err(err) = self.process_taken_descriptor(descriptor) {
                self.record_error(err);
                break;
            }
        }
        self.active_descriptor_drainers
            .fetch_sub(1, Ordering::AcqRel);
        self.maybe_spawn_descriptor_drainer(scope);
    }

    fn try_help_from_descriptor_queue(&self, count_drain: bool) -> AnnResult<bool> {
        let Some(descriptor) = self.try_take_descriptor() else {
            return Ok(false);
        };
        if count_drain {
            self.producer_help_drains.fetch_add(1, Ordering::Relaxed);
        }
        self.process_taken_descriptor(descriptor)?;
        Ok(true)
    }

    fn try_process_current_with_queued_cuda_descriptor(
        &self,
        current: LeafDescriptor,
        count_drain: bool,
    ) -> AnnResult<Option<LeafDescriptor>> {
        let Some(queued) = self.try_take_descriptor() else {
            return Ok(Some(current));
        };
        if count_drain {
            self.producer_help_drains.fetch_add(1, Ordering::Relaxed);
        }

        let mut descriptors = vec![queued, current];
        let queued_count = 1 + self.extend_cuda_descriptor_batch(&mut descriptors);
        let result = self.process_descriptors(descriptors);
        self.finish_leaf_execution_count(queued_count);
        result?;
        Ok(None)
    }

    fn process_taken_descriptor(&self, descriptor: LeafDescriptor) -> AnnResult<()> {
        let receive_start = Instant::now();
        let mut descriptors = vec![descriptor];
        let mut queued_count = 1;
        if self.context.params.pipeline_compute_mode == PiPNNPipelineComputeMode::Cuda {
            queued_count += self.extend_cuda_descriptor_batch(&mut descriptors);
        }
        self.descriptor_receive_wall_ms
            .fetch_add(elapsed_ms_usize(receive_start), Ordering::Relaxed);
        let result = self.process_descriptors(descriptors);
        self.finish_leaf_execution_count(queued_count);
        result
    }

    fn process_descriptors(&self, descriptors: Vec<LeafDescriptor>) -> AnnResult<()> {
        self.check_for_error()?;
        #[cfg(test)]
        if let Some(observer) = self.context.observer {
            for descriptor in &descriptors {
                observer.on_leaf_start(&descriptor.point_ids);
            }
        }

        let stage_start = Instant::now();
        let staged = self.stage_descriptors(descriptors)?;
        let stage_wall_ms = elapsed_ms_usize(stage_start);
        self.stage_wall_ms
            .fetch_add(stage_wall_ms, Ordering::Relaxed);
        self.staging_wait_ms
            .fetch_add(stage_wall_ms, Ordering::Relaxed);

        let compute_start = Instant::now();
        let completed_descriptors = staged.descriptors.len();
        let profile = self.compute_and_flush(staged)?;
        let compute_flush_wall_ms = elapsed_ms_usize(compute_start);
        self.compute_flush_wall_ms
            .fetch_add(compute_flush_wall_ms, Ordering::Relaxed);
        self.compute_wait_ms
            .fetch_add(compute_flush_wall_ms, Ordering::Relaxed);
        self.finish_profile(profile, completed_descriptors);
        Ok(())
    }

    fn extend_cuda_descriptor_batch(&self, descriptors: &mut Vec<LeafDescriptor>) -> usize {
        let extend_start = Instant::now();
        let mut queued_count = 0usize;

        let mut vector_bytes: usize = descriptors
            .iter()
            .map(|descriptor| descriptor.estimated_vector_bytes)
            .sum();
        let first_vector_bytes = descriptors
            .first()
            .map_or(1, |descriptor| descriptor.estimated_vector_bytes);
        let target_bytes = self.cuda_batch_target_bytes(first_vector_bytes);
        let mut stopped_on_empty_queue = false;

        while descriptors.len() < MAX_CUDA_BATCH_LEAVES && vector_bytes < target_bytes {
            let Some(next) = self.try_take_descriptor_without_inflight() else {
                stopped_on_empty_queue = true;
                break;
            };
            vector_bytes = vector_bytes.saturating_add(next.estimated_vector_bytes);
            descriptors.push(next);
            queued_count += 1;
        }
        if stopped_on_empty_queue {
            self.descriptor_queue_empty_count
                .fetch_add(1, Ordering::Relaxed);
        }
        self.descriptor_batch_extend_wall_ms
            .fetch_add(elapsed_ms_usize(extend_start), Ordering::Relaxed);
        self.cuda_batch_target_bytes
            .fetch_add(target_bytes, Ordering::Relaxed);
        self.cuda_batch_actual_bytes
            .fetch_add(vector_bytes, Ordering::Relaxed);
        if descriptors.len() < MAX_CUDA_BATCH_LEAVES && vector_bytes < target_bytes {
            self.cuda_batch_underfilled_count
                .fetch_add(1, Ordering::Relaxed);
        }
        queued_count
    }

    fn stage_descriptors(&self, descriptors: Vec<LeafDescriptor>) -> AnnResult<StagedLeafBatch> {
        let mut staged = StagedLeafBatch::default();
        for descriptor in descriptors {
            let _descriptor_meta = (
                descriptor.leaf_size,
                descriptor.depth,
                descriptor.run_id,
                descriptor.window_start,
            );
            staged.leaf_ids.push(descriptor.leaf_id);
            staged.estimated_htod_bytes = staged
                .estimated_htod_bytes
                .saturating_add(descriptor.estimated_vector_bytes);
            staged.estimated_dtoh_bytes = staged
                .estimated_dtoh_bytes
                .saturating_add(descriptor.estimated_topk_bytes);
            staged.descriptors.push(descriptor);
        }
        self.push_staged(staged)
    }

    fn compute_and_flush(&self, staged: StagedLeafBatch) -> AnnResult<LeafProfile> {
        let _staged_meta = (
            staged.leaf_ids.len(),
            staged.host_vector_buffer.as_ref().map_or(0, Vec::len),
            staged.sketch_handle,
            staged.estimated_htod_bytes,
            staged.estimated_dtoh_bytes,
        );
        if self.context.params.pipeline_compute_mode == PiPNNPipelineComputeMode::Cuda {
            let mut profile = self.process_descriptors_cuda(staged.descriptors)?;
            if !self.context.detailed_profiling {
                zero_leaf_timing(&mut profile);
            }
            return Ok(profile);
        }

        let mut aggregate = LeafProfile::default();
        for descriptor in staged.descriptors {
            let mut profile = match self.context.params.pipeline_compute_mode {
                PiPNNPipelineComputeMode::Passthrough => {
                    self.process_descriptor_passthrough(&descriptor)?
                }
                PiPNNPipelineComputeMode::CpuShadow => {
                    self.process_descriptor_cpu_shadow(&descriptor)?
                }
                PiPNNPipelineComputeMode::Cuda => self.process_descriptor_cuda(&descriptor)?,
                PiPNNPipelineComputeMode::Mock => {
                    self.process_descriptor_passthrough(&descriptor)?
                }
            };
            if !self.context.detailed_profiling {
                zero_leaf_timing(&mut profile);
            }
            aggregate.merge(profile);
        }
        Ok(aggregate)
    }

    fn push_staged(&self, mut staged: StagedLeafBatch) -> AnnResult<StagedLeafBatch> {
        loop {
            match self.staged_queue.push(staged) {
                Ok(()) => {
                    Self::update_peak(&self.peak_staged_queue, self.staged_queue.len());
                    return self.staged_queue.pop().ok_or_else(|| {
                        AnnError::log_index_error(
                            "Staged leaf queue unexpectedly empty".to_string(),
                        )
                    });
                }
                Err(next_staged) => {
                    staged = next_staged;
                    self.compute_blocked_ms.fetch_add(1, Ordering::Relaxed);
                    if rayon::yield_now().is_none() {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }

    fn push_topk(&self, mut compact: CompactTopKBatch) -> AnnResult<CompactTopKBatch> {
        let roundtrip_start = Instant::now();
        loop {
            match self.topk_queue.push(compact) {
                Ok(()) => {
                    Self::update_peak(&self.peak_topk_queue, self.topk_queue.len());
                    let compact = self.topk_queue.pop().ok_or_else(|| {
                        AnnError::log_index_error(
                            "Compact top-k queue unexpectedly empty".to_string(),
                        )
                    })?;
                    self.topk_queue_roundtrip_wall_ms
                        .fetch_add(elapsed_ms_usize(roundtrip_start), Ordering::Relaxed);
                    return Ok(compact);
                }
                Err(next_compact) => {
                    compact = next_compact;
                    self.compute_blocked_ms.fetch_add(1, Ordering::Relaxed);
                    if rayon::yield_now().is_none() {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }

    fn push_edge_flush(&self, mut batch: EdgeFlushBatch) -> AnnResult<EdgeFlushBatch> {
        let roundtrip_start = Instant::now();
        loop {
            match self.edge_queue.push(batch) {
                Ok(()) => {
                    Self::update_peak(&self.peak_edge_queue, self.edge_queue.len());
                    let batch = self.edge_queue.pop().ok_or_else(|| {
                        AnnError::log_index_error("Edge flush queue unexpectedly empty".to_string())
                    })?;
                    self.edge_queue_roundtrip_wall_ms
                        .fetch_add(elapsed_ms_usize(roundtrip_start), Ordering::Relaxed);
                    return Ok(batch);
                }
                Err(next_batch) => {
                    batch = next_batch;
                    self.edge_blocked_ms.fetch_add(1, Ordering::Relaxed);
                    if rayon::yield_now().is_none() {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }

    fn process_descriptor_passthrough(
        &self,
        descriptor: &LeafDescriptor,
    ) -> AnnResult<LeafProfile> {
        process_leaf_passthrough(
            self.budget,
            self.context,
            &descriptor.point_ids,
            &self.large_leaf_count,
            &self.large_leaf_block_tasks,
        )
    }

    fn process_descriptor_cpu_shadow(&self, descriptor: &LeafDescriptor) -> AnnResult<LeafProfile> {
        if self.context.metric != Metric::L2
            || descriptor.point_ids.len() >= self.budget.large_leaf_min_size
            || descriptor.point_ids.len() > self.context.params.kernel_safe_leaf_size()
        {
            self.cpu_shadow_unsupported_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return self.process_descriptor_passthrough(descriptor);
        }

        let (mut compact, mut profile) = compute_leaf_topk_l2_cpu_shadow(
            self.context.dataset,
            &descriptor.point_ids,
            self.context.params,
        )?;
        compact.leaf_id = descriptor.leaf_id;
        let compact = self.push_topk(compact)?;
        let edge_batch = self.push_edge_flush(EdgeFlushBatch {
            leaf_id: descriptor.leaf_id,
            estimated_dtoh_bytes: compact.estimated_total_dtoh_bytes(),
            topk: Some(compact),
        })?;
        let _edge_estimated_dtoh_bytes = edge_batch.estimated_dtoh_bytes;
        let compact = edge_batch.topk.ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Edge flush batch for leaf {} did not contain compact top-k payload",
                edge_batch.leaf_id
            ))
        })?;

        let edge_start = Instant::now();
        let flush_profile = flush_compact_topk_batch(
            &compact,
            self.context.sketches,
            self.context.params,
            self.context.edge_sink,
        )?;
        let edge_flush_wall_ms = elapsed_ms_usize(edge_start);
        self.edge_flush_wall_ms
            .fetch_add(edge_flush_wall_ms, Ordering::Relaxed);
        self.edge_wait_ms
            .fetch_add(edge_flush_wall_ms, Ordering::Relaxed);
        merge_cpu_shadow_flush_profile(&mut profile, flush_profile);
        Ok(profile)
    }

    fn process_descriptors_cuda(&self, descriptors: Vec<LeafDescriptor>) -> AnnResult<LeafProfile> {
        if self.context.compute_backend.kind() != PiPNNComputeBackendKind::Cuda {
            return Err(AnnError::log_index_error(
                "CUDA pipeline compute mode requires the CUDA compute backend; pass --pipnn-backend cuda and build with pipnn_cuda"
                    .to_string(),
            ));
        }
        if self.context.metric != Metric::L2 {
            return Err(AnnError::log_index_error(
                "CUDA pipeline compute mode currently supports only L2".to_string(),
            ));
        }

        let leaf_ids: Vec<u64> = descriptors
            .iter()
            .map(|descriptor| descriptor.leaf_id)
            .collect();
        let batch_leaf_count = descriptors.len();
        let batch_row_count: usize = descriptors
            .iter()
            .map(|descriptor| descriptor.point_ids.len())
            .sum();
        self.cuda_batch_count.fetch_add(1, Ordering::Relaxed);
        self.cuda_batch_leaves
            .fetch_add(batch_leaf_count, Ordering::Relaxed);
        self.cuda_batch_rows
            .fetch_add(batch_row_count, Ordering::Relaxed);
        let leaves: Vec<Vec<u32>> = descriptors
            .into_iter()
            .map(|descriptor| descriptor.point_ids)
            .collect();
        let (mut compacts, mut profile) = self.context.compute_backend.compute_leaf_topk_l2_batch(
            self.context.dataset,
            &leaves,
            self.context.params,
            self.budget.leaf_block_rows,
        )?;
        if compacts.len() != leaf_ids.len() {
            return Err(AnnError::log_index_error(format!(
                "CUDA compact top-k batch returned {} leaves for {} descriptors",
                compacts.len(),
                leaf_ids.len()
            )));
        }
        if profile.gpu_workspace_pressure_events > 0 {
            self.gpu_workspace_pressure_events
                .fetch_add(profile.gpu_workspace_pressure_events, Ordering::Relaxed);
        }
        if profile.gpu_cpu_fallbacks > 0 {
            self.gpu_cpu_fallbacks
                .fetch_add(profile.gpu_cpu_fallbacks, Ordering::Relaxed);
        }

        for (compact, leaf_id) in compacts.iter_mut().zip(leaf_ids) {
            compact.leaf_id = leaf_id;
            let compact = self.push_topk(std::mem::take(compact))?;
            let edge_batch = self.push_edge_flush(EdgeFlushBatch {
                leaf_id,
                estimated_dtoh_bytes: compact.estimated_total_dtoh_bytes(),
                topk: Some(compact),
            })?;
            let _edge_estimated_dtoh_bytes = edge_batch.estimated_dtoh_bytes;
            let compact = edge_batch.topk.ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Edge flush batch for leaf {} did not contain CUDA compact top-k payload",
                    edge_batch.leaf_id
                ))
            })?;

            let edge_start = Instant::now();
            let flush_profile = flush_compact_topk_batch(
                &compact,
                self.context.sketches,
                self.context.params,
                self.context.edge_sink,
            )?;
            let edge_flush_wall_ms = elapsed_ms_usize(edge_start);
            self.edge_flush_wall_ms
                .fetch_add(edge_flush_wall_ms, Ordering::Relaxed);
            self.edge_wait_ms
                .fetch_add(edge_flush_wall_ms, Ordering::Relaxed);
            merge_cpu_shadow_flush_profile(&mut profile, flush_profile);
        }
        Ok(profile)
    }

    fn process_descriptor_cuda(&self, descriptor: &LeafDescriptor) -> AnnResult<LeafProfile> {
        if self.context.compute_backend.kind() != PiPNNComputeBackendKind::Cuda {
            return Err(AnnError::log_index_error(
                "CUDA pipeline compute mode requires the CUDA compute backend; pass --pipnn-backend cuda and build with pipnn_cuda"
                    .to_string(),
            ));
        }
        if self.context.metric != Metric::L2 {
            return Err(AnnError::log_index_error(
                "CUDA pipeline compute mode currently supports only L2".to_string(),
            ));
        }

        let (mut compact, mut profile) = self.context.compute_backend.compute_leaf_topk_l2(
            self.context.dataset,
            &descriptor.point_ids,
            self.context.params,
            self.budget.leaf_block_rows,
        )?;
        if profile.gpu_workspace_pressure_events > 0 {
            self.gpu_workspace_pressure_events
                .fetch_add(profile.gpu_workspace_pressure_events, Ordering::Relaxed);
        }
        if profile.gpu_cpu_fallbacks > 0 {
            self.gpu_cpu_fallbacks
                .fetch_add(profile.gpu_cpu_fallbacks, Ordering::Relaxed);
        }
        compact.leaf_id = descriptor.leaf_id;
        let compact = self.push_topk(compact)?;
        let edge_batch = self.push_edge_flush(EdgeFlushBatch {
            leaf_id: descriptor.leaf_id,
            estimated_dtoh_bytes: compact.estimated_total_dtoh_bytes(),
            topk: Some(compact),
        })?;
        let _edge_estimated_dtoh_bytes = edge_batch.estimated_dtoh_bytes;
        let compact = edge_batch.topk.ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Edge flush batch for leaf {} did not contain CUDA compact top-k payload",
                edge_batch.leaf_id
            ))
        })?;

        let edge_start = Instant::now();
        let flush_profile = flush_compact_topk_batch(
            &compact,
            self.context.sketches,
            self.context.params,
            self.context.edge_sink,
        )?;
        let edge_flush_wall_ms = elapsed_ms_usize(edge_start);
        self.edge_flush_wall_ms
            .fetch_add(edge_flush_wall_ms, Ordering::Relaxed);
        self.edge_wait_ms
            .fetch_add(edge_flush_wall_ms, Ordering::Relaxed);
        merge_cpu_shadow_flush_profile(&mut profile, flush_profile);
        Ok(profile)
    }

    fn finish_profile(&self, mut profile: LeafProfile, completed_descriptors: usize) {
        if !self.context.detailed_profiling {
            zero_leaf_timing(&mut profile);
        }
        self.leaf_profile.lock().merge(profile);
        self.progress.inc(completed_descriptors as u64);
        if self.progress.position() % 1000 == 0 {
            self.progress
                .set_message(format!("Processed {}", self.progress.position()));
        }
    }
}

impl<'borrow, 'scope, 'a: 'scope> LeafEmitter for ScopedPipelineLeafEmitter<'borrow, 'scope, 'a> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.scheduler.check_for_error()?;
        let leaf_id = self.scheduler.next_leaf_id.fetch_add(1, Ordering::Relaxed) as u64;
        let mut descriptor = LeafDescriptor::from_leaf(
            leaf_id,
            leaf,
            self.scheduler.context.dataset.dim(),
            self.scheduler.context.params.leaf_knn,
        );
        if self.scheduler.context.params.pipeline_compute_mode == PiPNNPipelineComputeMode::Cuda {
            return self.scheduler.emit_cuda_descriptor(descriptor);
        }
        loop {
            let enqueue_start = Instant::now();
            match self.scheduler.try_enqueue_descriptor(descriptor) {
                Ok(()) => {
                    self.scheduler
                        .producer_wait_ms
                        .fetch_add(elapsed_ms_usize(enqueue_start), Ordering::Relaxed);
                    if self.scheduler.context.params.pipeline_compute_mode
                        != PiPNNPipelineComputeMode::Cuda
                    {
                        self.scheduler.maybe_spawn_descriptor_drainer(self.scope);
                    }
                    return Ok(());
                }
                Err(next_descriptor) => {
                    descriptor = next_descriptor;
                    let blocked_ms = elapsed_ms_usize(enqueue_start).max(1);
                    self.scheduler
                        .producer_blocked_ms
                        .fetch_add(blocked_ms, Ordering::Relaxed);
                    self.scheduler
                        .producer_wait_ms
                        .fetch_add(blocked_ms, Ordering::Relaxed);
                    if self.scheduler.context.params.pipeline_compute_mode
                        != PiPNNPipelineComputeMode::Cuda
                    {
                        self.scheduler.maybe_spawn_descriptor_drainer(self.scope);
                    }
                    if self.scheduler.context.params.pipeline_compute_mode
                        == PiPNNPipelineComputeMode::Cuda
                    {
                        match self
                            .scheduler
                            .try_process_current_with_queued_cuda_descriptor(descriptor, true)?
                        {
                            Some(next_descriptor) => {
                                descriptor = next_descriptor;
                            }
                            None => return Ok(()),
                        }
                    }
                    if self.scheduler.try_help_from_descriptor_queue(true)? {
                        continue;
                    }
                    if rayon::yield_now().is_none() {
                        self.scheduler
                            .producer_help_yields
                            .fetch_add(1, Ordering::Relaxed);
                        std::thread::yield_now();
                    }
                }
            }
        }
    }
}

impl<'borrow, 'scope, 'a: 'scope> LeafEmitter for ScopedLeafEmitter<'borrow, 'scope, 'a> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.scheduler.check_for_error()?;

        let mut leaf = leaf;
        loop {
            match self.scheduler.try_enqueue_leaf(leaf) {
                Ok(()) => {
                    self.scheduler.maybe_spawn_leaf_drainer(self.scope);
                    return Ok(());
                }
                Err(next_leaf) => {
                    leaf = next_leaf;
                    self.scheduler.maybe_spawn_leaf_drainer(self.scope);
                    // When the bounded backlog is full, free one slot inline before yielding.
                    // Using rayon::yield_now() as the primary backpressure path can execute more
                    // nested Rayon jobs on the current worker stack, which is exactly the
                    // pattern that triggered the observed RBC stack overflow.
                    if self.scheduler.try_help_from_backlog(true)? {
                        continue;
                    }

                    if rayon::yield_now().is_none() {
                        self.scheduler
                            .producer_help_yields
                            .fetch_add(1, Ordering::Relaxed);
                        std::thread::yield_now();
                    }
                }
            }
        }
    }
}

fn process_leaf_passthrough(
    budget: SchedulerBudget,
    context: LeafTaskContext<'_>,
    leaf: &[u32],
    large_leaf_count: &AtomicUsize,
    large_leaf_block_tasks: &AtomicUsize,
) -> AnnResult<LeafProfile> {
    if leaf.len() >= budget.large_leaf_min_size {
        large_leaf_count.fetch_add(1, Ordering::Relaxed);
        large_leaf_block_tasks.fetch_add(
            leaf.len().div_ceil(budget.leaf_block_rows),
            Ordering::Relaxed,
        );
        process_leaf_parallel_large_profiled_with_backend_and_sink(
            context.compute_backend,
            context.dataset,
            context.metric,
            context.sketches,
            leaf,
            context.params,
            context.edge_sink,
            budget.leaf_block_rows,
        )
    } else {
        with_thread_local_leaf_scratch(
            context.params.kernel_safe_leaf_size(),
            context.dataset.dim(),
            context.params.leaf_knn,
            |scratch| {
                process_leaf_profiled_with_sink(
                    context.dataset,
                    context.metric,
                    context.sketches,
                    leaf,
                    context.params,
                    context.edge_sink,
                    scratch,
                )
            },
        )
    }
}

fn zero_leaf_timing(profile: &mut LeafProfile) {
    profile.total_wall = Duration::ZERO;
    profile.load = Duration::ZERO;
    profile.distance = Duration::ZERO;
    profile.topk = Duration::ZERO;
    profile.hash = Duration::ZERO;
    profile.flush = Duration::ZERO;
    profile.host_load_wall = Duration::ZERO;
    profile.backend_compute_wall = Duration::ZERO;
    profile.compact_materialize_wall = Duration::ZERO;
    profile.hash_wall = Duration::ZERO;
    profile.edge_sink_wall = Duration::ZERO;
    profile.edge_sink_sort_wall = Duration::ZERO;
    profile.edge_sink_insert_wall = Duration::ZERO;
    profile.edge_sink_edges = 0;
    profile.edge_sink_flushes = 0;
    profile.flush_wall = Duration::ZERO;
    profile.gpu_h2d = Duration::ZERO;
    profile.gpu_d2h = Duration::ZERO;
    profile.gpu_kernel = Duration::ZERO;
    profile.gpu_sync = Duration::ZERO;
    profile.gpu_h2d_submit_wall = Duration::ZERO;
    profile.gpu_kernel_launch = Duration::ZERO;
    profile.gpu_kernel_elapsed = Duration::ZERO;
    profile.gpu_d2h_submit_wall = Duration::ZERO;
    profile.gpu_post_d2h_sync_wall = Duration::ZERO;
    profile.gpu_stream_total_elapsed = Duration::ZERO;
}

fn merge_cpu_shadow_flush_profile(profile: &mut LeafProfile, flush: LeafProfile) {
    profile.total_wall += flush.total_wall;
    profile.hash += flush.hash;
    profile.flush += flush.flush;
    profile.hash_wall += flush.hash_wall;
    profile.edge_sink_wall += flush.edge_sink_wall;
    profile.edge_sink_sort_wall += flush.edge_sink_sort_wall;
    profile.edge_sink_insert_wall += flush.edge_sink_insert_wall;
    profile.edge_sink_edges += flush.edge_sink_edges;
    profile.edge_sink_flushes += flush.edge_sink_flushes;
    profile.flush_wall += flush.flush_wall;
    profile.io_stats.merge(&flush.io_stats);
    profile.sketch_prefetch += flush.sketch_prefetch;
    profile.sketch_rows_prefetched += flush.sketch_rows_prefetched;
    profile.gpu_h2d_bytes += flush.gpu_h2d_bytes;
    profile.gpu_d2h_bytes += flush.gpu_d2h_bytes;
    profile.gpu_h2d += flush.gpu_h2d;
    profile.gpu_d2h += flush.gpu_d2h;
    profile.gpu_kernel += flush.gpu_kernel;
    profile.gpu_sync += flush.gpu_sync;
    profile.gpu_h2d_submit_wall += flush.gpu_h2d_submit_wall;
    profile.gpu_kernel_launch += flush.gpu_kernel_launch;
    profile.gpu_kernel_elapsed += flush.gpu_kernel_elapsed;
    profile.gpu_d2h_submit_wall += flush.gpu_d2h_submit_wall;
    profile.gpu_post_d2h_sync_wall += flush.gpu_post_d2h_sync_wall;
    profile.gpu_stream_total_elapsed += flush.gpu_stream_total_elapsed;
    profile.compact_lune_witness_enabled |= flush.compact_lune_witness_enabled;
    profile.compact_lune_witness_slots += flush.compact_lune_witness_slots;
    profile.compact_lune_witnesses += flush.compact_lune_witnesses;
    profile.compact_lune_prune_enabled |= flush.compact_lune_prune_enabled;
    profile.compact_lune_prune_protected_core = profile
        .compact_lune_prune_protected_core
        .max(flush.compact_lune_prune_protected_core);
    profile.compact_lune_pruned_slots += flush.compact_lune_pruned_slots;
    profile.compact_lune_pruned_rows += flush.compact_lune_pruned_rows;
    profile.gpu_workspace_peak_bytes = profile
        .gpu_workspace_peak_bytes
        .max(flush.gpu_workspace_peak_bytes);
    profile.gpu_workspace_budget_bytes = profile
        .gpu_workspace_budget_bytes
        .max(flush.gpu_workspace_budget_bytes);
    profile.gpu_workspace_pressure_events += flush.gpu_workspace_pressure_events;
    profile.gpu_cpu_fallbacks += flush.gpu_cpu_fallbacks;
    profile.hashprune_evidence.merge(flush.hashprune_evidence);
}

fn elapsed_ms_usize(start: Instant) -> usize {
    let millis = start.elapsed().as_millis();
    millis.min(usize::MAX as u128) as usize
}

fn with_thread_local_leaf_scratch<R, F>(
    c_max: usize,
    dim: usize,
    leaf_knn: usize,
    f: F,
) -> AnnResult<R>
where
    F: FnOnce(&mut LeafScratch) -> AnnResult<R>,
{
    TLS_LEAF_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let recreate = match slot.as_ref() {
            Some(scratch) => scratch.x.ncols() != dim || scratch.x.nrows() < c_max,
            None => true,
        };
        if recreate {
            *slot = Some(LeafScratch::new(c_max, dim, leaf_knn));
        }
        f(slot.as_mut().expect("thread-local scratch initialized"))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use indicatif::ProgressBar;
    use parking_lot::Mutex;
    use rayon::prelude::*;

    use super::{
        CompactTopKBatch, LeafProfile, LeafTaskContext, PiPNNComputeBackend, PipnnScheduler,
        SchedulerBudget, SchedulerTelemetry, SchedulerTestObserver, run_pipeline_scheduler_scope,
        run_scheduler_scope,
    };
    use crate::algorithm::pipnn::hash_prune::{HashPruneReservoir, SketchAccessor, SketchStore};
    use crate::algorithm::pipnn::hash_prune_evidence::HashPruneEvidenceProfile;
    use crate::algorithm::pipnn::leaf_build::{PendingEdgeSink, ReservoirEdgeSink};
    use crate::algorithm::pipnn::point_store::{InmemDatasetPointStore, PointStore};
    use crate::algorithm::pipnn::rbc_partition::GemmProfile;
    use crate::algorithm::pipnn::{
        PiPNNComputeBackendKind, PiPNNParams, PiPNNPipelineComputeMode, compute_backend,
    };
    use crate::common::{AnnError, AnnResult, Metric};
    use crate::model::InmemDataset;

    #[test]
    fn zero_leaf_timing_clears_legacy_and_semantic_timing_fields() {
        let mut profile = LeafProfile {
            total_wall: Duration::from_millis(1),
            load: Duration::from_millis(2),
            distance: Duration::from_millis(3),
            topk: Duration::from_millis(4),
            hash: Duration::from_millis(5),
            flush: Duration::from_millis(6),
            leaves: 7,
            points: 8,
            host_load_wall: Duration::from_millis(9),
            backend_compute_wall: Duration::from_millis(10),
            compact_materialize_wall: Duration::from_millis(11),
            hash_wall: Duration::from_millis(12),
            edge_sink_wall: Duration::from_millis(13),
            edge_sink_sort_wall: Duration::from_millis(14),
            edge_sink_insert_wall: Duration::from_millis(15),
            edge_sink_edges: 16,
            edge_sink_flushes: 17,
            flush_wall: Duration::from_millis(18),
            gpu_h2d: Duration::from_millis(19),
            gpu_d2h: Duration::from_millis(20),
            gpu_kernel: Duration::from_millis(21),
            gpu_sync: Duration::from_millis(22),
            gpu_h2d_submit_wall: Duration::from_millis(23),
            gpu_kernel_launch: Duration::from_millis(24),
            gpu_kernel_elapsed: Duration::from_millis(25),
            gpu_d2h_submit_wall: Duration::from_millis(26),
            gpu_post_d2h_sync_wall: Duration::from_millis(27),
            gpu_stream_total_elapsed: Duration::from_millis(28),
            ..LeafProfile::default()
        };

        super::zero_leaf_timing(&mut profile);

        assert_eq!(profile.leaves, 7);
        assert_eq!(profile.points, 8);
        assert_eq!(profile.total_wall, Duration::ZERO);
        assert_eq!(profile.load, Duration::ZERO);
        assert_eq!(profile.distance, Duration::ZERO);
        assert_eq!(profile.topk, Duration::ZERO);
        assert_eq!(profile.hash, Duration::ZERO);
        assert_eq!(profile.flush, Duration::ZERO);
        assert_eq!(profile.host_load_wall, Duration::ZERO);
        assert_eq!(profile.backend_compute_wall, Duration::ZERO);
        assert_eq!(profile.compact_materialize_wall, Duration::ZERO);
        assert_eq!(profile.hash_wall, Duration::ZERO);
        assert_eq!(profile.edge_sink_wall, Duration::ZERO);
        assert_eq!(profile.edge_sink_sort_wall, Duration::ZERO);
        assert_eq!(profile.edge_sink_insert_wall, Duration::ZERO);
        assert_eq!(profile.edge_sink_edges, 0);
        assert_eq!(profile.edge_sink_flushes, 0);
        assert_eq!(profile.flush_wall, Duration::ZERO);
        assert_eq!(profile.gpu_h2d, Duration::ZERO);
        assert_eq!(profile.gpu_d2h, Duration::ZERO);
        assert_eq!(profile.gpu_kernel, Duration::ZERO);
        assert_eq!(profile.gpu_sync, Duration::ZERO);
        assert_eq!(profile.gpu_h2d_submit_wall, Duration::ZERO);
        assert_eq!(profile.gpu_kernel_launch, Duration::ZERO);
        assert_eq!(profile.gpu_kernel_elapsed, Duration::ZERO);
        assert_eq!(profile.gpu_d2h_submit_wall, Duration::ZERO);
        assert_eq!(profile.gpu_post_d2h_sync_wall, Duration::ZERO);
        assert_eq!(profile.gpu_stream_total_elapsed, Duration::ZERO);
    }

    #[test]
    fn hashprune_evidence_merge_cpu_shadow_flush_profile_preserves_counts_and_flush_timings() {
        let mut profile = LeafProfile {
            total_wall: Duration::from_millis(10),
            leaves: 2,
            points: 4,
            hash_wall: Duration::from_millis(1),
            edge_sink_wall: Duration::from_millis(2),
            flush_wall: Duration::from_millis(3),
            hashprune_evidence: HashPruneEvidenceProfile {
                annotation_wall: Duration::from_millis(1),
                coalesce_wall: Duration::from_millis(2),
                reciprocal_checks: 3,
                class_counts: [1, 2, 3, 4, 5, 6],
                duplicate_records_removed: 7,
                spill_bytes_avoided: 8,
                ..HashPruneEvidenceProfile::default()
            },
            gpu_workspace_peak_bytes: 128,
            gpu_workspace_budget_bytes: 256,
            gpu_workspace_pressure_events: 1,
            gpu_cpu_fallbacks: 2,
            ..LeafProfile::default()
        };
        let flush = LeafProfile {
            total_wall: Duration::from_millis(20),
            leaves: 99,
            points: 199,
            hash: Duration::from_millis(4),
            flush: Duration::from_millis(5),
            hash_wall: Duration::from_millis(6),
            edge_sink_wall: Duration::from_millis(7),
            edge_sink_sort_wall: Duration::from_millis(8),
            edge_sink_insert_wall: Duration::from_millis(9),
            edge_sink_edges: 10,
            edge_sink_flushes: 11,
            flush_wall: Duration::from_millis(12),
            gpu_h2d_submit_wall: Duration::from_millis(13),
            gpu_kernel_launch: Duration::from_millis(14),
            gpu_kernel_elapsed: Duration::from_millis(15),
            gpu_d2h_submit_wall: Duration::from_millis(16),
            gpu_post_d2h_sync_wall: Duration::from_millis(17),
            gpu_stream_total_elapsed: Duration::from_millis(18),
            hashprune_evidence: HashPruneEvidenceProfile {
                annotation_wall: Duration::from_millis(10),
                coalesce_wall: Duration::from_millis(20),
                reciprocal_checks: 30,
                class_counts: [6, 5, 4, 3, 2, 1],
                duplicate_records_removed: 40,
                spill_bytes_avoided: 50,
                ..HashPruneEvidenceProfile::default()
            },
            gpu_workspace_peak_bytes: 512,
            gpu_workspace_budget_bytes: 1024,
            gpu_workspace_pressure_events: 3,
            gpu_cpu_fallbacks: 5,
            ..LeafProfile::default()
        };

        super::merge_cpu_shadow_flush_profile(&mut profile, flush);

        assert_eq!(profile.total_wall, Duration::from_millis(30));
        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert_eq!(profile.hash, Duration::from_millis(4));
        assert_eq!(profile.flush, Duration::from_millis(5));
        assert_eq!(profile.hash_wall, Duration::from_millis(7));
        assert_eq!(profile.edge_sink_wall, Duration::from_millis(9));
        assert_eq!(profile.edge_sink_sort_wall, Duration::from_millis(8));
        assert_eq!(profile.edge_sink_insert_wall, Duration::from_millis(9));
        assert_eq!(profile.edge_sink_edges, 10);
        assert_eq!(profile.edge_sink_flushes, 11);
        assert_eq!(profile.flush_wall, Duration::from_millis(15));
        assert_eq!(profile.gpu_h2d_submit_wall, Duration::from_millis(13));
        assert_eq!(profile.gpu_kernel_launch, Duration::from_millis(14));
        assert_eq!(profile.gpu_kernel_elapsed, Duration::from_millis(15));
        assert_eq!(profile.gpu_d2h_submit_wall, Duration::from_millis(16));
        assert_eq!(profile.gpu_post_d2h_sync_wall, Duration::from_millis(17));
        assert_eq!(profile.gpu_stream_total_elapsed, Duration::from_millis(18));
        assert_eq!(profile.gpu_workspace_peak_bytes, 512);
        assert_eq!(profile.gpu_workspace_budget_bytes, 1024);
        assert_eq!(profile.gpu_workspace_pressure_events, 4);
        assert_eq!(profile.gpu_cpu_fallbacks, 7);
        assert_eq!(
            profile.hashprune_evidence.annotation_wall,
            Duration::from_millis(11)
        );
        assert_eq!(
            profile.hashprune_evidence.coalesce_wall,
            Duration::from_millis(22)
        );
        assert_eq!(profile.hashprune_evidence.reciprocal_checks, 33);
        assert_eq!(profile.hashprune_evidence.class_counts, [7; 6]);
        assert_eq!(profile.hashprune_evidence.duplicate_records_removed, 47);
        assert_eq!(profile.hashprune_evidence.spill_bytes_avoided, 58);
    }

    #[derive(Default)]
    struct RecordingObserver {
        starts: parking_lot::Mutex<Vec<u32>>,
    }

    impl RecordingObserver {
        fn started(&self) -> Vec<u32> {
            self.starts.lock().clone()
        }
    }

    impl SchedulerTestObserver for RecordingObserver {
        fn on_leaf_start(&self, leaf: &[u32]) {
            self.starts.lock().push(leaf[0]);
        }
    }

    fn build_test_context<'a>(
        dataset: &'a InmemDatasetPointStore<'a>,
        sketches: &'a dyn SketchAccessor,
        params: &'a PiPNNParams,
        edge_sink: &'a dyn super::PendingEdgeSink,
        observer: Option<&'a dyn SchedulerTestObserver>,
    ) -> LeafTaskContext<'a> {
        build_test_context_with_backend(
            compute_backend::cpu_backend(),
            dataset,
            sketches,
            params,
            edge_sink,
            observer,
        )
    }

    fn build_test_context_with_backend<'a>(
        compute_backend: &'a dyn PiPNNComputeBackend,
        dataset: &'a InmemDatasetPointStore<'a>,
        sketches: &'a dyn SketchAccessor,
        params: &'a PiPNNParams,
        edge_sink: &'a dyn super::PendingEdgeSink,
        observer: Option<&'a dyn SchedulerTestObserver>,
    ) -> LeafTaskContext<'a> {
        LeafTaskContext {
            compute_backend,
            dataset,
            metric: Metric::L2,
            sketches,
            params,
            edge_sink,
            detailed_profiling: false,
            observer,
        }
    }

    #[derive(Default)]
    struct FakeCudaCompactBackend {
        single_calls: AtomicUsize,
        batch_calls: AtomicUsize,
        max_batch_leaves: AtomicUsize,
        batch_sleep: Duration,
    }

    impl PiPNNComputeBackend for FakeCudaCompactBackend {
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
                "fake backend does not implement assignment".to_string(),
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
                "CUDA compact path must not call full-matrix leaf processing".to_string(),
            ))
        }

        fn compute_leaf_topk_l2(
            &self,
            _dataset: &dyn PointStore,
            leaf: &[u32],
            params: &PiPNNParams,
            _block_rows: usize,
        ) -> AnnResult<(CompactTopKBatch, LeafProfile)> {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            let k = params.leaf_knn.min(leaf.len().saturating_sub(1));
            Ok((
                CompactTopKBatch {
                    point_ids: leaf.to_vec(),
                    rows: leaf.len(),
                    k,
                    topk_ids: vec![leaf[1], leaf[0]],
                    topk_dists: vec![1.0, 1.0],
                    estimated_dtoh_bytes: leaf.len()
                        * k
                        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    ..CompactTopKBatch::default()
                },
                LeafProfile {
                    leaves: 1,
                    points: leaf.len(),
                    ..LeafProfile::default()
                },
            ))
        }

        fn compute_leaf_topk_l2_batch(
            &self,
            _dataset: &dyn PointStore,
            leaves: &[Vec<u32>],
            params: &PiPNNParams,
            _block_rows: usize,
        ) -> AnnResult<(Vec<CompactTopKBatch>, LeafProfile)> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            self.max_batch_leaves
                .fetch_max(leaves.len(), Ordering::Relaxed);
            if self.batch_sleep > Duration::ZERO {
                std::thread::sleep(self.batch_sleep);
            }
            let mut batches = Vec::with_capacity(leaves.len());
            let mut profile = LeafProfile::default();
            for leaf in leaves {
                let k = params.leaf_knn.min(leaf.len().saturating_sub(1));
                batches.push(CompactTopKBatch {
                    point_ids: leaf.to_vec(),
                    rows: leaf.len(),
                    k,
                    topk_ids: vec![leaf[1], leaf[0]],
                    topk_dists: vec![1.0, 1.0],
                    estimated_dtoh_bytes: leaf.len()
                        * k
                        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    ..CompactTopKBatch::default()
                });
                profile.leaves += usize::from(!leaf.is_empty());
                profile.points += leaf.len();
            }
            Ok((batches, profile))
        }
    }

    struct FakeCudaBudgetFallbackBackend;

    impl PiPNNComputeBackend for FakeCudaBudgetFallbackBackend {
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
                "fake backend does not implement assignment".to_string(),
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
                "CUDA compact path must not call full-matrix leaf processing".to_string(),
            ))
        }

        fn compute_leaf_topk_l2(
            &self,
            _dataset: &dyn PointStore,
            leaf: &[u32],
            params: &PiPNNParams,
            _block_rows: usize,
        ) -> AnnResult<(CompactTopKBatch, LeafProfile)> {
            let k = params.leaf_knn.min(leaf.len().saturating_sub(1));
            Ok((
                CompactTopKBatch {
                    point_ids: leaf.to_vec(),
                    rows: leaf.len(),
                    k,
                    topk_ids: vec![leaf[1], leaf[0]],
                    topk_dists: vec![1.0, 1.0],
                    estimated_dtoh_bytes: leaf.len()
                        * k
                        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    ..CompactTopKBatch::default()
                },
                LeafProfile {
                    leaves: 1,
                    points: leaf.len(),
                    gpu_workspace_budget_bytes: 128,
                    gpu_workspace_peak_bytes: 256,
                    gpu_workspace_pressure_events: 1,
                    gpu_cpu_fallbacks: 1,
                    ..LeafProfile::default()
                },
            ))
        }
    }

    fn build_test_scheduler_inputs() -> (
        InmemDataset<f32>,
        SketchStore,
        PiPNNParams,
        Vec<Mutex<HashPruneReservoir>>,
    ) {
        let dataset = InmemDataset::new(4, 1.0, 1).expect("dataset");
        let sketches = SketchStore::from_test_data(vec![0.0; 4], 1);
        let params = PiPNNParams::default();
        let reservoirs = (0..4)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        (dataset, sketches, params, reservoirs)
    }

    fn single_thread_budget() -> SchedulerBudget {
        SchedulerBudget {
            worker_count: 1,
            leaf_backlog_capacity: 1,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        }
    }

    fn producer_active_budget() -> SchedulerBudget {
        SchedulerBudget {
            worker_count: 2,
            leaf_backlog_capacity: 1,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        }
    }

    #[test]
    fn backlog_full_drains_queued_leaf_before_current_leaf() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (profile, telemetry) = run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                Ok(())
            },
        )
        .expect("scheduler should complete");

        assert_eq!(observer.started(), vec![1, 2]);
        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 2);
        assert_eq!(telemetry.peak_leaf_backlog, 1);
        assert_eq!(telemetry.inline_leaf_fallbacks, 0);
        assert_eq!(telemetry.producer_help_drains, 1);
        assert_eq!(telemetry.producer_help_yields, 0);
    }

    #[test]
    fn scheduler_wait_until_idle_completes_backlog_without_deadlock() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = Arc::new(RecordingObserver::default());
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (_profile, telemetry) = run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(
                &store,
                &sketches,
                &params,
                &edge_sink,
                Some(observer.as_ref()),
            ),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                emitter.emit_leaf(vec![3])?;
                Ok(())
            },
        )
        .expect("scheduler should drain backlog");

        assert_eq!(observer.started(), vec![1, 2, 3]);
        assert!(telemetry.peak_leaf_backlog >= 1);
    }

    #[test]
    fn producer_active_backlog_full_still_drains_inline() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (_profile, telemetry) = run_scheduler_scope(
            &pool,
            producer_active_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                Ok(())
            },
        )
        .expect("scheduler should complete");

        assert_eq!(observer.started(), vec![1, 2]);
        assert_eq!(
            telemetry.producer_help_drains, 1,
            "backlog-full producer path must free one slot inline instead of relying on Rayon yield"
        );
    }

    #[test]
    fn telemetry_defaults_remain_zero_without_work() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = PipnnScheduler::new(
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        assert_eq!(
            scheduler.telemetry(),
            SchedulerTelemetry {
                peak_inflight_leaf_tasks: 0,
                peak_leaf_backlog: 0,
                inline_leaf_fallbacks: 0,
                producer_help_drains: 0,
                producer_help_yields: 0,
                large_leaf_count: 0,
                large_leaf_block_tasks: 0,
                ..SchedulerTelemetry::default()
            }
        );
    }

    #[test]
    fn producer_active_without_backlog_spawns_no_leaf_drainers() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = PipnnScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(0), 0);
    }

    #[test]
    fn producer_active_with_backlog_keeps_workers_reserved_for_rbc() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = PipnnScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let target = scheduler.spawned_leaf_worker_limit_for_backlog(64);
        assert!(target > 0, "expected some leaf workers once backlog exists");
        assert!(
            target < 41,
            "expected producer-active mode to reserve workers for RBC, got {target}"
        );
    }

    #[test]
    fn producer_active_with_backlog_can_still_use_most_workers_for_leaves() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = PipnnScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let target = scheduler.spawned_leaf_worker_limit_for_backlog(64);
        assert!(
            target >= 28,
            "expected producer-active mode to leave only a modest reserve for RBC, got {target}"
        );
    }

    #[test]
    fn producer_finished_can_use_full_leaf_worker_capacity() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = PipnnScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        scheduler.mark_producer_finished();

        assert_eq!(
            scheduler.spawned_leaf_worker_limit_for_backlog(64),
            scheduler.budget.worker_count.saturating_sub(1)
        );
    }

    #[test]
    fn scheduler_budget_shrinks_backlog_under_tight_memory_budget() {
        let tight = SchedulerBudget::for_memory_budget(42, 3500, 512 * 1024);
        let relaxed = SchedulerBudget::for_memory_budget(42, 3500, usize::MAX);

        assert!(tight.leaf_backlog_capacity < relaxed.leaf_backlog_capacity);
        assert!(tight.leaf_backlog_capacity >= 1);
    }

    #[test]
    fn scheduler_scope_runs_producer_inside_provided_rayon_pool() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let seen_threads = Arc::new(Mutex::new(HashSet::new()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .thread_name(|idx| format!("scheduler-test-{idx}"))
            .build()
            .expect("pool");

        let seen_threads_for_producer = Arc::clone(&seen_threads);
        run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
            move |_emitter| {
                (0..128usize).into_par_iter().for_each(|_| {
                    let name = std::thread::current()
                        .name()
                        .unwrap_or("<unnamed>")
                        .to_string();
                    seen_threads_for_producer.lock().insert(name);
                });
                Ok(())
            },
        )
        .expect("scheduler should complete");

        let seen_threads = seen_threads.lock();
        assert!(
            !seen_threads.is_empty(),
            "expected producer to execute parallel work"
        );
        assert!(
            seen_threads
                .iter()
                .all(|name| name.starts_with("scheduler-test-")),
            "expected nested parallel work to stay inside the provided pool, saw {seen_threads:?}"
        );
    }

    #[test]
    fn pipeline_passthrough_uses_bounded_descriptor_queue_and_processes_leaves() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let budget = SchedulerBudget {
            worker_count: 1,
            leaf_backlog_capacity: 1,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        };

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            budget,
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                Ok(())
            },
        )
        .expect("pipeline scheduler should complete");

        assert_eq!(observer.started(), vec![1, 2]);
        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 2);
        assert_eq!(telemetry.peak_leaf_descriptor_queue, 1);
        assert!(telemetry.producer_blocked_ms > 0);
    }

    #[test]
    fn pipeline_cpu_shadow_profile_counts_each_leaf_once() {
        let (dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        params.m_hash_bits = 1;
        params.leaf_knn = 1;
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::CpuShadow;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let budget = SchedulerBudget {
            worker_count: 1,
            leaf_backlog_capacity: 1,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        };

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            budget,
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                emitter.emit_leaf(vec![2, 3])?;
                Ok(())
            },
        )
        .expect("pipeline scheduler should complete");

        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert_eq!(telemetry.cpu_shadow_unsupported_fallbacks, 0);
    }

    #[test]
    fn pipeline_profile_records_compute_flush_and_legacy_aliases() {
        let (dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        params.m_hash_bits = 1;
        params.leaf_knn = 1;
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::CpuShadow;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                emitter.emit_leaf(vec![2, 3])?;
                Ok(())
            },
        )
        .expect("pipeline scheduler should complete");

        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert_eq!(
            telemetry.compute_wait_ms, telemetry.compute_flush_wall_ms,
            "legacy compute alias should match semantic compute+flush timing"
        );
        assert_eq!(
            telemetry.edge_wait_ms, telemetry.edge_flush_wall_ms,
            "legacy edge alias should match semantic edge flush timing"
        );
        assert!(
            telemetry.stage_wall_ms <= telemetry.staging_wait_ms,
            "semantic stage timing should not exceed legacy staging alias"
        );
        let _ = telemetry.topk_queue_roundtrip_wall_ms;
        let _ = telemetry.edge_queue_roundtrip_wall_ms;
        let _ = telemetry.edge_flush_wall_ms;
        assert_eq!(telemetry.compute_flush_wall_ms, telemetry.compute_wait_ms);
        assert_eq!(telemetry.edge_flush_wall_ms, telemetry.edge_wait_ms);
    }

    #[test]
    fn pipeline_cuda_mode_flushes_compact_topk_from_backend() {
        let (mut dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        dataset.data.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::Cuda;
        params.leaf_knn = 1;
        params.m_hash_bits = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let backend = FakeCudaCompactBackend::default();

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context_with_backend(&backend, &store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                emitter.emit_leaf(vec![2, 3])?;
                Ok(())
            },
        )
        .expect("CUDA compact top-k path should flush through CPU HashPrune");

        assert_eq!(backend.single_calls.load(Ordering::Relaxed), 0);
        assert_eq!(backend.batch_calls.load(Ordering::Relaxed), 1);
        assert_eq!(backend.max_batch_leaves.load(Ordering::Relaxed), 2);
        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert_eq!(telemetry.peak_topk_queue, 1);
        let neighbors: Vec<Vec<u32>> = reservoirs
            .into_iter()
            .map(|reservoir| reservoir.into_inner().into_neighbors())
            .collect();
        assert_eq!(neighbors[0], vec![1]);
        assert_eq!(neighbors[1], vec![0]);
        assert_eq!(neighbors[2], vec![3]);
        assert_eq!(neighbors[3], vec![2]);
    }

    #[test]
    fn pipeline_cuda_profile_records_batch_shape_and_underfill() {
        let (mut dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        dataset.data.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::Cuda;
        params.leaf_knn = 1;
        params.m_hash_bits = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let backend = FakeCudaCompactBackend::default();

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context_with_backend(&backend, &store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                emitter.emit_leaf(vec![2, 3])?;
                Ok(())
            },
        )
        .expect("CUDA compact top-k path should flush through CPU HashPrune");

        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert!(telemetry.cuda_batch_count >= 1);
        assert_eq!(telemetry.cuda_batch_leaves, 2);
        assert_eq!(telemetry.cuda_batch_rows, 4);
        assert!(telemetry.cuda_batch_target_bytes > 0);
        assert!(telemetry.cuda_batch_actual_bytes > 0);
        assert!(telemetry.cuda_batch_underfilled_count >= 1);
        assert!(telemetry.descriptor_queue_empty_count >= 1);
    }

    #[test]
    fn pipeline_cuda_ready_pending_batches_do_not_count_descriptor_queue_empty() {
        let (mut dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        dataset.data.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::Cuda;
        params.leaf_knn = 1;
        params.m_hash_bits = 1;
        params.gpu_workspace_budget_bytes = 2 * std::mem::size_of::<f32>();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let backend = FakeCudaCompactBackend::default();

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context_with_backend(&backend, &store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                emitter.emit_leaf(vec![2, 3])?;
                Ok(())
            },
        )
        .expect("CUDA compact top-k path should flush ready pending batches");

        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 4);
        assert_eq!(telemetry.cuda_batch_count, 2);
        assert_eq!(telemetry.cuda_batch_leaves, 2);
        assert_eq!(telemetry.descriptor_queue_empty_count, 0);
        assert_eq!(telemetry.cuda_batch_underfilled_count, 0);
    }

    #[test]
    fn pipeline_cuda_final_pending_flush_counts_post_producer_drain_time() {
        let (mut dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        dataset.data.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::Cuda;
        params.leaf_knn = 1;
        params.m_hash_bits = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let backend = FakeCudaCompactBackend {
            batch_sleep: Duration::from_millis(5),
            ..FakeCudaCompactBackend::default()
        };

        let (_profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context_with_backend(&backend, &store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                emitter.emit_leaf(vec![2, 3])?;
                Ok(())
            },
        )
        .expect("CUDA compact top-k path should flush final pending batch after producer");

        assert_eq!(telemetry.cuda_batch_count, 1);
        assert_eq!(telemetry.cuda_batch_underfilled_count, 1);
        assert!(telemetry.post_producer_leaf_drain_ms >= 5);
    }

    #[test]
    fn pipeline_cuda_mode_rejects_non_cuda_backend_without_gpu_execution() {
        let (dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::Cuda;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let budget = SchedulerBudget {
            worker_count: 1,
            leaf_backlog_capacity: 1,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        };

        let err = run_pipeline_scheduler_scope(
            &pool,
            budget,
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("CUDA pipeline compute mode"));
        assert!(err.to_string().contains("CUDA compute backend"));
    }

    #[test]
    fn pipeline_cuda_mode_records_gpu_workspace_fallback_without_gpu_execution() {
        let (mut dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        dataset.data.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        params.pipeline_compute_mode = PiPNNPipelineComputeMode::Cuda;
        params.leaf_knn = 1;
        params.m_hash_bits = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let backend = FakeCudaBudgetFallbackBackend;

        let (profile, telemetry) = run_pipeline_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context_with_backend(&backend, &store, &sketches, &params, &edge_sink, None),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![0, 1])?;
                Ok(())
            },
        )
        .expect("CUDA budget fallback should still flush compact CPU-shadow top-k");

        assert_eq!(profile.gpu_workspace_budget_bytes, 128);
        assert_eq!(profile.gpu_workspace_peak_bytes, 256);
        assert_eq!(profile.gpu_workspace_pressure_events, 1);
        assert_eq!(profile.gpu_cpu_fallbacks, 1);
        assert_eq!(telemetry.gpu_workspace_pressure_events, 1);
        assert_eq!(telemetry.gpu_cpu_fallbacks, 1);
    }
}
