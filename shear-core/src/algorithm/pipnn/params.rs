use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

use super::dco_sq_int8::SqInt8CpuKernel;
use super::root_risk_budget::RootRiskSigmaMode;
use crate::common::{AnnError, AnnResult, Metric};

static AUTO_OOM_MEMORY_BUDGET_BYTES: OnceLock<usize> = OnceLock::new();

fn percent_of(bytes: usize, percent: usize) -> usize {
    ((bytes as u128).saturating_mul(percent as u128) / 100) as usize
}

fn select_auto_oom_memory_budget_bytes(
    host_total_bytes: usize,
    host_available_bytes: usize,
    cgroup_limit_bytes: Option<usize>,
    cgroup_current_bytes: Option<usize>,
) -> usize {
    if host_total_bytes == 0 {
        return PiPNNParams::DEFAULT_OOM_MEMORY_BUDGET_BYTES;
    }

    let bounded_cgroup_limit = cgroup_limit_bytes.filter(|limit| *limit < host_total_bytes);
    let effective_total = bounded_cgroup_limit.unwrap_or(host_total_bytes);
    let effective_available = match bounded_cgroup_limit {
        Some(limit) => cgroup_current_bytes
            .map(|current| host_available_bytes.min(limit.saturating_sub(current)))
            .unwrap_or(0),
        None => host_available_bytes,
    };

    let total_cap = percent_of(effective_total, 80);
    let available_cap = percent_of(effective_available, 90);
    total_cap.min(available_cap)
}

fn read_memory_value(path: &Path) -> Option<usize> {
    let value = fs::read_to_string(path).ok()?;
    let value = value.trim();
    if value == "max" {
        return None;
    }
    value.parse::<u64>().ok()?.try_into().ok()
}

fn host_memory_bytes() -> Option<(usize, usize)> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kib = None;
    let mut available_kib = None;
    for line in meminfo.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("MemTotal:") => total_kib = fields.next()?.parse::<usize>().ok(),
            Some("MemAvailable:") => available_kib = fields.next()?.parse::<usize>().ok(),
            _ => {}
        }
    }
    let total = total_kib?.saturating_mul(1024);
    let available = available_kib.unwrap_or(total_kib?).saturating_mul(1024);
    Some((total, available))
}

fn cgroup_memory_file_paths(contents: &str) -> Option<(PathBuf, PathBuf)> {
    let mut unified_path = None;
    for line in contents.lines() {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?.trim_start_matches('/');
        if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            let base = Path::new("/sys/fs/cgroup/memory").join(path);
            return Some((
                base.join("memory.limit_in_bytes"),
                base.join("memory.usage_in_bytes"),
            ));
        }
        if hierarchy == "0" && controllers.is_empty() {
            let base = Path::new("/sys/fs/cgroup").join(path);
            unified_path = Some((base.join("memory.max"), base.join("memory.current")));
        }
    }
    unified_path
}

fn cgroup_memory_bytes() -> (Option<usize>, Option<usize>) {
    if let Some((limit_path, current_path)) = fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|contents| cgroup_memory_file_paths(&contents))
    {
        let limit = read_memory_value(&limit_path);
        let current = read_memory_value(&current_path);
        if limit.is_some() || current.is_some() {
            return (limit, current);
        }
    }

    for (limit_path, current_path) in [
        (
            Path::new("/sys/fs/cgroup/memory.max"),
            Path::new("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path::new("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path::new("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ] {
        let limit = read_memory_value(limit_path);
        let current = read_memory_value(current_path);
        if limit.is_some() || current.is_some() {
            return (limit, current);
        }
    }
    (None, None)
}

fn auto_oom_memory_budget_bytes() -> usize {
    *AUTO_OOM_MEMORY_BUDGET_BYTES.get_or_init(|| {
        let Some((host_total, host_available)) = host_memory_bytes() else {
            return PiPNNParams::DEFAULT_OOM_MEMORY_BUDGET_BYTES;
        };
        let (cgroup_limit, cgroup_current) = cgroup_memory_bytes();
        select_auto_oom_memory_budget_bytes(
            host_total,
            host_available,
            cgroup_limit,
            cgroup_current,
        )
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RootFanoutPolicy {
    #[default]
    Fixed,
    CellRadiusCertified,
    RiskBudgeted,
}

impl RootFanoutPolicy {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::CellRadiusCertified => "cell-radius-certified",
            Self::RiskBudgeted => "risk-budgeted",
        }
    }
}

impl std::fmt::Display for RootFanoutPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RootFanoutPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fixed" => Ok(Self::Fixed),
            "cell-radius-certified" | "cell_radius_certified" | "cell-radius" => {
                Ok(Self::CellRadiusCertified)
            }
            "risk-budgeted"
            | "risk_budgeted"
            | "risk-budgeted-core-drop"
            | "risk-budgeted-root-assign" => Ok(Self::RiskBudgeted),
            other => Err(format!(
                "Unsupported root fanout policy '{other}', expected one of: fixed, cell-radius-certified, risk-budgeted"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNComputeBackendKind {
    #[default]
    Cpu,
    Auto,
    Cuda,
}

impl PiPNNComputeBackendKind {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for PiPNNComputeBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNComputeBackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "Unsupported PiPNN backend '{other}', expected one of: cpu, auto, cuda"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeExecutionMode {
    #[default]
    FusedPersistent,
    TwoStage,
}

impl PiPNNGpuSubtreeExecutionMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FusedPersistent => "fused-persistent",
            Self::TwoStage => "two-stage",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeExecutionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fused-persistent" | "fused_persistent" | "persistent" | "fused" => {
                Ok(Self::FusedPersistent)
            }
            "two-stage" | "two_stage" | "twostage" => Ok(Self::TwoStage),
            other => Err(format!(
                "Unsupported GPU subtree execution mode '{other}', expected one of: fused-persistent, two-stage"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeLeafTopKBackend {
    #[default]
    PersistentCustom,
    PersistentTiledGemmFp32,
    PersistentBlockselectFp32,
    PersistentWmmaTf32,
    CuvsBruteforce,
    CublasGemm,
}

impl PiPNNGpuSubtreeLeafTopKBackend {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PersistentCustom => "persistent-custom",
            Self::PersistentTiledGemmFp32 => "persistent-tiled-gemm-fp32",
            Self::PersistentBlockselectFp32 => "persistent-blockselect-fp32",
            Self::PersistentWmmaTf32 => "persistent-wmma-tf32",
            Self::CuvsBruteforce => "cuvs-bruteforce",
            Self::CublasGemm => "cublas-gemm",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeLeafTopKBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeLeafTopKBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "persistent-custom" | "persistent_custom" | "persistent" => Ok(Self::PersistentCustom),
            "persistent-tiled-gemm-fp32"
            | "persistent_tiled_gemm_fp32"
            | "persistent-tiled-gemm"
            | "persistent_tiled_gemm" => Ok(Self::PersistentTiledGemmFp32),
            "persistent-blockselect-fp32"
            | "persistent_blockselect_fp32"
            | "persistent-blockselect"
            | "persistent_blockselect" => Ok(Self::PersistentBlockselectFp32),
            "persistent-wmma-tf32"
            | "persistent_wmma_tf32"
            | "persistent-wmma"
            | "persistent_wmma" => Ok(Self::PersistentWmmaTf32),
            "cuvs-bruteforce"
            | "cuvs_bruteforce"
            | "persistent-cuvs-bruteforce"
            | "persistent_cuvs_bruteforce" => Ok(Self::CuvsBruteforce),
            "cublas-gemm" | "cublas_gemm" | "cublas-gemm-leaf-topk" | "cublas_gemm_leaf_topk" => {
                Ok(Self::CublasGemm)
            }
            other => Err(format!(
                "Unsupported GPU subtree leaf top-k backend '{other}', expected one of: persistent-custom, persistent-tiled-gemm-fp32, persistent-blockselect-fp32, persistent-wmma-tf32, cuvs-bruteforce, cublas-gemm"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeAssignmentBackend {
    #[default]
    Current,
    PersistentCustomFp32,
    PersistentTiledGemmFp32,
    PersistentWmmaTf32,
    PersistentCuvsBruteforceFp32,
}

impl PiPNNGpuSubtreeAssignmentBackend {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::PersistentCustomFp32 => "persistent-custom-fp32",
            Self::PersistentTiledGemmFp32 => "persistent-tiled-gemm-fp32",
            Self::PersistentWmmaTf32 => "persistent-wmma-tf32",
            Self::PersistentCuvsBruteforceFp32 => "persistent-cuvs-bruteforce-fp32",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeAssignmentBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeAssignmentBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "current" | "default" => Ok(Self::Current),
            "persistent-custom-fp32"
            | "persistent_custom_fp32"
            | "persistent-custom"
            | "persistent_custom" => Ok(Self::PersistentCustomFp32),
            "persistent-tiled-gemm-fp32"
            | "persistent_tiled_gemm_fp32"
            | "persistent-tiled-gemm"
            | "persistent_tiled_gemm" => Ok(Self::PersistentTiledGemmFp32),
            "persistent-wmma-tf32"
            | "persistent_wmma_tf32"
            | "persistent-wmma"
            | "persistent_wmma" => Ok(Self::PersistentWmmaTf32),
            "persistent-cuvs-bruteforce-fp32"
            | "persistent_cuvs_bruteforce_fp32"
            | "persistent-cuvs-bruteforce"
            | "persistent_cuvs_bruteforce"
            | "cuvs-bruteforce"
            | "cuvs_bruteforce" => Ok(Self::PersistentCuvsBruteforceFp32),
            other => Err(format!(
                "Unsupported GPU subtree assignment backend '{other}', expected one of: current, persistent-custom-fp32, persistent-tiled-gemm-fp32, persistent-wmma-tf32, persistent-cuvs-bruteforce-fp32"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeDcoMode {
    #[default]
    None,
    PrefixProjection,
    DdcRes,
    Quantized,
    WmmaFilter,
    SqInt8Filter,
    RabitqWmmaFilter,
    RabitqSplitFilter,
    ShapeAware,
}

impl PiPNNGpuSubtreeDcoMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PrefixProjection => "prefix-projection",
            Self::DdcRes => "ddc-res",
            Self::Quantized => "quantized",
            Self::WmmaFilter => "wmma-filter",
            Self::SqInt8Filter => "sq-int8-filter",
            Self::RabitqWmmaFilter => "rabitq-wmma-filter",
            Self::RabitqSplitFilter => "rabitq-split-filter",
            Self::ShapeAware => "shape-aware",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeDcoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeDcoMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "disabled" => Ok(Self::None),
            "prefix-projection" | "prefix_projection" | "prefix" => Ok(Self::PrefixProjection),
            "ddc-res" | "ddc_res" | "ddcres" => Ok(Self::DdcRes),
            "projection-ddc" | "projection_ddc" | "ddc" => Err(
                "GPU subtree DCO mode 'projection-ddc' is reserved for the retired local prefix-projection prototype name; use ddc-res for benchmark-faithful ResInfer DDC_res or prefix-projection for the local prototype".to_string(),
            ),
            "quantized" | "quant" => Ok(Self::Quantized),
            "wmma-filter" | "wmma_filter" | "wmma" => Ok(Self::WmmaFilter),
            "sq-int8-filter" | "sq_int8_filter" | "sq-int8" | "sq_int8" => {
                Ok(Self::SqInt8Filter)
            }
            "rabitq-wmma-filter"
            | "rabitq_wmma_filter"
            | "rabitq-filter"
            | "rabitq_filter"
            | "rabitq" => Ok(Self::RabitqWmmaFilter),
            "rabitq-split-filter" | "rabitq_split_filter" | "rabitq-split" | "rabitq_split" => {
                Ok(Self::RabitqSplitFilter)
            }
            "shape-aware" | "shape_aware" | "shape" => Ok(Self::ShapeAware),
            other => Err(format!(
                "Unsupported GPU subtree DCO mode '{other}', expected one of: none, prefix-projection, ddc-res, quantized, wmma-filter, sq-int8-filter, rabitq-wmma-filter, rabitq-split-filter, shape-aware"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeDcoRerankMode {
    #[default]
    Exact,
    ExactStamp,
    None,
}

impl PiPNNGpuSubtreeDcoRerankMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ExactStamp => "exact-stamp",
            Self::None => "none",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeDcoRerankMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeDcoRerankMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "exact" | "on" | "rerank" => Ok(Self::Exact),
            "exact-stamp"
            | "exact_stamp"
            | "stamp"
            | "no-rerank-exact-stamp"
            | "no_rerank_exact_stamp" => Ok(Self::ExactStamp),
            "none" | "off" | "no-rerank" | "no_rerank" => Ok(Self::None),
            other => Err(format!(
                "Unsupported GPU subtree DCO rerank mode '{other}', expected one of: exact, exact-stamp, none"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNPartitionerKind {
    #[default]
    Rbc,
    KMeans,
    GpuSubtree,
}

impl PiPNNPartitionerKind {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rbc => "rbc",
            Self::KMeans => "kmeans",
            Self::GpuSubtree => "gpu-subtree",
        }
    }
}

impl std::fmt::Display for PiPNNPartitionerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNPartitionerKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rbc" => Ok(Self::Rbc),
            "kmeans" | "k-means" | "k_means" => Ok(Self::KMeans),
            "gpu-subtree" | "gpu_subtree" => Ok(Self::GpuSubtree),
            other => Err(format!(
                "Unsupported PiPNN partitioner '{other}', expected one of: rbc, kmeans, gpu-subtree"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeD0ThroughputModelMode {
    #[default]
    Off,
    Report,
    Enforce,
}

impl PiPNNGpuSubtreeD0ThroughputModelMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Report => "report",
            Self::Enforce => "enforce",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeD0ThroughputModelMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeD0ThroughputModelMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "report" | "report-only" | "report_only" => Ok(Self::Report),
            "enforce" | "enabled" | "on" => Ok(Self::Enforce),
            other => Err(format!(
                "Unsupported GPU-subtree D0 throughput model mode '{other}', expected one of: off, report, enforce"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeD1PlusLazySplitMode {
    Off,
    #[default]
    On,
}

impl PiPNNGpuSubtreeD1PlusLazySplitMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeD1PlusLazySplitMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeD1PlusLazySplitMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            other => Err(format!(
                "Unsupported GPU-subtree D1+ Lazy Split mode '{other}', expected one of: off, on"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiPNNGpuSubtreePolicy {
    Off,
    #[default]
    Report,
    Enforce,
}

impl PiPNNGpuSubtreePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Report => "report",
            Self::Enforce => "enforce",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreePolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "report" | "report-only" | "report_only" => Ok(Self::Report),
            "enforce" | "enabled" | "on" => Ok(Self::Enforce),
            other => Err(format!(
                "Unsupported GPU-subtree policy '{other}', expected one of: off, report, enforce"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeD0PartitionMode {
    #[default]
    RbcFanout,
    TagoreKMeansTopK,
    GpuKMeansCoarse,
}

impl PiPNNGpuSubtreeD0PartitionMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RbcFanout => "rbc-fanout",
            Self::TagoreKMeansTopK => "tagore-kmeans-topk",
            Self::GpuKMeansCoarse => "gpu-kmeans-coarse",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeD0PartitionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeD0PartitionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rbc" | "rbc-fanout" | "rbc_fanout" => Ok(Self::RbcFanout),
            "tagore" | "tagore-kmeans-topk" | "tagore_kmeans_topk" | "kmeans-topk"
            | "kmeans_topk" => Ok(Self::TagoreKMeansTopK),
            "gpu-kmeans-coarse" | "gpu_kmeans_coarse" | "gpu-kmeans" | "gpu_kmeans" => {
                Ok(Self::GpuKMeansCoarse)
            }
            other => Err(format!(
                "Unsupported GPU-subtree D0 partition mode '{other}', expected one of: rbc-fanout, tagore-kmeans-topk, gpu-kmeans-coarse"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeD1RootPartitionMode {
    #[default]
    Off,
    KMeansCoarse,
}

impl PiPNNGpuSubtreeD1RootPartitionMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::KMeansCoarse => "kmeans-coarse",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeD1RootPartitionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeD1RootPartitionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" | "rbc" | "rbc-fanout" | "rbc_fanout" => Ok(Self::Off),
            "kmeans-coarse" | "kmeans_coarse" | "gpu-kmeans-coarse" | "gpu_kmeans_coarse"
            | "gpu-kmeans" | "gpu_kmeans" => Ok(Self::KMeansCoarse),
            other => Err(format!(
                "Unsupported GPU-subtree D1 root partition mode '{other}', expected one of: off, kmeans-coarse"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeVectorStagingMode {
    #[default]
    Off,
    ProfileOnly,
    FanoutSlab,
    RollingCache,
}

impl PiPNNGpuSubtreeVectorStagingMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ProfileOnly => "profile-only",
            Self::FanoutSlab => "fanout-slab",
            Self::RollingCache => "rolling-cache",
        }
    }

    #[inline]
    pub fn records_repeat_profile(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeVectorStagingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeVectorStagingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "profile-only" | "profile_only" | "profile" => Ok(Self::ProfileOnly),
            "fanout-slab" | "fanout_slab" | "slab" => Ok(Self::FanoutSlab),
            "rolling-cache" | "rolling_cache" | "rolling" => Ok(Self::RollingCache),
            other => Err(format!(
                "Unsupported GPU-subtree vector staging mode '{other}', expected one of: off, profile-only, fanout-slab, rolling-cache"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeQcodeStagingMode {
    #[default]
    Off,
    RabitqResidual,
}

impl PiPNNGpuSubtreeQcodeStagingMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::RabitqResidual => "rabitq-residual",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeQcodeStagingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeQcodeStagingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "rabitq-residual" | "rabitq_residual" | "residual-rabitq" | "residual_rabitq" => {
                Ok(Self::RabitqResidual)
            }
            other => Err(format!(
                "Unsupported GPU-subtree qcode staging mode '{other}', expected one of: off, rabitq-residual"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSqStagingMode {
    #[default]
    Off,
    GpuD0,
    GpuD0ShardLocalD1,
    GpuD0ClusterBatchD1,
}

impl PiPNNGpuSubtreeSqStagingMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::GpuD0 => "gpu-d0",
            Self::GpuD0ShardLocalD1 => "gpu-d0-shard-local-d1",
            Self::GpuD0ClusterBatchD1 => "gpu-d0-cluster-batch-d1",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSqStagingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSqStagingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "gpu-d0" | "gpu_d0" => Ok(Self::GpuD0),
            "gpu-d0-shard-local-d1" | "gpu_d0_shard_local_d1" => Ok(Self::GpuD0ShardLocalD1),
            "gpu-d0-cluster-batch-d1" | "gpu_d0_cluster_batch_d1" => Ok(Self::GpuD0ClusterBatchD1),
            other => Err(format!(
                "Unsupported GPU-subtree SQ staging mode '{other}', expected one of: off, gpu-d0, gpu-d0-shard-local-d1, gpu-d0-cluster-batch-d1"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeQcodeD1PayloadMode {
    #[default]
    FullResident,
    ChildLocal,
}

impl PiPNNGpuSubtreeQcodeD1PayloadMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullResident => "full-resident",
            Self::ChildLocal => "child-local",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeQcodeD1PayloadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeQcodeD1PayloadMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "full-resident" | "full_resident" | "resident" => Ok(Self::FullResident),
            "child-local" | "child_local" | "local" => Ok(Self::ChildLocal),
            other => Err(format!(
                "Unsupported GPU-subtree qcode D1 payload mode '{other}', expected one of: full-resident, child-local"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeRabitqD1DistanceMode {
    #[default]
    DenseI8Wmma,
    SplitShortEx,
}

impl PiPNNGpuSubtreeRabitqD1DistanceMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DenseI8Wmma => "dense-i8-wmma",
            Self::SplitShortEx => "split-short-ex",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeRabitqD1DistanceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeRabitqD1DistanceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense-i8-wmma" | "dense_i8_wmma" | "wmma" => Ok(Self::DenseI8Wmma),
            "split-short-ex" | "split_short_ex" | "short-ex" | "short_ex" => Ok(Self::SplitShortEx),
            other => Err(format!(
                "Unsupported residual RaBitQ D1 distance mode '{other}', expected one of: dense-i8-wmma, split-short-ex"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeRabitqD0AssignMode {
    #[default]
    RowSerial,
    Tiled,
    Batch,
    WmmaBatch,
}

impl PiPNNGpuSubtreeRabitqD0AssignMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RowSerial => "row-serial",
            Self::Tiled => "tiled",
            Self::Batch => "batch",
            Self::WmmaBatch => "wmma-batch",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeRabitqD0AssignMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeRabitqD0AssignMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "row-serial" | "row_serial" | "serial" | "row" => Ok(Self::RowSerial),
            "tiled" | "tile" => Ok(Self::Tiled),
            "batch" | "batch-distance" | "batch_distance" | "batch-distance-kernel" => {
                Ok(Self::Batch)
            }
            "wmma-batch" | "wmma_batch" | "wmma-batch-distance" | "wmma_batch_distance" => {
                Ok(Self::WmmaBatch)
            }
            other => Err(format!(
                "Unsupported residual RaBitQ D0 assignment mode '{other}', expected one of: row-serial, tiled, batch, wmma-batch"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeRabitqD1TopKMode {
    #[default]
    RowSerial,
    WmmaBatch,
}

impl PiPNNGpuSubtreeRabitqD1TopKMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RowSerial => "row-serial",
            Self::WmmaBatch => "wmma-batch",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeRabitqD1TopKMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeRabitqD1TopKMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "row-serial" | "row_serial" | "serial" | "row" => Ok(Self::RowSerial),
            "wmma-batch" | "wmma_batch" | "wmma-batch-distance" | "wmma_batch_distance" => {
                Ok(Self::WmmaBatch)
            }
            other => Err(format!(
                "Unsupported residual RaBitQ D1+ top-k mode '{other}', expected one of: row-serial, wmma-batch"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeRabitqPlusMode {
    #[default]
    Off,
    Shadow,
    Build,
}

impl PiPNNGpuSubtreeRabitqPlusMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Build => "build",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeRabitqPlusMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeRabitqPlusMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "shadow" | "audit" | "profile" => Ok(Self::Shadow),
            "build" | "quantized-only" | "quantized_only" => Ok(Self::Build),
            other => Err(format!(
                "Unsupported GPU-subtree dense-i8 RabitQ+ mode '{other}', expected one of: off, shadow, build"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeRabitqPlusSidecarMode {
    #[default]
    Off,
    Require,
    CreateOnly,
    Rebuild,
}

impl PiPNNGpuSubtreeRabitqPlusSidecarMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Require => "require",
            Self::CreateOnly => "create-only",
            Self::Rebuild => "rebuild",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeRabitqPlusSidecarMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeRabitqPlusSidecarMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "require" | "required" | "read" | "read-only" | "readonly" => Ok(Self::Require),
            "create-only" | "create_only" | "create" | "generate-only" | "generate_only" => {
                Ok(Self::CreateOnly)
            }
            "rebuild" | "force" | "regenerate" => Ok(Self::Rebuild),
            other => Err(format!(
                "Unsupported GPU-subtree RabitQ sidecar mode '{other}', expected one of: off, require, create-only, rebuild"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSqWmmaMode {
    #[default]
    Off,
    Shadow,
    Build,
}

impl PiPNNGpuSubtreeSqWmmaMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Build => "build",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSqWmmaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSqWmmaMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "shadow" | "audit" | "profile" => Ok(Self::Shadow),
            "build" | "quantized-only" | "quantized_only" => Ok(Self::Build),
            other => Err(format!(
                "Unsupported GPU-subtree SQ+WMMA mode '{other}', expected one of: off, shadow, build"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreePersistentLargeTicketBatch {
    Auto,
    #[default]
    One,
    Two,
    Four,
    Eight,
}

impl PiPNNGpuSubtreePersistentLargeTicketBatch {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::One => "1",
            Self::Two => "2",
            Self::Four => "4",
            Self::Eight => "8",
        }
    }

    #[inline]
    pub(crate) fn explicit_batch(self) -> Option<u32> {
        match self {
            Self::Auto => None,
            Self::One => Some(1),
            Self::Two => Some(2),
            Self::Four => Some(4),
            Self::Eight => Some(8),
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreePersistentLargeTicketBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreePersistentLargeTicketBatch {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            "4" => Ok(Self::Four),
            "8" => Ok(Self::Eight),
            other => Err(format!(
                "Unsupported persistent large-ticket batch '{other}', expected one of: auto, 1, 2, 4, 8"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreePersistentSchedulerMode {
    Auto,
    #[default]
    Baseline,
    LocalCounters,
    LocalCountersBarrier,
    AdaptiveIdle,
}

impl PiPNNGpuSubtreePersistentSchedulerMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Baseline => "baseline",
            Self::LocalCounters => "local-counters",
            Self::LocalCountersBarrier => "local-counters-barrier",
            Self::AdaptiveIdle => "adaptive-idle",
        }
    }

    #[inline]
    pub(crate) fn resolved(self) -> Self {
        match self {
            Self::Auto => Self::Baseline,
            other => other,
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreePersistentSchedulerMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreePersistentSchedulerMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "baseline" => Ok(Self::Baseline),
            "local-counters" | "local_counters" => Ok(Self::LocalCounters),
            "local-counters-barrier" | "local_counters_barrier" => Ok(Self::LocalCountersBarrier),
            "adaptive-idle" | "adaptive_idle" => Ok(Self::AdaptiveIdle),
            other => Err(format!(
                "Unsupported persistent scheduler mode '{other}', expected one of: auto, baseline, local-counters, local-counters-barrier, adaptive-idle"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSq8D0AssignmentKernel {
    #[default]
    Auto,
    TiledScalar,
    WmmaBatch,
}

impl PiPNNGpuSubtreeSq8D0AssignmentKernel {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::TiledScalar => "tiled-scalar",
            Self::WmmaBatch => "wmma-batch",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSq8D0AssignmentKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSq8D0AssignmentKernel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "tiled-scalar" | "tiled_scalar" | "scalar" => Ok(Self::TiledScalar),
            "wmma-batch" | "wmma_batch" | "wmma" => Ok(Self::WmmaBatch),
            other => Err(format!(
                "Unsupported SQ8 D0 assignment kernel '{other}', expected one of: auto, tiled-scalar, wmma-batch"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSq8AssignmentKernel {
    Auto,
    #[default]
    WarpDp4a,
    Wmma,
}

impl PiPNNGpuSubtreeSq8AssignmentKernel {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::WarpDp4a => "warp-dp4a",
            Self::Wmma => "wmma",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSq8AssignmentKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSq8AssignmentKernel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "warp-dp4a" | "warp_dp4a" | "dp4a" => Ok(Self::WarpDp4a),
            "wmma" => Ok(Self::Wmma),
            other => Err(format!(
                "Unsupported SQ8 assignment kernel '{other}', expected one of: auto, warp-dp4a, wmma"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSq8LeafExecutor {
    #[default]
    Auto,
    WarpDp4a,
    Wmma,
}

impl PiPNNGpuSubtreeSq8LeafExecutor {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::WarpDp4a => "warp-dp4a",
            Self::Wmma => "wmma",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSq8LeafExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSq8LeafExecutor {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "warp-dp4a" | "warp_dp4a" | "dp4a" => Ok(Self::WarpDp4a),
            "wmma" => Ok(Self::Wmma),
            other => Err(format!(
                "Unsupported SQ8 leaf executor '{other}', expected one of: auto, warp-dp4a, wmma"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSq8LeafMode {
    Auto,
    #[default]
    Staged,
    RowOwned,
    D96K2,
}

impl PiPNNGpuSubtreeSq8LeafMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Staged => "staged",
            Self::RowOwned => "row-owned",
            Self::D96K2 => "d96-k2",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSq8LeafMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSq8LeafMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "staged" => Ok(Self::Staged),
            "row-owned" | "row_owned" => Ok(Self::RowOwned),
            "d96-k2" | "d96_k2" => Ok(Self::D96K2),
            other => Err(format!(
                "Unsupported SQ8 leaf mode '{other}', expected one of: auto, staged, row-owned, d96-k2"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeSqWmmaSidecarMode {
    #[default]
    Off,
    Require,
    CreateOnly,
    Rebuild,
}

impl PiPNNGpuSubtreeSqWmmaSidecarMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Require => "require",
            Self::CreateOnly => "create-only",
            Self::Rebuild => "rebuild",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeSqWmmaSidecarMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeSqWmmaSidecarMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "require" | "required" | "read" | "read-only" | "readonly" => Ok(Self::Require),
            "create-only" | "create_only" | "create" | "generate-only" | "generate_only" => {
                Ok(Self::CreateOnly)
            }
            "rebuild" | "force" | "regenerate" => Ok(Self::Rebuild),
            other => Err(format!(
                "Unsupported GPU-subtree SQ+WMMA sidecar mode '{other}', expected one of: off, require, create-only, rebuild"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeTurboQuantMode {
    #[default]
    Off,
    Shadow,
    Build,
}

impl PiPNNGpuSubtreeTurboQuantMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Build => "build",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeTurboQuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeTurboQuantMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "shadow" | "audit" | "profile" => Ok(Self::Shadow),
            "build" | "quantized-only" | "quantized_only" => Ok(Self::Build),
            other => Err(format!(
                "Unsupported GPU-subtree TurboQuant-MSE4 mode '{other}', expected one of: off, shadow, build"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeTurboQuantSidecarMode {
    #[default]
    Off,
    Require,
    CreateOnly,
    Rebuild,
}

impl PiPNNGpuSubtreeTurboQuantSidecarMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Require => "require",
            Self::CreateOnly => "create-only",
            Self::Rebuild => "rebuild",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeTurboQuantSidecarMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeTurboQuantSidecarMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "require" | "required" | "read" | "read-only" | "readonly" => Ok(Self::Require),
            "create-only" | "create_only" | "create" | "generate-only" | "generate_only" => {
                Ok(Self::CreateOnly)
            }
            "rebuild" | "force" | "regenerate" => Ok(Self::Rebuild),
            other => Err(format!(
                "Unsupported GPU-subtree TurboQuant-MSE4 sidecar mode '{other}', expected one of: off, require, create-only, rebuild"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeRabitqPlusRotator {
    #[default]
    FhtKac,
    None,
}

impl PiPNNGpuSubtreeRabitqPlusRotator {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FhtKac => "fht-kac",
            Self::None => "none",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeRabitqPlusRotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeRabitqPlusRotator {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fht-kac" | "fht_kac" | "fht" | "kac" => Ok(Self::FhtKac),
            "none" | "off" | "identity" => Ok(Self::None),
            other => Err(format!(
                "Unsupported GPU-subtree dense-i8 RabitQ+ rotator '{other}', expected one of: fht-kac, none"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNGpuSubtreeChildScheduleMode {
    #[default]
    Natural,
    OverlapGreedy,
}

impl PiPNNGpuSubtreeChildScheduleMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::OverlapGreedy => "overlap-greedy",
        }
    }
}

impl std::fmt::Display for PiPNNGpuSubtreeChildScheduleMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNGpuSubtreeChildScheduleMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "natural" | "none" | "off" => Ok(Self::Natural),
            "overlap-greedy" | "overlap_greedy" | "greedy" => Ok(Self::OverlapGreedy),
            other => Err(format!(
                "Unsupported GPU-subtree child schedule mode '{other}', expected one of: natural, overlap-greedy"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNOomIoMode {
    #[default]
    Auto,
    Direct,
    Buffered,
}

impl PiPNNOomIoMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Direct => "direct",
            Self::Buffered => "buffered",
        }
    }
}

impl std::fmt::Display for PiPNNOomIoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNOomIoMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "direct" | "strict" | "strict-direct" | "strict_direct" | "odirect" | "o_direct" => {
                Ok(Self::Direct)
            }
            "buffered" | "buffer" | "io-buffer" | "io_buffer" | "page-cache" | "page_cache"
            | "cached" => Ok(Self::Buffered),
            other => Err(format!(
                "Unsupported OOM I/O mode '{other}', expected one of: auto, direct, buffered"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNOnlineSuffixReduceMode {
    #[default]
    Off,
    DirectFirst,
    WritebackCache,
    LogAssisted,
}

impl PiPNNOnlineSuffixReduceMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::DirectFirst => "direct-first",
            Self::WritebackCache => "writeback-cache",
            Self::LogAssisted => "log-assisted",
        }
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    #[inline]
    pub fn uses_writeback_cache(self) -> bool {
        matches!(self, Self::WritebackCache)
    }

    #[inline]
    pub fn uses_log_assisted(self) -> bool {
        matches!(self, Self::LogAssisted)
    }

    #[inline]
    pub fn uses_state_cache(self) -> bool {
        matches!(self, Self::WritebackCache | Self::LogAssisted)
    }
}

impl std::fmt::Display for PiPNNOnlineSuffixReduceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNOnlineSuffixReduceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "direct-first" | "direct_first" | "direct" => Ok(Self::DirectFirst),
            "writeback-cache" | "writeback_cache" | "writeback" => Ok(Self::WritebackCache),
            "log-assisted" | "log_assisted" | "log" => Ok(Self::LogAssisted),
            other => Err(format!(
                "Unsupported online suffix reduce mode '{other}', expected one of: off, direct-first, writeback-cache, log-assisted"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiPNNOnlineSuffixBudgets {
    pub direct_queue_bytes: usize,
    pub state_cache_bytes: usize,
    pub state_page_cache_bytes: usize,
    pub state_writeback_bytes: usize,
    pub spill_buffer_bytes: usize,
    pub reorder_window_bytes: usize,
    pub decode_bytes: usize,
    pub read_window_bytes: usize,
    pub delta_budget_bytes: usize,
    pub delta_buffer_bytes: usize,
    pub compaction_bytes: usize,
}

impl PiPNNOnlineSuffixBudgets {
    #[inline]
    pub fn total_bytes(self) -> usize {
        self.direct_queue_bytes
            .saturating_add(self.state_cache_bytes)
            .saturating_add(self.state_page_cache_bytes)
            .saturating_add(self.state_writeback_bytes)
            .saturating_add(self.spill_buffer_bytes)
            .saturating_add(self.reorder_window_bytes)
            .saturating_add(self.decode_bytes)
            .saturating_add(self.delta_buffer_bytes)
            .saturating_add(self.compaction_bytes)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNPipelineComputeMode {
    #[default]
    Passthrough,
    CpuShadow,
    Cuda,
    Mock,
}

impl PiPNNPipelineComputeMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::CpuShadow => "cpu-shadow",
            Self::Cuda => "cuda",
            Self::Mock => "mock",
        }
    }
}

impl std::fmt::Display for PiPNNPipelineComputeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNPipelineComputeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "passthrough" | "pass-through" => Ok(Self::Passthrough),
            "cpu-shadow" | "cpu_shadow" | "cpushadow" => Ok(Self::CpuShadow),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "mock" => Ok(Self::Mock),
            other => Err(format!(
                "Unsupported PiPNN pipeline compute mode '{other}', expected one of: passthrough, cpu-shadow, cuda, mock"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNPruneBackend {
    #[default]
    HashPrune,
    LRPF,
}

impl PiPNNPruneBackend {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HashPrune => "hashprune",
            Self::LRPF => "lrpf",
        }
    }
}

impl std::fmt::Display for PiPNNPruneBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNPruneBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "hashprune" | "hash-prune" | "hash_prune" => Ok(Self::HashPrune),
            "lrpf" => Ok(Self::LRPF),
            other => Err(format!(
                "Unsupported PiPNN prune backend '{other}', expected one of: hashprune, lrpf"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNHashPruneEvidenceMode {
    #[default]
    Off,
    Report,
    Enforce,
}

impl PiPNNHashPruneEvidenceMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Report => "report",
            Self::Enforce => "enforce",
        }
    }
}

impl std::fmt::Display for PiPNNHashPruneEvidenceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNHashPruneEvidenceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "report" => Ok(Self::Report),
            "enforce" => Ok(Self::Enforce),
            other => Err(format!(
                "Unsupported HashPrune evidence mode '{other}', expected one of: off, report, enforce"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNHashPruneBackfillMode {
    #[default]
    Off,
    DistanceGated,
    DistanceGatedStratified,
    DistanceGatedMultihit,
    DistanceGatedTieredD52,
    DistanceGatedTieredRHalf,
    RHalfHashSlot1,
    RHalfDistanceFirstQ3,
    RHalfSoftTierQ3,
    RHalfHardTierQ3,
}

impl PiPNNHashPruneBackfillMode {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::DistanceGated => "distance-gated",
            Self::DistanceGatedStratified => "distance-gated-stratified",
            Self::DistanceGatedMultihit => "distance-gated-multihit",
            Self::DistanceGatedTieredD52 => "distance-gated-tiered-d52",
            Self::DistanceGatedTieredRHalf => "distance-gated-tiered-rhalf",
            Self::RHalfHashSlot1 => "rhalf-hash-slot-1",
            Self::RHalfDistanceFirstQ3 => "rhalf-distance-first-q3",
            Self::RHalfSoftTierQ3 => "rhalf-soft-tier-q3",
            Self::RHalfHardTierQ3 => "rhalf-hard-tier-q3",
        }
    }
}

impl std::fmt::Display for PiPNNHashPruneBackfillMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNHashPruneBackfillMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "distance-gated" | "distance_gated" => Ok(Self::DistanceGated),
            "distance-gated-stratified" | "distance_gated_stratified" => {
                Ok(Self::DistanceGatedStratified)
            }
            "distance-gated-multihit" | "distance_gated_multihit" => {
                Ok(Self::DistanceGatedMultihit)
            }
            "distance-gated-tiered-d52" | "distance_gated_tiered_d52" => {
                Ok(Self::DistanceGatedTieredD52)
            }
            "distance-gated-tiered-rhalf" | "distance_gated_tiered_rhalf" => {
                Ok(Self::DistanceGatedTieredRHalf)
            }
            "rhalf-hash-slot-1" | "rhalf_hash_slot_1" => Ok(Self::RHalfHashSlot1),
            "rhalf-distance-first-q3" | "rhalf_distance_first_q3" => Ok(Self::RHalfDistanceFirstQ3),
            "rhalf-soft-tier-q3" | "rhalf_soft_tier_q3" => Ok(Self::RHalfSoftTierQ3),
            "rhalf-hard-tier-q3" | "rhalf_hard_tier_q3" => Ok(Self::RHalfHardTierQ3),
            other => Err(format!(
                "Unsupported HashPrune backfill mode '{other}', expected one of: off, distance-gated, distance-gated-stratified, distance-gated-multihit, distance-gated-tiered-d52, distance-gated-tiered-rhalf, rhalf-hash-slot-1, rhalf-distance-first-q3, rhalf-soft-tier-q3, rhalf-hard-tier-q3"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PiPNNHashPruneBackfillAblation {
    #[default]
    None,
    BudgetQuarter,
    NoHalfCap,
    Quota1,
    Quota2,
    DistanceFirst,
    NoVeto,
}

impl PiPNNHashPruneBackfillAblation {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BudgetQuarter => "budget-quarter",
            Self::NoHalfCap => "no-half-cap",
            Self::Quota1 => "quota-1",
            Self::Quota2 => "quota-2",
            Self::DistanceFirst => "distance-first",
            Self::NoVeto => "no-veto",
        }
    }
}

impl std::fmt::Display for PiPNNHashPruneBackfillAblation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PiPNNHashPruneBackfillAblation {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "budget-quarter" | "budget_quarter" => Ok(Self::BudgetQuarter),
            "no-half-cap" | "no_half_cap" => Ok(Self::NoHalfCap),
            "quota-1" | "quota_1" => Ok(Self::Quota1),
            "quota-2" | "quota_2" => Ok(Self::Quota2),
            "distance-first" | "distance_first" => Ok(Self::DistanceFirst),
            "no-veto" | "no_veto" => Ok(Self::NoVeto),
            other => Err(format!(
                "Unsupported HashPrune backfill ablation '{other}', expected one of: none, budget-quarter, no-half-cap, quota-1, quota-2, distance-first, no-veto"
            )),
        }
    }
}

/// PiPNN 构图的超参数集合。
///
/// 这些参数尽量贴近论文中的记号，并给出适合小规模实验的默认值。
#[derive(Clone, Debug)]
pub struct PiPNNParams {
    /// HashPrune 残差哈希的位数（超平面个数），要求 <= 16。
    pub m_hash_bits: usize,
    /// 每个点 HashPrune 水库容量 l_max，对应目标出度上界。
    pub l_max: usize,
    /// Candidate pruning backend. HashPrune is the canonical default; LRPF is opt-in.
    pub prune_backend: PiPNNPruneBackend,
    /// Evidence-Aware HashPrune mode. Off is the strict-order baseline.
    pub hashprune_evidence_mode: PiPNNHashPruneEvidenceMode,
    /// Number of otherwise-unused reservoir slots available to evidence-qualified candidates.
    pub hashprune_evidence_overflow_budget: usize,
    /// Optional bounded fill of otherwise-unused HashPrune reservoir slots.
    pub hashprune_backfill_mode: PiPNNHashPruneBackfillMode,
    /// Closed research ablation for the frozen R-half mode.
    pub hashprune_backfill_ablation: PiPNNHashPruneBackfillAblation,
    /// Maximum emitted distance-gated backup edges per row.
    pub hashprune_backfill_budget: usize,
    /// Euclidean-distance ratio relative to the farthest retained primary.
    pub hashprune_backfill_alpha: f32,
    /// Optional per-hash quota override for named R-half policies.
    pub hashprune_rhalf_hash_quota: Option<usize>,
    /// Optional SoftTier priority alpha override for named R-half policies.
    pub hashprune_soft_tier_priority_alpha: Option<f32>,
    /// Optional maximum number of backfill edges retained per row for named R-half policies.
    pub hashprune_rhalf_max_backfill_per_row: Option<usize>,
    /// Optional LRPF pre-spill compression. Off preserves all LRPF candidates until reduction.
    pub lrpf_pre_spill_compress_enable: bool,
    /// LRPF pre-spill width multiplier. The effective width is clamped to [l_max, 4*l_max].
    pub lrpf_pre_spill_width_multiplier: f32,
    /// Number of nearest candidates protected before LRPF witness pruning.
    pub lrpf_protected_core: usize,
    /// Target LRPF output degree before clamping to `l_max`.
    pub lrpf_target_degree: usize,
    /// Number of farthest candidates reserved as LRPF navigation edges.
    /// The default is 0: long edges survive through witness pruning instead of a hard quota.
    pub lrpf_long_edge_quota: usize,
    /// 叶子最小大小 C_min。
    pub c_min: usize,
    /// 叶子最大大小 C_max。
    pub c_max: usize,
    /// RBC 采样 leader 的比例 P_samp（相对于子问题大小）。
    pub psamp_fraction: f64,
    /// 单层最多 leader 数，用于限制大子问题 leader 数量。
    pub max_leaders: usize,
    /// RBC 允许的最大深度，到达或超过该深度时直接将当前簇视为叶子。
    pub max_depth: usize,
    /// 父子簇规模的最小收缩率阈值。
    /// 当 parent_n / n < min_shrink_ratio 且 depth >= 3 时，认为收缩不明显，直接叶化。
    pub min_shrink_ratio: f32,
    /// 顶层 fanout（为每个点选择多少个最近 leader）。
    pub fanout_top: usize,
    /// 第二层 fanout（depth=1）。
    pub fanout_second: usize,
    /// Root-level fanout policy. Fixed keeps the historical `fanout_top` behavior.
    pub root_fanout_policy: RootFanoutPolicy,
    /// Number of owner-distance samples retained for each root cell radius certificate.
    pub root_radius_k: usize,
    /// Candidate root fanouts tried from smallest to largest by the cell-radius certificate.
    pub root_radius_candidate_rs: Vec<usize>,
    /// Points assigned with fixed fanout before cell-radius certificates are allowed.
    pub root_radius_warmup_points: usize,
    /// Record cell-radius telemetry while emitting fixed root fanout assignments.
    pub root_radius_dry_run: bool,
    /// OOM research mode: stop after depth-0 root materialization and preserve d00 runs.
    pub root_stop_after_d00: bool,
    /// Target average root fanout for budgeted utility root assignment.
    pub root_risk_target_avg_fanout: f32,
    /// Sigma multiplier used by budgeted utility root assignment.
    pub root_risk_sigma_lambda: f32,
    /// Optional explicit budget lambda; unset means calibrate on pilot points.
    pub root_risk_budget_lambda: Option<f32>,
    /// Sigma estimator used by budgeted utility root assignment.
    pub root_risk_sigma_mode: RootRiskSigmaMode,
    /// Points used for risk-budgeted pilot stats and budget calibration.
    pub root_risk_warmup_points: usize,
    /// Candidate root fanouts tried by risk budgeting.
    pub root_risk_candidate_rs: Vec<usize>,
    /// Record risk-budgeted telemetry while still emitting fixed root fanout assignments.
    pub root_risk_dry_run: bool,
    /// Experimental opt-in for full builds using risk-budgeted root assignments.
    pub root_risk_allow_full_build: bool,
    /// Streaming partition channel. RBC remains the default; k-means is an experimental
    /// low-replication channel.
    pub partitioner: PiPNNPartitionerKind,
    /// Backend used only for GPU-subtree depth-0/root assignment. `Cpu` preserves the historical
    /// OOM root spool path; `Cuda` routes root point-to-leader assignment through the CUDA GEMM
    /// compute backend while preserving the same D0 child-run format.
    pub gpu_subtree_root_assignment_backend: PiPNNComputeBackendKind,
    /// Fixed GPU subtree RBC assignment operator. Non-current variants are reserved until their
    /// resident device kernels are implemented and must not silently fall back.
    pub gpu_subtree_assignment_backend: PiPNNGpuSubtreeAssignmentBackend,
    /// Candidate-pruning mode for the GPU subtree assignment operator. `none` preserves the
    /// current exact assignment path.
    pub gpu_subtree_assignment_dco_mode: PiPNNGpuSubtreeDcoMode,
    /// Whether assignment DCO survivors are exact-reranked before child assignment.
    pub gpu_subtree_assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    /// Projection dimension or code width used by assignment DCO modes that need one.
    pub gpu_subtree_assignment_dco_proj_dim: usize,
    /// Maximum assignment DCO survivors kept per row before exact rerank or approximate output.
    pub gpu_subtree_assignment_dco_survivors: usize,
    /// Enable assignment DCO audit telemetry for false-prune sampling.
    pub gpu_subtree_assignment_dco_audit: bool,
    /// Candidate-pruning mode for GPU-subtree D0/root assignment. This is independent from the
    /// D1+ assignment DCO switch because it consumes root child-run assignment records.
    pub gpu_subtree_d0_dco_mode: PiPNNGpuSubtreeDcoMode,
    /// Whether D0/root DCO survivors are exact-reranked before child-run assignment.
    pub gpu_subtree_d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    /// Maximum D0/root DCO survivors kept per row before exact rerank or approximate output.
    pub gpu_subtree_d0_dco_survivors: usize,
    /// Enable D0/root DCO false-prune audit telemetry.
    pub gpu_subtree_d0_dco_audit: bool,
    /// CPU kernel used by D0 SQ-int8 approximate assignment.
    pub gpu_subtree_d0_sq_cpu_kernel: SqInt8CpuKernel,
    /// GPU subtree resident execution schedule. `fused-persistent` preserves the current single
    /// kernel pipeline; `two-stage` partitions first and runs leaf top-k in a second kernel.
    pub gpu_subtree_execution_mode: PiPNNGpuSubtreeExecutionMode,
    /// Persistent GPU subtree leaf top-k operator.
    pub gpu_subtree_leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend,
    /// Candidate-pruning mode for the GPU subtree leaf top-k operator. `none` preserves the
    /// current exact top-k path.
    pub gpu_subtree_leaf_dco_mode: PiPNNGpuSubtreeDcoMode,
    /// Whether leaf DCO survivors are exact-reranked before final top-k emission.
    pub gpu_subtree_leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode,
    /// Projection dimension or code width used by leaf DCO modes that need one.
    pub gpu_subtree_leaf_dco_proj_dim: usize,
    /// Maximum leaf DCO survivors kept per row before exact rerank or approximate output.
    pub gpu_subtree_leaf_dco_survivors: usize,
    /// Enable leaf DCO audit telemetry for false-prune sampling.
    pub gpu_subtree_leaf_dco_audit: bool,
    /// ResInfer DDC_res PCA matrix sidecar. This is required for faithful DDC_res.
    pub gpu_subtree_leaf_dco_ddc_res_pca_matrix: Option<PathBuf>,
    /// Confirm that the input data is already in the PCA coordinate system used by the sidecar.
    pub gpu_subtree_leaf_dco_ddc_res_projected_input: bool,
    /// ResInfer DDC_res incremental block size (`base_dim` in the reference code).
    pub gpu_subtree_leaf_dco_ddc_res_base_dim: usize,
    /// ResInfer DDC_res bound width (`sigma_count` in the reference code).
    pub gpu_subtree_leaf_dco_ddc_res_sigma_count: f32,
    /// Experimental dense-i8 RabitQ+ resident path. Off preserves raw f32 resident upload.
    pub gpu_subtree_rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode,
    /// Total dense-i8 RabitQ+ code bits used by the dense i8 resident code path.
    pub gpu_subtree_rabitq_plus_bits: usize,
    /// Deterministic rotation used before dense-i8 RabitQ+ quantization.
    pub gpu_subtree_rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    /// Seed used only by deterministic dense-i8 RabitQ+ quantization/sidecar rotation.
    pub gpu_subtree_rabitq_plus_seed: u64,
    /// Optional global dense-i8 RabitQ+ sidecar prefix used by shadow/build modes.
    pub gpu_subtree_rabitq_plus_sidecar_prefix: Option<PathBuf>,
    /// How RabitQ sidecar files are created or consumed.
    pub gpu_subtree_rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode,
    /// Experimental global-D0 qcode staging path. Off preserves the historical child-sidecar
    /// pipeline. v1 accepts only residual RabitQ.
    pub gpu_subtree_qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode,
    /// How D1+ consumes residual RaBitQ qcodes. Full-resident preserves the v1 global payload
    /// upload; child-local uploads only the current child's qcode rows.
    pub gpu_subtree_qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode,
    /// Distance kernel/data path used by residual RaBitQ D1+.
    pub gpu_subtree_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode,
    /// Candidate survivor count from the 1-bit short-code filter before ex-bit refine.
    pub gpu_subtree_rabitq_short_survivors: Option<usize>,
    /// Number of residual k-means centroids used by the RabitQ residual qcode staging path.
    pub gpu_subtree_rabitq_residual_centroids: usize,
    /// Raw rows uploaded per staging shard before GPU residual RabitQ quantization and D0
    /// assignment.
    pub gpu_subtree_qcode_staging_upload_shard_rows: usize,
    /// Optional global D0 leader count for qcode staging. Zero uses the staging planner default.
    pub gpu_subtree_qcode_global_d0_leaders: usize,
    /// Experimental residual RaBitQ global-D0 assignment kernel mode.
    pub gpu_subtree_rabitq_d0_assign_mode: PiPNNGpuSubtreeRabitqD0AssignMode,
    /// Experimental residual RaBitQ D1+ child top-k kernel mode.
    pub gpu_subtree_rabitq_d1_topk_mode: PiPNNGpuSubtreeRabitqD1TopKMode,
    /// Experimental SQ8 staging mode that creates the SQ sidecar during global D0 assignment.
    pub gpu_subtree_sq_staging_mode: PiPNNGpuSubtreeSqStagingMode,
    /// Raw rows uploaded per SQ8 GPU-D0 staging shard.
    pub gpu_subtree_sq_staging_upload_shard_rows: usize,
    /// Keep SQ8 GPU-D0 staging but use the raw FP32 D1+ path for diagnostics.
    pub gpu_subtree_sq_staging_raw_d1_diagnostic: bool,
    /// Row tile used by the experimental tiled residual RaBitQ D0 assignment path.
    pub gpu_subtree_rabitq_d0_row_tile: usize,
    /// Row chunk used to bound tiled residual RaBitQ D0 partial top-R buffers.
    pub gpu_subtree_rabitq_d0_row_chunk: usize,
    /// Leader tile used by the experimental tiled residual RaBitQ D0 assignment path.
    pub gpu_subtree_rabitq_d0_leader_tile: usize,
    /// Partial top-R candidates kept per row and leader tile.
    pub gpu_subtree_rabitq_d0_partial_r: usize,
    /// Effective SQ code bits used by the SQ-int8 resident/sidecar paths. Storage remains i8.
    pub gpu_subtree_sq_bits: usize,
    /// Use packed signed INT4 SQ codes and native INT4 WMMA for SQ build mode.
    pub gpu_subtree_sq_native_int4: bool,
    /// Experimental SQ-int8 WMMA resident path. Off preserves raw f32 resident upload.
    pub gpu_subtree_sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode,
    /// Optional global SQ-int8 WMMA sidecar prefix used by shadow/build modes.
    pub gpu_subtree_sq_wmma_sidecar_prefix: Option<PathBuf>,
    /// How SQ-int8 WMMA sidecar files are created or consumed.
    pub gpu_subtree_sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode,
    /// Experimental TurboQuant-MSE4 resident path. Off preserves raw/SQ+WMMA behavior.
    pub gpu_subtree_turboquant_mode: PiPNNGpuSubtreeTurboQuantMode,
    /// TurboQuant-MSE4 scalar quantizer bits. v1 accepts only 4.
    pub gpu_subtree_turboquant_bits: usize,
    /// Deterministic seed used by the fixed FHT-Kac rotator.
    pub gpu_subtree_turboquant_seed: u64,
    /// Optional global TurboQuant-MSE4 sidecar prefix used by shadow/build modes.
    pub gpu_subtree_turboquant_sidecar_prefix: Option<PathBuf>,
    /// How TurboQuant-MSE4 sidecar files are created or consumed.
    pub gpu_subtree_turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode,
    /// Low-dimensional prefix used by the TurboQuant-MSE4 cascade shadow/prefilter path.
    pub gpu_subtree_turboquant_cascade_low_dim: usize,
    /// Survivor cap for TurboQuant-MSE4 cascade telemetry/rerank experiments; 0 means no rerank.
    pub gpu_subtree_turboquant_cascade_survivor_cap: usize,
    /// Candidate-count threshold for adaptive TurboQuant-MSE4 survivor cap; 0 disables.
    pub gpu_subtree_turboquant_adaptive_survivor_threshold: usize,
    /// Survivor cap used when adaptive TurboQuant-MSE4 rerank is active; 0 disables.
    pub gpu_subtree_turboquant_adaptive_survivor_low_cap: usize,
    /// Query-row tile size used by persistent GPU leaf top-k workers.
    /// This changes only execution granularity, not semantic leaf/fanout policy.
    pub gpu_subtree_persistent_leaf_tile_rows: usize,
    /// Number of consecutive 16-row large-task tickets claimed by one CTA.
    pub gpu_subtree_persistent_large_ticket_batch: PiPNNGpuSubtreePersistentLargeTicketBatch,
    /// Persistent top-k scheduler specialization.
    pub gpu_subtree_persistent_scheduler_mode: PiPNNGpuSubtreePersistentSchedulerMode,
    /// Maximum requested sleep interval for the adaptive-idle scheduler.
    pub gpu_subtree_persistent_idle_backoff_ns: u32,
    /// Distance executor used by SQ8 global D0 assignment.
    pub gpu_subtree_sq8_d0_assignment_kernel: PiPNNGpuSubtreeSq8D0AssignmentKernel,
    /// Distance executor used by dense SQ8 large-node assignment.
    pub gpu_subtree_sq8_assignment_kernel: PiPNNGpuSubtreeSq8AssignmentKernel,
    /// Distance executor used by dense SQ8 persistent leaf scoring.
    pub gpu_subtree_sq8_leaf_executor: PiPNNGpuSubtreeSq8LeafExecutor,
    /// Experiment-only directory for deterministic TESSA production-task capture.
    pub gpu_subtree_tessa_capture_dir: Option<PathBuf>,
    /// Work ownership policy used by dense SQ8 persistent leaf scoring.
    pub gpu_subtree_sq8_leaf_mode: PiPNNGpuSubtreeSq8LeafMode,
    /// D0 child-run generator for GPU-subtree. The default preserves the RBC root-fanout path.
    pub gpu_subtree_d0_partition_mode: PiPNNGpuSubtreeD0PartitionMode,
    /// Optional one-level replacement for the D1 root fanout assignment inside each D0 child.
    /// Off preserves the persistent worker's sampled-leader RBC split.
    pub gpu_subtree_d1_root_partition_mode: PiPNNGpuSubtreeD1RootPartitionMode,
    /// Tagore-style D0 root partition count.
    pub gpu_subtree_d0_tagore_partitions: usize,
    /// Tagore-style D0 top-k assignment count.
    pub gpu_subtree_d0_tagore_k: usize,
    /// Fraction of rows sampled for Tagore-style root k-means training.
    pub gpu_subtree_d0_tagore_sample_fraction: f64,
    /// Enable GPU-subtree D0 capacity telemetry and oversized split-only safety repair.
    pub gpu_subtree_d0_capacity_enable: bool,
    /// Controls admission and capacity-retry splitting for D1+ GPU subtree tasks.
    pub gpu_subtree_d1plus_lazy_split_mode: PiPNNGpuSubtreeD1PlusLazySplitMode,
    /// Fraction of resident-admissible rows that marks an oversized D0 child.
    pub gpu_subtree_d0_capacity_high_ratio: f64,
    /// Fraction of resident-admissible rows used as the split-only repair target.
    pub gpu_subtree_d0_capacity_target_ratio: f64,
    /// Fraction of resident-admissible rows below which a child is counted as small for telemetry.
    pub gpu_subtree_d0_capacity_low_ratio: f64,
    /// Minimum row count considered efficient enough for a standalone GPU subtree.
    pub gpu_subtree_d0_capacity_min_efficient_rows: usize,
    /// Configured D0 root leader count used while capacity telemetry acts as a safety valve.
    pub gpu_subtree_d0_capacity_quality_min_root_leaders: usize,
    /// Hard cap for configured D0 root leaders.
    pub gpu_subtree_d0_capacity_root_leader_cap: usize,
    /// Maximum recursive split-only repair depth for one oversized D0 child.
    pub gpu_subtree_d0_capacity_split_recursion_limit: usize,
    /// Optional D0 capacity throughput model. Off preserves the memory-admission-only path.
    pub gpu_subtree_d0_throughput_model: PiPNNGpuSubtreeD0ThroughputModelMode,
    /// Allowed calibrated persistent-kernel growth relative to the memory-safe candidate.
    pub gpu_subtree_d0_throughput_persistent_growth_limit: f64,
    /// Maximum calibrated visible sidecar preload time for one D0 capacity candidate.
    pub gpu_subtree_d0_throughput_preload_visible_limit_ms: usize,
    /// Lowest candidate ratio considered by the throughput model.
    pub gpu_subtree_d0_throughput_min_ratio: f64,
    /// Optional JSON file used to seed D0 throughput calibration.
    pub gpu_subtree_d0_throughput_calibration_path: Option<PathBuf>,
    /// Optional p95 child row guardrail for D0 throughput enforce mode. Zero disables it.
    pub gpu_subtree_d0_throughput_child_p95_guardrail_rows: usize,
    pub gpu_subtree_persistent_launch_policy: PiPNNGpuSubtreePolicy,
    pub gpu_subtree_hbm_policy: PiPNNGpuSubtreePolicy,
    pub gpu_subtree_launch_calibration_cache: Option<PathBuf>,
    pub gpu_subtree_launch_calibration_refresh: bool,
    /// Experimental D1+ host vector staging mode. Off preserves the historical child-by-child
    /// path.
    pub gpu_subtree_vector_staging_mode: PiPNNGpuSubtreeVectorStagingMode,
    /// Scratch budget for fanout-aware vector slab staging. Zero selects an OOM-budget-derived
    /// cap.
    pub gpu_subtree_vector_staging_budget_bytes: usize,
    /// Maximum number of recent D1+ child slabs retained by rolling vector reuse.
    pub gpu_subtree_vector_staging_window_children: usize,
    /// D1+ child processing order. Natural preserves D0 child-run order.
    pub gpu_subtree_child_schedule_mode: PiPNNGpuSubtreeChildScheduleMode,
    /// Experimental witness-guided candidate thinning before compact top-k candidates enter
    /// HashPrune. Off preserves the historical HashPrune candidate pool.
    pub compact_lune_prune_enable: bool,
    /// Experimental compact top-k lune witness metadata generation. Off preserves the historical
    /// compact top-k payload unless compact_lune_prune is enabled.
    pub compact_lune_witness_enable: bool,
    /// Number of nearest per-row compact top-k candidates that may act as protected witness
    /// pivots and are never removed by compact lune pruning.
    pub compact_lune_prune_protected_core: usize,
    /// Experimental source-sharded bounded digest before candidate spill. Off preserves raw
    /// PendingEdge spill records.
    pub compact_spill_digest_enable: bool,
    /// Per-source candidate width multiplier for compact spill digest. The effective width is
    /// clamp(l_max * multiplier, l_max, 4 * l_max).
    pub compact_spill_digest_width_multiplier: f32,
    /// Number of assignment/update iterations for each k-means split.
    pub kmeans_iterations: usize,
    /// Multiplier for requested k-means child count before recursive leaf balancing.
    pub kmeans_oversplit_factor: usize,
    /// 叶内 k-NN 候选个数。
    pub leaf_knn: usize,
    /// 全矩阵叶内核允许处理的最大叶子大小。
    /// 超过该阈值的叶子会切换到 blockwise 精确 top-k 路径。
    pub max_full_matrix_leaf_size: usize,
    /// 是否启用最终 RobustPrune 精剪（在 InmemIndex 上复用 prune_neighbors 实现）。
    pub final_prune: bool,
    /// 最终精剪使用的 alpha 参数。
    pub final_prune_alpha: f32,
    /// 随机种子，控制 RBC leader 采样与超平面生成。
    pub random_seed: u64,
    /// 距离度量。当前实现主要针对 L2 进行了 GEMM 优化，对非 L2 会回退到逐对距离计算。
    pub metric: Metric,
    /// PiPNN 重计算核使用的执行后端。
    pub compute_backend: PiPNNComputeBackendKind,
    /// GPU 后端使用的设备编号。
    pub gpu_device_id: usize,
    /// GPU compact top-k workspace budget in bytes. Zero means use the conservative default.
    pub gpu_workspace_budget_bytes: usize,
    /// Optional directory for dumping real CUDA compact top-k batches for replay.
    pub cuda_compact_dump_dir: Option<PathBuf>,
    /// Maximum number of CUDA compact top-k batches to dump. Zero disables dumping.
    pub cuda_compact_dump_limit: usize,
    /// 是否启用默认关闭的 PiPNN Pipeline Runtime。
    pub pipeline_runtime_enable: bool,
    /// Pipeline compute lane 模式。v1 不要求真实 CUDA kernel。
    pub pipeline_compute_mode: PiPNNPipelineComputeMode,
    /// LeafDescriptorQueue 容量覆盖；未设置时使用 SchedulerBudget.leaf_backlog_capacity。
    pub pipeline_leaf_descriptor_queue_capacity: Option<usize>,
    /// StagedLeafQueue 容量覆盖；未设置时使用 worker_count.max(1)。
    pub pipeline_staged_queue_capacity: Option<usize>,
    /// CompactTopKQueue 容量覆盖；未设置时使用 worker_count * 2。
    pub pipeline_topk_queue_capacity: Option<usize>,
    /// EdgeFlushQueue 容量覆盖；未设置时使用 worker_count * 2。
    pub pipeline_edge_queue_capacity: Option<usize>,
    /// Producer active 时 compute lane 最大 inflight 覆盖。
    pub pipeline_max_inflight_compute_tasks_active: Option<usize>,
    /// Producer finished 后 compute lane 最大 inflight 覆盖。
    pub pipeline_max_inflight_compute_tasks_drain: Option<usize>,
    /// 是否启用 OOM research 模式。
    pub oom_enable: bool,
    /// OOM research 模式总内存预算。
    pub oom_memory_budget_bytes: usize,
    /// 是否在 OOM bulk I/O 路径上启用 strict direct I/O，避免 mmap/page-cache 作为主要依赖。
    pub oom_strict_io: bool,
    /// OOM bulk I/O mode. `Auto` preserves legacy `oom_strict_io` behavior.
    pub oom_io_mode: PiPNNOomIoMode,
    /// OOM resident reservoir budget share. 1.0 preserves legacy allocation.
    pub oom_reservoir_budget_share: f64,
    /// Optional diagnostic cap for OOM sidecar resident budget.
    pub oom_sidecar_resident_budget_cap_bytes: Option<usize>,
    /// OOM research 模式 sketch cache 预算。
    pub oom_sketch_cache_bytes: usize,
    /// OOM research 模式 raw vector cache 预算。该预算从 resident reservoir 预算中让出。
    pub oom_raw_vector_cache_bytes: usize,
    /// OOM research 模式 spill cache 预算。
    pub oom_spill_cache_bytes: usize,
    /// Online suffix reduction mode. Off preserves the external-reducer path.
    pub online_suffix_reduce_mode: PiPNNOnlineSuffixReduceMode,
    /// Direct candidate mailbox budget. Zero selects the bounded default.
    pub online_suffix_direct_queue_bytes: usize,
    /// Write-back source-reservoir cache budget. Zero selects the bounded default.
    pub online_suffix_state_cache_bytes: usize,
    /// Disk-reservoir page cache budget. Zero selects the bounded default.
    pub online_suffix_state_page_cache_bytes: usize,
    /// In-flight state-page writeback budget. Zero selects the bounded default.
    pub online_suffix_state_writeback_bytes: usize,
    /// Fallback spill writer budget. Zero selects the bounded default.
    pub online_suffix_spill_buffer_bytes: usize,
    /// Per-shard reorder window budget. Zero selects the bounded default.
    pub online_suffix_reorder_window_bytes: usize,
    /// Grouped state-read window budget. Zero selects the bounded default.
    pub online_suffix_read_window_bytes: usize,
    /// Optional pinned source prefix for write-back-cache mode.
    pub online_suffix_pinned_prefix_points: Option<usize>,
    /// Number of suffix reducer workers. Zero selects an automatic bounded value.
    pub online_suffix_reduce_workers: usize,
    /// Maximum logical bytes of uncompacted log-assisted delta/checkpoint state.
    pub online_suffix_delta_budget_bytes: usize,
    /// DRAM buffer budget used to form sorted log-assisted delta runs.
    pub online_suffix_delta_buffer_bytes: usize,
    /// Source IDs per log-assisted compaction range. Zero selects an automatic value.
    pub online_suffix_range_points: usize,
    /// Uncompacted bytes that schedule one range compaction. Zero selects the default.
    pub online_suffix_range_trigger_bytes: usize,
    /// Soft active-run reference for slot-bounded early compaction eligibility.
    /// A base-sized overlay is also required. Zero selects the default.
    pub online_suffix_max_runs_per_range: usize,
    /// Background range-compaction workers. Zero selects an automatic bounded value.
    pub online_suffix_compaction_workers: usize,
    /// Updates in one compacted epoch required for cache promotion.
    pub online_suffix_cache_hot_updates: usize,
    /// Physical disk-state slots allocated per logical suffix source.
    pub online_suffix_state_overprovision_ratio: f64,
    /// OOM research 模式临时目录。
    pub oom_temp_dir: PathBuf,
    /// 是否保留 OOM research 中间产物。
    pub oom_keep_artifacts: bool,
    /// 将 OOM profiling 额外写到 JSON 文件。
    pub oom_profile_json: Option<PathBuf>,
    /// Optional directory for default-off HashPrune SoftTier diagnostic artifacts.
    pub hashprune_backfill_diagnostics_prefix: Option<PathBuf>,
}

impl PiPNNParams {
    pub const DEFAULT_GPU_WORKSPACE_BUDGET_BYTES: usize = 512 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_CACHE_HOT_UPDATES: usize = 2;
    pub const DEFAULT_ONLINE_SUFFIX_COMPACTION_WORKSPACE_BYTES_PER_WORKER: usize = 64 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_DECODE_BYTES: usize = 64 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_DELTA_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_DELTA_BUFFER_BYTES: usize = 64 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_DIRECT_QUEUE_BYTES: usize = 128 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_MAX_RUNS_PER_RANGE: usize = 64;
    pub const DEFAULT_ONLINE_SUFFIX_RANGE_TRIGGER_BYTES: usize = 256 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_READ_WINDOW_BYTES: usize = 64 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_REORDER_WINDOW_BYTES: usize = 16 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_SPILL_BUFFER_BYTES: usize = 64 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_STATE_CACHE_BYTES: usize = 512 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_STATE_PAGE_CACHE_BYTES: usize = 512 * 1024 * 1024;
    pub const DEFAULT_ONLINE_SUFFIX_STATE_WRITEBACK_BYTES: usize = 128 * 1024 * 1024;
    pub const DEFAULT_OOM_MEMORY_BUDGET_BYTES: usize = 12 * 1024 * 1024 * 1024;
    pub const DEFAULT_WRITEBACK_STATE_PAGE_CACHE_BYTES: usize = 64 * 1024 * 1024;
    pub const MIN_OOM_MEMORY_BUDGET_BYTES: usize = 512 * 1024 * 1024;

    pub fn effective_online_suffix_budgets(&self) -> PiPNNOnlineSuffixBudgets {
        let uses_writeback_cache = self.online_suffix_reduce_mode.uses_writeback_cache();
        let uses_log_assisted = self.online_suffix_reduce_mode.uses_log_assisted();
        PiPNNOnlineSuffixBudgets {
            direct_queue_bytes: nonzero_or(
                self.online_suffix_direct_queue_bytes,
                Self::DEFAULT_ONLINE_SUFFIX_DIRECT_QUEUE_BYTES,
            ),
            state_cache_bytes: if self.online_suffix_reduce_mode.uses_state_cache() {
                nonzero_or(
                    self.online_suffix_state_cache_bytes,
                    Self::DEFAULT_ONLINE_SUFFIX_STATE_CACHE_BYTES,
                )
            } else {
                0
            },
            state_page_cache_bytes: if uses_log_assisted {
                0
            } else {
                nonzero_or(
                    self.online_suffix_state_page_cache_bytes,
                    if uses_writeback_cache {
                        Self::DEFAULT_WRITEBACK_STATE_PAGE_CACHE_BYTES
                    } else {
                        Self::DEFAULT_ONLINE_SUFFIX_STATE_PAGE_CACHE_BYTES
                    },
                )
            },
            state_writeback_bytes: if uses_log_assisted {
                0
            } else {
                nonzero_or(
                    self.online_suffix_state_writeback_bytes,
                    Self::DEFAULT_ONLINE_SUFFIX_STATE_WRITEBACK_BYTES,
                )
            },
            spill_buffer_bytes: nonzero_or(
                self.online_suffix_spill_buffer_bytes,
                Self::DEFAULT_ONLINE_SUFFIX_SPILL_BUFFER_BYTES,
            ),
            reorder_window_bytes: nonzero_or(
                self.online_suffix_reorder_window_bytes,
                Self::DEFAULT_ONLINE_SUFFIX_REORDER_WINDOW_BYTES,
            ),
            decode_bytes: if uses_writeback_cache || uses_log_assisted {
                0
            } else {
                Self::DEFAULT_ONLINE_SUFFIX_DECODE_BYTES
            },
            read_window_bytes: if uses_writeback_cache {
                nonzero_or(
                    self.online_suffix_read_window_bytes,
                    Self::DEFAULT_ONLINE_SUFFIX_READ_WINDOW_BYTES,
                )
            } else {
                0
            },
            delta_budget_bytes: if uses_log_assisted {
                nonzero_or(
                    self.online_suffix_delta_budget_bytes,
                    Self::DEFAULT_ONLINE_SUFFIX_DELTA_BUDGET_BYTES,
                )
            } else {
                0
            },
            delta_buffer_bytes: if uses_log_assisted {
                nonzero_or(
                    self.online_suffix_delta_buffer_bytes,
                    Self::DEFAULT_ONLINE_SUFFIX_DELTA_BUFFER_BYTES,
                )
            } else {
                0
            },
            compaction_bytes: if uses_log_assisted {
                let workers = if self.online_suffix_compaction_workers == 0 {
                    2
                } else {
                    self.online_suffix_compaction_workers
                };
                Self::DEFAULT_ONLINE_SUFFIX_COMPACTION_WORKSPACE_BYTES_PER_WORKER
                    .saturating_mul(workers)
            } else {
                0
            },
        }
    }

    #[inline]
    pub fn effective_online_suffix_range_trigger_bytes(&self) -> usize {
        nonzero_or(
            self.online_suffix_range_trigger_bytes,
            Self::DEFAULT_ONLINE_SUFFIX_RANGE_TRIGGER_BYTES,
        )
    }

    #[inline]
    pub fn effective_online_suffix_max_runs_per_range(&self) -> usize {
        nonzero_or(
            self.online_suffix_max_runs_per_range,
            Self::DEFAULT_ONLINE_SUFFIX_MAX_RUNS_PER_RANGE,
        )
    }

    #[inline]
    pub fn effective_online_suffix_compaction_workers(&self, pool_threads: usize) -> usize {
        let pool_threads = pool_threads.max(1);
        if self.online_suffix_compaction_workers == 0 {
            pool_threads.min(2)
        } else {
            self.online_suffix_compaction_workers.clamp(1, pool_threads)
        }
    }

    #[inline]
    pub fn effective_online_suffix_drain_compaction_workers(&self, pool_threads: usize) -> usize {
        self.effective_online_suffix_compaction_workers(pool_threads)
            .max(pool_threads.max(1).min(8))
    }

    #[inline]
    pub fn effective_online_suffix_cache_hot_updates(&self) -> usize {
        nonzero_or(
            self.online_suffix_cache_hot_updates,
            Self::DEFAULT_ONLINE_SUFFIX_CACHE_HOT_UPDATES,
        )
    }

    pub fn validate_online_suffix_reduce_config(&self) -> AnnResult<()> {
        if !self.online_suffix_reduce_mode.is_enabled() {
            return Ok(());
        }
        if !self.oom_enable {
            return Err(AnnError::log_index_config_error(
                "online_suffix_reduce".to_string(),
                "online suffix reduction requires oom_enable=true".to_string(),
            ));
        }
        let budgets = self.effective_online_suffix_budgets();
        if self.online_suffix_reduce_mode.uses_writeback_cache()
            && budgets.read_window_bytes > budgets.state_page_cache_bytes
        {
            return Err(AnnError::log_index_config_error(
                "online_suffix_read_window_bytes".to_string(),
                "read window must fit in the state page cache".to_string(),
            ));
        }
        if self.online_suffix_reduce_mode.uses_log_assisted()
            && (budgets.delta_buffer_bytes < 24
                || budgets.delta_buffer_bytes > budgets.delta_budget_bytes
                || self.effective_online_suffix_range_trigger_bytes() > budgets.delta_budget_bytes
                || self.effective_online_suffix_max_runs_per_range() == 0)
        {
            return Err(AnnError::log_index_config_error(
                "online_suffix_delta_budget_bytes".to_string(),
                "log-assisted requires a delta buffer of at least 24 bytes, buffer and range trigger within the delta budget, and a nonzero soft run reference"
                    .to_string(),
            ));
        }
        if !self.online_suffix_reduce_mode.uses_log_assisted()
            && (!self.online_suffix_state_overprovision_ratio.is_finite()
                || self.online_suffix_state_overprovision_ratio <= 1.0)
        {
            return Err(AnnError::log_index_config_error(
                "online_suffix_state_overprovision_ratio".to_string(),
                "state overprovision ratio must be finite and greater than 1.0".to_string(),
            ));
        }
        if self.l_max > u16::MAX as usize {
            return Err(AnnError::log_index_config_error(
                "l_max".to_string(),
                "online suffix state requires l_max <= 65535".to_string(),
            ));
        }
        Ok(())
    }

    pub fn validate_hashprune_evidence_config(&self) -> AnnResult<()> {
        let valid_pair = matches!(
            (
                self.hashprune_evidence_mode,
                self.hashprune_evidence_overflow_budget
            ),
            (PiPNNHashPruneEvidenceMode::Off, 0)
                | (PiPNNHashPruneEvidenceMode::Report, 0)
                | (PiPNNHashPruneEvidenceMode::Enforce, 1)
        );
        if !valid_pair {
            return Err(AnnError::log_index_config_error(
                "hashprune_evidence".to_string(),
                format!(
                    "invalid HashPrune evidence configuration: mode={} overflow_budget={}; expected off/0, report/0, or enforce/1",
                    self.hashprune_evidence_mode, self.hashprune_evidence_overflow_budget
                ),
            ));
        }
        if self.hashprune_evidence_mode != PiPNNHashPruneEvidenceMode::Off
            && self.prune_backend != PiPNNPruneBackend::HashPrune
        {
            return Err(AnnError::log_index_config_error(
                "hashprune_evidence".to_string(),
                "non-off HashPrune evidence modes require prune_backend=hashprune".to_string(),
            ));
        }
        Ok(())
    }

    pub fn validate_hashprune_backfill_config(&self) -> AnnResult<()> {
        let named_rhalf = matches!(
            self.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::RHalfHashSlot1
                | PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
                | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
                | PiPNNHashPruneBackfillMode::RHalfHardTierQ3
        );
        if let Some(cap) = self.hashprune_rhalf_max_backfill_per_row
            && (!named_rhalf || cap == 0 || cap > self.l_max / 2)
        {
            return Err(AnnError::log_index_config_error(
                "hashprune_rhalf_max_backfill_per_row".to_string(),
                format!(
                    "named R-half maximum backfill per row must be in 1..=floor(l_max/2), got cap={cap} l_max={}",
                    self.l_max
                ),
            ));
        }
        if let Some(quota) = self.hashprune_rhalf_hash_quota
            && (!named_rhalf || !(1..=3).contains(&quota))
        {
            return Err(AnnError::log_index_config_error(
                "hashprune_rhalf_hash_quota".to_string(),
                "named R-half hash quota must be in 1..=3".to_string(),
            ));
        }
        if let Some(alpha) = self.hashprune_soft_tier_priority_alpha
            && (self.hashprune_backfill_mode != PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
                || !alpha.is_finite()
                || alpha < 1.0)
        {
            return Err(AnnError::log_index_config_error(
                "hashprune_soft_tier_priority_alpha".to_string(),
                "SoftTier priority alpha must be finite and >= 1.0".to_string(),
            ));
        }
        if self.hashprune_backfill_mode != PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
            && self.hashprune_backfill_ablation != PiPNNHashPruneBackfillAblation::None
        {
            return Err(AnnError::log_index_config_error(
                "hashprune_backfill_ablation".to_string(),
                "HashPrune backfill ablations are valid only with distance-gated-tiered-rhalf"
                    .to_string(),
            ));
        }
        match self.hashprune_backfill_mode {
            PiPNNHashPruneBackfillMode::Off => {
                if self.hashprune_backfill_budget != 0 || self.hashprune_backfill_alpha != 1.0 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        format!(
                            "off HashPrune backfill requires budget=0 alpha=1.0, got budget={} alpha={}",
                            self.hashprune_backfill_budget, self.hashprune_backfill_alpha
                        ),
                    ));
                }
            }
            PiPNNHashPruneBackfillMode::DistanceGated
            | PiPNNHashPruneBackfillMode::DistanceGatedStratified
            | PiPNNHashPruneBackfillMode::DistanceGatedMultihit => {
                if !(1..=8).contains(&self.hashprune_backfill_budget) {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill_budget".to_string(),
                        "distance-gated HashPrune backfill requires budget in 1..=8".to_string(),
                    ));
                }
                if !self.hashprune_backfill_alpha.is_finite() || self.hashprune_backfill_alpha < 1.0
                {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill_alpha".to_string(),
                        "distance-gated HashPrune backfill requires finite alpha >= 1.0"
                            .to_string(),
                    ));
                }
                if self.prune_backend != PiPNNPruneBackend::HashPrune {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated HashPrune backfill requires prune_backend=hashprune"
                            .to_string(),
                    ));
                }
                if self.metric != Metric::L2 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated HashPrune backfill currently requires squared L2 distance"
                            .to_string(),
                    ));
                }
                if self.hashprune_evidence_mode != PiPNNHashPruneEvidenceMode::Off
                    || self.hashprune_evidence_overflow_budget != 0
                {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated backfill is incompatible with non-off HashPrune evidence"
                            .to_string(),
                    ));
                }
            }
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52 => {
                if self.hashprune_backfill_budget != 8 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill_budget".to_string(),
                        "distance-gated-tiered-d52 HashPrune backfill requires budget=8"
                            .to_string(),
                    ));
                }
                if !self.hashprune_backfill_alpha.is_finite() || self.hashprune_backfill_alpha < 1.0
                {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill_alpha".to_string(),
                        "distance-gated HashPrune backfill requires finite alpha >= 1.0"
                            .to_string(),
                    ));
                }
                if self.prune_backend != PiPNNPruneBackend::HashPrune {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated HashPrune backfill requires prune_backend=hashprune"
                            .to_string(),
                    ));
                }
                if self.metric != Metric::L2 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated HashPrune backfill currently requires squared L2 distance"
                            .to_string(),
                    ));
                }
                if self.hashprune_evidence_mode != PiPNNHashPruneEvidenceMode::Off
                    || self.hashprune_evidence_overflow_budget != 0
                {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated backfill is incompatible with non-off HashPrune evidence"
                            .to_string(),
                    ));
                }
            }
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf => {
                if self.l_max == 0 || self.l_max > usize::from(u16::MAX) {
                    return Err(AnnError::log_index_config_error(
                        "l_max".to_string(),
                        "distance-gated-tiered-rhalf requires l_max in 1..=u16::MAX".to_string(),
                    ));
                }
                if self.hashprune_backfill_budget != 0 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill_budget".to_string(),
                        "distance-gated-tiered-rhalf derives its budget from l_max and requires the legacy budget argument to be 0"
                            .to_string(),
                    ));
                }
                if self.hashprune_backfill_alpha != 1.1 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill_alpha".to_string(),
                        "distance-gated-tiered-rhalf freezes alpha at 1.1".to_string(),
                    ));
                }
                if self.prune_backend != PiPNNPruneBackend::HashPrune {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated-tiered-rhalf requires prune_backend=hashprune".to_string(),
                    ));
                }
                if self.metric != Metric::L2 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated-tiered-rhalf requires squared L2 distance".to_string(),
                    ));
                }
                if self.hashprune_evidence_mode != PiPNNHashPruneEvidenceMode::Off
                    || self.hashprune_evidence_overflow_budget != 0
                {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "distance-gated-tiered-rhalf is incompatible with non-off HashPrune evidence"
                            .to_string(),
                    ));
                }
            }
            PiPNNHashPruneBackfillMode::RHalfHashSlot1
            | PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3
            | PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
            | PiPNNHashPruneBackfillMode::RHalfHardTierQ3 => {
                if self.l_max == 0 || self.l_max > usize::from(u16::MAX) {
                    return Err(AnnError::log_index_config_error(
                        "l_max".to_string(),
                        "named R-half HashPrune backfill requires l_max in 1..=u16::MAX"
                            .to_string(),
                    ));
                }
                if self.hashprune_backfill_budget != 0 || self.hashprune_backfill_alpha != 1.0 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        format!(
                            "named R-half HashPrune backfill derives its policy and requires legacy budget=0 alpha=1.0, got budget={} alpha={}",
                            self.hashprune_backfill_budget, self.hashprune_backfill_alpha
                        ),
                    ));
                }
                if self.prune_backend != PiPNNPruneBackend::HashPrune {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "named R-half backfill requires prune_backend=hashprune".to_string(),
                    ));
                }
                if self.metric != Metric::L2 {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "named R-half backfill requires squared L2 distance".to_string(),
                    ));
                }
                if self.hashprune_evidence_mode != PiPNNHashPruneEvidenceMode::Off
                    || self.hashprune_evidence_overflow_budget != 0
                {
                    return Err(AnnError::log_index_config_error(
                        "hashprune_backfill".to_string(),
                        "named R-half backfill is incompatible with non-off HashPrune evidence"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn validate_hashprune_policies(&self) -> AnnResult<()> {
        self.validate_hashprune_evidence_config()?;
        self.validate_hashprune_backfill_config()?;
        self.validate_hashprune_backfill_diagnostics_config()
    }

    pub fn validate_hashprune_backfill_diagnostics_config(&self) -> AnnResult<()> {
        if self.hashprune_backfill_diagnostics_prefix.is_none() {
            return Ok(());
        }
        if self.prune_backend != PiPNNPruneBackend::HashPrune
            || self.hashprune_backfill_mode != PiPNNHashPruneBackfillMode::RHalfSoftTierQ3
            || self.hashprune_backfill_ablation != PiPNNHashPruneBackfillAblation::None
            || self.hashprune_backfill_budget != 0
            || self.hashprune_backfill_alpha != 1.0
            || self.hashprune_rhalf_hash_quota.is_some()
            || self.hashprune_soft_tier_priority_alpha.is_some()
            || self.hashprune_evidence_mode != PiPNNHashPruneEvidenceMode::Off
            || self.hashprune_evidence_overflow_budget != 0
            || self.metric != Metric::L2
            || !(1..=usize::from(u8::MAX)).contains(&self.l_max)
            || self.root_stop_after_d00
        {
            return Err(AnnError::log_index_config_error(
                "hashprune_backfill_diagnostics_prefix".to_string(),
                "HashPrune diagnostics require hashprune, rhalf-soft-tier-q3, ablation=none, budget=0, alpha=1.0, no named-policy tuning overrides, evidence=off, squared L2, l_max in 1..=255, and a complete non-root-stop build"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// 根据递归深度返回该层 fanout。
    #[inline]
    pub fn fanout_for_depth(&self, depth: usize) -> usize {
        match depth {
            0 => self.fanout_top,
            1 => self.fanout_second,
            _ => 1,
        }
    }

    /// 返回距离度量。
    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// 当前全矩阵叶内核可安全处理的叶子大小上限。
    #[inline]
    pub fn kernel_safe_leaf_size(&self) -> usize {
        self.max_full_matrix_leaf_size.max(1)
    }

    /// 根据当前簇规模和深度返回采样比例。
    #[inline]
    pub fn adaptive_psamp_fraction(&self, cluster_size: usize, depth: usize) -> f64 {
        let mut frac = self.psamp_fraction;

        if cluster_size >= 20_000_000 {
            frac *= 0.05;
        } else if cluster_size >= 10_000_000 {
            frac *= 0.1;
        } else if cluster_size >= 5_000_000 {
            frac *= 0.2;
        } else if cluster_size >= 1_000_000 {
            frac *= 0.4;
        } else if cluster_size >= 100_000 {
            frac *= 0.7;
        }

        if depth > 4 {
            frac *= 0.35;
        } else if depth > 2 {
            frac *= 0.5;
        } else if depth > 0 {
            frac *= 0.8;
        }

        frac.clamp(0.00005, 0.1)
    }

    /// 根据数据规模自适应放大 c_max，减少超大数据集的 RBC 顶层负担。
    #[inline]
    pub fn adaptive_c_max(&self, total_size: usize) -> usize {
        if total_size >= 20_000_000 {
            self.c_max.saturating_mul(2)
        } else if total_size >= 10_000_000 {
            self.c_max.saturating_mul(2)
        } else if total_size >= 5_000_000 {
            self.c_max.saturating_mul(2)
        } else {
            self.c_max
        }
    }

    /// 根据递归层级返回用于 RBC 分区的 fanout。
    #[inline]
    pub fn adaptive_fanout(&self, _cluster_size: usize, depth: usize) -> usize {
        self.fanout_for_depth(depth)
    }

    /// 根据数据集规模和硬件配置自动推荐 PiPNN 参数。
    ///
    /// 核心思路：PiPNN 总距离计算量应与 Vamana 在同一量级（乘以 GEMM 效率优势系数），
    /// 然后按 40/60 比例分配给 RBC 分区和叶内 GEMM，反推各参数。
    ///
    /// - `n`: 数据集点数
    /// - `d`: 向量维度
    /// - `r`: 目标最大出度（l_max）
    /// - `num_threads`: 构图线程数
    pub fn recommend(n: usize, d: usize, r: usize, num_threads: usize) -> Self {
        // Vamana 等效距离计算次数: n * L * R
        let l_vamana = r.max(100) as f64;
        let d_vamana = n as f64 * l_vamana * r as f64;

        // GEMM 相对 beam search 的效率优势（保守 10x）
        let gemm_advantage = 10.0_f64;
        let d_budget = d_vamana * gemm_advantage;

        // 预算分配：RBC 40%，叶内 GEMM 60%
        let d_rbc_budget = d_budget * 0.4;
        let d_leaf_budget = d_budget * 0.6;

        // fanout 保持原有经验值分层，但对超大数据集偏向更快构建。
        // wiki35m 这类规模会优先选择 10x2，而不是默认的 10x3。
        let fanout_top = 10_usize;
        let fanout_second = if n >= 20_000_000 { 2_usize } else { 3_usize };
        let f_total = (fanout_top * fanout_second) as f64;

        // 从叶内预算反推 c_max（注意 adaptive_c_max 会 ×2，这里推的是基础值）
        // D_leaf = n * F_total * adaptive_c_max / 2 ≤ D_leaf_budget
        // 对于 n >= 5M，adaptive_c_max = c_max * 2
        let adaptive_multiplier = if n >= 5_000_000 { 2.0 } else { 1.0 };
        let c_max_raw = 2.0 * d_leaf_budget / (n as f64 * f_total * adaptive_multiplier);
        let mut c_max = (c_max_raw as usize).clamp(512, 2048);
        if n >= 20_000_000 && fanout_second == 2 {
            // Keep wiki35m-class 10x2 runs aligned with the historical f10s2 baseline
            // so fanout changes do not silently inflate leaf size from 1280 -> 1920.
            c_max = c_max.min(1280);
        }

        // 从 RBC 预算反推 max_leaders
        // D_rbc ≈ n * leaders * 2（大多数情况 depth ≤ 2）
        let max_leaders_raw = d_rbc_budget / (n as f64 * 2.0);
        let max_leaders = (max_leaders_raw as usize).clamp(500, 10000);

        // psamp_fraction：确保顶层 leaders ≈ max_leaders
        // 对于 n >= 20M，adaptive_psamp_fraction 乘 0.05，再乘 depth=0 的 1.0
        // leaders = min(n * psamp * 0.05, max_leaders)
        // 希望 n * psamp * 0.05 ≈ max_leaders → psamp = max_leaders / (n * 0.05)
        let psamp = if n >= 20_000_000 {
            max_leaders as f64 / (n as f64 * 0.05)
        } else if n >= 10_000_000 {
            max_leaders as f64 / (n as f64 * 0.1)
        } else if n >= 5_000_000 {
            max_leaders as f64 / (n as f64 * 0.2)
        } else if n >= 1_000_000 {
            max_leaders as f64 / (n as f64 * 0.4)
        } else {
            max_leaders as f64 / (n as f64 * 0.7)
        };
        let psamp = psamp.clamp(0.001, 0.1);

        let leaf_knn = 2;

        println!("=== PiPNN Parameter Recommendation ===");
        println!(
            "  Dataset: n={}, d={}, R={}, threads={}",
            n, d, r, num_threads
        );
        println!("  Vamana equivalent D_dist = {:.2e}", d_vamana);
        println!("  PiPNN budget (10x GEMM advantage) = {:.2e}", d_budget);
        println!("  Recommended parameters:");
        println!(
            "    c_max = {} (adaptive = {})",
            c_max,
            c_max * adaptive_multiplier as usize
        );
        println!("    max_leaders = {}", max_leaders);
        println!("    psamp_fraction = {:.4}", psamp);
        println!(
            "    fanout_top = {}, fanout_second = {}",
            fanout_top, fanout_second
        );
        println!("    leaf_knn = {}", leaf_knn);

        let params = Self {
            l_max: r,
            c_max,
            psamp_fraction: psamp,
            max_leaders,
            fanout_top,
            fanout_second,
            root_fanout_policy: RootFanoutPolicy::Fixed,
            root_radius_k: 10,
            root_radius_candidate_rs: vec![4, 6, 8],
            root_radius_warmup_points: 500_000,
            root_radius_dry_run: false,
            root_stop_after_d00: false,
            root_risk_target_avg_fanout: 8.0,
            root_risk_sigma_lambda: 1.0,
            root_risk_budget_lambda: None,
            root_risk_sigma_mode: RootRiskSigmaMode::CellQ90Q50,
            root_risk_warmup_points: 500_000,
            root_risk_candidate_rs: vec![4, 6, 8, 10],
            root_risk_dry_run: false,
            root_risk_allow_full_build: false,
            partitioner: PiPNNPartitionerKind::Rbc,
            gpu_subtree_root_assignment_backend: PiPNNComputeBackendKind::Cpu,
            gpu_subtree_assignment_backend: PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32,
            gpu_subtree_assignment_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            gpu_subtree_assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            gpu_subtree_assignment_dco_proj_dim: 0,
            gpu_subtree_assignment_dco_survivors: 0,
            gpu_subtree_assignment_dco_audit: false,
            gpu_subtree_d0_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            gpu_subtree_d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            gpu_subtree_d0_dco_survivors: 0,
            gpu_subtree_d0_dco_audit: false,
            gpu_subtree_d0_sq_cpu_kernel: SqInt8CpuKernel::Auto,
            gpu_subtree_execution_mode: PiPNNGpuSubtreeExecutionMode::FusedPersistent,
            gpu_subtree_leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32,
            gpu_subtree_leaf_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            gpu_subtree_leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            gpu_subtree_leaf_dco_proj_dim: 0,
            gpu_subtree_leaf_dco_survivors: 0,
            gpu_subtree_leaf_dco_audit: false,
            gpu_subtree_leaf_dco_ddc_res_pca_matrix: None,
            gpu_subtree_leaf_dco_ddc_res_projected_input: false,
            gpu_subtree_leaf_dco_ddc_res_base_dim: 32,
            gpu_subtree_leaf_dco_ddc_res_sigma_count: 3.0,
            gpu_subtree_rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode::Off,
            gpu_subtree_rabitq_plus_bits: 8,
            gpu_subtree_rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            gpu_subtree_rabitq_plus_seed: 42,
            gpu_subtree_rabitq_plus_sidecar_prefix: None,
            gpu_subtree_rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode::Off,
            gpu_subtree_qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode::Off,
            gpu_subtree_qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident,
            gpu_subtree_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma,
            gpu_subtree_rabitq_short_survivors: None,
            gpu_subtree_rabitq_residual_centroids: 16,
            gpu_subtree_qcode_staging_upload_shard_rows: 524_288,
            gpu_subtree_qcode_global_d0_leaders: 0,
            gpu_subtree_rabitq_d0_assign_mode: PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial,
            gpu_subtree_rabitq_d1_topk_mode: PiPNNGpuSubtreeRabitqD1TopKMode::RowSerial,
            gpu_subtree_rabitq_d0_row_tile: 4,
            gpu_subtree_rabitq_d0_row_chunk: 4096,
            gpu_subtree_rabitq_d0_leader_tile: 64,
            gpu_subtree_rabitq_d0_partial_r: 8,
            gpu_subtree_sq_staging_mode: PiPNNGpuSubtreeSqStagingMode::Off,
            gpu_subtree_sq_staging_upload_shard_rows: 524_288,
            gpu_subtree_sq_staging_raw_d1_diagnostic: false,
            gpu_subtree_sq_bits: 8,
            gpu_subtree_sq_native_int4: false,
            gpu_subtree_sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode::Off,
            gpu_subtree_sq_wmma_sidecar_prefix: None,
            gpu_subtree_sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode::Off,
            gpu_subtree_turboquant_mode: PiPNNGpuSubtreeTurboQuantMode::Off,
            gpu_subtree_turboquant_bits: 4,
            gpu_subtree_turboquant_seed: 42,
            gpu_subtree_turboquant_sidecar_prefix: None,
            gpu_subtree_turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode::Off,
            gpu_subtree_turboquant_cascade_low_dim: 128,
            gpu_subtree_turboquant_cascade_survivor_cap: 0,
            gpu_subtree_turboquant_adaptive_survivor_threshold: 0,
            gpu_subtree_turboquant_adaptive_survivor_low_cap: 0,
            gpu_subtree_persistent_leaf_tile_rows: 128,
            gpu_subtree_persistent_large_ticket_batch:
                PiPNNGpuSubtreePersistentLargeTicketBatch::Auto,
            gpu_subtree_persistent_scheduler_mode: PiPNNGpuSubtreePersistentSchedulerMode::Auto,
            gpu_subtree_persistent_idle_backoff_ns: 350,
            gpu_subtree_sq8_d0_assignment_kernel: PiPNNGpuSubtreeSq8D0AssignmentKernel::Auto,
            gpu_subtree_sq8_assignment_kernel: PiPNNGpuSubtreeSq8AssignmentKernel::Auto,
            gpu_subtree_sq8_leaf_executor: PiPNNGpuSubtreeSq8LeafExecutor::Auto,
            gpu_subtree_tessa_capture_dir: None,
            gpu_subtree_sq8_leaf_mode: PiPNNGpuSubtreeSq8LeafMode::Auto,
            gpu_subtree_d0_partition_mode: PiPNNGpuSubtreeD0PartitionMode::RbcFanout,
            gpu_subtree_d1_root_partition_mode: PiPNNGpuSubtreeD1RootPartitionMode::Off,
            gpu_subtree_d0_tagore_partitions: 32,
            gpu_subtree_d0_tagore_k: 2,
            gpu_subtree_d0_tagore_sample_fraction: 0.01,
            gpu_subtree_d0_capacity_enable: true,
            gpu_subtree_d1plus_lazy_split_mode: PiPNNGpuSubtreeD1PlusLazySplitMode::On,
            gpu_subtree_d0_capacity_high_ratio: 0.60,
            gpu_subtree_d0_capacity_target_ratio: 0.50,
            gpu_subtree_d0_capacity_low_ratio: 0.15,
            gpu_subtree_d0_capacity_min_efficient_rows: 32 * 1024,
            gpu_subtree_d0_capacity_quality_min_root_leaders: 64,
            gpu_subtree_d0_capacity_root_leader_cap: 10_000,
            gpu_subtree_d0_capacity_split_recursion_limit: 8,
            gpu_subtree_d0_throughput_model: PiPNNGpuSubtreeD0ThroughputModelMode::Off,
            gpu_subtree_d0_throughput_persistent_growth_limit: 1.10,
            gpu_subtree_d0_throughput_preload_visible_limit_ms: 5000,
            gpu_subtree_d0_throughput_min_ratio: 0.50,
            gpu_subtree_d0_throughput_calibration_path: None,
            gpu_subtree_d0_throughput_child_p95_guardrail_rows: 0,
            gpu_subtree_persistent_launch_policy: PiPNNGpuSubtreePolicy::Report,
            gpu_subtree_hbm_policy: PiPNNGpuSubtreePolicy::Report,
            gpu_subtree_launch_calibration_cache: None,
            gpu_subtree_launch_calibration_refresh: false,
            gpu_subtree_vector_staging_mode: PiPNNGpuSubtreeVectorStagingMode::Off,
            gpu_subtree_vector_staging_budget_bytes: 0,
            gpu_subtree_vector_staging_window_children: 8,
            gpu_subtree_child_schedule_mode: PiPNNGpuSubtreeChildScheduleMode::Natural,
            compact_lune_prune_enable: false,
            compact_lune_witness_enable: false,
            compact_lune_prune_protected_core: 1,
            compact_spill_digest_enable: false,
            compact_spill_digest_width_multiplier: 2.0,
            kmeans_iterations: 4,
            kmeans_oversplit_factor: 2,
            leaf_knn,
            ..Self::default()
        };

        let est = params.estimate_build_time(n, d, num_threads);
        println!(
            "  Estimated build time: {:.0}s ({:.1} min)",
            est,
            est / 60.0
        );
        println!("=======================================");

        params
    }

    /// 估算给定参数下的构图时间（秒）。
    ///
    /// 基于距离计算次数 × 2d FLOP / (单核 GEMM 吞吐 × 线程数) 的模型，
    /// 加 15% overhead（内存分配、锁竞争等）。
    pub fn estimate_build_time(&self, n: usize, d: usize, num_threads: usize) -> f64 {
        let gemm_flops_per_core = 50e9_f64; // AVX2 f32 GEMM 典型单核吞吐
        let c_max = self.adaptive_c_max(n) as f64;
        let f_total = (self.fanout_top * self.fanout_second) as f64;

        // RBC 距离计算：depth 0 + depth 1
        let leaders_0 =
            (n as f64 * self.adaptive_psamp_fraction(n, 0)).min(self.max_leaders as f64);
        let cluster_size_1 = if leaders_0 > 0.0 {
            n as f64 / leaders_0
        } else {
            n as f64
        };
        let leaders_1 = (cluster_size_1 * self.adaptive_psamp_fraction(cluster_size_1 as usize, 1))
            .min(self.max_leaders as f64);
        let d_rbc = n as f64 * leaders_0 + n as f64 * self.fanout_top as f64 * leaders_1;

        // 叶内 GEMM 距离计算
        let d_leaf = n as f64 * f_total * c_max / 2.0;

        let d_total = d_rbc + d_leaf;
        let flop_total = d_total * 2.0 * d as f64;
        let t_compute = flop_total / (gemm_flops_per_core * num_threads.max(1) as f64);

        t_compute * 1.15 // 15% overhead
    }

    #[inline]
    pub fn effective_oom_memory_budget_bytes(&self) -> usize {
        if self.oom_memory_budget_bytes == 0 {
            auto_oom_memory_budget_bytes()
        } else {
            self.oom_memory_budget_bytes
                .max(Self::MIN_OOM_MEMORY_BUDGET_BYTES)
        }
    }

    #[inline]
    pub fn strict_oom_io_enabled(&self) -> bool {
        self.oom_enable && matches!(self.effective_oom_io_mode(), PiPNNOomIoMode::Direct)
    }

    #[inline]
    pub fn effective_oom_io_mode(&self) -> PiPNNOomIoMode {
        match self.oom_io_mode {
            PiPNNOomIoMode::Auto if self.oom_strict_io => PiPNNOomIoMode::Direct,
            PiPNNOomIoMode::Auto => PiPNNOomIoMode::Buffered,
            mode => mode,
        }
    }

    #[inline]
    pub fn effective_oom_reservoir_budget_share(&self) -> f64 {
        if self.oom_reservoir_budget_share.is_finite() {
            self.oom_reservoir_budget_share.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    #[inline]
    pub fn effective_gpu_workspace_budget_bytes(&self) -> usize {
        if self.gpu_workspace_budget_bytes == 0 {
            Self::DEFAULT_GPU_WORKSPACE_BUDGET_BYTES
        } else {
            self.gpu_workspace_budget_bytes
        }
    }

    #[inline]
    pub fn effective_gpu_subtree_vector_staging_budget_bytes(&self) -> usize {
        const DEFAULT_NO_OOM_BUDGET: usize = 1024 * 1024 * 1024;
        const MAX_AUTO_BUDGET: usize = 2 * 1024 * 1024 * 1024;
        if self.gpu_subtree_vector_staging_budget_bytes != 0 {
            return self.gpu_subtree_vector_staging_budget_bytes;
        }
        if self.oom_enable {
            (self.effective_oom_memory_budget_bytes() / 16).min(MAX_AUTO_BUDGET)
        } else {
            DEFAULT_NO_OOM_BUDGET
        }
    }
}

#[inline]
fn nonzero_or(value: usize, default_value: usize) -> usize {
    if value == 0 { default_value } else { value }
}

impl Default for PiPNNParams {
    fn default() -> Self {
        Self {
            m_hash_bits: 12,
            l_max: 64,
            prune_backend: PiPNNPruneBackend::HashPrune,
            hashprune_evidence_mode: PiPNNHashPruneEvidenceMode::Off,
            hashprune_evidence_overflow_budget: 0,
            hashprune_backfill_mode: PiPNNHashPruneBackfillMode::Off,
            hashprune_backfill_ablation: PiPNNHashPruneBackfillAblation::None,
            hashprune_backfill_budget: 0,
            hashprune_backfill_alpha: 1.0,
            hashprune_rhalf_hash_quota: None,
            hashprune_soft_tier_priority_alpha: None,
            hashprune_rhalf_max_backfill_per_row: None,
            lrpf_pre_spill_compress_enable: false,
            lrpf_pre_spill_width_multiplier: 2.0,
            lrpf_protected_core: 1,
            lrpf_target_degree: 64,
            lrpf_long_edge_quota: 0,
            c_min: 256,
            c_max: 2048,
            psamp_fraction: 0.001,
            max_leaders: 1000,
            max_depth: 10,
            min_shrink_ratio: 1.5, // 提高收缩率阈值，更激进地早停
            fanout_top: 10,
            fanout_second: 3,
            root_fanout_policy: RootFanoutPolicy::Fixed,
            root_radius_k: 10,
            root_radius_candidate_rs: vec![4, 6, 8],
            root_radius_warmup_points: 500_000,
            root_radius_dry_run: false,
            root_stop_after_d00: false,
            root_risk_target_avg_fanout: 8.0,
            root_risk_sigma_lambda: 1.0,
            root_risk_budget_lambda: None,
            root_risk_sigma_mode: RootRiskSigmaMode::CellQ90Q50,
            root_risk_warmup_points: 500_000,
            root_risk_candidate_rs: vec![4, 6, 8, 10],
            root_risk_dry_run: false,
            root_risk_allow_full_build: false,
            partitioner: PiPNNPartitionerKind::Rbc,
            gpu_subtree_root_assignment_backend: PiPNNComputeBackendKind::Cpu,
            gpu_subtree_assignment_backend: PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32,
            gpu_subtree_assignment_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            gpu_subtree_assignment_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            gpu_subtree_assignment_dco_proj_dim: 0,
            gpu_subtree_assignment_dco_survivors: 0,
            gpu_subtree_assignment_dco_audit: false,
            gpu_subtree_d0_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            gpu_subtree_d0_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            gpu_subtree_d0_dco_survivors: 0,
            gpu_subtree_d0_dco_audit: false,
            gpu_subtree_d0_sq_cpu_kernel: SqInt8CpuKernel::Auto,
            gpu_subtree_execution_mode: PiPNNGpuSubtreeExecutionMode::FusedPersistent,
            gpu_subtree_leaf_topk_backend: PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32,
            gpu_subtree_leaf_dco_mode: PiPNNGpuSubtreeDcoMode::None,
            gpu_subtree_leaf_dco_rerank_mode: PiPNNGpuSubtreeDcoRerankMode::Exact,
            gpu_subtree_leaf_dco_proj_dim: 0,
            gpu_subtree_leaf_dco_survivors: 0,
            gpu_subtree_leaf_dco_audit: false,
            gpu_subtree_leaf_dco_ddc_res_pca_matrix: None,
            gpu_subtree_leaf_dco_ddc_res_projected_input: false,
            gpu_subtree_leaf_dco_ddc_res_base_dim: 32,
            gpu_subtree_leaf_dco_ddc_res_sigma_count: 3.0,
            gpu_subtree_rabitq_plus_mode: PiPNNGpuSubtreeRabitqPlusMode::Off,
            gpu_subtree_rabitq_plus_bits: 8,
            gpu_subtree_rabitq_plus_rotator: PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            gpu_subtree_rabitq_plus_seed: 42,
            gpu_subtree_rabitq_plus_sidecar_prefix: None,
            gpu_subtree_rabitq_plus_sidecar_mode: PiPNNGpuSubtreeRabitqPlusSidecarMode::Off,
            gpu_subtree_qcode_staging_mode: PiPNNGpuSubtreeQcodeStagingMode::Off,
            gpu_subtree_qcode_d1_payload_mode: PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident,
            gpu_subtree_rabitq_d1_distance_mode: PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma,
            gpu_subtree_rabitq_short_survivors: None,
            gpu_subtree_rabitq_residual_centroids: 16,
            gpu_subtree_qcode_staging_upload_shard_rows: 524_288,
            gpu_subtree_qcode_global_d0_leaders: 0,
            gpu_subtree_rabitq_d0_assign_mode: PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial,
            gpu_subtree_rabitq_d1_topk_mode: PiPNNGpuSubtreeRabitqD1TopKMode::RowSerial,
            gpu_subtree_rabitq_d0_row_tile: 4,
            gpu_subtree_rabitq_d0_row_chunk: 4096,
            gpu_subtree_rabitq_d0_leader_tile: 64,
            gpu_subtree_rabitq_d0_partial_r: 8,
            gpu_subtree_sq_staging_mode: PiPNNGpuSubtreeSqStagingMode::Off,
            gpu_subtree_sq_staging_upload_shard_rows: 524_288,
            gpu_subtree_sq_staging_raw_d1_diagnostic: false,
            gpu_subtree_sq_bits: 8,
            gpu_subtree_sq_native_int4: false,
            gpu_subtree_sq_wmma_mode: PiPNNGpuSubtreeSqWmmaMode::Off,
            gpu_subtree_sq_wmma_sidecar_prefix: None,
            gpu_subtree_sq_wmma_sidecar_mode: PiPNNGpuSubtreeSqWmmaSidecarMode::Off,
            gpu_subtree_turboquant_mode: PiPNNGpuSubtreeTurboQuantMode::Off,
            gpu_subtree_turboquant_bits: 4,
            gpu_subtree_turboquant_seed: 42,
            gpu_subtree_turboquant_sidecar_prefix: None,
            gpu_subtree_turboquant_sidecar_mode: PiPNNGpuSubtreeTurboQuantSidecarMode::Off,
            gpu_subtree_turboquant_cascade_low_dim: 128,
            gpu_subtree_turboquant_cascade_survivor_cap: 0,
            gpu_subtree_turboquant_adaptive_survivor_threshold: 0,
            gpu_subtree_turboquant_adaptive_survivor_low_cap: 0,
            gpu_subtree_persistent_leaf_tile_rows: 128,
            gpu_subtree_persistent_large_ticket_batch:
                PiPNNGpuSubtreePersistentLargeTicketBatch::Auto,
            gpu_subtree_persistent_scheduler_mode: PiPNNGpuSubtreePersistentSchedulerMode::Auto,
            gpu_subtree_persistent_idle_backoff_ns: 350,
            gpu_subtree_sq8_d0_assignment_kernel: PiPNNGpuSubtreeSq8D0AssignmentKernel::Auto,
            gpu_subtree_sq8_assignment_kernel: PiPNNGpuSubtreeSq8AssignmentKernel::Auto,
            gpu_subtree_sq8_leaf_executor: PiPNNGpuSubtreeSq8LeafExecutor::Auto,
            gpu_subtree_tessa_capture_dir: None,
            gpu_subtree_sq8_leaf_mode: PiPNNGpuSubtreeSq8LeafMode::Auto,
            gpu_subtree_d0_partition_mode: PiPNNGpuSubtreeD0PartitionMode::RbcFanout,
            gpu_subtree_d1_root_partition_mode: PiPNNGpuSubtreeD1RootPartitionMode::Off,
            gpu_subtree_d0_tagore_partitions: 32,
            gpu_subtree_d0_tagore_k: 2,
            gpu_subtree_d0_tagore_sample_fraction: 0.01,
            gpu_subtree_d0_capacity_enable: true,
            gpu_subtree_d1plus_lazy_split_mode: PiPNNGpuSubtreeD1PlusLazySplitMode::On,
            gpu_subtree_d0_capacity_high_ratio: 0.60,
            gpu_subtree_d0_capacity_target_ratio: 0.50,
            gpu_subtree_d0_capacity_low_ratio: 0.15,
            gpu_subtree_d0_capacity_min_efficient_rows: 32 * 1024,
            gpu_subtree_d0_capacity_quality_min_root_leaders: 64,
            gpu_subtree_d0_capacity_root_leader_cap: 10_000,
            gpu_subtree_d0_capacity_split_recursion_limit: 8,
            gpu_subtree_d0_throughput_model: PiPNNGpuSubtreeD0ThroughputModelMode::Off,
            gpu_subtree_d0_throughput_persistent_growth_limit: 1.10,
            gpu_subtree_d0_throughput_preload_visible_limit_ms: 5000,
            gpu_subtree_d0_throughput_min_ratio: 0.50,
            gpu_subtree_d0_throughput_calibration_path: None,
            gpu_subtree_d0_throughput_child_p95_guardrail_rows: 0,
            gpu_subtree_persistent_launch_policy: PiPNNGpuSubtreePolicy::Report,
            gpu_subtree_hbm_policy: PiPNNGpuSubtreePolicy::Report,
            gpu_subtree_launch_calibration_cache: None,
            gpu_subtree_launch_calibration_refresh: false,
            gpu_subtree_vector_staging_mode: PiPNNGpuSubtreeVectorStagingMode::Off,
            gpu_subtree_vector_staging_budget_bytes: 0,
            gpu_subtree_vector_staging_window_children: 8,
            gpu_subtree_child_schedule_mode: PiPNNGpuSubtreeChildScheduleMode::Natural,
            compact_lune_prune_enable: false,
            compact_lune_witness_enable: false,
            compact_lune_prune_protected_core: 1,
            compact_spill_digest_enable: false,
            compact_spill_digest_width_multiplier: 2.0,
            kmeans_iterations: 4,
            kmeans_oversplit_factor: 2,
            leaf_knn: 2,
            max_full_matrix_leaf_size: 3500,
            final_prune: true,
            final_prune_alpha: 1.2,
            random_seed: 42,
            metric: Metric::L2,
            compute_backend: PiPNNComputeBackendKind::Cpu,
            gpu_device_id: 0,
            gpu_workspace_budget_bytes: 0,
            cuda_compact_dump_dir: None,
            cuda_compact_dump_limit: 0,
            pipeline_runtime_enable: false,
            pipeline_compute_mode: PiPNNPipelineComputeMode::Passthrough,
            pipeline_leaf_descriptor_queue_capacity: None,
            pipeline_staged_queue_capacity: None,
            pipeline_topk_queue_capacity: None,
            pipeline_edge_queue_capacity: None,
            pipeline_max_inflight_compute_tasks_active: None,
            pipeline_max_inflight_compute_tasks_drain: None,
            oom_enable: false,
            oom_memory_budget_bytes: 0,
            oom_strict_io: false,
            oom_io_mode: PiPNNOomIoMode::Auto,
            oom_reservoir_budget_share: 1.0,
            oom_sidecar_resident_budget_cap_bytes: None,
            oom_sketch_cache_bytes: 0,
            oom_raw_vector_cache_bytes: 0,
            oom_spill_cache_bytes: 0,
            online_suffix_reduce_mode: PiPNNOnlineSuffixReduceMode::Off,
            online_suffix_direct_queue_bytes: 0,
            online_suffix_state_cache_bytes: 0,
            online_suffix_state_page_cache_bytes: 0,
            online_suffix_state_writeback_bytes: 0,
            online_suffix_spill_buffer_bytes: 0,
            online_suffix_reorder_window_bytes: 0,
            online_suffix_read_window_bytes: 0,
            online_suffix_pinned_prefix_points: None,
            online_suffix_reduce_workers: 0,
            online_suffix_delta_budget_bytes: 0,
            online_suffix_delta_buffer_bytes: 0,
            online_suffix_range_points: 0,
            online_suffix_range_trigger_bytes: 0,
            online_suffix_max_runs_per_range: 0,
            online_suffix_compaction_workers: 0,
            online_suffix_cache_hot_updates: 0,
            online_suffix_state_overprovision_ratio: 2.0,
            oom_temp_dir: PathBuf::new(),
            oom_keep_artifacts: false,
            oom_profile_json: None,
            hashprune_backfill_diagnostics_prefix: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::{
        PiPNNComputeBackendKind, PiPNNGpuSubtreeAssignmentBackend,
        PiPNNGpuSubtreeChildScheduleMode, PiPNNGpuSubtreeD0PartitionMode,
        PiPNNGpuSubtreeD0ThroughputModelMode, PiPNNGpuSubtreeD1PlusLazySplitMode,
        PiPNNGpuSubtreeD1RootPartitionMode, PiPNNGpuSubtreeDcoMode, PiPNNGpuSubtreeDcoRerankMode,
        PiPNNGpuSubtreeExecutionMode, PiPNNGpuSubtreeLeafTopKBackend,
        PiPNNGpuSubtreePersistentLargeTicketBatch, PiPNNGpuSubtreePersistentSchedulerMode,
        PiPNNGpuSubtreePolicy, PiPNNGpuSubtreeQcodeD1PayloadMode, PiPNNGpuSubtreeQcodeStagingMode,
        PiPNNGpuSubtreeRabitqD0AssignMode, PiPNNGpuSubtreeRabitqD1DistanceMode,
        PiPNNGpuSubtreeRabitqD1TopKMode, PiPNNGpuSubtreeRabitqPlusMode,
        PiPNNGpuSubtreeRabitqPlusRotator, PiPNNGpuSubtreeRabitqPlusSidecarMode,
        PiPNNGpuSubtreeSq8AssignmentKernel, PiPNNGpuSubtreeSq8D0AssignmentKernel,
        PiPNNGpuSubtreeSq8LeafExecutor, PiPNNGpuSubtreeSq8LeafMode, PiPNNGpuSubtreeSqStagingMode,
        PiPNNGpuSubtreeTurboQuantMode, PiPNNGpuSubtreeTurboQuantSidecarMode,
        PiPNNGpuSubtreeVectorStagingMode, PiPNNHashPruneBackfillAblation,
        PiPNNHashPruneBackfillMode, PiPNNHashPruneEvidenceMode, PiPNNOnlineSuffixReduceMode,
        PiPNNOomIoMode, PiPNNParams, PiPNNPartitionerKind, PiPNNPipelineComputeMode,
        PiPNNPruneBackend, RootFanoutPolicy, RootRiskSigmaMode,
        select_auto_oom_memory_budget_bytes,
    };
    use crate::common::Metric;

    #[test]
    fn d1plus_lazy_split_mode_defaults_on_and_parses_off() {
        assert_eq!(
            PiPNNParams::default().gpu_subtree_d1plus_lazy_split_mode,
            PiPNNGpuSubtreeD1PlusLazySplitMode::On
        );
        assert_eq!(
            "off".parse::<PiPNNGpuSubtreeD1PlusLazySplitMode>().unwrap(),
            PiPNNGpuSubtreeD1PlusLazySplitMode::Off
        );
        assert_eq!(
            "on".parse::<PiPNNGpuSubtreeD1PlusLazySplitMode>()
                .unwrap()
                .as_str(),
            "on"
        );
    }

    #[test]
    fn prune_backend_parses_hashprune_and_lrpf() {
        assert_eq!(
            PiPNNPruneBackend::from_str("hashprune").unwrap(),
            PiPNNPruneBackend::HashPrune
        );
        assert_eq!(
            PiPNNPruneBackend::from_str("lrpf").unwrap(),
            PiPNNPruneBackend::LRPF
        );
    }

    #[test]
    fn default_prune_backend_is_hashprune_with_lrpf_pre_spill_off() {
        let params = PiPNNParams::default();
        assert_eq!(params.prune_backend, PiPNNPruneBackend::HashPrune);
        assert!(!params.lrpf_pre_spill_compress_enable);
        assert_eq!(params.lrpf_target_degree, 64);
        assert_eq!(params.lrpf_long_edge_quota, 0);
    }

    #[test]
    fn hashprune_evidence_defaults_to_off_with_zero_budget() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.hashprune_evidence_mode,
            PiPNNHashPruneEvidenceMode::Off
        );
        assert_eq!(params.hashprune_evidence_overflow_budget, 0);
        assert!(params.validate_hashprune_evidence_config().is_ok());
    }

    #[test]
    fn hashprune_evidence_rejects_invalid_pairs() {
        let mut params = PiPNNParams::default();
        params.hashprune_evidence_mode = PiPNNHashPruneEvidenceMode::Report;
        params.hashprune_evidence_overflow_budget = 1;
        assert!(params.validate_hashprune_evidence_config().is_err());

        params.hashprune_evidence_mode = PiPNNHashPruneEvidenceMode::Enforce;
        params.prune_backend = PiPNNPruneBackend::LRPF;
        assert!(params.validate_hashprune_evidence_config().is_err());
    }

    #[test]
    fn hashprune_backfill_config_defaults_to_canonical_off() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.hashprune_backfill_mode,
            PiPNNHashPruneBackfillMode::Off
        );
        assert_eq!(params.hashprune_backfill_budget, 0);
        assert_eq!(params.hashprune_backfill_alpha, 1.0);
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_named_rhalf_tuning_defaults_to_none() {
        let params = PiPNNParams::default();
        assert_eq!(params.hashprune_rhalf_hash_quota, None);
        assert_eq!(params.hashprune_soft_tier_priority_alpha, None);
        assert_eq!(params.hashprune_rhalf_max_backfill_per_row, None);
    }

    #[test]
    fn hashprune_named_rhalf_accepts_row_caps_through_r_half() {
        for mode in [
            PiPNNHashPruneBackfillMode::RHalfHashSlot1,
            PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
        ] {
            for cap in [1, 2, 3, 4, 32] {
                let mut params = PiPNNParams::default();
                params.l_max = 64;
                params.hashprune_backfill_mode = mode;
                params.hashprune_rhalf_max_backfill_per_row = Some(cap);
                assert!(params.validate_hashprune_backfill_config().is_ok());
            }
        }
    }

    #[test]
    fn hashprune_named_rhalf_rejects_invalid_row_caps() {
        for cap in [0, 33] {
            let mut params = PiPNNParams::default();
            params.l_max = 64;
            params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3;
            params.hashprune_rhalf_max_backfill_per_row = Some(cap);
            assert!(params.validate_hashprune_backfill_config().is_err());
        }
        for mode in [
            PiPNNHashPruneBackfillMode::Off,
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf,
        ] {
            let mut params = PiPNNParams::default();
            params.hashprune_backfill_mode = mode;
            params.hashprune_rhalf_max_backfill_per_row = Some(2);
            if mode == PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf {
                params.hashprune_backfill_alpha = 1.1;
            }
            assert!(params.validate_hashprune_backfill_config().is_err());
        }
    }

    #[test]
    fn hashprune_named_rhalf_accepts_orthogonal_tuning_domains() {
        for mode in [
            PiPNNHashPruneBackfillMode::RHalfHashSlot1,
            PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
        ] {
            for quota in 1..=3 {
                let mut params = PiPNNParams::default();
                params.hashprune_backfill_mode = mode;
                params.hashprune_rhalf_hash_quota = Some(quota);
                assert!(params.validate_hashprune_backfill_config().is_ok());
            }
        }
        for alpha in [1.0, 1.05, 1.1, 1.2, 1.4, 2.0] {
            let mut params = PiPNNParams::default();
            params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
            params.hashprune_soft_tier_priority_alpha = Some(alpha);
            assert!(params.validate_hashprune_backfill_config().is_ok());
        }
    }

    #[test]
    fn hashprune_named_rhalf_rejects_invalid_tuning_domains() {
        for quota in [0, 4] {
            let mut params = PiPNNParams::default();
            params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3;
            params.hashprune_rhalf_hash_quota = Some(quota);
            assert!(params.validate_hashprune_backfill_config().is_err());
        }
        for alpha in [f32::NAN, f32::INFINITY, 0.99] {
            let mut params = PiPNNParams::default();
            params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
            params.hashprune_soft_tier_priority_alpha = Some(alpha);
            assert!(params.validate_hashprune_backfill_config().is_err());
        }
        let mut wrong_mode = PiPNNParams::default();
        wrong_mode.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfHardTierQ3;
        wrong_mode.hashprune_soft_tier_priority_alpha = Some(1.2);
        assert!(wrong_mode.validate_hashprune_backfill_config().is_err());
    }

    #[test]
    fn hashprune_diagnostics_reject_named_policy_tuning_overrides() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_backfill_diagnostics_prefix = Some(PathBuf::from("/tmp/hashprune-diag"));

        let mut quota = params.clone();
        quota.hashprune_rhalf_hash_quota = Some(2);
        assert!(quota.validate_hashprune_policies().is_err());

        params.hashprune_soft_tier_priority_alpha = Some(1.2);
        assert!(params.validate_hashprune_policies().is_err());
    }

    #[test]
    fn hashprune_diagnostics_accept_named_row_cap() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_backfill_diagnostics_prefix = Some(PathBuf::from("/tmp/hashprune-diag"));
        params.hashprune_rhalf_max_backfill_per_row = Some(3);
        assert!(params.validate_hashprune_policies().is_ok());
    }

    #[test]
    fn hashprune_diagnostics_default_to_disabled() {
        let params = PiPNNParams::default();
        assert_eq!(params.hashprune_backfill_diagnostics_prefix, None);
        assert!(params.validate_hashprune_policies().is_ok());
    }

    #[test]
    fn hashprune_diagnostics_accept_only_named_softtier_l2_u8_degree() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_backfill_diagnostics_prefix = Some(PathBuf::from("/tmp/hashprune-diag"));
        assert!(params.validate_hashprune_policies().is_ok());

        let mutations: [fn(&mut PiPNNParams); 5] = [
            |p: &mut PiPNNParams| p.prune_backend = PiPNNPruneBackend::LRPF,
            |p: &mut PiPNNParams| {
                p.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::Off;
            },
            |p: &mut PiPNNParams| p.metric = Metric::Cosine,
            |p: &mut PiPNNParams| p.l_max = usize::from(u8::MAX) + 1,
            |p: &mut PiPNNParams| p.root_stop_after_d00 = true,
        ];
        for mutate in mutations {
            let mut invalid = params.clone();
            mutate(&mut invalid);
            assert!(invalid.validate_hashprune_policies().is_err());
        }
    }

    #[test]
    fn hashprune_named_rhalf_modes_parse_display_and_validate_defaults() {
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
            let parsed = raw.parse::<PiPNNHashPruneBackfillMode>().unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.to_string(), raw);

            let mut params = PiPNNParams::default();
            params.l_max = 64;
            params.hashprune_backfill_mode = parsed;
            assert!(params.validate_hashprune_backfill_config().is_ok());
        }
    }

    #[test]
    fn hashprune_named_rhalf_modes_reject_legacy_tuning_knobs() {
        for mode in [
            PiPNNHashPruneBackfillMode::RHalfHashSlot1,
            PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3,
            PiPNNHashPruneBackfillMode::RHalfSoftTierQ3,
            PiPNNHashPruneBackfillMode::RHalfHardTierQ3,
        ] {
            let mut params = PiPNNParams::default();
            params.l_max = 64;
            params.hashprune_backfill_mode = mode;

            let mut invalid = params.clone();
            invalid.hashprune_backfill_ablation = PiPNNHashPruneBackfillAblation::NoVeto;
            assert!(invalid.validate_hashprune_backfill_config().is_err());

            invalid = params.clone();
            invalid.hashprune_backfill_budget = 1;
            assert!(invalid.validate_hashprune_backfill_config().is_err());

            invalid = params;
            invalid.hashprune_backfill_alpha = 1.1;
            assert!(invalid.validate_hashprune_backfill_config().is_err());
        }
    }

    #[test]
    fn hashprune_tiered_d52_mode_parses_and_validates_k8_tuple() {
        assert_eq!(
            "distance-gated-tiered-d52"
                .parse::<PiPNNHashPruneBackfillMode>()
                .unwrap(),
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52
        );
        assert_eq!(
            "distance_gated_tiered_d52"
                .parse::<PiPNNHashPruneBackfillMode>()
                .unwrap(),
            PiPNNHashPruneBackfillMode::DistanceGatedTieredD52
        );
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGatedTieredD52;
        params.hashprune_backfill_budget = 8;
        params.hashprune_backfill_alpha = 1.1;
        assert!(params.validate_hashprune_backfill_config().is_ok());
    }

    #[test]
    fn hashprune_tiered_rhalf_config_parses_frozen_mode_and_ablations() {
        assert_eq!(
            "distance-gated-tiered-rhalf"
                .parse::<PiPNNHashPruneBackfillMode>()
                .unwrap(),
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
        );
        assert_eq!(
            "distance_gated_tiered_rhalf"
                .parse::<PiPNNHashPruneBackfillMode>()
                .unwrap(),
            PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf
        );
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
            assert_eq!(
                raw.parse::<PiPNNHashPruneBackfillAblation>().unwrap(),
                expected
            );
        }
    }

    #[test]
    fn hashprune_tiered_rhalf_config_accepts_only_frozen_contract() {
        let mut params = PiPNNParams::default();
        params.l_max = 64;
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf;
        params.hashprune_backfill_ablation = PiPNNHashPruneBackfillAblation::None;
        params.hashprune_backfill_budget = 0;
        params.hashprune_backfill_alpha = 1.1;
        assert!(params.validate_hashprune_backfill_config().is_ok());

        for ablation in [
            PiPNNHashPruneBackfillAblation::BudgetQuarter,
            PiPNNHashPruneBackfillAblation::NoHalfCap,
            PiPNNHashPruneBackfillAblation::Quota1,
            PiPNNHashPruneBackfillAblation::Quota2,
            PiPNNHashPruneBackfillAblation::DistanceFirst,
            PiPNNHashPruneBackfillAblation::NoVeto,
        ] {
            let mut ablated = params.clone();
            ablated.hashprune_backfill_ablation = ablation;
            assert!(ablated.validate_hashprune_backfill_config().is_ok());
        }

        let mut invalid = params.clone();
        invalid.l_max = 0;
        assert!(invalid.validate_hashprune_backfill_config().is_err());
        invalid = params.clone();
        invalid.l_max = usize::from(u16::MAX) + 1;
        assert!(invalid.validate_hashprune_backfill_config().is_err());
        invalid = params.clone();
        invalid.hashprune_backfill_budget = 1;
        assert!(invalid.validate_hashprune_backfill_config().is_err());
        invalid = params.clone();
        invalid.hashprune_backfill_alpha = 1.0;
        assert!(invalid.validate_hashprune_backfill_config().is_err());
        invalid = params.clone();
        invalid.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGated;
        invalid.hashprune_backfill_budget = 4;
        invalid.hashprune_backfill_ablation = PiPNNHashPruneBackfillAblation::Quota1;
        assert!(invalid.validate_hashprune_backfill_config().is_err());
    }

    #[test]
    fn hashprune_backfill_config_accepts_approved_experiment_tuples() {
        for mode in [
            PiPNNHashPruneBackfillMode::DistanceGated,
            PiPNNHashPruneBackfillMode::DistanceGatedStratified,
            PiPNNHashPruneBackfillMode::DistanceGatedMultihit,
        ] {
            for budget in [4, 8] {
                for alpha in [1.0, 1.1, 1.2] {
                    let mut params = PiPNNParams::default();
                    params.hashprune_backfill_mode = mode;
                    params.hashprune_backfill_budget = budget;
                    params.hashprune_backfill_alpha = alpha;
                    assert!(params.validate_hashprune_backfill_config().is_ok());
                }
            }
        }
    }

    #[test]
    fn hashprune_backfill_config_rejects_invalid_and_incompatible_tuples() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_budget = 1;
        assert!(params.validate_hashprune_backfill_config().is_err());

        params.hashprune_backfill_budget = 0;
        params.hashprune_backfill_alpha = 1.1;
        assert!(params.validate_hashprune_backfill_config().is_err());

        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGated;
        params.hashprune_backfill_alpha = 1.0;
        params.hashprune_backfill_budget = 0;
        assert!(params.validate_hashprune_backfill_config().is_err());

        params.hashprune_backfill_budget = 9;
        assert!(params.validate_hashprune_backfill_config().is_err());

        params.hashprune_backfill_budget = 8;
        for alpha in [0.99, f32::NAN, f32::INFINITY] {
            params.hashprune_backfill_alpha = alpha;
            assert!(params.validate_hashprune_backfill_config().is_err());
        }

        params.hashprune_backfill_alpha = 1.1;
        params.hashprune_evidence_mode = PiPNNHashPruneEvidenceMode::Report;
        assert!(params.validate_hashprune_backfill_config().is_err());

        params.hashprune_evidence_mode = PiPNNHashPruneEvidenceMode::Off;
        params.prune_backend = PiPNNPruneBackend::LRPF;
        assert!(params.validate_hashprune_backfill_config().is_err());

        params.prune_backend = PiPNNPruneBackend::HashPrune;
        params.metric = Metric::Cosine;
        assert!(params.validate_hashprune_backfill_config().is_err());
    }

    #[test]
    fn adaptive_psamp_uses_fixed_value_with_mild_depth_decay() {
        let mut params = PiPNNParams::default();
        params.psamp_fraction = 0.001;

        assert_eq!(params.adaptive_psamp_fraction(10_000, 0), 0.001);
        assert_eq!(params.adaptive_psamp_fraction(10_000_000, 0), 0.0001);
        assert_eq!(params.adaptive_psamp_fraction(10_000_000, 3), 0.00005);
    }

    #[test]
    fn adaptive_psamp_is_clamped_to_bounds() {
        let mut params = PiPNNParams::default();

        params.psamp_fraction = 0.5;
        assert_eq!(params.adaptive_psamp_fraction(1_000, 0), 0.1);

        params.psamp_fraction = 0.000001;
        assert_eq!(params.adaptive_psamp_fraction(1_000, 0), 0.00005);
    }

    #[test]
    fn adaptive_c_max_returns_user_configured_value() {
        let mut params = PiPNNParams::default();
        params.c_max = 2048;
        assert_eq!(params.adaptive_c_max(2_000_000), 2048);
        assert_eq!(params.adaptive_c_max(6_000_000), 4096);
        assert_eq!(params.adaptive_c_max(12_000_000), 4096);
        assert_eq!(params.adaptive_c_max(35_000_000), 4096);
    }

    #[test]
    fn adaptive_fanout_matches_depth_schedule() {
        let mut params = PiPNNParams::default();
        params.fanout_top = 10;
        params.fanout_second = 3;

        assert_eq!(params.adaptive_fanout(10_000, 0), 10);
        assert_eq!(params.adaptive_fanout(2_000_000, 1), 3);
        assert_eq!(params.adaptive_fanout(20_000_000, 4), 1);
    }

    #[test]
    fn recommend_produces_valid_params() {
        let params = PiPNNParams::recommend(35_000_000, 768, 64, 42);
        assert!(params.c_max >= 512 && params.c_max <= 2048);
        assert!(params.max_leaders >= 500 && params.max_leaders <= 10000);
        assert!(params.psamp_fraction >= 0.001 && params.psamp_fraction <= 0.1);
        assert_eq!(params.fanout_top, 10);
        assert_eq!(params.fanout_second, 2);
        assert_eq!(params.leaf_knn, 2);
        assert_eq!(params.l_max, 64);
    }

    #[test]
    fn recommend_wiki35m_10x2_matches_historical_c_max_baseline() {
        let params = PiPNNParams::recommend(35_167_920, 768, 64, 42);

        assert_eq!(params.fanout_top, 10);
        assert_eq!(params.fanout_second, 2);
        assert_eq!(params.c_max, 1280);
        assert_eq!(params.adaptive_c_max(35_167_920), 2560);
    }

    #[test]
    fn recommend_small_dataset() {
        let params = PiPNNParams::recommend(100_000, 128, 32, 8);
        assert!(params.c_max >= 512);
        assert!(params.max_leaders >= 500);
        assert_eq!(params.fanout_top, 10);
        assert_eq!(params.fanout_second, 3);
    }

    #[test]
    fn estimate_build_time_is_positive() {
        let params = PiPNNParams::recommend(35_000_000, 768, 64, 42);
        let est = params.estimate_build_time(35_000_000, 768, 42);
        assert!(est > 0.0);
        // 应该在合理范围内（几十秒到几千秒）
        assert!(est < 10000.0, "estimate too large: {}", est);
    }

    #[test]
    fn estimate_scales_with_threads() {
        let params = PiPNNParams::default();
        let t1 = params.estimate_build_time(1_000_000, 128, 1);
        let t8 = params.estimate_build_time(1_000_000, 128, 8);
        // 8 线程应该比 1 线程快约 8 倍
        assert!((t1 / t8 - 8.0).abs() < 0.1);
    }

    #[test]
    fn backend_kind_parses_known_values_case_insensitively() {
        assert_eq!(
            PiPNNComputeBackendKind::from_str("cpu").unwrap(),
            PiPNNComputeBackendKind::Cpu
        );
        assert_eq!(
            PiPNNComputeBackendKind::from_str("AUTO").unwrap(),
            PiPNNComputeBackendKind::Auto
        );
        assert_eq!(
            PiPNNComputeBackendKind::from_str("Cuda").unwrap(),
            PiPNNComputeBackendKind::Cuda
        );
    }

    #[test]
    fn backend_kind_rejects_unknown_values() {
        let err = PiPNNComputeBackendKind::from_str("metal").unwrap_err();
        assert!(err.contains("Unsupported PiPNN backend"));
    }

    #[test]
    fn gpu_subtree_execution_mode_parses_supported_values() {
        assert_eq!(
            PiPNNGpuSubtreeExecutionMode::from_str("fused-persistent").unwrap(),
            PiPNNGpuSubtreeExecutionMode::FusedPersistent
        );
        assert_eq!(
            PiPNNGpuSubtreeExecutionMode::from_str("two-stage").unwrap(),
            PiPNNGpuSubtreeExecutionMode::TwoStage
        );
        assert_eq!(
            PiPNNGpuSubtreeExecutionMode::from_str("two_stage").unwrap(),
            PiPNNGpuSubtreeExecutionMode::TwoStage
        );
        assert_eq!(
            PiPNNGpuSubtreeExecutionMode::TwoStage.to_string(),
            "two-stage"
        );
        assert!(PiPNNGpuSubtreeExecutionMode::from_str("pipeline").is_err());
    }

    #[test]
    fn pipnn_params_default_gpu_subtree_execution_mode_is_fused_persistent() {
        let params = PiPNNParams::default();

        assert_eq!(
            params.gpu_subtree_execution_mode,
            PiPNNGpuSubtreeExecutionMode::FusedPersistent
        );
    }

    #[test]
    fn partitioner_kind_parses_supported_modes() {
        assert_eq!(
            PiPNNPartitionerKind::from_str("rbc").unwrap(),
            PiPNNPartitionerKind::Rbc
        );
        assert_eq!(
            PiPNNPartitionerKind::from_str("kmeans").unwrap(),
            PiPNNPartitionerKind::KMeans
        );
        assert_eq!(
            PiPNNPartitionerKind::from_str("k-means").unwrap(),
            PiPNNPartitionerKind::KMeans
        );
        assert_eq!(
            PiPNNPartitionerKind::from_str("gpu-subtree").unwrap(),
            PiPNNPartitionerKind::GpuSubtree
        );
        assert_eq!(
            PiPNNPartitionerKind::from_str("gpu_subtree").unwrap(),
            PiPNNPartitionerKind::GpuSubtree
        );
        assert_eq!(PiPNNPartitionerKind::KMeans.to_string(), "kmeans");
        assert_eq!(PiPNNPartitionerKind::GpuSubtree.to_string(), "gpu-subtree");
    }

    #[test]
    fn partitioner_kind_rejects_unknown_values() {
        let err = PiPNNPartitionerKind::from_str("ivf").unwrap_err();
        assert!(err.contains("Unsupported PiPNN partitioner"));
    }

    #[test]
    fn gpu_subtree_d0_partition_mode_parses_tagore_aliases() {
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::from_str("rbc-fanout").unwrap(),
            PiPNNGpuSubtreeD0PartitionMode::RbcFanout
        );
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::from_str("tagore-kmeans-topk").unwrap(),
            PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK
        );
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::from_str("tagore").unwrap(),
            PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK
        );
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::TagoreKMeansTopK.to_string(),
            "tagore-kmeans-topk"
        );
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::from_str("gpu-kmeans-coarse").unwrap(),
            PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
        );
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::from_str("gpu_kmeans").unwrap(),
            PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse
        );
        assert_eq!(
            PiPNNGpuSubtreeD0PartitionMode::GpuKMeansCoarse.to_string(),
            "gpu-kmeans-coarse"
        );
    }

    #[test]
    fn gpu_subtree_d1_root_partition_mode_parses_kmeans_coarse() {
        assert_eq!(
            PiPNNGpuSubtreeD1RootPartitionMode::from_str("off").unwrap(),
            PiPNNGpuSubtreeD1RootPartitionMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeD1RootPartitionMode::from_str("rbc").unwrap(),
            PiPNNGpuSubtreeD1RootPartitionMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeD1RootPartitionMode::from_str("kmeans-coarse").unwrap(),
            PiPNNGpuSubtreeD1RootPartitionMode::KMeansCoarse
        );
        assert_eq!(
            PiPNNGpuSubtreeD1RootPartitionMode::from_str("gpu_kmeans_coarse").unwrap(),
            PiPNNGpuSubtreeD1RootPartitionMode::KMeansCoarse
        );
        assert_eq!(
            PiPNNGpuSubtreeD1RootPartitionMode::KMeansCoarse.to_string(),
            "kmeans-coarse"
        );
    }

    #[test]
    fn gpu_subtree_qcode_staging_mode_parses_rabitq_residual_only() {
        assert_eq!(
            PiPNNGpuSubtreeQcodeStagingMode::from_str("off").unwrap(),
            PiPNNGpuSubtreeQcodeStagingMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeQcodeStagingMode::from_str("rabitq-residual").unwrap(),
            PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual
        );
        assert_eq!(
            PiPNNGpuSubtreeQcodeStagingMode::from_str("rabitq_residual").unwrap(),
            PiPNNGpuSubtreeQcodeStagingMode::RabitqResidual
        );
        assert!(PiPNNGpuSubtreeQcodeStagingMode::from_str("sq-packed-s4").is_err());
    }

    #[test]
    fn gpu_subtree_sq_staging_mode_parses_gpu_d0_only() {
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("off").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("gpu-d0").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::GpuD0
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("gpu_d0").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::GpuD0
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("gpu-d0-shard-local-d1").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("gpu_d0_shard_local_d1").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("gpu-d0-cluster-batch-d1").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::from_str("gpu_d0_cluster_batch_d1").unwrap(),
            PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1
        );
        assert_eq!(PiPNNGpuSubtreeSqStagingMode::GpuD0.to_string(), "gpu-d0");
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1.to_string(),
            "gpu-d0-shard-local-d1"
        );
        assert_eq!(
            PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1.to_string(),
            "gpu-d0-cluster-batch-d1"
        );
        assert!(PiPNNGpuSubtreeSqStagingMode::GpuD0.is_enabled());
        assert!(PiPNNGpuSubtreeSqStagingMode::GpuD0ShardLocalD1.is_enabled());
        assert!(PiPNNGpuSubtreeSqStagingMode::GpuD0ClusterBatchD1.is_enabled());
        assert!(!PiPNNGpuSubtreeSqStagingMode::Off.is_enabled());
        assert!(PiPNNGpuSubtreeSqStagingMode::from_str("rabitq-residual").is_err());
        assert!(PiPNNGpuSubtreeSqStagingMode::from_str("sq4").is_err());
    }

    #[test]
    fn gpu_subtree_qcode_d1_payload_mode_parses_child_local() {
        assert_eq!(
            PiPNNGpuSubtreeQcodeD1PayloadMode::from_str("full-resident").unwrap(),
            PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident
        );
        assert_eq!(
            PiPNNGpuSubtreeQcodeD1PayloadMode::from_str("full_resident").unwrap(),
            PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident
        );
        assert_eq!(
            PiPNNGpuSubtreeQcodeD1PayloadMode::from_str("child-local").unwrap(),
            PiPNNGpuSubtreeQcodeD1PayloadMode::ChildLocal
        );
        assert_eq!(
            PiPNNGpuSubtreeQcodeD1PayloadMode::from_str("child_local").unwrap(),
            PiPNNGpuSubtreeQcodeD1PayloadMode::ChildLocal
        );
        assert_eq!(
            PiPNNGpuSubtreeQcodeD1PayloadMode::ChildLocal.to_string(),
            "child-local"
        );
        assert!(PiPNNGpuSubtreeQcodeD1PayloadMode::from_str("sq-sidecar").is_err());
    }

    #[test]
    fn gpu_subtree_rabitq_d1_distance_mode_parses_split_short_ex() {
        assert_eq!(
            PiPNNGpuSubtreeRabitqD1DistanceMode::from_str("dense-i8-wmma").unwrap(),
            PiPNNGpuSubtreeRabitqD1DistanceMode::DenseI8Wmma
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD1DistanceMode::from_str("split-short-ex").unwrap(),
            PiPNNGpuSubtreeRabitqD1DistanceMode::SplitShortEx
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD1DistanceMode::SplitShortEx.to_string(),
            "split-short-ex"
        );
        assert!(PiPNNGpuSubtreeRabitqD1DistanceMode::from_str("raw-query-code").is_err());
    }

    #[test]
    fn gpu_subtree_rabitq_d0_assign_mode_parses_row_serial_tiled_batch_and_wmma_batch() {
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("row-serial").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("row_serial").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("tiled").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::Tiled
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("batch").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::Batch
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("batch-distance").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::Batch
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("wmma-batch").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::WmmaBatch
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::from_str("wmma_batch").unwrap(),
            PiPNNGpuSubtreeRabitqD0AssignMode::WmmaBatch
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::Tiled.to_string(),
            "tiled"
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::Batch.to_string(),
            "batch"
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD0AssignMode::WmmaBatch.to_string(),
            "wmma-batch"
        );
        assert!(PiPNNGpuSubtreeRabitqD0AssignMode::from_str("centroid-gated").is_err());
    }

    #[test]
    fn gpu_subtree_rabitq_d1_topk_mode_parses_row_serial_and_wmma_batch() {
        assert_eq!(
            PiPNNGpuSubtreeRabitqD1TopKMode::from_str("row-serial").unwrap(),
            PiPNNGpuSubtreeRabitqD1TopKMode::RowSerial
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD1TopKMode::from_str("wmma-batch").unwrap(),
            PiPNNGpuSubtreeRabitqD1TopKMode::WmmaBatch
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqD1TopKMode::WmmaBatch.to_string(),
            "wmma-batch"
        );
        assert!(PiPNNGpuSubtreeRabitqD1TopKMode::from_str("full-resident").is_err());
    }

    #[test]
    fn default_params_keep_qcode_staging_disabled() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.gpu_subtree_qcode_staging_mode,
            PiPNNGpuSubtreeQcodeStagingMode::Off
        );
        assert_eq!(
            params.gpu_subtree_qcode_d1_payload_mode,
            PiPNNGpuSubtreeQcodeD1PayloadMode::FullResident
        );
        assert_eq!(params.gpu_subtree_rabitq_residual_centroids, 16);
        assert_eq!(params.gpu_subtree_qcode_staging_upload_shard_rows, 524_288);
        assert_eq!(params.gpu_subtree_qcode_global_d0_leaders, 0);
        assert_eq!(
            params.gpu_subtree_rabitq_d0_assign_mode,
            PiPNNGpuSubtreeRabitqD0AssignMode::RowSerial
        );
        assert_eq!(
            params.gpu_subtree_rabitq_d1_topk_mode,
            PiPNNGpuSubtreeRabitqD1TopKMode::RowSerial
        );
        assert_eq!(params.gpu_subtree_rabitq_d0_row_tile, 4);
        assert_eq!(params.gpu_subtree_rabitq_d0_row_chunk, 4096);
        assert_eq!(params.gpu_subtree_rabitq_d0_leader_tile, 64);
        assert_eq!(params.gpu_subtree_rabitq_d0_partial_r, 8);
        assert_eq!(
            params.gpu_subtree_sq_staging_mode,
            PiPNNGpuSubtreeSqStagingMode::Off
        );
        assert_eq!(params.gpu_subtree_sq_staging_upload_shard_rows, 524_288);
    }

    #[test]
    fn gpu_subtree_vector_staging_mode_parses_supported_values() {
        assert_eq!(
            PiPNNGpuSubtreeVectorStagingMode::from_str("off").unwrap(),
            PiPNNGpuSubtreeVectorStagingMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeVectorStagingMode::from_str("profile-only").unwrap(),
            PiPNNGpuSubtreeVectorStagingMode::ProfileOnly
        );
        assert_eq!(
            PiPNNGpuSubtreeVectorStagingMode::from_str("fanout_slab").unwrap(),
            PiPNNGpuSubtreeVectorStagingMode::FanoutSlab
        );
        assert_eq!(
            PiPNNGpuSubtreeVectorStagingMode::from_str("rolling-cache").unwrap(),
            PiPNNGpuSubtreeVectorStagingMode::RollingCache
        );
        assert_eq!(
            PiPNNGpuSubtreeVectorStagingMode::FanoutSlab.to_string(),
            "fanout-slab"
        );
        assert_eq!(
            PiPNNGpuSubtreeVectorStagingMode::RollingCache.to_string(),
            "rolling-cache"
        );
        assert!(PiPNNGpuSubtreeVectorStagingMode::ProfileOnly.records_repeat_profile());
        assert!(PiPNNGpuSubtreeVectorStagingMode::RollingCache.records_repeat_profile());
        assert!(!PiPNNGpuSubtreeVectorStagingMode::Off.records_repeat_profile());
    }

    #[test]
    fn gpu_subtree_rabitq_plus_modes_parse_supported_values() {
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusMode::from_str("off").unwrap(),
            PiPNNGpuSubtreeRabitqPlusMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusMode::from_str("shadow").unwrap(),
            PiPNNGpuSubtreeRabitqPlusMode::Shadow
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusMode::from_str("quantized_only").unwrap(),
            PiPNNGpuSubtreeRabitqPlusMode::Build
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusRotator::from_str("fht_kac").unwrap(),
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusRotator::from_str("identity").unwrap(),
            PiPNNGpuSubtreeRabitqPlusRotator::None
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusSidecarMode::from_str("require").unwrap(),
            PiPNNGpuSubtreeRabitqPlusSidecarMode::Require
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusSidecarMode::from_str("create_only").unwrap(),
            PiPNNGpuSubtreeRabitqPlusSidecarMode::CreateOnly
        );
        assert_eq!(
            PiPNNGpuSubtreeRabitqPlusSidecarMode::from_str("rebuild").unwrap(),
            PiPNNGpuSubtreeRabitqPlusSidecarMode::Rebuild
        );
        assert!(PiPNNGpuSubtreeRabitqPlusMode::Build.is_enabled());
        assert!(!PiPNNGpuSubtreeRabitqPlusMode::Off.is_enabled());
    }

    #[test]
    fn gpu_subtree_turboquant_modes_parse_supported_values() {
        assert_eq!(
            PiPNNGpuSubtreeTurboQuantMode::from_str("off").unwrap(),
            PiPNNGpuSubtreeTurboQuantMode::Off
        );
        assert_eq!(
            PiPNNGpuSubtreeTurboQuantMode::from_str("shadow").unwrap(),
            PiPNNGpuSubtreeTurboQuantMode::Shadow
        );
        assert_eq!(
            PiPNNGpuSubtreeTurboQuantMode::from_str("quantized_only").unwrap(),
            PiPNNGpuSubtreeTurboQuantMode::Build
        );
        assert_eq!(
            PiPNNGpuSubtreeTurboQuantSidecarMode::from_str("require").unwrap(),
            PiPNNGpuSubtreeTurboQuantSidecarMode::Require
        );
        assert_eq!(
            PiPNNGpuSubtreeTurboQuantSidecarMode::from_str("create_only").unwrap(),
            PiPNNGpuSubtreeTurboQuantSidecarMode::CreateOnly
        );
        assert_eq!(
            PiPNNGpuSubtreeTurboQuantSidecarMode::from_str("rebuild").unwrap(),
            PiPNNGpuSubtreeTurboQuantSidecarMode::Rebuild
        );
        assert!(PiPNNGpuSubtreeTurboQuantMode::Build.is_enabled());
        assert!(!PiPNNGpuSubtreeTurboQuantMode::Off.is_enabled());
    }

    #[test]
    fn pipnn_params_default_turboquant_is_off_mse4_seed42() {
        let params = PiPNNParams::default();

        assert_eq!(
            params.gpu_subtree_turboquant_mode,
            PiPNNGpuSubtreeTurboQuantMode::Off
        );
        assert_eq!(params.gpu_subtree_turboquant_bits, 4);
        assert_eq!(params.gpu_subtree_turboquant_seed, 42);
        assert!(params.gpu_subtree_turboquant_sidecar_prefix.is_none());
        assert_eq!(
            params.gpu_subtree_turboquant_sidecar_mode,
            PiPNNGpuSubtreeTurboQuantSidecarMode::Off
        );
        assert_eq!(params.gpu_subtree_turboquant_cascade_low_dim, 128);
        assert_eq!(params.gpu_subtree_turboquant_cascade_survivor_cap, 0);
        assert_eq!(params.gpu_subtree_turboquant_adaptive_survivor_threshold, 0);
        assert_eq!(params.gpu_subtree_turboquant_adaptive_survivor_low_cap, 0);
        assert_eq!(params.gpu_subtree_sq_bits, 8);
        assert!(!params.gpu_subtree_sq_native_int4);
    }

    #[test]
    fn gpu_subtree_child_schedule_mode_parses_supported_values() {
        assert_eq!(
            PiPNNGpuSubtreeChildScheduleMode::from_str("natural").unwrap(),
            PiPNNGpuSubtreeChildScheduleMode::Natural
        );
        assert_eq!(
            PiPNNGpuSubtreeChildScheduleMode::from_str("overlap_greedy").unwrap(),
            PiPNNGpuSubtreeChildScheduleMode::OverlapGreedy
        );
        assert_eq!(
            PiPNNGpuSubtreeChildScheduleMode::OverlapGreedy.to_string(),
            "overlap-greedy"
        );
    }

    #[test]
    fn oom_io_mode_parses_supported_values_and_preserves_strict_alias() {
        assert_eq!(
            PiPNNOomIoMode::from_str("auto").unwrap(),
            PiPNNOomIoMode::Auto
        );
        assert_eq!(
            PiPNNOomIoMode::from_str("strict-direct").unwrap(),
            PiPNNOomIoMode::Direct
        );
        assert_eq!(
            PiPNNOomIoMode::from_str("io_buffer").unwrap(),
            PiPNNOomIoMode::Buffered
        );
        assert_eq!(PiPNNOomIoMode::Buffered.to_string(), "buffered");
    }

    #[test]
    fn oom_io_mode_effective_mode_honors_legacy_strict_flag() {
        let mut params = PiPNNParams {
            oom_enable: true,
            ..PiPNNParams::default()
        };
        assert_eq!(params.effective_oom_io_mode(), PiPNNOomIoMode::Buffered);
        assert!(!params.strict_oom_io_enabled());

        params.oom_strict_io = true;
        assert_eq!(params.effective_oom_io_mode(), PiPNNOomIoMode::Direct);
        assert!(params.strict_oom_io_enabled());

        params.oom_strict_io = false;
        params.oom_io_mode = PiPNNOomIoMode::Direct;
        assert_eq!(params.effective_oom_io_mode(), PiPNNOomIoMode::Direct);
        assert!(params.strict_oom_io_enabled());

        params.oom_io_mode = PiPNNOomIoMode::Buffered;
        assert_eq!(params.effective_oom_io_mode(), PiPNNOomIoMode::Buffered);
        assert!(!params.strict_oom_io_enabled());
    }

    #[test]
    fn pipnn_params_default_gpu_subtree_persistent_leaf_tile_rows_is_128() {
        assert_eq!(
            PiPNNParams::default().gpu_subtree_persistent_leaf_tile_rows,
            128
        );
    }

    #[test]
    fn gpu_subtree_low_dim_sq8_modes_parse_and_use_validated_auto_defaults() {
        assert_eq!(
            PiPNNGpuSubtreePersistentLargeTicketBatch::from_str("auto").unwrap(),
            PiPNNGpuSubtreePersistentLargeTicketBatch::Auto
        );
        assert_eq!(
            PiPNNGpuSubtreePersistentLargeTicketBatch::from_str("8").unwrap(),
            PiPNNGpuSubtreePersistentLargeTicketBatch::Eight
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8AssignmentKernel::from_str("warp-dp4a").unwrap(),
            PiPNNGpuSubtreeSq8AssignmentKernel::WarpDp4a
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8AssignmentKernel::from_str("wmma").unwrap(),
            PiPNNGpuSubtreeSq8AssignmentKernel::Wmma
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8D0AssignmentKernel::from_str("auto").unwrap(),
            PiPNNGpuSubtreeSq8D0AssignmentKernel::Auto
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8D0AssignmentKernel::from_str("tiled-scalar").unwrap(),
            PiPNNGpuSubtreeSq8D0AssignmentKernel::TiledScalar
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8D0AssignmentKernel::from_str("wmma-batch").unwrap(),
            PiPNNGpuSubtreeSq8D0AssignmentKernel::WmmaBatch
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8LeafExecutor::from_str("auto").unwrap(),
            PiPNNGpuSubtreeSq8LeafExecutor::Auto
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8LeafExecutor::from_str("warp-dp4a").unwrap(),
            PiPNNGpuSubtreeSq8LeafExecutor::WarpDp4a
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8LeafExecutor::from_str("wmma").unwrap(),
            PiPNNGpuSubtreeSq8LeafExecutor::Wmma
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8LeafMode::from_str("row-owned").unwrap(),
            PiPNNGpuSubtreeSq8LeafMode::RowOwned
        );
        assert_eq!(
            PiPNNGpuSubtreeSq8LeafMode::from_str("d96-k2").unwrap(),
            PiPNNGpuSubtreeSq8LeafMode::D96K2
        );

        let params = PiPNNParams::default();
        assert_eq!(
            params.gpu_subtree_persistent_large_ticket_batch,
            PiPNNGpuSubtreePersistentLargeTicketBatch::Auto
        );
        assert_eq!(
            params.gpu_subtree_sq8_assignment_kernel,
            PiPNNGpuSubtreeSq8AssignmentKernel::Auto
        );
        assert_eq!(
            params.gpu_subtree_sq8_d0_assignment_kernel,
            PiPNNGpuSubtreeSq8D0AssignmentKernel::Auto
        );
        assert_eq!(
            params.gpu_subtree_sq8_leaf_executor,
            PiPNNGpuSubtreeSq8LeafExecutor::Auto
        );
        assert_eq!(
            params.gpu_subtree_sq8_leaf_mode,
            PiPNNGpuSubtreeSq8LeafMode::Auto
        );
    }

    #[test]
    fn gpu_subtree_persistent_scheduler_modes_parse_and_default_to_auto() {
        assert_eq!(
            PiPNNGpuSubtreePersistentSchedulerMode::from_str("baseline").unwrap(),
            PiPNNGpuSubtreePersistentSchedulerMode::Baseline
        );
        assert_eq!(
            PiPNNGpuSubtreePersistentSchedulerMode::from_str("local-counters").unwrap(),
            PiPNNGpuSubtreePersistentSchedulerMode::LocalCounters
        );
        assert_eq!(
            PiPNNGpuSubtreePersistentSchedulerMode::from_str("local-counters-barrier").unwrap(),
            PiPNNGpuSubtreePersistentSchedulerMode::LocalCountersBarrier
        );
        assert_eq!(
            PiPNNGpuSubtreePersistentSchedulerMode::from_str("adaptive-idle").unwrap(),
            PiPNNGpuSubtreePersistentSchedulerMode::AdaptiveIdle
        );

        let params = PiPNNParams::default();
        assert_eq!(
            params.gpu_subtree_persistent_scheduler_mode,
            PiPNNGpuSubtreePersistentSchedulerMode::Auto
        );
        assert_eq!(params.gpu_subtree_persistent_idle_backoff_ns, 350);
    }

    #[test]
    fn pipnn_params_default_d0_capacity_is_enabled_with_strict_thresholds() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.gpu_subtree_d0_partition_mode,
            PiPNNGpuSubtreeD0PartitionMode::RbcFanout
        );
        assert_eq!(params.gpu_subtree_d0_tagore_partitions, 32);
        assert_eq!(params.gpu_subtree_d0_tagore_k, 2);
        assert_eq!(params.gpu_subtree_d0_tagore_sample_fraction, 0.01);
        assert_eq!(
            params.gpu_subtree_vector_staging_mode,
            PiPNNGpuSubtreeVectorStagingMode::Off
        );
        assert_eq!(params.gpu_subtree_vector_staging_budget_bytes, 0);
        assert_eq!(params.gpu_subtree_vector_staging_window_children, 8);
        assert_eq!(
            params.gpu_subtree_child_schedule_mode,
            PiPNNGpuSubtreeChildScheduleMode::Natural
        );
        assert_eq!(
            params.effective_gpu_subtree_vector_staging_budget_bytes(),
            1024 * 1024 * 1024
        );
        assert!(params.gpu_subtree_d0_capacity_enable);
        assert_eq!(params.gpu_subtree_d0_capacity_high_ratio, 0.60);
        assert_eq!(params.gpu_subtree_d0_capacity_target_ratio, 0.50);
        assert_eq!(params.gpu_subtree_d0_capacity_low_ratio, 0.15);
        assert_eq!(params.gpu_subtree_d0_capacity_min_efficient_rows, 32 * 1024);
        assert_eq!(params.gpu_subtree_d0_capacity_quality_min_root_leaders, 64);
        assert_eq!(params.gpu_subtree_d0_capacity_root_leader_cap, 10_000);
        assert_eq!(params.gpu_subtree_d0_capacity_split_recursion_limit, 8);
    }

    #[test]
    fn gpu_subtree_d0_throughput_model_mode_parses_aliases() {
        assert_eq!(
            "off"
                .parse::<PiPNNGpuSubtreeD0ThroughputModelMode>()
                .unwrap(),
            PiPNNGpuSubtreeD0ThroughputModelMode::Off
        );
        assert_eq!(
            "report"
                .parse::<PiPNNGpuSubtreeD0ThroughputModelMode>()
                .unwrap(),
            PiPNNGpuSubtreeD0ThroughputModelMode::Report
        );
        assert_eq!(
            "enforce"
                .parse::<PiPNNGpuSubtreeD0ThroughputModelMode>()
                .unwrap(),
            PiPNNGpuSubtreeD0ThroughputModelMode::Enforce
        );
        assert!(
            "maybe"
                .parse::<PiPNNGpuSubtreeD0ThroughputModelMode>()
                .is_err()
        );
    }

    #[test]
    fn default_params_keep_d0_throughput_model_off() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.gpu_subtree_d0_throughput_model,
            PiPNNGpuSubtreeD0ThroughputModelMode::Off
        );
        assert_eq!(
            params.gpu_subtree_d0_throughput_persistent_growth_limit,
            1.10
        );
        assert_eq!(
            params.gpu_subtree_d0_throughput_preload_visible_limit_ms,
            5000
        );
        assert_eq!(params.gpu_subtree_d0_throughput_min_ratio, 0.50);
        assert!(params.gpu_subtree_d0_throughput_calibration_path.is_none());
        assert_eq!(params.gpu_subtree_d0_throughput_child_p95_guardrail_rows, 0);
    }

    #[test]
    fn gpu_subtree_policy_parses_all_modes() {
        assert_eq!("off".parse(), Ok(PiPNNGpuSubtreePolicy::Off));
        assert_eq!("report".parse(), Ok(PiPNNGpuSubtreePolicy::Report));
        assert_eq!("report-only".parse(), Ok(PiPNNGpuSubtreePolicy::Report));
        assert_eq!("report_only".parse(), Ok(PiPNNGpuSubtreePolicy::Report));
        assert_eq!("enforce".parse(), Ok(PiPNNGpuSubtreePolicy::Enforce));
        assert_eq!("enabled".parse(), Ok(PiPNNGpuSubtreePolicy::Enforce));
        assert_eq!("on".parse(), Ok(PiPNNGpuSubtreePolicy::Enforce));
        assert_eq!(
            "auto".parse::<PiPNNGpuSubtreePolicy>(),
            Err(
                "Unsupported GPU-subtree policy 'auto', expected one of: off, report, enforce"
                    .to_string()
            )
        );
    }

    #[test]
    fn gpu_subtree_policy_display_and_serde_names_are_canonical() {
        for (policy, expected) in [
            (PiPNNGpuSubtreePolicy::Off, "off"),
            (PiPNNGpuSubtreePolicy::Report, "report"),
            (PiPNNGpuSubtreePolicy::Enforce, "enforce"),
        ] {
            assert_eq!(policy.to_string(), expected);

            let value = serde_json::to_value(policy).unwrap();
            assert_eq!(value, serde_json::Value::String(expected.to_string()));
            assert_eq!(
                serde_json::from_value::<PiPNNGpuSubtreePolicy>(value).unwrap(),
                policy
            );
        }
    }

    #[test]
    fn gpu_subtree_policy_defaults_report_hardware_without_enforcing_it() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.gpu_subtree_persistent_launch_policy,
            PiPNNGpuSubtreePolicy::Report
        );
        assert_eq!(params.gpu_subtree_hbm_policy, PiPNNGpuSubtreePolicy::Report);
        assert_eq!(params.gpu_subtree_launch_calibration_cache, None);
        assert!(!params.gpu_subtree_launch_calibration_refresh);
    }

    #[test]
    fn gpu_subtree_policy_recommend_reports_hardware_without_enforcing_it() {
        let params = PiPNNParams::recommend(1_000_000, 128, 64, 8);
        assert_eq!(
            params.gpu_subtree_persistent_launch_policy,
            PiPNNGpuSubtreePolicy::Report
        );
        assert_eq!(params.gpu_subtree_hbm_policy, PiPNNGpuSubtreePolicy::Report);
        assert_eq!(params.gpu_subtree_launch_calibration_cache, None);
        assert!(!params.gpu_subtree_launch_calibration_refresh);
    }

    #[test]
    fn gpu_subtree_leaf_topk_backend_parses_measurement_modes() {
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-custom").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentCustom
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent_custom").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentCustom
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("cublas-gemm").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::CublasGemm
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("cublas-gemm-leaf-topk").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::CublasGemm
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-tiled-gemm-fp32").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentTiledGemmFp32
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-tiled-gemm").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentTiledGemmFp32
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-blockselect-fp32").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentBlockselectFp32
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-blockselect").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentBlockselectFp32
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-wmma-tf32").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-wmma").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("cuvs-bruteforce").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::CuvsBruteforce
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::from_str("persistent-cuvs-bruteforce").unwrap(),
            PiPNNGpuSubtreeLeafTopKBackend::CuvsBruteforce
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::PersistentCustom.to_string(),
            "persistent-custom"
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::CublasGemm.to_string(),
            "cublas-gemm"
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::PersistentTiledGemmFp32.to_string(),
            "persistent-tiled-gemm-fp32"
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::PersistentBlockselectFp32.to_string(),
            "persistent-blockselect-fp32"
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32.to_string(),
            "persistent-wmma-tf32"
        );
        assert_eq!(
            PiPNNGpuSubtreeLeafTopKBackend::CuvsBruteforce.to_string(),
            "cuvs-bruteforce"
        );
    }

    #[test]
    fn gpu_subtree_leaf_topk_backend_error_lists_persistent_cuvs_mode() {
        let err = PiPNNGpuSubtreeLeafTopKBackend::from_str("unknown-backend").unwrap_err();

        assert!(err.contains("Unsupported GPU subtree leaf top-k backend"));
        assert!(err.contains("persistent-custom"));
        assert!(err.contains("persistent-tiled-gemm-fp32"));
        assert!(err.contains("persistent-blockselect-fp32"));
        assert!(err.contains("persistent-wmma-tf32"));
        assert!(err.contains("cuvs-bruteforce"));
        assert!(err.contains("cublas-gemm"));
    }

    #[test]
    fn gpu_subtree_assignment_backend_parses_fixed_operator_modes() {
        assert_eq!(
            PiPNNGpuSubtreeAssignmentBackend::from_str("current").unwrap(),
            PiPNNGpuSubtreeAssignmentBackend::Current
        );
        assert_eq!(
            PiPNNGpuSubtreeAssignmentBackend::from_str("persistent-tiled-gemm-fp32").unwrap(),
            PiPNNGpuSubtreeAssignmentBackend::PersistentTiledGemmFp32
        );
        assert_eq!(
            PiPNNGpuSubtreeAssignmentBackend::from_str("persistent-wmma-tf32").unwrap(),
            PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
        );
        assert_eq!(
            PiPNNGpuSubtreeAssignmentBackend::from_str("persistent-cuvs-bruteforce-fp32").unwrap(),
            PiPNNGpuSubtreeAssignmentBackend::PersistentCuvsBruteforceFp32
        );
        assert_eq!(
            PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32.to_string(),
            "persistent-wmma-tf32"
        );
    }

    #[test]
    fn gpu_subtree_assignment_backend_error_lists_fixed_modes() {
        let err = PiPNNGpuSubtreeAssignmentBackend::from_str("auto").unwrap_err();

        assert!(err.contains("Unsupported GPU subtree assignment backend"));
        assert!(err.contains("current"));
        assert!(err.contains("persistent-tiled-gemm-fp32"));
        assert!(err.contains("persistent-wmma-tf32"));
        assert!(err.contains("persistent-cuvs-bruteforce-fp32"));
    }

    #[test]
    fn gpu_subtree_dco_mode_parses_display_names() {
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("none").unwrap(),
            PiPNNGpuSubtreeDcoMode::None
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("prefix-projection").unwrap(),
            PiPNNGpuSubtreeDcoMode::PrefixProjection
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("prefix_projection").unwrap(),
            PiPNNGpuSubtreeDcoMode::PrefixProjection
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("ddc-res").unwrap(),
            PiPNNGpuSubtreeDcoMode::DdcRes
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("ddc_res").unwrap(),
            PiPNNGpuSubtreeDcoMode::DdcRes
        );
        let err = PiPNNGpuSubtreeDcoMode::from_str("projection-ddc").unwrap_err();
        assert!(err.contains("ResInfer DDC_res"));
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("quantized").unwrap(),
            PiPNNGpuSubtreeDcoMode::Quantized
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("wmma-filter").unwrap(),
            PiPNNGpuSubtreeDcoMode::WmmaFilter
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("sq-int8-filter").unwrap(),
            PiPNNGpuSubtreeDcoMode::SqInt8Filter
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("sq-int8").unwrap(),
            PiPNNGpuSubtreeDcoMode::SqInt8Filter
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::SqInt8Filter.to_string(),
            "sq-int8-filter"
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("rabitq-wmma-filter").unwrap(),
            PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("rabitq").unwrap(),
            PiPNNGpuSubtreeDcoMode::RabitqWmmaFilter
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("rabitq-split-filter").unwrap(),
            PiPNNGpuSubtreeDcoMode::RabitqSplitFilter
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::RabitqSplitFilter.to_string(),
            "rabitq-split-filter"
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::from_str("shape-aware").unwrap(),
            PiPNNGpuSubtreeDcoMode::ShapeAware
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoMode::ShapeAware.to_string(),
            "shape-aware"
        );
        assert_eq!(PiPNNGpuSubtreeDcoMode::DdcRes.to_string(), "ddc-res");
        assert!(PiPNNGpuSubtreeDcoMode::PrefixProjection.is_enabled());
        assert!(!PiPNNGpuSubtreeDcoMode::None.is_enabled());
    }

    #[test]
    fn gpu_subtree_dco_rerank_mode_parses_display_names() {
        assert_eq!(
            PiPNNGpuSubtreeDcoRerankMode::from_str("exact").unwrap(),
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoRerankMode::from_str("no-rerank").unwrap(),
            PiPNNGpuSubtreeDcoRerankMode::None
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoRerankMode::from_str("exact-stamp").unwrap(),
            PiPNNGpuSubtreeDcoRerankMode::ExactStamp
        );
        assert_eq!(
            PiPNNGpuSubtreeDcoRerankMode::ExactStamp.to_string(),
            "exact-stamp"
        );
        assert_eq!(PiPNNGpuSubtreeDcoRerankMode::None.to_string(), "none");
    }

    #[test]
    fn pipnn_params_default_gpu_subtree_assignment_backend_is_persistent_wmma_tf32() {
        assert_eq!(
            PiPNNParams::default().gpu_subtree_assignment_backend,
            PiPNNGpuSubtreeAssignmentBackend::PersistentWmmaTf32
        );
    }

    #[test]
    fn pipnn_params_default_gpu_subtree_leaf_topk_backend_is_persistent_wmma_tf32() {
        assert_eq!(
            PiPNNParams::default().gpu_subtree_leaf_topk_backend,
            PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32
        );
    }

    #[test]
    fn pipnn_params_default_gpu_subtree_dco_is_disabled_with_exact_rerank_semantics() {
        let params = PiPNNParams::default();

        assert_eq!(
            params.gpu_subtree_assignment_dco_mode,
            PiPNNGpuSubtreeDcoMode::None
        );
        assert_eq!(
            params.gpu_subtree_assignment_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(params.gpu_subtree_assignment_dco_proj_dim, 0);
        assert_eq!(params.gpu_subtree_assignment_dco_survivors, 0);
        assert!(!params.gpu_subtree_assignment_dco_audit);
        assert_eq!(
            params.gpu_subtree_leaf_dco_mode,
            PiPNNGpuSubtreeDcoMode::None
        );
        assert_eq!(
            params.gpu_subtree_leaf_dco_rerank_mode,
            PiPNNGpuSubtreeDcoRerankMode::Exact
        );
        assert_eq!(params.gpu_subtree_leaf_dco_proj_dim, 0);
        assert_eq!(params.gpu_subtree_leaf_dco_survivors, 0);
        assert!(!params.gpu_subtree_leaf_dco_audit);
    }

    #[test]
    fn pipeline_compute_mode_parses_display_names() {
        assert_eq!(
            PiPNNPipelineComputeMode::from_str("passthrough").unwrap(),
            PiPNNPipelineComputeMode::Passthrough
        );
        assert_eq!(
            PiPNNPipelineComputeMode::from_str("cpu-shadow").unwrap(),
            PiPNNPipelineComputeMode::CpuShadow
        );
        assert_eq!(
            PiPNNPipelineComputeMode::from_str("cpu_shadow").unwrap(),
            PiPNNPipelineComputeMode::CpuShadow
        );
        assert_eq!(
            PiPNNPipelineComputeMode::from_str("MOCK").unwrap(),
            PiPNNPipelineComputeMode::Mock
        );
        assert_eq!(
            PiPNNPipelineComputeMode::from_str("cuda").unwrap(),
            PiPNNPipelineComputeMode::Cuda
        );
        assert_eq!(
            PiPNNPipelineComputeMode::from_str("GPU").unwrap(),
            PiPNNPipelineComputeMode::Cuda
        );
        assert_eq!(
            PiPNNPipelineComputeMode::CpuShadow.to_string(),
            "cpu-shadow"
        );
        assert_eq!(PiPNNPipelineComputeMode::Cuda.to_string(), "cuda");
    }

    #[test]
    fn root_fanout_policy_parses_cell_radius_aliases_and_rejects_witness() {
        assert_eq!(
            RootFanoutPolicy::from_str("fixed").unwrap(),
            RootFanoutPolicy::Fixed
        );
        assert_eq!(
            RootFanoutPolicy::from_str("cell-radius-certified").unwrap(),
            RootFanoutPolicy::CellRadiusCertified
        );
        assert_eq!(
            RootFanoutPolicy::from_str("cell_radius_certified").unwrap(),
            RootFanoutPolicy::CellRadiusCertified
        );
        assert_eq!(
            RootFanoutPolicy::from_str("cell-radius").unwrap(),
            RootFanoutPolicy::CellRadiusCertified
        );
        assert_eq!(
            RootFanoutPolicy::from_str("risk-budgeted").unwrap(),
            RootFanoutPolicy::RiskBudgeted
        );
        assert_eq!(
            RootFanoutPolicy::from_str("risk_budgeted").unwrap(),
            RootFanoutPolicy::RiskBudgeted
        );
        assert_eq!(
            RootFanoutPolicy::from_str("risk-budgeted-core-drop").unwrap(),
            RootFanoutPolicy::RiskBudgeted
        );
        assert_eq!(
            RootFanoutPolicy::from_str("risk-budgeted-root-assign").unwrap(),
            RootFanoutPolicy::RiskBudgeted
        );
        assert!(RootFanoutPolicy::from_str("witness").is_err());
        assert!(RootFanoutPolicy::from_str("witness-certified").is_err());
        assert!(RootFanoutPolicy::from_str("wc-ira").is_err());
        assert!(RootFanoutPolicy::from_str("coverage").is_err());
        assert!(RootFanoutPolicy::from_str("margin").is_err());
    }

    #[test]
    fn default_params_use_cpu_backend_and_device_zero() {
        let params = PiPNNParams::default();
        assert_eq!(params.compute_backend, PiPNNComputeBackendKind::Cpu);
        assert_eq!(params.gpu_device_id, 0);
        assert_eq!(params.gpu_workspace_budget_bytes, 0);
        assert_eq!(params.cuda_compact_dump_dir, None);
        assert_eq!(params.cuda_compact_dump_limit, 0);
    }

    #[test]
    fn recommend_keeps_cpu_backend_defaults() {
        let params = PiPNNParams::recommend(1_000_000, 128, 64, 8);
        assert_eq!(params.compute_backend, PiPNNComputeBackendKind::Cpu);
        assert_eq!(params.gpu_device_id, 0);
        assert_eq!(params.gpu_workspace_budget_bytes, 0);
        assert_eq!(params.cuda_compact_dump_dir, None);
        assert_eq!(params.cuda_compact_dump_limit, 0);
    }

    #[test]
    fn default_params_disable_oom_and_clear_optional_artifacts() {
        let params = PiPNNParams::default();

        assert!(!params.oom_enable);
        assert_eq!(params.oom_memory_budget_bytes, 0);
        assert!(!params.oom_strict_io);
        assert_eq!(params.oom_io_mode, PiPNNOomIoMode::Auto);
        assert_eq!(params.oom_reservoir_budget_share, 1.0);
        assert_eq!(params.oom_sidecar_resident_budget_cap_bytes, None);
        assert_eq!(params.oom_sketch_cache_bytes, 0);
        assert_eq!(params.oom_raw_vector_cache_bytes, 0);
        assert_eq!(params.oom_spill_cache_bytes, 0);
        assert_eq!(params.oom_temp_dir, PathBuf::new());
        assert!(!params.oom_keep_artifacts);
        assert_eq!(params.oom_profile_json, None);
        assert_eq!(params.gpu_workspace_budget_bytes, 0);
        assert_eq!(params.cuda_compact_dump_dir, None);
        assert_eq!(params.cuda_compact_dump_limit, 0);
        assert_eq!(
            params.effective_gpu_workspace_budget_bytes(),
            PiPNNParams::DEFAULT_GPU_WORKSPACE_BUDGET_BYTES
        );
        assert!(!params.pipeline_runtime_enable);
        assert_eq!(
            params.pipeline_compute_mode,
            PiPNNPipelineComputeMode::Passthrough
        );
        assert_eq!(params.pipeline_leaf_descriptor_queue_capacity, None);
        assert_eq!(params.pipeline_staged_queue_capacity, None);
        assert_eq!(params.pipeline_topk_queue_capacity, None);
        assert_eq!(params.pipeline_edge_queue_capacity, None);
        assert_eq!(params.pipeline_max_inflight_compute_tasks_active, None);
        assert_eq!(params.pipeline_max_inflight_compute_tasks_drain, None);
        assert_eq!(params.root_fanout_policy, RootFanoutPolicy::Fixed);
        assert_eq!(params.root_radius_k, 10);
        assert_eq!(params.root_radius_candidate_rs, vec![4, 6, 8]);
        assert!(!params.root_radius_candidate_rs.contains(&1));
        assert!(!params.root_radius_candidate_rs.contains(&2));
        assert_eq!(params.root_radius_warmup_points, 500_000);
        assert!(!params.root_radius_dry_run);
        assert!(!params.root_stop_after_d00);
        assert_eq!(params.root_risk_target_avg_fanout, 8.0);
        assert_eq!(params.root_risk_sigma_lambda, 1.0);
        assert_eq!(params.root_risk_budget_lambda, None);
        assert_eq!(params.root_risk_sigma_mode, RootRiskSigmaMode::CellQ90Q50);
        assert_eq!(params.root_risk_warmup_points, 500_000);
        assert_eq!(params.root_risk_candidate_rs, vec![4, 6, 8, 10]);
        assert!(!params.root_risk_dry_run);
        assert_eq!(params.partitioner, PiPNNPartitionerKind::Rbc);
        assert_eq!(params.kmeans_iterations, 4);
        assert_eq!(params.kmeans_oversplit_factor, 2);
    }

    #[test]
    fn online_suffix_reduce_mode_parses_supported_values() {
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("off").unwrap(),
            PiPNNOnlineSuffixReduceMode::Off
        );
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("direct-first").unwrap(),
            PiPNNOnlineSuffixReduceMode::DirectFirst
        );
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("direct_first").unwrap(),
            PiPNNOnlineSuffixReduceMode::DirectFirst
        );
        assert!(PiPNNOnlineSuffixReduceMode::from_str("lsm").is_err());
    }

    #[test]
    fn online_suffix_reduce_mode_parses_writeback_cache() {
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("writeback-cache").unwrap(),
            PiPNNOnlineSuffixReduceMode::WritebackCache
        );
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("writeback_cache").unwrap(),
            PiPNNOnlineSuffixReduceMode::WritebackCache
        );
    }

    #[test]
    fn online_suffix_reduce_mode_parses_log_assisted() {
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("log-assisted").unwrap(),
            PiPNNOnlineSuffixReduceMode::LogAssisted
        );
        assert_eq!(
            PiPNNOnlineSuffixReduceMode::from_str("log_assisted").unwrap(),
            PiPNNOnlineSuffixReduceMode::LogAssisted
        );
    }

    #[test]
    fn log_assisted_defaults_are_bounded() {
        let mut params = PiPNNParams::default();
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::LogAssisted;

        let budgets = params.effective_online_suffix_budgets();
        assert!(budgets.delta_budget_bytes > budgets.delta_buffer_bytes);
        assert!(budgets.delta_buffer_bytes >= 4096);
        assert_eq!(
            params.effective_online_suffix_range_trigger_bytes(),
            256 * 1024 * 1024
        );
        assert_eq!(params.effective_online_suffix_max_runs_per_range(), 64);
        assert!(params.effective_online_suffix_compaction_workers(8) >= 1);
    }

    #[test]
    fn log_assisted_drain_workers_expand_after_gpu_completion() {
        let mut params = PiPNNParams::default();
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::LogAssisted;

        assert_eq!(params.effective_online_suffix_compaction_workers(8), 2);
        assert_eq!(
            params.effective_online_suffix_drain_compaction_workers(8),
            8
        );
        assert_eq!(params.effective_online_suffix_compaction_workers(1), 1);
        assert_eq!(
            params.effective_online_suffix_drain_compaction_workers(1),
            1
        );

        params.online_suffix_compaction_workers = 16;
        assert_eq!(params.effective_online_suffix_compaction_workers(32), 16);
        assert_eq!(
            params.effective_online_suffix_drain_compaction_workers(32),
            16
        );
    }

    #[test]
    fn log_assisted_validation_rejects_buffer_larger_than_delta_budget() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::LogAssisted;
        params.online_suffix_delta_budget_bytes = 4096;
        params.online_suffix_delta_buffer_bytes = 8192;

        assert!(params.validate_online_suffix_reduce_config().is_err());
    }

    #[test]
    fn writeback_cache_reassigns_b0_state_budget_without_growing_total() {
        let mut params = PiPNNParams::default();
        let b0_total = params.effective_online_suffix_budgets().total_bytes();
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::WritebackCache;

        let budgets = params.effective_online_suffix_budgets();
        assert_eq!(budgets.state_cache_bytes, 512 * 1024 * 1024);
        assert_eq!(budgets.state_page_cache_bytes, 64 * 1024 * 1024);
        assert_eq!(budgets.read_window_bytes, 64 * 1024 * 1024);
        assert!(budgets.read_window_bytes <= budgets.state_page_cache_bytes);
        assert_eq!(budgets.total_bytes(), b0_total);
    }

    #[test]
    fn writeback_cache_honors_state_and_read_window_overrides() {
        let mut params = PiPNNParams::default();
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        params.online_suffix_state_cache_bytes = 256 * 1024 * 1024;
        params.online_suffix_state_page_cache_bytes = 32 * 1024 * 1024;
        params.online_suffix_read_window_bytes = 16 * 1024 * 1024;

        let budgets = params.effective_online_suffix_budgets();
        assert_eq!(budgets.state_cache_bytes, 256 * 1024 * 1024);
        assert_eq!(budgets.state_page_cache_bytes, 32 * 1024 * 1024);
        assert_eq!(budgets.read_window_bytes, 16 * 1024 * 1024);
        assert_eq!(params.online_suffix_pinned_prefix_points, None);
    }

    #[test]
    fn writeback_cache_rejects_read_window_larger_than_page_cache() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
        params.online_suffix_state_page_cache_bytes = 16 * 1024 * 1024;
        params.online_suffix_read_window_bytes = 32 * 1024 * 1024;

        assert!(params.validate_online_suffix_reduce_config().is_err());
    }

    #[test]
    fn online_suffix_defaults_are_off_and_bounded() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.online_suffix_reduce_mode,
            PiPNNOnlineSuffixReduceMode::Off
        );
        let budgets = params.effective_online_suffix_budgets();
        assert!(budgets.direct_queue_bytes > 0);
        assert!(budgets.state_page_cache_bytes > 0);
        assert!(budgets.state_writeback_bytes > 0);
        assert!(budgets.spill_buffer_bytes > 0);
        assert!(budgets.reorder_window_bytes > 0);
        assert!(budgets.decode_bytes > 0);
        assert_eq!(params.online_suffix_state_overprovision_ratio, 2.0);
    }

    #[test]
    fn online_suffix_validation_rejects_invalid_state_space() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::DirectFirst;
        params.online_suffix_state_overprovision_ratio = 1.0;
        assert!(params.validate_online_suffix_reduce_config().is_err());

        params.online_suffix_state_overprovision_ratio = 2.0;
        params.online_suffix_reduce_workers = 0;
        assert!(params.validate_online_suffix_reduce_config().is_ok());
    }

    #[test]
    fn auto_oom_memory_budget_uses_eighty_percent_of_effective_host_limit() {
        const GIB: usize = 1024 * 1024 * 1024;
        let budget = select_auto_oom_memory_budget_bytes(250 * GIB, 241 * GIB, None, None);

        assert_eq!(budget, 200 * GIB);
    }

    #[test]
    fn auto_oom_memory_budget_honors_cgroup_remaining_memory() {
        const GIB: usize = 1024 * 1024 * 1024;
        let budget = select_auto_oom_memory_budget_bytes(
            250 * GIB,
            241 * GIB,
            Some(64 * GIB),
            Some(8 * GIB),
        );

        assert_eq!(budget, 50 * GIB + 2 * GIB / 5);
    }

    #[test]
    fn auto_oom_memory_budget_does_not_rebound_when_cgroup_is_exhausted() {
        const GIB: usize = 1024 * 1024 * 1024;
        let budget = select_auto_oom_memory_budget_bytes(
            250 * GIB,
            241 * GIB,
            Some(64 * GIB),
            Some(64 * GIB),
        );

        assert_eq!(budget, 0);
    }

    #[test]
    fn auto_oom_memory_budget_treats_missing_bounded_cgroup_usage_as_unavailable() {
        const GIB: usize = 1024 * 1024 * 1024;
        let budget =
            select_auto_oom_memory_budget_bytes(250 * GIB, 241 * GIB, Some(64 * GIB), None);

        assert_eq!(budget, 0);
    }

    #[test]
    fn cgroup_memory_file_paths_resolve_nested_v1_memory_controller() {
        let cgroup = "11:cpuset:/batch\n10:memory,cpu:/jobs/shear\n";
        let (limit_path, current_path) = super::cgroup_memory_file_paths(cgroup).unwrap();

        assert_eq!(
            limit_path,
            PathBuf::from("/sys/fs/cgroup/memory/jobs/shear/memory.limit_in_bytes")
        );
        assert_eq!(
            current_path,
            PathBuf::from("/sys/fs/cgroup/memory/jobs/shear/memory.usage_in_bytes")
        );
    }

    #[test]
    fn effective_oom_memory_budget_uses_host_auto_budget_when_unset() {
        let params = PiPNNParams::default();
        assert_eq!(
            params.effective_oom_memory_budget_bytes(),
            super::auto_oom_memory_budget_bytes()
        );
    }

    #[test]
    fn oom_vector_staging_budget_uses_resolved_auto_memory_budget() {
        let mut params = PiPNNParams::default();
        params.oom_enable = true;
        assert_eq!(
            params.effective_gpu_subtree_vector_staging_budget_bytes(),
            (super::auto_oom_memory_budget_bytes() / 16).min(2 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn effective_oom_memory_budget_is_clamped_to_minimum() {
        let mut params = PiPNNParams::default();
        params.oom_memory_budget_bytes = 64 * 1024 * 1024;
        assert_eq!(
            params.effective_oom_memory_budget_bytes(),
            PiPNNParams::MIN_OOM_MEMORY_BUDGET_BYTES
        );
    }
}
