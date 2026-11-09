use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[cfg(feature = "pipnn_cuda")]
use super::compute_backend::TessaCudaReplaySession;
use super::dco_sq_int8::{SQ_INT8_DEFAULT_K_ALIGN, quantize_sq_int8_matrix};
use super::point_store::{DirectPointStore, PointStore};
use super::tessa_capture::{
    TESSA_CAPTURE_MANIFEST_FILE, TessaAssignmentTask, TessaCapture, TessaCaptureDatasetMetadata,
    TessaCaptureProvenance, TessaLeafTask, load_tessa_capture, merge_tessa_capture_fragments,
    tessa_capture_dataset_metadata, write_tessa_capture,
};
use crate::common::{AnnError, AnnResult};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum TessaReplayMode {
    A0Fp32Simt,
    A1Tf32Wmma,
    A2Sq8Dp4aBatch1,
    A3Sq8WmmaBatch8,
    Sq8Dp4aBatch8,
    Sq8WmmaBatch1,
}

impl TessaReplayMode {
    pub const ALL: [Self; 6] = [
        Self::A0Fp32Simt,
        Self::A1Tf32Wmma,
        Self::A2Sq8Dp4aBatch1,
        Self::A3Sq8WmmaBatch8,
        Self::Sq8Dp4aBatch8,
        Self::Sq8WmmaBatch1,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::A0Fp32Simt => "a0-fp32-simt",
            Self::A1Tf32Wmma => "a1-tf32-wmma",
            Self::A2Sq8Dp4aBatch1 => "a2-sq8-dp4a-batch1",
            Self::A3Sq8WmmaBatch8 => "a3-sq8-wmma-batch8",
            Self::Sq8Dp4aBatch8 => "sq8-dp4a-batch8",
            Self::Sq8WmmaBatch1 => "sq8-wmma-batch1",
        }
    }

    pub fn uses_sq8(self) -> bool {
        !matches!(self, Self::A0Fp32Simt | Self::A1Tf32Wmma)
    }

    pub fn replays_leaf_tasks(self) -> bool {
        matches!(
            self,
            Self::A0Fp32Simt | Self::A1Tf32Wmma | Self::A2Sq8Dp4aBatch1 | Self::A3Sq8WmmaBatch8
        )
    }

    pub fn assignment_ticket_batch(self) -> usize {
        match self {
            Self::A3Sq8WmmaBatch8 | Self::Sq8Dp4aBatch8 => 8,
            _ => 1,
        }
    }

    pub(crate) fn assignment_executor_code(self) -> u32 {
        match self {
            Self::A0Fp32Simt => 0,
            Self::A1Tf32Wmma => 1,
            Self::A2Sq8Dp4aBatch1 | Self::Sq8Dp4aBatch8 => 2,
            Self::A3Sq8WmmaBatch8 | Self::Sq8WmmaBatch1 => 3,
        }
    }

    pub(crate) fn leaf_executor_code(self) -> Option<u32> {
        match self {
            Self::A0Fp32Simt => Some(0),
            Self::A1Tf32Wmma => Some(0),
            Self::A2Sq8Dp4aBatch1 => Some(1),
            Self::A3Sq8WmmaBatch8 => Some(2),
            Self::Sq8Dp4aBatch8 | Self::Sq8WmmaBatch1 => None,
        }
    }
}

impl std::fmt::Display for TessaReplayMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TessaReplayPhase {
    Warmup,
    Measured,
}

impl TessaReplayPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Measured => "measured",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TessaReplayTaskKind {
    Assignment,
    Leaf,
    Total,
}

impl TessaReplayTaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Assignment => "assignment",
            Self::Leaf => "leaf",
            Self::Total => "total",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCandidateBlock {
    pub output_offset: usize,
    pub rows: usize,
    pub width: usize,
    pub pair_weight: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TessaOverlapSummary {
    pub task_count: usize,
    pub macro_mean: f64,
    pub pair_weighted_mean: f64,
    pub p5: f64,
    pub p50: f64,
    pub p95: f64,
    pub exact_task_fraction: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TessaDistributionSummary {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TessaCudaReplayCounters {
    pub overflow_flags: u64,
    pub watchdog_trips: u64,
    pub assignment_fallback_nodes: u64,
    pub assignment_overflow_nodes: u64,
    pub leaf_exact_fallback_rows: u64,
    pub leaf_survivor_overflow_rows: u64,
    pub large_split_retries: u64,
    pub large_balanced_fallbacks: u64,
    pub assignment_ticket_claims: u64,
    pub assignment_physical_tiles: u64,
    pub assignment_dp4a_tiles: u64,
    pub assignment_wmma_tiles: u64,
    pub leaf_dp4a_tiles: u64,
    pub leaf_wmma_tiles: u64,
    pub leaf_head: u64,
    pub leaf_pushes: u64,
    pub leaf_topk_pops: u64,
    pub leaf_col_tasks_started: u64,
    pub leaf_col_tasks_done: u64,
    pub leaf_finalize_started: u64,
    pub leaf_finalize_done: u64,
}

impl TessaCudaReplayCounters {
    pub fn fallback_count(self) -> u64 {
        self.assignment_fallback_nodes
            .saturating_add(self.assignment_overflow_nodes)
            .saturating_add(self.leaf_exact_fallback_rows)
            .saturating_add(self.leaf_survivor_overflow_rows)
            .saturating_add(self.large_split_retries)
            .saturating_add(self.large_balanced_fallbacks)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TessaCudaModeResult {
    pub assignment_event_ms: f64,
    pub leaf_event_ms: Option<f64>,
    pub total_event_ms: f64,
    pub assignment_ids: Vec<u32>,
    pub assignment_dists: Vec<f32>,
    pub leaf_ids: Vec<u32>,
    pub leaf_dists: Vec<f32>,
    pub counters: TessaCudaReplayCounters,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureFinalizeMetadata {
    pub dataset: TessaCaptureDatasetMetadata,
    pub provenance: TessaCaptureProvenance,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessaReplayRunConfig {
    pub manifest_path: PathBuf,
    pub base_path: PathBuf,
    pub output_dir: PathBuf,
    pub gpu_device: usize,
    pub seed: u64,
    pub warmup_sweeps: usize,
    pub measured_sweeps: usize,
    pub formal: bool,
}

impl TessaReplayRunConfig {
    pub fn validate(&self) -> AnnResult<()> {
        if self.formal && (self.warmup_sweeps != 5 || self.measured_sweeps != 20) {
            return Err(replay_error(format!(
                "formal replay requires exactly 5 warmup and 20 measured sweeps, got {}/{}",
                self.warmup_sweeps, self.measured_sweeps
            )));
        }
        if self.warmup_sweeps == 0 || self.measured_sweeps == 0 {
            return Err(replay_error(
                "warmup_sweeps and measured_sweeps must both be positive",
            ));
        }
        if self.seed == 0 {
            return Err(replay_error("replay seed must be non-zero"));
        }
        Ok(())
    }
}

pub fn finalize_tessa_capture(
    fragment_dir: &Path,
    capture_profile_path: &Path,
    provenance_path: &Path,
    output_dir: &Path,
    seed: u64,
) -> AnnResult<PathBuf> {
    let metadata: TessaCaptureFinalizeMetadata =
        serde_json::from_slice(&fs::read(provenance_path).map_err(|err| {
            replay_error(format!(
                "failed to read provenance {}: {err}",
                provenance_path.display()
            ))
        })?)
        .map_err(|err| {
            replay_error(format!(
                "failed to parse provenance {}: {err}",
                provenance_path.display()
            ))
        })?;
    if metadata.provenance.seed != seed {
        return Err(replay_error(format!(
            "capture seed mismatch: requested={seed} provenance={}",
            metadata.provenance.seed
        )));
    }
    let profile: serde_json::Value =
        serde_json::from_slice(&fs::read(capture_profile_path).map_err(|err| {
            replay_error(format!(
                "failed to read capture profile {}: {err}",
                capture_profile_path.display()
            ))
        })?)
        .map_err(|err| {
            replay_error(format!(
                "failed to parse capture profile {}: {err}",
                capture_profile_path.display()
            ))
        })?;
    validate_capture_profile_hashes(&profile, &metadata.provenance)?;
    let selection = merge_tessa_capture_fragments(fragment_dir, seed)?;
    write_tessa_capture(
        output_dir,
        &metadata.dataset,
        &metadata.provenance,
        &selection,
    )
}

fn validate_capture_profile_hashes(
    profile: &serde_json::Value,
    provenance: &TessaCaptureProvenance,
) -> AnnResult<()> {
    for (label, path, expected) in [
        (
            "root_leader_hash",
            &["root_fanout", "root_leader_hash"][..],
            provenance.root_leader_hash,
        ),
        (
            "d0_partition_hash",
            &["gpu_subtree", "d0_partition_hash"][..],
            provenance.d0_partition_hash,
        ),
        (
            "d0_task_list_hash",
            &["gpu_subtree", "d0_task_list_hash"][..],
            provenance.d0_task_list_hash,
        ),
    ] {
        let actual = json_u64_at(profile, path).ok_or_else(|| {
            replay_error(format!(
                "capture profile is missing numeric {} at {}",
                label,
                path.join(".")
            ))
        })?;
        if actual != expected {
            return Err(replay_error(format!(
                "capture profile {} mismatch: profile={} provenance={}",
                label, actual, expected
            )));
        }
    }
    Ok(())
}

fn json_u64_at(value: &serde_json::Value, path: &[&str]) -> Option<u64> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(serde_json::Value::as_u64)
}

pub fn run_tessa_operator_replay(config: &TessaReplayRunConfig) -> AnnResult<()> {
    config.validate()?;
    let capture_dir = if config.manifest_path.is_dir() {
        config.manifest_path.as_path()
    } else {
        config.manifest_path.parent().ok_or_else(|| {
            replay_error(format!(
                "manifest path {} has no parent directory",
                config.manifest_path.display()
            ))
        })?
    };
    if !config.manifest_path.is_dir()
        && config
            .manifest_path
            .file_name()
            .and_then(|name| name.to_str())
            != Some(TESSA_CAPTURE_MANIFEST_FILE)
    {
        return Err(replay_error(format!(
            "manifest must name {} or its containing directory, got {}",
            TESSA_CAPTURE_MANIFEST_FILE,
            config.manifest_path.display()
        )));
    }
    let dataset = tessa_capture_dataset_metadata(&config.base_path)?;
    let capture = load_tessa_capture(capture_dir, Some(&dataset))?;
    if capture.manifest.provenance.seed != config.seed {
        return Err(replay_error(format!(
            "replay seed mismatch: requested={} captured={}",
            config.seed, capture.manifest.provenance.seed
        )));
    }
    let prepared = prepare_tessa_replay(&capture, &config.base_path)?;
    run_prepared_tessa_operator_replay(config, &capture, prepared)
}

#[cfg(not(feature = "pipnn_cuda"))]
fn run_prepared_tessa_operator_replay(
    _config: &TessaReplayRunConfig,
    _capture: &TessaCapture,
    _prepared: TessaPreparedReplay,
) -> AnnResult<()> {
    Err(replay_error(
        "CUDA replay requires building shear_core with the pipnn_cuda feature",
    ))
}

#[cfg(feature = "pipnn_cuda")]
fn run_prepared_tessa_operator_replay(
    config: &TessaReplayRunConfig,
    capture: &TessaCapture,
    prepared: TessaPreparedReplay,
) -> AnnResult<()> {
    fs::create_dir_all(&config.output_dir).map_err(|err| {
        replay_error(format!(
            "failed to create replay output directory {}: {err}",
            config.output_dir.display()
        ))
    })?;
    for file in [
        "samples.tsv",
        "task_metrics.tsv",
        "summary.json",
        "summary.md",
        "kernel_resources.json",
        "failure.json",
    ] {
        let path = config.output_dir.join(file);
        if path.exists() {
            return Err(replay_error(format!(
                "refusing to overwrite replay output {}",
                path.display()
            )));
        }
    }

    let mut session = TessaCudaReplaySession::new(config.gpu_device, &prepared)?;
    let mut resource_json = serde_json::to_vec_pretty(&session.kernel_resource_attributes()?)
        .map_err(|err| replay_error(format!("failed to serialize replay kernel resources: {err}")))?;
    resource_json.push(b'\n');
    write_replay_file(&config.output_dir.join("kernel_resources.json"), &resource_json)?;
    let mut samples = Vec::new();
    let mut canonical = HashMap::<TessaReplayMode, TessaCudaModeResult>::new();
    for (phase, sweep_count) in [
        (TessaReplayPhase::Warmup, config.warmup_sweeps),
        (TessaReplayPhase::Measured, config.measured_sweeps),
    ] {
        for sweep in 0..sweep_count {
            let mut sweep_results = HashMap::new();
            for (order, mode) in tessa_rotated_mode_order(sweep).into_iter().enumerate() {
                let result = session.run_mode(mode)?;
                append_replay_samples(&mut samples, phase, sweep, order, mode, &result, &prepared)?;
                if phase == TessaReplayPhase::Measured {
                    if let Some(expected) = canonical.get(&mode) {
                        validate_deterministic_mode_output(mode, expected, &result)?;
                    } else {
                        canonical.insert(mode, result.clone());
                    }
                }
                sweep_results.insert(mode, result);
            }
            let dp4a = sweep_results
                .get(&TessaReplayMode::A2Sq8Dp4aBatch1)
                .expect("A2 is in every replay sweep");
            let wmma = sweep_results
                .get(&TessaReplayMode::A3Sq8WmmaBatch8)
                .expect("A3 is in every replay sweep");
            if let Some(failure) = sq8_exact_mismatch(dp4a, wmma, &prepared, phase, sweep) {
                let mismatched_kinds = [
                    failure.assignment.as_ref().map(|_| "assignment"),
                    failure.leaf.as_ref().map(|_| "leaf"),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" and ");
                let failure_path = config.output_dir.join("failure.json");
                let mut failure_json = serde_json::to_vec_pretty(&failure).map_err(|err| {
                    replay_error(format!("failed to serialize replay failure: {err}"))
                })?;
                failure_json.push(b'\n');
                write_replay_file(&failure_path, &failure_json)?;
                return Err(replay_error(format!(
                    "A2 SQ8-DP4A and A3 SQ8-WMMA {mismatched_kinds} candidate output differs; details: {}",
                    failure_path.display()
                )));
            }
        }
    }

    let samples_tsv = replay_samples_tsv(&samples);
    let task_metrics_tsv = replay_task_metrics_tsv(&canonical, &prepared)?;
    let summary = replay_summary_json(config, capture, &prepared, &samples, &canonical)?;
    let summary_md = replay_summary_markdown(&summary)?;
    write_replay_file(
        &config.output_dir.join("samples.tsv"),
        samples_tsv.as_bytes(),
    )?;
    write_replay_file(
        &config.output_dir.join("task_metrics.tsv"),
        task_metrics_tsv.as_bytes(),
    )?;
    let mut summary_json = serde_json::to_vec_pretty(&summary)
        .map_err(|err| replay_error(format!("failed to serialize replay summary: {err}")))?;
    summary_json.push(b'\n');
    write_replay_file(&config.output_dir.join("summary.json"), &summary_json)?;
    write_replay_file(&config.output_dir.join("summary.md"), summary_md.as_bytes())
}

#[cfg(feature = "pipnn_cuda")]
#[derive(Clone, Debug)]
struct TessaReplaySample {
    phase: TessaReplayPhase,
    sweep: usize,
    order: usize,
    mode: TessaReplayMode,
    kind: TessaReplayTaskKind,
    event_ms: f64,
    pairs: u64,
    logical_payload_bytes: u64,
    counters: TessaCudaReplayCounters,
}

#[cfg(feature = "pipnn_cuda")]
fn append_replay_samples(
    samples: &mut Vec<TessaReplaySample>,
    phase: TessaReplayPhase,
    sweep: usize,
    order: usize,
    mode: TessaReplayMode,
    result: &TessaCudaModeResult,
    prepared: &TessaPreparedReplay,
) -> AnnResult<()> {
    let assignment_payload = replay_payload_bytes(
        mode,
        prepared.assignment_unique_rows,
        prepared.dim,
        prepared.padded_dim,
    )?;
    samples.push(TessaReplaySample {
        phase,
        sweep,
        order,
        mode,
        kind: TessaReplayTaskKind::Assignment,
        event_ms: result.assignment_event_ms,
        pairs: prepared.assignment_pairs,
        logical_payload_bytes: assignment_payload,
        counters: result.counters,
    });
    let mut total_pairs = prepared.assignment_pairs;
    let mut total_payload = assignment_payload;
    if let Some(leaf_event_ms) = result.leaf_event_ms {
        let leaf_payload = replay_payload_bytes(
            mode,
            prepared.leaf_unique_rows,
            prepared.dim,
            prepared.padded_dim,
        )?;
        samples.push(TessaReplaySample {
            phase,
            sweep,
            order,
            mode,
            kind: TessaReplayTaskKind::Leaf,
            event_ms: leaf_event_ms,
            pairs: prepared.leaf_pairs,
            logical_payload_bytes: leaf_payload,
            counters: result.counters,
        });
        total_pairs = total_pairs.saturating_add(prepared.leaf_pairs);
        total_payload = total_payload.saturating_add(leaf_payload);
    }
    samples.push(TessaReplaySample {
        phase,
        sweep,
        order,
        mode,
        kind: TessaReplayTaskKind::Total,
        event_ms: result.total_event_ms,
        pairs: total_pairs,
        logical_payload_bytes: total_payload,
        counters: result.counters,
    });
    Ok(())
}

#[cfg(feature = "pipnn_cuda")]
fn replay_payload_bytes(
    mode: TessaReplayMode,
    rows: usize,
    dim: usize,
    padded_dim: usize,
) -> AnnResult<u64> {
    if mode.uses_sq8() {
        tessa_sq8_logical_payload_bytes(rows, padded_dim)
    } else {
        tessa_fp32_logical_payload_bytes(rows, dim)
    }
}

#[cfg(feature = "pipnn_cuda")]
fn validate_deterministic_mode_output(
    mode: TessaReplayMode,
    expected: &TessaCudaModeResult,
    actual: &TessaCudaModeResult,
) -> AnnResult<()> {
    if expected.assignment_ids != actual.assignment_ids
        || expected.leaf_ids != actual.leaf_ids
        || !same_f32_bits(&expected.assignment_dists, &actual.assignment_dists)
        || !same_f32_bits(&expected.leaf_dists, &actual.leaf_dists)
    {
        return Err(replay_error(format!(
            "mode {mode} produced non-deterministic candidate output across measured sweeps"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct TessaCandidateMismatch {
    index: usize,
    task_index: Option<usize>,
    row: Option<usize>,
    rank: Option<usize>,
    block_rows: Option<usize>,
    block_width: Option<usize>,
    block_pair_weight: Option<u64>,
    dp4a_id: Option<u32>,
    wmma_id: Option<u32>,
    dp4a_distance_bits: Option<u32>,
    wmma_distance_bits: Option<u32>,
}

const TESSA_MISMATCH_SAMPLE_LIMIT: usize = 64;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct TessaCandidateMismatchDetail {
    kind: String,
    dp4a_id_len: usize,
    wmma_id_len: usize,
    dp4a_distance_len: usize,
    wmma_distance_len: usize,
    id_mismatch_count: usize,
    distance_bit_mismatch_count: usize,
    mismatch_index_count: usize,
    mismatch_sample_limit: usize,
    mismatch_sample_truncated: bool,
    mismatches: Vec<TessaCandidateMismatch>,
    first_index: usize,
    task_index: Option<usize>,
    row: Option<usize>,
    rank: Option<usize>,
    block_rows: Option<usize>,
    block_width: Option<usize>,
    block_pair_weight: Option<u64>,
    dp4a_id: Option<u32>,
    wmma_id: Option<u32>,
    dp4a_distance_bits: Option<u32>,
    wmma_distance_bits: Option<u32>,
}

fn candidate_mismatch_detail(
    kind: &str,
    dp4a_ids: &[u32],
    dp4a_dists: &[f32],
    wmma_ids: &[u32],
    wmma_dists: &[f32],
    blocks: &[TessaCandidateBlock],
) -> Option<TessaCandidateMismatchDetail> {
    let id_mismatch_count = dp4a_ids
        .iter()
        .zip(wmma_ids)
        .filter(|(dp4a, wmma)| dp4a != wmma)
        .count()
        + dp4a_ids.len().abs_diff(wmma_ids.len());
    let distance_bit_mismatch_count = dp4a_dists
        .iter()
        .zip(wmma_dists)
        .filter(|(dp4a, wmma)| dp4a.to_bits() != wmma.to_bits())
        .count()
        + dp4a_dists.len().abs_diff(wmma_dists.len());
    if id_mismatch_count == 0 && distance_bit_mismatch_count == 0 {
        return None;
    }

    let output_len = dp4a_ids
        .len()
        .max(wmma_ids.len())
        .max(dp4a_dists.len())
        .max(wmma_dists.len());
    let mut mismatch_index_count = 0usize;
    let mut mismatches = Vec::with_capacity(TESSA_MISMATCH_SAMPLE_LIMIT);
    for index in 0..output_len {
        let differs = dp4a_ids.get(index) != wmma_ids.get(index)
            || dp4a_dists.get(index).map(|value| value.to_bits())
                != wmma_dists.get(index).map(|value| value.to_bits());
        if !differs {
            continue;
        }
        mismatch_index_count += 1;
        if mismatches.len() >= TESSA_MISMATCH_SAMPLE_LIMIT {
            continue;
        }
        let block_location = blocks.iter().enumerate().find_map(|(task_index, block)| {
            let block_len = block.rows.checked_mul(block.width)?;
            let block_end = block.output_offset.checked_add(block_len)?;
            if block.width > 0 && (block.output_offset..block_end).contains(&index) {
                let local_index = index - block.output_offset;
                Some((
                    task_index,
                    local_index / block.width,
                    local_index % block.width,
                    block,
                ))
            } else {
                None
            }
        });
        mismatches.push(TessaCandidateMismatch {
            index,
            task_index: block_location.map(|location| location.0),
            row: block_location.map(|location| location.1),
            rank: block_location.map(|location| location.2),
            block_rows: block_location.map(|location| location.3.rows),
            block_width: block_location.map(|location| location.3.width),
            block_pair_weight: block_location.map(|location| location.3.pair_weight),
            dp4a_id: dp4a_ids.get(index).copied(),
            wmma_id: wmma_ids.get(index).copied(),
            dp4a_distance_bits: dp4a_dists.get(index).map(|value| value.to_bits()),
            wmma_distance_bits: wmma_dists.get(index).map(|value| value.to_bits()),
        });
    }
    let first = mismatches
        .first()
        .expect("a counted mismatch has a sampled index");

    Some(TessaCandidateMismatchDetail {
        kind: kind.to_string(),
        dp4a_id_len: dp4a_ids.len(),
        wmma_id_len: wmma_ids.len(),
        dp4a_distance_len: dp4a_dists.len(),
        wmma_distance_len: wmma_dists.len(),
        id_mismatch_count,
        distance_bit_mismatch_count,
        mismatch_index_count,
        mismatch_sample_limit: TESSA_MISMATCH_SAMPLE_LIMIT,
        mismatch_sample_truncated: mismatch_index_count > mismatches.len(),
        first_index: first.index,
        task_index: first.task_index,
        row: first.row,
        rank: first.rank,
        block_rows: first.block_rows,
        block_width: first.block_width,
        block_pair_weight: first.block_pair_weight,
        dp4a_id: first.dp4a_id,
        wmma_id: first.wmma_id,
        dp4a_distance_bits: first.dp4a_distance_bits,
        wmma_distance_bits: first.wmma_distance_bits,
        mismatches,
    })
}

#[cfg(feature = "pipnn_cuda")]
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct TessaSq8MismatchFailure {
    schema_version: u32,
    message: String,
    phase: TessaReplayPhase,
    sweep: usize,
    dp4a_counters: TessaCudaReplayCounters,
    wmma_counters: TessaCudaReplayCounters,
    assignment: Option<TessaCandidateMismatchDetail>,
    leaf: Option<TessaCandidateMismatchDetail>,
}

#[cfg(feature = "pipnn_cuda")]
fn sq8_exact_mismatch(
    dp4a: &TessaCudaModeResult,
    wmma: &TessaCudaModeResult,
    prepared: &TessaPreparedReplay,
    phase: TessaReplayPhase,
    sweep: usize,
) -> Option<TessaSq8MismatchFailure> {
    let assignment = candidate_mismatch_detail(
        "assignment",
        &dp4a.assignment_ids,
        &dp4a.assignment_dists,
        &wmma.assignment_ids,
        &wmma.assignment_dists,
        &prepared.assignment_blocks,
    );
    let leaf = candidate_mismatch_detail(
        "leaf",
        &dp4a.leaf_ids,
        &dp4a.leaf_dists,
        &wmma.leaf_ids,
        &wmma.leaf_dists,
        &prepared.leaf_blocks,
    );
    if assignment.is_none() && leaf.is_none() {
        return None;
    }
    Some(TessaSq8MismatchFailure {
        schema_version: 2,
        message: "A2 SQ8-DP4A and A3 SQ8-WMMA candidate IDs or distance bits differ".to_string(),
        phase,
        sweep,
        dp4a_counters: dp4a.counters,
        wmma_counters: wmma.counters,
        assignment,
        leaf,
    })
}

#[cfg(feature = "pipnn_cuda")]
fn same_f32_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(feature = "pipnn_cuda")]
fn replay_samples_tsv(samples: &[TessaReplaySample]) -> String {
    let mut output = String::from(
        "phase\tsweep\torder\tmode\ttask_kind\tevent_ms\tpairs\tthroughput_pairs_per_s\tlogical_payload_bytes\toverflow_flags\tfallback_count\twatchdog_trips\tticket_claims\tphysical_tiles\tassignment_dp4a_tiles\tassignment_wmma_tiles\tleaf_dp4a_tiles\tleaf_wmma_tiles\n",
    );
    for sample in samples {
        let throughput = if sample.event_ms > 0.0 {
            sample.pairs as f64 * 1000.0 / sample.event_ms
        } else {
            0.0
        };
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.9}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            sample.phase.as_str(),
            sample.sweep,
            sample.order,
            sample.mode,
            sample.kind.as_str(),
            sample.event_ms,
            sample.pairs,
            throughput,
            sample.logical_payload_bytes,
            sample.counters.overflow_flags,
            sample.counters.fallback_count(),
            sample.counters.watchdog_trips,
            sample.counters.assignment_ticket_claims,
            sample.counters.assignment_physical_tiles,
            sample.counters.assignment_dp4a_tiles,
            sample.counters.assignment_wmma_tiles,
            sample.counters.leaf_dp4a_tiles,
            sample.counters.leaf_wmma_tiles,
        ));
    }
    output
}

#[cfg(feature = "pipnn_cuda")]
fn replay_task_metrics_tsv(
    canonical: &HashMap<TessaReplayMode, TessaCudaModeResult>,
    prepared: &TessaPreparedReplay,
) -> AnnResult<String> {
    let reference = canonical
        .get(&TessaReplayMode::A0Fp32Simt)
        .ok_or_else(|| replay_error("measured replay is missing A0 reference output"))?;
    let mut output = String::from("mode\ttask_kind\ttask_index\tpair_weight\toverlap\texact\n");
    for mode in TessaReplayMode::ALL {
        let actual = canonical
            .get(&mode)
            .ok_or_else(|| replay_error(format!("measured replay is missing mode {mode}")))?;
        append_task_metric_rows(
            &mut output,
            mode,
            TessaReplayTaskKind::Assignment,
            &reference.assignment_ids,
            &actual.assignment_ids,
            &prepared.assignment_blocks,
        )?;
        if mode.replays_leaf_tasks() {
            append_task_metric_rows(
                &mut output,
                mode,
                TessaReplayTaskKind::Leaf,
                &reference.leaf_ids,
                &actual.leaf_ids,
                &prepared.leaf_blocks,
            )?;
        }
    }
    Ok(output)
}

#[cfg(feature = "pipnn_cuda")]
fn append_task_metric_rows(
    output: &mut String,
    mode: TessaReplayMode,
    kind: TessaReplayTaskKind,
    reference: &[u32],
    actual: &[u32],
    blocks: &[TessaCandidateBlock],
) -> AnnResult<()> {
    for (index, block) in blocks.iter().enumerate() {
        let summary = tessa_overlap_summary(reference, actual, std::slice::from_ref(block))?;
        let len = block.rows * block.width;
        let end = block.output_offset + len;
        let exact = reference[block.output_offset..end] == actual[block.output_offset..end];
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.9}\t{}\n",
            mode,
            kind.as_str(),
            index,
            block.pair_weight,
            summary.macro_mean,
            u8::from(exact),
        ));
    }
    Ok(())
}

#[cfg(feature = "pipnn_cuda")]
fn replay_summary_json(
    config: &TessaReplayRunConfig,
    capture: &TessaCapture,
    prepared: &TessaPreparedReplay,
    samples: &[TessaReplaySample],
    canonical: &HashMap<TessaReplayMode, TessaCudaModeResult>,
) -> AnnResult<serde_json::Value> {
    let reference = canonical
        .get(&TessaReplayMode::A0Fp32Simt)
        .ok_or_else(|| replay_error("measured replay is missing A0 reference"))?;
    let mut modes = Vec::new();
    for mode in TessaReplayMode::ALL {
        let actual = canonical
            .get(&mode)
            .ok_or_else(|| replay_error(format!("measured replay is missing {mode}")))?;
        let assignment_ms = measured_event_values(samples, mode, TessaReplayTaskKind::Assignment);
        let total_ms = measured_event_values(samples, mode, TessaReplayTaskKind::Total);
        let leaf_ms = measured_event_values(samples, mode, TessaReplayTaskKind::Leaf);
        let assignment_distribution = tessa_distribution_summary(&assignment_ms)?;
        let total_distribution = tessa_distribution_summary(&total_ms)?;
        let leaf_distribution = if leaf_ms.is_empty() {
            None
        } else {
            Some(tessa_distribution_summary(&leaf_ms)?)
        };
        let assignment_overlap = tessa_overlap_summary(
            &reference.assignment_ids,
            &actual.assignment_ids,
            &prepared.assignment_blocks,
        )?;
        let leaf_overlap = if mode.replays_leaf_tasks() {
            Some(tessa_overlap_summary(
                &reference.leaf_ids,
                &actual.leaf_ids,
                &prepared.leaf_blocks,
            )?)
        } else {
            None
        };
        modes.push(serde_json::json!({
            "mode": mode.as_str(),
            "assignment_event_ms": assignment_distribution,
            "leaf_event_ms": leaf_distribution,
            "total_event_ms": total_distribution,
            "assignment_throughput_pairs_per_s": prepared.assignment_pairs as f64 * 1000.0 / assignment_distribution.median,
            "leaf_throughput_pairs_per_s": leaf_distribution.map(|distribution| prepared.leaf_pairs as f64 * 1000.0 / distribution.median),
            "assignment_logical_payload_bytes": replay_payload_bytes(mode, prepared.assignment_unique_rows, prepared.dim, prepared.padded_dim)?,
            "leaf_logical_payload_bytes": if mode.replays_leaf_tasks() { Some(replay_payload_bytes(mode, prepared.leaf_unique_rows, prepared.dim, prepared.padded_dim)?) } else { None },
            "assignment_overlap_at_rh": assignment_overlap,
            "leaf_overlap_at_k": leaf_overlap,
            "safety_counters": actual.counters,
        }));
    }
    Ok(serde_json::json!({
        "schema_version": 1,
        "formal": config.formal,
        "seed": config.seed,
        "warmup_sweeps": config.warmup_sweeps,
        "measured_sweeps": config.measured_sweeps,
        "gpu_device": config.gpu_device,
        "capture_manifest": capture.manifest,
        "assignment_pairs": prepared.assignment_pairs,
        "leaf_pairs": prepared.leaf_pairs,
        "a2_a3_candidate_ids_and_distance_bits_exact": true,
        "modes": modes,
    }))
}

#[cfg(feature = "pipnn_cuda")]
fn measured_event_values(
    samples: &[TessaReplaySample],
    mode: TessaReplayMode,
    kind: TessaReplayTaskKind,
) -> Vec<f64> {
    samples
        .iter()
        .filter(|sample| {
            sample.phase == TessaReplayPhase::Measured && sample.mode == mode && sample.kind == kind
        })
        .map(|sample| sample.event_ms)
        .collect()
}

#[cfg(feature = "pipnn_cuda")]
fn replay_summary_markdown(summary: &serde_json::Value) -> AnnResult<String> {
    let modes = summary["modes"]
        .as_array()
        .ok_or_else(|| replay_error("summary modes are not an array"))?;
    let mut output = String::from(
        "# TESSA operator replay\n\n| Mode | Assignment ms (median [min, max]) | Leaf ms | Total ms | Assignment overlap | Leaf overlap |\n|---|---:|---:|---:|---:|---:|\n",
    );
    for mode in modes {
        let distribution = |key: &str| -> String {
            let value = &mode[key];
            if value.is_null() {
                return "N/A".to_string();
            }
            format!(
                "{:.3} [{:.3}, {:.3}]",
                value["median"].as_f64().unwrap_or(0.0),
                value["min"].as_f64().unwrap_or(0.0),
                value["max"].as_f64().unwrap_or(0.0),
            )
        };
        output.push_str(&format!(
            "| {} | {} | {} | {} | {:.6} | {} |\n",
            mode["mode"].as_str().unwrap_or("unknown"),
            distribution("assignment_event_ms"),
            distribution("leaf_event_ms"),
            distribution("total_event_ms"),
            mode["assignment_overlap_at_rh"]["macro_mean"]
                .as_f64()
                .unwrap_or(0.0),
            mode["leaf_overlap_at_k"]["macro_mean"]
                .as_f64()
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "N/A".to_string()),
        ));
    }
    output.push_str("\nA2 and A3 candidate IDs and `f32` distance bits are exactly identical.\n");
    Ok(output)
}

#[cfg(feature = "pipnn_cuda")]
fn write_replay_file(path: &Path, bytes: &[u8]) -> AnnResult<()> {
    use std::io::Write;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| replay_error(format!("failed to create {}: {err}", path.display())))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|err| replay_error(format!("failed to write {}: {err}", path.display())))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TessaAssignmentLayout {
    pub member_offset: usize,
    pub member_count: usize,
    pub leader_offset: usize,
    pub leader_count: usize,
    pub r_h: usize,
    pub output_offset: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TessaLeafLayout {
    pub member_offset: usize,
    pub member_count: usize,
    pub k: usize,
    pub output_offset: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TessaPreparedReplay {
    pub dim: usize,
    pub padded_dim: usize,
    pub global_ids: Vec<u32>,
    pub vectors_row_major: Vec<f32>,
    pub raw_norms: Vec<f32>,
    pub sq8_codes: Vec<i8>,
    pub sq8_scales: Vec<f32>,
    pub sq8_norms: Vec<f32>,
    pub assignment_members: Vec<u32>,
    pub assignment_leaders: Vec<u32>,
    pub assignment_layouts: Vec<TessaAssignmentLayout>,
    pub leaf_members: Vec<u32>,
    pub leaf_layouts: Vec<TessaLeafLayout>,
    pub leaf_k: usize,
    pub assignment_blocks: Vec<TessaCandidateBlock>,
    pub leaf_blocks: Vec<TessaCandidateBlock>,
    pub assignment_pairs: u64,
    pub leaf_pairs: u64,
    pub assignment_unique_rows: usize,
    pub leaf_unique_rows: usize,
}

pub fn tessa_assignment_pairs(tasks: &[TessaAssignmentTask]) -> AnnResult<u64> {
    tasks.iter().try_fold(0_u64, |sum, task| {
        let members = u64::try_from(task.member_ids.len())
            .map_err(|_| replay_error("assignment member count exceeds u64"))?;
        let leaders = u64::try_from(task.leader_ids.len())
            .map_err(|_| replay_error("assignment leader count exceeds u64"))?;
        sum.checked_add(
            members
                .checked_mul(leaders)
                .ok_or_else(|| replay_error("assignment pair count overflow"))?,
        )
        .ok_or_else(|| replay_error("assignment pair count overflow"))
    })
}

pub fn tessa_leaf_pairs(tasks: &[TessaLeafTask]) -> AnnResult<u64> {
    tasks.iter().try_fold(0_u64, |sum, task| {
        let rows = u64::try_from(task.member_ids.len())
            .map_err(|_| replay_error("leaf member count exceeds u64"))?;
        sum.checked_add(
            rows.checked_mul(rows.saturating_sub(1))
                .ok_or_else(|| replay_error("leaf pair count overflow"))?,
        )
        .ok_or_else(|| replay_error("leaf pair count overflow"))
    })
}

pub fn tessa_fp32_logical_payload_bytes(unique_rows: usize, dim: usize) -> AnnResult<u64> {
    logical_payload_bytes(unique_rows, dim, 4, 4)
}

pub fn tessa_sq8_logical_payload_bytes(unique_rows: usize, padded_dim: usize) -> AnnResult<u64> {
    logical_payload_bytes(unique_rows, padded_dim, 1, 8)
}

fn logical_payload_bytes(
    unique_rows: usize,
    dimensions: usize,
    bytes_per_dimension: usize,
    metadata_bytes: usize,
) -> AnnResult<u64> {
    let row_bytes = dimensions
        .checked_mul(bytes_per_dimension)
        .and_then(|bytes| bytes.checked_add(metadata_bytes))
        .ok_or_else(|| replay_error("logical payload row size overflow"))?;
    let bytes = unique_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| replay_error("logical payload size overflow"))?;
    u64::try_from(bytes).map_err(|_| replay_error("logical payload size exceeds u64"))
}

pub fn tessa_nearest_rank_percentile(values: &[f64], percentile: f64) -> AnnResult<f64> {
    if values.is_empty() {
        return Err(replay_error("percentile requires at least one value"));
    }
    if !percentile.is_finite() || !(0.0..=1.0).contains(&percentile) {
        return Err(replay_error("percentile must be finite and in [0, 1]"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(replay_error("percentile input contains a non-finite value"));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    Ok(sorted[rank.max(1).min(sorted.len()) - 1])
}

pub fn tessa_distribution_summary(values: &[f64]) -> AnnResult<TessaDistributionSummary> {
    if values.is_empty() {
        return Err(replay_error(
            "distribution summary requires at least one value",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(replay_error(
            "distribution summary input contains a non-finite value",
        ));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    Ok(TessaDistributionSummary {
        median,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
    })
}

pub fn tessa_overlap_summary(
    reference_ids: &[u32],
    actual_ids: &[u32],
    blocks: &[TessaCandidateBlock],
) -> AnnResult<TessaOverlapSummary> {
    if reference_ids.len() != actual_ids.len() {
        return Err(replay_error(format!(
            "candidate output length mismatch: reference={} actual={}",
            reference_ids.len(),
            actual_ids.len()
        )));
    }
    if blocks.is_empty() {
        return Ok(TessaOverlapSummary::default());
    }
    let mut task_overlaps = Vec::with_capacity(blocks.len());
    let mut weighted_sum = 0.0_f64;
    let mut weight_sum = 0_u64;
    let mut exact_tasks = 0usize;
    for block in blocks {
        if block.width == 0 {
            return Err(replay_error("candidate block width must be positive"));
        }
        let len = block
            .rows
            .checked_mul(block.width)
            .ok_or_else(|| replay_error("candidate block length overflow"))?;
        let end = block
            .output_offset
            .checked_add(len)
            .ok_or_else(|| replay_error("candidate block range overflow"))?;
        let reference = reference_ids.get(block.output_offset..end).ok_or_else(|| {
            replay_error(format!(
                "candidate block {}..{} exceeds output length {}",
                block.output_offset,
                end,
                reference_ids.len()
            ))
        })?;
        let actual = &actual_ids[block.output_offset..end];
        let mut matches = 0usize;
        for (reference_row, actual_row) in reference
            .chunks_exact(block.width)
            .zip(actual.chunks_exact(block.width))
        {
            let actual_set = actual_row
                .iter()
                .copied()
                .filter(|&id| id != u32::MAX)
                .collect::<HashSet<_>>();
            matches += reference_row
                .iter()
                .copied()
                .filter(|&id| id != u32::MAX && actual_set.contains(&id))
                .count();
        }
        let overlap = if len == 0 {
            1.0
        } else {
            matches as f64 / len as f64
        };
        task_overlaps.push(overlap);
        weighted_sum += overlap * block.pair_weight as f64;
        weight_sum = weight_sum.saturating_add(block.pair_weight);
        exact_tasks += usize::from(reference == actual);
    }
    Ok(TessaOverlapSummary {
        task_count: blocks.len(),
        macro_mean: task_overlaps.iter().sum::<f64>() / task_overlaps.len() as f64,
        pair_weighted_mean: if weight_sum == 0 {
            0.0
        } else {
            weighted_sum / weight_sum as f64
        },
        p5: tessa_nearest_rank_percentile(&task_overlaps, 0.05)?,
        p50: tessa_nearest_rank_percentile(&task_overlaps, 0.50)?,
        p95: tessa_nearest_rank_percentile(&task_overlaps, 0.95)?,
        exact_task_fraction: exact_tasks as f64 / blocks.len() as f64,
    })
}

pub fn tessa_rotated_mode_order(sweep: usize) -> [TessaReplayMode; 6] {
    let mut modes = TessaReplayMode::ALL;
    let rotation = sweep % modes.len();
    modes.rotate_left(rotation);
    modes
}

pub(crate) fn prepare_tessa_replay(
    capture: &TessaCapture,
    base_path: &Path,
) -> AnnResult<TessaPreparedReplay> {
    let store = DirectPointStore::open(base_path)?;
    if store.len() as u64 != capture.manifest.dataset.rows
        || store.dim() as u32 != capture.manifest.dataset.dimension
    {
        return Err(replay_error(format!(
            "base shape mismatch: base={}x{} capture={}x{}",
            store.len(),
            store.dim(),
            capture.manifest.dataset.rows,
            capture.manifest.dataset.dimension
        )));
    }

    let assignment_ids = capture
        .assignment_tasks
        .iter()
        .flat_map(|task| task.member_ids.iter().chain(&task.leader_ids))
        .copied()
        .collect::<BTreeSet<_>>();
    let leaf_ids = capture
        .leaf_tasks
        .iter()
        .flat_map(|task| task.member_ids.iter())
        .copied()
        .collect::<BTreeSet<_>>();
    let global_ids = assignment_ids.union(&leaf_ids).copied().collect::<Vec<_>>();
    if global_ids.is_empty() {
        return Err(replay_error("capture contains no replay rows"));
    }
    let vectors_row_major = store.read_points(&global_ids)?;
    let dim = store.dim();
    let raw_norms = vectors_row_major
        .chunks_exact(dim)
        .map(|row| row.iter().map(|value| value * value).sum::<f32>())
        .collect::<Vec<_>>();
    let sq8 = quantize_sq_int8_matrix(
        &vectors_row_major,
        global_ids.len(),
        dim,
        SQ_INT8_DEFAULT_K_ALIGN,
    )?;
    let local_by_global = global_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local as u32))
        .collect::<HashMap<_, _>>();

    let mut assignment_members = Vec::new();
    let mut assignment_leaders = Vec::new();
    let mut assignment_layouts = Vec::with_capacity(capture.assignment_tasks.len());
    let mut assignment_blocks = Vec::with_capacity(capture.assignment_tasks.len());
    let mut assignment_output_offset = 0usize;
    for task in &capture.assignment_tasks {
        validate_task_padded_dim(task.padded_dim, sq8.padded_dim, "assignment")?;
        let r_h = task.r_h as usize;
        if r_h == 0 || r_h > 4 || r_h > task.leader_ids.len() {
            return Err(replay_error(format!(
                "assignment task {} has invalid r_h={} for {} leaders",
                task.task_hash,
                task.r_h,
                task.leader_ids.len()
            )));
        }
        let member_offset = assignment_members.len();
        map_global_ids(&task.member_ids, &local_by_global, &mut assignment_members)?;
        let leader_offset = assignment_leaders.len();
        map_global_ids(&task.leader_ids, &local_by_global, &mut assignment_leaders)?;
        let pair_weight = checked_pair_weight(task.member_ids.len(), task.leader_ids.len())?;
        assignment_blocks.push(TessaCandidateBlock {
            output_offset: assignment_output_offset,
            rows: task.member_ids.len(),
            width: r_h,
            pair_weight,
        });
        assignment_layouts.push(TessaAssignmentLayout {
            member_offset,
            member_count: task.member_ids.len(),
            leader_offset,
            leader_count: task.leader_ids.len(),
            r_h,
            output_offset: assignment_output_offset,
        });
        let output_len = task
            .member_ids
            .len()
            .checked_mul(r_h)
            .ok_or_else(|| replay_error("assignment output length overflow"))?;
        assignment_output_offset = assignment_output_offset
            .checked_add(output_len)
            .ok_or_else(|| replay_error("assignment output length overflow"))?;
    }

    let mut leaf_members = Vec::new();
    let mut leaf_layouts = Vec::with_capacity(capture.leaf_tasks.len());
    let mut leaf_blocks = Vec::with_capacity(capture.leaf_tasks.len());
    let mut leaf_output_offset = 0usize;
    let mut leaf_k = None;
    for task in &capture.leaf_tasks {
        validate_task_padded_dim(task.padded_dim, sq8.padded_dim, "leaf")?;
        let k = task.k as usize;
        if k == 0 || k >= task.member_ids.len() {
            return Err(replay_error(format!(
                "leaf task {} has invalid k={} for {} rows",
                task.task_hash,
                task.k,
                task.member_ids.len()
            )));
        }
        if let Some(expected) = leaf_k {
            if expected != k {
                return Err(replay_error(format!(
                    "captured leaf tasks must use one k, found {expected} and {k}"
                )));
            }
        } else {
            leaf_k = Some(k);
        }
        let member_offset = leaf_members.len();
        map_global_ids(&task.member_ids, &local_by_global, &mut leaf_members)?;
        let pair_weight = checked_pair_weight(
            task.member_ids.len(),
            task.member_ids.len().saturating_sub(1),
        )?;
        leaf_blocks.push(TessaCandidateBlock {
            output_offset: leaf_output_offset,
            rows: task.member_ids.len(),
            width: k,
            pair_weight,
        });
        leaf_layouts.push(TessaLeafLayout {
            member_offset,
            member_count: task.member_ids.len(),
            k,
            output_offset: leaf_output_offset,
        });
        let output_len = task
            .member_ids
            .len()
            .checked_mul(k)
            .ok_or_else(|| replay_error("leaf output length overflow"))?;
        leaf_output_offset = leaf_output_offset
            .checked_add(output_len)
            .ok_or_else(|| replay_error("leaf output length overflow"))?;
    }

    let leaf_k = leaf_k.ok_or_else(|| replay_error("capture contains no leaf tasks"))?;
    if assignment_layouts.is_empty() {
        return Err(replay_error("capture contains no assignment tasks"));
    }

    Ok(TessaPreparedReplay {
        dim,
        padded_dim: sq8.padded_dim,
        global_ids,
        vectors_row_major,
        raw_norms,
        sq8_codes: sq8.codes,
        sq8_scales: sq8.scales,
        sq8_norms: sq8.norms,
        assignment_members,
        assignment_leaders,
        assignment_layouts,
        leaf_members,
        leaf_layouts,
        leaf_k,
        assignment_blocks,
        leaf_blocks,
        assignment_pairs: tessa_assignment_pairs(&capture.assignment_tasks)?,
        leaf_pairs: tessa_leaf_pairs(&capture.leaf_tasks)?,
        assignment_unique_rows: assignment_ids.len(),
        leaf_unique_rows: leaf_ids.len(),
    })
}

fn validate_task_padded_dim(actual: u32, expected: usize, kind: &str) -> AnnResult<()> {
    if actual as usize != expected {
        return Err(replay_error(format!(
            "{kind} task padded_dim mismatch: captured={actual} prepared={expected}"
        )));
    }
    Ok(())
}

fn map_global_ids(
    ids: &[u32],
    local_by_global: &HashMap<u32, u32>,
    out: &mut Vec<u32>,
) -> AnnResult<()> {
    for id in ids {
        out.push(*local_by_global.get(id).ok_or_else(|| {
            replay_error(format!(
                "captured global row {id} is missing from replay union"
            ))
        })?);
    }
    Ok(())
}

fn checked_pair_weight(left: usize, right: usize) -> AnnResult<u64> {
    let pairs = left
        .checked_mul(right)
        .ok_or_else(|| replay_error("task pair weight overflow"))?;
    u64::try_from(pairs).map_err(|_| replay_error("task pair weight exceeds u64"))
}

fn replay_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(format!("TESSA replay: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::pipnn::tessa_capture::{
        make_tessa_assignment_task, make_tessa_leaf_task,
    };

    #[test]
    fn tessa_replay_pair_and_payload_formulas_are_exact() {
        let assignments = vec![
            make_tessa_assignment_task(42, 1, 0, 2, 16, vec![0, 1, 2], vec![3, 4]),
            make_tessa_assignment_task(42, 2, 0, 1, 16, vec![5, 6], vec![7, 8, 9]),
        ];
        let leaves = vec![
            make_tessa_leaf_task(42, 2, 16, vec![0, 1, 2, 3]),
            make_tessa_leaf_task(42, 2, 16, vec![4, 5, 6]),
        ];

        assert_eq!(tessa_assignment_pairs(&assignments).unwrap(), 12);
        assert_eq!(tessa_leaf_pairs(&leaves).unwrap(), 18);
        assert_eq!(tessa_fp32_logical_payload_bytes(7, 16).unwrap(), 476);
        assert_eq!(tessa_sq8_logical_payload_bytes(7, 16).unwrap(), 168);
    }

    #[test]
    fn tessa_replay_overlap_percentiles_and_exact_fraction_are_exact() {
        let reference = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let actual = vec![1, 2, 4, 9, 5, 10, 7, 8];
        let blocks = vec![
            TessaCandidateBlock {
                output_offset: 0,
                rows: 1,
                width: 2,
                pair_weight: 1,
            },
            TessaCandidateBlock {
                output_offset: 2,
                rows: 1,
                width: 2,
                pair_weight: 2,
            },
            TessaCandidateBlock {
                output_offset: 4,
                rows: 1,
                width: 2,
                pair_weight: 3,
            },
            TessaCandidateBlock {
                output_offset: 6,
                rows: 1,
                width: 2,
                pair_weight: 4,
            },
        ];

        let summary = tessa_overlap_summary(&reference, &actual, &blocks).unwrap();
        assert_eq!(summary.task_count, 4);
        assert_eq!(summary.macro_mean, 0.75);
        assert_eq!(summary.pair_weighted_mean, 0.75);
        assert_eq!(summary.p5, 0.5);
        assert_eq!(summary.p50, 0.5);
        assert_eq!(summary.p95, 1.0);
        assert_eq!(summary.exact_task_fraction, 0.5);
    }

    #[test]
    fn tessa_replay_nearest_rank_median_range_and_rotation_are_exact() {
        let values = [9.0, 1.0, 5.0, 3.0, 7.0];
        assert_eq!(tessa_nearest_rank_percentile(&values, 0.05).unwrap(), 1.0);
        assert_eq!(tessa_nearest_rank_percentile(&values, 0.50).unwrap(), 5.0);
        assert_eq!(tessa_nearest_rank_percentile(&values, 0.95).unwrap(), 9.0);
        assert_eq!(
            tessa_distribution_summary(&[4.0, 1.0, 3.0, 2.0]).unwrap(),
            TessaDistributionSummary {
                median: 2.5,
                min: 1.0,
                max: 4.0,
            }
        );
        assert_eq!(tessa_rotated_mode_order(0), TessaReplayMode::ALL);
        assert_eq!(
            tessa_rotated_mode_order(1),
            [
                TessaReplayMode::A1Tf32Wmma,
                TessaReplayMode::A2Sq8Dp4aBatch1,
                TessaReplayMode::A3Sq8WmmaBatch8,
                TessaReplayMode::Sq8Dp4aBatch8,
                TessaReplayMode::Sq8WmmaBatch1,
                TessaReplayMode::A0Fp32Simt,
            ]
        );
        assert_eq!(tessa_rotated_mode_order(6), TessaReplayMode::ALL);
    }

    #[test]
    fn tessa_replay_formal_contract_and_capture_hash_validation_are_strict() {
        let config = TessaReplayRunConfig {
            manifest_path: "manifest.json".into(),
            base_path: "base.fbin".into(),
            output_dir: "out".into(),
            gpu_device: 0,
            seed: 42,
            warmup_sweeps: 5,
            measured_sweeps: 20,
            formal: true,
        };
        assert!(config.validate().is_ok());
        assert!(
            TessaReplayRunConfig {
                warmup_sweeps: 4,
                ..config.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            TessaReplayRunConfig {
                measured_sweeps: 19,
                ..config
            }
            .validate()
            .is_err()
        );

        let provenance = TessaCaptureProvenance {
            seed: 42,
            git_commit: "abc".to_string(),
            binary_sha256: "11".repeat(32),
            command_sha256: "22".repeat(32),
            root_leader_hash: 7,
            d0_partition_hash: 11,
            d0_task_list_hash: 13,
        };
        let profile = serde_json::json!({
            "root_fanout": { "root_leader_hash": 7 },
            "gpu_subtree": {
                "d0_partition_hash": 11,
                "d0_task_list_hash": 13
            }
        });
        assert!(validate_capture_profile_hashes(&profile, &provenance).is_ok());
        let mut mismatch = profile;
        mismatch["gpu_subtree"]["d0_partition_hash"] = 17.into();
        assert!(validate_capture_profile_hashes(&mismatch, &provenance).is_err());
    }

    #[test]
    fn tessa_sq8_mismatch_diagnostic_locates_task_row_and_rank() {
        let blocks = vec![
            TessaCandidateBlock {
                output_offset: 0,
                rows: 2,
                width: 2,
                pair_weight: 8,
            },
            TessaCandidateBlock {
                output_offset: 4,
                rows: 1,
                width: 2,
                pair_weight: 4,
            },
        ];
        let dp4a_ids = vec![0, 1, 2, 3, 4, 5];
        let wmma_ids = vec![0, 1, 2, 3, 4, 9];
        let dp4a_dists = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let wmma_dists = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.5];

        let detail = candidate_mismatch_detail(
            "leaf",
            &dp4a_ids,
            &dp4a_dists,
            &wmma_ids,
            &wmma_dists,
            &blocks,
        )
        .unwrap();

        assert_eq!(detail.kind, "leaf");
        assert_eq!(detail.id_mismatch_count, 1);
        assert_eq!(detail.distance_bit_mismatch_count, 1);
        assert_eq!(detail.first_index, 5);
        assert_eq!(detail.task_index, Some(1));
        assert_eq!(detail.row, Some(0));
        assert_eq!(detail.rank, Some(1));
        assert_eq!(detail.dp4a_id, Some(5));
        assert_eq!(detail.wmma_id, Some(9));
        assert_eq!(detail.dp4a_distance_bits, Some(5.0f32.to_bits()));
        assert_eq!(detail.wmma_distance_bits, Some(5.5f32.to_bits()));
    }

    #[test]
    fn tessa_sq8_mismatch_diagnostic_keeps_multiple_locations() {
        let blocks = vec![TessaCandidateBlock {
            output_offset: 0,
            rows: 3,
            width: 2,
            pair_weight: 12,
        }];
        let dp4a_ids = vec![0, 1, 2, 3, 4, 5];
        let wmma_ids = vec![0, 9, 2, 3, 8, 5];
        let dp4a_dists = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let wmma_dists = vec![0.0, 1.5, 2.0, 3.0, 4.5, 5.0];

        let detail = candidate_mismatch_detail(
            "leaf",
            &dp4a_ids,
            &dp4a_dists,
            &wmma_ids,
            &wmma_dists,
            &blocks,
        )
        .unwrap();

        assert_eq!(detail.mismatch_index_count, 2);
        assert!(!detail.mismatch_sample_truncated);
        assert_eq!(detail.mismatches.len(), 2);
        assert_eq!(detail.mismatches[0].index, 1);
        assert_eq!(detail.mismatches[0].row, Some(0));
        assert_eq!(detail.mismatches[0].rank, Some(1));
        assert_eq!(detail.mismatches[1].index, 4);
        assert_eq!(detail.mismatches[1].row, Some(2));
        assert_eq!(detail.mismatches[1].rank, Some(0));
    }

    #[test]
    fn tessa_replay_failure_counters_are_serializable() {
        let counters = TessaCudaReplayCounters {
            leaf_head: 7,
            leaf_pushes: 8,
            leaf_topk_pops: 9,
            leaf_col_tasks_started: 10,
            leaf_col_tasks_done: 11,
            leaf_finalize_started: 12,
            leaf_finalize_done: 13,
            ..TessaCudaReplayCounters::default()
        };
        let json = serde_json::to_value(counters).unwrap();

        assert_eq!(json["leaf_head"], 7);
        assert_eq!(json["leaf_pushes"], 8);
        assert_eq!(json["leaf_topk_pops"], 9);
        assert_eq!(json["leaf_col_tasks_started"], 10);
        assert_eq!(json["leaf_col_tasks_done"], 11);
        assert_eq!(json["leaf_finalize_started"], 12);
        assert_eq!(json["leaf_finalize_done"], 13);
    }

    #[test]
    fn tessa_replay_resets_leaf_counters_before_each_launch() {
        let source = include_str!("compute_backend.rs");
        let run_mode = source
            .split("pub(crate) fn run_mode(&mut self, mode: TessaReplayMode)")
            .nth(1)
            .and_then(|tail| tail.split("fn launch_assignment").next())
            .expect("TESSA replay run_mode source should exist");
        let reset = run_mode
            .find("failed to clear leaf counters")
            .expect("leaf counters must be reset before every replay launch");
        let launch = run_mode
            .find("self.launch_leaf(mode)?")
            .expect("leaf replay launch should exist");

        assert!(reset < launch);
    }

    #[test]
    fn tessa_replay_safety_error_keeps_the_counter_breakdown() {
        let source = include_str!("compute_backend.rs");
        let validator = source
            .split("fn tessa_validate_mode_counters(")
            .nth(1)
            .and_then(|tail| tail.split("fn tessa_u32").next())
            .expect("TESSA replay counter validator source should exist");

        assert!(validator.contains("counters={counters:?}"));
    }

    #[test]
    fn tessa_staged_dp4a_self_only_column_is_not_counted_as_fallback() {
        let source = include_str!("compute_backend.rs");
        let scalar_filter = source
            .split(
                "static __device__ void gpu_persistent_compute_topk_leaf_tile_cta_sq_int8_scalar_filter",
            )
            .nth(1)
            .and_then(|tail| {
                tail.split("static __device__ bool gpu_persistent_sq_int8_should_use_wmma_filter")
                    .next()
            })
            .expect("SQ8 scalar leaf filter source should exist");

        assert!(scalar_filter.contains("if (!row_dco_enabled && candidate_count > 0u)"));
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn tessa_replay_warmups_are_labeled_and_excluded_from_measured_aggregation() {
        let sample = |phase, event_ms| TessaReplaySample {
            phase,
            sweep: 0,
            order: 0,
            mode: TessaReplayMode::A0Fp32Simt,
            kind: TessaReplayTaskKind::Assignment,
            event_ms,
            pairs: 10,
            logical_payload_bytes: 20,
            counters: TessaCudaReplayCounters::default(),
        };
        let samples = vec![
            sample(TessaReplayPhase::Warmup, 100.0),
            sample(TessaReplayPhase::Measured, 2.0),
        ];

        assert_eq!(
            measured_event_values(
                &samples,
                TessaReplayMode::A0Fp32Simt,
                TessaReplayTaskKind::Assignment,
            ),
            vec![2.0]
        );
        let tsv = replay_samples_tsv(&samples);
        assert!(tsv.contains("warmup\t0\t0\ta0-fp32-simt"));
        assert!(tsv.contains("measured\t0\t0\ta0-fp32-simt"));
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn cuda_tessa_replay_sq8_dp4a_and_wmma_are_bit_exact() {
        use cudarc::driver::CudaContext;

        if CudaContext::device_count().unwrap_or(0) == 0 {
            return;
        }
        let rows = 607usize;
        let dim = 1024usize;
        let global_ids = (0..rows as u32).collect::<Vec<_>>();
        let vectors_row_major = (0..rows)
            .flat_map(|row| {
                (0..dim).map(move |axis| {
                    let source_row = if row == 7 { 6 } else { row };
                    ((source_row * 17 + axis * 13) % 97) as f32 * 0.03125 - 1.0
                })
            })
            .collect::<Vec<_>>();
        let raw_norms = vectors_row_major
            .chunks_exact(dim)
            .map(|row| row.iter().map(|value| value * value).sum())
            .collect::<Vec<_>>();
        let sq8 = quantize_sq_int8_matrix(&vectors_row_major, rows, dim, SQ_INT8_DEFAULT_K_ALIGN)
            .unwrap();
        let assignment_members = (0..rows as u32).collect::<Vec<_>>();
        let assignment_leaders = (0..16u32).collect::<Vec<_>>();
        let leaf_members = assignment_members.clone();
        let prepared = TessaPreparedReplay {
            dim,
            padded_dim: sq8.padded_dim,
            global_ids,
            vectors_row_major,
            raw_norms,
            sq8_codes: sq8.codes,
            sq8_scales: sq8.scales,
            sq8_norms: sq8.norms,
            assignment_members,
            assignment_leaders,
            assignment_layouts: vec![TessaAssignmentLayout {
                member_offset: 0,
                member_count: rows,
                leader_offset: 0,
                leader_count: 16,
                r_h: 2,
                output_offset: 0,
            }],
            leaf_members,
            leaf_layouts: vec![TessaLeafLayout {
                member_offset: 0,
                member_count: rows,
                k: 2,
                output_offset: 0,
            }],
            leaf_k: 2,
            assignment_blocks: vec![TessaCandidateBlock {
                output_offset: 0,
                rows,
                width: 2,
                pair_weight: (rows * 16) as u64,
            }],
            leaf_blocks: vec![TessaCandidateBlock {
                output_offset: 0,
                rows,
                width: 2,
                pair_weight: (rows * (rows - 1)) as u64,
            }],
            assignment_pairs: (rows * 16) as u64,
            leaf_pairs: (rows * (rows - 1)) as u64,
            assignment_unique_rows: rows,
            leaf_unique_rows: rows,
        };
        let mut session =
            crate::algorithm::pipnn::compute_backend::TessaCudaReplaySession::new(0, &prepared)
                .unwrap();
        session.run_mode(TessaReplayMode::A0Fp32Simt).unwrap();
        session.run_mode(TessaReplayMode::A1Tf32Wmma).unwrap();
        let dp4a = session.run_mode(TessaReplayMode::A2Sq8Dp4aBatch1).unwrap();
        let wmma = session.run_mode(TessaReplayMode::A3Sq8WmmaBatch8).unwrap();

        assert_eq!(dp4a.assignment_ids, wmma.assignment_ids);
        assert_eq!(dp4a.leaf_ids, wmma.leaf_ids);
        assert_eq!(
            dp4a.assignment_dists
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wmma.assignment_dists
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            dp4a.leaf_dists
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wmma.leaf_dists
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn cuda_tessa_replay_staged_sq8_is_bit_exact_across_multiple_leaf_offsets() {
        use cudarc::driver::CudaContext;

        if CudaContext::device_count().unwrap_or(0) == 0 {
            return;
        }
        let leaf_sizes = [740usize, 728, 641];
        let rows = leaf_sizes.iter().sum::<usize>();
        let dim = 1024usize;
        let global_ids = (0..rows as u32).collect::<Vec<_>>();
        let vectors_row_major = (0..rows)
            .flat_map(|row| {
                (0..dim).map(move |axis| {
                    ((row * 131 + axis * 17 + row * axis * 3) % 251) as f32 * 0.0078125 - 0.75
                })
            })
            .collect::<Vec<_>>();
        let raw_norms = vectors_row_major
            .chunks_exact(dim)
            .map(|row| row.iter().map(|value| value * value).sum())
            .collect::<Vec<_>>();
        let sq8 = quantize_sq_int8_matrix(&vectors_row_major, rows, dim, SQ_INT8_DEFAULT_K_ALIGN)
            .unwrap();
        let assignment_rows = 64usize;
        let assignment_members = (0..assignment_rows as u32).collect::<Vec<_>>();
        let assignment_leaders = (0..16u32).collect::<Vec<_>>();
        let leaf_members = (0..rows as u32).collect::<Vec<_>>();
        let mut leaf_layouts = Vec::new();
        let mut leaf_blocks = Vec::new();
        let mut member_offset = 0usize;
        for member_count in leaf_sizes {
            leaf_layouts.push(TessaLeafLayout {
                member_offset,
                member_count,
                k: 2,
                output_offset: member_offset * 2,
            });
            leaf_blocks.push(TessaCandidateBlock {
                output_offset: member_offset * 2,
                rows: member_count,
                width: 2,
                pair_weight: (member_count * (member_count - 1)) as u64,
            });
            member_offset += member_count;
        }
        let leaf_pairs = leaf_blocks.iter().map(|block| block.pair_weight).sum();
        let prepared = TessaPreparedReplay {
            dim,
            padded_dim: sq8.padded_dim,
            global_ids,
            vectors_row_major,
            raw_norms,
            sq8_codes: sq8.codes,
            sq8_scales: sq8.scales,
            sq8_norms: sq8.norms,
            assignment_members,
            assignment_leaders,
            assignment_layouts: vec![TessaAssignmentLayout {
                member_offset: 0,
                member_count: assignment_rows,
                leader_offset: 0,
                leader_count: 16,
                r_h: 2,
                output_offset: 0,
            }],
            leaf_members,
            leaf_layouts,
            leaf_k: 2,
            assignment_blocks: vec![TessaCandidateBlock {
                output_offset: 0,
                rows: assignment_rows,
                width: 2,
                pair_weight: (assignment_rows * 16) as u64,
            }],
            leaf_blocks,
            assignment_pairs: (assignment_rows * 16) as u64,
            leaf_pairs,
            assignment_unique_rows: assignment_rows,
            leaf_unique_rows: rows,
        };
        let mut session =
            crate::algorithm::pipnn::compute_backend::TessaCudaReplaySession::new(0, &prepared)
                .unwrap();
        let dp4a = session.run_mode(TessaReplayMode::A2Sq8Dp4aBatch1).unwrap();
        let wmma = session.run_mode(TessaReplayMode::A3Sq8WmmaBatch8).unwrap();

        let mismatch = dp4a
            .leaf_ids
            .iter()
            .zip(&dp4a.leaf_dists)
            .zip(wmma.leaf_ids.iter().zip(&wmma.leaf_dists))
            .enumerate()
            .find(|(_, ((dp4a_id, dp4a_dist), (wmma_id, wmma_dist)))| {
                dp4a_id != wmma_id || dp4a_dist.to_bits() != wmma_dist.to_bits()
            });
        assert!(
            mismatch.is_none(),
            "first staged multi-leaf mismatch: {mismatch:?}; dp4a_counters={:?}; wmma_counters={:?}",
            dp4a.counters,
            wmma.counters,
        );
    }
}
