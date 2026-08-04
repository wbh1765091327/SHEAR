use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::common::{AnnError, AnnResult};

pub const TESSA_CAPTURE_VERSION: u32 = 1;
pub const TESSA_CAPTURE_MANIFEST_FILE: &str = "manifest.json";
pub const TESSA_ASSIGNMENT_MEMBERS_FILE: &str = "assignment_members.u32le";
pub const TESSA_ASSIGNMENT_LEADERS_FILE: &str = "assignment_leaders.u32le";
pub const TESSA_LEAF_MEMBERS_FILE: &str = "leaf_members.u32le";
pub const TESSA_ASSIGNMENT_ROW_BINS: usize = 3;
pub const TESSA_ASSIGNMENT_LEADER_BINS: usize = 4;
pub const TESSA_LEAF_SIZE_BINS: usize = 3;
pub const TESSA_ASSIGNMENT_BIN_LIMIT: usize = 16;
pub const TESSA_LEAF_BIN_LIMIT: usize = 64;

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaAssignmentTask {
    pub task_hash: u64,
    pub priority: u64,
    pub depth: u32,
    pub retry: u32,
    pub r_h: u32,
    pub padded_dim: u32,
    pub member_ids: Vec<u32>,
    pub leader_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaLeafTask {
    pub task_hash: u64,
    pub priority: u64,
    pub k: u32,
    pub padded_dim: u32,
    pub member_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureDatasetMetadata {
    pub canonical_base_path: String,
    pub rows: u64,
    pub dimension: u32,
    pub header_sha256: String,
}

impl TessaCaptureDatasetMetadata {
    fn has_same_identity(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.dimension == other.dimension
            && self.header_sha256 == other.header_sha256
    }
}

pub fn tessa_capture_dataset_metadata(path: &Path) -> AnnResult<TessaCaptureDatasetMetadata> {
    let canonical = fs::canonicalize(path).map_err(|err| {
        capture_error(format!(
            "failed to canonicalize capture dataset {}: {err}",
            path.display()
        ))
    })?;
    let (rows, dimension) = crate::utils::load_metadata_from_file(&canonical).map_err(|err| {
        capture_error(format!(
            "failed to read capture dataset header {}: {err}",
            canonical.display()
        ))
    })?;
    let mut header = [0_u8; 8];
    File::open(&canonical)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|err| {
            capture_error(format!(
                "failed to hash capture dataset header {}: {err}",
                canonical.display()
            ))
        })?;
    let metadata = TessaCaptureDatasetMetadata {
        canonical_base_path: canonical.to_string_lossy().into_owned(),
        rows: rows as u64,
        dimension: u32::try_from(dimension)
            .map_err(|_| capture_error(format!("dataset dimension {dimension} exceeds u32")))?,
        header_sha256: hex::encode(Sha256::digest(header)),
    };
    validate_dataset_metadata(&metadata)?;
    Ok(metadata)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureProvenance {
    pub seed: u64,
    pub git_commit: String,
    pub binary_sha256: String,
    pub command_sha256: String,
    pub root_leader_hash: u64,
    pub d0_partition_hash: u64,
    pub d0_task_list_hash: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureBinCounts {
    pub assignment_observed: [[u64; TESSA_ASSIGNMENT_LEADER_BINS]; TESSA_ASSIGNMENT_ROW_BINS],
    pub assignment_selected: [[u64; TESSA_ASSIGNMENT_LEADER_BINS]; TESSA_ASSIGNMENT_ROW_BINS],
    pub leaf_observed: [u64; TESSA_LEAF_SIZE_BINS],
    pub leaf_selected: [u64; TESSA_LEAF_SIZE_BINS],
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureSelection {
    pub assignment_tasks: Vec<TessaAssignmentTask>,
    pub leaf_tasks: Vec<TessaLeafTask>,
    pub bin_counts: TessaCaptureBinCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureFragment {
    pub version: u32,
    pub request_hash: u64,
    pub selection: TessaCaptureSelection,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureFiles {
    pub assignment_members: String,
    pub assignment_leaders: String,
    pub leaf_members: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaAssignmentTaskMetadata {
    pub task_hash: u64,
    pub priority: u64,
    pub depth: u32,
    pub retry: u32,
    pub r_h: u32,
    pub padded_dim: u32,
    pub member_offset: u64,
    pub member_len: u32,
    pub leader_offset: u64,
    pub leader_len: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaLeafTaskMetadata {
    pub task_hash: u64,
    pub priority: u64,
    pub k: u32,
    pub padded_dim: u32,
    pub member_offset: u64,
    pub member_len: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TessaCaptureManifest {
    pub version: u32,
    pub dataset: TessaCaptureDatasetMetadata,
    pub provenance: TessaCaptureProvenance,
    pub bin_counts: TessaCaptureBinCounts,
    pub files: TessaCaptureFiles,
    pub assignment_tasks: Vec<TessaAssignmentTaskMetadata>,
    pub leaf_tasks: Vec<TessaLeafTaskMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessaCapture {
    pub manifest: TessaCaptureManifest,
    pub assignment_tasks: Vec<TessaAssignmentTask>,
    pub leaf_tasks: Vec<TessaLeafTask>,
}

struct StableFnv1a64(u64);

impl StableFnv1a64 {
    fn new(domain: &[u8]) -> Self {
        let mut hash = Self(FNV1A64_OFFSET_BASIS);
        hash.write(domain);
        hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV1A64_PRIME);
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn finish(self) -> u64 {
        if self.0 == 0 { 1 } else { self.0 }
    }
}

pub fn assignment_row_bin(rows: usize) -> Option<usize> {
    match rows {
        256..=1024 => Some(0),
        1025..=8192 => Some(1),
        8193.. => Some(2),
        _ => None,
    }
}

pub fn assignment_leader_bin(leaders: usize) -> Option<usize> {
    match leaders {
        1..=16 => Some(0),
        17..=32 => Some(1),
        33..=64 => Some(2),
        65..=128 => Some(3),
        _ => None,
    }
}

pub fn leaf_size_bin(rows: usize) -> Option<usize> {
    match rows {
        256..=512 => Some(0),
        513..=896 => Some(1),
        897..=1280 => Some(2),
        _ => None,
    }
}

fn priority_for_task(seed: u64, task_hash: u64) -> u64 {
    let mut hash = StableFnv1a64::new(b"SHEAR-TESSA-PRIORITY\0");
    hash.write_u64(seed);
    hash.write_u64(task_hash);
    hash.finish()
}

fn assignment_task_hash(
    seed: u64,
    depth: u32,
    retry: u32,
    r_h: u32,
    padded_dim: u32,
    member_ids: &[u32],
    leader_ids: &[u32],
) -> u64 {
    let mut hash = StableFnv1a64::new(b"SHEAR-TESSA-ASSIGNMENT\0");
    hash.write_u64(seed);
    hash.write_u32(depth);
    hash.write_u32(retry);
    hash.write_u32(r_h);
    hash.write_u32(padded_dim);
    write_membership_digest(&mut hash, member_ids);
    hash.write_u64(leader_ids.len() as u64);
    for &leader_id in leader_ids {
        hash.write_u32(leader_id);
    }
    hash.finish()
}

fn leaf_task_hash(seed: u64, k: u32, padded_dim: u32, member_ids: &[u32]) -> u64 {
    let mut hash = StableFnv1a64::new(b"SHEAR-TESSA-LEAF\0");
    hash.write_u64(seed);
    hash.write_u32(k);
    hash.write_u32(padded_dim);
    write_membership_digest(&mut hash, member_ids);
    hash.finish()
}

#[inline]
fn mix_membership_id(value: u32) -> u64 {
    let mut mixed = u64::from(value).wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn membership_digest(ids: &[u32]) -> (u64, u64) {
    ids.iter().fold((0u64, 0u64), |(sum, xor), &id| {
        let mixed = mix_membership_id(id);
        (sum.wrapping_add(mixed), xor ^ mixed.rotate_left(id & 63))
    })
}

fn write_membership_digest(hash: &mut StableFnv1a64, ids: &[u32]) {
    let (sum, xor) = membership_digest(ids);
    hash.write_u64(ids.len() as u64);
    hash.write_u64(sum);
    hash.write_u64(xor);
}

pub fn tessa_request_hash(seed: u64, member_ids: &[u32]) -> u64 {
    let mut hash = StableFnv1a64::new(b"SHEAR-TESSA-REQUEST\0");
    hash.write_u64(seed);
    write_membership_digest(&mut hash, member_ids);
    hash.finish()
}

pub fn make_tessa_assignment_task(
    seed: u64,
    depth: u32,
    retry: u32,
    r_h: u32,
    padded_dim: u32,
    mut member_ids: Vec<u32>,
    leader_ids: Vec<u32>,
) -> TessaAssignmentTask {
    member_ids.sort_unstable();
    let task_hash = assignment_task_hash(
        seed,
        depth,
        retry,
        r_h,
        padded_dim,
        &member_ids,
        &leader_ids,
    );
    TessaAssignmentTask {
        task_hash,
        priority: priority_for_task(seed, task_hash),
        depth,
        retry,
        r_h,
        padded_dim,
        member_ids,
        leader_ids,
    }
}

pub fn make_tessa_leaf_task(
    seed: u64,
    k: u32,
    padded_dim: u32,
    mut member_ids: Vec<u32>,
) -> TessaLeafTask {
    member_ids.sort_unstable();
    let task_hash = leaf_task_hash(seed, k, padded_dim, &member_ids);
    TessaLeafTask {
        task_hash,
        priority: priority_for_task(seed, task_hash),
        k,
        padded_dim,
        member_ids,
    }
}

pub struct TessaCaptureSampler {
    seed: u64,
    assignment_bins:
        [[Vec<TessaAssignmentTask>; TESSA_ASSIGNMENT_LEADER_BINS]; TESSA_ASSIGNMENT_ROW_BINS],
    leaf_bins: [Vec<TessaLeafTask>; TESSA_LEAF_SIZE_BINS],
    bin_counts: TessaCaptureBinCounts,
}

impl TessaCaptureSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            assignment_bins: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            leaf_bins: std::array::from_fn(|_| Vec::new()),
            bin_counts: TessaCaptureBinCounts::default(),
        }
    }

    pub fn observe_assignment(&mut self, task: TessaAssignmentTask) -> AnnResult<bool> {
        let normalized = make_tessa_assignment_task(
            self.seed,
            task.depth,
            task.retry,
            task.r_h,
            task.padded_dim,
            task.member_ids,
            task.leader_ids,
        );
        let Some(row_bin) = assignment_row_bin(normalized.member_ids.len()) else {
            return Ok(false);
        };
        let Some(leader_bin) = assignment_leader_bin(normalized.leader_ids.len()) else {
            return Ok(false);
        };
        self.bin_counts.assignment_observed[row_bin][leader_bin] =
            self.bin_counts.assignment_observed[row_bin][leader_bin].saturating_add(1);
        let bin = &mut self.assignment_bins[row_bin][leader_bin];
        if let Some(existing) = bin
            .iter()
            .find(|existing| existing.task_hash == normalized.task_hash)
        {
            if existing != &normalized {
                return Err(capture_error(format!(
                    "assignment task hash collision for {}",
                    normalized.task_hash
                )));
            }
            return Ok(false);
        }
        bin.push(normalized);
        bin.sort_by_key(|task| (task.priority, task.task_hash));
        if bin.len() > TESSA_ASSIGNMENT_BIN_LIMIT {
            bin.pop();
        }
        Ok(true)
    }

    pub fn observe_leaf(&mut self, task: TessaLeafTask) -> AnnResult<bool> {
        let normalized = make_tessa_leaf_task(self.seed, task.k, task.padded_dim, task.member_ids);
        let Some(size_bin) = leaf_size_bin(normalized.member_ids.len()) else {
            return Ok(false);
        };
        self.bin_counts.leaf_observed[size_bin] =
            self.bin_counts.leaf_observed[size_bin].saturating_add(1);
        let bin = &mut self.leaf_bins[size_bin];
        if let Some(existing) = bin
            .iter()
            .find(|existing| existing.task_hash == normalized.task_hash)
        {
            if existing != &normalized {
                return Err(capture_error(format!(
                    "leaf task hash collision for {}",
                    normalized.task_hash
                )));
            }
            return Ok(false);
        }
        bin.push(normalized);
        bin.sort_by_key(|task| (task.priority, task.task_hash));
        if bin.len() > TESSA_LEAF_BIN_LIMIT {
            bin.pop();
        }
        Ok(true)
    }

    pub fn finish(mut self) -> TessaCaptureSelection {
        let mut assignment_tasks = Vec::new();
        for row_bin in 0..TESSA_ASSIGNMENT_ROW_BINS {
            for leader_bin in 0..TESSA_ASSIGNMENT_LEADER_BINS {
                let bin = &mut self.assignment_bins[row_bin][leader_bin];
                bin.sort_by_key(|task| (task.priority, task.task_hash));
                self.bin_counts.assignment_selected[row_bin][leader_bin] = bin.len() as u64;
                assignment_tasks.append(bin);
            }
        }
        let mut leaf_tasks = Vec::new();
        for size_bin in 0..TESSA_LEAF_SIZE_BINS {
            let bin = &mut self.leaf_bins[size_bin];
            bin.sort_by_key(|task| (task.priority, task.task_hash));
            self.bin_counts.leaf_selected[size_bin] = bin.len() as u64;
            leaf_tasks.append(bin);
        }
        TessaCaptureSelection {
            assignment_tasks,
            leaf_tasks,
            bin_counts: self.bin_counts,
        }
    }
}

pub fn write_tessa_capture_fragment(
    dir: &Path,
    request_hash: u64,
    selection: &TessaCaptureSelection,
) -> AnnResult<PathBuf> {
    fs::create_dir_all(dir).map_err(|err| {
        capture_error(format!(
            "failed to create TESSA fragment directory {}: {err}",
            dir.display()
        ))
    })?;
    let fragment = TessaCaptureFragment {
        version: TESSA_CAPTURE_VERSION,
        request_hash,
        selection: selection.clone(),
    };
    let mut bytes = serde_json::to_vec_pretty(&fragment)
        .map_err(|err| capture_error(format!("failed to serialize TESSA fragment: {err}")))?;
    bytes.push(b'\n');
    let path = dir.join(format!("fragment-{request_hash:016x}.json"));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    match options.open(&path) {
        Ok(mut file) => {
            file.write_all(&bytes).map_err(|err| {
                capture_error(format!(
                    "failed to write TESSA fragment {}: {err}",
                    path.display()
                ))
            })?;
            file.sync_all().map_err(|err| {
                capture_error(format!(
                    "failed to sync TESSA fragment {}: {err}",
                    path.display()
                ))
            })?;
        }
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(&path).map_err(|read_err| {
                capture_error(format!(
                    "failed to verify existing TESSA fragment {}: {read_err}",
                    path.display()
                ))
            })?;
            if existing != bytes {
                return Err(capture_error(format!(
                    "existing TESSA fragment {} differs from request {request_hash:016x}",
                    path.display()
                )));
            }
        }
        Err(err) => {
            return Err(capture_error(format!(
                "failed to create TESSA fragment {}: {err}",
                path.display()
            )));
        }
    }
    Ok(path)
}

pub fn merge_tessa_capture_fragments(dir: &Path, seed: u64) -> AnnResult<TessaCaptureSelection> {
    let mut paths = fs::read_dir(dir)
        .map_err(|err| {
            capture_error(format!(
                "failed to read TESSA fragment directory {}: {err}",
                dir.display()
            ))
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("fragment-") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(capture_error(format!(
            "no TESSA capture fragments found in {}",
            dir.display()
        )));
    }

    let mut sampler = TessaCaptureSampler::new(seed);
    let mut observed = TessaCaptureBinCounts::default();
    let mut request_hashes = HashSet::new();
    for path in paths {
        let bytes = fs::read(&path).map_err(|err| {
            capture_error(format!(
                "failed to read TESSA fragment {}: {err}",
                path.display()
            ))
        })?;
        let fragment: TessaCaptureFragment = serde_json::from_slice(&bytes).map_err(|err| {
            capture_error(format!(
                "failed to parse TESSA fragment {}: {err}",
                path.display()
            ))
        })?;
        if fragment.version != TESSA_CAPTURE_VERSION {
            return Err(capture_error(format!(
                "unsupported TESSA fragment version {} in {}",
                fragment.version,
                path.display()
            )));
        }
        if !request_hashes.insert(fragment.request_hash) {
            return Err(capture_error(format!(
                "duplicate TESSA request hash {:016x}",
                fragment.request_hash
            )));
        }
        for row_bin in 0..TESSA_ASSIGNMENT_ROW_BINS {
            for leader_bin in 0..TESSA_ASSIGNMENT_LEADER_BINS {
                observed.assignment_observed[row_bin][leader_bin] =
                    observed.assignment_observed[row_bin][leader_bin].saturating_add(
                        fragment.selection.bin_counts.assignment_observed[row_bin][leader_bin],
                    );
            }
        }
        for size_bin in 0..TESSA_LEAF_SIZE_BINS {
            observed.leaf_observed[size_bin] = observed.leaf_observed[size_bin]
                .saturating_add(fragment.selection.bin_counts.leaf_observed[size_bin]);
        }
        for task in fragment.selection.assignment_tasks {
            sampler.observe_assignment(task)?;
        }
        for task in fragment.selection.leaf_tasks {
            sampler.observe_leaf(task)?;
        }
    }
    let mut selection = sampler.finish();
    selection.bin_counts.assignment_observed = observed.assignment_observed;
    selection.bin_counts.leaf_observed = observed.leaf_observed;
    Ok(selection)
}

pub fn write_tessa_capture(
    dir: &Path,
    dataset: &TessaCaptureDatasetMetadata,
    provenance: &TessaCaptureProvenance,
    selection: &TessaCaptureSelection,
) -> AnnResult<PathBuf> {
    validate_dataset_metadata(dataset)?;
    validate_provenance(provenance)?;
    let mut normalized = selection.clone();
    normalized.assignment_tasks.sort_by_key(assignment_sort_key);
    normalized.leaf_tasks.sort_by_key(leaf_sort_key);
    validate_selection(&normalized, provenance.seed, dataset.rows)?;

    fs::create_dir_all(dir).map_err(|err| {
        capture_error(format!(
            "failed to create capture directory {}: {err}",
            dir.display()
        ))
    })?;

    let mut assignment_members = Vec::new();
    let mut assignment_leaders = Vec::new();
    let mut assignment_metadata = Vec::with_capacity(normalized.assignment_tasks.len());
    for task in &normalized.assignment_tasks {
        let member_len = u32::try_from(task.member_ids.len())
            .map_err(|_| capture_error("assignment member count exceeds u32"))?;
        let leader_len = u32::try_from(task.leader_ids.len())
            .map_err(|_| capture_error("assignment leader count exceeds u32"))?;
        assignment_metadata.push(TessaAssignmentTaskMetadata {
            task_hash: task.task_hash,
            priority: task.priority,
            depth: task.depth,
            retry: task.retry,
            r_h: task.r_h,
            padded_dim: task.padded_dim,
            member_offset: assignment_members.len() as u64,
            member_len,
            leader_offset: assignment_leaders.len() as u64,
            leader_len,
        });
        assignment_members.extend_from_slice(&task.member_ids);
        assignment_leaders.extend_from_slice(&task.leader_ids);
    }
    let mut leaf_members = Vec::new();
    let mut leaf_metadata = Vec::with_capacity(normalized.leaf_tasks.len());
    for task in &normalized.leaf_tasks {
        let member_len = u32::try_from(task.member_ids.len())
            .map_err(|_| capture_error("leaf member count exceeds u32"))?;
        leaf_metadata.push(TessaLeafTaskMetadata {
            task_hash: task.task_hash,
            priority: task.priority,
            k: task.k,
            padded_dim: task.padded_dim,
            member_offset: leaf_members.len() as u64,
            member_len,
        });
        leaf_members.extend_from_slice(&task.member_ids);
    }

    let files = TessaCaptureFiles {
        assignment_members: TESSA_ASSIGNMENT_MEMBERS_FILE.to_string(),
        assignment_leaders: TESSA_ASSIGNMENT_LEADERS_FILE.to_string(),
        leaf_members: TESSA_LEAF_MEMBERS_FILE.to_string(),
    };
    write_u32_le_file(&dir.join(&files.assignment_members), &assignment_members)?;
    write_u32_le_file(&dir.join(&files.assignment_leaders), &assignment_leaders)?;
    write_u32_le_file(&dir.join(&files.leaf_members), &leaf_members)?;

    let manifest = TessaCaptureManifest {
        version: TESSA_CAPTURE_VERSION,
        dataset: dataset.clone(),
        provenance: provenance.clone(),
        bin_counts: normalized.bin_counts,
        files,
        assignment_tasks: assignment_metadata,
        leaf_tasks: leaf_metadata,
    };
    let mut json = serde_json::to_vec_pretty(&manifest)
        .map_err(|err| capture_error(format!("failed to serialize capture manifest: {err}")))?;
    json.push(b'\n');
    let manifest_path = dir.join(TESSA_CAPTURE_MANIFEST_FILE);
    fs::write(&manifest_path, json).map_err(|err| {
        capture_error(format!(
            "failed to write capture manifest {}: {err}",
            manifest_path.display()
        ))
    })?;
    Ok(manifest_path)
}

pub fn load_tessa_capture(
    dir: &Path,
    expected_dataset: Option<&TessaCaptureDatasetMetadata>,
) -> AnnResult<TessaCapture> {
    let manifest_path = dir.join(TESSA_CAPTURE_MANIFEST_FILE);
    let bytes = fs::read(&manifest_path).map_err(|err| {
        capture_error(format!(
            "failed to read capture manifest {}: {err}",
            manifest_path.display()
        ))
    })?;
    let manifest: TessaCaptureManifest = serde_json::from_slice(&bytes).map_err(|err| {
        capture_error(format!(
            "failed to parse capture manifest {}: {err}",
            manifest_path.display()
        ))
    })?;
    if manifest.version != TESSA_CAPTURE_VERSION {
        return Err(capture_error(format!(
            "unsupported TESSA capture version {}",
            manifest.version
        )));
    }
    validate_dataset_metadata(&manifest.dataset)?;
    validate_provenance(&manifest.provenance)?;
    if let Some(expected) = expected_dataset {
        validate_dataset_metadata(expected)?;
        if !manifest.dataset.has_same_identity(expected) {
            return Err(capture_error(format!(
                "capture dataset identity mismatch: captured rows={} dim={} header_sha256={}, expected rows={} dim={} header_sha256={}",
                manifest.dataset.rows,
                manifest.dataset.dimension,
                manifest.dataset.header_sha256,
                expected.rows,
                expected.dimension,
                expected.header_sha256
            )));
        }
    }
    validate_relative_file_name(&manifest.files.assignment_members)?;
    validate_relative_file_name(&manifest.files.assignment_leaders)?;
    validate_relative_file_name(&manifest.files.leaf_members)?;
    let assignment_members = read_u32_le_file(&dir.join(&manifest.files.assignment_members))?;
    let assignment_leaders = read_u32_le_file(&dir.join(&manifest.files.assignment_leaders))?;
    let leaf_members = read_u32_le_file(&dir.join(&manifest.files.leaf_members))?;

    let mut next_member_offset = 0usize;
    let mut next_leader_offset = 0usize;
    let mut assignment_tasks = Vec::with_capacity(manifest.assignment_tasks.len());
    for metadata in &manifest.assignment_tasks {
        let member_ids = exact_u32_slice(
            &assignment_members,
            metadata.member_offset,
            metadata.member_len,
            &mut next_member_offset,
            "assignment members",
        )?
        .to_vec();
        let leader_ids = exact_u32_slice(
            &assignment_leaders,
            metadata.leader_offset,
            metadata.leader_len,
            &mut next_leader_offset,
            "assignment leaders",
        )?
        .to_vec();
        assignment_tasks.push(TessaAssignmentTask {
            task_hash: metadata.task_hash,
            priority: metadata.priority,
            depth: metadata.depth,
            retry: metadata.retry,
            r_h: metadata.r_h,
            padded_dim: metadata.padded_dim,
            member_ids,
            leader_ids,
        });
    }
    if next_member_offset != assignment_members.len()
        || next_leader_offset != assignment_leaders.len()
    {
        return Err(capture_error(format!(
            "capture assignment binary lengths do not match metadata: members {} of {}, leaders {} of {}",
            next_member_offset,
            assignment_members.len(),
            next_leader_offset,
            assignment_leaders.len()
        )));
    }

    let mut next_leaf_offset = 0usize;
    let mut leaf_tasks = Vec::with_capacity(manifest.leaf_tasks.len());
    for metadata in &manifest.leaf_tasks {
        let member_ids = exact_u32_slice(
            &leaf_members,
            metadata.member_offset,
            metadata.member_len,
            &mut next_leaf_offset,
            "leaf members",
        )?
        .to_vec();
        leaf_tasks.push(TessaLeafTask {
            task_hash: metadata.task_hash,
            priority: metadata.priority,
            k: metadata.k,
            padded_dim: metadata.padded_dim,
            member_ids,
        });
    }
    if next_leaf_offset != leaf_members.len() {
        return Err(capture_error(format!(
            "capture leaf binary length does not match metadata: {} of {}",
            next_leaf_offset,
            leaf_members.len()
        )));
    }

    let selection = TessaCaptureSelection {
        assignment_tasks,
        leaf_tasks,
        bin_counts: manifest.bin_counts.clone(),
    };
    validate_selection(&selection, manifest.provenance.seed, manifest.dataset.rows)?;
    Ok(TessaCapture {
        manifest,
        assignment_tasks: selection.assignment_tasks,
        leaf_tasks: selection.leaf_tasks,
    })
}

fn assignment_sort_key(task: &TessaAssignmentTask) -> (usize, usize, u64, u64) {
    (
        assignment_row_bin(task.member_ids.len()).unwrap_or(usize::MAX),
        assignment_leader_bin(task.leader_ids.len()).unwrap_or(usize::MAX),
        task.priority,
        task.task_hash,
    )
}

fn leaf_sort_key(task: &TessaLeafTask) -> (usize, u64, u64) {
    (
        leaf_size_bin(task.member_ids.len()).unwrap_or(usize::MAX),
        task.priority,
        task.task_hash,
    )
}

fn validate_selection(
    selection: &TessaCaptureSelection,
    seed: u64,
    dataset_rows: u64,
) -> AnnResult<()> {
    let mut hashes = HashSet::new();
    let mut selected_assignment = [[0u64; TESSA_ASSIGNMENT_LEADER_BINS]; TESSA_ASSIGNMENT_ROW_BINS];
    for task in &selection.assignment_tasks {
        let row_bin = assignment_row_bin(task.member_ids.len())
            .ok_or_else(|| capture_error("assignment task is outside configured row bins"))?;
        let leader_bin = assignment_leader_bin(task.leader_ids.len())
            .ok_or_else(|| capture_error("assignment task is outside configured leader bins"))?;
        selected_assignment[row_bin][leader_bin] += 1;
        if selected_assignment[row_bin][leader_bin] > TESSA_ASSIGNMENT_BIN_LIMIT as u64 {
            return Err(capture_error(format!(
                "assignment capture bin ({row_bin}, {leader_bin}) exceeds limit {}",
                TESSA_ASSIGNMENT_BIN_LIMIT
            )));
        }
        validate_sorted_ids(&task.member_ids, "assignment members")?;
        validate_ids(&task.member_ids, dataset_rows, "assignment member")?;
        validate_ids(&task.leader_ids, dataset_rows, "assignment leader")?;
        if task.r_h == 0 || task.r_h as usize > task.leader_ids.len() {
            return Err(capture_error(format!(
                "assignment task {} has invalid r_h {} for {} leaders",
                task.task_hash,
                task.r_h,
                task.leader_ids.len()
            )));
        }
        if task.padded_dim == 0 {
            return Err(capture_error(format!(
                "assignment task {} has zero padded_dim",
                task.task_hash
            )));
        }
        let expected_hash = assignment_task_hash(
            seed,
            task.depth,
            task.retry,
            task.r_h,
            task.padded_dim,
            &task.member_ids,
            &task.leader_ids,
        );
        validate_hash_and_priority(task.task_hash, task.priority, expected_hash, seed)?;
        if !hashes.insert(task.task_hash) {
            return Err(capture_error(format!(
                "duplicate TESSA task hash {}",
                task.task_hash
            )));
        }
    }

    let mut selected_leaf = [0u64; TESSA_LEAF_SIZE_BINS];
    for task in &selection.leaf_tasks {
        let size_bin = leaf_size_bin(task.member_ids.len())
            .ok_or_else(|| capture_error("leaf task is outside configured size bins"))?;
        selected_leaf[size_bin] += 1;
        if selected_leaf[size_bin] > TESSA_LEAF_BIN_LIMIT as u64 {
            return Err(capture_error(format!(
                "leaf capture bin {size_bin} exceeds limit {}",
                TESSA_LEAF_BIN_LIMIT
            )));
        }
        validate_sorted_ids(&task.member_ids, "leaf members")?;
        validate_ids(&task.member_ids, dataset_rows, "leaf member")?;
        if task.k == 0 || task.k as usize >= task.member_ids.len() || task.padded_dim == 0 {
            return Err(capture_error(format!(
                "leaf task {} has invalid k {} or padded_dim {} for {} members",
                task.task_hash,
                task.k,
                task.padded_dim,
                task.member_ids.len()
            )));
        }
        let expected_hash = leaf_task_hash(seed, task.k, task.padded_dim, &task.member_ids);
        validate_hash_and_priority(task.task_hash, task.priority, expected_hash, seed)?;
        if !hashes.insert(task.task_hash) {
            return Err(capture_error(format!(
                "duplicate TESSA task hash {}",
                task.task_hash
            )));
        }
    }

    if selection.bin_counts.assignment_selected != selected_assignment
        || selection.bin_counts.leaf_selected != selected_leaf
    {
        return Err(capture_error(
            "capture selected bin counts do not match task records",
        ));
    }
    for row_bin in 0..TESSA_ASSIGNMENT_ROW_BINS {
        for leader_bin in 0..TESSA_ASSIGNMENT_LEADER_BINS {
            if selection.bin_counts.assignment_observed[row_bin][leader_bin]
                < selected_assignment[row_bin][leader_bin]
            {
                return Err(capture_error(
                    "capture assignment observed count is smaller than selected count",
                ));
            }
        }
    }
    for (observed, selected) in selection.bin_counts.leaf_observed.iter().zip(selected_leaf) {
        if *observed < selected {
            return Err(capture_error(
                "capture leaf observed count is smaller than selected count",
            ));
        }
    }
    Ok(())
}

fn validate_hash_and_priority(
    task_hash: u64,
    priority: u64,
    expected_hash: u64,
    seed: u64,
) -> AnnResult<()> {
    if task_hash != expected_hash || priority != priority_for_task(seed, expected_hash) {
        return Err(capture_error(format!(
            "TESSA task hash or priority mismatch for task {task_hash}"
        )));
    }
    Ok(())
}

fn validate_dataset_metadata(dataset: &TessaCaptureDatasetMetadata) -> AnnResult<()> {
    if dataset.canonical_base_path.is_empty()
        || !Path::new(&dataset.canonical_base_path).is_absolute()
        || dataset.rows == 0
        || dataset.dimension == 0
        || !is_sha256(&dataset.header_sha256)
    {
        return Err(capture_error(format!(
            "invalid capture dataset metadata: path='{}' rows={} dim={} header_sha256='{}'",
            dataset.canonical_base_path, dataset.rows, dataset.dimension, dataset.header_sha256
        )));
    }
    Ok(())
}

fn validate_provenance(provenance: &TessaCaptureProvenance) -> AnnResult<()> {
    if provenance.git_commit.is_empty()
        || !is_sha256(&provenance.binary_sha256)
        || !is_sha256(&provenance.command_sha256)
        || provenance.root_leader_hash == 0
        || provenance.d0_partition_hash == 0
        || provenance.d0_task_list_hash == 0
    {
        return Err(capture_error("invalid TESSA capture provenance"));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_sorted_ids(ids: &[u32], label: &str) -> AnnResult<()> {
    if ids.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(capture_error(format!(
            "{label} are not in canonical sorted order"
        )));
    }
    Ok(())
}

fn validate_ids(ids: &[u32], dataset_rows: u64, label: &str) -> AnnResult<()> {
    if let Some(&point_id) = ids
        .iter()
        .find(|&&point_id| u64::from(point_id) >= dataset_rows)
    {
        return Err(capture_error(format!(
            "{label} id {point_id} is outside dataset rows {dataset_rows}"
        )));
    }
    Ok(())
}

fn validate_relative_file_name(file_name: &str) -> AnnResult<()> {
    let path = Path::new(file_name);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(capture_error(format!(
            "capture file name must be one relative path component: {file_name}"
        )));
    }
    Ok(())
}

fn exact_u32_slice<'a>(
    values: &'a [u32],
    offset: u64,
    len: u32,
    next_offset: &mut usize,
    label: &str,
) -> AnnResult<&'a [u32]> {
    let start = usize::try_from(offset)
        .map_err(|_| capture_error(format!("{label} offset exceeds usize")))?;
    if start != *next_offset {
        return Err(capture_error(format!(
            "{label} metadata is not contiguous: expected offset {}, got {}",
            *next_offset, start
        )));
    }
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| capture_error(format!("{label} range overflow")))?;
    let slice = values.get(start..end).ok_or_else(|| {
        capture_error(format!(
            "{label} range [{start}, {end}) exceeds {} words",
            values.len()
        ))
    })?;
    *next_offset = end;
    Ok(slice)
}

fn write_u32_le_file(path: &Path, values: &[u32]) -> AnnResult<()> {
    let file = File::create(path).map_err(|err| {
        capture_error(format!(
            "failed to create capture binary {}: {err}",
            path.display()
        ))
    })?;
    let mut writer = BufWriter::new(file);
    for &value in values {
        writer.write_all(&value.to_le_bytes()).map_err(|err| {
            capture_error(format!(
                "failed to write capture binary {}: {err}",
                path.display()
            ))
        })?;
    }
    writer.flush().map_err(|err| {
        capture_error(format!(
            "failed to flush capture binary {}: {err}",
            path.display()
        ))
    })
}

fn read_u32_le_file(path: &Path) -> AnnResult<Vec<u32>> {
    let bytes = fs::read(path).map_err(|err| {
        capture_error(format!(
            "failed to read capture binary {}: {err}",
            path.display()
        ))
    })?;
    if bytes.len() % 4 != 0 {
        return Err(capture_error(format!(
            "capture binary {} has non-u32 byte length {}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn capture_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(format!("TESSA capture: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::Value;
    use tempfile::tempdir;

    use super::*;

    fn hex_digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn dataset() -> TessaCaptureDatasetMetadata {
        TessaCaptureDatasetMetadata {
            canonical_base_path: "/data/base.fbin".to_string(),
            rows: 1_000_000,
            dimension: 1024,
            header_sha256: hex_digest(0x11),
        }
    }

    fn provenance() -> TessaCaptureProvenance {
        TessaCaptureProvenance {
            seed: 42,
            git_commit: "0123456789abcdef".to_string(),
            binary_sha256: hex_digest(0x22),
            command_sha256: hex_digest(0x33),
            root_leader_hash: 11,
            d0_partition_hash: 13,
            d0_task_list_hash: 17,
        }
    }

    fn assignment_task(task_index: usize, rows: usize, leaders: usize) -> TessaAssignmentTask {
        make_tessa_assignment_task(
            42,
            1,
            task_index as u32,
            leaders.min(8) as u32,
            1024,
            (0..rows)
                .map(|row| ((task_index * 10_007 + row) % 1_000_000) as u32)
                .rev()
                .collect(),
            (0..leaders)
                .map(|leader| ((task_index * 257 + leader) % 1_000_000) as u32)
                .collect(),
        )
    }

    fn leaf_task(task_index: usize, rows: usize) -> TessaLeafTask {
        make_tessa_leaf_task(
            42,
            2,
            1024,
            (0..rows)
                .map(|row| ((task_index * 4099 + row) % 1_000_000) as u32)
                .rev()
                .collect(),
        )
    }

    #[test]
    fn tessa_capture_bin_boundaries_are_exact() {
        assert_eq!(assignment_row_bin(255), None);
        assert_eq!(assignment_row_bin(256), Some(0));
        assert_eq!(assignment_row_bin(1024), Some(0));
        assert_eq!(assignment_row_bin(1025), Some(1));
        assert_eq!(assignment_row_bin(8192), Some(1));
        assert_eq!(assignment_row_bin(8193), Some(2));
        assert_eq!(assignment_leader_bin(0), None);
        assert_eq!(assignment_leader_bin(16), Some(0));
        assert_eq!(assignment_leader_bin(17), Some(1));
        assert_eq!(assignment_leader_bin(32), Some(1));
        assert_eq!(assignment_leader_bin(33), Some(2));
        assert_eq!(assignment_leader_bin(64), Some(2));
        assert_eq!(assignment_leader_bin(65), Some(3));
        assert_eq!(assignment_leader_bin(128), Some(3));
        assert_eq!(assignment_leader_bin(129), None);
        assert_eq!(leaf_size_bin(255), None);
        assert_eq!(leaf_size_bin(256), Some(0));
        assert_eq!(leaf_size_bin(512), Some(0));
        assert_eq!(leaf_size_bin(513), Some(1));
        assert_eq!(leaf_size_bin(896), Some(1));
        assert_eq!(leaf_size_bin(897), Some(2));
        assert_eq!(leaf_size_bin(1280), Some(2));
        assert_eq!(leaf_size_bin(1281), None);
    }

    #[test]
    fn tessa_capture_dataset_metadata_hashes_the_canonical_fbin_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("base.fbin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3_i32.to_le_bytes());
        bytes.extend_from_slice(&2_i32.to_le_bytes());
        bytes.extend_from_slice(&[0_u8; 3 * 2 * 4]);
        fs::write(&path, bytes).unwrap();

        let metadata = tessa_capture_dataset_metadata(&path).unwrap();
        assert_eq!(metadata.rows, 3);
        assert_eq!(metadata.dimension, 2);
        assert_eq!(
            metadata.canonical_base_path,
            fs::canonicalize(&path).unwrap().to_string_lossy()
        );
        assert_eq!(
            metadata.header_sha256,
            hex::encode(Sha256::digest(
                [3_i32.to_le_bytes(), 2_i32.to_le_bytes()].concat()
            ))
        );
    }

    #[test]
    fn tessa_capture_bottom_k_is_order_independent_and_bounded() {
        let row_sizes = [256, 1025, 8193];
        let leader_sizes = [16, 17, 33, 65];
        let leaf_sizes = [256, 513, 897];
        let mut assignments = Vec::new();
        let mut leaves = Vec::new();
        let mut task_index = 0usize;
        for rows in row_sizes {
            for leaders in leader_sizes {
                for _ in 0..20 {
                    assignments.push(assignment_task(task_index, rows, leaders));
                    task_index += 1;
                }
            }
        }
        for rows in leaf_sizes {
            for _ in 0..70 {
                leaves.push(leaf_task(task_index, rows));
                task_index += 1;
            }
        }

        let select = |assignment_order: Vec<usize>, leaf_order: Vec<usize>| {
            let mut sampler = TessaCaptureSampler::new(42);
            for index in assignment_order {
                sampler
                    .observe_assignment(assignments[index].clone())
                    .unwrap();
            }
            for index in leaf_order {
                sampler.observe_leaf(leaves[index].clone()).unwrap();
            }
            sampler.finish()
        };

        let forward = select(
            (0..assignments.len()).collect(),
            (0..leaves.len()).collect(),
        );
        let reverse = select(
            (0..assignments.len()).rev().collect(),
            (0..leaves.len()).rev().collect(),
        );
        let mut assignment_shuffled = (0..assignments.len()).collect::<Vec<_>>();
        assignment_shuffled.sort_by_key(|&index| assignments[index].priority ^ 0xa5a5_a5a5);
        let mut leaf_shuffled = (0..leaves.len()).collect::<Vec<_>>();
        leaf_shuffled.sort_by_key(|&index| leaves[index].priority ^ 0x5a5a_5a5a);
        let shuffled = select(assignment_shuffled, leaf_shuffled);

        let assignment_hashes = |selection: &TessaCaptureSelection| {
            selection
                .assignment_tasks
                .iter()
                .map(|task| task.task_hash)
                .collect::<Vec<_>>()
        };
        let leaf_hashes = |selection: &TessaCaptureSelection| {
            selection
                .leaf_tasks
                .iter()
                .map(|task| task.task_hash)
                .collect::<Vec<_>>()
        };
        assert_eq!(assignment_hashes(&forward), assignment_hashes(&reverse));
        assert_eq!(assignment_hashes(&forward), assignment_hashes(&shuffled));
        assert_eq!(leaf_hashes(&forward), leaf_hashes(&reverse));
        assert_eq!(leaf_hashes(&forward), leaf_hashes(&shuffled));
        assert_eq!(forward.assignment_tasks.len(), 12 * 16);
        assert_eq!(forward.leaf_tasks.len(), 3 * 64);
        assert_eq!(forward.bin_counts.assignment_observed, [[20; 4]; 3]);
        assert_eq!(forward.bin_counts.assignment_selected, [[16; 4]; 3]);
        assert_eq!(forward.bin_counts.leaf_observed, [70; 3]);
        assert_eq!(forward.bin_counts.leaf_selected, [64; 3]);
    }

    fn small_selection() -> TessaCaptureSelection {
        let mut sampler = TessaCaptureSampler::new(42);
        sampler
            .observe_assignment(assignment_task(1, 256, 16))
            .unwrap();
        sampler
            .observe_assignment(assignment_task(2, 513, 17))
            .unwrap();
        sampler.observe_leaf(leaf_task(3, 256)).unwrap();
        sampler.finish()
    }

    #[test]
    fn tessa_capture_fragment_round_trip_merge_and_idempotency() {
        let dir = tempdir().unwrap();
        let first = small_selection();
        let first_path = write_tessa_capture_fragment(dir.path(), 0x11, &first).unwrap();
        let first_bytes = fs::read(&first_path).unwrap();

        assert_eq!(
            write_tessa_capture_fragment(dir.path(), 0x11, &first).unwrap(),
            first_path
        );
        assert_eq!(fs::read(&first_path).unwrap(), first_bytes);

        let mut conflicting = first.clone();
        conflicting.bin_counts.leaf_observed[0] += 1;
        assert!(write_tessa_capture_fragment(dir.path(), 0x11, &conflicting).is_err());

        let mut second_sampler = TessaCaptureSampler::new(42);
        second_sampler
            .observe_assignment(assignment_task(4, 1025, 33))
            .unwrap();
        second_sampler.observe_leaf(leaf_task(5, 513)).unwrap();
        let second = second_sampler.finish();
        write_tessa_capture_fragment(dir.path(), 0x22, &second).unwrap();

        let merged = merge_tessa_capture_fragments(dir.path(), 42).unwrap();
        assert_eq!(merged.assignment_tasks.len(), 3);
        assert_eq!(merged.leaf_tasks.len(), 2);
        assert_eq!(merged.bin_counts.assignment_observed[0][0], 1);
        assert_eq!(merged.bin_counts.assignment_observed[0][1], 1);
        assert_eq!(merged.bin_counts.assignment_observed[1][2], 1);
        assert_eq!(merged.bin_counts.leaf_observed, [1, 1, 0]);
        assert_eq!(merged.bin_counts.assignment_selected[0][0], 1);
        assert_eq!(merged.bin_counts.assignment_selected[0][1], 1);
        assert_eq!(merged.bin_counts.assignment_selected[1][2], 1);
        assert_eq!(merged.bin_counts.leaf_selected, [1, 1, 0]);
    }

    fn corrupt_manifest(dir: &std::path::Path, update: impl FnOnce(&mut Value)) {
        let path = dir.join(TESSA_CAPTURE_MANIFEST_FILE);
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        update(&mut value);
        let mut bytes = serde_json::to_vec_pretty(&value).unwrap();
        bytes.push(b'\n');
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn tessa_capture_round_trip_is_little_endian_and_byte_stable() {
        let first = tempdir().unwrap();
        let second = tempdir().unwrap();
        let selection = small_selection();
        write_tessa_capture(first.path(), &dataset(), &provenance(), &selection).unwrap();
        write_tessa_capture(second.path(), &dataset(), &provenance(), &selection).unwrap();

        let loaded = load_tessa_capture(first.path(), Some(&dataset())).unwrap();
        assert_eq!(loaded.assignment_tasks, selection.assignment_tasks);
        assert_eq!(loaded.leaf_tasks, selection.leaf_tasks);
        let first_member = selection.assignment_tasks[0].member_ids[0];
        assert_eq!(
            &fs::read(first.path().join(TESSA_ASSIGNMENT_MEMBERS_FILE)).unwrap()[..4],
            &first_member.to_le_bytes()
        );
        for file in [
            TESSA_CAPTURE_MANIFEST_FILE,
            TESSA_ASSIGNMENT_MEMBERS_FILE,
            TESSA_ASSIGNMENT_LEADERS_FILE,
            TESSA_LEAF_MEMBERS_FILE,
        ] {
            assert_eq!(
                fs::read(first.path().join(file)).unwrap(),
                fs::read(second.path().join(file)).unwrap(),
                "file {file} was not deterministic"
            );
        }

        let mut alias = dataset();
        alias.canonical_base_path = "/same/data/through/another/mount.fbin".to_string();
        load_tessa_capture(first.path(), Some(&alias)).unwrap();
    }

    #[test]
    fn tessa_capture_rejects_corrupt_or_mismatched_inputs() {
        let wrong_version = tempdir().unwrap();
        write_tessa_capture(
            wrong_version.path(),
            &dataset(),
            &provenance(),
            &small_selection(),
        )
        .unwrap();
        corrupt_manifest(wrong_version.path(), |value| value["version"] = 2.into());
        assert!(load_tessa_capture(wrong_version.path(), None).is_err());

        let wrong_length = tempdir().unwrap();
        write_tessa_capture(
            wrong_length.path(),
            &dataset(),
            &provenance(),
            &small_selection(),
        )
        .unwrap();
        let members_path = wrong_length.path().join(TESSA_ASSIGNMENT_MEMBERS_FILE);
        let mut members = fs::read(&members_path).unwrap();
        members.truncate(members.len() - 4);
        fs::write(members_path, members).unwrap();
        assert!(load_tessa_capture(wrong_length.path(), None).is_err());

        let duplicate_hash = tempdir().unwrap();
        write_tessa_capture(
            duplicate_hash.path(),
            &dataset(),
            &provenance(),
            &small_selection(),
        )
        .unwrap();
        corrupt_manifest(duplicate_hash.path(), |value| {
            let hash = value["assignment_tasks"][0]["task_hash"].clone();
            value["assignment_tasks"][1]["task_hash"] = hash;
        });
        assert!(load_tessa_capture(duplicate_hash.path(), None).is_err());

        let out_of_range = tempdir().unwrap();
        write_tessa_capture(
            out_of_range.path(),
            &dataset(),
            &provenance(),
            &small_selection(),
        )
        .unwrap();
        let members_path = out_of_range.path().join(TESSA_ASSIGNMENT_MEMBERS_FILE);
        let mut members = fs::read(&members_path).unwrap();
        members[..4].copy_from_slice(&(dataset().rows as u32).to_le_bytes());
        fs::write(members_path, members).unwrap();
        assert!(load_tessa_capture(out_of_range.path(), None).is_err());

        let mismatch = tempdir().unwrap();
        write_tessa_capture(
            mismatch.path(),
            &dataset(),
            &provenance(),
            &small_selection(),
        )
        .unwrap();
        let mut expected = dataset();
        expected.header_sha256 = hex_digest(0xff);
        assert!(load_tessa_capture(mismatch.path(), Some(&expected)).is_err());
    }
}
