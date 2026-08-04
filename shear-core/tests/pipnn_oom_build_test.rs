use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "pipnn_cuda")]
use serde_json::Value;
#[cfg(feature = "pipnn_cuda")]
use shear_core::algorithm::pipnn::{
    PiPNNComputeBackendKind, PiPNNGpuSubtreeLeafTopKBackend, PiPNNGpuSubtreeSqStagingMode,
    PiPNNGpuSubtreeSqWmmaMode, PiPNNGpuSubtreeSqWmmaSidecarMode, PiPNNOnlineSuffixReduceMode,
    PiPNNOomIoMode, PiPNNPartitionerKind,
};
use shear_core::algorithm::pipnn::{PiPNNHashPruneBackfillMode, PiPNNParams};
use shear_core::common::Metric;
use shear_core::index::{InmemIndex, build_pipnn_graph_oom_to_file};
use shear_core::model::{IndexConfiguration, IndexWriteParametersBuilder};
use shear_core::utils::{load_metadata_from_file, round_up, save_bin_f32};

fn unique_temp_dir(label: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("shear_core-{label}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_test_fbin_with(path: &Path, rows: usize, dim: usize, value: impl Fn(usize, usize) -> f32) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let value = &value;
    let data: Vec<f32> = (0..rows)
        .flat_map(|row| (0..dim).map(move |col| value(row, col)))
        .collect();
    save_bin_f32(path, &data, rows, dim, 0).unwrap();
}

fn write_test_fbin(path: &Path, rows: usize, dim: usize) {
    write_test_fbin_with(path, rows, dim, |row, col| {
        row as f32 * 0.25 + col as f32 * 0.5
    });
}

fn test_config(data_path: &Path, num_threads: u32) -> IndexConfiguration {
    let (data_num, data_dim) = load_metadata_from_file(data_path).unwrap();
    let index_write_parameters = IndexWriteParametersBuilder::new(32, 16)
        .with_num_threads(num_threads)
        .build();
    IndexConfiguration::new(
        Metric::L2,
        data_dim,
        round_up(data_dim as u64, 32) as usize,
        data_num,
        false,
        0,
        false,
        0,
        1.2,
        index_write_parameters,
        0.0,
    )
}

#[test]
fn pipnn_oom_to_file_direct_store_round_trips_without_loading_dataset_into_index() {
    let dir = unique_temp_dir("pipnn-oom-file");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    write_test_fbin(&data_path, 48, 8);

    let config = test_config(&data_path, 1);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.c_min = 2;
    params.c_max = 8;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.fanout_top = 4;
    params.fanout_second = 2;

    build_pipnn_graph_oom_to_file(&data_path, 48, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    let loaded = index.load_graph(&graph_path, 48).unwrap();
    assert_eq!(loaded, 48);
    for vid in 0..48_u32 {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            !neighbors.get_neighbors().is_empty(),
            "expected node {vid} to have neighbors in direct-store OOM graph"
        );
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "expected node {vid} to respect write_range cap"
        );
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn hashprune_diagnostics_oom_build_writes_complete_artifacts() {
    let dir = unique_temp_dir("pipnn-oom-hashprune-diagnostics");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let diagnostics = dir.join("diagnostics");
    write_test_fbin_with(&data_path, 48, 8, |row, col| {
        ((row * 17 + col * 11) % 101) as f32 / 13.0
    });

    let config = test_config(&data_path, 1);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.c_min = 2;
    params.c_max = 8;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.fanout_top = 4;
    params.fanout_second = 2;
    params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
    params.hashprune_backfill_diagnostics_prefix = Some(diagnostics.clone());

    build_pipnn_graph_oom_to_file(&data_path, 48, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists());
    for name in [
        "primary_count.sidecar",
        "primary_count_manifest.json",
        "construction_metrics.json",
        "counterfactual_policy_summary.tsv",
        "alpha_sensitivity.tsv",
        "distance_layer_histograms.tsv",
    ] {
        let path = diagnostics.join(name);
        assert!(
            path.exists(),
            "missing diagnostic artifact {}",
            path.display()
        );
        assert!(!diagnostics.join(format!("{name}.partial")).exists());
    }

    let metrics: serde_json::Value =
        serde_json::from_slice(&fs::read(diagnostics.join("construction_metrics.json")).unwrap())
            .unwrap();
    assert_eq!(metrics["rows"].as_u64(), Some(48));
    assert_eq!(
        metrics["primary_edges"].as_u64().unwrap() + metrics["backfill_edges"].as_u64().unwrap(),
        metrics["retained_backfill_hist"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(degree, count)| degree as u64 * count.as_u64().unwrap())
            .sum::<u64>()
            + metrics["primary_degree_hist"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(degree, count)| degree as u64 * count.as_u64().unwrap())
                .sum::<u64>()
    );

    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pipnn_cuda")]
#[test]
fn pipnn_oom_gpu_subtree_persistent_multi_cta_small_graph_writes_loadable_index() {
    let dir = unique_temp_dir("pipnn-gpu-subtree-persistent-multi-cta");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("profile.json");
    write_test_fbin(&data_path, 96, 8);

    let config = test_config(&data_path, 2);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.oom_profile_json = Some(profile_path.clone());
    params.partitioner = PiPNNPartitionerKind::GpuSubtree;
    params.compute_backend = PiPNNComputeBackendKind::Cuda;
    params.gpu_device_id = 0;
    params.gpu_workspace_budget_bytes = 128 * 1024 * 1024;
    params.c_min = 4;
    params.c_max = 16;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.psamp_fraction = 1.0;
    params.fanout_top = 10;
    params.fanout_second = 3;
    params.leaf_knn = 4;

    build_pipnn_graph_oom_to_file(&data_path, 96, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");
    assert!(profile_path.exists(), "expected profile.json output");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    let loaded = index.load_graph(&graph_path, 96).unwrap();
    assert_eq!(loaded, 96);
    for vid in [0_u32, 1, 7, 31, 63, 95] {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            !neighbors.get_neighbors().is_empty(),
            "expected node {vid} to have neighbors in GPU subtree graph"
        );
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "expected node {vid} to respect write_range cap"
        );
    }

    let text = fs::read_to_string(&profile_path).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let gpu = &value["gpu_subtree"];
    assert_eq!(gpu["enabled"].as_bool(), Some(true));
    assert!(gpu["persistent_success_count"].as_u64().unwrap_or(0) > 0);
    assert_eq!(gpu["persistent_queue_overflow_flags"].as_u64(), Some(0));
    assert!(
        gpu["persistent_partition_worker_blocks"]
            .as_u64()
            .unwrap_or(0)
            > 1
    );
    assert!(gpu["persistent_topk_worker_blocks"].as_u64().unwrap_or(0) > 1);
    assert_eq!(
        gpu["persistent_partition_claims"].as_u64(),
        gpu["persistent_partition_pops"].as_u64()
    );
    assert_eq!(
        gpu["persistent_topk_claims"].as_u64(),
        gpu["persistent_topk_pops"].as_u64()
    );
    assert_eq!(
        gpu["persistent_leaf_pushes"].as_u64(),
        gpu["persistent_topk_claims"].as_u64()
    );

    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pipnn_cuda")]
#[test]
fn pipnn_oom_gpu_subtree_writeback_cache_emits_bounded_graph_and_profile() {
    let dir = unique_temp_dir("pipnn-gpu-subtree-writeback-cache");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("profile.json");
    write_test_fbin_with(&data_path, 96, 8, |row, col| {
        ((row * 17 + col * 11) % 101) as f32 / 13.0
    });

    let config = test_config(&data_path, 2);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.oom_profile_json = Some(profile_path.clone());
    params.oom_reservoir_budget_share = 0.25;
    params.partitioner = PiPNNPartitionerKind::GpuSubtree;
    params.compute_backend = PiPNNComputeBackendKind::Cuda;
    params.gpu_device_id = 0;
    params.gpu_workspace_budget_bytes = 128 * 1024 * 1024;
    params.c_min = 4;
    params.c_max = 16;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.psamp_fraction = 1.0;
    params.fanout_top = 10;
    params.fanout_second = 3;
    params.leaf_knn = 4;
    params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::WritebackCache;
    params.online_suffix_direct_queue_bytes = 64 * 1024;
    params.online_suffix_state_cache_bytes = 4096;
    params.online_suffix_state_page_cache_bytes = 8192;
    params.online_suffix_state_writeback_bytes = 4096;
    params.online_suffix_spill_buffer_bytes = 4096;
    params.online_suffix_reorder_window_bytes = 64 * 1024;
    params.online_suffix_read_window_bytes = 4096;
    params.online_suffix_pinned_prefix_points = Some(4);
    params.online_suffix_reduce_workers = 1;

    build_pipnn_graph_oom_to_file(&data_path, 96, &config, &params, &graph_path)
        .expect("write-back-cache GPU subtree graph build");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    assert_eq!(index.load_graph(&graph_path, 96).unwrap(), 96);
    for vid in 0..96_u32 {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "node {vid} exceeds the configured degree bound"
        );
    }

    let value: Value = serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
    let suffix = &value["online_suffix_reduce"];
    assert_eq!(suffix["active"].as_bool(), Some(true));
    assert_eq!(suffix["mode"].as_str(), Some("writeback-cache"));
    assert!(suffix["state_cache_misses"].as_u64().unwrap_or(0) > 0);
    assert!(
        suffix["state_cache_coalesced_updates"].as_u64().is_some(),
        "expected the B1 coalescing counter in the profile"
    );
    assert!(suffix["state_writeback_records"].as_u64().unwrap_or(0) > 0);
    assert_eq!(suffix["pinned_prefix_points"].as_u64(), Some(4));

    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pipnn_cuda")]
#[test]
fn pipnn_oom_gpu_subtree_log_assisted_emits_bounded_graph_and_profile() {
    let dir = unique_temp_dir("pipnn-gpu-subtree-log-assisted");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("profile.json");
    write_test_fbin_with(&data_path, 96, 8, |row, col| {
        ((row * 17 + col * 11) % 101) as f32 / 13.0
    });

    let config = test_config(&data_path, 2);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.oom_profile_json = Some(profile_path.clone());
    params.oom_reservoir_budget_share = 0.000001;
    params.partitioner = PiPNNPartitionerKind::GpuSubtree;
    params.compute_backend = PiPNNComputeBackendKind::Cuda;
    params.gpu_device_id = 0;
    params.gpu_workspace_budget_bytes = 128 * 1024 * 1024;
    params.c_min = 4;
    params.c_max = 16;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.psamp_fraction = 1.0;
    params.fanout_top = 10;
    params.fanout_second = 3;
    params.leaf_knn = 4;
    params.online_suffix_reduce_mode = PiPNNOnlineSuffixReduceMode::LogAssisted;
    params.online_suffix_direct_queue_bytes = 64 * 1024;
    params.online_suffix_state_cache_bytes = 4096;
    params.online_suffix_spill_buffer_bytes = 4096;
    params.online_suffix_reorder_window_bytes = 64 * 1024;
    params.online_suffix_reduce_workers = 1;
    params.online_suffix_delta_budget_bytes = 1024 * 1024;
    params.online_suffix_delta_buffer_bytes = 4096;
    params.online_suffix_range_points = 32;
    params.online_suffix_range_trigger_bytes = 4096;
    params.online_suffix_max_runs_per_range = 4;
    params.online_suffix_compaction_workers = 1;
    params.online_suffix_cache_hot_updates = 2;

    build_pipnn_graph_oom_to_file(&data_path, 96, &config, &params, &graph_path)
        .expect("log-assisted GPU subtree graph build");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    assert_eq!(index.load_graph(&graph_path, 96).unwrap(), 96);
    for vid in 0..96_u32 {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "node {vid} exceeds the configured degree bound"
        );
    }

    let value: Value = serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
    let suffix = &value["online_suffix_reduce"];
    assert_eq!(suffix["active"].as_bool(), Some(true));
    assert_eq!(suffix["mode"].as_str(), Some("log-assisted"));
    assert!(suffix["delta_records"].as_u64().unwrap_or(0) > 0);
    assert!(suffix["range_compactions"].as_u64().unwrap_or(0) > 0);
    assert_eq!(suffix["delta_uncompacted_bytes"].as_u64(), Some(0));
    assert_eq!(suffix["cold_miss_base_reads"].as_u64(), Some(0));
    assert_eq!(
        suffix["range_manifest_switches"].as_u64(),
        suffix["range_compactions"].as_u64()
    );
    assert_eq!(suffix["state_pages_written"].as_u64(), Some(0));

    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pipnn_cuda")]
#[test]
fn pipnn_oom_gpu_subtree_cublas_gemm_small_graph_writes_loadable_index() {
    let dir = unique_temp_dir("pipnn-gpu-subtree-cublas-gemm");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("profile.json");
    write_test_fbin(&data_path, 96, 8);

    let config = test_config(&data_path, 2);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.oom_profile_json = Some(profile_path.clone());
    params.partitioner = PiPNNPartitionerKind::GpuSubtree;
    params.compute_backend = PiPNNComputeBackendKind::Cuda;
    params.gpu_subtree_leaf_topk_backend = PiPNNGpuSubtreeLeafTopKBackend::CublasGemm;
    params.gpu_device_id = 0;
    params.gpu_workspace_budget_bytes = 128 * 1024 * 1024;
    params.c_min = 4;
    params.c_max = 16;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.psamp_fraction = 1.0;
    params.fanout_top = 1;
    params.fanout_second = 1;
    params.leaf_knn = 4;

    build_pipnn_graph_oom_to_file(&data_path, 96, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");
    assert!(profile_path.exists(), "expected profile.json output");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    let loaded = index.load_graph(&graph_path, 96).unwrap();
    assert_eq!(loaded, 96);
    for vid in [0_u32, 1, 7, 31, 63, 95] {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            !neighbors.get_neighbors().is_empty(),
            "expected node {vid} to have neighbors in cuBLAS GPU subtree graph"
        );
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "expected node {vid} to respect write_range cap"
        );
    }

    let text = fs::read_to_string(&profile_path).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let gpu = &value["gpu_subtree"];
    assert_eq!(gpu["enabled"].as_bool(), Some(true));
    assert_eq!(gpu["leaf_topk_backend"].as_str(), Some("cublas-gemm"));
    assert!(gpu["partition_only_total_ms"].as_u64().unwrap_or(0) > 0);
    assert!(gpu["library_topk_total_ms"].as_u64().unwrap_or(0) > 0);
    assert!(gpu["library_topk_gemm_ms"].as_u64().unwrap_or(0) > 0);
    assert!(gpu["library_topk_select_ms"].as_u64().unwrap_or(0) > 0);
    assert!(
        gpu.get("library_topk_d2h_ms").is_some(),
        "expected D2H timing field in profile: {gpu:?}"
    );
    assert!(gpu["library_topk_leaf_count"].as_u64().unwrap_or(0) > 0);
    assert!(gpu["library_topk_rows"].as_u64().unwrap_or(0) > 0);
    assert!(gpu["library_topk_batches"].as_u64().unwrap_or(0) > 0);
    assert_eq!(gpu["persistent_queue_overflow_flags"].as_u64(), Some(0));

    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pipnn_cuda")]
#[test]
fn pipnn_oom_gpu_subtree_sq_gpu_d0_staging_small_graph_writes_loadable_index() {
    let dir = unique_temp_dir("pipnn-gpu-subtree-sq-gpu-d0-staging");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("profile.json");
    let sidecar_prefix = dir.join("tiny.sq8");
    write_test_fbin(&data_path, 96, 8);

    let config = test_config(&data_path, 2);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.oom_profile_json = Some(profile_path.clone());
    params.partitioner = PiPNNPartitionerKind::GpuSubtree;
    params.compute_backend = PiPNNComputeBackendKind::Cuda;
    params.gpu_device_id = 0;
    params.gpu_workspace_budget_bytes = 128 * 1024 * 1024;
    params.c_min = 4;
    params.c_max = 16;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.psamp_fraction = 1.0;
    params.fanout_top = 4;
    params.fanout_second = 2;
    params.leaf_knn = 4;
    params.gpu_subtree_leaf_topk_backend = PiPNNGpuSubtreeLeafTopKBackend::PersistentWmmaTf32;
    params.gpu_subtree_sq_staging_mode = PiPNNGpuSubtreeSqStagingMode::GpuD0;
    params.gpu_subtree_sq_staging_upload_shard_rows = 17;
    params.gpu_subtree_sq_wmma_mode = PiPNNGpuSubtreeSqWmmaMode::Build;
    params.gpu_subtree_sq_wmma_sidecar_mode = PiPNNGpuSubtreeSqWmmaSidecarMode::Rebuild;
    params.gpu_subtree_sq_wmma_sidecar_prefix = Some(sidecar_prefix.clone());
    params.oom_io_mode = PiPNNOomIoMode::Direct;

    build_pipnn_graph_oom_to_file(&data_path, 96, &config, &params, &graph_path)
        .expect("SQ GPU-D0 staging graph build");

    assert!(graph_path.exists(), "expected _mem.index output");
    assert!(profile_path.exists(), "expected profile.json output");
    assert!(
        PathBuf::from(format!("{}.codes", sidecar_prefix.display())).exists(),
        "expected SQ sidecar codes output"
    );
    assert!(
        PathBuf::from(format!("{}.meta", sidecar_prefix.display())).exists(),
        "expected SQ sidecar meta output"
    );

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    let loaded = index.load_graph(&graph_path, 96).unwrap();
    assert_eq!(loaded, 96);
    for vid in [0_u32, 1, 7, 31, 63, 95] {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            !neighbors.get_neighbors().is_empty(),
            "expected node {vid} to have neighbors in SQ GPU-D0 staging graph"
        );
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "expected node {vid} to respect write_range cap"
        );
    }

    let text = fs::read_to_string(&profile_path).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let gpu = &value["gpu_subtree"];
    let sq = &gpu["sq_gpu_staging"];
    assert_eq!(gpu["enabled"].as_bool(), Some(true));
    assert_eq!(sq["enabled"].as_bool(), Some(true));
    assert_eq!(sq["mode"].as_str(), Some("gpu-d0"));
    assert_eq!(sq["bits"].as_u64(), Some(8));
    assert_eq!(sq["upload_shard_rows"].as_u64(), Some(17));
    assert!(sq["shard_count"].as_u64().unwrap_or(0) > 1);
    assert!(sq["leader_count"].as_u64().unwrap_or(0) > 0);
    assert_eq!(sq["sidecar_source"].as_str(), Some("gpu-d0"));
    assert_eq!(sq["sidecar_compatible"].as_bool(), Some(true));
    assert_eq!(sq["d0_distance"].as_str(), Some("sq8-code-code"));
    assert_eq!(sq["d0_leader_scope"].as_str(), Some("global"));
    for field in [
        "raw_h2d_ms",
        "qcode_d2h_ms",
        "assignment_fragment_d2h_ms",
        "leader_h2d_ms",
        "leader_quantize_ms",
        "d0_assign_distance_kernel_ms",
        "d0_assign_merge_kernel_ms",
    ] {
        assert!(
            sq.get(field).is_some(),
            "missing SQ GPU-D0 profile field {field}"
        );
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pipnn_cuda")]
#[test]
fn pipnn_oom_gpu_subtree_persistent_multi_cta_large_graph_writes_loadable_index() {
    let rows = 9216;
    let dim = 4;
    let dir = unique_temp_dir("pipnn-gpu-subtree-persistent-large");
    let data_path = dir.join("large.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("profile.json");
    write_test_fbin_with(&data_path, rows, dim, |row, col| {
        if col == 0 {
            row as f32 * 0.001
        } else {
            col as f32 * 0.01
        }
    });

    let config = test_config(&data_path, 2);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.oom_profile_json = Some(profile_path.clone());
    params.partitioner = PiPNNPartitionerKind::GpuSubtree;
    params.compute_backend = PiPNNComputeBackendKind::Cuda;
    params.gpu_device_id = 0;
    params.leaf_knn = 2;
    params.c_min = 128;
    params.c_max = 512;
    params.max_depth = 12;
    params.max_leaders = 2;
    params.fanout_top = 1;
    params.fanout_second = 1;
    params.psamp_fraction = 1.0;
    params.gpu_workspace_budget_bytes = 512 * 1024 * 1024;

    build_pipnn_graph_oom_to_file(&data_path, rows, &config, &params, &graph_path)
        .expect("persistent large graph build");

    assert!(graph_path.exists(), "expected _mem.index output");
    assert!(profile_path.exists(), "expected profile.json output");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    let loaded = index.load_graph(&graph_path, rows).unwrap();
    assert_eq!(loaded, rows);
    for vid in [0_u32, 1, 127, 1024, 4096, rows as u32 - 1] {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(
            !neighbors.get_neighbors().is_empty(),
            "expected node {vid} to have neighbors in GPU persistent large graph"
        );
        assert!(
            neighbors.get_neighbors().len() <= 16,
            "expected node {vid} to respect write_range cap"
        );
    }

    let text = fs::read_to_string(&profile_path).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let gpu = &value["gpu_subtree"];
    assert_eq!(gpu["enabled"].as_bool(), Some(true));
    assert!(gpu["persistent_success_count"].as_u64().unwrap_or(0) > 0);
    assert_eq!(gpu["persistent_queue_overflow_flags"].as_u64(), Some(0));
    assert!(
        gpu["persistent_large_node_splits"].as_u64().unwrap_or(0) > 0,
        "expected large-node splits in profile: {gpu:?}"
    );
    assert!(
        gpu["persistent_large_node_max_rows"].as_u64().unwrap_or(0) > 4096,
        "expected resident large-node max rows to cross small-node limit: {gpu:?}"
    );
    assert!(
        gpu["persistent_unique_resident_rows"].as_u64().unwrap_or(0) > 4096,
        "expected persistent resident work above small-node limit: {gpu:?}"
    );
    assert!(
        gpu["persistent_leaf_size_max"].as_u64().unwrap_or(u64::MAX) <= 1024,
        "expected leaves to respect persistent leaf max: {gpu:?}"
    );

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pipnn_oom_to_file_supports_strict_io_without_mmap_based_dataset_loading() {
    let dir = unique_temp_dir("pipnn-oom-file-strict");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    write_test_fbin(&data_path, 40, 8);

    let config = test_config(&data_path, 1);
    let mut params = PiPNNParams::default();
    params.final_prune = false;
    params.oom_enable = true;
    params.oom_strict_io = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_pipnn_oom");
    params.c_min = 2;
    params.c_max = 8;
    params.max_depth = 6;
    params.max_leaders = 24;
    params.fanout_top = 4;
    params.fanout_second = 2;

    build_pipnn_graph_oom_to_file(&data_path, 40, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");

    let mut index = InmemIndex::<f32>::new(config).unwrap();
    let loaded = index.load_graph(&graph_path, 40).unwrap();
    assert_eq!(loaded, 40);
    for vid in 0..40_u32 {
        let neighbors = index.final_graph.read_vertex_and_neighbors(vid).unwrap();
        assert!(!neighbors.get_neighbors().is_empty());
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pipnn_oom_to_file_rejects_strict_io_with_final_prune_until_strict_final_prune_exists() {
    let dir = unique_temp_dir("pipnn-oom-file-strict-final-prune");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    write_test_fbin(&data_path, 24, 8);

    let config = test_config(&data_path, 1);
    let mut params = PiPNNParams::default();
    params.final_prune = true;
    params.oom_enable = true;
    params.oom_strict_io = true;
    params.oom_temp_dir = dir.join("_pipnn_oom");

    let err =
        build_pipnn_graph_oom_to_file(&data_path, 24, &config, &params, &graph_path).unwrap_err();
    assert!(
        err.to_string()
            .contains("strict OOM PiPNN currently requires final_prune=false"),
        "unexpected error: {err}"
    );

    fs::remove_dir_all(&dir).unwrap();
}
