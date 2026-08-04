use std::path::Path;

use rand::SeedableRng;
use tracing::info;

use super::InmemIndex;
use crate::algorithm::pipnn::direct_io::DirectIoConfig;
use crate::algorithm::pipnn::point_store::{
    DirectPointStore, LimitedPointStore, PointStore, U8PointStore,
};
use crate::algorithm::pipnn::rbc_partition::root_radius_dry_run_json;
use crate::algorithm::pipnn::{PiPNNParams, build_pipnn_graph_with_store};
use crate::common::AnnResult;
use crate::index::DirectMemGraphWriter;
use crate::model::IndexConfiguration;

pub fn build_pipnn_graph_oom_to_file(
    dataset_path: &Path,
    num_points: usize,
    config: &IndexConfiguration,
    params: &PiPNNParams,
    graph_file: &Path,
) -> AnnResult<()> {
    if params.strict_oom_io_enabled() && params.final_prune {
        return Err(crate::common::AnnError::log_index_config_error(
            "oom_strict_io".to_string(),
            "strict OOM PiPNN currently requires final_prune=false because the final-prune file path still depends on InmemIndex-backed dataset loading".to_string(),
        ));
    }

    if !params.final_prune {
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let store = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
        build_pipnn_graph_oom_to_file_with_store(&store, num_points, config, params, graph_file)?;
        info!(
            "PiPNN lightweight OOM graph build done via direct f32 point store: num_points={} output={:?}",
            num_points, graph_file
        );
        return Ok(());
    }

    let mut index = InmemIndex::<f32>::new(config.clone())?;
    index.load_data_only_oom(dataset_path, num_points)?;
    index.build_graph_pipnn_oom_to_file(params, graph_file)?;

    info!(
        "PiPNN lightweight OOM graph build done: num_points={} output={:?}",
        num_points, graph_file
    );
    Ok(())
}

pub fn build_pipnn_graph_oom_to_file_u8(
    dataset_path: &Path,
    num_points: usize,
    config: &IndexConfiguration,
    params: &PiPNNParams,
    graph_file: &Path,
) -> AnnResult<()> {
    if params.final_prune {
        return Err(crate::common::AnnError::log_index_config_error(
            "final_prune".to_string(),
            "uint8 PiPNN build currently requires --final-prune=false because the final-prune OOM path still reloads data through InmemIndex<f32>".to_string(),
        ));
    }

    let io_cfg = if params.strict_oom_io_enabled() {
        DirectIoConfig::enabled_with_alignment(4096)
    } else {
        DirectIoConfig::disabled()
    };
    let store = U8PointStore::open_centered_with_config(dataset_path, io_cfg)?;
    build_pipnn_graph_oom_to_file_with_store(&store, num_points, config, params, graph_file)?;
    info!(
        "PiPNN lightweight OOM graph build done via centered uint8 point store: num_points={} output={:?}",
        num_points, graph_file
    );
    Ok(())
}

fn build_pipnn_graph_oom_to_file_with_store(
    store: &dyn PointStore,
    num_points: usize,
    config: &IndexConfiguration,
    params: &PiPNNParams,
    graph_file: &Path,
) -> AnnResult<()> {
    let limited_store = LimitedPointStore::new(store, num_points);
    if params.root_radius_dry_run || params.root_risk_dry_run {
        let all_indices: Vec<u32> = (0..num_points as u32).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(params.random_seed);
        let json = root_radius_dry_run_json(
            &limited_store,
            &all_indices,
            params.metric(),
            params,
            &mut rng,
        )?;
        println!("{json}");
        return Ok(());
    }
    let initial_start = if config.num_frozen_pts > 0 {
        config.max_points as u32
    } else {
        0
    };
    if params.root_stop_after_d00 {
        build_pipnn_graph_with_store(
            &limited_store,
            num_points,
            config.index_write_parameter.num_threads,
            params,
            |_, _| Ok(()),
        )?;
        info!(
            "PiPNN lightweight OOM stopped after root d00 materialization: num_points={}",
            num_points
        );
        return Ok(());
    }

    let mut writer =
        DirectMemGraphWriter::create(graph_file, initial_start, config.num_frozen_pts)?;
    let write_range = config.write_range();
    let build_result = build_pipnn_graph_with_store(
        &limited_store,
        num_points,
        config.index_write_parameter.num_threads,
        params,
        |vid, neighbors| {
            let capped = if neighbors.len() > write_range {
                &neighbors[..write_range]
            } else {
                neighbors.as_slice()
            };
            writer.write_node(vid, capped)
        },
    )?;
    if config.num_frozen_pts == 0 {
        if let Some(entry_point) = build_result.entry_point {
            writer.set_start(entry_point);
        }
    }
    writer.finish()?;
    Ok(())
}
