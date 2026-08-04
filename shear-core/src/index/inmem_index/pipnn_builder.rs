use std::ops::{Mul, Sub};
use std::path::Path;
use std::time::Instant;

use tracing::info;

use super::{DirectMemGraphWriter, InmemIndex};
use crate::algorithm::pipnn::{PiPNNParams, build_pipnn_graph};
use crate::common::{AnnError, AnnResult};
use crate::model::Neighbor;
use crate::model::graph::AdjacencyList;
use crate::model::scratch::InMemQueryScratch;
use crate::utils::thread_pool::execute_with_rayon_thread_pool;

impl<T> InmemIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    /// 仅从数据文件加载向量到 InmemDataset，不触发默认的搜索+剪枝构图流程。
    ///
    /// 该方法会执行与 `build` 相同的文件校验和数据加载逻辑，但不会调用 `link()`，
    /// 便于上层在自定义构图算法（如 PiPNN）中复用数据加载路径。
    pub fn load_data_only(&mut self, filename: &Path, num_points_to_load: usize) -> AnnResult<()> {
        self.load_data_only_impl(filename, num_points_to_load, false)
    }

    /// OOM mode variant of `load_data_only` that avoids preallocating the backing data buffer.
    pub fn load_data_only_oom(
        &mut self,
        filename: &Path,
        num_points_to_load: usize,
    ) -> AnnResult<()> {
        self.load_data_only_impl(filename, num_points_to_load, true)
    }

    fn load_data_only_impl(
        &mut self,
        filename: &Path,
        num_points_to_load: usize,
        oom_mode: bool,
    ) -> AnnResult<()> {
        use crate::utils::file_util::{file_exists, load_metadata_from_file};
        use crate::utils::set_rayon_num_threads;

        let filename = Path::new(filename);
        if !file_exists(filename) {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Data file {:?} does not exist.",
                filename
            )));
        }

        let (file_num_points, file_dim) = load_metadata_from_file(filename)?;
        if file_num_points > self.configuration.max_points {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has {} points, \n                but index can support only {} points as specified in configuration.",
                num_points_to_load, file_num_points, self.configuration.max_points
            )));
        }

        if num_points_to_load > file_num_points {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has only {} points.",
                num_points_to_load, file_num_points
            )));
        }

        if file_dim != self.configuration.dim {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Driver requests loading {} dimension, but file has {} dimension.",
                self.configuration.dim, file_dim
            )));
        }

        if self.configuration.use_pq_dist {
            // TODO: PQ
            todo!("PQ is not supported now");
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        if oom_mode {
            self.dataset = crate::model::InmemDataset::<T>::new_memory_mapped(
                self.configuration.max_points + self.configuration.num_frozen_pts,
                self.configuration.dim,
            )?;
        }
        self.dataset.set_memory_mapped(filename).unwrap();
        self.dataset.build_from_file(filename, num_points_to_load)?;

        self.num_active_pts = num_points_to_load;
        Ok(())
    }
}

impl InmemIndex<f32> {
    /// 使用 PiPNN 构图算法在当前 InmemIndex 上构建图。
    ///
    /// 要求在调用前已经通过 `load_data_only` 或等价方式加载好数据集，并设置好
    /// `num_active_pts`。该方法会直接覆盖 `final_graph` 中前 `num_active_pts` 个点的邻接表。
    pub fn build_graph_pipnn(&mut self, params: &PiPNNParams) -> AnnResult<()> {
        if self.num_active_pts < 1 {
            return Err(AnnError::log_index_error(
                "Error: Trying to build PiPNN graph with 0 points.".to_string(),
            ));
        }

        let t_all = Instant::now();
        if self.configuration.num_frozen_pts > 0 {
            self.start = self.configuration.max_points as u32;
        } else {
            self.start = 0;
        }

        let num_points = self.num_active_pts;
        let num_threads = self.configuration.index_write_parameter.num_threads;

        // 1) 调用算法层的 PiPNN 构图，直接写入 self.final_graph。
        let write_range = self.configuration.write_range();
        let build_result = build_pipnn_graph(
            &self.dataset,
            num_points,
            num_threads,
            params,
            |vid, neighbors| {
                let mut adj = AdjacencyList::for_range(write_range);
                for &nbr in &neighbors {
                    if adj.len() >= write_range {
                        break;
                    }
                    adj.push(nbr);
                }

                self.final_graph
                    .write_vertex_and_neighbors(vid)?
                    .set_neighbors(adj);
                Ok(())
            },
        )?;
        if self.configuration.num_frozen_pts == 0 {
            self.start = build_result.entry_point.unwrap_or(0);
        }

        // 3) 可选的最终细剪：在 PiPNN 输出的图上再跑一遍 per-node RobustPrune，
        //    保证方向多样性与度数上界。
        if params.final_prune {
            let t_prune = Instant::now();
            self.final_prune_pipnn(params)?;
            info!("PiPNN final prune done: elapsed={:?}", t_prune.elapsed());
        }

        info!(
            "PiPNN build_graph_pipnn done: elapsed={:?}",
            t_all.elapsed()
        );
        Ok(())
    }

    /// Build PiPNN graph and stream it directly to a `_mem.index` file without populating
    /// `final_graph`.
    pub fn build_graph_pipnn_oom_to_file(
        &mut self,
        params: &PiPNNParams,
        graph_file: &Path,
    ) -> AnnResult<()> {
        if self.num_active_pts < 1 {
            return Err(AnnError::log_index_error(
                "Error: Trying to build PiPNN graph with 0 points.".to_string(),
            ));
        }
        let t_all = Instant::now();
        if self.configuration.num_frozen_pts > 0 {
            self.start = self.configuration.max_points as u32;
        } else {
            self.start = 0;
        }

        let num_points = self.num_active_pts;
        let num_threads = self.configuration.index_write_parameter.num_threads;

        if params.root_radius_dry_run || params.root_risk_dry_run || params.root_stop_after_d00 {
            build_pipnn_graph(
                &self.dataset,
                num_points,
                num_threads,
                params,
                |_, _| Ok(()),
            )?;
            info!(
                "PiPNN build_graph_pipnn_oom_to_file stopped before graph output: elapsed={:?}",
                t_all.elapsed()
            );
            return Ok(());
        }

        if params.final_prune {
            let build_result = build_pipnn_graph(
                &self.dataset,
                num_points,
                num_threads,
                params,
                |vid, neighbors| {
                    let mut adj = AdjacencyList::for_range(self.configuration.write_range());
                    for &nbr in &neighbors {
                        if adj.len() >= self.configuration.write_range() {
                            break;
                        }
                        adj.push(nbr);
                    }

                    self.final_graph
                        .write_vertex_and_neighbors(vid)?
                        .set_neighbors(adj);
                    Ok(())
                },
            )?;
            if self.configuration.num_frozen_pts == 0 {
                self.start = build_result.entry_point.unwrap_or(0);
            }

            let t_prune = Instant::now();
            self.final_prune_pipnn(params)?;
            info!(
                "PiPNN OOM final prune done: elapsed={:?}",
                t_prune.elapsed()
            );

            let mut writer = DirectMemGraphWriter::create(
                graph_file,
                self.start,
                self.configuration.num_frozen_pts,
            )?;
            let write_range = self.configuration.write_range();
            for vid in 0..num_points as u32 {
                let neighbors = {
                    let guard = self.final_graph.read_vertex_and_neighbors(vid)?;
                    guard.get_neighbors().clone()
                };
                let capped = if neighbors.len() > write_range {
                    &neighbors[..write_range]
                } else {
                    neighbors.as_slice()
                };
                writer.write_node(vid, capped)?;
            }
            writer.finish()?;
        } else {
            let mut writer = DirectMemGraphWriter::create(
                graph_file,
                self.start,
                self.configuration.num_frozen_pts,
            )?;
            let write_range = self.configuration.write_range();

            let build_result = build_pipnn_graph(
                &self.dataset,
                num_points,
                num_threads,
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
            if self.configuration.num_frozen_pts == 0 {
                if let Some(entry_point) = build_result.entry_point {
                    writer.set_start(entry_point);
                }
            }
            writer.finish()?;
        }

        info!(
            "PiPNN build_graph_pipnn_oom_to_file done: elapsed={:?}",
            t_all.elapsed()
        );
        Ok(())
    }

    /// 在 PiPNN 输出的图上执行一次 per-node 的 RobustPrune，复用现有 `prune_neighbors`。
    fn final_prune_pipnn(&self, _params: &PiPNNParams) -> AnnResult<()> {
        let max_degree = self.configuration.index_write_parameter.max_degree;
        let search_list_size = self.configuration.index_write_parameter.search_list_size;
        let num_threads = self.configuration.index_write_parameter.num_threads;

        execute_with_rayon_thread_pool(0..self.num_active_pts, num_threads, |vid| {
            let vid_u32 = vid as u32;

            let neighbors_ids = {
                let guard = self.final_graph.read_vertex_and_neighbors(vid_u32)?;
                guard.get_neighbors().clone()
            };

            if neighbors_ids.is_empty() {
                return Ok(());
            }

            let mut pool: Vec<Neighbor> = Vec::with_capacity(neighbors_ids.len());
            let query = self.dataset.get_vertex(vid_u32)?;
            for &nid in neighbors_ids.iter() {
                let vertex = self.dataset.get_vertex(nid)?;
                let d = query.compare(&vertex, self.configuration.dist_metric);
                pool.push(Neighbor::new(nid, d));
            }

            let mut scratch = InMemQueryScratch::<f32>::new(
                search_list_size.max(max_degree),
                &self.configuration.index_write_parameter,
                self.dim,
            )?;

            let mut pruned = AdjacencyList::for_range(self.configuration.write_range());
            self.prune_neighbors(vid_u32, &mut pool, &mut pruned, &mut scratch)?;

            self.final_graph
                .write_vertex_and_neighbors(vid_u32)?
                .set_neighbors(pruned);
            Ok(())
        })
    }
}
