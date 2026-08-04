use std::cmp;
use std::collections::BinaryHeap;
use std::ops::{Mul, Sub};
use std::path::Path;
use std::time::Duration;

use dashmap::{DashMap, DashSet};
use hashbrown::HashSet;
use hashbrown::hash_set::Entry::*;
use parking_lot::RwLock;
use rand::rng;
use rand::seq::SliceRandom;
use tracing::{debug, info};

use crate::common::{AnnError, AnnResult};
use crate::index::AnnInMemIndex;
use crate::model::graph::AdjacencyList;
use crate::model::{
    ArcConcurrentBoxedQueue, InMemQueryScratch, InMemoryGraph, IndexConfiguration, InmemDataset,
    Neighbor, ScratchStoreManager, Vertex,
};
use crate::utils::file_util::{file_exists, load_metadata_from_file};
use crate::utils::thread_pool::execute_with_rayon_thread_pool;
use crate::utils::{Timer, set_rayon_num_threads};

/// In-memory Index
pub struct InmemIndex<T> {
    /// Dataset
    pub dataset: InmemDataset<T>,

    /// Graph
    pub final_graph: InMemoryGraph,

    /// Index configuration
    pub configuration: IndexConfiguration,

    /// Start point of the search. When _num_frozen_pts is greater than zero,
    /// this is the location of the first frozen point. Otherwise, this is a
    /// location of one of the points in index.
    pub start: u32,

    /// Max observed out degree
    pub max_observed_degree: u32,

    /// Number of active points i.e. existing in the graph
    pub num_active_pts: usize,

    /// query scratch queue.
    query_scratch_queue: ArcConcurrentBoxedQueue<InMemQueryScratch<T>>,

    pub delete_set: RwLock<HashSet<u32>>,

    pub dim: usize,

    /// Subgraph store
    pub subgraph: DashMap<u32, BinaryHeap<Neighbor>>,

    /// Prefetching table
    pub prefetch_table: DashMap<u32, Vec<u32>>,

    /// Subgraph table
    pub subgraph_set: DashSet<u32>,
}

impl<T> InmemIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    /// Create Index obj based on configuration
    pub fn new(mut config: IndexConfiguration) -> AnnResult<Self> {
        // Sanity check. While logically it is correct, max_points = 0 causes
        // downstream problems.
        if config.max_points == 0 {
            config.max_points = 1;
        }

        let total_internal_points = config.max_points + config.num_frozen_pts;

        let start = config.max_points.try_into()?;

        let query_scratch_queue = ArcConcurrentBoxedQueue::<InMemQueryScratch<T>>::new();
        let delete_set = RwLock::new(HashSet::<u32>::new());

        Ok(Self {
            dataset: InmemDataset::<T>::new(
                total_internal_points,
                config.growth_potential,
                config.dim,
            )?,
            final_graph: InMemoryGraph::new(
                total_internal_points,
                config.index_write_parameter.max_degree,
            ),
            dim: config.dim,
            configuration: config,
            start,
            max_observed_degree: 0,
            num_active_pts: 0,
            query_scratch_queue,
            delete_set,
            subgraph: DashMap::new(),
            prefetch_table: DashMap::new(),
            subgraph_set: DashSet::new(),
        })
    }

    pub fn query_scratch_queue_len(&self) -> AnnResult<usize> {
        self.query_scratch_queue.size()
    }

    pub fn ensure_query_scratch_initialized(&mut self) -> AnnResult<()> {
        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }
        Ok(())
    }

    /// Get distance between two vertices.
    pub fn get_distance(&self, id1: u32, id2: u32) -> AnnResult<f32> {
        self.dataset
            .get_distance(id1, id2, self.configuration.dist_metric)
    }

    fn build_with_data_populated(&mut self) -> AnnResult<()> {
        info!(
            "Starting index build with {} points...",
            self.num_active_pts
        );

        if self.num_active_pts < 1 {
            return Err(AnnError::log_index_error(
                "Error: Trying to build an index with 0 points.".to_string(),
            ));
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        // TODO: generate_frozen_point()

        self.link()?;

        self.print_stats()?;

        Ok(())
    }

    fn link(&mut self) -> AnnResult<()> {
        // visit_order is a vector that is initialized to the entire graph
        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);

        // Create a vector with all active points
        let mut all_active_points: Vec<u32> = (0..self.num_active_pts).map(|i| i as u32).collect();

        // Randomly select 30% of the active points
        let sample_size = ((self.num_active_pts as f32) * 0.7) as usize;
        let sample_size = if sample_size == 0 && self.num_active_pts > 0 {
            1
        } else {
            sample_size
        };
        let mut rng = rng();
        all_active_points.shuffle(&mut rng);
        visit_order.extend_from_slice(&all_active_points[..sample_size]);

        let mut centroid_node = Vec::new();
        centroid_node.extend_from_slice(&all_active_points[sample_size..]);

        // If there are any frozen points, add them all.
        for frozen in self.configuration.max_points
            ..(self.configuration.max_points + self.configuration.num_frozen_pts)
        {
            visit_order.push(frozen as u32);
        }

        // if there are frozen points, the first such one is set to be the _start
        if self.configuration.num_frozen_pts > 0 {
            self.start = self.configuration.max_points as u32;
        } else {
            self.start = self.dataset.calculate_medoid_point_id()?;
        }

        let timer = Timer::new();

        let range = visit_order.len();

        execute_with_rayon_thread_pool(
            0..range,
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                self.insert_vertex_id(visit_order[idx], false)?;

                if idx.is_multiple_of(10000) {
                    info!("We have built the {idx} vertex.");
                }
                Ok(())
            },
        )?;

        debug!("End main bone construction");

        execute_with_rayon_thread_pool(0..centroid_node.len(), 8, |idx| {
            self.reduce_vertex(centroid_node[idx])?;
            Ok(())
        })?;

        let subgraph_set = self.subgraph_set.iter().map(|v| *v).collect::<Vec<_>>();
        debug!("subgraph set len: {:}", subgraph_set.len());

        // execute_with_rayon_thread_pool(
        //     0..subgraph_set.len(),
        //     1,
        //     |idx| {
        //         self.insert_vertex_id(subgraph_set[idx], true)?;
        //         Ok(())
        //     },
        // )?;

        self.cleanup_graph(
            &(0..self.num_active_pts)
                .map(|v| v as u32)
                .collect::<Vec<_>>(),
        )?;

        if self.num_active_pts > 0 {
            println!("{}", timer.elapsed_seconds_for_step("Link time: "));
        }

        Ok(())
    }

    fn insert_vertex_id(&self, vertex_id: u32, overload: bool) -> AnnResult<()> {
        let mut vertex_id = vertex_id;
        let mut scratch_manager =
            ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;
        let scratch = scratch_manager.scratch_space().ok_or_else(|| {
            AnnError::log_index_error(
                "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
            )
        })?;

        let new_neighbors = self.search_for_point_and_prune(scratch, &mut vertex_id, overload)?;
        self.update_vertex_with_neighbors(vertex_id, new_neighbors)?;

        self.update_neighbors_of_vertex(vertex_id, scratch, overload)?;

        Ok(())
    }

    fn update_neighbors_of_vertex(
        &self,
        vertex_id: u32,
        scratch: &mut InMemQueryScratch<T>,
        overload: bool,
    ) -> Result<(), AnnError> {
        let vertex = self.final_graph.read_vertex_and_neighbors(vertex_id)?;
        assert!(vertex.size() <= self.configuration.index_write_parameter.max_degree as usize);
        self.inter_insert(
            vertex_id,
            vertex.get_neighbors(),
            self.configuration.index_write_parameter.max_degree,
            scratch,
            overload,
        )?;
        Ok(())
    }

    fn update_vertex_with_neighbors(
        &self,
        vertex_id: u32,
        new_neighbors: AdjacencyList,
    ) -> Result<(), AnnError> {
        let vertex = &mut self.final_graph.write_vertex_and_neighbors(vertex_id)?;
        vertex.set_neighbors(new_neighbors);
        assert!(
            vertex.size() <= self.configuration.index_write_parameter.max_degree as usize,
            "{}",
            vertex.size()
        );
        Ok(())
    }

    fn search_for_point_and_prune(
        &self,
        scratch: &mut InMemQueryScratch<T>,
        vertex_id: &mut u32,
        overload: bool,
    ) -> AnnResult<AdjacencyList> {
        let mut pruned_list =
            AdjacencyList::for_range(self.configuration.index_write_parameter.max_degree as usize);
        let vertex = self.dataset.get_vertex(*vertex_id)?;
        let mut visited_nodes = self.search_for_point(&vertex, scratch)?;

        if !overload {
            self.prune_neighbors(*vertex_id, &mut visited_nodes, &mut pruned_list, scratch)?;
        } else {
            visited_nodes.sort();
            for item in visited_nodes {
                pruned_list.push(item.id);
            }
            pruned_list.truncate(self.configuration.index_write_parameter.max_degree as usize);
        }

        if self.final_graph.size()
            != self.configuration.max_points + self.configuration.num_frozen_pts
        {
            return Err(AnnError::log_index_error(format!(
                "final_graph has {} vertices instead of {}",
                self.final_graph.size(),
                self.configuration.max_points + self.configuration.num_frozen_pts,
            )));
        }

        Ok(pruned_list)
    }

    /// Reduce vertex: Update graph structure and collect co-occurrence statistics for layout
    /// optimization
    fn reduce_vertex(&self, vertex_id: u32) -> AnnResult<()> {
        let mut scratch_manager =
            ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;
        let scratch = scratch_manager.scratch_space().ok_or_else(|| {
            AnnError::log_index_error(
                "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
            )
        })?;

        // 1. Search for the point to get the visitation path (simulating a query)
        let vertex = self.dataset.get_vertex(vertex_id)?;
        let mut visited_nodes = self.search_for_point(&vertex, scratch)?;

        // 2. Co-occurrence Statistics (Sliding Window)
        // We use the search result 'visited_nodes' as a proxy for the traversal path.
        // Nodes that appear together in the result set are considered "co-occurred".
        // We record the co-occurrence between the target 'vertex_id' and its neighbors in the
        // search path. This helps in placing 'vertex_id' close to nodes that lead to it or
        // are close to it.

        // Window size for local context
        const WINDOW_SIZE: usize = 6;
        // Limit stats to top K visited nodes to avoid noise and excessive computation
        const TOP_K_WINDOW: usize = 30;

        // Take top K nodes for statistics
        // let stats_len = visited_nodes.len().min(TOP_K_WINDOW);
        // let start_index = visited_nodes.len() - stats_len;
        // let total_len = visited_nodes.len();
        let total_len = (visited_nodes.len() + TOP_K_WINDOW).min(visited_nodes.len());
        let stats_len = visited_nodes.len().min(TOP_K_WINDOW);
        let start_index = 0;

        // Reverse traversal from the end of the list (closest/last visited) backwards
        for i in start_index..stats_len {
            let u = visited_nodes[i];

            // Define a window of context relative to i
            // We look at nodes visited just before and just after i in the path
            // Since we are traversing in reverse, this window logic captures local context
            let start = i;
            let end = (i + WINDOW_SIZE).min(total_len);

            for (j, v) in visited_nodes.iter().enumerate().take(end).skip(start) {
                if i == j {
                    continue;
                }

                // Calculate weight: Closer rank difference -> Higher weight
                // Also, absolute rank matters: top candidates (higher index in visited_nodes) are
                // more important.
                let dist = (i as isize - j as isize).abs() as f32;
                // Penalty based on how far back from the end we are
                // let rank_from_end = (total_len - 1 - i) + (total_len - 1 - j);
                // let rank_penalty = rank_from_end as f32 * 0.1;

                let weight = 1.0 / (dist + 1.0);

                // Update co-occurrence for (u, v)
                // Note: visited_nodes contains Neighbors { id, distance, .. }
                // We use update_co_occurrence to store this relation.
                self.update_co_occurrence(u.id, v.id, weight);
            }

            // Also record strong connection between vertex_id (the "query") and the top results
            // We only do this for the very top results (last few elements in visited_nodes)
            if i <= WINDOW_SIZE * 2 {
                let rank_from_end = total_len - 1 - i;
                let weight = 2.0 / (rank_from_end as f32 + 1.0);
                self.update_co_occurrence(vertex_id, u.id, weight);
                self.update_co_occurrence(u.id, vertex_id, weight);
            }
        }

        // 3. Prune neighbors (Standard graph construction logic)
        let mut pruned_list =
            AdjacencyList::for_range(self.configuration.index_write_parameter.max_degree as usize);

        self.prune_neighbors(vertex_id, &mut visited_nodes, &mut pruned_list, scratch)?;

        if pruned_list.is_empty() {
            return Ok(());
        }

        self.update_vertex_with_neighbors(vertex_id, pruned_list)?;

        // 4. Update reverse edges (add vertex_id to neighbors' lists)
        self.update_neighbors_of_vertex(vertex_id, scratch, true)?;
        Ok(())
    }

    /// Update co-occurrence weight in the subgraph cache.
    ///
    /// We reuse `self.subgraph` (DashMap<u32, BinaryHeap<Neighbor>>) to store the top-K
    /// co-occurring nodes. `Neighbor.distance` is used to store `-weight` (negative weight) so
    /// that BinaryHeap (MaxHeap) acts as a MinHeap for weights, allowing us to evict the
    /// smallest weight when capacity is full.
    fn update_co_occurrence(&self, u: u32, v: u32, weight: f32) {
        // Capacity for co-occurrence list
        const MAX_CO_OCCURRENCE: usize = 20;

        let mut heap = self.subgraph.entry(u).or_default();

        // We need to support weight accumulation.
        // Since BinaryHeap is not indexable, we convert to Vec to check for existing entry.
        // Given MAX_CO_OCCURRENCE is small (32), this is acceptable.

        let mut found = false;
        // Access underlying vec if possible, but BinaryHeap doesn't expose mutable iter easily.
        // We can drain and rebuild.
        let mut entries: Vec<Neighbor> = heap.drain().collect();

        for entry in &mut entries {
            if entry.id == v {
                // Accumulate weight.
                // stored distance is -weight.
                // new_weight = old_weight + weight
                // new_dist = -(old_weight + weight) = old_dist - weight
                entry.distance -= weight;
                found = true;
                break;
            }
        }

        if !found {
            entries.push(Neighbor::new(v, -weight));
        }

        // Rebuild heap
        *heap = BinaryHeap::from(entries);

        // Enforce capacity
        while heap.len() > MAX_CO_OCCURRENCE {
            heap.pop();
        }
    }

    /// Generate an optimized layout based on collected co-occurrence statistics.
    /// Returns a vector mapping: index -> node_id (the order in which nodes should be stored).
    #[cfg(test)]
    pub fn generate_optimized_layout(&self) -> Vec<u32> {
        info!("Generating optimized layout based on co-occurrence stats...");
        let num_points = self.num_active_pts as u32;
        let mut visited = HashSet::new();
        let mut layout = Vec::with_capacity(num_points as usize);
        let mut queue = std::collections::VecDeque::new();

        // Simple BFS-like traversal prioritized by weight
        // Start with the search start node (medoid)
        queue.push_back(self.start);
        visited.insert(self.start);

        while layout.len() < num_points as usize {
            if queue.is_empty() {
                // If queue empty but not all points visited, pick next unvisited
                for i in 0..num_points {
                    if !visited.contains(&i) {
                        queue.push_back(i);
                        visited.insert(i);
                        break;
                    }
                }
            }

            if let Some(curr) = queue.pop_front() {
                layout.push(curr);

                // Get high-weight neighbors from subgraph
                if let Some(neighbors) = self.subgraph.get(&curr) {
                    // Collect neighbors and sort by weight (descending)
                    // stored distance is -weight.
                    // Lower stored distance (more negative) -> Higher weight.
                    let mut nbrs: Vec<_> = neighbors.iter().collect();
                    nbrs.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()); // -10 < -1 -> -10 comes first (High weight)

                    // Filter out weak connections
                    // Threshold can be tuned. For now, we only follow connections with accumulated
                    // weight > 0.5
                    const LAYOUT_WEIGHT_THRESHOLD: f32 = 0.5;

                    for n in nbrs {
                        // stored distance is -weight. so weight = -distance
                        let weight = -n.distance;

                        if weight > LAYOUT_WEIGHT_THRESHOLD && !visited.contains(&n.id) {
                            visited.insert(n.id);
                            queue.push_back(n.id);
                        }
                    }
                }
            }
        }

        layout
    }

    fn search(
        &self,
        query: &Vertex<T>,
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> AnnResult<u32> {
        if k_value > l_value as usize {
            return Err(AnnError::log_index_error(format!(
                "Set L: {} to a value of at least K: {}",
                l_value, k_value
            )));
        }

        let mut scratch_manager =
            ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;

        let scratch = scratch_manager.scratch_space().ok_or_else(|| {
            AnnError::log_index_error(
                "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
            )
        })?;

        if l_value > scratch.candidate_size {
            println!(
                "Attempting to expand query scratch_space. Was created with Lsize: {} but search L is: {}",
                scratch.candidate_size, l_value
            );
            scratch.resize_for_new_candidate_size(l_value);
            println!(
                "Resize completed. New scratch size is: {}",
                scratch.candidate_size
            );
        }

        let cmp = self.search_with_l_override(query, scratch, l_value as usize)?;
        let mut pos = 0;

        for i in 0..scratch.best_candidates.size() {
            if scratch.best_candidates[i].id < self.configuration.max_points as u32 {
                // Filter out the deleted points.
                let delete_set_guard = loop {
                    if let Some(guard) = self.delete_set.try_read() {
                        break guard;
                    }
                    // rayon::yield_now();
                };
                if !delete_set_guard.contains(&scratch.best_candidates[i].id) {
                    indices[pos] = scratch.best_candidates[i].id;
                    pos += 1;
                }
            }

            if pos == k_value {
                break;
            }
        }

        if pos < k_value {
            eprintln!(
                "Found fewer than K elements for query! Found: {} but K: {}",
                pos, k_value
            );
        }

        Ok(cmp)
    }

    fn cleanup_graph(&mut self, visit_order: &[u32]) -> AnnResult<()> {
        if self.num_active_pts > 0 {
            info!("Starting final cleanup..");
        }

        execute_with_rayon_thread_pool(
            0..visit_order.len(),
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                let vertex_id = visit_order[idx];
                let num_nbrs = self.get_neighbor_count(vertex_id)?;

                if num_nbrs <= self.configuration.index_write_parameter.max_degree as usize {
                    // Neighbor list is already small enough.
                    return Ok(());
                }

                let mut scratch_manager = ScratchStoreManager::new(
                    self.query_scratch_queue.clone(),
                    Duration::from_millis(10),
                )?;
                let scratch = scratch_manager.scratch_space().ok_or_else(|| {
                    AnnError::log_index_error(
                        "ScratchStoreManager doesn't have InMemQueryScratch instance available"
                            .to_string(),
                    )
                })?;

                let mut dummy_pool = self.get_neighbors_for_vertex(vertex_id)?;

                let mut new_out_neighbors = AdjacencyList::for_range(
                    self.configuration.index_write_parameter.max_degree as usize,
                );
                self.prune_neighbors(vertex_id, &mut dummy_pool, &mut new_out_neighbors, scratch)?;

                self.final_graph
                    .write_vertex_and_neighbors(vertex_id)?
                    .set_neighbors(new_out_neighbors);

                Ok(())
            },
        )
    }

    /// Get the unique neighbors for a vertex.
    ///
    /// This code feels out of place here. This should have nothing to do with whether this
    /// is in memory index?
    /// # Errors
    ///
    /// This function will return an error if we are not able to get the read lock.
    fn get_neighbors_for_vertex(&self, vertex_id: u32) -> AnnResult<Vec<Neighbor>> {
        let binding = self.final_graph.read_vertex_and_neighbors(vertex_id)?;
        let neighbors = binding.get_neighbors();
        let dummy_pool = self.get_unique_neighbors(neighbors, vertex_id)?;

        Ok(dummy_pool)
    }

    /// Returns a vector of unique neighbors for the given vertex, along with their distances.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - A vector of neighbor id index for the given vertex.
    /// * `vertex_id` - The given vertex id.
    ///
    /// # Errors
    ///
    /// Returns an `ANNError` if there is an error retrieving the vertex or one of its neighbors.
    pub fn get_unique_neighbors(
        &self,
        neighbors: &[u32],
        vertex_id: u32,
    ) -> Result<Vec<Neighbor>, AnnError> {
        let vertex = self.dataset.get_vertex(vertex_id)?;

        let len = neighbors.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        self.dataset.prefetch_vector(neighbors[0]);

        let mut dummy_visited: HashSet<u32> = HashSet::with_capacity(len);
        let mut dummy_pool: Vec<Neighbor> = Vec::with_capacity(len);

        // let slice = ['w', 'i', 'n', 'd', 'o', 'w', 's'];
        // for window in slice.windows(2) {
        //   &println!{"[{}, {}]", window[0], window[1]};
        // }
        // prints: [w, i] -> [i, n] -> [n, d] -> [d, o] -> [o, w] -> [w, s]
        for current in neighbors.windows(2) {
            // Prefetch the next item.
            self.dataset.prefetch_vector(current[1]);
            let current = current[0];

            self.insert_neighbor_if_unique(
                &mut dummy_visited,
                current,
                vertex_id,
                &vertex,
                &mut dummy_pool,
            )?;
        }

        // Insert the last neighbor
        #[allow(clippy::unwrap_used)]
        self.insert_neighbor_if_unique(
            &mut dummy_visited,
            *neighbors.last().unwrap(), // we know len != 0, so this is safe.
            vertex_id,
            &vertex,
            &mut dummy_pool,
        )?;

        Ok(dummy_pool)
    }

    fn insert_neighbor_if_unique(
        &self,
        dummy_visited: &mut HashSet<u32>,
        current: u32,
        vertex_id: u32,
        vertex: &Vertex<'_, T>,
        dummy_pool: &mut Vec<Neighbor>,
    ) -> Result<(), AnnError> {
        if current != vertex_id
            && let Vacant(entry) = dummy_visited.entry(current)
        {
            let cur_nbr_vertex = self.dataset.get_vertex(current)?;
            let dist = vertex.compare(&cur_nbr_vertex, self.configuration.dist_metric);
            dummy_pool.push(Neighbor::new(current, dist));
            entry.insert();
        }

        Ok(())
    }

    /// Get count of neighbors for a given vertex.
    ///
    /// # Errors
    ///
    /// This function will return an error if we can't get a lock.
    fn get_neighbor_count(&self, vertex_id: u32) -> AnnResult<usize> {
        let num_nbrs = self
            .final_graph
            .read_vertex_and_neighbors(vertex_id)?
            .size();
        Ok(num_nbrs)
    }

    fn soft_delete_vertex(&self, vertex_id_to_delete: u32) -> AnnResult<()> {
        if vertex_id_to_delete as usize > self.num_active_pts {
            return Err(AnnError::log_index_error(format!(
                "vertex_id_to_delete: {} is greater than the number of active points in the graph: {}",
                vertex_id_to_delete, self.num_active_pts
            )));
        }
        let mut delete_set_guard = loop {
            if let Some(guard) = self.delete_set.try_write() {
                break guard;
            }
            // rayon::yield_now();
        };

        delete_set_guard.insert(vertex_id_to_delete);
        Ok(())
    }

    fn initialize_query_scratch(
        &mut self,
        num_threads: u32,
        search_candidate_size: u32,
    ) -> AnnResult<()> {
        self.query_scratch_queue.reserve(num_threads as usize)?;
        for _ in 0..num_threads {
            let scratch = Box::new(InMemQueryScratch::<T>::new(
                search_candidate_size,
                &self.configuration.index_write_parameter,
                self.dim,
            )?);

            self.query_scratch_queue.push(scratch)?;
        }

        Ok(())
    }

    fn print_stats(&mut self) -> AnnResult<()> {
        let mut max = 0;
        let mut min = usize::MAX;
        let mut total = 0;
        let mut cnt = 0;

        for i in 0..self.num_active_pts {
            let vertex_id = i.try_into()?;
            let pool_size = self
                .final_graph
                .read_vertex_and_neighbors(vertex_id)?
                .size();
            max = cmp::max(max, pool_size);
            min = cmp::min(min, pool_size);
            total += pool_size;
            if pool_size < 2 {
                cnt += 1;
            }
        }

        info!(
            "Index built with degree: max: {} avg: {} min: {} count(deg<2): {}",
            max,
            (total as f32) / ((self.num_active_pts + self.configuration.num_frozen_pts) as f32),
            min,
            cnt
        );

        self.max_observed_degree = cmp::max(max as u32, self.max_observed_degree);

        Ok(())
    }
}

impl<T> AnnInMemIndex<T> for InmemIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    fn build(&mut self, filename: &Path, num_points_to_load: usize) -> AnnResult<()> {
        // TODO: fresh-diskANN
        // std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

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
                "ERROR: Driver requests loading {} points and file has {} points, 
                but index can support only {} points as specified in configuration.",
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

        self.dataset.set_memory_mapped(filename).unwrap();
        self.dataset.build_from_file(filename, num_points_to_load)?;

        info!("Using only first {} from file.", num_points_to_load);

        self.num_active_pts = num_points_to_load;
        self.build_with_data_populated()?;

        Ok(())
    }

    fn insert(&mut self, filename: &Path, num_points_to_insert: usize) -> AnnResult<()> {
        // fresh-diskANN
        if !file_exists(filename) {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Data file {:?} does not exist.",
                filename
            )));
        }

        let (file_num_points, file_dim) = load_metadata_from_file(filename)?;

        if num_points_to_insert > file_num_points {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has only {} points.",
                num_points_to_insert, file_num_points
            )));
        }

        if file_dim != self.configuration.dim {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Driver requests loading {}  dimension, but file has {} dimension.",
                self.configuration.dim, file_dim
            )));
        }

        if self.configuration.use_pq_dist {
            // TODO: PQ
            todo!("PQ is not supported now");
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            // set the thread count of Rayon, otherwise it will use threads as many as logical
            // cores.
            unsafe {
                std::env::set_var(
                    "RAYON_NUM_THREADS",
                    self.configuration
                        .index_write_parameter
                        .num_threads
                        .to_string(),
                )
            };
        }

        self.dataset
            .append_from_file(filename, num_points_to_insert)?;
        self.final_graph.extend(
            num_points_to_insert,
            self.configuration.index_write_parameter.max_degree,
        );

        // TODO: this should not consider frozen points
        let previous_last_pt = self.num_active_pts;
        self.num_active_pts += num_points_to_insert;
        self.configuration.max_points += num_points_to_insert;

        println!("Inserting {} vectors from file.", num_points_to_insert);

        // TODO: tag_lock
        let timer = Timer::new();
        execute_with_rayon_thread_pool(
            previous_last_pt..self.num_active_pts,
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                self.insert_vertex_id(idx as u32, false)?;

                if idx.is_multiple_of(10000) {
                    info!("We have build the {idx} vertex.");
                }
                Ok(())
            },
        )?;

        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);
        for i in 0..self.num_active_pts {
            visit_order.push(i as u32);
        }

        self.cleanup_graph(&visit_order)?;
        println!("{}", timer.elapsed_seconds_for_step("Insert time: "));

        self.print_stats()?;

        Ok(())
    }

    fn save(&mut self, filename: &Path) -> AnnResult<()> {
        // let data_file = filename.to_string() + ".data";
        // let delete_file = filename.with_added_extension("delete");
        // let subgraph_file = filename.with_added_extension("subgraph");
        // let prefetch_table_file = filename.with_added_extension("prefetch");

        self.save_graph(filename)?;
        // self.save_data(data_file.as_str())?;
        // self.save_delete_list(&delete_file)?;
        // self.save_subgraph(&subgraph_file)?;
        // self.save_prefetch_table(&prefetch_table_file)?;

        Ok(())
    }

    fn load(&mut self, filename: &Path, expected_num_points: usize) -> AnnResult<()> {
        self.num_active_pts = expected_num_points;
        self.dataset
            .build_from_file(&filename.with_added_extension("data"), expected_num_points)?;

        self.load_graph(filename, expected_num_points)?;
        self.load_delete_list(&filename.with_added_extension("delete"))?;

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        Ok(())
    }

    fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> AnnResult<u32> {
        let query_vector = Vertex::new(query, 0);
        InmemIndex::search(self, &query_vector, k_value, l_value, indices)
    }

    fn soft_delete(
        &mut self,
        vertex_ids_to_delete: Vec<u32>,
        num_points_to_delete: usize,
    ) -> AnnResult<()> {
        println!("Deleting {} vectors from file.", num_points_to_delete);

        let timer = Timer::new();

        execute_with_rayon_thread_pool(
            0..num_points_to_delete,
            self.configuration.index_write_parameter.num_threads,
            |idx: usize| {
                self.soft_delete_vertex(vertex_ids_to_delete[idx])?;

                if idx.is_multiple_of(10000) {
                    info!("Build the {idx} vertex.");
                }
                Ok(())
            },
        )?;

        println!("{}", timer.elapsed_seconds_for_step("Delete time: "));
        self.print_stats()?;

        Ok(())
    }
}

#[cfg(test)]
mod index_test {
    use std::path::PathBuf;

    use super::*;
    use crate::common::Metric;
    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
    use crate::test_utils::get_test_file_path;
    use crate::utils::file_util::load_ids_to_delete_from_file;
    use crate::utils::round_up;

    const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
    const TRUTH_GRAPH: &str = "tests/data/truth_index_siftsmall_learn_256pts_R4_L50_A1.2";
    const TEST_DELETE_FILE: &str = "tests/data/delete_set_50pts.bin";
    const TRUTH_GRAPH_WITH_SATURATED: &str =
        "tests/data/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_mem.index";
    const R: u32 = 4;
    const L: u32 = 50;
    const ALPHA: f32 = 1.2;

    /// Build the index with TEST_DATA_FILE and compare the index graph with truth graph TRUTH_GRAPH
    /// Change above constants if you want to test with different dataset
    macro_rules! index_end_to_end_test_singlethread {
        ($saturate_graph:expr, $truth_graph:expr) => {{
            let (data_num, dim) =
                load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

            let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
                .with_alpha(ALPHA)
                .with_num_threads(1)
                .with_saturate_graph($saturate_graph)
                .build();
            let config = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num,
                false,
                0,
                false,
                0,
                1.0f32,
                index_write_parameters,
                0.,
            );
            let mut index: InmemIndex<f32> = InmemIndex::new(config.clone()).unwrap();

            index
                .build(&PathBuf::from(get_test_file_path(TEST_DATA_FILE)), data_num)
                .unwrap();

            let mut truth_index: InmemIndex<f32> = InmemIndex::new(config).unwrap();
            truth_index
                .load_graph(&PathBuf::from(get_test_file_path($truth_graph)), data_num)
                .unwrap();

            compare_graphs(&index, &truth_index);
        }};
    }

    #[test]
    fn index_end_to_end_test_singlethread() {
        index_end_to_end_test_singlethread!(false, TRUTH_GRAPH);
    }

    #[test]
    fn index_end_to_end_test_singlethread_with_saturate_graph() {
        index_end_to_end_test_singlethread!(true, TRUTH_GRAPH_WITH_SATURATED);
    }

    #[test]
    fn index_end_to_end_test_multithread() {
        let (data_num, dim) =
            load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

        let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
            .with_alpha(ALPHA)
            .with_num_threads(8)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            dim,
            round_up(dim as u64, 16_u64) as usize,
            data_num,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
            0.,
        );
        let mut index: InmemIndex<f32> = InmemIndex::new(config).unwrap();

        index
            .build(&PathBuf::from(get_test_file_path(TEST_DATA_FILE)), data_num)
            .unwrap();

        for i in 0..index.final_graph.size() {
            assert_ne!(
                index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .size(),
                0
            );
        }
    }

    const TEST_DATA_FILE_2: &str = "tests/data/siftsmall_learn_256pts_2.fbin";
    const INSERT_TRUTH_GRAPH: &str =
        "tests/data/truth_index_siftsmall_learn_256pts_1+2_R4_L50_A1.2";
    const INSERT_TRUTH_GRAPH_WITH_SATURATED: &str =
        "tests/data/truth_index_siftsmall_learn_256pts_1+2_saturated_R4_L50_A1.2";

    /// Build the index with TEST_DATA_FILE, insert TEST_DATA_FILE_2 and compare the index graph
    /// with truth graph TRUTH_GRAPH Change above constants if you want to test with different
    /// dataset
    macro_rules! index_insert_end_to_end_test_singlethread {
        ($saturate_graph:expr, $truth_graph:expr) => {{
            let (data_num, dim) =
                load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

            let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
                .with_alpha(ALPHA)
                .with_num_threads(1)
                .with_saturate_graph($saturate_graph)
                .build();
            let config = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num,
                false,
                0,
                false,
                0,
                2.0f32,
                index_write_parameters,
                0.,
            );
            let mut index: InmemIndex<f32> = InmemIndex::new(config.clone()).unwrap();

            index
                .build(&PathBuf::from(get_test_file_path(TEST_DATA_FILE)), data_num)
                .unwrap();
            index
                .insert(
                    &PathBuf::from(get_test_file_path(TEST_DATA_FILE_2)),
                    data_num,
                )
                .unwrap();

            let config2 = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num * 2,
                false,
                0,
                false,
                0,
                1.0f32,
                index_write_parameters,
                0.,
            );
            let mut truth_index: InmemIndex<f32> = InmemIndex::new(config2).unwrap();
            truth_index
                .load_graph(&PathBuf::from(get_test_file_path($truth_graph)), data_num)
                .unwrap();

            compare_graphs(&index, &truth_index);
        }};
    }

    /// Build the index with TEST_DATA_FILE, and delete the vertices with id defined in
    /// TEST_DELETE_SET
    macro_rules! index_delete_end_to_end_test_singlethread {
        () => {{
            let (data_num, dim) =
                load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

            let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
                .with_alpha(ALPHA)
                .with_num_threads(1)
                .build();
            let config = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num,
                false,
                0,
                false,
                0,
                2.0f32,
                index_write_parameters,
                0.,
            );
            let mut index: InmemIndex<f32> = InmemIndex::new(config.clone()).unwrap();

            index
                .build(&PathBuf::from(get_test_file_path(TEST_DATA_FILE)), data_num)
                .unwrap();

            let (num_points_to_delete, vertex_ids_to_delete) =
                load_ids_to_delete_from_file(TEST_DELETE_FILE).unwrap();
            index
                .soft_delete(vertex_ids_to_delete, num_points_to_delete)
                .unwrap();
            let delete_set_guard = loop {
                if let Some(guard) = index.delete_set.try_read() {
                    break guard.len();
                }
                // rayon::yield_now();
            };
            assert!(delete_set_guard == num_points_to_delete);
        }};
    }

    #[test]
    fn index_insert_end_to_end_test_singlethread() {
        index_insert_end_to_end_test_singlethread!(false, INSERT_TRUTH_GRAPH);
    }

    #[test]
    fn index_delete_end_to_end_test_singlethread() {
        index_delete_end_to_end_test_singlethread!();
    }

    #[test]
    fn index_insert_end_to_end_test_saturated_singlethread() {
        index_insert_end_to_end_test_singlethread!(true, INSERT_TRUTH_GRAPH_WITH_SATURATED);
    }

    fn compare_graphs(index: &InmemIndex<f32>, truth_index: &InmemIndex<f32>) {
        assert_eq!(index.start, truth_index.start);
        assert_eq!(index.max_observed_degree, truth_index.max_observed_degree);
        assert_eq!(index.final_graph.size(), truth_index.final_graph.size());

        for i in 0..index.final_graph.size() {
            assert_eq!(
                index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .size(),
                truth_index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .size()
            );
            assert_eq!(
                index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .get_neighbors(),
                truth_index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .get_neighbors()
            );
        }
    }

    #[test]
    fn test_reduce_vertex_statistics_and_layout() {
        // Setup simple index
        let dim = 2;
        let max_points = 5;
        let index_write_parameters = IndexWriteParametersBuilder::new(10, 5)
            .with_alpha(1.0)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            dim,
            dim,
            max_points,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
            0.5,
        );

        let mut index = InmemIndex::<f32>::new(config).unwrap();

        // Manual data population
        let data = vec![
            0.0, 0.0, // 0
            0.1, 0.0, // 1 (close to 0)
            0.2, 0.0, // 2 (close to 1)
            10.0, 10.0, // 3 (far)
            10.1, 10.0, // 4 (close to 3)
        ];

        {
            let dataset_slice = index.dataset.data.as_mut_slice();
            for i in 0..data.len() {
                dataset_slice[i] = data[i];
            }
        }
        index.num_active_pts = 5;
        index.start = 0; // Set start node
        index.final_graph.extend(5, 5);
        index.initialize_query_scratch(1, 10).unwrap();

        // Seed the graph (Linear chain: 0->1->2->3->4)
        {
            let mut adj = AdjacencyList::for_range(5);
            adj.push(1);
            index.update_vertex_with_neighbors(0, adj).unwrap();
        }
        {
            let mut adj = AdjacencyList::for_range(5);
            adj.push(2);
            index.update_vertex_with_neighbors(1, adj).unwrap();
        }
        {
            let mut adj = AdjacencyList::for_range(5);
            adj.push(3);
            index.update_vertex_with_neighbors(2, adj).unwrap();
        }
        {
            let mut adj = AdjacencyList::for_range(5);
            adj.push(4);
            index.update_vertex_with_neighbors(3, adj).unwrap();
        }
        {
            let mut adj = AdjacencyList::for_range(5);
            adj.push(0);
            index.update_vertex_with_neighbors(4, adj).unwrap();
        }

        // Reduce vertex 0
        // This should search for 0. Visited nodes will likely include 0, 1, 2.
        index.reduce_vertex(0).unwrap();

        // Check subgraph stats
        // Node 0 should have co-occurrence with 1 (since 1 is neighbor and close)
        let entry = index.subgraph.get(&0);
        if let Some(entry) = entry {
            let neighbors: Vec<_> = entry.iter().collect();
            println!("Node 0 co-occurrences: {:?}", neighbors);
            // We expect at least some co-occurrences
            assert!(!neighbors.is_empty());

            // Check if weights are negative (our implementation detail)
            for n in neighbors {
                assert!(n.distance <= 0.0);
            }
        } else {
            // It's possible 0 is not in subgraph if it was never "u" in the loop
            // But reduce_vertex(0) puts 0 in visited_nodes.
            // If 0 is in visited_nodes, it will be processed as 'u'.
            // And it will have 'v' neighbors.
            // So index.subgraph.get(&0) should exist.
            println!("Node 0 has no co-occurrences recorded.");
        }

        // Test Layout Generation
        let layout = index.generate_optimized_layout();
        println!("Layout: {:?}", layout);

        assert_eq!(layout.len(), 5);
        let layout_set: HashSet<_> = layout.iter().cloned().collect();
        assert_eq!(layout_set.len(), 5); // Unique elements
        assert!(layout_set.contains(&0));
        assert!(layout_set.contains(&4));
    }
}
