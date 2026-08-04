// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#![warn(missing_debug_implementations, missing_docs)]

//! Search algorithm for index construction and query

use std::ops::{Mul, Sub};

use hashbrown::hash_set::Entry::*;

use crate::common::{AnnError, AnnResult};
use crate::index::InmemIndex;
use crate::model::scratch::InMemQueryScratch;
use crate::model::{Neighbor, Vertex};

impl<T> InmemIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    /// Search for query using given L value, for benchmarking purposes
    /// # Arguments
    /// * `query` - query vertex
    /// * `scratch` - in-memory query scratch
    /// * `search_list_size` - search list size to use for the benchmark
    pub fn search_with_l_override(
        &self,
        query: &Vertex<T>,
        scratch: &mut InMemQueryScratch<T>,
        search_list_size: usize,
    ) -> AnnResult<u32> {
        let init_ids = self.get_init_ids()?;
        self.init_graph_for_point(query, init_ids, scratch)?;
        // Scratch is created using largest L val from search_memory_index, so we artifically make
        // it smaller here This allows us to use the same scratch for all L values without
        // having to rebuild the query scratch
        scratch.best_candidates.set_capacity(search_list_size);
        let (_, cmp) = self.greedy_search(query, scratch)?;

        Ok(cmp)
    }

    /// search for point
    /// # Arguments
    /// * `query` - query vertex
    /// * `scratch` - in-memory query scratch
    pub fn search_for_point(
        &self,
        query: &Vertex<T>,
        scratch: &mut InMemQueryScratch<T>,
    ) -> AnnResult<Vec<Neighbor>> {
        let init_ids = self.get_init_ids()?;
        self.init_graph_for_point(query, init_ids, scratch)?;
        let (mut visited_nodes, _) = self.greedy_search(query, scratch)?;

        visited_nodes.retain(|&element| element.id != query.vertex_id());
        Ok(visited_nodes)
    }

    /// Returns the locations of start point and frozen points suitable for use with
    /// iterate_to_fixed_point.
    fn get_init_ids(&self) -> AnnResult<Vec<u32>> {
        let mut init_ids = Vec::with_capacity(1 + self.configuration.num_frozen_pts);
        init_ids.push(self.start);

        for frozen in self.configuration.max_points
            ..(self.configuration.max_points + self.configuration.num_frozen_pts)
        {
            let frozen_u32 = frozen.try_into()?;
            if frozen_u32 != self.start {
                init_ids.push(frozen_u32);
            }
        }

        Ok(init_ids)
    }

    /// Initialize graph for point
    /// # Arguments
    /// * `query` - query vertex
    /// * `init_ids` - initial nodes from which search starts
    /// * `scratch` - in-memory query scratch
    /// * `search_list_size_override` - override for search list size in index config
    fn init_graph_for_point(
        &self,
        query: &Vertex<T>,
        init_ids: Vec<u32>,
        scratch: &mut InMemQueryScratch<T>,
    ) -> AnnResult<()> {
        scratch
            .best_candidates
            .reserve(self.configuration.index_write_parameter.search_list_size as usize);
        scratch.query.memcpy(query.vector())?;

        if !scratch.id_scratch.is_empty() {
            return Err(AnnError::log_index_error(
                "id_scratch is not empty.".to_string(),
            ));
        }

        let query_vertex =
            Vertex::<T>::try_from((&scratch.query[..], query.vertex_id())).map_err(|err| {
                AnnError::log_index_error(format!(
                    "TryFromSliceError: failed to get Vertex for query, err={}",
                    err
                ))
            })?;

        for id in init_ids {
            if (id as usize) >= self.configuration.max_points + self.configuration.num_frozen_pts {
                return Err(AnnError::log_index_error(format!(
                    "vertex_id {} is out of valid range of points {}",
                    id,
                    self.configuration.max_points + self.configuration.num_frozen_pts
                )));
            }

            if let Vacant(entry) = scratch.node_visited_robinset.entry(id) {
                entry.insert();

                let vertex = self.dataset.get_vertex(id)?;

                let distance = vertex.compare(&query_vertex, self.configuration.dist_metric);
                let neighbor = Neighbor::new(id, distance);
                scratch.best_candidates.insert(neighbor);
            }
        }

        Ok(())
    }

    /// GreedySearch against query node
    /// Returns visited nodes
    /// # Arguments
    /// * `query` - query vertex
    /// * `scratch` - in-memory query scratch
    fn greedy_search(
        &self,
        query: &Vertex<T>,
        scratch: &mut InMemQueryScratch<T>,
    ) -> AnnResult<(Vec<Neighbor>, u32)> {
        // let vertex_id = query.vertex_id();
        let mut visited_nodes =
            Vec::with_capacity((3 * scratch.candidate_size + scratch.max_degree) as usize);

        // TODO: uncomment hops?
        // let mut hops: u32 = 0;
        let mut cmps: u32 = 0;

        let query_vertexes = vec![
            Vertex::<T>::try_from((&scratch.query[..], query.vertex_id())).map_err(|err| {
                AnnError::log_index_error(format!(
                    "TryFromSliceError: failed to get Vertex for query, err={}",
                    err
                ))
            })?,
        ];
        // loop {
        //     match self.subgraph.try_get(&vertex_id) {
        //         TryResult::Present(subgraph) => {
        //             for item in subgraph.value() {
        //                 query_vertexes.push(Vertex::<T>::try_from((&scratch.query[..],
        // item.id)).unwrap());             }
        //             break;
        //         }
        //         TryResult::Absent => break,
        //         TryResult::Locked => sleep(Duration::from_millis(1)),
        //     }
        // }

        while scratch.best_candidates.has_notvisited_node() {
            let closest_node = scratch.best_candidates.closest_notvisited();

            // Add node to visited nodes to create pool for prune later
            // TODO: search_invocation and use_filter
            visited_nodes.push(closest_node);

            // Find which of the nodes in des have not been visited before
            scratch.id_scratch.clear();

            let max_vertex_id = self.configuration.max_points + self.configuration.num_frozen_pts;

            let adj = self
                .final_graph
                .read_vertex_and_neighbors(closest_node.id)?
                .get_neighbors()
                .edges
                .clone();
            for id in adj {
                let current_vertex_id = id;
                debug_assert!(
                    (current_vertex_id as usize) < max_vertex_id,
                    "current_vertex_id {} is out of valid range of points {}",
                    current_vertex_id,
                    max_vertex_id
                );
                if current_vertex_id as usize >= max_vertex_id {
                    continue;
                }

                // quickly de-dup. Remember, we are in a read lock
                // we want to exit out of it quickly
                if scratch.node_visited_robinset.insert(current_vertex_id) {
                    scratch.id_scratch.push(current_vertex_id);
                }
            }

            let len = scratch.id_scratch.len();
            for (m, &id) in scratch.id_scratch.iter().enumerate() {
                if m + 1 < len {
                    let next_node = unsafe { *scratch.id_scratch.get_unchecked(m + 1) };
                    self.dataset.prefetch_vector(next_node);
                }

                let vertex = self.dataset.get_vertex(id)?;
                let mut distance = f32::MAX;
                for query in &query_vertexes {
                    distance = distance.min(query.compare(&vertex, self.configuration.dist_metric));
                }

                // Insert <id, dist> pairs into the pool of candidates
                scratch.best_candidates.insert(Neighbor::new(id, distance));
            }

            cmps += len as u32;
        }

        Ok((visited_nodes, cmps))
    }
}

#[cfg(test)]
mod search_test {
    use super::*;
    use crate::common::Metric;
    use crate::model::IndexConfiguration;
    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
    use crate::model::graph::AdjacencyList;
    use crate::test_utils::inmem_index_initialization::create_index_with_test_data;

    #[test]
    fn get_init_ids_no_forzen_pts() {
        let index_write_parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            256,
            256,
            256,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
            0.,
        );

        let index = InmemIndex::<f32>::new(config).unwrap();
        let init_ids = index.get_init_ids().unwrap();
        assert_eq!(init_ids.len(), 1);
        assert_eq!(init_ids[0], 256);
    }

    #[test]
    fn get_init_ids_with_forzen_pts() {
        let index_write_parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            256,
            256,
            256,
            false,
            0,
            false,
            2,
            1f32,
            index_write_parameters,
            0.,
        );

        let index = InmemIndex::<f32>::new(config).unwrap();
        let init_ids = index.get_init_ids().unwrap();
        assert_eq!(init_ids.len(), 2);
        assert_eq!(init_ids[0], 256);
        assert_eq!(init_ids[1], 257);
    }

    #[test]
    fn search_for_point_initial_call() {
        let index = create_index_with_test_data();
        let query = index.dataset.get_vertex(0).unwrap();

        let mut scratch = InMemQueryScratch::new(
            index.configuration.index_write_parameter.search_list_size,
            &index.configuration.index_write_parameter,
            128,
        )
        .unwrap();
        let visited_nodes = index.search_for_point(&query, &mut scratch).unwrap();
        assert_eq!(visited_nodes.len(), 1);
        assert_eq!(scratch.best_candidates.size(), 1);
        assert_eq!(scratch.best_candidates[0].id, 72);
        assert_eq!(scratch.best_candidates[0].distance, 125678.0_f32);
        assert!(scratch.best_candidates[0].visited);
    }

    fn set_neighbors(index: &InmemIndex<f32>, vertex_id: u32, neighbors: Vec<u32>) {
        index
            .final_graph
            .write_vertex_and_neighbors(vertex_id)
            .unwrap()
            .set_neighbors(AdjacencyList::from(neighbors));
    }
    #[test]
    fn search_for_point_works_with_edges() {
        let index = create_index_with_test_data();
        let query = index.dataset.get_vertex(14).unwrap();

        set_neighbors(&index, 0, vec![12, 72, 5, 9]);
        set_neighbors(&index, 1, vec![2, 12, 10, 4]);
        set_neighbors(&index, 2, vec![1, 72, 9]);
        set_neighbors(&index, 3, vec![13, 6, 5, 11]);
        set_neighbors(&index, 4, vec![1, 3, 7, 9]);
        set_neighbors(&index, 5, vec![3, 0, 8, 11, 13]);
        set_neighbors(&index, 6, vec![3, 72, 7, 10, 13]);
        set_neighbors(&index, 7, vec![72, 4, 6]);
        set_neighbors(&index, 8, vec![72, 5, 9, 12]);
        set_neighbors(&index, 9, vec![8, 4, 0, 2]);
        set_neighbors(&index, 10, vec![72, 1, 9, 6]);
        set_neighbors(&index, 11, vec![3, 0, 5]);
        set_neighbors(&index, 12, vec![1, 0, 8, 9]);
        set_neighbors(&index, 13, vec![3, 72, 5, 6]);
        set_neighbors(&index, 72, vec![7, 2, 10, 8, 13]);

        let mut scratch = InMemQueryScratch::new(
            index.configuration.index_write_parameter.search_list_size,
            &index.configuration.index_write_parameter,
            128,
        )
        .unwrap();
        let visited_nodes = index.search_for_point(&query, &mut scratch).unwrap();
        assert_eq!(visited_nodes.len(), 15);
        assert_eq!(scratch.best_candidates.size(), 15);
        assert_eq!(scratch.best_candidates[0].id, 2);
        assert_eq!(scratch.best_candidates[0].distance, 120899.0_f32);
        assert_eq!(scratch.best_candidates[1].id, 8);
        assert_eq!(scratch.best_candidates[1].distance, 145538.0_f32);
        assert_eq!(scratch.best_candidates[2].id, 72);
        assert_eq!(scratch.best_candidates[2].distance, 146046.0_f32);
        assert_eq!(scratch.best_candidates[3].id, 4);
        assert_eq!(scratch.best_candidates[3].distance, 148462.0_f32);
        assert_eq!(scratch.best_candidates[4].id, 7);
        assert_eq!(scratch.best_candidates[4].distance, 148912.0_f32);
        assert_eq!(scratch.best_candidates[5].id, 10);
        assert_eq!(scratch.best_candidates[5].distance, 154570.0_f32);
        assert_eq!(scratch.best_candidates[6].id, 1);
        assert_eq!(scratch.best_candidates[6].distance, 159448.0_f32);
        assert_eq!(scratch.best_candidates[7].id, 12);
        assert_eq!(scratch.best_candidates[7].distance, 170698.0_f32);
        assert_eq!(scratch.best_candidates[8].id, 9);
        assert_eq!(scratch.best_candidates[8].distance, 177205.0_f32);
        assert_eq!(scratch.best_candidates[9].id, 0);
        assert_eq!(scratch.best_candidates[9].distance, 259996.0_f32);
        assert_eq!(scratch.best_candidates[10].id, 6);
        assert_eq!(scratch.best_candidates[10].distance, 371819.0_f32);
        assert_eq!(scratch.best_candidates[11].id, 5);
        assert_eq!(scratch.best_candidates[11].distance, 385240.0_f32);
        assert_eq!(scratch.best_candidates[12].id, 3);
        assert_eq!(scratch.best_candidates[12].distance, 413899.0_f32);
        assert_eq!(scratch.best_candidates[13].id, 13);
        assert_eq!(scratch.best_candidates[13].distance, 416386.0_f32);
        assert_eq!(scratch.best_candidates[14].id, 11);
        assert_eq!(scratch.best_candidates[14].distance, 449266.0_f32);
    }
}
