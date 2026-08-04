#![warn(missing_docs)]

//! ANN in-memory index abstraction

use std::ops::{Mul, Sub};
use std::path::Path;

use super::InmemIndex;
use crate::common::AnnResult;
use crate::model::IndexConfiguration;

/// ANN inmem-index abstraction for custom <T, N>
pub trait AnnInMemIndex<T>: Sync + Send
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    /// Build index
    fn build(&mut self, filename: &Path, num_points_to_load: usize) -> AnnResult<()>;

    /// Save index
    fn save(&mut self, filename: &Path) -> AnnResult<()>;

    /// Load index
    fn load(&mut self, filename: &Path, expected_num_points: usize) -> AnnResult<()>;

    /// insert index
    fn insert(&mut self, filename: &Path, num_points_to_insert: usize) -> AnnResult<()>;

    /// Search the index for K nearest neighbors of query using given L value, for benchmarking
    /// purposes
    fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> AnnResult<u32>;

    /// Soft deletes the nodes with the ids in the given array.
    fn soft_delete(
        &mut self,
        vertex_ids_to_delete: Vec<u32>,
        num_points_to_delete: usize,
    ) -> AnnResult<()>;
}

/// Create Index<T, N> based on configuration
pub fn create_inmem_index<'a, T>(
    config: IndexConfiguration,
) -> AnnResult<Box<dyn AnnInMemIndex<T> + 'a>>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'a,
    T: Sub<Output = T> + Mul<Output = T>,
{
    let index = Box::new(InmemIndex::<T>::new(config)?);
    Ok(index as Box<dyn AnnInMemIndex<T>>)
}

#[cfg(test)]
mod dataset_test {
    use std::path::PathBuf;

    use super::*;
    use crate::common::Metric;
    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;

    #[test]
    #[should_panic(expected = "ERROR: Data file fake_file does not exist.")]
    fn create_index_test() {
        let index_write_parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .with_saturate_graph(false)
            .with_num_threads(1)
            .build();

        let config = IndexConfiguration::new(
            Metric::L2,
            128,
            256,
            1_000_000,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
            0.,
        );
        let mut index = create_inmem_index::<f32>(config).unwrap();
        index.build(&PathBuf::from("fake_file"), 100).unwrap();
    }
}
