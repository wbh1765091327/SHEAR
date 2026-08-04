use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::{Mul, Sub};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use hashbrown::HashSet;
use tracing::info;

use super::InmemIndex;
use crate::common::{AnnError, AnnResult};
use crate::model::InMemoryGraph;
use crate::model::graph::AdjacencyList;
use crate::utils::{file_exists, save_data_in_base_dimensions};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphRecordEncoding {
    SequentialDegreeOnly,
    ExplicitVertexIds,
}

fn detect_graph_record_encoding(
    reader: &mut BufReader<File>,
    payload_bytes: usize,
    expected_num_points: usize,
) -> AnnResult<GraphRecordEncoding> {
    let mut payload = vec![0u8; payload_bytes];
    reader.read_exact(&mut payload)?;
    let words: Vec<u32> = payload
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    if graph_words_fit_sequential(&words, expected_num_points) {
        return Ok(GraphRecordEncoding::SequentialDegreeOnly);
    }
    if graph_words_fit_explicit_ids(&words, expected_num_points) {
        return Ok(GraphRecordEncoding::ExplicitVertexIds);
    }

    Err(AnnError::log_index_error(
        "ERROR: Unable to determine graph record encoding".to_string(),
    ))
}

fn graph_words_fit_sequential(words: &[u32], expected_num_points: usize) -> bool {
    let mut offset = 0usize;
    let mut points = 0usize;
    while offset < words.len() {
        let degree = words[offset] as usize;
        if degree == 0 || offset + 1 + degree > words.len() {
            return false;
        }
        offset += 1 + degree;
        points += 1;
    }

    offset == words.len() && points == expected_num_points
}

fn graph_words_fit_explicit_ids(words: &[u32], expected_num_points: usize) -> bool {
    let mut offset = 0usize;
    let mut seen = HashSet::new();
    while offset < words.len() {
        if offset + 1 >= words.len() {
            return false;
        }
        let uid = words[offset];
        let degree = words[offset + 1] as usize;
        if uid as usize >= expected_num_points || degree == 0 || !seen.insert(uid) {
            return false;
        }
        offset += 2 + degree;
        if offset > words.len() {
            return false;
        }
    }

    offset == words.len() && seen.len() == expected_num_points
}

fn load_graph_records_sequential(
    reader: &mut BufReader<File>,
    payload_bytes: usize,
    expected_num_points: usize,
    final_graph: &InMemoryGraph,
) -> AnnResult<(u32, usize, u32)> {
    let mut bytes_read = 0usize;
    let mut num_edges = 0u32;
    let mut nodes_read = 0usize;
    let mut max_observed_degree = 0u32;

    while bytes_read < payload_bytes {
        let num_nbrs = reader.read_u32::<LittleEndian>()?;
        if num_nbrs == 0 {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Point found with no out-neighbors, point# {}",
                nodes_read
            )));
        }
        if nodes_read >= expected_num_points {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Graph contains more than {} points",
                expected_num_points
            )));
        }

        let mut tmp: Vec<u32> = Vec::with_capacity(num_nbrs as usize);
        for _ in 0..num_nbrs {
            tmp.push(reader.read_u32::<LittleEndian>()?);
        }
        final_graph
            .write_vertex_and_neighbors(nodes_read as u32)?
            .set_neighbors(AdjacencyList::from(tmp));

        max_observed_degree = max_observed_degree.max(num_nbrs);
        num_edges += num_nbrs;
        nodes_read += 1;
        bytes_read += 4 * (num_nbrs as usize + 1);
    }

    if nodes_read != expected_num_points {
        return Err(AnnError::log_index_error(format!(
            "ERROR: Graph contains {} points, expected {}",
            nodes_read, expected_num_points
        )));
    }

    Ok((num_edges, nodes_read, max_observed_degree))
}

fn load_graph_records_with_vertex_ids(
    reader: &mut BufReader<File>,
    payload_bytes: usize,
    expected_num_points: usize,
    final_graph: &InMemoryGraph,
) -> AnnResult<(u32, usize, u32)> {
    let mut bytes_read = 0usize;
    let mut num_edges = 0u32;
    let mut nodes_read = 0usize;
    let mut max_observed_degree = 0u32;
    let mut seen = HashSet::new();

    while bytes_read < payload_bytes {
        let uid = reader.read_u32::<LittleEndian>()?;
        if uid as usize >= expected_num_points {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Graph record uid {} exceeds expected point count {}",
                uid, expected_num_points
            )));
        }
        if !seen.insert(uid) {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Duplicate graph record for point {}",
                uid
            )));
        }

        let num_nbrs = reader.read_u32::<LittleEndian>()?;
        if num_nbrs == 0 {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Point found with no out-neighbors, point# {}",
                uid
            )));
        }

        let mut tmp: Vec<u32> = Vec::with_capacity(num_nbrs as usize);
        for _ in 0..num_nbrs {
            tmp.push(reader.read_u32::<LittleEndian>()?);
        }

        final_graph
            .write_vertex_and_neighbors(uid)?
            .set_neighbors(AdjacencyList::from(tmp));

        max_observed_degree = max_observed_degree.max(num_nbrs);
        num_edges += num_nbrs;
        nodes_read += 1;
        bytes_read += 4 * (num_nbrs as usize + 2);
    }

    if nodes_read != expected_num_points {
        return Err(AnnError::log_index_error(format!(
            "ERROR: Graph contains {} points, expected {}",
            nodes_read, expected_num_points
        )));
    }

    Ok((num_edges, nodes_read, max_observed_degree))
}

impl<T> InmemIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    T: Sub<Output = T> + Mul<Output = T>,
{
    pub fn load_graph(&mut self, filename: &Path, expected_num_points: usize) -> AnnResult<usize> {
        // let file_offset = 0; // will need this for single file format support

        let mut in_file = BufReader::new(File::open(Path::new(filename))?);
        // in_file.seek(SeekFrom::Start(file_offset as u64))?;

        let expected_file_size: usize = in_file.read_u64::<LittleEndian>()? as usize;
        self.max_observed_degree = in_file.read_u32::<LittleEndian>()?;
        self.start = in_file.read_u32::<LittleEndian>()?;
        let file_frozen_pts: usize = in_file.read_u64::<LittleEndian>()? as usize;

        let vamana_metadata_size = 24;

        println!(
            "From graph header, expected_file_size: {}, max_observed_degree: {}, start: {}, file_frozen_pts: {}",
            expected_file_size, self.max_observed_degree, self.start, file_frozen_pts
        );

        if file_frozen_pts != self.configuration.num_frozen_pts {
            if file_frozen_pts == 1 {
                return Err(AnnError::log_index_config_error(
                    "num_frozen_pts".to_string(), 
                    "ERROR: When loading index, detected dynamic index, but constructor asks for static index. Exitting.".to_string())
                );
            } else {
                return Err(AnnError::log_index_config_error(
                    "num_frozen_pts".to_string(), 
                    "ERROR: When loading index, detected static index, but constructor asks for dynamic index. Exitting.".to_string())
                );
            }
        }

        info!("Loading vamana graph {:?}...", filename);

        let expected_max_points = expected_num_points - file_frozen_pts;

        // If user provides more points than max_points
        // resize the _final_graph to the larger size.
        if self.configuration.max_points < expected_max_points {
            info!(
                "Number of points in data: {} is greater than max_points: {} Setting max points to: {}",
                expected_max_points, self.configuration.max_points, expected_max_points
            );

            self.configuration.max_points = expected_max_points;
            self.final_graph = InMemoryGraph::new(
                self.configuration.max_points + self.configuration.num_frozen_pts,
                self.configuration.index_write_parameter.max_degree,
            );
        }

        let payload_bytes = expected_file_size
            .checked_sub(vamana_metadata_size)
            .ok_or_else(|| {
                AnnError::log_index_error("ERROR: Graph header size overflow".to_string())
            })?;
        if !payload_bytes.is_multiple_of(std::mem::size_of::<u32>()) {
            return Err(AnnError::log_index_error(format!(
                "ERROR: Graph payload size {} is not aligned to u32 records",
                payload_bytes
            )));
        }

        let encoding =
            detect_graph_record_encoding(&mut in_file, payload_bytes, expected_num_points)?;
        in_file.seek(SeekFrom::Start(vamana_metadata_size as u64))?;

        let (num_edges, nodes_read, max_observed_degree) = match encoding {
            GraphRecordEncoding::SequentialDegreeOnly => load_graph_records_sequential(
                &mut in_file,
                payload_bytes,
                expected_num_points,
                &self.final_graph,
            )?,
            GraphRecordEncoding::ExplicitVertexIds => load_graph_records_with_vertex_ids(
                &mut in_file,
                payload_bytes,
                expected_num_points,
                &self.final_graph,
            )?,
        };

        println!(
            "Done. Index has {} nodes and {} out-edges, _start is set to {}",
            nodes_read, num_edges, self.start
        );

        self.max_observed_degree = max_observed_degree;
        Ok(nodes_read as usize)
    }

    /// Save the graph index on a file as an adjacency list.
    /// For each point, store the number of neighbors followed by the neighbor ids.
    pub fn save_graph(&mut self, graph_file: &Path) -> AnnResult<u64> {
        let file: File = File::create(graph_file)?;
        let mut out = BufWriter::new(file);

        let file_offset: u64 = 0;
        out.seek(SeekFrom::Start(file_offset))?;
        let mut index_size: u64 = 24;
        let mut max_degree: u32 = 0;
        out.write_all(&index_size.to_le_bytes())?;
        out.write_all(&self.max_observed_degree.to_le_bytes())?;
        out.write_all(&self.start.to_le_bytes())?;
        out.write_all(&(self.configuration.num_frozen_pts as u64).to_le_bytes())?;

        for i in 0..self.num_active_pts + self.configuration.num_frozen_pts {
            let uid = i as u32;
            let neighbors_list = self.final_graph.read_vertex_and_neighbors(uid)?;
            let gk = neighbors_list.size() as u32;
            out.write_all(&gk.to_le_bytes())?;

            for neighbor in neighbors_list.get_neighbors().iter() {
                out.write_all(&neighbor.to_le_bytes())?;
            }

            if gk > max_degree {
                max_degree = gk;
            }
            index_size += (std::mem::size_of::<u32>() * (gk as usize + 1)) as u64;
        }

        info!("Max observed degree: {:}", max_degree);
        out.seek(SeekFrom::Start(file_offset))?;
        out.write_all(&index_size.to_le_bytes())?;
        out.write_all(&max_degree.to_le_bytes())?;
        out.flush()?;
        Ok(index_size)
    }

    /// Save the data on a file.
    pub fn save_data(&mut self, data_file: &str) -> AnnResult<usize> {
        // Note: at this point, either _nd == _max_points or any frozen points have
        // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
        // location limit.
        Ok(save_data_in_base_dimensions(
            data_file,
            &mut self.dataset.data,
            self.num_active_pts + self.configuration.num_frozen_pts,
            self.configuration.dim,
            self.configuration.aligned_dim,
            0,
        )?)
    }

    /// Save the delete list to a file only if the delete list length is not zero.
    pub fn save_delete_list(&mut self, delete_list_file: &Path) -> AnnResult<usize> {
        let mut delete_file_size = 0;
        let delete_set = self.delete_set.read();
        let delete_set_len = delete_set.len() as u32;

        if delete_set_len != 0 {
            let file: File = File::create(delete_list_file)?;
            let mut writer = BufWriter::new(file);

            // Write the length of the set.
            writer.write_all(&delete_set_len.to_le_bytes())?;
            delete_file_size += std::mem::size_of::<u32>();

            // Write the elements of the set.
            for &item in delete_set.iter() {
                writer.write_all(&item.to_be_bytes())?;
                delete_file_size += std::mem::size_of::<u32>();
            }

            writer.flush()?;
        }

        Ok(delete_file_size)
    }

    /// Save the subgraph to a file
    pub fn save_subgraph(&mut self, subgraph_file: &Path) -> AnnResult<()> {
        let file: File = File::create(subgraph_file)?;
        let mut out = BufWriter::new(file);

        let file_offset: u64 = 0;
        out.seek(SeekFrom::Start(file_offset))?;

        // Write the size of the subgraph
        let graph_sz = self.subgraph.len() as u64;
        out.write_all(&graph_sz.to_le_bytes())?;

        // Write the start point
        out.write_all(&self.start.to_le_bytes())?;

        // Magic number
        let magic: u32 = 0x3f3f3f3f;

        let mut subgraph_num = 0;

        info!("Subgraph node size: {}", self.subgraph.len());
        // Iterate through the subgraph store
        for entry in self.subgraph.iter() {
            let uid = *entry.key();
            let que = entry.value();
            let sz = que.len() as u32;
            subgraph_num += sz;

            if sz > 0 {
                // Write the uid and size
                out.write_all(&uid.to_le_bytes())?;
                out.write_all(&sz.to_le_bytes())?;

                // Create a vector of neighbor ids
                let mut n_array: Vec<u32> = Vec::with_capacity(sz as usize);
                for neighbor in que.iter() {
                    n_array.push(neighbor.id);
                }

                // Write the neighbor ids
                for &id in n_array.iter() {
                    out.write_all(&id.to_le_bytes())?;
                }
            } else {
                // Write magic number twice if size is 0
                out.write_all(&magic.to_le_bytes())?;
                out.write_all(&magic.to_le_bytes())?;
            }
        }

        info!("Subgraph num: {subgraph_num}");

        out.flush()?;

        // Create metadata file for subgraph
        let metadata_file = subgraph_file.with_extension("metadata");
        let metadata_file: File = File::create(&metadata_file)?;
        let mut metadata_out = BufWriter::new(metadata_file);

        // Write the size of the subgraph
        metadata_out.write_all(&graph_sz.to_le_bytes())?;

        // Write metadata for each subgraph
        for entry in self.subgraph.iter() {
            let uid = *entry.key();
            let que = entry.value();
            let query = self.dataset.get_vertex(uid)?;

            // Find the maximum distance among neighbors
            let max_distance = que
                .iter()
                .map(|neighbor| {
                    query.compare(
                        &self.dataset.get_vertex(neighbor.id).unwrap(),
                        self.configuration.dist_metric,
                    )
                })
                .fold(0.0_f32, |a, b| a.max(b));

            // Write the uid and max distance
            metadata_out.write_all(&uid.to_le_bytes())?;
            metadata_out.write_all(&max_distance.to_le_bytes())?;
        }

        metadata_out.flush()?;

        Ok(())
    }

    /// Save the prefetch table to a file
    pub fn save_prefetch_table(&mut self, prefetch_file: &Path) -> AnnResult<()> {
        let file: File = File::create(prefetch_file)?;
        let mut out = BufWriter::new(file);

        let file_offset: u64 = 0;
        out.seek(SeekFrom::Start(file_offset))?;

        // Write the size of the prefetch table
        let table_sz = self.prefetch_table.len() as u64;
        out.write_all(&table_sz.to_le_bytes())?;

        let mut prefetch_num = 0;

        info!("Prefetch table size: {}", self.prefetch_table.len());
        // Iterate through the prefetch table
        for entry in self.prefetch_table.iter() {
            let uid = *entry.key();
            let prefetch_list = entry.value();
            let sz = prefetch_list.len() as u32;
            prefetch_num += sz;

            // Write the uid and size
            out.write_all(&uid.to_le_bytes())?;
            out.write_all(&sz.to_le_bytes())?;

            // Write the prefetch ids
            for &id in prefetch_list.iter() {
                out.write_all(&id.to_le_bytes())?;
            }
        }

        info!("Prefetch table num: {prefetch_num}");

        out.flush()?;

        Ok(())
    }

    // load the deleted list from the delete file if it exists.
    pub fn load_delete_list(&mut self, delete_list_file: &Path) -> AnnResult<usize> {
        let mut len = 0;

        if file_exists(delete_list_file) {
            let file = File::open(delete_list_file)?;
            let mut reader = BufReader::new(file);

            len = reader.read_u32::<LittleEndian>()? as usize;

            let mut delete_set = self.delete_set.write();
            for _ in 0..len {
                let item = reader.read_u32::<LittleEndian>()?;
                delete_set.insert(item);
            }
        }

        Ok(len)
    }
}

#[cfg(test)]
mod index_test {
    use std::fs;
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::*;
    use crate::common::Metric;
    use crate::index::AnnInMemIndex;
    use crate::model::IndexConfiguration;
    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
    use crate::utils::{load_metadata_from_file, round_up};

    const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
    const R: u32 = 4;
    const L: u32 = 50;
    const ALPHA: f32 = 1.2;

    #[test]
    fn save_graph_test() {
        let parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            10,
            16,
            16,
            false,
            0,
            false,
            8,
            1f32,
            parameters,
            0.,
        );
        let mut index = InmemIndex::<f32>::new(config).unwrap();
        let final_graph = InMemoryGraph::new(10, 3);
        let num_active_pts = 2_usize;
        index.final_graph = final_graph;
        index.num_active_pts = num_active_pts;
        let graph_file = "test_save_graph_data.bin";
        let result = index.save_graph(&PathBuf::from(graph_file));
        assert!(result.is_ok());

        fs::remove_file(graph_file).expect("Failed to delete file");
    }

    #[test]
    fn save_graph_round_trips_vertex_ids() {
        let parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            4,
            16,
            6,
            false,
            0,
            false,
            0,
            1f32,
            parameters,
            0.,
        );
        let mut writer = InmemIndex::<f32>::new(config.clone()).unwrap();
        writer.num_active_pts = 6;
        writer
            .final_graph
            .write_vertex_and_neighbors(0)
            .unwrap()
            .set_neighbors(AdjacencyList::from(vec![4, 1]));
        writer
            .final_graph
            .write_vertex_and_neighbors(1)
            .unwrap()
            .set_neighbors(AdjacencyList::from(vec![3]));
        writer
            .final_graph
            .write_vertex_and_neighbors(2)
            .unwrap()
            .set_neighbors(AdjacencyList::from(vec![5, 0, 1]));
        writer
            .final_graph
            .write_vertex_and_neighbors(3)
            .unwrap()
            .set_neighbors(AdjacencyList::from(vec![2]));
        writer
            .final_graph
            .write_vertex_and_neighbors(4)
            .unwrap()
            .set_neighbors(AdjacencyList::from(vec![1, 2]));
        writer
            .final_graph
            .write_vertex_and_neighbors(5)
            .unwrap()
            .set_neighbors(AdjacencyList::from(vec![2]));

        let dir = tempdir().unwrap();
        let graph_file = dir.path().join("roundtrip_mem.index");
        writer.save_graph(&graph_file).unwrap();

        let mut reader = InmemIndex::<f32>::new(config).unwrap();
        let loaded = reader.load_graph(&graph_file, 6).unwrap();

        assert_eq!(loaded, 6);
        assert_eq!(
            reader
                .final_graph
                .read_vertex_and_neighbors(0)
                .unwrap()
                .get_neighbors(),
            &AdjacencyList::from(vec![4, 1])
        );
        assert_eq!(
            reader
                .final_graph
                .read_vertex_and_neighbors(1)
                .unwrap()
                .get_neighbors(),
            &AdjacencyList::from(vec![3])
        );
        assert_eq!(
            reader
                .final_graph
                .read_vertex_and_neighbors(2)
                .unwrap()
                .get_neighbors(),
            &AdjacencyList::from(vec![5, 0, 1])
        );
        assert_eq!(
            reader
                .final_graph
                .read_vertex_and_neighbors(3)
                .unwrap()
                .get_neighbors(),
            &AdjacencyList::from(vec![2])
        );
        assert_eq!(
            reader
                .final_graph
                .read_vertex_and_neighbors(4)
                .unwrap()
                .get_neighbors(),
            &AdjacencyList::from(vec![1, 2])
        );
        assert_eq!(
            reader
                .final_graph
                .read_vertex_and_neighbors(5)
                .unwrap()
                .get_neighbors(),
            &AdjacencyList::from(vec![2])
        );
    }

    #[test]
    fn save_data_test() {
        let (data_num, dim) = load_metadata_from_file(TEST_DATA_FILE).unwrap();

        let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
            .with_alpha(ALPHA)
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
            .build(&PathBuf::from(TEST_DATA_FILE), data_num)
            .unwrap();

        let data_file = "test.data";
        let result = index.save_data(data_file);
        assert_eq!(
            result.unwrap(),
            2 * std::mem::size_of::<u32>()
                + (index.num_active_pts + index.configuration.num_frozen_pts)
                    * index.configuration.dim
                    * (std::mem::size_of::<f32>())
        );
        fs::remove_file(data_file).expect("Failed to delete file");
    }
}
