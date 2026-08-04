use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::common::AnnResult;

#[inline]
fn write_u32_slice_le<W: Write>(writer: &mut W, values: &[u32]) -> std::io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        writer.write_all(bytemuck::cast_slice(values))
    }
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)
    }
}

#[derive(Debug)]
pub struct DirectMemGraphWriter {
    writer: BufWriter<File>,
    index_size: u64,
    max_degree: u32,
    start: u32,
    num_frozen_pts: u64,
}

impl DirectMemGraphWriter {
    pub const BUFFER_BYTES: usize = 8 * 1024 * 1024;

    pub fn create(path: &Path, start: u32, num_frozen_pts: usize) -> AnnResult<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::with_capacity(Self::BUFFER_BYTES, file);
        writer.write_all(&24u64.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        writer.write_all(&start.to_le_bytes())?;
        writer.write_all(&(num_frozen_pts as u64).to_le_bytes())?;

        Ok(Self {
            writer,
            index_size: 24,
            max_degree: 0,
            start,
            num_frozen_pts: num_frozen_pts as u64,
        })
    }

    pub fn write_node(&mut self, _uid: u32, neighbors: &[u32]) -> AnnResult<()> {
        let degree = neighbors.len() as u32;
        self.writer.write_all(&degree.to_le_bytes())?;
        write_u32_slice_le(&mut self.writer, neighbors)?;
        self.index_size += ((neighbors.len() + 1) * std::mem::size_of::<u32>()) as u64;
        self.max_degree = self.max_degree.max(degree);
        Ok(())
    }

    pub fn set_start(&mut self, start: u32) {
        self.start = start;
    }

    pub fn finish(mut self) -> AnnResult<()> {
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&self.index_size.to_le_bytes())?;
        self.writer.write_all(&self.max_degree.to_le_bytes())?;
        self.writer.write_all(&self.start.to_le_bytes())?;
        self.writer.write_all(&self.num_frozen_pts.to_le_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};

    use tempfile::tempdir;

    use crate::algorithm::pipnn::hash_prune::HashPruneReservoir;
    use crate::algorithm::pipnn::hash_prune_backfill::{
        HashPruneBackfillPolicy, HashPruneBackfillProfile,
    };
    use crate::common::Metric;
    use crate::index::InmemIndex;
    use crate::model::{IndexConfiguration, IndexWriteParametersBuilder};

    #[derive(Default)]
    struct CountingWriter {
        writes: usize,
        bytes: Vec<u8>,
    }

    impl Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct ByteCountWriter {
        writes: u64,
        bytes: u64,
    }

    impl Write for ByteCountWriter {
        #[inline(never)]
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            self.bytes += buf.len() as u64;
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn write_u32_slice_le_uses_one_bulk_write() {
        let values = [0_u32, 1, 0x1234_5678, u32::MAX];
        let mut writer = CountingWriter::default();
        super::write_u32_slice_le(&mut writer, &values).unwrap();
        let expected = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(writer.writes, 1);
        assert_eq!(writer.bytes, expected);
    }

    #[test]
    #[ignore = "release-only contiguous writer microbenchmark"]
    fn benchmark_u32_neighbor_slice_bulk_write() {
        const ROWS: usize = 1_000_000;
        let neighbors = (0_u32..64).collect::<Vec<_>>();

        let mut scalar = ByteCountWriter::default();
        let scalar_start = std::time::Instant::now();
        for _ in 0..ROWS {
            for neighbor in &neighbors {
                scalar.write_all(&neighbor.to_le_bytes()).unwrap();
            }
        }
        let scalar_wall = scalar_start.elapsed();
        std::hint::black_box(&scalar);

        let mut bulk = ByteCountWriter::default();
        let bulk_start = std::time::Instant::now();
        for _ in 0..ROWS {
            super::write_u32_slice_le(&mut bulk, &neighbors).unwrap();
        }
        let bulk_wall = bulk_start.elapsed();
        std::hint::black_box(&bulk);

        assert_eq!(bulk.bytes, scalar.bytes);
        assert_eq!(bulk.writes, ROWS as u64);
        assert_eq!(scalar.writes, ROWS as u64 * neighbors.len() as u64);
        eprintln!(
            "neighbor writer rows={ROWS} degree={} scalar_ms={} bulk_ms={}",
            neighbors.len(),
            scalar_wall.as_millis(),
            bulk_wall.as_millis()
        );
    }

    fn reference_graph_bytes(start: u32, num_frozen_pts: u64, rows: &[Vec<u32>]) -> Vec<u8> {
        let payload_bytes = rows
            .iter()
            .map(|row| (row.len() + 1) * std::mem::size_of::<u32>())
            .sum::<usize>();
        let index_size = 24_u64 + payload_bytes as u64;
        let max_degree = rows.iter().map(Vec::len).max().unwrap_or(0) as u32;
        let mut bytes = Vec::with_capacity(index_size as usize);
        bytes.extend_from_slice(&index_size.to_le_bytes());
        bytes.extend_from_slice(&max_degree.to_le_bytes());
        bytes.extend_from_slice(&start.to_le_bytes());
        bytes.extend_from_slice(&num_frozen_pts.to_le_bytes());
        for row in rows {
            bytes.extend_from_slice(&(row.len() as u32).to_le_bytes());
            for neighbor in row {
                bytes.extend_from_slice(&neighbor.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn direct_mem_graph_writer_uses_large_sequential_buffer() {
        assert_eq!(super::DirectMemGraphWriter::BUFFER_BYTES, 8 * 1024 * 1024);
    }

    #[test]
    fn direct_mem_graph_writer_bulk_rows_match_scalar_file_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bulk.index");
        let rows = (0_u32..512)
            .map(|row| {
                (0_u32..row % 65)
                    .map(|offset| row.wrapping_mul(65).wrapping_add(offset))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut writer = super::DirectMemGraphWriter::create(&path, 17, 3).unwrap();
        for (uid, row) in rows.iter().enumerate() {
            writer.write_node(uid as u32, row).unwrap();
        }
        writer.finish().unwrap();
        assert_eq!(
            std::fs::read(path).unwrap(),
            reference_graph_bytes(17, 3, &rows)
        );
    }

    #[test]
    fn hashprune_backfill_off_writer_bytes_match_baseline_same_trace() {
        let dir = tempdir().unwrap();
        let baseline_path = dir.path().join("baseline.index");
        let off_path = dir.path().join("backfill-off.index");
        let traces = [
            vec![(1, 10, 1.0), (2, 10, 1.5), (3, 20, 2.0)],
            vec![(0, 30, 0.5), (2, 40, 3.0), (3, 30, 0.4)],
            vec![],
            vec![(4, 50, 4.0), (0, 60, 2.0), (1, 70, 1.0)],
            vec![(3, 80, 2.0), (2, 90, 1.0), (1, 80, 1.5)],
        ];

        let mut baseline_writer =
            super::DirectMemGraphWriter::create(&baseline_path, 17, 0).unwrap();
        let mut off_writer = super::DirectMemGraphWriter::create(&off_path, 17, 0).unwrap();
        let mut profile = HashPruneBackfillProfile::default();
        for (source, trace) in traces.iter().enumerate() {
            let mut baseline = HashPruneReservoir::new(4);
            let mut off = HashPruneReservoir::new(4);
            for &(target, hash, dist) in trace {
                baseline.insert(target, hash, dist);
                off.insert(target, hash, dist);
            }
            baseline_writer
                .write_node(source as u32, &baseline.into_neighbors())
                .unwrap();
            off_writer
                .write_node(
                    source as u32,
                    &off.into_neighbors_with_backfill(
                        source,
                        HashPruneBackfillPolicy::off(),
                        &mut profile,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        baseline_writer.finish().unwrap();
        off_writer.finish().unwrap();

        assert_eq!(profile, HashPruneBackfillProfile::default());
        assert_eq!(
            std::fs::read(baseline_path).unwrap(),
            std::fs::read(off_path).unwrap()
        );
    }

    #[test]
    fn direct_mem_graph_writer_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "pipnn-oom-writer-{}-{}.index",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut writer = super::DirectMemGraphWriter::create(&path, 17, 0).unwrap();
        writer.write_node(0, &[3, 4]).unwrap();
        writer.write_node(1, &[4]).unwrap();
        writer.write_node(2, &[1, 0, 4]).unwrap();
        writer.write_node(3, &[0]).unwrap();
        writer.write_node(4, &[2, 1]).unwrap();
        writer.finish().unwrap();

        let config = IndexConfiguration::new(
            Metric::L2,
            4,
            16,
            5,
            false,
            0,
            false,
            0,
            1.0,
            IndexWriteParametersBuilder::new(16, 8)
                .with_num_threads(1)
                .build(),
            0.0,
        );
        let mut index = InmemIndex::<f32>::new(config).unwrap();
        let loaded = index.load_graph(&path, 5).unwrap();

        assert_eq!(loaded, 5);
        assert_eq!(
            index
                .final_graph
                .read_vertex_and_neighbors(0)
                .unwrap()
                .get_neighbors()
                .as_slice(),
            &[3, 4]
        );
        assert_eq!(
            index
                .final_graph
                .read_vertex_and_neighbors(1)
                .unwrap()
                .get_neighbors()
                .as_slice(),
            &[4]
        );
        assert_eq!(
            index
                .final_graph
                .read_vertex_and_neighbors(2)
                .unwrap()
                .get_neighbors()
                .as_slice(),
            &[1, 0, 4]
        );
        assert_eq!(
            index
                .final_graph
                .read_vertex_and_neighbors(3)
                .unwrap()
                .get_neighbors()
                .as_slice(),
            &[0]
        );
        assert_eq!(
            index
                .final_graph
                .read_vertex_and_neighbors(4)
                .unwrap()
                .get_neighbors()
                .as_slice(),
            &[2, 1]
        );

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn direct_mem_graph_writer_can_patch_start_before_finish() {
        let path = std::env::temp_dir().join(format!(
            "pipnn-oom-writer-patched-start-{}-{}.index",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut writer = super::DirectMemGraphWriter::create(&path, 0, 0).unwrap();
        writer.write_node(0, &[1]).unwrap();
        writer.write_node(1, &[0]).unwrap();
        writer.set_start(1);
        writer.finish().unwrap();

        let config = IndexConfiguration::new(
            Metric::L2,
            4,
            16,
            2,
            false,
            0,
            false,
            0,
            1.0,
            IndexWriteParametersBuilder::new(16, 8)
                .with_num_threads(1)
                .build(),
            0.0,
        );
        let mut index = InmemIndex::<f32>::new(config).unwrap();
        index.load_graph(&path, 2).unwrap();

        assert_eq!(index.start, 1);

        std::fs::remove_file(path).unwrap();
    }
}
