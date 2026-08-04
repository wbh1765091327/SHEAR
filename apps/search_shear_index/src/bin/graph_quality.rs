use std::collections::VecDeque;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use clap::Parser;
use memmap2::{Mmap, MmapOptions};
use shear_core::utils::file_util::load_metadata_from_file;

#[derive(Parser)]
struct Args {
    /// Path prefix where _mem.index lives.
    #[clap(long)]
    index_path_prefix: PathBuf,

    /// Base data .fbin. Used for point count and edge-distance sampling.
    #[clap(long)]
    data_path: PathBuf,

    /// TSV output path.
    #[clap(long)]
    output: PathBuf,

    /// Maximum directed edges sampled for reciprocal and edge-distance metrics.
    #[clap(long, default_value_t = 1_000_000)]
    max_edge_samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (num_points, dim) = load_metadata_from_file(&args.data_path)?;
    let graph_path = args.index_path_prefix.join("_mem.index");
    let graph = MmapGraph::open(&graph_path, num_points)?;
    let vectors = MmapFbinF32::open(&args.data_path)?;
    ensure!(
        vectors.num_points() == num_points && vectors.dim() == dim,
        "data metadata mismatch: header reports ({num_points}, {dim}), mmap reports ({}, {})",
        vectors.num_points(),
        vectors.dim()
    );

    let summary = summarize_graph(&graph, Some(&vectors), args.max_edge_samples)?;
    write_graph_summary_tsv(&args.output, &summary)?;
    eprintln!("Graph quality metrics saved to {:?}", args.output);
    Ok(())
}

trait GraphAccess {
    fn start(&self) -> u32;
    fn num_nodes(&self) -> usize;
    fn neighbors(&self, id: usize) -> Option<&[u32]>;
}

trait VectorAccess {
    fn row(&self, id: usize) -> Option<&[f32]>;
}

#[derive(Debug)]
struct MmapFbinF32 {
    _file: File,
    mmap: Mmap,
    num_points: usize,
    dim: usize,
}

impl MmapFbinF32 {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open fbin {:?}", path))?;
        ensure!(
            file.metadata()?.len() >= 8,
            "fbin file {:?} is too small for an 8-byte header",
            path
        );
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let num_points = read_u32_at(&mmap, 0)? as usize;
        let dim = read_u32_at(&mmap, 4)? as usize;
        let payload_bytes = num_points
            .checked_mul(dim)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .context("fbin payload byte length overflow")?;
        let expected_len = 8usize
            .checked_add(payload_bytes)
            .context("fbin byte length overflow")?;
        ensure!(
            mmap.len() == expected_len,
            "fbin file {:?} length {} != expected {} for {num_points}x{dim} f32",
            path,
            mmap.len(),
            expected_len
        );
        Ok(Self {
            _file: file,
            mmap,
            num_points,
            dim,
        })
    }

    fn num_points(&self) -> usize {
        self.num_points
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

impl VectorAccess for MmapFbinF32 {
    fn row(&self, id: usize) -> Option<&[f32]> {
        if id >= self.num_points {
            return None;
        }
        let byte_offset = 8usize.checked_add(
            id.checked_mul(self.dim)?
                .checked_mul(std::mem::size_of::<f32>())?,
        )?;
        let byte_len = self.dim.checked_mul(std::mem::size_of::<f32>())?;
        let bytes = self.mmap.get(byte_offset..byte_offset + byte_len)?;
        Some(bytemuck::cast_slice(bytes))
    }
}

#[derive(Debug)]
struct MmapGraph {
    _file: File,
    mmap: Mmap,
    offsets: Vec<u64>,
    start: u32,
}

impl MmapGraph {
    fn open(path: &Path, expected_num_points: usize) -> Result<Self> {
        ensure!(
            expected_num_points > 0,
            "graph must have at least one point"
        );
        let file = File::open(path).with_context(|| format!("open graph {:?}", path))?;
        ensure!(
            file.metadata()?.len() >= GRAPH_HEADER_BYTES as u64,
            "graph file {:?} is too small for a {}-byte header",
            path,
            GRAPH_HEADER_BYTES
        );
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let expected_file_size = read_u64_at(&mmap, 0)? as usize;
        let start = read_u32_at(&mmap, 12)?;
        ensure!(
            expected_file_size == mmap.len(),
            "graph header expected_file_size {} != actual file size {} for {:?}",
            expected_file_size,
            mmap.len(),
            path
        );
        ensure!(
            (start as usize) < expected_num_points,
            "graph start node {} is outside expected {} points",
            start,
            expected_num_points
        );
        ensure!(
            (mmap.len() - GRAPH_HEADER_BYTES).is_multiple_of(std::mem::size_of::<u32>()),
            "graph payload size {} is not aligned to u32 records",
            mmap.len() - GRAPH_HEADER_BYTES
        );

        let offsets = scan_graph_offsets(&mmap, expected_num_points)?;
        Ok(Self {
            _file: file,
            mmap,
            offsets,
            start,
        })
    }
}

impl GraphAccess for MmapGraph {
    fn start(&self) -> u32 {
        self.start
    }

    fn num_nodes(&self) -> usize {
        self.offsets.len()
    }

    fn neighbors(&self, id: usize) -> Option<&[u32]> {
        let degree_offset = *self.offsets.get(id)? as usize;
        let degree = read_u32_at(&self.mmap, degree_offset).ok()? as usize;
        let neighbors_offset = degree_offset.checked_add(std::mem::size_of::<u32>())?;
        let neighbors_bytes = degree.checked_mul(std::mem::size_of::<u32>())?;
        let bytes = self
            .mmap
            .get(neighbors_offset..neighbors_offset + neighbors_bytes)?;
        Some(bytemuck::cast_slice(bytes))
    }
}

#[cfg(test)]
#[derive(Debug)]
struct InMemoryGraph {
    start: u32,
    graph: Vec<Vec<u32>>,
}

#[cfg(test)]
impl InMemoryGraph {
    fn new(start: u32, graph: Vec<Vec<u32>>) -> Self {
        Self { start, graph }
    }
}

#[cfg(test)]
impl GraphAccess for InMemoryGraph {
    fn start(&self) -> u32 {
        self.start
    }

    fn num_nodes(&self) -> usize {
        self.graph.len()
    }

    fn neighbors(&self, id: usize) -> Option<&[u32]> {
        self.graph.get(id).map(Vec::as_slice)
    }
}

#[cfg(test)]
#[derive(Debug)]
struct InMemoryVectors {
    rows: Vec<Vec<f32>>,
}

#[cfg(test)]
impl InMemoryVectors {
    fn new(rows: Vec<Vec<f32>>) -> Self {
        Self { rows }
    }
}

#[cfg(test)]
impl VectorAccess for InMemoryVectors {
    fn row(&self, id: usize) -> Option<&[f32]> {
        self.rows.get(id).map(Vec::as_slice)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct GraphSummary {
    nodes: usize,
    edges: usize,
    self_loop_edges: usize,
    start: u32,
    max_out_degree: usize,
    avg_out_degree: f64,
    p50_out_degree: usize,
    p95_out_degree: usize,
    zero_out_nodes: usize,
    avg_in_degree: f64,
    p50_in_degree: usize,
    p95_in_degree: usize,
    zero_in_nodes: usize,
    reciprocal_sampled_edges: usize,
    reciprocal_directed_edges: usize,
    reciprocal_directed_ratio: f64,
    directed_reachable_from_start: usize,
    directed_reachable_from_start_ratio: f64,
    edge_distance_samples: usize,
    edge_distance_p50: Option<f32>,
    edge_distance_p95: Option<f32>,
    edge_distance_p99: Option<f32>,
}

fn summarize_graph(
    graph: &impl GraphAccess,
    vectors: Option<&dyn VectorAccess>,
    max_edge_samples: usize,
) -> Result<GraphSummary> {
    let nodes = graph.num_nodes();
    ensure!(nodes > 0, "graph must have at least one node");
    ensure!(
        (graph.start() as usize) < nodes,
        "graph start node {} is outside {} nodes",
        graph.start(),
        nodes
    );

    let mut out_degrees = Vec::with_capacity(nodes);
    let mut in_degrees = vec![0usize; nodes];
    let mut edges = 0usize;
    let mut self_loop_edges = 0usize;

    for id in 0..nodes {
        let neighbors = graph
            .neighbors(id)
            .with_context(|| format!("missing graph neighbors for node {id}"))?;
        out_degrees.push(neighbors.len());
        edges = edges
            .checked_add(neighbors.len())
            .context("edge count overflow")?;
        for &neighbor in neighbors {
            let neighbor = neighbor as usize;
            ensure!(
                neighbor < nodes,
                "node {id} has out-of-range neighbor {neighbor} for {nodes} nodes"
            );
            if neighbor == id {
                self_loop_edges += 1;
            }
            in_degrees[neighbor] += 1;
        }
    }

    let mut reciprocal_sampled_edges = 0usize;
    let mut reciprocal_directed_edges = 0usize;
    let mut edge_distances = Vec::new();
    let sample_limit = max_edge_samples.min(edges);
    let sample_stride = if sample_limit == 0 {
        usize::MAX
    } else {
        edges.div_ceil(sample_limit)
    };
    let mut edge_ordinal = 0usize;

    for id in 0..nodes {
        let neighbors = graph.neighbors(id).unwrap();
        for &neighbor in neighbors {
            let should_sample = sample_limit > 0
                && reciprocal_sampled_edges < sample_limit
                && edge_ordinal.is_multiple_of(sample_stride);
            if should_sample {
                reciprocal_sampled_edges += 1;
                if graph
                    .neighbors(neighbor as usize)
                    .unwrap_or_default()
                    .contains(&(id as u32))
                {
                    reciprocal_directed_edges += 1;
                }
                if let Some(vectors) = vectors {
                    if let (Some(lhs), Some(rhs)) =
                        (vectors.row(id), vectors.row(neighbor as usize))
                    {
                        ensure!(
                            lhs.len() == rhs.len(),
                            "vector dim mismatch between node {id} and neighbor {neighbor}"
                        );
                        edge_distances.push(l2_distance(lhs, rhs));
                    }
                }
            }
            edge_ordinal += 1;
        }
    }

    let directed_reachable_from_start = directed_reachable_count(graph)?;
    let mut out_for_p = out_degrees.clone();
    let mut in_for_p = in_degrees.clone();

    Ok(GraphSummary {
        nodes,
        edges,
        self_loop_edges,
        start: graph.start(),
        max_out_degree: out_degrees.iter().copied().max().unwrap_or(0),
        avg_out_degree: average_usize(&out_degrees),
        p50_out_degree: percentile_usize(&mut out_for_p, 0.50),
        p95_out_degree: percentile_usize(&mut out_degrees, 0.95),
        zero_out_nodes: out_for_p.iter().filter(|&&degree| degree == 0).count(),
        avg_in_degree: average_usize(&in_degrees),
        p50_in_degree: percentile_usize(&mut in_for_p, 0.50),
        p95_in_degree: percentile_usize(&mut in_degrees, 0.95),
        zero_in_nodes: in_for_p.iter().filter(|&&degree| degree == 0).count(),
        reciprocal_sampled_edges,
        reciprocal_directed_edges,
        reciprocal_directed_ratio: if reciprocal_sampled_edges == 0 {
            0.0
        } else {
            reciprocal_directed_edges as f64 / reciprocal_sampled_edges as f64
        },
        directed_reachable_from_start,
        directed_reachable_from_start_ratio: directed_reachable_from_start as f64 / nodes as f64,
        edge_distance_samples: edge_distances.len(),
        edge_distance_p50: percentile_f32(&mut edge_distances, 0.50),
        edge_distance_p95: percentile_f32(&mut edge_distances, 0.95),
        edge_distance_p99: percentile_f32(&mut edge_distances, 0.99),
    })
}

fn directed_reachable_count(graph: &impl GraphAccess) -> Result<usize> {
    let nodes = graph.num_nodes();
    let mut visited = vec![false; nodes];
    let mut queue = VecDeque::new();
    let start = graph.start() as usize;
    visited[start] = true;
    queue.push_back(start);
    let mut count = 0usize;

    while let Some(id) = queue.pop_front() {
        count += 1;
        for &neighbor in graph
            .neighbors(id)
            .with_context(|| format!("missing graph neighbors for node {id}"))?
        {
            let neighbor = neighbor as usize;
            ensure!(
                neighbor < nodes,
                "node {id} has out-of-range neighbor {neighbor} for {nodes} nodes"
            );
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    Ok(count)
}

fn write_graph_summary_tsv(path: &Path, summary: &GraphSummary) -> Result<()> {
    let mut writer = std::io::BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "nodes\tedges\tself_loop_edges\tstart\tmax_out_degree\tavg_out_degree\tp50_out_degree\tp95_out_degree\tzero_out_nodes\tavg_in_degree\tp50_in_degree\tp95_in_degree\tzero_in_nodes\treciprocal_sampled_edges\treciprocal_directed_edges\treciprocal_directed_ratio\tdirected_reachable_from_start\tdirected_reachable_from_start_ratio\tedge_distance_samples\tedge_distance_p50\tedge_distance_p95\tedge_distance_p99"
    )?;
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{:.6}\t{}\t{}\t{}\t{}",
        summary.nodes,
        summary.edges,
        summary.self_loop_edges,
        summary.start,
        summary.max_out_degree,
        summary.avg_out_degree,
        summary.p50_out_degree,
        summary.p95_out_degree,
        summary.zero_out_nodes,
        summary.avg_in_degree,
        summary.p50_in_degree,
        summary.p95_in_degree,
        summary.zero_in_nodes,
        summary.reciprocal_sampled_edges,
        summary.reciprocal_directed_edges,
        summary.reciprocal_directed_ratio,
        summary.directed_reachable_from_start,
        summary.directed_reachable_from_start_ratio,
        summary.edge_distance_samples,
        format_optional_f32(summary.edge_distance_p50),
        format_optional_f32(summary.edge_distance_p95),
        format_optional_f32(summary.edge_distance_p99)
    )?;
    writer.flush()?;
    Ok(())
}

const GRAPH_HEADER_BYTES: usize = 24;

fn scan_graph_offsets(mmap: &[u8], expected_num_points: usize) -> Result<Vec<u64>> {
    if let Some(offsets) = try_scan_sequential_graph_offsets(mmap, expected_num_points)? {
        return Ok(offsets);
    }
    if let Some(offsets) = try_scan_explicit_graph_offsets(mmap, expected_num_points)? {
        return Ok(offsets);
    }
    bail!("unable to determine graph record encoding for {expected_num_points} points")
}

fn try_scan_sequential_graph_offsets(
    mmap: &[u8],
    expected_num_points: usize,
) -> Result<Option<Vec<u64>>> {
    let mut offsets = Vec::with_capacity(expected_num_points);
    let mut offset = GRAPH_HEADER_BYTES;

    for _ in 0..expected_num_points {
        let Some(degree) = read_u32_at_opt(mmap, offset) else {
            return Ok(None);
        };
        let degree = degree as usize;
        let record_bytes = match (1usize)
            .checked_add(degree)
            .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        if offset
            .checked_add(record_bytes)
            .is_none_or(|end| end > mmap.len())
        {
            return Ok(None);
        }
        offsets.push(offset as u64);
        offset += record_bytes;
    }

    if offset == mmap.len() {
        Ok(Some(offsets))
    } else {
        Ok(None)
    }
}

fn try_scan_explicit_graph_offsets(
    mmap: &[u8],
    expected_num_points: usize,
) -> Result<Option<Vec<u64>>> {
    let mut offsets = vec![u64::MAX; expected_num_points];
    let mut seen_count = 0usize;
    let mut offset = GRAPH_HEADER_BYTES;

    while offset < mmap.len() {
        if offset + 2 * std::mem::size_of::<u32>() > mmap.len() {
            return Ok(None);
        }
        let uid = read_u32_at(mmap, offset)? as usize;
        if uid >= expected_num_points || offsets[uid] != u64::MAX {
            return Ok(None);
        }
        let degree_offset = offset + std::mem::size_of::<u32>();
        let degree = read_u32_at(mmap, degree_offset)? as usize;
        let record_bytes = match (2usize)
            .checked_add(degree)
            .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        if offset
            .checked_add(record_bytes)
            .is_none_or(|end| end > mmap.len())
        {
            return Ok(None);
        }
        offsets[uid] = degree_offset as u64;
        seen_count += 1;
        offset += record_bytes;
    }

    if seen_count == expected_num_points && offsets.iter().all(|&item| item != u64::MAX) {
        Ok(Some(offsets))
    } else {
        Ok(None)
    }
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    let chunk = bytes
        .get(offset..offset + std::mem::size_of::<u32>())
        .context("u32 read offset is outside mapped file")?;
    Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_u32_at_opt(bytes: &[u8], offset: usize) -> Option<u32> {
    let chunk = bytes.get(offset..offset + std::mem::size_of::<u32>())?;
    Some(u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    let chunk = bytes
        .get(offset..offset + std::mem::size_of::<u64>())
        .context("u64 read offset is outside mapped file")?;
    Ok(u64::from_le_bytes(chunk.try_into().unwrap()))
}

fn average_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<usize>() as f64 / values.len() as f64
}

fn percentile_usize(values: &mut [usize], percentile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let rank = ((values.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
    values[rank.min(values.len() - 1)]
}

fn percentile_f32(values: &mut [f32], percentile: f64) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((values.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
    Some(values[rank.min(values.len() - 1)])
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

fn format_optional_f32(value: Option<f32>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_summary_reports_degree_reciprocity_reachability_and_edge_distances() {
        let graph = InMemoryGraph::new(0, vec![vec![1, 2], vec![0, 2], vec![3], vec![2]]);
        let vectors = InMemoryVectors::new(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![0.0, 5.0],
        ]);

        let summary = summarize_graph(&graph, Some(&vectors), 100).unwrap();

        assert_eq!(summary.nodes, 4);
        assert_eq!(summary.edges, 6);
        assert_eq!(summary.self_loop_edges, 0);
        assert_eq!(summary.max_out_degree, 2);
        assert_eq!(summary.zero_out_nodes, 0);
        assert_eq!(summary.zero_in_nodes, 0);
        assert_eq!(summary.p95_in_degree, 3);
        assert_eq!(summary.reciprocal_sampled_edges, 6);
        assert_eq!(summary.reciprocal_directed_edges, 4);
        assert!((summary.reciprocal_directed_ratio - (4.0 / 6.0)).abs() < 1e-9);
        assert_eq!(summary.directed_reachable_from_start, 4);
        assert_eq!(summary.edge_distance_samples, 6);
        assert_eq!(summary.edge_distance_p50, Some(4.0));
        assert_eq!(summary.edge_distance_p95, Some(9.0));
        assert_eq!(summary.edge_distance_p99, Some(9.0));
    }

    #[test]
    fn graph_summary_counts_self_loops() {
        let graph = InMemoryGraph::new(0, vec![vec![0, 1], vec![]]);

        let summary = summarize_graph(&graph, None, 100).unwrap();

        assert_eq!(summary.edges, 2);
        assert_eq!(summary.self_loop_edges, 1);
    }
}
