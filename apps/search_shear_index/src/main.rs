use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read as _, Write as _};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{Parser, ValueEnum};
use memmap2::{Mmap, MmapOptions};
use shear_core::model::hashprune_provenance::HashPrunePrimaryCountSidecar;
use shear_core::model::rabitq::graph_search::Rabitq1Sidecar;
use shear_core::utils::file_util::load_metadata_from_file;

mod rabitq_search;

use rabitq_search::{
    GraphAccess, Rabitq1Estimator, RabitqSearchScratch, SearchOutcome, SearchStats, VectorAccess,
    duration_ns, search_adaptive_exact, search_post_rerank,
};

/// Search a _mem.index (Vamana/PiPNN adjacency-list format).
#[derive(Parser)]
struct Args {
    /// Path prefix where _mem.index lives
    #[clap(long)]
    index_path_prefix: PathBuf,

    /// Path to base data .fbin file (raw f32 vectors)
    #[clap(long)]
    data_path: PathBuf,

    /// Query vectors in .fbin format
    #[clap(long)]
    query_file: PathBuf,

    /// Result output .fbin
    #[clap(long)]
    result_path: PathBuf,

    /// Ground truth file
    #[clap(long)]
    gt_file: PathBuf,

    /// Recall@K
    #[clap(long, default_value_t = 10)]
    recall_at: usize,

    /// Search list size (L)
    #[clap(long, default_value_t = 100)]
    search_list_size: usize,

    /// Number of threads
    #[clap(long, short = 'T', default_value_t = 1)]
    num_threads: usize,

    /// Storage mode for base vectors and graph adjacency.
    #[clap(long, value_enum, default_value = "mmap")]
    storage: StorageMode,

    /// Optional query limit for validator smoke runs.
    #[clap(long)]
    max_queries: Option<usize>,

    /// Optional TSV path for aggregate navigation metrics.
    #[clap(long)]
    metrics_path: Option<PathBuf>,

    /// Distance policy used while traversing the graph.
    #[clap(long, value_enum, default_value = "exact")]
    search_backend: SearchBackend,

    /// Dataset-level resident 1-bit RaBitQ sidecar.
    #[clap(long)]
    rabitq1_sidecar: Option<PathBuf>,

    /// Primary-prefix boundary for diagnostic HashPrune search attribution.
    #[clap(long)]
    hashprune_primary_count_sidecar: Option<PathBuf>,

    /// Per-query search counters; written after the timed search region.
    #[clap(long)]
    query_metrics_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SearchBackend {
    Exact,
    Rabitq1PostRerank,
    Rabitq1AdaptiveExact,
}

impl SearchBackend {
    fn as_str(self) -> &'static str {
        match self {
            SearchBackend::Exact => "exact",
            SearchBackend::Rabitq1PostRerank => "rabitq1-post-rerank",
            SearchBackend::Rabitq1AdaptiveExact => "rabitq1-adaptive-exact",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum StorageMode {
    /// Memory-map the base vectors and graph, building only a compact graph offset table.
    Mmap,
    /// Load the full base vectors and graph into resident memory.
    Memory,
}

impl StorageMode {
    fn as_str(self) -> &'static str {
        match self {
            StorageMode::Mmap => "mmap",
            StorageMode::Memory => "memory",
        }
    }
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.recall_at > 0, "--recall-at must be positive");
    ensure!(
        args.search_list_size >= args.recall_at,
        "--search-list-size must be at least --recall-at"
    );
    ensure!(args.num_threads > 0, "--num-threads must be positive");
    ensure!(
        args.max_queries != Some(0),
        "--max-queries must be positive"
    );
    match args.search_backend {
        SearchBackend::Exact => ensure!(
            args.rabitq1_sidecar.is_none(),
            "exact backend does not accept --rabitq1-sidecar"
        ),
        SearchBackend::Rabitq1PostRerank | SearchBackend::Rabitq1AdaptiveExact => {
            ensure!(
                args.rabitq1_sidecar.is_some(),
                "RaBitQ1 backend requires --rabitq1-sidecar"
            );
            ensure!(
                args.storage == StorageMode::Mmap,
                "RaBitQ1 backends require --storage mmap"
            );
        }
    }
    if args.hashprune_primary_count_sidecar.is_some() {
        ensure!(
            args.query_metrics_path.is_some(),
            "--hashprune-primary-count-sidecar requires --query-metrics-path"
        );
        ensure!(
            args.search_backend == SearchBackend::Rabitq1AdaptiveExact,
            "HashPrune provenance requires --search-backend rabitq1-adaptive-exact"
        );
        ensure!(
            args.storage == StorageMode::Mmap,
            "HashPrune provenance requires --storage mmap"
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    run(Args::parse())
}

fn run(args: Args) -> anyhow::Result<()> {
    validate_args(&args)?;
    let total_t0 = std::time::Instant::now();

    // 1. Read base metadata.
    let data_path = args.data_path.clone();
    let t0 = std::time::Instant::now();
    let (num_points, dim) = load_metadata_from_file(&data_path)?;
    eprintln!("Base data: {num_points} points, dim={dim}");
    eprintln!("Stage base metadata: {:.2}s", t0.elapsed().as_secs_f64());

    let sidecar = if args.search_backend == SearchBackend::Exact {
        None
    } else {
        let sidecar_path = args
            .rabitq1_sidecar
            .as_ref()
            .context("RaBitQ1 sidecar path is missing")?;
        let loaded = Rabitq1Sidecar::load(sidecar_path, &data_path)
            .with_context(|| format!("load RaBitQ1 sidecar {sidecar_path:?}"))?;
        ensure!(
            loaded.header().rows == num_points as u64 && loaded.header().dim == dim as u32,
            "RaBitQ1 sidecar metadata does not match base data"
        );
        eprintln!(
            "RaBitQ1 sidecar: rows={}, dim={}, padded_dim={}, stride={}, resident_bytes={}, load={:.3}s",
            loaded.header().rows,
            loaded.header().dim,
            loaded.header().padded_dim,
            loaded.header().bin_stride,
            loaded.resident_bytes(),
            loaded.load_seconds()
        );
        Some(Arc::new(loaded))
    };
    let sidecar_load_seconds = sidecar
        .as_ref()
        .map_or(0.0, |sidecar| sidecar.load_seconds());
    let sidecar_resident_bytes = sidecar
        .as_ref()
        .map_or(0, |sidecar| sidecar.resident_bytes());

    // 2. Locate graph.
    let graph_path = args.index_path_prefix.join("_mem.index");

    // 3. Load queries.
    let t0 = std::time::Instant::now();
    let (query_file_num_queries, query_dim) = load_metadata_from_file(&args.query_file)?;
    ensure!(query_dim == dim, "query dim {query_dim} != data dim {dim}");
    let queries = load_fbin_f32(&args.query_file)?;
    let num_queries = args
        .max_queries
        .map(|limit| limit.min(query_file_num_queries))
        .unwrap_or(query_file_num_queries);
    ensure!(num_queries > 0, "query count must be positive");
    eprintln!("Queries: {num_queries}");
    if num_queries != query_file_num_queries {
        eprintln!("Query limit: using first {num_queries} of {query_file_num_queries} queries");
    }
    eprintln!("Stage load queries: {:.2}s", t0.elapsed().as_secs_f64());

    // 4. Search
    let k = args.recall_at;
    let l = args.search_list_size;

    let (all_results, all_result_flags, all_stats, search_elapsed, graph_encoding, graph_bytes) =
        if args.storage == StorageMode::Mmap {
            let t0 = std::time::Instant::now();
            let data = MmapFbinF32::open(&data_path)?;
            ensure!(
                data.num_points() == num_points && data.dim() == dim,
                "mmap data metadata mismatch: metadata reports ({num_points}, {dim}), mmap reports ({}, {})",
                data.num_points(),
                data.dim()
            );
            eprintln!(
                "Base data mmap: {} points, dim={} ({:.2}s)",
                data.num_points(),
                data.dim(),
                t0.elapsed().as_secs_f64()
            );

            let t0 = std::time::Instant::now();
            let mut graph = MmapGraph::open(&graph_path, num_points)?;
            if let Some(primary_count_path) = &args.hashprune_primary_count_sidecar {
                graph
                    .attach_primary_counts(primary_count_path)
                    .with_context(|| {
                        format!("attach HashPrune primary-count sidecar {primary_count_path:?}")
                    })?;
            }
            eprintln!(
                "Graph mmap: {} nodes, start={}, max_degree={}, encoding={} ({:.2}s)",
                graph.num_nodes(),
                graph.start(),
                graph.max_degree(),
                graph.encoding_name(),
                t0.elapsed().as_secs_f64()
            );

            let graph_encoding = graph.encoding_name().to_owned();
            let graph_bytes = graph.file_bytes() as u64;
            let t0 = std::time::Instant::now();
            let outcomes = match args.search_backend {
                SearchBackend::Exact => {
                    if args.num_threads <= 1 {
                        let mut scratch = SearchScratch::new(graph.num_nodes(), l);
                        (0..num_queries)
                            .map(|qi| {
                                let query = &queries[qi * dim..(qi + 1) * dim];
                                search_one_mmap_with_stats(query, &data, &graph, k, l, &mut scratch)
                            })
                            .collect::<Vec<_>>()
                    } else {
                        parallel_search_mmap(
                            &queries,
                            &data,
                            &graph,
                            k,
                            l,
                            num_queries,
                            args.num_threads,
                        )
                    }
                }
                SearchBackend::Rabitq1PostRerank | SearchBackend::Rabitq1AdaptiveExact => {
                    parallel_search_rabitq1(
                        &queries,
                        &data,
                        &graph,
                        sidecar.as_ref().context("RaBitQ1 sidecar is unavailable")?,
                        args.search_backend,
                        k,
                        l,
                        num_queries,
                        args.num_threads,
                    )?
                }
            };
            let elapsed = t0.elapsed();
            eprintln!(
                "Search: {num_queries} queries in {:.2}s ({:.1} QPS)",
                elapsed.as_secs_f64(),
                num_queries as f64 / elapsed.as_secs_f64()
            );
            let (results, result_flags, stats, elapsed) = split_search_outcomes(outcomes, elapsed)?;
            (
                results,
                result_flags,
                stats,
                elapsed,
                graph_encoding,
                graph_bytes,
            )
        } else {
            let graph_bytes = std::fs::metadata(&graph_path)?.len();
            let t0 = std::time::Instant::now();
            let data = load_fbin_f32(&data_path)?;
            eprintln!(
                "Stage load base data (memory): {:.2}s",
                t0.elapsed().as_secs_f64()
            );

            let t0 = std::time::Instant::now();
            let (start_node, graph) = load_graph(&graph_path, num_points)?;
            eprintln!(
                "Graph: {} nodes, start={}, max_degree={} ({:.2}s)",
                graph.len(),
                start_node,
                graph.iter().map(|v| v.len()).max().unwrap_or(0),
                t0.elapsed().as_secs_f64()
            );

            let t0 = std::time::Instant::now();
            let outcomes = if args.num_threads <= 1 {
                (0..num_queries)
                    .map(|qi| {
                        let query = &queries[qi * dim..(qi + 1) * dim];
                        search_one_with_stats(query, &data, dim, &graph, start_node, k, l)
                    })
                    .collect::<Vec<_>>()
            } else {
                parallel_search(
                    &queries,
                    &data,
                    dim,
                    &graph,
                    start_node,
                    k,
                    l,
                    num_queries,
                    args.num_threads,
                )
            };
            let elapsed = t0.elapsed();
            eprintln!(
                "Search: {num_queries} queries in {:.2}s ({:.1} QPS)",
                elapsed.as_secs_f64(),
                num_queries as f64 / elapsed.as_secs_f64()
            );
            let (results, result_flags, stats, elapsed) = split_search_outcomes(outcomes, elapsed)?;
            (
                results,
                result_flags,
                stats,
                elapsed,
                "resident-decoded".to_owned(),
                graph_bytes,
            )
        };

    // 5. Compute recall
    let t0 = std::time::Instant::now();
    let gt = load_gt(&args.gt_file, num_queries, args.recall_at)?;
    let mut total_recall = 0.0f64;
    let mut gt_hits = Vec::with_capacity(num_queries);
    let mut backfill_final_gt_hits = Vec::with_capacity(num_queries);
    for qi in 0..num_queries {
        let gt_set: HashSet<u32> = gt[qi * args.recall_at..(qi + 1) * args.recall_at]
            .iter()
            .copied()
            .collect();
        let hits = all_results[qi]
            .iter()
            .filter(|&&id| gt_set.contains(&id))
            .count();
        let attributed_hits = all_results[qi]
            .iter()
            .zip(&all_result_flags[qi])
            .filter(|(id, flag)| **flag && gt_set.contains(id))
            .count() as u64;
        gt_hits.push(hits);
        backfill_final_gt_hits.push(attributed_hits);
        total_recall += hits as f64 / args.recall_at as f64;
    }
    let recall = total_recall / num_queries as f64;
    eprintln!("Recall@{}: {:.4}", args.recall_at, recall);
    eprintln!("Stage recall: {:.2}s", t0.elapsed().as_secs_f64());

    if let Some(metrics_path) = &args.metrics_path {
        let t0 = std::time::Instant::now();
        write_search_metrics_tsv(
            metrics_path,
            args.storage,
            num_queries,
            k,
            l,
            recall,
            search_elapsed.as_secs_f64(),
            &all_stats,
            args.search_backend,
            sidecar_load_seconds,
            sidecar_resident_bytes,
            dim,
            &graph_encoding,
            graph_bytes,
            &backfill_final_gt_hits,
        )?;
        eprintln!("Search metrics saved to {:?}", metrics_path);
        eprintln!(
            "Stage save search metrics: {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }

    if let Some(query_metrics_path) = &args.query_metrics_path {
        write_query_metrics_tsv(
            query_metrics_path,
            l,
            dim,
            &gt_hits,
            &backfill_final_gt_hits,
            &all_stats,
        )?;
        eprintln!("Per-query search metrics saved to {:?}", query_metrics_path);
    }

    // 6. Save results
    let t0 = std::time::Instant::now();
    save_result_fbin(&args.result_path, &all_results, k)?;
    eprintln!("Results saved to {:?}", args.result_path);
    eprintln!("Stage save results: {:.2}s", t0.elapsed().as_secs_f64());
    eprintln!(
        "Total validator wall: {:.2}s",
        total_t0.elapsed().as_secs_f64()
    );

    Ok(())
}

fn parallel_search(
    queries: &[f32],
    data: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    start: u32,
    k: usize,
    l: usize,
    num_queries: usize,
    num_threads: usize,
) -> Vec<SearchOutcome> {
    let results: std::sync::Mutex<Vec<Option<SearchOutcome>>> =
        std::sync::Mutex::new(vec![None; num_queries]);

    std::thread::scope(|s| {
        let chunk = (num_queries + num_threads - 1) / num_threads;
        for tid in 0..num_threads {
            let lo = tid * chunk;
            let hi = (lo + chunk).min(num_queries);
            if lo >= hi {
                continue;
            }
            let results = &results;
            s.spawn(move || {
                for qi in lo..hi {
                    let query = &queries[qi * dim..(qi + 1) * dim];
                    let result = search_one_with_stats(query, data, dim, graph, start, k, l);
                    results.lock().unwrap()[qi] = Some(result);
                }
            });
        }
    });

    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.unwrap())
        .collect()
}

fn load_fbin_f32(path: &PathBuf) -> anyhow::Result<Vec<f32>> {
    let (n, d) = load_metadata_from_file(path)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.read_u32::<LittleEndian>()?; // n
    reader.read_u32::<LittleEndian>()?; // d
    let byte_len = n * d * 4;
    let mut buf = vec![0u8; byte_len];
    reader.read_exact(&mut buf)?;
    Ok(bytemuck::cast_slice(&buf).to_vec())
}

fn load_graph(path: &PathBuf, expected_num_points: usize) -> anyhow::Result<(u32, Vec<Vec<u32>>)> {
    let mut reader = BufReader::new(File::open(path)?);
    let _total_size = reader.read_u64::<LittleEndian>()?;
    let _max_degree = reader.read_u32::<LittleEndian>()?;
    let start = reader.read_u32::<LittleEndian>()?;
    let _frozen_pts = reader.read_u64::<LittleEndian>()?;

    let mut graph = Vec::with_capacity(expected_num_points);
    for _ in 0..expected_num_points {
        let degree = reader.read_u32::<LittleEndian>()?;
        let mut neighbors = vec![0u32; degree as usize];
        for nbr in &mut neighbors {
            *nbr = reader.read_u32::<LittleEndian>()?;
        }
        graph.push(neighbors);
    }
    Ok((start, graph))
}

fn load_gt(path: &PathBuf, num_queries: usize, recall_at: usize) -> anyhow::Result<Vec<u32>> {
    let (n, d) = load_metadata_from_file(path)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.read_u32::<LittleEndian>()?; // n
    reader.read_u32::<LittleEndian>()?; // d
    let total = n * d;
    let mut all = vec![0u32; total];
    for id in &mut all {
        *id = reader.read_u32::<LittleEndian>()?;
    }
    let nq = num_queries.min(n);
    let mut gt = vec![0u32; nq * recall_at];
    for qi in 0..nq {
        for ki in 0..recall_at.min(d) {
            gt[qi * recall_at + ki] = all[qi * d + ki];
        }
    }
    Ok(gt)
}

struct MmapFbinF32 {
    _file: File,
    mmap: Mmap,
    num_points: usize,
    dim: usize,
}

impl MmapFbinF32 {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path).with_context(|| format!("open fbin {:?}", path))?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.len() >= 8,
            "fbin file {:?} is too small for an 8-byte header",
            path
        );
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let num_points = read_u32_at(&mmap, 0)? as usize;
        let dim = read_u32_at(&mmap, 4)? as usize;
        let payload_bytes = num_points
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .context("fbin payload byte length overflow")?;
        let expected_len = 8usize
            .checked_add(payload_bytes)
            .context("fbin file byte length overflow")?;
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

    fn row(&self, id: usize) -> Option<&[f32]> {
        if id >= self.num_points {
            return None;
        }
        let float_offset = id.checked_mul(self.dim)?;
        let byte_offset =
            8usize.checked_add(float_offset.checked_mul(std::mem::size_of::<f32>())?)?;
        let byte_len = self.dim.checked_mul(std::mem::size_of::<f32>())?;
        let row_bytes = self.mmap.get(byte_offset..byte_offset + byte_len)?;
        Some(bytemuck::cast_slice(row_bytes))
    }
}

impl VectorAccess for MmapFbinF32 {
    fn dim(&self) -> usize {
        self.dim()
    }

    fn row(&self, id: usize) -> Option<&[f32]> {
        self.row(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphRecordEncoding {
    SequentialDegreeOnly,
    ExplicitVertexIds,
}

struct MmapGraph {
    _file: File,
    mmap: Mmap,
    offsets: Vec<u64>,
    start: u32,
    max_degree: u32,
    encoding: GraphRecordEncoding,
    primary_counts: Option<HashPrunePrimaryCountSidecar>,
}

impl MmapGraph {
    fn open(path: &Path, expected_num_points: usize) -> anyhow::Result<Self> {
        let file = File::open(path).with_context(|| format!("open graph {:?}", path))?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.len() >= GRAPH_HEADER_BYTES as u64,
            "graph file {:?} is too small for a {}-byte header",
            path,
            GRAPH_HEADER_BYTES
        );
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let expected_file_size = read_u64_at(&mmap, 0)? as usize;
        let header_max_degree = read_u32_at(&mmap, 8)?;
        let start = read_u32_at(&mmap, 12)?;
        let _frozen_pts = read_u64_at(&mmap, 16)?;
        ensure!(
            expected_file_size == mmap.len(),
            "graph header expected_file_size {} != actual file size {} for {:?}",
            expected_file_size,
            mmap.len(),
            path
        );
        ensure!(
            start as usize <= expected_num_points.saturating_sub(1),
            "graph start node {} is outside expected {} points",
            start,
            expected_num_points
        );
        ensure!(
            (mmap.len() - GRAPH_HEADER_BYTES).is_multiple_of(std::mem::size_of::<u32>()),
            "graph payload size {} is not aligned to u32 records",
            mmap.len() - GRAPH_HEADER_BYTES
        );

        let (encoding, offsets, observed_max_degree) =
            scan_graph_offsets(&mmap, expected_num_points)?;
        ensure!(
            observed_max_degree == header_max_degree,
            "graph header max_degree {} != observed max_degree {}",
            header_max_degree,
            observed_max_degree
        );

        Ok(Self {
            _file: file,
            mmap,
            offsets,
            start,
            max_degree: observed_max_degree,
            encoding,
            primary_counts: None,
        })
    }

    fn num_nodes(&self) -> usize {
        self.offsets.len()
    }

    fn start(&self) -> u32 {
        self.start
    }

    fn max_degree(&self) -> u32 {
        self.max_degree
    }

    fn encoding_name(&self) -> &'static str {
        match self.encoding {
            GraphRecordEncoding::SequentialDegreeOnly => "sequential-degree-only",
            GraphRecordEncoding::ExplicitVertexIds => "explicit-vertex-ids",
        }
    }

    fn file_bytes(&self) -> usize {
        self.mmap.len()
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

    fn attach_primary_counts(&mut self, path: &Path) -> anyhow::Result<()> {
        ensure!(
            self.encoding == GraphRecordEncoding::SequentialDegreeOnly,
            "HashPrune primary-count sidecar requires sequential-degree-only graph encoding"
        );
        let sidecar = HashPrunePrimaryCountSidecar::load(path, self.num_nodes())?;
        let degrees = (0..self.num_nodes()).map(|source| {
            self.neighbors(source)
                .expect("offset scan already validated every graph row")
                .len() as u32
        });
        sidecar.validate_graph_degrees(self.start, degrees)?;
        self.primary_counts = Some(sidecar);
        Ok(())
    }
}

impl GraphAccess for MmapGraph {
    fn num_nodes(&self) -> usize {
        self.num_nodes()
    }

    fn neighbors(&self, id: usize) -> Option<&[u32]> {
        self.neighbors(id)
    }

    fn primary_count(&self, id: usize) -> Option<usize> {
        self.primary_counts.as_ref().map_or_else(
            || self.neighbors(id).map(<[u32]>::len),
            |sidecar| sidecar.primary_count(id),
        )
    }
}

const GRAPH_HEADER_BYTES: usize = 24;

fn scan_graph_offsets(
    mmap: &[u8],
    expected_num_points: usize,
) -> anyhow::Result<(GraphRecordEncoding, Vec<u64>, u32)> {
    if let Some(result) = try_scan_sequential_graph_offsets(mmap, expected_num_points)? {
        return Ok(result);
    }
    if let Some(result) = try_scan_explicit_graph_offsets(mmap, expected_num_points)? {
        return Ok(result);
    }
    bail!(
        "unable to determine graph record encoding for {} points",
        expected_num_points
    )
}

fn try_scan_sequential_graph_offsets(
    mmap: &[u8],
    expected_num_points: usize,
) -> anyhow::Result<Option<(GraphRecordEncoding, Vec<u64>, u32)>> {
    let mut offsets = Vec::with_capacity(expected_num_points);
    let mut offset = GRAPH_HEADER_BYTES;
    let mut max_degree = 0u32;

    for id in 0..expected_num_points {
        let Some(degree) = read_u32_at_opt(mmap, offset) else {
            return Ok(None);
        };
        let degree_usize = degree as usize;
        let record_bytes = match (1usize)
            .checked_add(degree_usize)
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

        let neighbors_offset = offset + std::mem::size_of::<u32>();
        for nbr_idx in 0..degree_usize {
            let nbr = read_u32_at(
                mmap,
                neighbors_offset + nbr_idx * std::mem::size_of::<u32>(),
            )?;
            if nbr as usize >= expected_num_points {
                return Ok(None);
            }
        }

        offsets.push(offset as u64);
        max_degree = max_degree.max(degree);
        offset += record_bytes;

        if id + 1 == expected_num_points && offset != mmap.len() {
            return Ok(None);
        }
    }

    if offset != mmap.len() {
        return Ok(None);
    }

    Ok(Some((
        GraphRecordEncoding::SequentialDegreeOnly,
        offsets,
        max_degree,
    )))
}

fn try_scan_explicit_graph_offsets(
    mmap: &[u8],
    expected_num_points: usize,
) -> anyhow::Result<Option<(GraphRecordEncoding, Vec<u64>, u32)>> {
    let mut offsets = vec![u64::MAX; expected_num_points];
    let mut seen_count = 0usize;
    let mut offset = GRAPH_HEADER_BYTES;
    let mut max_degree = 0u32;

    while offset < mmap.len() {
        if offset + 2 * std::mem::size_of::<u32>() > mmap.len() {
            return Ok(None);
        }
        let uid = read_u32_at(mmap, offset)? as usize;
        if uid >= expected_num_points || offsets[uid] != u64::MAX {
            return Ok(None);
        }
        let degree_offset = offset + std::mem::size_of::<u32>();
        let degree = read_u32_at(mmap, degree_offset)?;
        if degree == 0 {
            return Ok(None);
        }
        let degree_usize = degree as usize;
        let record_bytes = match (2usize)
            .checked_add(degree_usize)
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

        let neighbors_offset = offset + 2 * std::mem::size_of::<u32>();
        for nbr_idx in 0..degree_usize {
            let nbr = read_u32_at(
                mmap,
                neighbors_offset + nbr_idx * std::mem::size_of::<u32>(),
            )?;
            if nbr as usize >= expected_num_points {
                return Ok(None);
            }
        }

        offsets[uid] = degree_offset as u64;
        seen_count += 1;
        max_degree = max_degree.max(degree);
        offset += record_bytes;
    }

    if seen_count != expected_num_points
        || offsets
            .iter()
            .any(|&record_offset| record_offset == u64::MAX)
    {
        return Ok(None);
    }

    Ok(Some((
        GraphRecordEncoding::ExplicitVertexIds,
        offsets,
        max_degree,
    )))
}

fn read_u32_at(bytes: &[u8], offset: usize) -> anyhow::Result<u32> {
    let chunk = bytes
        .get(offset..offset + std::mem::size_of::<u32>())
        .context("u32 read offset is outside mapped file")?;
    Ok(u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_u32_at_opt(bytes: &[u8], offset: usize) -> Option<u32> {
    let chunk = bytes.get(offset..offset + std::mem::size_of::<u32>())?;
    Some(u32::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> anyhow::Result<u64> {
    let chunk = bytes
        .get(offset..offset + std::mem::size_of::<u64>())
        .context("u64 read offset is outside mapped file")?;
    Ok(u64::from_le_bytes(chunk.try_into().unwrap()))
}

struct SearchScratch {
    visited_epoch: Vec<u32>,
    epoch: u32,
    candidates: BinaryHeap<Reverse<(OrderedF32, u32)>>,
    best: BinaryHeap<(OrderedF32, u32)>,
}

impl SearchScratch {
    fn new(num_points: usize, l: usize) -> Self {
        Self {
            visited_epoch: vec![0; num_points],
            epoch: 0,
            candidates: BinaryHeap::with_capacity(l.max(1)),
            best: BinaryHeap::with_capacity(l.max(1)),
        }
    }

    fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited_epoch.fill(0);
            self.epoch = 1;
        }
        self.candidates.clear();
        self.best.clear();
    }

    fn mark_visited(&mut self, id: usize) -> bool {
        if id >= self.visited_epoch.len() || self.visited_epoch[id] == self.epoch {
            return false;
        }
        self.visited_epoch[id] = self.epoch;
        true
    }
}

fn split_search_outcomes(
    outcomes: Vec<SearchOutcome>,
    elapsed: std::time::Duration,
) -> anyhow::Result<(
    Vec<Vec<u32>>,
    Vec<Vec<bool>>,
    Vec<SearchStats>,
    std::time::Duration,
)> {
    let mut ids = Vec::with_capacity(outcomes.len());
    let mut flags = Vec::with_capacity(outcomes.len());
    let mut stats = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        ensure!(
            outcome.ids.len() == outcome.first_discovery_via_backfill.len(),
            "search outcome provenance length does not match result length"
        );
        ids.push(outcome.ids);
        flags.push(outcome.first_discovery_via_backfill);
        stats.push(outcome.stats);
    }
    Ok((ids, flags, stats, elapsed))
}

/// Greedy BFS search on the graph (standard Vamana/DiskANN algorithm).
#[cfg(test)]
fn search_one(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    start: u32,
    k: usize,
    l: usize,
) -> Vec<u32> {
    search_one_with_stats(query, data, dim, graph, start, k, l).ids
}

fn search_one_with_stats(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &[Vec<u32>],
    start: u32,
    k: usize,
    l: usize,
) -> SearchOutcome {
    let l_eff = l.max(k + 1);
    // Max-heap of (-dist, id) => gives us the farthest element on top
    // We use (Reverse(dist), id) in a max-heap => smallest -dist (i.e. farthest) on top
    // Actually: BinaryHeap is max-heap. With Reverse<OrderedFloat>, smaller floats pop first.
    // We want to pop the farthest when trimming, so use Reverse for the candidates.

    let mut stats = SearchStats::default();
    let mut visited = vec![false; graph.len()];
    visited[start as usize] = true;
    stats.visited_nodes += 1;

    let d0 = l2_distance(
        query,
        &data[start as usize * dim..(start as usize + 1) * dim],
    );
    stats.distance_comparisons += 1;

    // Candidates: min-heap by distance (closest first to expand)
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l_eff);
    candidates.push(Reverse((OrderedF32(d0), start)));

    // Best results: max-heap by distance (farthest on top, for trimming)
    let mut best: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l_eff);
    best.push((OrderedF32(d0), start));

    while let Some(Reverse((cd, cid))) = candidates.pop() {
        // Prune: if closest candidate is farther than L-th best, stop
        if best.len() >= l_eff {
            if let Some(&(OrderedF32(worst), _)) = best.peek() {
                if cd.0 > worst {
                    stats.early_stopped = true;
                    break;
                }
            }
        }
        stats.expanded_nodes += 1;

        let neighbors = &graph[cid as usize];
        for &nbr in neighbors {
            let ni = nbr as usize;
            if ni >= visited.len() || visited[ni] {
                continue;
            }
            visited[ni] = true;
            stats.visited_nodes += 1;
            let dist = l2_distance(query, &data[ni * dim..(ni + 1) * dim]);
            stats.distance_comparisons += 1;

            candidates.push(Reverse((OrderedF32(dist), nbr)));
            best.push((OrderedF32(dist), nbr));

            while best.len() > l_eff {
                best.pop();
            }
        }
    }

    // Sort by distance, take top-k
    let mut sorted: Vec<(OrderedF32, u32)> = best.into_iter().collect();
    sorted.sort_by(|a, b| {
        a.0.0
            .partial_cmp(&b.0.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let result_len = sorted.len().min(k);
    SearchOutcome {
        ids: sorted.into_iter().take(k).map(|(_, id)| id).collect(),
        first_discovery_via_backfill: vec![false; result_len],
        stats,
    }
}

#[cfg(test)]
fn search_one_mmap(
    query: &[f32],
    data: &MmapFbinF32,
    graph: &MmapGraph,
    k: usize,
    l: usize,
    scratch: &mut SearchScratch,
) -> Vec<u32> {
    search_one_mmap_with_stats(query, data, graph, k, l, scratch).ids
}

fn search_one_mmap_with_stats(
    query: &[f32],
    data: &MmapFbinF32,
    graph: &MmapGraph,
    k: usize,
    l: usize,
    scratch: &mut SearchScratch,
) -> SearchOutcome {
    let l_eff = l.max(k + 1);
    scratch.reset();
    let mut stats = SearchStats::default();

    let start = graph.start();
    if !scratch.mark_visited(start as usize) {
        return SearchOutcome {
            ids: Vec::new(),
            first_discovery_via_backfill: Vec::new(),
            stats,
        };
    }
    stats.visited_nodes += 1;

    let Some(start_row) = data.row(start as usize) else {
        return SearchOutcome {
            ids: Vec::new(),
            first_discovery_via_backfill: Vec::new(),
            stats,
        };
    };
    let d0 = l2_distance(query, start_row);
    stats.distance_comparisons += 1;
    scratch.candidates.push(Reverse((OrderedF32(d0), start)));
    scratch.best.push((OrderedF32(d0), start));

    while let Some(Reverse((cd, cid))) = scratch.candidates.pop() {
        if scratch.best.len() >= l_eff {
            if let Some(&(OrderedF32(worst), _)) = scratch.best.peek() {
                if cd.0 > worst {
                    stats.early_stopped = true;
                    break;
                }
            }
        }
        stats.expanded_nodes += 1;

        let Some(neighbors) = graph.neighbors(cid as usize) else {
            continue;
        };
        for &nbr in neighbors {
            let ni = nbr as usize;
            if !scratch.mark_visited(ni) {
                continue;
            }
            stats.visited_nodes += 1;
            let Some(row) = data.row(ni) else {
                continue;
            };
            let dist = l2_distance(query, row);
            stats.distance_comparisons += 1;

            scratch.candidates.push(Reverse((OrderedF32(dist), nbr)));
            scratch.best.push((OrderedF32(dist), nbr));

            while scratch.best.len() > l_eff {
                scratch.best.pop();
            }
        }
    }

    let mut sorted: Vec<(OrderedF32, u32)> = scratch.best.iter().copied().collect();
    sorted.sort_by(|a, b| {
        a.0.0
            .partial_cmp(&b.0.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let result_len = sorted.len().min(k);
    SearchOutcome {
        ids: sorted.into_iter().take(k).map(|(_, id)| id).collect(),
        first_discovery_via_backfill: vec![false; result_len],
        stats,
    }
}

fn parallel_search_mmap(
    queries: &[f32],
    data: &MmapFbinF32,
    graph: &MmapGraph,
    k: usize,
    l: usize,
    num_queries: usize,
    num_threads: usize,
) -> Vec<SearchOutcome> {
    let results: std::sync::Mutex<Vec<Option<SearchOutcome>>> =
        std::sync::Mutex::new(vec![None; num_queries]);

    std::thread::scope(|s| {
        let chunk = (num_queries + num_threads - 1) / num_threads;
        for tid in 0..num_threads {
            let lo = tid * chunk;
            let hi = (lo + chunk).min(num_queries);
            if lo >= hi {
                continue;
            }
            let results = &results;
            s.spawn(move || {
                let mut scratch = SearchScratch::new(graph.num_nodes(), l);
                for qi in lo..hi {
                    let query = &queries[qi * data.dim()..(qi + 1) * data.dim()];
                    let result = search_one_mmap_with_stats(query, data, graph, k, l, &mut scratch);
                    results.lock().unwrap()[qi] = Some(result);
                }
            });
        }
    });

    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.unwrap())
        .collect()
}

fn search_one_rabitq1(
    query: &[f32],
    data: &MmapFbinF32,
    graph: &MmapGraph,
    sidecar: &Rabitq1Sidecar,
    backend: SearchBackend,
    k: usize,
    l: usize,
    scratch: &mut RabitqSearchScratch,
) -> anyhow::Result<SearchOutcome> {
    let query_started = std::time::Instant::now();
    let prepared = sidecar.prepare_query(query)?;
    let query_preprocess_ns = duration_ns(query_started.elapsed());
    let mut estimator = Rabitq1Estimator::new(prepared, sidecar.center_ids(), sidecar.bin_data());
    let mut outcome = match backend {
        SearchBackend::Rabitq1PostRerank => search_post_rerank(
            graph,
            data,
            query,
            graph.start(),
            k,
            l,
            &mut estimator,
            scratch,
        )?,
        SearchBackend::Rabitq1AdaptiveExact => search_adaptive_exact(
            graph,
            data,
            query,
            graph.start(),
            k,
            l,
            &mut estimator,
            scratch,
        )?,
        SearchBackend::Exact => bail!("exact backend cannot use the RaBitQ1 query path"),
    };
    outcome.stats.query_preprocess_ns = query_preprocess_ns;
    Ok(outcome)
}

fn parallel_search_rabitq1(
    queries: &[f32],
    data: &MmapFbinF32,
    graph: &MmapGraph,
    sidecar: &Arc<Rabitq1Sidecar>,
    backend: SearchBackend,
    k: usize,
    l: usize,
    num_queries: usize,
    num_threads: usize,
) -> anyhow::Result<Vec<SearchOutcome>> {
    let chunks = std::thread::scope(
        |scope| -> anyhow::Result<Vec<Vec<(usize, SearchOutcome)>>> {
            let chunk_size = num_queries.div_ceil(num_threads);
            let mut handles = Vec::new();
            for thread_id in 0..num_threads {
                let lo = thread_id * chunk_size;
                let hi = (lo + chunk_size).min(num_queries);
                if lo >= hi {
                    continue;
                }
                let sidecar = Arc::clone(sidecar);
                handles.push(scope.spawn(
                    move || -> anyhow::Result<Vec<(usize, SearchOutcome)>> {
                        let mut scratch = RabitqSearchScratch::new(graph.num_nodes(), l, k)?;
                        let mut outcomes = Vec::with_capacity(hi - lo);
                        for query_id in lo..hi {
                            let query =
                                &queries[query_id * data.dim()..(query_id + 1) * data.dim()];
                            outcomes.push((
                                query_id,
                                search_one_rabitq1(
                                    query,
                                    data,
                                    graph,
                                    &sidecar,
                                    backend,
                                    k,
                                    l,
                                    &mut scratch,
                                )?,
                            ));
                        }
                        Ok(outcomes)
                    },
                ));
            }

            let mut chunks = Vec::with_capacity(handles.len());
            for handle in handles {
                chunks.push(
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("RaBitQ1 search worker panicked"))??,
                );
            }
            Ok(chunks)
        },
    )?;

    let mut ordered = std::iter::repeat_with(|| None)
        .take(num_queries)
        .collect::<Vec<Option<SearchOutcome>>>();
    for chunk in chunks {
        for (query_id, outcome) in chunk {
            ordered[query_id] = Some(outcome);
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(query_id, outcome)| {
            outcome.ok_or_else(|| anyhow::anyhow!("missing RaBitQ1 result for query {query_id}"))
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct OrderedF32(f32);

impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SearchStatsSummary {
    avg_visited_nodes: f64,
    p95_visited_nodes: usize,
    avg_expanded_nodes: f64,
    p95_expanded_nodes: usize,
    avg_distance_comparisons: f64,
    p95_distance_comparisons: usize,
    early_stopped_queries: usize,
}

fn summarize_search_stats(stats: &[SearchStats]) -> SearchStatsSummary {
    if stats.is_empty() {
        return SearchStatsSummary::default();
    }

    let mut visited = Vec::with_capacity(stats.len());
    let mut expanded = Vec::with_capacity(stats.len());
    let mut distance_comparisons = Vec::with_capacity(stats.len());
    let mut early_stopped_queries = 0usize;

    for item in stats {
        visited.push(item.visited_nodes);
        expanded.push(item.expanded_nodes);
        distance_comparisons.push(item.distance_comparisons);
        early_stopped_queries += usize::from(item.early_stopped);
    }

    SearchStatsSummary {
        avg_visited_nodes: average_usize(&visited),
        p95_visited_nodes: percentile_usize(&mut visited, 0.95),
        avg_expanded_nodes: average_usize(&expanded),
        p95_expanded_nodes: percentile_usize(&mut expanded, 0.95),
        avg_distance_comparisons: average_usize(&distance_comparisons),
        p95_distance_comparisons: percentile_usize(&mut distance_comparisons, 0.95),
        early_stopped_queries,
    }
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

fn write_search_metrics_tsv(
    path: &Path,
    storage: StorageMode,
    num_queries: usize,
    recall_at: usize,
    search_list_size: usize,
    recall: f64,
    search_seconds: f64,
    stats: &[SearchStats],
    search_backend: SearchBackend,
    sidecar_load_seconds: f64,
    sidecar_resident_bytes: usize,
    dim: usize,
    graph_encoding: &str,
    graph_bytes: u64,
    backfill_final_gt_hits: &[u64],
) -> anyhow::Result<()> {
    ensure!(
        backfill_final_gt_hits.len() == stats.len(),
        "backfill GT-hit count does not match query count"
    );
    let summary = summarize_search_stats(stats);
    let total_distance_comparisons = stats
        .iter()
        .map(|item| item.distance_comparisons)
        .sum::<usize>();
    let recall_per_1k_distance_comparisons = if total_distance_comparisons == 0 {
        0.0
    } else {
        recall * 1000.0 * num_queries as f64 / total_distance_comparisons as f64
    };
    let qps = if search_seconds > 0.0 {
        num_queries as f64 / search_seconds
    } else {
        0.0
    };
    let total_one_bit_evaluations = stats
        .iter()
        .map(|item| item.one_bit_evaluations)
        .sum::<u64>();
    let total_lower_bound_rejects = stats
        .iter()
        .map(|item| item.lower_bound_rejects)
        .sum::<u64>();
    let total_navigation_exact_evaluations = if search_backend == SearchBackend::Exact {
        total_distance_comparisons as u64
    } else {
        stats
            .iter()
            .map(|item| item.navigation_exact_evaluations)
            .sum::<u64>()
    };
    let total_adaptive_exact_refinements = stats
        .iter()
        .map(|item| item.adaptive_exact_refinements)
        .sum::<u64>();
    let total_post_rerank_exact_evaluations = stats
        .iter()
        .map(|item| item.post_rerank_exact_evaluations)
        .sum::<u64>();
    let total_exact_evaluations = if search_backend == SearchBackend::Exact {
        total_distance_comparisons as u64
    } else {
        stats
            .iter()
            .map(|item| item.total_exact_evaluations)
            .sum::<u64>()
    };
    let logical_exact_vector_bytes = total_exact_evaluations
        .checked_mul(dim as u64)
        .and_then(|value| value.checked_mul(4))
        .context("logical exact-vector byte count overflow")?;
    let ns_to_seconds = |total: u64| total as f64 / 1_000_000_000.0;
    let query_preprocess_seconds = ns_to_seconds(
        stats
            .iter()
            .map(|item| item.query_preprocess_ns)
            .sum::<u64>(),
    );
    let one_bit_estimate_seconds = ns_to_seconds(
        stats
            .iter()
            .map(|item| item.one_bit_estimate_ns)
            .sum::<u64>(),
    );
    let exact_l2_seconds = ns_to_seconds(stats.iter().map(|item| item.exact_l2_ns).sum::<u64>());
    let navigation_seconds =
        ns_to_seconds(stats.iter().map(|item| item.navigation_ns).sum::<u64>());
    let sum_counter =
        |select: fn(&SearchStats) -> u64| stats.iter().map(select).fold(0_u64, u64::saturating_add);
    let backfill_edges_scanned_total = sum_counter(|item| item.backfill_edges_scanned);
    let backfill_first_discoveries_total = sum_counter(|item| item.backfill_first_discoveries);
    let backfill_beam_admissions_total = sum_counter(|item| item.backfill_beam_admissions);
    let backfill_nodes_expanded_total = sum_counter(|item| item.backfill_nodes_expanded);
    let backfill_exact_refinements_total = sum_counter(|item| item.backfill_exact_refinements);
    let backfill_lower_bound_rejects_total = sum_counter(|item| item.backfill_lower_bound_rejects);
    let backfill_final_top10_total = sum_counter(|item| item.backfill_final_top10);
    let backfill_final_gt_hits_total = backfill_final_gt_hits
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let queries_using_backfill = stats
        .iter()
        .filter(|item| item.backfill_first_discoveries > 0)
        .count();

    let mut writer = std::io::BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "storage\tqueries\trecall_at\tsearch_list_size\trecall\tsearch_seconds\tqps\tavg_visited_nodes\tp95_visited_nodes\tavg_expanded_nodes\tp95_expanded_nodes\tavg_distance_comparisons\tp95_distance_comparisons\tearly_stopped_queries\trecall_per_1k_distance_comparisons\tsearch_backend\tsidecar_load_seconds\tsidecar_resident_bytes\ttotal_one_bit_evaluations\ttotal_lower_bound_rejects\ttotal_navigation_exact_evaluations\ttotal_adaptive_exact_refinements\ttotal_post_rerank_exact_evaluations\ttotal_exact_evaluations\tlogical_exact_vector_bytes\tquery_preprocess_seconds\tone_bit_estimate_seconds\texact_l2_seconds\tnavigation_seconds\tgraph_encoding\tgraph_bytes\tbackfill_edges_scanned_total\tbackfill_first_discoveries_total\tbackfill_beam_admissions_total\tbackfill_nodes_expanded_total\tbackfill_exact_refinements_total\tbackfill_lower_bound_rejects_total\tbackfill_final_top10_total\tbackfill_final_gt_hits_total\tqueries_using_backfill"
    )?;
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.3}\t{:.3}\t{}\t{:.3}\t{}\t{:.3}\t{}\t{}\t{:.6}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        storage.as_str(),
        num_queries,
        recall_at,
        search_list_size,
        recall,
        search_seconds,
        qps,
        summary.avg_visited_nodes,
        summary.p95_visited_nodes,
        summary.avg_expanded_nodes,
        summary.p95_expanded_nodes,
        summary.avg_distance_comparisons,
        summary.p95_distance_comparisons,
        summary.early_stopped_queries,
        recall_per_1k_distance_comparisons,
        search_backend.as_str(),
        sidecar_load_seconds,
        sidecar_resident_bytes,
        total_one_bit_evaluations,
        total_lower_bound_rejects,
        total_navigation_exact_evaluations,
        total_adaptive_exact_refinements,
        total_post_rerank_exact_evaluations,
        total_exact_evaluations,
        logical_exact_vector_bytes,
        query_preprocess_seconds,
        one_bit_estimate_seconds,
        exact_l2_seconds,
        navigation_seconds,
        graph_encoding,
        graph_bytes,
        backfill_edges_scanned_total,
        backfill_first_discoveries_total,
        backfill_beam_admissions_total,
        backfill_nodes_expanded_total,
        backfill_exact_refinements_total,
        backfill_lower_bound_rejects_total,
        backfill_final_top10_total,
        backfill_final_gt_hits_total,
        queries_using_backfill,
    )?;
    writer.flush()?;
    Ok(())
}

fn write_query_metrics_tsv(
    path: &Path,
    search_list_size: usize,
    dim: usize,
    gt_hits: &[usize],
    backfill_final_gt_hits: &[u64],
    stats: &[SearchStats],
) -> anyhow::Result<()> {
    ensure!(
        gt_hits.len() == stats.len() && backfill_final_gt_hits.len() == stats.len(),
        "per-query metric inputs have inconsistent query counts"
    );
    let mut writer = std::io::BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "query_id\tL\tgt_hits\tvisited_nodes\texpanded_nodes\tdistance_comparisons\tone_bit_evaluations\tlower_bound_rejects\tnavigation_exact_evaluations\tadaptive_exact_refinements\tpost_rerank_exact_evaluations\ttotal_exact_evaluations\tlogical_exact_vector_bytes\tquery_preprocess_ns\tone_bit_estimate_ns\texact_l2_ns\tnavigation_ns\tbackfill_edges_scanned\tbackfill_first_discoveries\tbackfill_beam_admissions\tbackfill_nodes_expanded\tbackfill_exact_refinements\tbackfill_lower_bound_rejects\tbackfill_final_top10\tbackfill_final_gt_hits\tquery_used_backfill\tfirst_backfill_admission_expansion_ordinal"
    )?;
    for (query_id, item) in stats.iter().enumerate() {
        let logical_exact_vector_bytes = item
            .total_exact_evaluations
            .checked_mul(dim as u64)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
            .context("per-query logical exact-vector byte count overflow")?;
        let first_ordinal = item
            .first_backfill_admission_expansion_ordinal
            .map_or_else(|| "NA".to_owned(), |value| value.to_string());
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            query_id,
            search_list_size,
            gt_hits[query_id],
            item.visited_nodes,
            item.expanded_nodes,
            item.distance_comparisons,
            item.one_bit_evaluations,
            item.lower_bound_rejects,
            item.navigation_exact_evaluations,
            item.adaptive_exact_refinements,
            item.post_rerank_exact_evaluations,
            item.total_exact_evaluations,
            logical_exact_vector_bytes,
            item.query_preprocess_ns,
            item.one_bit_estimate_ns,
            item.exact_l2_ns,
            item.navigation_ns,
            item.backfill_edges_scanned,
            item.backfill_first_discoveries,
            item.backfill_beam_admissions,
            item.backfill_nodes_expanded,
            item.backfill_exact_refinements,
            item.backfill_lower_bound_rejects,
            item.backfill_final_top10,
            backfill_final_gt_hits[query_id],
            u8::from(item.backfill_first_discoveries > 0),
            first_ordinal,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

fn save_result_fbin(path: &PathBuf, results: &[Vec<u32>], k: usize) -> anyhow::Result<()> {
    let n = results.len() as u32;
    let mut writer = std::io::BufWriter::new(File::create(path)?);
    writer.write_u32::<LittleEndian>(n)?;
    writer.write_u32::<LittleEndian>(k as u32)?;
    for result in results {
        for i in 0..k {
            writer.write_u32::<LittleEndian>(result.get(i).copied().unwrap_or(0))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shear_core::model::hashprune_provenance::{
        HashPruneGraphEncoding, HashPrunePrimaryCountWriter,
    };
    use shear_core::model::rabitq::graph_search::{BuildRabitq1Options, build_rabitq1_sidecar};
    use tempfile::tempdir;

    fn minimal_args() -> Vec<&'static str> {
        vec![
            "search_shear_index",
            "--index-path-prefix",
            "/tmp/index",
            "--data-path",
            "/tmp/base.fbin",
            "--query-file",
            "/tmp/query.fbin",
            "--result-path",
            "/tmp/result.bin",
            "--gt-file",
            "/tmp/gt.bin",
        ]
    }

    fn diagnostic_cli_args() -> Vec<&'static str> {
        let mut argv = minimal_args();
        argv.extend([
            "--storage",
            "mmap",
            "--search-backend",
            "rabitq1-adaptive-exact",
            "--rabitq1-sidecar",
            "/tmp/base.rabitq1",
            "--hashprune-primary-count-sidecar",
            "/tmp/primary_count.sidecar",
            "--query-metrics-path",
            "/tmp/query.tsv",
        ]);
        argv
    }

    fn without_option_pair(mut argv: Vec<&'static str>, option: &'static str) -> Vec<&'static str> {
        let index = argv.iter().position(|value| *value == option).unwrap();
        argv.drain(index..=index + 1);
        argv
    }

    #[test]
    fn primary_count_sidecar_requires_adaptive_mmap_and_query_metrics() {
        let valid = Args::try_parse_from(diagnostic_cli_args()).unwrap();
        assert!(validate_args(&valid).is_ok());

        let missing_metrics = without_option_pair(diagnostic_cli_args(), "--query-metrics-path");
        assert!(validate_args(&Args::try_parse_from(missing_metrics).unwrap()).is_err());

        let mut exact = Args::try_parse_from(diagnostic_cli_args()).unwrap();
        exact.search_backend = SearchBackend::Exact;
        assert!(validate_args(&exact).is_err());
        let mut memory = Args::try_parse_from(diagnostic_cli_args()).unwrap();
        memory.storage = StorageMode::Memory;
        assert!(validate_args(&memory).is_err());
    }

    #[test]
    fn query_metrics_without_provenance_is_valid_for_standard_runs() {
        let argv = without_option_pair(diagnostic_cli_args(), "--hashprune-primary-count-sidecar");
        let args = Args::try_parse_from(argv).unwrap();
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn rabitq1_backend_requires_sidecar_and_mmap() {
        let mut args = minimal_args();
        args.extend(["--search-backend", "rabitq1-post-rerank"]);
        let parsed = Args::try_parse_from(args).unwrap();
        assert!(
            validate_args(&parsed)
                .unwrap_err()
                .to_string()
                .contains("sidecar")
        );

        let mut args = minimal_args();
        args.extend([
            "--search-backend",
            "rabitq1-adaptive-exact",
            "--rabitq1-sidecar",
            "/tmp/test.rabitq1",
            "--storage",
            "memory",
        ]);
        let parsed = Args::try_parse_from(args).unwrap();
        assert!(
            validate_args(&parsed)
                .unwrap_err()
                .to_string()
                .contains("mmap")
        );
    }

    #[test]
    fn search_cli_rejects_invalid_common_and_exact_only_options() {
        let mut args = minimal_args();
        args.extend(["--recall-at", "11", "--search-list-size", "10"]);
        assert!(validate_args(&Args::try_parse_from(args).unwrap()).is_err());

        let mut args = minimal_args();
        args.extend(["--num-threads", "0"]);
        assert!(validate_args(&Args::try_parse_from(args).unwrap()).is_err());

        let mut args = minimal_args();
        args.extend(["--rabitq1-sidecar", "/tmp/test.rabitq1"]);
        assert!(validate_args(&Args::try_parse_from(args).unwrap()).is_err());
    }

    fn write_fbin(path: &std::path::Path, rows: &[&[f32]]) {
        let mut writer = std::io::BufWriter::new(File::create(path).unwrap());
        writer.write_u32::<LittleEndian>(rows.len() as u32).unwrap();
        writer
            .write_u32::<LittleEndian>(rows[0].len() as u32)
            .unwrap();
        for row in rows {
            for &value in *row {
                writer.write_f32::<LittleEndian>(value).unwrap();
            }
        }
    }

    fn write_sequential_graph(path: &std::path::Path, start: u32, graph: &[Vec<u32>]) {
        let graph_bytes: usize = graph.iter().map(|neighbors| 4 + neighbors.len() * 4).sum();
        let mut writer = std::io::BufWriter::new(File::create(path).unwrap());
        writer
            .write_u64::<LittleEndian>((24 + graph_bytes) as u64)
            .unwrap();
        writer
            .write_u32::<LittleEndian>(graph.iter().map(Vec::len).max().unwrap_or(0) as u32)
            .unwrap();
        writer.write_u32::<LittleEndian>(start).unwrap();
        writer.write_u64::<LittleEndian>(0).unwrap();
        for neighbors in graph {
            writer
                .write_u32::<LittleEndian>(neighbors.len() as u32)
                .unwrap();
            for &neighbor in neighbors {
                writer.write_u32::<LittleEndian>(neighbor).unwrap();
            }
        }
    }

    fn write_explicit_graph(path: &std::path::Path, start: u32, graph: &[Vec<u32>]) {
        let graph_bytes: usize = graph.iter().map(|neighbors| 8 + neighbors.len() * 4).sum();
        let mut writer = std::io::BufWriter::new(File::create(path).unwrap());
        writer
            .write_u64::<LittleEndian>((24 + graph_bytes) as u64)
            .unwrap();
        writer
            .write_u32::<LittleEndian>(graph.iter().map(Vec::len).max().unwrap_or(0) as u32)
            .unwrap();
        writer.write_u32::<LittleEndian>(start).unwrap();
        writer.write_u64::<LittleEndian>(0).unwrap();
        for (uid, neighbors) in graph.iter().enumerate() {
            writer.write_u32::<LittleEndian>(uid as u32).unwrap();
            writer
                .write_u32::<LittleEndian>(neighbors.len() as u32)
                .unwrap();
            for &neighbor in neighbors {
                writer.write_u32::<LittleEndian>(neighbor).unwrap();
            }
        }
    }

    fn build_rabitq1_fixture(
        dir: &std::path::Path,
    ) -> (PathBuf, PathBuf, PathBuf, Arc<Rabitq1Sidecar>, Vec<f32>) {
        let data_path = dir.join("data.fbin");
        let sequential_path = dir.join("sequential.index");
        let explicit_path = dir.join("explicit.index");
        let sidecar_path = dir.join("data.rabitq1");
        let rows = (0..16)
            .map(|id| {
                let mut row = vec![0.0_f32; 64];
                row[0] = id as f32;
                row[1] = (id * id) as f32 / 16.0;
                row[2] = 1.0;
                row
            })
            .collect::<Vec<_>>();
        let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        write_fbin(&data_path, &row_refs);
        let graph = (0..16)
            .map(|id| {
                (0..16)
                    .filter(|neighbor| *neighbor != id)
                    .map(|neighbor| neighbor as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        write_sequential_graph(&sequential_path, 0, &graph);
        write_explicit_graph(&explicit_path, 0, &graph);
        build_rabitq1_sidecar(
            &data_path,
            &sidecar_path,
            &BuildRabitq1Options {
                center_count: 16,
                sample_rows: 16,
                kmeans_iterations: 20,
                seed: 42,
                batch_rows: 5,
                threads: 2,
            },
        )
        .unwrap();
        let sidecar = Arc::new(Rabitq1Sidecar::load(&sidecar_path, &data_path).unwrap());
        let mut queries = vec![0.0_f32; 2 * 64];
        queries[0] = 12.4;
        queries[1] = 9.0;
        queries[2] = 1.0;
        queries[64] = 3.2;
        queries[65] = 0.7;
        queries[66] = 1.0;
        (data_path, sequential_path, explicit_path, sidecar, queries)
    }

    fn write_primary_count_sidecar(
        path: &Path,
        graph: &MmapGraph,
        primary_count: usize,
        graph_start: u32,
    ) {
        let mut writer = HashPrunePrimaryCountWriter::create(
            path,
            graph.num_nodes(),
            graph.max_degree() as usize,
        )
        .unwrap();
        for source in 0..graph.num_nodes() {
            let degree = graph.neighbors(source).unwrap().len();
            writer
                .write_row(source as u64, primary_count.min(degree), degree)
                .unwrap();
        }
        writer
            .finish(graph_start, HashPruneGraphEncoding::SequentialDegreeOnly)
            .unwrap();
    }

    fn integer_counters(stats: &[SearchStats]) -> Vec<[u64; 10]> {
        stats
            .iter()
            .map(|item| {
                [
                    item.visited_nodes as u64,
                    item.expanded_nodes as u64,
                    item.distance_comparisons as u64,
                    item.early_stopped as u64,
                    item.one_bit_evaluations,
                    item.lower_bound_rejects,
                    item.navigation_exact_evaluations,
                    item.adaptive_exact_refinements,
                    item.post_rerank_exact_evaluations,
                    item.total_exact_evaluations,
                ]
            })
            .collect()
    }

    #[test]
    fn mmap_fbin_exposes_rows_without_loading_payload() {
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("data.fbin");
        write_fbin(
            &data_path,
            &[&[0.0, 0.0], &[1.0, 0.0], &[0.0, 2.0], &[3.0, 4.0]],
        );

        let data = MmapFbinF32::open(&data_path).unwrap();

        assert_eq!(data.num_points(), 4);
        assert_eq!(data.dim(), 2);
        assert_eq!(data.row(1).unwrap(), &[1.0, 0.0]);
        assert_eq!(data.row(3).unwrap(), &[3.0, 4.0]);
        assert!(data.row(4).is_none());
    }

    #[test]
    fn mmap_graph_exposes_sequential_neighbors_by_offset() {
        let dir = tempdir().unwrap();
        let graph_path = dir.path().join("_mem.index");
        let graph = vec![vec![1, 2], vec![2], vec![0]];
        write_sequential_graph(&graph_path, 0, &graph);

        let mmap_graph = MmapGraph::open(&graph_path, graph.len()).unwrap();

        assert_eq!(mmap_graph.num_nodes(), 3);
        assert_eq!(mmap_graph.start(), 0);
        assert_eq!(mmap_graph.max_degree(), 2);
        assert_eq!(mmap_graph.neighbors(0).unwrap(), &[1, 2]);
        assert_eq!(mmap_graph.neighbors(1).unwrap(), &[2]);
        assert_eq!(mmap_graph.neighbors(2).unwrap(), &[0]);
        assert!(mmap_graph.neighbors(3).is_none());
    }

    #[test]
    fn mmap_graph_accepts_zero_degree_sequential_row() {
        let dir = tempdir().unwrap();
        let graph_path = dir.path().join("_mem.index");
        let graph = vec![vec![1], vec![], vec![0]];
        write_sequential_graph(&graph_path, 0, &graph);

        let mmap_graph = MmapGraph::open(&graph_path, graph.len()).unwrap();

        assert_eq!(mmap_graph.neighbors(0).unwrap(), &[1]);
        assert_eq!(mmap_graph.neighbors(1).unwrap(), &[] as &[u32]);
        assert_eq!(mmap_graph.neighbors(2).unwrap(), &[0]);
    }

    #[test]
    fn mmap_graph_exposes_explicit_id_neighbors_by_offset() {
        let dir = tempdir().unwrap();
        let graph_path = dir.path().join("_mem.index");
        let graph = vec![vec![1, 2], vec![2], vec![0]];
        write_explicit_graph(&graph_path, 0, &graph);

        let mmap_graph = MmapGraph::open(&graph_path, graph.len()).unwrap();

        assert_eq!(mmap_graph.neighbors(0).unwrap(), &[1, 2]);
        assert_eq!(mmap_graph.neighbors(1).unwrap(), &[2]);
        assert_eq!(mmap_graph.neighbors(2).unwrap(), &[0]);
    }

    #[test]
    fn mmap_graph_rejects_a_neighbor_outside_the_dataset() {
        let dir = tempdir().unwrap();
        let graph_path = dir.path().join("_mem.index");
        write_sequential_graph(&graph_path, 0, &[vec![1], vec![2], vec![3]]);

        assert!(MmapGraph::open(&graph_path, 3).is_err());
    }

    #[test]
    fn mmap_search_matches_in_memory_search_on_small_graph() {
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("data.fbin");
        let graph_path = dir.path().join("_mem.index");
        let graph = vec![vec![1, 2], vec![0, 3], vec![3], vec![1]];
        let rows: [&[f32]; 4] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0], &[2.0, 0.0]];
        write_fbin(&data_path, &rows);
        write_sequential_graph(&graph_path, 0, &graph);

        let data_vec = load_fbin_f32(&data_path).unwrap();
        let mmap_data = MmapFbinF32::open(&data_path).unwrap();
        let mmap_graph = MmapGraph::open(&graph_path, graph.len()).unwrap();
        let query = [1.8, 0.0];

        let memory_result = search_one(&query, &data_vec, 2, &graph, 0, 2, 4);
        let mut scratch = SearchScratch::new(graph.len(), 4);
        let mmap_result = search_one_mmap(&query, &mmap_data, &mmap_graph, 2, 4, &mut scratch);

        assert_eq!(mmap_result, memory_result);
    }

    #[test]
    fn mmap_search_reports_navigation_stats() {
        let dir = tempdir().unwrap();
        let data_path = dir.path().join("data.fbin");
        let graph_path = dir.path().join("_mem.index");
        let graph = vec![vec![1, 2], vec![0, 3], vec![3], vec![1]];
        let rows: [&[f32]; 4] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0], &[2.0, 0.0]];
        write_fbin(&data_path, &rows);
        write_sequential_graph(&graph_path, 0, &graph);

        let mmap_data = MmapFbinF32::open(&data_path).unwrap();
        let mmap_graph = MmapGraph::open(&graph_path, graph.len()).unwrap();
        let query = [1.8, 0.0];
        let mut scratch = SearchScratch::new(graph.len(), 4);

        let outcome =
            search_one_mmap_with_stats(&query, &mmap_data, &mmap_graph, 2, 4, &mut scratch);

        assert_eq!(outcome.ids, vec![3, 1]);
        assert_eq!(outcome.stats.visited_nodes, 4);
        assert_eq!(outcome.stats.distance_comparisons, 4);
        assert_eq!(outcome.stats.expanded_nodes, 4);
        assert!(!outcome.stats.early_stopped);
    }

    #[test]
    fn rabitq1_small_file_backends_are_encoding_and_thread_deterministic() {
        let dir = tempdir().unwrap();
        let (data_path, sequential_path, explicit_path, sidecar, queries) =
            build_rabitq1_fixture(dir.path());
        let data = MmapFbinF32::open(&data_path).unwrap();
        let sequential = MmapGraph::open(&sequential_path, 16).unwrap();
        let explicit = MmapGraph::open(&explicit_path, 16).unwrap();
        for query in queries.chunks_exact(64) {
            let mut prepared = sidecar.prepare_query(query).unwrap();
            let stride = sidecar.header().bin_stride as usize;
            for id in 0..16 {
                prepared
                    .reference_estimate(
                        sidecar.center_ids()[id],
                        &sidecar.bin_data()[id * stride..(id + 1) * stride],
                    )
                    .unwrap_or_else(|error| {
                        panic!("reference estimate failed for ID {id}: {error}")
                    });
            }
        }

        for backend in [
            SearchBackend::Rabitq1PostRerank,
            SearchBackend::Rabitq1AdaptiveExact,
        ] {
            let sequential_outcomes = parallel_search_rabitq1(
                &queries,
                &data,
                &sequential,
                &sidecar,
                backend,
                2,
                16,
                2,
                1,
            )
            .unwrap();
            let explicit_outcomes =
                parallel_search_rabitq1(&queries, &data, &explicit, &sidecar, backend, 2, 16, 2, 1)
                    .unwrap();
            assert_eq!(
                sequential_outcomes
                    .iter()
                    .map(|outcome| &outcome.ids)
                    .collect::<Vec<_>>(),
                explicit_outcomes
                    .iter()
                    .map(|outcome| &outcome.ids)
                    .collect::<Vec<_>>()
            );

            let eight_thread_outcomes = parallel_search_rabitq1(
                &queries,
                &data,
                &sequential,
                &sidecar,
                backend,
                2,
                16,
                2,
                8,
            )
            .unwrap();
            assert_eq!(
                sequential_outcomes
                    .iter()
                    .map(|outcome| &outcome.ids)
                    .collect::<Vec<_>>(),
                eight_thread_outcomes
                    .iter()
                    .map(|outcome| &outcome.ids)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                integer_counters(
                    &sequential_outcomes
                        .iter()
                        .map(|outcome| outcome.stats)
                        .collect::<Vec<_>>()
                ),
                integer_counters(
                    &eight_thread_outcomes
                        .iter()
                        .map(|outcome| outcome.stats)
                        .collect::<Vec<_>>()
                )
            );

            let first_path = dir.path().join(format!("{backend:?}-one.fbin"));
            let second_path = dir.path().join(format!("{backend:?}-eight.fbin"));
            save_result_fbin(
                &first_path,
                &sequential_outcomes
                    .iter()
                    .map(|outcome| outcome.ids.clone())
                    .collect::<Vec<_>>(),
                2,
            )
            .unwrap();
            save_result_fbin(
                &second_path,
                &eight_thread_outcomes
                    .iter()
                    .map(|outcome| outcome.ids.clone())
                    .collect::<Vec<_>>(),
                2,
            )
            .unwrap();
            assert_eq!(
                std::fs::read(first_path).unwrap(),
                std::fs::read(second_path).unwrap()
            );
        }

        let post = parallel_search_rabitq1(
            &queries,
            &data,
            &sequential,
            &sidecar,
            SearchBackend::Rabitq1PostRerank,
            2,
            16,
            2,
            1,
        )
        .unwrap();
        for (query_id, outcome) in post.iter().enumerate() {
            let query = &queries[query_id * 64..(query_id + 1) * 64];
            let mut brute_force = (0..16_u32)
                .map(|id| {
                    (
                        OrderedF32(l2_distance(query, data.row(id as usize).unwrap())),
                        id,
                    )
                })
                .collect::<Vec<_>>();
            brute_force.sort_by(|left, right| left.cmp(right));
            assert_eq!(
                outcome.ids,
                brute_force[..2]
                    .iter()
                    .map(|(_, id)| *id)
                    .collect::<Vec<_>>()
            );
        }
        let repeated = parallel_search_rabitq1(
            &queries,
            &data,
            &sequential,
            &sidecar,
            SearchBackend::Rabitq1PostRerank,
            2,
            16,
            2,
            1,
        )
        .unwrap();
        assert_eq!(
            post.iter().map(|outcome| &outcome.ids).collect::<Vec<_>>(),
            repeated
                .iter()
                .map(|outcome| &outcome.ids)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn primary_count_diagnostics_preserve_results_and_standard_counters() {
        let dir = tempdir().unwrap();
        let (data_path, sequential_path, _, sidecar, queries) = build_rabitq1_fixture(dir.path());
        let data = MmapFbinF32::open(&data_path).unwrap();
        let mut graph = MmapGraph::open(&sequential_path, 16).unwrap();
        let standard = parallel_search_rabitq1(
            &queries,
            &data,
            &graph,
            &sidecar,
            SearchBackend::Rabitq1AdaptiveExact,
            2,
            16,
            2,
            1,
        )
        .unwrap();

        let primary_path = dir.path().join("primary-count.sidecar");
        write_primary_count_sidecar(&primary_path, &graph, 8, graph.start());
        graph.attach_primary_counts(&primary_path).unwrap();
        let diagnostic = parallel_search_rabitq1(
            &queries,
            &data,
            &graph,
            &sidecar,
            SearchBackend::Rabitq1AdaptiveExact,
            2,
            16,
            2,
            1,
        )
        .unwrap();

        let standard_ids = standard
            .iter()
            .map(|outcome| outcome.ids.clone())
            .collect::<Vec<_>>();
        let diagnostic_ids = diagnostic
            .iter()
            .map(|outcome| outcome.ids.clone())
            .collect::<Vec<_>>();
        assert_eq!(standard_ids, diagnostic_ids);
        assert_eq!(
            integer_counters(
                &standard
                    .iter()
                    .map(|outcome| outcome.stats)
                    .collect::<Vec<_>>()
            ),
            integer_counters(
                &diagnostic
                    .iter()
                    .map(|outcome| outcome.stats)
                    .collect::<Vec<_>>()
            )
        );

        let standard_result = dir.path().join("standard.fbin");
        let diagnostic_result = dir.path().join("diagnostic.fbin");
        save_result_fbin(&standard_result, &standard_ids, 2).unwrap();
        save_result_fbin(&diagnostic_result, &diagnostic_ids, 2).unwrap();
        assert_eq!(
            std::fs::read(standard_result).unwrap(),
            std::fs::read(diagnostic_result).unwrap()
        );

        let stats = diagnostic
            .iter()
            .map(|outcome| outcome.stats)
            .collect::<Vec<_>>();
        let query_metrics = dir.path().join("query.tsv");
        write_query_metrics_tsv(&query_metrics, 16, 64, &[2, 1], &[1, 0], &stats).unwrap();
        let contents = std::fs::read_to_string(query_metrics).unwrap();
        assert!(contents.contains("backfill_final_gt_hits"));
        assert_eq!(contents.lines().count(), 3);
    }

    #[test]
    fn primary_count_sidecar_rejects_wrong_start_before_search() {
        let dir = tempdir().unwrap();
        let (_, sequential_path, _, _, _) = build_rabitq1_fixture(dir.path());
        let mut graph = MmapGraph::open(&sequential_path, 16).unwrap();
        let primary_path = dir.path().join("wrong-start.sidecar");
        write_primary_count_sidecar(&primary_path, &graph, 8, graph.start() + 1);
        assert!(graph.attach_primary_counts(&primary_path).is_err());
    }

    #[test]
    fn corrupt_rabitq1_sidecar_fails_before_result_creation() {
        let dir = tempdir().unwrap();
        let (data_path, _, _, sidecar, _) = build_rabitq1_fixture(dir.path());
        let sidecar_path = dir.path().join("data.rabitq1");
        drop(sidecar);
        let mut bytes = std::fs::read(&sidecar_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&sidecar_path, bytes).unwrap();
        let result_path = dir.path().join("must-not-exist.fbin");
        let error = run(Args {
            index_path_prefix: dir.path().to_path_buf(),
            data_path,
            query_file: dir.path().join("missing-query.fbin"),
            result_path: result_path.clone(),
            gt_file: dir.path().join("missing-gt.bin"),
            recall_at: 2,
            search_list_size: 16,
            num_threads: 1,
            storage: StorageMode::Mmap,
            max_queries: None,
            metrics_path: None,
            search_backend: SearchBackend::Rabitq1PostRerank,
            rabitq1_sidecar: Some(sidecar_path),
            hashprune_primary_count_sidecar: None,
            query_metrics_path: None,
        })
        .unwrap_err();
        assert!(error.to_string().contains("sidecar"));
        assert!(!result_path.exists());
    }
}
