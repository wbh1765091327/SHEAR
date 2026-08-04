use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use shear_core::index::DirectMemGraphWriter;
use shear_core::model::HashPrunePrimaryCountSidecar;

const GRAPH_HEADER_BYTES: u64 = 24;

#[derive(Clone, Debug, Parser)]
struct Args {
    #[clap(long)]
    source_graph: PathBuf,
    #[clap(long)]
    primary_count_sidecar: PathBuf,
    #[clap(long)]
    output_graph: PathBuf,
    #[clap(long)]
    validation_json: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct GraphHeader {
    index_size: u64,
    max_degree: u32,
    start: u32,
    num_frozen_points: u64,
}

fn main() -> Result<()> {
    run(Args::parse())
}

fn run(args: Args) -> Result<()> {
    let sidecar_bytes = fs::metadata(&args.primary_count_sidecar)
        .with_context(|| format!("stat {:?}", args.primary_count_sidecar))?
        .len();
    ensure!(
        sidecar_bytes >= 128,
        "primary-count sidecar is shorter than its header"
    );
    let rows = usize::try_from(sidecar_bytes - 128)?;
    let sidecar = HashPrunePrimaryCountSidecar::load(&args.primary_count_sidecar, rows)?;
    ensure!(
        !args.output_graph.exists() && !args.validation_json.exists(),
        "refusing to overwrite output graph or validation report"
    );
    let output_partial = append_suffix(&args.output_graph, ".partial");
    let validation_partial = append_suffix(&args.validation_json, ".partial");
    ensure!(
        !output_partial.exists() && !validation_partial.exists(),
        "refusing to overwrite partial output"
    );

    let mut reader = BufReader::new(File::open(&args.source_graph)?);
    let header = read_header(&mut reader)?;
    ensure!(
        header.index_size == fs::metadata(&args.source_graph)?.len(),
        "source graph size mismatch"
    );
    ensure!(
        header.start == sidecar.header().graph_start,
        "source start and sidecar start mismatch"
    );
    ensure!(
        header.max_degree <= sidecar.header().degree_cap,
        "source degree exceeds sidecar cap"
    );
    ensure!(rows == sidecar.counts().len(), "sidecar row mismatch");

    let mut writer = DirectMemGraphWriter::create(
        &output_partial,
        header.start,
        usize::try_from(header.num_frozen_points)?,
    )?;
    let mut source_edges = 0_u64;
    let mut shadow_edges = 0_u64;
    let mut source_max_degree = 0_u32;
    let mut shadow_max_degree = 0_u32;
    for source in 0..rows {
        let degree = read_u32(&mut reader)?;
        ensure!(
            degree <= header.max_degree,
            "source {source} degree exceeds header cap"
        );
        let mut neighbors = vec![0_u32; degree as usize];
        for neighbor in &mut neighbors {
            *neighbor = read_u32(&mut reader)?;
            ensure!(
                (*neighbor as usize) < rows,
                "source {source} has out-of-range neighbor"
            );
        }
        let primary = sidecar
            .primary_count(source)
            .context("missing primary count")?;
        ensure!(
            primary <= neighbors.len(),
            "source {source} primary count exceeds degree"
        );
        writer.write_node(source as u32, &neighbors[..primary])?;
        source_edges += degree as u64;
        shadow_edges += primary as u64;
        source_max_degree = source_max_degree.max(degree);
        shadow_max_degree = shadow_max_degree.max(primary as u32);
    }
    let mut trailing = [0_u8; 1];
    ensure!(
        reader.read(&mut trailing)? == 0,
        "source graph has trailing bytes or explicit-ID records"
    );
    ensure!(
        source_max_degree == header.max_degree,
        "source header max degree mismatch"
    );
    writer.set_start(header.start);
    writer.finish()?;

    let validated = validate_sequential_graph(&output_partial, rows)?;
    ensure!(
        validated.start == header.start,
        "shadow graph start changed"
    );
    ensure!(
        validated.num_frozen_points == header.num_frozen_points,
        "shadow frozen count changed"
    );
    ensure!(
        validated.max_degree == shadow_max_degree,
        "shadow maximum degree mismatch"
    );
    fs::rename(&output_partial, &args.output_graph)?;
    sync_parent(&args.output_graph)?;

    let report = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"hashprune_shadow_plain_validation_v1\",\n",
            "  \"rows\": {rows},\n",
            "  \"source_start\": {start},\n",
            "  \"num_frozen_points\": {frozen},\n",
            "  \"source_edges\": {source_edges},\n",
            "  \"shadow_edges\": {shadow_edges},\n",
            "  \"removed_backfill_edges\": {removed},\n",
            "  \"source_max_degree\": {source_max_degree},\n",
            "  \"shadow_max_degree\": {shadow_max_degree},\n",
            "  \"all_rows_prefix_exact\": true\n",
            "}}"
        ),
        rows = rows,
        start = header.start,
        frozen = header.num_frozen_points,
        source_edges = source_edges,
        shadow_edges = shadow_edges,
        removed = source_edges - shadow_edges,
        source_max_degree = source_max_degree,
        shadow_max_degree = shadow_max_degree,
    );
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&validation_partial)?;
    file.write_all(report.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&validation_partial, &args.validation_json)?;
    sync_parent(&args.validation_json)
}

fn validate_sequential_graph(path: &Path, rows: usize) -> Result<GraphHeader> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header(&mut reader)?;
    ensure!(
        header.index_size == fs::metadata(path)?.len(),
        "shadow size mismatch"
    );
    let mut observed_max = 0_u32;
    for _ in 0..rows {
        let degree = read_u32(&mut reader)?;
        observed_max = observed_max.max(degree);
        for _ in 0..degree {
            let neighbor = read_u32(&mut reader)?;
            ensure!((neighbor as usize) < rows, "shadow neighbor out of range");
        }
    }
    let mut trailing = [0_u8; 1];
    ensure!(
        reader.read(&mut trailing)? == 0,
        "shadow graph has trailing bytes"
    );
    ensure!(
        header.max_degree == observed_max,
        "shadow header max degree mismatch"
    );
    Ok(header)
}

fn read_header(reader: &mut impl Read) -> Result<GraphHeader> {
    let index_size = read_u64(reader)?;
    ensure!(
        index_size >= GRAPH_HEADER_BYTES,
        "graph index size is too small"
    );
    Ok(GraphHeader {
        index_size,
        max_degree: read_u32(reader)?,
        start: read_u32(reader)?,
        num_frozen_points: read_u64(reader)?,
    })
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn sync_parent(path: &Path) -> Result<()> {
    let parent = path.parent().context("output path has no parent")?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shear_core::model::{HashPruneGraphEncoding, HashPrunePrimaryCountWriter};
    use tempfile::tempdir;

    fn write_fixture(root: &Path, sidecar_start: u32) -> (Args, Vec<Vec<u32>>, Vec<usize>) {
        let rows = vec![vec![1, 2, 3], vec![], vec![0, 1, 3], vec![2, 1]];
        let primary_counts = vec![2, 0, 3, 1];
        let source_graph = root.join("source.index");
        let sidecar_path = root.join("primary.sidecar");
        let mut graph_writer = DirectMemGraphWriter::create(&source_graph, 3, 0).unwrap();
        for (source, neighbors) in rows.iter().enumerate() {
            graph_writer.write_node(source as u32, neighbors).unwrap();
        }
        graph_writer.set_start(3);
        graph_writer.finish().unwrap();

        let mut sidecar =
            HashPrunePrimaryCountWriter::create(&sidecar_path, rows.len(), 3).unwrap();
        for (source, (&primary, neighbors)) in primary_counts.iter().zip(&rows).enumerate() {
            sidecar
                .write_row(source as u64, primary, neighbors.len())
                .unwrap();
        }
        sidecar
            .finish(sidecar_start, HashPruneGraphEncoding::SequentialDegreeOnly)
            .unwrap();
        (
            Args {
                source_graph,
                primary_count_sidecar: sidecar_path,
                output_graph: root.join("shadow.index"),
                validation_json: root.join("validation.json"),
            },
            rows,
            primary_counts,
        )
    }

    fn read_rows(path: &Path, rows: usize) -> (GraphHeader, Vec<Vec<u32>>) {
        let mut reader = BufReader::new(File::open(path).unwrap());
        let header = read_header(&mut reader).unwrap();
        let output = (0..rows)
            .map(|_| {
                let degree = read_u32(&mut reader).unwrap() as usize;
                (0..degree)
                    .map(|_| read_u32(&mut reader).unwrap())
                    .collect()
            })
            .collect();
        (header, output)
    }

    #[test]
    fn shadow_builder_keeps_exact_primary_prefixes() {
        let dir = tempdir().unwrap();
        let (args, rows, primary_counts) = write_fixture(dir.path(), 3);

        run(args.clone()).unwrap();

        let (header, output_rows) = read_rows(&args.output_graph, rows.len());
        assert_eq!(header.start, 3);
        assert_eq!(header.num_frozen_points, 0);
        assert_eq!(header.max_degree, 3);
        assert_eq!(
            output_rows,
            rows.iter()
                .zip(primary_counts)
                .map(|(row, primary)| row[..primary].to_vec())
                .collect::<Vec<_>>()
        );
        let report = fs::read_to_string(args.validation_json).unwrap();
        assert!(report.contains("\"removed_backfill_edges\": 2"));
        assert!(report.contains("\"all_rows_prefix_exact\": true"));
    }

    #[test]
    fn shadow_builder_rejects_start_mismatch_without_validation_report() {
        let dir = tempdir().unwrap();
        let (args, _, _) = write_fixture(dir.path(), 2);

        assert!(run(args.clone()).is_err());
        assert!(!args.output_graph.exists());
        assert!(!args.validation_json.exists());
    }
}
