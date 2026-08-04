use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use shear_core::model::rabitq::graph_search::{
    BuildRabitq1Options, Rabitq1Sidecar, build_rabitq1_sidecar,
};

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    data_path: PathBuf,
    #[clap(long)]
    output: PathBuf,
    #[clap(long, default_value_t = 16)]
    centroids: usize,
    #[clap(long, default_value_t = 262_144)]
    sample_rows: usize,
    #[clap(long, default_value_t = 20)]
    kmeans_iters: usize,
    #[clap(long, default_value_t = 42)]
    seed: u64,
    #[clap(long, default_value_t = 8_192)]
    batch_rows: usize,
    #[clap(long, default_value_t = 8)]
    threads: usize,
    #[clap(long)]
    validate_only: bool,
}

fn validate_args(args: &Args) -> Result<()> {
    ensure!(args.centroids == 16, "--centroids must equal 16");
    ensure!(args.kmeans_iters == 20, "--kmeans-iters must equal 20");
    ensure!(args.sample_rows > 0, "--sample-rows must be positive");
    ensure!(args.batch_rows > 0, "--batch-rows must be positive");
    ensure!(args.threads > 0, "--threads must be positive");
    ensure!(
        args.output
            .extension()
            .is_some_and(|extension| extension == "rabitq1"),
        "--output must have the .rabitq1 extension"
    );
    if args.validate_only {
        ensure!(
            args.output.is_file(),
            "--validate-only requires an existing sidecar"
        );
    }
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    validate_args(args)?;
    if args.validate_only {
        let sidecar = Rabitq1Sidecar::load(&args.output, &args.data_path)
            .with_context(|| format!("validate sidecar {:?}", args.output))?;
        println!("{}", sidecar.validation_report().to_pretty_json()?);
        return Ok(());
    }

    let options = BuildRabitq1Options {
        center_count: args.centroids,
        sample_rows: args.sample_rows,
        kmeans_iterations: args.kmeans_iters,
        seed: args.seed,
        batch_rows: args.batch_rows,
        threads: args.threads,
    };
    let report = build_rabitq1_sidecar(&args.data_path, &args.output, &options)
        .with_context(|| format!("build sidecar {:?}", args.output))?;
    println!("{}", report.to_pretty_json()?);
    Ok(())
}

fn main() -> Result<()> {
    run(&Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::process::Command;

    fn minimal_args() -> Vec<&'static str> {
        vec![
            "build_graph_rabitq_1bit",
            "--data-path",
            "/tmp/base.fbin",
            "--output",
            "/tmp/base.rabitq1",
        ]
    }

    fn build_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.fbin");
        let sidecar_path = dir.path().join("base.rabitq1");
        let mut base = Vec::with_capacity(8 + 32 * 67 * 4);
        base.extend_from_slice(&32_u32.to_le_bytes());
        base.extend_from_slice(&67_u32.to_le_bytes());
        for index in 0..32 * 67 {
            let value = ((index * 31 + 11) % 257) as f32 / 43.0;
            base.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&base_path, base).unwrap();
        build_rabitq1_sidecar(
            &base_path,
            &sidecar_path,
            &BuildRabitq1Options {
                center_count: 16,
                sample_rows: 32,
                kmeans_iterations: 20,
                seed: 42,
                batch_rows: 7,
                threads: 2,
            },
        )
        .unwrap();
        (dir, base_path, sidecar_path)
    }

    fn sha256(path: &std::path::Path) -> String {
        let output = Command::new("sha256sum").arg(path).output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .to_owned()
    }

    #[test]
    fn cli_defaults_match_the_frozen_preprocessing_contract() {
        let args = Args::try_parse_from(minimal_args()).unwrap();
        validate_args(&args).unwrap();
        assert_eq!(args.centroids, 16);
        assert_eq!(args.sample_rows, 262_144);
        assert_eq!(args.kmeans_iters, 20);
        assert_eq!(args.seed, 42);
        assert_eq!(args.batch_rows, 8_192);
        assert_eq!(args.threads, 8);
        assert!(!args.validate_only);
    }

    #[test]
    fn cli_rejects_noncanonical_or_zero_options() {
        for extra in [
            vec!["--centroids", "15"],
            vec!["--kmeans-iters", "19"],
            vec!["--batch-rows", "0"],
            vec!["--threads", "0"],
        ] {
            let mut argv = minimal_args();
            argv.extend(extra);
            let args = Args::try_parse_from(argv).unwrap();
            assert!(validate_args(&args).is_err());
        }

        let args = Args::try_parse_from([
            "build_graph_rabitq_1bit",
            "--data-path",
            "/tmp/base.fbin",
            "--output",
            "/tmp/base.bin",
        ])
        .unwrap();
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn validate_only_preserves_the_sidecar_bytes_and_metadata() {
        let (_dir, base_path, sidecar_path) = build_fixture();
        let before = std::fs::metadata(&sidecar_path).unwrap();
        let before_hash = sha256(&sidecar_path);
        let args = Args {
            data_path: base_path,
            output: sidecar_path.clone(),
            centroids: 16,
            sample_rows: 262_144,
            kmeans_iters: 20,
            seed: 42,
            batch_rows: 8_192,
            threads: 8,
            validate_only: true,
        };

        run(&args).unwrap();

        let after = std::fs::metadata(&sidecar_path).unwrap();
        assert_eq!(after.len(), before.len());
        assert_eq!(after.modified().unwrap(), before.modified().unwrap());
        assert_eq!(sha256(&sidecar_path), before_hash);
    }

    #[test]
    fn validate_only_rejects_a_corrupt_sidecar() {
        let (_dir, base_path, sidecar_path) = build_fixture();
        let mut bytes = std::fs::read(&sidecar_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&sidecar_path, bytes).unwrap();
        let args = Args {
            data_path: base_path,
            output: sidecar_path,
            centroids: 16,
            sample_rows: 262_144,
            kmeans_iters: 20,
            seed: 42,
            batch_rows: 8_192,
            threads: 8,
            validate_only: true,
        };

        assert!(run(&args).is_err());
    }
}
