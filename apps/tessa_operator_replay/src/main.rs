use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use shear_core::algorithm::pipnn::tessa_replay::{
    TessaReplayRunConfig, finalize_tessa_capture, run_tessa_operator_replay,
};
use shear_core::common::AnnResult;

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Debug, Subcommand)]
enum Command {
    FinalizeCapture(FinalizeCaptureArgs),
    Run(RunArgs),
}

#[derive(Clone, Debug, Args)]
struct FinalizeCaptureArgs {
    #[arg(long)]
    fragment_dir: PathBuf,

    #[arg(long)]
    capture_profile: PathBuf,

    #[arg(long)]
    provenance_json: PathBuf,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Clone, Debug, Args)]
struct RunArgs {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    base_path: PathBuf,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 0)]
    gpu_device: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 5)]
    warmup_sweeps: usize,

    #[arg(long, default_value_t = 20)]
    measured_sweeps: usize,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    formal: bool,
}

fn main() -> AnnResult<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> AnnResult<()> {
    match cli.command {
        Command::FinalizeCapture(args) => {
            finalize_tessa_capture(
                &args.fragment_dir,
                &args.capture_profile,
                &args.provenance_json,
                &args.output_dir,
                args.seed,
            )?;
            Ok(())
        }
        Command::Run(args) => run_tessa_operator_replay(&TessaReplayRunConfig {
            manifest_path: args.manifest,
            base_path: args.base_path,
            output_dir: args.output_dir,
            gpu_device: args.gpu_device,
            seed: args.seed,
            warmup_sweeps: args.warmup_sweeps,
            measured_sweeps: args.measured_sweeps,
            formal: args.formal,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tessa_operator_replay_cli_requires_both_subcommand_contracts() {
        assert!(Cli::try_parse_from(["tessa_operator_replay", "finalize-capture"]).is_err());
        assert!(Cli::try_parse_from(["tessa_operator_replay", "run"]).is_err());

        let finalize = Cli::try_parse_from([
            "tessa_operator_replay",
            "finalize-capture",
            "--fragment-dir",
            "fragments",
            "--capture-profile",
            "profile.json",
            "--provenance-json",
            "provenance.json",
            "--output-dir",
            "capture",
        ])
        .unwrap();
        let Command::FinalizeCapture(finalize) = finalize.command else {
            panic!("expected finalize-capture")
        };
        assert_eq!(finalize.seed, 42);

        let replay = Cli::try_parse_from([
            "tessa_operator_replay",
            "run",
            "--manifest",
            "capture/manifest.json",
            "--base-path",
            "base.fbin",
            "--output-dir",
            "replay",
        ])
        .unwrap();
        let Command::Run(replay) = replay.command else {
            panic!("expected run")
        };
        assert_eq!(replay.seed, 42);
        assert_eq!(replay.gpu_device, 0);
        assert_eq!(replay.warmup_sweeps, 5);
        assert_eq!(replay.measured_sweeps, 20);
        assert!(replay.formal);
    }

    #[test]
    fn tessa_operator_replay_cli_rejects_nonformal_counts_in_formal_mode() {
        let cli = Cli::try_parse_from([
            "tessa_operator_replay",
            "run",
            "--manifest",
            "capture/manifest.json",
            "--base-path",
            "base.fbin",
            "--output-dir",
            "replay",
            "--warmup-sweeps",
            "4",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run")
        };
        let config = TessaReplayRunConfig {
            manifest_path: args.manifest,
            base_path: args.base_path,
            output_dir: args.output_dir,
            gpu_device: args.gpu_device,
            seed: args.seed,
            warmup_sweeps: args.warmup_sweeps,
            measured_sweeps: args.measured_sweeps,
            formal: args.formal,
        };
        assert!(config.validate().is_err());
    }
}
