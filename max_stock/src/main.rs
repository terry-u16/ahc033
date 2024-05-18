mod common;
mod dp;
mod problem;

use std::{error::Error, path::PathBuf};

use clap::{Parser, Subcommand};
use common::ChangeMinMax;
use problem::Input;

#[derive(Debug, Parser)]
struct Cli {
    #[clap(subcommand)]
    subcommand: SubCommnad,
}

#[derive(Debug, Subcommand)]
enum SubCommnad {
    Single {
        #[clap(short = 's', long = "seed")]
        seed: usize,
        #[clap(short = 'd', long = "dir")]
        dir: PathBuf,
    },
    Multi {
        #[clap(short = 's', long = "start")]
        start: usize,
        #[clap(short = 'e', long = "end")]
        end: usize,
        #[clap(short = 'd', long = "dir")]
        dir: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    match args.subcommand {
        SubCommnad::Single { seed, dir } => {
            let path = get_path(&dir, seed);
            let input = problem::Input::read_input(&path)?;
            let (max, hist) = dp::dp(&input);

            for (turn, hist) in hist.iter().enumerate() {
                println!("turn: {:>2} | {}", turn, hist);
            }

            println!("max_buffer: {}", max);
        }
        SubCommnad::Multi { start, end, dir } => {
            if start >= end {
                println!("start must be less than end");
                return Ok(());
            }

            let case_count = end - start;
            let mut max_buf = 0;
            let mut min_buf = std::u32::MAX;
            let mut sum = 0;
            let mut counts = [0; Input::N * Input::N];

            for seed in start..end {
                let path = get_path(&dir, seed);
                let input = problem::Input::read_input(&path)?;
                let (max, _) = dp::dp(&input);
                sum += max;
                max_buf.change_max(max);
                min_buf.change_min(max);
                counts[max as usize] += 1;
            }

            println!("min_buffer: {}", min_buf);
            println!("max_buffer: {}", max_buf);
            println!("average_buffer: {:.2}", sum as f64 / case_count as f64);

            let div = (case_count as f64 / 100.0).max(1.0);

            for i in min_buf..=max_buf {
                let cnt = counts[i as usize];
                let percent = (cnt as f64 / case_count as f64) * 100.0;
                let bar_len = (cnt as f64 / div).round() as usize;
                println!("{:>2}: {:>5.2}% {}", i, percent, "|".repeat(bar_len))
            }
        }
    }

    Ok(())
}

fn get_path(dir: &PathBuf, seed: usize) -> PathBuf {
    dir.join(format!("{:0>4}.txt", seed))
}
