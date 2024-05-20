mod common;
mod data_structures;
mod grid;
mod problem;
mod solver;

use crate::{
    common::ChangeMinMax,
    solver::{greedy::GreedySolver, single_crane::SingleCraneSolver, Solver},
};
use problem::Input;
use rand::{Rng as _, SeedableRng};
use rand_pcg::Pcg64Mcg;
use solver::Solver as _;

fn main() -> Result<(), &'static str> {
    let input = Input::read_input();
    let since = std::time::Instant::now();
    let mut best_result = SingleCraneSolver.solve(&input)?;
    let mut best_score = best_result.score();
    let mut rng = Pcg64Mcg::from_entropy();

    while since.elapsed().as_secs_f64() < 1.9 {
        let solver = GreedySolver::new(rng.gen(), best_score as usize);
        let result = match solver.solve(&input) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("{}", err);
                continue;
            }
        };
        let score = result.score();

        if best_score.change_min(score) {
            best_result = result;
            best_score = score;
            eprintln!("score updated!: {}", score);
        }
    }

    print!("{}", best_result.output());
    eprintln!("Score: {}", best_result.score());

    Ok(())
}
