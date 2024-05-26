mod common;
mod data_structures;
mod grid;
mod problem;
mod solver;

use crate::{
    common::ChangeMinMax,
    solver::{beam::BeamSolver, single_crane::SingleCraneSolver, Solver},
};
use problem::Input;
use rand::{Rng as _, SeedableRng};
use rand_pcg::Pcg64Mcg;

fn main() -> Result<(), &'static str> {
    let input = Input::read_input();
    let mut best_result = SingleCraneSolver.solve(&input)?;
    let mut best_score = best_result.score();
    let mut rng = Pcg64Mcg::from_entropy();

    let solver = BeamSolver::new(rng.gen(), best_score as usize);
    match solver.solve(&input) {
        Ok(result) => {
            let score = result.score();

            if best_score.change_min(score) {
                best_result = result;
                eprintln!("score updated!: {}", score);
            }
        }
        Err(err) => {
            eprintln!("{}", err);
        }
    };

    print!("{}", best_result.output());
    eprintln!("Score: {}", best_result.score());

    Ok(())
}
