mod bayesian;
mod beam;
mod common;
mod data_structures;
mod grid;
mod problem;
mod solver;

use crate::{
    common::ChangeMinMax,
    solver::{
        beam_storage4::BeamSolver4, beam_storage6::BeamSolver6, beam_storage8::BeamSolver8,
        single_crane::SingleCraneSolver, Solver,
    },
};
use problem::Input;
use rand::{Rng as _, SeedableRng};
use rand_pcg::Pcg64Mcg;

fn main() -> Result<(), &'static str> {
    let input = Input::read_input();
    let since = std::time::Instant::now();
    let mut best_result = SingleCraneSolver.solve(&input)?;
    let mut best_score = best_result.score();
    let mut rng = Pcg64Mcg::from_entropy();

    let solver = BeamSolver4::new(rng.gen(), best_score as usize);
    solve(solver, &input, &mut best_score, &mut best_result);

    eprintln!("elapsed: {:?}", since.elapsed());

    let solver = BeamSolver6::new(rng.gen(), best_score as usize);
    solve(solver, &input, &mut best_score, &mut best_result);

    eprintln!("elapsed: {:?}", since.elapsed());

    if best_score >= 100 && since.elapsed().as_secs_f64() <= 0.5 {
        let solver = BeamSolver8::new(rng.gen(), best_score as usize);
        solve(solver, &input, &mut best_score, &mut best_result);
    }

    print!("{}", best_result.output());
    eprintln!("Score: {}", best_result.score());

    Ok(())
}

fn solve(
    solver: impl Solver,
    input: &Input,
    best_score: &mut u32,
    best_result: &mut solver::SolverResult,
) {
    match solver.solve(&input) {
        Ok(result) => {
            let score = result.score();

            if best_score.change_min(score) {
                *best_result = result;
                eprintln!("score updated!: {}", score);
            }
        }
        Err(err) => {
            eprintln!("{}", err);
        }
    }
}
