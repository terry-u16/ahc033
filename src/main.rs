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
        beam::BeamSolver, beam_storage8::BeamSolver8, single_crane::SingleCraneSolver, Solver,
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

            // 一時保管場所は基本的に6箇所だが、6箇所で足りない場合があるため8箇所も試す
            if since.elapsed().as_secs_f64() <= 0.1 {
                eprintln!("Trying BeamSolver8...");
                let solver = BeamSolver8::new(rng.gen(), best_score as usize);

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
                }
            }
        }
    };

    print!("{}", best_result.output());
    eprintln!("Score: {}", best_result.score());

    Ok(())
}
