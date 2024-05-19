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
use solver::Solver as _;

fn main() -> Result<(), &'static str> {
    let input = Input::read_input();
    //let solvers: Vec<Box<dyn Solver>> = vec![Box::new(SingleCraneSolver), Box::new(GreedySolver)];
    let solvers: Vec<Box<dyn Solver>> = vec![Box::new(GreedySolver)];
    let mut best_result = None;
    let mut best_score = u32::MAX;

    for solver in solvers {
        let result = match solver.solve(&input) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("{}", err);
                continue;
            }
        };
        let score = result.score();

        if best_score.change_min(score) {
            best_result = Some(result);
            best_score = score;
        }
    }

    let result = best_result.ok_or("Failed to solve")?;

    print!("{}", result.output());
    eprintln!("Score: {}", result.score());

    Ok(())
}
