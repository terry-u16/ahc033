mod common;
mod data_structures;
mod grid;
mod problem;
mod solver;

use crate::solver::single_crane::SingleCraneSolver;
use problem::Input;
use solver::Solver as _;

fn main() -> Result<(), &'static str> {
    let input = Input::read_input();
    let result = SingleCraneSolver.solve(&input)?;
    print!("{}", result.output());
    eprintln!("Score: {}", result.score());

    Ok(())
}
