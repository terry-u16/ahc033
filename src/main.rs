mod common;
mod data_structures;
mod grid;
mod problem;
mod solver;

use problem::Input;

fn main() -> Result<(), ()> {
    let input = Input::read_input();
    let output = solver::solve(&input)?;
    print!("{}", output);

    Ok(())
}
