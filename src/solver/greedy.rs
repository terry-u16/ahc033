use crate::problem::{Output, Yard};

use super::Solver;

mod task_assign;
mod task_gen;

pub struct GreedySolver;

impl Solver for GreedySolver {
    fn solve(&self, input: &crate::problem::Input) -> Result<super::SolverResult, &'static str> {
        let mut tasks = task_gen::generate_tasks(input)?;

        for s in tasks
            .iter()
            .map(|t| format!("{:>2} {} -> {}", t.container().index(), t.from(), t.to()))
        {
            eprintln!("{}", s);
        }

        let mut yard = Yard::new(&input);
        let mut output = Output::new();

        //while !yard.is_end() {}

        let tasks = task_assign::assign_tasks(&mut yard, &mut tasks);

        eprintln!("assigned");

        for t in tasks.iter() {
            eprintln!("{:?}", t);
        }

        Err("not yet implemented")
    }
}
