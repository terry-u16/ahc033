use crate::problem::{Operation, Output, Yard};

use super::{Solver, SolverResult};

mod task_assign;
mod task_execute;
mod task_gen;

pub struct GreedySolver;

impl Solver for GreedySolver {
    fn solve(&self, input: &crate::problem::Input) -> Result<super::SolverResult, &'static str> {
        let mut all_tasks = task_gen::generate_tasks(input)?;

        for s in all_tasks.iter().map(|t| {
            format!(
                "{:>2} | {:>2} {} -> {}",
                t.index(),
                t.container().index(),
                t.from(),
                t.to()
            )
        }) {
            eprintln!("{}", s);
        }

        let mut yard = Yard::new(&input);
        let mut output = Output::new();

        while !yard.is_end() {
            let tasks = task_assign::assign_tasks(&mut yard, &mut all_tasks);
            let operations = task_execute::execute(&mut yard, &tasks);

            for (i, op) in operations.iter().enumerate() {
                if let Operation::Drop = op {
                    all_tasks[tasks[i].as_ref().unwrap().index()].complete();
                }
            }

            if let Err(err) = yard.apply(&operations) {
                eprintln!("{}", err);
                return Ok(SolverResult::new(output, &yard));
            };

            yard.carry_in_and_ship();
            output.push(&operations);

            if output.len() > 100 {
                break;
            }
        }

        Ok(SolverResult::new(output, &yard))
    }
}
