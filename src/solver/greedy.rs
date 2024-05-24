use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use crate::problem::{Operation, Output, Yard};

use super::{Solver, SolverResult};

mod task_assign;
mod task_execute;
mod task_gen;
mod task_order;

pub struct GreedySolver {
    seed: u64,
    max_turn: usize,
}

impl GreedySolver {
    pub fn new(seed: u64, max_turn: usize) -> Self {
        Self { seed, max_turn }
    }
}

impl Solver for GreedySolver {
    fn solve(&self, input: &crate::problem::Input) -> Result<super::SolverResult, &'static str> {
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        let mut all_tasks = task_gen::generate_tasks(input, &mut rng)?;

        //for s in all_tasks.iter().map(|t| {
        //    format!(
        //        "{:>2} | {:>2} {} -> {}",
        //        t.index(),
        //        t.container().index(),
        //        t.from(),
        //        t.to()
        //    )
        //}) {
        //    eprintln!("{}", s);
        //}

        let mut yard = Yard::new(&input);
        let mut output = Output::new();
        task_order::order_tasks(input, &all_tasks);

        while !yard.is_end() {
            let tasks = task_assign::assign_tasks(&mut yard, &mut all_tasks);
            let operations = task_execute::execute(&mut yard, &tasks, &mut rng);
            //eprintln!("turn: {:>3} | {:?}", output.len(), operations);

            for (i, op) in operations.iter().enumerate() {
                if let Operation::Drop = op {
                    all_tasks[tasks[i].as_ref().unwrap().index()].complete();
                }
            }

            // for debug
            // if let Err(err) = yard.apply(&operations) {
            //     eprintln!("{}", err);
            //     return Ok(SolverResult::new(output, &yard));
            // };

            yard.apply(&operations)?;
            yard.carry_in_and_ship();
            output.push(&operations);

            if output.len() > self.max_turn {
                break;
            }
        }

        Ok(SolverResult::new(output, &yard))
    }
}
