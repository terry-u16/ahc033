use super::Solver;

mod task_gen;

pub struct GreedySolver;

impl Solver for GreedySolver {
    fn solve(&self, input: &crate::problem::Input) -> Result<super::SolverResult, &'static str> {
        let tasks = task_gen::generate_tasks(input)?;

        for s in tasks
            .iter()
            .map(|t| format!("{:>2} {} -> {}", t.container().index(), t.from(), t.to()))
        {
            eprintln!("{}", s);
        }

        Err("not yet implemented")
    }
}
