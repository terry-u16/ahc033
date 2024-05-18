pub mod single_crane;

use crate::problem::{Input, Output, Yard};

pub trait Solver {
    fn solve(&self, input: &Input) -> Result<SolverResult, &'static str>;
}

#[derive(Debug, Clone)]
pub struct SolverResult {
    output: Output,
    score: u32,
}

impl SolverResult {
    fn new(output: Output, yard: &Yard) -> Self {
        let score = output.len() as u32 + yard.inversions() * 100;
        Self { output, score }
    }

    pub fn output(&self) -> &Output {
        &self.output
    }

    pub fn score(&self) -> u32 {
        self.score
    }
}
