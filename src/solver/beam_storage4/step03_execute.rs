mod step03_01_dag;
mod step03_02_beam;

use super::{step02_order::SubTask, Precalc, StorageFlag};
use crate::problem::{Input, Operation};

pub(super) fn execute(
    input: &Input,
    precalc: &Precalc,
    tasks: &[Vec<SubTask>; Input::N],
    max_turn: usize,
) -> Result<Vec<[Operation; Input::N]>, &'static str> {
    let tasks = step03_01_dag::critical_path_analysis(tasks, precalc);
    step03_02_beam::beam(input, precalc, tasks, max_turn)
}
