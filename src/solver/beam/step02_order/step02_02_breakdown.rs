use crate::problem::{Grid, Input};

use super::{Env, Recorder, State, SubTask};

pub(super) fn breakdown(env: &Env, state: &State) -> Result<[Vec<SubTask>; Input::N], &'static str> {
    let subtasks: SubTaskRecorder = state.simulate(env, usize::MAX)?;
    Ok(subtasks.tasks)
}

struct SubTaskRecorder {
    tasks: [Vec<SubTask>; 5],
    grid: Grid<usize>,
}

impl Recorder for SubTaskRecorder {
    fn new() -> Self {
        Self {
            tasks: [vec![], vec![], vec![], vec![], vec![]],
            grid: Grid::with_default(),
        }
    }

    fn record_pick(&mut self, crane: usize, coord: crate::grid::Coord) {
        let task = SubTask::Pick(coord, self.grid[coord]);
        self.tasks[crane].push(task);
        self.grid[coord] += 1;
    }

    fn record_drop(&mut self, crane: usize, coord: crate::grid::Coord) {
        let task = SubTask::Drop(coord, self.grid[coord]);
        self.tasks[crane].push(task);
        self.grid[coord] += 1;
    }

    fn record_turn(&mut self, _turns: [usize; Input::N]) {
        // シミュレーションが終わったので、EOOを追加
        for tasks in self.tasks.iter_mut() {
            tasks.push(SubTask::EndOfOrder);
        }
    }
}
