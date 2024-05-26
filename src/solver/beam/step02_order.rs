mod annealing;
mod breakdown;

use super::{step01_gen::Task, DistDict, Precalc};
use crate::{
    grid::Coord,
    problem::{Container, CraneState, Grid, Input},
};
use itertools::Itertools;
use std::array;

pub(super) fn order_tasks(
    input: &Input,
    precalc: &Precalc,
    tasks: &[Task],
) -> Result<[Vec<SubTask>; Input::N], &'static str> {
    let env = Env::new(&input, &precalc.dist_dict);
    let state1 = State::new(tasks, |_| 0);
    let state2 = State::new(tasks, |i| i % Input::N);

    let state = if state1.calc_score(&env, 1000).unwrap_or(f64::MAX)
        < state2.calc_score(&env, 1000).unwrap_or(f64::MAX)
    {
        state1
    } else {
        state2
    };

    let state = annealing::annealing(&env, state, 0.5);

    let since = std::time::Instant::now();
    let result: Turns = state.simulate(&env, 200)?;
    eprintln!("elapsed: {:?}", since.elapsed());
    eprintln!("{:?}", result);
    eprintln!("{}", result.calc_score());

    breakdown::breakdown(&env, &state)
}

#[derive(Debug, Clone, Copy)]
enum TaskType {
    Direct(Container, Coord, Coord),
    ToTemporary(Container, Coord, Coord),
    FromTemporary(Container, Coord, Coord),
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    dist_dict: &'a DistDict,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input, dist_dict: &'a DistDict) -> Self {
        Self { input, dist_dict }
    }
}

#[derive(Debug, Clone)]
struct State {
    tasks: [Vec<TaskType>; Input::N],
}

impl State {
    fn new(tasks: &[Task], assign_fn: impl Fn(usize) -> usize) -> Self {
        let all_tasks = tasks
            .iter()
            .map(|t| {
                if t.from().col() == 0 {
                    if t.to().col() == Input::N - 1 {
                        TaskType::Direct(t.container(), t.from(), t.to())
                    } else {
                        TaskType::ToTemporary(t.container(), t.from(), t.to())
                    }
                } else {
                    TaskType::FromTemporary(t.container(), t.from(), t.to())
                }
            })
            .collect_vec();

        let mut tasks = [vec![], vec![], vec![], vec![], vec![]];

        for (i, &task) in all_tasks.iter().enumerate() {
            tasks[assign_fn(i)].push(task);
        }

        Self { tasks }
    }

    fn calc_score(&self, env: &Env, max_turn: usize) -> Result<f64, &'static str> {
        let turns: Turns = self.simulate(env, max_turn)?;
        Ok(turns.calc_score())
    }

    fn simulate<T: Recorder>(&self, env: &Env, max_turn: usize) -> Result<T, &'static str> {
        let mut recorder = T::new();
        let mut in_ptr = [0; Input::N];
        let mut out_next: [_; Input::N] = array::from_fn(|i| i * Input::N);
        let mut task_ptr = [0; Input::N];
        let mut cranes: [_; Input::N] = array::from_fn(|i| CraneState::Empty(Coord::new(i, 0)));
        let mut yard = Grid::new([None; Input::N * Input::N]);
        let mut avail_turns = Grid::new([0; Input::N * Input::N]);
        let mut last_turns = [0; Input::N];
        let mut no_progress = 0;

        for row in 0..Input::N {
            yard[Coord::new(row, 0)] = Some(env.input.containers()[row][0]);
        }

        for turn in 1.. {
            if turn > max_turn {
                return Err("turn limit exceeded");
            }

            let prev_state = cranes.clone();
            let flag = env.dist_dict.get_flag(|c| yard[c].is_some());

            for crane_i in 0..Input::N {
                let crane = &mut cranes[crane_i];
                let Some(task) = self.tasks[crane_i].get(task_ptr[crane_i]).copied() else {
                    continue;
                };

                match *crane {
                    CraneState::Empty(coord) => {
                        let (container, from) = match task {
                            TaskType::Direct(container, from, _) => (container, from),
                            TaskType::ToTemporary(container, from, _) => (container, from),
                            TaskType::FromTemporary(container, from, _) => (container, from),
                        };

                        let can_pick = yard[from] == Some(container) && avail_turns[from] <= turn;

                        if coord == from && can_pick {
                            // コンテナをPickする
                            *crane = CraneState::Holding(container, from);
                            avail_turns[from] = turn + 2;
                            recorder.record_pick(crane_i, coord);

                            if let TaskType::FromTemporary(_, _, _) = task {
                                yard[from] = None;
                            } else {
                                let row = from.row();
                                let in_ptr = &mut in_ptr[row];
                                *in_ptr += 1;
                                yard[from] = env.input.containers()[row].get(*in_ptr).copied();
                            }
                        } else {
                            // コンテナに向けて移動
                            let next = env.dist_dict.next(flag, coord, from, false);
                            *crane = CraneState::Empty(next);
                        }
                    }
                    CraneState::Holding(container_t, coord) => {
                        let (container, to) = match task {
                            TaskType::Direct(container, _, to) => (container, to),
                            TaskType::ToTemporary(container, _, to) => (container, to),
                            TaskType::FromTemporary(container, _, to) => (container, to),
                        };
                        assert_eq!(container, container_t);

                        let can_drop = if let TaskType::ToTemporary(_, _, _) = task {
                            yard[to] == None && avail_turns[to] <= turn
                        } else {
                            out_next[to.row()] == container.index() && avail_turns[to] <= turn
                        };

                        if coord == to && can_drop {
                            // コンテナをDropする
                            *crane = CraneState::Empty(coord);
                            avail_turns[to] = turn + 2;
                            task_ptr[crane_i] += 1;
                            last_turns[crane_i] = turn;
                            recorder.record_drop(crane_i, coord);

                            if let TaskType::ToTemporary(_, _, _) = task {
                                yard[to] = Some(container);
                            } else {
                                out_next[to.row()] += 1;
                            }
                        } else {
                            // コンテナに向けて移動
                            let consider_container = !Input::is_large_crane(crane_i);
                            let next = env.dist_dict.next(flag, coord, to, consider_container);
                            *crane = CraneState::Holding(container, next);
                        }
                    }
                    CraneState::Destroyed => continue,
                }
            }

            if task_ptr
                .iter()
                .zip(&self.tasks)
                .all(|(&ptr, tasks)| ptr == tasks.len())
            {
                recorder.record_turn(last_turns);
                return Ok(recorder);
            }

            if prev_state == cranes {
                no_progress += 1;

                // available_turn制約により1ターンは状況が変化しない可能性がある
                // 2ターン以上状況が変化しない場合は終了
                if no_progress >= 2 {
                    return Err("no progress");
                }
            } else {
                no_progress = 0;
            }
        }

        unreachable!();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubTask {
    Pick(Coord, usize),
    Drop(Coord, usize),
    EndOfOrder,
}

impl SubTask {
    pub fn coord(&self) -> Option<Coord> {
        match self {
            SubTask::Pick(coord, _) => Some(*coord),
            SubTask::Drop(coord, _) => Some(*coord),
            SubTask::EndOfOrder => None,
        }
    }

    pub fn index(&self) -> Option<usize> {
        match self {
            SubTask::Pick(_, index) => Some(*index),
            SubTask::Drop(_, index) => Some(*index),
            SubTask::EndOfOrder => None,
        }
    }
}

trait Recorder {
    fn new() -> Self;
    fn record_pick(&mut self, crane: usize, coord: Coord);
    fn record_drop(&mut self, crane: usize, coord: Coord);
    fn record_turn(&mut self, turns: [usize; Input::N]);
}

#[derive(Debug, Clone, Copy)]
struct Turns {
    turns: [usize; Input::N],
}

impl Turns {
    fn new(turns: [usize; Input::N]) -> Self {
        Self { turns }
    }

    fn calc_score(&self) -> f64 {
        // logsumexp
        const KAPPA: f64 = 3.0;
        let logsumexp = self
            .turns
            .iter()
            .map(|&t| (t as f64 / KAPPA).exp())
            .sum::<f64>()
            .ln()
            * KAPPA;
        logsumexp
    }
}

impl Recorder for Turns {
    fn new() -> Self {
        Self::new([0; Input::N])
    }

    fn record_pick(&mut self, _crane: usize, _coord: Coord) {
        // do nothing
    }

    fn record_drop(&mut self, _crane: usize, _coord: Coord) {
        // do nothing
    }

    fn record_turn(&mut self, turns: [usize; Input::N]) {
        self.turns = turns;
    }
}
