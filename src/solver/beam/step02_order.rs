mod step02_01_annealing;
mod step02_02_breakdown;

use super::{step01a_gen_dp::Task, DistDict, Precalc};
use crate::{
    common::ChangeMinMax,
    grid::Coord,
    problem::{Container, CraneState, Grid, Input},
};
use rand::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::array;

pub(super) fn order_tasks(
    input: &Input,
    precalc: &Precalc,
    tasks: Vec<Vec<Task>>,
) -> Result<[Vec<SubTask>; Input::N], &'static str> {
    let env = Env::new(&input, &precalc.dist_dict);
    let mut rng = Pcg64Mcg::from_entropy();

    let mut best_state = None;
    let mut best_score = f64::INFINITY;

    if tasks.len() == 0 {
        return Err("no tasks");
    }

    let step1_duration = 0.5 / tasks.len() as f64;

    for tasks in tasks {
        let mut state = State::new(&tasks, |c, i| c.unwrap_or(i % Input::N));

        loop {
            if state.calc_score(&env, 1000).is_ok() {
                break;
            }

            eprintln!("retrying...");
            state = State::new(&tasks, |_, _| rng.gen_range(0..Input::N));
        }

        let state = step02_01_annealing::annealing(&env, state, step1_duration);

        if best_score.change_min(
            state
                .calc_score_best_state(&env, usize::MAX)
                .unwrap_or(f64::INFINITY),
        ) {
            best_state = Some(state);
        }
    }

    let Some(state) = best_state else {
        return Err("no valid state found");
    };

    let step2_duration = 1.0;
    let state = step02_01_annealing::annealing(&env, state, step2_duration);

    step02_02_breakdown::breakdown(&env, &state)
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
    fn new(tasks: &[Task], mut assign_fn: impl FnMut(Option<usize>, usize) -> usize) -> Self {
        let mut assigned_tasks = [vec![], vec![], vec![], vec![], vec![]];

        for (i, task) in tasks.iter().enumerate() {
            let t = if task.from().col() == 0 {
                if task.to().col() == Input::N - 1 {
                    TaskType::Direct(task.container(), task.from(), task.to())
                } else {
                    TaskType::ToTemporary(task.container(), task.from(), task.to())
                }
            } else {
                TaskType::FromTemporary(task.container(), task.from(), task.to())
            };

            assigned_tasks[assign_fn(task.crane(), i)].push(t);
        }

        Self {
            tasks: assigned_tasks,
        }
    }

    fn calc_score(&self, env: &Env, max_turn: usize) -> Result<f64, &'static str> {
        let turns: Turns = self.simulate(env, max_turn)?;
        Ok(turns.calc_score(env))
    }

    fn calc_score_best_state(&self, env: &Env, max_turn: usize) -> Result<f64, &'static str> {
        let turns: Turns = self.simulate(env, max_turn)?;
        Ok(turns.calc_score_best_state(env))
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

    fn calc_score(&self, env: &Env) -> f64 {
        // logsumexp
        let kappa = env.input.params().kappa_step02();
        let logsumexp = self
            .turns
            .iter()
            .map(|&t| (t as f64 / kappa).exp())
            .sum::<f64>()
            .ln()
            * kappa;
        logsumexp
    }

    fn calc_score_best_state(&self, env: &Env) -> f64 {
        // logsumexp
        let kappa = env.input.params().kappa_step02_best();
        let logsumexp = self
            .turns
            .iter()
            .map(|&t| (t as f64 / kappa).exp())
            .sum::<f64>()
            .ln()
            * kappa;
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
