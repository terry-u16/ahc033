use std::{array, collections::VecDeque};

use itertools::Itertools;

use crate::{
    common::ChangeMinMax,
    grid::{Coord, CoordDiff, ADJACENTS},
    problem::{Container, CraneState, Grid, Input},
};

use super::task_gen::Task;

pub fn order_tasks(input: &Input, tasks: &[Task]) -> Result<(), &'static str> {
    let env = Env::new(&input);
    let state = State::new(tasks);
    let result = state.simulate(&env, 200)?;
    eprintln!("{:?}", result);
    eprintln!("{}", result.calc_score());

    todo!();
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
    dist_dict: DistDict,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let dist_dict = DistDict::new();
        Self { input, dist_dict }
    }
}

#[derive(Debug, Clone)]
struct DistDict {
    dists: Vec<Grid<Grid<usize>>>,
}

impl DistDict {
    const STORAGES: [Coord; 11] = [
        Coord::new(0, 0),
        Coord::new(1, 0),
        Coord::new(2, 0),
        Coord::new(3, 0),
        Coord::new(4, 0),
        Coord::new(0, 2),
        Coord::new(2, 2),
        Coord::new(4, 2),
        Coord::new(0, 3),
        Coord::new(2, 3),
        Coord::new(4, 3),
    ];

    fn new() -> Self {
        let mut dists = vec![];

        for flag in 0..1 << Self::STORAGES.len() {
            let mut board = Grid::new([false; Input::N * Input::N]);

            for (i, &storage) in Self::STORAGES.iter().enumerate() {
                if (flag & (1 << i)) > 0 {
                    board[storage] = true;
                }
            }

            let mut d =
                Grid::new([Grid::new([usize::MAX; Input::N * Input::N]); Input::N * Input::N]);

            for row in 0..Input::N {
                for col in 0..Input::N {
                    let c = Coord::new(row, col);
                    Self::bfs(&board, &mut d[c], c);
                }
            }

            dists.push(d);
        }

        Self { dists }
    }

    fn bfs(board: &Grid<bool>, dists: &mut Grid<usize>, from: Coord) {
        dists[from] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(from);

        while let Some(coord) = queue.pop_front() {
            let next_dist = dists[coord] + 1;

            for &adj in ADJACENTS.iter() {
                let next = coord + adj;

                if next.in_map(Input::N) && !board[next] && dists[next].change_min(next_dist) {
                    queue.push_back(next);
                }
            }
        }
    }

    fn get_flag(&self, f: impl Fn(Coord) -> bool) -> StorageFlag {
        let mut flag = 0;

        for (i, &storage) in Self::STORAGES.iter().enumerate() {
            if f(storage) {
                flag |= 1 << i;
            }
        }

        StorageFlag(flag)
    }

    fn dist(&self, flag: StorageFlag, from: Coord, to: Coord, consider_contianer: bool) -> usize {
        if !consider_contianer {
            return from.dist(&to);
        }

        self.dists[flag.0][from][to]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StorageFlag(usize);

#[derive(Debug, Clone)]
struct State {
    tasks: [Vec<TaskType>; Input::N],
}

impl State {
    fn new(tasks: &[Task]) -> Self {
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
            tasks[i % 5].push(task);
        }

        Self { tasks }
    }

    fn simulate(&self, env: &Env, max_turn: usize) -> Result<Turns, &'static str> {
        let mut in_ptr = [0; Input::N];
        let mut out_next: [_; Input::N] = array::from_fn(|i| i * Input::N);
        let mut task_ptr = [0; Input::N];
        let mut cranes: [_; Input::N] = array::from_fn(|i| CraneState::Empty(Coord::new(i, 0)));
        let mut yard = Grid::new([None; Input::N * Input::N]);
        let mut avail_turns = Grid::new([0; Input::N * Input::N]);
        let mut last_turns = [0; Input::N];

        for row in 0..Input::N {
            yard[Coord::new(row, 0)] = Some(env.input.containers()[row][0]);
        }

        fn get_best_move(coord: Coord, dist: impl Fn(Coord) -> Option<usize>) -> Coord {
            const ADJ: [CoordDiff; 5] = [
                CoordDiff::new(0, 1),
                CoordDiff::new(1, 0),
                CoordDiff::new(0, -1),
                CoordDiff::new(-1, 0),
                CoordDiff::new(0, 0),
            ];

            let mut best = coord;
            let mut best_dist = usize::MAX;

            for &adj in ADJ.iter() {
                let next = coord + adj;

                if !next.in_map(Input::N) {
                    continue;
                }

                if let Some(d) = dist(next) {
                    if best_dist.change_min(d) {
                        best = next;
                    }
                }
            }

            best
        }

        for turn in 1.. {
            if turn > max_turn {
                return Err("turn limit exceeded");
            }

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
                            avail_turns[from] = turn + 1;

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
                            let next = get_best_move(coord, |c| {
                                if c != from || can_pick {
                                    Some(env.dist_dict.dist(flag, c, from, false))
                                } else {
                                    None
                                }
                            });
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
                            avail_turns[to] = turn + 1;
                            task_ptr[crane_i] += 1;
                            last_turns[crane_i] = turn;

                            if let TaskType::ToTemporary(_, _, _) = task {
                                yard[to] = Some(container);
                            } else {
                                out_next[to.row()] += 1;
                            }
                        } else {
                            // コンテナに向けて移動
                            let consider_container = !Input::is_large_crane(crane_i);
                            let next = get_best_move(coord, |c| {
                                if c != to || can_drop {
                                    Some(env.dist_dict.dist(flag, c, to, consider_container))
                                } else {
                                    None
                                }
                            });
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
                return Ok(Turns::new(last_turns));
            }
        }

        unreachable!();
    }
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
