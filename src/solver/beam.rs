use std::{array, collections::VecDeque};

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, ADJACENTS},
    problem::{Grid, Input, Output, Yard},
};

use super::{Solver, SolverResult};

mod task_execute;
mod task_gen;
mod task_order;

pub struct BeamSolver {
    seed: u64,
    max_turn: usize,
}

impl BeamSolver {
    pub fn new(seed: u64, max_turn: usize) -> Self {
        Self { seed, max_turn }
    }
}

impl Solver for BeamSolver {
    fn solve(&self, input: &crate::problem::Input) -> Result<super::SolverResult, &'static str> {
        let mut rng = Pcg64Mcg::seed_from_u64(self.seed);
        let precalc = Precalc::new();
        let all_tasks = task_gen::generate_tasks(input, &precalc, &mut rng)?;

        let subtasks = task_order::order_tasks(input, &precalc, &all_tasks)?;

        let operations = task_execute::execute(input, &precalc, &subtasks, self.max_turn)?;
        let mut yard = Yard::new(&input);
        let mut output = Output::new();

        for op in operations.iter() {
            yard.apply(op)?;
            yard.carry_in_and_ship();
            output.push(op);
        }

        Ok(SolverResult::new(output, &yard))
    }
}

#[derive(Debug, Clone)]
struct Precalc {
    dist_dict: DistDict,
    exp_table: [f64; 1000],
}

impl Precalc {
    const KAPPA: f64 = 1.0;

    fn new() -> Self {
        let exp_table = array::from_fn(|i| (i as f64 / Self::KAPPA).exp());

        Self {
            dist_dict: DistDict::new(),
            exp_table,
        }
    }
}

#[derive(Debug, Clone)]
struct DistDict {
    dists: [Vec<Grid<Grid<usize>>>; 2],
    next: [Vec<Grid<Grid<Coord>>>; 2],
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
        let mut dists_with_container = vec![];
        let mut next_with_container = vec![];

        for flag in 0..1 << Self::STORAGES.len() {
            let mut board = Grid::new([false; Input::N * Input::N]);

            for (i, &storage) in Self::STORAGES.iter().enumerate() {
                if (flag & (1 << i)) > 0 {
                    board[storage] = true;
                }
            }

            let mut d =
                Grid::new([Grid::new([usize::MAX; Input::N * Input::N]); Input::N * Input::N]);
            let mut next = Grid::new(
                [Grid::new([Coord::new(0, 0); Input::N * Input::N]); Input::N * Input::N],
            );

            for row in 0..Input::N {
                for col in 0..Input::N {
                    let c = Coord::new(row, col);
                    Self::bfs(&board, &mut d[c], &mut next[c], c);
                }
            }

            dists_with_container.push(d);
            next_with_container.push(next);
        }

        // コンテナなし = flagが0
        let dists_without_container =
            vec![dists_with_container[0].clone(); dists_with_container.len()];
        let next_without_container =
            vec![next_with_container[0].clone(); next_with_container.len()];

        Self {
            dists: [dists_without_container, dists_with_container],
            next: [next_without_container, next_with_container],
        }
    }

    fn bfs(board: &Grid<bool>, dists: &mut Grid<usize>, next: &mut Grid<Coord>, from: Coord) {
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

        // INFになるのを防ぐ
        let mut temp = dists.clone();

        for row in 0..Input::N {
            for col in 0..Input::N {
                let c = Coord::new(row, col);

                for &adj in ADJACENTS.iter() {
                    let next = c + adj;

                    if next.in_map(Input::N) {
                        temp[c].change_min(dists[next].saturating_add(1));
                    }
                }
            }
        }

        *dists = temp;

        for row in 0..Input::N {
            for col in 0..Input::N {
                let c = Coord::new(row, col);
                let mut best = c;
                let mut best_dist = dists[best];

                for &adj in ADJACENTS.iter() {
                    let next = c + adj;

                    if next.in_map(Input::N) && best_dist.change_min(dists[next]) {
                        best = next;
                    }
                }

                next[c] = best;
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

    fn dist(&self, flag: StorageFlag, from: Coord, to: Coord, consider_container: bool) -> usize {
        // [to][from]の順になることに注意
        self.dists[consider_container as usize][flag.0][to][from]
    }

    fn next(&self, flag: StorageFlag, from: Coord, to: Coord, consider_container: bool) -> Coord {
        self.next[consider_container as usize][flag.0][to][from]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StorageFlag(usize);
