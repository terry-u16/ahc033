mod step01_01_dp;
mod step01_02_beam;

use crate::{
    common::ChangeMinMax as _,
    grid::Coord,
    problem::{Container, Grid, Input},
    solver::beam::step01_gen::step01_01_dp::dp,
};
use itertools::Itertools;
use rand::prelude::*;
use rand::Rng;

const N: usize = Input::N;
const NP1: usize = N + 1;
const N2: usize = N * N;

#[derive(Debug, Clone)]
pub struct Task {
    container: Container,
    from: Coord,
    to: Coord,
}

impl Task {
    pub fn new(container: Container, from: Coord, to: Coord) -> Self {
        Self {
            container,
            from,
            to,
        }
    }

    pub fn container(&self) -> Container {
        self.container
    }

    pub fn from(&self) -> Coord {
        self.from
    }

    pub fn to(&self) -> Coord {
        self.to
    }
}

pub(super) fn generate_tasks(input: &Input, rng: &mut impl Rng) -> Result<Vec<Task>, &'static str> {
    let (_, history) = dp(input);
    eprintln!("{:?}", history);

    return Ok(step01_02_beam::beam(input));

    let mut tasks = vec![];
    let mut containers = input
        .containers()
        .map(|c| c.iter().copied().rev().collect_vec());

    let mut storages = [
        Coord::new(0, 3),
        Coord::new(2, 3),
        Coord::new(4, 3),
        Coord::new(0, 2),
        Coord::new(2, 2),
        Coord::new(4, 2),
    ];

    let mut board = Grid::new([false; Input::CONTAINER_COUNT]);
    board[Coord::new(0, 0)] = true;
    board[Coord::new(1, 0)] = true;
    board[Coord::new(2, 0)] = true;
    board[Coord::new(3, 0)] = true;
    board[Coord::new(4, 0)] = true;

    let mut positions = [None; Input::CONTAINER_COUNT];
    let mut next_shippings = [0, 5, 10, 15, 20];

    for &row in history.iter() {
        let container = containers[row].pop().unwrap();
        let from = Coord::new(row, 0);
        let to = Input::get_goal(container);

        // 搬入口クリア
        if containers[row].is_empty() {
            board[from] = false;
        }

        if container.index() == next_shippings[to.row()] {
            next_shippings[to.row()] += 1;
            let task = Task::new(container, from, to);
            tasks.push(task);
        } else {
            // ベストな場所を探す
            let mut best_pos = None;
            let mut best_cost = usize::MAX;
            storages.shuffle(rng);

            for &cand in storages.iter() {
                if board[cand] {
                    continue;
                }

                let cost = from.dist(&cand) + cand.dist(&to);

                if best_cost.change_min(cost) {
                    best_pos = Some(cand);
                }
            }

            match best_pos {
                Some(best_pos) => {
                    positions[container.index()] = Some(best_pos);
                    let task = Task::new(container, from, best_pos);
                    tasks.push(task);
                    board[best_pos] = true;
                }
                None => {
                    return Err("storage positions are occupied");
                }
            }
        }

        loop {
            let mut next = None;
            const ORDER: [usize; 25] = [
                0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14,
                19, 24,
            ];

            for &container in ORDER.iter() {
                let Some(pos) = positions[container] else {
                    continue;
                };

                if next_shippings[Input::get_goal(Container::new(container)).row()] != container {
                    continue;
                }

                next = Some((container, pos));
                break;
            }

            let Some((container, pos)) = next else {
                break;
            };

            board[pos] = false;
            positions[container] = None;
            next_shippings[Input::get_goal(Container::new(container)).row()] += 1;
            let task = Task::new(
                Container::new(container),
                pos,
                Input::get_goal(Container::new(container)),
            );
            tasks.push(task);
        }
    }

    Ok(tasks)
}
