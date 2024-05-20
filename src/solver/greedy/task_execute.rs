use itertools::Itertools;
use rand::prelude::*;
use std::collections::VecDeque;

use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, ADJACENTS},
    problem::{CraneState, Grid, Input, Operation, Yard},
};

use super::task_gen::Task;

pub fn execute(
    yard: &Yard,
    tasks: &[Option<Task>; Input::N],
    rng: &mut impl Rng,
) -> [Operation; Input::N] {
    let mut operations = [
        Operation::None,
        Operation::None,
        Operation::None,
        Operation::None,
        Operation::None,
    ];

    let (dists_container, dists_no_container) = calc_dists_all(yard);
    let mut crane_order = (0..Input::N)
        .filter(|&i| yard.cranes()[i] != CraneState::Destroyed)
        .collect_vec();
    crane_order.sort_by_key(|&i| tasks[i].as_ref().map_or(usize::MAX, |t| t.index()));

    // 操作の候補を列挙
    let mut candidates = [vec![], vec![], vec![], vec![], vec![]];
    let moves = [
        Operation::None,
        Operation::Up,
        Operation::Right,
        Operation::Down,
        Operation::Left,
    ];

    for (i, &crane_i) in crane_order.iter().enumerate() {
        let task = &tasks[crane_i];
        let crane = yard.cranes()[crane_i];
        let candidates = &mut candidates[i];

        match (task, crane) {
            (None, CraneState::Empty(_)) => candidates.push(Operation::Destroy),
            (None, CraneState::Holding(_, _)) => unreachable!("Holding crane shold have task"),
            (None, CraneState::Destroyed) => unreachable!("Destroyed crane shold not be here"),
            (Some(task), CraneState::Empty(coord)) => {
                if task.from() == coord {
                    if yard.grid()[coord] == Some(task.container()) {
                        candidates.push(Operation::Pick);
                    } else {
                        for &op in moves[1..].iter() {
                            let next = coord + op.dir();
                            if next.in_map(Input::N) {
                                candidates.push(op);
                            }
                        }
                    }
                } else {
                    for &op in moves.iter() {
                        let next = coord + op.dir();
                        if next.in_map(Input::N) {
                            candidates.push(op);
                        }
                    }
                }
            }
            (Some(task), CraneState::Holding(_, coord)) => {
                if task.to() == coord {
                    candidates.push(Operation::Drop);
                } else {
                    for &op in moves.iter() {
                        let next = coord + op.dir();
                        if next.in_map(Input::N)
                            && (Input::is_large_crane(crane_i) || yard.grid()[next].is_none())
                        {
                            candidates.push(op);
                        }
                    }
                }
            }
            (Some(_), CraneState::Destroyed) => unreachable!("Destroyed crane shold not be here"),
        }

        candidates.shuffle(rng);
    }

    let mut best_operations = operations.clone();
    let mut cant_in = yard.grid().map(|_| false);
    let mut cant_move = yard.grid().map(|_| [false; 8]);
    let mut max_dists = yard.grid().map(|_| 0);
    let mut best_score = i32::MAX;

    dfs(
        &mut operations,
        &mut best_operations,
        &candidates,
        &mut cant_in,
        &mut cant_move,
        &mut max_dists,
        &dists_container,
        &dists_no_container,
        yard.cranes(),
        tasks,
        &crane_order,
        0,
        0,
        &mut best_score,
    );

    best_operations
}

fn dfs(
    operations: &mut [Operation; Input::N],
    best_operations: &mut [Operation; Input::N],
    candidates: &[Vec<Operation>],
    cant_in: &mut Grid<bool>,
    cant_move: &mut Grid<[bool; 8]>,
    max_dists: &mut Grid<i32>,
    dists_container: &Grid<Grid<i32>>,
    dists_no_container: &Grid<Grid<i32>>,
    cranes: &[CraneState; Input::N],
    tasks: &[Option<Task>; Input::N],
    crane_order: &[usize],
    depth: usize,
    score: i32,
    best_score: &mut i32,
) {
    if depth == crane_order.len() {
        if best_score.change_min(score) {
            *best_operations = *operations;
        }

        return;
    }

    let crane_i = crane_order[depth];
    let crane = cranes[crane_i];
    let task = &tasks[crane_i];
    let dists_back = dists_no_container;
    let dists_forward = if Input::is_large_crane(crane_i) {
        dists_no_container
    } else {
        dists_container
    };

    for &op in candidates[depth].iter() {
        let coord = crane.coord().unwrap();
        let next = coord + op.dir();
        let op_usize = op as usize;

        if (cant_in[next] && op != Operation::Destroy) || cant_move[coord][op_usize] {
            continue;
        }

        let mut new_score = score;
        let dist = if let Some(task) = task {
            match crane {
                CraneState::Empty(_) => {
                    dists_back[next][task.from()] + dists_forward[task.from()][task.to()]
                }
                CraneState::Holding(_, _) => dists_forward[next][task.to()],
                CraneState::Destroyed => 0,
            }
        } else {
            0
        };

        let score_mul = 1 << (Input::N - depth - 1) as i32;

        let old_max_dist = if let Some(goal) = task.as_ref().map(|t| t.to()) {
            let old_max_dist = max_dists[goal];
            new_score += (max_dists[goal] - dist).max(0) * 2 * score_mul;
            max_dists[goal].change_max(dist);
            old_max_dist
        } else {
            0
        };

        new_score += dist * score_mul;
        operations[crane_i] = op;

        if op != Operation::Destroy {
            cant_in[next] = true;
        }

        // クロスする移動もNG
        if op_usize < 4 {
            cant_move[next][op_usize ^ 2] = true;
        }

        dfs(
            operations,
            best_operations,
            candidates,
            cant_in,
            cant_move,
            max_dists,
            dists_container,
            dists_no_container,
            cranes,
            tasks,
            crane_order,
            depth + 1,
            new_score,
            best_score,
        );

        if op != Operation::Destroy {
            cant_in[next] = false;
        }

        if op_usize < 4 {
            cant_move[next][op_usize ^ 2] = false;
        }

        if let Some(goal) = task.as_ref().map(|t| t.to()) {
            max_dists[goal] = old_max_dist;
        }
    }
}

fn calc_dists_all(yard: &Yard) -> (Grid<Grid<i32>>, Grid<Grid<i32>>) {
    let map_container = yard.grid().map(|c| c.is_none());
    let map_no_container = yard.grid().map(|_| true);

    let mut starts = Grid::new([Coord::new(0, 0); Input::N * Input::N]);

    for row in 0..Input::N {
        for col in 0..Input::N {
            let c = Coord::new(row, col);
            starts[c] = c;
        }
    }

    let dists_container = starts.map(|c| bfs(&map_container, c));
    let dists_no_container = starts.map(|c| bfs(&map_no_container, c));

    (dists_container, dists_no_container)
}

fn bfs(map: &Grid<bool>, start: Coord) -> Grid<i32> {
    let mut dists = Grid::new([i32::MAX / 2; Input::N * Input::N]);
    let mut queue = VecDeque::new();
    dists[start] = 0;
    queue.push_back(start);

    while let Some(coord) = queue.pop_front() {
        for &adj in ADJACENTS.iter() {
            let next = coord + adj;
            let next_dist = dists[coord] + 1;

            if next.in_map(Input::N) && map[next] && dists[next].change_min(next_dist) {
                queue.push_back(next);
            }
        }
    }

    // コンテナがあるマスがINFにならないようにする
    let mut dists_no_inf = dists.clone();

    for row in 0..Input::N {
        for col in 0..Input::N {
            let c = Coord::new(row, col);

            for &adj in ADJACENTS.iter() {
                let next = c + adj;

                if next.in_map(Input::N) {
                    let d = dists[next] + 1;
                    dists_no_inf[c].change_min(d);
                }
            }
        }
    }

    dists_no_inf
}
