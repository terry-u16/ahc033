use std::collections::BinaryHeap;

use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, ADJACENTS},
    problem::{CraneState, Grid, Input, Operation, Yard},
};

use super::task_gen::Task;

pub fn execute(yard: &Yard, tasks: &[Option<Task>; Input::N]) -> [Operation; Input::N] {
    let mut operations = [
        Operation::None,
        Operation::None,
        Operation::None,
        Operation::None,
        Operation::None,
    ];

    let mut cranes = yard.cranes().clone();

    for (i, task) in tasks.iter().enumerate() {
        if task.is_none() && cranes[i] != CraneState::Destroyed {
            operations[i] = Operation::Destroy;
        }
    }

    let mut done = [false; Input::CONTAINER_COUNT];
    let mut reserved = Grid::new([false; Input::N * Input::N]);

    for crane in cranes.iter() {
        let Some(c) = crane.coord() else {
            continue;
        };

        reserved[c] = true;
    }

    loop {
        let mut min_index = usize::MAX;
        let mut first_index = None;

        for (i, task) in tasks.iter().enumerate() {
            if let Some(task) = task {
                if !done[i] && min_index.change_min(task.index()) {
                    first_index = Some(i);
                }
            }
        }

        let Some(crane_index) = first_index else {
            break;
        };

        let task = tasks[crane_index].as_ref().unwrap();
        let crane = cranes[crane_index];
        done[crane_index] = true;

        let operation = match crane {
            CraneState::Empty(c) => {
                let op = if task.from() == c {
                    Operation::Pick
                } else {
                    move_to(
                        &mut cranes,
                        crane_index,
                        &task.board(),
                        &yard,
                        &mut reserved,
                        c,
                        task.from(),
                        false,
                    )
                };

                reserve(
                    &mut cranes,
                    &task.board(),
                    &mut reserved,
                    c,
                    task.from(),
                    false,
                );
                reserve(
                    &mut cranes,
                    &task.board(),
                    &mut reserved,
                    task.from(),
                    task.to(),
                    !Input::is_large_crane(crane_index),
                );

                op
            }
            CraneState::Holding(_, c) => {
                let op = if task.to() == c {
                    Operation::Drop
                } else {
                    move_to(
                        &mut cranes,
                        crane_index,
                        &task.board(),
                        &yard,
                        &mut reserved,
                        c,
                        task.to(),
                        !Input::is_large_crane(crane_index),
                    )
                };

                reserve(
                    &mut cranes,
                    &task.board(),
                    &mut reserved,
                    c,
                    task.to(),
                    !Input::is_large_crane(crane_index),
                );

                op
            }
            CraneState::Destroyed => unreachable!("crane is destroyed"),
        };

        operations[crane_index] = operation;
    }

    operations
}

fn reserve(
    cranes: &mut [CraneState],
    board: &Grid<bool>,
    reserved: &mut Grid<bool>,
    current: Coord,
    to: Coord,
    consider_containers: bool,
) {
    let path = dijkstra(cranes, board, current, to, consider_containers, reserved);

    for &c in path.iter() {
        if reserved[c] {
            //break;
        }

        reserved[c] = true;
    }
}

fn move_to(
    cranes: &mut [CraneState],
    crane_index: usize,
    board: &Grid<bool>,
    yard: &Yard,
    reserved: &mut Grid<bool>,
    current: Coord,
    to: Coord,
    consider_containers: bool,
) -> Operation {
    let path = dijkstra(cranes, board, current, to, consider_containers, reserved);
    let can_move = !reserved[path[0]] && (!consider_containers || yard.grid()[path[0]].is_none());

    if can_move {
        reserved[current] = false;
        reserved[path[0]] = true;

        cranes[crane_index] = match cranes[crane_index] {
            CraneState::Empty(_) => CraneState::Empty(path[0]),
            CraneState::Holding(container, _) => CraneState::Holding(container, path[0]),
            CraneState::Destroyed => unreachable!("crane is destroyed"),
        };

        let mut operation = Operation::None;
        const DIRECTIONS: [Operation; 4] = [
            Operation::Up,
            Operation::Right,
            Operation::Down,
            Operation::Left,
        ];

        for (&adj, &dir) in ADJACENTS.iter().zip(DIRECTIONS.iter()) {
            if current + adj == path[0] {
                operation = dir;
                break;
            }
        }

        operation
    } else {
        Operation::None
    }
}

fn dijkstra(
    cranes: &[CraneState],
    board: &Grid<bool>,
    start: Coord,
    goal: Coord,
    consider_containers: bool,
    reserved: &Grid<bool>,
) -> Vec<Coord> {
    let mut cost_map = Grid::new([1; Input::CONTAINER_COUNT]);

    // クレーンコスト
    for crane in cranes.iter() {
        let Some(coord) = crane.coord() else {
            continue;
        };

        cost_map[coord].change_max(3);
    }

    // コンテナコスト
    if consider_containers {
        for row in 0..Input::N {
            for col in 0..Input::N {
                let coord = Coord::new(row, col);

                if board[coord] {
                    cost_map[coord].change_max(100);
                }
            }
        }
    }

    // 予約コスト
    for row in 0..Input::N {
        for col in 0..Input::N {
            let c = Coord::new(row, col);

            if reserved[c] {
                cost_map[c] += 1;
            }
        }
    }

    let mut dists = Grid::new([i64::MAX / 2; Input::N * Input::N]);
    let mut from = Grid::new([start; Input::N * Input::N]);
    dists[start] = 0;
    let mut queue = BinaryHeap::new();
    queue.push(DijkstraState::new(0, start));

    while let Some(DijkstraState { cost, coord }) = queue.pop() {
        if coord == goal {
            break;
        } else if dists[coord] < cost {
            continue;
        }

        for &adj in ADJACENTS.iter() {
            let next = coord + adj;

            if next.in_map(Input::N) && dists[next].change_min(cost + cost_map[next]) {
                queue.push(DijkstraState::new(cost + cost_map[next], next));
                from[next] = coord;
            }
        }
    }

    let mut current = goal;
    let mut path = vec![];

    while current != start {
        path.push(current);
        current = from[current];
    }

    path.reverse();
    path
}

#[derive(Debug, Clone, Copy)]
struct DijkstraState {
    cost: i64,
    coord: Coord,
}

impl DijkstraState {
    fn new(cost: i64, coord: Coord) -> Self {
        Self { cost, coord }
    }
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.cmp(&other.cost).reverse()
    }
}
