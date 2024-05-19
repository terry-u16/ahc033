use std::collections::BinaryHeap;

use super::task_gen::Task;
use crate::{
    common::ChangeMinMax,
    grid::{Coord, ADJACENTS},
    problem::{CraneState, Grid, Input, Yard},
};
use ac_library::MinCostFlowGraph;
use itertools::Itertools;

pub fn assign_tasks(yard: &Yard, tasks: &[Task]) -> [Option<Task>; Input::N] {
    let mut assigns = [None, None, None, None, None];
    let mut candidates = vec![];
    let mut containers = [false; Input::CONTAINER_COUNT];

    let mut task_count = 0;
    let available_crane_count = yard
        .cranes()
        .iter()
        .filter(|c| match c {
            CraneState::Empty(_) => true,
            CraneState::Holding(_, _) => true,
            CraneState::Destroyed => false,
        })
        .count();

    for task in tasks.iter() {
        if task.is_completed() || containers[task.container().index()] {
            continue;
        }

        let crane = yard.cranes().iter().enumerate().find(|(_, c)| match c {
            CraneState::Empty(_) => false,
            CraneState::Holding(container, _) => *container == task.container(),
            CraneState::Destroyed => false,
        });

        if let Some((index, _)) = crane {
            assigns[index] = Some(task.clone());
            containers[task.container().index()] = true;
        } else {
            candidates.push(task.clone());
        }

        task_count += 1;

        if task_count >= available_crane_count {
            break;
        }
    }

    matching(yard, &candidates, assigns)
}

fn matching(
    yard: &Yard,
    tasks: &[Task],
    mut assigns: [Option<Task>; Input::N],
) -> [Option<Task>; Input::N] {
    let mut edge_indices = vec![];
    let cranes = assigns
        .iter()
        .enumerate()
        .filter(|(_, t)| t.is_none())
        .map(|(i, _)| i)
        .collect_vec();
    let crane_count = cranes.len();
    let task_count = tasks.len();

    let mut graph = MinCostFlowGraph::new(crane_count + task_count + 2);

    for (i, &crane) in cranes.iter().enumerate() {
        for (j, task) in tasks.iter().enumerate() {
            let Some(crane_coord) = yard.cranes()[crane].coord() else {
                continue;
            };

            // 自分のことは考慮しない
            let mut cranes = yard.cranes().clone();
            cranes[crane] = CraneState::Destroyed;

            // 回送コスト
            let cost1 = dijkstra(&cranes, task.board(), crane_coord, task.from(), false);

            // 運搬コスト
            let consider = !Input::is_large_crane(crane);
            let cost2 = dijkstra(&cranes, task.board(), task.from(), task.to(), consider);

            let cost_mul = 1 << (40 - j * 10);

            graph.add_edge(i, j + crane_count, 1, (cost1 + cost2) * cost_mul);
            edge_indices.push((i, j));
        }
    }

    let source = crane_count + task_count;
    let sink = source + 1;

    for i in 0..crane_count {
        graph.add_edge(source, i, 1, 0);
    }

    for j in 0..task_count {
        graph.add_edge(j + crane_count, sink, 1, 0);
    }

    graph.flow(source, sink, i64::MAX);

    for (index, &(i, j)) in edge_indices.iter().enumerate() {
        if graph.get_edge(index).flow > 0 {
            assigns[cranes[i]] = Some(tasks[j].clone());
        }
    }

    assigns
}

fn dijkstra(
    cranes: &[CraneState],
    board: &Grid<bool>,
    start: Coord,
    goal: Coord,
    consider_containers: bool,
) -> i64 {
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

    let mut dists = Grid::new([i64::MAX / 2; Input::N * Input::N]);
    dists[start] = 0;
    let mut queue = BinaryHeap::new();
    queue.push(DijkstraState::new(0, start));

    while let Some(DijkstraState { cost, coord }) = queue.pop() {
        if coord == goal {
            return cost;
        } else if dists[coord] < cost {
            continue;
        }

        for &adj in ADJACENTS.iter() {
            let next = coord + adj;

            if next.in_map(Input::N) && dists[next].change_min(cost + cost_map[next]) {
                queue.push(DijkstraState::new(cost + cost_map[next], next));
            }
        }
    }

    dists[goal]
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
