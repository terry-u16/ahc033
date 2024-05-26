use crate::{
    grid::Coord,
    problem::Input,
    solver::beam::{step02_order::SubTask, Precalc},
};
use itertools::Itertools;
use std::{collections::HashMap, iter::repeat};

const DEPOTS: [Coord; 16] = [
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
    Coord::new(0, 4),
    Coord::new(1, 4),
    Coord::new(2, 4),
    Coord::new(3, 4),
    Coord::new(4, 4),
];

pub(super) fn critical_path_analysis(
    tasks: &[Vec<SubTask>; Input::N],
    precalc: &Precalc,
) -> TaskSet {
    let all_tasks = tasks.iter().flatten().copied().collect_vec();
    let cranes = (0..Input::N)
        .flat_map(|i| repeat(i).take(tasks[i].len()).collect_vec())
        .collect_vec();
    let mut init_ptr = [0; Input::N];
    let mut end_ptr = [0; Input::N];

    for (i, t) in tasks.iter().take(Input::N - 1).enumerate() {
        init_ptr[i + 1] = init_ptr[i] + t.len() as u8;
        end_ptr[i] = init_ptr[i + 1] as u8;
    }

    end_ptr[Input::N - 1] = all_tasks.len() as u8;

    let mut grid = HashMap::new();

    for &d in DEPOTS.iter() {
        grid.insert(d, vec![]);
    }

    for (i, t) in all_tasks.iter().enumerate() {
        let (Some(coord), Some(index)) = (t.coord(), t.index()) else {
            continue;
        };

        let v = grid.get_mut(&coord).unwrap();

        if v.len() <= index {
            v.resize(index + 1, !0);
        }
        grid.get_mut(&coord).unwrap()[index] = i;
    }

    // DAGを構築
    let mut graph = vec![vec![]; all_tasks.len()];

    // 同クレーン内の依存関係
    for i in 0..all_tasks.len() - 1 {
        if let (Some(c0), Some(c1)) = (all_tasks[i].coord(), all_tasks[i + 1].coord()) {
            graph[i + 1].push((i, c0.dist(&c1) + 1));
        } else if all_tasks[i + 1] == SubTask::EndOfOrder {
            graph[i + 1].push((i, 0));
        }
    }

    // 位置別の依存関係
    for (i, t) in all_tasks.iter().enumerate() {
        let (Some(coord), Some(index)) = (t.coord(), t.index()) else {
            continue;
        };

        let v = grid.get(&coord).unwrap();
        let Some(prev) = v.get(index.wrapping_sub(1)).copied() else {
            continue;
        };

        if cranes[prev] != cranes[i] {
            graph[i].push((prev, 2));
        }
    }

    let mut dp = vec![0.0; all_tasks.len()];
    let mut indegrees = vec![0; all_tasks.len()];
    let mut stack = vec![];

    for edges in graph.iter() {
        for &(v, _) in edges.iter() {
            indegrees[v] += 1;
        }
    }

    for v in 0..all_tasks.len() {
        if indegrees[v] == 0 {
            stack.push(v);
            dp[v] = precalc.exp_table[0];
        }
    }

    while let Some(v) = stack.pop() {
        for &(u, cost) in graph[v].iter() {
            dp[u] += dp[v] * precalc.exp_table[cost];
            indegrees[u] -= 1;

            if indegrees[u] == 0 {
                stack.push(u);
            }
        }
    }

    assert!(indegrees.iter().all(|&x| x == 0));

    TaskSet::new(all_tasks, dp, init_ptr, end_ptr)
}

#[derive(Debug, Clone)]
pub struct TaskSet {
    pub tasks: Vec<SubTask>,
    pub dp: Vec<f64>,
    pub init_ptr: [u8; Input::N],
    pub end_ptr: [u8; Input::N],
}

impl TaskSet {
    pub fn new(
        tasks: Vec<SubTask>,
        dp: Vec<f64>,
        init_ptr: [u8; Input::N],
        end_ptr: [u8; Input::N],
    ) -> Self {
        Self {
            tasks,
            dp,
            init_ptr,
            end_ptr,
        }
    }
}
