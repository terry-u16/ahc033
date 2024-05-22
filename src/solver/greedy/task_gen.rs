use std::vec;

use crate::{
    common::ChangeMinMax as _,
    grid::Coord,
    problem::{Container, Grid, Input},
};
use itertools::{iproduct, Itertools};
use once_cell::sync::Lazy;
use rand::prelude::*;
use rand::Rng;

const N: usize = Input::N;
const NP1: usize = N + 1;
const N2: usize = N * N;

#[derive(Debug, Clone)]
pub struct Task {
    index: usize,
    container: Container,
    from: Coord,
    to: Coord,
    is_completed: bool,
    board: Grid<bool>,
}

impl Task {
    pub fn new(
        index: usize,
        container: Container,
        from: Coord,
        to: Coord,
        board: Grid<bool>,
    ) -> Self {
        Self {
            index,
            container,
            from,
            to,
            is_completed: false,
            board,
        }
    }

    pub fn index(&self) -> usize {
        self.index
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

    pub fn is_completed(&self) -> bool {
        self.is_completed
    }

    pub fn complete(&mut self) {
        self.is_completed = true;
    }

    pub fn board(&self) -> &Grid<bool> {
        &self.board
    }
}

const KAPPA: f64 = 1.0;

#[derive(Debug, Clone)]
struct State {
    temporary: [bool; Input::CONTAINER_COUNT],
}

impl State {
    fn new() -> Self {
        Self {
            temporary: [true; Input::CONTAINER_COUNT],
        }
    }

    fn calc_score(&self, input: &Input) -> Result<f64, ()> {
        const CONT: usize = Input::CONTAINER_COUNT;

        // [搬入口積み * 25, 一時下ろし * 25, 一時積み * 25, 搬出口下ろし * 25, START]
        let mut graph = vec![vec![]; CONT * 4 + 1];

        // 積み→下ろし
        for row in 0..Input::N {
            let from = Coord::new(row, 0);

            for i in 0..Input::N {
                let container = input.containers()[row][i].index();
                let goal = Input::get_goal(Container::new(container));

                if self.temporary[container] {
                    // 一時保管あり
                    const TEMP_COL: usize = 3;
                    let temp = Coord::new(goal.row(), TEMP_COL);
                    graph[container + CONT].push((container, from.dist(&temp) + 2));
                    graph[container + CONT * 2].push((container + CONT * 1, 0));
                    graph[container + CONT * 3].push((container + CONT * 2, temp.dist(&goal) + 2));
                } else {
                    // 一時保管なし
                    graph[container + CONT * 3].push((container, from.dist(&goal) + 2));
                }
            }
        }

        // 積み → 積み
        for row in 0..Input::N {
            for i in 0..Input::N - 1 {
                let container0 = input.containers()[row][i].index();
                let container1 = input.containers()[row][i + 1].index();
                graph[container1].push((container0, 2));
            }
        }

        // 下ろし → 下ろし
        for i in 0..Input::N {
            for i in i * Input::N..(i + 1) * Input::N - 1 {
                let container0 = i;
                let container1 = i + 1;
                graph[container1 + CONT * 3].push((container0 + CONT * 3, 2));
            }
        }

        // start
        for row in 0..Input::N {
            let container = input.containers()[row][0].index();
            graph[container].push((CONT * 4, 0));
        }

        let mut inv_graph = vec![vec![]; CONT * 4 + 1];

        for (v, edges) in graph.iter().enumerate() {
            for &(u, cost) in edges.iter() {
                inv_graph[u].push((v, cost));
            }
        }

        let exp_table = EXP_TABLE.as_slice();
        let mut dp = vec![0.0; CONT * 4 + 1];
        let mut stack = Vec::new();
        let mut indegrees = vec![0; CONT * 4 + 1];

        for edge in graph.iter() {
            for &(v, _) in edge.iter() {
                indegrees[v] += 1;
            }
        }

        for v in 0..CONT * 4 + 1 {
            if indegrees[v] == 0 {
                for &(u, cost) in graph[v].iter() {
                    indegrees[u] -= 1;

                    if indegrees[u] == 0 {
                        stack.push(u);
                    }
                }
            }
        }

        while let Some(v) = stack.pop() {
            let mut sum = 0.0;

            for &(u, cost) in inv_graph[v].iter() {
                sum += ((dp[u] + cost as f64) / KAPPA).exp();
            }

            for &(u, _) in graph[v].iter() {
                indegrees[u] -= 1;
                if indegrees[u] == 0 {
                    stack.push(u);
                }
            }

            dp[v] = KAPPA * sum.ln();
        }

        if indegrees.iter().any(|&x| x != 0) {
            Err(())
        } else {
            let logsumexp = dp[CONT * 4];
            Ok(logsumexp)
        }
    }
}

static EXP_TABLE: Lazy<Vec<f64>> = Lazy::new(|| {
    let mut table = vec![];

    for i in 0..=1000 {
        table.push((i as f64 / KAPPA).exp());
    }

    table
});

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input).unwrap();
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e1;
    let temp1 = 1e-1;
    let mut inv_temp = 1.0 / temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let index = rng.gen_range(0..Input::CONTAINER_COUNT);
        solution.temporary[index] ^= true;

        // スコア計算
        let Ok(new_score) = solution.calc_score(input) else {
            solution.temporary[index] ^= true;
            continue;
        };
        let score_diff = new_score - current_score;

        if score_diff <= 0.0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;

            if best_score.change_min(current_score) {
                best_solution = solution.clone();
                update_count += 1;
            }
        } else {
            solution.temporary[index] ^= true;
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("init score : {}", init_score);
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    best_solution
}

pub fn generate_tasks(input: &Input, rng: &mut impl Rng) -> Result<Vec<Task>, &'static str> {
    let state = State::new();
    let state = annealing(input, state, 1.0);
    eprintln!("state: {:?}", state);

    panic!();

    let (max_stock, history) = dp(input);
    //eprintln!("max_stock: {}", max_stock);

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
            let task = Task::new(tasks.len(), container, from, to, board.clone());
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
                    let task = Task::new(tasks.len(), container, from, best_pos, board.clone());
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
                tasks.len(),
                Container::new(container),
                pos,
                Input::get_goal(Container::new(container)),
                board.clone(),
            );
            tasks.push(task);
        }
    }

    Ok(tasks)
}

/// 滞留コンテナ数の最大値を最小にするDPを行い、搬入順（何番目にどの行から搬入するか）を返す
fn dp(input: &Input) -> (u128, Vec<usize>) {
    let mut counts = vec![vec![vec![vec![vec![0; NP1]; NP1]; NP1]; NP1]; NP1];

    for i0 in 0..NP1 {
        for i1 in 0..NP1 {
            for i2 in 0..NP1 {
                for i3 in 0..NP1 {
                    for i4 in 0..NP1 {
                        let indices = [i0, i1, i2, i3, i4];
                        counts[i0][i1][i2][i3][i4] = count(input, indices);
                    }
                }
            }
        }
    }

    let mut dp = vec![vec![vec![vec![vec![u128::MAX / 2; NP1]; NP1]; NP1]; NP1]; NP1];
    dp[0][0][0][0][0] = 0;

    let mut from = vec![vec![vec![vec![vec![([0; N], 0); NP1]; NP1]; NP1]; NP1]; NP1];

    for (i0, i1, i2, i3, i4) in iproduct!(0..NP1, 0..NP1, 0..NP1, 0..NP1, 0..NP1) {
        let old_indices = [i0, i1, i2, i3, i4];
        let current_dp = dp[i0][i1][i2][i3][i4];
        let current_cnt = counts[i0][i1][i2][i3][i4];

        for idx in 0..N {
            let mut indices = old_indices;
            indices[idx] += 1;

            if indices[idx] > N {
                continue;
            }

            let cnt = counts[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]];
            let new_cnt = current_cnt.max(cnt);
            let cost = current_dp + (1 << (new_cnt * 4));

            if dp[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]].change_min(cost) {
                from[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]] =
                    (old_indices, idx);
            }
        }
    }

    let mut current = [N; N];
    let mut history = vec![];

    while current != [0; N] {
        let (prev, index) = from[current[0]][current[1]][current[2]][current[3]][current[4]];
        history.push(index);
        current = prev;
    }

    history.reverse();

    (dp[N][N][N][N][N], history)
}

fn count(input: &Input, indices: [usize; N]) -> u32 {
    let mut contains = [false; N2];

    for (i, &c) in indices.iter().enumerate() {
        for &c in input.containers()[i][..c].iter() {
            contains[c.index()] = true;
        }
    }

    let mut count = indices.iter().map(|&i| i as u32).sum::<u32>();

    for i in 0..N {
        let slice = &contains[i * N..(i + 1) * N];
        count -= slice.iter().take_while(|&&b| b).count() as u32;
    }

    count
}
