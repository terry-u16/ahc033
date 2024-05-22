use crate::{
    common::ChangeMinMax as _,
    grid::Coord,
    problem::{Container, Grid, Input},
};
use itertools::{iproduct, Itertools};
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

#[derive(Debug, Clone)]
struct State {
    priority_direct: [[usize; Input::CONTAINER_COUNT]; Input::N],
    priority_temporary: [[usize; Input::CONTAINER_COUNT]; Input::N],
}

impl State {
    fn init() -> Self {
        let mut state = Self {
            priority_direct: [[0; Input::CONTAINER_COUNT]; Input::N],
            priority_temporary: [[0; Input::CONTAINER_COUNT]; Input::N],
        };

        for i in 0..Input::N {
            for j in 0..Input::CONTAINER_COUNT {
                state.priority_direct[i][j] = j;
                state.priority_temporary[i][j] = j + Input::N;
            }
        }

        state
    }

    fn neigh(
        &mut self,
        rng: &mut impl Rng,
    ) -> (
        usize,
        [usize; Input::CONTAINER_COUNT],
        [usize; Input::CONTAINER_COUNT],
    ) {
        let crane = rng.gen_range(0..Input::N);
        let prev_direct = self.priority_direct[crane];
        let prev_temporary = self.priority_temporary[crane];
        let direct = &mut self.priority_direct[crane];
        let temporary = &mut self.priority_temporary[crane];

        // insertする
        let mut order = vec![0; Input::CONTAINER_COUNT * 2];

        for i in 0..Input::CONTAINER_COUNT {
            order[direct[i]] = i;
            order[temporary[i]] = i;
        }

        let i = rng.gen_range(0..order.len());
        let v = order.remove(i);
        let i = rng.gen_range(0..=order.len());
        order.insert(i, v);
        let mut counts = [0; Input::CONTAINER_COUNT];

        // priorityに戻す
        for i in 0..order.len() {
            let v = order[i];

            if counts[v] == 0 {
                direct[v] = i;
            } else {
                temporary[v] = i;
            }

            counts[v] += 1;
        }

        (crane, prev_direct, prev_temporary)
    }

    fn revert(
        &mut self,
        crane: usize,
        prev_direct: [usize; Input::CONTAINER_COUNT],
        prev_temporary: [usize; Input::CONTAINER_COUNT],
    ) {
        self.priority_direct[crane] = prev_direct;
        self.priority_temporary[crane] = prev_temporary;
    }

    fn simulate(&self, input: &Input, recorder: &mut impl Recorder) {
        let i01234 = [0, 1, 2, 3, 4];
        let mut containers_in = i01234.map(|i| Some(input.containers()[i][0]));
        let mut crane_pos = i01234.map(|r| Coord::new(r, 0));
        let mut pointers = [0; Input::N];
        let mut temps: Vec<Container> = vec![];
        let mut last_turns = [0; Input::N];
        let mut completed_count = 0;
        let mut next_ship = [0, 5, 10, 15, 20];

        // 搬入口・搬出口・仮置き場の最後の作業が終わってマスに進入できるようになるターン
        let mut start_avaliable_turns = [i32::MIN; Input::N];
        let mut goal_avaliable_turns = [i32::MIN; Input::N];
        let mut temp_available_turns = [i32::MIN; Input::CONTAINER_COUNT];
        const TEMP_COL: usize = 3;

        while completed_count < Input::CONTAINER_COUNT {
            let mut turn = i32::MAX;
            let mut crane = !0;

            for i in 0..Input::N {
                if turn.change_min(last_turns[i]) {
                    crane = i;
                }
            }

            let mut best_carry = None;
            let mut best_priority = usize::MAX;

            // 搬入口から運び出す
            for (row, &container) in containers_in.iter().enumerate() {
                let Some(container) = container else {
                    continue;
                };
                let start = Coord::new(row, 0);
                let goal = Input::get_goal(container);

                if next_ship[goal.row()] == container.index() {
                    // 搬出口に運ぶ
                    if best_priority.change_min(self.priority_direct[crane][container.index()]) {
                        best_carry = Some(Carry::Direct(container, start, goal));
                    }
                } else {
                    // 仮置き場に運ぶ
                    if best_priority.change_min(self.priority_direct[crane][container.index()]) {
                        let goal = Coord::new(goal.row(), TEMP_COL);
                        best_carry = Some(Carry::ToTemporary(container, start, goal));
                    }
                };
            }

            // 仮置き場から運び出す
            for (index, &container) in temps.iter().enumerate() {
                let goal = Input::get_goal(container);
                let start = Coord::new(goal.row(), TEMP_COL);

                if next_ship[goal.row()] == container.index() {
                    // 搬出口に運ぶ
                    if best_priority.change_min(self.priority_temporary[crane][container.index()]) {
                        best_carry = Some(Carry::FromTemporary(container, start, goal, index));
                    }
                }
            }

            let carry = best_carry.expect("no carry found");

            // タスク実行
            let current_pos = crane_pos[crane];

            match carry {
                Carry::Direct(_, from, to) => {
                    let avail_turn = start_avaliable_turns[from.row()];
                    let pick_turn = (turn + current_pos.dist(&from) as i32 + 1).max(avail_turn + 1);
                    let avail_turn = goal_avaliable_turns[to.row()];
                    let ship_turn = (pick_turn + from.dist(&to) as i32 + 1).max(avail_turn + 1);
                    start_avaliable_turns[from.row()] = pick_turn + 1;
                    goal_avaliable_turns[to.row()] = ship_turn + 1;

                    pointers[from.row()] += 1;
                    next_ship[to.row()] += 1;
                    containers_in[from.row()] = input.containers()[from.row()]
                        .get(pointers[from.row()])
                        .copied();
                    crane_pos[crane] = to;
                    last_turns[crane] = ship_turn;
                    completed_count += 1;
                    recorder.record(crane, ship_turn, carry);
                }
                Carry::ToTemporary(container, from, to) => {
                    let avail_turn = start_avaliable_turns[from.row()];
                    let pick_turn = (turn + current_pos.dist(&from) as i32 + 1).max(avail_turn + 1);
                    let temp_turn = pick_turn + from.dist(&to) as i32 + 1;
                    start_avaliable_turns[from.row()] = pick_turn + 1;
                    temp_available_turns[container.index()] = temp_turn + 1;

                    temps.push(container);
                    pointers[from.row()] += 1;
                    containers_in[from.row()] = input.containers()[from.row()]
                        .get(pointers[from.row()])
                        .copied();
                    crane_pos[crane] = to;
                    last_turns[crane] = temp_turn;
                    recorder.record(crane, temp_turn, carry);
                }
                Carry::FromTemporary(container, from, to, index) => {
                    let avail_turn = temp_available_turns[container.index()];
                    let pick_turn = (turn + current_pos.dist(&from) as i32 + 1).max(avail_turn + 1);
                    let avail_turn = goal_avaliable_turns[to.row()];
                    let ship_turn = (pick_turn + from.dist(&to) as i32 + 1).max(avail_turn + 1);
                    goal_avaliable_turns[to.row()] = ship_turn + 1;

                    temps.swap_remove(index);
                    next_ship[to.row()] += 1;
                    crane_pos[crane] = to;
                    last_turns[crane] = ship_turn;
                    completed_count += 1;
                    recorder.record(crane, ship_turn, carry);
                }
            };
        }
    }

    fn calc_score(&self, input: &Input) -> f64 {
        const KAPPA: f64 = 3.0;
        let mut turns = TurnRecorder::new();
        self.simulate(input, &mut turns);

        // logsumexp関数
        let result = turns
            .last_turns
            .iter()
            .map(|&t| ((t as f64) / KAPPA).exp())
            .sum::<f64>()
            .ln()
            * KAPPA;

        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carry {
    Direct(Container, Coord, Coord),
    ToTemporary(Container, Coord, Coord),
    FromTemporary(Container, Coord, Coord, usize),
}

trait Recorder {
    fn record(&mut self, crane: usize, turn: i32, carry: Carry);
}

struct TurnRecorder {
    last_turns: [i32; Input::N],
}

impl TurnRecorder {
    fn new() -> Self {
        Self {
            last_turns: [0; Input::N],
        }
    }
}

impl Recorder for TurnRecorder {
    fn record(&mut self, crane: usize, turn: i32, _carry: Carry) {
        self.last_turns[crane] = turn;
    }
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input);
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
    let temp1 = 1e-2;
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
        let (crane, prev_direct, prev_temporary) = solution.neigh(&mut rng);

        // スコア計算
        let new_score = solution.calc_score(input);
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
            solution.revert(crane, prev_direct, prev_temporary);
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
    let state = State::init();
    let state = annealing(input, state, 1.0);
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
