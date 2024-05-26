use std::array;

use crate::{
    common::ChangeMinMax as _,
    data_structures::{History, HistoryIndex},
    grid::Coord,
    problem::{Container, Grid, Input},
};
use itertools::{iproduct, Itertools};
use rand::prelude::*;
use rand::Rng;

use super::{DistDict, Precalc};

const N: usize = Input::N;
const NP1: usize = N + 1;
const N2: usize = N * N;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Task {
    crane: u8,
    container: Container,
    from: Coord,
    to: Coord,
}

impl Task {
    const DUMMY: Self = Self {
        crane: !0,
        container: Container::new(!0),
        from: Coord::new(!0, !0),
        to: Coord::new(!0, !0),
    };

    const fn new(crane: u8, container: Container, from: Coord, to: Coord) -> Self {
        Self {
            crane,
            container,
            from,
            to,
        }
    }

    pub fn crane(&self) -> usize {
        self.crane as usize
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

pub(super) fn generate_tasks(
    input: &Input,
    precalc: &Precalc,
    rng: &mut impl Rng,
) -> Result<Vec<Task>, &'static str> {
    let env = Env::new(&input, &precalc.dist_dict);
    let mut beam = vec![vec![vec![]; State::STORAGE_COUNT + 1]; u8::MAX as usize];
    beam[0][0].push(State::init(input));
    let mut history = History::new();
    const BEAM_WIDTH: usize = 3000;

    for turn in 0.. {
        let completed = beam
            .iter()
            .flatten()
            .flatten()
            .filter(|s| s.is_completed())
            .next();

        if let Some(&completed) = completed {
            let hist = history.collect(completed.history);
            eprintln!("turn: {}", turn);
            todo!();
        }

        for storage in 0..=State::STORAGE_COUNT {
            let mut b = vec![];
            std::mem::swap(&mut b, &mut beam[turn as usize][storage as usize]);

            if b.len() > BEAM_WIDTH {
                b.select_nth_unstable(BEAM_WIDTH);
                b.truncate(BEAM_WIDTH);
            }

            for state in b.iter_mut() {
                state.gen_next(&env, &mut beam, &mut history);
            }
        }
    }

    todo!();
}

struct Env<'a> {
    input: &'a Input,
    dist_dict: &'a DistDict,
    index_grid: Grid<usize>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input, dist_dict: &'a DistDict) -> Self {
        let mut index_grid = Grid::new([!0; Input::N * Input::N]);

        for (i, c) in State::POS.iter().enumerate() {
            index_grid[c] = i;
        }

        Self {
            input,
            dist_dict,
            index_grid,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct State {
    in_ptr: [u8; Input::N],
    container_avail_turns: [u8; Input::N + Self::STORAGE_COUNT],
    containers: [Option<Container>; Input::N + Self::STORAGE_COUNT],
    cranes: [Coord; Input::N],
    out_next: [u8; Input::N],
    out_avail_turns: [u8; Input::N],
    crane_avail_turns: [u8; Input::N],
    crane_score_per_turn: [f32; Input::N],
    temp_count: u8,
    finished_container_count: u8,
    score: f32,
    history: HistoryIndex,
}

impl State {
    const STORAGE_COUNT: usize = 6;
    const POS: [Coord; Input::N * 2 + Self::STORAGE_COUNT] = [
        // 搬入口
        Coord::new(0, 0),
        Coord::new(1, 0),
        Coord::new(2, 0),
        Coord::new(3, 0),
        Coord::new(4, 0),
        // 一時保管
        Coord::new(0, 2),
        Coord::new(2, 2),
        Coord::new(4, 2),
        Coord::new(0, 3),
        Coord::new(2, 3),
        Coord::new(4, 3),
        // 搬出口
        Coord::new(0, 4),
        Coord::new(1, 4),
        Coord::new(2, 4),
        Coord::new(3, 4),
        Coord::new(4, 4),
    ];
    const CRANE_AVAIL: u8 = u8::MAX;

    fn init(input: &Input) -> Self {
        let mut containers = [None; Input::N + Self::STORAGE_COUNT];
        for i in 0..Input::N {
            containers[i] = Some(input.containers()[i][0]);
        }

        Self {
            in_ptr: [0; Input::N],
            container_avail_turns: [0; Input::N + Self::STORAGE_COUNT],
            containers,
            temp_count: 0,
            finished_container_count: 0,
            cranes: array::from_fn(|i| Coord::new(i, 0)),
            out_next: [0, 5, 10, 15, 20],
            out_avail_turns: [0; Input::N],
            crane_avail_turns: [0; Input::N],
            crane_score_per_turn: [0.0; Input::N],
            score: 0.0,
            history: HistoryIndex::ROOT,
        }
    }

    fn is_completed(&self) -> bool {
        self.finished_container_count == Input::CONTAINER_COUNT as u8
    }

    fn gen_next(&mut self, env: &Env, beam: &mut Vec<Vec<Vec<Self>>>, history: &mut History<Task>) {
        let mut container_counts = [0; Input::N + Self::STORAGE_COUNT];
        let mut container_suffix_sum = [0; Input::N + Self::STORAGE_COUNT];

        for i in 0..Input::N {
            container_counts[i] = Input::N - self.in_ptr[i] as usize;
        }

        for i in Input::N..Input::N + Self::STORAGE_COUNT {
            container_counts[i] = self.containers[i].is_some() as usize;
        }

        container_suffix_sum[Input::N + Self::STORAGE_COUNT - 1] =
            container_counts[Input::N + Self::STORAGE_COUNT - 1];

        for i in (0..Input::N + Self::STORAGE_COUNT - 1).rev() {
            container_suffix_sum[i] = container_suffix_sum[i + 1] + container_counts[i];
        }

        let turn = self.crane_avail_turns.iter().copied().min().unwrap();
        let mut available_crane_count = 0;

        for (c, t) in self
            .crane_avail_turns
            .iter_mut()
            .zip(self.crane_score_per_turn.iter_mut())
        {
            if *c == turn {
                *c = Self::CRANE_AVAIL;
                *t = 0.0;
                available_crane_count += 1;
            }
        }

        // コンテナ数とクレーン数の小さい方の数だけ操作を割り付ける必要がある
        let remaining = container_suffix_sum[0].min(available_crane_count);

        let mut state = *self;
        state.dfs(
            env,
            beam,
            history,
            &container_suffix_sum,
            turn,
            0,
            remaining,
        );
    }

    fn dfs(
        &mut self,
        env: &Env,
        beam: &mut Vec<Vec<Vec<Self>>>,
        history: &mut History<Task>,
        container_suffix_sum: &[usize; Input::N + Self::STORAGE_COUNT],
        turn: u8,
        search_i: usize,
        remaining: usize,
    ) {
        if remaining == 0 {
            let next_turn = self.crane_avail_turns.iter().copied().min().unwrap();
            let score_per_turn = self.crane_score_per_turn.iter().copied().sum::<f32>();
            assert!(next_turn > turn);

            for t in self.crane_avail_turns.iter_mut() {
                if *t == Self::CRANE_AVAIL {
                    *t = next_turn;
                }
            }

            self.score += score_per_turn * (next_turn - turn) as f32;
            beam[next_turn as usize][self.temp_count as usize].push(*self);
            return;
        }

        for container_i in search_i..Input::N + Self::STORAGE_COUNT {
            if container_suffix_sum[container_i] < remaining {
                break;
            }

            let Some(container) = self.containers[container_i] else {
                continue;
            };

            let from = Self::POS[container_i];
            let goal = Input::get_goal(container);

            let to_out = self.out_next[goal.row()] == container.index() as u8;
            let to_temp = container_i < Input::N && self.temp_count < Self::STORAGE_COUNT as u8;

            if !to_out && !to_temp {
                continue;
            }

            // craneのアサイン
            let mut best_crane = !0;
            let mut best_dist = usize::MAX;

            for (crane, (&t, &c)) in self
                .crane_avail_turns
                .iter()
                .zip(self.cranes.iter())
                .enumerate()
            {
                if t == Self::CRANE_AVAIL && best_dist.change_min(c.dist(&from)) {
                    best_crane = crane;
                }
            }

            let crane_pos = self.cranes[best_crane];

            // in側の更新
            let mut state = *self;
            let storage_flag = env
                .dist_dict
                .get_flag(|c| state.containers[env.index_grid[c]].is_some());

            // pickを開始するターン
            let move_len = env.dist_dict.dist(storage_flag, crane_pos, from, false) as u8;
            let pick_turn = (turn + move_len).max(state.container_avail_turns[container_i]);
            state.container_avail_turns[container_i] = turn + 2;

            // pick_work: Pickの有効作業量。一時保管場所からのPickは無駄な作業なので0
            let pick_work = if container_i < Input::N {
                let in_ptr = &mut state.in_ptr[container_i];
                *in_ptr += 1;
                state.containers[container_i] = env.input.containers()[container_i]
                    .get(*in_ptr as usize)
                    .copied();
                1
            } else {
                state.containers[container_i] = None;
                state.temp_count -= 1;
                0
            };

            // クレーンの戻り距離に応じた仕事量
            // 左方向の移動は作業に貢献しており、それ以外の移動は貢献していないと見なす
            let crane_back_work = crane_pos.col().saturating_sub(from.col()) as i32;

            // out側の更新
            if to_out {
                // そのまま搬出
                state.out_next[goal.row()] += 1;
                state.finished_container_count += 1;

                let dist = from.dist(&goal);
                let prev_potential = dist as i32;

                // クレーンの左移動 + Pick + ゴールまでの移動 + Dropが仕事量
                let total_work = crane_back_work + pick_work + prev_potential + 1;

                // dropが開始するターン
                let move_len =
                    env.dist_dict
                        .dist(storage_flag, from, goal, !Input::is_large_crane(best_crane))
                        as u8;
                let drop_turn = (pick_turn + 1 + move_len).max(state.out_avail_turns[goal.row()]);
                state.out_avail_turns[goal.row()] = drop_turn + 2;

                // クレーンが使用可能になるターン
                let crane_avail_turn = drop_turn + 1;
                let total_turn = crane_avail_turn - turn;
                let score_per_turn = total_work as f32 / total_turn as f32;
                state.crane_score_per_turn[best_crane] = score_per_turn;
                state.crane_avail_turns[best_crane] = crane_avail_turn;

                // historyの追加
                let task = Task::new(best_crane as u8, container, from, goal);
                state.history = history.push(task, state.history);
                state.dfs(
                    env,
                    beam,
                    history,
                    container_suffix_sum,
                    turn,
                    container_i,
                    remaining - 1,
                );
            } else {
                // 一時保管場所に運ぶ
                let mut candidates = (Input::N..Self::STORAGE_COUNT + Input::N)
                    .filter(|&i| state.containers[i].is_none())
                    .collect_vec();
                candidates.sort_unstable_by_key(|&i| from.dist(&Self::POS[i]));
                const TAKE_COUNT: usize = 3;
                state.temp_count += 1;

                for &temp_i in candidates.iter().take(TAKE_COUNT) {
                    let mut state = state;
                    state.containers[temp_i] = Some(container);
                    let to = Self::POS[temp_i];
                    let prev_potential = from.dist(&goal) as i32;
                    let new_potential = to.dist(&goal) as i32;
                    let potential_diff = prev_potential - new_potential;

                    // クレーンの左移動 + Pick + ゴールまでの移動 + Dropが仕事量
                    // potential_diffが負になること、一時保管場所にDropする操作は進捗を生まないことに注意
                    let total_work = crane_back_work + pick_work + potential_diff;
                    let move_len = env.dist_dict.dist(
                        storage_flag,
                        from,
                        to,
                        !Input::is_large_crane(best_crane),
                    ) as u8;
                    let drop_turn =
                        (pick_turn + 1 + move_len).max(state.container_avail_turns[temp_i]);
                    state.container_avail_turns[temp_i] = drop_turn + 2;

                    let crane_avail_turn = drop_turn + 1;
                    let total_turn = crane_avail_turn - turn;
                    let score_per_turn = total_work as f32 / total_turn as f32;
                    state.crane_score_per_turn[best_crane] = score_per_turn;
                    state.crane_avail_turns[best_crane] = crane_avail_turn;

                    // historyの追加
                    let task = Task::new(best_crane as u8, container, from, to);
                    state.history = history.push(task, state.history);
                    state.dfs(
                        env,
                        beam,
                        history,
                        container_suffix_sum,
                        turn,
                        container_i,
                        remaining - 1,
                    );
                }
            }
        }
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 浮動小数点数の比較は誤差を考慮する
        let a = (self.score * 100.0).round() as i32;
        let b = (other.score * 100.0).round() as i32;
        a.cmp(&b).reverse()
    }
}
