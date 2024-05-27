use super::Task;
use crate::{
    common::ChangeMinMax,
    data_structures::{History, HistoryIndex},
    grid::Coord,
    problem::{Container, Grid, Input},
};
use itertools::Itertools;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::collections::HashSet;

const HALF_STORAGE_LEN: usize = 3;
const STORAGE_LEN: usize = 6;
const LEFT_STORAGES: [Coord; HALF_STORAGE_LEN] =
    [Coord::new(0, 2), Coord::new(2, 2), Coord::new(4, 2)];
const RIGHT_STORAGES: [Coord; HALF_STORAGE_LEN] =
    [Coord::new(0, 3), Coord::new(2, 3), Coord::new(4, 3)];

pub(super) fn beam(input: &Input) -> Result<Vec<Task>, &'static str> {
    let env = Env::new(input);
    let mut all_states = vec![State::init(&env)];
    let mut beam = vec![vec![]; STORAGE_LEN + 1];
    let mut transitions = vec![];
    let mut next_beam = vec![vec![]; STORAGE_LEN + 1];
    next_beam[0].push(0);
    let mut history = History::new();
    let mut best_score = u32::MAX;
    let mut hashes = HashSet::new();
    let mut best_state = State::init(&env);
    const BEAM_WIDTH: usize = 10000;

    for _turn in 0..Input::CONTAINER_COUNT * 2 {
        std::mem::swap(&mut beam, &mut next_beam);
        hashes.clear();
        transitions.clear();

        for next_beam in next_beam.iter_mut() {
            next_beam.clear();
        }

        for &state_id in beam.iter().flatten() {
            let state = &all_states[state_id];
            state.transit(&env, &mut transitions, state_id);
        }

        transitions.sort_unstable();

        let mut best_out = 0;

        for &transition in transitions.iter() {
            if transition.score > best_score {
                break;
            }

            if next_beam[transition.storage_count()].len() >= BEAM_WIDTH
                || !hashes.insert(transition.hash)
            {
                continue;
            }

            let state_id = all_states.len();
            next_beam[transition.storage_count()].push(state_id);

            let state = &all_states[transition.state_id()];
            let hist_index = history.push(transition, state.hist_index);
            let state = transition.apply(&env, state.clone(), hist_index);

            if state.is_completed() && best_score.change_min(state.score) {
                best_state = state.clone();
            }

            best_out.change_max(state.out_count);
            all_states.push(state);
        }
    }

    if !best_state.is_completed() {
        return Err("failed to generate tasks");
    }

    let history = history.collect(best_state.hist_index);
    eprintln!("best_score: {}", best_state.score);

    Ok(history.iter().map(|t| t.to_task()).collect_vec())
}

struct Env<'a> {
    input: &'a Input,
    hashes: Vec<Grid<u64>>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let mut hashes = vec![];
        let mut rng = Pcg64Mcg::new(42);

        for _ in 0..Input::CONTAINER_COUNT {
            let mut h = Grid::with_default();

            for row in 0..Input::N {
                for col in 0..Input::N {
                    h[Coord::new(row, col)] = rng.gen();
                }
            }

            hashes.push(h);
        }

        Self { input, hashes }
    }
}

#[derive(Debug, Clone)]
struct State {
    in_ptrs: [u8; Input::N],
    out_next: [u8; Input::N],
    left_storages: [Option<Container>; HALF_STORAGE_LEN],
    right_storages: [Option<Container>; HALF_STORAGE_LEN],
    hist_index: HistoryIndex,
    hash: u64,
    score: u32,
    storage_count: u8,
    out_count: u8,
}

impl State {
    fn init(env: &Env) -> Self {
        let mut hash = 0;

        for (row, container) in env.input.containers().iter().enumerate() {
            hash ^= env.hashes[container[0].index()][Coord::new(row, 0)];
        }

        Self {
            in_ptrs: [0; Input::N],
            out_next: [0, 5, 10, 15, 20],
            left_storages: [None; HALF_STORAGE_LEN],
            right_storages: [None; HALF_STORAGE_LEN],
            hist_index: HistoryIndex::ROOT,
            hash,
            score: 0,
            storage_count: 0,
            out_count: 0,
        }
    }

    fn is_completed(&self) -> bool {
        self.out_count == Input::CONTAINER_COUNT as u8
    }

    fn score_diff(move_cost: usize, storage_count: u8) -> u32 {
        move_cost as u32 * 100000 + storage_count as u32
    }

    fn transit(&self, env: &Env, beam: &mut Vec<Transition>, state_id: usize) {
        let state_id = state_id as u32;

        // 搬入口から
        for (row, &in_ptr) in self.in_ptrs.iter().enumerate() {
            let Some(&container) = env.input.containers()[row].get(in_ptr as usize) else {
                continue;
            };

            let hashmap = env.hashes[container.index()];
            let hash = self.hash ^ hashmap[Coord::new(row, 0)];
            let goal = Input::get_goal(container);

            // 一時保管（左）
            for (index, &to) in LEFT_STORAGES.iter().enumerate() {
                if self.left_storages[index].is_some() {
                    continue;
                }

                let dist_diff = row
                    .abs_diff(to.row())
                    .saturating_sub(row.abs_diff(goal.row()));
                let storage_count = self.storage_count + 1;
                let score_diff = Self::score_diff(dist_diff * 2, storage_count);
                let transition = Transition::new(
                    hash ^ hashmap[to],
                    state_id,
                    Storage::In(row as u8),
                    Storage::Left(index as u8),
                    self.score + score_diff,
                    storage_count,
                    container,
                );
                beam.push(transition);
            }

            // 一時保管（右）
            for (index, &to) in RIGHT_STORAGES.iter().enumerate() {
                if self.right_storages[index].is_some() {
                    continue;
                }

                let dist_diff = row
                    .abs_diff(to.row())
                    .saturating_sub(row.abs_diff(goal.row()));

                // コンテナを跨ぐ必要があるケース
                let detour = if row == to.row() && self.right_storages[index].is_some() {
                    2
                } else {
                    0
                };

                let storage_count = self.storage_count + 1;
                let score_diff = Self::score_diff(dist_diff * 2 + detour, storage_count);

                let transition = Transition::new(
                    hash ^ hashmap[to],
                    state_id,
                    Storage::In(row as u8),
                    Storage::Right(index as u8),
                    self.score + score_diff,
                    storage_count,
                    container,
                );
                beam.push(transition);
            }

            // ゴールへ
            if self.out_next[goal.row()] == container.index() as u8 {
                // コンテナを跨ぐ必要があるケース
                let detour = if row == goal.row()
                    && (row & 1 == 0)
                    && (self.left_storages[row >> 1].is_some()
                        || self.right_storages[row >> 1].is_some())
                {
                    2
                } else {
                    0
                };

                let storage_count = self.storage_count;
                let score_diff = Self::score_diff(detour, storage_count);

                let transition = Transition::new(
                    hash,
                    state_id,
                    Storage::In(row as u8),
                    Storage::Out(goal.row() as u8),
                    self.score + score_diff,
                    storage_count,
                    container,
                );
                beam.push(transition);
            }
        }

        // 一時保管（左）から
        for (index, container) in self.left_storages.iter().enumerate() {
            let &Some(container) = container else {
                continue;
            };

            let hashmap = env.hashes[container.index()];
            let from = LEFT_STORAGES[index];
            let hash = self.hash ^ hashmap[from];
            let goal = Input::get_goal(container);
            let row = from.row();

            // ゴールへ
            if self.out_next[goal.row()] == container.index() as u8 {
                let detour = if row == goal.row() && self.right_storages[index].is_some() {
                    2
                } else {
                    0
                };

                let storage_count = self.storage_count - 1;
                let score_diff = Self::score_diff(detour, storage_count);

                let transition = Transition::new(
                    hash,
                    state_id,
                    Storage::Left(index as u8),
                    Storage::Out(goal.row() as u8),
                    self.score + score_diff,
                    storage_count,
                    container,
                );
                beam.push(transition);
            }
        }

        // 一時保管（右）から
        for (index, container) in self.right_storages.iter().enumerate() {
            let &Some(container) = container else {
                continue;
            };

            let hashmap = env.hashes[container.index()];
            let from = RIGHT_STORAGES[index];
            let hash = self.hash ^ hashmap[from];
            let goal = Input::get_goal(container);

            // ゴールへ
            if self.out_next[goal.row()] == container.index() as u8 {
                let storage_count = self.storage_count - 1;
                let score_diff = Self::score_diff(0, storage_count);
                let transition = Transition::new(
                    hash,
                    state_id,
                    Storage::Right(index as u8),
                    Storage::Out(goal.row() as u8),
                    self.score + score_diff,
                    storage_count,
                    container,
                );
                beam.push(transition);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Transition {
    hash: u64,
    state_id: u32,
    from: Storage,
    to: Storage,
    score: u32,
    storage_count: u8,
    container: Container,
}

impl Transition {
    fn new(
        hash: u64,
        state_id: u32,
        from: Storage,
        to: Storage,
        score: u32,
        storage_count: u8,
        container: Container,
    ) -> Self {
        Self {
            hash,
            state_id,
            from,
            to,
            score,
            storage_count,
            container,
        }
    }

    fn state_id(&self) -> usize {
        self.state_id as usize
    }

    fn storage_count(&self) -> usize {
        self.storage_count as usize
    }

    fn to_task(&self) -> Task {
        Task::new(self.container, self.from.coord(), self.to.coord())
    }

    fn apply(&self, env: &Env, mut state: State, hist_index: HistoryIndex) -> State {
        state.hash = self.hash;
        state.score = self.score;
        state.storage_count = self.storage_count;
        state.hist_index = hist_index;

        let container = match self.from {
            Storage::In(idx) => {
                let ptr = &mut state.in_ptrs[idx as usize];
                let c = env.input.containers()[idx as usize][*ptr as usize];
                *ptr += 1;
                c
            }
            Storage::Left(idx) => {
                let c = state.left_storages[idx as usize].take().unwrap();
                c
            }
            Storage::Right(idx) => {
                let c = state.right_storages[idx as usize].take().unwrap();
                c
            }
            Storage::Out(_) => unreachable!(),
        };

        match self.to {
            Storage::In(_) => {
                unreachable!()
            }
            Storage::Left(idx) => {
                state.left_storages[idx as usize] = Some(container);
            }
            Storage::Right(idx) => {
                state.right_storages[idx as usize] = Some(container);
            }
            Storage::Out(idx) => {
                state.out_next[idx as usize] += 1;
                state.out_count += 1;
            }
        }

        state
    }
}

impl PartialEq for Transition {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Transition {}

impl PartialOrd for Transition {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.score.cmp(&other.score))
    }
}

impl Ord for Transition {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Storage {
    In(u8),
    Left(u8),
    Right(u8),
    Out(u8),
}

impl Storage {
    fn coord(&self) -> Coord {
        match self {
            Storage::In(idx) => Coord::new(*idx as usize, 0),
            Storage::Left(idx) => LEFT_STORAGES[*idx as usize],
            Storage::Right(idx) => RIGHT_STORAGES[*idx as usize],
            Storage::Out(idx) => Coord::new(*idx as usize, Input::N - 1),
        }
    }
}
