use super::{Env, State, TaskType};
use crate::{
    common::ChangeMinMax as _,
    grid::Coord,
    problem::{Container, Input},
};
use itertools::Itertools;
use rand::prelude::*;

const STORAGES: [Coord; 6] = [
    Coord::new(0, 2),
    Coord::new(2, 2),
    Coord::new(4, 2),
    Coord::new(0, 3),
    Coord::new(2, 3),
    Coord::new(4, 3),
];

pub(super) fn annealing(env: &Env, initial_solution: State, duration: f64) -> State {
    let mut state = initial_solution;
    let mut best_state = state.clone();
    let mut current_score = state.calc_score(env, usize::MAX).unwrap();
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e0;
    let temp1 = 1e-1;
    let mut temp = temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let neigh_type = rng.gen_range(0..7);
        let neigh = match neigh_type {
            0 => SwapAfter::gen(&state, &mut rng),
            1 => Move::gen(&state, &mut rng),
            2 => SwapSingle::gen(&state, &mut rng),
            3 => SwapInCrane::gen(&state, &mut rng),
            4 => ChangeStoragePos::gen(&state, &mut rng),
            5 => SwapStoragePos::gen(&state, &mut rng),
            6 => SwapStorageAll::gen(&state, &mut rng),
            _ => unreachable!(),
        };
        let Some(neigh) = neigh else {
            continue;
        };
        let new_state = neigh.neigh(env, state.clone());

        // スコア計算
        let threshold = current_score - temp * rng.gen_range(0.0f64..1.0).ln();
        let Ok(new_score) = new_state.calc_score(env, threshold as usize) else {
            continue;
        };

        if new_score <= threshold {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;
            state = new_state;

            if best_score.change_min(current_score) {
                best_state = state.clone();
                update_count += 1;
            }
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

    best_state
}

trait Neigh {
    fn neigh(&self, env: &Env, state: State) -> State;
}

struct SwapAfter {
    crane0: usize,
    crane1: usize,
    index0: usize,
    index1: usize,
}

impl SwapAfter {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let crane0 = rng.gen_range(0..Input::N);
        let crane1 = (crane0 + rng.gen_range(1..Input::N)) % Input::N;
        let index0 = rng.gen_range(0..=state.tasks[crane0].len());
        let index1 = rng.gen_range(0..=state.tasks[crane1].len());

        Some(Box::new(Self {
            crane0,
            crane1,
            index0,
            index1,
        }))
    }
}

impl Neigh for SwapAfter {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        let temp0 = state.tasks[self.crane0].drain(self.index0..).collect_vec();
        let temp1 = state.tasks[self.crane1].drain(self.index1..).collect_vec();
        state.tasks[self.crane0].extend(temp1);
        state.tasks[self.crane1].extend(temp0);

        state
    }
}

struct Move {
    crane0: usize,
    index0: usize,
    crane1: usize,
    index1: usize,
}

impl Move {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let candidates = (0..Input::N)
            .filter(|&i| state.tasks[i].len() >= 1)
            .collect_vec();

        let crane0 = *candidates.choose(rng)?;
        let crane1 = (crane0 + rng.gen_range(1..Input::N)) % Input::N;
        let index0 = rng.gen_range(0..state.tasks[crane0].len());
        let index1 = rng.gen_range(0..=state.tasks[crane1].len());

        Some(Box::new(Self {
            crane0,
            index0,
            crane1,
            index1,
        }))
    }
}

impl Neigh for Move {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        let task = state.tasks[self.crane0].remove(self.index0);
        state.tasks[self.crane1].insert(self.index1, task);

        state
    }
}

struct SwapSingle {
    crane0: usize,
    crane1: usize,
    index0: usize,
    index1: usize,
}

impl SwapSingle {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let candidates = (0..Input::N)
            .filter(|&i| state.tasks[i].len() >= 1)
            .collect_vec();

        if candidates.len() < 2 {
            return None;
        }

        let mut cranes = candidates.choose_multiple(rng, 2);
        let crane0 = *cranes.next().unwrap();
        let crane1 = *cranes.next().unwrap();

        let index0 = rng.gen_range(0..state.tasks[crane0].len());
        let index1 = rng.gen_range(0..state.tasks[crane1].len());
        Some(Box::new(Self {
            crane0,
            crane1,
            index0,
            index1,
        }))
    }
}

impl Neigh for SwapSingle {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        let temp = state.tasks[self.crane0][self.index0];
        state.tasks[self.crane0][self.index0] = state.tasks[self.crane1][self.index1];
        state.tasks[self.crane1][self.index1] = temp;

        state
    }
}

struct SwapInCrane {
    crane: usize,
    index: usize,
}

impl SwapInCrane {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let candidates = (0..Input::N)
            .filter(|&i| state.tasks[i].len() >= 2)
            .collect_vec();
        let crane = *candidates.choose(rng)?;
        let index = rng.gen_range(0..state.tasks[crane].len() - 1);

        Some(Box::new(Self { crane, index }))
    }
}

impl Neigh for SwapInCrane {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        state.tasks[self.crane].swap(self.index, self.index + 1);
        state
    }
}

struct ChangeStoragePos {
    container: Container,
    pos: Coord,
}

impl ChangeStoragePos {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let mut candidates = vec![];

        for tasks in state.tasks.iter() {
            for &task in tasks.iter() {
                if let TaskType::ToTemporary(container, _, to) = task {
                    candidates.push((container, to));
                }
            }
        }

        let &(container, _) = candidates.choose(rng)?;
        let &pos = STORAGES.choose(rng).unwrap();

        Some(Box::new(Self { container, pos }))
    }
}

impl Neigh for ChangeStoragePos {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        for tasks in state.tasks.iter_mut() {
            for task in tasks.iter_mut() {
                match task {
                    TaskType::ToTemporary(container, _, to) => {
                        if *container == self.container {
                            *to = self.pos;
                        }
                    }
                    TaskType::FromTemporary(container, from, _) => {
                        if *container == self.container {
                            *from = self.pos;
                        }
                    }
                    _ => {}
                }
            }
        }

        state
    }
}

struct SwapStoragePos {
    container0: Container,
    container1: Container,
    pos0: Coord,
    pos1: Coord,
}

impl SwapStoragePos {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let mut candidates = vec![];

        for tasks in state.tasks.iter() {
            for &task in tasks.iter() {
                if let TaskType::ToTemporary(container, _, to) = task {
                    candidates.push((container, to));
                }
            }
        }

        if candidates.len() < 2 {
            return None;
        }

        let mut chosen = candidates.choose_multiple(rng, 2);

        let &(container0, pos0) = chosen.next().unwrap();
        let &(container1, pos1) = chosen.next().unwrap();

        Some(Box::new(Self {
            container0,
            container1,
            pos0,
            pos1,
        }))
    }
}

impl Neigh for SwapStoragePos {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        for tasks in state.tasks.iter_mut() {
            for task in tasks.iter_mut() {
                match task {
                    TaskType::ToTemporary(container, _, to) => {
                        if *container == self.container0 {
                            *to = self.pos1;
                        } else if *container == self.container1 {
                            *to = self.pos0;
                        }
                    }
                    TaskType::FromTemporary(container, from, _) => {
                        if *container == self.container0 {
                            *from = self.pos1;
                        } else if *container == self.container1 {
                            *from = self.pos0;
                        }
                    }
                    _ => {}
                }
            }
        }

        state
    }
}

struct SwapStorageAll {
    pos0: Coord,
    pos1: Coord,
}

impl SwapStorageAll {
    fn gen(_state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let mut pos = STORAGES.choose_multiple(rng, 2);
        let pos0 = *pos.next().unwrap();
        let pos1 = *pos.next().unwrap();
        Some(Box::new(Self { pos0, pos1 }))
    }
}

impl Neigh for SwapStorageAll {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        for tasks in state.tasks.iter_mut() {
            for task in tasks.iter_mut() {
                match task {
                    TaskType::ToTemporary(_, _, to) => {
                        if *to == self.pos0 {
                            *to = self.pos1;
                        } else if *to == self.pos1 {
                            *to = self.pos0;
                        }
                    }
                    TaskType::FromTemporary(_, from, _) => {
                        if *from == self.pos0 {
                            *from = self.pos1;
                        } else if *from == self.pos1 {
                            *from = self.pos0;
                        }
                    }
                    _ => {}
                }
            }
        }

        state
    }
}
