use std::{array, collections::VecDeque};

use crate::{
    common::ChangeMinMax,
    grid::{Coord, CoordDiff, ADJACENTS},
    problem::{Container, CraneState, Grid, Input},
};
use itertools::Itertools;
use rand::prelude::*;

use super::task_gen::Task;

pub fn order_tasks(input: &Input, tasks: &[Task]) -> Result<(), &'static str> {
    let env = Env::new(&input);
    let state1 = State::new(tasks, |_| 0);
    let state2 = State::new(tasks, |i| i % Input::N);

    let state = if state1.calc_score(&env, 1000).unwrap_or(f64::MAX)
        < state2.calc_score(&env, 1000).unwrap_or(f64::MAX)
    {
        state1
    } else {
        state2
    };

    let state = annealing(&env, state, 1.0);

    let since = std::time::Instant::now();
    let result = state.simulate(&env, 200)?;
    eprintln!("elapsed: {:?}", since.elapsed());
    eprintln!("{:?}", result);
    eprintln!("{}", result.calc_score());

    todo!();
}

#[derive(Debug, Clone, Copy)]
enum TaskType {
    Direct(Container, Coord, Coord),
    ToTemporary(Container, Coord, Coord),
    FromTemporary(Container, Coord, Coord),
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    dist_dict: DistDict,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let dist_dict = DistDict::new();
        Self { input, dist_dict }
    }
}

#[derive(Debug, Clone)]
struct DistDict {
    dists: [Vec<Grid<Grid<usize>>>; 2],
    next: [Vec<Grid<Grid<Coord>>>; 2],
}

impl DistDict {
    const STORAGES: [Coord; 11] = [
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
    ];

    fn new() -> Self {
        let mut dists_with_container = vec![];
        let mut next_with_container = vec![];

        for flag in 0..1 << Self::STORAGES.len() {
            let mut board = Grid::new([false; Input::N * Input::N]);

            for (i, &storage) in Self::STORAGES.iter().enumerate() {
                if (flag & (1 << i)) > 0 {
                    board[storage] = true;
                }
            }

            let mut d =
                Grid::new([Grid::new([usize::MAX; Input::N * Input::N]); Input::N * Input::N]);
            let mut next = Grid::new(
                [Grid::new([Coord::new(0, 0); Input::N * Input::N]); Input::N * Input::N],
            );

            for row in 0..Input::N {
                for col in 0..Input::N {
                    let c = Coord::new(row, col);
                    Self::bfs(&board, &mut d[c], &mut next[c], c);
                }
            }

            dists_with_container.push(d);
            next_with_container.push(next);
        }

        // コンテナなし = flagが0
        let dists_without_container =
            vec![dists_with_container[0].clone(); dists_with_container.len()];
        let next_without_container =
            vec![next_with_container[0].clone(); next_with_container.len()];

        Self {
            dists: [dists_without_container, dists_with_container],
            next: [next_without_container, next_with_container],
        }
    }

    fn bfs(board: &Grid<bool>, dists: &mut Grid<usize>, next: &mut Grid<Coord>, from: Coord) {
        dists[from] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(from);

        while let Some(coord) = queue.pop_front() {
            let next_dist = dists[coord] + 1;

            for &adj in ADJACENTS.iter() {
                let next = coord + adj;

                if next.in_map(Input::N) && !board[next] && dists[next].change_min(next_dist) {
                    queue.push_back(next);
                }
            }
        }

        for row in 0..Input::N {
            for col in 0..Input::N {
                let c = Coord::new(row, col);
                let mut best = c;
                let mut best_dist = dists[best];

                for &adj in ADJACENTS.iter() {
                    let next = c + adj;

                    if next.in_map(Input::N) && best_dist.change_min(dists[next]) {
                        best = next;
                    }
                }

                next[c] = best;
            }
        }
    }

    fn get_flag(&self, f: impl Fn(Coord) -> bool) -> StorageFlag {
        let mut flag = 0;

        for (i, &storage) in Self::STORAGES.iter().enumerate() {
            if f(storage) {
                flag |= 1 << i;
            }
        }

        StorageFlag(flag)
    }

    fn dist(&self, flag: StorageFlag, from: Coord, to: Coord, consider_contianer: bool) -> usize {
        // [to][from]の順になることに注意
        self.dists[consider_contianer as usize][flag.0][to][from]
    }

    fn next(&self, flag: StorageFlag, from: Coord, to: Coord, consider_contianer: bool) -> Coord {
        self.next[consider_contianer as usize][flag.0][to][from]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StorageFlag(usize);

#[derive(Debug, Clone)]
struct State {
    tasks: [Vec<TaskType>; Input::N],
}

impl State {
    fn new(tasks: &[Task], assign_fn: impl Fn(usize) -> usize) -> Self {
        let all_tasks = tasks
            .iter()
            .map(|t| {
                if t.from().col() == 0 {
                    if t.to().col() == Input::N - 1 {
                        TaskType::Direct(t.container(), t.from(), t.to())
                    } else {
                        TaskType::ToTemporary(t.container(), t.from(), t.to())
                    }
                } else {
                    TaskType::FromTemporary(t.container(), t.from(), t.to())
                }
            })
            .collect_vec();

        let mut tasks = [vec![], vec![], vec![], vec![], vec![]];

        for (i, &task) in all_tasks.iter().enumerate() {
            tasks[assign_fn(i)].push(task);
        }

        Self { tasks }
    }

    fn calc_score(&self, env: &Env, max_turn: usize) -> Result<f64, &'static str> {
        let turns = self.simulate(env, max_turn)?;
        Ok(turns.calc_score())
    }

    fn simulate(&self, env: &Env, max_turn: usize) -> Result<Turns, &'static str> {
        let mut in_ptr = [0; Input::N];
        let mut out_next: [_; Input::N] = array::from_fn(|i| i * Input::N);
        let mut task_ptr = [0; Input::N];
        let mut cranes: [_; Input::N] = array::from_fn(|i| CraneState::Empty(Coord::new(i, 0)));
        let mut yard = Grid::new([None; Input::N * Input::N]);
        let mut avail_turns = Grid::new([0; Input::N * Input::N]);
        let mut last_turns = [0; Input::N];
        let mut no_progress = 0;

        for row in 0..Input::N {
            yard[Coord::new(row, 0)] = Some(env.input.containers()[row][0]);
        }

        for turn in 1.. {
            if turn > max_turn {
                return Err("turn limit exceeded");
            }

            let prev_state = cranes.clone();
            let flag = env.dist_dict.get_flag(|c| yard[c].is_some());

            for crane_i in 0..Input::N {
                let crane = &mut cranes[crane_i];
                let Some(task) = self.tasks[crane_i].get(task_ptr[crane_i]).copied() else {
                    continue;
                };

                match *crane {
                    CraneState::Empty(coord) => {
                        let (container, from) = match task {
                            TaskType::Direct(container, from, _) => (container, from),
                            TaskType::ToTemporary(container, from, _) => (container, from),
                            TaskType::FromTemporary(container, from, _) => (container, from),
                        };

                        let can_pick = yard[from] == Some(container) && avail_turns[from] <= turn;

                        if coord == from && can_pick {
                            // コンテナをPickする
                            *crane = CraneState::Holding(container, from);
                            avail_turns[from] = turn + 2;

                            if let TaskType::FromTemporary(_, _, _) = task {
                                yard[from] = None;
                            } else {
                                let row = from.row();
                                let in_ptr = &mut in_ptr[row];
                                *in_ptr += 1;
                                yard[from] = env.input.containers()[row].get(*in_ptr).copied();
                            }
                        } else {
                            // コンテナに向けて移動
                            let next = env.dist_dict.next(flag, coord, from, false);
                            *crane = CraneState::Empty(next);
                        }
                    }
                    CraneState::Holding(container_t, coord) => {
                        let (container, to) = match task {
                            TaskType::Direct(container, _, to) => (container, to),
                            TaskType::ToTemporary(container, _, to) => (container, to),
                            TaskType::FromTemporary(container, _, to) => (container, to),
                        };
                        assert_eq!(container, container_t);

                        let can_drop = if let TaskType::ToTemporary(_, _, _) = task {
                            yard[to] == None && avail_turns[to] <= turn
                        } else {
                            out_next[to.row()] == container.index() && avail_turns[to] <= turn
                        };

                        if coord == to && can_drop {
                            // コンテナをDropする
                            *crane = CraneState::Empty(coord);
                            avail_turns[to] = turn + 2;
                            task_ptr[crane_i] += 1;
                            last_turns[crane_i] = turn;

                            if let TaskType::ToTemporary(_, _, _) = task {
                                yard[to] = Some(container);
                            } else {
                                out_next[to.row()] += 1;
                            }
                        } else {
                            // コンテナに向けて移動
                            let consider_container = !Input::is_large_crane(crane_i);
                            let next = env.dist_dict.next(flag, coord, to, consider_container);
                            *crane = CraneState::Holding(container, next);
                        }
                    }
                    CraneState::Destroyed => continue,
                }
            }

            if task_ptr
                .iter()
                .zip(&self.tasks)
                .all(|(&ptr, tasks)| ptr == tasks.len())
            {
                return Ok(Turns::new(last_turns));
            }

            if prev_state == cranes {
                no_progress += 1;

                // available_turn制約により1ターンは状況が変化しない可能性がある
                // 2ターン以上状況が変化しない場合は終了
                if no_progress >= 2 {
                    return Err("no progress");
                }
            } else {
                no_progress = 0;
            }
        }

        unreachable!();
    }
}

#[derive(Debug, Clone, Copy)]
struct Turns {
    turns: [usize; Input::N],
}

impl Turns {
    fn new(turns: [usize; Input::N]) -> Self {
        Self { turns }
    }

    fn calc_score(&self) -> f64 {
        // logsumexp
        const KAPPA: f64 = 3.0;
        let logsumexp = self
            .turns
            .iter()
            .map(|&t| (t as f64 / KAPPA).exp())
            .sum::<f64>()
            .ln()
            * KAPPA;
        logsumexp
    }
}

const STORAGES: [Coord; 6] = [
    Coord::new(0, 2),
    Coord::new(2, 2),
    Coord::new(4, 2),
    Coord::new(0, 3),
    Coord::new(2, 3),
    Coord::new(4, 3),
];

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

struct UseStorage {
    container: Container,
    pos: Coord,
    crane: usize,
    index: usize,
}

impl UseStorage {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let mut candidates = vec![];
        let mut index = 0;

        for tasks in state.tasks.iter() {
            for (i, &task) in tasks.iter().enumerate() {
                if let TaskType::Direct(container, _, _) = task {
                    candidates.push(container);
                    index = i;
                }
            }
        }

        let &container = candidates.choose(rng)?;
        let &pos = STORAGES.choose(rng).unwrap();
        let crane = rng.gen_range(0..Input::N);
        let index = rng.gen_range(
            (index.saturating_sub(1).min(state.tasks[crane].len()))..=state.tasks[crane].len(),
        );

        Some(Box::new(Self {
            container,
            pos,
            crane,
            index,
        }))
    }
}

impl Neigh for UseStorage {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        'main: for tasks in state.tasks.iter_mut() {
            for task in tasks.iter_mut() {
                match *task {
                    TaskType::Direct(container, from, _) => {
                        if container == self.container {
                            *task = TaskType::ToTemporary(container, from, self.pos);
                            break 'main;
                        }
                    }
                    _ => {}
                }
            }
        }

        state.tasks[self.crane].insert(
            self.index,
            TaskType::FromTemporary(self.container, self.pos, Input::get_goal(self.container)),
        );

        state
    }
}

struct DisuseStorage {
    container: Container,
}

impl DisuseStorage {
    fn gen(state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let mut candidates = vec![];

        for tasks in state.tasks.iter() {
            for &task in tasks.iter() {
                if let TaskType::ToTemporary(container, _, _) = task {
                    candidates.push(container);
                }
            }
        }

        let &container = candidates.choose(rng)?;

        Some(Box::new(Self { container }))
    }
}

impl Neigh for DisuseStorage {
    fn neigh(&self, _env: &Env, mut state: State) -> State {
        let mut to_crane = !0;
        let mut to_index = !0;
        let mut to_in = Coord::new(0, 0);
        let mut from_crane = !0;
        let mut from_index = !0;
        let mut from_out = Coord::new(0, 0);

        for (crane, tasks) in state.tasks.iter_mut().enumerate() {
            for (index, task) in tasks.iter_mut().enumerate() {
                match *task {
                    TaskType::ToTemporary(container, from, _) => {
                        if container == self.container {
                            to_crane = crane;
                            to_index = index;
                            to_in = from;
                        }
                    }
                    TaskType::FromTemporary(container, _, to) => {
                        if container == self.container {
                            from_crane = crane;
                            from_index = index;
                            from_out = to;
                        }
                    }
                    _ => {}
                }
            }
        }

        state.tasks[to_crane][to_index] = TaskType::Direct(self.container, to_in, from_out);
        state.tasks[from_crane].remove(from_index);
        state
    }
}

fn annealing(env: &Env, initial_solution: State, duration: f64) -> State {
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
            //7 => UseStorage::gen(&state, &mut rng),
            //8 => DisuseStorage::gen(&state, &mut rng),
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