mod dag;

use self::dag::TaskSet;
use super::{step02_order::SubTask, Precalc, StorageFlag};
use crate::{
    data_structures::{History, HistoryIndex},
    grid::Coord,
    problem::{Container, CraneState, Grid, Input, Operation},
};
use std::array;

pub(super) fn execute(
    input: &Input,
    precalc: &Precalc,
    tasks: &[Vec<SubTask>; Input::N],
    max_turn: usize,
) -> Result<Vec<[Operation; Input::N]>, &'static str> {
    const BEAM_SIZE: usize = 1 << (Input::N * 2);
    let tasks = dag::critical_path_analysis(tasks, precalc);
    let env = Env::new(input, precalc, tasks);
    let mut history = History::new();
    let mut beam = vec![None; BEAM_SIZE];
    beam[0] = Some(State::init(&env));
    let mut turn = 0;

    while !beam
        .iter()
        .any(|s| s.map_or(false, |s| s.is_completed(&env)))
    {
        turn += 1;

        if turn > max_turn {
            return Err("turn limit exceeded");
        }

        let mut next_beam = vec![None; BEAM_SIZE];
        for state in beam {
            let Some(state) = state else {
                continue;
            };

            state.gen_next(&env, &mut next_beam, &mut history);
        }

        beam = next_beam;
    }

    let best = beam
        .iter()
        .filter(|s| s.is_some_and(|s| s.is_completed(&env)))
        .next()
        .unwrap()
        .unwrap();

    Ok(history.collect(best.history))
}

struct Env<'a> {
    _input: &'a Input,
    precalc: &'a Precalc,
    tasks: TaskSet,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input, precalc: &'a Precalc, tasks: TaskSet) -> Self {
        Self {
            _input: input,
            precalc,
            tasks,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct State {
    task_ptr: [u8; Input::N],
    grid_ptr: Grid<u8>,
    cranes: [CraneState; Input::N],
    board: Grid<i8>,
    history: HistoryIndex,
    score: f64,
}

impl State {
    fn init(env: &Env) -> Self {
        let task_ptr = env.tasks.init_ptr;
        let cranes = array::from_fn(|i| CraneState::Empty(Coord::new(i, 0)));
        let history = HistoryIndex::ROOT;
        let mut board = Grid::with_default();
        board[Coord::new(0, 0)] = 5;
        board[Coord::new(1, 0)] = 5;
        board[Coord::new(2, 0)] = 5;
        board[Coord::new(3, 0)] = 5;
        board[Coord::new(4, 0)] = 5;
        board[Coord::new(0, 4)] = -5;
        board[Coord::new(1, 4)] = -5;
        board[Coord::new(2, 4)] = -5;
        board[Coord::new(3, 4)] = -5;
        board[Coord::new(4, 4)] = -5;

        let storage_flag = env.precalc.dist_dict.get_flag(|c| board[c] > 0);
        let grid_ptr = Grid::with_default();

        let mut scores = [0.0; Input::N];

        for (i, &task_ptr) in task_ptr.iter().enumerate() {
            let crane = cranes[i];
            let task = &env.tasks.tasks[task_ptr as usize];
            let edge_cost = if let (Some(c0), Some(c1)) = (cranes[i].coord(), task.coord()) {
                let consider_container = !Input::is_large_crane(i) && crane.is_holding();
                env.precalc
                    .dist_dict
                    .dist(storage_flag, c0, c1, consider_container)
                    + 1
            } else {
                0
            };

            scores[i] = env.tasks.dp[task_ptr as usize] * env.precalc.exp_table[edge_cost];
        }

        let score = scores.iter().sum::<f64>();

        Self::new(task_ptr, grid_ptr, cranes, board, score, history)
    }

    fn new(
        task_ptr: [u8; Input::N],
        grid_ptr: Grid<u8>,
        cranes: [CraneState; Input::N],
        board: Grid<i8>,
        score: f64,
        history: HistoryIndex,
    ) -> Self {
        Self {
            task_ptr,
            grid_ptr,
            cranes,
            board,
            history,
            score,
        }
    }

    fn gen_next(
        &self,
        env: &Env,
        beam: &mut Vec<Option<State>>,
        history: &mut History<[Operation; Input::N]>,
    ) {
        // 操作の候補を列挙
        let mut candidates = [vec![], vec![], vec![], vec![], vec![]];
        const MOVES: [Operation; 5] = [
            Operation::None,
            Operation::Up,
            Operation::Right,
            Operation::Down,
            Operation::Left,
        ];

        let storage_flag = env.precalc.dist_dict.get_flag(|c| self.board[c] > 0);

        for crane_i in 0..Input::N {
            let crane = self.cranes[crane_i];
            let candidates = &mut candidates[crane_i];
            let task = env.tasks.tasks[self.task_ptr[crane_i] as usize];

            if crane == CraneState::Destroyed {
                candidates.push(Operation::None);
            } else if task == SubTask::EndOfOrder {
                candidates.push(Operation::Destroy);
            } else {
                let coord = crane.coord().unwrap();
                let destination = task.coord().unwrap();

                if coord == destination && self.grid_ptr[coord] == task.index().unwrap() as u8 {
                    // pick / drop
                    let op = match task {
                        SubTask::Pick(_, _) => Operation::Pick,
                        SubTask::Drop(_, _) => Operation::Drop,
                        SubTask::EndOfOrder => unreachable!(),
                    };
                    candidates.push(op);
                } else {
                    let nearest_crane_dist = self
                        .cranes
                        .iter()
                        .filter_map(|c| c.coord())
                        .map(|c| c.dist(&coord))
                        .filter(|&d| d > 0)
                        .min()
                        .unwrap_or(usize::MAX);
                    let consider_container = !Input::is_large_crane(crane_i) && crane.is_holding();
                    let current_dist = env.precalc.dist_dict.dist(
                        storage_flag,
                        coord,
                        destination,
                        consider_container,
                    );

                    if crane.is_holding() && !Input::is_large_crane(crane_i) {
                        // コンテナを考慮して移動
                        for &op in MOVES.iter() {
                            let next = coord + op.dir();
                            if next.in_map(Input::N) && self.board[next] <= 0 {
                                let next_dist = env.precalc.dist_dict.dist(
                                    storage_flag,
                                    next,
                                    destination,
                                    consider_container,
                                );

                                // 他のクレーンが遠いときは損しない移動方法だけを考慮
                                if nearest_crane_dist <= 1 || next_dist <= current_dist {
                                    candidates.push(op);
                                }
                            }
                        }
                    } else {
                        // コンテナを無視して移動
                        for &op in MOVES.iter() {
                            let next = coord + op.dir();
                            if next.in_map(Input::N) {
                                let next_dist = env.precalc.dist_dict.dist(
                                    storage_flag,
                                    next,
                                    destination,
                                    consider_container,
                                );

                                // 他のクレーンが遠いときは損しない移動方法だけを考慮
                                if nearest_crane_dist <= 1 || next_dist <= current_dist {
                                    candidates.push(op);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut cant_in = Grid::new([false; Input::N * Input::N]);
        let mut cant_move = Grid::new([[false; 8]; Input::N * Input::N]);
        let mut state = self.clone();

        state.dfs(
            env,
            &candidates,
            &mut cant_in,
            &mut cant_move,
            history,
            &mut [Operation::None; Input::N],
            beam,
            storage_flag,
            0.0,
            0,
            0,
        );
    }

    fn dfs(
        &mut self,
        env: &Env,
        candidates: &[Vec<Operation>],
        cant_in: &mut Grid<bool>,
        cant_move: &mut Grid<[bool; 8]>,
        history: &mut History<[Operation; Input::N]>,
        operations: &mut [Operation; Input::N],
        beam: &mut Vec<Option<State>>,
        storage_flag: StorageFlag,
        score: f64,
        hash: usize,
        depth: usize,
    ) {
        if depth == Input::N {
            let old_score = beam[hash].map_or(f64::MAX, |s| s.score);

            if score < old_score {
                let mut new_state = Self::new(
                    self.task_ptr,
                    self.grid_ptr,
                    self.cranes,
                    self.board,
                    score,
                    self.history,
                );

                let hist_index = history.push(operations.clone(), self.history);
                new_state.history = hist_index;
                beam[hash] = Some(new_state);
            }

            return;
        }

        let crane_i = depth;
        let crane = self.cranes[crane_i];
        let task = env.tasks.tasks[self.task_ptr[crane_i] as usize];

        for &op in candidates[depth].iter() {
            if crane == CraneState::Destroyed {
                operations[crane_i] = Operation::None;

                self.dfs(
                    env,
                    candidates,
                    cant_in,
                    cant_move,
                    history,
                    operations,
                    beam,
                    storage_flag,
                    score,
                    hash,
                    depth + 1,
                );

                continue;
            }

            let coord = crane.coord().unwrap();
            let next = coord + op.dir();
            let op_usize = op as usize;

            if (cant_in[next] && op != Operation::Destroy) || cant_move[coord][op_usize] {
                continue;
            }

            let board_change = match op {
                Operation::Pick => -1,
                Operation::Drop => 1,
                _ => 0,
            };
            let current_crane = self.cranes[crane_i];
            let task_ptr_diff = match op {
                Operation::Pick => 1,
                Operation::Drop => 1,
                _ => 0,
            };
            self.board[coord] += board_change;
            self.grid_ptr[coord] = self.grid_ptr[coord].wrapping_add_signed(task_ptr_diff);
            self.task_ptr[crane_i] = self.task_ptr[crane_i].wrapping_add_signed(task_ptr_diff);
            self.cranes[crane_i] = if task == SubTask::EndOfOrder {
                CraneState::Destroyed
            } else {
                match op {
                    Operation::Pick => CraneState::Holding(Container::new(!0), coord),
                    Operation::Drop => CraneState::Empty(coord),
                    _ => match current_crane {
                        CraneState::Empty(_) => CraneState::Empty(next),
                        CraneState::Holding(container, _) => CraneState::Holding(container, next),
                        CraneState::Destroyed => CraneState::Destroyed,
                    },
                }
            };

            operations[crane_i] = op;

            if op != Operation::Destroy {
                cant_in[next] = true;
            }

            // クロスする移動もNG
            if op_usize < 4 {
                cant_move[next][op_usize ^ 2] = true;
            }

            // ハッシュ値を更新
            let new_hash = hash * 4 + (next.row() % 2 * 2 + next.col() % 2);

            // スコア
            let new_crane = self.cranes[crane_i].coord();
            let new_task_coord = env.tasks.tasks[self.task_ptr[crane_i] as usize].coord();
            let edge_cost = if let (Some(c0), Some(c1)) = (new_crane, new_task_coord) {
                let consider_container =
                    !Input::is_large_crane(crane_i) && self.cranes[crane_i].is_holding();
                env.precalc
                    .dist_dict
                    .dist(storage_flag, c0, c1, consider_container)
                    + 1
            } else {
                0
            };

            let task_ptr = self.task_ptr[crane_i];
            let score_diff = env.tasks.dp[task_ptr as usize] * env.precalc.exp_table[edge_cost];

            self.dfs(
                env,
                candidates,
                cant_in,
                cant_move,
                history,
                operations,
                beam,
                storage_flag,
                score + score_diff,
                new_hash,
                depth + 1,
            );

            self.board[coord] -= board_change;
            self.grid_ptr[coord] = self.grid_ptr[coord].wrapping_add_signed(-task_ptr_diff);
            self.task_ptr[crane_i] = self.task_ptr[crane_i].wrapping_add_signed(-task_ptr_diff);
            self.cranes[crane_i] = current_crane;

            if op != Operation::Destroy {
                cant_in[next] = false;
            }

            if op_usize < 4 {
                cant_move[next][op_usize ^ 2] = false;
            }
        }
    }

    fn is_completed(&self, env: &Env) -> bool {
        self.task_ptr
            .iter()
            .all(|&ptr| env.tasks.tasks[ptr as usize] == SubTask::EndOfOrder)
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
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).unwrap()
    }
}
