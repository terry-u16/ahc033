use std::fmt::Display;

use proconio::input;

use crate::{
    data_structures::{ConstQueue, ConstVec},
    grid::{ConstMap2d, Coord, CoordDiff},
};

pub type Grid<T> = ConstMap2d<T, { Input::N }, { Input::N * Input::N }>;

const SEQUENCE: [usize; Input::CONTAINER_COUNT] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

#[derive(Debug, Clone)]
pub struct Input {
    containers: [[Container; Input::N]; Input::N],
}

impl Input {
    pub const N: usize = 5;
    pub const CONTAINER_COUNT: usize = Self::N * Self::N;

    fn new(containers: [[Container; Input::N]; Input::N]) -> Self {
        Self { containers }
    }

    pub fn read_input() -> Self {
        input! {
            n: usize,
            c: [[usize; n]; n],
        }

        let mut containers = [[Container::new(0); Self::N]; Self::N];

        for i in 0..Self::N {
            for j in 0..Self::N {
                containers[i][j] = Container::new(c[i][j]);
            }
        }

        Self::new(containers)
    }

    pub const fn containers(&self) -> &[[Container; Input::N]; Input::N] {
        &self.containers
    }

    pub const fn is_large_crane(crane: usize) -> bool {
        crane == 0
    }

    pub const fn get_goal(container: Container) -> Coord {
        Coord::new(container.index() / Input::N, Input::N - 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Container(u8);

impl Container {
    pub const fn new(index: usize) -> Self {
        Self(index as u8)
    }

    pub const fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Up,
    Down,
    Left,
    Right,
    Pick,
    Drop,
    Destroy,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CraneState {
    Empty(Coord),
    Holding(Container, Coord),
    Destroyed,
}

impl CraneState {
    pub const fn coord(&self) -> Option<Coord> {
        match self {
            Self::Empty(coord) => Some(*coord),
            Self::Holding(_, coord) => Some(*coord),
            Self::Destroyed => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Yard {
    grid: Grid<Option<Container>>,
    cranes: [CraneState; Input::N],
    waiting: [ConstQueue<Container, { Input::N }>; Input::N],
    shipped: [ConstVec<Container, { Input::N }>; Input::N],
    shipped_flags: [u8; Input::N],
    shipped_count: u8,
    inversions: u16,
}

impl Yard {
    pub fn new(input: &Input) -> Self {
        let mut grid = Grid::with_default();
        let cranes = [0, 1, 2, 3, 4].map(|i| CraneState::Empty(Coord::new(i, 0)));
        let mut waiting = input.containers.map(|row| ConstQueue::new(row));

        for row in 0..Input::N {
            grid[Coord::new(row, 0)] = Some(waiting[row].pop_front().unwrap());
        }

        let shipped = [ConstVec::new(); Input::N];
        let shipped_flags = [0; Input::N];
        let shipped_count = 0;
        let inversions = 0;

        Self {
            grid,
            cranes,
            waiting,
            shipped,
            shipped_count,
            shipped_flags,
            inversions,
        }
    }

    pub const fn grid(&self) -> &Grid<Option<Container>> {
        &self.grid
    }

    pub const fn cranes(&self) -> &[CraneState; Input::N] {
        &self.cranes
    }

    pub const fn waiting(&self) -> &[ConstQueue<Container, { Input::N }>; Input::N] {
        &self.waiting
    }

    pub const fn shipped(&self) -> &[ConstVec<Container, { Input::N }>; Input::N] {
        &self.shipped
    }

    pub const fn inversions(&self) -> u32 {
        self.inversions as u32
    }

    pub const fn shipped_count(&self) -> usize {
        self.shipped_count as usize
    }

    pub fn apply(&mut self, operations: &[Operation; Input::N]) -> Result<(), &'static str> {
        let prev_coords = self.cranes.map(|c| c.coord());

        for crane in 0..Input::N {
            match operations[crane] {
                Operation::Up => self.move_crane(crane, -1, 0),
                Operation::Down => self.move_crane(crane, 1, 0),
                Operation::Left => self.move_crane(crane, 0, -1),
                Operation::Right => self.move_crane(crane, 0, 1),
                Operation::Pick => self.pick(crane),
                Operation::Drop => self.drop(crane),
                Operation::Destroy => self.destroy(crane),
                Operation::None => Ok(()),
            }?;
        }

        let current_coords = self.cranes.map(|c| c.coord());

        // クレーン交叉判定
        for (i, (p0, c0)) in prev_coords.iter().zip(current_coords.iter()).enumerate() {
            let (Some(p0), Some(c0)) = (p0, c0) else {
                continue;
            };

            let p1 = &prev_coords[i + 1..];
            let c1 = &current_coords[i + 1..];

            for (p1, c1) in p1.iter().zip(c1.iter()) {
                let (Some(p1), Some(c1)) = (p1, c1) else {
                    continue;
                };

                if c0 == c1 || (c0 == p1 && c1 == p0) {
                    return Err("Two cranes are in the same cell");
                }
            }
        }

        Ok(())
    }

    fn move_crane(&mut self, crane: usize, dr: isize, dc: isize) -> Result<(), &'static str> {
        let new_state = match self.cranes[crane] {
            CraneState::Empty(coord) => CraneState::Empty(coord + CoordDiff::new(dr, dc)),
            CraneState::Holding(container, coord) => {
                let new_coord = coord + CoordDiff::new(dr, dc);

                if !Input::is_large_crane(crane) && self.grid[new_coord].is_some() {
                    return Err("Cannot move to a cell with a container");
                }

                CraneState::Holding(container, coord + CoordDiff::new(dr, dc))
            }
            CraneState::Destroyed => return Err("Cannot move a destroyed crane"),
        };

        self.cranes[crane] = new_state;
        Ok(())
    }

    fn pick(&mut self, crane: usize) -> Result<(), &'static str> {
        let CraneState::Empty(coord) = self.cranes[crane] else {
            return Err("Tried to pick up a container while holding one");
        };

        let container = self.grid[coord]
            .take()
            .ok_or("Tried to pick up a container but the cell is empty")?;
        self.cranes[crane] = CraneState::Holding(container, coord);

        Ok(())
    }

    fn drop(&mut self, crane: usize) -> Result<(), &'static str> {
        let CraneState::Holding(container, coord) = self.cranes[crane] else {
            return Err("Tried to drop a container while not holding one");
        };

        self.cranes[crane] = CraneState::Empty(coord);
        let place = self.grid[coord].as_mut();

        if place.is_some() {
            return Err("Tried to drop a container but the cell is not empty");
        }

        self.grid[coord] = Some(container);
        Ok(())
    }

    fn destroy(&mut self, crane: usize) -> Result<(), &'static str> {
        match self.cranes[crane] {
            CraneState::Empty(_) => {
                self.cranes[crane] = CraneState::Destroyed;
                Ok(())
            }
            CraneState::Holding(_, _) => Err("Tried to destroy a crane holding a container"),
            CraneState::Destroyed => Err("Tried to destroy a destroyed crane"),
        }
    }

    pub fn carry_in_and_ship(&mut self) {
        // 搬入
        for row in 0..Input::N {
            let cell = &mut self.grid[Coord::new(row, 0)];

            if cell.is_none() {
                if let Some(container) = self.waiting[row].pop_front() {
                    *cell = Some(container);
                }
            }
        }

        // 搬出
        for row in 0..Input::N {
            let cell = &mut self.grid[Coord::new(row, Input::N - 1)];

            if let Some(container) = cell.take() {
                let min = Input::N * row;
                let max = min + Input::N;
                assert!(min <= container.index() && container.index() < max);

                for &c in self.shipped[row].as_slice().iter() {
                    if container < c {
                        self.inversions += 1;
                    }
                }

                self.shipped_flags[row] |= 1 << (container.index() - min);
                self.shipped[row].push(container);
                self.shipped_count += 1;
            }
        }
    }

    pub const fn is_end(&self) -> bool {
        self.shipped_count == Input::CONTAINER_COUNT as u8
    }

    pub fn is_next_ship(&self, container: Container) -> bool {
        let row = container.index() / Input::N;
        self.next_ship_of(row) == Some(container)
    }

    pub fn next_ship_of(&self, row: usize) -> Option<Container> {
        let flag = self.shipped_flags[row];
        let min = Input::N * row;

        if flag < (1 << 5) {
            let i = (!flag).trailing_zeros() as usize;
            Some(Container::new(min + i))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    operations: [Vec<Operation>; Input::N],
}

impl Output {
    pub fn new() -> Self {
        Self {
            operations: [vec![], vec![], vec![], vec![], vec![]],
        }
    }

    pub fn push(&mut self, operations: &[Operation; Input::N]) {
        for (ops, &operation) in self.operations.iter_mut().zip(operations.iter()) {
            ops.push(operation);
        }
    }

    pub fn len(&self) -> usize {
        self.operations.iter().map(|ops| ops.len()).max().unwrap()
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for operations in self.operations.iter() {
            for operation in operations.iter() {
                match operation {
                    Operation::Up => write!(f, "U")?,
                    Operation::Down => write!(f, "D")?,
                    Operation::Left => write!(f, "L")?,
                    Operation::Right => write!(f, "R")?,
                    Operation::Pick => write!(f, "P")?,
                    Operation::Drop => write!(f, "Q")?,
                    Operation::Destroy => write!(f, "B")?,
                    Operation::None => write!(f, ".")?,
                }
            }

            writeln!(f)?;
        }

        Ok(())
    }
}
