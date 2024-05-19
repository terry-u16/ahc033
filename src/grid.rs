use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Coord {
    row: u8,
    col: u8,
}

#[allow(dead_code)]
impl Coord {
    pub const fn new(row: usize, col: usize) -> Self {
        Self {
            row: row as u8,
            col: col as u8,
        }
    }

    pub const fn row(&self) -> usize {
        self.row as usize
    }

    pub const fn col(&self) -> usize {
        self.col as usize
    }

    pub fn in_map(&self, size: usize) -> bool {
        self.row < size as u8 && self.col < size as u8
    }

    pub const fn to_index(&self, size: usize) -> usize {
        self.row as usize * size + self.col as usize
    }

    pub const fn dist(&self, other: &Self) -> usize {
        Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
    }

    const fn dist_1d(x0: u8, x1: u8) -> usize {
        x0.abs_diff(x1) as usize
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.row(), self.col())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoordDiff {
    dr: i8,
    dc: i8,
}

#[allow(dead_code)]
impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self {
            dr: dr as i8,
            dc: dc as i8,
        }
    }

    pub const fn invert(&self) -> Self {
        Self {
            dr: -self.dr,
            dc: -self.dc,
        }
    }

    pub const fn dr(&self) -> isize {
        self.dr as isize
    }

    pub const fn dc(&self) -> isize {
        self.dc as isize
    }
}

impl Add<CoordDiff> for Coord {
    type Output = Coord;

    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord {
            row: self.row.wrapping_add_signed(rhs.dr),
            col: self.col.wrapping_add_signed(rhs.dc),
        }
    }
}

impl AddAssign<CoordDiff> for Coord {
    fn add_assign(&mut self, rhs: CoordDiff) {
        self.row = self.row.wrapping_add_signed(rhs.dr);
        self.col = self.col.wrapping_add_signed(rhs.dc);
    }
}

#[allow(dead_code)]
pub const ADJACENTS: [CoordDiff; 4] = [
    CoordDiff::new(-1, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(0, -1),
];

#[allow(dead_code)]
pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

#[derive(Debug, Clone)]
pub struct Map2d<T> {
    size: usize,
    map: Vec<T>,
}

#[allow(dead_code)]
impl<T> Map2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        debug_assert!(size * size == map.len());
        Self { size, map }
    }
}

#[allow(dead_code)]
impl<T: Default + Clone> Map2d<T> {
    pub fn with_default(size: usize) -> Self {
        let map = vec![T::default(); size * size];
        Self { size, map }
    }
}

impl<T> Index<Coord> for Map2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self.map[coordinate.to_index(self.size)]
    }
}

impl<T> IndexMut<Coord> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(self.size)]
    }
}

impl<T> Index<&Coord> for Map2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: &Coord) -> &Self::Output {
        &self.map[coordinate.to_index(self.size)]
    }
}

impl<T> IndexMut<&Coord> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(self.size)]
    }
}

impl<T> Index<usize> for Map2d<T> {
    type Output = [T];

    #[inline]
    fn index(&self, row: usize) -> &Self::Output {
        let begin = row * self.size;
        let end = begin + self.size;
        &self.map[begin..end]
    }
}

impl<T> IndexMut<usize> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let begin = row * self.size;
        let end = begin + self.size;
        &mut self.map[begin..end]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstMap2d<T, const N: usize, const N2: usize> {
    map: [T; N2],
}

#[allow(dead_code)]
impl<T, const N: usize, const N2: usize> ConstMap2d<T, N, N2> {
    pub fn new(map: [T; N2]) -> Self {
        Self { map }
    }
}

#[allow(dead_code)]
impl<T: Default + Clone + Copy, const N: usize, const N2: usize> ConstMap2d<T, N, N2> {
    pub fn with_default() -> Self {
        let map = [T::default(); N2];
        Self { map }
    }
}

impl<T, const N: usize, const N2: usize> Index<Coord> for ConstMap2d<T, N, N2> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self.map[coordinate.to_index(N)]
    }
}

impl<T, const N: usize, const N2: usize> IndexMut<Coord> for ConstMap2d<T, N, N2> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(N)]
    }
}

impl<T, const N: usize, const N2: usize> Index<&Coord> for ConstMap2d<T, N, N2> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: &Coord) -> &Self::Output {
        &self.map[coordinate.to_index(N)]
    }
}

impl<T, const N: usize, const N2: usize> IndexMut<&Coord> for ConstMap2d<T, N, N2> {
    #[inline]
    fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
        &mut self.map[coordinate.to_index(N)]
    }
}

impl<T, const N: usize, const N2: usize> Index<usize> for ConstMap2d<T, N, N2> {
    type Output = [T];

    #[inline]
    fn index(&self, row: usize) -> &Self::Output {
        let begin = row * N;
        let end = begin + N;
        &self.map[begin..end]
    }
}

impl<T, const N: usize, const N2: usize> IndexMut<usize> for ConstMap2d<T, N, N2> {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let begin = row * N;
        let end = begin + N;
        &mut self.map[begin..end]
    }
}
