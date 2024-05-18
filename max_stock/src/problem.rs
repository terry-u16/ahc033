use std::{error::Error, path::Path};

use proconio::{input, source::once::OnceSource};

#[derive(Debug, Clone)]
pub struct Input {
    containers: [[usize; Input::N]; Input::N],
}

impl Input {
    pub const N: usize = 5;

    fn new(containers: [[usize; Input::N]; Input::N]) -> Self {
        Self { containers }
    }

    pub fn read_input(path: &Path) -> Result<Self, Box<dyn Error>> {
        let source = std::fs::read(path)?;
        let source = OnceSource::new(source.as_slice());
        input! {
            from source,
            n: usize,
            c: [[usize; n]; n],
        }

        let mut containers = [[0; Self::N]; Self::N];

        for i in 0..Self::N {
            for j in 0..Self::N {
                containers[i][j] = c[i][j];
            }
        }

        Ok(Self::new(containers))
    }

    pub const fn contaniers(&self) -> &[[usize; Input::N]; Input::N] {
        &self.containers
    }
}
