use std::fmt::Display;

use itertools::iproduct;

use crate::{common::ChangeMinMax, problem::Input};

const N: usize = Input::N;
const NP1: usize = N + 1;
const N2: usize = N * N;

pub fn dp(input: &Input) -> (u32, Vec<Hist>) {
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

    let mut dp = vec![vec![vec![vec![vec![u32::MAX / 2; NP1]; NP1]; NP1]; NP1]; NP1];
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
            let new_cnt = current_dp.max(current_cnt).max(cnt);
            if dp[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]].change_min(new_cnt) {
                from[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]] =
                    (old_indices, idx);
            }
        }
    }

    let mut current = [N; N];
    let mut history = vec![];

    while current != [0; N] {
        let (prev, index) = from[current[0]][current[1]][current[2]][current[3]][current[4]];
        let container = input.contaniers()[index][current[index] - 1];
        let buf_size = counts[current[0]][current[1]][current[2]][current[3]][current[4]]
            .max(counts[prev[0]][prev[1]][prev[2]][prev[3]][prev[4]])
            as usize;
        history.push(Hist::new(index, container, buf_size));
        current = prev;
    }

    history.reverse();

    (dp[N][N][N][N][N], history)
}

fn count(input: &Input, indices: [usize; N]) -> u32 {
    let mut contains = [false; N2];

    for (i, &c) in indices.iter().enumerate() {
        for &i in input.contaniers()[i][..c].iter() {
            contains[i] = true;
        }
    }

    let mut count = indices.iter().map(|&i| i as u32).sum::<u32>();

    for i in 0..N {
        let slice = &contains[i * N..(i + 1) * N];
        count -= slice.iter().take_while(|&&b| b).count() as u32;
    }

    count
}

#[derive(Debug, Clone, Copy)]
pub struct Hist {
    index: usize,
    container: usize,
    buf_size: usize,
}

impl Hist {
    pub fn new(index: usize, container: usize, buf_size: usize) -> Self {
        Self {
            index,
            container,
            buf_size,
        }
    }
}

impl Display for Hist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "row {:>2}, container: {:>2}, buf_size: {:>2}",
            self.index, self.container, self.buf_size
        )
    }
}
