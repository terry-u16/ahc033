use itertools::iproduct;

use crate::{common::ChangeMinMax as _, problem::Input};

use super::{N, N2, NP1};

/// 滞留コンテナ数の最大値を最小にするDPを行い、搬入順（何番目にどの行から搬入するか）を返す
pub(super) fn dp(input: &Input) -> (u128, Vec<usize>) {
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
