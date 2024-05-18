# フィールドの最大滞留コンテナ数を求めるやつ

フィールドの最大滞留コンテナ数をDPで求める。

## 単一ケース

```sh
cargo run --release -- single -s SEED -d PATH_TO_INPUT_DIR
```

### 実行結果（seed=0）

```text
turn:  0 | row  2, container:  9, buf_size:  6
turn:  1 | row  2, container:  6, buf_size:  7
turn:  2 | row  2, container: 21, buf_size:  8
turn:  3 | row  2, container: 20, buf_size:  9
turn:  4 | row  4, container: 23, buf_size:  8
turn:  5 | row  4, container: 22, buf_size:  9
turn:  6 | row  4, container:  0, buf_size:  7
turn:  7 | row  1, container: 11, buf_size:  7
turn:  8 | row  1, container:  2, buf_size:  8
turn:  9 | row  1, container:  1, buf_size:  9
turn: 10 | row  4, container: 12, buf_size:  8
turn: 11 | row  1, container:  5, buf_size:  9
turn: 12 | row  3, container:  4, buf_size:  5
turn: 13 | row  3, container: 19, buf_size:  6
turn: 14 | row  3, container:  3, buf_size:  7
turn: 15 | row  3, container: 16, buf_size:  6
turn: 16 | row  0, container: 10, buf_size:  7
turn: 17 | row  0, container: 17, buf_size:  5
turn: 18 | row  0, container: 15, buf_size:  6
turn: 19 | row  0, container: 13, buf_size:  2
max_buffer: 9
```

## 複数ケース

```sh
cargo run --release -- multi -s START_SEED -e END_SEED -d PATH_TO_INPUT_DIR
```

### 実行結果（5000ケース）

```text
min_buffer: 6
max_buffer: 10
average_buffer: 6.38
 6: 70.74% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 7: 21.82% ||||||||||||||||||||||
 8:  6.62% |||||||
 9:  0.78% |
10:  0.04%
```
