# フィールドの最大滞留コンテナ数を求めるやつ

フィールドの最大滞留コンテナ数をDPで求める。
「滞留」とは、搬入口でも搬出口でもない場所に一時的に保管していることを指す。

## 単一ケース

```sh
cargo run --release -- single -s SEED -d PATH_TO_INPUT_DIR
```

### 実行結果（seed=0）

```text
turn:  0 | row  2, container:  7, buf_size:  1
turn:  1 | row  2, container:  9, buf_size:  2
turn:  2 | row  1, container: 14, buf_size:  3
turn:  3 | row  1, container: 11, buf_size:  4
turn:  4 | row  1, container:  2, buf_size:  5
turn:  5 | row  1, container:  1, buf_size:  6
turn:  6 | row  1, container:  5, buf_size:  6
turn:  7 | row  2, container:  6, buf_size:  6
turn:  8 | row  4, container: 18, buf_size:  6
turn:  9 | row  3, container:  8, buf_size:  6
turn: 10 | row  2, container: 21, buf_size:  6
turn: 11 | row  2, container: 20, buf_size:  6
turn: 12 | row  4, container: 23, buf_size:  6
turn: 13 | row  4, container: 22, buf_size:  6
turn: 14 | row  4, container:  0, buf_size:  5
turn: 15 | row  4, container: 12, buf_size:  4
turn: 16 | row  3, container:  4, buf_size:  5
turn: 17 | row  3, container: 19, buf_size:  6
turn: 18 | row  3, container:  3, buf_size:  6
turn: 19 | row  3, container: 16, buf_size:  6
turn: 20 | row  0, container: 24, buf_size:  6
turn: 21 | row  0, container: 10, buf_size:  6
turn: 22 | row  0, container: 17, buf_size:  5
turn: 23 | row  0, container: 15, buf_size:  5
turn: 24 | row  0, container: 13, buf_size:  1
max_buffer: 6
```

## 複数ケース

```sh
cargo run --release -- multi -s START_SEED -e END_SEED -d PATH_TO_INPUT_DIR
```

### 実行結果（5000ケース）

```text
min_buffer: 0
max_buffer: 8
average_buffer: 3.58
 0:  0.06% 
 1:  2.32% ||
 2: 13.06% |||||||||||||
 3: 32.40% ||||||||||||||||||||||||||||||||
 4: 32.58% |||||||||||||||||||||||||||||||||
 5: 15.64% ||||||||||||||||
 6:  3.68% ||||
 7:  0.24%
 8:  0.02%
```
