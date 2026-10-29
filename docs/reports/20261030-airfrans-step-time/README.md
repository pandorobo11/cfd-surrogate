# AirfRANS の学習時間の見積もり（ダミー入力での計測）

計測日: 2026-10-30。状態: **計測済み**。本物の AirfRANS のデータは使っていない。精度の結果はない。

## 目的

Transolver 公式の AirfRANS 用の設定を Mac mini（Apple M5 Pro、メモリ24GB、PyTorch MPS）で学習すると、どれだけ時間がかかるかを見積もる。縮小した設定で、どれだけ短くなるかも測る。

## 共通の条件

| 項目 | 設定 |
|---|---|
| モデル | 公式 `Airfoil-Design-AirfRANS/models/Transolver.py`（commit `75e0f6764380`）。`get_grid` の `.cuda()` だけを装置指定に置き換えた |
| 入力 | ダミーの乱数。1点あたり7成分（公式と同じ `space_dim=7`）、座標2成分、出力4成分 |
| 固定の設定 | 頭数8、`mlp_ratio=2`、スライス数32、`unified_pos=1`（公式と同じ） |
| 学習1ステップ | バッチサイズ1、順伝播、MSE、逆伝播、Adam の更新。5回の空回しのあと30回の平均 |
| 推論 | 勾配なしで18万点（AirfRANS の1ケースの節点数の目安）。5回の平均 |
| 装置 | `mps`。`PYTORCH_ENABLE_MPS_FALLBACK` は未設定。CPU に逃がした演算はない |
| 環境 | [標準ベンチマークの記録](../20261029-standard-benchmark-repro/README.md#環境)と同じ（torch 2.15.0） |

## 計測した設定と結果

| 設定ID | 隠れ次元 | 層数 | 学習時の点数 | パラメータ数 | 学習1ステップ | 18万点の推論 | MPS が確保したメモリ |
|---|---|---|---|---|---|---|---|
| STEP-OFFICIAL | 256 | 8 | 32,000 | 5,996,868 | 287 ms | 605 ms | 4.20 GiB |
| STEP-R128L4-32K | 128 | 4 | 32,000 | 785,060 | 83 ms | 162 ms | 2.15 GiB |
| STEP-R128L4-16K | 128 | 4 | 16,000 | 785,060 | 43 ms | 161 ms | 2.14 GiB |

STEP-OFFICIAL は公式の `main.py` と `params.yaml` の設定（隠れ次元256、8層、学習時に32,000点へ間引き）と同じ。メモリは推論のあとの値で、同じプロセスで前の設定を解放してから測った。

## 学習時間の見積もり

1学習ステップの時間 × 学習ケース数 × 398エポック（公式の `params.yaml`）で求めた。公式コードは学習用のケースの1割を検証に回すので、学習ケース数は `full` が720、`scarce` が180になる。

| 設定ID | `full` | `scarce` |
|---|---|---|
| STEP-OFFICIAL | 約23時間 | 約5.7時間 |
| STEP-R128L4-32K | 約6.6時間 | 約1.7時間 |
| STEP-R128L4-16K | 約3.4時間 | 約0.9時間 |

この見積もりに含まないもの:

- 検証。公式コードは10エポックごとに、検証ケースを20回ずつ間引いて推論する。`full` では合計1〜2時間ほどの見込み。
- 毎エポックの点の間引きとデータの組み立て（CPU、Python）。まだ測っていない。

そのため、公式の設定で `full` を学習すると、実際には25時間以上かかると見込む。公式の README は A100 1枚で20〜24時間としている。

メモリ: 公式コードは学習データ全体をメモリに読み込む。データ（およそ6GB）と学習時のメモリ（約4GB）を足しても24GBに収まる見込みだが、余裕は大きくない。

## 再現手順

計測に使ったスクリプトは [bench_airfrans_step.py](bench_airfrans_step.py)。引数に公式リポジトリの `Airfoil-Design-AirfRANS` のパスを渡す。

```sh
local/venv-bench/bin/python docs/reports/20261030-airfrans-step-time/bench_airfrans_step.py \
  external/Transolver/Airfoil-Design-AirfRANS
```
