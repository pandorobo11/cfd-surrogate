# 標準ベンチマークでの論文値の再現（計画）

作成日: 2026-10-29。状態: **REPRO-02（Elasticity）は実行済み（2026-10-30）。REPRO-01（Airfoil）は試走だけで、当面は見送る（10月30日の判断）**。

## 目的

Transolver 公式の標準ベンチマーク用コードを Mac mini（Apple M5 Pro、メモリ24GB、PyTorch MPS）で動かし、論文に近い精度が出るかを確かめる。自作の部分（[表面だけの Cp・Cf 予測](../20261029-surface-cp-cf/README.md)など）で問題が起きたときに、環境の問題と切り分けるための基準にする。

2つのデータセットを使う。どちらも小さく、論文の値と直接比べられる。

| データセット | 中身 | 使うモデル | 確かめること |
|---|---|---|---|
| Airfoil（Geo-FNO） | 遷音速 Euler 流れ。翼型の形状から周りのマッハ数の場を予測する。構造格子 221×51（11,271点）。学習1000、テスト200 | `Transolver_Structured_Mesh_2D` | 翼まわりの流れで論文の精度が出るか |
| Elasticity（Geo-FNO） | 弾性体の内部応力。点群972点。学習1000、テスト200 | `Transolver_Irregular_Mesh` | 不規則な点群用のモデル（AirfRANS と同じ種類）が MPS で正しく動くか |

Airfoil は一様流の条件（マッハ数、迎角）が固定で、形状だけが変わる。流れ条件を変えた予測の検証にはならない。

## 取得するもの

| 対象 | 取得元 | 版 | ライセンス |
|---|---|---|---|
| Transolver 公式リポジトリ | `https://github.com/thuml/Transolver`（`PDE-Solving-StandardBenchmark/` を使う） | commit `75e0f6764380` | MIT |
| Airfoil と Elasticity のデータ | Transolver の README が示す Geo-FNO のデータ（Google Drive のフォルダ） | 取得時に記録 | 取得時に確認して記録する |

使うファイル:

- Airfoil: `NACA_Cylinder_X.npy`、`NACA_Cylinder_Y.npy`、`NACA_Cylinder_Q.npy`。目標は `Q` の5番目の成分（マッハ数）。
- Elasticity: `elasticity/Meshes/Random_UnitCell_sigma_10.npy`、`elasticity/Meshes/Random_UnitCell_XY_10.npy`。

置き場所: `data/geofno/`（Git 管理外）。容量は取得時に記録する。

## 公式コードへの変更

- `.cuda()` の直書きを、`mps`（使えなければ `cpu`）を選ぶ装置指定に置き換える。
- それ以外の設定は変えない。

変更は上流のコードなのでこのリポジトリには入れない。変更内容は再現手順に文章で記録し、差分は `local/` に置く。

## 共通の条件

| 項目 | 設定 |
|---|---|
| データ分割・指標 | 公式スクリプトのとおり。指標はテストの相対L2誤差 |
| 学習 | 学習率 1e-3、勾配のノルムの上限 0.1、500エポック（公式スクリプトの既定） |
| 装置 | `mps`。`PYTORCH_ENABLE_MPS_FALLBACK=1` を使ったかを記録する |
| 精度 | fp32 |

## 実行する学習（計画）

| 実行ID | データセット | モデル | 層数 | 隠れ次元 | 頭数 | スライス数 | バッチサイズ | エポック数 | 論文の値（相対L2） |
|---|---|---|---|---|---|---|---|---|---|
| REPRO-01 | Airfoil | `Transolver_Structured_Mesh_2D` | 8 | 128 | 8 | 64 | 4 | 500 | 0.0053 |
| REPRO-02 | Elasticity | `Transolver_Irregular_Mesh` | 8 | 128 | 8 | 64 | 1 | 500 | 0.0064 |

設定は公式の `scripts/Transolver_Airfoil.sh` と `scripts/Transolver_Elas.sh` のとおり。論文の値は arXiv 版（2402.02366）の標準ベンチマークの表から取った。

## 進め方と判断の目安

1. 1エポックだけ走らせて、時間とピークメモリを測る。
2. 500エポックが現実的な時間（目安として半日以内）なら、そのまま最後まで走らせる。長すぎる場合はエポック数を減らす。そのときは再現ではなく縮小した実行として記録し、論文の値とは並べない。
3. テストの相対L2誤差が論文の値の1.2倍以内なら、環境に大きな問題はないとみなす。この基準は目安で、論文の値は1回の学習の結果と見込む。
4. 大きく外れたら、MPS で CPU に逃がした演算がないか、乱数の種による差、公式コードの版の違いを順に調べる。

時間とメモリ（1エポックの時間、ピークメモリ）は精度とは別の表に記録する。

## 実行結果

| 実行ID | 状態 | テストの相対L2（最終エポック） | 論文の値 | 論文の値との比 | 判断 |
|---|---|---|---|---|---|
| REPRO-02（Elasticity） | 500エポック完了 | 0.00605 | 0.0064 | 0.95 | 目安（1.2倍以内）に収まる。MPS の環境に大きな問題はないとみなす |
| REPRO-01（Airfoil） | 試走のみ | — | 0.0053 | — | 見送り。500エポックで約17時間かかり、環境の検証は REPRO-02 で足りるため、AirfRANS の作業を優先する |

- 学習は1回だけで、公式スクリプトは乱数の種を固定していない。論文の値よりわずかに小さいのは、乱数や版の違いによる揺れの範囲と見込む。
- 最終エポックの値を使った。公式スクリプトには、テストの誤差で最良のエポックを選ぶ仕組みがない。
- 途中のテストの相対L2: 0エポック 0.492、100エポック 0.0234、200エポック 0.0132、300エポック 0.00916、400エポック 0.00662、499エポック 0.00605。最終エポックの学習損失は 0.00448。

### 時間とメモリ（REPRO-02 の本番）

| 項目 | 値 |
|---|---|
| 1エポックの時間（500エポックの平均） | 23.9秒 |
| 全体の時間 | 11,925秒（約3時間19分） |
| MPS が確保したメモリ | 0.12 GiB |
| プロセスのピークメモリ | 0.58 GB |
| `PYTORCH_ENABLE_MPS_FALLBACK` | 未設定 |

学習中は `caffeinate -i` でスリープを止めた。ほかの重い処理は同時に走らせていない。

## 試走の時間とメモリ（2026-10-30）

公式スクリプトの設定のまま、エポック数だけを減らして走らせた。精度の比較には使わない。

| 実行ID | 試走のエポック数 | 1エポックの時間 | MPS が確保したメモリ | プロセスのピークメモリ | パラメータ数 | 500エポックの見積もり |
|---|---|---|---|---|---|---|
| REPRO-02（Elasticity） | 3 | 23.4秒（最初のエポックは25.6秒） | 0.12 GiB | 0.60 GB | 713,665 | 約3.3時間 |
| REPRO-01（Airfoil） | 2 | 121秒 | 4.19 GiB | 6.6 GB | 2,810,817 | 約17時間 |

- 1エポックの時間には、学習1000ケースとテスト200ケースの評価を含む。
- MPS が確保したメモリは `torch.mps.driver_allocated_memory()` の値。プロセスのピークメモリは `/usr/bin/time -l` の peak memory footprint。
- どちらも `PYTORCH_ENABLE_MPS_FALLBACK` を設定せずに最後まで動いた。CPU に逃がした演算はない。
- Airfoil は500エポックの見積もりが目安の半日を超える。

## 再現手順

### 環境

| 項目 | 版 |
|---|---|
| 機材 | Mac mini（Apple M5 Pro、メモリ24GB）、macOS 27.0.1 |
| Python | 3.12.12（`uv venv` で作成） |
| ライブラリ | torch 2.15.0、numpy 2.5.4、timm 1.0.30、einops 0.8.2、scipy 1.18.1、matplotlib 3.11.2、tqdm 4.70.1 |
| 装置 | `mps`。`PYTORCH_ENABLE_MPS_FALLBACK` は未設定 |

公式の `requirements.txt` は torch 1.10.1 と Python 3.8 を指定している。CUDA を使わない Mac では、この版のとおりには揃えられない。今回は最新の版を使った。`dgl` と `h5py` は Elasticity と Airfoil では使わないので入れていない。

```sh
uv venv --python 3.12 local/venv-bench
VIRTUAL_ENV=local/venv-bench uv pip install torch numpy einops timm matplotlib tqdm scipy gdown
git clone https://github.com/thuml/Transolver.git external/Transolver
git -C external/Transolver checkout 75e0f67643806a81cd1d3f6adc88dd8c02416fe7
```

### データ

Transolver の README が示す Google Drive のフォルダ（ID `1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8`）から、必要な5ファイルだけを `gdown` で取得した。合計1.5GB。フォルダにライセンスの記載は見当たらなかった。

| ファイル（`data/geofno/` からの相対パス） | Drive のファイルID | バイト数 | SHA-256 |
|---|---|---|---|
| `elasticity/Meshes/Random_UnitCell_sigma_10.npy` | `1Ia5izgUum-IQLdO6PW70HO8AdAqA_IVb` | 15,552,080 | `eb8102b5…a1a` |
| `elasticity/Meshes/Random_UnitCell_XY_10.npy` | `1I-fO-RsFvD3nqBuFrg67R0yqTFdD_gpA` | 31,104,080 | `29c615b7…65f` |
| `airfoil/naca/NACA_Cylinder_Q.npy` | `1AjW0t0YolY680J6xTQJ_g5bqTSJAZDZc` | 1,122,591,728 | `f79dfdaa…58c` |
| `airfoil/naca/NACA_Cylinder_X.npy` | `16EY0obqsccypaDFVY0wlsXX73SlD5TMy` | 224,518,448 | `23902420…35f` |
| `airfoil/naca/NACA_Cylinder_Y.npy` | `1rJUPtIhTAsG8TQnqV5mljjgQlyz0NJvJ` | 224,518,448 | `710c77e4…155` |

```sh
local/venv-bench/bin/gdown <ファイルID> -O data/geofno/<相対パス>
```

### 公式コードへの変更

差分は `local/transolver-mps.diff` に置いた（Git 管理外）。変更点は次のとおり。

- `exp_elas.py`、`exp_airfoil.py`、`utils/normalizer.py` に、`mps`（使えなければ `cpu`）を選ぶ `DEVICE` を加え、`.cuda()` を `.to(DEVICE)` に置き換えた。正規化クラスの `cuda()` メソッドは名前を残し、中身だけを `DEVICE` への移動に変えた。
- `exp_elas.py` と `exp_airfoil.py` の学習ループに、エポックごとの時間と `torch.mps.driver_allocated_memory()` を表示する行を加えた。表示だけで、計算は変えていない。

### 実行したコマンド

`external/Transolver/PDE-Solving-StandardBenchmark/` で実行した。引数は公式の `scripts/Transolver_*.sh` と同じで、`--gpu`、`--epochs`、`--save_name`、`--data_path` だけを変えた。

```sh
# REPRO-02 の試走（3エポック）
/usr/bin/time -l python exp_elas.py --gpu 0 --model Transolver_Irregular_Mesh \
  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --max_grad_norm 0.1 \
  --batch-size 1 --slice_num 64 --unified_pos 0 --ref 8 --eval 0 \
  --epochs 3 --save_name elas_probe --data_path ../../../data/geofno

# REPRO-01 の試走（2エポック）
/usr/bin/time -l python exp_airfoil.py --gpu 0 --model Transolver_Structured_Mesh_2D \
  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --max_grad_norm 0.1 \
  --batch-size 4 --slice_num 64 --unified_pos 0 --ref 8 --eval 0 \
  --epochs 2 --save_name airfoil_probe --data_path ../../../data/geofno/airfoil/naca
```

```sh
# REPRO-02 の本番（500エポック）
caffeinate -i /usr/bin/time -l python exp_elas.py --gpu 0 --model Transolver_Irregular_Mesh \
  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --max_grad_norm 0.1 \
  --batch-size 1 --slice_num 64 --unified_pos 0 --ref 8 --eval 0 \
  --epochs 500 --save_name elas_REPRO-02 --data_path ../../../data/geofno
```

エポック数を減らすと、余弦スケジュールの周期（`T_max`）も短くなる。試走の誤差は500エポックの途中経過とは違う。
