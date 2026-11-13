# CFD サロゲートモデル 作業メモ

CFD の結果を、形状や気流条件から予測するサロゲートモデルの作業フォルダです。これまでの対象は、Transolver による翼まわりの流れ場と表面の量の予測、舵角の違う表面格子の補間です。Transolver 本体のソースやデータセットは含みません。

調査の結論と記録は主題ごとにまとめ、[調査記録の索引](docs/README.md)から辿れます。

| 主題 | 本文 |
|---|---|
| 環境とデータセットの選定 | [docs/setup/feasibility.md](docs/setup/feasibility.md) |
| AirfRANS の翼面の量の予測 | [docs/surface/conclusions.md](docs/surface/conclusions.md) |
| 舵角の違う表面格子の補間 | [docs/reports/20261114-fin-remap-su2/README.md](docs/reports/20261114-fin-remap-su2/README.md) |

- `docs/reports/`: 1回ごとの調査の記録（計画、実使用設定、集計CSV、図、再現手順）。
- `data/`: データセット（AirfRANS など）。ローカル専用。
- `results/`: 学習結果、チェックポイント、ログ。ローカル専用。
- `external/`: Transolver 公式リポジトリなどの clone。ローカル専用。
- `local/`: 環境固有の設定。ローカル専用。

ローカル専用ディレクトリはGit管理対象外です。新しいcloneでは必要に応じて作成します。

```sh
mkdir -p data results external local
```

## 共有する情報

GitHubには汎用的なメモを保存します。個人名、メールアドレス、アカウント名、ホームディレクトリの絶対パス、端末識別子、認証情報、データセット本体、学習済み重みは含めません。非公開リポジトリでも、コミット前に差分を確認します。
