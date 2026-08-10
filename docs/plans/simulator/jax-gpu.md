# JAX 化と GPU 検証 — 結果

> 目的: エージェントと環境を JAX 化し、**実際に GPU 上で高速化できているかを検証する**。
> 結論: **GPU では高速化できなかった。JAX 化自体は 20x 効いたが、その加速は CPU 上で得られたもの。**

## 0. 結論サマリ

| 実行環境 | batch | env-steps/s | official 比 |
|---|---|---|---|
| official (`env.run`) | 1 | 850 | 1x |
| fast (interpreter 直叩き) | 1 | 13,337 | 16x |
| **JAX / CPU** | 1 | 50,973 | 60x |
| **JAX / CPU** | 1024 | 191,096 | **225x** |
| **JAX / CPU** | 8192 | 276,042 | **325x** |
| **JAX / Metal GPU** | 1 | 219 | **0.26x（official より遅い）** |
| **JAX / Metal GPU** | 64 | 922 | **1.1x** |

GPU は CPU の **1/118**（batch 64 時点で 922 対 109,236）。batch 256 以上は
24 step すら 2 分で完了せず、計測を打ち切った。

## 1. 検証環境

| 項目 | 内容 |
|---|---|
| マシン | Apple M2 Max、GPU 30 コア、32 GB |
| GPU バックエンド | `jax-metal` 0.1.1 + JAX 0.4.34（唯一のローカル GPU 経路） |
| CUDA | **なし**（darwin。`nvidia-smi` 不在） |
| RunPod | 認証情報・SDK ともに未設定のため使用せず |

`jax-metal` は JAX 0.11 とは非互換（`StableHLO_v1.14.0` の
`unknown attribute code: 22` で 2000×2000 matmul すら失敗）。0.4.34 に固定して
はじめて動作した。

### GPU は実在し、行列積では速い

バックエンドが偽物でないことを先に確認した:

```
GPU matmul 2048^3: 2.91 ms -> 5,903 GFLOP/s
numpy CPU  2048^3: 10.30 ms -> 1,668 GFLOP/s   → 3.5x
```

つまり **GPU 自体は正常に動いている**。遅いのは環境シミュレーションの
ワークロードとの相性である。

## 2. なぜ GPU で遅いのか — scatter が原因

同一コードを両バックエンドでマイクロベンチした（B=64）:

| 操作 | CPU | Metal GPU | GPU/CPU |
|---|---|---|---|
| 要素演算 `(B,2,10,10)` ×50 | 1.5 ms | 24.6 ms | 16x 遅い |
| `fori_loop` ×24（自明な body）×20 | 4.4 ms | 34.7 ms | 8x 遅い |
| **`.at[].set` scatter ×50** | **1.0 ms** | **222.7 ms** | **223x 遅い** |
| market 風 `fori_loop` ×32 ×20 | 0.3 ms | 18.7 ms | 62x 遅い |

**`jax-metal` に効率的な scatter 実装がない。** 1 回あたり約 4.5 ms。
本環境の 1 step はタイル操作ごとに scatter を要し、合計 15 回程度発生するので、
step あたり数十 ms が scatter だけで消える。batch を増やしても scatter の
レイテンシは減らないため、**スケールしない**。

タイルベースの環境は本質的に「特定タイルだけ書き換える」操作の集合なので、
この欠落は致命的である。

## 3. 実装したもの

```
backend/src/simulate/jaxenv/
  state.py       EnvState（struct-of-arrays、batch 軸先頭）+ 市場価格モデル
  env.py         jit 済み step。全ルールを jnp.where のマスク更新で表現
  town.py        ショップ解放スケジュールの host 側再現
  benchmark.py   official / fast / jax の env-steps/s 比較
```

### 実装範囲（過大に読まないための明示）

**実装した**: 作物（5 種）、植付・水やり・収穫・DIG・移動、市場の
SELL / BUY_SEED（per-unit lockstep）、価格モデル（9 商品・上下で別形状関数）、
町中心の需要、日次処理（水やり判定・雑草化・継続産出・shed への搬入）。

**実装していない**: 雇用 hand、家畜、肥料、`BUY_LAND`、ショップ需要の
device 側適用、雑草の自然発生。

## 4. 等価性の検証

**再実装なので `fast` と違って構造的な等価性がない。** 実測で確認した。

| 検証 | 結果 |
|---|---|
| 市場価格 9 商品 × 223 在庫点（`I0` 境界含む） | **2,007 点すべて一致** |
| 小麦ループ 15 日 = 360 step（money/seeds/shed/market_inv） | **全 step 一致** |
| batch 1 と batch 16 で同一結果 | 一致 |
| batch 内の環境が互いに干渉しない | 確認 |

比較は `weedSpawnChance=0` かつショップ無効の条件で行った。理由は §5。

## 5. 途中で見つかった engine の事実

| 発見 | 影響 |
|---|---|
| **町中心の需要に日次倍率はない** | `abstract.md` は「11 日目以降 2 倍、21 日目以降 4 倍」と書くが、この engine 版のコードは常に 1 個・間隔 24。**abstract が実装と食い違っている** |
| `BUY_SEED` は市場在庫を減らさない | 種は市場の板から引かれず「発行」される |
| **ショップ解放の RNG は盤面依存** | `_spawn_weeds` が同じ RNG を先に消費し、その回数は**空きタイル数**に等しい。実測で 1 日目 50 回 → 雑草が出ると 49 → 48 と減る。よって解放スケジュールは seed だけでは決まらず、事前テーブル化できない |

3 番目のため、ショップを含めた完全な等価性は「host 側で batch と歩調を合わせて
RNG を回す」以外に方法がない。今回は**両エンジンでショップを無効化**して
決定論的コアを比較する方針を採った（`town.py` の docstring に記録）。

## 6. 評価 — この方向は妥当か

**GPU 目的としては失敗、CPU 上の JAX 化としては成功。**

- 高速化の要因は **GPU ではなく vectorization + jit**。batch 8192 で
  276k env-steps/s は official の 325 倍で、これは CPU で得られている
- **Metal では逆に遅くなる**ので、この環境を Apple GPU に載せる意味はない
- CUDA GPU で速くなるかは**未検証**。CUDA は scatter を実装しているので
  Metal のような崩壊は起きないと予想されるが、**実測していないので断言しない**。
  検証には RunPod（認証情報が必要）か CUDA マシンが要る

### 次に取るべき手段（推奨順）

1. **CPU 上の JAX env をそのまま使う** — 実装済み・検証済み・276k steps/s。
   パラメータ探索や自己対戦データ生成には十分
2. **CUDA GPU で再計測** — RunPod 認証情報を設定すれば `benchmark.py` が
   そのまま走る。scatter 性能が Metal と違うかを見るのが要点
3. **RL に進むなら** 実装範囲（hand / 家畜 / 土地）の拡張が先。現状は
   小麦ループ相当しか表現できない

## 7. 再現手順

```bash
# CPU（リポジトリの venv でそのまま）
cd backend/src && PYTHONPATH=. python -m simulate.jaxenv.benchmark \
    --steps 720 --batches 1 64 1024 4096 8192 --skip-cpu

# 等価性テスト
dev/test tests/e2e/src/simulate/test_jaxenv_equivalence.py

# Metal GPU を再現する場合（別 venv 必須。JAX 0.4.34 に固定）
uv venv --python 3.11 /tmp/metalenv
VIRTUAL_ENV=/tmp/metalenv uv pip install 'jax==0.4.34' 'jaxlib==0.4.34' \
    jax-metal kaggle-environments
```

> `jax-metal` を**リポジトリの venv に入れてはいけない**。入れると
> `jax.devices()` が METAL を返し、既存テストがその上で走ってしまう。
> 検証後にアンインストール済み。

### 依存関係について

`jax` は `pyproject.toml` に追加していない。**`kaggle-environments` が
`jax` を直接 requires しているため、engine がある環境には必ず存在する**
（`uv pip show kaggle-environments` の Requires に `jax` がある）。
当初 `[dependency-groups] jax` を追加したが、冗長なので取り消した。

テスト側の `pytest.importorskip("jax")` はそのため実際には発火しない。
将来 `kaggle-environments` が依存を落とした場合の保険として残している。

## 8. テスト

`dev/test` で **141 件パス**（JAX 等価性 6 件を含む）、ruff / mypy strict クリーン。
JAX 関連テストは `tests/e2e/src/simulate/test_jaxenv_equivalence.py`
（実エピソードを回すため `e2e` 分類）。
