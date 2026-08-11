# JAX 化と GPU 検証 — 結果

> 目的: エージェントと環境を JAX 化し、**実際に GPU 上で高速化できているかを検証する**。
> 結論: **CUDA では 42,900x 高速化した。ただし Apple Metal では逆に遅くなる。**
> GPU バックエンドによって結果が正反対になるため、「GPU なら速い」とは言えない。

## 0. 結論サマリ

| 実行環境 | batch | env-steps/s | official 比 |
|---|---|---|---|
| official (`env.run`) | 1 | 850 | 1x |
| fast (interpreter 直叩き) | 1 | 13,337 | 16x |
| JAX / CPU (M2 Max) | 1 | 50,973 | 60x |
| JAX / CPU (M2 Max) | 8192 | 276,042 | 325x |
| **JAX / Metal GPU** | 1 | 219 | **0.26x（official より遅い）** |
| **JAX / Metal GPU** | 64 | 922 | **1.1x** |
| **JAX / CUDA (RTX 4090)** | 64 | 226,014 | 266x |
| **JAX / CUDA (RTX 4090)** | 8192 | 25,942,145 | 30,520x |
| **JAX / CUDA (RTX 4090)** | **65536** | **36,489,836** | **42,929x** |
| JAX / CUDA (RTX 4090) | 262144 | 31,956,572 | 37,596x（飽和後） |

- **CUDA では狙いどおり効いた。** batch 65536 で **3,650 万 env-steps/s**。
  それ以上は頭打ちになり 262144 では低下する（飽和点は 65536 付近）
- **Metal では逆に遅い。** CPU の約 1/118。batch 256 以上は 24 step すら
  2 分で終わらず計測を打ち切った
- CUDA は同じ batch 64 で Metal の **245 倍**（226,014 対 922）

## 1. 検証環境

2 系統で計測した。

| 項目 | ローカル | CUDA |
|---|---|---|
| マシン | Apple M2 Max、GPU 30 コア、32 GB | RunPod RTX 4090 (24GB) / Community Cloud |
| バックエンド | `jax-metal` 0.1.1 + JAX 0.4.34 | JAX 0.11.0 + `jax[cuda12]` |
| ドライバ | — | 550.54.15 / CUDA 12.4 |
| コスト | — | $0.34/h × 約 14 分 ≈ **$0.08** |

`jax-metal` は JAX 0.11 とは非互換（`StableHLO_v1.14.0` の
`unknown attribute code: 22` で 2000×2000 matmul すら失敗）。0.4.34 に固定して
はじめて動作した。**CUDA 側は JAX 0.11 のままで問題なく動く。**

RunPod pod は計測後に terminate 済み（`get_pods()` で残存なしを確認）。
`~/.runpod/config.toml` の API キーを `backend/.env` に転記して使用した
（`backend/.env` は gitignore 済み）。

### GPU は実在し、行列積では速い

バックエンドが偽物でないことを先に確認した:

```
GPU matmul 2048^3: 2.91 ms -> 5,903 GFLOP/s
numpy CPU  2048^3: 10.30 ms -> 1,668 GFLOP/s   → 3.5x
```

つまり **GPU 自体は正常に動いている**。遅いのは環境シミュレーションの
ワークロードとの相性である。

## 2. Metal が遅い原因 — scatter。CUDA にはこの問題がない

同一コードを 3 バックエンドでマイクロベンチした（B=64、×50 回）:

| 操作 | CPU | Metal GPU | **CUDA (4090)** |
|---|---|---|---|
| 要素演算 `(B,2,10,10)` | 1.5 ms | 24.6 ms | 6.0 ms |
| **`.at[].set` scatter** | **1.0 ms** | **222.7 ms** | **5.7 ms** |
| scatter / 要素演算 の比 | 0.7x | **9.1x** | **0.95x** |

**決定的な差はこの比率にある。** CUDA では scatter が要素演算と**同じコスト**
（5.7 対 6.0 ms、比 0.95）で、ペナルティが存在しない。一方 Metal は 9.1 倍かかる。

`jax-metal` に効率的な scatter 実装がないのが Metal 側の原因である。1 回あたり
約 4.5 ms、本環境の 1 step はタイル操作ごとに scatter を要し合計 15 回程度
発生するので、step あたり数十 ms が scatter だけで消える。batch を増やしても
このレイテンシは減らないため **Metal ではスケールしない**。

タイルベースの環境は本質的に「特定タイルだけ書き換える」操作の集合なので、
scatter 性能がそのまま環境の性能になる。**CUDA を使うべき理由はここにある。**

### 行列積の素の性能（参考）

| バックエンド | 2048³ matmul | GFLOP/s |
|---|---|---|
| numpy CPU | 10.30 ms | 1,668 |
| Metal GPU | 2.91 ms | 5,903 |
| **CUDA (RTX 4090)** | **0.40 ms** | **43,269** |

Metal も行列積では CPU の 3.5x 出る。つまり **Metal の GPU 自体は正常で、
遅いのは scatter という特定の操作**である。

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

### CUDA 上でも同一値を確認した

速度だけ出ても値が違えば意味がないので、**RTX 4090 上で同じ 15 日エピソードを
再実行**し、CPU で公式 engine と一致させた値と突き合わせた:

```
CUDA batch1  : (3057.0, 10, 2, 10001)
CUDA batch64 : (3057.0, 10, 2, 10001)
公式 engine   : (3057.0, 10, 2, 10001)   → 一致
市場価格 @I0  : WHEAT 25（公式と一致）
```

float32 の丸めが絡む市場価格を含めて一致したので、**バックエンドを変えても
数値は変わらない**ことを確認できた。

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

**CUDA 前提なら明確に成功。ただし「GPU なら速い」は誤り。**

- **CUDA では 42,929x**（batch 65536）。1 シーズン 720 turn が実質 0.02 ms 相当で、
  RL の学習ループに載せられる水準に達した
- **Metal では 0.26x**（official より遅い）。同じコード・同じ JAX で
  バックエンドを替えるだけで **245 倍**の差が出る
- 差の原因は scatter の実装有無に尽きる（§2）。汎用的な教訓としては
  **「タイルベース環境を GPU に載せる際は scatter 性能を先に測る」**

### 実務上の使い分け

| 用途 | 推奨 | 根拠 |
|---|---|---|
| 提出前の検証・リプレイ | `official` (`env.run`) | 本番と同じ経路 |
| ローカルの大量対戦評価 | `fast` (interpreter 直) | 13k steps/s、ルール乖離ゼロ |
| パラメータ探索（Mac 上） | JAX / CPU | 276k steps/s。Metal は使わない |
| RL 学習・大規模自己対戦 | **JAX / CUDA** | 36.5M steps/s |

### 次に取るべき手段（推奨順）

1. **実装範囲の拡張** — これが最大のボトルネック。現状は小麦ループ相当しか
   表現できず、hand / 家畜 / 土地 / 肥料が未実装。**速度はもう足りているので、
   次の投資先は速度ではなく忠実度**
2. **RL 学習ループの構築** — 36.5M steps/s あるので環境側は律速にならない。
   RunPod での実行は今回の手順（§7）で再現できる
3. **飽和点を意識した batch 選択** — 65536 が最大で、262144 では低下する。
   RTX 4090 では 65536 前後を使うのが最適

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

### CUDA (RunPod) の再現手順

`dev/runpod` は push 済み SHA と AWS/DVC 設定を要求する重い経路なので、
自己完結した benchmark には SDK 直叩きの一時 pod を使った。

```bash
# 1. GPU deps と API キー
cd backend && uv sync --group gpu
#    ~/.runpod/config.toml の apikey を backend/.env の RUNPOD_API_KEY に転記
#    (backend/.env は gitignore 済み)

# 2. 在庫と価格の確認
dev/runpod stock --min-memory-gb 16 --max-dph 1.0

# 3. pod 作成 (RTX 4090 / COMMUNITY)。jaxenv だけを tar で送る
tar -czf jaxenv.tar.gz -C backend/src/simulate jaxenv
#    create_pod(onstart=...) で sshd を起動し jax[cuda12] を pre-install、
#    以後は SSH 経由で benchmark を実行する

# 4. SSH で実行 (pod の python は 3.10 だが pip は 3.13 に入るので注意)
ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <port> root@<ip> \
  'cd /work && /usr/bin/python3.13 -m jaxenv.benchmark \
      --steps 720 --batches 1 64 1024 8192 65536 --skip-cpu'

# 5. 必ず terminate し、残存がないことを確認
#    runpod.terminate_pod(pod_id); runpod.get_pods() -> []
```

踏んだ点を 2 つ:

- **RunPod SDK にログ取得 API がない**（`create_pod` / `get_pod` /
  `terminate_pod` のみ）。pod 内 stdout だけに出した結果は回収できないので、
  SSH で対話実行するのが確実
- **`pip` と `python` の Python バージョンが違う**。image の `python` は 3.10 だが
  `pip` は 3.13 の dist-packages に入るため、`python -m` では `ModuleNotFoundError`
  になる。`/usr/bin/python3.13` を明示する

pod は必ず終了させる。今回は onstart 側に watchdog（`sleep` 後に
`runpodctl remove pod`）を仕込み、launcher の `finally` と手動 terminate
スクリプトを合わせた三重の停止経路にした。

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

---

# 追記 (2026-08-11): フル実装への拡張と GPU 実測の再挑戦

## 実装範囲を subset から全ルールへ拡張

本ドキュメント上部の計測は **wheat loop が使う op のみの subset 実装**に対する
ものだった。RL 学習用途では機能欠落が許容できないため、公式 interpreter
(1073 行) の全ルールを port した。

追加: unit op 18 種 (BUILD_COOP/PASTURE, PLACE, FEED, CARE,
COLLECT_FERTILIZER, FERTILIZE, DROP, PICKUP)、market op 7 種 (BUY_PRODUCT,
BUY_ANIMAL, HIRE, BUY_LAND)、家畜・雇用 hand・土地購入・肥料・町ショップ・雑草。

### 等価性の担保方法を変更した

subset 版はスクリプト固定のテストだったが、フル実装では**差分ファズ**にした。
全 op を無作為に叩き、公式 engine と同一 action 列で駆動して毎ターン全状態
(tile 8 面 / 各 unit の所持品 / shed / seeds / 家畜 / 資金 / 市場在庫 / 位置)
を突き合わせる。seed 100 本 × 3/6/10/14/30 日で不一致ゼロ。

このファズが発見した実装バグ 4 件は、いずれも「静かに低スコア化」する種類で
スクリプト固定テストでは検出できなかった:

1. tie-break の sentinel が 0 で、unit 0 以外が PLANT できない
2. 種の atomic 事前チェックを LOCKED マスク後に評価していた
3. **op-major に適用していた** — engine は unit-major。同ターン内で hand A の
   DIG が空けた tile に hand B が PLANT する挙動が再現できていなかった。
   `fori_loop` で unit-major に変更し、同時に同一 tile 衝突が構造的に消えた
4. PLACE の shed-drop fallthrough 欠落。かつ animal 配置は LOCKED ガード後、
   shed drop はガード前に解決する必要がある

## CPU 実測 (フル実装)

| batch | env-steps/s | 公式比 |
|---|---|---|
| 1   | 1,125 | 1.5x |
| 64  | 3,871 | 5.0x |
| 512 | 5,113 | 6.7x |

subset 版の 707x から大きく落ちた。unit ループ 16 周と全 op の scatter が
入ったため。**tile 環境の性能は scatter 性能で決まる**という上部の結論とは
整合しており、CUDA なら改善が見込まれる。

## GPU 実測は未達成 — RunPod のインフラ障害

pod 8 台、約 3.5 時間、概算 $2〜3 を投じたが **GPU 実測に到達できなかった**。
前回 (RTX 4090 で 42,929x) は成功しているので、今回に限った障害と思われる。

### 切り分け結果: simulator は無関係

simulator のコードを一切使わず `echo ALIVE; hostname` だけを実行した:

| 試行 | 結果 |
|---|---|
| pod 起動 / SSH ポート割当 | 成功 |
| SSH 接続 10 回 | **成功 1 / 失敗 9** |
| 続く接続 100 回 | **成功 0** |

pod は `RUNNING` を報告し続けながら接続の 87〜95% を拒否する。成功した回では
hostname が正常に返るので pod 自体は生きている。

### 失敗の内訳

| # | 構成 | 失敗内容 |
|---|---|---|
| 1 | 4090 SECURE | sshd が接続をほぼ毎回切断 |
| 2 | 4090 COMMUNITY | pip が 1.26MB で停止 (3 分間 0 bytes) |
| 3 | `nvidia/jax:jax` | JAX nightly が sm_89 非対応 (`no supported devices`) |
| 4 | `nvidia/jax:jax-2024-10-16` | 約 20GB の image pull が 30 分で未完了 |
| 5 | 3090 COMMUNITY | **同一 host:port が複数マシンに負荷分散**され、upload したファイルが次の接続で消える |
| 6 | 4090 SECURE | pip が 4.5MB で停止 (150 秒間 0 KB/s) |
| 7 | 4090 SECURE (uv) | **4.5GB まで到達**したが同じく停止 |
| 8 | 4090 SECURE (wheel 直送) | 8MB chunk すら `Broken pipe` で転送不可 |

### 分かったこと

- **`uv` は `pip` より大幅に強い。** pip が数 MB で諦めた所を uv は 4.5GB まで
  到達した (並列 DL + リトライ)。pod 上の依存導入は uv を使うべき
- **`jax[cuda12]` の 4.5GB のうち大半は NVIDIA の CUDA ライブラリ。**
  JAX 本体 4 wheel (jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt) は
  **合計 284MB** しかない。RunPod の PyTorch image は CUDA ランタイムを
  同梱しているので、ローカルで wheel を落として転送すれば PyPI を経由せずに
  導入できる — この方針自体は正しいが、今回は転送路が死んでいて使えなかった
- **`setsid nohup` でも SSH 切断でジョブが死ぬことがある。** リトライループを
  ジョブ内に持たせる必要がある

### 次に試すなら

1. 時間を置いて RunPod を再試行 (一時障害の可能性)
2. Colab / Lambda Labs 等の別 CUDA 環境。bench 一式は `state.py` / `env.py` /
   `bench.py` / `fidelity.py` の 4 ファイルで自己完結しており、CPU 側で取得した
   状態 digest と突き合わせれば GPU 上の忠実性も確認できる
3. ローカル (M2 Max / Metal) は**不適**。上部の通り scatter が 9.1x 遅く、
   フル実装では subset の 0.26x より更に悪化する見込み

pod は全て terminate 済み、`dev/runpod ps` で残存なしを確認。
