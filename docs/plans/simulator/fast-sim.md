# 高速シミュレータ 方針

> 前提: [`design.md`](design.md) の評価基盤は実装済み（113 テスト / 120 ep 25.4 秒）。
> 本文書はその上に載せる**高速化レイヤ**の方針。段階的に 3 Phase。

## 0. プロファイル結果 — ゲームロジックは 1.7% しかない

フルシーズン 1 エピソード（`starter` vs `pass`, 720 turn）を `cProfile` で計測:

| 内訳 | 時間 | 割合 |
|---|---|---|
| `structify`（dict → 属性アクセス化） | 1.00s | **30.5%** |
| その他 stdlib | 0.90s | 27.5% |
| `deepcopy` | 0.84s | **25.7%** |
| `jsonschema` バリデーション | 0.39s | **11.9%** |
| core framework | 0.08s | 2.6% |
| **ゲームロジック本体（`kaggriculture.py`）** | **0.06s** | **1.7%** |

**結論: 遅いのはゲームではなくフレームワーク。** `env.run()` は 1 ステップごとに
state を deepcopy し、structify し、jsonschema で検証している。ゲームルールを
自前で書き直しても取り戻せるのは 1.7% で、**引き換えにルール乖離のリスクを負う**。

したがって方針は「**ルールは公式のまま、フレームワークだけ剥がす**」。

## 1. Phase 1 — interpreter 直叩き（30x, ルール乖離リスク ゼロ）

`kaggriculture.interpreter(state, env)` を自分のループから直接呼ぶ。
実測済み:

```
env.run           824 ms / 720 steps
interpreter 直叩き  27 ms / 720 steps     → 30x
```

**5 seed で最終所持金が bit-identical であることを実測確認済み**
（seed 1 / 2 / 7 / 42 / 99、`starter` vs `pass`）。公式の interpreter を
そのまま呼ぶので当然だが、下記の罠を踏むと静かにズレる。

### ★ 罠: `observation.step` はフレームワークが管理している

`core.py:626` が `new_state[0].observation.step` を毎ステップ設定している。
interpreter 自身は `step` を**書かない**（読むだけ）。よって直叩きループで
step を進め忘れると:

- `day` / `hour` が 0 のまま凍結
- 日次処理（水やり判定・雑草化・家畜の産出・町の消費）が一切走らない
- エピソードは「動いているように見えて」所持金が初期値付近で止まる

実際にこれを踏み、全 seed で `money=2960` に張り付いた。修正は 1 行:

```py
for n in range(steps - 1):
    state[0].action = agent0(state[0].observation)
    state[1].action = agent1(state[1].observation)
    interpreter(state, env)
    state[0].observation.step = n + 1     # ← これ
```

同様にフレームワーク側の責務で、直叩きでは**自前で持つ必要があるもの**:

| 項目 | 誰の責務 | 直叩きでの扱い |
|---|---|---|
| `observation.step` | framework | 手動でインクリメント（上記） |
| `reward` / `status` | interpreter（最終ステップのみ） | 最終所持金は `farms[i]["money"]` から直接読む |
| agent の例外捕捉 | framework | 自前で `try/except` → `PASS` |
| agent の実行時間計測 | framework | 必要なら `perf_counter` で自前計測 |
| `env.steps` 履歴 | framework | **持たない**（これが速さの源。リプレイが要る時は Phase 1 を使わない） |

### 位置づけと API

```
backend/src/simulate/
  fast.py        FastEpisode — interpreter 直叩きループ
```

- **dev ライブラリとして実装**（`backend/src/**`）。提出物には入らない。
- 既存の `EpisodeOutcome` を返し、`matrix.py` から `--engine fast` で選べるようにする。
- **既定は今のまま `env.run`**。`fast` は opt-in にする — フレームワークを剥がす
  ということは、将来 `kaggle-environments` が更新された時に追従が必要になるということ。

### 使える／使えない場面

| 用途 | Phase 1 で可 | 理由 |
|---|---|---|
| 大量対戦の勝率評価 | ○ | 最終所持金だけ要る。120 ep が 25 秒 → 1 秒 |
| パラメータ探索（売却量・雇用数の掃引） | ○ | 数千エピソードが現実的に |
| クラッシュ / timeout 検出 | △ | 自前計測が必要。**厳密な検証は `env.run` 側で行う** |
| リプレイ出力 | ✗ | 履歴を持たないのが速さの源。`--replay` は `env.run` 経路のみ |
| 提出前の最終確認 | ✗ | 本番と同じ経路で通すべき。`dev/submit --dry-run` は現状維持 |

> **`fast` は探索用、`env.run` は検証用**という役割分担を崩さない。
> 速い方だけで判断すると、フレームワーク層に起因するバグを見落とす。

## 2. Phase 2 — エージェント内 rollout（要 前提確認）

先読みする agent（1手先の市場価格・収量シミュレーション）向け。

実測した性能:

| 操作 | コスト | スループット |
|---|---|---|
| `deepcopy(state)` | 151 us | 6,633 clone/sec |
| 620 step の rollout | 14.4 ms | 70 rollout/sec |

clone した state で回しても**元の state は不変**であることを確認済み（`deepcopy` で十分、
`interpreter` の in-place 変更は clone 内に閉じる）。
`actTimeout = 1` 秒なので、1 ターンあたり**数十本の rollout** が予算内に収まる。

### ★ 着手前に必ず確認すること

**提出コードが `kaggle_environments` を import できるかは未確認。**
`docs/competition/AGENTS.md` はローカル検証での `pip install kaggle-environments` に
言及しているだけで、**エージェント実行環境で import 可能とは書いていない**。

- import できる → interpreter を rollout に流用できる（実装は軽い）
- import できない → ルールの**自前実装が必要**になり、コストと乖離リスクが跳ね上がる

判定方法: 提出可能な最小 agent に `import kaggle_environments` を仕込み、
`dev/submit --dry-run` ではなく**実際に 1 回提出**して episode が完走するか見る。
提出枠を 1 消費するので、Phase 2 に本当に進む段階で行う。

この確認が取れるまで Phase 2 は着手しない。取れなかった場合は、
先読みの対象を**市場価格モデルだけに絞る**のが現実的（`market_price()` は
純関数で、`MARKET_PARAMS` を写すだけで再現できる。盤面全体の再実装は要らない）。

## 3. Phase 3 — RL 用の環境（現時点では起こさない）

数百万ステップ規模が必要になった場合のみ。Phase 1 の 27ms/episode でも
100 万ステップ = 約 40 秒相当だが、RL は**並列 env と観測のテンソル化**が要るので
別物になる。

方針の選択肢（着手時に再評価）:

1. **Phase 1 + プロセス並列** — 実装ゼロ。まずこれで足りるか測る
2. **numpy ベクトル化した自前実装** — 盤面を配列で持ち N 環境同時に進める。
   公式との等価性テスト（Phase 1 を ground truth にした差分検証）が必須
3. **既存実装の流用** — 他コンペの高速化事例を調査してから

**Phase 3 は「Phase 1 で足りないと実測で示せてから」着手する。** 現時点で
必要という根拠はない。

## 4. 実装順序（Phase 1 のみ具体化）

| # | 作業 | 検証 | 状態 |
|---|---|---|---|
| 1 | `fast.py` に `run_fast_episode` | `step` 進行を忘れた場合に**テストが落ちる**こと | 完了 |
| 2 | 公式との等価性テスト | 複数 seed × 複数エージェント組で最終所持金一致 | 完了 |
| 3 | `matrix.py` / CLI に `--engine fast` | 既定は `env.run` のまま | 完了 |
| 4 | ベンチマーク記録 | 30x が維持されているか | 完了 |

### 等価性テストが本命

Phase 1 の価値は「速いこと」ではなく「**速くて、かつ公式と一致すること**」。
`step` の罠のように、ズレても例外は出ず結果だけ静かに変わる。よって:

- 等価性テストは `e2e/` に置き、**複数 seed で最終所持金の完全一致**を assert する
- `starter` / `pass` / `rulebase/case1` の組み合わせで回す
- **`kaggle-environments` のバージョンを上げた時に必ず走らせる**（追従の検知装置）

これは `design.md` §0 と同じ思想 — 静かに壊れるものにテストで蓋をする。

## 6. Phase 1 実装結果

**テスト 135 件（うち等価性 17 件）/ ruff・mypy strict クリーン。**

### 実測スループット

| ワークロード | official | fast | 短縮 |
|---|---|---|---|
| 120 ep（3 相手 × 20 ep × swap） | 25.4 s | **1.9 s** | 13x |
| 1000 ep（500 ep × swap） | 約 3.5 分（推定） | **10.4 s** | — |
| 1 ep 単体（720 turn, プロセス並列なし） | 824 ms | **27 ms** | 30x |

CLI 経路の短縮率（13x）が単体（30x）より小さいのは、既にプロセス並列で
11 ワーカー使っており、プロセス起動と `make()` の初期化が定数コストとして残るため。

deterministic な組み合わせでは**両エンジンの数値が完全一致**している
（`rulebase/case1` vs starter: どちらも mean money 9,033 / vs pass: 9,896）。

### 等価性テストは mutation で検証した

「テストがあること」自体は保証にならないので、`fast.py` に意図的なバグを入れて
落ちるか確認した:

| 変異 | 結果 |
|---|---|
| `state[0].observation.step = n + 1` を削除 | **17 件中 15 件が失敗** — 検出できている |
| ループ上限を `steps - 1` → `steps` に変更（off-by-one） | **17 件全て成功 = 検出できない** |

2 番目は当初「off-by-one を守る」とテストの docstring に書いていたが、調べると
`interpreter` は `env.done` で早期 return するため**余分な 1 回は無害な no-op**
だった。テストの不備ではなくバグが存在しないケースなので、
docstring から誤った主張を削除し、実際に守れている性質（clock 進行）だけを
記述するよう修正した。

### 使い分け（実装済みのガード）

- 既定は `--engine official`。`fast` は明示的な opt-in。
- `--engine fast --replay all` は**エラーで停止**する（履歴を持たないので
  リプレイを作れない。黙って 0 件書くのを避ける）。
- `fast` 実行時はレポートのヘッダに `engine=fast` が出る
  （fast の数値を検証済みの数値と混同しないため）。
- `dev/submit --dry-run` は `official` 経路のまま。

## 5. やらないこと

- **ゲームロジックの自前実装（Phase 1 の範囲では）** — 1.7% のために乖離リスクを負わない
- **`fast` を既定にすること** — 検証は本番と同じ経路で行う
- **Phase 2 / 3 の先行実装** — Phase 2 は import 可否の確認待ち、Phase 3 は必要性の実証待ち
