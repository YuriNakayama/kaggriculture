# ローカルシミュレータ 設計

> 目的: **エージェント評価基盤**。複数 case × 複数対戦相手 × 複数 seed のマトリクスを並列実行し、勝率・収支・**失敗の理由**を集計する。
> 方針: 既存 `backend/src/simulate/` を**再設計**（`runner.py` は互換ラッパとして残す）。動作確認用に `rulebase/case1` の最小 case も同時に作る。

## 0. 実測した前提（設計の根拠）

すべて本ワークツリーの `kaggle-environments 1.32.6` で計測済み。

| 事項 | 実測値 | 設計への影響 |
|---|---|---|
| 1 エピソード（720 turn, `starter` vs `random`） | **約 1.3 秒** | 高速。Python の素実装で十分、独自の高速 sim は不要 |
| 8 並列（`ProcessPoolExecutor`, spawn） | 8 ep / **1.5 秒** | プロセス並列がそのまま効く。50 ep × 3 相手 でも十数秒 |
| `env.toJSON()` のサイズ | **約 4.8 MB / ep** | 全 ep のリプレイ保存は非現実的。**既定は保存しない**（§4） |
| `actTimeout` | **1 秒 / turn** | ローカルでも遵守が必須。超過を検出できないと本番で無言に壊れる |
| `state[i]["status"]` | クラッシュ時も timeout 時も **`DONE`** | ★ 現行 `runner.py` の失敗判定が機能しない（§1） |

### ★ 最重要: 失敗はステータスに出ない

`status` / `reward` だけを見る現行実装では、**壊れたエージェントが「弱いエージェント」に見える**。実測した挙動:

| agent 形態 | `debug=False` | `debug=True` |
|---|---|---|
| 例外を投げる（callable / `main.py` ファイル どちらも） | `status=DONE`、以降 `PASS` 扱いで完走 | **`env.run` 自体が例外を送出** |
| 2.5 秒かけて timeout | `status=DONE`、完走 | `status=DONE`、完走（**両モードで検出不能**） |

したがって失敗検出は 3 系統を併用する（これが本設計の中核）:

1. **クラッシュ** → `debug=True` で `env.run` を `try/except` 包囲。例外を掴んだら「その case は壊れている」と断定する。
2. **timeout** → `env.logs[step][player]["duration"]` が唯一の情報源。`actTimeout` の一定割合（既定 0.5 秒）超過を **警告**、1 秒超過を **失敗**として集計。
3. **silent no-op** → 不正 action は無警告で捨てられる。行動を engine に渡す前に自前で検証する（§3 `ActionValidator`）。

`status != "DONE"` の判定も残すが、**主たる失敗検出としては当てにしない**。

## 1. モジュール構成

```
backend/src/simulate/
  __init__.py        公開 API の再エクスポート
  __main__.py        CLI (typer)。python -m simulate
  config.py          MatchSpec / SimConfig — 実行したい評価の宣言
  episode.py         1 エピソード実行 + 失敗検出（debug 包囲・duration 監視）
  matrix.py          case × opponent × seed の展開と並列実行
  report.py          集計・表描画・JSON 出力
  validate.py        ActionValidator — silent no-op を落とす前に捕まえる
  agents.py          エージェント解決（builtin / case / ファイルパス）
  runner.py          互換ラッパ（run_episode / run_match を維持）
```

`runner.py` の `run_episode` / `run_match` / `EpisodeResult` / `MatchSummary` は**シグネチャを保ったまま**新実装への委譲にする。既存の呼び出し元（`dev/simulate`、将来の `dataset/`）を壊さない。

## 2. データモデル

```py
@dataclass(frozen=True)
class MatchSpec:
    """「何を何と何 seed で戦わせるか」の 1 セル。"""
    case: str                  # "rulebase/case1" | "starter" | パス
    opponent: str
    seed: int
    steps: int = 720
    swap_sides: bool = False   # 先手/後手の有利を打ち消す

@dataclass(frozen=True)
class TurnCost:
    step: int
    player: int
    duration: float

@dataclass(frozen=True)
class EpisodeOutcome:
    spec: MatchSpec
    rewards: tuple[float, float]     # 常に (自分, 相手) に正規化（swap 後も）
    statuses: tuple[str, str]
    steps: int
    crashed: bool                    # debug=True で例外を掴んだ
    crash_repr: str | None
    timeouts: tuple[TurnCost, ...]   # duration > actTimeout
    slow_turns: tuple[TurnCost, ...] # duration > warn 閾値
    max_duration: float
    invalid_actions: int             # ActionValidator の検出数
    replay_path: Path | None

    @property
    def failed(self) -> bool:
        return self.crashed or bool(self.timeouts) or any(s != "DONE" for s in self.statuses)
```

`rewards` を必ず「自分視点」に正規化するのが要点。`swap_sides` で席を入れ替えても集計側が向きを気にしなくて済む。

## 3. `ActionValidator` — silent no-op を可視化する

engine は不正 action を無警告で捨てるので、**評価時だけ**エージェントの出力を覆って検証する。提出コードには一切入らない（`backend/src/**` は tarball に含まれない）。

検証項目（engine の実装に対応させる）:

| 検証 | 根拠 |
|---|---|
| `farmer` がちょうど 1 op / `hands` が hand 数と一致 | 数が合わないと該当ユニットが `PASS` に落ちる |
| op 名が既知集合に含まれる | 未知 op は no-op |
| `market` の要素数 ≤ `maxMarketOrdersPerTurn`（既定 10） | 11 件目以降は無警告で破棄 |
| 移動先が盤内 / タイル操作が locked quadrant でない | locked 上の操作は全 no-op |
| `PICKUP` / `PLACE` 時に shed 隣接 4 タイル上にいる | 非隣接は no-op |
| 同一 turn の `PLANT` 要求数 ≤ 保有 seed | engine は**その作物の PLANT を全部** drop する（atomic 検証） |
| 資金不足の購入 | no-op |

検出は `EpisodeOutcome.invalid_actions` に積み、レポートで内訳を出す。`--strict` では 1 件でも失敗扱いにする（CI 用）。

> これは「エージェントが意図した行動」と「engine が実際に適用した行動」の乖離を測る仕組み。§0 の通りバグはスコアにしか現れないため、この層が実質的な型検査になる。

## 4. リプレイの扱い

1 ep 4.8 MB なので**既定では保存しない**。`--replay` の 3 モード:

| モード | 挙動 |
|---|---|
| `none`（既定） | 保存しない |
| `failed` | `failed` な ep のみ保存（デバッグ用途で最も有用） |
| `all` | 全保存。ep 数 × 5 MB を承知の上で |

出力先は `data/output/simulate/{yyyymmdd_hhmmss}/`（`docs/` ではなく `data/output/`。`rules/docs.md` の規約通り）。集計 JSON `summary.json` も同じディレクトリに置く。

## 5. 並列実行

- `ProcessPoolExecutor`（spawn）。既定 `workers = min(8, cpu_count - 1)`。
- **ワーカ内で `env` を作り、`EpisodeOutcome`（軽量 dataclass）だけを返す。** `env` 自体は返さない — 5 MB の pickle が並列度ぶん飛ぶのを避ける。
- `--workers 1` で逐次実行に落とせる（`pdb` を使うデバッグ時）。
- seed は `seed_base + i` で決定論的に割り当て、`summary.json` に記録して再現可能にする。

## 6. CLI

```bash
# 単発（現行互換）
dev/simulate --case rulebase/case1 --opponent starter

# 評価マトリクス（本設計の主用途）
dev/simulate --case rulebase/case1 \
             --opponent starter,random,pass \
             --episodes 50 --workers 8 --swap-sides \
             --replay failed --json summary.json

# case 同士の対戦（リグレッション比較）
dev/simulate --case rulebase/case2 --opponent rulebase/case1 --episodes 100
```

`--opponent` をカンマ区切りで複数受け、`case × opponent × episodes` を展開する。

### 出力

```
rulebase/case1   (50 ep x 720 steps, swap-sides, 8 workers, 41.2s)

opponent      W/L/T        win%    mean money   mean margin   worst
starter       31/18/1      62.0%       4,182         +271     -1,850
random        50/0/0      100.0%       4,205       +4,001       +822
pass          50/0/0      100.0%       4,198       +4,198     +3,010

health
  crashes         : 0
  timeouts        : 0        (max turn 0.031s / limit 1.000s)
  slow turns >0.5s: 0
  invalid actions : 0
  status != DONE  : 0
```

`health` ブロックを常に出すのが要点 — §0 の失敗が黙って通り過ぎないようにする。

### exit code

| code | 条件 |
|---|---|
| 0 | 全 ep 完走、失敗なし |
| 1 | crash / timeout / `status != DONE` が 1 件以上 |
| 2 | `--strict` かつ invalid action が 1 件以上 |

これで `dev/simulate` を CI とプリサブミット検証にそのまま使える。

## 7. 最小 case: `rulebase/case1`

評価基盤の動作確認には**動く被験体**が必要なので同時に作る。目標は勝つことではなく `starter` に対して**壊れずに完走する**こと。

```
backend/pipeline/rulebase/case1/
  main.py        agent(obs) -> dict。単一ファイル、標準ライブラリのみ
```

小麦ループ（`abstract.md` の罠を踏まないことを優先）:

1. NW quadrant のみ使う（land 購入なし）
2. `BUY_SEED WHEAT` → `PLANT` → **植えた当日に `WATER`**（`consecutive_unwatered` は植付時点で 1 から始まるため猶予なし）
3. 2 日目以降 `HARVEST` → shed へ `PLACE`
4. 売却は **1 turn あたり少量に分散**。wheat の上側は `log / 0.20` で供給過多に鈍感なので比較的安全な売り先
5. 最終日は在庫を売り切る（未売却在庫は 0 点）

`main.py` は全体を `try/except` で包み、例外時は `{"farmer": ["PASS"], "hands": [], "market": []}` を返す。

## 8. テスト

`backend/tests/unit/src/simulate/` に配置（既存の空 `__init__.py` がある）。

| テスト | 内容 |
|---|---|
| `test_validate.py` | `ActionValidator` — 市場注文 11 件目、未知 op、shed 非隣接 `PICKUP`、seed 超過 `PLANT` を検出すること |
| `test_episode.py` | わざとクラッシュ / 2.5 秒 sleep するエージェントを渡し、`crashed` / `timeouts` が立つこと（§0 の実測を回帰テストに固定） |
| `test_matrix.py` | spec 展開と `swap_sides` の視点正規化。`--workers 1` で決定論 |
| `test_report.py` | 集計と exit code の対応 |
| `test_agents.py` | builtin / case / パス の解決と、未知 spec でのエラー |

`steps` を小さく（20〜60）した短縮エピソードで回して高速に保つ。フル 720 turn を通すのは e2e 1 本だけに留める。

## 9. 実装順序

| # | 作業 | 検証 | 状態 |
|---|---|---|---|
| 1 | `config.py` / `agents.py` | `test_agents.py` / `test_config.py` | 完了 |
| 2 | `episode.py`（失敗検出 3 系統） | `test_episode.py` — 本設計の要 | 完了 |
| 3 | `validate.py` | `test_validate.py` | 完了 |
| 4 | `matrix.py` 並列実行 | `test_matrix.py` / `test_matrix_run.py` | 完了 |
| 5 | `report.py` + `__main__.py` | `test_report.py` / `test_cli.py` | 完了 |
| 6 | `runner.py` を委譲に置換 | `test_runner_compat.py` | 完了 |
| 7 | `rulebase/case1/main.py` | `test_case1.py` / `dev/submit --dry-run` | 完了 |

各段で `dev/lint` / `dev/test` を通す。

## 11. 実装結果

**テスト 113 件 / ruff・mypy strict クリーン。** テスト配置は `rules/backend/tests.md`
に従い、実エピソードを走らせるものは `e2e/`、CLI happy path は `integration/`、
純ロジックは `unit/` に置いた。

### 実装中に見つかった engine の追加事実

| 発見 | 影響 |
|---|---|
| `Agent.act()` は **`(action, log)` タプル**を返す（builtin は action 単体） | validator が最初これでタプルを検証し、`action_not_dict` の**誤検出 600 件**を出した。`build_agent` を直接使い、`co_argcount` で arity を判定して解決 |
| builtin `random_agent` は `random.Random()` を**引数なし**で生成 | env の `seed` では制御されない。再現性テストは決定論エージェント同士でしか成立しない |
| 市場注文 `SELL`/`BUY_*` は **3 要素必須**（`["SELL","WHEAT"]` は破棄） | validator の `order_missing_quantity` として検出 |
| ユニット inventory は日末に**自動で shed へ流れる** | 収穫後に `PLACE` で運ぶ必要はない。`SELL` は shed から引くので売却は翌日以降 |

### `rulebase/case1` の評価（120 ep, swap-sides, `--strict`）

```
opponent      W/L/T        win%    mean money   mean margin
starter       40/0/0     100.0%       9,033        +5,546
random        40/0/0     100.0%       9,381        +9,377
pass          40/0/0     100.0%       9,896        +6,896

health: crashes 0 / timeouts 0 / slow 0 / invalid actions 0
        (max turn 0.038s / limit 1.000s)
```

`dev/submit --case rulebase/case1 --dry-run` も通過（tarball 3.8 KB, `main.py` 単一）。

### 評価基盤が実際に機能した例

case1 の初版は**ユニットが移動せず 1 タイルしか耕さない**実装だった。
シミュレータは勝率 66.7% / mean money 3,557 と報告し、移動ロジック追加後は
**100% / 9,033** になった。`invalid actions` カウンタは前述の validator バグ
（誤検出 600 件）を即座に露出させており、§0 で設計した 3 系統の検出が
狙いどおり「スコアにしか現れないはずのバグ」を可視化している。

### 既知の制約

- **`max_duration` は wall-clock** なので、pytest の並列ワーカーが CPU を食い合うと
  値が膨らむ。case1 の budget テストは `act_timeout / 2` という緩い境界にしてあり、
  厳密な判定は `timeouts` カウンタ側で行う。
- `validate.py` のカバレッジは 84%。未到達は主に防御的な型チェック分岐。

## 10. 意図的にやらないこと

- **独自の高速 forward sim** — 1 ep 1.3 秒なら評価用途では不要。エージェント内の先読みに必要になった時点で別レイヤとして起こす（engine の state は共有 dict を in-place 変更するので、その際は deep copy が必須という調査結果だけ記録しておく）。
- **リプレイ解析ツール** — 本設計は集計まで。turn 単位の追跡は `--replay failed` で落としたリプレイを別途食わせる想定。
- **Kaggle publicScore との突き合わせ** — ローカル評価はローカル評価として扱う。
