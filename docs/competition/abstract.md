# Kaggriculture コンペ概要

> 公式資料の日本語要約。原文は [`README.md`](README.md)（ゲームルール全文）と [`AGENTS.md`](AGENTS.md)（提出手順）を参照。
> 数値の一次情報は常に原文が正。本ファイルは方針判断のための要約。

| 項目 | 内容 |
|---|---|
| コンペ | [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) |
| 種別 | Featured / Simulation（2 人対戦） |
| 賞金 | $50,000 |
| 締切 | 2026-09-30 23:59 UTC |
| エンジン | `kaggle-environments` の `make("kaggriculture")` |

## ゲームの骨子

2 人の農場経営シム。**720 ターン（24 ターン/日 × 30 日）**のシーズンを通して、種・家畜の購入、植付、水やり、収穫、家畜の世話、動的市場での売買を行う。**シーズン終了時点の所持金が多い方が勝ち**（同点あり）。**未売却の在庫は加算されない** — 売り切るまでが仕事。

- 開始資金 `startingMoney = 3000`
- 農場は `boardSize × boardSize`（既定 10×10）を 4 つの 5×5 quadrant に分割。開始時は NW のみ解放。`BUY_LAND` で NE / SW / SE を **$1k / $2k / $4k** で購入
- 各プレイヤーは farmer 1 体 + その日雇った farm hand。**各ユニットが毎ターン独立に 1 行動**
- 雇用コストは `farmHandCostMult * fib(n)`（n = その日の既雇用数）→ 1, 1, 2, 3, 5, 8, 13, ...。**日ごとにリセット**され、hand は日末に消える

## 作物・家畜

| 種類 | 種コスト | 基準価格 | 初収穫 | 最大収量到達 | 継続産出 | 収量/タイル/日 |
|---|---|---|---|---|---|---|
| Wheat | 10 | 25 | 2 日 | 4 日 | なし | 0.80 |
| Carrot | 20 | 35 | 2 日 | 3 日 | なし | 0.75 |
| Tomato | 50 | 60 | 8 日 | 11 日 | 毎日 ×4 | 0.33 |
| Strawberry | 100 | 120 | 10 日 | 16 日 | 隔日 ×4 | 0.24 |
| Melon | 80 | 250 | 10 日 | 10 日 | なし | 0.55 |
| Goose/Egg | 300 | 50 | 4 日 | — | 毎日・無期限 | 1.00 |
| Cow/Milk | 400 | 160 | 8 日 | — | 2 日毎・無期限 | 0.50 |
| Sheep/Wool | 500 | 200 | 6 日 | — | 3 日毎・無期限 | 0.33 |
| Fertilizer | 100 | — | — | — | — | — |

**注意点（罠になりやすい箇所）**:

- **植えた日が「水やり漏れ 1 日目」としてカウントされる**。`consecutive_unwatered` は種を植えた時点で 1 から始まり、その日に水をやらなければ日末に 2 に達してその夜に雑草化する。新規植付に猶予はない
- 一方、**新規配置の家畜は `consecutive_unfed = 0` から始まる**ので初日は餌なしでも生存する
- Wheat / Carrot が表の Max Yield（6 / 4）に届くのは**肥料ありの場合のみ**。水やりだけでは 4 / 3 止まり
- Tomato / Strawberry は「継続産出」だが**無期限ではない** — 産出 4 回で打ち止め、その後は雑草化する
- Melon の bonus window は 6〜12 日目だが、水やりのみだと 10 日目で上限 6 に到達し 11〜12 日目は無意味
- 家畜の餌は **wheat**。市場から `BUY_PRODUCT` でも買える
- `COLLECT_FERTILIZER` は生存中の全家畜が毎日 1 個産出。**蓄積しない**ので毎日回収しないと消える
- `CARE` は「餌あり かつ 世話あり」の日に `pending_care_bonus` を +1 貯め、次回の産出時に一括で上乗せされる

## 行動

```py
{
  "farmer": [op, ...args],          # メイン farmer の 1 行動
  "hands":  [[op, ...args], ...],   # hand ごとに 1 行動（hands の順序どおり）
  "market": [[op, ...args], ...],   # 市場注文の順序付きリスト
}
```

- **移動**: `NORTH` / `SOUTH` / `EAST` / `WEST` / `PASS`。**locked タイルは通過可能**だが、その上でのタイル操作は全て no-op
- **shed 操作**: `PICKUP <item> [n]` / `PLACE <item> [n]` / `DROP`。shed は `tiles` に現れない中央の存在で、「隣接」とは中央 4 タイル `(4,4)` `(5,4)` `(4,5)` `(5,5)` のいずれかに立つこと
- **作物**: `PLANT <crop>` / `WATER` / `HARVEST` / `FERTILIZE`
- **家畜**: `BUILD_COOP` / `BUILD_PASTURE` / `FEED` / `COLLECT_FERTILIZER` / `CARE`
- **地形**: `DIG`（作物・雑草・**空の**coop/pasture を撤去。家畜が乗っていると no-op）
- **市場**: `BUY_SEED` / `BUY_PRODUCT` / `BUY_ANIMAL` / `SELL` / `HIRE` / `BUY_LAND`

> **市場注文は 1 ターンあたり `maxMarketOrdersPerTurn`（既定 10）まで。超過分は警告なく破棄される。**
> 不正な行動も全て silent no-op なので、**エラーが表に出ない**。ローカル検証を厚くする必要がある。

## 市場メカニクス（このコンペの肝）

全商品が市場在庫 `I0 = 10,000` から開始。価格は在庫量の関数:

```
price(inv) = base + sign · amp · f(|inv − I0|)
  sign = +1  if inv < I0   (品薄 → 高騰)
  sign = −1  if inv > I0   (供給過多 → 下落)
  f    ∈ { linear, sq, sqrt, log, log10 }
```

**`I0` の上下で異なる形状関数・異なる感度**を持つため、資源ごとに戦略性が大きく異なる:

| 資源 | Base | T | 下側 func/target | 上側 func/target | P(I0−T) | P(I0+T) | P(I0+2T) |
|---|---|---|---|---|---|---|---|
| Wheat | 25 | 400 | sqrt / 0.80 | log / 0.20 | $45 | $20 | $19 |
| Carrot | 35 | 450 | log / 0.20 | sqrt / 0.70 | $42 | $10 | $1 |
| Tomato | 60 | 200 | linear / 0.40 | sqrt / 0.60 | $84 | $24 | $9 |
| Strawberry | 120 | 100 | sqrt / 0.70 | linear / 1.60 | $204 | $1 | $1 |
| Melon | 250 | 300 | log / 0.20 | sq / 3.60 | $300 | $1 | $1 |
| Egg | 50 | 332 | linear / 0.40 | log / 0.20 | $70 | $40 | $39 |
| Milk | 160 | 122 | sqrt / 0.60 | linear / 1.60 | $256 | $1 | $1 |
| Wool | 200 | 105 | log / 0.20 | sq / 3.20 | $240 | $1 | $1 |
| Fertilizer | 100 | 200 | linear / 0.40 | linear / 0.40 | $140 | $60 | $20 |

**戦略上の含意**:

- **高額品（strawberry / melon / milk / wool）は `above_target > 1`**。少し売り込むだけで $1 のフロアまで暴落する。**売却タイミングと分散が損益を支配する**
- Wheat は品薄で高騰しやすく供給過多には鈍感 → 安定した売り先。Carrot は逆
- 売買は**両プレイヤー同時に 1 単位ずつ**処理される。相手の売り注文が自分の売値を削る
- 買値は「買った後の在庫」で、売値は「売る前の在庫」で見積もられるので、**同一商品を即座に買って売ると損益ゼロ**（裁定不可）
- 市場から買い戻せるのは **wheat と fertilizer のみ**。売却は全商品可能
- 価格は $1 が下限。フロアで売った分は市場在庫に加算されない

## 町の需要

- **町の中心**は `townCenterSellInterval`（既定 12 ターン）ごとに fertilizer 以外の全商品を 1 個消費。**11 日目以降 2 倍、21 日目以降 4 倍**
- **ショップ**は `townShopUnlockInterval`（既定 3 日）ごとにランダムに 1 軒解放され、以後シーズン終了まで残る。各店は需要商品を `townShopSellInterval`（既定 4 ターン）ごとに 1 個消費（単一商品の店は 2 倍）
- 需要は単調増加する → **シーズン後半ほど売り圧を吸収できる**

| 店 | 需要 |
|---|---|
| Bakery | egg, wheat |
| Pizza Shop | milk, tomato, wheat |
| Brunch Spot | egg, wheat, strawberry |
| Yarn Store | wool (2x) |
| Ice Cream Shop | strawberry, milk, wheat |
| Pet Cafe | carrot (2x) |
| Smoothie Shop | strawberry, milk |
| Farmers Market | wheat, carrot, tomato, strawberry |

## 観測フォーマット

```py
{
  "player": int,           # 0 or 1
  "step":   int,           # 通算ターン（framework 供給）
  "day":    int,           # 0-indexed
  "hour":   int,           # 0-indexed, 0..turnsPerDay-1
  "farms":  [farm, farm],  # 両者の農場は公開
  "market": {"inventory": {...}, "prices": {...}},   # 共有
  "town":   {"unlocked_shops": [...]},               # 共有
  "private": {             # 自分のみ。相手の private は見えない
    "shed":        {item: count},
    "seeds":       {crop: count},
    "inventories": [farmer_inv, hand_inv, ...],
  },
}
```

`farm` は `money` / `tiles[y][x]` / `farmer [x,y]` / `hands [[x,y],...]` / `unlocked_quadrants` / `hires_today`。
`tile` は `None`（空き）/ `"LOCKED"` / plant dict / weed dict / animal structure dict のいずれか。

**不完全情報の範囲**: 相手の農場・所持金・タイル状態は**見える**。見えないのは相手の shed・種・ユニット inventory のみ。市場は完全に共有。

## 提出

- `main.py` を root に置き、その中に `agent` 関数を定義
- 単一ファイル: `kaggle competitions submit kaggriculture -f main.py -m "..."`
- 複数ファイル: `main.py` を root に据えた `tar.gz` を提出

## ローカル検証

```py
from kaggle_environments import make

env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run([agent, "random"])   # env.run(["main.py", "random"]) でファイル指定も可
```

組込エージェント: `"pass"` / `"random"` / `"starter"`（決定論的ベースライン）。

## 設定可能パラメータ

| パラメータ | 既定 | 説明 |
|---|---|---|
| `episodeSteps` | 720 | シーズン総ターン数 |
| `boardSize` | 10 | 農場の一辺 |
| `startingMoney` | 3000 | 開始資金 |
| `maxMarketOrdersPerTurn` | 10 | 1 ターンの市場注文上限（超過は破棄） |
| `turnsPerDay` | 24 | 1 日のターン数 |
| `shedCapacity` | 100 | shed の非種アイテム上限（溢れた分は破棄） |
| `weedSpawnChance` | 0.005 | 空きタイルの雑草発生率/日 |
| `townShopUnlockInterval` | 3 | ショップ解放間隔（日） |
| `townShopSellInterval` | 4 | ショップ消費間隔（ターン） |
| `townCenterSellInterval` | 12 | 町中心の消費間隔（ターン） |
| `seed` | null | 決定論的生成用シード |

種コストと基準価格は設定不可（上表のとおり固定）。市場パラメータは `env.configuration["marketParams"]` で資源ごとに部分上書き可能。
