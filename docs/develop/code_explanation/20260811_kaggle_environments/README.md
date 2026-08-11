# kaggle-environments / kaggriculture 実装分析

`kaggle_environments` パッケージに含まれる Kaggriculture 環境の実装を、3 段階（ディレクトリ構成 → 責務と入出力 → 処理ステップ）で読み解いたもの。

**HTML で保存している**理由は、SVG 図 18 点が本文の主要な情報を担っているため。Markdown では再現できない。差分レビューの対象外として扱う（`.claude/rules/docs.md` の「実装解説の例外」を参照）。

## ページ構成

ブラウザで開くと相互リンクで行き来できる。

| | ファイル | 内容 |
|---|---|---|
| ① | [`1-environment.html`](1-environment.html) | 用語 / ファイル構成 / 状態の持ち方 / どの変数を誰が更新するか / 1 ターンの処理順 / 実装上の落とし穴 |
| ② | [`2-interface.html`](2-interface.html) | `obs` の全キー（実測値）/ 倉庫・手持ち・種の制約 / 農夫と雇い人の 18 命令 / 市場注文の 6 命令 |
| ③ | [`3-economy.html`](3-economy.html) | 価格計算のロジック / 9 商品の下落曲線 / 売却タイミングの実測 / 商品別の投資〜売却カード |

## 解析対象

- `kaggle_environments/envs/kaggriculture/kaggriculture.py`（1,073 行）— ルール本体
- `kaggle_environments/core.py`（772 行）— フレームワーク

本文中の行番号はこの 2 ファイル内の位置を指す。バージョンは `kaggle-environments` 1.32.6（`backend/pyproject.toml` で固定）。

## 数値の扱い

掲載している数値は**すべて実際に環境を実行して計測したもの**で、仕様書からの引き写しではない。主な計測項目:

- 作物・動物の成長曲線（日ごとの収穫可能量）
- 価格関数を再現した 9 商品の下落曲線と、床に到達する個数
- 売却の約定順序（両プレイヤーが同一価格・同額になることの確認）
- 分割売却の効果（同一ターン内では変化なし、ターンをまたぐと町の需要ぶん回復）
- 倉庫が満杯のときの経路別挙動（購入は部分約定、`DROP` は超過分が消滅）

エンジンのバージョンが上がった場合、これらは再計測が必要になる。

## 公開版

同じ内容を Artifact としても公開している（非公開設定）。

- ① https://claude.ai/code/artifact/de7c1587-35b1-4c34-9b12-76bba3009b3f
- ② https://claude.ai/code/artifact/b25cbed0-a90a-46bf-b78b-5aee49d44a92
- ③ https://claude.ai/code/artifact/4e85e5d7-e62a-4618-82e6-42afe104143d

リポジトリ内の HTML はページ間リンクを相対パスに書き換えてあるため、`git clone` しただけでオフラインで閲覧できる。両者を更新する場合は内容を同期させること。
