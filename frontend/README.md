# Playable Frontend

ブラウザで Kaggriculture を遊ぶ・観る・調べるための静的アプリ。公式ビジュアライザ
([Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments) の
Apache-2.0 ソース、詳細は [`UPSTREAM.md`](UPSTREAM.md)) をフォークし、以下を追加している:

- **人間 vs AI 対戦** — 対戦相手は TS 移植 bot (starter / random) と、
  `backend/pipeline/` の Python エージェント (Pyodide でブラウザ内実行、無改変)
- **合法手ハイライト** — エンジンが黙って捨てる操作 (silent no-op) をグレーアウト+事後通知
- **リプレイビューア** — `dev/simulate --replay` や Kaggle スクレイプ品の JSON を D&D 再生
- **AI vs AI 観戦** — 速度スライダー付き自動進行
- **セッション自動保存** — localStorage、リロード/タブ落ちから Resume 可能

## 使い方

```bash
dev/play            # 開発サーバ (http://localhost:5173, --host で LAN 公開)
dev/play --build    # 本番ビルド (frontend/playable/dist)
dev/play --test     # フロントエンドテスト一式
```

新しい case を `backend/pipeline/<family>/<caseN>/` に追加すると、次回の
`dev/play` / ビルドで自動的に対戦相手リストへ載る (`scripts/collect-agents.mjs`)。
条件は純 Python + numpy のみ (Pyodide 制約)。

## エンジンパリティ

ゲーム進行はブラウザ内の TS 移植エンジンが担う。本家 Python エンジンとの一致は
`playable/src/engine/__tests__/parity.spec.ts` が担保する — Python エンジンで
720 ステップのリファレンストレースを生成し、同じ行動列を TS エンジンで再生して
全ステップの状態を突き合わせる。乱数 (雑草・店舗解禁) も CPython 互換 RNG
(`engine/rng.ts`) により一致する。

トレース再生成 (kaggle-environments 更新時は CI が自動実行):

```bash
cd backend && uv run python ../frontend/scripts/gen_parity_trace.py \
    --steps 720 --seed 123 --out ../frontend/playable/parity-trace.json
```

## デプロイ

Amplify Hosting (`amplify.yml` + `infra/module/application/amplify_hosting/`) で
`https://farm.avifauna.click` に配信。main への push で自動ビルド。
