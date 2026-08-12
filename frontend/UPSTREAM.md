# Upstream provenance

The `playable/`, `default/`, `web/` directories and `kaggriculture.json` are
vendored (copied) from the Apache-2.0 licensed upstream repository:

- Repository: https://github.com/Kaggle/kaggle-environments
- Commit: `bded87b0d7879078c726a93a4884d044f79c4eed` (master, retrieved 2026-08-11)
- Source paths:
  - `kaggle_environments/envs/kaggriculture/visualizer/playable/` → `playable/`
  - `kaggle_environments/envs/kaggriculture/visualizer/default/` → `default/`
  - `web/{core/, vite.config.base.ts, tsconfig.base.json}` → `web/`
  - `kaggle_environments/envs/kaggriculture/kaggriculture.json` → `kaggriculture.json`
  - `LICENSE` → `LICENSE.upstream`

Why vendored: the PyPI wheel (`kaggle-environments==1.32.6`) ships the playable
visualizer without its `gameWorker-*.js` chunk, so it cannot run as packaged.
Building from source fixes that and lets us extend the UI (Pyodide agents,
replay browser, spectate mode).

## Modified files (keep this list current)

| File | Change |
|------|--------|
| `playable/vite.config.ts` | monorepo path `../../../../../web` → `../web` |
| `playable/tsconfig.json` | same path rewrite |
| `default/vite.config.ts` | same path rewrite |
| `default/tsconfig.json` | same path rewrite |
| `playable/src/engine/constants.ts` | spec import `../../../../kaggriculture.json` → `../../../kaggriculture.json` |
| `playable/src/engine/__tests__/market.spec.ts` | MELON@10100 expectation 225 → 150 (matches authoritative Python `market_price`; upstream test was stale) |
| `default/replays/` | removed (28 MB sample replay; use real replays from `data/` instead) |
| `playable/src/worker/protocol.ts` | added `py` slot kind, `PROGRESS` message, `agentErrors` on STATE |
| `playable/src/worker/gameWorker.ts` | Pyodide agent loading on INIT, py-slot action collection with error surfacing |
| `playable/src/worker/workerClient.ts` | progress callback + `lastAgentErrors` |
| `playable/src/ui/useGameWorker.ts` | expose `progress` / `agentErrors` |
| `playable/src/ui/SetupScreen.tsx` | Python-case dropdown fed by `agents/manifest.json` |
| `playable/src/ui/GameScreen.tsx` | loading progress display, agent-error banner, py slot labels |
| `playable/src/ui/GameOverModal.tsx` | py slot labels |
| `playable/package.json` | added `pyodide` devDependency (node-side bridge tests) |
| `playable/src/engine/rng.ts` | rewritten to be bit-compatible with CPython `random.Random` (init_by_array seeding, getrandbits-based `choice`); upstream version diverged from Python rollouts by design |
| `playable/src/ui/ActionPanel.tsx` | legal-op graying, cost/stock labels, silent-no-op notes |
| `playable/src/App.tsx`, `playable/src/ui/SetupScreen.tsx` | Replays mode, session resume |
| `playable/src/style.css` | responsive/mobile rules, replay browser + spectate styles |

Files added by this repo under `playable/src/`: `pyagent/`, `engine/legality.ts`,
`ui/ReplayScreen.tsx`, `engine/__tests__/{legality,parity,rng}.spec.ts`,
`pyagent/__tests__/`.

Files added by this repo (not upstream): `package.json`, `pnpm-workspace.yaml`,
`UPSTREAM.md`, plus everything under `playable/src/pyagent/` and
`scripts/` once added.
