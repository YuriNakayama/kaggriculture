/**
 * Integration test for the Pyodide agent bridge: loads real cases from
 * public/agents (produced by scripts/collect-agents.mjs) into a node-side
 * Pyodide and drives them with observations from the TS engine.
 *
 * Verifies the harness-compatible exec (last callable, bare imports for
 * multi-module cases) and the JSON call bridge — the exact code the worker
 * runs, minus fetch/CDN.
 */

import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { loadPyodide, type PyodideInterface } from 'pyodide';
import { initGameState, resolveConfig } from '../../engine/state';
import { step } from '../../engine/interpreter';
import type { GameState, PlayerAction } from '../../engine/types';
import { LOADER_PY } from '../loader';

const AGENTS_DIR = join(__dirname, '..', '..', '..', 'public', 'agents');
const hasAgents = existsSync(join(AGENTS_DIR, 'manifest.json'));

function buildObservation(s: GameState, player: number) {
  return {
    player,
    step: s.step,
    day: s.day,
    hour: s.hour,
    numAgents: s.numAgents,
    farms: s.farms,
    private: s.privates[player],
    market: s.market,
    town: s.town,
  };
}

describe.skipIf(!hasAgents)('pyodide agent bridge', () => {
  let py: PyodideInterface;

  beforeAll(async () => {
    // Vitest's transform breaks pyodide's own path detection — point it at the
    // package dir explicitly.
    const require = createRequire(import.meta.url);
    py = await loadPyodide({ indexURL: dirname(require.resolve('pyodide')) });
    py.runPython(LOADER_PY);
  }, 120_000);

  function loadCase(id: string): void {
    const slug = id.replace('/', '__');
    const dir = join(AGENTS_DIR, slug);
    const caseDir = `/cases/${slug}`;
    py.FS.mkdirTree(caseDir);
    for (const f of readdirSync(dir)) {
      py.FS.writeFile(`${caseDir}/${f}`, readFileSync(join(dir, f), 'utf8'));
    }
    py.globals.get('_load_case')(id, caseDir);
  }

  function callCase(id: string, obs: unknown): { action: PlayerAction; error?: string } {
    const raw = py.globals.get('_call_agent')(id, JSON.stringify(obs)) as string;
    return JSON.parse(raw);
  }

  it('loads every collected case and gets a first-turn action', () => {
    const manifest = JSON.parse(readFileSync(join(AGENTS_DIR, 'manifest.json'), 'utf8')) as {
      id: string;
    }[];
    expect(manifest.length).toBeGreaterThan(0);
    const config = resolveConfig({ seed: 42 });
    const state = initGameState(2, config, 42);
    for (const { id } of manifest) {
      loadCase(id);
      const result = callCase(id, buildObservation(state, 1));
      expect(result.error, `${id}: ${result.error}`).toBeUndefined();
      expect(result.action).toHaveProperty('farmer');
      expect(Array.isArray(result.action.farmer)).toBe(true);
    }
  }, 60_000);

  it('drives case1 vs case2 through 24 engine turns without errors', () => {
    const config = resolveConfig({ seed: 7 });
    let state = initGameState(2, config, 7);
    for (let t = 0; t < 24; t++) {
      const a0 = callCase('rulebase/case1', buildObservation(state, 0));
      const a1 = callCase('rulebase/case2', buildObservation(state, 1));
      expect(a0.error, `case1 turn ${t}: ${a0.error}`).toBeUndefined();
      expect(a1.error, `case2 turn ${t}: ${a1.error}`).toBeUndefined();
      state = step(state, [a0.action, a1.action], config);
    }
    expect(state.step).toBe(24);
  }, 60_000);
});
