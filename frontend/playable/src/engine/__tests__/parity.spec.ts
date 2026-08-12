/**
 * Engine parity: replay a Python-engine reference trace through the TS port
 * and compare per-step observable state. Regenerate the trace with:
 *
 *   cd backend && uv run python ../frontend/scripts/gen_parity_trace.py \
 *       --steps 240 --seed 123 --out ../frontend/playable/parity-trace.json
 *
 * A failure here means the TS engine drifted from the authoritative Python
 * engine (upstream update or porting bug) — fix the port, don't relax the test.
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { step } from '../interpreter';
import { initGameState, resolveConfig } from '../state';
import type { PlayerAction } from '../types';

const TRACE_PATH = join(__dirname, '..', '..', '..', 'parity-trace.json');

interface TraceStep {
  t: number;
  actions: (PlayerAction | null)[] | null;
  day: number;
  hour: number;
  money: number[];
  farmer: [number, number][];
  num_hands: number[];
  market_inventory: Record<string, number>;
}

interface Trace {
  seed: number;
  episodeSteps: number;
  config: Record<string, unknown>;
  steps: TraceStep[];
  rewards: number[];
}

const PASS: PlayerAction = { farmer: ['PASS'], hands: [], market: [] };

describe.skipIf(!existsSync(TRACE_PATH))('python-engine parity', () => {
  it('replays the reference trace with identical state at every step', () => {
    const trace = JSON.parse(readFileSync(TRACE_PATH, 'utf8')) as Trace;
    const config = resolveConfig({ episodeSteps: trace.episodeSteps, seed: trace.seed });
    let s = initGameState(2, config, trace.seed);

    const check = (ref: TraceStep) => {
      const ctx = `step ${ref.t}`;
      expect(s.day, `${ctx} day`).toBe(ref.day);
      expect(s.hour, `${ctx} hour`).toBe(ref.hour);
      for (let p = 0; p < 2; p++) {
        expect(s.farms[p].money, `${ctx} p${p} money`).toBe(ref.money[p]);
        expect(s.farms[p].farmer, `${ctx} p${p} farmer pos`).toEqual(ref.farmer[p]);
        expect(s.farms[p].hands.length, `${ctx} p${p} hands`).toBe(ref.num_hands[p]);
      }
      expect(s.market.inventory, `${ctx} market inventory`).toEqual(ref.market_inventory);
    };

    check(trace.steps[0]);
    for (const ref of trace.steps.slice(1)) {
      const actions = (ref.actions ?? []).map((a) => a ?? PASS);
      s = step(s, actions, config);
      check(ref);
    }
    expect(s.done).toBe(true);
    expect(s.scores).toEqual(trace.rewards);
  });
});
