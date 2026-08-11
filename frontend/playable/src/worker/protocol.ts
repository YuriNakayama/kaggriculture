/**
 * Postmessage wire protocol between the main thread and the game worker.
 * Mirrors the orbit_wars shape — INIT once, then STEP per turn (passing in
 * only the human players' actions; the worker fills in AI actions itself).
 */

import type { Config, GameState, PlayerAction } from '../engine/types';

export type SlotConfig =
  | { kind: 'human' }
  | { kind: 'ai'; agentId: string }
  // Python agent from backend/pipeline (e.g. "rulebase/case2"), run via Pyodide.
  | { kind: 'py'; caseId: string };

/** Sparse map: player id → action. Missing entries fall through to AI or PASS. */
export type HumanActions = Record<number, PlayerAction>;

export type Req =
  | { type: 'INIT'; reqId: string; config: Config; numAgents: number; slots: SlotConfig[] }
  | { type: 'STEP'; reqId: string; humanActions: HumanActions }
  | { type: 'RESET'; reqId: string }
  | { type: 'GET_STATE'; reqId: string };

export type Res =
  | { type: 'STATE'; reqId: string; state: GameState; agentErrors?: Record<number, string> }
  | { type: 'ERROR'; reqId: string; message: string }
  // Progress while Pyodide / a Python case is loading during INIT.
  | { type: 'PROGRESS'; reqId: string; message: string };
