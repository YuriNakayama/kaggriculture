/// <reference lib="webworker" />
/**
 * Game worker: owns the authoritative GameState and runs `step()` on demand.
 * AI slots are evaluated here so the main thread never sees their cost.
 * Mirrors `orbit_wars/visualizer/playable/src/worker/gameWorker.ts`.
 */

import { AGENTS } from '../ai';
import type { Observation } from '../ai/types';
import { initGameState, pickSeed } from '../engine/state';
import { step } from '../engine/interpreter';
import type { Config, GameState, PlayerAction } from '../engine/types';
import { loadPyAgent, type PyAgentHandle } from '../pyagent/loader';
import type { HumanActions, Req, Res, SlotConfig } from './protocol';

let state: GameState | null = null;
let config: Config | null = null;
let slots: SlotConfig[] = [];
const pyAgents = new Map<number, PyAgentHandle>();

function buildObservation(s: GameState, player: number): Observation {
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

function collectActions(
  s: GameState,
  humanActions: HumanActions
): { actions: PlayerAction[]; agentErrors: Record<number, string> } {
  const actions: PlayerAction[] = [];
  const agentErrors: Record<number, string> = {};
  for (let pid = 0; pid < slots.length; pid++) {
    const slot = slots[pid];
    if (slot.kind === 'human') {
      actions.push(humanActions[pid] ?? { farmer: ['PASS'], hands: [], market: [] });
      continue;
    }
    if (slot.kind === 'py') {
      const handle = pyAgents.get(pid);
      if (!handle) {
        actions.push({ farmer: ['PASS'], hands: [], market: [] });
        agentErrors[pid] = `Python agent ${slot.caseId} not loaded`;
        continue;
      }
      const result = handle.call(buildObservation(s, pid));
      actions.push(result.action);
      if (result.error) agentErrors[pid] = result.error;
      continue;
    }
    const agent = AGENTS[slot.agentId];
    if (!agent) {
      actions.push({ farmer: ['PASS'], hands: [], market: [] });
      continue;
    }
    try {
      actions.push(agent.fn(buildObservation(s, pid)));
    } catch (err) {
      console.error('AI agent error', slot.agentId, err);
      actions.push({ farmer: ['PASS'], hands: [], market: [] });
      agentErrors[pid] = err instanceof Error ? err.message : String(err);
    }
  }
  return { actions, agentErrors };
}

function resolveSeed(cfg: Config): number {
  return cfg.seed ?? pickSeed();
}

function send(res: Res): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(res);
}

self.addEventListener('message', async (evt: MessageEvent<Req>) => {
  const msg = evt.data;
  try {
    switch (msg.type) {
      case 'INIT': {
        config = msg.config;
        slots = msg.slots;
        // Load Python agents (Pyodide) before the first state ships, so the
        // UI's INIT promise doubles as the loading gate. Progress messages
        // stream back for the loading indicator.
        pyAgents.clear();
        for (let pid = 0; pid < slots.length; pid++) {
          const slot = slots[pid];
          if (slot.kind !== 'py') continue;
          const handle = await loadPyAgent(slot.caseId, (message) =>
            send({ type: 'PROGRESS', reqId: msg.reqId, message })
          );
          pyAgents.set(pid, handle);
        }
        state = initGameState(msg.numAgents, msg.config, resolveSeed(msg.config));
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'RESET': {
        if (!config) throw new Error('Worker not initialized');
        state = initGameState(slots.length, config, resolveSeed(config));
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'STEP': {
        if (!state || !config) throw new Error('Worker not initialized');
        if (state.done) {
          send({ type: 'STATE', reqId: msg.reqId, state });
          break;
        }
        const { actions, agentErrors } = collectActions(state, msg.humanActions);
        state = step(state, actions, config);
        send({
          type: 'STATE',
          reqId: msg.reqId,
          state,
          ...(Object.keys(agentErrors).length ? { agentErrors } : {}),
        });
        break;
      }
      case 'GET_STATE': {
        if (!state) throw new Error('Worker not initialized');
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
    }
  } catch (err) {
    send({ type: 'ERROR', reqId: msg.reqId, message: err instanceof Error ? err.message : String(err) });
  }
});
