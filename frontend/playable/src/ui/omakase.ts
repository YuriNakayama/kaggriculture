/**
 * おまかせ農場モード: 農作業 (farmer + hands) を内蔵 starter 方針に委譲し、
 * ユーザーは市場注文だけを操作する。このコンペの本質である市場ゲームに
 * 集中するための自由度制限モード。
 */

import { starterAgent } from '../ai/starter';
import type { Observation } from '../ai/types';
import type { GameState, UnitAction } from '../engine/types';

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

/** starter 方針の農作業 op (市場注文は破棄)。失敗時は全 PASS。 */
export function omakaseFarmOps(state: GameState, player: number): { farmer: UnitAction; hands: UnitAction[] } {
  try {
    const a = starterAgent(buildObservation(state, player));
    const numHands = state.farms[player].hands.length;
    const hands = (a.hands ?? []).slice(0, numHands);
    while (hands.length < numHands) hands.push(['PASS']);
    return { farmer: a.farmer ?? ['PASS'], hands };
  } catch {
    return { farmer: ['PASS'], hands: state.farms[player].hands.map(() => ['PASS'] as UnitAction) };
  }
}
