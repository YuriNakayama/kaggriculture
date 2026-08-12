/**
 * おまかせ農場モード: 農作業 (farmer + hands) を内蔵 starter 方針に委譲し、
 * ユーザーは市場注文だけを操作する。このコンペの本質である市場ゲームに
 * 集中するための自由度制限モード。
 */

import { starterAgent } from '../ai/starter';
import type { Observation } from '../ai/types';
import type { GameState, MarketOrder, UnitAction } from '../engine/types';

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

/**
 * starter 方針の農作業 op と、生産に必要な購入系注文 (BUY_* / HIRE)。
 * SELL は含めない — 販売判断こそユーザーの仕事、が本モードの趣旨。
 * 失敗時は全 PASS。
 */
export function omakaseAction(
  state: GameState,
  player: number
): { farmer: UnitAction; hands: UnitAction[]; autoMarket: MarketOrder[] } {
  try {
    const a = starterAgent(buildObservation(state, player));
    const numHands = state.farms[player].hands.length;
    const hands = (a.hands ?? []).slice(0, numHands);
    while (hands.length < numHands) hands.push(['PASS']);
    const autoMarket = (a.market ?? []).filter((o) => Array.isArray(o) && o[0] !== 'SELL');
    return { farmer: a.farmer ?? ['PASS'], hands, autoMarket };
  } catch {
    return {
      farmer: ['PASS'],
      hands: state.farms[player].hands.map(() => ['PASS'] as UnitAction),
      autoMarket: [],
    };
  }
}

/** ユーザー注文を優先しつつ自動購入を合流 (エンジン上限 10 件)。 */
export function mergeMarket(user: MarketOrder[], auto: MarketOrder[]): MarketOrder[] {
  return [...user, ...auto].slice(0, 10);
}
