/**
 * おまかせ農場の回帰テスト: 購入系注文を通すことで実際に「種を買う→植える→
 * 水をやる」が回ることを実エンジンで検証する (市場注文を全破棄していた
 * 初期実装では farmer が永久 PASS になっていた)。
 */

import { describe, expect, it } from 'vitest';
import { step } from '../../engine/interpreter';
import { initGameState, resolveConfig } from '../../engine/state';
import type { PlayerAction } from '../../engine/types';
import { mergeMarket, omakaseAction } from '../omakase';

const config = resolveConfig({ seed: 9 });
const PASS: PlayerAction = { farmer: ['PASS'], hands: [], market: [] };

describe('omakase mode', () => {
  it('buys seeds and gets a plant into the ground within a day', () => {
    let s = initGameState(2, config, 9);
    let planted = false;
    for (let t = 0; t < 24 && !planted; t++) {
      const auto = omakaseAction(s, 0);
      const action: PlayerAction = { farmer: auto.farmer, hands: auto.hands, market: auto.autoMarket };
      s = step(s, [action, PASS], config);
      planted = s.farms[0].tiles.some((row) =>
        row.some((tile) => tile !== null && tile !== 'LOCKED' && (tile as { kind?: string }).kind === 'PLANT')
      );
    }
    expect(planted).toBe(true);
  });

  it('mergeMarket keeps user orders first and caps at 10', () => {
    const user = Array.from({ length: 6 }, (_, i) => ['SELL', 'WHEAT', i + 1]) as never[];
    const auto = Array.from({ length: 6 }, () => ['BUY_SEED', 'CARROT', 1]) as never[];
    const merged = mergeMarket(user, auto);
    expect(merged).toHaveLength(10);
    expect(merged[0]).toEqual(['SELL', 'WHEAT', 1]);
    expect(merged[9]).toEqual(['BUY_SEED', 'CARROT', 1]);
  });
});
