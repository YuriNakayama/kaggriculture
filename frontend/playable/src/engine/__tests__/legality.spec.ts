import { describe, expect, it } from 'vitest';
import { auditAction, legalMarket, legalUnitOps } from '../legality';
import { initGameState, resolveConfig } from '../state';

const config = resolveConfig({ seed: 1 });

describe('legalUnitOps', () => {
  it('initial state: moves ok, tile ops on empty unlocked tile', () => {
    const s = initGameState(2, config, 1);
    const legal = legalUnitOps(s, 0, 0);
    expect(legal.PASS).toBe(true);
    expect(legal.BUILD_COOP).toBe(true); // empty tile
    expect(legal.WATER).toBe(false); // no plant
    expect(legal.HARVEST).toBe(false);
    expect(legal.DIG).toBe(false); // empty tile: nothing to dig
    expect(legal.FEED).toBe(false); // no animal
    expect(legal.PLANT).toBe(false); // no seeds at game start
  });

  it('PLANT becomes legal once a seed is in stock', () => {
    const s = initGameState(2, config, 1);
    s.privates[0].seeds.WHEAT = 1;
    expect(legalUnitOps(s, 0, 0).PLANT).toBe(true);
  });
});

describe('legalMarket', () => {
  it('initial state: seeds affordable, selling impossible, land depends on money', () => {
    const s = initGameState(2, config, 1);
    const m = legalMarket(s, 0);
    expect(m.buySeed.WHEAT?.legal).toBe(true);
    expect(Object.values(m.sell).every((x) => !x?.legal)).toBe(true);
    expect(m.hire.cost).toBeGreaterThan(0);
    expect(m.buyLand.quadrant).toBe('NE');
  });
});

describe('auditAction', () => {
  it('flags silently-discarded sub-actions', () => {
    const s = initGameState(2, config, 1);
    const notes = auditAction(s, 0, {
      farmer: ['WATER'], // no plant under farmer → no-op
      hands: [],
      market: [['SELL', 'MELON', 3]], // nothing in shed → no-op
    });
    expect(notes.length).toBe(2);
    expect(notes[0]).toContain('WATER');
    expect(notes[1]).toContain('SELL');
  });

  it('is quiet for a legal action', () => {
    const s = initGameState(2, config, 1);
    const notes = auditAction(s, 0, {
      farmer: ['PASS'],
      hands: [],
      market: [['BUY_SEED', 'WHEAT', 2]],
    });
    expect(notes).toEqual([]);
  });
});
