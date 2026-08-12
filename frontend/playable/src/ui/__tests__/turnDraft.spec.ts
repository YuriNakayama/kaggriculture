import { describe, expect, it } from 'vitest';
import { defaultUnit, fromUnitAction, toMarketOrder, toUnitAction, type UnitDraft } from '../useTurnDraft';

describe('draft ⇄ action round-trip', () => {
  it('toUnitAction handles argument ops', () => {
    expect(toUnitAction({ ...defaultUnit, op: 'PLANT', crop: 'MELON' })).toEqual(['PLANT', 'MELON']);
    expect(toUnitAction({ ...defaultUnit, op: 'PICKUP', item: 'WHEAT', qty: 3 })).toEqual(['PICKUP', 'WHEAT', 3]);
    expect(toUnitAction({ ...defaultUnit, op: 'WATER' })).toEqual(['WATER']);
  });

  it('fromUnitAction inverts toUnitAction', () => {
    const drafts: UnitDraft[] = [
      { ...defaultUnit, op: 'PLANT', crop: 'STRAWBERRY' },
      { ...defaultUnit, op: 'PLACE', item: 'EGG', qty: 5 },
      { ...defaultUnit, op: 'NORTH' },
      { ...defaultUnit, op: 'PASS' },
    ];
    for (const d of drafts) {
      const round = fromUnitAction(toUnitAction(d));
      expect(toUnitAction(round)).toEqual(toUnitAction(d));
    }
  });

  it('toMarketOrder rejects invalid combinations', () => {
    expect(toMarketOrder({ kind: 'BUY_PRODUCT', crop: 'WHEAT', animal: 'GOOSE', product: 'MILK', qty: 1 })).toBeNull();
    expect(toMarketOrder({ kind: 'SELL', crop: 'WHEAT', animal: 'GOOSE', product: 'COW', qty: 1 })).toBeNull();
    expect(toMarketOrder({ kind: 'SELL', crop: 'WHEAT', animal: 'GOOSE', product: 'MILK', qty: 2 })).toEqual([
      'SELL',
      'MILK',
      2,
    ]);
  });
});
