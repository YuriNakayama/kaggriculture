/**
 * Shared per-turn draft state: what the human intends each unit to do plus the
 * market order list. Owned by GameScreen so BOTH the ActionPanel (dropdowns)
 * and the board's tap-to-act interaction edit the same draft.
 *
 * Pure helpers (draft ⇄ PlayerAction) are exported for tests and for the
 * repeat-last-turn feature.
 */

import { useEffect, useRef, useState } from 'react';
import type {
  AnimalId,
  CropId,
  GameState,
  MarketOrder,
  PlayerAction,
  ShedItemId,
  UnitAction,
} from '../engine/types';

export type UnitOp =
  | 'PASS'
  | 'NORTH'
  | 'SOUTH'
  | 'EAST'
  | 'WEST'
  | 'WATER'
  | 'HARVEST'
  | 'FERTILIZE'
  | 'DIG'
  | 'BUILD_COOP'
  | 'BUILD_PASTURE'
  | 'FEED'
  | 'COLLECT_FERTILIZER'
  | 'CARE'
  | 'PLANT'
  | 'PICKUP'
  | 'PLACE';

export interface UnitDraft {
  op: UnitOp;
  crop: CropId;
  item: ShedItemId;
  qty: number;
}

export type MarketKind = 'HIRE' | 'BUY_LAND' | 'BUY_SEED' | 'BUY_PRODUCT' | 'BUY_ANIMAL' | 'SELL';

export interface MarketDraft {
  kind: MarketKind;
  crop: CropId;
  animal: AnimalId;
  product: ShedItemId;
  qty: number;
}

export const defaultUnit: UnitDraft = { op: 'PASS', crop: 'WHEAT', item: 'WHEAT', qty: 1 };
export const defaultMarket: MarketDraft = {
  kind: 'BUY_SEED',
  crop: 'WHEAT',
  animal: 'GOOSE',
  product: 'WHEAT',
  qty: 1,
};

export function toUnitAction(d: UnitDraft): UnitAction {
  switch (d.op) {
    case 'PLANT':
      return ['PLANT', d.crop];
    case 'PICKUP':
      return ['PICKUP', d.item, Math.max(1, Math.floor(d.qty))];
    case 'PLACE':
      return ['PLACE', d.item, Math.max(1, Math.floor(d.qty))];
    default:
      return [d.op];
  }
}

export function toMarketOrder(d: MarketDraft): MarketOrder | null {
  const qty = Math.max(1, Math.floor(d.qty));
  switch (d.kind) {
    case 'HIRE':
      return ['HIRE'];
    case 'BUY_LAND':
      return ['BUY_LAND'];
    case 'BUY_SEED':
      return ['BUY_SEED', d.crop, qty];
    case 'BUY_PRODUCT':
      if (d.product !== 'WHEAT' && d.product !== 'FERTILIZER') return null;
      return ['BUY_PRODUCT', d.product, qty];
    case 'BUY_ANIMAL':
      return ['BUY_ANIMAL', d.animal, qty];
    case 'SELL':
      if (d.product === 'GOOSE' || d.product === 'COW' || d.product === 'SHEEP') return null;
      return ['SELL', d.product, qty];
    default:
      return null;
  }
}

/** Inverse of toUnitAction — used by repeat-last-turn. */
export function fromUnitAction(a: UnitAction | undefined): UnitDraft {
  if (!Array.isArray(a) || (a as unknown[]).length === 0) return { ...defaultUnit };
  const [op, arg, qty] = a as [UnitOp, string?, number?];
  const d: UnitDraft = { ...defaultUnit, op };
  if (op === 'PLANT' && arg) d.crop = arg as CropId;
  if ((op === 'PICKUP' || op === 'PLACE') && arg) {
    d.item = arg as ShedItemId;
    d.qty = qty ?? 1;
  }
  return d;
}

export interface TurnDraft {
  farmer: UnitDraft;
  hands: UnitDraft[];
  orders: MarketDraft[];
  setFarmer(d: UnitDraft): void;
  setHand(i: number, d: UnitDraft): void;
  setOrders(orders: MarketDraft[]): void;
  /** Board-tap entry point: set a unit's op (+optional args) in one call. */
  setUnitDraft(unit: number, patch: Partial<UnitDraft>): void;
  /** Bundle drafts into the submittable PlayerAction. */
  buildAction(): PlayerAction;
  /** buildAction with one unit's draft overridden — for instant (tap-to-run) ops. */
  buildActionWith(unit: number, patch: Partial<UnitDraft>): PlayerAction;
  /** Reset unit drafts to PASS and clear orders (after submit). */
  afterSubmit(action: PlayerAction): void;
  /** Restore the previously submitted action's unit ops (not market orders). */
  repeatLast(): void;
  hasLast: boolean;
}

export function useTurnDraft(state: GameState | null, player: number | null): TurnDraft {
  const numHands = player !== null && state ? (state.farms[player]?.hands.length ?? 0) : 0;

  const [farmer, setFarmer] = useState<UnitDraft>({ ...defaultUnit });
  const [hands, setHands] = useState<UnitDraft[]>([]);
  const [orders, setOrders] = useState<MarketDraft[]>([]);
  const lastRef = useRef<PlayerAction | null>(null);
  const [hasLast, setHasLast] = useState(false);

  // Keep the hands array length in sync with the live farm. New hands default
  // to PASS; extra entries are trimmed if a hand was lost.
  useEffect(() => {
    setHands((prev) => {
      if (prev.length === numHands) return prev;
      const next = prev.slice(0, numHands);
      while (next.length < numHands) next.push({ ...defaultUnit });
      return next;
    });
  }, [numHands]);

  const setHand = (i: number, d: UnitDraft) => {
    setHands((prev) => {
      const next = [...prev];
      next[i] = d;
      return next;
    });
  };

  const setUnitDraft = (unit: number, patch: Partial<UnitDraft>) => {
    if (unit === 0) setFarmer((prev) => ({ ...prev, ...patch }));
    else setHand(unit - 1, { ...(hands[unit - 1] ?? defaultUnit), ...patch });
  };

  const buildAction = (): PlayerAction => ({
    farmer: toUnitAction(farmer),
    hands: hands.map(toUnitAction),
    market: orders.map(toMarketOrder).filter((o): o is MarketOrder => o !== null),
  });

  const buildActionWith = (unit: number, patch: Partial<UnitDraft>): PlayerAction => {
    const patched = { ...(unit === 0 ? farmer : (hands[unit - 1] ?? defaultUnit)), ...patch };
    return {
      farmer: unit === 0 ? toUnitAction(patched) : toUnitAction(farmer),
      hands: hands.map((h, i) => (i === unit - 1 ? toUnitAction(patched) : toUnitAction(h))),
      market: orders.map(toMarketOrder).filter((o): o is MarketOrder => o !== null),
    };
  };

  const afterSubmit = (action: PlayerAction) => {
    lastRef.current = action;
    setHasLast(true);
    setFarmer({ ...defaultUnit });
    setHands((prev) => prev.map(() => ({ ...defaultUnit })));
    setOrders([]);
  };

  const repeatLast = () => {
    const last = lastRef.current;
    if (!last) return;
    setFarmer(fromUnitAction(last.farmer));
    setHands((prev) => prev.map((_, i) => fromUnitAction(last.hands[i])));
    // Market orders are deliberately NOT repeated — prices moved.
  };

  return {
    farmer,
    hands,
    orders,
    setFarmer,
    setHand,
    setOrders,
    setUnitDraft,
    buildAction,
    buildActionWith,
    afterSubmit,
    repeatLast,
    hasLast,
  };
}
