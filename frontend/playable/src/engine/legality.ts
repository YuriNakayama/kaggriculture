/**
 * Advisory legality checks for the UI. Mirrors the no-op conditions in
 * `applyUnitAction` / `processMarket` (interpreter.ts) so the ActionPanel can
 * gray out ops that the engine would silently discard — the game's biggest
 * trap is that invalid actions are no-ops, not errors.
 *
 * Advisory only: the engine remains the judge. Where legality depends on an
 * op's argument (PLANT crop, PICKUP item…), an op counts as legal if SOME
 * argument makes it legal; per-argument helpers refine the dropdowns.
 */

import { ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, PRODUCTS } from './constants';
import { hireCost } from './interpreter';
import { marketPrice } from './market';
import { isShedAdjacent } from './state';
import type {
  AnimalId,
  AnimalTile,
  CropId,
  Farm,
  GameState,
  PlantTile,
  Position,
  ProductId,
  ShedItemId,
  Tile,
  UnitAction,
} from './types';
import { LOCKED } from './types';

const MOVES: Record<'NORTH' | 'SOUTH' | 'EAST' | 'WEST', [number, number]> = {
  NORTH: [0, -1],
  SOUTH: [0, 1],
  EAST: [1, 0],
  WEST: [-1, 0],
};

function isPlant(tile: Tile): tile is PlantTile {
  return tile !== null && tile !== LOCKED && (tile as { kind: string }).kind === 'PLANT';
}

function isAnimal(tile: Tile): tile is AnimalTile {
  return tile !== null && tile !== LOCKED && 'animal' in (tile as object);
}

function invCount(state: GameState, player: number, unit: number, item: ShedItemId): number {
  return state.privates[player].inventories[unit]?.[item] ?? 0;
}

function invHasAny(state: GameState, player: number, unit: number): boolean {
  return Object.values(state.privates[player].inventories[unit] ?? {}).some((v) => (v ?? 0) > 0);
}

function unitPos(farm: Farm, unit: number): Position | null {
  if (unit === 0) return farm.farmer;
  return farm.hands[unit - 1] ?? null;
}

export type UnitOpName =
  | 'PASS'
  | 'NORTH'
  | 'SOUTH'
  | 'EAST'
  | 'WEST'
  | 'PICKUP'
  | 'PLACE'
  | 'PLANT'
  | 'WATER'
  | 'HARVEST'
  | 'FERTILIZE'
  | 'DIG'
  | 'BUILD_COOP'
  | 'BUILD_PASTURE'
  | 'FEED'
  | 'COLLECT_FERTILIZER'
  | 'CARE';

/** Which crops can this unit PLANT right now (tile empty + seed in stock)? */
export function legalPlantCrops(state: GameState, player: number): CropId[] {
  const seeds = state.privates[player].seeds;
  return (Object.keys(CROPS) as CropId[]).filter((c) => (seeds[c] ?? 0) > 0);
}

/** Which items can this unit PICKUP from the shed (stock > 0)? */
export function legalPickupItems(state: GameState, player: number): ShedItemId[] {
  const shed = state.privates[player].shed;
  return (Object.keys(shed) as ShedItemId[]).filter((i) => (shed[i] ?? 0) > 0);
}

/** Which items can this unit PLACE right now (in its inventory)? */
export function legalPlaceItems(state: GameState, player: number, unit: number): ShedItemId[] {
  const inv = state.privates[player].inventories[unit] ?? {};
  return (Object.keys(inv) as ShedItemId[]).filter((i) => (inv[i] ?? 0) > 0);
}

/**
 * One entry per unit op: `true` = some argument choice makes it a real move,
 * `false` = the engine would silently discard it.
 */
export function legalUnitOps(state: GameState, player: number, unit: number): Record<UnitOpName, boolean> {
  const farm = state.farms[player];
  const pos = unitPos(farm, unit);
  const out = {} as Record<UnitOpName, boolean>;
  const boardSize = farm.tiles.length;

  out.PASS = true;
  for (const dir of Object.keys(MOVES) as (keyof typeof MOVES)[]) {
    if (!pos) {
      out[dir] = false;
      continue;
    }
    const nx = pos[0] + MOVES[dir][0];
    const ny = pos[1] + MOVES[dir][1];
    out[dir] = nx >= 0 && nx < boardSize && ny >= 0 && ny < boardSize && farm.tiles[ny][nx] !== LOCKED;
  }

  if (!pos) {
    for (const op of [
      'PICKUP',
      'PLACE',
      'PLANT',
      'WATER',
      'HARVEST',
      'FERTILIZE',
      'DIG',
      'BUILD_COOP',
      'BUILD_PASTURE',
      'FEED',
      'COLLECT_FERTILIZER',
      'CARE',
    ] as UnitOpName[]) {
      out[op] = false;
    }
    return out;
  }

  const [x, y] = pos;
  const tile = farm.tiles[y][x];
  const onShed = isShedAdjacent([x, y], boardSize);
  const locked = tile === LOCKED;

  out.PICKUP = onShed && legalPickupItems(state, player).length > 0;
  out.PLANT = !locked && tile === null && legalPlantCrops(state, player).length > 0;
  out.WATER = !locked && isPlant(tile) && !tile.watered_today;
  out.HARVEST =
    !locked &&
    ((isPlant(tile) &&
      tile.yield_units > 0 &&
      state.day - tile.planted_day >= CROPS[tile.crop].first_yield_day) ||
      (isAnimal(tile) && tile.yield_units > 0));
  out.FERTILIZE = !locked && isPlant(tile) && invCount(state, player, unit, 'FERTILIZER') > 0;
  out.DIG = !locked && tile !== null && !isAnimal(tile);
  out.BUILD_COOP = !locked && tile === null;
  out.BUILD_PASTURE = !locked && tile === null;
  out.FEED = !locked && isAnimal(tile) && !tile.fed_today && invCount(state, player, unit, 'WHEAT') > 0;
  out.COLLECT_FERTILIZER = !locked && isAnimal(tile) && tile.fertilizer_available;
  out.CARE = !locked && isAnimal(tile) && !tile.cared_today;

  // PLACE: animal onto a matching empty structure, or any carried item into
  // the shed (if adjacent and there is room).
  let canPlace = false;
  if (!locked && tile !== null && !isAnimal(tile) && (tile.kind === 'COOP' || tile.kind === 'PASTURE')) {
    canPlace = (Object.keys(ANIMALS) as AnimalId[]).some(
      (a) => ANIMALS[a].structure === tile.kind && invCount(state, player, unit, a) > 0
    );
  }
  if (!canPlace && onShed && invHasAny(state, player, unit)) {
    let total = 0;
    for (const v of Object.values(state.privates[player].shed)) total += v ?? 0;
    canPlace = total < 100; // shedCapacity default; advisory
  }
  out.PLACE = canPlace;

  return out;
}

export interface MarketLegality {
  hire: { legal: boolean; cost: number };
  buyLand: { legal: boolean; cost: number | null; quadrant: string | null };
  buySeed: Partial<Record<CropId, { legal: boolean; cost: number }>>;
  buyProduct: Partial<Record<'WHEAT' | 'FERTILIZER', { legal: boolean; cost: number }>>;
  buyAnimal: Partial<Record<AnimalId, { legal: boolean; cost: number }>>;
  sell: Partial<Record<ProductId, { legal: boolean; stock: number; price: number }>>;
}

export function legalMarket(state: GameState, player: number): MarketLegality {
  const farm = state.farms[player];
  const priv = state.privates[player];
  const money = farm.money;
  const market = state.market;

  const nExtra = farm.unlocked_quadrants.length - 1;
  const landCost = nExtra < LAND_ORDER.length ? LAND_PRICES[nExtra] : null;

  const out: MarketLegality = {
    hire: { legal: money >= hireCost(farm.hires_today), cost: hireCost(farm.hires_today) },
    buyLand: {
      legal: landCost !== null && money >= landCost,
      cost: landCost,
      quadrant: nExtra < LAND_ORDER.length ? LAND_ORDER[nExtra] : null,
    },
    buySeed: {},
    buyProduct: {},
    buyAnimal: {},
    sell: {},
  };

  for (const c of Object.keys(CROPS) as CropId[]) {
    out.buySeed[c] = { legal: money >= CROPS[c].seed, cost: CROPS[c].seed };
  }
  for (const p of ['WHEAT', 'FERTILIZER'] as const) {
    const cost = marketPrice(p, market.inventory[p] - 1, market.params);
    out.buyProduct[p] = { legal: money >= cost, cost };
  }
  for (const a of Object.keys(ANIMALS) as AnimalId[]) {
    out.buyAnimal[a] = { legal: money >= ANIMALS[a].cost, cost: ANIMALS[a].cost };
  }
  for (const p of PRODUCTS) {
    const stock = priv.shed[p] ?? 0;
    out.sell[p] = {
      legal: stock > 0,
      stock,
      price: marketPrice(p, market.inventory[p], market.params),
    };
  }
  return out;
}

/**
 * Post-hoc audit of a submitted action: returns one human-readable line per
 * sub-action the engine would have discarded as a silent no-op.
 */
export function auditAction(
  state: GameState,
  player: number,
  action: { farmer: UnitAction; hands: UnitAction[]; market: unknown[] }
): string[] {
  const notes: string[] = [];
  const units: { label: string; act: UnitAction }[] = [
    { label: 'Farmer', act: action.farmer },
    ...action.hands.map((a, i) => ({ label: `Hand ${i + 1}`, act: a })),
  ];
  for (let u = 0; u < units.length; u++) {
    const { label, act } = units[u];
    if (!Array.isArray(act) || (act as unknown[]).length === 0) continue;
    const op = act[0] as UnitOpName;
    const legal = legalUnitOps(state, player, u);
    if (op in legal && !legal[op]) {
      notes.push(`${label}: ${act.join(' ')} — no effect (illegal here)`);
    }
  }
  const m = legalMarket(state, player);
  const maxOrders = 10;
  (action.market as unknown[][]).forEach((order, i) => {
    if (!Array.isArray(order) || order.length === 0) return;
    if (i >= maxOrders) {
      notes.push(`Market #${i + 1}: ${order.join(' ')} — dropped (over the ${maxOrders}-order cap)`);
      return;
    }
    const [kind, item] = order as [string, string?];
    const bad =
      (kind === 'HIRE' && !m.hire.legal) ||
      (kind === 'BUY_LAND' && !m.buyLand.legal) ||
      (kind === 'BUY_SEED' && item !== undefined && !m.buySeed[item as CropId]?.legal) ||
      (kind === 'BUY_PRODUCT' && item !== undefined && !m.buyProduct[item as 'WHEAT' | 'FERTILIZER']?.legal) ||
      (kind === 'BUY_ANIMAL' && item !== undefined && !m.buyAnimal[item as AnimalId]?.legal) ||
      (kind === 'SELL' && item !== undefined && !m.sell[item as ProductId]?.legal);
    if (bad) notes.push(`Market #${i + 1}: ${order.join(' ')} — likely no effect (funds/stock)`);
  });
  return notes;
}
