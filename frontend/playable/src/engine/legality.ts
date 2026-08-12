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

export interface OpVerdict {
  legal: boolean;
  /** Japanese explanation of WHY the op would be a silent no-op. Set iff !legal. */
  reason?: string;
}

export type UnitOpVerdicts = Record<UnitOpName, OpVerdict>;

const ok: OpVerdict = { legal: true };
const no = (reason: string): OpVerdict => ({ legal: false, reason });

const TILE_OPS: UnitOpName[] = [
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
];

/**
 * Legality of every unit op AT AN ARBITRARY TILE, with Japanese reasons for
 * illegal ones. Used both for the current tile (ActionPanel) and for
 * "what could I do over there?" previews on board tap. Movement verdicts are
 * always evaluated from the unit's actual position.
 */
export function legalUnitOpsAt(
  state: GameState,
  player: number,
  unit: number,
  at?: Position
): UnitOpVerdicts {
  const farm = state.farms[player];
  const pos = unitPos(farm, unit);
  const out = {} as UnitOpVerdicts;
  const boardSize = farm.tiles.length;

  out.PASS = ok;
  for (const dir of Object.keys(MOVES) as (keyof typeof MOVES)[]) {
    if (!pos) {
      out[dir] = no('ユニットがいない');
      continue;
    }
    const nx = pos[0] + MOVES[dir][0];
    const ny = pos[1] + MOVES[dir][1];
    if (nx < 0 || nx >= boardSize || ny < 0 || ny >= boardSize) out[dir] = no('盤面の外に出られない');
    else if (farm.tiles[ny][nx] === LOCKED) out[dir] = no('ロック中の区画 (BUY_LAND で解放)');
    else out[dir] = ok;
  }

  const target = at ?? pos;
  if (!target) {
    for (const op of TILE_OPS) out[op] = no('ユニットがいない');
    return out;
  }

  const [x, y] = target;
  const tile = farm.tiles[y][x];
  const onShed = isShedAdjacent([x, y], boardSize);
  const locked = tile === LOCKED;
  const priv = state.privates[player];

  if (locked) {
    for (const op of TILE_OPS) out[op] = no('ロック中の区画 (BUY_LAND で解放)');
    return out;
  }

  out.PICKUP = !onShed
    ? no('倉庫のアクセスタイル (中央4マス) に立つ必要がある')
    : legalPickupItems(state, player).length > 0
      ? ok
      : no('倉庫が空');

  out.PLANT =
    tile !== null
      ? no('空きタイルでないと植えられない')
      : legalPlantCrops(state, player).length > 0
        ? ok
        : no('種を持っていない (市場で BUY_SEED)');

  out.WATER = !isPlant(tile)
    ? no('このタイルに作物がない')
    : tile.watered_today
      ? no('今日はもう水やり済み')
      : ok;

  out.HARVEST = isPlant(tile)
    ? tile.yield_units <= 0
      ? no('収穫できる実がまだない')
      : state.day - tile.planted_day < CROPS[tile.crop].first_yield_day
        ? no(`初収穫は植えてから${CROPS[tile.crop].first_yield_day}日後`)
        : ok
    : isAnimal(tile)
      ? tile.yield_units > 0
        ? ok
        : no('収穫できる生産物がまだない')
      : no('このタイルに作物も動物もいない');

  out.FERTILIZE = !isPlant(tile)
    ? no('このタイルに作物がない')
    : invCount(state, player, unit, 'FERTILIZER') > 0
      ? ok
      : no('FERTILIZER を持っていない (PICKUP か COLLECT_FERTILIZER)');

  out.DIG =
    tile === null ? no('掘るものがない') : isAnimal(tile) ? no('動物のいる建物は撤去できない') : ok;

  out.BUILD_COOP = tile === null ? ok : no('空きタイルにしか建てられない');
  out.BUILD_PASTURE = tile === null ? ok : no('空きタイルにしか建てられない');

  out.FEED = !isAnimal(tile)
    ? no('このタイルに動物がいない')
    : tile.fed_today
      ? no('今日はもう餌やり済み')
      : invCount(state, player, unit, 'WHEAT') > 0
        ? ok
        : no('WHEAT を持っていない (倉庫から PICKUP)');

  out.COLLECT_FERTILIZER = !isAnimal(tile)
    ? no('このタイルに動物がいない')
    : tile.fertilizer_available
      ? ok
      : no('肥料がまだ溜まっていない');

  out.CARE = !isAnimal(tile) ? no('このタイルに動物がいない') : tile.cared_today ? no('今日はもう世話済み') : ok;

  // PLACE: animal onto a matching empty structure, or any carried item into
  // the shed (if adjacent and there is room).
  if (tile !== null && !isAnimal(tile) && (tile.kind === 'COOP' || tile.kind === 'PASTURE')) {
    const match = (Object.keys(ANIMALS) as AnimalId[]).some(
      (a) => ANIMALS[a].structure === tile.kind && invCount(state, player, unit, a) > 0
    );
    out.PLACE = match ? ok : no(`この建物に入れられる動物を持っていない`);
  } else if (onShed) {
    if (!invHasAny(state, player, unit)) out.PLACE = no('持ち物が空');
    else {
      let total = 0;
      for (const v of Object.values(priv.shed)) total += v ?? 0;
      out.PLACE = total < 100 ? ok : no('倉庫が満杯 (100)'); // shedCapacity default; advisory
    }
  } else {
    out.PLACE = no('倉庫のアクセスタイルか、空きの建物の上でのみ置ける');
  }

  return out;
}

/**
 * One entry per unit op: `true` = some argument choice makes it a real move,
 * `false` = the engine would silently discard it. (Boolean view of
 * `legalUnitOpsAt` at the unit's current tile.)
 */
export function legalUnitOps(state: GameState, player: number, unit: number): Record<UnitOpName, boolean> {
  const verdicts = legalUnitOpsAt(state, player, unit);
  const out = {} as Record<UnitOpName, boolean>;
  for (const [op, v] of Object.entries(verdicts) as [UnitOpName, OpVerdict][]) out[op] = v.legal;
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
