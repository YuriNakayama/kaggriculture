/**
 * スマートコマンド: 「全部に水やり」等の一括指示を、ユニットごとのタスク列
 * (歩いて→その場で op) に展開し、毎ターン1 op ずつ消化する。エンジンには
 * 常に普通の UnitAction が渡るため、リプレイ/再開ログは通常プレイと同一。
 */

import { CROPS } from '../engine/constants';
import { isShedAdjacent, shedAccessTiles } from '../engine/state';
import type { AnimalTile, Farm, GameState, PlantTile, Position, ShedItemId, Tile, UnitAction } from '../engine/types';
import { LOCKED } from '../engine/types';
import { findPath } from './pathfind';

export type SmartCommand = 'WATER_ALL' | 'HARVEST_ALL' | 'CLEAR_WEEDS' | 'TEND_ANIMALS' | 'DEPOSIT_ALL';

export type UnitTask =
  // op: null = 移動のみ (到着したら完了、追加ターンを消費しない)
  | { kind: 'op-at'; target: Position; op: UnitAction | null }
  | { kind: 'deposit' };

export type TaskQueues = Record<number, UnitTask[]>;

function isPlant(t: Tile): t is PlantTile {
  return t !== null && t !== LOCKED && (t as { kind: string }).kind === 'PLANT';
}
function isAnimal(t: Tile): t is AnimalTile {
  return t !== null && t !== LOCKED && 'animal' in (t as object);
}
function isWeed(t: Tile): boolean {
  return t !== null && t !== LOCKED && (t as { kind: string }).kind === 'WEED';
}

function unitPositions(state: GameState, player: number): Position[] {
  const farm = state.farms[player];
  return [farm.farmer, ...farm.hands];
}

/** Tiles the command wants visited, with the op to run there. */
export function commandTargets(state: GameState, player: number, cmd: SmartCommand): { target: Position; op: UnitAction }[] {
  const farm = state.farms[player];
  const out: { target: Position; op: UnitAction }[] = [];
  const size = farm.tiles.length;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const t = farm.tiles[y][x];
      if (cmd === 'WATER_ALL' && isPlant(t) && !t.watered_today) out.push({ target: [x, y], op: ['WATER'] });
      else if (cmd === 'HARVEST_ALL') {
        const ready =
          (isPlant(t) && t.yield_units > 0 && state.day - t.planted_day >= CROPS[t.crop].first_yield_day) ||
          (isAnimal(t) && t.yield_units > 0);
        if (ready) out.push({ target: [x, y], op: ['HARVEST'] });
      } else if (cmd === 'CLEAR_WEEDS' && isWeed(t)) out.push({ target: [x, y], op: ['DIG'] });
      else if (cmd === 'TEND_ANIMALS' && isAnimal(t) && !t.cared_today) out.push({ target: [x, y], op: ['CARE'] });
    }
  }
  return out;
}

/**
 * Greedy nearest-assignment of targets to units. Each unit's "position" for
 * distance advances to its last assigned target so routes stay coherent.
 */
export function assignCommand(state: GameState, player: number, cmd: SmartCommand): TaskQueues {
  const queues: TaskQueues = {};
  const units = unitPositions(state, player);
  if (cmd === 'DEPOSIT_ALL') {
    units.forEach((_, u) => {
      const inv = state.privates[player].inventories[u] ?? {};
      if (Object.values(inv).some((n) => (n ?? 0) > 0)) queues[u] = [{ kind: 'deposit' }];
    });
    return queues;
  }
  const targets = commandTargets(state, player, cmd);
  const cursor: Position[] = units.map((p) => [p[0], p[1]]);
  const dist = (a: Position, b: Position) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  for (const t of targets.sort((a, b) => dist(a.target, cursor[0]) - dist(b.target, cursor[0]))) {
    let best = 0;
    let bestCost = Infinity;
    for (let u = 0; u < units.length; u++) {
      const cost = dist(cursor[u], t.target) + (queues[u]?.length ?? 0) * 2; // load balancing
      if (cost < bestCost) {
        bestCost = cost;
        best = u;
      }
    }
    (queues[best] ??= []).push({ kind: 'op-at', target: t.target, op: t.op });
    cursor[best] = t.target;
  }
  return queues;
}

/** Next single-turn op for a task. done=true → pop the task after this op. */
export function nextOp(
  state: GameState,
  player: number,
  unit: number,
  task: UnitTask
): { op: UnitAction; done: boolean } | null {
  const farm: Farm = state.farms[player];
  const pos = unit === 0 ? farm.farmer : farm.hands[unit - 1];
  if (!pos) return null;

  if (task.kind === 'op-at') {
    if (pos[0] === task.target[0] && pos[1] === task.target[1]) {
      if (task.op === null) return null; // 到着のみで完了 — このターンは消費しない
      return { op: task.op, done: true };
    }
    const path = findPath(farm, pos, task.target);
    if (!path || path.length === 0) return null; // unreachable → drop task
    return { op: [path[0]], done: false };
  }

  // deposit: walk to the nearest shed-access tile, then PLACE items until empty.
  const inv = state.privates[player].inventories[unit] ?? {};
  const items = (Object.keys(inv) as ShedItemId[]).filter((i) => (inv[i] ?? 0) > 0);
  if (items.length === 0) return null; // nothing left → drop task
  if (isShedAdjacent(pos, farm.tiles.length)) {
    const item = items[0];
    return { op: ['PLACE', item, inv[item] ?? 1], done: items.length === 1 };
  }
  const sheds = shedAccessTiles(farm.tiles.length);
  let best: ReturnType<typeof findPath> = null;
  for (const s of sheds) {
    const p = findPath(farm, pos, s);
    if (p && (best === null || p.length < best.length)) best = p;
  }
  if (!best || best.length === 0) return null;
  return { op: [best[0]], done: false };
}

/** Expand every unit's queue head into this turn's action (PASS when idle). */
export function autoActions(
  state: GameState,
  player: number,
  queues: TaskQueues
): { action: { farmer: UnitAction; hands: UnitAction[]; market: never[] }; nextQueues: TaskQueues; idle: boolean } {
  const numUnits = 1 + state.farms[player].hands.length;
  const ops: UnitAction[] = [];
  const nextQueues: TaskQueues = {};
  let idle = true;
  for (let u = 0; u < numUnits; u++) {
    let queue = [...(queues[u] ?? [])];
    let op: UnitAction = ['PASS'];
    while (queue.length > 0) {
      const step = nextOp(state, player, u, queue[0]);
      if (step === null) {
        queue.shift(); // task impossible/finished — try the next one
        continue;
      }
      op = step.op;
      idle = false;
      if (step.done) queue = queue.slice(1);
      break;
    }
    if (queue.length > 0) nextQueues[u] = queue;
    ops.push(op);
  }
  return { action: { farmer: ops[0], hands: ops.slice(1), market: [] }, nextQueues, idle };
}

/** Total queued tasks (for the UI badge). */
export function queueSize(queues: TaskQueues): number {
  return Object.values(queues).reduce((a, q) => a + q.length, 0);
}

/**
 * 1日分のルーチン: 収穫 → 水やり → 世話 → 雑草 の順に全ターゲットを
 * 全ユニットへ配分し、持ち物のあるユニットには最後に倉庫収納を積む。
 * 自動進行側が毎ターン再生成して呼ぶことで「湧いた仕事」も拾う。
 */
export function dailyRoutineQueues(state: GameState, player: number): TaskQueues {
  const queues: TaskQueues = {};
  const units = unitPositions(state, player);
  const cursor: Position[] = units.map((p) => [p[0], p[1]]);
  const dist = (a: Position, b: Position) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

  const ordered: { target: Position; op: UnitAction }[] = [
    ...commandTargets(state, player, 'HARVEST_ALL'),
    ...commandTargets(state, player, 'WATER_ALL'),
    ...commandTargets(state, player, 'TEND_ANIMALS'),
    ...commandTargets(state, player, 'CLEAR_WEEDS'),
  ];
  for (const t of ordered) {
    let best = 0;
    let bestCost = Infinity;
    for (let u = 0; u < units.length; u++) {
      const cost = dist(cursor[u], t.target) + (queues[u]?.length ?? 0) * 2;
      if (cost < bestCost) {
        bestCost = cost;
        best = u;
      }
    }
    (queues[best] ??= []).push({ kind: 'op-at', target: t.target, op: t.op });
    cursor[best] = t.target;
  }
  units.forEach((_, u) => {
    const inv = state.privates[player].inventories[u] ?? {};
    if (Object.values(inv).some((n) => (n ?? 0) > 0)) (queues[u] ??= []).push({ kind: 'deposit' });
  });
  return queues;
}
