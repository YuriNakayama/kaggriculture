/**
 * BFS movement planning on a farm board. Walkability mirrors the engine's
 * movement rule exactly (interpreter.applyUnitAction): any in-bounds tile that
 * is not LOCKED is passable — plants, weeds, structures and other units do
 * not block movement.
 */

import type { Farm, Position, UnitAction } from '../engine/types';
import { LOCKED } from '../engine/types';

const DIRS: [string, number, number][] = [
  ['NORTH', 0, -1],
  ['SOUTH', 0, 1],
  ['EAST', 1, 0],
  ['WEST', -1, 0],
];

export type MoveOp = 'NORTH' | 'SOUTH' | 'EAST' | 'WEST';

/** Shortest path from → to as a list of move ops. null = unreachable. */
export function findPath(farm: Farm, from: Position, to: Position): MoveOp[] | null {
  const size = farm.tiles.length;
  const inBounds = (x: number, y: number) => x >= 0 && x < size && y >= 0 && y < size;
  if (!inBounds(to[0], to[1]) || farm.tiles[to[1]][to[0]] === LOCKED) return null;
  if (from[0] === to[0] && from[1] === to[1]) return [];

  const key = (x: number, y: number) => y * size + x;
  const prev = new Map<number, { k: number; op: MoveOp }>();
  const seen = new Set<number>([key(from[0], from[1])]);
  const queue: Position[] = [from];

  while (queue.length) {
    const [x, y] = queue.shift()!;
    for (const [op, dx, dy] of DIRS) {
      const nx = x + dx;
      const ny = y + dy;
      if (!inBounds(nx, ny) || farm.tiles[ny][nx] === LOCKED) continue;
      const nk = key(nx, ny);
      if (seen.has(nk)) continue;
      seen.add(nk);
      prev.set(nk, { k: key(x, y), op: op as MoveOp });
      if (nx === to[0] && ny === to[1]) {
        // Reconstruct.
        const ops: MoveOp[] = [];
        let cur = nk;
        while (cur !== key(from[0], from[1])) {
          const p = prev.get(cur)!;
          ops.push(p.op);
          cur = p.k;
        }
        return ops.reverse();
      }
      queue.push([nx, ny]);
    }
  }
  return null;
}

/** Positions visited along a path (excluding start), for overlay preview. */
export function pathPositions(from: Position, ops: MoveOp[]): Position[] {
  const out: Position[] = [];
  let [x, y] = from;
  for (const op of ops) {
    if (op === 'NORTH') y -= 1;
    else if (op === 'SOUTH') y += 1;
    else if (op === 'EAST') x += 1;
    else x -= 1;
    out.push([x, y]);
  }
  return out;
}

/** A queued destination (and an op to run on arrival) for one unit. */
export interface MoveIntent {
  target: Position;
  thenOp?: UnitAction;
}
