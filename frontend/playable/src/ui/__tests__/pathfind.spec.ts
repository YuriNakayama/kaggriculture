import { describe, expect, it } from 'vitest';
import { initGameState, resolveConfig } from '../../engine/state';
import { findPath, pathPositions } from '../pathfind';

const config = resolveConfig({ seed: 1 });

describe('findPath', () => {
  it('routes within the unlocked quadrant and refuses locked targets', () => {
    const s = initGameState(2, config, 1);
    const farm = s.farms[0];
    const from = farm.farmer; // spawns near the shed
    expect(findPath(farm, from, [0, 0])).not.toBeNull();
    expect(findPath(farm, from, [9, 9])).toBeNull(); // SE locked at start
    expect(findPath(farm, from, from)).toEqual([]);
  });

  it('path length equals manhattan distance on an open board', () => {
    const s = initGameState(2, config, 1);
    const farm = s.farms[0];
    const path = findPath(farm, [0, 0], [3, 4])!;
    expect(path.length).toBe(7);
    const visited = pathPositions([0, 0], path);
    expect(visited[visited.length - 1]).toEqual([3, 4]);
  });
});
