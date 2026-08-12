/**
 * スマートコマンドを実エンジンで走らせ、タスクが実際に完遂されることを検証。
 */

import { describe, expect, it } from 'vitest';
import { step } from '../../engine/interpreter';
import { initGameState, resolveConfig } from '../../engine/state';
import type { GameState, PlantTile } from '../../engine/types';
import { assignCommand, autoActions, commandTargets, dailyRoutineQueues, queueSize } from '../smartTasks';

const config = resolveConfig({ seed: 5 });
const PASS = { farmer: ['PASS'] as ['PASS'], hands: [], market: [] };

function runQueues(s: GameState, player: number, queues: ReturnType<typeof assignCommand>, maxTurns = 60): GameState {
  let cur = s;
  let q = queues;
  for (let i = 0; i < maxTurns && queueSize(q) > 0; i++) {
    const { action, nextQueues } = autoActions(cur, player, q);
    cur = step(cur, player === 0 ? [action, PASS] : [PASS, action], config);
    q = nextQueues;
  }
  return cur;
}

describe('smart commands on the real engine', () => {
  it('WATER_ALL walks to every unwatered plant and waters it', () => {
    let s = initGameState(2, config, 5);
    // Plant three wheat tiles by hand.
    s.privates[0].seeds.WHEAT = 3;
    const spots: [number, number][] = [
      [1, 1],
      [3, 2],
      [2, 4],
    ];
    for (const [x, y] of spots) {
      s.farms[0].tiles[y][x] = {
        kind: 'PLANT',
        crop: 'WHEAT',
        planted_day: 0,
        watered_today: false,
        consecutive_unwatered: 0,
        yield_units: 0,
        max_lifespan_step: 9999,
        fertilized_until_day: -1,
      } as PlantTile;
    }
    expect(commandTargets(s, 0, 'WATER_ALL')).toHaveLength(3);
    const queues = assignCommand(s, 0, 'WATER_ALL');
    const done = runQueues(s, 0, queues);
    for (const [x, y] of spots) {
      expect((done.farms[0].tiles[y][x] as PlantTile).watered_today).toBe(true);
    }
  });

  it('DEPOSIT_ALL carries the farmer inventory into the shed', () => {
    const s = initGameState(2, config, 5);
    s.privates[0].inventories[0] = { WHEAT: 4 };
    const queues = assignCommand(s, 0, 'DEPOSIT_ALL');
    expect(queueSize(queues)).toBe(1);
    const done = runQueues(s, 0, queues);
    expect(done.privates[0].shed.WHEAT ?? 0).toBe(4);
    expect(done.privates[0].inventories[0].WHEAT ?? 0).toBe(0);
  });

  it('dailyRoutineQueues waters, digs weeds, and deposits in one run', () => {
    let s = initGameState(2, config, 5);
    s.privates[0].inventories[0] = { WHEAT: 2 };
    s.farms[0].tiles[1][1] = {
      kind: 'PLANT',
      crop: 'WHEAT',
      planted_day: 0,
      watered_today: false,
      consecutive_unwatered: 0,
      yield_units: 0,
      max_lifespan_step: 9999,
      fertilized_until_day: -1,
    } as PlantTile;
    s.farms[0].tiles[2][3] = { kind: 'WEED' };
    const queues = dailyRoutineQueues(s, 0);
    expect(queueSize(queues)).toBeGreaterThanOrEqual(3); // water + weed + deposit
    const done = runQueues(s, 0, queues);
    expect((done.farms[0].tiles[1][1] as PlantTile).watered_today).toBe(true);
    expect(done.farms[0].tiles[2][3]).toBeNull();
    expect(done.privates[0].shed.WHEAT ?? 0).toBe(2);
  });

  it('autoActions passes when idle', () => {
    const s = initGameState(2, config, 5);
    const { action, idle } = autoActions(s, 0, {});
    expect(idle).toBe(true);
    expect(action.farmer).toEqual(['PASS']);
  });
});
