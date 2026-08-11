/**
 * ハンド役割自動化: 日雇いに「水やり係」等を割り当てると、毎ターンの draft
 * を自動で埋める。ユーザーは農夫と市場に集中できる。手動で op を変えれば
 * そのターンは手動が優先される (自動記入は state 更新直後の一度だけ)。
 */

import { useEffect, useState } from 'react';
import type { GameState, Position, UnitAction } from '../engine/types';
import { findPath } from './pathfind';
import { commandTargets, type SmartCommand } from './smartTasks';
import { fromUnitAction, type TurnDraft } from './useTurnDraft';

export type HandRole = 'MANUAL' | 'WATER' | 'HARVEST' | 'TEND';

export const ROLE_LABELS: Record<HandRole, string> = {
  MANUAL: '手動',
  WATER: '💧 水やり係',
  HARVEST: '🌾 収穫係',
  TEND: '🐄 世話係',
};

const ROLE_COMMAND: Record<Exclude<HandRole, 'MANUAL'>, SmartCommand> = {
  WATER: 'WATER_ALL',
  HARVEST: 'HARVEST_ALL',
  TEND: 'TEND_ANIMALS',
};

/** ユニット1体の役割行動: 最寄りの未処理ターゲットへ移動 / その場で実行。 */
export function roleOp(
  state: GameState,
  player: number,
  unit: number,
  role: Exclude<HandRole, 'MANUAL'>,
  claimed: Set<string>
): { op: UnitAction; claim?: string } | null {
  const farm = state.farms[player];
  const pos: Position | undefined = unit === 0 ? farm.farmer : farm.hands[unit - 1];
  if (!pos) return null;
  const targets = commandTargets(state, player, ROLE_COMMAND[role]).filter(
    (t) => !claimed.has(`${t.target[0]},${t.target[1]}`)
  );
  if (targets.length === 0) return null;
  let best: { target: Position; op: UnitAction; path: ReturnType<typeof findPath> } | null = null;
  for (const t of targets) {
    const path = findPath(farm, pos, t.target);
    if (!path) continue;
    if (!best || path.length < best.path!.length) best = { ...t, path };
  }
  if (!best) return null;
  const claim = `${best.target[0]},${best.target[1]}`;
  if (best.path!.length === 0) return { op: best.op, claim };
  return { op: [best.path![0]], claim };
}

export function useHandRoles(state: GameState | null, player: number | null, draft: TurnDraft) {
  const [roles, setRoles] = useState<Record<number, HandRole>>({});

  const setRole = (unit: number, role: HandRole) => setRoles((prev) => ({ ...prev, [unit]: role }));

  // Fill hand drafts right after each new state (and when roles change).
  useEffect(() => {
    if (!state || player === null || state.done) return;
    const claimed = new Set<string>();
    const numHands = state.farms[player].hands.length;
    for (let unit = 1; unit <= numHands; unit++) {
      const role = roles[unit] ?? 'MANUAL';
      if (role === 'MANUAL') continue;
      const r = roleOp(state, player, unit, role, claimed);
      if (!r) {
        draft.setUnitDraft(unit, fromUnitAction(['PASS']));
        continue;
      }
      if (r.claim) claimed.add(r.claim);
      draft.setUnitDraft(unit, fromUnitAction(r.op));
    }
    // Draft setters are stable; re-run per turn / role change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.step, player, roles]);

  return { roles, setRole };
}
