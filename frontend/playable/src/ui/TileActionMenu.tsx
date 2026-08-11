/**
 * Context menu for a tapped board tile: shows what the selected unit could do
 * there (legal ops as buttons, illegal ones grayed with the Japanese reason),
 * movement, and unit selection. Rendered as a bottom sheet on mobile and a
 * floating card on desktop (pure CSS).
 */

import { useState } from 'react';
import { ANIMALS } from '../engine/constants';
import { legalPickupItems, legalPlaceItems, legalPlantCrops, type UnitOpVerdicts } from '../engine/legality';
import { OP_HELP } from './opHelp';
import type { AnimalTile, CropId, GameState, PlantTile, Position, ShedItemId, Tile, UnitAction } from '../engine/types';
import { LOCKED } from '../engine/types';
import type { UnitDraft } from './useTurnDraft';

const TILE_OP_ORDER = [
  'HARVEST',
  'WATER',
  'PLANT',
  'DIG',
  'FEED',
  'CARE',
  'COLLECT_FERTILIZER',
  'FERTILIZE',
  'PICKUP',
  'PLACE',
  'BUILD_COOP',
  'BUILD_PASTURE',
] as const;

type TileOp = (typeof TILE_OP_ORDER)[number];

function tileSummary(tile: Tile): string {
  if (tile === LOCKED) return 'ロック区画';
  if (tile === null) return '空き地';
  if ((tile as PlantTile).kind === 'PLANT') {
    const p = tile as PlantTile;
    return `${p.crop} (実り ${p.yield_units})${p.watered_today ? ' 💧済' : ''}`;
  }
  if ('animal' in (tile as object)) {
    const a = tile as AnimalTile;
    return `${a.kind}: ${a.animal} (生産物 ${a.yield_units})`;
  }
  if ((tile as { kind: string }).kind === 'WEED') return '雑草';
  return (tile as { kind: string }).kind === 'COOP' ? '小屋 (空)' : '牧草地 (空)';
}

interface Props {
  state: GameState;
  player: number;
  unit: number;
  unitLabel: string;
  pos: Position;
  verdicts: UnitOpVerdicts;
  onSameTile: boolean;
  /** null = unreachable */
  pathLength: number | null;
  unitsHere: { unit: number; label: string }[];
  onPick(patch: Partial<UnitDraft>): void;
  onMoveHere(thenOp?: UnitAction): void;
  onSelectUnit(unit: number): void;
  onClose(): void;
}

export function TileActionMenu(props: Props) {
  const { state, player, unit, pos, verdicts, onSameTile, pathLength, unitsHere } = props;
  const [submenu, setSubmenu] = useState<'PLANT' | 'PICKUP' | 'PLACE' | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const tile = state.farms[player].tiles[pos[1]][pos[0]];

  const pick = (op: TileOp, patch: Partial<UnitDraft> = {}) => {
    if (onSameTile) props.onPick({ op, ...patch });
    else {
      // Far/adjacent tile: walk there, then do it on arrival.
      const draft: UnitDraft = { op, crop: 'WHEAT', item: 'WHEAT', qty: 1, ...patch } as UnitDraft;
      const action: UnitAction =
        op === 'PLANT'
          ? ['PLANT', draft.crop]
          : op === 'PICKUP' || op === 'PLACE'
            ? [op, draft.item, draft.qty]
            : [op];
      props.onMoveHere(action);
    }
    props.onClose();
  };

  const argOptions = (op: 'PLANT' | 'PICKUP' | 'PLACE'): { value: string; label: string }[] => {
    if (op === 'PLANT') return legalPlantCrops(state, player).map((c) => ({ value: c, label: c }));
    if (op === 'PICKUP')
      return legalPickupItems(state, player).map((i) => ({
        value: i,
        label: `${i} (倉庫 x${state.privates[player].shed[i] ?? 0})`,
      }));
    const inv = state.privates[player].inventories[unit] ?? {};
    // PLACE onto a structure narrows to matching animals; otherwise anything held.
    const isStructure =
      tile !== null && tile !== LOCKED && !('animal' in (tile as object)) && (tile as { kind: string }).kind !== 'PLANT' && (tile as { kind: string }).kind !== 'WEED';
    return legalPlaceItems(state, player, unit)
      .filter((i) => !isStructure || (i in ANIMALS && ANIMALS[i as keyof typeof ANIMALS].structure === (tile as { kind: string }).kind))
      .map((i) => ({ value: i, label: `${i} (所持 x${inv[i as ShedItemId] ?? 0})` }));
  };

  return (
    <div className="tile-menu" role="dialog" aria-label="タイル操作">
      <div className="tile-menu-head">
        <strong>
          ({pos[0]}, {pos[1]}) {tileSummary(tile)}
        </strong>
        <span>
          <button
            type="button"
            className="tile-menu-help-toggle"
            onClick={() => setShowHelp((v) => !v)}
            title="各操作の説明を表示"
          >
            ⓘ
          </button>
          <button type="button" className="tile-menu-close" onClick={props.onClose}>
            ×
          </button>
        </span>
      </div>

      {unitsHere.length > 0 && (
        <div className="tile-menu-units">
          {unitsHere
            .filter((u) => u.unit !== unit)
            .map((u) => (
              <button key={u.unit} type="button" onClick={() => props.onSelectUnit(u.unit)}>
                👤 {u.label} を選択
              </button>
            ))}
        </div>
      )}

      {!onSameTile && (
        <div className="tile-menu-section">
          <div className="tile-menu-subhead">🚶 移動</div>
          <button
            type="button"
            className="tile-menu-move"
            disabled={pathLength === null}
            onClick={() => {
              props.onMoveHere();
              props.onClose();
            }}
          >
            ここへ移動するだけ{pathLength !== null ? ` (${pathLength}ターン)` : ' — 到達不能'}
          </button>
        </div>
      )}

      {submenu === null ? (
        <div className="tile-menu-ops">
          <div className="tile-menu-subhead">
            {onSameTile ? '🎯 この場所で実行' : '🎯 移動して到着後に実行 (任意)'}
          </div>
          {TILE_OP_ORDER.map((op) => {
            const v = verdicts[op];
            const needsArg = op === 'PLANT' || op === 'PICKUP' || op === 'PLACE';
            return (
              <button
                key={op}
                type="button"
                disabled={!v.legal}
                title={v.legal ? OP_HELP[op] : `${v.reason ?? ''}\n${OP_HELP[op] ?? ''}`}
                onClick={() => (needsArg ? setSubmenu(op as 'PLANT' | 'PICKUP' | 'PLACE') : pick(op))}
              >
                {op}
                {!v.legal && v.reason ? <span className="tile-menu-reason"> — {v.reason}</span> : null}
                {showHelp && OP_HELP[op] ? <div className="tile-menu-desc">{OP_HELP[op]}</div> : null}
              </button>
            );
          })}
        </div>
      ) : (
        <div className="tile-menu-ops">
          <div className="tile-menu-subhead">{submenu} — 対象を選択</div>
          {argOptions(submenu).map((o) => (
            <button
              key={o.value}
              type="button"
              onClick={() => pick(submenu, submenu === 'PLANT' ? { crop: o.value as CropId } : { item: o.value as ShedItemId, qty: 1 })}
            >
              {o.label}
            </button>
          ))}
          <button type="button" onClick={() => setSubmenu(null)}>
            ← 戻る
          </button>
        </div>
      )}
    </div>
  );
}
