/**
 * Board-tap play state: which unit is selected, the open tile menu, and
 * per-unit movement intents ("walk to (x,y), then do OP"). Intents live in
 * the UI only — every turn they are expanded into ONE ordinary move op in the
 * shared TurnDraft, so the submitted actions and the resume log stay
 * completely standard.
 */

import { useEffect, useMemo, useState } from 'react';
import { legalUnitOpsAt, type UnitOpVerdicts } from '../engine/legality';
import type { GameState, Position, UnitAction } from '../engine/types';
import { overlayKey, type OverlayMap } from './FarmView';
import { findPath, pathPositions, type MoveIntent } from './pathfind';
import { fromUnitAction, type TurnDraft } from './useTurnDraft';

function unitPosition(state: GameState, player: number, unit: number): Position | null {
  const farm = state.farms[player];
  if (unit === 0) return farm.farmer;
  return farm.hands[unit - 1] ?? null;
}

export interface BoardPlay {
  selectedUnit: number;
  setSelectedUnit(unit: number): void;
  tapPos: Position | null;
  closeMenu(): void;
  onTileTap(x: number, y: number): void;
  overlays: OverlayMap;
  /** Verdicts for the tapped tile (for TileActionMenu). */
  tapVerdicts: UnitOpVerdicts | null;
  onSameTile: boolean;
  pathLength: number | null;
  unitsHere: { unit: number; label: string }[];
  moveHere(thenOp?: UnitAction): void;
  /** Number of units with an active movement intent. */
  activeIntents: number;
  clearIntents(): void;
}

export function useBoardPlay(state: GameState | null, player: number | null, draft: TurnDraft): BoardPlay {
  const [selectedUnit, setSelectedUnit] = useState(0);
  const [tapPos, setTapPos] = useState<Position | null>(null);
  const [intents, setIntents] = useState<Record<number, MoveIntent>>({});

  const numUnits = player !== null && state ? 1 + (state.farms[player]?.hands.length ?? 0) : 1;

  // Clamp selection when hands vanish at end of day.
  useEffect(() => {
    if (selectedUnit >= numUnits) setSelectedUnit(0);
  }, [numUnits, selectedUnit]);

  // Advance movement intents whenever a new state lands: write the next step
  // (or the arrival op) into the shared draft.
  useEffect(() => {
    if (!state || player === null) return;
    setIntents((prev) => {
      const next: Record<number, MoveIntent> = {};
      for (const [unitStr, intent] of Object.entries(prev)) {
        const unit = Number(unitStr);
        const pos = unitPosition(state, player, unit);
        if (!pos) continue; // hand disappeared
        if (pos[0] === intent.target[0] && pos[1] === intent.target[1]) {
          if (intent.thenOp) draft.setUnitDraft(unit, fromUnitAction(intent.thenOp));
          continue; // arrived — drop the intent
        }
        const path = findPath(state.farms[player], pos, intent.target);
        if (!path || path.length === 0) continue; // unreachable — drop
        draft.setUnitDraft(unit, { op: path[0] });
        next[unit] = intent;
      }
      return next;
    });
    // Draft setters are stable; state.step identifies a new turn.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.step, player]);

  const onTileTap = (x: number, y: number) => setTapPos([x, y]);
  const closeMenu = () => setTapPos(null);

  const selectedPos = state && player !== null ? unitPosition(state, player, selectedUnit) : null;

  const tapVerdicts = useMemo(() => {
    if (!state || player === null || !tapPos) return null;
    return legalUnitOpsAt(state, player, selectedUnit, tapPos);
  }, [state, player, selectedUnit, tapPos]);

  const onSameTile = !!(tapPos && selectedPos && tapPos[0] === selectedPos[0] && tapPos[1] === selectedPos[1]);

  const pathToTap = useMemo(() => {
    if (!state || player === null || !tapPos || !selectedPos || onSameTile) return null;
    return findPath(state.farms[player], selectedPos, tapPos);
  }, [state, player, tapPos, selectedPos, onSameTile]);

  const unitsHere = useMemo(() => {
    if (!state || player === null || !tapPos) return [];
    const out: { unit: number; label: string }[] = [];
    for (let u = 0; u < numUnits; u++) {
      const p = unitPosition(state, player, u);
      if (p && p[0] === tapPos[0] && p[1] === tapPos[1]) {
        out.push({ unit: u, label: u === 0 ? 'Farmer' : `Hand ${u}` });
      }
    }
    return out;
  }, [state, player, tapPos, numUnits]);

  const moveHere = (thenOp?: UnitAction) => {
    if (!state || player === null || !tapPos || !selectedPos) return;
    const path = findPath(state.farms[player], selectedPos, tapPos);
    if (!path) return;
    if (path.length === 0) {
      if (thenOp) draft.setUnitDraft(selectedUnit, fromUnitAction(thenOp));
      return;
    }
    draft.setUnitDraft(selectedUnit, { op: path[0] });
    setIntents((prev) => ({ ...prev, [selectedUnit]: { target: tapPos, thenOp } }));
  };

  const overlays = useMemo<OverlayMap>(() => {
    const out: OverlayMap = {};
    if (!state || player === null) return out;
    // Path previews for every active intent (selected unit's path brighter).
    for (const [unitStr, intent] of Object.entries(intents)) {
      const unit = Number(unitStr);
      const pos = unitPosition(state, player, unit);
      if (!pos) continue;
      const path = findPath(state.farms[player], pos, intent.target);
      if (!path) continue;
      for (const p of pathPositions(pos, path)) out[overlayKey(p[0], p[1])] = 'ov-path';
      out[overlayKey(intent.target[0], intent.target[1])] = 'ov-target';
    }
    if (tapPos) out[overlayKey(tapPos[0], tapPos[1])] = `${out[overlayKey(tapPos[0], tapPos[1])] ?? ''} ov-tap`.trim();
    if (selectedPos) {
      const k = overlayKey(selectedPos[0], selectedPos[1]);
      out[k] = `${out[k] ?? ''} ov-selected`.trim();
    }
    return out;
  }, [state, player, intents, tapPos, selectedPos]);

  return {
    selectedUnit,
    setSelectedUnit,
    tapPos,
    closeMenu,
    onTileTap,
    overlays,
    tapVerdicts,
    onSameTile,
    pathLength: pathToTap ? pathToTap.length : onSameTile ? 0 : null,
    unitsHere,
    moveHere,
    activeIntents: Object.keys(intents).length,
    clearIntents: () => setIntents({}),
  };
}
