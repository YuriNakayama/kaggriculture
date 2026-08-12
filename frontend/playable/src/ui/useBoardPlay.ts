/**
 * Board-tap play state: which unit is selected, the open tile menu, and the
 * overlay highlights. Movement execution lives in GameScreen's auto-run
 * (smart-task queues) — tapping "move" starts an auto-run that advances turns
 * by itself; no per-turn Submit involved.
 */

import { useEffect, useMemo, useState } from 'react';
import { legalUnitOpsAt, type UnitOpVerdicts } from '../engine/legality';
import type { GameState, Position } from '../engine/types';
import { overlayKey, type OverlayMap } from './FarmView';
import { findPath, pathPositions } from './pathfind';

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
  tapVerdicts: UnitOpVerdicts | null;
  onSameTile: boolean;
  pathLength: number | null;
  unitsHere: { unit: number; label: string }[];
}

export function useBoardPlay(state: GameState | null, player: number | null): BoardPlay {
  const [selectedUnit, setSelectedUnit] = useState(0);
  const [tapPos, setTapPos] = useState<Position | null>(null);

  const numUnits = player !== null && state ? 1 + (state.farms[player]?.hands.length ?? 0) : 1;

  // Clamp selection when hands vanish at end of day.
  useEffect(() => {
    if (selectedUnit >= numUnits) setSelectedUnit(0);
  }, [numUnits, selectedUnit]);

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

  const overlays = useMemo<OverlayMap>(() => {
    const out: OverlayMap = {};
    if (!state || player === null) return out;
    // Preview the walking route while the menu is open.
    if (pathToTap && selectedPos) {
      for (const p of pathPositions(selectedPos, pathToTap)) out[overlayKey(p[0], p[1])] = 'ov-path';
    }
    if (tapPos) out[overlayKey(tapPos[0], tapPos[1])] = `${out[overlayKey(tapPos[0], tapPos[1])] ?? ''} ov-tap`.trim();
    if (selectedPos) {
      const k = overlayKey(selectedPos[0], selectedPos[1]);
      out[k] = `${out[k] ?? ''} ov-selected`.trim();
    }
    return out;
  }, [state, player, tapPos, selectedPos, pathToTap]);

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
  };
}
