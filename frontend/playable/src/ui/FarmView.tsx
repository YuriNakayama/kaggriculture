/**
 * React wrapper around the default visualizer's imperative render pipeline.
 * Mounts an empty <div>, builds the shell + cached refs once, then calls
 * `renderObservation` whenever the live GameState changes.
 *
 * Tap-to-act support (playable-only, `default/` stays unmodified):
 * - one delegated click listener on the human player's panel resolves taps
 *   to (x, y) via each cell's data-col/data-row;
 * - the otherwise-unused `.cell-overlay` layer carries highlight classes
 *   (selection ring, path preview, tap target) painted after every render.
 */

import { useEffect, useRef } from 'react';
import {
  buildShell,
  collectRefs,
  renderObservation,
  type BoardSize,
  type LayoutRefs,
} from '../../../default/src/renderFarm';
import type { Config, GameState } from '../engine/types';
import { createPriceHistory, stateToView, type PriceHistoryTracker } from './buildView';

/** "x,y" → extra class names for that cell's overlay layer. */
export type OverlayMap = Record<string, string>;

export const overlayKey = (x: number, y: number) => `${x},${y}`;

interface FarmViewProps {
  state: GameState;
  config: Config;
  playerNames: string[];
  /** Seat whose panel accepts taps and shows overlays. null = spectate. */
  humanPlayerId?: number | null;
  overlays?: OverlayMap;
  onTileTap?(x: number, y: number): void;
}

export function FarmView({ state, config, playerNames, humanPlayerId = null, overlays, onTileTap }: FarmViewProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const refsRef = useRef<LayoutRefs | null>(null);
  const historyRef = useRef<PriceHistoryTracker | null>(null);
  const shellKeyRef = useRef<string>('');
  const onTileTapRef = useRef(onTileTap);
  onTileTapRef.current = onTileTap;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const board: BoardSize = { rows: config.boardSize, cols: config.boardSize };
    const key = `${board.rows}x${board.cols}|${playerNames.join('|')}`;
    if (shellKeyRef.current !== key) {
      buildShell(host, board, playerNames);
      refsRef.current = collectRefs(host, board);
      shellKeyRef.current = key;
    }
    if (!historyRef.current) {
      historyRef.current = createPriceHistory(config.turnsPerDay);
    }
    const priceHistory = historyRef.current.record(state);
    const view = stateToView(state, priceHistory);
    if (refsRef.current) renderObservation(refsRef.current, view, config);
    if (refsRef.current) paintOverlays(refsRef.current, humanPlayerId, overlays);
  }, [state, config, playerNames, humanPlayerId, overlays]);

  // Delegated tap listener on the human player's grid (survives re-renders —
  // the shell is built once and renderObservation never replaces .cell nodes).
  useEffect(() => {
    const host = hostRef.current;
    if (!host || humanPlayerId === null) return;
    const panel = host.querySelector<HTMLElement>(`.farm-panel[data-player="${humanPlayerId + 1}"]`);
    const grid = panel?.querySelector<HTMLElement>('.farm-grid');
    if (!grid) return;
    const listener = (e: Event) => {
      const cell = (e.target as HTMLElement).closest<HTMLElement>('.cell');
      if (!cell || !grid.contains(cell)) return;
      const x = Number(cell.dataset.col);
      const y = Number(cell.dataset.row);
      if (Number.isFinite(x) && Number.isFinite(y)) onTileTapRef.current?.(x, y);
    };
    grid.addEventListener('click', listener);
    grid.classList.add('tappable-grid');
    return () => {
      grid.removeEventListener('click', listener);
      grid.classList.remove('tappable-grid');
    };
    // Re-attach only if the shell was rebuilt (key change) or seat changed.
  }, [humanPlayerId, config.boardSize, playerNames]);

  return <div ref={hostRef} className="farm-view-root" />;
}

function paintOverlays(refs: LayoutRefs, humanPlayerId: number | null, overlays?: OverlayMap): void {
  if (humanPlayerId === null) return;
  const player = refs.players[humanPlayerId];
  if (!player) return;
  for (let y = 0; y < player.cells.length; y++) {
    for (let x = 0; x < player.cells[y].length; x++) {
      const overlayEl = player.cells[y][x]?.el.querySelector<HTMLElement>('.cell-overlay');
      if (!overlayEl) continue;
      const extra = overlays?.[overlayKey(x, y)];
      const next = extra ? `cell-overlay ${extra}` : 'cell-overlay';
      if (overlayEl.className !== next) overlayEl.className = next;
    }
  }
}
