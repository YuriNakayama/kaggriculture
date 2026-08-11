import { useEffect, useMemo, useRef, useState } from 'react';
import type { PlayerAction } from '../engine/types';
import { ActionPanel } from './ActionPanel';
import { FarmView } from './FarmView';
import { GameOverModal } from './GameOverModal';
import { HUD } from './HUD';
import { TileActionMenu } from './TileActionMenu';
import { useBoardPlay } from './useBoardPlay';
import { useGameWorker, type SetupResult } from './useGameWorker';
import { useTurnDraft } from './useTurnDraft';

interface Props {
  setup: SetupResult;
  onExit(): void;
}

export function GameScreen({ setup, onExit }: Props) {
  const { state, busy, error, progress, agentErrors, stepGame, reset } = useGameWorker(setup);

  const humanPlayerId = useMemo<number | null>(() => {
    const idx = setup.slots.findIndex((s) => s.kind === 'human');
    return idx >= 0 ? idx : null;
  }, [setup.slots]);

  const playerNames = useMemo(
    () =>
      setup.slots.map((s, i) => {
        if (s.kind === 'human') return `Player ${i + 1} (You)`;
        return `Player ${i + 1} (${s.kind === 'py' ? s.caseId : s.agentId})`;
      }),
    [setup.slots]
  );

  const draft = useTurnDraft(state, humanPlayerId);
  const board = useBoardPlay(state, humanPlayerId, draft);
  const [mobileFocusOwn, setMobileFocusOwn] = useState(true);

  const handleSubmit = (action: PlayerAction) => {
    if (humanPlayerId === null) return;
    void stepGame({ [humanPlayerId]: action });
  };

  const handleAiStep = () => {
    if (humanPlayerId !== null) return;
    void stepGame({});
  };

  // --- AI-vs-AI spectate: autoplay with a speed control ---
  const [playing, setPlaying] = useState(false);
  const [turnsPerSec, setTurnsPerSec] = useState(4);
  const busyRef = useRef(busy);
  busyRef.current = busy;
  const done = state?.done ?? false;

  useEffect(() => {
    if (!playing || humanPlayerId !== null || done) return;
    const id = setInterval(() => {
      if (!busyRef.current) void stepGame({});
    }, 1000 / turnsPerSec);
    return () => clearInterval(id);
    // stepGame is stable enough for this loop; re-arm on speed/state changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, turnsPerSec, humanPlayerId, done]);

  useEffect(() => {
    if (done) setPlaying(false);
  }, [done]);

  if (error && !state) {
    return <div className="placeholder">Worker error: {error}</div>;
  }
  if (!state) return <div className="placeholder">{progress ?? 'Initializing engine…'}</div>;

  const agentErrorBanner = Object.entries(agentErrors)
    .map(([pid, msg]) => `Player ${Number(pid) + 1} agent raised (fell back to PASS): ${msg.trim().split('\n').pop()}`)
    .join(' / ');

  return (
    <>
      <HUD state={state} busy={busy} error={error ?? (agentErrorBanner || null)} onReset={() => void reset()} onExit={onExit} />
      <div className="game-body">
        <div
          className={`game-main${
            humanPlayerId !== null && mobileFocusOwn ? ` mobile-focus-p${humanPlayerId + 1}` : ''
          }`}
        >
          {humanPlayerId !== null && (
            <button
              type="button"
              className="mobile-view-toggle"
              onClick={() => setMobileFocusOwn((v) => !v)}
              title="自分の畑だけ拡大表示 / 全体表示を切替 (モバイル)"
            >
              {mobileFocusOwn ? '🔍 自分の畑のみ' : '👀 全体表示'}
            </button>
          )}
          <FarmView
            state={state}
            config={setup.config}
            playerNames={playerNames}
            humanPlayerId={humanPlayerId}
            overlays={board.overlays}
            onTileTap={board.onTileTap}
          />
          {humanPlayerId !== null && board.tapPos && board.tapVerdicts && (
            <TileActionMenu
              state={state}
              player={humanPlayerId}
              unit={board.selectedUnit}
              unitLabel={board.selectedUnit === 0 ? 'Farmer' : `Hand ${board.selectedUnit}`}
              pos={board.tapPos}
              verdicts={board.tapVerdicts}
              onSameTile={board.onSameTile}
              pathLength={board.pathLength}
              unitsHere={board.unitsHere}
              onPick={(patch) => draft.setUnitDraft(board.selectedUnit, patch)}
              onMoveHere={board.moveHere}
              onSelectUnit={(u) => board.setSelectedUnit(u)}
              onClose={board.closeMenu}
            />
          )}
        </div>
        {humanPlayerId !== null ? (
          <ActionPanel state={state} player={humanPlayerId} busy={busy} draft={draft} onSubmit={handleSubmit} />
        ) : (
          <aside className="action-panel">
            <h3>AI vs AI</h3>
            <p className="action-hints">Spectate mode — play automatically or step one turn at a time.</p>
            <div className="action-row">
              <button type="button" onClick={() => setPlaying((p) => !p)} disabled={state.done}>
                {playing ? 'Pause' : 'Play'}
              </button>
              <button type="button" onClick={handleAiStep} disabled={busy || state.done || playing}>
                Step
              </button>
            </div>
            <div className="action-row">
              <label className="action-label" htmlFor="spectate-speed">
                Speed
              </label>
              <input
                id="spectate-speed"
                type="range"
                min={1}
                max={24}
                value={turnsPerSec}
                onChange={(e) => setTurnsPerSec(Number(e.target.value))}
              />
              <span className="action-hints">{turnsPerSec} turns/s</span>
            </div>
            <div className="action-hints">
              Day {state.day + 1}, hour {state.hour} — step {state.step}
              {state.done ? ' (finished)' : ''}
            </div>
            <div className="spectate-scores">
              {state.farms.map((f, i) => (
                <div key={i} className="spectate-score">
                  <span>{playerNames[i]}</span>
                  <strong>${f.money.toLocaleString()}</strong>
                </div>
              ))}
            </div>
          </aside>
        )}
      </div>
      {state.done && (
        <GameOverModal
          state={state}
          slots={setup.slots}
          humanPlayerId={humanPlayerId}
          onReplay={() => void reset()}
          onExit={onExit}
        />
      )}
    </>
  );
}
