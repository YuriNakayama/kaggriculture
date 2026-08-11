import { useEffect, useMemo, useRef, useState } from 'react';
import type { MarketOrder, PlayerAction, UnitAction } from '../engine/types';
import { ActionPanel } from './ActionPanel';
import { FarmView } from './FarmView';
import { GameOverModal } from './GameOverModal';
import { HUD } from './HUD';
import { mergeMarket, omakaseAction } from './omakase';
import { SmartCommandBar, type AutoStatus } from './SmartCommandBar';
import { useHandRoles } from './useHandRoles';
import { assignCommand, autoActions, commandTargets, queueSize, type SmartCommand, type TaskQueues } from './smartTasks';
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
  const handRoles = useHandRoles(state, humanPlayerId, draft);
  const [mobileFocusOwn, setMobileFocusOwn] = useState(true);
  const [omakase, setOmakase] = useState(false);

  const handleSubmit = (action: PlayerAction) => {
    if (humanPlayerId === null || !state) return;
    // おまかせ農場: 農作業と購入は starter 方針、販売はユーザーの注文を通す。
    let merged = action;
    if (omakase) {
      const auto = omakaseAction(state, humanPlayerId);
      merged = { farmer: auto.farmer, hands: auto.hands, market: mergeMarket(action.market, auto.autoMarket) };
    }
    void stepGame({ [humanPlayerId]: merged });
  };

  const handleAiStep = () => {
    if (humanPlayerId !== null) return;
    void stepGame({});
  };

  // --- Smart commands: expand task queues into one op per unit per turn and
  // auto-advance until done / morning / an attention event. ---
  const [auto, setAuto] = useState<AutoStatus>({ running: false, label: null, remaining: 0, stopNote: null });
  const autoStopRef = useRef(false);

  const runAuto = async (label: string, initialQueues: TaskQueues, untilMorning: boolean) => {
    if (humanPlayerId === null || !state || state.done) return;
    autoStopRef.current = false;
    let s = state;
    let queues = initialQueues;
    const startDay = s.day;
    const startHarvest = commandTargets(s, humanPlayerId, 'HARVEST_ALL').length;
    setAuto({ running: true, label, remaining: queueSize(queues), stopNote: null });
    let note: string | null = 'タスク完了';
    for (;;) {
      if (autoStopRef.current) {
        note = '手動停止';
        break;
      }
      if (s.done) {
        note = 'シーズン終了';
        break;
      }
      let action: { farmer: UnitAction; hands: UnitAction[]; market: MarketOrder[] };
      let nextQueues: TaskQueues;
      let idle: boolean;
      if (omakase) {
        const auto = omakaseAction(s, humanPlayerId);
        action = { farmer: auto.farmer, hands: auto.hands, market: auto.autoMarket };
        nextQueues = {};
        idle = false;
      } else {
        const r = autoActions(s, humanPlayerId, queues);
        action = { ...r.action, market: [] };
        nextQueues = r.nextQueues;
        idle = r.idle;
      }
      if (!untilMorning && idle) break;
      const ns = await stepGame({ [humanPlayerId]: action });
      if (!ns) {
        note = 'エラーで停止';
        break;
      }
      s = ns;
      queues = nextQueues;
      setAuto((a) => ({ ...a, remaining: queueSize(queues) }));
      if (untilMorning) {
        if (s.day !== startDay) {
          note = '朝になったので停止';
          break;
        }
        const harvestNow = commandTargets(s, humanPlayerId, 'HARVEST_ALL').length;
        if (harvestNow > startHarvest) {
          note = `収穫可能が ${harvestNow} 件になったので停止`;
          break;
        }
      } else if (queueSize(queues) === 0) {
        break;
      }
    }
    setAuto({ running: false, label: null, remaining: 0, stopNote: note });
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
      <HUD
        state={state}
        config={setup.config}
        humanPlayerId={humanPlayerId}
        busy={busy}
        error={error ?? (agentErrorBanner || null)}
        onReset={() => void reset()}
        onExit={onExit}
      />
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
          <div className="action-col">
            <SmartCommandBar
              state={state}
              player={humanPlayerId}
              busy={busy}
              auto={auto}
              onCommand={(cmd: SmartCommand) =>
                void runAuto(
                  cmd === 'DEPOSIT_ALL' ? '倉庫へ収納' : '一括作業',
                  assignCommand(state, humanPlayerId, cmd),
                  false
                )
              }
              onUntilMorning={() => void runAuto('翌朝まで自動進行', {}, true)}
              onStop={() => {
                autoStopRef.current = true;
              }}
            />
            <ActionPanel
              state={state}
              player={humanPlayerId}
              busy={busy || auto.running}
              draft={draft}
              roles={handRoles.roles}
              onRoleChange={handRoles.setRole}
              omakase={omakase}
              onOmakaseChange={setOmakase}
              onSubmit={handleSubmit}
            />
          </div>
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
