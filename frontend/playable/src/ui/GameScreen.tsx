import { useEffect, useMemo, useRef, useState } from 'react';
import type { MarketOrder, PlayerAction, Position, UnitAction } from '../engine/types';
import { ActionPanel } from './ActionPanel';
import { FarmView, overlayKey, type OverlayMap } from './FarmView';
import { findPath, pathPositions } from './pathfind';
import { GameOverModal } from './GameOverModal';
import { HUD } from './HUD';
import { mergeMarket, omakaseAction } from './omakase';
import { SmartCommandBar, type AutoStatus } from './SmartCommandBar';
import { useHandRoles } from './useHandRoles';
import {
  assignCommand,
  autoActions,
  commandTargets,
  dailyRoutineQueues,
  queueSize,
  type SmartCommand,
  type TaskQueues,
} from './smartTasks';
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
  const board = useBoardPlay(state, humanPlayerId);
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
  // Live queues during an auto-run, for path overlays on the board.
  const [runQueues, setRunQueues] = useState<TaskQueues>({});

  const runAuto = async (
    label: string,
    initialQueues: TaskQueues,
    untilMorning: boolean,
    // 毎ターン、キューが尽きたら再生成する (1日オート用)。設定時は
    // 収穫可能での注意停止を行わない — ルーチン自身が収穫するため。
    regenerate?: (s: NonNullable<typeof state>) => TaskQueues
  ) => {
    if (humanPlayerId === null || !state || state.done) return;
    if (auto.running) return; // 二重起動防止 (自動進行中の追加タップは無視)
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
        if (regenerate && queueSize(queues) === 0) queues = regenerate(s);
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
      setRunQueues(queues);
      setAuto((a) => ({ ...a, remaining: queueSize(queues) }));
      if (untilMorning) {
        if (s.day !== startDay) {
          note = '朝になったので停止';
          break;
        }
        const harvestNow = commandTargets(s, humanPlayerId, 'HARVEST_ALL').length;
        if (!regenerate && harvestNow > startHarvest) {
          note = `収穫可能が ${harvestNow} 件になったので停止`;
          break;
        }
      } else if (queueSize(queues) === 0) {
        break;
      }
    }
    setRunQueues({});
    setAuto({ running: false, label: null, remaining: 0, stopNote: note });
  };

  // タップ操作の即時実行: 選んだ op でそのままターンを送る (Submit 不要)。
  const instantOp = (patch: Parameters<typeof draft.buildActionWith>[1]) => {
    if (humanPlayerId === null || !state || state.done || busy || auto.running) return;
    const action = draft.buildActionWith(board.selectedUnit, patch);
    draft.afterSubmit(action);
    void stepGame({ [humanPlayerId]: action });
  };

  // 移動 (+到着後 op) はタスク化して自動進行。thenOp なしは到着で即完了。
  const moveTo = (target: Position, thenOp?: UnitAction) => {
    void runAuto('移動中', { [board.selectedUnit]: [{ kind: 'op-at', target, op: thenOp ?? null }] }, false);
  };

  // Auto-run path overlays (merged under the tap/selection overlays).
  const queueOverlays = useMemo<OverlayMap>(() => {
    const out: OverlayMap = {};
    if (!state || humanPlayerId === null) return out;
    for (const [uStr, tasks] of Object.entries(runQueues)) {
      const u = Number(uStr);
      const pos = u === 0 ? state.farms[humanPlayerId].farmer : state.farms[humanPlayerId].hands[u - 1];
      const task = tasks[0];
      if (!pos || !task || task.kind !== 'op-at') continue;
      const path = findPath(state.farms[humanPlayerId], pos, task.target);
      if (!path) continue;
      for (const p of pathPositions(pos, path)) out[overlayKey(p[0], p[1])] = 'ov-path';
      out[overlayKey(task.target[0], task.target[1])] = 'ov-target';
    }
    return out;
  }, [state, humanPlayerId, runQueues]);

  const mergedOverlays = useMemo<OverlayMap>(
    () => ({ ...queueOverlays, ...board.overlays }),
    [queueOverlays, board.overlays]
  );

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
            overlays={mergedOverlays}
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
              onPick={instantOp}
              onMoveHere={(thenOp) => {
                if (board.tapPos) moveTo(board.tapPos, thenOp);
              }}
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
              onDayAuto={() =>
                void runAuto('1日オート (農作業を自動消化)', dailyRoutineQueues(state, humanPlayerId), true, (s) =>
                  dailyRoutineQueues(s, humanPlayerId)
                )
              }
              onNextTurn={() => {
                const action = draft.buildAction();
                draft.afterSubmit(action);
                handleSubmit(action);
              }}
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
