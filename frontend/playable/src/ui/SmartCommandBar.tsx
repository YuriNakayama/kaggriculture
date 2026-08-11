/**
 * スマートコマンドバー: 一括作業の指示と自動進行。「全部に水やり」等を押すと
 * タスクをユニットへ割り振り、完了まで自動でターンを消化する。注目イベント
 * (収穫可能の出現・日替わり) で自動停止する。
 */

import { commandTargets, type SmartCommand } from './smartTasks';
import type { GameState } from '../engine/types';

export interface AutoStatus {
  running: boolean;
  label: string | null;
  remaining: number;
  stopNote: string | null;
}

interface Props {
  state: GameState;
  player: number;
  busy: boolean;
  auto: AutoStatus;
  onCommand(cmd: SmartCommand): void;
  onUntilMorning(): void;
  onNextTurn(): void;
  onStop(): void;
}

const COMMANDS: { cmd: SmartCommand; label: string; help: string }[] = [
  { cmd: 'WATER_ALL', label: '🌊 水やり', help: '未給水の作物すべてに水やり (全ユニットで分担)' },
  { cmd: 'HARVEST_ALL', label: '🌾 収穫', help: '収穫可能な作物・生産物をすべて回収' },
  { cmd: 'TEND_ANIMALS', label: '🐄 世話', help: '未ケアの動物すべてを世話' },
  { cmd: 'CLEAR_WEEDS', label: '🧹 雑草', help: '雑草をすべて除去' },
  { cmd: 'DEPOSIT_ALL', label: '📦 収納', help: '全ユニットの持ち物を倉庫へ運ぶ' },
];

export function SmartCommandBar({ state, player, busy, auto, onCommand, onUntilMorning, onNextTurn, onStop }: Props) {
  return (
    <div className="smart-bar">
      <div className="smart-bar-title">スマートコマンド (自動で移動・実行)</div>
      <div className="smart-bar-buttons">
        {COMMANDS.map(({ cmd, label, help }) => {
          const n = cmd === 'DEPOSIT_ALL' ? undefined : commandTargets(state, player, cmd).length;
          return (
            <button
              key={cmd}
              type="button"
              title={help}
              disabled={busy || auto.running || state.done || n === 0}
              onClick={() => onCommand(cmd)}
            >
              {label}
              {n !== undefined ? ` (${n})` : ''}
            </button>
          );
        })}
        <button
          type="button"
          title="このターンを実行して1ターン進める (役割ハンド・保留中の市場注文もこのとき実行)"
          disabled={busy || auto.running || state.done}
          onClick={onNextTurn}
        >
          ⏭ 1ターン
        </button>
        <button
          type="button"
          title="タスクを消化しつつ翌朝まで自動でターンを進める (収穫可能が現れたら停止)"
          disabled={busy || auto.running || state.done}
          onClick={onUntilMorning}
        >
          ⏩ 翌朝まで
        </button>
      </div>
      {auto.running && (
        <div className="smart-bar-status">
          <span>
            ▶ {auto.label} — 残タスク {auto.remaining}
          </span>
          <button type="button" onClick={onStop}>
            ⏹ 停止
          </button>
        </div>
      )}
      {!auto.running && auto.stopNote && <div className="smart-bar-note">{auto.stopNote}</div>}
    </div>
  );
}
