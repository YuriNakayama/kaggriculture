import { useRef } from 'react';
import { legalMarket } from '../engine/legality';
import type { Config, GameState } from '../engine/types';

interface Props {
  state: GameState;
  config: Config;
  humanPlayerId: number | null;
  busy: boolean;
  error: string | null;
  onReset(): void;
  onExit(): void;
}

export function HUD({ state, config, humanPlayerId, busy, error, onReset, onExit }: Props) {
  const totalDays = Math.ceil(config.episodeSteps / config.turnsPerDay);
  const turnsLeftToday = config.turnsPerDay - state.hour;
  const stepsLeft = config.episodeSteps - state.step;

  // Money delta since the previous rendered state (per player).
  const prevMoneyRef = useRef<number[]>(state.farms.map((f) => f.money));
  const deltas = state.farms.map((f, i) => f.money - (prevMoneyRef.current[i] ?? f.money));
  prevMoneyRef.current = state.farms.map((f) => f.money);

  const shedTotal =
    humanPlayerId !== null
      ? Object.values(state.privates[humanPlayerId].shed).reduce((a, b) => a + (b ?? 0), 0)
      : 0;
  const shedFullish = shedTotal >= config.shedCapacity * 0.8;
  const hireCost = humanPlayerId !== null ? legalMarket(state, humanPlayerId).hire.cost : null;
  const endgame = stepsLeft <= config.turnsPerDay * 3 && !state.done;

  return (
    <header className="hud">
      <div className="hud-meta">
        <span className="hud-clock" title={`残り ${stepsLeft} ターン`}>
          📅 Day {state.day + 1}/{totalDays} ・ {state.hour}時 (今日あと{turnsLeftToday})
        </span>
        {humanPlayerId !== null && (
          <span
            className={`hud-tag${shedFullish ? ' hud-tag-warn' : ''}`}
            title="倉庫使用量。日没時にあふれた持ち物は破棄される"
          >
            📦 {shedTotal}/{config.shedCapacity}
          </span>
        )}
        {hireCost !== null && (
          <span className="hud-tag" title="次の日雇いコスト (同日 n 人目でフィボナッチ増)">
            👷 ${hireCost}
          </span>
        )}
        {endgame && (
          <span className="hud-tag hud-tag-warn" title="シーズン終了時、倉庫・持ち物の在庫は無価値。売り切ろう">
            ⚠ 残り{stepsLeft}ターン — 在庫は無価値に
          </span>
        )}
        {busy && <span className="hud-tag">working…</span>}
        {state.done && <span className="hud-tag hud-tag-done">game over</span>}
        {error && <span className="hud-tag hud-tag-error">error: {error}</span>}
      </div>
      <div className="hud-scores">
        {state.farms.map((f, i) => (
          <div key={i} className="hud-score">
            <span className="hud-score-label">
              P{i + 1}
              {i === humanPlayerId ? ' (You)' : ''}
            </span>
            <span className="hud-score-value">
              ${f.money.toLocaleString()}
              {deltas[i] !== 0 && (
                <span className={deltas[i] > 0 ? 'hud-delta-up' : 'hud-delta-down'}>
                  {' '}
                  {deltas[i] > 0 ? '+' : ''}
                  {deltas[i].toLocaleString()}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
      <div className="hud-buttons">
        <button type="button" onClick={onReset}>
          Reset
        </button>
        <button type="button" onClick={onExit}>
          New Game
        </button>
      </div>
    </header>
  );
}
