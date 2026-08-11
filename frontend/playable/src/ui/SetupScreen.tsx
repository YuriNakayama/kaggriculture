import { useEffect, useId, useState } from 'react';
import { AGENTS, DEFAULT_AGENT_ID } from '../ai';
import { resolveConfig } from '../engine/state';
import { fetchManifest, type CaseManifestEntry } from '../pyagent/loader';
import type { SlotConfig } from '../worker/protocol';
import { loadSavedSession, type SavedSession, type SetupResult } from './useGameWorker';
import strawberryUrl from '../../../default/src/assets/sprites/market_strawberry.png';
import woodBgUrl from '../../../default/src/assets/sprites/wood_bg.svg';

type SlotPick = { kind: 'human' } | { kind: 'ai'; agentId: string } | { kind: 'py'; caseId: string };

const DEFAULT_SLOTS: SlotPick[] = [{ kind: 'human' }, { kind: 'ai', agentId: DEFAULT_AGENT_ID }];

interface Props {
  onStart(result: SetupResult): void;
  onReplays?(): void;
}

const DAY_OPTIONS = [1, 3, 5, 10, 30] as const;
const DEFAULT_DAYS = 30;

export function SetupScreen({ onStart, onReplays }: Props) {
  const [slots, setSlots] = useState<SlotPick[]>(DEFAULT_SLOTS);
  const [seedText, setSeedText] = useState('');
  const [days, setDays] = useState<number>(DEFAULT_DAYS);
  const [pyCases, setPyCases] = useState<CaseManifestEntry[]>([]);
  const [saved, setSaved] = useState<SavedSession | null>(null);

  useEffect(() => {
    setSaved(loadSavedSession());
  }, []);

  useEffect(() => {
    // Python cases are optional — if the manifest is absent (e.g. collect-agents
    // not run), the dropdown simply offers the built-in TS bots.
    fetchManifest()
      .then(setPyCases)
      .catch(() => setPyCases([]));
  }, []);
  const idPrefix = useId();
  const daysId = `${idPrefix}-days`;
  const seedId = `${idPrefix}-seed`;

  const setSlot = (idx: number, slot: SlotPick) => {
    setSlots((prev) => {
      const next = [...prev];
      next[idx] = slot;
      return next;
    });
  };

  const handleStart = () => {
    const seedNum = seedText.trim() === '' ? Math.floor(Math.random() * 0x7fffffff) : Number(seedText);
    const seed = Number.isFinite(seedNum) ? Math.floor(seedNum) : Math.floor(Math.random() * 0x7fffffff);
    const base = resolveConfig({ seed });
    const config = { ...base, episodeSteps: days * base.turnsPerDay };
    const finalSlots: SlotConfig[] = slots.map((s): SlotConfig => {
      if (s.kind === 'human') return { kind: 'human' };
      if (s.kind === 'py') return { kind: 'py', caseId: s.caseId };
      return { kind: 'ai', agentId: s.agentId };
    });
    onStart({ config, numAgents: finalSlots.length, slots: finalSlots });
  };

  return (
    <div className="setup sketched-border" style={{ backgroundImage: `url(${woodBgUrl})` }}>
      <h1>
        <img className="setup-strawberry" src={strawberryUrl} alt="" />
        Kaggriculture
      </h1>
      <p className="setup-sub">Pick who controls each farm, then start.</p>

      {slots.map((slot, i) => {
        const slotId = `${idPrefix}-slot-${i}`;
        return (
          <div key={i} className="setup-row">
            <label htmlFor={slotId}>Player {i + 1}</label>
            <select
              id={slotId}
              value={slot.kind === 'human' ? '__human' : slot.kind === 'py' ? `py:${slot.caseId}` : slot.agentId}
              onChange={(e) => {
                const v = e.target.value;
                if (v === '__human') setSlot(i, { kind: 'human' });
                else if (v.startsWith('py:')) setSlot(i, { kind: 'py', caseId: v.slice(3) });
                else setSlot(i, { kind: 'ai', agentId: v });
              }}
            >
              <option value="__human">Human</option>
              {Object.entries(AGENTS).map(([id, info]) => (
                <option key={id} value={id}>
                  AI — {info.label}
                </option>
              ))}
              {pyCases.map((c) => (
                <option key={c.id} value={`py:${c.id}`}>
                  AI (Python) — {c.label}
                </option>
              ))}
            </select>
          </div>
        );
      })}

      <div className="setup-row">
        <label htmlFor={daysId}>Days</label>
        <select id={daysId} value={days} onChange={(e) => setDays(Number(e.target.value))}>
          {DAY_OPTIONS.map((d) => (
            <option key={d} value={d}>
              {d} {d === 1 ? 'day' : 'days'}
            </option>
          ))}
        </select>
      </div>

      <div className="setup-row">
        <label htmlFor={seedId}>Seed</label>
        <input
          id={seedId}
          type="text"
          placeholder="random"
          value={seedText}
          onChange={(e) => setSeedText(e.target.value)}
          style={{ width: 120 }}
        />
      </div>

      <div className="setup-row setup-actions">
        {onReplays && (
          <button className="setup-start" onClick={onReplays}>
            Replays
          </button>
        )}
        {saved && (
          <button
            className="setup-start"
            onClick={() => onStart({ ...saved.setup, resumeLog: saved.log })}
            title={`Saved ${new Date(saved.savedAt).toLocaleString()} — turn ${saved.log.length}`}
          >
            Resume (turn {saved.log.length})
          </button>
        )}
        <button className="setup-start" onClick={handleStart}>
          Start Game
        </button>
      </div>
    </div>
  );
}
