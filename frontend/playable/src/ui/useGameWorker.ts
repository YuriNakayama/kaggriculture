/**
 * Hook that owns a WorkerClient and tracks the latest GameState it produces.
 * Re-mounts (new worker, fresh init) whenever the `setup` reference changes.
 */

import { useEffect, useRef, useState } from 'react';
import type { Config, GameState } from '../engine/types';
import { WorkerClient } from '../worker/workerClient';
import type { HumanActions, SlotConfig } from '../worker/protocol';

export interface SetupResult {
  config: Config;
  numAgents: number;
  slots: SlotConfig[];
  /** Human-action log to replay after INIT (session resume). */
  resumeLog?: HumanActions[];
}

const SESSION_KEY = 'kaggriculture:session';

export interface SavedSession {
  setup: Omit<SetupResult, 'resumeLog'>;
  log: HumanActions[];
  savedAt: number;
}

export function loadSavedSession(): SavedSession | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SavedSession;
    return parsed?.setup?.config && Array.isArray(parsed.log) ? parsed : null;
  } catch {
    return null;
  }
}

export function clearSavedSession(): void {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    /* private mode etc. — autosave is best-effort */
  }
}

function saveSession(setup: SetupResult, log: HumanActions[]): void {
  try {
    const { resumeLog: _drop, ...bare } = setup;
    localStorage.setItem(SESSION_KEY, JSON.stringify({ setup: bare, log, savedAt: Date.now() }));
  } catch {
    /* best-effort */
  }
}

export function useGameWorker(setup: SetupResult) {
  const clientRef = useRef<WorkerClient | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [agentErrors, setAgentErrors] = useState<Record<number, string>>({});

  const logRef = useRef<HumanActions[]>([]);

  useEffect(() => {
    const client = new WorkerClient(setProgress);
    clientRef.current = client;
    logRef.current = [];
    setBusy(true);
    setError(null);
    (async () => {
      let s = await client.init(setup.config, setup.numAgents, setup.slots);
      // Resume: replay the saved human-action log. AI moves are recomputed —
      // deterministic for the engine, Python cases and starter; the `random`
      // TS bot may diverge (it is not seeded).
      if (setup.resumeLog?.length) {
        for (let i = 0; i < setup.resumeLog.length; i++) {
          setProgress(`Resuming… turn ${i + 1}/${setup.resumeLog.length}`);
          s = await client.step(setup.resumeLog[i]);
          logRef.current.push(setup.resumeLog[i]);
        }
      }
      return s;
    })()
      .then((s) => {
        setState(s);
        setProgress(null);
        setBusy(false);
      })
      .catch((e) => {
        setError(String(e));
        setProgress(null);
        setBusy(false);
      });
    return () => {
      client.terminate();
      clientRef.current = null;
    };
  }, [setup]);

  const stepGame = async (humanActions: HumanActions): Promise<void> => {
    const client = clientRef.current;
    if (!client) return;
    setBusy(true);
    setError(null);
    try {
      const next = await client.step(humanActions);
      logRef.current.push(humanActions);
      if (next.done) clearSavedSession();
      else saveSession(setup, logRef.current);
      setState(next);
      setAgentErrors(client.lastAgentErrors);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = async (): Promise<void> => {
    const client = clientRef.current;
    if (!client) return;
    setBusy(true);
    setError(null);
    try {
      const next = await client.reset();
      logRef.current = [];
      clearSavedSession();
      setState(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return { state, busy, error, progress, agentErrors, stepGame, reset };
}
