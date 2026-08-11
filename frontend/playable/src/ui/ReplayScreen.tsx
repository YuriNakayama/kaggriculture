/**
 * Replay browser: drop (or pick) a replay JSON — a dev/simulate dump or a
 * Kaggle-scraped episode from data/lake/kaggle_episodes/replays/ — and it
 * plays in the official default visualizer, iframed from replay/index.html.
 *
 * The visualizer posts {ready: true} when it can accept data and then
 * receives {environment, agents} via postMessage (the same contract the
 * Kaggle notebook player uses).
 */

import { useEffect, useRef, useState } from 'react';

interface ReplayJson {
  steps?: unknown[][];
  info?: { Agents?: unknown[]; TeamNames?: string[] };
  rewards?: number[];
}

interface Props {
  onExit(): void;
}

export function ReplayScreen({ onExit }: Props) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const [replay, setReplay] = useState<ReplayJson | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [iframeReady, setIframeReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // The iframe announces readiness with {ready: true}; send the replay then
  // (and re-send if a new file is dropped later).
  useEffect(() => {
    const onMessage = (evt: MessageEvent) => {
      if (evt.source === iframeRef.current?.contentWindow && evt.data?.ready) {
        setIframeReady(true);
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, []);

  useEffect(() => {
    if (!iframeReady || !replay) return;
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    const playerCount = Array.isArray(replay.steps?.[0]) ? replay.steps![0].length : 2;
    const teamNames = replay.info?.TeamNames ?? [];
    const agents =
      replay.info?.Agents ??
      Array.from({ length: playerCount }, (_, i) => ({ index: i, name: teamNames[i] || `Player ${i + 1}` }));
    win.postMessage({ environment: replay, agents }, '*');
  }, [iframeReady, replay]);

  const loadFile = (file: File) => {
    setError(null);
    file
      .text()
      .then((text) => {
        const parsed = JSON.parse(text) as ReplayJson;
        if (!Array.isArray(parsed.steps)) throw new Error('no "steps" array — not a replay JSON');
        setReplay(parsed);
        setFileName(file.name);
      })
      .catch((e) => setError(`Could not load ${file.name}: ${e instanceof Error ? e.message : e}`));
  };

  return (
    <div className="replay-screen">
      <div className="replay-toolbar">
        <button type="button" onClick={onExit}>
          ← Back
        </button>
        <label className="replay-pick">
          Open replay JSON
          <input
            type="file"
            accept=".json,application/json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadFile(f);
              e.target.value = '';
            }}
          />
        </label>
        {fileName && <span className="replay-filename">{fileName}</span>}
        {error && <span className="replay-error">{error}</span>}
      </div>
      <div
        className={`replay-body${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) loadFile(f);
        }}
      >
        {!replay && (
          <div className="replay-hint">
            <p>Drop a replay JSON here (or use “Open replay JSON”).</p>
            <p className="replay-hint-sub">
              Sources: <code>dev/simulate --replay all</code> output, or scraped episodes under{' '}
              <code>data/lake/kaggle_episodes/replays/</code>.
            </p>
          </div>
        )}
        <iframe
          ref={iframeRef}
          src="replay/index.html"
          title="Replay visualizer"
          className="replay-frame"
          style={{ visibility: replay ? 'visible' : 'hidden' }}
        />
      </div>
    </div>
  );
}
