/**
 * Pyodide bridge: run the repo's Python agents (backend/pipeline/<family>/<case>)
 * unmodified inside the game worker.
 *
 * Loading reproduces the Kaggle harness semantics: the case dir goes first on
 * sys.path, main.py is exec'd without __name__/__file__, and the LAST callable
 * defined in the module globals becomes the agent. Case-local modules are
 * evicted from sys.modules after load so multi-module cases with identical
 * module names (case2 and case7 both have config.py etc.) can coexist.
 *
 * Per-turn calls cross the JS/Python boundary as JSON strings — no proxies to
 * leak, and the exact same dict shapes the harness delivers. An agent that
 * raises falls back to all-PASS, mirroring the competition contract where an
 * exception would forfeit; the error is surfaced to the UI instead.
 */

import type { Observation } from '../ai/types';
import type { PlayerAction } from '../engine/types';

export const PYODIDE_VERSION = '314.0.4';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.mjs`;

export interface CaseManifestEntry {
  id: string;
  label: string;
  dir: string;
  files: string[];
  needsNumpy: boolean;
}

export interface PyAgentHandle {
  /** Returns the action, or PASS + error message if the agent raised. */
  call(obs: Observation): { action: PlayerAction; error?: string };
}

interface PyodideApi {
  FS: { mkdirTree(path: string): void; writeFile(path: string, data: string): void };
  loadPackage(name: string): Promise<void>;
  runPython(code: string): unknown;
  globals: { get(name: string): { (...args: unknown[]): unknown; destroy?(): void } };
}

let pyodidePromise: Promise<PyodideApi> | null = null;
let manifestPromise: Promise<CaseManifestEntry[]> | null = null;

const PASS: PlayerAction = { farmer: ['PASS'], hands: [], market: [] };

export const LOADER_PY = `
import json, sys, traceback

_agents = {}

def _load_case(case_id, case_dir):
    saved_modules = set(sys.modules)
    sys.path.insert(0, case_dir)
    try:
        with open(case_dir + "/main.py") as f:
            src = f.read()
        g = {}
        exec(compile(src, "main.py", "exec"), g)
        callables = [v for v in g.values() if callable(v)]
        if not callables:
            raise RuntimeError("no callable defined in main.py")
        _agents[case_id] = callables[-1]
    finally:
        sys.path.remove(case_dir)
        for m in [m for m in sys.modules if m not in saved_modules]:
            del sys.modules[m]

def _call_agent(case_id, obs_json):
    try:
        obs = json.loads(obs_json)
        action = _agents[case_id](obs)
        return json.dumps({"action": action})
    except Exception:
        return json.dumps({
            "action": {"farmer": ["PASS"], "hands": [], "market": []},
            "error": traceback.format_exc(limit=5),
        })
`;

export async function fetchManifest(): Promise<CaseManifestEntry[]> {
  manifestPromise ??= fetch('agents/manifest.json').then((r) => {
    if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
    return r.json() as Promise<CaseManifestEntry[]>;
  });
  return manifestPromise;
}

async function getPyodide(onProgress?: (msg: string) => void): Promise<PyodideApi> {
  pyodidePromise ??= (async () => {
    onProgress?.('Downloading Python runtime…');
    const mod = (await import(/* @vite-ignore */ PYODIDE_URL)) as {
      loadPyodide(opts: { indexURL: string }): Promise<PyodideApi>;
    };
    onProgress?.('Starting Python runtime…');
    const py = await mod.loadPyodide({ indexURL: PYODIDE_URL.replace(/pyodide\.mjs$/, '') });
    py.runPython(LOADER_PY);
    return py;
  })();
  return pyodidePromise;
}

export async function loadPyAgent(
  caseId: string,
  onProgress?: (msg: string) => void
): Promise<PyAgentHandle> {
  const [py, manifest] = await Promise.all([getPyodide(onProgress), fetchManifest()]);
  const entry = manifest.find((m) => m.id === caseId);
  if (!entry) throw new Error(`unknown Python agent: ${caseId}`);

  if (entry.needsNumpy) {
    onProgress?.('Loading numpy…');
    await py.loadPackage('numpy');
  }

  onProgress?.(`Loading ${caseId}…`);
  const caseDir = `/cases/${entry.dir.replace(/[^A-Za-z0-9_/]/g, '_')}`;
  py.FS.mkdirTree(caseDir);
  for (const file of entry.files) {
    const res = await fetch(`${entry.dir}/${file}`);
    if (!res.ok) throw new Error(`fetch ${entry.dir}/${file}: ${res.status}`);
    py.FS.writeFile(`${caseDir}/${file}`, await res.text());
  }
  const loadCase = py.globals.get('_load_case');
  try {
    loadCase(caseId, caseDir);
  } finally {
    loadCase.destroy?.();
  }

  const callAgent = py.globals.get('_call_agent');
  return {
    call(obs: Observation) {
      const raw = callAgent(caseId, JSON.stringify(obs)) as string;
      const parsed = JSON.parse(raw) as { action: PlayerAction; error?: string };
      return parsed.error ? { action: PASS, error: parsed.error } : { action: parsed.action };
    },
  };
}
