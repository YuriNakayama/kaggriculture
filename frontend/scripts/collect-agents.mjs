#!/usr/bin/env node
/**
 * Collect Python agent cases from backend/pipeline into the playable app's
 * public assets, and emit a manifest the UI reads for its agent dropdown.
 *
 *   backend/pipeline/<family>/<case>/*.py  →  playable/public/agents/<family>__<case>/
 *                                             playable/public/agents/manifest.json
 *
 * Run automatically by dev/play and the Amplify build. New cases appear in the
 * UI on the next build with no frontend changes.
 */

import { cpSync, existsSync, mkdirSync, readdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const pipelineDir = join(repoRoot, 'backend', 'pipeline');
const outDir = join(here, '..', 'playable', 'public', 'agents');

// Modules Pyodide cannot satisfy (C extensions etc.). numpy IS available via
// loadPackage, so it is allowed; anything else non-stdlib gets a warning.
const KNOWN_OK = new Set(['numpy']);

function pyFilesIn(dir) {
  return readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isFile() && e.name.endsWith('.py'))
    .map((e) => e.name);
}

function scanImports(dir, files) {
  const found = new Set();
  for (const f of files) {
    const src = readFileSync(join(dir, f), 'utf8');
    for (const m of src.matchAll(/^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)/gm)) {
      found.add(m[1]);
    }
  }
  return found;
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const manifest = [];
if (!existsSync(pipelineDir)) {
  console.error(`pipeline dir not found: ${pipelineDir}`);
  process.exit(1);
}

for (const family of readdirSync(pipelineDir, { withFileTypes: true })) {
  if (!family.isDirectory() || family.name.startsWith('_')) continue;
  const familyDir = join(pipelineDir, family.name);
  for (const c of readdirSync(familyDir, { withFileTypes: true })) {
    if (!c.isDirectory() || c.name.startsWith('_')) continue;
    const caseDir = join(familyDir, c.name);
    if (!existsSync(join(caseDir, 'main.py'))) continue;

    const files = pyFilesIn(caseDir);
    const slug = `${family.name}__${c.name}`;
    const dest = join(outDir, slug);
    mkdirSync(dest, { recursive: true });
    for (const f of files) cpSync(join(caseDir, f), join(dest, f));

    const localModules = new Set(files.map((f) => f.replace(/\.py$/, '')));
    const stdlibOrLocal = (name) =>
      localModules.has(name) || KNOWN_OK.has(name) || !['polars', 'pyarrow', 'torch', 'pandas', 'scipy', 'sklearn'].includes(name);
    const imports = [...scanImports(caseDir, files)];
    const suspect = imports.filter((i) => !stdlibOrLocal(i));
    if (suspect.length) {
      console.warn(`WARN ${slug}: imports likely unavailable in Pyodide: ${suspect.join(', ')}`);
    }
    const needsNumpy = imports.includes('numpy');

    manifest.push({
      id: `${family.name}/${c.name}`,
      label: `${family.name}/${c.name}`,
      dir: `agents/${slug}`,
      files,
      needsNumpy,
    });
  }
}

manifest.sort((a, b) => a.id.localeCompare(b.id));
writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`collect-agents: ${manifest.length} case(s) → ${outDir}`);
for (const m of manifest) console.log(`  - ${m.id} (${m.files.length} files${m.needsNumpy ? ', numpy' : ''})`);
