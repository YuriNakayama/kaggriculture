#!/usr/bin/env node
/**
 * Copy the built default (replay) visualizer — a single self-contained
 * index.html thanks to vite-plugin-singlefile — into the playable app's
 * public dir, where ReplayScreen iframes it at `replay/index.html`.
 */
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', 'default', 'dist', 'index.html');
const destDir = join(here, '..', 'playable', 'public', 'replay');

if (!existsSync(src)) {
  console.error(`copy-replay: ${src} not found — build the default visualizer first`);
  process.exit(1);
}
mkdirSync(destDir, { recursive: true });
cpSync(src, join(destDir, 'index.html'));
console.log(`copy-replay: → ${join(destDir, 'index.html')}`);
