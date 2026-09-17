// TASK13260.166. Browser execution is allowed only after the full-UAT entry gate.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { privateJson } from './matrix-launcher.mjs';

process.umask(0o077);
const packet = path.dirname(fileURLToPath(import.meta.url));
const installedCli = '/Users/macbook-dev/.npm/_npx/31e32ef8478fbf80/node_modules/@playwright/cli/playwright-cli.js';
const cells = new Set(['sqlite-single', 'sqlite-multi', 'pg-single', 'pg-multi']);
const inside = (file, root) => file.startsWith(root + path.sep);
const sha = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const write = (file, text) => fs.writeFileSync(file, text, { mode: 0o600, flag: 'wx' });
const requireThat = condition => { if (!condition) throw Error('Invalid browser profile or command'); };

function context(runId, cell, root) {
  requireThat(/^[A-Za-z0-9][A-Za-z0-9._-]{0,90}$/.test(runId || '') && cells.has(cell));
  root = fs.realpathSync(root);
  const session = `${runId}-${cell}`;
  const p = privateJson(path.join(root, `${session}.profile.private.json`));
  const expectedRoot = path.join(root, 'profiles', `tldw-onboarding-uat-${session}`);
  requireThat(p.runId === runId && p.name === cell && p.browserSessionName === session && p.root === expectedRoot);
  requireThat(p.preparationStatus === 'complete' && p.preparationId && /^[a-f0-9]{40}$/.test(p.sourceCommit));
  requireThat(fs.realpathSync(p.root) === expectedRoot && inside(fs.realpathSync(p.sourceRoot), root));
  requireThat(p.sourceRoot === fs.realpathSync(p.sourceRoot) && !fs.existsSync(path.join(p.sourceRoot, '.git')));
  requireThat(p.credentialsPath === path.join(expectedRoot, 'credentials.private.json'));
  requireThat(fs.realpathSync(p.credentialsPath) === p.credentialsPath);
  const initialized = privateJson(path.join(expectedRoot, 'initialized.private.json'));
  requireThat(initialized.status === 'completed' && initialized.code === 0 && initialized.token && initialized.preparationHash === sha(p));
  const c = privateJson(p.credentialsPath);
  requireThat(c.providerSecrets === undefined || Array.isArray(c.providerSecrets));
  const values = [c.apiKey, c.jwtSecret, c.apiHashSecret, ...Object.values(c.accounts || {}).map(a => a.password), ...(c.providerSecrets || [])];
  requireThat(values.length >= 4 && values.every(v => typeof v === 'string' && v.length > 0));
  if (cell.startsWith('pg-')) {
    const config = path.join(root, 'holders', session, 'runtime.pg-config.private.json');
    requireThat(p.pgConfigPath === config && fs.realpathSync(config) === config);
    const password = privateJson(config).password;
    requireThat(typeof password === 'string' && password.length > 0); values.push(password);
  }
  const secrets = [...new Set(values.flatMap(v => [v, encodeURIComponent(v), JSON.stringify(v).slice(1, -1)]))].sort((a, b) => b.length - a.length);
  return { profile: p, redact: text => secrets.reduce((s, value) => s.split(value).join('[REDACTED]'), text).replace(/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g, '[REDACTED JWT]') };
}

export function runBrowser(runId, cell, args, { packetRoot = packet, cli = installedCli, spawn = spawnSync } = {}) {
  requireThat(args.length > 0 && args.every(a => typeof a === 'string') && !args.some(a => /^-s|^--session/.test(a)));
  const { profile: p, redact } = context(runId, cell, packetRoot);
  const evidence = path.join(p.root, 'browser-evidence');
  fs.mkdirSync(evidence, { recursive: true, mode: 0o700 });
  requireThat(fs.realpathSync(evidence) === evidence);
  fs.chmodSync(evidence, 0o700);
  const prefix = path.join(evidence, `${Date.now()}-${crypto.randomUUID()}`);
  const result = spawn(process.execPath, [cli, `-s=${p.browserSessionName}`, ...args], { cwd: p.sourceRoot, encoding: 'utf8', timeout: 55000, maxBuffer: 20 * 1024 * 1024 });
  const raw = (result.stdout || '') + (result.stderr || ''), rawPath = `${prefix}.private.txt`;
  write(rawPath, raw);
  if (result.error || result.signal || result.status === null) return { code: 1, output: 'Browser subprocess failed; partial output retained privately.', rawPath };
  let output = redact(raw), code = result.status ?? 1, index = 0;
  const refs = new Set([...raw.matchAll(/\.playwright-cli\/[A-Za-z0-9._/-]+\.yml/g)].map(m => m[0]));
  for (const ref of refs) {
    try {
      const snapshotRoot = path.join(p.sourceRoot, '.playwright-cli');
      requireThat(fs.realpathSync(snapshotRoot) === snapshotRoot);
      const source = fs.realpathSync(path.resolve(p.sourceRoot, ref));
      requireThat(inside(source, snapshotRoot) && fs.statSync(source).isFile());
      const text = redact(fs.readFileSync(source, 'utf8'));
      const destination = `${prefix}-${++index}.snapshot.txt`;
      write(destination, text);
      output = output.split(ref).join(destination) + '\n' + text;
    } catch {
      output = output.split(ref).join('[snapshot unavailable within owned archive]');
      if (code === 0) code = 1;
    }
  }
  return { code, output, rawPath };
}

export function main(argv, options) {
  try {
    const [runId, cell, ...args] = argv;
    return runBrowser(runId, cell, args, options);
  } catch {
    return { code: 1, output: 'Browser command setup failed; check the owned initialized profile and arguments.' };
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = main(process.argv.slice(2));
  console.log(result.output); process.exitCode = result.code;
}
