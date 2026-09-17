// TASK13260.196: only synthetic files, import inspection, sockets and children.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import vm from 'node:vm';
import { EventEmitter } from 'node:events';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const controller = path.resolve(here, '../../uat-next-matrix-20260916');
const launcherSource = fs.readFileSync(process.env.UPGRADE_TEST_LAUNCHER || path.join(controller, 'matrix-launcher.mjs'), 'utf8');
const upgradeFile = process.env.UPGRADE_TEST_SOURCE || path.join(controller, 'matrix-upgrade.mjs');
// A missing action has no behavior: retain assertion failures, not import errors, for the initial RED.
const upgradeSource = fs.existsSync(upgradeFile) ? fs.readFileSync(upgradeFile, 'utf8') : 'export async function launchUpgrade() {}';
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const plain = value => JSON.parse(JSON.stringify(value));
const write = (file, value) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, typeof value === 'string' ? value : JSON.stringify(value), { mode: 0o600 }); fs.chmodSync(file, 0o600); };

async function fixture(t, cell = 'pg-single') {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'upgrade196-')));
  write(path.join(root, 'matrix-launcher.mjs'), launcherSource);
  write(path.join(root, 'matrix-upgrade.mjs'), upgradeSource);
  const originalRun = 'old-run', upgradeRun = 'new-code';
  const oldRoot = path.join(root, 'repair-sources', originalRun, cell), newRoot = path.join(root, 'repair-sources', upgradeRun, cell);
  for (const source of [oldRoot, newRoot]) {
    write(path.join(source, 'entry.py'), 'frozen entry\n');
    write(path.join(source, 'apps/tldw-frontend/package.json'), '{}');
  }
  const inspect = options => {
    const s = options['source-root'];
    return { sourceRoot: s, sourceCommit: options['source-commit'], pythonVenv: path.join(root, 'dependencies/python-venv'), python: path.join(root, 'dependencies/python-venv/bin/python'), sourcePaths: [s, path.join(s, 'apps/mcp-unified/src'), path.join(s, 'packages/tldw_profile_core/src')], sourceHashes: { 'entry.py': sha(fs.readFileSync(path.join(s, 'entry.py'))) }, pythonOrigins: { app: path.join(s, 'entry.py') }, frontend: path.join(s, 'apps/tldw-frontend'), nextCli: path.join(s, 'apps/tldw-frontend/node_modules/next/dist/bin/next'), dependencyOrigins: { next: path.join(s, 'apps/tldw-frontend/node_modules/next/package.json') } };
  };
  const runtime = path.join(root, 'profiles', `tldw-onboarding-uat-${originalRun}-${cell}`);
  const p = { ...inspect({ 'source-root': oldRoot, 'source-commit': 'a'.repeat(40) }), name: cell, runId: originalRun, root: runtime, repoRoot: oldRoot, preparationStatus: 'complete', preparationId: 'original-preparation', spec: { mode: cell === 'pg-single' ? 'single_user' : 'multi_user', engine: 'postgresql', api: 19101, web: 19181 }, configPath: path.join(runtime, 'config/config.txt'), envPath: path.join(runtime, 'config/.env'), credentialsPath: path.join(runtime, 'credentials.private.json'), databaseDir: path.join(runtime, 'Databases'), userDatabasesDir: path.join(runtime, 'Databases/users'), databasePaths: { evaluations: path.join(runtime, 'Databases/evals.db') }, nextDistDir: '.next-old', browserSessionName: `${originalRun}-${cell}`, pgConfigPath: path.join(root, 'holders', `${originalRun}-${cell}`, 'runtime.pg-config.private.json'), pgReceiptPath: path.join(root, 'holders', `${originalRun}-${cell}`, `${cell}.pg-receipt.private.json`) };
  const cfg = { host: '127.0.0.1', port: 55475, user: 'tldw_matrix_1234567890abcdef', password: 'synthetic-pg-only', purpose: 'matrix-runtime', cell, run_id: originalRun };
  const holder = { profile: cell, run_id: originalRun, source_root: oldRoot, source_commit: p.sourceCommit, python_venv: p.pythonVenv, status: 'held', pid: 1234, auth_fixture: 'pg_temp_db', content_fixture: 'pg_temp_db_session', provisioning: { user: 'provisioner-only', host: cfg.host, port: cfg.port }, runtime_role: { name: cfg.user, login: true, memberships: 0, superuser: false, bypassrls: false, inherit: false, createdb: false, createrole: false, replication: false }, auth: { ...cfg, database: 'tldw_test_aaaaaaaa' }, content: { ...cfg, database: 'tldw_test_bbbbbbbb' } };
  const record = path.join(root, `${originalRun}-${cell}.profile.private.json`);
  write(record, p); write(p.pgConfigPath, cfg); write(p.pgReceiptPath, holder);
  write(p.credentialsPath, { apiKey: 'synthetic-api-only', jwtSecret: 'synthetic-jwt-only', apiHashSecret: 'synthetic-hash-only' });
  write(p.configPath, '[Setup]\nsetup_completed = true\n'); write(p.envPath, 'SYNTHETIC_SAVED_ENV=unchanged\n');
  const initialized = path.join(runtime, 'initialized.private.json');
  write(initialized, { status: 'completed', code: 0, token: 'old-initializer', preparationHash: sha(JSON.stringify(p)) });
  write(path.join(oldRoot, 'Databases/system_ops.json.lock'), '');
  const preparation = path.join(root, 'repair-sources', upgradeRun, 'preparation'), revision = 'b'.repeat(40);
  const tar = path.join(preparation, `${revision}.tar`); write(tar, 'synthetic archive bytes');
  const manifest = { root: newRoot, revision, fileCount: 2, files: ['entry.py', 'apps/tldw-frontend/package.json'].map(file => ({ path: file, kind: 'file', bytes: fs.statSync(path.join(newRoot, file)).size, sha256: sha(fs.readFileSync(path.join(newRoot, file))) })) };
  write(path.join(preparation, `${cell}-source-manifest.json`), manifest);
  const gate = { purpose: 'targeted-acceptance', status: 'RELEASED', revision, runId: upgradeRun, cells: ['pg-single', 'pg-multi'], dataPolicy: 'existing-profile-upgrade', originalRunId: originalRun };
  write(path.join(preparation, 'gate.json'), gate);
  write(path.join(preparation, 'complete.json'), { purpose: gate.purpose, revision, runId: upgradeRun, cells: gate.cells, archive: tar, archiveSha256: sha(fs.readFileSync(tar)), noProfileFixtureRuntimeOrBrowserStarted: true });
  write(path.join(preparation, 'python-reuse.json'), { pythonVenv: p.pythonVenv, reused: true, copiedOrModified: false });
  const calls = { spawn: [], inspect: [], role: [], ports: [], helpers: [], signals: [] };
  const h = { roleStatus: 0, inspect: options => { calls.inspect.push(plain(options)); if (h.originFailure) throw Error('Synthetic origin rejection'); return inspect(options); }, helpers: async source => { calls.helpers.push(source); return { buildLiveTierBackendEnv: ({ profile }) => ({ TLDW_CONFIG_FILE: profile.configPath, TLDW_ENV_FILE: profile.envPath, USER_DB_BASE_DIR: profile.userDatabasesDir, ACP_RUNNER_CWD: path.join(profile.sourceRoot, 'tools/tldw-agent') }) }; } };
  const fakeProcess = new EventEmitter();
  Object.assign(fakeProcess, { argv: [], env: {}, execPath: '/synthetic/node', umask() {}, kill(pid, signal) { calls.signals.push({ pid, signal }); }, exitCode: undefined });
  const context = vm.createContext({ process: fakeProcess, console: { log() {}, error() {} }, __harness: h, Buffer });
  const builtins = new Map();
  const link = async name => {
    if (name === './matrix-launcher.mjs') return matrix;
    if (builtins.has(name)) return builtins.get(name);
    let values;
    if (name === 'node:child_process') values = { spawn: (command, args, options) => {
      if (h.spawnThrows) throw Error('Synthetic spawn failure');
      const child = new EventEmitter(); Object.assign(child, { pid: 4321, kill: signal => calls.signals.push({ pid: child.pid, signal }) });
      calls.spawn.push({ command, args: plain(args), options, child }); return child;
    }, spawnSync: (command, args, options) => { calls.role.push({ command, args: plain(args), options }); return { status: h.roleStatus, stdout: '{}', stderr: 'synthetic hidden details' }; } };
    else if (name === 'node:net') values = { default: { createServer: () => { const server = new EventEmitter(); server.listen = (port, host, callback) => { calls.ports.push({ port, host }); if (h.portBusy) server.emit('error', { code: 'EADDRINUSE' }); else callback(); }; server.close = callback => callback(); return server; } } };
    else values = await import(name);
    const result = new vm.SyntheticModule(Object.keys(values), function () { for (const [key, value] of Object.entries(values)) this.setExport(key, value); }, { context }); builtins.set(name, result); return result;
  };
  const matrix = new vm.SourceTextModule(launcherSource + '\ninspectInputs = __harness.inspect; helpers = __harness.helpers;', { context, initializeImportMeta: meta => { meta.url = pathToFileURL(path.join(root, 'matrix-launcher.mjs')).href; }, importModuleDynamically: () => { throw Error('Real app import forbidden'); } });
  await matrix.link(link); await matrix.evaluate();
  const upgrade = new vm.SourceTextModule(upgradeSource, { context, initializeImportMeta: meta => { meta.url = pathToFileURL(path.join(root, 'matrix-upgrade.mjs')).href; } });
  await upgrade.link(link); await upgrade.evaluate();
  const protectedFiles = [record, initialized, p.pgReceiptPath, p.pgConfigPath, p.credentialsPath, p.configPath, p.envPath];
  const originalHashes = protectedFiles.map(file => [file, sha(fs.readFileSync(file))]);
  t.after(() => { for (const { child } of calls.spawn) child.emit('exit', 0, null); fs.rmSync(root, { recursive: true, force: true }); });
  const run = (action = 'backend') => upgrade.namespace.launchUpgrade(action, cell, originalRun, upgradeRun);
  const finish = (code = 0, signal = null) => calls.spawn.at(-1).child.emit('exit', code, signal);
  const receiptRoot = path.join(root, 'targeted-upgrades', upgradeRun, cell);
  return { root, p, cfg, holder, record, initialized, originalRun, upgradeRun, cell, newRoot, preparation, gate, manifest, tar, h, calls, fakeProcess, api: upgrade.namespace, matrix: matrix.namespace, run, finish, receiptRoot, originalHashes };
}

test('original launcher exposes exactly the reusable initialization/environment operations', async t => {
  const f = await fixture(t);
  for (const name of ['requireInitialized', 'runtimeEnv', 'frontendEnv']) assert.equal(typeof f.matrix[name], 'function', name);
});

test('backend upgrades code with original auth/data/ports and untouched origin proofs', async t => {
  const f = await fixture(t); await f.run();
  assert.equal(f.calls.spawn.length, 1);
  const call = f.calls.spawn[0];
  assert.equal(call.command, f.p.python); assert.equal(call.options.cwd, f.p.root);
  assert.deepEqual(call.args, ['-m', 'uvicorn', 'tldw_Server_API.app.main:app', '--host', '127.0.0.1', '--port', '19101']);
  assert.equal(call.options.env.TLDW_CONFIG_FILE, f.p.configPath);
  assert.equal(call.options.env.TLDW_ENV_FILE, f.p.envPath);
  assert.equal(call.options.env.USER_DB_BASE_DIR, f.p.userDatabasesDir);
  assert.equal(call.options.env.SINGLE_USER_API_KEY, 'synthetic-api-only');
  assert.equal(call.options.env.WORKFLOWS_ARTIFACTS_DIR, path.join(f.p.root, 'workflow-artifacts'));
  assert.equal(call.options.env.PYTHONPATH, [f.newRoot, path.join(f.newRoot, 'apps/mcp-unified/src'), path.join(f.newRoot, 'packages/tldw_profile_core/src')].join(path.delimiter));
  assert.deepEqual(f.calls.helpers, [f.p.sourceRoot]);
  assert.equal(f.calls.role.length, 1); assert.equal(f.calls.role[0].command, f.p.python);
  assert.deepEqual(f.calls.role[0].args, [path.join(f.root, 'pg_role_adapter.py'), f.p.pgConfigPath, f.p.pgReceiptPath]);
  f.finish();
  assert.deepEqual(f.originalHashes.map(([file]) => [file, sha(fs.readFileSync(file))]), f.originalHashes);
  assert.equal(fs.existsSync(path.join(f.p.root, 'backend-process.private.json')), false);
});

test('frontend uses new Next code and owned build but retains the original API/browser origin', async t => {
  const f = await fixture(t); write(path.join(f.newRoot, 'apps/tldw-frontend/.env.local'), 'UNSAFE_FILE_SETTING=value\n');
  await f.run('frontend'); assert.equal(f.calls.spawn.length, 1);
  const call = f.calls.spawn[0];
  assert.equal(call.options.cwd, path.join(f.newRoot, 'apps/tldw-frontend'));
  assert.equal(call.args[0], path.join(f.newRoot, 'apps/tldw-frontend/node_modules/next/dist/bin/next'));
  assert.equal(call.options.env.TLDW_INTERNAL_API_ORIGIN, 'http://127.0.0.1:19101');
  assert.equal(call.options.env.UNSAFE_FILE_SETTING, '');
  assert.equal(call.options.env.NEXT_PUBLIC_X_API_KEY, '');
  assert.equal(call.options.env.TLDW_NEXT_DIST_DIR, `.next-upgrade-${f.upgradeRun}-${f.cell}`);
  assert.equal(json(f.record).browserSessionName, f.p.browserSessionName);
  assert.equal(fs.existsSync(path.join(f.p.frontend, f.p.nextDistDir)), false);
  f.finish();
});

test('multi-user upgrade retains original JWT identity and its existing auth/content databases', async t => {
  const f = await fixture(t, 'pg-multi'); await f.run();
  const env = f.calls.spawn[0].options.env;
  assert.equal(env.AUTH_MODE, 'multi_user'); assert.equal(env.JWT_SECRET_KEY, 'synthetic-jwt-only');
  assert.equal(new URL(env.DATABASE_URL).pathname, '/tldw_test_aaaaaaaa');
  assert.equal(new URL(env.TLDW_CONTENT_PG_DSN).pathname, '/tldw_test_bbbbbbbb');
  f.finish(); assert.deepEqual(f.originalHashes.map(([file]) => [file, sha(fs.readFileSync(file))]), f.originalHashes);
});

const refusals = [
  ['incomplete original preparation', f => write(f.record, { ...f.p, preparationStatus: 'preparing' })],
  ['wrong original profile cell', f => write(f.record, { ...f.p, name: 'pg-multi' })],
  ['missing old initialization', f => fs.unlinkSync(f.initialized)],
  ['mismatched old initialization', f => write(f.initialized, { ...json(f.initialized), preparationHash: 'foreign' })],
  ['old source fingerprint change', f => write(path.join(f.p.sourceRoot, 'entry.py'), 'changed')],
  ['released holder', f => write(f.p.pgReceiptPath, { ...f.holder, status: 'released' })],
  ['foreign holder source', f => write(f.p.pgReceiptPath, { ...f.holder, source_root: f.newRoot })],
  ['privileged holder role', f => write(f.p.pgReceiptPath, { ...f.holder, runtime_role: { ...f.holder.runtime_role, bypassrls: true } })],
  ['live role probe denial', f => { f.h.roleStatus = 1; }],
  ['missing explicit upgrade policy', f => write(path.join(f.preparation, 'gate.json'), { ...f.gate, dataPolicy: 'fresh' })],
  ['gate belongs to another original profile', f => write(path.join(f.preparation, 'gate.json'), { ...f.gate, originalRunId: 'foreign' })],
  ['unreleased source', f => write(path.join(f.preparation, 'gate.json'), { ...f.gate, status: 'HELD' })],
  ['wrong source manifest cell', f => write(path.join(f.preparation, `${f.cell}-source-manifest.json`), { ...f.manifest, root: f.p.sourceRoot })],
  ['changed new tracked source', f => write(path.join(f.newRoot, 'entry.py'), 'changed')],
  ['tracked source traversal', f => write(path.join(f.preparation, `${f.cell}-source-manifest.json`), { ...f.manifest, files: [{ path: '../../outside', kind: 'file', sha256: 'f'.repeat(64) }], fileCount: 1 })],
  ['new source symlink escape', f => { fs.unlinkSync(path.join(f.newRoot, 'entry.py')); fs.symlinkSync(path.join(f.p.sourceRoot, 'entry.py'), path.join(f.newRoot, 'entry.py')); }],
  ['changed archive bytes', f => write(f.tar, 'changed')],
  ['origin inspection failure', f => { f.h.originFailure = true; }],
  ['archive already bound to a profile', f => write(path.join(f.root, 'foreign-pg-single.profile.private.json'), { sourceRoot: f.newRoot })],
  ['occupied API port', f => { f.h.portBusy = true; }],
];
for (const [name, mutate] of refusals) test(`refuses ${name} before launching or writing upgrade state`, async t => {
  const f = await fixture(t); mutate(f);
  await assert.rejects(f.run()); assert.equal(f.calls.spawn.length, 0); assert.equal(fs.existsSync(f.receiptRoot), false);
});

for (const rel of ['Databases/system_ops.json', 'Databases/document_upload_drafts.db', 'Databases/document_upload_drafts.db-wal', 'Databases/document_upload_drafts.db-shm', 'Databases/webscraper', 'Databases/downloads/audio']) test(`existing archive data blocks relocation: ${rel}`, async t => {
  const f = await fixture(t); write(path.join(f.p.sourceRoot, rel), 'retained-state');
  await assert.rejects(f.run(), /archive|storage|state/i); assert.equal(f.calls.spawn.length, 0);
});

test('a dangling root-state symlink and nonempty lock are not treated as absent', async t => {
  const f = await fixture(t); const lock = path.join(f.p.sourceRoot, 'Databases/system_ops.json.lock'); write(lock, 'state');
  await assert.rejects(f.run()); fs.unlinkSync(lock); fs.symlinkSync(path.join(f.root, 'absent'), lock);
  await assert.rejects(f.run()); assert.equal(f.calls.spawn.length, 0);
});

test('unowned Next build is rejected without adopting or deleting its contents', async t => {
  const f = await fixture(t), existing = path.join(f.newRoot, 'apps/tldw-frontend', `.next-upgrade-${f.upgradeRun}-${f.cell}`, 'sentinel'); write(existing, 'keep');
  await assert.rejects(f.run('frontend')); assert.equal(fs.readFileSync(existing, 'utf8'), 'keep'); assert.equal(f.calls.spawn.length, 0);
});

test('backend and frontend share one immutable upgrade binding with separate attempts', async t => {
  const f = await fixture(t); await f.run(); f.finish();
  const bound = fs.readFileSync(path.join(f.receiptRoot, 'binding.private.json'), 'utf8');
  await f.run('frontend'); f.finish();
  assert.equal(fs.readFileSync(path.join(f.receiptRoot, 'binding.private.json'), 'utf8'), bound);
  const records = fs.readdirSync(f.receiptRoot).filter(name => name.endsWith('.process.private.json'));
  assert.equal(records.length, 2);
  for (const name of records) { const r = json(path.join(f.receiptRoot, name)); assert.equal(r.purpose, 'upgrade-targeted-acceptance'); assert.equal(r.sourceCommit, f.gate.revision); assert.equal(r.code, 0); assert.equal(r.status, 'exited'); assert.equal(JSON.stringify(r).includes('synthetic-pg-only'), false); }
});

test('changed original proof cannot rebind an existing upgrade', async t => {
  const f = await fixture(t); await f.run(); f.finish();
  write(f.initialized, { ...json(f.initialized), token: 'different-init' });
  await assert.rejects(f.run()); assert.equal(f.calls.spawn.length, 1);
});

test('upgrade binding retains the exact controller/helper byte identity', async t => {
  const f = await fixture(t); await f.run(); f.finish();
  const binding = json(path.join(f.receiptRoot, 'binding.private.json'));
  assert.equal(binding.launcherHash, sha(launcherSource));
  assert.equal(binding.upgradeHelperHash, sha(upgradeSource));
});

test('child failure retains its truthful failure receipt and leaves original initialization alone', async t => {
  const f = await fixture(t); await f.run(); f.finish(7);
  const name = fs.readdirSync(f.receiptRoot).find(name => name.endsWith('.process.private.json'));
  assert.equal(json(path.join(f.receiptRoot, name)).resultCode, 7); assert.equal(f.fakeProcess.exitCode, 7);
  assert.equal(json(f.initialized).token, 'old-initializer');
});

test('cancellation forwards only to the owned child and records no readiness claim', async t => {
  const f = await fixture(t); await f.run(); f.fakeProcess.emit('SIGTERM'); f.finish(null, 'SIGTERM');
  assert.deepEqual(f.calls.signals.filter(x => x.signal !== 0), [{ pid: 4321, signal: 'SIGTERM' }]);
  const name = fs.readdirSync(f.receiptRoot).find(name => name.endsWith('.process.private.json'));
  const receipt = json(path.join(f.receiptRoot, name)); assert.equal(receipt.resultCode, 1); assert.equal(receipt.signal, 'SIGTERM'); assert.equal(receipt.ready, undefined);
});

test('a child exiting zero after requested cancellation still records the interruption', async t => {
  const f = await fixture(t); await f.run(); f.fakeProcess.emit('SIGINT'); f.finish(0);
  const name = fs.readdirSync(f.receiptRoot).find(name => name.endsWith('.process.private.json'));
  const receipt = json(path.join(f.receiptRoot, name));
  assert.equal(receipt.resultCode, 1); assert.equal(receipt.requestedSignal, 'SIGINT');
});

test('archive-local data through a symlinked parent cannot appear absent', async t => {
  const f = await fixture(t), databaseDir = path.join(f.p.sourceRoot, 'Databases'), elsewhere = path.join(f.root, 'other-state');
  fs.rmSync(databaseDir, { recursive: true }); fs.mkdirSync(elsewhere); fs.symlinkSync(elsewhere, databaseDir);
  await assert.rejects(f.run(), /archive|storage|state/i); assert.equal(f.calls.spawn.length, 0);
});

test('asynchronous spawn errors settle once and remove only owned signal listeners', async t => {
  const f = await fixture(t), unrelated = () => {};
  f.fakeProcess.on('SIGTERM', unrelated); await f.run();
  f.calls.spawn[0].child.emit('error', Error('synthetic private failure')); f.finish(0);
  const name = fs.readdirSync(f.receiptRoot).find(name => name.endsWith('.process.private.json'));
  assert.equal(json(path.join(f.receiptRoot, name)).resultCode, 1);
  assert.deepEqual(f.fakeProcess.listeners('SIGTERM'), [unrelated]);
});

test('spawn error preserves a failed attempt without overwriting the old profile', async t => {
  const f = await fixture(t); f.h.spawnThrows = true;
  await assert.rejects(f.run());
  const name = fs.readdirSync(f.receiptRoot).find(name => name.endsWith('.process.private.json'));
  assert.equal(json(path.join(f.receiptRoot, name)).status, 'spawn-failed');
  assert.equal(json(f.record).sourceRoot, f.p.sourceRoot);
});

test('action and run inputs reject reset, same-run and traversal before boundaries', async t => {
  const f = await fixture(t);
  for (const args of [['initialize', f.cell, f.originalRun, f.upgradeRun], ['backend', f.cell, f.originalRun, f.originalRun], ['backend', f.cell, '../outside', f.upgradeRun]]) await assert.rejects(f.api.launchUpgrade(...args));
  assert.equal(f.calls.inspect.length, 0); assert.equal(f.calls.spawn.length, 0);
});
