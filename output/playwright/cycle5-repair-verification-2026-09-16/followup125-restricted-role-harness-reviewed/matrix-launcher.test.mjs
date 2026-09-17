// No CLI actions: actual control flow with fake external boundaries and temp files.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import vm from 'node:vm';
import { EventEmitter } from 'node:events';
import { fileURLToPath, pathToFileURL } from 'node:url';

const packet = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(process.env.MATRIX_TEST_SOURCE || path.join(packet, 'matrix-launcher.mjs'), 'utf8');
const write = (file, value) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, typeof value === 'string' ? value : JSON.stringify(value), { mode: 0o600 }); };
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');

async function fixture(t) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'matrix-guard-test-')));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const sourceRoot = path.join(root, 'source');
  const runtimeRoot = path.join(root, 'profiles/tldw-onboarding-uat-run-sqlite-single');
  write(path.join(sourceRoot, 'tldw_Server_API/Config_Files/config.txt'), '[Setup]\nsetup_completed = false\n');
  const inspected = { sourceRoot, sourceCommit: 'a'.repeat(40), pythonVenv: path.join(root, 'dependencies'), python: '/fake/private/python', sourcePaths: [sourceRoot], sourceHashes: { entry: 'frozen' }, pythonOrigins: {}, frontend: path.join(sourceRoot, 'apps/tldw-frontend'), dependencyOrigins: {}, nextCli: '/fake/next-cli' };
  const p = { ...inspected, name: 'sqlite-single', runId: 'run', spec: { mode: 'single_user', engine: 'sqlite', api: 19001, web: 19002 }, root: runtimeRoot, logsDir: path.join(runtimeRoot, 'logs'), nextDistDir: '.next-fixture', preparationStatus: 'complete', preparationId: 'preparation-fixture' };
  fs.mkdirSync(p.logsDir, { recursive: true });
  const record = path.join(root, 'run-sqlite-single.profile.private.json');
  write(record, p);
  const fakeProcess = new EventEmitter();
  Object.assign(fakeProcess, { argv: [], env: {}, execPath: '/fake/node', umask() {}, kill() {}, exitCode: undefined });
  const calls = { spawn: [], ports: [], helpers: 0 };
  const h = {
    inspect: () => inspected,
    env: async () => ({ SINGLE_USER_API_KEY: 'synthetic-only', DATABASE_URL: 'sqlite:///unused' }),
    helpers: async () => ({
      buildLiveTierProfile: ({ runId }) => {
        calls.helpers++;
        if (h.failHelper) throw Error('Synthetic helper failure');
        const temp = path.join(root, 'profiles', `tldw-onboarding-uat-${runId}`);
        return { root: temp, databaseDir: path.join(temp, 'Databases'), userDatabasesDir: path.join(temp, 'Databases/users'), logsDir: path.join(temp, 'logs'), configPath: path.join(temp, 'Config_Files/config.txt'), envPath: path.join(temp, 'Config_Files/.env'), fixtureRoot: path.join(root, 'fixtures'), databasePaths: { evaluations: path.join(temp, 'evaluations.db') }, systemLogFilePath: path.join(temp, 'logs/system.log') };
      },
      scrubHostConfigurationValues: text => text,
    }),
    port: async port => calls.ports.push(port),
    spawn: (command, args, options) => {
      const child = new EventEmitter(); Object.assign(child, { pid: 12345, kill: () => {} });
      calls.spawn.push({ command, args, options, child }); return child;
    },
  };
  const context = vm.createContext({ process: fakeProcess, console: { log() {}, error() {} }, __harness: h });
  const module = new vm.SourceTextModule(source + '\ninspectInputs = __harness.inspect; helpers = __harness.helpers; runtimeEnv = __harness.env; frontendEnv = () => ({}); portFree = __harness.port; export { prepare, launch, readPgReceipt };', { context, initializeImportMeta: meta => { meta.url = pathToFileURL(path.join(root, 'matrix-launcher.mjs')).href; }, importModuleDynamically: () => { throw Error('Real dynamic import forbidden'); } });
  await module.link(async name => {
    let values;
    if (name === 'node:child_process') values = { spawn: h.spawn, spawnSync: (...args) => { if (h.spawnSync) return h.spawnSync(...args); throw Error('Real process preflight forbidden'); } };
    else if (name === 'node:net') values = { default: { createServer: () => { throw Error('Real network forbidden'); } } };
    else values = await import(name);
    const linked = new vm.SyntheticModule(Object.keys(values), function () { for (const [key, value] of Object.entries(values)) this.setExport(key, value); }, { context });
    return linked;
  });
  await module.evaluate();
  return { root, p, record, h, calls, fakeProcess, api: module.namespace };
}

const initialize = f => f.api.launch('initialize', f.p.name, { 'run-id': f.p.runId });
const initializedFile = f => path.join(f.p.root, 'initialized.private.json');
function completion(f, overrides = {}) {
  const attempt = JSON.parse(f.calls.spawn.at(-1).options.env.MATRIX_INIT_REQUEST);
  write(attempt.receiptPath, { status: 'completed', token: attempt.token, preparationHash: attempt.preparationHash, ...overrides });
  return attempt;
}
function finish(f, code = 0, signal = null) { f.calls.spawn.at(-1).child.emit('exit', code, signal); }

test('zero exit without normal-return proof cannot mark initialization complete', async t => {
  const f = await fixture(t); await initialize(f); finish(f);
  assert.equal(fs.existsSync(initializedFile(f)), false);
  assert.equal(f.fakeProcess.exitCode, 1);
  assert.equal(JSON.parse(fs.readFileSync(path.join(f.p.root, 'initialize-process.private.json'))).resultCode, 1);
});

test('forwarded cancellation cannot mark initialization complete even with exit zero', async t => {
  const f = await fixture(t); await initialize(f); f.fakeProcess.emit('SIGINT'); finish(f);
  assert.equal(fs.existsSync(initializedFile(f)), false);
});

for (const [label, overrides] of [['stale attempt', { token: 'old-attempt' }], ['other preparation', { preparationHash: 'other-profile' }], ['incomplete', { status: 'started' }]]) {
  test(`initialization rejects ${label} proof`, async t => {
    const f = await fixture(t); await initialize(f); completion(f, overrides); finish(f);
    assert.equal(fs.existsSync(initializedFile(f)), false); assert.equal(f.fakeProcess.exitCode, 1);
  });
}

test('nonzero child exit rejects an otherwise matching completion proof', async t => {
  const f = await fixture(t); await initialize(f); completion(f); finish(f, 1);
  assert.equal(fs.existsSync(initializedFile(f)), false);
});

test('matching normal completion records identity and allows restart without resetting', async t => {
  const f = await fixture(t); await initialize(f); const attempt = completion(f); finish(f);
  const receipt = JSON.parse(fs.readFileSync(initializedFile(f)));
  assert.equal(receipt.preparationHash, hash(f.p)); assert.equal(receipt.token, attempt.token);
  await f.api.launch('backend', f.p.name, { 'run-id': f.p.runId }); finish(f);
  await f.api.launch('backend', f.p.name, { 'run-id': f.p.runId }); finish(f);
  assert.deepEqual(f.calls.ports, [19001, 19001]);
  await assert.rejects(initialize(f), /already completed/);
});

test('failed initialization remains retryable and an old attempt cannot satisfy retry', async t => {
  const f = await fixture(t); await initialize(f); const old = completion(f); finish(f, 1);
  await initialize(f); const next = JSON.parse(f.calls.spawn.at(-1).options.env.MATRIX_INIT_REQUEST);
  assert.notEqual(next.token, old.token); finish(f);
  assert.equal(fs.existsSync(initializedFile(f)), false);
});

for (const action of ['backend', 'frontend']) {
  for (const state of ['missing', 'wrong-preparation', 'legacy-exit-only']) {
    test(`${action} rejects ${state} initialization before port or spawn`, async t => {
      const f = await fixture(t);
      if (state !== 'missing') write(initializedFile(f), state === 'wrong-preparation' ? { status: 'completed', preparationHash: 'other' } : { at: 'old', code: 0 });
      await assert.rejects(f.api.launch(action, f.p.name, { 'run-id': f.p.runId }), /initializ/i);
      assert.equal(f.calls.spawn.length, 0); assert.equal(f.calls.ports.length, 0);
    });
  }
}

test('incomplete preparation cannot initialize or start', async t => {
  const f = await fixture(t); write(f.record, { ...f.p, preparationStatus: 'preparing' });
  await assert.rejects(initialize(f), /prepar/i); assert.equal(f.calls.spawn.length, 0);
});

for (const name of ['sqlite-single', 'pg-single']) {
  test(`another run/cell cannot reuse a bound source root (${name})`, async t => {
    const f = await fixture(t);
    await assert.rejects(f.api.prepare(name, { 'run-id': 'another', 'source-root': f.p.sourceRoot, 'python-venv': f.p.pythonVenv, 'source-commit': f.p.sourceCommit, 'api-port': '19003', 'web-port': '19004' }), /source.*(bound|owned)/i);
    assert.equal(f.calls.helpers, 0);
  });
}

test('preparation reserves a new source before a helper can fail', async t => {
  const f = await fixture(t); fs.unlinkSync(f.record); fs.rmSync(f.p.root, { recursive: true }); f.h.failHelper = true;
  await assert.rejects(f.api.prepare(f.p.name, { 'run-id': f.p.runId, 'api-port': '19001', 'web-port': '19002' }), /Synthetic helper failure/);
  const record = JSON.parse(fs.readFileSync(f.record));
  assert.equal(record.sourceRoot, f.p.sourceRoot); assert.equal(record.preparationStatus, 'preparing');
  assert.equal(f.calls.helpers, 1);
});

test('successful new preparation writes a completed matching record', async t => {
  const f = await fixture(t); fs.unlinkSync(f.record); fs.rmSync(f.p.root, { recursive: true });
  await f.api.prepare(f.p.name, { 'run-id': f.p.runId, 'api-port': '19001', 'web-port': '19002' });
  const record = JSON.parse(fs.readFileSync(f.record));
  assert.equal(record.preparationStatus, 'complete'); assert.ok(record.preparationId); assert.equal(record.sourceRoot, f.p.sourceRoot);
});

test('existing profile mode ownership refusal remains before spawn', async t => {
  const f = await fixture(t); write(f.record, { ...f.p, spec: { ...f.p.spec, mode: 'multi_user' } });
  await assert.rejects(initialize(f), /ownership mismatch/); assert.equal(f.calls.spawn.length, 0);
});

test('a failed child never creates a success marker', async t => {
  const f = await fixture(t); await initialize(f); finish(f, 1);
  assert.equal(fs.existsSync(initializedFile(f)), false);
});

test('existing profile preparation is never overwritten', async t => {
  const f = await fixture(t);
  await assert.rejects(f.api.prepare(f.p.name, { 'run-id': f.p.runId, 'api-port': '19001', 'web-port': '19002' }), /already exists/);
  assert.equal(f.calls.helpers, 0);
});

test('official PG receipt validates exact local ownership without connecting', async t => {
  const f = await fixture(t), { p, receipt } = roleFixture(f);
  const pgReceiptPath = p.pgReceiptPath;
  assert.equal(f.api.readPgReceipt(p).content.database, receipt.content.database);
  write(pgReceiptPath, { ...receipt, run_id: 'foreign' });
  assert.throws(() => f.api.readPgReceipt(p), /ownership/);
  write(pgReceiptPath, { ...receipt, content: receipt.auth });
  assert.throws(() => f.api.readPgReceipt(p), /distinct/);
});

function roleFixture(f) {
  const user = 'tldw_matrix_1234567890abcdef';
  const cfg = { host: '127.0.0.1', port: 55475, user, password: 'synthetic-role-password', purpose: 'matrix-runtime', cell: 'pg-single', run_id: 'run' };
  const pgConfigPath = path.join(f.root, 'runtime.private.json'), pgReceiptPath = path.join(f.root, 'holder.private.json');
  const flags = { name: user, login: true, superuser: false, bypassrls: false, inherit: false, createdb: false, createrole: false, replication: false, memberships: 0 };
  const receipt = { profile: 'pg-single', run_id: 'run', source_root: f.p.sourceRoot, source_commit: f.p.sourceCommit, python_venv: f.p.pythonVenv, status: 'held', pid: 12345, auth_fixture: 'pg_temp_db', content_fixture: 'pg_temp_db_session', provisioning: { user: 'fixture-admin', host: cfg.host, port: cfg.port }, runtime_role: flags, auth: { ...cfg, database: 'tldw_test_aaaaaaaa' }, content: { ...cfg, database: 'tldw_test_bbbbbbbb' } };
  write(pgConfigPath, cfg); write(pgReceiptPath, receipt);
  const p = { ...f.p, name: 'pg-single', spec: { ...f.p.spec, engine: 'postgresql' }, pgReceiptPath, pgConfigPath };
  write(path.join(f.root, 'run-pg-single.profile.private.json'), p);
  return { p, cfg, receipt, save: () => write(pgReceiptPath, receipt) };
}

test('matrix runtime role accepts separated checked provisioning and runtime identities', async t => {
  const f = await fixture(t), pg = roleFixture(f);
  assert.equal(f.api.readPgReceipt(pg.p).runtime_role.name, pg.cfg.user);
});

for (const flag of ['superuser', 'bypassrls', 'inherit', 'createdb', 'createrole', 'replication']) {
  test(`matrix runtime role rejects elevated ${flag} receipt`, async t => {
    const f = await fixture(t), pg = roleFixture(f); pg.receipt.runtime_role[flag] = true; pg.save();
    assert.throws(() => f.api.readPgReceipt(pg.p), /role/i);
  });
}

test('matrix runtime role rejects legacy administrator receipt', async t => {
  const f = await fixture(t), pg = roleFixture(f); delete pg.receipt.runtime_role; delete pg.receipt.provisioning; pg.save();
  assert.throws(() => f.api.readPgReceipt(pg.p), /role|provision/i);
});

test('matrix runtime role rejects administrator membership and reused provisioning identity', async t => {
  const f = await fixture(t), pg = roleFixture(f); pg.receipt.runtime_role.memberships = 1; pg.save();
  assert.throws(() => f.api.readPgReceipt(pg.p), /role/i);
  pg.receipt.runtime_role.memberships = 0; pg.receipt.provisioning.user = pg.cfg.user; pg.save();
  assert.throws(() => f.api.readPgReceipt(pg.p), /role|provision/i);
});

test('matrix runtime role checks live privileges before launching initializer', async t => {
  const f = await fixture(t); roleFixture(f); const probes = [];
  f.h.spawnSync = (...args) => { probes.push(args); return { status: 1, stdout: '', stderr: 'synthetic denied' }; };
  await assert.rejects(f.api.launch('initialize', 'pg-single', { 'run-id': 'run' }), /role/i);
  assert.equal(f.calls.spawn.length, 0);
  assert.equal(probes.length, 1);
});

test('matrix runtime role live probe uses explicit copied python and no passwords in argv', async t => {
  const f = await fixture(t), pg = roleFixture(f); const probes = [];
  f.h.spawnSync = (...args) => { probes.push(args); return { status: 0, stdout: '{}', stderr: '' }; };
  await f.api.launch('initialize', 'pg-single', { 'run-id': 'run' });
  assert.equal(probes.length, 1);
  assert.equal(probes[0][0], pg.p.python);
  assert.equal(JSON.stringify(probes[0][1]).includes(pg.cfg.password), false);
  assert.equal(f.calls.spawn.length, 1);
  finish(f, 1);
});

for (const action of ['backend', 'frontend']) {
  test(`matrix runtime live role failure blocks ${action} before port or process`, async t => {
    const f = await fixture(t), pg = roleFixture(f);
    write(initializedFile(f), { status: 'completed', preparationHash: hash(pg.p), code: 0, token: 'synthetic-initialized' });
    f.h.spawnSync = () => ({ status: 1, stdout: '', stderr: 'synthetic elevated role' });
    await assert.rejects(f.api.launch(action, 'pg-single', { 'run-id': 'run' }), /role/i);
    assert.equal(f.calls.ports.length, 0);
    assert.equal(f.calls.spawn.length, 0);
  });
}
