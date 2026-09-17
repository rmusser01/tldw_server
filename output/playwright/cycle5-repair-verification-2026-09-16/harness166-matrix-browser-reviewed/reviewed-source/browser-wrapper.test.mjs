import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { main, runBrowser } from '../uat-next-matrix-20260916/matrix-browser.mjs';

const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const secret = 'synthetic-quote"-slash\\-space /?';
function fixture(t, { pg = false } = {}) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'uat166-synthetic-')));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const cell = pg ? 'pg-multi' : 'sqlite-single', runId = 'synthetic-run', session = `${runId}-${cell}`;
  const runtime = path.join(root, 'profiles', `tldw-onboarding-uat-${session}`);
  const source = path.join(root, 'sources', cell);
  fs.mkdirSync(runtime, { recursive: true, mode: 0o700 });
  fs.mkdirSync(path.join(source, '.playwright-cli'), { recursive: true, mode: 0o700 });
  const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value), { mode: 0o600 });
  const profile = { name: cell, runId, root: runtime, sourceRoot: source, sourceCommit: 'a'.repeat(40), preparationId: 'synthetic', preparationStatus: 'complete', browserSessionName: session, credentialsPath: path.join(runtime, 'credentials.private.json') };
  const record = path.join(root, `${session}.profile.private.json`);
  write(profile.credentialsPath, { apiKey: secret, jwtSecret: 'synthetic-jwt-secret', apiHashSecret: 'synthetic-api-hash', accounts: { admin: { password: 'synthetic-admin-password' } } });
  if (pg) {
    const holder = path.join(root, 'holders', session); fs.mkdirSync(holder, { recursive: true, mode: 0o700 });
    profile.pgConfigPath = path.join(holder, 'runtime.pg-config.private.json');
    write(profile.pgConfigPath, { password: 'synthetic-pg-password' });
  }
  const save = () => { write(record, profile); write(path.join(runtime, 'initialized.private.json'), { status: 'completed', code: 0, token: 'synthetic-proof', preparationHash: hash(profile) }); };
  save();
  const scenario = path.join(root, 'scenario.json'), cli = path.join(root, 'fake-cli.mjs');
  fs.writeFileSync(cli, `import fs from 'node:fs';\nconst c=JSON.parse(fs.readFileSync(${JSON.stringify(scenario)},'utf8'));\nif(c.snapshot)fs.writeFileSync('.playwright-cli/page-test.yml',c.snapshot);\nconsole.log(JSON.stringify({args:process.argv.slice(2),cwd:process.cwd()}));\nconsole.log(c.text??'');\nprocess.exitCode=c.code??0;\n`, { mode: 0o600 });
  write(scenario, {});
  return { root, cell, runId, runtime, source, profile, record, save, write, scenario, cli, options: { packetRoot: root, cli }, invoke: (...args) => main([runId, cell, ...args], { packetRoot: root, cli }) };
}

test('uses recorded session/archive and privately preserves raw command evidence', t => {
  const f = fixture(t); const r = f.invoke('snapshot');
  assert.equal(r.code, 0); const observed = JSON.parse(r.output.split('\n')[0]);
  assert.deepEqual(observed.args, [`-s=${f.profile.browserSessionName}`, 'snapshot']);
  assert.equal(observed.cwd, f.source);
  assert.equal(fs.statSync(r.rawPath).mode & 0o777, 0o600);
  assert.ok(r.rawPath.startsWith(f.runtime + path.sep));
  const again = f.invoke('snapshot'); assert.notEqual(r.rawPath, again.rawPath);
});

test('redacts raw, JSON-escaped, URL-encoded, account and JWT values', t => {
  const f = fixture(t);
  const jwt = 'eyJhbGciOiJub25lIn0.eyJzdWIiOiJzeW50aGV0aWMifQ.c3ludGhldGlj';
  const values = [secret, JSON.stringify(secret).slice(1, -1), encodeURIComponent(secret), 'synthetic-admin-password', 'synthetic-jwt-secret', 'synthetic-api-hash', jwt];
  f.write(f.scenario, { text: values.join('\n') }); const r = f.invoke('snapshot');
  assert.equal(r.code, 0); for (const value of values) assert.ok(!r.output.includes(value));
  assert.ok(fs.readFileSync(r.rawPath, 'utf8').includes(secret));
});

test('redacts matching PostgreSQL runtime password', t => {
  const f = fixture(t, { pg: true }); f.write(f.scenario, { text: 'synthetic-pg-password' });
  const r = f.invoke('snapshot'); assert.equal(r.code, 0); assert.ok(!r.output.includes('synthetic-pg-password'));
});

test('redacts operator-supplied provider credentials registered before UI entry', t => {
  const f = fixture(t); const c = JSON.parse(fs.readFileSync(f.profile.credentialsPath));
  const providerSecret = 'synthetic-provider /? key'; c.providerSecrets = [providerSecret]; f.write(f.profile.credentialsPath, c);
  f.write(f.scenario, { text: `${providerSecret}\n${encodeURIComponent(providerSecret)}` });
  const r = f.invoke('snapshot'); assert.equal(r.code, 0);
  assert.ok(!r.output.includes(providerSecret)); assert.ok(!r.output.includes(encodeURIComponent(providerSecret)));
});

test('replaces raw snapshot references with confined redacted text copies', t => {
  const f = fixture(t); f.write(f.scenario, { text: '[Snapshot](.playwright-cli/page-test.yml)', snapshot: `textbox: ${secret}` });
  const r = f.invoke('snapshot'); assert.equal(r.code, 0);
  assert.ok(!r.output.includes(secret)); assert.ok(!r.output.includes('](.playwright-cli/page-test.yml)'));
  const snapshots = fs.readdirSync(path.dirname(r.rawPath)).filter(n => n.endsWith('.snapshot.txt'));
  assert.equal(snapshots.length, 1);
  const file = path.join(path.dirname(r.rawPath), snapshots[0]);
  assert.equal(fs.statSync(file).mode & 0o777, 0o600); assert.ok(!fs.readFileSync(file, 'utf8').includes(secret));
});

test('does not follow a snapshot symlink outside the owned archive', t => {
  const f = fixture(t); const outside = path.join(f.root, 'outside.yml'); fs.writeFileSync(outside, 'OUTSIDE_CONTENT');
  fs.symlinkSync(outside, path.join(f.source, '.playwright-cli/page-test.yml'));
  f.write(f.scenario, { text: '[Snapshot](.playwright-cli/page-test.yml)' });
  const r = f.invoke('snapshot'); assert.equal(r.code, 1); assert.ok(!r.output.includes('OUTSIDE_CONTENT'));
});

test('preserves command failure exit code and sanitized diagnostics', t => {
  const f = fixture(t); f.write(f.scenario, { code: 7, text: `failed ${secret}` });
  const r = f.invoke('snapshot'); assert.equal(r.code, 7); assert.match(r.output, /failed \[REDACTED\]/);
});

for (const change of ['owner', 'session', 'source', 'credentials', 'initialization', 'mode']) {
  test(`rejects ${change} mismatch before subprocess execution`, t => {
    const f = fixture(t);
    assert.equal(f.invoke('snapshot').code, 0);
    if (change === 'owner') f.profile.name = 'sqlite-multi';
    if (change === 'session') f.profile.browserSessionName = 'another-session';
    if (change === 'source') f.profile.sourceRoot = os.tmpdir();
    if (change === 'credentials') f.profile.credentialsPath = path.join(f.root, 'outside.json');
    f.save();
    if (change === 'initialization') f.write(path.join(f.runtime, 'initialized.private.json'), { status: 'completed', code: 0, token: 'wrong', preparationHash: 'wrong' });
    if (change === 'mode') fs.chmodSync(f.record, 0o644);
    const r = f.invoke('snapshot'); assert.equal(r.code, 1); assert.equal(r.rawPath, undefined);
  });
}

test('rejects session overrides and invalid run/cell values', t => {
  const f = fixture(t);
  assert.equal(f.invoke('snapshot').code, 0);
  for (const flag of ['-s=foreign', '-sforeign', '--session=foreign']) assert.equal(f.invoke(flag, 'snapshot').code, 1);
  assert.equal(main(['../escape', f.cell, 'snapshot'], f.options).code, 1);
  assert.equal(main([f.runId, 'invalid', 'snapshot'], f.options).code, 1);
});

test('safe setup errors never print malformed credential contents', t => {
  const f = fixture(t); assert.equal(f.invoke('snapshot').code, 0); fs.writeFileSync(f.profile.credentialsPath, `malformed ${secret}`);
  const r = f.invoke('snapshot'); assert.equal(r.code, 1); assert.ok(!r.output.includes(secret));
});

test('subprocess infrastructure errors suppress partial output', t => {
  const f = fixture(t);
  const r = runBrowser(f.runId, f.cell, ['snapshot'], { ...f.options, spawn: () => ({ status: null, stdout: secret.slice(0, 8), stderr: secret, error: new Error('synthetic timeout') }) });
  assert.equal(r.code, 1); assert.ok(!r.output.includes(secret.slice(0, 8)));
  assert.ok(fs.readFileSync(r.rawPath, 'utf8').includes(secret));
});

for (const args of [['kill-all'], ['close-all'], ['attach', 'http://synthetic.invalid'], ['open', '--profile=/synthetic/foreign'], ['open', '--config=/synthetic/foreign'], ['open', '--persistent']]) {
  test(`rejects command or browser override: ${args.join(' ')}`, t => {
    const f = fixture(t); let called = false;
    const r = main([f.runId, f.cell, ...args], { ...f.options, spawn: () => { called = true; return { status: 0, stdout: '' }; } });
    assert.equal(r.code, 1); assert.equal(called, false);
  });
}

test('child environment removes host browser/config/storage and Node overrides', t => {
  const f = fixture(t); let child;
  const keys = ['PLAYWRIGHT_MCP_CDP_ENDPOINT', 'PLAYWRIGHT_MCP_USER_DATA_DIR', 'PLAYWRIGHT_MCP_STORAGE_STATE', 'PLAYWRIGHT_MCP_CONFIG', 'PWTEST_CLI_GLOBAL_CONFIG', 'NODE_OPTIONS'];
  const old = Object.fromEntries(keys.map(k => [k, process.env[k]]));
  t.after(() => { for (const k of keys) { if (old[k] === undefined) delete process.env[k]; else process.env[k] = old[k]; } });
  for (const k of keys) process.env[k] = 'synthetic-foreign-setting';
  const r = main([f.runId, f.cell, 'open', '--headed'], { ...f.options, spawn: (_exe, _args, options) => { child = options.env ?? process.env; return { status: 0, stdout: '' }; } });
  assert.equal(r.code, 0); for (const k of keys) assert.equal(child[k], undefined);
  assert.equal(child.NO_UPDATE_NOTIFIER, '1'); assert.equal(child.HOME, process.env.HOME);
});

for (const kind of ['local', 'global']) {
  test(`rejects implicit ${kind} config before subprocess creation`, t => {
    const f = fixture(t), globalConfig = path.join(f.root, 'global', '.playwright', 'cli.config.json');
    let called = false; const options = { ...f.options, globalConfig, spawn: () => { called = true; return { status: 0, stdout: '' }; } };
    assert.equal(main([f.runId, f.cell, 'open'], options).code, 0); called = false;
    const file = kind === 'global' ? globalConfig : path.join(f.source, '.playwright', 'cli.config.json');
    fs.mkdirSync(path.dirname(file), { recursive: true }); f.write(file, { browser: { cdpEndpoint: 'http://synthetic.invalid' } });
    const r = main([f.runId, f.cell, 'open'], options); assert.equal(r.code, 1); assert.equal(called, false);
  });
}

test('supports reviewed interaction and evidence options in the owned session', t => {
  const f = fixture(t);
  for (const args of [['open', 'http://127.0.0.1:1', '--headed'], ['run-code', '--filename=synthetic.js'], ['screenshot', '--full-page'], ['fill', 'e1', 'synthetic', '--submit'], ['snapshot', '--depth=3']]) {
    assert.equal(f.invoke(...args).code, 0);
  }
});

test('rejects malformed short options while allowing numeric interaction arguments', t => {
  const f = fixture(t);
  for (const arg of ['-1sforeign', '-1s=foreign', '--', '--=foreign']) assert.equal(f.invoke('open', arg).code, 1);
  assert.equal(f.invoke('mousewheel', '-100', '0').code, 0);
});

test('installed config resolver sees an isolated browser with the exact child environment', async t => {
  const f = fixture(t); let child;
  const r = main([f.runId, f.cell, 'open'], { ...f.options, spawn: (_exe, _args, options) => { child = options.env; return { status: 0, stdout: '' }; } });
  assert.equal(r.code, 0);
  const { tools } = createRequire(import.meta.url)('/Users/macbook-dev/.npm/_npx/31e32ef8478fbf80/node_modules/playwright-core/lib/coreBundle.js');
  const previous = process.cwd(); t.after(() => process.chdir(previous)); process.chdir(f.source);
  const config = await tools.resolveCLIConfigForCLI(f.root, f.profile.browserSessionName, {}, child);
  assert.equal(config.browser.isolated, true);
  for (const key of ['cdpEndpoint', 'remoteEndpoint', 'userDataDir']) assert.equal(config.browser[key], undefined);
  assert.equal(config.configFile, undefined);
});
