import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
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


for (const args of [['kill-all'], ['attach','http://synthetic.invalid:9222'], ['open','--profile=/synthetic/foreign-profile'], ['open','--config=/synthetic/foreign-config.json']]) {
  test(`rejects foreign/global browser command ${args[0]} ${args[1] || ''}`, t => {
    const f = fixture(t); let called = false;
    const result = main([f.runId, f.cell, ...args], {...f.options, spawn: () => { called = true; return {status: 0, stdout: '', stderr: ''}; }});
    assert.equal(called, false, 'unsafe CLI invocation must be rejected before spawn');
    assert.equal(result.code, 1);
  });
}

test('does not inherit browser attachment, profile, or config overrides', t => {
  const f = fixture(t); let inherited;
  const key = 'PLAYWRIGHT_MCP_CDP_ENDPOINT'; const old = process.env[key];
  t.after(() => { if (old === undefined) delete process.env[key]; else process.env[key] = old; });
  process.env[key] = 'http://synthetic.invalid:9222';
  const result = main([f.runId, f.cell, 'open'], {...f.options, spawn: (_exe, _args, options) => { inherited = (options.env ?? process.env)[key]; return {status: 0, stdout: '', stderr: ''}; }});
  assert.equal(result.code, 0);
  assert.equal(inherited, undefined, 'implicit inherited environment can redirect browser attachment');
});
