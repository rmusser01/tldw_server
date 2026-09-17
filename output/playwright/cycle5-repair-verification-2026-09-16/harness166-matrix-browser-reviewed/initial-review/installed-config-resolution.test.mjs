// Read configuration only. Never starts the CLI, a daemon, or a browser.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { tools } = require('/Users/macbook-dev/.npm/_npx/31e32ef8478fbf80/node_modules/playwright-core/lib/coreBundle.js');

test('installed resolver consumes an inherited CDP endpoint without browser launch', async t => {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'uat166-config-only-')));
  const cwd = process.cwd();
  t.after(() => { process.chdir(cwd); fs.rmSync(root, {recursive:true, force:true}); });
  process.chdir(root);
  const config = await tools.resolveCLIConfigForCLI(root, 'synthetic', {}, {
    PWTEST_CLI_GLOBAL_CONFIG: root,
    PLAYWRIGHT_MCP_CDP_ENDPOINT: 'http://synthetic.invalid:9222'
  });
  assert.equal(config.browser.cdpEndpoint, 'http://synthetic.invalid:9222');
  assert.equal(config.browser.isolated, false);
});

test('installed resolver merges global browser configuration even with explicit empty local config', async t => {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'uat166-config-only-')));
  const cwd = process.cwd();
  t.after(() => { process.chdir(cwd); fs.rmSync(root, {recursive:true, force:true}); });
  process.chdir(root);
  fs.mkdirSync(path.join(root, '.playwright'));
  fs.writeFileSync(path.join(root, '.playwright', 'cli.config.json'), JSON.stringify({browser:{cdpEndpoint:'http://synthetic.invalid:9223'}}));
  const local = path.join(root, 'explicit-empty.json'); fs.writeFileSync(local, '{}');
  const config = await tools.resolveCLIConfigForCLI(root, 'synthetic', {config:local}, {PWTEST_CLI_GLOBAL_CONFIG:root});
  assert.equal(config.browser.cdpEndpoint, 'http://synthetic.invalid:9223');
  assert.equal(config.browser.isolated, false);
});
