/** Asset/CSP smoke in the actual WXT build; not extension Notes/history qualification. */
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const web = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(resolve(web, 'package.json'));
const { chromium } = require('playwright');
const extension = resolve(web, '../extension/.output/chrome-mv3');
const profile = mkdtempSync(resolve(tmpdir(), 'i1-monaco-extension-'));
const context = await chromium.launchPersistentContext(profile, {
  channel: 'chromium',
  headless: true,
  args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
});
try {
  const background = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
  const origin = new URL(background.url()).origin;
  // URL.origin for the extension scheme is opaque in Node; use the actual host explicitly.
  const base = `chrome-extension://${new URL(background.url()).host}`;
  const page = await context.newPage();
  const requests = [];
  page.on('request', (request) => {
    if (request.url().includes('/i1-monaco/')) requests.push(request.url());
  });
  await page.goto(`${base}/options.html`);
  await page.addScriptTag({ url: `${base}/i1-monaco/vs/loader.js` });
  const result = await page.evaluate(async (base) => {
    await new Promise((resolve, reject) => {
      window.require.config({ paths: { vs: `${base}/i1-monaco/vs` } });
      window.require(['vs/editor/editor.main'], resolve, reject);
    });
    const source = '{"s":"a&b <tag> 😀  ","n":9007199254740993}\r\n';
    const model = window.monaco.editor.createModel(source, 'plaintext');
    const actual = model.getValue(undefined, true);
    model.dispose();
    const workerUrl = `${base}/i1-monaco/editor-worker.js`;
    await new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl);
      const timer = setTimeout(() => {
        worker.terminate();
        reject(new Error('Worker readiness timed out'));
      }, 5000);
      worker.onmessage = (event) => {
        if (event.data?.type === 'vscode-worker-ready') {
          clearTimeout(timer);
          worker.terminate();
          resolve(true);
        }
      };
      worker.onerror = (event) => {
        clearTimeout(timer);
        worker.terminate();
        reject(new Error(event.message));
      };
    });
    return { source, actual, workerUrl };
  }, base);
  assert.equal(result.actual, result.source);
  assert(requests.some((url) => /assets\/editor\.worker-.*\.js$/.test(url)));
  assert(requests.every((url) => url.startsWith(base + '/i1-monaco/')));
  console.log(
    JSON.stringify(
      { browser: context.browser()?.version(), origin, base, ...result, requests },
      null,
      2
    )
  );
} finally {
  await context.close();
  rmSync(profile, { recursive: true, force: true });
}
