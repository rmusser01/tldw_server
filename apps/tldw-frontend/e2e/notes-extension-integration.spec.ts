/** Untouched installed extension artifact; no manifest/CSP/preflight patches. */
import { test, expect, chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { withOwnedNote, withOwnedExtensionProfile } from './utils/notes-owned';
import { TEST_CONFIG } from './utils/helpers';
import {
  createRealServerWorkflowStorageSeed,
  createRealServerWorkflowTldwConfig,
  REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED,
} from '../../test-utils/real-server-workflows';

test('installed Chrome extension opens actual structured Notes with local native assets and trusted worker', async ({
  request,
}, info) => {
  test.setTimeout(90_000);
  const title = `I3 extension ${Date.now()}`;
  const headers = { 'X-API-KEY': TEST_CONFIG.apiKey };
  test.skip(
    process.env.I3_EXTENSION_PRIOR === 'presentation' && !process.env.I3_STORED_PRESENTATION_ID,
    'Stored Presentation fixture requires explicit owned I3_STORED_PRESENTATION_ID; generation API remains unqualified.'
  );
  await withOwnedNote(
    request,
    info,
    { title, content: '{"a":1}', content_language: 'json' },
    async (note) => {
      const extension = path.resolve('../extension/.output/chrome-mv3');
      await withOwnedExtensionProfile(async (profile) => {
        let context: Awaited<ReturnType<typeof chromium.launchPersistentContext>> | undefined;
        try {
          context = await chromium.launchPersistentContext(profile, {
            channel: 'chromium',
            headless: true,
            ignoreDefaultArgs: ['--disable-extensions'],
            args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
          });
          const inventory = await context.newPage();
          await inventory.goto('chrome://extensions');
          const installed = await inventory.evaluate(
            () =>
              new Promise<
                Array<{
                  id: string;
                  name: string;
                  state: string;
                  runtimeErrors: unknown[];
                  manifestErrors: unknown[];
                }>
              >((resolve) => {
                const chrome = (
                  globalThis as typeof globalThis & {
                    chrome: {
                      developerPrivate: {
                        getExtensionsInfo: (
                          options: unknown,
                          callback: (
                            value: Array<{
                              id: string;
                              name: string;
                              state: string;
                              runtimeErrors: unknown[];
                              manifestErrors: unknown[];
                            }>
                          ) => void
                        ) => void;
                      };
                    };
                  }
                ).chrome;
                chrome.developerPrivate.getExtensionsInfo(
                  { includeDisabled: true, includeTerminated: true },
                  resolve
                );
              })
          );
          const product = installed.find((entry) => entry.name.includes('tldw'));
          writeFileSync(
            info.outputPath('extension-inventory.json'),
            JSON.stringify(
              {
                installed: installed.map(({ id, name, state, runtimeErrors, manifestErrors }) => ({
                  id,
                  name,
                  state,
                  runtimeErrors,
                  manifestErrors,
                })),
              },
              null,
              2
            )
          );
          expect(product).toBeTruthy();
          const id = product!.id,
            page = await context.newPage();
          const workers: string[] = [],
            violations: string[] = [];
          page.on('worker', (worker) => workers.push(worker.url()));
          page.on('console', (message) => {
            if (message.text().startsWith('i3-csp:')) violations.push(message.text());
          });
          await page.goto(`chrome-extension://${id}/options.html`);
          const seed = {
            ...createRealServerWorkflowStorageSeed(),
            tldwConfig: createRealServerWorkflowTldwConfig(
              TEST_CONFIG.serverUrl,
              TEST_CONFIG.apiKey
            ),
          };
          await page.evaluate(async (seed) => {
            const chrome = (
              globalThis as typeof globalThis & {
                chrome: {
                  storage: {
                    local: { set: (value: unknown) => Promise<void> };
                    sync: { set: (value: unknown) => Promise<void> };
                  };
                };
              }
            ).chrome;
            await chrome.storage.local.set(seed);
            await chrome.storage.sync.set(seed);
          }, seed);
          await page.evaluate((seed) => {
            for (const [key, value] of Object.entries(seed)) localStorage.setItem(key, value);
          }, REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED);
          await page.addInitScript(() =>
            document.addEventListener('securitypolicyviolation', (event) =>
              console.error('i3-csp:' + event.violatedDirective)
            )
          );
          const prior = process.env.I3_EXTENSION_PRIOR;
          if (prior === 'watchlists') {
            await page.goto(`chrome-extension://${id}/options.html#/watchlists`);
            await page.getByRole('tab', { name: 'Reports', exact: true }).click();
            await page.getByRole('button', { name: /Templates/ }).click();
            await page.getByRole('button', { name: 'Create Template', exact: true }).click();
            const dialog = page.getByRole('dialog', { name: 'Create Template', exact: true });
            await dialog.getByText('Advanced', { exact: true }).click();
            await dialog.getByRole('tab', { name: 'Editor', exact: true }).click();
            await expect(dialog.locator('.monaco-editor')).toBeVisible();
            await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
          } else if (prior === 'presentation') {
            await page.goto(
              `chrome-extension://${id}/options.html#/presentation-studio/${encodeURIComponent(process.env.I3_STORED_PRESENTATION_ID!)}`
            );
            await expect(
              page.getByText('Standalone HTML editing is available only in the WebUI.', {
                exact: true,
              })
            ).toBeVisible();
            await expect(page.locator('.monaco-editor')).toHaveCount(0);
          }
          const timeOrigin = await page.evaluate(() => performance.timeOrigin);
          await page.goto(`chrome-extension://${id}/options.html#/notes`);
          if (prior) expect(await page.evaluate(() => performance.timeOrigin)).toBe(timeOrigin);
          expect(
            await page.evaluate(() => [
              localStorage.getItem('__tldw_test_bypass'),
              localStorage.getItem('__tldw_allow_offline'),
            ])
          ).toEqual([null, null]);
          await page
            .getByRole('button', { name: 'Skip tour', exact: true })
            .click({ timeout: 15000 });
          await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
          await expect(page.getByTestId('notes-native-source')).toBeVisible();
          await page.getByRole('button', { name: 'Format document', exact: true }).click();
          await expect
            .poll(() =>
              page.evaluate(() => {
                const monaco = (
                  window as typeof window & {
                    monaco?: {
                      editor: {
                        getEditors: () => Array<{
                          getDomNode: () => HTMLElement | null;
                          getModel: () => {
                            getValue: (eol?: undefined, bom?: boolean) => string;
                          } | null;
                        }>;
                      };
                    };
                  }
                ).monaco;
                return monaco?.editor
                  .getEditors()
                  .find((e) => e.getDomNode()?.isConnected)
                  ?.getModel()
                  ?.getValue(undefined, true);
              })
            )
            .toBe('{\n  "a": 1\n}');
          await page.getByTestId('notes-save-button').click();
          await expect
            .poll(async () => {
              const saved = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
                headers,
              });
              const body = await saved.json();
              return [body.content, body.content_language];
            })
            .toEqual(['{\n  "a": 1\n}', 'json']);
          expect(violations).toEqual([]);
          expect(workers.some((url) => !url.includes('/monaco/'))).toBeTruthy();
          const nativeAssets = await page.evaluate(async () => {
            const factory = (
              globalThis as typeof globalThis & {
                MonacoEnvironment: { getWorker: (module: string, label: string) => Worker };
              }
            ).MonacoEnvironment;
            const results = [];
            for (const label of ['editor', 'json', 'css', 'html', 'typescript']) {
              const worker = await factory.getWorker('workerMain.js', label);
              try {
                await new Promise<void>((resolve, reject) => {
                  const timer = setTimeout(
                    () => reject(Error('Native asset readiness timeout')),
                    2000
                  );
                  worker.onmessage = (event) => {
                    if (event.data?.type === 'vscode-worker-ready') {
                      clearTimeout(timer);
                      resolve();
                    }
                  };
                  worker.onerror = (event) => {
                    clearTimeout(timer);
                    reject(Error(`${event.message} at ${event.filename}:${event.lineno}`));
                  };
                });
                results.push({ label, ready: true });
              } catch (error) {
                results.push({ label, ready: false, error: String(error) });
              } finally {
                worker.terminate();
              }
            }
            return results;
          });
          expect(nativeAssets).toHaveLength(5);
          writeFileSync(
            info.outputPath('native-assets.json'),
            JSON.stringify({ nativeAssets, workers, violations }, null, 2)
          );
          expect(nativeAssets.every((item) => item.ready)).toBeTruthy();
          const evidence = info.outputPath('installed-extension-runtime.json');
          writeFileSync(
            evidence,
            JSON.stringify({
              browser: context.browser()?.version(),
              workers,
              violations,
              untouchedManifest: true,
              prior: process.env.I3_EXTENSION_PRIOR ?? null,
              standaloneHtmlEditingQualified: false,
            })
          );
          await info.attach('installed-extension-runtime.json', {
            path: evidence,
            contentType: 'application/json',
          });
        } finally {
          await context?.close();
        }
      });
    }
  );
});
