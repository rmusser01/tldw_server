/** Existing real editor routes precede Notes in one browser document/context. */
import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { seedAuth, TEST_CONFIG } from './utils/helpers';
import type { OnMount } from '@monaco-editor/react';

test.use({ actionTimeout: 15_000 });

for (const prior of ['watchlists', 'presentation'])
  test(`${prior} local native editor precedes actual Notes`, async ({
    page,
    request,
    browser,
  }, info) => {
    const headers = { 'X-API-KEY': TEST_CONFIG.apiKey },
      title = `I3 loader order ${Date.now()}`;
    const created = await request.post(`${TEST_CONFIG.serverUrl}/api/v1/notes/`, {
      headers,
      data: { title, content: '{"a":1}', content_language: 'json' },
    });
    expect(created.ok()).toBeTruthy();
    const note = await created.json();
    const requests: string[] = [];
    page.on('request', (request) => {
      if (/monaco|\/vs\//.test(request.url())) requests.push(request.url());
    });
    try {
      if (process.env.I3_COOKIE_AUTH === '1')
        await page.addInitScript(() => {
          localStorage.setItem('__tldw_first_run_complete', 'true');
          localStorage.setItem('assistant_setup_dismissed', 'true');
          localStorage.setItem('notes-tutorial-shown', JSON.stringify('1'));
          localStorage.setItem('isMigrated', 'true');
        });
      else await seedAuth(page);
      await page.addInitScript(() => {
        localStorage.removeItem('__tldw_test_bypass');
        localStorage.removeItem('__tldw_allow_offline');
      });
      if (prior === 'watchlists') {
        await page.goto('/watchlists');
        await page.getByRole('tab', { name: 'Reports', exact: true }).click();
        await page.getByRole('button', { name: /Templates/ }).click();
        await page.getByRole('button', { name: 'Create Template', exact: true }).click();
        await page
          .getByRole('dialog', { name: 'Create Template', exact: true })
          .getByText('Advanced', { exact: true })
          .click();
        await page
          .getByRole('dialog', { name: 'Create Template', exact: true })
          .getByRole('tab', { name: 'Editor', exact: true })
          .click();
        await expect(
          page.getByTestId('template-editor-modal').locator('.monaco-editor')
        ).toBeVisible();
        await page.getByRole('button', { name: 'Cancel', exact: true }).click();
      } else {
        const fixture = JSON.parse(
          readFileSync('/private/tmp/i3-evidence/presentation-fixture.json', 'utf8')
        );
        await page.goto(`/presentation-studio/${fixture.id}`);
        await expect(page.locator('.monaco-editor')).toBeVisible();
        await expect(
          page.getByRole('textbox', { name: 'HTML source', exact: true })
        ).toBeAttached();
      }
      const assets = await page.evaluate(async () => {
        const factory = globalThis.MonacoEnvironment!;
        const results = [];
        for (const label of ['editor', 'json', 'css', 'html', 'typescript']) {
          const worker = await factory.getWorker!('workerMain.js', label);
          try {
            await new Promise<void>((resolve, reject) => {
              const timer = setTimeout(() => reject(Error('Native asset readiness timeout')), 2000);
              worker.onmessage = (event) => {
                if (event.data?.type === 'vscode-worker-ready') {
                  clearTimeout(timer);
                  resolve();
                }
              };
              worker.onerror = (event) => {
                clearTimeout(timer);
                reject(Error(event.message));
              };
            });
            results.push({ label, ready: true });
          } finally {
            worker.terminate();
          }
        }
        return results;
      });
      expect(assets).toHaveLength(5);
      const documentOrigin = await page.evaluate(() => performance.timeOrigin);
      await page.getByRole('button', { name: 'Notes', exact: true }).click();
      await expect(page).toHaveURL(/\/notes$/);
      expect(await page.evaluate(() => performance.timeOrigin)).toBe(documentOrigin);
      await page.getByTestId(`notes-open-button-${note.id}`).click();
      await expect(
        page.getByRole('button', { name: 'Format document', exact: true })
      ).toBeEnabled();
      await page.getByRole('button', { name: 'Format document', exact: true }).click();
      await expect
        .poll(() =>
          page.evaluate(() => {
            const monaco = (
              window as typeof window & {
                monaco?: { editor: { getEditors: () => Array<Parameters<OnMount>[0]> } };
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
      expect(
        requests.every((url) => new URL(url).origin === new URL(page.url()).origin)
      ).toBeTruthy();
      expect(
        await page.evaluate(() => [
          localStorage.getItem('__tldw_test_bypass'),
          localStorage.getItem('__tldw_allow_offline'),
        ])
      ).toEqual([null, null]);
      const evidence = info.outputPath('prior-native-order.json');
      writeFileSync(
        evidence,
        JSON.stringify({
          prior,
          browser: browser.version(),
          sameDocument: true,
          nativeAssets: assets,
          localRequests: requests,
          generationCreationQualified: false,
        })
      );
      await info.attach('prior-native-order.json', {
        path: evidence,
        contentType: 'application/json',
      });
    } finally {
      await page.close();
      const response = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
        headers,
      });
      if (response.ok()) {
        const saved = await response.json();
        expect(
          (
            await request.delete(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
              headers: { ...headers, 'expected-version': String(saved.version) },
            })
          ).ok()
        ).toBeTruthy();
      }
    }
  });
