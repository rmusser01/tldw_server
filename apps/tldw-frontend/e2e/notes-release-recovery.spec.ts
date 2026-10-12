/** Empty source recovery through the normal Notes owner and real server. */
import { test, expect } from '@playwright/test';
import type { OnMount } from '@monaco-editor/react';
import { seedAuth, TEST_CONFIG } from './utils/helpers';
import { withOwnedNote } from './utils/notes-owned';

test('empty native draft survives navigation and app reload with its language', async ({
  page,
  request,
}, info) => {
  const title = `I5 empty recovery ${Date.now()}`;
  await withOwnedNote(
    request,
    info,
    { title, content: '{"kept":true}', content_language: 'json' },
    async (note) => {
      await seedAuth(page);
      await page.addInitScript(() => {
        localStorage.removeItem('__tldw_test_bypass');
        localStorage.removeItem('__tldw_allow_offline');
      });
      await page.goto('/notes');
      await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
      const source = () =>
        page.evaluate(() => {
          const editor = (
            window as typeof window & {
              monaco?: { editor: { getEditors: () => Array<Parameters<OnMount>[0]> } };
            }
          ).monaco?.editor
            .getEditors()
            .find((value) => value.getDomNode()?.isConnected);
          return editor?.getModel()?.getValue(undefined, true);
        });
      await expect.poll(source).toBe('{"kept":true}');
      const input = page.getByRole('textbox', { name: 'Note content', exact: true });
      const mac = await page.evaluate(() => /Macintosh/.test(navigator.userAgent));
      await input.focus();
      await input.press(mac ? 'Meta+a' : 'Control+a');
      await input.press('Backspace');
      await expect.poll(source).toBe('');
      await page.getByRole('button', { name: 'New note', exact: true }).click();
      await page.goto(`/notes?source_ref_id=${encodeURIComponent(note.id)}`);
      await expect.poll(source).toBe('');
      await expect(
        page.getByTestId('notes-document-tools').getByText('JSON', { exact: true })
      ).toBeVisible();
      const canonical = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
        headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
      });
      expect(canonical.ok()).toBeTruthy();
      expect(await canonical.json()).toMatchObject({
        content: '{"kept":true}',
        content_language: 'json',
      });
      const text = await page.locator('body').innerText();
      expect(text).toMatch(/Unsaved|not saved|Unable to save|could not save|Save failed|pending/i);
      await info.attach('empty-recovery.png', {
        body: await page.screenshot(),
        contentType: 'image/png',
      });
    }
  );
});
