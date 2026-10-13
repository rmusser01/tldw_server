import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { seedAuth } from './utils/helpers';
import { withOwnedNote } from './utils/notes-owned';

for (const [profile, source] of [
  ['JSON', '{"x":9007199254740993}\n'],
  ['fenced', '\ufeffa\r\nb\nc\r'],
]) {
  test(`actual ${profile} source and portable downloads`, async ({
    page,
    request,
    browser,
  }, info) => {
    const title = `I4 ${profile} ${Date.now()}`;
    await withOwnedNote(
      request,
      info,
      { title, content: source, content_language: 'json' },
      async (note) => {
        await seedAuth(page);
        await page.addInitScript(() => {
          localStorage.removeItem('__tldw_test_bypass');
          localStorage.removeItem('__tldw_allow_offline');
        });
        const response = await page.goto('/notes');
        const csp = response?.headers()['content-security-policy'];
        expect(csp).toBeTruthy();
        expect(
          await page.evaluate(() => [
            localStorage.getItem('__tldw_test_bypass'),
            localStorage.getItem('__tldw_allow_offline'),
          ])
        ).toEqual([null, null]);
        await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
        // Wait for actual canonical loading, not only the selected list row.
        await expect(page.getByRole('textbox', { name: 'Note title', exact: true })).toHaveValue(
          title
        );
        if (profile === 'JSON')
          await expect(
            page.getByRole('button', { name: 'Check document', exact: true })
          ).toBeEnabled();
        const artifacts: Array<{ name: string; bytes: number }> = [];
        for (const [label, portable] of [
          ['Download source', false],
          ['Export note (.note.json)', true],
        ] as const) {
          await page.getByTestId('notes-overflow-menu-button').click();
          const pending = page.waitForEvent('download');
          await page.getByText(label, { exact: true }).click();
          const download = await pending;
          const target = info.outputPath(portable ? 'portable.note.json' : 'source.json');
          await download.saveAs(target);
          const bytes = readFileSync(target);
          const suffix = portable ? '.note.json' : '.json';
          expect(download.suggestedFilename()).toBe(title.replace(/[^a-z0-9-_]+/gi, '-') + suffix);
          if (portable)
            expect(JSON.parse(bytes.toString('utf8'))).toMatchObject({
              format_version: 'tldw.note.v1',
              id: note.id,
              title,
              content: source,
              content_language: 'json',
              keywords: [],
            });
          else expect(bytes).toEqual(Buffer.from(source, 'utf8'));
          artifacts.push({ name: download.suggestedFilename(), bytes: bytes.length });
        }
        await page.getByTestId('notes-overflow-menu-button').click();
        await page.screenshot({ path: info.outputPath('desktop-actions.png') });
        await page.getByTestId('notes-overflow-menu-button').click();
        await page.setViewportSize({ width: 390, height: 844 });
        await page.getByTestId('notes-overflow-menu-button').click();
        await expect(page.getByText('Download source', { exact: true })).toBeVisible();
        await expect(page.getByText('Export note (.note.json)', { exact: true })).toBeVisible();
        const mobileExport = page.getByText('Export note (.note.json)', { exact: true });
        await mobileExport.scrollIntoViewIfNeeded();
        const box = await mobileExport.boundingBox();
        expect(box && box.y >= 0 && box.y + box.height <= 844).toBeTruthy();
        await page.screenshot({ path: info.outputPath('mobile-actions.png') });
        const mobileDownload = page.waitForEvent('download');
        await mobileExport.click();
        const mobileFile = await mobileDownload;
        const mobilePath = info.outputPath('mobile.note.json');
        await mobileFile.saveAs(mobilePath);
        expect(JSON.parse(readFileSync(mobilePath, 'utf8')).content).toBe(source);

        writeFileSync(
          info.outputPath('downloads.json'),
          JSON.stringify({
            profile,
            artifacts,
            browser: browser.version(),
            sourceBytes: Buffer.byteLength(source),
            csp,
          }) + '\n'
        );
      }
    );
  });
}
