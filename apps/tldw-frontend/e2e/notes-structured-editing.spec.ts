/** Production Notes qualification; all API records belong to this synthetic run. */
import { test, expect, type Page, type APIRequestContext } from '@playwright/test';
import { appendFileSync, writeFileSync } from 'node:fs';
import type { OnMount } from '@monaco-editor/react';
import { seedAuth, TEST_CONFIG } from './utils/helpers';

const source = (page: Page) =>
  page.evaluate(() => {
    const monaco = (
      window as typeof window & {
        monaco?: {
          editor: {
            getEditors: () => Array<{
              getDomNode: () => HTMLElement | null;
              getModel: () => { getValue: (eol?: unknown, bom?: boolean) => string } | null;
            }>;
          };
        };
      }
    ).monaco;
    return monaco?.editor
      .getEditors()
      .find((editor) => editor.getDomNode()?.isConnected)
      ?.getModel()
      ?.getValue(undefined, true);
  });
const nativeState = (page: Page) =>
  page.evaluate(() => {
    const editor = (
      window as typeof window & {
        monaco?: { editor: { getEditors: () => Array<Parameters<OnMount>[0]> } };
      }
    ).monaco?.editor
      .getEditors()
      .find((editor) => editor.getDomNode()?.isConnected);
    const model = editor?.getModel(),
      selection = editor?.getSelection();
    if (!editor || !model) return null;
    const exact = model.getValue(undefined, true),
      plain = model.getValue();
    return {
      prefix: exact === plain ? 0 : exact === '\ufeff' + plain ? 1 : null,
      version: model.getVersionId(),
      top: editor.getScrollTop(),
      left: editor.getScrollLeft(),
      selected: selection ? model.getValueInRange(selection) : null,
      anchor: selection ? model.getOffsetAt(selection.getSelectionStart()) : null,
      active: selection ? model.getOffsetAt(selection.getPosition()) : null,
      length: plain.length,
    };
  });
const input = (page: Page) => page.getByRole('textbox', { name: 'Note content', exact: true });
async function keys(page: Page) {
  const mac = await page.evaluate(() => /Macintosh/.test(navigator.userAgent));
  return {
    end: mac ? 'Meta+ArrowDown' : 'Control+End',
    undo: mac ? 'Meta+z' : 'Control+z',
    redo: mac ? 'Meta+Shift+z' : 'Control+y',
    all: mac ? 'Meta+a' : 'Control+a',
  };
}
async function owned(
  page: Page,
  request: APIRequestContext,
  text: string,
  language: string | null,
  run: (note: { id: string; title: string }) => Promise<void>
) {
  const headers = { 'X-API-KEY': TEST_CONFIG.apiKey };
  const title = `I3 ${Date.now()} ${Math.random().toString(36).slice(2)}`;
  const response = await request.post(`${TEST_CONFIG.serverUrl}/api/v1/notes/`, {
    headers,
    data: { title, content: text, content_language: language },
  });
  expect(response.ok()).toBeTruthy();
  const note = await response.json();
  appendFileSync(
    '/private/tmp/i3-evidence/owned-notes.jsonl',
    JSON.stringify({ id: note.id, title, createdAt: Date.now() }) + '\n'
  );
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
    await page.goto('/notes');
    expect(
      await page.evaluate(() => [
        localStorage.getItem('__tldw_test_bypass'),
        localStorage.getItem('__tldw_allow_offline'),
      ])
    ).toEqual([null, null]);
    await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
    await run({ ...note, title });
  } finally {
    await page.close();
    const response = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
      headers,
    });
    if (response.ok()) {
      const current = await response.json();
      expect(
        (
          await request.delete(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
            headers: { ...headers, 'expected-version': String(current.version) },
          })
        ).ok()
      ).toBeTruthy();
    }
  }
}
for (const [profile, a, c] of [
  ['LF', '{"a":1}', '{\n  "a": 1\n}'],
  ['CRLF', '{\r\n"a":1\r\n}\r\n', '{\r\n  "a": 1\r\n}'],
])
  test(`actual native ${profile} A/B/C/D undo and redo`, async ({
    page,
    request,
    browser,
  }, info) => {
    const workers: string[] = [];
    const violations: string[] = [];
    page.on('worker', (worker) => workers.push(worker.url()));
    await page.addInitScript(() =>
      document.addEventListener('securitypolicyviolation', (event) =>
        console.error('i3-csp:' + event.violatedDirective)
      )
    );
    page.on('console', (message) => {
      if (message.text().startsWith('i3-csp:')) violations.push(message.text());
    });
    await owned(page, request, a, 'json', async () => {
      await expect.poll(() => source(page)).toBe(a);
      await expect(
        page.getByRole('button', { name: 'Format document', exact: true })
      ).toBeEnabled();
      const shortcuts = await keys(page);
      const editor = input(page);
      const trace: Record<string, unknown> = { browser: browser.version(), profile };
      await editor.focus();
      await editor.press(shortcuts.end);
      await editor.pressSequentially(' ');
      await expect.poll(() => source(page)).toBe(a + ' ');
      trace.B = true;
      await page.getByRole('button', { name: 'Format document', exact: true }).click();
      await expect.poll(() => source(page)).toBe(c);
      await expect(editor).toBeFocused();
      trace.C = true;
      await editor.press(shortcuts.end);
      await editor.pressSequentially(' ');
      await expect.poll(() => source(page)).toBe(c + ' ');
      trace.D = true;
      for (const [name, key, expected] of [
        ['undoD', shortcuts.undo, c],
        ['undoC', shortcuts.undo, a + ' '],
        ['redoC', shortcuts.redo, c],
        ['undoC2', shortcuts.undo, a + ' '],
        ['undoB', shortcuts.undo, a],
      ]) {
        await editor.press(key);
        await expect.poll(() => source(page)).toBe(expected);
        trace[name] = true;
      }
      expect(violations).toEqual([]);
      expect(workers.length).toBeGreaterThan(0);
      const evidence = info.outputPath('native-history.json');
      writeFileSync(evidence, JSON.stringify({ ...trace, workerUrls: workers }));
      await info.attach('native-history.json', { path: evidence, contentType: 'application/json' });
    });
  });

test('syntax errors remain saveable and metadata/source round trip through the actual API', async ({
  page,
  request,
}) => {
  await owned(page, request, '{"a":1}', 'json', async (note) => {
    await expect.poll(() => source(page)).toBe('{"a":1}');
    const shortcuts = await keys(page);
    await input(page).focus();
    await input(page).press(shortcuts.all);
    await input(page).pressSequentially('{"a":');
    await expect.poll(() => source(page)).toBe('{"a":');
    await page.getByRole('button', { name: 'Check document', exact: true }).click();
    await expect(page.getByRole('list', { name: 'Problems' })).toBeVisible();
    await page.getByTestId('notes-save-button').click();
    await expect
      .poll(async () => {
        const response = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
          headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
        });
        const saved = await response.json();
        return { content: saved.content, language: saved.content_language };
      })
      .toEqual({ content: '{"a":', language: 'json' });
  });
});

test('native composition defers checking and formatting (synthetic flag control only)', async ({
  page,
  request,
}) => {
  await owned(page, request, '{"a":1}', 'json', async () => {
    await expect.poll(() => source(page)).toBe('{"a":1}');
    await input(page).focus();
    await input(page).dispatchEvent('compositionstart');
    await expect(page.getByRole('button', { name: 'Format document', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Check document', exact: true })).toBeDisabled();
    await input(page).dispatchEvent('compositionend');
    await expect(page.getByRole('button', { name: 'Format document', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Format document', exact: true }).click();
    await expect.poll(() => source(page)).toBe('{\n  "a": 1\n}');
  });
});
for (const text of ['a\r\nb\n', 'a\rb'])
  test(`source-only representation retains exact title/save (${text.charCodeAt(0) === 0xfeff ? 'BOM' : text.includes('\r\n') ? 'mixed' : 'CR'})`, async ({
    page,
    request,
  }) => {
    await owned(page, request, text, 'json', async (note) => {
      await expect(input(page)).toHaveAttribute('aria-readonly', 'true');
      expect(await input(page).textContent()).toBe(text);
      await page
        .getByRole('textbox', { name: 'Note title', exact: true })
        .fill(note.title + ' edited');
      await page.getByTestId('notes-save-button').click();
      await expect
        .poll(async () => {
          const reply = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
            headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
          });
          const saved = await reply.json();
          return [saved.title, saved.content, saved.content_language];
        })
        .toEqual([note.title + ' edited', text, 'json']);
    });
  });

// Initially fenced product representation: this is the installed-native RED control.
for (const origin of ['model-owned', 'pasted'])
  test(`actual ${origin} BOM native format history and exact save`, async ({ page, request }) => {
    const a = '\ufeffroot:\n    a: 1',
      c = '\ufeffroot:\n  a: 1 ';
    await owned(page, request, origin === 'model-owned' ? a : a.slice(1), 'yaml', async (note) => {
      await expect
        .poll(() => source(page), { timeout: 5000 })
        .toBe(origin === 'model-owned' ? a : a.slice(1));
      if (origin === 'pasted') {
        await input(page).focus();
        const mac = await page.evaluate(() => /Macintosh/.test(navigator.userAgent));
        await input(page).press(mac ? 'Meta+ArrowUp' : 'Control+Home');
        await page.keyboard.insertText('\ufeff');
        await expect.poll(() => source(page)).toBe(a);
      }
      await expect(
        page.getByRole('button', { name: 'Format document', exact: true })
      ).toBeEnabled();
      expect((await nativeState(page))?.prefix).toBe(origin === 'model-owned' ? 1 : 0);
      const shortcuts = await keys(page),
        editor = input(page);
      await editor.focus();
      await editor.press(shortcuts.end);
      await editor.pressSequentially(' ');
      await expect.poll(() => source(page)).toBe(a + ' ');
      await page.getByRole('button', { name: 'Format document', exact: true }).click();
      await expect.poll(() => source(page)).toBe(c);
      await page.getByTestId('notes-save-button').click();
      await expect
        .poll(async () => {
          const response = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
            headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
          });
          const saved = await response.json();
          return [saved.content, saved.content_language];
        })
        .toEqual([c, 'yaml']);
      await page.getByRole('button', { name: 'Format document', exact: true }).click();
      await expect(page.getByTestId('notes-document-status')).toHaveText('Already formatted');
      await editor.press(shortcuts.end);
      await editor.pressSequentially(' ');
      for (const [key, expected] of [
        [shortcuts.undo, c],
        [shortcuts.undo, a + ' '],
        [shortcuts.redo, c],
        [shortcuts.undo, a + ' '],
        [shortcuts.undo, a],
      ]) {
        await editor.press(key);
        await expect.poll(() => source(page)).toBe(expected);
      }
    });
  });

for (const origin of ['model-owned', 'pasted'])
  test(`actual ${origin} BOM Unicode selection, viewport and Problems navigation`, async ({
    page,
    request,
  }) => {
    const body =
      'label: "😀é\t"\nroot:\n' +
      Array.from({ length: 35 }, (_, i) => `    item${i}: ${i}\n`).join('');
    await owned(
      page,
      request,
      origin === 'model-owned' ? '\ufeff' + body : body,
      'yaml',
      async () => {
        await expect
          .poll(() => source(page))
          .toBe(origin === 'model-owned' ? '\ufeff' + body : body);
        if (origin === 'pasted') {
          await input(page).focus();
          const mac = await page.evaluate(() => /Macintosh/.test(navigator.userAgent));
          await input(page).press(mac ? 'Meta+ArrowUp' : 'Control+Home');
          await page.keyboard.insertText('\ufeff');
        }
        await page.evaluate(() => {
          const e = (
            window as typeof window & {
              monaco?: { editor: { getEditors: () => Array<Parameters<OnMount>[0]> } };
            }
          ).monaco?.editor
            .getEditors()
            .find((editor) => editor.getDomNode()?.isConnected);
          const m = e?.getModel();
          if (!e || !m) throw Error('Native model missing');
          const plain = m.getValue(),
            start = plain.indexOf('😀'),
            a = m.getPositionAt(start + 4),
            b = m.getPositionAt(start);
          e.setSelection({
            selectionStartLineNumber: a.lineNumber,
            selectionStartColumn: a.column,
            positionLineNumber: b.lineNumber,
            positionColumn: b.column,
          });
          e.setScrollPosition({ scrollTop: 150, scrollLeft: 0 });
          e.focus();
        });
        const before = await nativeState(page);
        expect(before?.selected).toBe('😀é');
        expect(before!.anchor).toBeGreaterThan(before!.active!);
        await page.getByRole('button', { name: 'Format document', exact: true }).click();
        await expect
          .poll(() => source(page))
          .toBe('\ufeff' + body.replaceAll('    item', '  item'));
        const after = await nativeState(page);
        expect(after?.selected).toBe('😀é');
        expect(after!.anchor).toBeGreaterThan(after!.active!);
        expect(after?.top).toBe(before?.top);
        const shortcuts = await keys(page);
        await input(page).press(shortcuts.all);
        await page.keyboard.insertText((origin === 'model-owned' ? '' : '\ufeff') + '["😀é\t",');
        await expect.poll(() => source(page)).toBe('\ufeff["😀é\t",');
        // Model-owned prefix survives replace-all; use its exact reader as canonical.
        await page.getByRole('button', { name: 'Check document', exact: true }).click();
        await expect(page.getByRole('list', { name: 'Problems' })).toBeVisible();
        await page.getByRole('list', { name: 'Problems' }).getByRole('button').last().click();
        const state = await nativeState(page);
        expect(state!.active).toBe(state!.length);
      }
    );
  });

test('JSON BOM stays editable/saveable while the strict engine refuses format', async ({
  page,
  request,
}) => {
  const a = '\ufeff{"emoji":"😀é\t"}';
  await owned(page, request, a, 'json', async (note) => {
    await expect.poll(() => source(page)).toBe(a);
    await page.getByRole('button', { name: 'Format document', exact: true }).click();
    await expect(page.getByRole('list', { name: 'Problems' })).toBeVisible();
    expect(await source(page)).toBe(a);
    await page.getByRole('list', { name: 'Problems' }).getByRole('button').first().click();
    expect((await nativeState(page))?.active).toBe(0);
    const shortcuts = await keys(page);
    await input(page).press(shortcuts.end);
    await input(page).pressSequentially(' ');
    await page.getByTestId('notes-save-button').click();
    await expect
      .poll(async () => {
        const r = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
          headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
        });
        return (await r.json()).content;
      })
      .toBe(a + ' ');
  });
});

test('actual live native edit retires a delayed real-worker format and finite checks clean up', async ({
  page,
  request,
}) => {
  // Delay only transferable parser frames; installed worker and app owners remain real.
  await page.addInitScript(() => {
    const original = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (
      message: unknown,
      options?: Transferable[] | StructuredSerializeOptions
    ) {
      if (message instanceof ArrayBuffer) {
        setTimeout(() => original.call(this, message, options as Transferable[]), 350);
      } else original.call(this, message, options as Transferable[]);
    };
  });
  await owned(page, request, '{"a":1}', 'json', async () => {
    await expect.poll(() => source(page)).toBe('{"a":1}');
    await expect(page.getByTestId('notes-document-status')).toHaveText(
      'Checked; no problems found'
    );
    await page.getByRole('button', { name: 'Format document', exact: true }).click();
    const shortcuts = await keys(page);
    await input(page).press(shortcuts.end);
    await input(page).pressSequentially(' ');
    await expect.poll(() => source(page)).toBe('{"a":1} ');
    await expect(page.getByTestId('notes-document-status')).toHaveText(
      'Checked; no problems found'
    );
    expect(await source(page)).toBe('{"a":1} ');
    const baseline = new Set(page.workers().map((w) => w.url()));
    for (let i = 0; i < 8; i++) {
      await page.getByRole('button', { name: 'Check document', exact: true }).click();
      await expect(page.getByTestId('notes-document-status')).toHaveText(
        'Checked; no problems found'
      );
    }
    await expect
      .poll(() =>
        page
          .workers()
          .filter((w) => !baseline.has(w.url()))
          .map((w) => w.url())
      )
      .toEqual([]);
    await page.getByRole('button', { name: 'Format document', exact: true }).click();
    await page.goto('/notes');
    await page.getByRole('button', { name: 'Open note', exact: false }).first().waitFor();
    expect(await source(page)).toBeUndefined();
  });
});

test('actual native read-only option rejects BOM format without mutating source', async ({
  page,
  request,
}) => {
  const text = '\ufeffroot:\n    a: 1';
  await owned(page, request, text, 'yaml', async () => {
    await expect.poll(() => source(page)).toBe(text);
    await page.evaluate(() => {
      const e = (
        window as typeof window & {
          monaco?: { editor: { getEditors: () => Array<Parameters<OnMount>[0]> } };
        }
      ).monaco?.editor
        .getEditors()
        .find((e) => e.getDomNode()?.isConnected);
      e?.updateOptions({ readOnly: true });
    });
    await page.getByRole('button', { name: 'Format document', exact: true }).click();
    expect(await source(page)).toBe(text);
    expect(await page.getByTestId('notes-document-status').textContent()).not.toBe('Formatted');
  });
});

test('actual manual Check and Format share byte/depth bounds without changing native draft', async ({
  page,
  request,
}) => {
  test.setTimeout(60_000);
  const mark = (step: string) =>
    appendFileSync(
      '/private/tmp/i3-evidence/bounds-steps.jsonl',
      JSON.stringify({ step, time: Date.now() }) + '\n'
    );
  await owned(page, request, '{}', 'json', async () => {
    await expect.poll(() => source(page)).toBe('{}');
    const shortcuts = await keys(page);
    for (const text of [' '.repeat(1024 * 1024) + '{}', '['.repeat(101) + '0' + ']'.repeat(101)]) {
      mark('paste-start-' + text.length);
      await input(page).press(shortcuts.all);
      mark('select-all-complete');
      if (text.length > 1024 * 1024) {
        await page.evaluate(() => {
          const e = (
            window as typeof window & {
              monaco?: { editor: { getEditors: () => Array<Parameters<OnMount>[0]> } };
            }
          ).monaco?.editor
            .getEditors()
            .find((e) => e.getDomNode()?.isConnected);
          e?.trigger('i3.native-paste-control', 'paste', { text: ' '.repeat(1024 * 1024) + '{}' });
        });
      } else await page.keyboard.insertText(text);
      mark('paste-end');
      await expect.poll(() => source(page)).toBe(text);
      mark('source-match');
      const title = page.getByRole('textbox', { name: 'Note title', exact: true });
      await title.fill('I3 bounds liveness');
      await expect(title).toHaveValue('I3 bounds liveness');
      mark('title-owner-liveness');
      await page.getByRole('button', { name: 'Check document', exact: true }).click();
      await expect(page.getByTestId('notes-document-status')).toHaveText(
        'Check reached a resource limit; source kept'
      );
      await page.getByRole('button', { name: 'Format document', exact: true }).click();
      await expect(page.getByTestId('notes-document-status')).toHaveText(
        'Check reached a resource limit; source kept'
      );
      expect(await source(page)).toBe(text);
      mark('check-format-source-kept');
    }
    await input(page).press(shortcuts.all);
    mark('16KiB-CDP-start');
    await page.keyboard.insertText(' '.repeat(16384) + '{}');
    mark('16KiB-CDP-returned');
    await expect.poll(() => source(page)).toBe(' '.repeat(16384) + '{}');
    await page.getByRole('button', { name: 'Check document', exact: true }).click();
    await expect(page.getByTestId('notes-document-status')).toHaveText(
      'Checked; no problems found'
    );
    mark('16KiB-source-check-live');
  });
});

test('actual structured controls remain accessible at desktop/mobile and future mode saves literally', async ({
  page,
  request,
}, info) => {
  await owned(page, request, '{"a":}', 'json', async () => {
    await expect(page.getByRole('list', { name: 'Problems', exact: true })).toBeVisible();
    await expect(page.getByTestId('notes-document-status')).toHaveAttribute('aria-live', 'off');
    await page.getByRole('button', { name: 'Check document', exact: true }).click();
    await expect(page.getByTestId('notes-document-status')).toHaveAttribute('aria-live', 'polite');
    for (const [name, width, height] of [
      ['desktop', 1280, 900],
      ['mobile', 390, 844],
    ] as const) {
      await page.setViewportSize({ width, height });
      const closeList = page.getByRole('button', { name: 'Close notes list', exact: true });
      if (await closeList.isVisible())
        await closeList.click({ position: { x: width - 10, y: 10 } });

      if (name === 'mobile')
        await expect
          .poll(() =>
            page
              .locator('aside')
              .last()
              .evaluate((element) => element.getBoundingClientRect().right)
          )
          .toBeLessThanOrEqual(0);
      const controls = page.getByTestId('notes-document-tools');
      await expect(controls).toBeVisible();
      for (const button of ['Check document', 'Format document'])
        await expect(controls.getByRole('button', { name: button, exact: true })).toBeVisible();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)
      ).toBeTruthy();
      await page.screenshot({ path: info.outputPath(`structured-${name}.png`) });
    }
  });
  const futurePage = await page.context().newPage();
  await owned(futurePage, request, ' literal <b>😀</b> ', '__default', async (note) => {
    await expect(futurePage.getByRole('combobox', { name: 'Language', exact: true })).toBeVisible();
    await expect(futurePage.getByText('Unsupported: __default', { exact: true })).toBeVisible();
    await expect(
      futurePage.getByRole('button', { name: 'Format document', exact: true })
    ).toBeDisabled();
    const editor = input(futurePage);
    await editor.focus();
    const key = await keys(futurePage);
    await futurePage.keyboard.press(key.end);
    await futurePage.keyboard.insertText('X');
    await futurePage.getByTestId('notes-save-button').click();
    await expect
      .poll(async () => {
        const saved = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
          headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
        });
        const body = await saved.json();
        return [body.content, body.content_language];
      })
      .toEqual([' literal <b>😀</b> X', '__default']);
  });
});

test('pending actual Markdown attachment cannot rewrite a newly structured source', async ({
  page,
  request,
}, info) => {
  const steps: string[] = [];
  const mark = (step: string) => {
    steps.push(step);
    writeFileSync(info.outputPath('attachment-steps.json'), JSON.stringify(steps));
  };
  page.setDefaultTimeout(15000);
  await owned(page, request, 'body', 'markdown', async () => {
    await page.getByTestId('notes-input-mode-wysiwyg').click();
    mark('wysiwyg');
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let entered = false;
    await page.route('**/*/attachments*', async (route) => {
      entered = true;
      mark('transport-held');
      await gate;
      await route.continue();
    });
    mark('route-installed');
    await page.getByTestId('notes-attachment-input').setInputFiles({
      name: 'i3-source-control.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('harmless task-owned attachment'),
    });
    mark('file-input-returned');
    await expect.poll(() => entered).toBe(true);
    const language = page.getByRole('combobox', { name: 'Language', exact: true });
    await language.focus();
    await language.press('ArrowDown');
    mark('language-open');
    await page
      .locator('.ant-select-item-option-content')
      .getByText('YAML', { exact: true })
      .click();
    mark('yaml-selected');
    await expect.poll(() => source(page)).toBe('body');
    const uploaded = page.waitForResponse(
      (response) =>
        response.url().includes('/attachments') && response.request().method() === 'POST'
    );
    release();
    const response = await uploaded;
    expect(response.ok()).toBeTruthy();
    await page.getByRole('button', { name: 'Check document', exact: true }).click();
    await expect(page.getByTestId('notes-document-status')).toHaveText(
      'Checked; no problems found'
    );
    expect(await source(page)).toBe('body');
  });
});
