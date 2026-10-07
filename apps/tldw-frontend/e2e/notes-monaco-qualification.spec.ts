/** Opt-in removable real Notes-page candidate; skipped runs are not qualification. */
import { test, expect, type Page, type APIRequestContext } from '@playwright/test';
import type {} from '../../packages/ui/src/components/Notes/__tests__/fixtures/MonacoNotesQualification';
import { seedAuth, TEST_CONFIG } from './utils/helpers';

test.skip(
  process.env.TLDW_I1_MONACO !== '1',
  'Apply the candidate patch and prepare local assets first; not production I3 coverage.'
);
const A = '{"a":1}',
  C = '{\n  "a": 1\n}';
const source = (page: Page) => page.getByTestId('monaco-source').textContent();
const snapshot = (page: Page) => page.evaluate(() => window.__i1Monaco!.snapshot());
const editor = (page: Page) => page.getByRole('textbox', { name: 'Note content', exact: true });
async function bindings(page: Page) {
  const mac = await page.evaluate(() => /Macintosh/.test(navigator.userAgent));
  return {
    end: mac ? 'Meta+ArrowDown' : 'Control+End',
    undo: mac ? 'Meta+z' : 'Control+z',
    redo: mac ? 'Meta+Shift+z' : 'Control+y',
  };
}
async function ownedNote(
  page: Page,
  request: APIRequestContext,
  text: string,
  run: (title: string) => Promise<void>,
  language = 'json'
) {
  const headers = { 'X-API-KEY': TEST_CONFIG.apiKey };
  const title = `I1 Monaco ${Date.now()} ${Math.random().toString(36).slice(2)}`;
  const response = await request.post(`${TEST_CONFIG.serverUrl}/api/v1/notes/`, {
    headers,
    data: { title, content: text, content_language: language },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
  const note = await response.json();
  try {
    await seedAuth(page);
    await page.goto('/notes');
    await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
    await expect.poll(() => source(page)).toBe(text);
    await run(title);
  } finally {
    await page.close();
    const current = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
      headers,
    });
    const removed = await request.delete(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
      headers: { ...headers, 'expected-version': String((await current.json()).version) },
    });
    expect(removed.ok(), await removed.text()).toBeTruthy();
  }
}
async function ready(page: Page) {
  await expect(page.getByRole('button', { name: 'Format document', exact: true })).toBeEnabled();
  await editor(page).focus();
}
async function apply(page: Page, replacement: string) {
  await page.evaluate((text) => window.__i1Monaco!.capture(text), replacement);
  await page.getByRole('button', { name: 'Apply captured format', exact: true }).click();
}
async function history(page: Page, a: string, c: string) {
  await ready(page);
  const keys = await bindings(page),
    input = editor(page);
  const trace: Record<string, unknown> = { shortcuts: keys, A: await source(page) };
  await input.press(keys.end);
  await input.pressSequentially(' ');
  await expect.poll(() => source(page)).toBe(a + ' ');
  trace.B = await source(page);
  await apply(page, c);
  await expect.poll(() => source(page)).toBe(c);
  trace.C = await source(page);
  await expect(input).toBeFocused();
  await input.press(keys.end);
  await input.pressSequentially(' ');
  await expect.poll(() => source(page)).toBe(c + ' ');
  trace.D = await source(page);
  for (const [name, key, expected] of [
    ['undoD', keys.undo, c],
    ['undoC', keys.undo, a + ' '],
    ['redoC', keys.redo, c],
    ['undoC2', keys.undo, a + ' '],
    ['undoB', keys.undo, a],
  ]) {
    await input.press(key);
    await expect.poll(() => source(page)).toBe(expected);
    trace[name] = await source(page);
  }
  return trace;
}
for (const [profile, a, c] of [
  ['LF', A, C],
  ['CRLF', '{\r\n"a":1\r\n}\r\n', '{\r\n  "a": 1\r\n}\r\n'],
]) {
  test(`native ${profile} A/B/C/D history through the real Notes handler`, async ({
    page,
    request,
    browser,
  }, info) => {
    await ownedNote(page, request, a, async () => {
      const trace = { browser: browser.version(), profile, ...(await history(page, a, c)) };
      console.log(JSON.stringify(trace));
      await info.attach('native-trace.json', {
        body: JSON.stringify(trace, null, 2),
        contentType: 'application/json',
      });
    });
  });
}

test('current identity/source, no-op, focus, read-only and editability admission', async ({
  page,
  request,
}) => {
  await ownedNote(page, request, A, async () => {
    await ready(page);
    const before = await snapshot(page);
    for (const field of Object.keys(before.key)) {
      const changed = typeof before.key[field] === 'string' ? 'other' : before.key[field] + 1;
      expect(
        await page.evaluate(
          ({ field, changed, c }) => {
            const probe = window.__i1Monaco!;
            probe.capture(c);
            return probe.complete({ key: { [field]: changed } });
          },
          { field, changed, c: C }
        )
      ).toBe(false);
    }
    expect(
      await page.evaluate((c) => {
        const p = window.__i1Monaco!;
        p.capture(c);
        return p.complete({ editable: false });
      }, C)
    ).toBe(false);
    expect(
      await page.evaluate((c) => {
        const p = window.__i1Monaco!;
        p.capture(c);
        p.readOnly(true);
        return p.complete();
      }, C)
    ).toBe(false);
    await page.evaluate(() => window.__i1Monaco!.readOnly(false));
    // Captured output cannot cross a real editor blur; after refocus it remains usable.
    await page.evaluate((c) => window.__i1Monaco!.capture(c), C);
    await page.getByRole('button', { name: 'Apply captured format', exact: true }).focus();
    expect(await page.evaluate(() => window.__i1Monaco!.complete())).toBe(false);
    await editor(page).focus();
    expect(await page.evaluate(() => window.__i1Monaco!.complete())).toBe(true);
    const keys = await bindings(page);
    await editor(page).press(keys.undo);
    await expect.poll(() => source(page)).toBe(A);
    // Same source is not a transaction. Typing while a result is pending invalidates it.
    expect(
      await page.evaluate((a) => {
        const p = window.__i1Monaco!;
        p.capture(a);
        return p.complete();
      }, A)
    ).toBe(false);
    await page.evaluate((c) => window.__i1Monaco!.capture(c), C);
    await editor(page).press(keys.end);
    await editor(page).pressSequentially(' ');
    await expect.poll(() => source(page)).toBe(A + ' ');
    expect(await page.evaluate(() => window.__i1Monaco!.complete())).toBe(false);
    await editor(page).press(keys.undo);
    await expect.poll(() => source(page)).toBe(A);
  });
});

test('literal whitespace, non-BMP caret and oriented selections survive format and native undo', async ({
  page,
  request,
}) => {
  const a = '{"s":"😀x  y","n":9007199254740993}',
    c = '{\n  "s": "😀x  y",\n  "n": 9007199254740993\n}';
  await ownedNote(page, request, a, async () => {
    await ready(page);
    const keys = await bindings(page);
    const spaces = a.indexOf('  '),
      mapped = c.indexOf('  y');
    for (const [start, end, wantStart, wantEnd] of [
      [spaces, spaces + 2, mapped, mapped + 2],
      [spaces + 2, spaces, mapped + 2, mapped],
      [spaces + 1, spaces + 1, mapped + 1, mapped + 1],
      [a.indexOf('😀') + 2, a.indexOf('😀') + 2, c.indexOf('😀') + 2, c.indexOf('😀') + 2],
    ]) {
      await page.evaluate(([a, b]) => window.__i1Monaco!.select(a, b), [start, end]);
      await apply(page, c);
      await expect.poll(() => source(page)).toBe(c);
      expect((await snapshot(page)).selection).toEqual([wantStart, wantEnd]);
      await editor(page).press(keys.undo);
      await expect.poll(() => source(page)).toBe(a);
      expect((await snapshot(page)).selection).toEqual([start, end]);
    }
  });
});

test('mounted synthetic composition defers a pending format until composition ends', async ({
  page,
  request,
}) => {
  await ownedNote(page, request, A, async () => {
    await ready(page);
    await page.evaluate((c) => window.__i1Monaco!.capture(c), C);
    await editor(page).dispatchEvent('compositionstart', { data: '' });
    expect((await snapshot(page)).composing).toBe(true);
    expect(await page.evaluate(() => window.__i1Monaco!.complete())).toBe(false);
    await expect.poll(() => source(page)).toBe(A);
    await editor(page).dispatchEvent('compositionend', { data: '' });
    expect((await snapshot(page)).composing).toBe(false);
    expect(await page.evaluate(() => window.__i1Monaco!.complete())).toBe(true);
    await expect.poll(() => source(page)).toBe(C);
  });
});

for (const [profile, text] of [
  ['BOM', '\ufeff{"s":"😀  a&b <tag>"}'],
  ['mixed', 'a\r\nb\n'],
  ['lone CR', 'a\rb'],
]) {
  test(`refuses ${profile} before editable mount and retains exact source`, async ({
    page,
    request,
  }) => {
    await ownedNote(page, request, text, async () => {
      await expect(
        page.getByRole('status').filter({ hasText: 'Representation refused' })
      ).toBeVisible();
      expect(await editor(page).textContent()).toBe(text);
      await expect(editor(page)).toHaveAttribute('aria-readonly', 'true');
      expect(await source(page)).toBe(text);
      expect(await page.evaluate(() => window.__i1Monaco)).toBeUndefined();
    });
  });
}

test('exact lexemes, viewport, local loader and worker requests', async ({
  page,
  request,
}, info) => {
  const text =
    's: "a&b <tag> 😀  \\u0001"\nnum: [9007199254740993, 1e400, -0, 0.12345678901234567890]\n' +
    Array.from({ length: 120 }, (_, i) => `k${i}: ${'x'.repeat(160)}  # comment\n`).join('');
  const replacement = '# formatted\n' + text;
  const requests: string[] = [],
    workers: string[] = [],
    leaks: string[] = [];
  page.on('request', (r) => {
    requests.push(r.url());
    if (
      r.postData()?.includes('9007199254740993') &&
      !r.url().startsWith(TEST_CONFIG.serverUrl + '/api/v1/notes')
    )
      leaks.push(r.url());
  });
  page.on('worker', (w) => workers.push(w.url()));
  await ownedNote(
    page,
    request,
    text,
    async (title) => {
      await ready(page);
      await page.evaluate(() => {
        window.__i1Monaco!.select(6000, 6000);
        window.__i1Monaco!.scroll(900, 120);
      });
      await expect
        .poll(async () =>
          page.evaluate(() => {
            window.__i1Monaco!.scroll(900, 120);
            return window.__i1Monaco!.snapshot().left;
          })
        )
        .toBe(120);
      const before = await snapshot(page);
      expect(before.top).toBe(900);
      await apply(page, replacement);
      await expect.poll(() => source(page)).toBe(replacement);
      const after = await snapshot(page);
      expect(after.top).toBe(before.top);
      expect(after.left).toBe(before.left);
      expect(after.source).toBe(replacement);
      // Native explicit suggestion asks Monaco's installed editor worker for document words.
      await editor(page).press('Control+Space');
      await expect.poll(() => workers.length).toBeGreaterThan(0);
      const assets = requests.filter((url) => url.includes('/i1-monaco/'));
      expect(assets.some((url) => url.endsWith('/loader.js'))).toBe(true);
      expect(assets.some((url) => url.endsWith('/editor/editor.main.js'))).toBe(true);
      expect(
        workers.every(
          (url) => url === new URL('/i1-monaco/editor-worker.js', TEST_CONFIG.webUrl).href
        )
      ).toBe(true);
      // Firefox does not report importScripts requests through page.on('request').
      // Inspect the real worker global after its local bootstrap loaded the distribution.
      await expect
        .poll(async () =>
          Promise.all(
            page
              .workers()
              .map((worker) =>
                worker.evaluate(() => typeof (globalThis as { monaco?: unknown }).monaco)
              )
          )
        )
        .toEqual(['object']);
      const workerResources = await page
        .workers()[0]
        .evaluate(() => performance.getEntriesByType('resource').map((entry) => entry.name));
      expect(
        assets.every((url) => new URL(url).origin === new URL(TEST_CONFIG.webUrl).origin)
      ).toBe(true);
      expect(requests.filter((url) => /cdn|jsdelivr|unpkg/.test(url))).toEqual([]);
      expect(leaks).toEqual([]);
      const trace = {
        assets,
        workers,
        workerResources,
        leaks,
        source: after.source,
        viewport: { before: [before.top, before.left], after: [after.top, after.left] },
      };
      console.log(JSON.stringify({ assetCount: assets.length, workers, viewport: trace.viewport }));
      await info.attach('assets-source.json', {
        body: JSON.stringify(
          { ...trace, assets: requests.filter((url) => url.includes('/i1-monaco/')) },
          null,
          2
        ),
        contentType: 'application/json',
      });
      await editor(page).press('Escape');
      await page.getByTestId('notes-save-button').click();
      const id = after.key.document_id;
      await expect
        .poll(async () => {
          const saved = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${id}`, {
            headers: { 'X-API-KEY': TEST_CONFIG.apiKey },
          });
          expect(saved.ok()).toBeTruthy();
          return (await saved.json()).content;
        })
        .toBe(replacement);
      await page.reload();
      await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
      await expect.poll(() => source(page)).toBe(replacement);
    },
    'yaml'
  );
});

test('real note navigation disposes models and invalidates old pending formats', async ({
  page,
  request,
}) => {
  await ownedNote(page, request, A, async () => {
    await ready(page);
    const first = await snapshot(page);
    await page.evaluate((c) => window.__i1Monaco!.capture(c), C);
    const old = await page.evaluateHandle(() => window.__i1Monaco!);
    const headers = { 'X-API-KEY': TEST_CONFIG.apiKey };
    const title = `I1 second ${Date.now()}`,
      content = 'second 😀  a&b <tag>  ';
    const response = await request.post(`${TEST_CONFIG.serverUrl}/api/v1/notes/`, {
      headers,
      data: { title, content, content_language: 'json' },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const note = await response.json();
    try {
      // Search/refetch through the real list; no model setters or history methods.
      await page.getByRole('textbox', { name: 'Search notes', exact: true }).fill(title);
      await page.getByRole('button', { name: `Open note ${title}`, exact: true }).click();
      await expect.poll(() => source(page)).toBe(content);
      await ready(page);
      const second = await snapshot(page);
      expect(second.uri).not.toBe(first.uri);
      expect(second.models).toBe(1);
      expect(second.retired).toContainEqual({ uri: first.uri, disposed: true });
      expect(await old.evaluate((probe) => probe.complete())).toBe(false);
      await editor(page).press((await bindings(page)).undo);
      await expect.poll(() => source(page)).toBe(content);
    } finally {
      await old.dispose();
      const current = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
        headers,
      });
      const removed = await request.delete(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
        headers: { ...headers, 'expected-version': String((await current.json()).version) },
      });
      expect(removed.ok(), await removed.text()).toBeTruthy();
    }
  });
});

test('a real optimistic save conflict refuses the pending format and retains the draft', async ({
  page,
  request,
}) => {
  await ownedNote(page, request, A, async () => {
    await ready(page);
    const keys = await bindings(page);
    await editor(page).press(keys.end);
    await editor(page).pressSequentially(' ');
    await expect.poll(() => source(page)).toBe(A + ' ');
    await page.evaluate((c) => window.__i1Monaco!.capture(c), C);
    const id = (await snapshot(page)).key.document_id,
      headers = { 'X-API-KEY': TEST_CONFIG.apiKey };
    const current = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${id}`, { headers });
    const updated = await request.put(`${TEST_CONFIG.serverUrl}/api/v1/notes/${id}`, {
      headers: { ...headers, 'expected-version': String((await current.json()).version) },
      data: { content: '{"external":true}' },
    });
    expect(updated.ok(), await updated.text()).toBeTruthy();
    await page.getByTestId('notes-save-button').click();
    await expect(
      page.getByText(
        'Save conflict: this note changed on the server. Your local edits are still in the editor.',
        { exact: true }
      )
    ).toBeVisible();
    await editor(page).focus();
    expect(await page.evaluate(() => window.__i1Monaco!.complete())).toBe(false);
    expect(await source(page)).toBe(A + ' ');
  });
});

test('keyboard toolbar activation restores focus and stays one native undo step', async ({
  page,
  request,
}) => {
  await ownedNote(page, request, A, async () => {
    await ready(page);
    const keys = await bindings(page);
    await editor(page).press(keys.end);
    await editor(page).pressSequentially(' ');
    await expect.poll(() => source(page)).toBe(A + ' ');
    const button = page.getByRole('button', { name: 'Format document', exact: true });
    await button.focus();
    await button.press('Enter');
    await expect.poll(() => source(page)).toBe(C);
    await expect(editor(page)).toBeFocused();
    await editor(page).press(keys.undo);
    await expect.poll(() => source(page)).toBe(A + ' ');
  });
});
