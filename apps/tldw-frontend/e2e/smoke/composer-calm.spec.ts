import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const prepareOfflinePreview = async (page: Page) => {
  await page.route('http://127.0.0.1:8000/**', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Offline preview' }),
    })
  );
  await page.route('**/api/v1/llm/models/metadata**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ models: [] }),
    })
  );
  await page.route('**/api/v1/llm/providers**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ providers: [] }),
    })
  );
  await page.addInitScript(() => {
    localStorage.setItem('assistant_setup_dismissed', 'true');
    localStorage.setItem('__tldw_first_run_complete', 'true');
    localStorage.setItem('__tldw_test_bypass', 'true');
    localStorage.setItem('stickyChatInput', 'true');
    localStorage.setItem(
      'tldwConfig',
      JSON.stringify({
        serverUrl: 'http://127.0.0.1:8000',
        authMode: 'single-user',
        apiKey: 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY',
      })
    );
  });
};

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  for (const theme of ['dark', 'light'] as const) {
    test(`calm composer ${viewport.name} ${theme}`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await prepareOfflinePreview(page);
      await page.addInitScript((value) => localStorage.setItem('theme', value), theme);
      await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
      await page.setViewportSize(viewport);
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      const input = page.getByTestId('chat-input');
      await expect(input).toBeVisible({ timeout: 30_000 });
      const switchTheme = page
        .getByRole('button', { name: `Switch to ${theme} theme`, exact: true })
        .first();
      if (await switchTheme.isVisible()) await switchTheme.click();
      await expect(page.locator('html')).toHaveClass(theme === 'dark' ? /dark/ : /light/);
      const tools = page.getByTestId('composer-options-toggle');
      await expect(tools).toHaveAccessibleName('Tools');
      await expect(tools).toHaveAttribute('aria-expanded', 'false');
      // The short-viewport collapse (AX-04) stays out of taller windows.
      await expect(page.getByTestId('composer-short-options-toggle')).toBeHidden();
      for (const control of [
        page.getByTestId('model-selector'),
        page.getByTestId('attachment-button'),
        page.getByRole('button', { name: 'Send message', exact: true }),
      ]) {
        await expect(control).toBeVisible();
        const box = await control.boundingBox();
        expect(box!.width).toBeGreaterThan(24);
        expect(box!.x).toBeGreaterThanOrEqual(0);
        expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width);
        expect(box!.y + box!.height).toBeLessThanOrEqual(viewport.height);
      }
      // '@' mentions are browser tabs, which only the extension reads (CC-11).
      const hints = page.getByText('/ commands', { exact: true });
      const saved = page.getByTestId('composer-persistence');
      await expect(hints).toBeVisible();
      await expect(saved).toBeVisible();
      const model = page.getByTestId('model-selector');
      const delivery = page.getByRole('button', { name: 'Send message', exact: true });
      const rows = viewport.name === 'mobile'
        ? [[tools, hints], [saved, model, delivery]]
        : [[tools, hints, saved, model, delivery]];
      for (const row of rows) {
        const boxes = await Promise.all(row.map(async (control) =>
          (await control.boundingBox())!
        ));
        expect(Math.max(...boxes.map((box) => box.y))).toBeLessThan(
          Math.min(...boxes.map((box) => box.y + box.height))
        );
      }
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath(`${viewport.name}-${theme}-closed.png`),
      });
      await input.fill('Keep this draft while I choose a tool.');
      await tools.click();
      const panel = page.getByRole('region', { name: 'Composer tools' });
      await expect(panel).toBeVisible();
      await expect(
        page.getByRole('button', { name: 'Search & Context', exact: true })
      ).toBeVisible();
      await expect(
        page.getByRole('button', { name: 'Advanced controls', exact: true })
      ).toHaveCount(0);
      await expect(input).toHaveValue('Keep this draft while I choose a tool.');
      const relatedRows = [
        [model, page.getByTestId('dictation-button')],
        [page.getByTestId('chat-prompt-select'), panel.getByRole('button', { name: 'System prompts', exact: true })],
        [panel.getByRole('button', { name: 'Chat Settings', exact: true }), panel.getByRole('button', { name: 'Generation style', exact: true })],
        [panel.getByTestId('composer-formatting-guide-toggle'), panel.getByTestId('openui-request-mode-toggle')],
        [panel.getByTestId('knowledge-search-toggle'), panel.getByTestId('web-search-toggle')],
      ];
      for (const row of relatedRows) {
        const boxes = await Promise.all(row.map(async (control) =>
          (await control.boundingBox())!
        ));
        expect(Math.max(...boxes.map((box) => box.y))).toBeLessThan(
          Math.min(...boxes.map((box) => box.y + box.height))
        );
      }
      const accessibility = await new AxeBuilder({ page })
        .include('[data-testid="composer-toolbar"]')
        .analyze();
      expect(accessibility.violations).toEqual([]);
      await page.screenshot({
        animations: 'disabled',
        path: testInfo.outputPath(`${viewport.name}-${theme}-tools.png`),
      });
      await page.keyboard.press('Escape');
      await expect(panel).toBeHidden();
      await expect(tools).toBeFocused();
      expect(errors).toEqual([]);
    });
  }
}

// CO-06 (#3113): at 1024x768 the review found the chrome left the
// transcript about 340px. Connected, with a model chosen and Tools closed,
// the header, view toolbar and composer must leave it most of the window.
const MODEL_ID = 'openai:gpt-4.1-mini';
const connectWithOneModel = async (page: Page) => {
  // Registered after prepareOfflinePreview, so these win over its 503s: the
  // server answers, with nothing stored, and one configured model.
  await page.route('http://127.0.0.1:8000/**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
  );
  await page.route('**/api/v1/health', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', checks: {} }),
    })
  );
  await page.route('**/api/v1/llm/providers**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        providers: [
          {
            id: 'openai',
            name: 'OpenAI',
            apiProvider: 'openai',
            models: ['gpt-4.1-mini'],
            configured: true,
            available: true,
          },
        ],
      }),
    })
  );
  await page.route('**/api/v1/llm/models/metadata**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: [
          {
            id: MODEL_ID,
            model: 'gpt-4.1-mini',
            name: 'GPT-4.1 Mini',
            provider: 'openai',
            apiProvider: 'openai',
            capabilities: ['chat'],
          },
        ],
      }),
    })
  );
  await page.addInitScript((modelId) => {
    localStorage.setItem('selectedModel', JSON.stringify(modelId));
  }, MODEL_ID);
};

test('leaves the transcript most of a 1024x768 window', async ({ page }, testInfo) => {
  const viewport = { width: 1024, height: 768 };
  await prepareOfflinePreview(page);
  await connectWithOneModel(page);
  await page.addInitScript(() => {
    localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
  });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize(viewport);
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30_000 });
  const exitFocus = page.getByRole('button', { name: 'Exit focus', exact: true });
  if (await exitFocus.isVisible()) await exitFocus.click();
  // The everyday chrome: the conversation header, a chosen model and a
  // closed Tools tray.
  await expect(page.getByTestId('chat-conversation-header')).toBeVisible();
  await expect(page.getByTestId('model-selector')).not.toHaveText(/select a model|loading/i, {
    timeout: 30_000,
  });
  await expect(page.getByTestId('composer-options-toggle')).toHaveAttribute(
    'aria-expanded',
    'false'
  );
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise(requestAnimationFrame);
    await new Promise(requestAnimationFrame);
  });
  const transcript = (await page.getByTestId('playground-chat-transcript').boundingBox())!;
  const dock = (await page.getByTestId('playground-chat-composer-dock').boundingBox())!;
  await page.screenshot({ animations: 'disabled', path: testInfo.outputPath('1024x768.png') });
  expect(transcript.y + transcript.height).toBeLessThanOrEqual(dock.y + 1);
  expect(
    transcript.height,
    `transcript ${transcript.height}px from y=${transcript.y}; composer dock ${dock.height}px`
  ).toBeGreaterThanOrEqual(viewport.height * 0.6);
});

// CC-10 (#3101), decision [2]: Search & Context stays in the composer with its
// own height cap, so the textarea and Send stay in view while it is open.
for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 1024, height: 768 },
]) {
  test(`keeps the input and Send in view with Search & Context open at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await prepareOfflinePreview(page);
    await page.addInitScript(() => {
      localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
    });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.setViewportSize(viewport);
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });
    const input = page.getByTestId('chat-input');
    await expect(input).toBeVisible({ timeout: 30_000 });
    const tools = page.getByTestId('composer-options-toggle');
    if ((await tools.getAttribute('aria-expanded')) !== 'true') await tools.click();
    await page.getByTestId('knowledge-search-toggle').click();
    if ((await tools.getAttribute('aria-expanded')) === 'true') await tools.click();
    await expect(tools).toHaveAttribute('aria-expanded', 'false');
    const section = page.getByTestId('composer-knowledge-section');
    await expect(section.getByRole('heading', { name: 'Search & Context', exact: true })).toBeVisible();
    const send = page.getByRole('button', { name: 'Send message', exact: true });
    for (const control of [input, send]) {
      await expect(control).toBeInViewport({ ratio: 1 });
      const box = (await control.boundingBox())!;
      const hit = await page.evaluate(
        ({ x, y }) => document.elementFromPoint(x, y)?.closest('textarea, button')?.outerHTML.slice(0, 80) ?? null,
        { x: box.x + box.width / 2, y: box.y + box.height / 2 }
      );
      expect(hit, 'nothing covers the control').not.toBeNull();
    }
    const sectionBox = (await section.boundingBox())!;
    expect(sectionBox.height).toBeLessThanOrEqual(Math.min(viewport.height * 0.3, 352) + 1);
  });
}

// AX-04 (#3124), decision [14]: at 200-400% zoom the window is a few hundred
// CSS px tall. The toolbar row collapses first behind "Composer options"; the
// textarea and Send stay, and the message log keeps a usable height. The log
// is measured by its scroll viewport (the transcript), since an empty chat's
// role=log has no height of its own.
// Landscape phones of 768px or wider open in cockpit by default (AX-04
// review), and a saved cockpit layout with a rail panel open must keep the
// composer too.
for (const { viewport, seed } of [
  { viewport: { width: 320, height: 256 }, seed: '' },
  { viewport: { width: 640, height: 400 }, seed: '' },
  { viewport: { width: 844, height: 390 }, seed: '' },
  { viewport: { width: 640, height: 400 }, seed: 'cockpit+rail' },
]) {
  test(`reflows the composer at ${viewport.width}x${viewport.height}${seed ? ` (${seed})` : ''}`, async ({ page }, testInfo) => {
    await prepareOfflinePreview(page);
    if (seed)
      await page.addInitScript(() => {
        localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
        localStorage.setItem('playgroundChatContextRailVisible', 'true');
        localStorage.setItem('playgroundChatMobileCockpitPanel', JSON.stringify('context'));
      });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.setViewportSize(viewport);
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });
    const input = page.getByTestId('chat-input');
    await expect(input).toBeVisible({ timeout: 30_000 });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
    });
    const send = page.getByRole('button', { name: 'Send message', exact: true });
    const transcript = page.getByTestId('playground-chat-transcript');
    const dock = page.getByTestId('playground-chat-composer-dock');
    const expectUncovered = async () => {
      for (const control of [input, send]) {
        await expect(control).toBeInViewport({ ratio: 1 });
        const box = (await control.boundingBox())!;
        const covered = await control.evaluate(
          (element, point) => {
            const hit = document.elementFromPoint(point.x, point.y);
            return !(hit && (hit === element || element.contains(hit)));
          },
          { x: box.x + box.width / 2, y: box.y + box.height / 2 }
        );
        expect(covered, 'the control is not covered').toBe(false);
      }
    };
    await expect(page.getByTestId('composer-options-toggle')).toBeHidden();
    await expectUncovered();
    const log = (await transcript.boundingBox())!;
    const composer = (await dock.boundingBox())!;
    expect(log.height, `log ${log.height}px, composer ${composer.height}px`).toBeGreaterThanOrEqual(100);
    testInfo.annotations.push({
      type: 'composer share',
      description: `${viewport.width}x${viewport.height}${seed ? ` ${seed}` : ''}: composer ${Math.round(composer.height)}px (${Math.round((composer.height / viewport.height) * 100)}%), log ${Math.round(log.height)}px`,
    });
    await page.screenshot({
      animations: 'disabled',
      path: testInfo.outputPath(`reflow-${viewport.width}x${viewport.height}${seed ? '-rail' : ''}.png`),
    });

    const options = page.getByTestId('composer-short-options-toggle');
    await expect(options).toHaveAccessibleName('Composer options');
    await options.click();
    await expect(options).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByTestId('composer-options-toggle')).toBeVisible();
    expect((await transcript.boundingBox())!.height).toBeGreaterThanOrEqual(100);
    await options.click();
    await expect(page.getByTestId('composer-options-toggle')).toBeHidden();
  });
}

// CC-10 review: after a search, the capped box brings its results into view,
// so the first source and Generate answer show without scrolling.
const RAG_DOCUMENTS = Array.from({ length: 5 }, (_, index) => ({
  id: `doc-${index + 1}`,
  content: `Embedding models are evaluated on held-out queries. Result ${index + 1}.`,
  score: 0.9 - index * 0.1,
  metadata: { title: `Evaluation note ${index + 1}`, type: 'note' },
}));
for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 1024, height: 768 },
]) {
  test(`brings Search & Context results into view after a search at ${viewport.width}x${viewport.height}`, async ({ page }, testInfo) => {
    await prepareOfflinePreview(page);
    await connectWithOneModel(page);
    await page.route('**/api/v1/rag/search**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ documents: RAG_DOCUMENTS, query: 'embedding model evaluation' }),
      })
    );
    await page.addInitScript(() => {
      localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
    });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.setViewportSize(viewport);
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30_000 });
    const tools = page.getByTestId('composer-options-toggle');
    if ((await tools.getAttribute('aria-expanded')) !== 'true') await tools.click();
    await page.getByTestId('knowledge-search-toggle').click();
    if ((await tools.getAttribute('aria-expanded')) === 'true') await tools.click();
    const section = page.getByTestId('composer-knowledge-section');
    const query = section.getByRole('textbox', { name: 'Search your knowledge...' });
    await expect(query).toBeEnabled({ timeout: 30_000 });
    await query.fill('embedding model evaluation');
    await query.press('Enter');
    const header = section.getByTestId('qa-results-header');
    await expect(header).toContainText('Source Chunks (5)');
    const generate = header.getByRole('button', { name: 'Generate answer' });
    const firstResult = section.getByRole('listitem').first();
    await page.evaluate(() => new Promise(requestAnimationFrame));
    const box = (await section.boundingBox())!;
    for (const [name, locator] of [
      ['Generate answer', generate],
      ['the first result', firstResult],
    ] as const) {
      const target = (await locator.boundingBox())!;
      expect(target.y, `${name} top ${target.y} inside ${box.y}..${box.y + box.height}`).toBeGreaterThanOrEqual(box.y - 1);
      expect(target.y + Math.min(target.height, 40), `${name} visible in the box`).toBeLessThanOrEqual(box.y + box.height + 1);
      testInfo.annotations.push({
        type: 'search results',
        description: `${viewport.width}x${viewport.height}: ${name} at +${Math.round(target.y - box.y)}px in a ${Math.round(box.height)}px box`,
      });
    }
    await expect(page.getByRole('button', { name: 'Send message', exact: true })).toBeInViewport();
  });
}

// Decision [3]: toolbar controls are 32px on desktop and 44px on touch.
const sizedControls = (page: Page) => [
  page.getByTestId('composer-options-toggle'),
  page.getByTestId('attachment-button'),
  page.getByTestId('model-selector'),
  page.getByTestId('composer-persistence'),
];
const openComposerAt1440 = async (page: Page) => {
  await prepareOfflinePreview(page);
  await page.addInitScript(() => {
    localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30_000 });
};

test('sizes toolbar controls at 32px on a desktop pointer', async ({ page }) => {
  await openComposerAt1440(page);
  expect(await page.evaluate(() => matchMedia('(pointer: coarse)').matches)).toBe(false);
  for (const control of sizedControls(page)) {
    const box = (await control.boundingBox())!;
    expect(Math.round(box.height), await control.getAttribute('data-testid') ?? '').toBe(32);
  }
});

test.describe('on a touch screen', () => {
  test.use({ hasTouch: true });
  test('sizes toolbar controls at 44px for a coarse pointer, even at 1440px', async ({ page }) => {
    await openComposerAt1440(page);
    expect(await page.evaluate(() => matchMedia('(pointer: coarse)').matches)).toBe(true);
    for (const control of sizedControls(page)) {
      const box = (await control.boundingBox())!;
      expect(Math.round(box.height), await control.getAttribute('data-testid') ?? '').toBe(44);
    }
  });
});

test('remembers whether Tools was expanded across reload', async ({ page }) => {
  await prepareOfflinePreview(page);
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  const tools = page.getByTestId('composer-options-toggle');
  await expect(tools).toHaveAttribute('aria-expanded', 'false');
  await tools.click();
  await expect(tools).toHaveAttribute('aria-expanded', 'true');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(tools).toHaveAttribute('aria-expanded', 'true');
});

for (const variant of ['v1', 'v3', 'v5']) {
  test(`keeps one delivery control in the optional ${variant} shell`, async ({ page }) => {
    await prepareOfflinePreview(page);
    await page.addInitScript(
      (value) => localStorage.setItem('tldw:composerVariant', value),
      variant
    );
    await page.goto('/chat?nextgenComposer=1', { waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('nextgen-composer-wrapper')).toBeVisible();
    await expect(page.getByTestId('chat-input')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Send message', exact: true })).toHaveCount(1);
    await expect(page.getByTestId('model-selector')).toBeVisible();
  });
}
