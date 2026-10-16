/** Shared preview contracts from #3117; real saved content, CSS and browser effects. */
import { readFile } from 'node:fs/promises';
import type { Locator, Page } from '@playwright/test';
import { test, expect, skipIfServerUnavailable } from '../utils/fixtures';
import { NotesPage, WritingPlaygroundPage } from '../utils/page-objects';
import { generateTestId, TEST_CONFIG } from '../utils/helpers';
import { launchExtension, SidePanelChat } from '../utils/extension-sidepanel';
import {
  createCharacter,
  createChatWithMessages,
  createSeedApi,
  warmBackendOnce,
} from '../utils/seed-api';

test.use({ trace: 'retain-on-failure' });

const code = Array.from({ length: 16 }, (_, index) =>
  index === 9
    ? '    indented_line = 10'
    : `line_${index + 1} = "${'unequal'.repeat(index === 8 ? 24 : 1)}"`
).join('\n');
const prose =
  '# Preview contract\n\n- Parent item\n  - Nested item\n\n1. Ordered item\n2. Second item\n\n- [x] Complete task\n- [ ] Pending task\n\n> Quoted passage\n\n[Safe link](https://example.com)';
const content = `${prose}\n\n\`\`\`python\n${code}\n\`\`\`\n\n\`\`\`html\n<p onclick="alert('unsafe')">Inline safe preview</p><script>alert('unsafe')</script>\n\`\`\``;

async function assertProse(surface: Locator) {
  await expect(surface.locator('.prose')).toBeVisible();
  await expect(surface.locator('.prose > ul:not(.contains-task-list) > li').first()).toHaveCSS(
    'list-style-type',
    'disc'
  );
  await expect(surface.locator('.prose ul ul > li')).toHaveCSS('list-style-type', 'disc');
  await expect(surface.locator('.prose ol > li').first()).toHaveCSS('list-style-type', 'decimal');
  const tasks = surface.locator('.prose .task-list-item');
  await expect(tasks).toHaveCount(2);
  for (const task of await tasks.all()) {
    await expect(task).toHaveCSS('list-style-type', 'none');
    await expect(task.locator('input[type="checkbox"]')).toBeVisible();
  }
  await expect(tasks.first().locator('input')).toBeChecked();
  await expect(tasks.last().locator('input')).not.toBeChecked();
  const colors = await surface.locator('.prose').evaluate((element) => {
    const text = getComputedStyle(element).color;
    let background = '';
    for (let node: Element | null = element; node; node = node.parentElement) {
      const candidate = getComputedStyle(node).backgroundColor;
      if (candidate !== 'rgba(0, 0, 0, 0)' && candidate !== 'transparent') {
        background = candidate;
        break;
      }
    }
    const luminance = (color: string) => {
      const channels = (color.match(/[\d.]+/g) || [])
        .slice(0, 3)
        .map(Number)
        .map((value) => {
          const channel = value / 255;
          return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
        });
      return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
    };
    const a = luminance(text),
      b = luminance(background);
    return { text, background, contrast: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) };
  });
  expect(colors.background).not.toBe('');
  expect(colors.contrast).toBeGreaterThanOrEqual(4.5);
}

async function assertCodeColumns(block: Locator) {
  const rows = block.locator('pre .table-row');
  await expect(rows).toHaveCount(16);
  const positions = await rows.evaluateAll((elements) =>
    elements.map((element) => element.lastElementChild!.getBoundingClientRect().left)
  );
  expect(Math.max(...positions) - Math.min(...positions)).toBeLessThanOrEqual(1);
  await expect(rows.nth(8).locator('span').first()).toHaveText('9');
  await expect(rows.nth(9).locator('span').first()).toHaveText('10');
  expect(await rows.nth(9).locator(':scope > span').last().textContent()).toBe(
    '    indented_line = 10'
  );
  await expect(rows.nth(9).locator(':scope > span').last()).toHaveCSS('white-space', 'pre');
}

async function assertControls(block: Locator) {
  const buttons = block.locator('button:visible');
  const boxes = await buttons.evaluateAll((elements) =>
    elements.map((element) => {
      const { left, top, right, bottom } = element.getBoundingClientRect();
      return { left, top, right, bottom };
    })
  );
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i],
        b = boxes[j];
      expect(
        Math.min(a.right, b.right) <= Math.max(a.left, b.left) ||
          Math.min(a.bottom, b.bottom) <= Math.max(a.top, b.top)
      ).toBe(true);
    }
  for (const button of await buttons.all()) {
    await button.scrollIntoViewIfNeeded();
    expect(
      await button.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        const hit = document.elementFromPoint(
          rect.left + rect.width / 2,
          rect.top + rect.height / 2
        );
        return hit === element || element.contains(hit);
      })
    ).toBe(true);
  }
}

async function assertSourceActions(page: Page, block: Locator) {
  await block.getByRole('button', { name: /copy to clipboard/i }).click();
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(code);
  const downloadEvent = page.waitForEvent('download');
  await block.getByRole('button', { name: /download code/i }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toMatch(/^code_[0-9TZ-]+\.py$/);
  const downloadedPath = await download.path();
  expect(downloadedPath).not.toBeNull();
  expect(await readFile(downloadedPath!, 'utf8')).toBe(code);
}

for (const surface of ['WebUI', 'extension options'] as const) {
  test(`NE-05/NE-07/NE-N1: ${surface} preview and split preserve list markers and code controls`, async ({
    authedPage,
    serverInfo,
    request,
  }, testInfo) => {
    skipIfServerUnavailable(serverInfo);
    const api = createSeedApi(request);
    await warmBackendOnce(api);
    const title = generateTestId('uxr-preview-code');
    const response = await api.post('/api/v1/notes/', { title, content, keywords: [] });
    expect(response.ok()).toBe(true);
    const noteId = String((await response.json()).id);
    const extension =
      surface === 'extension options'
        ? await launchExtension({ serverUrl: TEST_CONFIG.serverUrl, apiKey: TEST_CONFIG.apiKey })
        : null;
    const page = extension ? await extension.context.newPage() : authedPage;
    const url = (route: string) =>
      extension ? `chrome-extension://${extension.extensionId}/options.html#${route}` : route;
    try {
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
      if (extension) {
        await extension.context
          .serviceWorkers()[0]
          .evaluate(() =>
            chrome.storage.local.set({ 'notes-tutorial-shown': JSON.stringify('1') })
          );
      }
      await page.goto(url('/notes'));
      const notes = new NotesPage(page);
      await page.getByTestId(`notes-open-button-${noteId}`).click();
      await expect(notes.contentTextarea).toHaveValue(content);
      const prompt = page.getByText('New to this page?', { exact: true });
      if (extension) {
        await expect(prompt).toBeVisible();
        await page.getByRole('button', { name: 'Not now', exact: true }).click();
        await expect(prompt).toBeHidden();
      }
      await page.getByRole('button', { name: 'Preview', exact: true }).click();
      let preview = page.getByTestId('notes-preview-surface');
      await assertProse(preview);
      await expect(preview.getByRole('button', { name: /view code/i })).toHaveCount(0);
      let block = preview
        .locator('[data-artifact-origin]')
        .filter({ has: page.getByText('python', { exact: true }) });
      await block.getByRole('button', { name: /^expand/i }).click();
      await assertCodeColumns(block);
      await assertControls(block);
      await assertSourceActions(page, block);
      await expect(page.locator('html')).toHaveClass(/dark/);
      await testInfo.attach(`${surface}-dark-preview`, {
        body: await page.screenshot(),
        contentType: 'image/png',
      });
      await page.getByRole('button', { name: 'Split', exact: true }).click();
      preview = page.getByTestId('notes-split-preview-surface');
      await assertProse(preview);
      block = preview
        .locator('[data-artifact-origin]')
        .filter({ has: page.getByText('python', { exact: true }) });
      await assertCodeColumns(block);
      await page.setViewportSize({ width: 360, height: 900 });
      await expect(page.getByTestId('notes-mobile-open-list-button')).toBeVisible();
      await testInfo.attach(`${surface}-mobile-mode-before-selection`, {
        body: Buffer.from(JSON.stringify({ splitVisible: await preview.isVisible() })),
        contentType: 'application/json',
      });
      await page.getByTestId('notes-overflow-menu-button').click();
      const splitMode = page.getByRole('menuitem', { name: 'Split', exact: true });
      if (await splitMode.isEnabled()) await splitMode.click();
      else await page.keyboard.press('Escape');
      preview = page.getByTestId('notes-split-preview-surface');
      await expect(preview).toBeVisible();
      block = preview
        .locator('[data-artifact-origin]')
        .filter({ has: page.getByText('python', { exact: true }) });
      await assertCodeColumns(block);
      await assertControls(block);
      await assertSourceActions(page, block);
      expect(
        await block.locator('pre').evaluate((element) => element.scrollWidth > element.clientWidth)
      ).toBe(true);
      const html = preview
        .locator('[data-artifact-origin]')
        .filter({ has: page.getByText('html', { exact: true }) });
      await assertControls(html);
      await html.getByRole('button', { name: /^preview$/i }).click();
      await expect(html.locator('iframe')).toHaveAttribute('sandbox', 'allow-scripts');
      await expect(
        html.frameLocator('iframe').getByText('Inline safe preview', { exact: true })
      ).toBeVisible();
      await expect(html.frameLocator('iframe').locator('[onclick]')).toHaveCount(0);
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.goto(url('/settings/ui'));
      const appearance = page.getByTestId('theme-display-disclosure');
      await appearance.locator('summary').click();
      await appearance.getByText('Light', { exact: true }).click();
      await expect(page.locator('html')).toHaveClass(/light/);
      await page.goto(url('/notes'));
      await page.getByTestId(`notes-open-button-${noteId}`).click();
      await page.getByRole('button', { name: 'Preview', exact: true }).click();
      await assertProse(page.getByTestId('notes-preview-surface'));
      await testInfo.attach(`${surface}-light-preview`, {
        body: await page.screenshot(),
        contentType: 'image/png',
      });
    } finally {
      await extension?.close();
    }
  });
}

test('NE-07: sidepanel artifacts keep aligned code rows and Mermaid actions', async ({
  serverInfo,
  request,
}) => {
  skipIfServerUnavailable(serverInfo);
  const api = createSeedApi(request);
  await warmBackendOnce(api);
  const title = generateTestId('uxr-code-artifact');
  const characterId = await createCharacter(api, title);
  const chatId = await createChatWithMessages(api, {
    characterId,
    title,
    messages: [
      { role: 'user', content: `Show saved code and diagram\n\n\`\`\`python\n${code}\n\`\`\`` },
      {
        role: 'assistant',
        content: `\`\`\`mermaid\ngraph TD\n  A-->B\n\`\`\``,
      },
    ],
  });
  const extension = await launchExtension({
    serverUrl: TEST_CONFIG.serverUrl,
    apiKey: TEST_CONFIG.apiKey,
  });
  try {
    await test.step('Enable user Markdown and verify its persisted native setting', async () => {
      const settings = await extension.context.newPage();
      await settings.setViewportSize({ width: 1280, height: 900 });
      await settings.goto(
        `chrome-extension://${extension.extensionId}/options.html#/settings/chat`
      );
      const markdownSetting = settings.getByRole('switch', {
        name: 'Enable Markdown formatting for User messages',
        exact: true,
      });
      if (!(await markdownSetting.isChecked())) await markdownSetting.click();
      await expect(markdownSetting).toBeChecked();
      await expect(
        settings.getByRole('switch', { name: 'Use Chat Bubble for User Messages', exact: true })
      ).toBeChecked();
      await expect
        .poll(() =>
          settings.evaluate(async () => {
            const stored = await chrome.storage.sync.get('useMarkdownForUserMessage');
            return stored.useMarkdownForUserMessage;
          })
        )
        .toBe(JSON.stringify(true));
      await expect(markdownSetting).toBeChecked();
      await settings.close();
    });
    const page = await test.step('Open the actual extension sidepanel', () =>
      extension.openSidePanel());
    await page.setViewportSize({ width: 1280, height: 900 });
    const panel = new SidePanelChat(page);
    await test.step('Select the saved conversation and its normal user-bubble presentation', async () => {
      await panel.openServerChatFromSearch(title, chatId);
      await panel.openSidebar();
      const proMode = panel.sidebar
        .getByRole('group', { name: 'Mode', exact: true })
        .getByRole('button', { name: 'Pro', exact: true });
      await proMode.click();
      await expect(proMode).toHaveAttribute('aria-pressed', 'true');
      await expect(page.getByTestId('artifacts-panel')).toBeHidden();
      await panel.closeSidebar();
      await page.getByTestId('conversation-context-trigger').click();
      await page.getByTestId('chat-character-select').click();
      await page.getByRole('menuitem', { name: 'Hide large portraits', exact: true }).click();
      await expect(page.getByTestId('artifacts-panel')).toBeVisible();
      await page.getByTestId('artifacts-panel-close').click();
      await expect(page.getByTestId('artifacts-panel')).toBeHidden();
      const contextTrigger = page.getByTestId('conversation-context-trigger');
      if ((await contextTrigger.getAttribute('aria-expanded')) === 'true') {
        await contextTrigger.click();
      }
      await expect(page.getByTestId('conversation-context-trigger')).toHaveAttribute(
        'aria-expanded',
        'false'
      );
    });
    await test.step('Open code from the saved user message', async () => {
      await panel.transcript.getByRole('button', { name: 'View code', exact: true }).click();
    });
    const artifact = page.getByTestId('artifacts-panel');
    await expect(artifact).toBeVisible();
    await assertCodeColumns(artifact);
    await page.getByTestId('artifacts-jump-source').click();
    await expect(artifact).toBeHidden();
    await test.step('Open the saved assistant Mermaid artifact', async () => {
      await panel.transcript
        .getByRole('button', { name: 'View Mermaid diagram', exact: true })
        .click();
    });
    const artifactDiagram = artifact
      .getByRole('img', { name: 'Mermaid diagram', exact: true })
      .locator('svg');
    const inlineDiagram = panel.transcript
      .getByRole('img', { name: 'Mermaid diagram', exact: true })
      .locator('svg');
    await expect(artifactDiagram).toBeVisible();
    await expect(inlineDiagram).toBeVisible();
    const inlineDiagramId = await inlineDiagram.getAttribute('id');
    const artifactDiagramId = await artifactDiagram.getAttribute('id');
    expect(inlineDiagramId).toBeTruthy();
    expect(artifactDiagramId).toBeTruthy();
    expect(artifactDiagramId).not.toBe(inlineDiagramId);
    await expect(artifact.getByRole('button', { name: 'Open Mermaid preview' })).toBeVisible();
    await page.getByTestId('artifacts-panel-close').click();
    await expect(artifact).toBeHidden();
  } finally {
    await test.step('Close the isolated extension context', () => extension.close());
  }
});

test('NE-05: Writing Playground shared preview keeps prose typography', async ({
  authedPage,
  serverInfo,
  request,
}) => {
  skipIfServerUnavailable(serverInfo);
  await warmBackendOnce(createSeedApi(request));
  const writing = new WritingPlaygroundPage(authedPage);
  await writing.goto();
  await writing.assertPageReady();
  if (!(await writing.newSessionButton.isVisible())) await writing.toggleLibraryButton.click();
  await writing.newSessionButton.click();
  const dialog = authedPage.getByRole('dialog', { name: 'New session', exact: true });
  await dialog
    .getByPlaceholder('e.g. Draft ideas', { exact: true })
    .fill(generateTestId('uxr-prose'));
  await dialog.getByRole('button', { name: 'OK', exact: true }).click();
  await expect(dialog).toBeHidden();
  await authedPage.getByText('Plain', { exact: true }).click();
  await authedPage.getByPlaceholder('Start writing your prompt...', { exact: true }).fill(prose);
  await authedPage.getByText('Preview', { exact: true }).click();
  const preview = authedPage
    .locator('.prose')
    .filter({ has: authedPage.getByRole('heading', { name: 'Preview contract', exact: true }) });
  await expect(preview).toBeVisible();
  await expect(preview.locator('h1')).toHaveCSS('font-weight', '800');
  await expect(preview.locator('blockquote')).toHaveCSS('border-left-width', '4px');
  await expect(preview.locator('a')).toHaveAttribute('href', 'https://example.com');
  await assertProse(preview.locator('..'));
});
