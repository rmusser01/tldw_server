import { expect, test, type Locator, type Page } from '@playwright/test';

const bypassChatGates = async (page: Page) => {
  await page.route('**/api/v1/llm/models/metadata**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ models: [] }),
    });
  });
  await page.route('**/api/v1/llm/providers**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ providers: [] }),
    });
  });

  await page.addInitScript(() => {
    const authConfig = {
      serverUrl: 'http://127.0.0.1:8000',
      authMode: 'single-user',
      // lgtm[js/clear-text-storage-of-sensitive-data] synthetic CI key only
      apiKey: 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY',
    };

    window.localStorage.setItem('assistant_setup_dismissed', 'true');
    window.localStorage.setItem('__tldw_first_run_complete', 'true');
    window.localStorage.setItem('__tldw_test_bypass', 'true');
    window.localStorage.setItem('tldwConfig', JSON.stringify(authConfig));
    // lgtm[js/clear-text-storage-of-sensitive-data] test-only legacy auth compatibility key
    window.localStorage.setItem('apiKey', authConfig.apiKey);
    window.localStorage.setItem('authMode', authConfig.authMode);
    window.localStorage.setItem('stickyChatInput', 'true');
    window.localStorage.setItem('playgroundComposerOptionsExpanded', 'false');
  });
};

const waitForStickyChat = async (page: Page) => {
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('playground-chat-composer-dock')).toBeVisible({
    timeout: 30_000,
  });
};

const expectDockWithinViewport = async (page: Page) => {
  const dockBox = await page.getByTestId('playground-chat-composer-dock').boundingBox();
  expect(dockBox).not.toBeNull();
  expect(dockBox!.y).toBeGreaterThanOrEqual(0);
  expect(dockBox!.y + dockBox!.height).toBeLessThanOrEqual(page.viewportSize()!.height);
};

const scrollTranscriptToBottom = async (page: Page, transcript: Locator) => {
  const dock = page.getByTestId('playground-chat-composer-dock');

  await expect(transcript).toBeVisible();
  const forceScrollTranscript = () =>
    transcript.evaluate((el) => {
      let filler = el.querySelector<HTMLElement>('[data-testid="sticky-chat-scroll-filler"]');
      if (!filler) {
        filler = document.createElement('div');
        filler.setAttribute('data-testid', 'sticky-chat-scroll-filler');
        filler.style.flex = '0 0 auto';
        el.appendChild(filler);
      }

      filler.style.height = `${Math.max(1200, el.clientHeight * 2)}px`;
      const maxScrollTop = el.scrollHeight - el.clientHeight;
      if (maxScrollTop <= 0) {
        return 0;
      }

      el.scrollTop = maxScrollTop;
      return el.scrollTop;
    });

  await expect.poll(forceScrollTranscript).toBeGreaterThan(0);
  await forceScrollTranscript();
  await expect(dock).toBeVisible();
  await expectDockWithinViewport(page);
};

test.describe('chat sticky composer dock', () => {
  test('desktop sticky /chat keeps the composer visible while the transcript scrolls', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await bypassChatGates(page);
    await page.setViewportSize({ width: 1440, height: 960 });
    await waitForStickyChat(page);

    const transcript = page.getByTestId('playground-chat-transcript');
    await expect(transcript).toBeVisible();
    await scrollTranscriptToBottom(page, transcript);
  });

  test('mobile-sized sticky /chat keeps the composer visible after focusing the input', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await bypassChatGates(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await waitForStickyChat(page);

    const chatInput = page.getByTestId('chat-input');
    await expect(chatInput).toBeVisible();
    await chatInput.focus();

    await expectDockWithinViewport(page);
  });

  test('desktop sticky /chat keeps the dock scoped to the chat column when artifacts are open', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await bypassChatGates(page);
    await page.setViewportSize({ width: 1440, height: 960 });
    await waitForStickyChat(page);

    await page.evaluate(() => {
      const artifactsStore = (
        window as Window & {
          __tldw_useArtifactsStore?: {
            getState: () => {
              openArtifact: (
                artifact: {
                  id: string;
                  title: string;
                  content: string;
                  kind: 'code';
                  language?: string;
                },
                options?: { auto?: boolean }
              ) => void;
            };
          };
        }
      ).__tldw_useArtifactsStore;

      artifactsStore?.getState().openArtifact(
        {
          id: 'dock-smoke-artifact',
          title: 'Smoke artifact',
          content: "console.log('dock')",
          kind: 'code',
          language: 'ts',
        },
        { auto: false }
      );
    });

    const artifactsPanel = page.locator('[data-testid="artifacts-panel"]').first();
    await expect(artifactsPanel).toBeVisible({ timeout: 30_000 });
    await expectDockWithinViewport(page);

    const dockBox = await page.getByTestId('playground-chat-composer-dock').boundingBox();
    const panelBox = await artifactsPanel.boundingBox();
    expect(dockBox).not.toBeNull();
    expect(panelBox).not.toBeNull();
    expect(dockBox!.x + dockBox!.width).toBeLessThanOrEqual(panelBox!.x + 1);
  });
});
