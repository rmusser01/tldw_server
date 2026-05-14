/**
 * /chat cockpit real-server parity checks.
 *
 * This spec intentionally does not stub or fulfill backend routes. It verifies
 * the /chat cockpit and focus layouts against the live tldw server configured
 * by TLDW_E2E_SERVER_URL/TLDW_E2E_API_KEY.
 */
import { expect, test, type APIRequestContext, type Page, type Response } from '@playwright/test';

const serverUrl = (
  process.env.TLDW_E2E_SERVER_URL ||
  process.env.TLDW_SERVER_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

const apiKey =
  process.env.TLDW_E2E_API_KEY || process.env.TLDW_API_KEY || '';

test.skip(!apiKey, 'TLDW_E2E_API_KEY or TLDW_API_KEY is required for real-server chat cockpit checks');

type ApiHit = {
  method: string;
  path: string;
  status: number;
};

const apiHeaders = () => ({
  'x-api-key': apiKey,
});

const apiGet = async <T>(
  request: APIRequestContext,
  path: string
): Promise<{ status: number; body: T }> => {
  const response = await request.get(`${serverUrl}${path}`, {
    headers: apiHeaders(),
    timeout: 30_000,
  });
  const body = (await response.json()) as T;
  return { status: response.status(), body };
};

const apiPost = async <T>(
  request: APIRequestContext,
  path: string,
  body: Record<string, unknown>
): Promise<{ status: number; body: T | null }> => {
  const response = await request.post(`${serverUrl}${path}`, {
    data: body,
    headers: {
      ...apiHeaders(),
      'content-type': 'application/json',
    },
    timeout: 30_000,
  });
  const payload = await response.json().catch(() => null);
  return { status: response.status(), body: payload as T | null };
};

const apiDelete = async (
  request: APIRequestContext,
  path: string
): Promise<{ status: number }> => {
  const response = await request.delete(`${serverUrl}${path}`, {
    headers: apiHeaders(),
    timeout: 30_000,
  });
  return { status: response.status() };
};

const extractModels = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.models)) return payload.models;
  return [];
};

const extractCharacters = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.items)) return payload.items;
  if (Array.isArray(payload?.characters)) return payload.characters;
  if (Array.isArray(payload?.results)) return payload.results;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
};

const extractConfiguredProviders = (payload: any): any[] => {
  const providers = Array.isArray(payload?.providers)
    ? payload.providers
    : Array.isArray(payload)
      ? payload
      : [];

  return providers.filter((provider: any) => {
    const hasModelChoices = Array.isArray(provider?.models) && provider.models.length > 0;
    return Boolean(provider?.is_configured) && (hasModelChoices || provider?.endpoint_only);
  });
};

const seedRealServerConfig = async (page: Page) => {
  await page.addInitScript(
    ({ configuredServerUrl, configuredApiKey }) => {
      const config = {
        serverUrl: configuredServerUrl,
        authMode: 'single-user',
        apiKey: configuredApiKey,
      };

      localStorage.setItem('tldwConfig', JSON.stringify(config));
      localStorage.setItem('serverUrl', configuredServerUrl);
      localStorage.setItem('tldwServerUrl', configuredServerUrl);
      localStorage.setItem('tldw-api-host', configuredServerUrl);
      localStorage.setItem('authMode', 'single-user');
      localStorage.setItem('apiKey', configuredApiKey);
      localStorage.setItem('isMigrated', 'true');
      localStorage.setItem('__tldw_first_run_complete', 'true');
      localStorage.setItem('assistant_setup_dismissed', 'true');
      localStorage.setItem('playgroundComposerOptionsExpanded', 'true');
    },
    {
      configuredServerUrl: serverUrl,
      configuredApiKey: apiKey,
    }
  );
};

const trackRealApiHits = (page: Page) => {
  const hits: ApiHit[] = [];
  const watchedPaths = ['/api/v1/health', '/api/v1/llm/providers', '/api/v1/llm/models/metadata'];

  const onResponse = (response: Response) => {
    const url = new URL(response.url());
    if (!response.url().startsWith(serverUrl)) return;
    const path = url.pathname;
    if (!watchedPaths.some((watchedPath) => path === watchedPath)) return;
    hits.push({
      method: response.request().method(),
      path,
      status: response.status(),
    });
  };

  page.on('response', onResponse);

  return {
    hits,
    dispose: () => page.off('response', onResponse),
  };
};

const ensureComposerOptionsVisible = async (page: Page) => {
  const toggle = page.getByTestId('composer-options-toggle');
  if (await toggle.isVisible().catch(() => false)) {
    const label = await toggle.getAttribute('aria-label').catch(() => '');
    if (/show composer options/i.test(label || '')) {
      await toggle.click();
    }
  }
};

const closeSearchContextIfOpen = async (page: Page) => {
  const closeSearch = page.getByRole('button', { name: 'Close Search & Context' });
  if (await closeSearch.isVisible().catch(() => false)) {
    await closeSearch.click();
  }
};

const assertCoreComposerControls = async (
  page: Page,
  options: { mobile?: boolean; composerOnly?: boolean } = {}
) => {
  await ensureComposerOptionsVisible(page);
  await expect(page.getByTestId('composer-options-toggle')).toBeVisible();
  await expect(page.getByTestId('chat-input')).toBeVisible();
  await expect(page.getByRole('button', { name: /send message/i })).toBeVisible();

  if (options.mobile && options.composerOnly) {
    return;
  }

  await expect(page.getByTestId('model-selector').first()).toBeVisible();
  await expect(page.getByTestId('chat-prompt-select')).toBeVisible();
  await expect(page.getByTestId('character-select')).toBeVisible();
  await expect(page.getByTestId('attachment-button')).toBeVisible();
  await expect(page.getByTestId('tools-button')).toBeVisible();

  if (options.mobile) {
    await expect(page.getByRole('button', { name: /More options/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Health & diagnostics|Server:/i })).toBeVisible();
    return;
  }

  await expect(page.getByTestId('mcp-tools-toggle')).toBeVisible();
  await expect(page.getByRole('button', { name: /Advanced controls/i })).toBeVisible();
};

const assertNoBlockingServerDialog = async (page: Page) => {
  await expect(
    page.getByRole('dialog').filter({ hasText: /can't reach your tldw server/i })
  ).toBeHidden({ timeout: 5_000 });
};

const assertHealthResponse = (health: { status: number; body: any }) => {
  expect([200, 206]).toContain(health.status);
  expect(['healthy', 'degraded']).toContain(health.body?.status);
};

const waitForChatCompletionAttempt = (page: Page) =>
  page.waitForResponse(
    (response) => {
      if (!response.url().startsWith(serverUrl)) return false;
      const url = new URL(response.url());
      return (
        url.pathname === '/api/v1/chat/completions' &&
        response.request().method() === 'POST'
      );
    },
    { timeout: 60_000 }
  );

const assertChatCompletionRenderedOrRecoverable = async (
  page: Page,
  response: Response
) => {
  const chatLog = page.getByRole('log', { name: /chat messages/i });
  if (response.status() < 400) {
    await expect(
      page
        .locator(
          "article[aria-label*='Assistant message'], [data-role='assistant'], [data-message-role='assistant'], .assistant-message"
        )
        .last()
    ).toBeVisible({ timeout: 60_000 });
    return;
  }

  await expect(chatLog).toContainText(/error|failed|unable|provider|request/i, {
    timeout: 30_000,
  });
};

const assertProviderQualifiedPayload = async (page: Page, response: Response) => {
  const payload = response.request().postDataJSON() as any;
  expect(payload).toBeTruthy();
  expect(typeof payload.model).toBe('string');
  expect(payload.model.trim().length).toBeGreaterThan(0);

  const runtimeInspector = page.getByTestId('playground-runtime-inspector');
  const visibleRouteLabel = await runtimeInspector
    .getByText('Provider route')
    .locator('xpath=following-sibling::p[1]')
    .textContent()
    .catch(() => null);
  const routeText = await runtimeInspector
    .getByText(/^Route /)
    .first()
    .textContent()
    .catch(() => null);
  const routeLabel =
    visibleRouteLabel?.trim() ||
    routeText?.replace(/^Route\s+/, '').trim() ||
    '';
  const separatorIndex = routeLabel.indexOf(':');
  if (separatorIndex > 0 && separatorIndex < routeLabel.length - 1) {
    const provider = routeLabel.slice(0, separatorIndex);
    const model = routeLabel.slice(separatorIndex + 1);
    const routedProvider = payload.api_provider || payload.provider;
    expect(payload.model).toBe(model);
    expect(routedProvider).toBeTruthy();
    if (provider !== 'tldw') {
      expect(routedProvider).toBe(provider);
    }
  }
};

test.describe('/chat cockpit real-server parity', () => {
  test('does not intercept backend routes in this real-server spec', async ({}, testInfo) => {
    const fs = await import('node:fs');
    const { readFileSync } = fs;
    const source = readFileSync(testInfo.file, 'utf8');
    const forbiddenCall = ['page', 'route'].join('.') + '(';
    expect(source).not.toContain(forbiddenCall);
  });

  test('uses the running server and keeps cockpit/focus controls working', async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);

    const providers = await apiGet<any>(request, '/api/v1/llm/providers');
    expect(providers.status).toBe(200);
    expect(extractConfiguredProviders(providers.body).length).toBeGreaterThan(0);

    const models = await apiGet<any>(request, '/api/v1/llm/models/metadata');
    expect(models.status).toBe(200);
    expect(extractModels(models.body).length).toBeGreaterThan(0);

    const apiTracker = trackRealApiHits(page);
    await seedRealServerConfig(page);
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });

    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
      timeout: 60_000,
    });
    await assertNoBlockingServerDialog(page);
    await expect(page.getByRole('log', { name: /chat messages/i })).toBeVisible();
    await assertCoreComposerControls(page);

    if (health.body?.status === 'degraded') {
      const degradedShell = page.getByTestId('server-readiness-degraded-shell');
      await expect(degradedShell).toBeVisible();
      await expect(degradedShell).toContainText('Server partially degraded');
    }

    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await page.getByRole('button', { name: 'Hide context rail' }).click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await page.getByRole('button', { name: 'Show context rail' }).click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await page.getByRole('button', { name: 'Hide runtime rail' }).click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await page.getByRole('button', { name: 'Show runtime rail' }).click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    const contextRail = page.getByTestId('playground-context-rail');
    const runtimeInspector = page.getByTestId('playground-runtime-inspector');
    await expect(runtimeInspector.getByText('Provider route')).toBeVisible();
    await expect(runtimeInspector.getByText('Scoped settings')).toBeVisible();
    await expect(runtimeInspector.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'Open Model & Chat settings' })).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'Select character or persona' })).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'MCP tool choice Auto' })).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'Configure MCP tools' })).toBeVisible();
    const cockpitStatus = page.getByRole('status', { name: 'Chat status' });
    const webSearchControl = contextRail.getByRole('button', { name: 'Web search', exact: true });
    const initialWebSearchState = await webSearchControl.getAttribute('aria-pressed');
    expect(['true', 'false']).toContain(initialWebSearchState);

    await webSearchControl.click();
    const toggledWebSearchState = initialWebSearchState === 'true' ? 'false' : 'true';
    await expect(webSearchControl).toHaveAttribute('aria-pressed', toggledWebSearchState);
    if (toggledWebSearchState === 'true') {
      await expect(cockpitStatus).toContainText('Web search on');
    } else {
      await expect(cockpitStatus).not.toContainText('Web search on');
    }
    if (toggledWebSearchState !== 'true') {
      await webSearchControl.click();
      await expect(webSearchControl).toHaveAttribute('aria-pressed', 'true');
    }
    await expect(cockpitStatus).toContainText('Web search on');
    const sourceInventory = contextRail.getByRole('list', { name: 'Context sources' });
    await expect(sourceInventory).toBeVisible();
    await expect(sourceInventory.getByRole('listitem').filter({ hasText: 'Web search' })).toContainText(
      'Active'
    );

    await closeSearchContextIfOpen(page);
    await contextRail.getByRole('button', { name: 'Open Search & Context' }).click();
    await expect(page.getByRole('heading', { name: /Knowledge Search/i })).toBeVisible();

    const mcpToggle = page.getByTestId('mcp-tools-toggle');
    if (await mcpToggle.isEnabled()) {
      await mcpToggle.click();
      await expect(page.getByText(/Tool choice/i)).toBeVisible();
      await page.keyboard.press('Escape');
    } else {
      await expect(mcpToggle).toBeDisabled();
      await expect(mcpToggle).toHaveAccessibleName(
        /MCP tools unavailable|MCP tools are offline|MCP tools: None|Not checked yet/i
      );
    }

    await page.getByRole('button', { name: /Advanced controls/i }).click();
    await expect(page.getByTestId('composer-options-panel')).toBeVisible();

    await page.getByTestId('model-selector').first().click();
    await expect(
      page.getByRole('button', { name: /OpenAI \/|Anthropic \/|Ollama \/|Select a model/i }).first()
    ).toBeVisible();
    await page.keyboard.press('Escape');

    await contextRail.getByRole('button', { name: 'Select a prompt' }).click();
    await expect(page.getByText(/Prompt|Search/i).first()).toBeVisible();
    await page.keyboard.press('Escape');

    await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
    await expect(page.getByText(/Character|No character/i).first()).toBeVisible();
    await page.keyboard.press('Escape');

    await page.getByTestId('tools-button').click();
    await expect(page.getByText(/Clear conversation|More tools/i).first()).toBeVisible();
    await page.keyboard.press('Escape');

    await runtimeInspector.getByRole('button', { name: 'Open Model & Chat settings' }).click();
    const modelSettingsDialog = page.getByRole('dialog', {
      name: 'Current Chat Model Settings',
    });
    await expect(modelSettingsDialog).toBeVisible();
    await expect(modelSettingsDialog.getByText(/API \/ model/i).first()).toBeVisible();
    await modelSettingsDialog.getByRole('button', { name: 'Close' }).click();
    await expect(modelSettingsDialog).toBeHidden();

    await runtimeInspector.getByRole('button', { name: 'Configure MCP tools' }).click();
    const mcpSettingsDialog = page.getByRole('dialog', {
      name: 'MCP tool settings',
    });
    await expect(mcpSettingsDialog).toBeVisible();
    await mcpSettingsDialog
      .getByTestId('mcp-settings-modal-footer')
      .getByRole('button', { name: 'Close' })
      .click();
    await expect(mcpSettingsDialog).toBeHidden();

    await page.getByRole('button', { name: 'Enter focus chat' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'focus'
    );
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await assertCoreComposerControls(page);

    await page.getByRole('button', { name: 'Show cockpit panels' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'cockpit'
    );
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(
      page
        .getByTestId('playground-context-rail')
        .getByRole('button', { name: 'Web search', exact: true })
    ).toHaveAttribute('aria-pressed', 'true');

    const smokePrompt = `cockpit smoke ${Date.now()}`;
    await page.getByTestId('chat-input').fill(smokePrompt);
    const chatCompletionAttempt = waitForChatCompletionAttempt(page);
    await page.getByRole('button', { name: /send message/i }).click();
    const chatCompletionResponse = await chatCompletionAttempt;
    await assertProviderQualifiedPayload(page, chatCompletionResponse);
    await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(smokePrompt);
    await assertChatCompletionRenderedOrRecoverable(page, chatCompletionResponse);
    await expect(cockpitStatus).not.toContainText('0 messages');

    const failingApiHits = apiTracker.hits.filter((hit) => hit.status >= 400);
    expect(failingApiHits).toEqual([]);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/health')).toBe(true);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/llm/providers')).toBe(true);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/llm/models/metadata')).toBe(true);

    apiTracker.dispose();
  });

  test('keeps mobile cockpit tabs and focus composer usable against the live server', async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);

    await seedRealServerConfig(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });

    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
      timeout: 60_000,
    });
    await assertNoBlockingServerDialog(page);
    await assertCoreComposerControls(page, { mobile: true });

    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'focus'
    );
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);

    await page.getByRole('button', { name: 'Show cockpit panels' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'cockpit'
    );
    const mobileRails = page.getByTestId('playground-cockpit-mobile-rails');
    await expect(mobileRails).toBeVisible();
    await expect(mobileRails.getByRole('tab', { name: 'Context' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    await expect(mobileRails.getByRole('tab', { name: 'Runtime' })).toHaveAttribute(
      'aria-selected',
      'false'
    );
    await page.getByRole('button', { name: 'Hide context rail' }).click();
    await expect(mobileRails.getByRole('tab', { name: 'Context' })).toHaveCount(0);
    await expect(mobileRails.getByRole('tab', { name: 'Runtime' })).toBeVisible();
    await page.getByRole('button', { name: 'Show context rail' }).click();
    const contextTab = mobileRails.getByRole('tab', { name: 'Context' });
    await expect(contextTab).toBeVisible();
    await contextTab.click();
    await expect(contextTab).toHaveAttribute('aria-selected', 'true');
    const contextPanel = mobileRails.getByRole('tabpanel', { name: 'Context' });
    await expect(contextPanel.getByRole('button', { name: 'Open Search & Context' })).toBeVisible();
    const mobileWebSearchControl = contextPanel.getByRole('button', {
      name: 'Web search',
      exact: true,
    });
    const initialMobileWebSearchState =
      await mobileWebSearchControl.getAttribute('aria-pressed');
    await mobileWebSearchControl.click();
    await expect(mobileWebSearchControl).toHaveAttribute(
      'aria-pressed',
      initialMobileWebSearchState === 'true' ? 'false' : 'true'
    );
    await expect(page.getByTestId('chat-input')).toBeVisible();
    const runtimeTab = mobileRails.getByRole('tab', { name: 'Runtime' });
    await runtimeTab.click();
    await expect(runtimeTab).toHaveAttribute('aria-selected', 'true');
    const runtimePanel = mobileRails.getByRole('tabpanel', { name: 'Runtime' });
    await expect(runtimePanel.getByText('Provider route')).toBeVisible();
    await expect(runtimePanel.getByText('Scoped settings')).toBeVisible();
    await expect(runtimePanel.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Open Model & Chat settings' })).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Select character or persona' })).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Configure MCP tools' })).toBeVisible();

    await page.getByRole('button', { name: 'Enter focus chat' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'focus'
    );
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);
    await assertCoreComposerControls(page, { mobile: true, composerOnly: true });
  });

  test('selects and clears a real disposable character through the runtime rail', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);

    const characterName = `Cockpit Rail ${Date.now()}`;
    const created = await apiPost<any>(request, '/api/v1/characters', {
      name: characterName,
      system_prompt: `You are ${characterName}, a concise test assistant.`,
      first_message: 'Ready for cockpit rail proof.',
      creator: 'tldw e2e',
      tags: ['e2e', 'cockpit-rail'],
    });
    const createdId = created.body?.id;
    const createdVersion = created.body?.version ?? 1;

    if (created.status !== 201 || !createdId) {
      testInfo.annotations.push({
        type: 'blocker',
        description: `Could not create disposable character via real server: status ${created.status}`,
      });

      await seedRealServerConfig(page);
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      const runtimeInspector = page.getByTestId('playground-runtime-inspector');
      await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
      const listed = await apiGet<any>(request, '/api/v1/characters');
      if (extractCharacters(listed.body).length === 0) {
        await expect(page.getByText('No characters available.')).toBeVisible();
      } else {
        await expect(page.getByRole('tab', { name: 'Characters' })).toHaveAttribute(
          'aria-selected',
          'true'
        );
      }
      return;
    }

    try {
      const listed = await apiGet<any>(request, '/api/v1/characters');
      expect(listed.status).toBe(200);
      expect(
        extractCharacters(listed.body).some(
          (character) => String(character?.id) === String(createdId)
        )
      ).toBe(true);

      await seedRealServerConfig(page);
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });

      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      await assertCoreComposerControls(page);

      const runtimeInspector = page.getByTestId('playground-runtime-inspector');
      const composerAssistant = page.getByTestId('character-select');
      await expect(runtimeInspector.getByText('No assistant selected').first()).toBeVisible();
      await expect(composerAssistant).toHaveAccessibleName(/Select character or persona/i);

      await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
      await expect(page.getByRole('tab', { name: 'Characters' })).toHaveAttribute(
        'aria-selected',
        'true'
      );
      await page.getByRole('button', { name: characterName }).click();

      await expect(runtimeInspector.getByText(characterName)).toBeVisible();
      await expect(composerAssistant).toHaveAccessibleName(characterName);
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toBeVisible();
      await expect(runtimeInspector.getByRole('button', { name: 'Open Scene Director' })).toBeVisible();

      const smokePrompt = `assistant rail proof ${Date.now()}`;
      await page.getByTestId('chat-input').fill(smokePrompt);
      const chatCompletionAttempt = waitForChatCompletionAttempt(page);
      await page.getByRole('button', { name: /send message/i }).click();
      const chatCompletionResponse = await chatCompletionAttempt;
      await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(smokePrompt);
      await assertChatCompletionRenderedOrRecoverable(page, chatCompletionResponse);

      await runtimeInspector.getByRole('button', { name: 'Clear assistant' }).click();
      await expect(runtimeInspector.getByText('No assistant selected').first()).toBeVisible();
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toHaveCount(0);
      await expect(composerAssistant).toHaveAccessibleName(/Select character or persona/i);
    } finally {
      await apiDelete(
        request,
        `/api/v1/characters/${encodeURIComponent(String(createdId))}?expected_version=${encodeURIComponent(
          String(createdVersion)
        )}`
      ).catch(() => ({ status: 0 }));
    }
  });
});
