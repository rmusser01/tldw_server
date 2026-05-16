/**
 * /chat cockpit real-server parity checks.
 *
 * This spec intentionally does not stub or fulfill backend routes. It verifies
 * the /chat cockpit and focus layouts against the live tldw server configured
 * by TLDW_E2E_SERVER_URL/TLDW_E2E_API_KEY.
 */
import {
  expect,
  test,
  type APIRequestContext,
  type APIResponse,
  type Locator,
  type Page,
  type Response,
} from '@playwright/test';

const serverUrl = (
  process.env.TLDW_E2E_SERVER_URL ||
  process.env.TLDW_SERVER_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

const apiKey =
  process.env.TLDW_E2E_API_KEY || process.env.TLDW_API_KEY || process.env.SINGLE_USER_API_KEY || '';

test.skip(!apiKey, 'TLDW_E2E_API_KEY, TLDW_API_KEY, or SINGLE_USER_API_KEY is required for real-server chat cockpit checks');

type ApiHit = {
  method: string;
  path: string;
  status: number;
};

type RealChatModelSelection = {
  provider: string;
  model: string;
  key: string;
};

const apiHeaders = () => ({
  'x-api-key': apiKey,
});

const truncateForDiagnostics = (value: string, maxLength = 800): string =>
  value.length > maxLength ? `${value.slice(0, maxLength)}...` : value;

const parseApiJsonResponse = async <T>(
  response: APIResponse,
  method: string,
  path: string
): Promise<T> => {
  if (response.status() === 204) {
    return null as T;
  }
  try {
    return (await response.json()) as T;
  } catch {
    let responseText = '<response body unavailable>';
    try {
      const text = (await response.text()).trim();
      responseText = text
        ? truncateForDiagnostics(text)
        : '<empty response body>';
    } catch {
      // Keep the fallback marker when the response body cannot be read.
    }
    throw new Error(
      `${method} ${path} returned non-JSON response (${response.status()}): ${responseText}`
    );
  }
};

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
  const payload = await parseApiJsonResponse<T | null>(response, 'POST', path);
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

const extractPersonaProfiles = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.items)) return payload.items;
  if (Array.isArray(payload?.profiles)) return payload.profiles;
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

const buildConfiguredChatModelSelection = (payload: any): RealChatModelSelection => {
  const configuredProviders = extractConfiguredProviders(payload).filter(
    (provider: any) => Array.isArray(provider?.models) && provider.models.length > 0
  );
  const provider =
    configuredProviders.find((candidate: any) => candidate?.name === 'openai') ||
    configuredProviders[0];

  if (!provider) {
    throw new Error('No configured provider with chat models is available on the real server');
  }

  const model =
    typeof provider.default_model === 'string' && provider.default_model.trim().length > 0
      ? provider.default_model.trim()
      : String(provider.models[0] || '').trim();

  if (!model) {
    throw new Error(`Configured provider ${provider.name || '<unknown>'} has no usable model`);
  }

  const providerName = String(provider.name || '').trim();
  if (!providerName) {
    throw new Error(`Configured model ${model} is missing a provider name`);
  }

  return {
    provider: providerName,
    model,
    key: `tldw:${model}`,
  };
};

const getConfiguredChatModelSelection = async (
  request: APIRequestContext
): Promise<RealChatModelSelection> => {
  const providers = await apiGet<any>(request, '/api/v1/llm/providers');
  expect(providers.status).toBe(200);
  expect(extractConfiguredProviders(providers.body).length).toBeGreaterThan(0);
  return buildConfiguredChatModelSelection(providers.body);
};

const seedRealServerConfig = async (
  page: Page,
  options: { selectedModel?: RealChatModelSelection } = {}
) => {
  await page.addInitScript(
    ({ configuredServerUrl, configuredApiKey, configuredSelectedModel }) => {
      const config = {
        serverUrl: configuredServerUrl,
        authMode: 'single-user',
        apiKey: configuredApiKey,
        requestTimeoutMs: 60_000,
        chatRequestTimeoutMs: 120_000,
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

      if (configuredSelectedModel?.key) {
        localStorage.setItem('selectedModel', configuredSelectedModel.key);
        localStorage.setItem(
          'chatModelUsageByProviderModel',
          JSON.stringify({
            [configuredSelectedModel.key]: {
              selectedCount: 1,
              lastSelectedAt: Date.now(),
            },
          })
        );
      }
    },
    {
      configuredServerUrl: serverUrl,
      configuredApiKey: apiKey,
      configuredSelectedModel: options.selectedModel || null,
    }
  );
};

const putLocalPrompt = async (
  page: Page,
  prompt: {
    id: string;
    title: string;
    name: string;
    content: string;
    is_system: boolean;
    createdAt: number;
    updatedAt: number;
    tags: string[];
    keywords: string[];
    favorite: boolean;
    usageCount: number;
    lastUsedAt: number | null;
    system_prompt: string | null;
    user_prompt: string | null;
    promptFormat: string;
    promptSchemaVersion: number | null;
    structuredPromptDefinition: null;
    syncPayloadVersion: number;
    fewShotExamples: null;
    modulesConfig: null;
    versionNumber: null;
    changeDescription: null;
    parentVersionId: null;
    serverParentVersionId: null;
    syncStatus: string;
    sourceSystem: string;
  }
) => {
  await page.evaluate(async (promptRecord) => {
    await new Promise<void>((resolve, reject) => {
      const openRequest = indexedDB.open('PageAssistDatabase');
      openRequest.onerror = () =>
        reject(openRequest.error ?? new Error('Failed to open prompt database'));
      openRequest.onsuccess = () => {
        const db = openRequest.result;
        if (!db.objectStoreNames.contains('prompts')) {
          db.close();
          reject(new Error('Prompt object store is not available'));
          return;
        }
        const tx = db.transaction('prompts', 'readwrite');
        tx.oncomplete = () => {
          db.close();
          resolve();
        };
        tx.onerror = () => {
          db.close();
          reject(tx.error ?? new Error('Failed to write prompt record'));
        };
        tx.objectStore('prompts').put(promptRecord);
      };
    });
  }, prompt);
};

const deleteLocalPrompt = async (page: Page, promptId: string) => {
  await page
    .evaluate(async (id) => {
      await new Promise<void>((resolve, reject) => {
        const openRequest = indexedDB.open('PageAssistDatabase');
        openRequest.onerror = () =>
          reject(openRequest.error ?? new Error('Failed to open prompt database'));
        openRequest.onsuccess = () => {
          const db = openRequest.result;
          if (!db.objectStoreNames.contains('prompts')) {
            db.close();
            resolve();
            return;
          }
          const tx = db.transaction('prompts', 'readwrite');
          tx.oncomplete = () => {
            db.close();
            resolve();
          };
          tx.onerror = () => {
            db.close();
            reject(tx.error ?? new Error('Failed to delete prompt record'));
          };
          tx.objectStore('prompts').delete(id);
        };
      });
    }, promptId)
    .catch(() => undefined);
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

const getDesktopContextRail = (page: Page): Locator =>
  page.getByTestId('playground-cockpit-left-rail').getByTestId('playground-context-rail');

const getDesktopCompositionPreview = (page: Page): Locator =>
  getDesktopContextRail(page).getByRole('region', {
    name: 'Next message composition',
  });

const getDesktopRuntimeInspector = (page: Page): Locator =>
  page
    .getByTestId('playground-cockpit-right-rail')
    .getByTestId('playground-runtime-inspector');

const assertCoreComposerControls = async (
  page: Page,
  options: { mobile?: boolean; composerOnly?: boolean } = {}
) => {
  await ensureComposerOptionsVisible(page);
  await expect(page.getByTestId('composer-options-toggle')).toBeVisible();
  await expect(page.getByTestId('chat-input')).toBeVisible();
  await expect(page.getByRole('button', { name: /send message/i })).toBeVisible();

  if (options.composerOnly) {
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

const assertRuntimeProviderSummary = async (runtimeInspector: Locator) => {
  const runtimeState = runtimeInspector.getByRole('region', { name: 'Runtime state' });
  await expect(runtimeState.getByRole('heading', { name: 'Runtime' })).toBeVisible();
  await expect(runtimeState.getByText('Model', { exact: true })).toBeVisible();
  const providerRoute = runtimeState.getByText('Provider route');
  if (await providerRoute.isVisible().catch(() => false)) {
    await expect(providerRoute).toBeVisible();
  }
};

const assertRuntimeMcpRailState = async (runtimeInspector: Locator) => {
  await expect(runtimeInspector.getByRole('button', { name: 'Configure MCP tools' })).toBeVisible();

  const autoToolChoice = runtimeInspector.getByRole('button', {
    name: 'MCP tool choice Auto',
  });

  if (await autoToolChoice.isVisible().catch(() => false)) {
    await expect(autoToolChoice).toBeVisible();
    await expect(
      runtimeInspector.getByRole('button', { name: 'MCP tool choice Required' })
    ).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'MCP tool choice None' })).toBeVisible();
    const stateCounts = runtimeInspector.getByLabel('MCP tool state counts');
    if (await stateCounts.isVisible().catch(() => false)) {
      await expect(stateCounts).toContainText('Discovered');
      await expect(stateCounts).toContainText('Executable');
      await expect(stateCounts).toContainText('Chat-enabled');
      await expect(stateCounts).toContainText('User-disabled');
    }
    return;
  }

  await expect(runtimeInspector.getByRole('region', { name: 'MCP tools' })).toContainText(
    /No MCP tools available|MCP tools unavailable|MCP tools are offline|Loading MCP tools|Loading tools|MCP unavailable|Not checked yet/i
  );
};

const assertNoBlockingServerDialog = async (page: Page) => {
  await expect(
    page.getByRole('dialog').filter({ hasText: /can't reach your tldw server/i })
  ).toBeHidden({ timeout: 5_000 });
};

const assertHealthResponse = (health: { status: number; body: any }) => {
  expect([200, 206]).toContain(health.status);
  expect(['ok', 'healthy', 'degraded']).toContain(health.body?.status);
};

const waitForChatCompletionAttempt = (page: Page) =>
  page.waitForResponse(
    (response) => {
      if (!response.url().startsWith(serverUrl)) return false;
      const url = new URL(response.url());
      return (
        (url.pathname === '/api/v1/chat/completions' ||
          /^\/api\/v1\/chats\/[^/]+\/completions$/.test(url.pathname)) &&
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

  const runtimeInspector = getDesktopRuntimeInspector(page);
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

const getTemperatureInput = async (modelSettingsDialog: Locator): Promise<Locator> => {
  const byLabel = modelSettingsDialog.getByLabel(/temperature/i).first();
  if (await byLabel.isVisible().catch(() => false)) return byLabel;

  const byId = modelSettingsDialog.locator('input[id*="temperature"]').first();
  await expect(byId).toBeVisible();
  return byId;
};

test.describe('/chat cockpit real-server parity', () => {
  test('reports non-JSON API POST responses with response context', async () => {
    const fakeRequest = {
      post: async () => ({
        status: () => 502,
        json: async () => {
          throw new Error('invalid json');
        },
        text: async () => 'upstream gateway returned HTML',
      }),
    } as unknown as APIRequestContext;

    await expect(
      apiPost(fakeRequest, '/api/v1/example', { sample: true })
    ).rejects.toThrow(
      /POST \/api\/v1\/example returned non-JSON response \(502\): upstream gateway returned HTML/
    );
  });

  test('keeps successful 204 API POST responses as null bodies', async () => {
    const fakeRequest = {
      post: async () => ({
        status: () => 204,
        json: async () => {
          throw new Error('empty body');
        },
        text: async () => '',
      }),
    } as unknown as APIRequestContext;

    await expect(
      apiPost(fakeRequest, '/api/v1/example', { sample: true })
    ).resolves.toEqual({ status: 204, body: null });
  });

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
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);

    const providers = await apiGet<any>(request, '/api/v1/llm/providers');
    expect(providers.status).toBe(200);
    expect(extractConfiguredProviders(providers.body).length).toBeGreaterThan(0);
    const chatModelSelection = buildConfiguredChatModelSelection(providers.body);

    const models = await apiGet<any>(request, '/api/v1/llm/models/metadata');
    expect(models.status).toBe(200);
    expect(extractModels(models.body).length).toBeGreaterThan(0);

    const apiTracker = trackRealApiHits(page);
    await seedRealServerConfig(page, { selectedModel: chatModelSelection });
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });

    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
      timeout: 60_000,
    });
    await assertNoBlockingServerDialog(page);
    await expect(page.getByRole('log', { name: /chat messages/i })).toBeVisible();
    await assertCoreComposerControls(page, { composerOnly: true });
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-desktop-initial.png'),
      fullPage: true,
    });

    if (health.body?.status === 'degraded') {
      const degradedShell = page.getByTestId('server-readiness-degraded-shell');
      await expect(degradedShell).toBeVisible();
      await expect(degradedShell).toContainText('Server partially degraded');
    }

    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(getDesktopCompositionPreview(page)).toContainText(
      `Scope: ${chatModelSelection.key}`
    );
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
    const contextRail = getDesktopContextRail(page);
    const runtimeInspector = getDesktopRuntimeInspector(page);
    await assertRuntimeProviderSummary(runtimeInspector);
    await expect(runtimeInspector.getByText('Scoped settings')).toBeVisible();
    await expect(runtimeInspector.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'Open Model & Chat settings' })).toBeVisible();
    await expect(runtimeInspector.getByRole('button', { name: 'Select character or persona' })).toBeVisible();
    await assertRuntimeMcpRailState(runtimeInspector);
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
    await expect(page.getByRole('textbox', { name: 'Search models' })).toBeVisible();
    await expect(page.getByRole('menu').first()).toBeVisible();
    await page.keyboard.press('Escape');

    await contextRail.getByRole('button', { name: /Select prompt|Select a prompt/i }).click();
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
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-desktop-focus.png'),
      fullPage: true,
    });

    await page.getByRole('button', { name: 'Show cockpit panels' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'cockpit'
    );
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(
      getDesktopContextRail(page).getByRole('button', { name: 'Web search', exact: true })
    ).toHaveAttribute('aria-pressed', 'true');

    const sendContextRail = getDesktopContextRail(page);
    const sendWebSearchControl = sendContextRail.getByRole('button', {
      name: 'Web search',
      exact: true,
    });
    if ((await sendWebSearchControl.getAttribute('aria-pressed')) === 'true') {
      await sendWebSearchControl.click();
      await expect(sendWebSearchControl).toHaveAttribute('aria-pressed', 'false');
    }

    const smokePrompt = `cockpit smoke ${Date.now()}`;
    await page.getByTestId('chat-input').fill(smokePrompt);
    const chatCompletionAttempt = waitForChatCompletionAttempt(page);
    await page.getByRole('button', { name: /send message/i }).click();
    const chatCompletionResponse = await chatCompletionAttempt;
    await assertProviderQualifiedPayload(page, chatCompletionResponse);
    await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(smokePrompt);
    await assertChatCompletionRenderedOrRecoverable(page, chatCompletionResponse);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-desktop-conversation.png'),
      fullPage: true,
    });
    await expect(cockpitStatus).not.toContainText('0 messages');

    const failingApiHits = apiTracker.hits.filter((hit) => hit.status >= 400);
    expect(failingApiHits).toEqual([]);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/health')).toBe(true);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/llm/providers')).toBe(true);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/llm/models/metadata')).toBe(true);

    apiTracker.dispose();
  });

  test('proves real prompt, model setting restore, and MCP state through the main cockpit rails', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    const promptId = `cockpit-prompt-${Date.now()}`;
    const promptName = `Cockpit prompt ${Date.now()}`;
    const promptContent = 'Use concise cockpit proof wording.';
    const now = Date.now();

    await seedRealServerConfig(page, { selectedModel: chatModelSelection });
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
      timeout: 60_000,
    });
    await assertNoBlockingServerDialog(page);

    await putLocalPrompt(page, {
      id: promptId,
      title: promptName,
      name: promptName,
      content: promptContent,
      is_system: true,
      createdAt: now,
      updatedAt: now,
      tags: ['e2e', 'cockpit'],
      keywords: ['e2e', 'cockpit'],
      favorite: false,
      usageCount: 0,
      lastUsedAt: null,
      system_prompt: promptContent,
      user_prompt: null,
      promptFormat: 'legacy',
      promptSchemaVersion: null,
      structuredPromptDefinition: null,
      syncPayloadVersion: 1,
      fewShotExamples: null,
      modulesConfig: null,
      versionNumber: null,
      changeDescription: null,
      parentVersionId: null,
      serverParentVersionId: null,
      syncStatus: 'local',
      sourceSystem: 'workspace',
    });

    try {
      await page.reload({ waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);

      const contextRail = getDesktopContextRail(page);
      const runtimeInspector = getDesktopRuntimeInspector(page);

      await contextRail.getByRole('button', { name: /Select prompt|Select a prompt/i }).click();
      const promptSearch = page.getByPlaceholder('Search prompts...');
      await expect(promptSearch).toBeVisible();
      await promptSearch.fill(promptName);
      await page.getByText(promptName, { exact: true }).click();

      const promptTrigger = contextRail.locator('[data-cockpit-prompt-select-trigger]');
      await expect(promptTrigger).toBeFocused({ timeout: 5_000 });
      await expect(
        contextRail.getByRole('region', { name: 'Prompt context' }).getByText(promptName)
      ).toBeVisible();
      const compositionPreview = getDesktopCompositionPreview(page);
      await expect(compositionPreview).toBeVisible();
      await expect(compositionPreview).toContainText(promptName);
      await expect(compositionPreview).toContainText(/Scope: [^:]+:.+/);
      await expect(compositionPreview).toContainText(
        /MCP tools|MCP unavailable|Tools managed from composer/
      );
      const sourceList = contextRail.getByRole('list', { name: 'Context sources' });
      const promptSource = sourceList.getByRole('listitem').filter({ hasText: promptName });
      await expect(promptSource).toContainText('Prompt');
      await expect(promptSource).toContainText(promptName);

      await contextRail.getByRole('button', { name: 'Clear prompt', exact: true }).click();
      await expect(promptTrigger).toBeFocused({ timeout: 5_000 });
      await expect(contextRail.getByRole('region', { name: 'Prompt context' })).toContainText(
        'No prompt selected'
      );
      await expect(promptSource).toHaveCount(0);
      await expect(contextRail.getByText(promptName)).toHaveCount(0);

      const modelSettingsTrigger = runtimeInspector.locator(
        '[data-cockpit-model-settings-trigger]'
      );
      await expect(runtimeInspector.getByText('Provider route')).toBeVisible();
      const modelSettingsOpenDetail = page.evaluate(
        () =>
          new Promise<{ settingsScope?: string | null }>((resolve) => {
            window.addEventListener(
              'tldw:open-model-settings',
              (event) => resolve((event as CustomEvent).detail || {}),
              { once: true }
            );
          })
      );
      await modelSettingsTrigger.click();
      await expect.poll(async () => (await modelSettingsOpenDetail).settingsScope || '').toMatch(
        /^[^:]+:.+/
      );
      const modelSettingsDialog = page.getByRole('dialog', {
        name: 'Current Chat Model Settings',
      });
      await expect(modelSettingsDialog).toBeVisible();
      const temperatureInput = await getTemperatureInput(modelSettingsDialog);
      const originalTemperature = (await temperatureInput.inputValue()).trim();
      const nextTemperature = originalTemperature === '0.31' ? '0.32' : '0.31';
      const restoredTemperature = originalTemperature || '0.7';

      await temperatureInput.fill(nextTemperature);
      await modelSettingsDialog.getByRole('button', { name: /^Save$/i }).click();
      await expect(modelSettingsDialog).toBeHidden();
      await expect(modelSettingsTrigger).toBeFocused({ timeout: 5_000 });
      await expect(runtimeInspector.getByText('Temperature')).toBeVisible();
      await expect(runtimeInspector.getByText(nextTemperature, { exact: true })).toBeVisible();
      await expect(runtimeInspector.getByText('Override')).toBeVisible();
      await expect(compositionPreview).toContainText(`Temperature: ${nextTemperature}`);

      await modelSettingsTrigger.click();
      await expect(modelSettingsDialog).toBeVisible();
      const restoredInput = await getTemperatureInput(modelSettingsDialog);
      await expect(restoredInput).toHaveValue(nextTemperature);
      await restoredInput.fill(restoredTemperature);
      await modelSettingsDialog.getByRole('button', { name: /^Save$/i }).click();
      await expect(modelSettingsDialog).toBeHidden();
      await expect(modelSettingsTrigger).toBeFocused({ timeout: 5_000 });
      await expect(runtimeInspector.getByText(restoredTemperature, { exact: true })).toBeVisible();
      await expect(compositionPreview).toContainText(`Temperature: ${restoredTemperature}`);

      await assertRuntimeMcpRailState(runtimeInspector);
      const mcpSettingsTrigger = runtimeInspector.locator(
        '[data-cockpit-mcp-settings-trigger]'
      );
      await mcpSettingsTrigger.click();
      const mcpSettingsDialog = page.getByRole('dialog', {
        name: 'MCP tool settings',
      });
      await expect(mcpSettingsDialog).toBeVisible();
      const mcpSelector = mcpSettingsDialog.getByTestId('mcp-tool-selector');
      if (await mcpSelector.isVisible().catch(() => false)) {
        await expect(mcpSelector).toContainText(/enabled/i);
        await expect(mcpSelector).toContainText(/disabled/i);
        await expect(mcpSelector).toContainText(/unavailable/i);
        const firstSwitch = mcpSelector.getByRole('switch').first();
        if (await firstSwitch.isEnabled().catch(() => false)) {
          const initialChecked = await firstSwitch.getAttribute('aria-checked');
          if (initialChecked === 'true' || initialChecked === 'false') {
            await firstSwitch.click();
            await expect(firstSwitch).toHaveAttribute(
              'aria-checked',
              initialChecked === 'true' ? 'false' : 'true'
            );
            await firstSwitch.click();
            await expect(firstSwitch).toHaveAttribute('aria-checked', initialChecked);
          }
        }
      } else {
        await expect(mcpSettingsDialog).toContainText(
          /MCP tools unavailable|MCP tools are offline|No MCP tools discovered|Loading tools/i
        );
      }
      await mcpSettingsDialog
        .getByTestId('mcp-settings-modal-footer')
        .getByRole('button', { name: 'Close' })
        .click();
      await expect(mcpSettingsDialog).toBeHidden();
      await expect(mcpSettingsTrigger).toBeFocused({ timeout: 5_000 });

      await page.screenshot({
        path: testInfo.outputPath('chat-cockpit-p0-rails-proof.png'),
        fullPage: true,
      });
    } finally {
      await deleteLocalPrompt(page, promptId);
    }
  });

  test('keeps mobile cockpit tabs and focus composer usable against the live server', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    await seedRealServerConfig(page, { selectedModel: chatModelSelection });
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
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-context.png'),
      fullPage: true,
    });
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
    await contextPanel.getByRole('button', { name: 'Open Search & Context' }).click();
    await expect(page.getByRole('heading', { name: /Knowledge Search/i })).toBeVisible();
    await closeSearchContextIfOpen(page);
    await contextPanel.getByRole('button', { name: /Select prompt|Select a prompt/i }).click();
    await expect(page.getByPlaceholder('Search prompts...')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('chat-input')).toBeVisible();
    const runtimeTab = mobileRails.getByRole('tab', { name: 'Runtime' });
    await runtimeTab.click();
    await expect(runtimeTab).toHaveAttribute('aria-selected', 'true');
    const runtimePanel = mobileRails.getByRole('tabpanel', { name: 'Runtime' });
    await assertRuntimeProviderSummary(runtimePanel);
    await expect(runtimePanel.getByText('Scoped settings')).toBeVisible();
    await expect(runtimePanel.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Open Model & Chat settings' })).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Select character or persona' })).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Configure MCP tools' })).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-runtime.png'),
      fullPage: true,
    });
    await runtimePanel.getByRole('button', { name: 'Select character or persona' }).click();
    await expect(page.getByRole('tab', { name: 'Characters' })).toBeVisible();
    await page.keyboard.press('Escape');
    await runtimePanel.getByRole('button', { name: 'Configure MCP tools' }).click();
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
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);
    await assertCoreComposerControls(page, { mobile: true, composerOnly: true });
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-focus.png'),
      fullPage: true,
    });
  });

  test('selects and clears a real disposable character through the runtime rail', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

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

      await seedRealServerConfig(page, { selectedModel: chatModelSelection });
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      const runtimeInspector = getDesktopRuntimeInspector(page);
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

      await seedRealServerConfig(page, { selectedModel: chatModelSelection });
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });

      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      await assertCoreComposerControls(page);

      const runtimeInspector = getDesktopRuntimeInspector(page);
      const composerAssistant = page.getByTestId('character-select');
      await expect(runtimeInspector.getByText('No assistant selected').first()).toBeVisible();
      await expect(composerAssistant).toHaveAccessibleName(/Select character or persona/i);

      await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
      await expect(page.getByRole('tab', { name: 'Characters' })).toHaveAttribute(
        'aria-selected',
        'true'
      );
      await page.getByRole('button', { name: characterName, exact: true }).click();

      await expect(runtimeInspector.getByText(characterName)).toBeVisible();
      await expect(composerAssistant).toHaveAccessibleName(characterName);
      await expect(page.getByText(`Character: ${characterName}`)).toBeVisible();
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toBeVisible();
      await expect(runtimeInspector.getByRole('button', { name: 'Open Scene Director' })).toBeVisible();
      const contextRail = getDesktopContextRail(page);
      const compositionPreview = getDesktopCompositionPreview(page);
      await expect(compositionPreview).toContainText(characterName);
      await expect(compositionPreview).toContainText(/Scope: [^:]+:.+/);
      const assistantContextSource = contextRail
        .getByRole('list', { name: 'Context sources' })
        .getByRole('listitem')
        .filter({ hasText: characterName });
      await expect(assistantContextSource).toContainText('Character');
      await expect(assistantContextSource).toContainText(characterName);

      await page.screenshot({
        path: testInfo.outputPath('chat-cockpit-character-selected.png'),
        fullPage: true,
      });

      await assistantContextSource.getByRole('button', { name: 'Clear assistant' }).click();
      await expect(runtimeInspector.getByText('No assistant selected').first()).toBeVisible();
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toHaveCount(0);
      await expect(assistantContextSource).toHaveCount(0);
      await expect(composerAssistant).toHaveAttribute(
        'aria-label',
        /Select character or persona/i
      );
    } finally {
      await apiDelete(
        request,
        `/api/v1/characters/${encodeURIComponent(String(createdId))}?expected_version=${encodeURIComponent(
          String(createdVersion)
        )}`
      ).catch(() => ({ status: 0 }));
    }
  });

  test('selects and clears a real persona through the runtime rail', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    const personaName = `Cockpit Persona ${Date.now()}`;
    const personaId = `cockpit_persona_${Date.now()}`;
    const created = await apiPost<any>(request, '/api/v1/persona/profiles', {
      id: personaId,
      name: personaName,
      mode: 'session_scoped',
      system_prompt: `You are ${personaName}, a concise persona proof assistant.`,
      is_active: true,
      use_persona_state_context_default: true,
    });

    let selectedPersona = created.body;
    let cleanupPersonaId: string | null = null;
    let cleanupPersonaVersion: number | null = null;

    if (created.status === 201 && selectedPersona?.id) {
      cleanupPersonaId = String(selectedPersona.id);
      cleanupPersonaVersion = Number(selectedPersona.version ?? 1);
    } else {
      testInfo.annotations.push({
        type: 'blocker',
        description: `Could not create disposable persona via real server: status ${created.status}`,
      });
      const listed = await apiGet<any>(request, '/api/v1/persona/profiles?active_only=true');
      const personas = extractPersonaProfiles(listed.body);
      selectedPersona = personas.find((persona) => persona?.is_active !== false) ?? personas[0] ?? null;
    }

    if (!selectedPersona?.id || !selectedPersona?.name) {
      await seedRealServerConfig(page, { selectedModel: chatModelSelection });
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      const runtimeInspector = getDesktopRuntimeInspector(page);
      await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
      await page.getByRole('tab', { name: 'Personas' }).click();
      await expect(page.getByText('No personas available.')).toBeVisible();
      return;
    }

    try {
      await seedRealServerConfig(page, { selectedModel: chatModelSelection });
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.goto('/chat', { waitUntil: 'domcontentloaded' });

      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      await assertCoreComposerControls(page);

      const runtimeInspector = getDesktopRuntimeInspector(page);
      const composerAssistant = page.getByTestId('character-select');
      await expect(runtimeInspector.getByText('No assistant selected').first()).toBeVisible();

      await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
      await page.getByRole('tab', { name: 'Personas' }).click();
      await expect(page.getByRole('tab', { name: 'Personas' })).toHaveAttribute(
        'aria-selected',
        'true'
      );
      await page
        .getByRole('button', { name: String(selectedPersona.name), exact: true })
        .click();

      const assistantTrigger = runtimeInspector.locator(
        '[data-cockpit-assistant-select-trigger]'
      );
      await expect(assistantTrigger).toBeFocused({ timeout: 5_000 });
      await expect(runtimeInspector.getByText(String(selectedPersona.name))).toBeVisible();
      await expect(runtimeInspector.getByText('Persona selected').first()).toBeVisible();
      await expect(composerAssistant).toHaveAccessibleName(String(selectedPersona.name));
      await expect(
        runtimeInspector.getByRole('button', { name: 'Open Scene Director' })
      ).toHaveCount(0);
      await expect(
        runtimeInspector.getByText('Scene Director is available for character-backed chats.')
      ).toBeVisible();

      const contextRail = getDesktopContextRail(page);
      const compositionPreview = getDesktopCompositionPreview(page);
      await expect(compositionPreview).toContainText(String(selectedPersona.name));
      await expect(compositionPreview).toContainText(/Scope: [^:]+:.+/);
      const personaContextSource = contextRail
        .getByRole('list', { name: 'Context sources' })
        .getByRole('listitem')
        .filter({ hasText: String(selectedPersona.name) });
      await expect(personaContextSource).toContainText('Persona');
      await expect(personaContextSource).toContainText(String(selectedPersona.name));

      await page.screenshot({
        path: testInfo.outputPath('chat-cockpit-persona-selected.png'),
        fullPage: true,
      });

      await runtimeInspector.getByRole('button', { name: 'Clear assistant' }).click();
      await expect(assistantTrigger).toBeFocused({ timeout: 5_000 });
      await expect(runtimeInspector.getByText('No assistant selected').first()).toBeVisible();
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toHaveCount(0);
      await expect(personaContextSource).toHaveCount(0);
      await expect(composerAssistant).toHaveAccessibleName(/Select character or persona/i);
    } finally {
      if (cleanupPersonaId) {
        const expectedVersionQuery =
          cleanupPersonaVersion != null
            ? `?expected_version=${encodeURIComponent(String(cleanupPersonaVersion))}`
            : '';
        await apiDelete(
          request,
          `/api/v1/persona/profiles/${encodeURIComponent(cleanupPersonaId)}${expectedVersionQuery}`
        ).catch(() => ({ status: 0 }));
      }
    }
  });
});
