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
import { captureAllApiCalls, type CapturedApiCall } from '../utils/api-assertions';
import { waitForStreamComplete } from '../utils/journey-helpers';

const serverUrl = (
  process.env.TLDW_E2E_SERVER_URL ||
  process.env.TLDW_SERVER_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

const apiKey =
  process.env.TLDW_E2E_API_KEY || process.env.TLDW_API_KEY || process.env.SINGLE_USER_API_KEY || '';
const expectStreamingControlEvidence =
  process.env.TLDW_E2E_EXPECT_STREAMING_CONTROLS === 'true';

test.skip(
  !apiKey,
  'TLDW_E2E_API_KEY, TLDW_API_KEY, or SINGLE_USER_API_KEY is required for real-server chat cockpit checks'
);

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

type DisposableCharacter = {
  id: string;
  name: string;
  firstMessage: string;
  version: number;
};

type DisposablePersona = {
  id: string;
  name: string;
  version: number;
};

const apiHeaders = () => ({
  'x-api-key': apiKey,
});

const truncateForDiagnostics = (value: string, maxLength = 800): string =>
  value.length > maxLength ? `${value.slice(0, maxLength)}...` : value;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

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
      responseText = text ? truncateForDiagnostics(text) : '<empty response body>';
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

const apiGetWithRetry = async <T>(
  request: APIRequestContext,
  path: string,
  options?: { attempts?: number; retryDelayMs?: number }
): Promise<{ status: number; body: T }> => {
  const attempts = Math.max(1, options?.attempts ?? 3);
  const retryDelayMs = options?.retryDelayMs ?? 1_500;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const response = await apiGet<T>(request, path);
    if (response.status !== 429 || attempt === attempts) {
      return response;
    }
    await sleep(retryDelayMs * attempt);
  }

  throw new Error(`Unreachable retry loop for GET ${path}`);
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

const apiDelete = async (request: APIRequestContext, path: string): Promise<{ status: number }> => {
  const response = await request.delete(`${serverUrl}${path}`, {
    headers: apiHeaders(),
    timeout: 30_000,
  });
  return { status: response.status() };
};

const apiPostWithRetry = async <T>(
  request: APIRequestContext,
  path: string,
  body: Record<string, unknown>,
  options?: { attempts?: number; retryDelayMs?: number }
): Promise<{ status: number; body: T | null }> => {
  const attempts = Math.max(1, options?.attempts ?? 3);
  const retryDelayMs = options?.retryDelayMs ?? 1_500;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const response = await apiPost<T>(request, path, body);
    if (response.status !== 429 || attempt === attempts) {
      return response;
    }
    await sleep(retryDelayMs * attempt);
  }

  throw new Error(`Unreachable retry loop for POST ${path}`);
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

const normalizeCockpitProviderKey = (provider: string): string => {
  const normalized = provider.trim().toLowerCase();
  if (normalized === 'llama.cpp') return 'llamacpp';
  if (normalized === 'local-llm') return 'local';
  return normalized;
};

const normalizeConfiguredChatModelId = (providerName: string, value: unknown): string => {
  let model = String(value || '')
    .trim()
    .replace(/^tldw:/i, '');
  const separatorIndex = model.indexOf(':');
  if (separatorIndex <= 0 || separatorIndex === model.length - 1) return model;

  const prefix = normalizeCockpitProviderKey(model.slice(0, separatorIndex));
  const configuredProvider = normalizeCockpitProviderKey(providerName);
  if (prefix === configuredProvider) {
    model = model.slice(separatorIndex + 1).trim();
  }

  return model;
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

  const providerName = String(provider.name || '').trim();
  if (!providerName) {
    throw new Error('Configured chat model provider is missing a provider name');
  }

  const rawModel =
    typeof provider.default_model === 'string' && provider.default_model.trim().length > 0
      ? provider.default_model.trim()
      : String(provider.models[0] || '').trim();
  const model = normalizeConfiguredChatModelId(providerName, rawModel);

  if (!model) {
    throw new Error(`Configured provider ${provider.name || '<unknown>'} has no usable model`);
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
  options: {
    selectedModel?: RealChatModelSelection;
    persistedServerChatId?: string | null;
  } = {}
) => {
  await page.addInitScript(
    ({
      configuredServerUrl,
      configuredApiKey,
      configuredSelectedModel,
      configuredPersistedServerChatId,
    }) => {
      const fnv1a36 = (value: string) => {
        let hash = 2166136261;
        for (let i = 0; i < value.length; i += 1) {
          hash ^= value.charCodeAt(i);
          hash +=
            (hash << 1) +
            (hash << 4) +
            (hash << 7) +
            (hash << 8) +
            (hash << 24);
        }
        return (hash >>> 0).toString(36);
      };
      const normalizeServerIdentity = (rawServerUrl: string) => {
        const raw = String(rawServerUrl || '').trim();
        if (!raw) return '';
        try {
          const parsed = new URL(raw);
          const protocol = parsed.protocol.toLowerCase();
          const hostname = parsed.hostname.toLowerCase();
          const includePort = Boolean(
            parsed.port &&
              !(
                (protocol === 'http:' && parsed.port === '80') ||
                (protocol === 'https:' && parsed.port === '443')
              )
          );
          const port = includePort ? `:${parsed.port}` : '';
          const pathname = parsed.pathname.replace(/\/+$/, '');
          return `${protocol}//${hostname}${port}${pathname}`;
        } catch {
          return raw.replace(/\/+$/, '').toLowerCase();
        }
      };
      const serverFingerprint = (() => {
        const normalized = normalizeServerIdentity(configuredServerUrl);
        return normalized ? `server:${fnv1a36(normalized)}` : 'server:unknown';
      })();
      const apiKeyScope = String(configuredApiKey || '').trim()
        ? `key:${fnv1a36(String(configuredApiKey).trim())}`
        : 'key:none';
      const persistedScopeKey = `${serverFingerprint}:auth:single-user:org:none:user:single-user:${apiKeyScope}`;
      const config = {
        serverUrl: configuredServerUrl,
        authMode: 'single-user',
        apiKey: configuredApiKey,
        requestTimeoutMs: 60_000,
        chatRequestTimeoutMs: 120_000,
        chatStartupTimeoutMs: 60_000,
        chatStreamIdleTimeoutMs: 120_000,
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
      if (configuredPersistedServerChatId) {
        localStorage.setItem(
          'tldw-playground-session',
          JSON.stringify({
            state: {
              historyId: null,
              serverChatId: configuredPersistedServerChatId,
              scopeKey: persistedScopeKey,
              chatMode: 'normal',
              webSearch: false,
              compareMode: false,
              compareSelectedModels: [],
              ragMediaIds: null,
              ragSearchMode: 'hybrid',
              ragTopK: null,
              ragEnableGeneration: true,
              ragEnableCitations: true,
              queuedMessages: [],
              lastUpdated: Date.now(),
            },
            version: 0,
          })
        );
      }

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
      configuredPersistedServerChatId: options.persistedServerChatId ?? null,
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

const assertVisibleTooltipForControl = async (
  page: Page,
  control: Locator,
  expectedText: string
) => {
  const describedBy = await control.getAttribute('aria-describedby');
  expect(describedBy).toBeTruthy();
  const describedByIds = (describedBy ?? '').split(/\s+/).filter(Boolean);
  expect(describedByIds.length).toBeGreaterThan(0);

  let tooltip: Locator | null = null;
  for (const id of describedByIds) {
    const candidate = page.locator(`[id=${JSON.stringify(id)}]`);
    if ((await candidate.count()) === 0) continue;

    const role = await candidate.getAttribute('role');
    const text = (await candidate.textContent())?.trim();
    if (role === 'tooltip' && text === expectedText) {
      tooltip = candidate;
      break;
    }
  }

  expect(
    tooltip,
    `Expected aria-describedby (${describedByIds.join(
      ', '
    )}) to include tooltip text "${expectedText}"`
  ).not.toBeNull();
  await control.hover();
  await expect(tooltip!).toHaveText(expectedText);
  await expect(tooltip!).toHaveCSS('opacity', '1');
};

const getDesktopContextRail = (page: Page): Locator =>
  page.getByTestId('playground-cockpit-left-rail').getByTestId('playground-context-rail');

const getDesktopCompositionPreview = (page: Page): Locator =>
  getDesktopContextRail(page).getByRole('region', {
    name: 'Next message composition',
  });

const getDesktopRuntimeInspector = (page: Page): Locator =>
  page.getByTestId('playground-cockpit-right-rail').getByTestId('playground-runtime-inspector');

const assertRuntimeAssistantCleared = async (runtimeInspector: Locator) => {
  await expect(runtimeInspector.getByText('No runtime assistant selected').first()).toBeVisible();
  await expect(runtimeInspector.getByText('No assistant selected')).toHaveCount(0);
  await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toHaveCount(0);
};

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
    await expect(
      runtimeInspector.getByRole('button', { name: 'MCP tool choice None' })
    ).toBeVisible();
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

const assertNoHorizontalOverflow = async (page: Page) => {
  const metrics = await page.evaluate(() => ({
    innerWidth: window.innerWidth,
    docScrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }));

  expect(metrics.docScrollWidth).toBeLessThanOrEqual(metrics.innerWidth + 1);
  expect(metrics.bodyScrollWidth).toBeLessThanOrEqual(metrics.innerWidth + 1);
};

const assertNoVerticalOverlap = async (first: Locator, second: Locator, label: string) => {
  const firstBox = await first.boundingBox();
  const secondBox = await second.boundingBox();

  expect(firstBox, `${label}: first element is measurable`).not.toBeNull();
  expect(secondBox, `${label}: second element is measurable`).not.toBeNull();
  expect(firstBox!.y + firstBox!.height).toBeLessThanOrEqual(secondBox!.y + 1);
};

const assertHealthResponse = (health: { status: number; body: any }) => {
  expect([200, 206]).toContain(health.status);
  expect(['ok', 'healthy', 'degraded']).toContain(health.body?.status);
};

const waitForChatCompletionAttempt = (page: Page, timeout = 15_000) => {
  const backendOrigin = new URL(serverUrl).origin;
  const pageOrigin = new URL(page.url()).origin;

  return page.waitForResponse(
    (response) => {
      const url = new URL(response.url());
      if (url.origin !== backendOrigin && url.origin !== pageOrigin) return false;
      return (
        (url.pathname === '/api/v1/chat/completions' ||
          /^\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(url.pathname) ||
          /^\/api\/v1\/chats\/[^/]+\/completions$/.test(url.pathname)) &&
        response.request().method() === 'POST'
      );
    },
    { timeout }
  );
};

type ControlCandidate = {
  label: string;
  locator: Locator;
  requireEnabled?: boolean;
};

const waitForFirstAvailableControl = async (
  candidates: ControlCandidate[],
  timeout = 10_000
): Promise<ControlCandidate | null> => {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    for (const candidate of candidates) {
      const visible = await candidate.locator.isVisible().catch(() => false);
      if (!visible) continue;
      const enabled = candidate.requireEnabled
        ? await candidate.locator.isEnabled().catch(() => false)
        : true;
      if (enabled) return candidate;
    }
    await sleep(100);
  }
  return null;
};

const clickFirstAvailableControl = async (
  candidates: ControlCandidate[]
): Promise<ControlCandidate | null> => {
  for (const candidate of candidates) {
    const visible = await candidate.locator.isVisible().catch(() => false);
    if (!visible) continue;
    const enabled = candidate.requireEnabled
      ? await candidate.locator.isEnabled().catch(() => false)
      : true;
    if (!enabled) continue;
    const clicked = await candidate.locator
      .click({ timeout: 1_000 })
      .then(() => true)
      .catch(() => false);
    if (clicked) return candidate;
  }
  return null;
};


const assertChatCompletionRenderedOrRecoverable = async (
  page: Page,
  response?: Response | null
) => {
  const chatLog = page.getByRole('log', { name: /chat messages/i });
  if (!response || response.status() < 400) {
    await waitForStreamComplete(page, 60_000);
    const assistantMessage = chatLog
      .locator(
        "article[aria-label*='Assistant message'], [data-role='assistant'], [data-message-role='assistant'], .assistant-message"
      )
      .last();
    await expect(assistantMessage).toBeVisible({ timeout: 5_000 });

    const emptyResponseNotice = assistantMessage.getByRole('status', {
      name: 'Empty assistant response',
    });
    if ((await emptyResponseNotice.count()) > 0) {
      await expect(emptyResponseNotice).toContainText('No response text was returned.');
      const runtimeInspector = getDesktopRuntimeInspector(page);
      await expect(
        runtimeInspector.getByRole('status', {
          name: 'Empty assistant response',
        })
      ).toContainText('No response text returned.');
      await expect(
        runtimeInspector.getByRole('button', { name: 'Regenerate last response' })
      ).toBeEnabled();
    }
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
  const routeLabel = visibleRouteLabel?.trim() || routeText?.replace(/^Route\s+/, '').trim() || '';
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

const escapeCssAttrValue = (value: string): string =>
  value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');

const selectConfiguredCockpitModel = async (
  page: Page,
  selection: RealChatModelSelection
): Promise<string> => {
  await ensureComposerOptionsVisible(page);
  await page.getByTestId('model-selector').first().click();

  const scopeToggle = page.getByTestId('model-list-scope-toggle');
  await expect(scopeToggle).toBeVisible();
  await expect(scopeToggle).toHaveText(/Search all models/);
  await expect(page.getByText('Usable configured models')).toBeVisible();

  await scopeToggle.click();
  await expect(scopeToggle).toHaveText(/Configured/);
  await expect(page.getByText('All known models')).toBeVisible();

  await scopeToggle.click();
  await expect(scopeToggle).toHaveText(/Search all models/);
  await expect(page.getByText('Usable configured models')).toBeVisible();

  await page.getByLabel('Search models').fill(selection.model);

  const modelKey = `${normalizeCockpitProviderKey(selection.provider)}:${selection.model}`;
  const option = page
    .locator(
      `[data-testid="model-selector-option"][data-model-key="${escapeCssAttrValue(modelKey)}"]`
    )
    .first();
  await expect(option).toBeVisible({ timeout: 30_000 });
  await option.click();

  return modelKey;
};

const openDesktopChatCockpit = async (
  page: Page,
  selection: RealChatModelSelection,
  options: { persistedServerChatId?: string | null } = {}
) => {
  await seedRealServerConfig(page, {
    selectedModel: selection,
    persistedServerChatId: options.persistedServerChatId ?? null,
  });
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
    timeout: 60_000,
  });
  await assertNoBlockingServerDialog(page);
  await assertCoreComposerControls(page);
  await expect(page.getByTestId('character-control-rail')).toHaveCount(0);
};

const createDisposableCharacter = async (
  request: APIRequestContext,
  name: string
): Promise<DisposableCharacter> => {
  const firstMessage = 'Ready for overlay continuity proof.';
  const created = await apiPostWithRetry<any>(request, '/api/v1/characters', {
    name,
    system_prompt: `You are ${name}, a concise overlay test assistant.`,
    first_message: firstMessage,
    creator: 'tldw e2e',
    tags: ['e2e', 'overlay-rail'],
  }, {
    attempts: 6,
    retryDelayMs: 2_000,
  });

  expect(created.status).toBe(201);
  expect(created.body?.id).toBeTruthy();

  return {
    id: String(created.body.id),
    firstMessage,
    name,
    version: Number(created.body.version ?? 1),
  };
};

const createDisposablePersona = async (
  request: APIRequestContext,
  personaId: string,
  name: string
): Promise<DisposablePersona> => {
  const created = await apiPostWithRetry<any>(request, '/api/v1/persona/profiles', {
    id: personaId,
    name,
    mode: 'session_scoped',
    system_prompt: `You are ${name}, a concise overlay test persona.`,
    is_active: true,
    use_persona_state_context_default: true,
  }, {
    attempts: 6,
    retryDelayMs: 2_000,
  });

  expect(created.status).toBe(201);
  expect(created.body?.id).toBeTruthy();

  await expect
    .poll(
      async () => {
        const catalog = await apiGetWithRetry<any>(request, '/api/v1/persona/catalog');
        return extractPersonaProfiles(catalog.body).some((item) => {
          const candidateId =
            typeof item?.id === 'string' || typeof item?.id === 'number'
              ? String(item.id)
              : '';
          return candidateId === String(created.body?.id ?? personaId);
        });
      },
      {
        timeout: 30_000,
        message: `Timed out waiting for persona ${personaId} to appear in the persona catalog`,
      }
    )
    .toBe(true);

  return {
    id: String(created.body.id),
    name,
    version: Number(created.body.version ?? 1),
  };
};

const cleanupDisposableCharacter = async (
  request: APIRequestContext,
  character: DisposableCharacter | null
) => {
  if (!character) return;
  await apiDelete(
    request,
    `/api/v1/characters/${encodeURIComponent(character.id)}?expected_version=${encodeURIComponent(
      String(character.version)
    )}`
  ).catch(() => ({ status: 0 }));
};

const cleanupDisposablePersona = async (
  request: APIRequestContext,
  persona: DisposablePersona | null
) => {
  if (!persona) return;
  await apiDelete(
    request,
    `/api/v1/persona/profiles/${encodeURIComponent(persona.id)}?expected_version=${encodeURIComponent(
      String(persona.version)
    )}`
  ).catch(() => ({ status: 0 }));
};

const selectAssistantFromRuntimeRail = async (
  page: Page,
  options: {
    tab: 'Characters' | 'Personas';
    assistantName: string;
  }
) => {
  const runtimeInspector = getDesktopRuntimeInspector(page);
  await runtimeInspector
    .getByRole('button', { name: 'Select character or persona' })
    .click();
  const panel = page.getByTestId('assistant-select-panel');
  await expect(panel).toBeVisible();
  await page.getByRole('tab', { name: options.tab }).click();
  await expect(page.getByRole('tab', { name: options.tab })).toHaveAttribute(
    'aria-selected',
    'true'
  );
  const assistantButton = page.getByRole('button', {
    name: options.assistantName,
    exact: true,
  });
  const retryButton = page.getByRole('button', {
    name: options.tab === 'Personas' ? 'Retry personas' : 'Retry characters',
  });
  for (let attempt = 0; attempt < 6; attempt += 1) {
    if (await assistantButton.isVisible().catch(() => false)) {
      break;
    }
    if (await retryButton.isVisible().catch(() => false)) {
      await retryButton.click();
    }
    await page.waitForTimeout(1_500);
  }
  await expect(assistantButton).toBeVisible({ timeout: 30_000 });
  await assistantButton.click();
  await expect(panel).toBeHidden({ timeout: 10_000 });
};

const findChatCreateCall = (calls: CapturedApiCall[]): CapturedApiCall | undefined =>
  [...calls].reverse().find((call) => {
    const url = new URL(call.url);
    return call.method === 'POST' && /^\/api\/v1\/chats\/?$/.test(url.pathname);
  });

const extractConversationChatIdFromCall = (call: CapturedApiCall | undefined): string | null => {
  if (!call) return null;
  const url = new URL(call.url);
  const match = url.pathname.match(/^\/api\/v1\/chats\/([^/]+)\/(?:complete-v2|completions)$/);
  if (match?.[1]) {
    return match[1];
  }

  const requestBody =
    call.requestBody && typeof call.requestBody === 'object'
      ? (call.requestBody as Record<string, unknown>)
      : null;
  const conversationId =
    requestBody?.conversation_id ?? requestBody?.conversationId ?? requestBody?.chat_id;
  return typeof conversationId === 'string' && conversationId.trim().length > 0
    ? conversationId.trim()
    : typeof conversationId === 'number'
      ? String(conversationId)
      : null;
};

const extractConversationChatIdFromResponse = (response: Response | null): string | null => {
  if (!response) return null;
  const url = new URL(response.url());
  const match = url.pathname.match(/^\/api\/v1\/chats\/([^/]+)\/(?:complete-v2|completions)$/);
  return match?.[1] ? match[1] : null;
};

const extractCreatedChatIdFromCreateCall = (call: CapturedApiCall | undefined): string | null => {
  if (!call) return null;
  const responseBody =
    call.responseBody && typeof call.responseBody === 'object'
      ? (call.responseBody as Record<string, unknown>)
      : null;
  const responseId = responseBody?.id;
  if (typeof responseId === 'string' && responseId.trim().length > 0) {
    return responseId.trim();
  }
  if (typeof responseId === 'number' && Number.isFinite(responseId)) {
    return String(responseId);
  }
  return null;
};

const findConversationTurnCall = (calls: CapturedApiCall[]): CapturedApiCall | undefined =>
  [...calls].reverse().find((call) => {
    const url = new URL(call.url);
    return (
      call.method === 'POST' &&
      (url.pathname === '/api/v1/chat/completions' ||
        /^\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(url.pathname) ||
        /^\/api\/v1\/chats\/[^/]+\/completions$/.test(url.pathname))
    );
  });

const sendChatTurnAndCapture = async (
  page: Page,
  prompt: string,
  options: {
    stopStreamingAfterRequest?: boolean;
    stopTimeoutMs?: number;
    settleAfterStopMs?: number;
  } = {}
): Promise<{
  response: Response | null;
  calls: CapturedApiCall[];
}> => {
  const capture = captureAllApiCalls(page);
  const completionAttempt = waitForChatCompletionAttempt(page, 90_000).catch(() => null);

  await page.getByTestId('chat-input').fill(prompt);
  await page.getByRole('button', { name: /send message/i }).click();

  const response = await completionAttempt;
  if (options.stopStreamingAfterRequest) {
    const stopStreaming = page.getByRole('button', { name: /stop streaming response/i });
    const stopTimeoutMs = options.stopTimeoutMs ?? 30_000;
    const stopStreamingVisible = await stopStreaming
      .waitFor({ state: 'visible', timeout: stopTimeoutMs })
      .then(() => true)
      .catch(() => false);
    if (stopStreamingVisible) {
      await stopStreaming.click().catch(() => undefined);
    }
    const settleAfterStopMs = options.settleAfterStopMs ?? 1_000;
    if (settleAfterStopMs > 0) {
      await page.waitForTimeout(settleAfterStopMs);
    }
  }
  const calls = await capture.stop();
  try {
    if (!options.stopStreamingAfterRequest) {
      await assertChatCompletionRenderedOrRecoverable(page, response);
      await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(prompt);
    }
  } catch (error) {
    console.log(
      '[sendChatTurnAndCapture:failure]',
      JSON.stringify(
        calls.map((call) => ({
          method: call.method,
          url: call.url,
          status: call.status,
          requestBody: call.requestBody,
          responseBody: call.responseBody
        })),
        null,
        2
      )
    );
    throw error;
  }

  return {
    response,
    calls,
  };
};

const waitForSuccessfulChatMessagesLoad = (
  page: Page,
  chatId: string,
  timeout = 60_000
) => {
  const backendOrigin = new URL(serverUrl).origin;
  return page.waitForResponse(
    (response) => {
      const url = new URL(response.url());
      return (
        response.request().method() === 'GET' &&
        url.origin === backendOrigin &&
        url.pathname === `/api/v1/chats/${encodeURIComponent(chatId)}/messages` &&
        response.status() === 200
      );
    },
    { timeout }
  );
};

const getChatDetails = async (
  request: APIRequestContext,
  chatId: string
): Promise<{ status: number; body: any }> =>
  apiGetWithRetry<any>(request, `/api/v1/chats/${encodeURIComponent(chatId)}`);

type PlaygroundSessionSnapshot = {
  historyId?: string | null;
  serverChatId?: string | null;
} | null;

const readPlaygroundSessionSnapshot = async (
  page: Page
): Promise<PlaygroundSessionSnapshot> =>
  page.evaluate(() => {
    const raw = window.localStorage.getItem('tldw-playground-session');
    if (!raw) return null;

    try {
      const parsed = JSON.parse(raw) as
        | { state?: { historyId?: string | null; serverChatId?: string | null } | null }
        | { historyId?: string | null; serverChatId?: string | null }
        | null;
      if (parsed && typeof parsed === 'object' && 'state' in parsed) {
        return parsed.state ?? null;
      }
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
      return null;
    }
  });

const waitForPersistedServerChatId = async (page: Page): Promise<string> => {
  await expect
    .poll(
      async () => {
        const snapshot = await readPlaygroundSessionSnapshot(page);
        return snapshot?.serverChatId ?? null;
      },
      {
        timeout: 15_000,
      }
    )
    .toBeTruthy();

  const snapshot = await readPlaygroundSessionSnapshot(page);
  if (!snapshot?.serverChatId) {
    throw new Error('Expected playground session to persist a serverChatId');
  }
  return snapshot.serverChatId;
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

    await expect(apiPost(fakeRequest, '/api/v1/example', { sample: true })).rejects.toThrow(
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

    await expect(apiPost(fakeRequest, '/api/v1/example', { sample: true })).resolves.toEqual({
      status: 204,
      body: null,
    });
  });

  test('normalizes configured provider model IDs before deriving cockpit keys', () => {
    expect(
      buildConfiguredChatModelSelection({
        providers: [
          {
            name: 'openai',
            is_configured: true,
            default_model: 'tldw:openai:gpt-4.1-mini',
            models: ['openai:gpt-4.1-mini'],
          },
        ],
      })
    ).toEqual({
      provider: 'openai',
      model: 'gpt-4.1-mini',
      key: 'tldw:gpt-4.1-mini',
    });

    expect(
      buildConfiguredChatModelSelection({
        providers: [
          {
            name: 'llama.cpp',
            is_configured: true,
            models: ['llama3:latest'],
          },
        ],
      }).model
    ).toBe('llama3:latest');
  });

  test('does not intercept backend routes in this real-server spec', async ({
    browserName: _browserName,
  }, testInfo) => {
    const fs = await import('node:fs');
    const { readFileSync } = fs;
    const source = readFileSync(testInfo.file, 'utf8');
    const forbiddenCall = ['page', 'route'].join('.') + '(';
    expect(source).not.toContain(forbiddenCall);
  });

  test('keeps sidechannel tooltip controls side-local against the running server', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(90_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    await seedRealServerConfig(page, { selectedModel: chatModelSelection });
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });

    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
      timeout: 60_000,
    });
    await assertNoBlockingServerDialog(page);
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();

    const collapseContextSidechannel = page
      .getByTestId('playground-cockpit-left-rail')
      .getByRole('button', { name: 'Collapse context sidechannel' });
    await collapseContextSidechannel.evaluate((element) => {
      const probe = document.createElement('span');
      probe.id = 'playground-cockpit-describedby-probe';
      probe.textContent = 'Extra sidechannel description';
      element.insertAdjacentElement('beforebegin', probe);
      const describedBy = element.getAttribute('aria-describedby');
      element.setAttribute('aria-describedby', [probe.id, describedBy].filter(Boolean).join(' '));
    });
    await assertVisibleTooltipForControl(
      page,
      collapseContextSidechannel,
      'Collapse context sidechannel'
    );
    await collapseContextSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);

    const restoreContextSidechannel = page.getByTestId('playground-cockpit-left-rail-restore');
    await assertVisibleTooltipForControl(
      page,
      restoreContextSidechannel,
      'Restore context sidechannel'
    );
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-context-restore-side-tooltip.png'),
      fullPage: true,
    });
    await restoreContextSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();

    const collapseRuntimeSidechannel = page
      .getByTestId('playground-cockpit-right-rail')
      .getByRole('button', { name: 'Collapse runtime sidechannel' });
    await assertVisibleTooltipForControl(
      page,
      collapseRuntimeSidechannel,
      'Collapse runtime sidechannel'
    );
    await collapseRuntimeSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);

    const restoreRuntimeSidechannel = page.getByTestId('playground-cockpit-right-rail-restore');
    await assertVisibleTooltipForControl(
      page,
      restoreRuntimeSidechannel,
      'Restore runtime sidechannel'
    );
    await restoreRuntimeSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
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
    await assertNoHorizontalOverflow(page);
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
    const modeSummary = page.getByTestId('playground-cockpit-mode-summary');
    await expect(modeSummary).toHaveText('Context and runtime rails visible.');
    const cockpitStatus = page.getByRole('status', { name: 'Chat status' });
    await expect(cockpitStatus).toBeVisible();
    if (health.body?.status === 'degraded') {
      await expect(cockpitStatus).toContainText('Degraded');
      await expect(cockpitStatus).toContainText('Chat remains available.');
    }
    await expect(getDesktopCompositionPreview(page)).toContainText(
      `Scope: ${chatModelSelection.key}`
    );

    await page
      .getByTestId('playground-cockpit-left-rail')
      .getByRole('button', { name: 'Collapse context sidechannel' })
      .click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-left-rail-restore')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-desktop-context-collapsed-side-only.png'),
      fullPage: true,
    });
    await page.getByTestId('playground-cockpit-left-rail-restore').click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context and runtime rails visible.');

    await page
      .getByTestId('playground-cockpit-right-rail')
      .getByRole('button', { name: 'Collapse runtime sidechannel' })
      .click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail-restore')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);
    await page.getByTestId('playground-cockpit-right-rail-restore').click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context and runtime rails visible.');

    await page.getByRole('button', { name: 'Hide context rail' }).click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context rail hidden. Runtime rail visible.');
    await page.getByRole('button', { name: 'Hide runtime rail' }).click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(modeSummary).toHaveText('Cockpit rails hidden. Status remains visible.');
    await expect(cockpitStatus).toBeVisible();
    await page.getByRole('button', { name: 'Show context rail' }).click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(modeSummary).toHaveText('Runtime rail hidden. Context rail visible.');
    await page.getByRole('button', { name: 'Show runtime rail' }).click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context and runtime rails visible.');
    const contextRail = getDesktopContextRail(page);
    const runtimeInspector = getDesktopRuntimeInspector(page);
    await assertRuntimeProviderSummary(runtimeInspector);
    await expect(runtimeInspector.getByText('Provider:model settings')).toBeVisible();
    await expect(runtimeInspector.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    await expect(
      runtimeInspector.getByRole('button', { name: 'Open model settings' })
    ).toBeVisible();
    await expect(
      runtimeInspector.getByRole('button', { name: 'Select character or persona' })
    ).toBeVisible();
    await assertRuntimeMcpRailState(runtimeInspector);
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
    await expect(
      sourceInventory.getByRole('listitem').filter({ hasText: 'Web search' })
    ).toContainText('Active');

    await closeSearchContextIfOpen(page);
    await contextRail.getByRole('button', { name: 'Open Search & Context' }).click();
    await expect(page.getByRole('heading', { name: /Knowledge Search/i })).toBeVisible();

    const mcpToggle = page.getByTestId('mcp-tools-toggle');
    await expect(mcpToggle).toHaveAccessibleName(
      /MCP tools|MCP tools unavailable|MCP tools are offline|MCP tools: None|Not checked yet/i
    );
    if (await mcpToggle.isEnabled()) {
      await mcpToggle.click();
      await expect(page.getByText(/Tool choice/i)).toBeVisible();
      await page.keyboard.press('Escape');
    }

    await page.getByRole('button', { name: /Advanced controls/i }).click();
    await expect(page.getByTestId('composer-options-panel')).toBeVisible();

    const modelSelector = page.getByTestId('model-selector').first();
    await modelSelector.focus();
    await expect(modelSelector).toBeFocused();
    await page.keyboard.press('Enter');
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

    await runtimeInspector.getByRole('button', { name: 'Open model settings' }).click();
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
    await expect(modeSummary).toHaveText(
      'Focus mode hides rails. Chat and composer remain active.'
    );
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await assertCoreComposerControls(page, { composerOnly: true });
    await assertNoHorizontalOverflow(page);
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
    await assertNoHorizontalOverflow(page);
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
    const chatCompletionAttempt = waitForChatCompletionAttempt(page).catch(() => null);
    await page.getByRole('button', { name: /send message/i }).click();
    const chatCompletionResponse = await chatCompletionAttempt;
    await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(smokePrompt);
    if (chatCompletionResponse) {
      await assertProviderQualifiedPayload(page, chatCompletionResponse);
    }
    await assertChatCompletionRenderedOrRecoverable(page, chatCompletionResponse);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-desktop-conversation.png'),
      fullPage: true,
    });
    await expect(cockpitStatus).not.toContainText('0 messages');

    const failingApiHits = apiTracker.hits.filter((hit) => hit.status >= 400);
    expect(failingApiHits).toEqual([]);
    expect(apiTracker.hits.some((hit) => hit.path === '/api/v1/health')).toBe(true);

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
      const sessionBeforeReload = await readPlaygroundSessionSnapshot(page);
      const firstReloadMessagesPromise = sessionBeforeReload?.serverChatId
        ? waitForSuccessfulChatMessagesLoad(page, sessionBeforeReload.serverChatId)
        : null;
      await page.reload({ waitUntil: 'domcontentloaded' });
      if (firstReloadMessagesPromise) {
        await firstReloadMessagesPromise;
      }
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
      await expect(promptTrigger).toBeVisible();
      await expect(
        contextRail.getByRole('region', { name: 'Prompt management' }).getByText(promptName)
      ).toBeVisible();
      const compositionPreview = getDesktopCompositionPreview(page);
      await expect(compositionPreview).toBeVisible();
      await expect(compositionPreview).toContainText(promptName);
      await expect(compositionPreview).toContainText(/Scope: [^:]+:.+/);
      await expect(compositionPreview).toContainText(
        /MCP tools|MCP unavailable|MCP tools managed from composer/
      );
      const sourceList = contextRail.getByRole('list', { name: 'Context sources' });
      const promptSource = sourceList.getByRole('listitem').filter({ hasText: promptName });
      await expect(promptSource).toContainText('Prompt');
      await expect(promptSource).toContainText(promptName);

      await contextRail.getByRole('button', { name: 'Clear prompt', exact: true }).click();
      await expect(promptTrigger).toBeFocused({ timeout: 5_000 });
      await expect(contextRail.getByRole('region', { name: 'Prompt management' })).toContainText(
        'Ready to add prompt'
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
      await expect
        .poll(async () => (await modelSettingsOpenDetail).settingsScope || '')
        .toMatch(/^[^:]+:.+/);
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
      const mcpSettingsTrigger = runtimeInspector.locator('[data-cockpit-mcp-settings-trigger]');
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
    await assertNoHorizontalOverflow(page);

    const mobileDraft = `mobile cockpit draft ${Date.now()}`;
    const expectMobileDraftPreserved = async () => {
      await expect(page.getByTestId('chat-input')).toHaveValue(mobileDraft);
    };
    const expectMobileDraftReachable = async () => {
      await expectMobileDraftPreserved();
      const draftBox = await page.getByTestId('chat-input').boundingBox();
      expect(draftBox?.width ?? 0).toBeGreaterThanOrEqual(180);
    };
    const expectNoMobileBottomControls = async () => {
      await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);
      await expect(page.getByTestId('composer-bottom-bar')).toHaveCount(0);
    };
    const panelControlledByTab = async (tab: Locator) => {
      const tabId = await tab.getAttribute('id');
      const panelId = await tab.getAttribute('aria-controls');
      expect(tabId).toBeTruthy();
      expect(panelId).toBeTruthy();
      const panel = page.locator(`[id="${panelId}"]`);
      await expect(panel).toHaveAttribute('role', 'tabpanel');
      await expect(panel).toHaveAttribute('aria-labelledby', tabId as string);
      return panel;
    };
    await page.getByTestId('chat-input').fill(mobileDraft);
    await expectMobileDraftReachable();
    await expectNoMobileBottomControls();

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
    await expect(mobileRails).toHaveAttribute('data-mobile-panel', 'context');
    await expect(mobileRails.getByTestId('playground-cockpit-mobile-panel-summary')).toHaveText(
      'Context panel active. Composer draft remains available below.'
    );
    await expectMobileDraftPreserved();
    await expectNoMobileBottomControls();
    const initialContextTab = mobileRails.getByRole('tab', { name: 'Context' });
    const initialRuntimeTab = mobileRails.getByRole('tab', { name: 'Runtime' });
    const initialContextPanel = await panelControlledByTab(initialContextTab);
    const initialRuntimePanel = await panelControlledByTab(initialRuntimeTab);
    await expect(initialContextTab).toHaveAttribute('aria-selected', 'true');
    await expect(initialRuntimeTab).toHaveAttribute('aria-selected', 'false');
    await expect(initialContextPanel).toBeVisible();
    await expect(initialRuntimePanel).toBeHidden();
    await assertNoHorizontalOverflow(page);
    await assertNoVerticalOverlap(
      mobileRails,
      page.getByTestId('chat-input'),
      'mobile cockpit context rails should not overlap composer'
    );
    const initialContextPanelBox = await initialContextPanel.boundingBox();
    expect(initialContextPanelBox, 'mobile context panel is measurable').not.toBeNull();
    expect(initialContextPanelBox!.height).toBeLessThanOrEqual(260);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-context.png'),
      fullPage: true,
    });
    await page.getByRole('button', { name: 'Hide context rail' }).click();
    await expect(mobileRails.getByRole('tab', { name: 'Context' })).toHaveCount(0);
    await expect(mobileRails.getByRole('tab', { name: 'Runtime' })).toBeVisible();
    await expect(mobileRails).toHaveAttribute('data-mobile-panel', 'runtime');
    await expect(mobileRails.getByTestId('playground-cockpit-mobile-panel-summary')).toHaveText(
      'Runtime panel active. Composer draft remains available below.'
    );
    await expectMobileDraftPreserved();
    await page.getByRole('button', { name: 'Show context rail' }).click();
    const contextTab = mobileRails.getByRole('tab', { name: 'Context' });
    await expect(contextTab).toBeVisible();
    await contextTab.click();
    await expect(contextTab).toHaveAttribute('aria-selected', 'true');
    await expect(mobileRails).toHaveAttribute('data-mobile-panel', 'context');
    const contextPanelTarget = await panelControlledByTab(contextTab);
    const runtimePanelTarget = await panelControlledByTab(
      mobileRails.getByRole('tab', { name: 'Runtime' })
    );
    await expect(contextPanelTarget).toBeVisible();
    await expect(runtimePanelTarget).toBeHidden();
    await expectMobileDraftPreserved();
    const contextPanel = mobileRails.getByRole('tabpanel', { name: 'Context' });
    await expect(contextPanel.getByRole('button', { name: 'Open Search & Context' })).toBeVisible();
    const mobileWebSearchControl = contextPanel.getByRole('button', {
      name: 'Web search',
      exact: true,
    });
    const initialMobileWebSearchState = await mobileWebSearchControl.getAttribute('aria-pressed');
    await mobileWebSearchControl.click();
    await expect(mobileWebSearchControl).toHaveAttribute(
      'aria-pressed',
      initialMobileWebSearchState === 'true' ? 'false' : 'true'
    );
    await mobileWebSearchControl.click();
    await expect(mobileWebSearchControl).toHaveAttribute(
      'aria-pressed',
      initialMobileWebSearchState || 'false'
    );
    if ((await mobileWebSearchControl.getAttribute('aria-pressed')) === 'true') {
      await mobileWebSearchControl.click();
      await expect(mobileWebSearchControl).toHaveAttribute('aria-pressed', 'false');
    }
    await expectMobileDraftPreserved();
    await contextPanel.getByRole('button', { name: 'Open Search & Context' }).click();
    await expect(page.getByRole('heading', { name: /Knowledge Search/i })).toBeVisible();
    await closeSearchContextIfOpen(page);
    await expectMobileDraftPreserved();
    await contextPanel.getByRole('button', { name: /Select prompt|Select a prompt/i }).click();
    const promptSearch = page.getByPlaceholder('Search prompts...').first();
    await expect(promptSearch).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(promptSearch).toBeHidden();
    await expect(page.getByTestId('chat-input')).toBeVisible();
    await expectMobileDraftReachable();
    const runtimeTab = mobileRails.getByRole('tab', { name: 'Runtime' });
    await runtimeTab.click();
    await expect(runtimeTab).toHaveAttribute('aria-selected', 'true');
    await expect(mobileRails).toHaveAttribute('data-mobile-panel', 'runtime');
    await expect(runtimePanelTarget).toBeVisible();
    await expect(contextPanelTarget).toBeHidden();
    await expect(mobileRails.getByTestId('playground-cockpit-mobile-panel-summary')).toHaveText(
      'Runtime panel active. Composer draft remains available below.'
    );
    await expectMobileDraftPreserved();
    const runtimePanel = mobileRails.getByRole('tabpanel', { name: 'Runtime' });
    await assertRuntimeProviderSummary(runtimePanel);
    await expect(runtimePanel.getByText('Provider:model settings')).toBeVisible();
    await expect(runtimePanel.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    const mobileModelSettingsTrigger = runtimePanel.getByRole('button', {
      name: 'Open model settings',
    });
    await expect(mobileModelSettingsTrigger).toBeVisible();
    await expect(
      runtimePanel.getByRole('button', { name: 'Select character or persona' })
    ).toBeVisible();
    await expect(runtimePanel.getByRole('button', { name: 'Configure MCP tools' })).toBeVisible();
    await assertNoHorizontalOverflow(page);
    await assertNoVerticalOverlap(
      mobileRails,
      page.getByTestId('chat-input'),
      'mobile cockpit runtime rails should not overlap composer'
    );
    const runtimePanelBox = await runtimePanel.boundingBox();
    expect(runtimePanelBox, 'mobile runtime panel is measurable').not.toBeNull();
    expect(runtimePanelBox!.height).toBeLessThanOrEqual(260);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-runtime.png'),
      fullPage: true,
    });
    await mobileModelSettingsTrigger.click();
    const modelSettingsDialog = page.getByRole('dialog', {
      name: 'Current Chat Model Settings',
    });
    await expect(modelSettingsDialog).toBeVisible();
    await expect(modelSettingsDialog.getByText(/API \/ model/i).first()).toBeVisible();
    await modelSettingsDialog.getByRole('button', { name: 'Close' }).click();
    await expect(modelSettingsDialog).toBeHidden();
    await expect(mobileModelSettingsTrigger).toBeFocused({ timeout: 5_000 });
    await expectMobileDraftPreserved();
    await runtimePanel.getByRole('button', { name: 'Select character or persona' }).click();
    await expect(page.getByRole('tab', { name: 'Characters' })).toBeVisible();
    await page.keyboard.press('Escape');
    await expectMobileDraftPreserved();
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
    await expectMobileDraftReachable();
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-active-draft.png'),
      fullPage: true,
    });

    await mobileRails.getByRole('button', { name: 'Return to focus chat' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'focus'
    );
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);
    await assertCoreComposerControls(page, { mobile: true, composerOnly: true });
    await expectMobileDraftReachable();
    await assertNoHorizontalOverflow(page);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-focus.png'),
      fullPage: true,
    });
    await expectNoMobileBottomControls();
    await page.getByRole('button', { name: 'Show cockpit panels' }).click();
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'cockpit'
    );
    await expect(mobileRails).toHaveAttribute('data-mobile-panel', 'runtime');
    await expectMobileDraftPreserved();
    await expect(runtimePanelTarget).toBeVisible();
    await assertNoHorizontalOverflow(page);
  });

  test('sends a real mobile focus conversation against the live server', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(150_000);

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
    await assertCoreComposerControls(page, { mobile: true, composerOnly: true });
    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'focus'
    );
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);
    await expect(page.getByTestId('composer-bottom-bar')).toHaveCount(0);

    const mobileWebSearchControl = page.getByRole('button', {
      name: 'Web search',
      exact: true,
    });
    if (
      (await mobileWebSearchControl.isVisible().catch(() => false)) &&
      (await mobileWebSearchControl.getAttribute('aria-pressed')) === 'true'
    ) {
      await mobileWebSearchControl.click();
      await expect(mobileWebSearchControl).toHaveAttribute('aria-pressed', 'false');
    }

    const mobileSmokePrompt = 'mobile cockpit smoke deterministic prompt';
    await page.getByTestId('chat-input').fill(mobileSmokePrompt);
    await page.getByRole('button', { name: /send message/i }).click();
    await expect(page.locator("article[data-role='user']").last()).toContainText(
      mobileSmokePrompt
    );
    await assertChatCompletionRenderedOrRecoverable(page, null);
    await expect(
      page.locator('.ant-notification-notice').filter({
        hasText: 'Chat now saved on server',
      })
    ).toHaveCount(0);
    await expect(page.getByTestId('chat-input')).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-conversation.png'),
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
      await assertRuntimeAssistantCleared(runtimeInspector);
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
      await expect(
        runtimeInspector.getByRole('button', { name: 'Open Scene Director' })
      ).toBeVisible();
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

      const plainReturnCapture = captureAllApiCalls(page);
      await assistantContextSource.getByRole('button', { name: 'Clear assistant' }).click();
      await assertRuntimeAssistantCleared(runtimeInspector);
      await expect(assistantContextSource).toHaveCount(0);
      await expect(composerAssistant).toHaveAttribute('aria-label', /Select character or persona/i);

      const plainPrompt = `Plain return after character clear ${Date.now()}`;
      const plainCompletionAttempt = waitForChatCompletionAttempt(page, 90_000).catch(() => null);
      await page.getByTestId('chat-input').fill(plainPrompt);
      await page.getByRole('button', { name: /send message/i }).click();
      await plainCompletionAttempt;
      const stopStreaming = page.getByRole('button', { name: /stop streaming response/i });
      if (await stopStreaming.isVisible({ timeout: 10_000 }).catch(() => false)) {
        await stopStreaming.click().catch(() => undefined);
      }
      await page.waitForTimeout(1_500);
      const plainReturnCalls = await plainReturnCapture.stop();
      const plainCreateCall = findChatCreateCall(plainReturnCalls);
      expect(plainCreateCall).toBeDefined();
      expect(plainCreateCall?.requestBody).toEqual(
        expect.objectContaining({
          source: 'webui-chat',
        })
      );
      const plainCreatePayload =
        plainCreateCall?.requestBody && typeof plainCreateCall.requestBody === 'object'
          ? (plainCreateCall.requestBody as Record<string, unknown>)
          : {};
      expect(plainCreatePayload).not.toHaveProperty('character_id');
      expect(plainCreatePayload).not.toHaveProperty('assistant_kind');
      expect(plainCreatePayload).not.toHaveProperty('assistant_id');
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

    let selectedPersona: { id?: unknown; name?: unknown; version?: unknown } | null = null;
    let cleanupPersonaId: string | null = null;
    let cleanupPersonaVersion: number | null = null;

    try {
      selectedPersona = await createDisposablePersona(request, personaId, personaName);
      cleanupPersonaId = String(selectedPersona.id);
      cleanupPersonaVersion = Number(selectedPersona.version ?? 1);
    } catch (error) {
      testInfo.annotations.push({
        type: 'blocker',
        description: `Could not create disposable persona via real server: ${
          error instanceof Error ? error.message : String(error)
        }`,
      });
      const listed = await apiGet<any>(request, '/api/v1/persona/profiles?active_only=true');
      const personas = extractPersonaProfiles(listed.body);
      selectedPersona =
        personas.find((persona) => persona?.is_active !== false) ?? personas[0] ?? null;
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
      await expect(
        page.getByText(/No personas available\.|Could not load personas\./)
      ).toBeVisible();
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
      await assertRuntimeAssistantCleared(runtimeInspector);

      await selectAssistantFromRuntimeRail(page, {
        tab: 'Personas',
        assistantName: String(selectedPersona.name),
      });

      const assistantTrigger = runtimeInspector.locator('[data-cockpit-assistant-select-trigger]');
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
      await assertRuntimeAssistantCleared(runtimeInspector);
      await expect(personaContextSource).toHaveCount(0);
      await expect(composerAssistant).toHaveAttribute('aria-label', /Select character or persona/i);
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

  test('starts a tracked character chat from the runtime rail and restores tracked mode after reload', async ({
    page,
    request,
  }) => {
    test.setTimeout(150_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    const timestamp = Date.now();
    let character: DisposableCharacter | null = null;

    try {
      character = await createDisposableCharacter(
        request,
        `Tracked Character ${timestamp}`
      );

      await openDesktopChatCockpit(page, chatModelSelection);

      const trackedStartCapture = captureAllApiCalls(page);
      await selectAssistantFromRuntimeRail(page, {
        tab: 'Characters',
        assistantName: character.name,
      });
      const trackedStartCalls = await trackedStartCapture.stop();
      await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(
        character.firstMessage
      );
      const createCall = findChatCreateCall(trackedStartCalls);
      expect(createCall).toBeDefined();
      const trackedCharacterId = (createCall?.requestBody as Record<string, unknown> | null)
        ?.character_id;
      expect(String(trackedCharacterId ?? "")).toBe(character.id);

      const chatId =
        extractCreatedChatIdFromCreateCall(createCall) ??
        (await waitForPersistedServerChatId(page));

      const chatDetails = await getChatDetails(request, chatId);
      expect(chatDetails.status).toBe(200);
      expect(chatDetails.body).toMatchObject({
        id: chatId,
        character_id: Number(character.id),
        assistant_kind: 'character',
      });

      await page.reload({ waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      await assertCoreComposerControls(page);

      const runtimeInspector = getDesktopRuntimeInspector(page);
      await expect(runtimeInspector).toContainText('Character selected', {
        timeout: 60_000,
      });
      await expect(runtimeInspector).toContainText(character.name);
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toBeVisible();
      await expect(page.getByTestId('character-control-rail')).toHaveCount(0);
    } finally {
      await cleanupDisposableCharacter(request, character);
    }
  });

  test('starts a tracked persona chat from the runtime rail and restores tracked mode after reload', async ({
    page,
    request,
  }) => {
    test.setTimeout(150_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    const timestamp = Date.now();
    let persona: DisposablePersona | null = null;

    try {
      persona = await createDisposablePersona(
        request,
        `tracked_persona_${timestamp}`,
        `Tracked Persona ${timestamp}`
      );

      await openDesktopChatCockpit(page, chatModelSelection);

      await selectAssistantFromRuntimeRail(page, {
        tab: 'Personas',
        assistantName: persona.name,
      });

      const prompt = `Tracked persona proof ${timestamp}`;
      const turn = await sendChatTurnAndCapture(page, prompt, {
        stopStreamingAfterRequest: true,
      });
      const createCall = findChatCreateCall(turn.calls);

      expect(createCall).toBeDefined();
      expect(createCall?.requestBody).toMatchObject({
        assistant_kind: 'persona',
        assistant_id: persona.id,
      });

      const chatId =
        extractCreatedChatIdFromCreateCall(createCall) ??
        extractConversationChatIdFromResponse(turn.response) ??
        extractConversationChatIdFromCall(findConversationTurnCall(turn.calls)) ??
        (await waitForPersistedServerChatId(page));

      const chatDetails = await getChatDetails(request, chatId);
      expect(chatDetails.status).toBe(200);
      expect(chatDetails.body).toMatchObject({
        id: chatId,
        assistant_kind: 'persona',
        assistant_id: persona.id,
      });

      await page.reload({ waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      await assertCoreComposerControls(page);

      const runtimeInspector = getDesktopRuntimeInspector(page);
      await expect(runtimeInspector).toContainText('Persona selected', {
        timeout: 60_000,
      });
      await expect(runtimeInspector).toContainText(persona.name);
      await expect(runtimeInspector.getByRole('button', { name: 'Clear assistant' })).toBeVisible();
      await expect(page.getByTestId('character-control-rail')).toHaveCount(0);
    } finally {
      await cleanupDisposablePersona(request, persona);
    }
  });

  test('proves model provider confidence through a real cockpit selection and conversation', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    const apiTracker = trackRealApiHits(page);
    await seedRealServerConfig(page);
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });

    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
      timeout: 60_000,
    });
    await assertNoBlockingServerDialog(page);

    const modelKey = await selectConfiguredCockpitModel(page, chatModelSelection);
    const runtimeInspector = getDesktopRuntimeInspector(page);
    const compositionPreview = getDesktopCompositionPreview(page);

    await expect(compositionPreview).toContainText(`Scope: ${modelKey}`);
    await expect(compositionPreview).not.toContainText(`Scope: ${chatModelSelection.key}`);
    await expect(runtimeInspector.getByText('Provider route')).toBeVisible();
    await expect(runtimeInspector.getByText(modelKey, { exact: true })).toBeVisible();

    const prompt = `Cockpit model provider proof ${Date.now()}: answer in one short sentence.`;
    await page.getByTestId('chat-input').fill(prompt);
    const chatCompletionAttempt = waitForChatCompletionAttempt(page).catch(() => null);
    await page.getByRole('button', { name: /send message/i }).click();
    const chatCompletionResponse = await chatCompletionAttempt;

    await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(prompt);
    expect(chatCompletionResponse).toBeTruthy();
    if (chatCompletionResponse) {
      await assertProviderQualifiedPayload(page, chatCompletionResponse);
      expect(chatCompletionResponse.request().postDataJSON()).toMatchObject({
        model: chatModelSelection.model,
      });
    }
    await assertChatCompletionRenderedOrRecoverable(page, chatCompletionResponse);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-model-provider-conversation.png'),
      fullPage: true,
    });

    const failingApiHits = apiTracker.hits.filter((hit) => hit.status >= 400);
    expect(failingApiHits).toEqual([]);
    apiTracker.dispose();
  });

  test('captures streaming stop and regenerate controls through the real cockpit', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(150_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    await openDesktopChatCockpit(page, chatModelSelection);

    const contextRail = getDesktopContextRail(page);
    const webSearchControl = contextRail.getByRole('button', {
      name: 'Web search',
      exact: true,
    });
    if ((await webSearchControl.getAttribute('aria-pressed')) === 'true') {
      await webSearchControl.click();
      await expect(webSearchControl).toHaveAttribute('aria-pressed', 'false');
    }

    const runtimeInspector = getDesktopRuntimeInspector(page);
    const regenerateControl = runtimeInspector.getByRole('button', {
      name: 'Regenerate last response',
    });
    await expect(regenerateControl).toBeDisabled();

    const prompt = `Cockpit streaming controls proof ${Date.now()}: reply with a concise numbered list.`;
    const completionAttempt = waitForChatCompletionAttempt(page, 90_000).catch(() => null);

    await page.getByTestId('chat-input').fill(prompt);
    await page.getByRole('button', { name: /send message/i }).click();
    const completionResponse = await completionAttempt;
    expect(completionResponse).toBeTruthy();

    const statusStripStop = page
      .getByRole('status', { name: 'Chat status' })
      .getByRole('button', { name: 'Stop generation' });
    const runtimeStop = runtimeInspector.getByRole('button', { name: 'Stop generation' });
    const messageStop = page
      .getByRole('button', { name: /Stop streaming response/i })
      .first();
    const stopCandidates = [
      { label: 'status strip stop', locator: statusStripStop },
      { label: 'runtime rail stop', locator: runtimeStop, requireEnabled: true },
      { label: 'message stop', locator: messageStop },
    ];
    const stopControl = await waitForFirstAvailableControl(stopCandidates, 15_000);

    if (!stopControl) {
      const note =
        'No streaming stop control became observable before the provider response completed.';
      testInfo.annotations.push({
        type: expectStreamingControlEvidence ? 'failure-context' : 'streaming-control',
        description: note,
      });
      if (expectStreamingControlEvidence) {
        throw new Error(note);
      }
      await assertChatCompletionRenderedOrRecoverable(page, completionResponse);
    } else {
      const clickedStopControl =
        (await clickFirstAvailableControl([stopControl, ...stopCandidates])) ?? null;
      if (!clickedStopControl && expectStreamingControlEvidence) {
        throw new Error(
          `Streaming stop control "${stopControl.label}" appeared but disappeared before it could be clicked.`
        );
      }
      await page.screenshot({
        path: testInfo.outputPath('chat-cockpit-streaming-stop-clicked.png'),
        fullPage: true,
      });
      await expect(runtimeStop).toBeDisabled({ timeout: 30_000 });
      await expect(statusStripStop).toHaveCount(0);
    }

    if (completionResponse) {
      await assertProviderQualifiedPayload(page, completionResponse);
    }

    await expect(regenerateControl).toBeEnabled({ timeout: 30_000 });
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-regenerate-ready.png'),
      fullPage: true,
    });

    const regenerateAttempt = waitForChatCompletionAttempt(page, 90_000).catch(() => null);
    await regenerateControl.click();
    const regenerateResponse = await regenerateAttempt;
    expect(regenerateResponse).toBeTruthy();
    await assertChatCompletionRenderedOrRecoverable(page, regenerateResponse);
    if (regenerateResponse) {
      await assertProviderQualifiedPayload(page, regenerateResponse);
    }
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-regenerated-response.png'),
      fullPage: true,
    });
  });
});
