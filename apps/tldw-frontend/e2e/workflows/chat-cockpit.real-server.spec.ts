import AxeBuilder from "@axe-core/playwright";
import { readStore } from "../../../extension/tests/e2e/utils/history-selection";
import type { HistoryBookmark } from "../../../packages/ui/src/db/dexie/types";
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
  type Request,
  type Response,
} from '@playwright/test';
import { captureAllApiCalls, type CapturedApiCall } from '../utils/api-assertions';
import { waitForStreamComplete } from '../utils/journey-helpers';
import { parseRoleplayResumeAppendRequest, parseRoleplayResumeAppendReceipt, parseRoleplayResumeCompletionRequest, parseRoleplayResumeCompletionReceipt } from '../../../packages/ui/src/services/tldw/roleplay-resume';

const serverUrl = (
  process.env.TLDW_E2E_SERVER_URL ||
  process.env.TLDW_SERVER_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

const apiKey =
  process.env.TLDW_E2E_API_KEY || process.env.TLDW_API_KEY || process.env.SINGLE_USER_API_KEY || '';
const expectStreamingControlEvidence = process.env.TLDW_E2E_EXPECT_STREAMING_CONTROLS === 'true';

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

type DeterministicChatReadiness = RealChatModelSelection & {
  providerId: 'custom-openai-api';
  modelId: string;
  reason?: string;
};

type CockpitLayoutSeed = {
  mode?: 'cockpit' | 'focus';
  contextRailVisible?: boolean;
  runtimeRailVisible?: boolean;
  mobilePanel?: 'context' | 'runtime' | null;
};

type DesktopCockpitRail = 'left' | 'right';

const desktopCockpitRailConfig: Record<
  DesktopCockpitRail,
  { attr: 'data-left-rail' | 'data-right-rail'; railId: string; restoreId: string }
> = {
  left: {
    attr: 'data-left-rail',
    railId: 'playground-cockpit-left-rail',
    restoreId: 'playground-cockpit-left-rail-restore',
  },
  right: {
    attr: 'data-right-rail',
    railId: 'playground-cockpit-right-rail',
    restoreId: 'playground-cockpit-right-rail-restore',
  },
};

const mobileCockpitPanelConfig = {
  context: {
    buttonId: 'playground-cockpit-mobile-context-tab',
    panelId: 'playground-cockpit-mobile-context-panel',
  },
  runtime: {
    buttonId: 'playground-cockpit-mobile-runtime-tab',
    panelId: 'playground-cockpit-mobile-runtime-panel',
  },
} as const;

const restoreDesktopCockpitRail = async (page: Page, rail: DesktopCockpitRail) => {
  const config = desktopCockpitRailConfig[rail];
  const shell = page
    .locator('[data-testid="playground-cockpit-shell"][data-mode="cockpit"]')
    .first();
  const railLocator = page.locator(`[data-testid="${config.railId}"]:visible`).first();
  await expect(shell).toHaveAttribute('data-mode', 'cockpit', { timeout: 10_000 });
  if ((await shell.getAttribute(config.attr)) === 'visible') {
    await expect(railLocator).toBeVisible({ timeout: 10_000 });
    return;
  }

  await expect
    .poll(
      async () =>
        page.evaluate(
          async ({ attr, restoreId }) => {
            const isVisible = (element: Element) => {
              const box = element.getBoundingClientRect();
              const style = window.getComputedStyle(element);
              return (
                box.width > 0 &&
                box.height > 0 &&
                style.display !== 'none' &&
                style.visibility !== 'hidden' &&
                !element.closest('[hidden],[aria-hidden="true"]')
              );
            };
            const shell = Array.from(
              document.querySelectorAll<HTMLElement>(
                '[data-testid="playground-cockpit-shell"][data-mode="cockpit"]'
              )
            ).find(isVisible);
            if (!shell) return 'missing shell';
            if (shell.getAttribute(attr) === 'visible') return 'visible';

            const restoreSelector = `[data-testid="${restoreId}"]`;
            const restoreControl =
              Array.from(shell.querySelectorAll<HTMLElement>(restoreSelector)).find(isVisible) ??
              Array.from(document.querySelectorAll<HTMLElement>(restoreSelector)).find(isVisible);
            if (!restoreControl) return 'missing visible restore control';

            restoreControl.click();
            await new Promise<void>((resolve) =>
              requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
            );
            return shell.getAttribute(attr) ?? 'missing rail state';
          },
          { attr: config.attr, restoreId: config.restoreId }
        ),
      { message: `restore ${rail} cockpit rail`, timeout: 10_000 }
    )
    .toBe('visible');
  await expect(shell).toHaveAttribute(config.attr, 'visible', { timeout: 10_000 });
  await expect(railLocator).toBeVisible({ timeout: 10_000 });
};

const waitForMobileCockpitPanel = async (page: Page, panel: 'context' | 'runtime') => {
  const config = mobileCockpitPanelConfig[panel];
  await expect
    .poll(
      async () =>
        page.evaluate(
          ({ panel, panelId, buttonId }) => {
            const shell = document.querySelector('[data-testid="playground-cockpit-shell"]');
            if (shell?.getAttribute('data-mode') !== 'cockpit') {
              return `mode:${shell?.getAttribute('data-mode') ?? 'missing'}`;
            }

            const rails = Array.from(
              document.querySelectorAll<HTMLElement>('[data-testid="playground-cockpit-mobile-rails"]')
            ).find((candidate) => {
              const box = candidate.getBoundingClientRect();
              return (
                candidate.getAttribute('data-mobile-panel') === panel &&
                box.width > 0 &&
                box.height > 0
              );
            });
            if (!rails) return 'missing rails';
            const button = rails.querySelector<HTMLButtonElement>(`button#${buttonId}`);
            const panelNode = rails.querySelector<HTMLElement>(`section#${panelId}`);
            if (!button || !panelNode) return 'missing disclosure pair';
            if (button.type !== 'button') return 'wrong button type';
            if (button.getAttribute('aria-controls') !== panelId) return 'wrong button target';
            if (button.getAttribute('aria-expanded') !== 'true') return 'collapsed button';
            if (panelNode.getAttribute('aria-labelledby') !== buttonId) return 'wrong panel label';
            if (panelNode.hidden || panelNode.getAttribute('aria-hidden') !== 'false') {
              return 'hidden panel';
            }
            const panelBox = panelNode.getBoundingClientRect();
            return panelBox.width > 0 && panelBox.height > 0 ? 'ready' : 'unmeasurable panel';
          },
          { panel, panelId: config.panelId, buttonId: config.buttonId }
        ),
      { message: `mobile cockpit ${panel} panel ready`, timeout: 15_000 }
    )
    .toBe('ready');
  const rails = page
    .locator(
      `[data-testid="playground-cockpit-mobile-rails"]:visible:has(#${config.panelId})`
    )
    .first();
  await expect(rails).toBeVisible({ timeout: 10_000 });
  await expect(rails).toHaveAttribute('data-mobile-panel', panel);
  const button = rails.getByRole('button', {
    name: panel === 'context' ? 'Context' : 'Chat settings',
    exact: true,
  });
  await expect(button).toHaveAttribute('id', config.buttonId);
  await expect(button).toHaveAttribute('type', 'button');
  await expect(button).toHaveAttribute('aria-controls', config.panelId);
  await expect(button).toHaveAttribute('aria-expanded', 'true');
  const region = rails.getByRole('region', {
    name: panel === 'context' ? 'Context' : 'Chat settings',
    exact: true,
  });
  await expect(region).toHaveAttribute('id', config.panelId);
  await expect(region).toHaveAttribute('aria-labelledby', config.buttonId);
  await expect(region).toBeVisible();
  return rails;
};

const switchChatLayoutMode = async (page: Page, targetMode: 'cockpit' | 'focus') => {
  const shell = page.getByTestId('playground-cockpit-shell');
  if ((await shell.getAttribute('data-mode')) !== targetMode) {
    const triggerName = targetMode === 'cockpit' ? 'Exit focus' : 'Enter focus chat';
    try {
      await page
        .getByRole('button', { name: triggerName, exact: true })
        .click({ timeout: 10_000 });
    } catch (error) {
      await expect(shell).toHaveAttribute('data-mode', targetMode, { timeout: 5_000 }).catch(() => {
        throw error;
      });
    }
  }

  await expect(shell).toHaveAttribute('data-mode', targetMode, { timeout: 10_000 });
};

const waitForPlaygroundDraftSaved = async (page: Page) => {
  await expect(page.getByRole('status').filter({ hasText: 'Draft saved' })).toBeVisible({
    timeout: 5_000,
  });
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

const desktopCockpitLayoutSeed: CockpitLayoutSeed = {
  mode: 'cockpit',
  contextRailVisible: true,
  runtimeRailVisible: true,
  mobilePanel: 'context',
};

const mobileFocusWithCockpitPanelsSeed: CockpitLayoutSeed = {
  mode: 'focus',
  contextRailVisible: true,
  runtimeRailVisible: true,
  mobilePanel: 'context',
};

const mobileFocusOnlyLayoutSeed: CockpitLayoutSeed = {
  mode: 'focus',
  contextRailVisible: false,
  runtimeRailVisible: false,
  mobilePanel: 'context',
};

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
  if (normalized === 'llama.cpp' || normalized === 'llama') return 'llamacpp';
  if (normalized === 'local-llm') return 'local';
  if (normalized === 'custom_openai_api') return 'custom-openai-api';
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

const buildConfiguredChatModelSelection = (payload: any, metadata: unknown): RealChatModelSelection => {
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

  const configuredProvider = normalizeCockpitProviderKey(providerName);
  const rows = Array.isArray(metadata) ? metadata : Array.isArray((metadata as any)?.models) ? (metadata as any).models : [];
  // Match normalizeTldwModels ID precedence; Playwright cannot import the UI's ESM package in this CJS fixture.
  const catalog = rows.map((entry: any) => {
    const text = (value: unknown) => typeof value === 'string' ? value.trim() : '';
    const rawModel = text(entry?.model) || text(entry?.model_id);
    const rawName = text(entry?.name), rawId = text(entry?.id);
    const id = rawModel || (rawName && !/\s/.test(rawName) && /[/:._-]/.test(rawName) ? rawName : '') || rawId || rawName;
    return { id, provider: text(entry?.provider), type: entry?.type };
  });
  const matches = catalog.filter(entry =>
    normalizeCockpitProviderKey(entry.provider) === configuredProvider &&
    normalizeConfiguredChatModelId(providerName, entry.id) === model &&
    (!entry.type || entry.type === 'chat')
  );
  if (matches.length !== 1) {
    throw new Error(`Configured chat model ${providerName}/${model} needs one unique model catalog entry`);
  }
  return {
    provider: providerName,
    model,
    key: `${configuredProvider}:${matches[0].id}`,
  };
};

const getConfiguredChatModelSelection = async (
  request: APIRequestContext
): Promise<RealChatModelSelection> => {
  const [providers, metadata] = await Promise.all([
    apiGet<unknown>(request, '/api/v1/llm/providers'),
    apiGet<unknown>(request, '/api/v1/llm/models/metadata'),
  ]);
  expect(providers.status).toBe(200);
  expect(metadata.status).toBe(200);
  expect(extractConfiguredProviders(providers.body).length).toBeGreaterThan(0);
  return buildConfiguredChatModelSelection(providers.body, metadata.body);
};

const getDeterministicChatReadiness = async (
  request: APIRequestContext
): Promise<DeterministicChatReadiness> => {
  const providers = await apiGet<unknown>(request, '/api/v1/llm/providers');
  expect(providers.status).toBe(200);

  const provider = extractConfiguredProviders(providers.body).find(
    (candidate) =>
      normalizeCockpitProviderKey(String(candidate?.name || '')) === 'custom-openai-api' &&
      Array.isArray(candidate?.models) &&
      candidate.models.length > 0
  );

  if (!provider) {
    throw new Error(
      'Deterministic live Chat UAT requires a configured custom-openai-api provider with a chat model'
    );
  }

  const providerName = String(provider.name || '').trim();
  const rawModel =
    typeof provider.default_model === 'string' && provider.default_model.trim().length > 0
      ? provider.default_model.trim()
      : String(provider.models[0] || '').trim();
  const modelId = normalizeConfiguredChatModelId(providerName, rawModel);

  if (!modelId) {
    throw new Error('Configured custom-openai-api provider has no usable chat model');
  }

  return {
    provider: providerName,
    providerId: 'custom-openai-api',
    model: modelId,
    modelId,
    key: `custom-openai-api:${modelId}`,
  };
};

const seedRealServerConfig = async (
  page: Page,
  options: {
    selectedModel?: RealChatModelSelection;
    persistedServerChatId?: string | null;
    cockpitLayout?: CockpitLayoutSeed | null;
    uiMode?: 'casual' | 'pro';
  } = {}
) => {
  await page.addInitScript(
    ({
      configuredServerUrl,
      configuredApiKey,
      configuredSelectedModel,
      configuredPersistedServerChatId,
      configuredCockpitLayout,
      configuredUiMode,
    }) => {
      const fnv1a36 = (value: string) => {
        let hash = 2166136261;
        for (let i = 0; i < value.length; i += 1) {
          hash ^= value.charCodeAt(i);
          hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
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
        authSource: 'manual',
        credentialSource: 'manual',
        apiKeyPersistence: 'device',
        apiKeyServerOrigin: new URL(configuredServerUrl).origin,
        apiKey: configuredApiKey,
        requestTimeoutMs: 60_000,
        chatRequestTimeoutMs: 120_000,
        chatStartupTimeoutMs: 60_000,
        chatStreamIdleTimeoutMs: 120_000,
      };

      const writeJsonStorageValue = (key: string, value: unknown) => {
        localStorage.setItem(key, JSON.stringify(value));
      };

      // lgtm[js/clear-text-storage-of-sensitive-data]: test-only browser auth seed for real-server E2E.
      localStorage.setItem('tldwConfig', JSON.stringify(config));
      localStorage.setItem('serverUrl', configuredServerUrl);
      localStorage.setItem('tldwServerUrl', configuredServerUrl);
      localStorage.setItem('tldw-api-host', configuredServerUrl);
      localStorage.setItem('authMode', 'single-user');
      // lgtm[js/clear-text-storage-of-sensitive-data]: test-only legacy auth compatibility key.
      localStorage.setItem('apiKey', configuredApiKey);
      localStorage.setItem('isMigrated', 'true');
      localStorage.setItem('__tldw_first_run_complete', 'true');
      localStorage.setItem('assistant_setup_dismissed', 'true');
      localStorage.setItem('playgroundComposerOptionsExpanded', 'true');
      writeJsonStorageValue('tldw-ui-mode', { state: { mode: configuredUiMode }, version: 1 });
      if (configuredCockpitLayout) {
        if (configuredCockpitLayout.mode) {
          writeJsonStorageValue('playgroundChatLayoutMode', configuredCockpitLayout.mode);
        }
        if (typeof configuredCockpitLayout.contextRailVisible === 'boolean') {
          writeJsonStorageValue(
            'playgroundChatContextRailVisible',
            configuredCockpitLayout.contextRailVisible
          );
        }
        if (typeof configuredCockpitLayout.runtimeRailVisible === 'boolean') {
          writeJsonStorageValue(
            'playgroundChatRuntimeRailVisible',
            configuredCockpitLayout.runtimeRailVisible
          );
        }
        if ('mobilePanel' in configuredCockpitLayout) {
          writeJsonStorageValue(
            'playgroundChatMobileCockpitPanel',
            configuredCockpitLayout.mobilePanel
          );
        }
      }
      if (configuredPersistedServerChatId) {
        // lgtm[js/clear-text-storage-of-sensitive-data]: synthetic E2E persisted chat fixture.
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
      configuredCockpitLayout: options.cockpitLayout ?? null,
      configuredUiMode: options.uiMode ?? 'casual',
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
  const watchedPaths = ['/health', '/api/v1/llm/providers', '/api/v1/llm/models/metadata'];

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
  const mode = page.locator('[data-ui-mode]').first();
  await expect(mode).toHaveAttribute('data-ui-mode', /casual|pro/);
  const casual = page.getByTestId('composer-casual-advanced-chip');
  if ((await mode.getAttribute('data-ui-mode')) === 'casual') {
    await expect(casual).toBeEnabled();
    if ((await casual.getAttribute('aria-expanded')) !== 'true') await casual.click();
    await expect(casual).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByTestId('composer-casual-advanced-controls-row')).toBeVisible();
    return;
  }
  const toggle = page.getByTestId('composer-options-toggle');
  await expect(toggle).toBeVisible();
  const label = await toggle.getAttribute('aria-label');
  if (/show composer options/i.test(label || '')) await toggle.click();
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
  const mode = page.locator('[data-ui-mode]').first();
  await expect(mode).toHaveAttribute('data-ui-mode', /casual|pro/);
  const casual = page.getByTestId('composer-casual-advanced-chip');
  const isCasual = (await mode.getAttribute('data-ui-mode')) === 'casual';
  await expect(isCasual ? casual : page.getByTestId('composer-options-toggle')).toBeVisible();
  await expect(page.getByTestId('chat-input')).toBeVisible();
  await expect(page.getByRole('button', { name: /send message/i })).toBeVisible();
  if (options.composerOnly) return;
  await ensureComposerOptionsVisible(page);
  await expect(page.getByTestId('model-selector').first()).toBeVisible();
  await expect(page.getByTestId('attachment-button')).toBeVisible();
  if (isCasual) {
    const customize = page.getByTestId('composer-casual-advanced-controls-row');
    const research = customize.getByRole('group', { name: 'Research & context' });
    const character = customize.getByRole('group', { name: 'Character & your role' });
    const response = customize.getByRole('group', { name: 'Response settings' });
    await expect(research).toBeVisible();
    await expect(character).toBeVisible();
    await expect(response).toBeVisible();
    await expect(character.getByTestId('character-select')).toBeVisible();
    await expect(response.getByTestId('chat-prompt-select')).toBeVisible();
    await expect(research.getByTestId('tools-button')).toBeVisible();
    await expect(research.getByTestId('mcp-tools-toggle')).toBeVisible();
    if (options.mobile)
      await expect(
        research.getByRole('button', { name: /Health & diagnostics|Server:/i })
      ).toBeVisible();
    return;
  }
  await expect(page.getByTestId('chat-prompt-select')).toBeVisible();
  await expect(page.getByTestId('character-select')).toBeVisible();
  await expect(page.getByTestId('tools-button')).toBeVisible();
  if (options.mobile) {
    await expect(page.getByRole('button', { name: /More options/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Health & diagnostics|Server:/i })).toBeVisible();
    return;
  }
  await expect(page.getByTestId('mcp-tools-toggle')).toBeVisible();
  await expect(page.getByRole('button', { name: /Advanced controls/i })).toBeVisible();
};

const fillEditableComposer = async (page: Page, value: string) => {
  const input = page.getByTestId('chat-input');
  await expect(input).toBeEditable({ timeout: 15_000 });
  await input.fill(value);
  await expect(input).toHaveValue(value);
};

const ensureDesktopCockpitRailsVisible = async (page: Page) => {
  await restoreDesktopCockpitRail(page, 'left');
  await restoreDesktopCockpitRail(page, 'right');
};

const assertRuntimeProviderSummary = async (runtimeInspector: Locator) => {
  const runtimeState = runtimeInspector.getByRole('region', { name: 'Runtime state' });
  await expect(runtimeState.getByRole('heading', { name: 'Chat settings' })).toBeVisible();
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

type MobilePanelMeasurement =
  | { status: 'ok'; panelBottom: number; panelHeight: number; composerTop: number }
  | { status: 'error'; message: string };

const measureMobilePanelComposer = async (
  page: Page,
  panel: 'context' | 'runtime'
): Promise<MobilePanelMeasurement> => {
  const config = mobileCockpitPanelConfig[panel];
  return page.evaluate(
    ({ panel, panelId }) => {
      const isVisible = (element: Element) => {
        const box = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        return (
          box.width > 0 &&
          box.height > 0 &&
          style.display !== 'none' &&
          style.visibility !== 'hidden' &&
          !element.closest('[hidden],[aria-hidden="true"]')
        );
      };
      const rails = Array.from(
        document.querySelectorAll<HTMLElement>('[data-testid="playground-cockpit-mobile-rails"]')
      ).find(
        (candidate) =>
          candidate.getAttribute('data-mobile-panel') === panel && isVisible(candidate)
      );
      if (!rails) return { status: 'error' as const, message: 'missing visible rails' };
      const panelNode = rails.querySelector<HTMLElement>(`#${panelId}`);
      if (!panelNode || !isVisible(panelNode)) {
        return { status: 'error' as const, message: 'first element is not measurable' };
      }
      const composer = document.querySelector<HTMLElement>('[data-testid="chat-input"]');
      if (!composer || !isVisible(composer)) {
        return { status: 'error' as const, message: 'second element is not measurable' };
      }
      const panelBox = panelNode.getBoundingClientRect();
      const composerBox = composer.getBoundingClientRect();
      return {
        status: 'ok' as const,
        panelBottom: panelBox.y + panelBox.height,
        panelHeight: panelBox.height,
        composerTop: composerBox.y,
      };
    },
    { panel, panelId: config.panelId }
  );
};

const assertNoMobilePanelComposerOverlap = async (
  page: Page,
  panel: 'context' | 'runtime',
  label: string
) => {
  await expect
    .poll(
      async () => {
        const measurement = await measureMobilePanelComposer(page, panel);
        if (measurement.status === 'error') return measurement.message;
        return measurement.panelBottom <= measurement.composerTop + 1
          ? 'ok'
          : `overlap: first bottom ${measurement.panelBottom}, second top ${measurement.composerTop}`;
      },
      { message: label, timeout: 10_000 }
    )
    .toBe('ok');
};

const assertMobileComposerDraftVisible = async (page: Page) => {
  await expect.poll(async () => {
    const input = await page.getByTestId('chat-input').boundingBox();
    const viewport = await page.locator('[data-testid="playground-chat-composer-region"], [data-testid="playground-chat-composer-dock"]').boundingBox();
    return Boolean(input && viewport && input.y >= viewport.y - 1 && input.y + input.height <= viewport.y + viewport.height + 1);
  }, { message: 'Mobile draft should be visible inside its composer scroll area' }).toBe(true);
};

const MAX_MOBILE_PANEL_HEIGHT_PX = 260;

const assertMobilePanelMeasuredHeight = async (
  page: Page,
  panel: 'context' | 'runtime',
  label: string
) => {
  await expect
    .poll(
      async () => {
        const measurement = await measureMobilePanelComposer(page, panel);
        if (measurement.status === 'error') return measurement.message;
        return measurement.panelHeight <= MAX_MOBILE_PANEL_HEIGHT_PX
          ? 'ok'
          : `too tall: ${measurement.panelHeight}`;
      },
      { message: label, timeout: 10_000 }
    )
    .toBe('ok');
};

const assertHealthResponse = (health: { status: number; body: any }) => {
  expect([200, 206]).toContain(health.status);
  expect(['ok', 'healthy', 'degraded']).toContain(health.body?.status);
};

const isChatCompletionRequest = (page: Page, request: Request) => {
  const backendOrigin = new URL(serverUrl).origin;
  const pageOrigin = new URL(page.url()).origin;
  const url = new URL(request.url());
  if (url.origin !== backendOrigin && url.origin !== pageOrigin) return false;
  return (
    (url.pathname === '/api/v1/chat/completions' ||
      /^\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(url.pathname) ||
      /^\/api\/v1\/chats\/[^/]+\/completions$/.test(url.pathname)) &&
    request.method() === 'POST'
  );
};

const waitForChatCompletionAttempt = (page: Page, timeout = 15_000) => {
  return page.waitForResponse(
    (response) => isChatCompletionRequest(page, response.request()),
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

const assertProviderQualifiedPayload = async (page: Page, responseOrRequest: Response | Request) => {
  const request = 'request' in responseOrRequest ? responseOrRequest.request() : responseOrRequest;
  const payload = request.postDataJSON() as any;
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
    cockpitLayout: desktopCockpitLayoutSeed,
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
  const created = await apiPostWithRetry<any>(
    request,
    '/api/v1/characters',
    {
      name,
      system_prompt: `You are ${name}, a concise overlay test assistant.`,
      first_message: firstMessage,
      creator: 'tldw e2e',
      tags: ['e2e', 'overlay-rail'],
    },
    {
      attempts: 6,
      retryDelayMs: 2_000,
    }
  );

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
  const created = await apiPostWithRetry<any>(
    request,
    '/api/v1/persona/profiles',
    {
      id: personaId,
      name,
      mode: 'session_scoped',
      system_prompt: `You are ${name}, a concise overlay test persona.`,
      is_active: true,
      use_persona_state_context_default: true,
    },
    {
      attempts: 6,
      retryDelayMs: 2_000,
    }
  );

  expect(created.status).toBe(201);
  expect(created.body?.id).toBeTruthy();

  await expect
    .poll(
      async () => {
        const catalog = await apiGetWithRetry<any>(request, '/api/v1/persona/catalog');
        return extractPersonaProfiles(catalog.body).some((item) => {
          const candidateId =
            typeof item?.id === 'string' || typeof item?.id === 'number' ? String(item.id) : '';
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
    applyAsOverlay?: boolean;
  }
) => {
  await expect(page.getByRole('button', { name: /send message/i })).toBeEnabled();
  const runtimeInspector = getDesktopRuntimeInspector(page);
  await runtimeInspector.getByRole('button', { name: 'Select character or persona' }).click();
  const panel = page.getByTestId('assistant-select-panel');
  await expect(panel).toBeVisible();
  await expect(panel.getByRole('textbox', { name: 'Search characters and personas', exact: true })).toBeFocused();
  await panel.getByRole('tab', { name: options.tab }).click();
  await expect(page.getByRole('tab', { name: options.tab })).toHaveAttribute(
    'aria-selected',
    'true'
  );
  const assistantButton = panel.getByRole('button', {
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
  if (options.applyAsOverlay) {
    await expect(assistantButton).toBeDisabled();
    await panel.getByRole('button', { name: 'Use a persona prompt overlay' }).click();
    await expect(assistantButton).toBeEnabled();
  }
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
      await stopStreaming.click({ timeout: 2_000 }).catch(() => undefined);
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
          responseBody: call.responseBody,
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

const expectDeterministicMockReply = async (page: Page): Promise<void> => {
  const assistantMessage = page
    .getByRole('log', { name: /chat messages/i })
    .locator(
      "article[aria-label*='Assistant message'], [data-role='assistant'], [data-message-role='assistant'], .assistant-message"
    )
    .last();

  await expect(assistantMessage).toContainText(/mock provider returned a deterministic success/i, {
    timeout: 30_000,
  });
};

const waitForSuccessfulChatMessagesLoad = (page: Page, chatId: string, timeout = 60_000) => {
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

const readPlaygroundSessionSnapshot = async (page: Page): Promise<PlaygroundSessionSnapshot> =>
  page.evaluate(() => {
    const raw = window.localStorage.getItem('tldw-playground-session');
    if (!raw) return null;

    try {
      const toRecord = (value: unknown): Record<string, unknown> | null =>
        value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
      const toNullableString = (value: unknown): string | null =>
        typeof value === 'string' ? value : null;
      const toSnapshot = (
        value: unknown
      ): { historyId: string | null; serverChatId: string | null } | null => {
        const record = toRecord(value);
        if (!record) return null;
        return {
          historyId: toNullableString(record.historyId),
          serverChatId: toNullableString(record.serverChatId),
        };
      };

      const parsed = JSON.parse(raw) as unknown;
      const parsedRecord = toRecord(parsed);
      if (!parsedRecord) return null;

      return 'state' in parsedRecord ? toSnapshot(parsedRecord.state) : toSnapshot(parsedRecord);
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
      }, { models: [{ name: 'gpt-4.1-mini', provider: 'openai', type: 'chat' }] })
    ).toEqual({
      provider: 'openai',
      model: 'gpt-4.1-mini',
      key: 'openai:gpt-4.1-mini',
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
      }, { models: [{ model: 'llama3:latest', provider: 'llama.cpp', type: 'chat' }] })
    ).toEqual({ provider: 'llama.cpp', model: 'llama3:latest', key: 'llamacpp:llama3:latest' });
  });

  test('qualifies deployed name-only metadata with its explicit provider', () => {
    expect(buildConfiguredChatModelSelection({ providers: [{ name: 'openai', is_configured: true,
      default_model: 'gpt-4.1-mini', models: ['gpt-4.1-mini'] }] },
      { models: [{ name: 'gpt-4.1-mini', provider: 'openai' }] })).toEqual({
      provider: 'openai', model: 'gpt-4.1-mini', key: 'openai:gpt-4.1-mini'
    });
  });

  test('rejects missing or ambiguous configured model catalog entries', () => {
    const providers = { providers: [{ name: 'openai', is_configured: true, models: ['gpt-4.1-mini'] }] };
    expect(() => buildConfiguredChatModelSelection(providers, { models: [] })).toThrow(/unique.*catalog/i);
    const entry = { id: 'openai:gpt-4.1-mini', provider: 'openai', type: 'chat' };
    expect(() => buildConfiguredChatModelSelection(providers, { models: [entry, entry] })).toThrow(/unique.*catalog/i);
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

    await seedRealServerConfig(page, {
      selectedModel: chatModelSelection,
      cockpitLayout: desktopCockpitLayoutSeed,
    });
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
      .getByRole('button', { name: 'Hide context' });
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
      'Hide context'
    );
    await collapseContextSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);

    const restoreContextSidechannel = page.getByTestId('playground-cockpit-left-rail-restore');
    await assertVisibleTooltipForControl(
      page,
      restoreContextSidechannel,
      'Show context'
    );
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-context-restore-side-tooltip.png'),
      fullPage: true,
    });
    await restoreContextSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();

    const collapseRuntimeSidechannel = page
      .getByTestId('playground-cockpit-right-rail')
      .getByRole('button', { name: 'Hide chat settings' });
    await assertVisibleTooltipForControl(
      page,
      collapseRuntimeSidechannel,
      'Hide chat settings'
    );
    await collapseRuntimeSidechannel.click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);

    const restoreRuntimeSidechannel = page.getByTestId('playground-cockpit-right-rail-restore');
    await assertVisibleTooltipForControl(
      page,
      restoreRuntimeSidechannel,
      'Show chat settings'
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
    const models = await apiGet<any>(request, '/api/v1/llm/models/metadata');
    expect(models.status).toBe(200);
    expect(extractModels(models.body).length).toBeGreaterThan(0);
    const chatModelSelection = buildConfiguredChatModelSelection(providers.body, models.body);

    const apiTracker = trackRealApiHits(page);
    await seedRealServerConfig(page, {
      selectedModel: chatModelSelection,
      cockpitLayout: desktopCockpitLayoutSeed,
      uiMode: 'pro',
    });
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
    await expect(modeSummary).toHaveText('Context and chat settings visible.');
    await expect(getDesktopCompositionPreview(page)).toContainText(
      `Scope: ${chatModelSelection.key}`
    );

    await page
      .getByTestId('playground-cockpit-left-rail')
      .getByRole('button', { name: 'Hide context' })
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
    await expect(modeSummary).toHaveText('Context and chat settings visible.');

    await page
      .getByTestId('playground-cockpit-right-rail')
      .getByRole('button', { name: 'Hide chat settings' })
      .click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail-restore')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-collapsed-composition-summary')).toHaveCount(0);
    await page.getByTestId('playground-cockpit-right-rail-restore').click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context and chat settings visible.');

    await page
      .getByTestId('playground-cockpit-left-rail')
      .getByRole('button', { name: 'Hide context' })
      .click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context hidden. Chat settings visible.');
    await page
      .getByTestId('playground-cockpit-right-rail')
      .getByRole('button', { name: 'Hide chat settings' })
      .click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toHaveCount(0);
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(modeSummary).toHaveText('Cockpit rails hidden. Chat and composer remain active.');
    await page.getByTestId('playground-cockpit-left-rail-restore').click();
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toHaveCount(0);
    await expect(modeSummary).toHaveText('Chat settings hidden. Context visible.');
    await page.getByTestId('playground-cockpit-right-rail-restore').click();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(modeSummary).toHaveText('Context and chat settings visible.');
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
    if (toggledWebSearchState !== 'true') {
      await webSearchControl.click();
      await expect(webSearchControl).toHaveAttribute('aria-pressed', 'true');
    }
    const sourceInventory = contextRail.getByRole('list', { name: 'Context sources' });
    await expect(sourceInventory).toBeVisible();
    await expect(
      sourceInventory.getByRole('listitem').filter({ hasText: 'Web search' })
    ).toContainText('Selected for the next message.');

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

    const advanced = page.getByTestId('composer-advanced-toggle');
    await expect(advanced).toBeEnabled();
    if (await advanced.getAttribute('aria-expanded') !== 'true') await advanced.click();
    await expect(page.getByTestId('composer-options-panel')).toBeVisible();
    await expect(page.getByTestId('composer-advanced-toggle')).toHaveAttribute('aria-expanded', 'true');
    await expect(page.locator('#composer-pro-advanced-controls-row')).toBeVisible();

    const modelSelector = page.getByTestId('model-selector').first();
    await modelSelector.focus();
    await expect(modelSelector).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('textbox', { name: 'Search models' })).toBeVisible();
    await expect(page.getByRole('menu').first()).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('textbox', { name: 'Search models' })).toBeHidden();
    await expect(modelSelector).toBeFocused();

    const promptTrigger = contextRail.getByRole('button', { name: /Select prompt|Select a prompt/i });
    await promptTrigger.click();
    const promptSearch = page.getByPlaceholder('Search prompts...', { exact: true });
    await expect(promptSearch).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(promptSearch).toBeHidden();
    await expect(promptTrigger).toBeFocused();

    const assistantTrigger = runtimeInspector.getByRole('button', { name: 'Select character or persona' });
    await assistantTrigger.click();
    const assistantSearch = page.getByRole('textbox', { name: 'Search characters and personas', exact: true });
    await expect(assistantSearch).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('assistant-select-panel')).toBeHidden();
    await expect(assistantTrigger).toBeFocused();

    const toolsTrigger = page.getByTestId('tools-button');
    const toolsBeforePointerFocus = await toolsTrigger.boundingBox();
    expect(toolsBeforePointerFocus).not.toBeNull();
    await toolsTrigger.click();
    const toolsPopup = page.locator('.playground-more-tools:visible');
    await expect(toolsPopup.getByRole('button', { name: 'Clear conversation', exact: true })).toBeVisible();
    const toolsAfterPointerFocus = await toolsTrigger.boundingBox();
    expect(toolsAfterPointerFocus).not.toBeNull();
    expect(Math.abs(toolsAfterPointerFocus!.y - toolsBeforePointerFocus!.y)).toBeLessThanOrEqual(1);
    await page.keyboard.press('Escape');
    await expect(toolsPopup).toBeHidden();
    await expect(toolsTrigger).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(toolsPopup.getByRole('button', { name: 'Clear conversation', exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(toolsPopup).toBeHidden();
    await expect(toolsTrigger).toBeFocused();

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

    await switchChatLayoutMode(page, 'focus');
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

    await switchChatLayoutMode(page, 'cockpit');
    await expect(modeSummary).toHaveText('Context and chat settings visible.');
    await expect(page.getByTestId('playground-cockpit-left-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-right-rail')).toBeVisible();
    await expect(page.getByTestId('playground-cockpit-left-rail-restore')).toBeHidden();
    await expect(page.getByTestId('playground-cockpit-right-rail-restore')).toBeHidden();
    await page.evaluate(() => {
      localStorage.setItem('playgroundChatLayoutMode', JSON.stringify('cockpit'));
      localStorage.setItem('playgroundChatContextRailVisible', JSON.stringify(true));
      localStorage.setItem('playgroundChatRuntimeRailVisible', JSON.stringify(true));
    });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({ timeout: 60_000 });
    await assertNoBlockingServerDialog(page);
    await ensureDesktopCockpitRailsVisible(page);
    await assertNoHorizontalOverflow(page);

    const sendContextRail = getDesktopContextRail(page);
    const sendWebSearchControl = sendContextRail.getByRole('button', {
      name: 'Web search',
      exact: true,
    });
    if ((await sendWebSearchControl.getAttribute('aria-pressed')) !== 'true') {
      await sendWebSearchControl.click();
    }
    await expect(sendWebSearchControl).toHaveAttribute('aria-pressed', 'true');
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

    const failingApiHits = apiTracker.hits.filter((hit) => hit.status >= 400);
    expect(failingApiHits).toEqual([]);
    expect(apiTracker.hits.some((hit) => hit.path === '/health')).toBe(true);

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

    await seedRealServerConfig(page, {
      selectedModel: chatModelSelection,
      cockpitLayout: desktopCockpitLayoutSeed,
    });
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

      for (const dismissal of ['Close', 'Escape'] as const) {
        await modelSettingsTrigger.click();
        await expect(modelSettingsDialog).toBeVisible();
        if (dismissal === 'Close') {
          await modelSettingsDialog.getByRole('button', { name: 'Close', exact: true }).click();
        } else {
          await page.keyboard.press('Escape');
        }
        await expect(modelSettingsDialog).toBeHidden();
        await expect(modelSettingsTrigger).toBeFocused({ timeout: 5_000 });
        await expect(runtimeInspector.getByText(restoredTemperature, { exact: true })).toBeVisible();
      }

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

  test('keeps mobile cockpit panels and focus composer usable against the live server', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(120_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    await seedRealServerConfig(page, {
      selectedModel: chatModelSelection,
      cockpitLayout: mobileFocusWithCockpitPanelsSeed,
    });
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
    const panelControlledByButton = async (panel: 'context' | 'runtime') => {
      const config = mobileCockpitPanelConfig[panel];
      const button = page.locator(`button#${config.buttonId}`);
      const region = page.locator(`section#${config.panelId}`);
      await expect(button).toHaveAttribute('type', 'button');
      await expect(button).toHaveAttribute('aria-controls', config.panelId);
      await expect(region).toHaveAttribute('aria-labelledby', config.buttonId);
      return region;
    };
    await fillEditableComposer(page, mobileDraft);
    await expectMobileDraftReachable();
    await waitForPlaygroundDraftSaved(page);
    await expectNoMobileBottomControls();

    await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
      'data-mode',
      'focus'
    );
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);

    await switchChatLayoutMode(page, 'cockpit');
    let mobileRails = await waitForMobileCockpitPanel(page, 'context');
    await expect(mobileRails.getByTestId('playground-cockpit-mobile-panel-summary')).toHaveText(
      'Context panel active. Composer draft remains available below.'
    );
    await expectMobileDraftPreserved();
    await expectNoMobileBottomControls();
    const contextButton = mobileRails.getByRole('button', { name: 'Context', exact: true });
    const runtimeButton = mobileRails.getByRole('button', { name: 'Chat settings', exact: true });
    await expect(mobileRails.getByRole('group', { name: 'Mobile cockpit panels' })).toBeVisible();
    await expect(mobileRails.getByRole('tab')).toHaveCount(0);
    const initialContextPanel = await panelControlledByButton('context');
    const initialRuntimePanel = await panelControlledByButton('runtime');
    await expect(contextButton).toHaveAttribute('aria-expanded', 'true');
    await expect(runtimeButton).toHaveAttribute('aria-expanded', 'false');
    const expectMobilePanelsClosed = async () => {
      await expect(mobileRails).toHaveAttribute('data-mobile-panel', 'none');
      await expect(contextButton).toHaveAttribute('aria-expanded', 'false');
      await expect(runtimeButton).toHaveAttribute('aria-expanded', 'false');
      await expect(initialContextPanel).toHaveAttribute('hidden', '');
      await expect(initialRuntimePanel).toHaveAttribute('hidden', '');
      await expect(initialContextPanel).toHaveAttribute('aria-hidden', 'true');
      await expect(initialRuntimePanel).toHaveAttribute('aria-hidden', 'true');
      await expect(initialContextPanel).toBeHidden();
      await expect(initialRuntimePanel).toBeHidden();
      await expect(mobileRails.getByRole('region')).toHaveCount(0);
      await expect(mobileRails.getByTestId('playground-cockpit-mobile-panel-summary')).toHaveText(
        'Cockpit panels hidden. Composer draft remains available below.'
      );
      await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
        'data-left-rail',
        'visible'
      );
      await expect(page.getByTestId('playground-cockpit-shell')).toHaveAttribute(
        'data-right-rail',
        'visible'
      );
      await expectMobileDraftReachable();
    };
    await expect(initialContextPanel).toBeVisible();
    await expect(initialRuntimePanel).toBeHidden();
    await assertNoHorizontalOverflow(page);
    await assertNoMobilePanelComposerOverlap(
      page,
      'context',
      'mobile cockpit context rails should not overlap composer'
    );
    await assertMobilePanelMeasuredHeight(page, 'context', 'mobile context panel is measurable');
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-context.png'),
      fullPage: true,
    });
    await contextButton.click();
    await expectMobilePanelsClosed();
    await runtimeButton.focus();
    await runtimeButton.press('Enter');
    mobileRails = await waitForMobileCockpitPanel(page, 'runtime');
    await expect(contextButton).toHaveAttribute('aria-expanded', 'false');
    await expect(initialContextPanel).toBeHidden();
    await expectMobileDraftPreserved();
    await runtimeButton.press('Space');
    await expectMobilePanelsClosed();
    await contextButton.click();
    mobileRails = await waitForMobileCockpitPanel(page, 'context');
    await expect(contextButton).toHaveAttribute('aria-expanded', 'true');
    await expect(runtimeButton).toHaveAttribute('aria-expanded', 'false');
    const contextPanelTarget = await panelControlledByButton('context');
    const runtimePanelTarget = await panelControlledByButton('runtime');
    await expect(contextPanelTarget).toBeVisible();
    await expect(runtimePanelTarget).toBeHidden();
    await expectMobileDraftPreserved();
    const contextPanel = contextPanelTarget;
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
    // Retire earlier composer layout timers before focusing its sibling rail.
    await page.waitForTimeout(1000);
    await runtimeButton.click();
    await expect(runtimeButton).toBeFocused();
    mobileRails = await waitForMobileCockpitPanel(page, 'runtime');
    await expect(runtimeButton).toHaveAttribute('aria-expanded', 'true');
    await expect(contextButton).toHaveAttribute('aria-expanded', 'false');
    const runtimePanel = mobileRails.getByRole('region', { name: 'Chat settings', exact: true });
    await expect(runtimePanel).toBeVisible();
    await expect(contextPanelTarget).toBeHidden();
    await expect(mobileRails.getByTestId('playground-cockpit-mobile-panel-summary')).toHaveText(
      'Runtime panel active. Composer draft remains available below.'
    );
    await expectMobileDraftPreserved();
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
    await assertMobileComposerDraftVisible(page);
    await expect(runtimeButton).toBeFocused();
    await assertNoMobilePanelComposerOverlap(
      page,
      'runtime',
      'mobile cockpit runtime rails should not overlap composer'
    );
    await assertMobilePanelMeasuredHeight(page, 'runtime', 'mobile runtime panel is measurable');
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

    await switchChatLayoutMode(page, 'focus');
    await expect(page.getByTestId('playground-cockpit-mobile-rails')).toHaveCount(0);
    await assertCoreComposerControls(page, { mobile: true, composerOnly: true });
    await expectMobileDraftReachable();
    await assertNoHorizontalOverflow(page);
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-mobile-focus.png'),
      fullPage: true,
    });
    await expectNoMobileBottomControls();
    await switchChatLayoutMode(page, 'cockpit');
    mobileRails = await waitForMobileCockpitPanel(page, 'runtime');
    await expectMobileDraftPreserved();
    await expect(mobileRails.getByRole('region', { name: 'Chat settings', exact: true })).toBeVisible();
    await assertNoHorizontalOverflow(page);
  });

  test('reserves composer space below runtime rails on a short mobile viewport', async ({ page, request }) => {
    const selection = await getConfiguredChatModelSelection(request);
    await seedRealServerConfig(page, { selectedModel: selection, cockpitLayout: { mode: 'cockpit', contextRailVisible: true, runtimeRailVisible: true, mobilePanel: 'runtime' } });
    await page.setViewportSize({ width: 390, height: 600 });
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });
    const rails = await waitForMobileCockpitPanel(page, 'runtime');
    const panel = rails.getByRole('region', { name: 'Chat settings', exact: true });
    await expect(panel.getByRole('heading', { name: 'MCP tools' })).toBeVisible();
    await fillEditableComposer(page, 'Keep this short-screen draft.');
    await assertMobileComposerDraftVisible(page);
    await assertNoMobilePanelComposerOverlap(page, 'runtime', 'short mobile runtime panel should leave composer space');
    await expect(page.getByTestId('chat-input')).toHaveValue('Keep this short-screen draft.');
    await panel.getByRole('button', { name: 'Configure MCP tools' }).focus();
    await expect(panel.getByRole('button', { name: 'Configure MCP tools' })).toBeFocused();
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

    await seedRealServerConfig(page, {
      selectedModel: chatModelSelection,
      cockpitLayout: mobileFocusOnlyLayoutSeed,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    const catalogResponse = page.waitForResponse(response => response.request().method() === 'GET' &&
      new URL(response.url()).pathname === '/api/v1/llm/models/metadata' && response.status() === 200);
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

    const browserCatalog = extractModels(await (await catalogResponse).json());
    expect(browserCatalog.some(entry => normalizeCockpitProviderKey(String(entry.provider)) === normalizeCockpitProviderKey(chatModelSelection.provider) &&
      normalizeConfiguredChatModelId(chatModelSelection.provider, entry.model || entry.model_id || entry.id || entry.name) === chatModelSelection.model)).toBe(true);
    await expect(page.getByTestId('model-selector-option')).toHaveCount(0);
    const mobileSmokePrompt = 'mobile cockpit smoke deterministic prompt';
    await page.getByTestId('chat-input').fill(mobileSmokePrompt);
    await page.getByRole('button', { name: /send message/i }).click();
    await expect(page.locator("article[data-role='user']").last()).toContainText(mobileSmokePrompt);
    await assertChatCompletionRenderedOrRecoverable(page, null);
    await expect(page.getByText('Chat now saved on server', { exact: true })).toHaveCount(0);
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
    const chatModelSelection = await getDeterministicChatReadiness(request);
    expect(chatModelSelection).toMatchObject({
      providerId: 'custom-openai-api',
      modelId: expect.any(String),
    });

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

      await seedRealServerConfig(page, {
        selectedModel: chatModelSelection,
        cockpitLayout: desktopCockpitLayoutSeed,
      });
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

      await seedRealServerConfig(page, {
        selectedModel: chatModelSelection,
        cockpitLayout: desktopCockpitLayoutSeed,
      });
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

      await assistantContextSource.getByRole('button', { name: 'Clear assistant' }).click();
      await assertRuntimeAssistantCleared(runtimeInspector);
      await expect(assistantContextSource).toHaveCount(0);
      await expect(composerAssistant).toHaveAttribute('aria-label', /Select character or persona/i);

      const plainPrompt = `Plain return after character clear ${Date.now()}`;
      const plainTurn = await sendChatTurnAndCapture(page, plainPrompt);
      const plainCompletionCall = findConversationTurnCall(plainTurn.calls);
      expect(plainCompletionCall).toBeDefined();
      const plainCompletionPayload =
        plainCompletionCall?.requestBody && typeof plainCompletionCall.requestBody === 'object'
          ? (plainCompletionCall.requestBody as Record<string, unknown>)
          : {};
      expect(plainCompletionPayload).not.toHaveProperty('character_id');
      expect(plainCompletionPayload).not.toHaveProperty('assistant_kind');
      expect(plainCompletionPayload).not.toHaveProperty('assistant_id');

      const plainChatId = extractConversationChatIdFromCall(plainCompletionCall);
      expect(plainChatId).toBeTruthy();
      const plainChatDetails = await getChatDetails(request, String(plainChatId));
      expect(plainChatDetails.status).toBe(200);
      expect(plainChatDetails.body).toMatchObject({
        id: plainChatId,
        character_id: null,
        assistant_kind: null,
        assistant_id: null,
      });
      await expectDeterministicMockReply(page);
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
      await seedRealServerConfig(page, {
        selectedModel: chatModelSelection,
        cockpitLayout: desktopCockpitLayoutSeed,
      });
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
      await seedRealServerConfig(page, {
        selectedModel: chatModelSelection,
        cockpitLayout: desktopCockpitLayoutSeed,
      });
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
        applyAsOverlay: true,
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
    const chatModelSelection = await getDeterministicChatReadiness(request);
    expect(chatModelSelection).toMatchObject({
      providerId: 'custom-openai-api',
      modelId: expect.any(String),
    });

    const timestamp = Date.now();
    let character: DisposableCharacter | null = null;

    try {
      character = await createDisposableCharacter(request, `Tracked Character ${timestamp}`);

      await openDesktopChatCockpit(page, chatModelSelection);

      await selectAssistantFromRuntimeRail(page, {
        tab: 'Characters',
        assistantName: character.name,
      });
      const openingCapture = captureAllApiCalls(page);
      await page.getByRole('button', { name: 'Start with this greeting', exact: true }).click();
      await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(character.firstMessage);
      const openingCalls = await openingCapture.stop();
      expect(openingCalls.filter(call => call.method === 'POST' && /\/(?:chat\/completions|complete-v2|completions)$/.test(new URL(call.url).pathname))).toEqual([]);
      const createCall = findChatCreateCall(openingCalls);
      expect(createCall).toBeDefined();
      const trackedCharacterId = (createCall?.requestBody as Record<string, unknown> | null)
        ?.character_id;
      expect(String(trackedCharacterId ?? '')).toBe(character.id);

      const chatId =
        extractCreatedChatIdFromCreateCall(createCall) ??
        (await waitForPersistedServerChatId(page));

      expect(createCall?.status).toBe(201);
      expect(new URL(createCall!.url).searchParams.get('seed_first_message')).toBe('true');
      expect(new URL(createCall!.url).searchParams.get('greeting_strategy')).toBe('default');
      const opening = await apiGet<any>(request, `/api/v1/chats/${chatId}/messages?include_metadata=true&render_placeholders=false`);
      expect(opening.status).toBe(200);
      expect(opening.body.messages).toHaveLength(1);
      const greeting = opening.body.messages[0];
      expect(greeting).toMatchObject({ conversation_id: chatId, sender: 'assistant', content: character.firstMessage, parent_message_id: null, version: 1 });
      const captured = openingCalls.filter(call => call.method === 'POST' && call.status === 200 && new URL(call.url).pathname === `/api/v1/chat/conversations/${chatId}/history/selection`).at(-1)?.responseBody as any;
      expect(captured.view).toMatchObject({ owner_key: expect.stringMatching(/^native-history-v1:sha256:/), conversation_id: chatId, cursor: { kind: 'after_message', message_id: greeting.id } });
      expect(captured.snapshot).toMatchObject({ owner_key: captured.view.owner_key, conversation_id: chatId, source_digest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/) });
      expect(captured.rows).toHaveLength(1);
      expect(captured.rows[0]).toMatchObject({ id: greeting.id, role: 'assistant', settled: true, parent_id: null });

      const prompt = `Tracked character proof ${timestamp}`;
      const turn = await sendChatTurnAndCapture(page, prompt);
      const appendCall = turn.calls.find(call => call.method === 'POST' && new URL(call.url).pathname === `/api/v1/chats/${chatId}/roleplay/messages`);
      const completionCall = turn.calls.find(call => call.method === 'POST' && new URL(call.url).pathname === `/api/v1/chats/${chatId}/complete-v2`);
      expect(appendCall?.status).toBe(201);
      expect(completionCall?.status).toBe(200);
      const input = parseRoleplayResumeAppendRequest(appendCall?.requestBody);
      expect(input.content).toBe(prompt);
      expect(input.expected.tail).toEqual({ message_id: greeting.id, message_version: greeting.version });
      const admission = parseRoleplayResumeAppendReceipt(chatId, input, appendCall?.responseBody);
      const sent = parseRoleplayResumeCompletionRequest(completionCall?.requestBody);
      expect(sent).toMatchObject({ stream: false, save_to_db: true, include_character_context: true, append_user_message: false,
        tldw_roleplay_resume_v1: { input_message_id: admission.message_id, input_message_version: admission.message_version, expected: admission.fence } });
      const settlement = parseRoleplayResumeCompletionReceipt(chatId, sent, completionCall?.responseBody);
      expect(settlement).toMatchObject({ character_id: Number(character.id), saved: true });
      const canonical = await apiGet<any>(request, `/api/v1/chats/${chatId}/messages?include_metadata=true&render_placeholders=false`);
      expect(canonical.status).toBe(200);
      expect(canonical.body.messages).toHaveLength(3);
      expect(canonical.body.messages[0]).toEqual(greeting);
      expect(canonical.body.messages[1]).toMatchObject({ id: admission.message_id, content: prompt, sender: 'user', parent_message_id: greeting.id });
      expect(canonical.body.messages[2]).toMatchObject({ id: settlement.assistant_message_id, content: settlement.assistant_content, sender: 'assistant', parent_message_id: admission.message_id });
      const chatDetails = await getChatDetails(request, chatId);
      expect(chatDetails.status).toBe(200);
      expect(chatDetails.body).toMatchObject({ id: chatId, character_id: Number(character.id), assistant_kind: 'character' });
      await expectDeterministicMockReply(page);

      await page.reload({ waitUntil: 'domcontentloaded' });
      await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible({
        timeout: 60_000,
      });
      await assertNoBlockingServerDialog(page);
      await expect(page.getByTestId('character-chat-readiness-panel')).toBeHidden({
        timeout: 60_000,
      });
      await assertCoreComposerControls(page);
      await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(prompt);
      await expectDeterministicMockReply(page);
      const reopened = await apiGet<any>(request, `/api/v1/chats/${chatId}/messages?include_metadata=true&render_placeholders=false`);
      expect(reopened.status).toBe(200);
      expect(reopened.body.messages).toEqual(canonical.body.messages);

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

  test('explains unavailable tracked persona chat without writes and offers an explicit ordinary-chat overlay', async ({ page, request }) => {
    test.setTimeout(150_000);
    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getDeterministicChatReadiness(request);
    expect(chatModelSelection).toMatchObject({ providerId: 'custom-openai-api', modelId: expect.any(String) });
    const timestamp = Date.now();
    let persona: DisposablePersona | null = null;
    try {
      persona = await createDisposablePersona(request, `tracked_persona_${timestamp}`, `Tracked Persona ${timestamp}`);
      await openDesktopChatCockpit(page, chatModelSelection);
      const prompt = `Preserved persona draft ${timestamp}`;
      const input = page.getByTestId('chat-input');
      await input.fill(prompt);
      const originalSession = await readPlaygroundSessionSnapshot(page);
      const originalMessages = await page.getByRole('log', { name: /chat messages/i }).textContent();
      const writes: string[] = [];
      const recordWrite = (hit: Request) => {
        if (hit.url().includes('/api/') && ['POST', 'PUT', 'PATCH', 'DELETE'].includes(hit.method()))
          writes.push(`${hit.method()} ${new URL(hit.url()).pathname}`);
      };
      page.on('request', recordWrite);
      try {
        const runtime = getDesktopRuntimeInspector(page);
        await runtime.getByRole('button', { name: 'Select character or persona' }).click();
        const panel = page.getByTestId('assistant-select-panel');
        await panel.getByRole('tab', { name: 'Personas' }).click();
        const choice = panel.getByRole('button', { name: persona.name, exact: true });
        await expect(choice).toBeVisible({ timeout: 30_000 });
        await expect(choice).toBeDisabled();
        await expect(choice).toHaveAccessibleDescription(/tracked persona chat.*unavailable.*overlay/i);
        await expect(panel.getByRole('status')).toContainText('choose a character for tracked chat');
        await choice.evaluate((element: HTMLButtonElement) => element.click());
        await expect(choice).toBeDisabled();
        await expect(panel).toBeVisible();
        await expect(input).toHaveValue(prompt);
        expect(await readPlaygroundSessionSnapshot(page)).toEqual(originalSession);
        expect(await page.getByRole('log', { name: /chat messages/i }).textContent()).toBe(originalMessages);
        await assertRuntimeAssistantCleared(runtime);
        expect(writes).toEqual([]);
        await panel.getByRole('button', { name: 'Use a persona prompt overlay' }).click();
        await expect(choice).toBeEnabled();
        await choice.click();
        await expect(panel).toBeHidden();
      } finally {
        page.removeListener('request', recordWrite);
      }
      await expect(input).toHaveValue(prompt);
      const turn = await sendChatTurnAndCapture(page, prompt);
      expect(turn.response?.status()).toBe(200);
      const ordinaryCompletion = turn.calls.find(call => call.method === 'POST' && /\/chat\/completions$/.test(new URL(call.url).pathname));
      expect(ordinaryCompletion).toBeDefined();
      expect(JSON.stringify(ordinaryCompletion?.requestBody)).toContain(`You are ${persona.name}, a concise overlay test persona.`);
      await expect(getDesktopRuntimeInspector(page)).toContainText(persona.name);
      for (const call of turn.calls.filter(call => call.method === 'POST')) {
        expect(call.requestBody).not.toMatchObject({ assistant_kind: 'persona' });
        expect(new URL(call.url).pathname).not.toMatch(/\/complete-v2$/);
      }
      await expectDeterministicMockReply(page);
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
    await seedRealServerConfig(page, { cockpitLayout: desktopCockpitLayoutSeed });
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
    expect(modelKey).toBe(chatModelSelection.key);
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

  test('captures streaming stop and the selected-history regeneration gate through the real cockpit', async ({
    page,
    request,
  }, testInfo) => {
    test.setTimeout(150_000);

    const health = await apiGet<any>(request, '/api/v1/health');
    assertHealthResponse(health);
    const chatModelSelection = await getConfiguredChatModelSelection(request);

    const catalogResponse = page.waitForResponse(response => response.request().method() === 'GET' &&
      new URL(response.url()).pathname === '/api/v1/llm/models/metadata' && response.status() === 200);
    await openDesktopChatCockpit(page, chatModelSelection);
    const browserCatalog = extractModels(await (await catalogResponse).json());
    expect(browserCatalog.some(entry => normalizeCockpitProviderKey(String(entry.provider)) === normalizeCockpitProviderKey(chatModelSelection.provider) &&
      normalizeConfiguredChatModelId(chatModelSelection.provider, entry.model || entry.model_id || entry.id || entry.name) === chatModelSelection.model)).toBe(true);
    await expect(page.getByTestId('model-selector-option')).toHaveCount(0);

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
    const completionAttempt = page.waitForRequest(
      (request) => isChatCompletionRequest(page, request),
      { timeout: 90_000 }
    );

    await page.getByTestId('chat-input').fill(prompt);
    await page.getByRole('button', { name: /send message/i }).click();
    const completionRequest = await completionAttempt;
    await assertProviderQualifiedPayload(page, completionRequest);

    const runtimeStop = runtimeInspector.getByRole('button', { name: 'Stop generation' });
    const messageStop = page.getByRole('button', { name: /Stop streaming response/i }).first();
    const stopCandidates = [
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
      await assertChatCompletionRenderedOrRecoverable(page);
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
    }

    await expect(runtimeStop).toBeDisabled({ timeout: 30_000 });
    await expect(messageStop).toBeHidden({ timeout: 30_000 });
    const assistantMessages = page
      .getByRole('log', { name: /chat messages/i })
      .locator("article[aria-label*='Assistant message']");
    if ((await assistantMessages.count()) === 0) {
      // Stopping before the first token correctly removes the empty assistant
      // stub. Complete another turn so the history gate has a response to gate.
      await expect(regenerateControl).toBeDisabled();
      await expect(
        runtimeInspector.getByText('Regenerate becomes available after an assistant response.')
      ).toBeVisible();
      // Stop may precede the server settlement receipt; reload the saved history before a new turn.
      const originalChatId = completionRequest.postDataJSON().conversation_id ?? completionRequest.postDataJSON().tldw_history_selection_v1?.conversation_id;
      expect(typeof originalChatId).toBe('string');
      const originalRecoveries = ((await readStore(page, 'historySelections')) as HistoryBookmark[])
        .filter(row => row.conversation_id === originalChatId).flatMap(row => Object.values(row.pending_turns ?? {}));
      const stoppedInput = originalRecoveries.find(turn => turn.input_text === prompt);
      expect(stoppedInput).toBeDefined();
      await expect(page.getByText('The server may have saved a reply. Text that arrived is retained here. Reload this chat and open Review conversation history. Choose a saved message to continue after before trying again. Recovered requests are never sent automatically.', { exact: true })).toBeVisible();
      const automaticTurnAttempts: string[] = [];
      const observeAutomaticTurn = (hit: Request) => {
        if (hit.method() !== 'POST') return;
        const path = new URL(hit.url()).pathname;
        if (path.endsWith('/completions') || path.endsWith('/complete-v2') ||
          path.endsWith('/complete') || path.endsWith('/messages')) automaticTurnAttempts.push(path);
      };
      page.on('request', observeAutomaticTurn);
      try {
        await page.reload({ waitUntil: 'domcontentloaded' });
        await assertNoBlockingServerDialog(page);
        const coldRecovery = ((await readStore(page, 'historySelections')) as HistoryBookmark[])
          .filter(row => row.conversation_id === originalChatId).flatMap(row => Object.values(row.pending_turns ?? {}))
          .find(turn => turn.operation_id === stoppedInput!.operation_id);
        expect(coldRecovery).toEqual(stoppedInput);
        await page.getByRole('button', { name: 'Review conversation history', exact: true }).click();
        const savedSource = page.getByRole('list', { name: 'Complete source messages', exact: true });
        await expect(savedSource).toBeVisible();
        const continueAfterSaved = savedSource.locator('[data-history-message-id]').last()
          .getByRole('button', { name: 'Continue after this message', exact: true });
        await expect(continueAfterSaved).toBeEnabled();
        await continueAfterSaved.click();
        await page.getByRole('button', { name: 'Cancel review', exact: true }).click();
        await expect(page.locator("article[data-role='user']").filter({ hasText: prompt })).toHaveCount(1);
        await expect(page.getByRole('button', { name: /send message/i })).toBeEnabled();
        expect(automaticTurnAttempts).toEqual([]);
      } finally {
        page.removeListener('request', observeAutomaticTurn);
      }
      const followUpCompletion = waitForChatCompletionAttempt(page, 90_000);
      await page.getByTestId('chat-input').fill('Reply with one short sentence to verify the saved-history regeneration gate.');
      await page.getByRole('button', { name: /send message/i }).click();
      const followUpResponse = await followUpCompletion;
      expect(followUpResponse.status()).toBeLessThan(400);
      await assertProviderQualifiedPayload(page, followUpResponse);
      await waitForStreamComplete(page, 60_000);
    }
    await expect(assistantMessages.last()).toBeVisible();
    await expect(regenerateControl).toBeDisabled({ timeout: 30_000 });
    await expect(runtimeInspector.getByText('Regeneration is unavailable for selected history.')).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath('chat-cockpit-regenerate-selected-history-gated.png'),
      fullPage: true,
    });
  });
});

test('refreshes a real empty native source after late settlement without replay or changing its boundary', async ({ page, request }) => {
  test.setTimeout(90_000);
  const model = await getConfiguredChatModelSelection(request);
  const created = await apiPost<any>(request, '/api/v1/chats/', { title: `Empty refresh 13673 ${Date.now()}`, state: 'in-progress', source: 'e2e' });
  expect(created.status).toBe(201);
  const id = String(created.body.id);
  const firstCapture = page.waitForResponse(response => new URL(response.url()).pathname.includes('/history/selection') && response.status() === 200);
  await openDesktopChatCockpit(page, model, { persistedServerChatId: id });
  const captured = await (await firstCapture).json();
  expect(captured.status).toBe('captured');
  expect(captured.snapshot.nodes).toEqual([]);
  await expect.poll(async () => ((await readStore(page, 'historySelections')) as HistoryBookmark[]).filter(row => row.conversation_id === id).length).toBeGreaterThan(0);
  const original = ((await readStore(page, 'historySelections')) as HistoryBookmark[]).filter(row => row.conversation_id === id).at(-1)!;
  const recovery = { operation_id: crypto.randomUUID(), origin_view: original.view, owner_key: original.owner_key,
    conversation_id: id, selection_digest: 'synthetic-original-empty', request_context_digest: 'synthetic-original-command',
    created_at: Date.now(), input_text: '  Synthetic original uncertain input 13673  ', input_images: [], result_text: '', state: 'unknown', persistence: 'server' };
  await page.evaluate(async ({ original, recovery }) => {
    const opening = indexedDB.open('PageAssistDatabase');
    const db = await new Promise<IDBDatabase>((resolve, reject) => { opening.onsuccess = () => resolve(opening.result); opening.onerror = () => reject(opening.error); });
    const tx = db.transaction('historySelections', 'readwrite');
    tx.objectStore('historySelections').put({ ...original, pending_turns: { ...original.pending_turns, [recovery.operation_id]: recovery } });
    await new Promise<void>((resolve, reject) => { tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error); });
    db.close();
  }, { original, recovery });
  await page.reload({ waitUntil: 'domcontentloaded' });
  const refresh = page.getByRole('button', { name: 'Refresh history', exact: true });
  await expect(refresh).toBeVisible();
  await expect(page.getByText("Save status unconfirmed", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Review conversation history', exact: true })).toHaveCount(0);
  const attempts: string[] = [];
  const observe = (hit: Request) => { const path = new URL(hit.url()).pathname; if (hit.method() === 'POST' && (path.endsWith('/messages') || path.endsWith('/completions') || path.endsWith('/complete-v2'))) attempts.push(path); };
  page.on('request', observe);
  try {
    const user = await apiPost<any>(request, `/api/v1/chats/${id}/messages`, { role: 'user', content: recovery.input_text });
    expect(user.status).toBe(201);
    const answer = await apiPost<any>(request, `/api/v1/chats/${id}/messages`, { role: 'assistant', content: 'Synthetic late saved reply 13673', parent_message_id: user.body.id });
    expect(answer.status).toBe(201);
    const refreshed = page.waitForResponse(response => new URL(response.url()).pathname.includes('/history/selection') && response.status() === 200);
    await refresh.focus();
    await refresh.press("Enter");
    const current = await (await refreshed).json();
    expect(current.snapshot.nodes).toHaveLength(2);
    expect(current.view.cursor).toEqual({ kind: 'empty' });
    await expect(page.getByRole("button", { name: "Review conversation history", exact: true })).toBeFocused();
    await expect(page.getByText("Save status unconfirmed", { exact: true }).first()).toBeVisible();
    await page.getByRole('button', { name: 'Review conversation history', exact: true }).click();
    await expect(page.getByRole('list', { name: 'Complete source messages', exact: true }).locator('[data-history-message-id]')).toHaveCount(2);
    const records = ((await readStore(page, 'historySelections')) as HistoryBookmark[]).filter(row => row.conversation_id === id);
    expect(records.flatMap(row => Object.values(row.pending_turns ?? {})).find(turn => turn.operation_id === recovery.operation_id)).toEqual(recovery);
    expect(attempts).toEqual([]);
  } finally { page.removeListener('request', observe); }
});

test('qualifies chat draft, keyboard semantics and both themes at narrow widths and renderer zoom', async ({ page, request }, testInfo) => {
  test.setTimeout(120_000);
  const model = await getConfiguredChatModelSelection(request);
  await openDesktopChatCockpit(page, model);
  const input = page.getByTestId('chat-input');
  const draft = '  Acceptance matrix draft\nTrailing spaces  ';
  await input.fill(draft);
  const cells: unknown[] = [];
  const violationsByCell: unknown[] = [];
  for (const width of [320, 390, 1280]) for (const theme of ['light', 'dark']) for (const rendererZoom of [1, 2]) {
    await page.evaluate(() => { document.documentElement.style.zoom = '1'; });
    await page.setViewportSize({ width, height: 960 });
    if (await page.evaluate(() => document.documentElement.classList.contains('dark')) !== (theme === 'dark')) await page.getByTestId('playground-chat-shell').getByTestId('chat-header-theme-toggle').click();
    await expect.poll(() => page.evaluate(() => document.documentElement.classList.contains('dark'))).toBe(theme === 'dark');
    await page.evaluate(zoom => { document.documentElement.style.zoom = String(zoom); }, rendererZoom);
    await expect(input).toBeVisible();
    await expect(input).toHaveValue(draft);
    await expect(input).toHaveAccessibleName(/message/i);
    await input.focus();
    await expect(input).toBeFocused();
    const send = page.getByRole('button', { name: /send message/i });
    await expect(send).toBeEnabled();
    await expect.poll(() => page.evaluate(() => document.getAnimations().filter(animation => animation.playState === 'running' && animation.effect?.getTiming().iterations !== Infinity).length)).toBe(0);
    const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    const violations = results.violations.map(issue => ({ id: issue.id, impact: issue.impact, targets: issue.nodes.map(node => node.target), nodes: issue.nodes.map(node => ({ target: node.target, html: node.html, summary: node.failureSummary, checks: node.any.map(check => check.data) })) }));
    cells.push({ width, theme, rendererZoom, draftPreserved: true, violations });
    await testInfo.attach(`chat-${width}-${theme}-renderer-${rendererZoom}`, { body: await page.screenshot({ fullPage: true }), contentType: 'image/png' });
    if (violations.length) violationsByCell.push({ width, theme, rendererZoom, violations });
  }
  await testInfo.attach('chat-acceptance-matrix', { body: JSON.stringify({ cells, limits: 'CSS renderer zoom is a layout stress check; native browser-toolbar zoom and physical assistive technology remain unrun. No participant results or server-load claim.' }), contentType: 'application/json' });
  expect(violationsByCell).toEqual([]);
});

for (const format of ['v2', 'png'] as const) test('imports a real '+format+' card and preserves greeting, two turns and reviewed Scene settings across reopen', async ({ page, request }, testInfo) => {
  test.setTimeout(180_000);
  const model = await getConfiguredChatModelSelection(request);
  const name = 'Disposable imported '+format+' '+Date.now();
  let seed: DisposableCharacter | null = null;
  let imported: DisposableCharacter | null = null;
  try {
    seed = await createDisposableCharacter(request, name);
    const exported = await request.get(serverUrl+'/api/v1/characters/'+seed.id+'/export?format='+format, { headers: apiHeaders() });
    expect(exported.status()).toBe(200);
    const buffer = await exported.body();
    const renamedSeed = await request.put(serverUrl+"/api/v1/characters/"+seed.id+"?expected_version="+seed.version, { headers: apiHeaders(), data: { name: name+" export seed" } });
    expect(renamedSeed.ok()).toBe(true);
    seed.version = (await renamedSeed.json()).version;
    await cleanupDisposableCharacter(request, seed); seed = null;
    await openDesktopChatCockpit(page, model);
    await page.goto('/characters', { waitUntil: 'domcontentloaded' });
    const chooserEvent = page.waitForEvent('filechooser', { timeout: 5_000 }).catch(() => null);
    await page.getByRole('button', { name: "Import character", exact: true }).click();
    const chooser = await chooserEvent;
    if (!chooser) await testInfo.attach('public-import-dom', { body: JSON.stringify(await page.getByTestId('character-import-dropzone').evaluate(node => ({ markup: node.outerHTML, buttons: node.querySelectorAll('button').length, inputs: node.querySelectorAll('input[type=file]').length }))), contentType: 'application/json' });
    expect(chooser, 'Public Import must open its native file chooser').not.toBeNull();
    await chooser!.setFiles({ name: format === 'png' ? 'synthetic-card.png' : 'synthetic-card.json', mimeType: format === 'png' ? 'image/png' : 'application/json', buffer });
    const preview = page.getByRole('dialog', { name: 'Import preview' });
    await expect(preview).toContainText(format === "png" ? "synthetic-card.png" : name);
    const importResponse = page.waitForResponse(r => r.request().method() === 'POST' && /\/api\/v1\/characters\/import\/?$/.test(new URL(r.url()).pathname));
    await preview.getByRole('button', { name: /Confirm import/i }).click();
    expect((await importResponse).ok()).toBe(true);
    await preview.getByRole('button', { name: 'OK', exact: true }).click();
    const records = await apiGet<any>(request, '/api/v1/characters?limit=1000');
    const record = extractCharacters(records.body).find(row => row.name === name && !row.deleted);
    expect(record?.id).toBeTruthy();
    imported = { id: String(record.id), name, firstMessage: record.first_message, version: record.version ?? 1 };
    await page.goto('/chat', { waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('playground-cockpit-shell')).toBeVisible();
    await selectAssistantFromRuntimeRail(page, { tab: 'Characters', assistantName: name });
    await page.getByRole('button', { name: 'Start with this greeting', exact: true }).click();
    await expect(page.getByRole('log', { name: /chat messages/i })).toContainText(imported.firstMessage);
    const chatId = await waitForPersistedServerChatId(page);
    for (const prompt of ['Imported card first turn '+format, 'Imported card second turn '+format]) {
      const turn = await sendChatTurnAndCapture(page, prompt);
      expect(turn.calls.find(call => call.method === 'POST' && new URL(call.url).pathname === '/api/v1/chats/'+chatId+'/roleplay/messages')?.status).toBe(201);
      expect(turn.calls.find(call => call.method === 'POST' && new URL(call.url).pathname === '/api/v1/chats/'+chatId+'/complete-v2')?.status).toBe(200);
    }
    const canonical = await apiGet<any>(request, '/api/v1/chats/'+chatId+'/messages?include_metadata=true&render_placeholders=false');
    expect(canonical.body.messages).toHaveLength(5);
    expect(canonical.body.messages[0].content).toBe(imported.firstMessage);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('log', { name: /chat messages/i })).toContainText('Imported card second turn '+format);
    const customize = page.getByRole('button', { name: 'Customize chat', exact: true });
    if (await customize.isVisible() && await customize.getAttribute('aria-expanded') === 'false') await customize.click();
    await page.getByTestId('composer-role-play-setup').click();
    const scene = page.getByRole('dialog', { name: 'Role-play setup', exact: true });
    await expect(scene).toBeVisible();
    const notes = 'Reviewed imported Scene '+format;
    const notesInput = scene.getByRole('textbox', { name: 'Scene notes', exact: true });
    await expect(notesInput).toBeEditable();
    await notesInput.fill(notes);
    await expect(notesInput).toHaveValue(notes);
    const sceneCapture = captureAllApiCalls(page);
    await scene.getByRole('button', { name: 'Apply', exact: true }).click();
    await expect(scene).toBeHidden();
    const sceneCalls = await sceneCapture.stop();
    await testInfo.attach("scene-apply-io", { body: JSON.stringify(sceneCalls.map(call => ({ method: call.method, path: new URL(call.url).pathname, status: call.status, requestContainsOwnNote: JSON.stringify(call.requestBody ?? null).includes(notes), responseContainsOwnNote: JSON.stringify(call.responseBody ?? null).includes(notes) }))), contentType: "application/json" });
    await page.getByTestId('composer-role-play-setup').click();
    await expect(scene.getByRole('textbox', { name: 'Scene notes', exact: true })).toHaveValue(notes);
    await scene.getByRole('button', { name: 'Cancel', exact: true }).click();
    const afterReview = await apiGet<any>(request, '/api/v1/chats/'+chatId+'/messages?include_metadata=true&render_placeholders=false');
    expect(afterReview.body.messages).toEqual(canonical.body.messages);
    await testInfo.attach('imported-card-journey', { body: JSON.stringify({ format, rows: 5, turns: 2, coldReopen: true, sceneNotesReview: true, limits: 'Scene settings review/change does not establish a saved frozen model or provider override. No packaged extension or physical AT claim.' }), contentType: 'application/json' });
  } finally {
    await cleanupDisposableCharacter(request, seed);
    await cleanupDisposableCharacter(request, imported);
  }
});


test('qualifies settled persistence badge contrast in both themes', async ({ page, request }, testInfo) => {
  const model = await getConfiguredChatModelSelection(request);
  const created = await apiPost<any>(request, '/api/v1/chats/', { title: 'Badge contrast '+Date.now(), state: 'in-progress', source: 'e2e' });
  expect(created.status).toBe(201);
  const id = String(created.body.id);
  const user = await apiPost<any>(request, '/api/v1/chats/'+id+'/messages', { role: 'user', content: 'Synthetic saved badge question' });
  expect(user.status).toBe(201);
  expect((await apiPost<any>(request, '/api/v1/chats/'+id+'/messages', { role: 'assistant', content: 'Synthetic saved badge reply', parent_message_id: user.body.id })).status).toBe(201);
  await openDesktopChatCockpit(page, model, { persistedServerChatId: id });
  const shell = page.getByTestId('playground-chat-shell');
  const badge = page.getByTestId('chat-header-persistence-badge');
  const evidence: unknown[] = [];
  for (const kind of ['server', 'temporary']) {
    if (kind === 'temporary') await page.getByRole('button', { name: 'Temporary chat (on this device)', exact: true }).click();
    await expect(badge).toHaveText(kind === 'server' ? 'Saved on server' : 'Temporary');
    for (const theme of ['light', 'dark']) {
      if (await page.evaluate(() => document.documentElement.classList.contains('dark')) !== (theme === 'dark')) await shell.getByTestId('chat-header-theme-toggle').click();
      await expect.poll(() => page.evaluate(() => document.documentElement.classList.contains('dark'))).toBe(theme === 'dark');
      await expect.poll(() => page.evaluate(() => document.getAnimations().filter(animation => animation.playState === 'running' && animation.effect?.getTiming().iterations !== Infinity).length)).toBe(0);
      const results = await new AxeBuilder({ page }).include('[data-testid="chat-header-persistence-badge"]').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      expect(results.violations).toEqual([]);
      evidence.push({ kind, theme, styles: await badge.evaluate(node => { const css = getComputedStyle(node);return { color: css.color, background: css.backgroundColor, fontSize: css.fontSize }; }), violations: [] });
    }
  }
  await testInfo.attach('settled-badge-contrast', { body: JSON.stringify({ evidence, limits: 'Saved and Temporary are actual runtime states. Failed shares the warning background/foreground projection qualified by mounted header tests; no forced runtime save failure or physical AT claim.' }), contentType: 'application/json' });
});
