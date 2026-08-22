import {
  expect,
  test,
  type APIRequestContext,
  type Browser,
  type BrowserContext,
  type Page,
  type Route,
} from '@playwright/test';
import { readFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import path from 'node:path';

type DefinitionId =
  | 'chat.rag.answer'
  | 'chat.rag.question_rewrite'
  | 'chat.web_search.answer'
  | 'media.text.translation';

type ServicePromptDetail = {
  id: DefinitionId;
  default_parts: Record<string, string>;
  effective_parts: Record<string, string>;
  source: 'packaged' | 'user';
  revision: string | null;
};

type RuntimeConfig = {
  webUrl: string;
  serverUrl: string;
  apiKey: string;
  captureUrl: URL;
};

type UpstreamCapture = {
  method: string;
  path: string;
  body: Record<string, unknown>;
};

const DISPOSABLE_API_KEY = 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY';
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);
const touchedDefinitions = new Set<DefinitionId>();
const upstreamCaptures: UpstreamCapture[] = [];
let runtimeConfig: RuntimeConfig | null = null;
let captureServer: Server | null = null;

const golden = JSON.parse(
  readFileSync(
    path.resolve(
      __dirname,
      '../../../packages/ui/src/utils/__fixtures__/service-prompt-rendering.json'
    ),
    'utf8'
  )
) as {
  defaults: Record<DefinitionId, Record<string, string>>;
};

const requireEnvironmentValue = (name: string): string => {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`${name} is required for this completion gate.`);
  }
  return value;
};

const requireLoopbackUrl = (name: string, raw: string): URL => {
  const value = new URL(raw);
  if (!LOOPBACK_HOSTS.has(value.hostname)) {
    throw new Error(`${name} must use a loopback host; received ${value.hostname}.`);
  }
  if (value.protocol !== 'http:') {
    throw new Error(`${name} must use http on the disposable local harness.`);
  }
  return value;
};

const loadRuntimeConfig = (): RuntimeConfig => {
  const webUrl = requireLoopbackUrl('TLDW_WEB_URL', requireEnvironmentValue('TLDW_WEB_URL'));
  const serverUrl = requireLoopbackUrl(
    'TLDW_E2E_SERVER_URL',
    requireEnvironmentValue('TLDW_E2E_SERVER_URL')
  );
  const captureUrl = requireLoopbackUrl(
    'TLDW_E2E_CAPTURE_URL',
    requireEnvironmentValue('TLDW_E2E_CAPTURE_URL')
  );
  const apiKey = requireEnvironmentValue('TLDW_E2E_API_KEY');
  if (apiKey !== DISPOSABLE_API_KEY) {
    throw new Error(
      'TLDW_E2E_API_KEY must be the documented fake key before prompt overrides are mutated.'
    );
  }
  return {
    webUrl: webUrl.toString().replace(/\/$/, ''),
    serverUrl: serverUrl.toString().replace(/\/$/, ''),
    apiKey,
    captureUrl,
  };
};

const getRuntimeConfig = (): RuntimeConfig => {
  if (!runtimeConfig) {
    throw new Error('Runtime E2E configuration was not initialized.');
  }
  return runtimeConfig;
};

const apiHeaders = (): Record<string, string> => ({
  'Content-Type': 'application/json',
  'X-API-KEY': getRuntimeConfig().apiKey,
});

const readJsonResponse = async <T>(
  response: Awaited<ReturnType<APIRequestContext['fetch']>>,
  label: string
): Promise<T> => {
  const status = response.status();
  const text = await response.text();
  if (status < 200 || status >= 300) {
    throw new Error(`${label} failed (${status}): ${text.slice(0, 500)}`);
  }
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    throw new Error(`${label} returned non-JSON content.`, { cause: error });
  }
};

const apiJson = async <T>(
  request: APIRequestContext,
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  apiPath: string,
  data?: unknown
): Promise<T> => {
  const response = await request.fetch(`${getRuntimeConfig().serverUrl}${apiPath}`, {
    method,
    headers: apiHeaders(),
    ...(typeof data === 'undefined' ? {} : { data }),
  });
  return readJsonResponse<T>(response, `${method} ${apiPath}`);
};

const getDefinition = (
  request: APIRequestContext,
  definitionId: DefinitionId
): Promise<ServicePromptDetail> =>
  apiJson(request, 'GET', `/api/v1/service-prompts/${encodeURIComponent(definitionId)}`);

const saveDefinition = async (
  request: APIRequestContext,
  definitionId: DefinitionId,
  parts: Record<string, string>
): Promise<ServicePromptDetail> => {
  const current = await getDefinition(request, definitionId);
  touchedDefinitions.add(definitionId);
  const saved = await apiJson<ServicePromptDetail>(
    request,
    'PUT',
    `/api/v1/service-prompts/${encodeURIComponent(definitionId)}`,
    { parts, expected_revision: current.revision }
  );
  expect(saved.source).toBe('user');
  expect(saved.effective_parts).toEqual(parts);
  expect(saved.revision).toBeTruthy();
  return saved;
};

const resetDefinition = async (
  request: APIRequestContext,
  definitionId: DefinitionId
): Promise<ServicePromptDetail> => {
  const current = await getDefinition(request, definitionId);
  if (!current.revision) {
    expect(current.source).toBe('packaged');
    return current;
  }
  const packaged = await apiJson<ServicePromptDetail>(
    request,
    'DELETE',
    `/api/v1/service-prompts/${encodeURIComponent(definitionId)}?expected_revision=${encodeURIComponent(current.revision)}`
  );
  expect(packaged.source).toBe('packaged');
  expect(packaged.revision).toBeNull();
  expect(packaged.effective_parts).toEqual(golden.defaults[definitionId]);
  return packaged;
};

const cleanupTouchedDefinitions = async (request: APIRequestContext): Promise<void> => {
  for (const definitionId of [...touchedDefinitions]) {
    await resetDefinition(request, definitionId);
    touchedDefinitions.delete(definitionId);
  }
};

const assertHealth = async (url: string, headers?: HeadersInit) => {
  const response = await fetch(url, { headers, redirect: 'follow' });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`Health preflight failed for ${url}: ${response.status} ${body.slice(0, 300)}`);
  }
};

const readRequestBody = async (
  request: import('node:http').IncomingMessage
): Promise<Record<string, unknown>> => {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
};

const startCaptureServer = async (captureUrl: URL): Promise<Server> => {
  const basePath = captureUrl.pathname.replace(/\/$/, '');
  const server = createServer(async (request, response) => {
    const requestUrl = new URL(request.url || '/', captureUrl);
    if (request.method !== 'POST' || requestUrl.pathname !== `${basePath}/chat/completions`) {
      response.writeHead(404, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ error: 'not_found' }));
      return;
    }

    try {
      const body = await readRequestBody(request);
      upstreamCaptures.push({
        method: request.method,
        path: requestUrl.pathname,
        body,
      });
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(
        JSON.stringify({
          id: 'service-prompts-e2e',
          object: 'chat.completion',
          created: 1,
          model: body.model || 'service-prompts-e2e',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: 'deterministic translation',
              },
              finish_reason: 'stop',
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        })
      );
    } catch {
      response.writeHead(400, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ error: 'invalid_json' }));
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(Number(captureUrl.port), captureUrl.hostname, () => {
      server.off('error', reject);
      resolve();
    });
  });
  return server;
};

const closeCaptureServer = async (): Promise<void> => {
  if (!captureServer) return;
  await new Promise<void>((resolve, reject) => {
    captureServer?.close((error) => (error ? reject(error) : resolve()));
  });
  captureServer = null;
};

type ChatCapture = {
  body: Record<string, unknown>;
};

type WorkflowKind =
  | 'main-rag'
  | 'tab-chat'
  | 'document-chat'
  | 'sidepanel-rag'
  | 'normal-web'
  | 'compare-web';

type WorkflowRun = {
  kind: WorkflowKind;
  calls: ChatCapture[];
};

const MODEL_A = 'openai:gpt-4.1-mini';
const MODEL_B = 'openai:gpt-4.1';
const TRANSPORT_MODEL_A = 'gpt-4.1-mini';
const TRANSPORT_MODEL_B = 'gpt-4.1';
const RETRIEVED_CONTEXT = 'Deterministic retrieved context for Service Prompt E2E.';
const TAB_CONTEXT = 'Deterministic current-tab context for Service Prompt E2E.';
const SIDEPANEL_RESTORE_SENTINEL = 'Service Prompt Sidepanel restoration complete.';
const SEARCH_RESULT = {
  title: 'Service Prompt E2E result',
  url: 'https://example.test/service-prompts',
  content: 'Deterministic web-search evidence for Service Prompt E2E.',
};
const PRIOR_USER = 'What was established earlier?';
const PRIOR_ASSISTANT = 'The earlier deterministic answer.';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
};

const fulfillJson = async (route: Route, body: unknown, status = 200): Promise<void> => {
  if (route.request().method() === 'OPTIONS') {
    await route.fulfill({ status: 204, headers: corsHeaders });
    return;
  }
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers: corsHeaders,
    body: JSON.stringify(body),
  });
};

const installDeterministicWorkflowRoutes = async (
  context: BrowserContext,
  calls: ChatCapture[]
): Promise<void> => {
  await context.route('**/api/v1/llm/models/metadata**', (route) =>
    fulfillJson(route, {
      models: [
        {
          id: MODEL_A,
          model: MODEL_A,
          name: 'Service Prompt model A',
          provider: 'openai',
          apiProvider: 'custom-openai-api',
          capabilities: ['chat'],
          configured: true,
          available: true,
        },
        {
          id: MODEL_B,
          model: MODEL_B,
          name: 'Service Prompt model B',
          provider: 'openai',
          apiProvider: 'custom-openai-api',
          capabilities: ['chat'],
          configured: true,
          available: true,
        },
      ],
    })
  );
  await context.route('**/api/v1/llm/providers**', (route) =>
    fulfillJson(route, {
      providers: [
        {
          id: 'openai',
          name: 'OpenAI-compatible E2E',
          apiProvider: 'custom-openai-api',
          models: [MODEL_A, MODEL_B],
          configured: true,
          available: true,
        },
      ],
    })
  );
  await context.route('**/api/v1/rag/search', (route) =>
    fulfillJson(route, {
      results: [
        {
          id: 'service-prompt-rag-result',
          content: RETRIEVED_CONTEXT,
          metadata: {
            title: 'Service Prompt fixture',
            source: 'Service Prompt fixture',
            type: 'text',
            url: 'https://example.test/rag-source',
          },
          score: 1,
        },
      ],
    })
  );
  await context.route('**/api/v1/research/websearch', (route) =>
    fulfillJson(route, {
      web_search_results_dict: { results: [SEARCH_RESULT] },
    })
  );
  await context.route('**/api/v1/media/document-upload/preflight', (route) => {
    const payload =
      (route.request().postDataJSON() as {
        files?: Array<{
          client_id?: unknown;
          filename?: unknown;
          size_bytes?: unknown;
        }>;
      } | null) || {};
    return fulfillJson(route, {
      files: (payload.files || []).map((file) => ({
        client_id: String(file.client_id || ''),
        filename: String(file.filename || 'service-prompt-e2e.md'),
        media_type: 'document',
        default_mode: 'add_to_chat',
        modes: {
          add_to_chat: { available: true, status: 'available' },
          ingest_to_library: { available: false, status: 'unavailable' },
          ocr_pages: { available: false, status: 'unavailable' },
        },
        max_size_bytes: 1_000_000,
        max_pages: null,
        max_chat_tokens: 24_000,
        estimated_pages: null,
        estimated_tokens: 16,
        requires_send_time_estimate: false,
      })),
    });
  });
  await context.route('**/api/v1/chat/completions', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: corsHeaders });
      return;
    }
    const body = (route.request().postDataJSON() as Record<string, unknown> | null) || {};
    if (body.stream === false) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: corsHeaders,
        body: JSON.stringify({
          id: 'service-prompts-e2e-rewrite',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: 'Which prompt reached the workflow after the prior exchange?',
              },
              finish_reason: 'stop',
            },
          ],
        }),
      });
      calls.push({ body });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: {
        ...corsHeaders,
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
      body:
        `data: ${JSON.stringify({
          id: 'service-prompts-e2e',
          choices: [{ delta: { content: 'deterministic completion' } }],
        })}\n\n` + 'data: [DONE]\n\n',
    });
    calls.push({ body });
  });
};

const seedBrowserContext = async (
  context: BrowserContext,
  enableSidepanelRag: boolean
): Promise<void> => {
  const config = getRuntimeConfig();
  await context.addInitScript(
    ({ serverUrl, apiKey, selectedModel, enableSidepanelRag, restoreSentinel }) => {
      const tldwConfig = {
        serverUrl,
        authMode: 'single-user',
        apiKey,
        requestTimeoutMs: 60_000,
        chatRequestTimeoutMs: 60_000,
        chatStartupTimeoutMs: 30_000,
        chatStreamIdleTimeoutMs: 30_000,
      };
      localStorage.setItem('tldwConfig', JSON.stringify(tldwConfig));
      localStorage.setItem('serverUrl', serverUrl);
      localStorage.setItem('tldwServerUrl', serverUrl);
      localStorage.setItem('tldw-api-host', serverUrl);
      localStorage.setItem('authMode', 'single-user');
      localStorage.setItem('apiKey', apiKey);
      localStorage.setItem('isMigrated', 'true');
      localStorage.setItem('__tldw_first_run_complete', 'true');
      localStorage.setItem('assistant_setup_dismissed', 'true');
      localStorage.setItem('__tldw_test_bypass', 'true');
      localStorage.setItem('selectedModel', JSON.stringify(selectedModel));
      localStorage.setItem('sidepanelTemporaryChat', 'true');
      localStorage.setItem('ff_compareMode', 'true');
      localStorage.setItem('playgroundChatContextRailVisible', 'true');
      localStorage.setItem('playgroundChatRuntimeRailVisible', 'false');
      localStorage.setItem('plasmo-sync:sidepanelTemporaryChat', 'true');
      localStorage.setItem('plasmo-sync:ff_compareMode', 'true');
      localStorage.setItem('plasmo-sync:playgroundChatContextRailVisible', 'true');
      localStorage.setItem('plasmo-sync:playgroundChatRuntimeRailVisible', 'false');
      if (enableSidepanelRag) {
        localStorage.setItem('chatWithWebsiteEmbedding', 'true');
        localStorage.setItem('plasmo-sync:chatWithWebsiteEmbedding', 'true');
        const restoreTabId = 'service-prompt-restore-tab';
        localStorage.setItem(
          'sidepanelChatTabsState',
          JSON.stringify({
            tabs: [
              {
                id: restoreTabId,
                label: 'Restored service prompt chat',
                historyId: null,
                serverChatId: null,
                serverChatTopic: null,
                updatedAt: 1,
              },
            ],
            activeTabId: restoreTabId,
            snapshotsById: {
              [restoreTabId]: {
                history: [{ role: 'assistant', content: restoreSentinel }],
                messages: [
                  {
                    id: 'service-prompt-restore-ready',
                    isBot: true,
                    name: selectedModel,
                    message: restoreSentinel,
                    sources: [],
                    images: [],
                  },
                ],
                chatMode: 'normal',
                historyId: null,
                webSearch: false,
                toolChoice: 'none',
                selectedModel,
                selectedSystemPrompt: null,
                selectedQuickPrompt: null,
                temporaryChat: true,
                useOCR: false,
                serverChatId: null,
                serverChatState: null,
                serverChatTopic: null,
                serverChatClusterId: null,
                serverChatSource: null,
                serverChatExternalRef: null,
                queuedMessages: [],
                modelSettings: { apiProvider: 'custom-openai-api' },
              },
            },
          })
        );
      }

    },
    {
      serverUrl: config.serverUrl,
      apiKey: config.apiKey,
      selectedModel: MODEL_A,
      enableSidepanelRag,
      restoreSentinel: SIDEPANEL_RESTORE_SENTINEL,
    }
  );
};

const waitForWorkflowStateToRender = async (
  page: Page,
  kind: WorkflowKind,
  withHistory: boolean
): Promise<void> => {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      })
  );

  if (withHistory) {
    await expect(page.getByText(PRIOR_USER, { exact: true }).first()).toBeVisible({
      timeout: 30_000,
    });
  }

  if (kind === 'main-rag') {
    const rail = page
      .getByTestId('playground-cockpit-left-rail')
      .getByTestId('playground-context-rail');
    await expect(rail).toBeVisible({ timeout: 30_000 });
    await expect(rail).toContainText('Service Prompt E2E knowledge');
    return;
  }

  if (kind === 'tab-chat') {
    await expect(page.getByTestId('composer-context-strip').first()).toContainText(
      /Context\s*External/,
      { timeout: 30_000 }
    );
    return;
  }

  if (kind === 'document-chat') {
    const rail = page
      .getByTestId('playground-cockpit-left-rail')
      .getByTestId('playground-context-rail');
    await expect(rail).toBeVisible({ timeout: 30_000 });
    await expect(rail).toContainText('service-prompt-e2e.md');
    return;
  }

  if (kind === 'sidepanel-rag') {
    await page.keyboard.press('Control+e');
    await page.waitForFunction(
      () => {
        const testWindow = window as typeof window & {
          __tldw_useStoreMessageOption?: {
            getState?: () => { chatMode?: string };
          };
        };
        return testWindow.__tldw_useStoreMessageOption?.getState?.().chatMode === 'rag';
      },
      undefined,
      { timeout: 30_000 }
    );
    return;
  }

  if (kind === 'normal-web') {
    await expect(page.getByTestId('composer-context-strip').first()).toContainText(
      /Web search\s*On/,
      { timeout: 30_000 }
    );
    return;
  }

  await expect(page.getByTestId('composer-context-strip').first()).toContainText(
    /Compare\s*2 models/,
    { timeout: 30_000 }
  );
};

const priorMessages = [
  {
    isBot: false,
    name: 'You',
    message: PRIOR_USER,
    sources: [],
    images: [],
    id: 'service-prompt-prior-user',
  },
  {
    isBot: true,
    name: MODEL_A,
    message: PRIOR_ASSISTANT,
    sources: [],
    images: [],
    id: 'service-prompt-prior-assistant',
    parentMessageId: 'service-prompt-prior-user',
  },
];

const priorHistory = [
  { role: 'user', content: PRIOR_USER },
  { role: 'assistant', content: PRIOR_ASSISTANT },
];

const configureWorkflowState = async (
  page: Page,
  kind: WorkflowKind,
  withHistory: boolean
): Promise<void> => {
  await page.waitForFunction(
    () => {
      const testWindow = window as typeof window & {
        __tldw_useStoreMessageOption?: { setState?: unknown };
      };
      return Boolean(testWindow.__tldw_useStoreMessageOption?.setState);
    },
    undefined,
    { timeout: 60_000 }
  );
  await page.evaluate(
    ({ kind, modelA, modelB, seedMessages, seedHistory, tabContext }) => {
      type WritableTestStore = {
        setState: (state: Record<string, unknown>) => void;
      };
      const testWindow = window as typeof window & {
        __tldw_useStoreMessageOption: WritableTestStore;
        __tldw_useStoreChatModelSettings?: WritableTestStore;
      };
      if (kind === 'tab-chat' || kind === 'sidepanel-rag') {
        type ScriptTab = {
          id: number;
          title: string;
          url: string;
        };
        type ScriptBrowser = {
          scripting?: {
            executeScript: () => Promise<Array<{ result: Record<string, unknown> }>>;
          };
          tabs?: {
            query: (
              query: Record<string, unknown>,
              callback?: (tabs: ScriptTab[]) => void
            ) => Promise<ScriptTab[]>;
          };
        };
        const executeScript = async () => [
          {
            result:
              kind === 'tab-chat'
                ? {
                    html: `<main>${tabContext}</main>`,
                    title: 'Service Prompt fixture tab',
                    url: 'https://example.test/current-tab',
                    isPDF: false,
                  }
                : {
                    content: `<main>${tabContext}</main>`,
                    url: 'https://example.test/current-tab',
                    type: 'html',
                  },
          },
        ];
        const tabs: ScriptTab[] = [
          {
            id: 1,
            title: 'Service Prompt fixture tab',
            url: 'https://example.test/current-tab',
          },
        ];
        const queryTabs = async (
          _query: Record<string, unknown>,
          callback?: (items: ScriptTab[]) => void
        ) => {
          callback?.(tabs);
          return tabs;
        };
        const globals = globalThis as unknown as {
          browser?: ScriptBrowser;
          chrome?: ScriptBrowser;
        };
        for (const api of [globals.browser, globals.chrome]) {
          if (!api) continue;
          api.scripting = { executeScript };
          if (kind === 'sidepanel-rag') api.tabs = { query: queryTabs };
        }
      }
      const store = testWindow.__tldw_useStoreMessageOption;
      const baseState = {
        selectedModel: modelA,
        temporaryChat: true,
        historyId: 'temp',
        serverChatId: null,
        messages: seedMessages,
        history: seedHistory,
        streaming: false,
        isProcessing: false,
        chatMode: 'normal',
        webSearch: false,
        selectedKnowledge: null,
        fileRetrievalEnabled: false,
        ragMediaIds: null,
        documentContext: null,
        contextFiles: [],
        uploadedFiles: [],
        compareMode: false,
        compareSelectedModels: [],
      };
      const modeState: Record<string, unknown> = {};
      if (kind === 'main-rag') {
        Object.assign(modeState, {
          chatMode: 'rag',
          selectedKnowledge: {
            id: 'service-prompt-e2e-knowledge',
            title: 'Service Prompt E2E knowledge',
          },
        });
      } else if (kind === 'tab-chat') {
        Object.assign(modeState, {
          documentContext: [
            {
              type: 'tab',
              tabId: 1,
              title: 'Service Prompt fixture tab',
              url: 'https://example.test/current-tab',
            },
          ],
        });
      } else if (kind === 'normal-web') {
        Object.assign(modeState, { webSearch: true });
      } else if (kind === 'compare-web') {
        Object.assign(modeState, {
          webSearch: true,
          compareMode: true,
          compareSelectedModels: [modelA, modelB],
        });
      }
      store.setState({ ...baseState, ...modeState });
      testWindow.__tldw_useStoreChatModelSettings?.setState({
        apiProvider: 'custom-openai-api',
      });
    },
    {
      kind,
      modelA: MODEL_A,
      modelB: MODEL_B,
      seedMessages: withHistory ? priorMessages : [],
      seedHistory: withHistory ? priorHistory : [],
      tabContext: TAB_CONTEXT,
    }
  );
};

const runWorkflow = async (
  browser: Browser,
  kind: WorkflowKind,
  question: string,
  options: { withHistory?: boolean; expectedCalls?: number } = {}
): Promise<WorkflowRun> => {
  const context = await browser.newContext({
    baseURL: getRuntimeConfig().webUrl,
  });
  const calls: ChatCapture[] = [];
  await seedBrowserContext(context, kind === 'sidepanel-rag');
  await installDeterministicWorkflowRoutes(context, calls);
  const page = await context.newPage();
  try {
    const isSidepanel = kind === 'sidepanel-rag';
    await page.goto(isSidepanel ? '/__debug__/sidepanel-chat?nextgenComposer=1' : '/chat', {
      waitUntil: 'domcontentloaded',
    });
    const withHistory = options.withHistory === true;
    if (isSidepanel) {
      await expect(page.getByText(SIDEPANEL_RESTORE_SENTINEL, { exact: true })).toBeVisible({
        timeout: 30_000,
      });
    }
    const input = page.getByTestId('chat-input').first();
    await expect(input).toBeVisible({ timeout: 60_000 });
    await configureWorkflowState(page, kind, withHistory);
    if (kind === 'document-chat') {
      await page.locator('#document-upload').setInputFiles({
        name: 'service-prompt-e2e.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('Deterministic document context.'),
      });
    }
    await waitForWorkflowStateToRender(page, kind, withHistory);
    await input.fill(question);

    const send = isSidepanel
      ? page.getByRole('button', { name: 'Send message' }).first()
      : page.getByTestId('composer-inline-send-control').getByRole('button').first();
    await expect(send).toBeVisible({ timeout: 30_000 });
    await send.click();
    await expect
      .poll(() => calls.length, { timeout: 60_000 })
      .toBeGreaterThanOrEqual(options.expectedCalls ?? 1);
    return { kind, calls };
  } finally {
    await context.close();
  }
};

const contentToText = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((part) =>
      part && typeof part === 'object' && 'text' in part
        ? String((part as { text?: unknown }).text || '')
        : ''
    )
    .join('');
};

const payloadTexts = (capture: ChatCapture | UpstreamCapture): string[] => {
  const messages = capture.body.messages;
  return Array.isArray(messages)
    ? messages.map((message) =>
        contentToText(
          message && typeof message === 'object' ? (message as { content?: unknown }).content : ''
        )
      )
    : [];
};

const captureText = (capture: ChatCapture | UpstreamCapture): string =>
  payloadTexts(capture).join('\n');

const replaceTemplateValues = (template: string, values: Record<string, string>): string => {
  let rendered = template;
  for (const [key, value] of Object.entries(values)) {
    rendered = rendered.replaceAll(`{${key}}`, value);
  }
  return rendered;
};

const assertMarkerRun = (run: WorkflowRun, marker: string): void => {
  const matching = run.calls.filter((call) => captureText(call).includes(marker));
  expect(matching, `${run.kind} should emit its customized marker`).toHaveLength(1);
};

const expectedAnswerContext = (kind: WorkflowKind): string => {
  switch (kind) {
    case 'main-rag':
    case 'document-chat':
    case 'sidepanel-rag':
      return `<doc id='0'>${RETRIEVED_CONTEXT}</doc>`;
    case 'tab-chat':
      return `# Service Prompt fixture tab (example.test) \n\n${TAB_CONTEXT}`;
    default:
      throw new Error(`Unexpected answer workflow: ${kind}`);
  }
};

const assertCustomAnswerRun = (run: WorkflowRun, marker: string, question: string): void => {
  expect(
    run.calls.flatMap(payloadTexts),
    `${run.kind} should render its exact customized answer context`
  ).toContain(`${marker}\nContext=${expectedAnswerContext(run.kind)}\nQuestion=${question}`);
};

const assertPackagedAnswerRun = (
  run: WorkflowRun,
  question: string,
  removedMarker: string
): void => {
  expect(run.calls.map(captureText).join('\n')).not.toContain(removedMarker);
  const template = golden.defaults['chat.rag.answer'].template;
  const context = expectedAnswerContext(run.kind);
  expect(
    run.calls.flatMap(payloadTexts),
    `${run.kind} should emit the exact packaged answer context`
  ).toContain(
    replaceTemplateValues(template, { context, question })
  );
};

const expectedRewriteHistory = (kind: WorkflowKind, question: string): string[] => {
  const priorHistory = [`Human: ${PRIOR_USER}`, `Assistant: ${PRIOR_ASSISTANT}`];
  switch (kind) {
    case 'main-rag':
      return priorHistory;
    case 'document-chat':
    case 'sidepanel-rag':
      return [...priorHistory, `Human: ${question}`];
    default:
      throw new Error(`Unexpected rewrite workflow: ${kind}`);
  }
};

const assertCustomRewriteRun = (run: WorkflowRun, marker: string, question: string): void => {
  const history = expectedRewriteHistory(run.kind, question).join('\n');
  expect(
    run.calls.flatMap(payloadTexts),
    `${run.kind} should render its exact customized rewrite history`
  ).toContain(`${marker}\nHistory=${history}\nQuestion=${question}`);
};

const assertPackagedRewriteRun = (
  run: WorkflowRun,
  question: string,
  removedMarker: string
): void => {
  expect(run.calls.map(captureText).join('\n')).not.toContain(removedMarker);
  const history = expectedRewriteHistory(run.kind, question);
  const priorHistory = [`Human: ${PRIOR_USER}`, `Assistant: ${PRIOR_ASSISTANT}`];
  const alternateHistory =
    run.kind === 'main-rag' ? [...priorHistory, `Human: ${question}`] : priorHistory;
  const template = golden.defaults['chat.rag.question_rewrite'].template;
  const expectedPrompt = replaceTemplateValues(template, {
    chat_history: history.join('\n'),
    question,
  });
  const alternatePrompt = replaceTemplateValues(template, {
    chat_history: alternateHistory.join('\n'),
    question,
  });
  const payloads = run.calls.flatMap(payloadTexts);
  expect(payloads, `${run.kind} should emit its exact packaged rewrite template`).toContain(
    expectedPrompt
  );
  expect(payloads).not.toContain(alternatePrompt);
};

const extractCustomWebSearchResults = (capture: ChatCapture, marker: string): string => {
  const candidate = payloadTexts(capture).find((text) => text.includes(marker));
  expect(candidate, 'custom web prompt should reach the chat request').toBeDefined();
  const resultsDelimiter = '\nResults=';
  const resultsStart = candidate!.indexOf(resultsDelimiter, candidate!.indexOf(marker));
  expect(resultsStart, 'custom web prompt should render search_results').toBeGreaterThanOrEqual(0);
  return candidate!.slice(resultsStart + resultsDelimiter.length);
};

const assertPackagedWebRun = (
  run: WorkflowRun,
  removedMarker: string,
  formattedSearchResults: string
): void => {
  const combined = run.calls.map(captureText).join('\n');
  expect(combined).not.toContain(removedMarker);
  const template = golden.defaults['chat.web_search.answer'].template;
  const [prefix, afterDate] = template.split('{current_date_time}');
  const [between] = afterDate.split('{search_results}');

  for (const call of run.calls) {
    const messages = Array.isArray(call.body.messages) ? call.body.messages : [];
    const candidate = messages
      .filter(
        (message) =>
          message &&
          typeof message === 'object' &&
          (message as { role?: unknown }).role === 'system'
      )
      .map((message) => contentToText((message as { content?: unknown }).content))
      .find(
        (text) => text.includes('<search-results>') && text.includes(SEARCH_RESULT.url)
      );
    expect(candidate, `${run.kind} branch should emit packaged web prompt`).toBeDefined();
    expect(candidate!.startsWith(prefix)).toBe(true);
    const betweenStart = candidate!.indexOf(between, prefix.length);
    expect(betweenStart).toBeGreaterThanOrEqual(prefix.length);
    const currentDateTime = candidate!.slice(prefix.length, betweenStart);
    expect(currentDateTime).toMatch(
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/
    );
    expect(candidate).toBe(
      replaceTemplateValues(template, {
        current_date_time: currentDateTime,
        search_results: formattedSearchResults,
      }).trim()
    );
  }
};

const callTranslation = async (
  request: APIRequestContext,
  text: string,
  targetLanguage: string
): Promise<void> => {
  const response = await apiJson<{
    translated_text: string;
    target_language: string;
  }>(request, 'POST', '/api/v1/translate', {
    text,
    target_language: targetLanguage,
    source_language: 'English',
    provider: 'custom-openai-api',
    model: 'service-prompts-e2e-model',
  });
  expect(response.translated_text).toBe('deterministic translation');
  expect(response.target_language).toBe(targetLanguage);
};

const assertTranslationCapture = (
  capture: UpstreamCapture,
  expectedSystem: string,
  expectedUser: string,
  removedMarker?: string
): void => {
  const captureBase = getRuntimeConfig().captureUrl.pathname.replace(/\/$/, '');
  expect(capture.method).toBe('POST');
  expect(capture.path).toBe(`${captureBase}/chat/completions`);
  const messages = Array.isArray(capture.body.messages)
    ? (capture.body.messages as Array<Record<string, unknown>>)
    : [];
  const system = messages.find((message) => message.role === 'system');
  const user = messages.find((message) => message.role === 'user');
  expect(contentToText(system?.content)).toBe(expectedSystem);
  expect(contentToText(user?.content)).toBe(expectedUser);
  if (removedMarker) {
    expect(captureText(capture)).not.toContain(removedMarker);
  }
};

test.beforeAll(async () => {
  runtimeConfig = loadRuntimeConfig();
  await assertHealth(getRuntimeConfig().webUrl);
  await assertHealth(`${getRuntimeConfig().serverUrl}/api/v1/health`, {
    'X-API-KEY': getRuntimeConfig().apiKey,
  });
  await assertHealth(`${getRuntimeConfig().serverUrl}/api/v1/service-prompts`, {
    'X-API-KEY': getRuntimeConfig().apiKey,
  });
  captureServer = await startCaptureServer(getRuntimeConfig().captureUrl);
});

test.afterEach(async ({ request }) => {
  await cleanupTouchedDefinitions(request);
  upstreamCaptures.splice(0);
});

test.afterAll(async () => {
  await closeCaptureServer();
});

test.describe.serial('service prompt runtime propagation', () => {
  test('chat.rag.answer reaches Main, Tab, Document, and legacy Sidepanel RAG', async ({
    browser,
    request,
  }) => {
    test.setTimeout(300_000);
    const marker = `SERVICE_PROMPT_ANSWER_${Date.now()}`;
    const question = 'Which answer prompt reached this workflow?';
    const kinds: WorkflowKind[] = ['main-rag', 'tab-chat', 'document-chat', 'sidepanel-rag'];

    await saveDefinition(request, 'chat.rag.answer', {
      template: `${marker}\nContext={context}\nQuestion={question}`,
    });
    for (const kind of kinds) {
      const run = await runWorkflow(browser, kind, question);
      assertCustomAnswerRun(run, marker, question);
    }

    await resetDefinition(request, 'chat.rag.answer');
    for (const kind of kinds) {
      const run = await runWorkflow(browser, kind, question);
      assertPackagedAnswerRun(run, question, marker);
    }
  });

  test('chat.rag.question_rewrite reaches Main, Document, and legacy Sidepanel RAG', async ({
    browser,
    request,
  }) => {
    test.setTimeout(300_000);
    const marker = `SERVICE_PROMPT_REWRITE_${Date.now()}`;
    const question = 'What did the prior exchange establish?';
    const kinds: WorkflowKind[] = ['main-rag', 'document-chat', 'sidepanel-rag'];

    await saveDefinition(request, 'chat.rag.question_rewrite', {
      template: `${marker}\nHistory={chat_history}\nQuestion={question}`,
    });
    for (const kind of kinds) {
      const run = await runWorkflow(browser, kind, question, {
        withHistory: true,
        expectedCalls: 2,
      });
      assertCustomRewriteRun(run, marker, question);
    }

    await resetDefinition(request, 'chat.rag.question_rewrite');
    for (const kind of kinds) {
      const run = await runWorkflow(browser, kind, question, {
        withHistory: true,
        expectedCalls: 2,
      });
      assertPackagedRewriteRun(run, question, marker);
    }
  });

  test('chat.web_search.answer reaches normal Chat and every Compare branch', async ({
    browser,
    request,
  }) => {
    test.setTimeout(300_000);
    const marker = `SERVICE_PROMPT_WEB_${Date.now()}`;
    const question = 'What does the deterministic web result say?';

    await saveDefinition(request, 'chat.web_search.answer', {
      template: `${marker}\nAt={current_date_time}\nResults={search_results}`,
    });
    const customNormal = await runWorkflow(browser, 'normal-web', question);
    const customCompare = await runWorkflow(browser, 'compare-web', question, {
      expectedCalls: 2,
    });
    assertMarkerRun(customNormal, marker);
    expect(customCompare.calls).toHaveLength(2);
    const formattedSearchResults = extractCustomWebSearchResults(customNormal.calls[0]!, marker);
    expect(formattedSearchResults).toContain(SEARCH_RESULT.title);
    expect(formattedSearchResults).toContain(SEARCH_RESULT.url);
    expect(formattedSearchResults).toContain(SEARCH_RESULT.content);
    for (const call of customCompare.calls) {
      expect(captureText(call)).toContain(marker);
      expect(extractCustomWebSearchResults(call, marker)).toBe(formattedSearchResults);
    }
    expect(customCompare.calls.map((call) => String(call.body.model)).sort()).toEqual(
      [TRANSPORT_MODEL_A, TRANSPORT_MODEL_B].sort()
    );

    await resetDefinition(request, 'chat.web_search.answer');
    const packagedNormal = await runWorkflow(browser, 'normal-web', question);
    const packagedCompare = await runWorkflow(browser, 'compare-web', question, {
      expectedCalls: 2,
    });
    expect(packagedCompare.calls).toHaveLength(2);
    assertPackagedWebRun(packagedNormal, marker, formattedSearchResults);
    assertPackagedWebRun(packagedCompare, marker, formattedSearchResults);
  });

  test('media.text.translation reaches the configured OpenAI-compatible upstream', async ({
    request,
  }) => {
    test.setTimeout(120_000);
    const marker = `SERVICE_PROMPT_TRANSLATION_${Date.now()}`;
    const sourceText = 'Hello from the Service Prompt runtime matrix.';
    const targetLanguage = 'French';
    const customSystem = `${marker}:SYSTEM`;
    const customUserTemplate = `${marker}:USER target={target_language} text={text}`;

    await saveDefinition(request, 'media.text.translation', {
      system: customSystem,
      user_template: customUserTemplate,
    });
    upstreamCaptures.splice(0);
    await callTranslation(request, sourceText, targetLanguage);
    expect(upstreamCaptures).toHaveLength(1);
    assertTranslationCapture(
      upstreamCaptures[0],
      customSystem,
      replaceTemplateValues(customUserTemplate, {
        target_language: targetLanguage,
        text: sourceText,
      })
    );

    await resetDefinition(request, 'media.text.translation');
    upstreamCaptures.splice(0);
    await callTranslation(request, sourceText, targetLanguage);
    expect(upstreamCaptures).toHaveLength(1);
    assertTranslationCapture(
      upstreamCaptures[0],
      golden.defaults['media.text.translation'].system,
      replaceTemplateValues(golden.defaults['media.text.translation'].user_template, {
        target_language: targetLanguage,
        text: sourceText,
      }),
      marker
    );
  });
});
