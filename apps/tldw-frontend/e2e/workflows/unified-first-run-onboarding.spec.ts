import type { Page, Route } from '@playwright/test';
import { assertNoCriticalErrors, expect, test } from '../utils/fixtures';
import { seedAuth, stubNotificationsApi } from '../utils/helpers';

type FirstRunStatus =
  | 'not_started'
  | 'in_progress'
  | 'skipped'
  | 'first_chat_complete'
  | 'completed';

type FirstRunState = {
  status: FirstRunStatus;
  current_step: string | null;
  completed_steps: string[];
  skipped_steps: string[];
  step_data: Record<string, Record<string, unknown>>;
  acknowledged_steps: string[];
  first_chat: {
    completed: boolean;
    provider: string | null;
    model: string | null;
    response_id: string | null;
    completed_at: string | null;
  };
  skip_reason: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

const corsHeaders = {
  'access-control-allow-origin': '*',
  'access-control-allow-headers': '*',
  'access-control-allow-methods': 'GET,POST,OPTIONS',
};

const json = async (route: Route, body: unknown, status = 200): Promise<void> => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers: corsHeaders,
    body: JSON.stringify(body),
  });
};

const nowIso = () => '2026-05-31T12:00:00Z';

const unique = (values: string[]) => Array.from(new Set(values));

const createFirstRunState = (patch: Partial<FirstRunState> = {}): FirstRunState => ({
  status: 'not_started',
  current_step: 'setup_path',
  completed_steps: [],
  skipped_steps: [],
  step_data: {},
  acknowledged_steps: [],
  first_chat: {
    completed: false,
    provider: null,
    model: null,
    response_id: null,
    completed_at: null,
  },
  skip_reason: null,
  created_at: nowIso(),
  updated_at: nowIso(),
  completed_at: null,
  ...patch,
});

function requestJson(route: Route): Record<string, unknown> {
  try {
    return route.request().postDataJSON() as Record<string, unknown>;
  } catch {
    return {};
  }
}

async function prepareFirstRunPage(page: Page): Promise<void> {
  await seedAuth(page);
  await stubNotificationsApi(page);
  await page.addInitScript(() => {
    localStorage.removeItem('__tldw_first_run_complete');
    localStorage.removeItem('tldw:first-source-milestone-dismissed');
  });
}

async function installUnifiedFirstRunApi(
  page: Page,
  initialState: FirstRunState = createFirstRunState()
) {
  let state = initialState;
  const setupMutations: Array<{ path: string; body: Record<string, unknown> }> = [];
  const firstChatRequests: Record<string, unknown>[] = [];
  const completeRequests: Record<string, unknown>[] = [];

  const markStepComplete = (step: string, data: Record<string, unknown> = {}) => {
    const completedStep = step === 'providers' ? 'providers' : step;
    state = {
      ...state,
      status: state.status === 'completed' ? 'completed' : 'in_progress',
      current_step: completedStep,
      completed_steps: unique([...state.completed_steps, completedStep]),
      acknowledged_steps: data.acknowledged
        ? unique([...state.acknowledged_steps, completedStep])
        : state.acknowledged_steps,
      step_data: {
        ...state.step_data,
        [completedStep]: data,
      },
      updated_at: nowIso(),
    };
  };

  await page.route(/\/api\/v1\/.*(?:\?.*)?$/, async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

    if (method === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: corsHeaders });
      return;
    }

    if (path === '/api/v1/health' || path === '/api/v1/health/live') {
      await json(route, { status: 'ok', version: 'e2e' });
      return;
    }

    if (path === '/api/v1/setup/first-run/state' && method === 'GET') {
      await json(route, state);
      return;
    }

    if (path === '/api/v1/setup/first-run/metadata' && method === 'GET') {
      await json(route, {
        auth_mode: 'single_user',
        bundled_single_user_auth_available: true,
        manual_auth_required: false,
        setup_required: state.status !== 'completed',
        setup_completed: state.status === 'completed',
        remote_setup_enabled: false,
        connection: {
          frontend_origin: 'http://localhost:8080',
          api_origin: 'http://127.0.0.1:8000',
          browser_access: 'local',
        },
        setup_paths: [
          { key: 'docker_single_user', label: 'Solo, Docker' },
          { key: 'local_single_user', label: 'Solo, local install' },
        ],
        multi_user_exit: { guide_path: '/Docs/AuthNZ/Multi_User_Setup.md' },
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/state' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      const step = String(body.step || '');
      const data =
        body.data && typeof body.data === 'object' && !Array.isArray(body.data)
          ? (body.data as Record<string, unknown>)
          : {};
      markStepComplete(step, data);
      await json(route, state);
      return;
    }

    if (path === '/api/v1/setup/first-run/skip' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      state = {
        ...state,
        status: 'skipped',
        current_step: null,
        skip_reason: String(body.reason || 'user_skip'),
        updated_at: nowIso(),
      };
      await json(route, state);
      return;
    }

    if (path === '/api/v1/setup/first-run/providers/catalog' && method === 'GET') {
      await json(route, {
        providers: [
          {
            provider_key: 'openai',
            label: 'OpenAI',
            provider_type: 'hosted_api_key',
            supports_preflight: true,
            recommended_for_first_chat: true,
            model_field: 'gpt-4.1-mini',
          },
          {
            provider_key: 'ollama',
            label: 'Ollama',
            provider_type: 'local_endpoint',
            supports_preflight: true,
            recommended_for_first_chat: false,
            default_base_url: 'http://127.0.0.1:11434/v1',
            model_field: 'llama3.1',
          },
        ],
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/providers' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      await json(route, {
        provider_key: body.provider_key,
        status: 'saved',
        masked_api_key: 'sk-...test',
        default_model: body.model,
        requires_restart: false,
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/providers/validate' && method === 'POST') {
      const body = requestJson(route);
      await json(route, {
        provider_key: body.provider_key,
        status: 'accepted',
        validation_level: 'local_syntax',
        can_gate_first_chat: true,
        models: [],
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/ingest-defaults' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      markStepComplete('ingest_defaults', body);
      await json(route, {
        status: 'saved',
        step: 'ingest_defaults',
        requires_restart: false,
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/audio-defaults' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      markStepComplete('audio_defaults', body);
      await json(route, {
        status: 'saved',
        step: 'audio_defaults',
        requires_restart: false,
      });
      return;
    }

    if (path === '/api/v1/setup/audio/recommendations' && method === 'GET') {
      await json(route, { recommendations: [] });
      return;
    }

    if (path === '/api/v1/setup/first-run/optional-advanced' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      markStepComplete('optional_advanced', body);
      await json(route, {
        status: 'saved',
        step: 'optional_advanced',
        requires_restart: false,
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/first-chat' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      firstChatRequests.push(body);
      state = {
        ...state,
        status: 'first_chat_complete',
        first_chat: {
          completed: true,
          provider: String(body.provider || 'openai'),
          model: String(body.model || 'gpt-4.1-mini'),
          response_id: 'chatcmpl-e2e-first-run',
          completed_at: nowIso(),
        },
        updated_at: nowIso(),
      };
      await json(route, {
        status: 'ready',
        provider: body.provider,
        model: body.model,
        response_id: 'chatcmpl-e2e-first-run',
        response_text: 'Hello from the mocked first chat.',
      });
      return;
    }

    if (path === '/api/v1/setup/first-run/complete' && method === 'POST') {
      const body = requestJson(route);
      setupMutations.push({ path, body });
      completeRequests.push(body);
      state = {
        ...state,
        status: 'completed',
        current_step: null,
        completed_steps: unique([...state.completed_steps, 'first_chat']),
        acknowledged_steps: unique([...state.acknowledged_steps, 'first_chat']),
        completed_at: nowIso(),
        updated_at: nowIso(),
      };
      await json(route, {
        success: true,
        message: 'First-run setup completed.',
        requires_restart: false,
        install_plan_submitted: false,
      });
      return;
    }

    if (method === 'GET' && path === '/api/v1/llm/providers') {
      await json(route, { providers: [] });
      return;
    }

    await json(route, method === 'GET' ? {} : { status: 'ok' });
  });

  return {
    get state() {
      return state;
    },
    setupMutations,
    firstChatRequests,
    completeRequests,
  };
}

test.describe('unified first-run onboarding', () => {
  test('uses a focused setup shell until the user explicitly skips', async ({
    page,
    diagnostics,
  }) => {
    await prepareFirstRunPage(page);
    const mock = await installUnifiedFirstRunApi(page);

    await page.goto('/');

    await expect(page.getByRole('heading', { name: /first-time setup/i })).toBeVisible();
    await expect(page.getByTestId('chat-toggle-shortcuts')).toHaveCount(0);

    await page.getByRole('button', { name: /skip for now/i }).click();

    await expect(page.getByRole('heading', { name: /first-time setup/i })).toHaveCount(0);
    await expect(page.getByTestId('chat-toggle-shortcuts')).toBeVisible();
    expect(mock.state.status).toBe('skipped');
    expect(
      mock.setupMutations.some((request) => request.path === '/api/v1/setup/first-run/skip')
    ).toBe(true);

    await assertNoCriticalErrors(diagnostics);
  });

  test('shows the first-source milestone after completed setup without backend mutation on dismiss', async ({
    page,
    diagnostics,
  }) => {
    await prepareFirstRunPage(page);
    const mock = await installUnifiedFirstRunApi(
      page,
      createFirstRunState({
        status: 'completed',
        current_step: null,
        completed_steps: ['first_chat'],
        acknowledged_steps: ['first_chat'],
        first_chat: {
          completed: true,
          provider: 'openai',
          model: 'gpt-4.1-mini',
          response_id: 'chatcmpl-existing',
          completed_at: nowIso(),
        },
        completed_at: nowIso(),
      })
    );

    await page.goto('/');

    await expect(page.getByTestId('chat-toggle-shortcuts')).toBeVisible();
    await expect(page.getByRole('heading', { name: /add your first source/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /add source/i })).toBeVisible();

    await page.getByRole('button', { name: /dismiss/i }).click();

    await expect(page.getByRole('heading', { name: /add your first source/i })).toHaveCount(0);
    expect(mock.setupMutations).toHaveLength(0);

    await assertNoCriticalErrors(diagnostics);
  });

  test('requires a successful backend first-chat response before completion', async ({
    page,
    diagnostics,
  }) => {
    await prepareFirstRunPage(page);
    const mock = await installUnifiedFirstRunApi(page);

    await page.goto('/');

    await page.getByRole('button', { name: /solo, docker/i }).click();
    await expect(page.getByRole('heading', { name: /privacy and security/i })).toBeVisible();
    await page.getByLabel(/i understand local or remote setup access/i).check();
    await page.getByRole('button', { name: /^continue$/i }).click();

    await expect(page.getByRole('heading', { name: /chat provider/i })).toBeVisible();
    await page.getByLabel(/select openai/i).check();
    await page.getByLabel(/openai api key/i).fill('sk-test-onboarding');
    await page.getByLabel(/default model/i).fill('gpt-4.1-mini');
    await page.getByRole('button', { name: /validate openai/i }).click();
    await expect(page.getByText(/first chat verifies this provider/i)).toBeVisible();
    await page.getByRole('button', { name: /save provider/i }).click();
    await expect(page.getByText(/saved as sk-\.\.\.test/i)).toBeVisible();
    await page.getByRole('button', { name: /^continue$/i }).click();

    await expect(page.getByRole('heading', { name: /ingest defaults/i })).toBeVisible();
    await page.getByRole('button', { name: /^continue$/i }).click();

    await expect(page.getByRole('heading', { name: /audio, stt, and tts/i })).toBeVisible();
    await page.getByRole('button', { name: /^continue$/i }).click();

    await expect(page.getByRole('heading', { name: /optional advanced setup/i })).toBeVisible();
    await page.getByRole('button', { name: /^continue$/i }).click();

    await expect(page.getByRole('heading', { name: /first chat/i })).toBeVisible();
    expect(mock.completeRequests).toHaveLength(0);

    await page.getByRole('button', { name: /send test chat/i }).click();

    await expect(page.getByRole('heading', { name: /add your first source/i })).toBeVisible();
    await expect(page.getByRole('radio', { name: /web url/i })).toBeChecked();
    await page.getByRole('button', { name: /add source/i }).click();
    const firstSourceDetail = await page.evaluate(
      () => (window as any).__tldwPendingQuickIngestOpen?.detail
    );
    expect(firstSourceDetail).toMatchObject({
      source: 'first_source_milestone',
      firstSource: true,
      firstSourceKind: 'web_url',
    });
    expect(mock.firstChatRequests).toHaveLength(1);
    expect(mock.firstChatRequests[0]).toMatchObject({
      provider: 'openai',
      model: 'gpt-4.1-mini',
    });
    expect(mock.completeRequests).toHaveLength(1);
    expect(mock.state.status).toBe('completed');

    await assertNoCriticalErrors(diagnostics);
  });
});
