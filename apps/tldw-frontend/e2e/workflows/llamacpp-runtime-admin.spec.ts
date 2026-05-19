import type { Page, Route } from '@playwright/test';
import { test, expect, assertNoCriticalErrors } from '../utils/fixtures';
import { AdminPage } from '../utils/page-objects';

const fulfillJson = async (
  route: Route,
  data: unknown,
  status = 200
): Promise<void> => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(data),
  });
};

const savedConfig = {
  enabled: true,
  executable_path: '/opt/llama-server',
  models_dir: '/srv/models',
  default_host: '127.0.0.1',
  default_port: 8080,
  default_threads: 8,
  default_n_gpu_layers: 0,
  default_ctx_size: 4096,
  allow_unvalidated_args: false,
  allow_cli_secrets: false,
  port_autoselect: true,
  port_probe_max: 10,
  allowed_paths: ['/srv/models'],
  registered_model_paths: [],
  imported_asset_folders: ['/srv/models/imported'],
  log_output_file: null,
};

const ggufAsset = {
  asset_id: 'gguf:toy-chat',
  kind: 'gguf',
  identity_basis: 'resolved_path',
  path: '/srv/models/toy-chat.gguf',
  resolved_path: '/srv/models/toy-chat.gguf',
  display_name: 'Toy Chat GGUF',
  source: 'models_dir',
  size_bytes: 2_200_000_000,
  modified_at: '2026-05-18T10:00:00Z',
  metadata: {
    quantization: 'Q4_K_M',
    parameter_hint: '7B',
    context_hint: 4096,
    family_hint: 'toy',
  },
  capabilities: ['chat'],
  mmproj_asset_ids: ['mmproj:toy-vision'],
  base_model_asset_ids: [],
  warnings: ['Filename metadata only.'],
};

const mmprojAsset = {
  asset_id: 'mmproj:toy-vision',
  kind: 'mmproj',
  identity_basis: 'resolved_path',
  path: '/srv/models/toy-mmproj.gguf',
  resolved_path: '/srv/models/toy-mmproj.gguf',
  display_name: 'Toy Vision Projector',
  source: 'models_dir',
  size_bytes: 320_000_000,
  modified_at: '2026-05-18T10:05:00Z',
  metadata: {
    family_hint: 'toy',
  },
  capabilities: ['vision'],
  mmproj_asset_ids: [],
  base_model_asset_ids: ['gguf:toy-chat'],
  warnings: [],
};

const profiles = [
  {
    profile_id: 'chat-runtime',
    name: 'Chat runtime',
    enabled: true,
    mode: 'chat',
    model_id: 'gguf:toy-chat',
    model_path: null,
    mmproj_model_id: null,
    mmproj_path: null,
    mmproj_display_name: null,
    capabilities: { chat: true },
    modalities: { input: ['text'], output: ['text'] },
    capability_warnings: [],
    host: '127.0.0.1',
    port: 8181,
    port_policy: 'explicit',
    server_args: { ctx_size: 4096 },
    autostart: true,
    restart_policy: { max_restarts: 2 },
    provider_alias: 'llamacpp-chat-runtime',
    tags: ['chat'],
  },
  {
    profile_id: 'vision-runtime',
    name: 'Vision runtime',
    enabled: true,
    mode: 'vision',
    model_id: 'gguf:toy-chat',
    model_path: null,
    mmproj_model_id: 'mmproj:toy-vision',
    mmproj_path: '/srv/models/toy-mmproj.gguf',
    mmproj_display_name: 'Toy Vision Projector',
    capabilities: { chat: true, vision: true },
    modalities: { input: ['text', 'image'], output: ['text'] },
    capability_warnings: ['Projector pairing is inferred from local assets.'],
    host: '127.0.0.1',
    port: 8182,
    port_policy: 'explicit',
    server_args: {},
    autostart: false,
    restart_policy: { max_restarts: 1 },
    provider_alias: 'llamacpp-vision-runtime',
    tags: ['vision'],
  },
  {
    profile_id: 'embedding-runtime',
    name: 'Embedding runtime',
    enabled: true,
    mode: 'embedding',
    model_id: 'gguf:toy-chat',
    model_path: null,
    mmproj_model_id: null,
    mmproj_path: null,
    mmproj_display_name: null,
    capabilities: { embeddings: true },
    modalities: { input: ['text'], output: ['embedding'] },
    capability_warnings: [],
    host: '127.0.0.1',
    port: 8183,
    port_policy: 'explicit',
    server_args: { embedding: true },
    autostart: false,
    restart_policy: { max_restarts: 1 },
    provider_alias: 'llamacpp-embedding-runtime',
    tags: ['embedding'],
  },
];

const runtimes = [
  {
    profile_id: 'chat-runtime',
    state: 'running',
    pid: 4242,
    host: '127.0.0.1',
    port: 8181,
    endpoint: 'http://127.0.0.1:8181',
    model_id: 'gguf:toy-chat',
    model_path: '/srv/models/toy-chat.gguf',
    mmproj_model_id: null,
    mmproj_path: null,
    mmproj_display_name: null,
    capabilities: { chat: true },
    modalities: { input: ['text'], output: ['text'] },
    capability_warnings: [],
    resolved_args: ['--model', '/srv/models/toy-chat.gguf', '--port', '8181'],
    started_at: '2026-05-18T10:10:00Z',
    stopped_at: null,
    last_health_at: '2026-05-18T10:11:00Z',
    restart_count: 0,
    next_restart_at: null,
    exit_code: null,
    last_error: null,
    log_tail_available: true,
    warnings: ['GPU probe unavailable.'],
    health: { ok: true },
    message: null,
  },
  {
    profile_id: 'vision-runtime',
    state: 'stopped',
    pid: null,
    host: '127.0.0.1',
    port: 8182,
    endpoint: null,
    model_id: 'gguf:toy-chat',
    model_path: '/srv/models/toy-chat.gguf',
    mmproj_model_id: 'mmproj:toy-vision',
    mmproj_path: '/srv/models/toy-mmproj.gguf',
    mmproj_display_name: 'Toy Vision Projector',
    capabilities: { chat: true, vision: true },
    modalities: { input: ['text', 'image'], output: ['text'] },
    capability_warnings: ['Projector pairing is inferred from local assets.'],
    resolved_args: [],
    started_at: null,
    stopped_at: '2026-05-18T10:12:00Z',
    last_health_at: null,
    restart_count: 0,
    next_restart_at: null,
    exit_code: 0,
    last_error: null,
    log_tail_available: false,
    warnings: ['Hardware fit warning only.'],
    health: {},
    message: 'Stopped by operator.',
  },
  {
    profile_id: 'embedding-runtime',
    state: 'failed',
    pid: null,
    host: '127.0.0.1',
    port: 8183,
    endpoint: null,
    model_id: 'gguf:toy-chat',
    model_path: '/srv/models/toy-chat.gguf',
    mmproj_model_id: null,
    mmproj_path: null,
    mmproj_display_name: null,
    capabilities: { embeddings: true },
    modalities: { input: ['text'], output: ['embedding'] },
    capability_warnings: [],
    resolved_args: [],
    started_at: '2026-05-18T10:13:00Z',
    stopped_at: '2026-05-18T10:14:00Z',
    last_health_at: null,
    restart_count: 2,
    next_restart_at: null,
    exit_code: 1,
    last_error: 'llama-server exited with code 1',
    log_tail_available: true,
    warnings: ['Restart limit reached.'],
    health: { ok: false },
    message: 'Startup failed.',
  },
];

async function mockManagedRuntimeAdmin(page: Page): Promise<{
  useInChatProfileIds: string[];
}> {
  const useInChatProfileIds: string[] = [];

  await page.route('**/api/v1/llamacpp/config', (route) =>
    fulfillJson(route, {
      saved_config: savedConfig,
      active_config: {
        handler_configured: true,
        enabled: true,
        executable_path: '/opt/llama-server',
        models_dir: '/srv/models',
        default_host: '127.0.0.1',
        default_port: 8080,
        active_model: null,
        active_host: null,
        active_port: null,
        active_pid: null,
      },
      restart_required: false,
      restart_reasons: [],
      env_overrides: {},
      warnings: [],
    })
  );

  await page.route('**/api/v1/llamacpp/status', (route) =>
    fulfillJson(route, {
      state: 'stopped',
      model: null,
      port: 8080,
      backend: 'llamacpp',
    })
  );

  await page.route('**/api/v1/llamacpp/inventory', (route) =>
    fulfillJson(route, {
      models: [
        {
          model_id: 'gguf:toy-chat',
          display_name: 'Toy Chat GGUF',
          basename: 'toy-chat.gguf',
          source: 'models_dir',
          path: '/srv/models/toy-chat.gguf',
          size_bytes: 2_200_000_000,
          modified_at: '2026-05-18T10:00:00Z',
          metadata: {
            quantization: 'Q4_K_M',
            parameter_hint: '7B',
            context_hint: 4096,
          },
          warnings: [],
        },
      ],
      warnings: [],
      scan_limited: false,
    })
  );

  await page.route('**/api/v1/llamacpp/hardware', (route) =>
    fulfillJson(route, {
      ram_total_bytes: 16_000_000_000,
      ram_available_bytes: 8_000_000_000,
      cpu_count: 8,
      gpus: [],
      warnings: ['GPU probe unavailable.'],
    })
  );

  await page.route('**/api/v1/llamacpp/assets', (route) =>
    fulfillJson(route, {
      assets: [ggufAsset, mmprojAsset],
      warnings: ['One asset had filename-derived metadata.'],
      scan_limited: false,
    })
  );

  await page.route('**/api/v1/llamacpp/assets/downloads', (route) =>
    fulfillJson(route, { jobs: [] })
  );

  await page.route('**/api/v1/llamacpp/profiles/*/use-in-chat', (route) => {
    const match = route.request().url().match(/\/llamacpp\/profiles\/([^/]+)\/use-in-chat$/);
    useInChatProfileIds.push(decodeURIComponent(match?.[1] || ''));
    return fulfillJson(route, {
      provider: 'llamacpp',
      endpoint: 'http://127.0.0.1:8181',
      updated: true,
      effective: true,
      warnings: [],
    });
  });

  await page.route('**/api/v1/llamacpp/profiles', (route) =>
    fulfillJson(route, { profiles })
  );

  await page.route('**/api/v1/llamacpp/instances', (route) =>
    fulfillJson(route, {
      runtimes,
      warnings: ['Runtime reconciliation is using bounded restart policy.'],
    })
  );

  return { useInChatProfileIds };
}

test.describe('llama.cpp managed runtime admin smoke', () => {
  test('shows assets, profile runtimes, warnings, and running-only chat wiring', async ({
    authedPage,
    diagnostics,
  }) => {
    const api = await mockManagedRuntimeAdmin(authedPage);
    const admin = new AdminPage(authedPage);

    await admin.gotoSection('llamacpp');
    await admin.assertSectionReady('llamacpp');

    await expect(authedPage.getByText('Assets', { exact: true })).toBeVisible();
    await expect(authedPage.getByText('Toy Chat GGUF').first()).toBeVisible();
    await expect(authedPage.getByText('One asset had filename-derived metadata.')).toBeVisible();

    const runtimePanel = authedPage.getByLabel('Runtime instances');
    await expect(runtimePanel).toBeVisible();
    await expect(runtimePanel.getByText('Chat runtime')).toBeVisible();
    await expect(runtimePanel.getByText('Vision runtime')).toBeVisible();
    await expect(runtimePanel.getByText('Embedding runtime')).toBeVisible();
    await expect(runtimePanel.getByText('running')).toBeVisible();
    await expect(runtimePanel.getByText('stopped')).toBeVisible();
    await expect(runtimePanel.getByText('failed')).toBeVisible();
    await expect(runtimePanel.getByText('Toy Vision Projector').first()).toBeVisible();
    await expect(runtimePanel.getByText('GPU probe unavailable.')).toBeVisible();
    await expect(runtimePanel.getByText('Restart limit reached.')).toBeVisible();
    await expect(
      runtimePanel.getByText('Projector pairing is inferred from local assets.')
    ).toBeVisible();

    await expect(
      authedPage.getByRole('button', { name: 'Use Chat runtime in Chat' })
    ).toBeVisible();
    await expect(
      authedPage.getByRole('button', { name: 'Use Vision runtime in Chat' })
    ).toHaveCount(0);
    await expect(
      authedPage.getByRole('button', { name: 'Use Embedding runtime in Chat' })
    ).toHaveCount(0);

    await authedPage.getByRole('button', { name: 'Use Chat runtime in Chat' }).click();
    expect(api.useInChatProfileIds).toEqual(['chat-runtime']);

    await assertNoCriticalErrors(diagnostics);
  });
});
