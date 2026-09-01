/* @vitest-environment jsdom */

import type { ReactNode } from 'react';
import { act, cleanup, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import WebhooksPage from '../page';
import { useWebhookSecretCommands } from '../use-webhook-secret-commands';
import { WebhookApiError, WebhookContractError, WebhookTransportError } from '@/lib/http';
import type {
  WebhookCatalog,
  WebhookDelivery,
  WebhookDeliveryAttempt,
  WebhookRegistration,
  WebhookSecretResponse,
  WebhookStatus,
} from '@/types';

const mocks = vi.hoisted(() => ({
  canonical: {
    getWebhookStatus: vi.fn(),
    getWebhookCatalog: vi.fn(),
    getWebhooks: vi.fn(),
    getWebhook: vi.fn(),
    createWebhook: vi.fn(),
    updateWebhook: vi.fn(),
    deleteWebhook: vi.fn(),
    rotateWebhookSecret: vi.fn(),
    testWebhook: vi.fn(),
    getWebhookDeliveries: vi.fn(),
    redeliverWebhook: vi.fn(),
  },
  privileged: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
  toastWarning: vi.fn(),
  guard: vi.fn(),
}));

vi.mock('@/lib/api-client', () => ({
  canonicalWebhookApi: mocks.canonical,
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: (props: {
    children: ReactNode;
    role?: string[];
    requireAuth?: boolean;
    variant?: string;
  }) => {
    mocks.guard(props);
    return <div data-testid="permission-guard">{props.children}</div>;
  },
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => mocks.privileged,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: mocks.toastSuccess,
    error: mocks.toastError,
    warning: mocks.toastWarning,
    info: vi.fn(),
  }),
}));

const STATUS: WebhookStatus = {
  mode: 'on',
  route_selection: 'canonical',
  schema_ready: true,
  key_state: 'available',
  delivery_capability_ready: true,
  delivery: {
    canonical_schema_version: 5,
    schema_ready: true,
    delivery_schema_ready: true,
    migration_complete: true,
    key_ready: true,
    key_primary_match: true,
    jobs_database_ready: true,
    queue_ready: true,
    job_type_ready: true,
    jobs_backend: 'postgres',
    worker: { component: 'worker', ready: true, reason_code: null, heartbeat_age_seconds: 3 },
    reconciler: {
      component: 'reconciler',
      ready: true,
      reason_code: null,
      heartbeat_age_seconds: 4,
    },
    retention: {
      component: 'retention',
      ready: true,
      reason_code: null,
      heartbeat_age_seconds: 5,
    },
    backlog: { pending: 1, enqueue_claimed: 0, queued: 2, processing: 1, retry_wait: 0 },
    oldest_nonterminal_age_seconds: 11,
    acquisition_ready: true,
    acquisition_reason_code: null,
    delivery_capability_ready: true,
  },
  limits: {
    registrations: 100,
    active_registrations: 25,
    current_registrations: 1,
    current_active_registrations: 0,
    registrations_over_limit: false,
    active_registrations_over_limit: false,
  },
  migration: {
    phase: 'complete',
    imported_count: 0,
    unresolved_count: 0,
    rejected_count: 0,
    secret_rotation_required_count: 0,
    legacy_file_restore_permitted: true,
    rollback_window_expires_at: '2026-08-29T12:00:00Z',
  },
};

const CATALOG: WebhookCatalog = {
  api_version: '1',
  events: [
    { event_type: 'incident.created', description: 'An incident was created.' },
    { event_type: 'catalog.only', description: 'Only supplied by this test catalog.' },
  ],
  registration_limit: 100,
  active_limit: 25,
};

const REGISTRATION: WebhookRegistration = {
  id: 41,
  description: 'Incident receiver',
  target_display: 'https://receiver.example',
  target_hostname: 'receiver.example',
  event_types: ['incident.created'],
  active: false,
  timeout_seconds: 10,
  revision: 2,
  delivery_config_version: 1,
  secret_version: 1,
  secret_rotation_required: false,
  created_by: 7,
  updated_by: 7,
  created_at: '2026-08-22T12:00:00Z',
  updated_at: '2026-08-22T12:00:00Z',
};

const SIGNING_SECRET = `whsec_${'a'.repeat(64)}`;
const SECOND_SIGNING_SECRET = `whsec_${'b'.repeat(64)}`;
const DELIVERY_ID = '11111111-1111-4111-8111-111111111111';
const DELIVERY: WebhookDelivery = {
  id: DELIVERY_ID,
  event_id: '22222222-2222-4222-8222-222222222222',
  event_type: 'incident.created',
  webhook_id: 41,
  kind: 'automatic',
  state: 'succeeded',
  delivery_config_version: 1,
  secret_version: 1,
  attempt_count: 1,
  status_code: 204,
  latency_ms: 42,
  reason_code: null,
  expires_at: '2026-08-23T12:00:00Z',
  created_at: '2026-08-22T12:00:00Z',
  updated_at: '2026-08-22T12:00:01Z',
  terminal_at: '2026-08-22T12:00:01Z',
  redelivery_of_id: null,
  completed_after_config_change: false,
};
const ATTEMPT: WebhookDeliveryAttempt = {
  id: '33333333-3333-4333-8333-333333333333',
  sequence: 1,
  state: 'succeeded',
  request_timeout_seconds: 10,
  status_code: 204,
  latency_ms: 42,
  reason_code: null,
  requested_retry_delay_seconds: null,
  started_at: '2026-08-22T12:00:00Z',
  finished_at: '2026-08-22T12:00:01Z',
};
const SECRET_RESPONSE: WebhookSecretResponse = {
  registration: REGISTRATION,
  signing_secret: SIGNING_SECRET,
  replayed: false,
};

const strongResponse = <T,>(data: T, etag = '"admin-webhook-41-r2"', status = 200) => ({
  data,
  etag,
  status,
  requestId: 'request-41',
  retryAfterSeconds: null,
});

const canonicalPage = (items = [REGISTRATION], total = items.length, offset = 0) => ({
  items,
  total,
  limit: 20,
  offset,
});

const renderReadyPage = async () => {
  render(<WebhooksPage />);
  await screen.findByRole('heading', { name: 'Webhooks' });
  await screen.findByText(REGISTRATION.target_display);
};

const openCreateDialog = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole('button', { name: /add webhook/i }));
  return screen.findByRole('dialog', { name: /add webhook/i });
};

const submitCanonicalCreate = async (user: ReturnType<typeof userEvent.setup>) => {
  const dialog = await openCreateDialog(user);
  await user.type(
    within(dialog).getByLabelText(/destination url/i),
    'https://private.example/hooks/secret',
  );
  await user.type(within(dialog).getByLabelText(/^description/i), 'Private receiver');
  const timeout = within(dialog).getByLabelText(/timeout/i);
  await user.clear(timeout);
  await user.type(timeout, '12');
  await user.click(within(dialog).getByLabelText(/catalog\.only/i));
  await user.click(within(dialog).getByRole('button', { name: /^create$/i }));
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('crypto', {
    getRandomValues: vi.fn((target: Uint8Array) => {
      target.forEach((_value, index) => {
        target[index] = index;
      });
      return target;
    }),
  });
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
  });

  mocks.canonical.getWebhookStatus.mockResolvedValue(STATUS);
  mocks.canonical.getWebhookCatalog.mockResolvedValue(CATALOG);
  mocks.canonical.getWebhooks.mockResolvedValue(canonicalPage());
  mocks.canonical.getWebhook.mockResolvedValue(strongResponse(REGISTRATION));
  mocks.canonical.createWebhook.mockResolvedValue(
    strongResponse(SECRET_RESPONSE, '"admin-webhook-41-r2"', 201),
  );
  mocks.canonical.updateWebhook.mockResolvedValue(
    strongResponse({ ...REGISTRATION, revision: 3 }, '"admin-webhook-41-r3"'),
  );
  mocks.canonical.deleteWebhook.mockResolvedValue({ deleted: true, id: 41 });
  mocks.canonical.rotateWebhookSecret.mockResolvedValue(strongResponse({
    ...SECRET_RESPONSE,
    registration: { ...REGISTRATION, revision: 3, secret_version: 2 },
  }, '"admin-webhook-41-r3"'));
  mocks.canonical.getWebhookDeliveries.mockResolvedValue({
    items: [{ delivery: DELIVERY, attempts: [ATTEMPT] }],
    total: 1,
    limit: 50,
    offset: 0,
  });
  mocks.canonical.testWebhook.mockResolvedValue({
    data: {
      delivery: { ...DELIVERY, kind: 'test', event_type: 'webhook.test' },
      attempt: ATTEMPT,
      idempotent_replay: false,
      in_progress: false,
    },
    status: 200,
    etag: null,
    requestId: 'request-test',
    retryAfterSeconds: null,
  });
  mocks.canonical.redeliverWebhook.mockResolvedValue({
    data: {
      delivery: {
        ...DELIVERY,
        id: '44444444-4444-4444-8444-444444444444',
        kind: 'manual',
        state: 'pending',
        status_code: null,
        latency_ms: null,
        terminal_at: null,
        redelivery_of_id: DELIVERY_ID,
      },
      idempotent_replay: false,
    },
    status: 202,
    etag: null,
    requestId: 'request-redelivery',
    retryAfterSeconds: null,
  });
  mocks.privileged.mockResolvedValue({
    reason: 'Reviewed current state',
    adminPassword: 'AdminPass123!',
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('canonical webhook control plane page', () => {
  it('loads canonical status before catalog/list and renders only server catalog events', async () => {
    const user = userEvent.setup();
    await renderReadyPage();

    expect(mocks.canonical.getWebhookStatus).toHaveBeenCalledTimes(1);
    expect(mocks.canonical.getWebhookStatus.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.canonical.getWebhookCatalog.mock.invocationCallOrder[0],
    );
    const dialog = await openCreateDialog(user);
    expect(within(dialog).getByText('catalog.only')).toBeInTheDocument();
    expect(within(dialog).queryByText('user.deleted')).not.toBeInTheDocument();
  });

  it('creates inactive from the catalog and requires copy plus acknowledgement before dismissal', async () => {
    const user = userEvent.setup();
    await renderReadyPage();

    await submitCanonicalCreate(user);

    await waitFor(() => expect(mocks.canonical.createWebhook).toHaveBeenCalledTimes(1));
    const [body, key] = mocks.canonical.createWebhook.mock.calls[0] ?? [];
    expect(body).toEqual({
      url: 'https://private.example/hooks/secret',
      event_types: ['catalog.only'],
      description: 'Private receiver',
      timeout_seconds: 12,
    });
    expect(body).not.toHaveProperty('active');
    expect(key).toMatch(/^[0-9a-f]{32}$/);

    const secretDialog = await screen.findByRole('dialog', { name: /signing secret/i });
    expect((within(secretDialog).getByLabelText(/^signing secret$/i) as HTMLInputElement).value)
      .toBe(SIGNING_SECRET);
    const done = within(secretDialog).getByRole('button', { name: /done/i });
    expect(done).toBeDisabled();
    await user.click(within(secretDialog).getByRole('button', { name: /copy signing secret/i }));
    await user.click(within(secretDialog).getByLabelText(/i have stored this signing secret/i));
    expect(done).not.toBeDisabled();
    await user.click(done);
    expect(screen.queryByDisplayValue(SIGNING_SECRET)).not.toBeInTheDocument();
    expect(JSON.stringify(mocks.toastSuccess.mock.calls)).not.toContain(SIGNING_SECRET);
    expect(JSON.stringify(mocks.toastSuccess.mock.calls)).not.toContain('private.example');
  });

  it('rejects an unsafe destination before creating an idempotent command', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    const dialog = await openCreateDialog(user);

    await user.type(
      within(dialog).getByLabelText(/destination url/i),
      'https://operator:secret@receiver.example/hooks/events#private',
    );
    await user.click(within(dialog).getByLabelText(/catalog\.only/i));
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }));

    expect(await within(dialog).findByText(/must not include credentials or a fragment/i))
      .toBeInTheDocument();
    expect(within(dialog).getByLabelText(/destination url/i).getAttribute('aria-invalid'))
      .toBe('true');
    expect(mocks.canonical.createWebhook).not.toHaveBeenCalled();
  });

  it('does not mark the destination invalid for a timeout validation error', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    const dialog = await openCreateDialog(user);
    const destination = within(dialog).getByLabelText(/destination url/i);
    await user.type(destination, 'https://receiver.example/hooks/events');
    await user.click(within(dialog).getByLabelText(/catalog\.only/i));
    const timeout = within(dialog).getByLabelText(/timeout/i);
    await user.clear(timeout);
    await user.type(timeout, '31');
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }));

    expect(await within(dialog).findByText(/timeout must be a whole number/i)).toBeInTheDocument();
    expect(destination.getAttribute('aria-invalid')).toBe('false');
    expect(destination.hasAttribute('aria-describedby')).toBe(false);
    expect(mocks.canonical.createWebhook).not.toHaveBeenCalled();
  });

  it('warns on premature close and clears secret and retry state synchronously on pagehide', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    await submitCanonicalCreate(user);

    const secretDialog = await screen.findByRole('dialog', { name: /signing secret/i });
    await user.click(within(secretDialog).getByRole('button', { name: 'Close' }));
    expect(screen.getByText(/copy and acknowledge the secret before closing/i)).toBeInTheDocument();
    expect(screen.getByDisplayValue(SIGNING_SECRET)).toBeInTheDocument();

    await act(async () => {
      const event = new Event('pagehide');
      Object.defineProperty(event, 'persisted', { value: false });
      window.dispatchEvent(event);
    });
    expect(screen.queryByDisplayValue(SIGNING_SECRET)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /retry same command/i })).not.toBeInTheDocument();

    await act(async () => {
      const event = new Event('pageshow');
      Object.defineProperty(event, 'persisted', { value: true });
      window.dispatchEvent(event);
    });
    expect(screen.queryByDisplayValue(SIGNING_SECRET)).not.toBeInTheDocument();
  });

  it('does not restore a secret when a create response resolves after pagehide', async () => {
    const user = userEvent.setup();
    let resolveCreate: ((value: ReturnType<typeof strongResponse>) => void) | undefined;
    mocks.canonical.createWebhook.mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    await renderReadyPage();

    await submitCanonicalCreate(user);
    await waitFor(() => expect(mocks.canonical.createWebhook).toHaveBeenCalledTimes(1));
    await act(async () => {
      window.dispatchEvent(new Event('pagehide'));
    });
    const createDialog = screen.getByRole('dialog', { name: /add webhook/i });
    expect(within(createDialog).getByRole('button', { name: /^create$/i })).not.toBeDisabled();
    await act(async () => {
      resolveCreate?.(strongResponse(SECRET_RESPONSE, '"admin-webhook-41-r2"', 201));
      await Promise.resolve();
    });

    expect(screen.queryByDisplayValue(SIGNING_SECRET)).not.toBeInTheDocument();
    expect(mocks.toastSuccess).not.toHaveBeenCalledWith(
      'Webhook created',
      expect.any(String),
    );
  });

  it('does not apply a stale clipboard result to a later signing secret', async () => {
    const user = userEvent.setup();
    let resolveCopy: (() => void) | undefined;
    const writeText = vi.fn(() => new Promise<void>((resolve) => {
      resolveCopy = resolve;
    }));
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    mocks.canonical.createWebhook.mockReset()
      .mockResolvedValueOnce(strongResponse(SECRET_RESPONSE, '"admin-webhook-41-r2"', 201))
      .mockResolvedValueOnce(strongResponse({
        ...SECRET_RESPONSE,
        signing_secret: SECOND_SIGNING_SECRET,
      }, '"admin-webhook-41-r3"', 201));
    await renderReadyPage();

    await submitCanonicalCreate(user);
    const firstSecretDialog = await screen.findByRole('dialog', { name: /signing secret/i });
    await user.click(within(firstSecretDialog).getByRole('button', { name: /copy signing secret/i }));
    await act(async () => {
      window.dispatchEvent(new Event('pagehide'));
    });
    await submitCanonicalCreate(user);
    const secondSecretDialog = await screen.findByRole('dialog', { name: /signing secret/i });
    expect(within(secondSecretDialog).getByDisplayValue(SECOND_SIGNING_SECRET))
      .toBeInTheDocument();

    await act(async () => {
      resolveCopy?.();
      await Promise.resolve();
    });

    expect(within(secondSecretDialog).queryByText(/copied to clipboard/i))
      .not.toBeInTheDocument();
    expect(within(secondSecretDialog).getByRole('button', { name: /^done$/i }))
      .toBeDisabled();
  });

  it('retries a lost create response with the same idempotency key only on operator request', async () => {
    const user = userEvent.setup();
    mocks.canonical.createWebhook
      .mockRejectedValueOnce(new WebhookTransportError(502, 'proxy-request-1'))
      .mockResolvedValueOnce(
        strongResponse({ ...SECRET_RESPONSE, replayed: true }, '"admin-webhook-41-r2"', 201),
      );
    await renderReadyPage();

    await submitCanonicalCreate(user);
    const retry = await screen.findByRole('button', { name: /retry same command/i });
    expect(mocks.canonical.createWebhook).toHaveBeenCalledTimes(1);
    await user.click(retry);
    await screen.findByDisplayValue(SIGNING_SECRET);

    expect(mocks.canonical.createWebhook).toHaveBeenCalledTimes(2);
    expect(mocks.canonical.createWebhook.mock.calls[0]?.[1]).toBe(
      mocks.canonical.createWebhook.mock.calls[1]?.[1],
    );
    expect(screen.getByText(/response was recovered from the original command/i)).toBeInTheDocument();
  });

  it.each([
    new WebhookApiError(503, 'operation_failed', 'Service unavailable', 'request-create-503'),
    new WebhookContractError(201, 'Malformed committed response', 'request-create-contract'),
  ])('preserves a create command after an ambiguous API result', async (failure) => {
    const user = userEvent.setup();
    mocks.canonical.createWebhook
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce(
        strongResponse({ ...SECRET_RESPONSE, replayed: true }, '"admin-webhook-41-r2"', 201),
      );
    await renderReadyPage();

    await submitCanonicalCreate(user);
    const retry = await screen.findByRole('button', { name: /retry same command/i });
    expect(screen.getByText(/result is ambiguous/i)).toBeInTheDocument();
    await user.click(retry);
    await screen.findByDisplayValue(SIGNING_SECRET);

    expect(mocks.canonical.createWebhook).toHaveBeenCalledTimes(2);
    expect(mocks.canonical.createWebhook.mock.calls[0]?.[1]).toBe(
      mocks.canonical.createWebhook.mock.calls[1]?.[1],
    );
  });

  it('gates catalog/list while migration is incomplete', async () => {
    mocks.canonical.getWebhookStatus.mockResolvedValue({
      ...STATUS,
      mode: 'migrate',
      migration: { ...STATUS.migration, phase: 'migration_pending' },
    });

    render(<WebhooksPage />);

    expect(await screen.findByText(/migration is not complete/i)).toBeInTheDocument();
    expect(mocks.canonical.getWebhookCatalog).not.toHaveBeenCalled();
    expect(mocks.canonical.getWebhooks).not.toHaveBeenCalled();
  });

  it('keeps healthy runtime status visible when registration loading fails', async () => {
    mocks.canonical.getWebhooks.mockRejectedValue(new Error('list unavailable'));

    render(<WebhooksPage />);

    expect(await screen.findByText(/worker ready/i)).toBeInTheDocument();
    await waitFor(() => expect(mocks.toastError).toHaveBeenCalledWith(
      'Unable to load webhooks',
      'Webhook registrations could not be loaded.',
    ));
    expect(screen.queryByText(/webhook status could not be loaded/i)).not.toBeInTheDocument();
  });

  it('shows operational status and blocks activation for unavailable delivery or required rotation', async () => {
    const rotationRequired = { ...REGISTRATION, secret_rotation_required: true };
    mocks.canonical.getWebhookStatus.mockResolvedValue({
      ...STATUS,
      key_state: 'unavailable',
      delivery_capability_ready: false,
      delivery: {
        ...STATUS.delivery,
        key_ready: false,
        worker: {
          component: 'worker',
          ready: false,
          reason_code: 'key_unavailable',
          heartbeat_age_seconds: null,
        },
        acquisition_ready: false,
        acquisition_reason_code: 'key_unavailable',
        delivery_capability_ready: false,
      },
      limits: { ...STATUS.limits, registrations_over_limit: true },
      migration: {
        ...STATUS.migration,
        secret_rotation_required_count: 1,
        legacy_file_restore_permitted: false,
      },
    });
    mocks.canonical.getWebhooks.mockResolvedValue(canonicalPage([rotationRequired]));

    await renderReadyPage();

    expect(screen.getByText(/signing key is unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/registration limit is exceeded/i)).toBeInTheDocument();
    expect(screen.getByText(/delivery capability is unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/1 registration requires a new signing secret/i)).toBeInTheDocument();
    expect(screen.getByText(/legacy restore is unavailable/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^enable$/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /generate a new secret/i })).toBeDisabled();
  });

  it('paginates from server totals and exposes canonical test and delivery controls', async () => {
    const second = { ...REGISTRATION, id: 42, target_display: 'https://second.example' };
    mocks.canonical.getWebhooks
      .mockResolvedValueOnce(canonicalPage([REGISTRATION], 21, 0))
      .mockResolvedValueOnce(canonicalPage([second], 21, 20));
    const user = userEvent.setup();
    await renderReadyPage();

    expect(screen.getByRole('button', { name: /run test/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /show delivery history/i })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /next page/i }));

    expect(await screen.findByText('https://second.example')).toBeInTheDocument();
    expect(screen.queryByText(REGISTRATION.target_display)).not.toBeInTheDocument();
    expect(mocks.canonical.getWebhooks).toHaveBeenLastCalledWith({ limit: 20, offset: 20 });
  });

  it('uses a fresh GET/ETag for metadata PATCH and omits the destination URL', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    await user.click(screen.getByRole('button', { name: /edit metadata/i }));
    const dialog = await screen.findByRole('dialog', { name: /edit webhook metadata/i });
    const description = within(dialog).getByLabelText(/^description/i);
    await user.clear(description);
    await user.type(description, 'Reviewed receiver');
    await user.click(within(dialog).getByRole('button', { name: /save changes/i }));

    await waitFor(() => expect(mocks.canonical.updateWebhook).toHaveBeenCalledTimes(1));
    expect(mocks.canonical.getWebhook.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.canonical.updateWebhook.mock.invocationCallOrder[0],
    );
    const [, body, etag] = mocks.canonical.updateWebhook.mock.calls[0] ?? [];
    expect(body).toMatchObject({ description: 'Reviewed receiver' });
    expect(body).not.toHaveProperty('url');
    expect(etag).toBe('"admin-webhook-41-r2"');
    expect(mocks.privileged).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.stringContaining('revision 2'),
      confirmationOnly: true,
    }));
  });

  it('keeps destination replacement blank and sends only the explicit new URL', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    await user.click(screen.getByRole('button', { name: /replace destination/i }));
    const dialog = await screen.findByRole('dialog', { name: /replace webhook destination/i });
    const destination = within(dialog).getByLabelText(/new destination url/i);
    expect((destination as HTMLInputElement).value).toBe('');
    expect((destination as HTMLInputElement).value).not.toBe(REGISTRATION.target_display);
    await user.type(destination, 'https://replacement.example/private');
    await user.click(within(dialog).getByRole('button', { name: /save destination/i }));

    await waitFor(() => expect(mocks.canonical.updateWebhook).toHaveBeenCalledTimes(1));
    expect(mocks.canonical.updateWebhook).toHaveBeenCalledWith(
      41,
      { url: 'https://replacement.example/private' },
      '"admin-webhook-41-r2"',
    );
  });

  it('rejects an unsafe replacement destination before loading a fresh ETag', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    await user.click(screen.getByRole('button', { name: /replace destination/i }));
    const dialog = await screen.findByRole('dialog', { name: /replace webhook destination/i });

    await user.type(within(dialog).getByLabelText(/new destination url/i), 'javascript:alert(1)');
    await user.click(within(dialog).getByRole('button', { name: /save destination/i }));

    expect(await within(dialog).findByText(/absolute HTTP or HTTPS URL/i)).toBeInTheDocument();
    expect(mocks.canonical.getWebhook).not.toHaveBeenCalled();
    expect(mocks.canonical.updateWebhook).not.toHaveBeenCalled();
  });

  it.each([412, 428])('does not automatically retry a conditional mutation after HTTP %i', async (status) => {
    const user = userEvent.setup();
    const fresh = { ...REGISTRATION, description: 'Changed elsewhere', revision: 3 };
    mocks.canonical.getWebhook
      .mockResolvedValueOnce(strongResponse(REGISTRATION))
      .mockResolvedValueOnce(strongResponse(fresh, '"admin-webhook-41-r3"'));
    mocks.canonical.updateWebhook.mockRejectedValue(new WebhookApiError(
      status,
      status === 412
        ? 'admin_webhook_precondition_failed'
        : 'admin_webhook_precondition_required',
      'Webhook precondition failed',
      'request-conflict',
    ));
    await renderReadyPage();
    await user.click(screen.getByRole('button', { name: /edit metadata/i }));
    const dialog = await screen.findByRole('dialog', { name: /edit webhook metadata/i });
    await user.click(within(dialog).getByRole('button', { name: /save changes/i }));

    expect(await screen.findByText(/review the current webhook before retrying/i)).toBeInTheDocument();
    expect(screen.getAllByText(/changed elsewhere/i)).toHaveLength(2);
    expect(mocks.canonical.updateWebhook).toHaveBeenCalledTimes(1);
    expect(mocks.canonical.getWebhook).toHaveBeenCalledTimes(2);

    await user.click(screen.getByRole('button', { name: /edit metadata/i }));
    const refreshedEditor = await screen.findByRole('dialog', { name: /edit webhook metadata/i });
    expect((within(refreshedEditor).getByLabelText(/^description/i) as HTMLInputElement).value)
      .toBe('Changed elsewhere');
  });

  it('uses a fresh ETag for delete and rotate and reveals a new secret only for inactive rows', async () => {
    const user = userEvent.setup();
    await renderReadyPage();

    await user.click(screen.getByRole('button', { name: /generate a new secret/i }));
    await waitFor(() => expect(mocks.canonical.rotateWebhookSecret).toHaveBeenCalledTimes(1));
    expect(mocks.canonical.rotateWebhookSecret.mock.calls[0]?.slice(0, 2)).toEqual([
      41,
      '"admin-webhook-41-r2"',
    ]);
    expect(await screen.findByDisplayValue(SIGNING_SECRET)).toBeInTheDocument();

    await act(async () => {
      window.dispatchEvent(new Event('pagehide'));
    });
    await user.click(screen.getByRole('button', { name: /delete webhook/i }));
    await waitFor(() => expect(mocks.canonical.deleteWebhook).toHaveBeenCalledWith(
      41,
      '"admin-webhook-41-r2"',
    ));
  });

  it('locks other secret rotations while one rotation command is pending', async () => {
    const secondRegistration = {
      ...REGISTRATION,
      id: 42,
      target_display: 'https://second.example',
      target_hostname: 'second.example',
    };
    mocks.canonical.getWebhooks.mockResolvedValue(
      canonicalPage([REGISTRATION, secondRegistration]),
    );
    mocks.canonical.getWebhook.mockImplementation(async (id: number) => (
      id === secondRegistration.id
        ? strongResponse(secondRegistration, '"admin-webhook-42-r2"')
        : strongResponse(REGISTRATION)
    ));
    mocks.canonical.rotateWebhookSecret.mockReturnValue(new Promise(() => {}));
    const user = userEvent.setup();
    await renderReadyPage();

    const rotateButtons = screen.getAllByRole('button', { name: /generate a new secret/i });
    await user.click(rotateButtons[0]!);
    await waitFor(() => expect(mocks.canonical.rotateWebhookSecret).toHaveBeenCalledTimes(1));

    expect(screen.getAllByRole('button', { name: /generate a new secret/i })[1])
      .toBeDisabled();
  });

  it('keeps bounded status failures visible without compatibility fallback', async () => {
    const user = userEvent.setup();
    mocks.canonical.getWebhookStatus.mockRejectedValueOnce(new WebhookContractError(
      503,
      'Webhook API returned an invalid status response',
      'request-status',
    ));

    render(<WebhooksPage />);

    expect(await screen.findByText(/invalid status response/i)).toBeInTheDocument();
    expect(screen.getByText(/request-status/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /retry status/i }));
    await waitFor(() => expect(mocks.canonical.getWebhookStatus).toHaveBeenCalledTimes(2));
  });

  it('warns before navigation while a one-time secret is visible', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    await submitCanonicalCreate(user);
    await screen.findByDisplayValue(SIGNING_SECRET);

    const event = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
  });

  it('blocks same-tab SPA navigation while a one-time secret is visible', async () => {
    const user = userEvent.setup();
    await renderReadyPage();
    await submitCanonicalCreate(user);
    await screen.findByDisplayValue(SIGNING_SECRET);
    const link = document.createElement('a');
    link.href = '/incidents';
    document.body.append(link);

    const allowed = link.dispatchEvent(new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
    }));

    expect(allowed).toBe(false);
    link.remove();
  });

  it('renders runtime readiness, backlog, and oldest-work age from canonical status', async () => {
    await renderReadyPage();

    expect(screen.getByText(/worker ready/i)).toBeInTheDocument();
    expect(screen.getByText(/signing key ready/i)).toBeInTheDocument();
    expect(screen.getByText(/reconciler ready/i)).toBeInTheDocument();
    expect(screen.getByText(/retention ready/i)).toBeInTheDocument();
    expect(screen.getByText(/4 nonterminal deliveries/i)).toBeInTheDocument();
    expect(screen.getByText(/oldest work 11s/i)).toBeInTheDocument();
  });

  it('expands sanitized canonical delivery and attempt history', async () => {
    const user = userEvent.setup();
    await renderReadyPage();

    await user.click(screen.getByRole('button', { name: /show delivery history/i }));

    await waitFor(() => expect(mocks.canonical.getWebhookDeliveries).toHaveBeenCalledWith(
      41,
      { limit: 50, offset: 0 },
    ));
    expect(screen.getAllByText('incident.created')).toHaveLength(2);
    expect(screen.getAllByText(/HTTP 204 \(2xx\)/i)).toHaveLength(2);
    expect(screen.getByText(/attempt 1/i)).toBeInTheDocument();
    expect(screen.getByText(/config v1, secret v1/i)).toBeInTheDocument();
    expect(screen.getByText(/original delivery/i)).toBeInTheDocument();
    expect(screen.getByText(/10s timeout/i)).toBeInTheDocument();
    expect(screen.getByText(/no requested retry delay/i)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain('/hooks/private');
  });

  it('runs a persisted test with a fresh ETag and keeps one ambiguous retry key in memory', async () => {
    const user = userEvent.setup();
    mocks.canonical.testWebhook
      .mockRejectedValueOnce(new WebhookTransportError(502, 'request-test-lost'))
      .mockResolvedValueOnce({
        data: {
          delivery: { ...DELIVERY, kind: 'test', event_type: 'webhook.test' },
          attempt: ATTEMPT,
          idempotent_replay: true,
          in_progress: false,
        },
        status: 200,
        etag: null,
        requestId: 'request-test-replay',
        retryAfterSeconds: null,
      });
    await renderReadyPage();

    await user.click(screen.getByRole('button', { name: /run test/i }));
    await waitFor(() => expect(mocks.canonical.testWebhook).toHaveBeenCalledTimes(1));
    const retry = await screen.findByRole('button', { name: /retry same test/i });
    await user.click(retry);

    await waitFor(() => expect(mocks.canonical.testWebhook).toHaveBeenCalledTimes(2));
    const first = mocks.canonical.testWebhook.mock.calls[0] ?? [];
    const second = mocks.canonical.testWebhook.mock.calls[1] ?? [];
    expect(first.slice(0, 3)).toEqual([41, { delivery_config_version: 1 }, '"admin-webhook-41-r2"']);
    expect(first[3]).toBe(second[3]);
    expect(screen.getByText(/test delivery succeeded/i)).toBeInTheDocument();
  });

  it('requires explicit changed-configuration confirmation before manual redelivery', async () => {
    const user = userEvent.setup();
    const changed = { ...REGISTRATION, delivery_config_version: 2, revision: 3 };
    mocks.canonical.getWebhook.mockResolvedValue(strongResponse(
      changed,
      '"admin-webhook-41-r3"',
    ));
    await renderReadyPage();
    await user.click(screen.getByRole('button', { name: /show delivery history/i }));
    await screen.findByText(/attempt 1/i);

    await user.click(screen.getByRole('button', { name: /redeliver incident\.created/i }));

    await waitFor(() => expect(mocks.privileged).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.stringMatching(/configuration changed.*receiver\.example/is),
      confirmationOnly: true,
    })));
    expect(mocks.canonical.redeliverWebhook).toHaveBeenCalledWith(
      41,
      DELIVERY_ID,
      { delivery_config_version: 2, confirm_changed_configuration: true },
      '"admin-webhook-41-r3"',
      expect.stringMatching(/^[0-9a-f]{32}$/),
    );
  });

  it('clears an ambiguous test retry command on pagehide', async () => {
    const user = userEvent.setup();
    mocks.canonical.testWebhook.mockRejectedValue(
      new WebhookTransportError(502, 'request-test-lost'),
    );
    await renderReadyPage();
    await user.click(screen.getByRole('button', { name: /run test/i }));
    await screen.findByRole('button', { name: /retry same test/i });

    await act(async () => {
      window.dispatchEvent(new Event('pagehide'));
    });

    expect(screen.queryByRole('button', { name: /retry same test/i })).not.toBeInTheDocument();
  });

  it('uses a route-level platform-admin permission guard', async () => {
    await renderReadyPage();

    expect(mocks.guard).toHaveBeenCalledWith(expect.objectContaining({
      role: ['admin', 'super_admin', 'owner'],
      requireAuth: true,
      variant: 'route',
    }));
  });
});

describe('one-time secret command ownership', () => {
  it('atomically rejects a second command while the first owns sensitive state', async () => {
    const firstRun = vi.fn(() => new Promise<ReturnType<typeof strongResponse>>(() => {}));
    const secondRun = vi.fn().mockResolvedValue(
      strongResponse(SECRET_RESPONSE, '"admin-webhook-42-r2"'),
    );
    const showError = vi.fn();
    const { result } = renderHook(() => useWebhookSecretCommands({
      clearCreateForm: vi.fn(),
      loadControlPlane: vi.fn().mockResolvedValue(undefined),
      offset: 0,
      recoverConditionalConflict: vi.fn().mockResolvedValue(undefined),
      setCreateOpen: vi.fn(),
      showError,
      success: vi.fn(),
    }));
    const first = {
      command: {
        idempotencyKey: 'first-command',
        canRetry: false,
        run: firstRun,
        retry: firstRun,
      },
      operation: 'rotate' as const,
      webhookId: 41,
    };
    const second = {
      command: {
        idempotencyKey: 'second-command',
        canRetry: false,
        run: secondRun,
        retry: secondRun,
      },
      operation: 'rotate' as const,
      webhookId: 42,
    };

    await act(async () => {
      void result.current.startSecretCommand(first);
      void result.current.startSecretCommand(second);
      await Promise.resolve();
    });

    expect(firstRun).toHaveBeenCalledTimes(1);
    expect(secondRun).not.toHaveBeenCalled();
    expect(showError).toHaveBeenCalledWith(
      'Signing secret command blocked',
      expect.stringMatching(/finish the current signing-secret command/i),
    );
  });
});
