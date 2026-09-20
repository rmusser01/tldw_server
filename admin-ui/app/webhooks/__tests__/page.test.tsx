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
  WebhookRegistration,
  WebhookSecretResponse,
  WebhookStatus,
} from '@/types';

const mocks = vi.hoisted(() => ({
  detect: vi.fn(),
  canonical: {
    getWebhookStatus: vi.fn(),
    getWebhookCatalog: vi.fn(),
    getWebhooks: vi.fn(),
    getWebhook: vi.fn(),
    createWebhook: vi.fn(),
    updateWebhook: vi.fn(),
    deleteWebhook: vi.fn(),
    rotateWebhookSecret: vi.fn(),
  },
  legacy: {
    getWebhooks: vi.fn(),
    createWebhook: vi.fn(),
    updateWebhook: vi.fn(),
    deleteWebhook: vi.fn(),
    testWebhook: vi.fn(),
    getWebhookDeliveries: vi.fn(),
  },
  privileged: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
  toastWarning: vi.fn(),
  guard: vi.fn(),
}));

vi.mock('@/lib/api-client', () => ({
  detectWebhookApi: mocks.detect,
  canonicalWebhookApi: mocks.canonical,
  legacyWebhookApi: mocks.legacy,
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
  mode: 'migrate',
  route_selection: 'canonical',
  schema_ready: true,
  key_state: 'available',
  delivery_capability_ready: false,
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
});

const canonicalPage = (items = [REGISTRATION], total = items.length, offset = 0) => ({
  items,
  total,
  limit: 20,
  offset,
});

const deferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
};

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

  mocks.detect.mockResolvedValue({ kind: 'canonical', status: STATUS, client: mocks.canonical });
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
  mocks.privileged.mockResolvedValue({
    reason: 'Reviewed current state',
    adminPassword: 'AdminPass123!',
  });
  mocks.legacy.getWebhooks.mockResolvedValue({ items: [], total: 0 });
  mocks.legacy.getWebhookDeliveries.mockResolvedValue({ items: [], total: 0 });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('canonical webhook control plane page', () => {
  it('selects mode before loading catalog/list and renders only server catalog events', async () => {
    const user = userEvent.setup();
    await renderReadyPage();

    expect(mocks.detect).toHaveBeenCalledTimes(1);
    expect(mocks.detect.mock.invocationCallOrder[0]).toBeLessThan(
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

  it('gates catalog/list while migration is incomplete and never probes legacy CRUD', async () => {
    mocks.detect.mockResolvedValue({
      kind: 'canonical',
      status: { ...STATUS, migration: { ...STATUS.migration, phase: 'migration_pending' } },
      client: mocks.canonical,
    });

    render(<WebhooksPage />);

    expect(await screen.findByText(/migration is not complete/i)).toBeInTheDocument();
    expect(mocks.canonical.getWebhookCatalog).not.toHaveBeenCalled();
    expect(mocks.canonical.getWebhooks).not.toHaveBeenCalled();
    expect(mocks.legacy.getWebhooks).not.toHaveBeenCalled();
  });

  it('shows operational status and blocks activation for unavailable delivery or required rotation', async () => {
    const rotationRequired = { ...REGISTRATION, secret_rotation_required: true };
    mocks.detect.mockResolvedValue({
      kind: 'canonical',
      status: {
        ...STATUS,
        key_state: 'unavailable',
        limits: { ...STATUS.limits, registrations_over_limit: true },
        migration: {
          ...STATUS.migration,
          secret_rotation_required_count: 1,
          legacy_file_restore_permitted: false,
        },
      },
      client: mocks.canonical,
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

  it('paginates from server totals and exposes no canonical test or delivery controls', async () => {
    const second = { ...REGISTRATION, id: 42, target_display: 'https://second.example' };
    mocks.canonical.getWebhooks
      .mockResolvedValueOnce(canonicalPage([REGISTRATION], 21, 0))
      .mockResolvedValueOnce(canonicalPage([second], 21, 20));
    const user = userEvent.setup();
    await renderReadyPage();

    expect(screen.queryByRole('button', { name: /^test$/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/delivery history/i)).not.toBeInTheDocument();
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

  it('keeps bounded status failures visible and never downgrades to legacy CRUD', async () => {
    const user = userEvent.setup();
    mocks.detect.mockRejectedValueOnce(new WebhookContractError(
      503,
      'Webhook API returned an invalid status response',
      'request-status',
    ));

    render(<WebhooksPage />);

    expect(await screen.findByText(/invalid status response/i)).toBeInTheDocument();
    expect(screen.getByText(/request-status/i)).toBeInTheDocument();
    expect(mocks.legacy.getWebhooks).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /retry status/i }));
    await waitFor(() => expect(mocks.detect).toHaveBeenCalledTimes(2));
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

  it('uses a route-level platform-admin permission guard', async () => {
    await renderReadyPage();

    expect(mocks.guard).toHaveBeenCalledWith(expect.objectContaining({
      role: ['admin', 'super_admin', 'owner'],
      requireAuth: true,
      variant: 'route',
    }));
  });
});

describe('legacy webhook compatibility mode', () => {
  it('shows the compatibility banner and legacy test/delivery controls only when status selects legacy', async () => {
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [{
        id: 'legacy-1',
        targetUrl: 'https://legacy.example/hook',
        eventTypes: ['incident.created'],
        enabled: true,
        createdAt: null,
        updatedAt: null,
      }],
      total: 1,
    });
    mocks.legacy.testWebhook.mockResolvedValue({
      id: 'delivery-1',
      webhookId: 'legacy-1',
      eventType: 'webhook.test',
      statusCode: 200,
      responseTimeMs: 15,
      success: true,
      error: null,
      attemptedAt: '2026-08-22T12:00:00Z',
      payloadPreview: null,
    });
    const user = userEvent.setup();

    render(<WebhooksPage />);

    expect(await screen.findByText(/legacy compatibility mode/i)).toBeInTheDocument();
    expect(screen.getByText('https://legacy.example/hook')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^test$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /show delivery history/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /generate a new secret/i })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /^test$/i }));
    await waitFor(() => expect(mocks.legacy.testWebhook).toHaveBeenCalledWith('legacy-1'));
    expect(mocks.canonical.getWebhookCatalog).not.toHaveBeenCalled();
  });

  it('keeps an older delivery-history response out of the newly expanded row', async () => {
    const firstHistory = deferred<{ items: Array<Record<string, unknown>>; total: number }>();
    const secondHistory = deferred<{ items: Array<Record<string, unknown>>; total: number }>();
    const firstDelivery = {
      id: 'delivery-first',
      webhookId: 'legacy-1',
      eventType: 'first.completed',
      statusCode: 200,
      responseTimeMs: 15,
      success: true,
      error: null,
      attemptedAt: '2026-08-22T12:00:00Z',
      payloadPreview: null,
    };
    const secondDelivery = {
      ...firstDelivery,
      id: 'delivery-second',
      webhookId: 'legacy-2',
      eventType: 'second.completed',
    };
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [
        {
          id: 'legacy-1',
          targetUrl: 'https://legacy-one.example/hook',
          eventTypes: ['incident.created'],
          enabled: true,
          createdAt: null,
          updatedAt: null,
        },
        {
          id: 'legacy-2',
          targetUrl: 'https://legacy-two.example/hook',
          eventTypes: ['incident.created'],
          enabled: true,
          createdAt: null,
          updatedAt: null,
        },
      ],
      total: 2,
    });
    mocks.legacy.getWebhookDeliveries.mockImplementation((webhookId: string) => (
      webhookId === 'legacy-1' ? firstHistory.promise : secondHistory.promise
    ));
    const user = userEvent.setup();
    render(<WebhooksPage />);

    const firstRow = (await screen.findByText('https://legacy-one.example/hook')).closest('tr');
    const secondRow = screen.getByText('https://legacy-two.example/hook').closest('tr');
    expect(firstRow).not.toBeNull();
    expect(secondRow).not.toBeNull();
    await user.click(within(firstRow!).getByRole('button', { name: /show delivery history/i }));
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledWith(
      'legacy-1',
      { limit: 50, offset: 0 },
    ));
    await user.click(within(secondRow!).getByRole('button', { name: /show delivery history/i }));
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledWith(
      'legacy-2',
      { limit: 50, offset: 0 },
    ));

    await act(async () => {
      firstHistory.resolve({ items: [firstDelivery], total: 1 });
      await firstHistory.promise;
    });

    expect(screen.getByText(/loading delivery history/i)).toBeInTheDocument();
    expect(screen.queryByText('first.completed')).not.toBeInTheDocument();

    await act(async () => {
      secondHistory.resolve({ items: [secondDelivery], total: 1 });
      await secondHistory.promise;
    });
    expect(await screen.findByText('second.completed')).toBeInTheDocument();
    expect(screen.queryByText('first.completed')).not.toBeInTheDocument();
  });

  it('does not refresh a tested webhook after the admin expands another row', async () => {
    const testDelivery = deferred<Record<string, unknown>>();
    const historyByWebhook = {
      'legacy-1': {
        items: [],
        total: 0,
      },
      'legacy-2': {
        items: [{
          id: 'delivery-second',
          webhookId: 'legacy-2',
          eventType: 'second.completed',
          statusCode: 200,
          responseTimeMs: 15,
          success: true,
          error: null,
          attemptedAt: '2026-08-22T12:00:00Z',
          payloadPreview: null,
        }],
        total: 1,
      },
    };
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [
        {
          id: 'legacy-1',
          targetUrl: 'https://legacy-one.example/hook',
          eventTypes: ['incident.created'],
          enabled: true,
          createdAt: null,
          updatedAt: null,
        },
        {
          id: 'legacy-2',
          targetUrl: 'https://legacy-two.example/hook',
          eventTypes: ['incident.created'],
          enabled: true,
          createdAt: null,
          updatedAt: null,
        },
      ],
      total: 2,
    });
    mocks.legacy.getWebhookDeliveries.mockImplementation((webhookId: 'legacy-1' | 'legacy-2') => (
      Promise.resolve(historyByWebhook[webhookId])
    ));
    mocks.legacy.testWebhook.mockReturnValue(testDelivery.promise);
    const user = userEvent.setup();
    render(<WebhooksPage />);

    const firstRow = (await screen.findByText('https://legacy-one.example/hook')).closest('tr');
    const secondRow = screen.getByText('https://legacy-two.example/hook').closest('tr');
    expect(firstRow).not.toBeNull();
    expect(secondRow).not.toBeNull();
    await user.click(within(firstRow!).getByRole('button', { name: /show delivery history/i }));
    await screen.findByText(/no legacy deliveries recorded/i);
    await user.click(within(firstRow!).getByRole('button', { name: /^test$/i }));
    await waitFor(() => expect(mocks.legacy.testWebhook).toHaveBeenCalledWith('legacy-1'));
    await user.click(within(secondRow!).getByRole('button', { name: /show delivery history/i }));
    expect(await screen.findByText('second.completed')).toBeInTheDocument();

    await act(async () => {
      testDelivery.resolve({
        id: 'delivery-test',
        webhookId: 'legacy-1',
        eventType: 'webhook.test',
        statusCode: 200,
        responseTimeMs: 15,
        success: true,
        error: null,
        attemptedAt: '2026-08-22T12:00:00Z',
        payloadPreview: null,
      });
      await testDelivery.promise;
    });

    await waitFor(() => expect(mocks.toastSuccess).toHaveBeenCalledWith(
      'Legacy test delivery succeeded',
    ));
    expect(mocks.legacy.getWebhookDeliveries.mock.calls.filter(
      ([webhookId]) => webhookId === 'legacy-1',
    )).toHaveLength(1);
    expect(screen.getByText('second.completed')).toBeInTheDocument();
  });

  it('does not refresh from an earlier expansion after the same row is re-expanded', async () => {
    const testDelivery = deferred<Record<string, unknown>>();
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [{
        id: 'legacy-1',
        targetUrl: 'https://legacy.example/hook',
        eventTypes: ['incident.created'],
        enabled: true,
        createdAt: null,
        updatedAt: null,
      }],
      total: 1,
    });
    mocks.legacy.testWebhook.mockReturnValue(testDelivery.promise);
    const user = userEvent.setup();
    render(<WebhooksPage />);

    const row = (await screen.findByText('https://legacy.example/hook')).closest('tr');
    expect(row).not.toBeNull();
    await user.click(within(row!).getByRole('button', { name: /show delivery history/i }));
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(1));
    await user.click(within(row!).getByRole('button', { name: /^test$/i }));
    await waitFor(() => expect(mocks.legacy.testWebhook).toHaveBeenCalledWith('legacy-1'));
    await user.click(within(row!).getByRole('button', { name: /hide delivery history/i }));
    await user.click(within(row!).getByRole('button', { name: /show delivery history/i }));
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(2));

    await act(async () => {
      testDelivery.resolve({
        id: 'delivery-old-expansion',
        webhookId: 'legacy-1',
        eventType: 'webhook.test',
        statusCode: 200,
        responseTimeMs: 15,
        success: true,
        error: null,
        attemptedAt: '2026-08-22T12:00:00Z',
        payloadPreview: null,
      });
      await testDelivery.promise;
    });

    await waitFor(() => expect(mocks.toastSuccess).toHaveBeenCalledWith(
      'Legacy test delivery succeeded',
    ));
    expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(2);
  });

  it('only lets the latest overlapping legacy test refresh delivery history', async () => {
    const firstTest = deferred<Record<string, unknown>>();
    const secondTest = deferred<Record<string, unknown>>();
    const delivery = {
      id: 'delivery-test',
      webhookId: 'legacy-1',
      eventType: 'webhook.test',
      statusCode: 200,
      responseTimeMs: 15,
      success: true,
      error: null,
      attemptedAt: '2026-08-22T12:00:00Z',
      payloadPreview: null,
    };
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [{
        id: 'legacy-1',
        targetUrl: 'https://legacy.example/hook',
        eventTypes: ['incident.created'],
        enabled: true,
        createdAt: null,
        updatedAt: null,
      }],
      total: 1,
    });
    mocks.legacy.testWebhook
      .mockReturnValueOnce(firstTest.promise)
      .mockReturnValueOnce(secondTest.promise);
    const user = userEvent.setup();
    render(<WebhooksPage />);

    const row = (await screen.findByText('https://legacy.example/hook')).closest('tr');
    expect(row).not.toBeNull();
    await user.click(within(row!).getByRole('button', { name: /show delivery history/i }));
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(1));
    const testButton = within(row!).getByRole('button', { name: /^test$/i });
    await user.click(testButton);
    await user.click(testButton);
    await waitFor(() => expect(mocks.legacy.testWebhook).toHaveBeenCalledTimes(2));

    await act(async () => {
      secondTest.resolve({ ...delivery, id: 'delivery-newer-test' });
      await secondTest.promise;
    });
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(2));

    await act(async () => {
      firstTest.resolve({ ...delivery, id: 'delivery-older-test' });
      await firstTest.promise;
    });

    expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(2);
  });

  it('invalidates an older test history refresh when a newer same-row test starts', async () => {
    const olderHistory = deferred<{ items: Array<Record<string, unknown>>; total: number }>();
    const newerTest = deferred<Record<string, unknown>>();
    const delivery = {
      id: 'delivery-test',
      webhookId: 'legacy-1',
      eventType: 'webhook.test',
      statusCode: 200,
      responseTimeMs: 15,
      success: true,
      error: null,
      attemptedAt: '2026-08-22T12:00:00Z',
      payloadPreview: null,
    };
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [{
        id: 'legacy-1',
        targetUrl: 'https://legacy.example/hook',
        eventTypes: ['incident.created'],
        enabled: true,
        createdAt: null,
        updatedAt: null,
      }],
      total: 1,
    });
    mocks.legacy.getWebhookDeliveries
      .mockResolvedValueOnce({ items: [], total: 0 })
      .mockReturnValueOnce(olderHistory.promise)
      .mockResolvedValueOnce({
        items: [{ ...delivery, id: 'history-newer', eventType: 'newer.completed' }],
        total: 1,
      });
    mocks.legacy.testWebhook
      .mockResolvedValueOnce({ ...delivery, id: 'test-older' })
      .mockReturnValueOnce(newerTest.promise);
    const user = userEvent.setup();
    render(<WebhooksPage />);

    const row = (await screen.findByText('https://legacy.example/hook')).closest('tr');
    expect(row).not.toBeNull();
    await user.click(within(row!).getByRole('button', { name: /show delivery history/i }));
    await screen.findByText(/no legacy deliveries recorded/i);
    const testButton = within(row!).getByRole('button', { name: /^test$/i });
    await user.click(testButton);
    await waitFor(() => expect(mocks.legacy.getWebhookDeliveries).toHaveBeenCalledTimes(2));
    expect(screen.getByText(/loading delivery history/i)).toBeInTheDocument();
    await user.click(testButton);
    await waitFor(() => expect(mocks.legacy.testWebhook).toHaveBeenCalledTimes(2));

    await act(async () => {
      olderHistory.resolve({
        items: [{ ...delivery, id: 'history-older', eventType: 'older.completed' }],
        total: 1,
      });
      await olderHistory.promise;
    });

    expect(screen.queryByText('older.completed')).not.toBeInTheDocument();

    await act(async () => {
      newerTest.resolve({ ...delivery, id: 'test-newer' });
      await newerTest.promise;
    });
    expect(await screen.findByText('newer.completed')).toBeInTheDocument();
  });

  it('rejects an unsafe destination before calling the legacy API', async () => {
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    const user = userEvent.setup();
    render(<WebhooksPage />);
    await screen.findByRole('heading', { name: 'Webhooks' });
    const dialog = await openCreateDialog(user);

    await user.type(within(dialog).getByLabelText(/destination url/i), 'file:///tmp/callback');
    await user.type(within(dialog).getByLabelText(/^events$/i), 'incident.created');
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }));

    expect(await within(dialog).findByText(/absolute HTTP or HTTPS URL/i)).toBeInTheDocument();
    expect(mocks.legacy.createWebhook).not.toHaveBeenCalled();
  });

  it('does not show a stale creation error after pagehide invalidates the request', async () => {
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    let rejectCreate: ((error: Error) => void) | undefined;
    mocks.legacy.createWebhook.mockReturnValue(new Promise((_resolve, reject) => {
      rejectCreate = reject;
    }));
    const user = userEvent.setup();
    render(<WebhooksPage />);
    await screen.findByRole('heading', { name: 'Webhooks' });
    const dialog = await openCreateDialog(user);
    await user.type(
      within(dialog).getByLabelText(/destination url/i),
      'https://legacy.example/hook',
    );
    await user.type(within(dialog).getByLabelText(/^events$/i), 'incident.created');
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(mocks.legacy.createWebhook).toHaveBeenCalledTimes(1));
    await act(async () => {
      window.dispatchEvent(new Event('pagehide'));
      rejectCreate?.(new Error('late failure'));
      await Promise.resolve();
    });

    expect(mocks.toastError).not.toHaveBeenCalledWith(
      'Webhook creation failed',
      expect.any(String),
    );
  });

  it('does not restore a legacy secret when creation resolves after pagehide', async () => {
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    let resolveCreate: ((value: { signingSecret: string }) => void) | undefined;
    mocks.legacy.createWebhook.mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    const user = userEvent.setup();
    render(<WebhooksPage />);
    await screen.findByRole('heading', { name: 'Webhooks' });
    const dialog = await openCreateDialog(user);
    await user.type(
      within(dialog).getByLabelText(/destination url/i),
      'https://legacy.example/hook',
    );
    await user.type(within(dialog).getByLabelText(/^events$/i), 'incident.created');
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(mocks.legacy.createWebhook).toHaveBeenCalledTimes(1));
    await act(async () => {
      window.dispatchEvent(new Event('pagehide'));
      resolveCreate?.({ signingSecret: SIGNING_SECRET });
      await Promise.resolve();
    });

    expect(screen.queryByDisplayValue(SIGNING_SECRET)).not.toBeInTheDocument();
    expect(mocks.toastSuccess).not.toHaveBeenCalledWith('Legacy webhook created');
  });

  it('locks the legacy form and row mutations while creation is in flight', async () => {
    mocks.detect.mockResolvedValue({
      kind: 'legacy',
      status: { ...STATUS, route_selection: 'legacy' },
      client: mocks.legacy,
    });
    mocks.legacy.getWebhooks.mockResolvedValue({
      items: [{
        id: 'legacy-1',
        targetUrl: 'https://legacy.example/existing',
        eventTypes: ['incident.created'],
        enabled: true,
        createdAt: null,
        updatedAt: null,
      }],
      total: 1,
    });
    mocks.legacy.createWebhook.mockReturnValue(new Promise(() => {}));
    const user = userEvent.setup();
    render(<WebhooksPage />);
    await screen.findByText('https://legacy.example/existing');
    const dialog = await openCreateDialog(user);
    const destination = within(dialog).getByLabelText(/destination url/i);
    const events = within(dialog).getByLabelText(/^events$/i);
    await user.type(destination, 'https://legacy.example/new');
    await user.type(events, 'incident.created');
    await user.click(within(dialog).getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(mocks.legacy.createWebhook).toHaveBeenCalledTimes(1));

    expect(destination).toBeDisabled();
    expect(events).toBeDisabled();
    expect(screen.getByRole('button', { name: /^test$/i, hidden: true })).toBeDisabled();
    expect(screen.getByRole('button', { name: /show delivery history/i, hidden: true }))
      .toBeDisabled();
    expect(screen.getByRole('button', { name: /^disable$/i, hidden: true })).toBeDisabled();
    expect(screen.getByRole('button', { name: /delete legacy webhook/i, hidden: true }))
      .toBeDisabled();
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

  it('atomically rejects a second non-retryable legacy secret command', async () => {
    const firstRun = vi.fn(() => new Promise<{
      signing_secret: string;
      replayed: boolean;
    }>(() => {}));
    const secondRun = vi.fn().mockResolvedValue({
      signing_secret: SECOND_SIGNING_SECRET,
      replayed: false,
    });
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

    await act(async () => {
      void result.current.startLegacySecretCommand(firstRun);
      void result.current.startLegacySecretCommand(secondRun);
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
