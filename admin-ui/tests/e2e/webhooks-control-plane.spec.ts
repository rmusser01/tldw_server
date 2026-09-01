import { expect, test, type Page, type Route } from '@playwright/test';

import type {
  WebhookDelivery,
  WebhookDeliveryAttempt,
  WebhookRegistration,
  WebhookStatus,
} from '../../types';
import { installAdminApiRoutes, setAuthenticatedSession } from './smoke-helpers';

const DESTINATION_URL = 'https://receiver.example/private/control-plane-hook';
const CREATE_SECRET = `whsec_${'c'.repeat(64)}`;
const ROTATED_SECRET = `whsec_${'r'.repeat(64)}`;
const SECOND_ROTATED_SECRET = `whsec_${'s'.repeat(64)}`;
const DELIVERY_ID = '11111111-1111-4111-8111-111111111111';

const STATUS: WebhookStatus = {
  mode: 'on',
  route_selection: 'canonical',
  schema_ready: true,
  key_state: 'available',
  delivery_capability_ready: true,
  delivery: {
    canonical_schema_version: 1,
    schema_ready: true,
    delivery_schema_ready: true,
    migration_complete: true,
    key_ready: true,
    key_primary_match: true,
    jobs_database_ready: true,
    queue_ready: true,
    job_type_ready: true,
    jobs_backend: 'postgres',
    worker: {
      component: 'worker',
      ready: true,
      reason_code: null,
      heartbeat_age_seconds: 2,
    },
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
      heartbeat_age_seconds: 7,
    },
    backlog: {
      pending: 0,
      enqueue_claimed: 0,
      queued: 1,
      processing: 0,
      retry_wait: 0,
    },
    oldest_nonterminal_age_seconds: 9,
    acquisition_ready: true,
    acquisition_reason_code: null,
    delivery_capability_ready: true,
  },
  limits: {
    registrations: 100,
    active_registrations: 25,
    current_registrations: 0,
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
  expires_at: '2026-08-29T12:00:00Z',
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
  request_timeout_seconds: 12,
  status_code: 204,
  latency_ms: 42,
  reason_code: null,
  requested_retry_delay_seconds: null,
  started_at: '2026-08-22T12:00:00Z',
  finished_at: '2026-08-22T12:00:01Z',
};

type CapturedRequest = {
  method: string;
  url: string;
  body: string | null;
  headers: Record<string, string>;
};

type RotationReplay = {
  registration: WebhookRegistration;
  secret: string;
  calls: number;
};

const webhookEtag = (registration: WebhookRegistration) => (
  `"admin-webhook-${registration.id}-r${registration.revision}"`
);

const fulfillJson = async (
  route: Route,
  payload: unknown,
  options: { status?: number; requestId?: string; etag?: string } = {},
) => {
  await route.fulfill({
    status: options.status ?? 200,
    contentType: 'application/json',
    headers: {
      'x-request-id': options.requestId ?? 'request-webhooks-e2e',
      ...(options.etag ? { etag: options.etag } : {}),
    },
    body: JSON.stringify(payload),
  });
};

const approvePrivilegedAction = async (
  page: Page,
  title: string | RegExp,
  confirmName: string | RegExp,
) => {
  const dialog = page.getByRole('dialog', { name: title });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByLabel(/^reason$/i)).toHaveCount(0);
  await expect(dialog.getByLabel(/current password/i)).toHaveCount(0);
  await dialog.getByRole('button', { name: confirmName, exact: true }).click();
};

test('manages the canonical webhook lifecycle without leaking one-time secrets', async ({
  context,
  page,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await installAdminApiRoutes(page);
  await setAuthenticatedSession(page);

  let registration: WebhookRegistration | null = null;
  let patchAttempts = 0;
  let conflictInjected = false;
  let deleteIfMatch: string | null = null;
  const capturedRequests: CapturedRequest[] = [];
  const rotationReplays = new Map<string, RotationReplay>();
  const browserMessages: string[] = [];

  page.on('console', (message) => browserMessages.push(message.text()));
  page.on('pageerror', (error) => browserMessages.push(error.message));

  await page.route('**/api/proxy/admin/webhooks**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace('/api/proxy', '');
    const method = request.method();
    const headers = await request.allHeaders();
    capturedRequests.push({ method, url: request.url(), body: request.postData(), headers });

    if (method === 'GET' && path === '/admin/webhooks/status') {
      await fulfillJson(route, {
        ...STATUS,
        limits: {
          ...STATUS.limits,
          current_registrations: registration ? 1 : 0,
        },
      });
      return;
    }

    if (method === 'GET' && path === '/admin/webhooks/catalog') {
      await fulfillJson(route, {
        api_version: '1',
        events: [
          { event_type: 'incident.created', description: 'An incident was created.' },
          { event_type: 'user.created', description: 'A user was created.' },
        ],
        registration_limit: 100,
        active_limit: 25,
      });
      return;
    }

    if (method === 'GET' && path === '/admin/webhooks') {
      const offset = Number(url.searchParams.get('offset') ?? '0');
      await fulfillJson(route, {
        items: registration && offset === 0 ? [registration] : [],
        total: registration ? 1 : 0,
        limit: 20,
        offset,
      });
      return;
    }

    if (method === 'POST' && path === '/admin/webhooks') {
      const body = request.postDataJSON() as {
        url: string;
        event_types: string[];
        description: string;
        timeout_seconds: number;
        active?: boolean;
      };
      expect(body.active).toBeUndefined();
      expect(body.url).toBe(DESTINATION_URL);
      expect(headers['idempotency-key']).toMatch(/^[0-9a-f]{32}$/);
      registration = {
        id: 41,
        description: body.description,
        target_display: 'https://receiver.example',
        target_hostname: 'receiver.example',
        event_types: body.event_types,
        active: false,
        timeout_seconds: body.timeout_seconds,
        revision: 1,
        delivery_config_version: 1,
        secret_version: 1,
        secret_rotation_required: false,
        created_by: 1,
        updated_by: 1,
        created_at: '2026-08-22T12:00:00Z',
        updated_at: '2026-08-22T12:00:00Z',
      };
      await fulfillJson(route, {
        registration,
        signing_secret: CREATE_SECRET,
        replayed: false,
      }, { status: 201, etag: webhookEtag(registration) });
      return;
    }

    if (method === 'GET' && path === '/admin/webhooks/41' && registration) {
      await fulfillJson(route, registration, { etag: webhookEtag(registration) });
      return;
    }

    if (method === 'GET' && path === '/admin/webhooks/41/deliveries' && registration) {
      expect(url.searchParams.get('limit')).toBe('50');
      expect(url.searchParams.get('offset')).toBe('0');
      await fulfillJson(route, {
        items: [{ delivery: DELIVERY, attempts: [ATTEMPT] }],
        total: 1,
        limit: 50,
        offset: 0,
      });
      return;
    }

    if (method === 'POST' && path === '/admin/webhooks/41/test' && registration) {
      expect(headers['if-match']).toBe(webhookEtag(registration));
      expect(headers['idempotency-key']).toMatch(/^[0-9a-f]{32}$/);
      expect(request.postDataJSON()).toEqual({
        delivery_config_version: registration.delivery_config_version,
      });
      await fulfillJson(route, {
        delivery: { ...DELIVERY, kind: 'test', event_type: 'webhook.test' },
        attempt: ATTEMPT,
        idempotent_replay: false,
        in_progress: false,
      });
      return;
    }

    if (
      method === 'POST'
      && path === `/admin/webhooks/41/deliveries/${DELIVERY_ID}/redeliver`
      && registration
    ) {
      expect(headers['if-match']).toBe(webhookEtag(registration));
      expect(headers['idempotency-key']).toMatch(/^[0-9a-f]{32}$/);
      expect(request.postDataJSON()).toEqual({
        delivery_config_version: registration.delivery_config_version,
        confirm_changed_configuration: false,
      });
      await fulfillJson(route, {
        delivery: {
          ...DELIVERY,
          id: '44444444-4444-4444-8444-444444444444',
          kind: 'manual',
          state: 'pending',
          attempt_count: 0,
          status_code: null,
          latency_ms: null,
          terminal_at: null,
          redelivery_of_id: DELIVERY_ID,
        },
        idempotent_replay: false,
      }, { status: 202 });
      return;
    }

    if (method === 'PATCH' && path === '/admin/webhooks/41' && registration) {
      patchAttempts += 1;
      expect(headers['if-match']).toBe(webhookEtag(registration));
      const body = request.postDataJSON() as Partial<WebhookRegistration> & { url?: string };
      expect(body).not.toHaveProperty('target_display');

      if (patchAttempts === 2) {
        conflictInjected = true;
        registration = {
          ...registration,
          description: 'Changed elsewhere',
          revision: registration.revision + 1,
          updated_at: '2026-08-22T12:02:00Z',
        };
        await fulfillJson(route, {
          error: {
            code: 'admin_webhook_precondition_failed',
            message: 'Webhook changed before this update.',
            request_id: 'request-webhook-conflict',
          },
        }, { status: 412, requestId: 'request-webhook-conflict' });
        return;
      }

      registration = {
        ...registration,
        description: body.description ?? registration.description,
        event_types: body.event_types ?? registration.event_types,
        timeout_seconds: body.timeout_seconds ?? registration.timeout_seconds,
        revision: registration.revision + 1,
        updated_at: '2026-08-22T12:01:00Z',
      };
      await fulfillJson(route, registration, { etag: webhookEtag(registration) });
      return;
    }

    if (method === 'POST' && path === '/admin/webhooks/41/rotate-secret' && registration) {
      const idempotencyKey = headers['idempotency-key'];
      expect(idempotencyKey).toMatch(/^[0-9a-f]{32}$/);
      const replay = rotationReplays.get(idempotencyKey);
      if (replay) {
        replay.calls += 1;
        expect(headers['if-match']).toBe(webhookEtag({
          ...replay.registration,
          revision: replay.registration.revision - 1,
        }));
        await fulfillJson(route, {
          registration: replay.registration,
          signing_secret: replay.secret,
          replayed: true,
        }, { etag: webhookEtag(replay.registration) });
        return;
      }

      expect(headers['if-match']).toBe(webhookEtag(registration));
      const secret = rotationReplays.size === 0 ? ROTATED_SECRET : SECOND_ROTATED_SECRET;
      registration = {
        ...registration,
        revision: registration.revision + 1,
        secret_version: registration.secret_version + 1,
        secret_rotation_required: false,
        updated_at: '2026-08-22T12:03:00Z',
      };
      rotationReplays.set(idempotencyKey, { registration, secret, calls: 1 });
      await fulfillJson(route, { detail: 'Backend unavailable' }, {
        status: 502,
        requestId: 'request-rotate-response-lost',
      });
      return;
    }

    if (method === 'DELETE' && path === '/admin/webhooks/41' && registration) {
      deleteIfMatch = headers['if-match'] ?? null;
      expect(deleteIfMatch).toBe(webhookEtag(registration));
      registration = null;
      await fulfillJson(route, { deleted: true, id: 41 });
      return;
    }

    await fulfillJson(route, { detail: `Unhandled webhook route: ${method} ${path}` }, {
      status: 500,
    });
  });

  await page.goto('/webhooks');
  await expect(page.getByRole('heading', { name: 'Webhooks', exact: true })).toBeVisible();
  await expect(page.getByText(/no webhooks configured/i)).toBeVisible();

  await page.getByRole('button', { name: /add webhook/i }).click();
  const createDialog = page.getByRole('dialog', { name: /add webhook/i });
  await createDialog.getByLabel(/destination url/i).fill(DESTINATION_URL);
  await createDialog.getByLabel(/^description$/i).fill('Private beta receiver');
  await createDialog.getByLabel(/timeout/i).fill('12');
  await createDialog.getByLabel(/user\.created/i).check();
  await createDialog.getByRole('button', { name: /^create$/i }).click();

  const createSecretDialog = page.getByRole('dialog', { name: /signing secret/i });
  await expect(createSecretDialog.getByLabel(/^signing secret$/i)).toHaveValue(CREATE_SECRET);
  await expect(createSecretDialog.getByRole('button', { name: /^done$/i })).toBeDisabled();
  await createSecretDialog.getByRole('button', { name: /copy signing secret/i }).click();
  await createSecretDialog.getByLabel(/i have stored this signing secret/i).check();
  await createSecretDialog.getByRole('button', { name: /^done$/i }).click();
  await expect(page.locator('#webhook-signing-secret')).toHaveCount(0);
  await expect(page.getByText('https://receiver.example', { exact: true })).toBeVisible();
  await expect(page.getByText(DESTINATION_URL, { exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: /edit metadata/i }).click();
  const metadataDialog = page.getByRole('dialog', { name: /edit webhook metadata/i });
  await metadataDialog.getByLabel(/^description$/i).fill('Reviewed receiver');
  await metadataDialog.getByRole('button', { name: /save changes/i }).click();
  await approvePrivilegedAction(page, /^save changes$/i, /^save changes$/i);
  await expect(page.getByText('Reviewed receiver', { exact: true })).toBeVisible();

  const firstPatch = capturedRequests.find((request) => request.method === 'PATCH');
  expect(firstPatch?.headers['if-match']).toBe('"admin-webhook-41-r1"');
  expect(firstPatch?.body).not.toContain('"url"');

  await page.getByRole('button', { name: /edit metadata/i }).click();
  await page.getByRole('dialog', { name: /edit webhook metadata/i })
    .getByRole('button', { name: /save changes/i })
    .click();
  await approvePrivilegedAction(page, /^save changes$/i, /^save changes$/i);
  await expect(page.getByText(/review the current webhook before retrying/i)).toBeVisible();
  await expect(page.getByText('Changed elsewhere', { exact: true })).toBeVisible();
  expect(conflictInjected).toBe(true);
  expect(patchAttempts).toBe(2);

  await page.getByRole('button', { name: /edit metadata/i }).click();
  const refreshedMetadataDialog = page.getByRole('dialog', { name: /edit webhook metadata/i });
  await expect(refreshedMetadataDialog.getByLabel(/^description$/i)).toHaveValue('Changed elsewhere');
  await refreshedMetadataDialog.getByRole('button', { name: /save changes/i }).click();
  const reReview = page.getByRole('dialog', { name: /^save changes$/i });
  await expect(reReview).toBeVisible();
  await reReview.getByRole('button', { name: /^cancel$/i }).click();
  expect(patchAttempts).toBe(2);
  await page.getByRole('dialog', { name: /edit webhook metadata/i })
    .getByRole('button', { name: /^cancel$/i })
    .click();

  await expect(page.getByLabel('Webhook delivery runtime')).toContainText('Worker ready');
  await expect(page.getByLabel('Webhook delivery runtime')).toContainText('1 nonterminal delivery');

  await page.getByRole('button', { name: /^run test$/i }).click();
  await approvePrivilegedAction(page, /^run webhook test$/i, /^run test$/i);
  await expect(page.getByText(/test delivery succeeded/i)).toBeVisible();

  await page.getByRole('button', { name: /show delivery history/i }).click();
  const history = page.getByRole('region', { name: /delivery history/i });
  await expect(history.getByText('incident.created', { exact: true })).toBeVisible();
  await expect(history.getByText('HTTP 204 (2xx)', { exact: true }).first()).toBeVisible();
  await expect(history.getByText(/attempt 1: succeeded/i)).toBeVisible();
  await history.getByRole('button', { name: /redeliver incident\.created/i }).click();
  await approvePrivilegedAction(page, /^redeliver webhook event$/i, /^redeliver event$/i);
  await expect(page.getByText(/manual redelivery accepted/i)).toBeVisible();

  await page.getByRole('button', { name: /generate a new secret/i }).click();
  await approvePrivilegedAction(
    page,
    /generate a new signing secret/i,
    /generate secret/i,
  );
  const retryRotate = page.getByRole('button', { name: /retry same command/i });
  await expect(retryRotate).toBeVisible();
  await retryRotate.click();
  await expect(page.getByRole('textbox', { name: /^signing secret$/i }))
    .toHaveValue(ROTATED_SECRET);
  await expect(page.getByText(/response was recovered from the original command/i)).toBeVisible();

  const firstRotation = [...rotationReplays.values()][0];
  expect(firstRotation?.calls).toBe(2);
  const rotateRequests = capturedRequests.filter((request) => (
    request.method === 'POST' && request.url.includes('/rotate-secret')
  ));
  expect(rotateRequests).toHaveLength(2);
  expect(rotateRequests[0]?.headers['idempotency-key']).toBe(
    rotateRequests[1]?.headers['idempotency-key'],
  );

  await page.evaluate(() => {
    window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: false }));
  });
  await expect(page.locator('#webhook-signing-secret')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /retry same command/i })).toHaveCount(0);
  await page.evaluate(() => {
    window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }));
  });
  await expect(page.locator('#webhook-signing-secret')).toHaveCount(0);

  await page.getByRole('button', { name: /generate a new secret/i }).click();
  await approvePrivilegedAction(
    page,
    /generate a new signing secret/i,
    /generate secret/i,
  );
  await expect(page.getByRole('button', { name: /retry same command/i })).toBeVisible();
  await page.evaluate(() => {
    window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: false }));
  });
  await expect(page.getByRole('button', { name: /retry same command/i })).toHaveCount(0);
  await page.evaluate(() => {
    window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }));
  });
  await expect(page.getByText(SECOND_ROTATED_SECRET)).toHaveCount(0);
  await expect(page.getByRole('button', { name: /retry same command/i })).toHaveCount(0);

  await page.getByRole('button', { name: /delete webhook/i }).click();
  await approvePrivilegedAction(page, /^delete webhook$/i, /^delete webhook$/i);
  await expect(page.getByText(/no webhooks configured/i)).toBeVisible();
  expect(deleteIfMatch).toBe('"admin-webhook-41-r5"');

  const leakedRequestMaterial = capturedRequests
    .map((request) => `${request.url}\n${request.body ?? ''}`)
    .join('\n');
  for (const secret of [CREATE_SECRET, ROTATED_SECRET, SECOND_ROTATED_SECRET]) {
    expect(leakedRequestMaterial).not.toContain(secret);
    expect(browserMessages.join('\n')).not.toContain(secret);
  }
  expect(browserMessages.join('\n')).not.toContain(DESTINATION_URL);

  const browserPersistence = await page.evaluate(() => ({
    localStorage: JSON.stringify({ ...localStorage }),
    sessionStorage: JSON.stringify({ ...sessionStorage }),
    href: window.location.href,
  }));
  for (const secret of [CREATE_SECRET, ROTATED_SECRET, SECOND_ROTATED_SECRET]) {
    expect(JSON.stringify(browserPersistence)).not.toContain(secret);
  }
  expect(capturedRequests.some((request) => /\/test(?:[/?]|$)/.test(request.url))).toBe(true);
  expect(capturedRequests.some((request) => /\/deliveries(?:[/?]|$)/.test(request.url))).toBe(true);
  expect(capturedRequests.some((request) => /\/redeliver(?:[/?]|$)/.test(request.url))).toBe(true);
});
