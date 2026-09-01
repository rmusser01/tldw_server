/* @vitest-environment node */

import { beforeEach, describe, expect, it, vi } from 'vitest';

const httpMocks = vi.hoisted(() => ({
  requestJson: vi.fn(),
  requestJsonWithMetadata: vi.fn(),
  requestText: vi.fn(),
}));

vi.mock('./http', () => {
  class ApiError extends Error {
    status: number;
    detail?: unknown;

    constructor(status: number, message: string, detail?: unknown) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.detail = detail;
    }
  }

  class WebhookContractError extends ApiError {
    requestId: string | null;

    constructor(status: number, message: string, requestId: string | null = null) {
      super(status, message);
      this.name = 'WebhookContractError';
      this.requestId = requestId;
    }
  }

  return {
    ApiError,
    WebhookContractError,
    requestJson: httpMocks.requestJson,
    requestJsonWithMetadata: httpMocks.requestJsonWithMetadata,
    requestText: httpMocks.requestText,
  };
});

import * as apiClientModule from './api-client';
import { api, canonicalWebhookApi } from './api-client';
import { WebhookContractError } from './http';
import type {
  WebhookCreateRequest,
  WebhookDelivery,
  WebhookDeliveryAttempt,
  WebhookRegistration,
  WebhookSecretResponse,
  WebhookStatus,
} from '@/types';

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

const CREATE_BODY: WebhookCreateRequest = {
  url: 'https://receiver.example/hooks/private',
  event_types: ['incident.created'],
  description: 'Incident receiver',
  timeout_seconds: 10,
};

const SECRET_RESPONSE: WebhookSecretResponse = {
  registration: REGISTRATION,
  signing_secret: `whsec_${'a'.repeat(64)}`,
  replayed: false,
};

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
    worker: {
      component: 'worker',
      ready: true,
      reason_code: null,
      heartbeat_age_seconds: 3,
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
      heartbeat_age_seconds: 5,
    },
    backlog: {
      pending: 1,
      enqueue_claimed: 0,
      queued: 2,
      processing: 1,
      retry_wait: 0,
    },
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

const DELIVERY_ID = '11111111-1111-4111-8111-111111111111';
const EVENT_ID = '22222222-2222-4222-8222-222222222222';
const ATTEMPT_ID = '33333333-3333-4333-8333-333333333333';

const DELIVERY: WebhookDelivery = {
  id: DELIVERY_ID,
  event_id: EVENT_ID,
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
  id: ATTEMPT_ID,
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

describe('canonical webhook API client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('carries idempotency and strong conditional headers through mutations', async () => {
    httpMocks.requestJsonWithMetadata
      .mockResolvedValueOnce({
        data: SECRET_RESPONSE,
        status: 201,
        etag: '"admin-webhook-41-r2"',
        requestId: 'request-create',
      })
      .mockResolvedValueOnce({
        data: { ...REGISTRATION, revision: 3 },
        status: 200,
        etag: '"admin-webhook-41-r3"',
        requestId: 'request-patch',
      })
      .mockResolvedValueOnce({
        data: {
          ...SECRET_RESPONSE,
          registration: { ...REGISTRATION, revision: 4, secret_version: 2 },
        },
        status: 200,
        etag: '"admin-webhook-41-r4"',
        requestId: 'request-rotate',
      });

    await api.createWebhook(CREATE_BODY, '0123456789abcdef0123456789abcdef');
    await api.updateWebhook(41, { description: 'Updated' }, '"admin-webhook-41-r2"');
    await api.rotateWebhookSecret(
      41,
      '"admin-webhook-41-r3"',
      'abcdef0123456789abcdef0123456789',
    );

    expect(httpMocks.requestJsonWithMetadata).toHaveBeenNthCalledWith(
      1,
      '/admin/webhooks',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Idempotency-Key': '0123456789abcdef0123456789abcdef' },
        body: JSON.stringify(CREATE_BODY),
      }),
    );
    expect(httpMocks.requestJsonWithMetadata).toHaveBeenNthCalledWith(
      2,
      '/admin/webhooks/41',
      expect.objectContaining({
        method: 'PATCH',
        headers: { 'If-Match': '"admin-webhook-41-r2"' },
      }),
    );
    expect(httpMocks.requestJsonWithMetadata).toHaveBeenNthCalledWith(
      3,
      '/admin/webhooks/41/rotate-secret',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'If-Match': '"admin-webhook-41-r3"',
          'Idempotency-Key': 'abcdef0123456789abcdef0123456789',
        },
      }),
    );
  });

  it.each([
    ['missing', null],
    ['weak', 'W/"admin-webhook-41-r2"'],
    ['malformed', '"admin-webhook-41-r0"'],
    ['wrong registration', '"admin-webhook-42-r2"'],
    ['wrong revision', '"admin-webhook-41-r3"'],
  ])('rejects a %s response ETag', async (_label, etag) => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: REGISTRATION,
      status: 200,
      etag,
      requestId: 'request-get',
    });

    await expect(api.getWebhook(41)).rejects.toBeInstanceOf(WebhookContractError);
  });

  it('rejects malformed caller ETags before issuing PATCH, DELETE, or rotate', async () => {
    await expect(
      api.updateWebhook(41, { active: false }, 'W/"admin-webhook-41-r2"'),
    ).rejects.toBeInstanceOf(WebhookContractError);
    await expect(
      api.deleteWebhook(41, '"admin-webhook-42-r2"'),
    ).rejects.toBeInstanceOf(WebhookContractError);
    await expect(
      api.rotateWebhookSecret(41, 'malformed', '0123456789abcdef0123456789abcdef'),
    ).rejects.toBeInstanceOf(WebhookContractError);
    expect(httpMocks.requestJsonWithMetadata).not.toHaveBeenCalled();
  });

  it('validates DELETE status and acknowledgement without requiring an ETag response', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: { deleted: true, id: 41 },
      status: 200,
      etag: null,
      requestId: 'request-delete',
    });

    await expect(api.deleteWebhook(41, '"admin-webhook-41-r2"')).resolves.toEqual({
      deleted: true,
      id: 41,
    });
    expect(httpMocks.requestJsonWithMetadata).toHaveBeenCalledWith(
      '/admin/webhooks/41',
      {
        method: 'DELETE',
        headers: { 'If-Match': '"admin-webhook-41-r2"' },
      },
    );
  });

  it('loads and validates sanitized delivery history', async () => {
    httpMocks.requestJson.mockResolvedValue({
      items: [{ delivery: DELIVERY, attempts: [ATTEMPT] }],
      total: 1,
      limit: 50,
      offset: 0,
    });

    await expect(canonicalWebhookApi.getWebhookDeliveries(41, {
      limit: 50,
      offset: 0,
    })).resolves.toEqual({
      items: [{ delivery: DELIVERY, attempts: [ATTEMPT] }],
      total: 1,
      limit: 50,
      offset: 0,
    });
    expect(httpMocks.requestJson).toHaveBeenCalledWith(
      '/admin/webhooks/41/deliveries?limit=50&offset=0',
    );
  });

  it('submits persisted tests with current configuration and accepts terminal 200 results', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
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

    const result = await canonicalWebhookApi.testWebhook(
      41,
      { delivery_config_version: 1 },
      '"admin-webhook-41-r2"',
      '0123456789abcdef0123456789abcdef',
    );

    expect(result.status).toBe(200);
    expect(result.retryAfterSeconds).toBeNull();
    expect(result.data.in_progress).toBe(false);
    expect(httpMocks.requestJsonWithMetadata).toHaveBeenCalledWith(
      '/admin/webhooks/41/test',
      {
        method: 'POST',
        headers: {
          'If-Match': '"admin-webhook-41-r2"',
          'Idempotency-Key': '0123456789abcdef0123456789abcdef',
        },
        body: JSON.stringify({ delivery_config_version: 1 }),
      },
    );
  });

  it('requires a bounded Retry-After for in-progress 202 test results', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: {
        delivery: { ...DELIVERY, kind: 'test', state: 'processing', status_code: null },
        attempt: { ...ATTEMPT, state: 'processing', finished_at: null, status_code: null },
        idempotent_replay: true,
        in_progress: true,
      },
      status: 202,
      etag: null,
      requestId: 'request-test-processing',
      retryAfterSeconds: 2,
    });

    await expect(canonicalWebhookApi.testWebhook(
      41,
      { delivery_config_version: 1 },
      '"admin-webhook-41-r2"',
      '0123456789abcdef0123456789abcdef',
    )).resolves.toMatchObject({ status: 202, retryAfterSeconds: 2 });

    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: {
        delivery: { ...DELIVERY, kind: 'test', state: 'processing', status_code: null },
        attempt: { ...ATTEMPT, state: 'processing', finished_at: null, status_code: null },
        idempotent_replay: true,
        in_progress: true,
      },
      status: 202,
      etag: null,
      requestId: 'request-test-processing',
      retryAfterSeconds: null,
    });
    await expect(canonicalWebhookApi.testWebhook(
      41,
      { delivery_config_version: 1 },
      '"admin-webhook-41-r2"',
      '0123456789abcdef0123456789abcdef',
    )).rejects.toBeInstanceOf(WebhookContractError);
  });

  it('accepts only canonical 202 manual redelivery acknowledgements', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: {
        delivery: {
          ...DELIVERY,
          id: '44444444-4444-4444-8444-444444444444',
          event_id: '55555555-5555-4555-8555-555555555555',
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
      requestId: 'request-redeliver',
      retryAfterSeconds: null,
    });

    await expect(canonicalWebhookApi.redeliverWebhook(
      41,
      DELIVERY_ID,
      { delivery_config_version: 1, confirm_changed_configuration: false },
      '"admin-webhook-41-r2"',
      'abcdef0123456789abcdef0123456789',
    )).resolves.toMatchObject({ status: 202 });
    expect(httpMocks.requestJsonWithMetadata).toHaveBeenCalledWith(
      `/admin/webhooks/41/deliveries/${DELIVERY_ID}/redeliver`,
      {
        method: 'POST',
        headers: {
          'If-Match': '"admin-webhook-41-r2"',
          'Idempotency-Key': 'abcdef0123456789abcdef0123456789',
        },
        body: JSON.stringify({
          delivery_config_version: 1,
          confirm_changed_configuration: false,
        }),
      },
    );
  });

  it.each([
    { ...DELIVERY, id: 'not-a-uuid' },
    { ...DELIVERY, webhook_id: '41' },
    { ...DELIVERY, state: 'unknown' },
    { ...DELIVERY, target_url: 'https://secret.example/path' },
  ])('fails closed on malformed or unsanitized delivery records', async (delivery) => {
    httpMocks.requestJson.mockResolvedValue({
      items: [{ delivery, attempts: [ATTEMPT] }],
      total: 1,
      limit: 50,
      offset: 0,
    });

    await expect(canonicalWebhookApi.getWebhookDeliveries(41)).rejects.toBeInstanceOf(
      WebhookContractError,
    );
  });
});

describe('canonical-only webhook client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('loads strict canonical status with runtime readiness', async () => {
    httpMocks.requestJson.mockResolvedValue(STATUS);

    await expect(canonicalWebhookApi.getWebhookStatus()).resolves.toEqual(STATUS);
    expect(httpMocks.requestJson).toHaveBeenCalledTimes(1);
    expect(httpMocks.requestJson).toHaveBeenCalledWith('/admin/webhooks/status');
  });

  it.each([
    { ...STATUS, route_selection: 'unknown' },
    { ...STATUS, route_selection: 'legacy' },
    { ...STATUS, delivery: { ...STATUS.delivery, jobs_backend: 'memory' } },
    { ...STATUS, delivery_capability_ready: false },
    { ...STATUS, delivery: { ...STATUS.delivery, worker: { ...STATUS.delivery.worker, component: 'retention' } } },
    { ...STATUS, delivery: { ...STATUS.delivery, backlog: { ...STATUS.delivery.backlog, pending: -1 } } },
    { ...STATUS, migration: { ...STATUS.migration, legacy_file_restore_permitted: 'yes' } },
    { ...STATUS, migration: { ...STATUS.migration, rollback_window_expires_at: '2026-08-22 12:00:00' } },
    { route_selection: 'canonical' },
  ])('rejects malformed successful status without compatibility probing', async (body) => {
    httpMocks.requestJson.mockResolvedValue(body);

    await expect(canonicalWebhookApi.getWebhookStatus()).rejects.toBeInstanceOf(
      WebhookContractError,
    );
    expect(httpMocks.requestJson).toHaveBeenCalledTimes(1);
  });

  it('keeps status failures visible and exports no legacy detector or client', async () => {
    const failure = new Error('status unavailable');
    httpMocks.requestJson.mockRejectedValue(failure);

    await expect(canonicalWebhookApi.getWebhookStatus()).rejects.toBe(failure);
    expect(httpMocks.requestJson).toHaveBeenCalledTimes(1);
    expect(apiClientModule).not.toHaveProperty('detectWebhookApi');
    expect(apiClientModule).not.toHaveProperty('legacyWebhookApi');
  });
});

describe('incident webhook command client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('submits the reviewed narrative as an idempotent durable command', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: {
        incident_id: 'inc-41',
        event_id: EVENT_ID,
        event_type: 'incident.notify',
        command_id: '66666666-6666-4666-8666-666666666666',
        accepted: true,
        replayed: false,
      },
      status: 202,
      etag: null,
      requestId: 'request-incident-notify',
      retryAfterSeconds: null,
    });

    await expect(api.notifyIncidentWebhooks(
      'inc-41',
      { narrative: 'Customer impact is limited to delayed imports.' },
      '0123456789abcdef0123456789abcdef',
    )).resolves.toMatchObject({ accepted: true, replayed: false });
    expect(httpMocks.requestJsonWithMetadata).toHaveBeenCalledWith(
      '/admin/incidents/inc-41/notify-webhooks',
      {
        method: 'POST',
        headers: { 'Idempotency-Key': '0123456789abcdef0123456789abcdef' },
        body: JSON.stringify({
          narrative: 'Customer impact is limited to delayed imports.',
        }),
      },
    );
  });

  it('fails closed on malformed durable command acceptance', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: {
        incident_id: 'inc-41',
        event_id: EVENT_ID,
        event_type: 'incident.notify',
        command_id: 'not-a-uuid',
        accepted: true,
        replayed: false,
      },
      status: 202,
      etag: null,
      requestId: 'request-incident-notify',
      retryAfterSeconds: null,
    });

    await expect(api.notifyIncidentWebhooks(
      'inc-41',
      { narrative: null },
      '0123456789abcdef0123456789abcdef',
    )).rejects.toBeInstanceOf(WebhookContractError);
  });

  it('rejects a valid incident command body returned with the wrong success status', async () => {
    httpMocks.requestJsonWithMetadata.mockResolvedValue({
      data: {
        incident_id: 'inc-41',
        event_id: EVENT_ID,
        event_type: 'incident.notify',
        command_id: '66666666-6666-4666-8666-666666666666',
        accepted: true,
        replayed: false,
      },
      status: 200,
      etag: null,
      requestId: 'request-incident-notify',
      retryAfterSeconds: null,
    });

    await expect(api.notifyIncidentWebhooks(
      'inc-41',
      { narrative: null },
      '0123456789abcdef0123456789abcdef',
    )).rejects.toBeInstanceOf(WebhookContractError);
  });
});
