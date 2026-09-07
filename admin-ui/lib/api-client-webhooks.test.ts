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

import {
  api,
  canonicalWebhookApi,
  detectWebhookApi,
  legacyWebhookApi,
} from './api-client';
import { WebhookContractError } from './http';
import type {
  WebhookCreateRequest,
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

  it('keeps test, delivery, and redelivery operations out of the canonical client', () => {
    expect(canonicalWebhookApi).not.toHaveProperty('testWebhook');
    expect(canonicalWebhookApi).not.toHaveProperty('getWebhookDeliveries');
    expect(canonicalWebhookApi).not.toHaveProperty('redeliverWebhook');
  });
});

describe('webhook API detection and compatibility isolation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('accepts the delivery diagnostics extension without exposing it as page state', async () => {
    const delivery = {
      canonical_schema_version: 1,
      schema_ready: true,
      delivery_schema_ready: true,
      migration_complete: true,
      key_ready: true,
      key_primary_match: true,
      jobs_database_ready: true,
      queue_ready: true,
      job_type_ready: true,
      jobs_backend: 'sqlite',
      worker: { component: 'worker', ready: false, reason_code: 'mode_migrate', heartbeat_age_seconds: null },
      reconciler: { component: 'reconciler', ready: false, reason_code: 'mode_migrate', heartbeat_age_seconds: null },
      retention: { component: 'retention', ready: false, reason_code: 'mode_migrate', heartbeat_age_seconds: null },
      backlog: { pending: 0, enqueue_claimed: 0, queued: 0, processing: 0, retry_wait: 0 },
      oldest_nonterminal_age_seconds: null,
      acquisition_ready: false,
      acquisition_reason_code: 'mode_migrate',
      delivery_capability_ready: false,
    };
    httpMocks.requestJson.mockResolvedValue({ ...STATUS, delivery });

    await expect(detectWebhookApi()).resolves.toEqual({
      kind: 'canonical', status: STATUS, client: canonicalWebhookApi,
    });
  });

  it.each([
    ['canonical', canonicalWebhookApi],
    ['legacy', legacyWebhookApi],
  ] as const)('selects only the status-declared %s client', async (routeSelection, client) => {
    httpMocks.requestJson.mockResolvedValue({
      ...STATUS,
      route_selection: routeSelection,
    });

    await expect(detectWebhookApi()).resolves.toEqual({
      kind: routeSelection,
      status: { ...STATUS, route_selection: routeSelection },
      client,
    });
    expect(httpMocks.requestJson).toHaveBeenCalledTimes(1);
    expect(httpMocks.requestJson).toHaveBeenCalledWith('/admin/webhooks/status');
  });

  it.each([
    { ...STATUS, delivery: null },
    { ...STATUS, delivery: [] },
    { ...STATUS, delivery: 'unavailable' },
    { ...STATUS, delivery: {}, delivery_capability_ready: 'yes' },
    { ...STATUS, delivery: {}, unexpected: true },
    { ...STATUS, route_selection: 'unknown' },
    { ...STATUS, migration: { ...STATUS.migration, legacy_file_restore_permitted: 'yes' } },
    { ...STATUS, migration: { ...STATUS.migration, rollback_window_expires_at: '2026-08-22 12:00:00' } },
    { route_selection: 'legacy' },
  ])('rejects a malformed successful status response without probing legacy CRUD', async (body) => {
    httpMocks.requestJson.mockResolvedValue(body);

    await expect(detectWebhookApi()).rejects.toBeInstanceOf(WebhookContractError);
    expect(httpMocks.requestJson).toHaveBeenCalledTimes(1);
  });

  it('keeps status failures visible instead of downgrading to legacy', async () => {
    const failure = new Error('status unavailable');
    httpMocks.requestJson.mockRejectedValue(failure);

    await expect(detectWebhookApi()).rejects.toBe(failure);
    expect(httpMocks.requestJson).toHaveBeenCalledTimes(1);
  });

  it('adapts legacy DTOs to a separate string-ID view with no ETag or rotate support', async () => {
    httpMocks.requestJson.mockResolvedValue({
      items: [{
        id: 'legacy-1',
        url: 'https://legacy.example/hook',
        events: ['incident.created'],
        enabled: true,
        created_at: null,
        updated_at: null,
      }],
      total: 1,
      pagination: {
        mode: 'offset',
        limit: 50,
        offset: 0,
        total: 1,
        has_more: false,
        next_offset: null,
      },
    });

    const page = await legacyWebhookApi.getWebhooks();

    expect(page.items).toEqual([{
      id: 'legacy-1',
      targetUrl: 'https://legacy.example/hook',
      eventTypes: ['incident.created'],
      enabled: true,
      createdAt: null,
      updatedAt: null,
    }]);
    expect(page.items[0]).not.toHaveProperty('etag');
    expect(page.items[0]).not.toHaveProperty('revision');
    expect(legacyWebhookApi).not.toHaveProperty('rotateWebhookSecret');
  });

  it('validates and adapts legacy delivery DTOs without exposing backend field names', async () => {
    const delivery = {
      id: 'delivery-1',
      webhook_id: 'legacy-1',
      event_type: 'incident.created',
      status_code: 204,
      response_time_ms: 42,
      success: true,
      error: null,
      attempted_at: '2026-08-22T12:00:00Z',
      payload_preview: null,
    };
    httpMocks.requestJson
      .mockResolvedValueOnce(delivery)
      .mockResolvedValueOnce({
        items: [delivery],
        total: 1,
        limit: 50,
        offset: 0,
        pagination: {
          mode: 'offset',
          limit: 50,
          offset: 0,
          total: 1,
          has_more: false,
          next_offset: null,
        },
      });

    const tested = await legacyWebhookApi.testWebhook('legacy-1');
    const history = await legacyWebhookApi.getWebhookDeliveries('legacy-1', { limit: 50 });

    expect(tested).toEqual({
      id: 'delivery-1',
      webhookId: 'legacy-1',
      eventType: 'incident.created',
      statusCode: 204,
      responseTimeMs: 42,
      success: true,
      error: null,
      attemptedAt: '2026-08-22T12:00:00Z',
      payloadPreview: null,
    });
    expect(history).toEqual({ items: [tested], total: 1 });
    expect(tested).not.toHaveProperty('webhook_id');
    expect(tested).not.toHaveProperty('payload_preview');
  });

  it('fails closed on malformed legacy delivery records', async () => {
    httpMocks.requestJson.mockResolvedValue({
      id: 'delivery-1',
      webhook_id: 'legacy-1',
      event_type: 'incident.created',
      status_code: 204,
      response_time_ms: 42,
      success: 'yes',
      error: null,
      attempted_at: null,
      payload_preview: null,
    });

    await expect(legacyWebhookApi.testWebhook('legacy-1')).rejects.toBeInstanceOf(
      WebhookContractError,
    );
  });
});
