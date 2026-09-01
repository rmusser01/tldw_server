import { createHmac } from 'node:crypto';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { type AddressInfo } from 'node:net';

import { type Page } from '@playwright/test';

import { postAdminE2EJson } from './helpers/admin-e2e-support';
import { expect, test } from './helpers/fixtures';

const WEBHOOK_HEADERS = [
  'x-tldw-webhook-event',
  'x-tldw-webhook-event-id',
  'x-tldw-webhook-delivery-id',
  'x-tldw-webhook-timestamp',
  'x-tldw-webhook-secret-version',
  'x-tldw-webhook-signature',
] as const;
const UUID4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu;

type CapturedWebhook = {
  body: Buffer;
  headers: Record<string, string>;
  payload: Record<string, unknown>;
};

type WebhookRegistration = {
  id: number;
  description: string;
};

type WebhookDelivery = {
  id: string;
  event_id: string;
  event_type: string;
  kind: 'automatic' | 'manual' | 'test';
  state: string;
};

type DeliveryHistory = {
  items: Array<{ delivery: WebhookDelivery }>;
};

const readBody = async (request: IncomingMessage): Promise<Buffer> => {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
};

const startReceiver = async (): Promise<{
  captures: CapturedWebhook[];
  errors: string[];
  server: Server;
  setSigningSecret: (secret: string) => void;
  url: string;
}> => {
  const captures: CapturedWebhook[] = [];
  const errors: string[] = [];
  const identities = new Set<string>();
  let signingSecret: string | null = null;

  const server = createServer(async (request, response) => {
    try {
      const body = await readBody(request);
      const headers = Object.fromEntries(
        Object.entries(request.headers)
          .filter(([name]) => name.startsWith('x-tldw-webhook-'))
          .map(([name, value]) => [name, Array.isArray(value) ? value.join(',') : String(value)]),
      );
      const payload = JSON.parse(body.toString('utf8')) as Record<string, unknown>;
      const expectedHeaders = [
        ...WEBHOOK_HEADERS,
        ...(headers['x-tldw-webhook-test'] === 'true' ? ['x-tldw-webhook-test'] : []),
      ].sort();

      if (request.method !== 'POST' || request.url !== '/admin-webhooks') {
        throw new Error(`unexpected receiver request ${request.method ?? 'unknown'} ${request.url ?? ''}`);
      }
      if (JSON.stringify(Object.keys(headers).sort()) !== JSON.stringify(expectedHeaders)) {
        throw new Error(`unexpected webhook headers: ${Object.keys(headers).sort().join(',')}`);
      }
      if (!signingSecret) {
        throw new Error('webhook arrived before its one-time signing secret was stored');
      }
      if (!UUID4.test(headers['x-tldw-webhook-event-id'] ?? '')) {
        throw new Error('event id is not UUIDv4');
      }
      if (!UUID4.test(headers['x-tldw-webhook-delivery-id'] ?? '')) {
        throw new Error('delivery id is not UUIDv4');
      }
      if (!/^\d+$/u.test(headers['x-tldw-webhook-timestamp'] ?? '')) {
        throw new Error('timestamp header is not an integer');
      }
      if (!/^[1-9]\d*$/u.test(headers['x-tldw-webhook-secret-version'] ?? '')) {
        throw new Error('secret version header is not positive');
      }
      if (payload.type !== headers['x-tldw-webhook-event']) {
        throw new Error('body event type does not match its header');
      }

      const timestamp = headers['x-tldw-webhook-timestamp'];
      const expectedSignature = `v1=${createHmac('sha256', signingSecret)
        .update(timestamp)
        .update('.')
        .update(body)
        .digest('hex')}`;
      if (headers['x-tldw-webhook-signature'] !== expectedSignature) {
        throw new Error('signature does not authenticate the exact request body');
      }

      const identity = `${headers['x-tldw-webhook-event-id']}:${headers['x-tldw-webhook-delivery-id']}`;
      if (identities.has(identity)) {
        throw new Error(`duplicate receiver identity ${identity}`);
      }
      identities.add(identity);
      captures.push({ body, headers, payload });
      response.writeHead(204);
      response.end();
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
      response.writeHead(500);
      response.end();
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address() as AddressInfo;

  return {
    captures,
    errors,
    server,
    setSigningSecret: (secret) => {
      signingSecret = secret;
    },
    url: `http://127.0.0.1:${address.port}/admin-webhooks`,
  };
};

const closeServer = async (server: Server): Promise<void> => {
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
};

const countReceiverCaptures = (
  receiver: { captures: CapturedWebhook[]; errors: string[] },
  predicate: (capture: CapturedWebhook) => boolean,
): number => {
  if (receiver.errors.length > 0) {
    throw new Error(`Controlled receiver rejected a webhook: ${receiver.errors.join('; ')}`);
  }
  return receiver.captures.filter(predicate).length;
};

const approvePrivilegedAction = async (
  page: Page,
  title: string | RegExp,
  confirmName: string | RegExp,
): Promise<void> => {
  const dialog = page.getByRole('dialog', { name: title });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByLabel(/^reason$/i)).toHaveCount(0);
  await expect(dialog.getByLabel(/current password/i)).toHaveCount(0);
  await dialog.getByRole('button', { name: confirmName, exact: true }).click();
};

const fetchProxyJson = async <T>(page: Page, path: string): Promise<T> => page.evaluate(
  async (requestPath) => {
    const response = await fetch(requestPath, { credentials: 'include' });
    const body = await response.json().catch(() => null) as unknown;
    if (!response.ok) {
      throw new Error(`GET ${requestPath} failed with ${response.status}: ${JSON.stringify(body)}`);
    }
    return body;
  },
  path,
) as Promise<T>;

const postProxyJson = async <T>(page: Page, path: string, body: unknown): Promise<T> => page.evaluate(
  async ({ requestPath, requestBody }) => {
    const response = await fetch(requestPath, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      credentials: 'include',
    });
    const responseBody = await response.json().catch(() => null) as unknown;
    if (!response.ok) {
      throw new Error(`POST ${requestPath} failed with ${response.status}: ${JSON.stringify(responseBody)}`);
    }
    return responseBody;
  },
  { requestPath: path, requestBody: body },
) as Promise<T>;

const removeExistingRegistrations = async (page: Page): Promise<void> => {
  await page.evaluate(async () => {
    const listResponse = await fetch('/api/proxy/admin/webhooks?limit=100&offset=0', {
      credentials: 'include',
    });
    const list = await listResponse.json() as { items?: Array<{ id?: unknown }> };
    if (!listResponse.ok || !Array.isArray(list.items)) {
      throw new Error(`Could not list existing webhooks (${listResponse.status})`);
    }

    for (const item of list.items) {
      if (!Number.isInteger(item.id)) {
        throw new Error('Webhook list returned an invalid registration id');
      }
      const detailResponse = await fetch(`/api/proxy/admin/webhooks/${item.id}`, {
        credentials: 'include',
      });
      const etag = detailResponse.headers.get('etag');
      if (!detailResponse.ok || !etag) {
        throw new Error(`Could not load webhook ${item.id} for cleanup`);
      }
      const deleteResponse = await fetch(`/api/proxy/admin/webhooks/${item.id}`, {
        method: 'DELETE',
        headers: { 'If-Match': etag },
        credentials: 'include',
      });
      if (!deleteResponse.ok) {
        throw new Error(`Could not delete webhook ${item.id} during cleanup (${deleteResponse.status})`);
      }
    }
  });
};

const storeOneTimeSecret = async (page: Page): Promise<string> => {
  const dialog = page.getByRole('dialog', { name: /^signing secret$/i });
  await expect(dialog).toBeVisible();
  const secret = await dialog.getByRole('textbox', { name: /^signing secret$/i }).inputValue();
  expect(secret).toMatch(/^whsec_[A-Za-z0-9_-]{32,}$/u);
  await dialog.getByRole('button', { name: /copy signing secret/i }).click();
  await dialog.getByRole('checkbox', { name: /I have stored this signing secret/i }).click();
  await dialog.getByRole('button', { name: /^done$/i }).click();
  await expect(dialog).toHaveCount(0);
  await expect(page.getByText(secret, { exact: true })).toHaveCount(0);
  return secret;
};

test('proves the canonical webhook lifecycle against the real backend and receiver', async ({
  context,
  page,
  projectEnv,
  seededSession,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== 'chromium-real-jwt',
    'Webhook delivery activation runs only in the multi-user JWT project',
  );
  test.setTimeout(180_000);

  const receiver = await startReceiver();
  const browserMessages: string[] = [];
  const browserRequests: string[] = [];
  page.on('console', (message) => browserMessages.push(message.text()));
  page.on('pageerror', (error) => browserMessages.push(error.message));
  page.on('request', (request) => {
    browserRequests.push(`${request.url()}\n${request.postData() ?? ''}`);
  });

  try {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await postAdminE2EJson(
      projectEnv.apiBaseUrl,
      '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
    );
    await seededSession.as('admin', 'jwt_admin');

    await expect.poll(async () => page.evaluate(async () => {
      const response = await fetch('/api/proxy/admin/webhooks/status', { credentials: 'include' });
      const body = await response.json().catch(() => null) as { delivery_capability_ready?: boolean } | null;
      return response.ok && body?.delivery_capability_ready === true;
    }), { timeout: 30_000 }).toBe(true);

    await removeExistingRegistrations(page);
    await page.goto('/webhooks');
    await expect(page.getByRole('heading', { name: /^webhooks$/i })).toBeVisible();
    const runtime = page.getByLabel('Webhook delivery runtime');
    for (const readyState of [
      'Signing key ready',
      'Worker ready',
      'Reconciler ready',
      'Retention ready',
      'Acquisition ready',
    ]) {
      await expect(runtime).toContainText(readyState);
    }

    const suffix = `${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
    const description = `Private beta receiver ${suffix}`;
    await page.getByRole('button', { name: /add webhook/i }).click();
    const createDialog = page.getByRole('dialog', { name: /^add webhook$/i });
    await createDialog.getByLabel(/^destination URL$/i).fill(receiver.url);
    await createDialog.getByLabel(/^description$/i).fill(description);
    await createDialog.getByLabel(/timeout/i).fill('5');
    await createDialog.getByRole('checkbox', { name: /^user\.created\b/i }).click();
    await createDialog.getByRole('checkbox', { name: /^incident\.created\b/i }).click();
    await createDialog.getByRole('button', { name: /^create$/i }).click();
    const initialSecret = await storeOneTimeSecret(page);
    receiver.setSigningSecret(initialSecret);

    const registrationRow = page.getByRole('row').filter({ hasText: description });
    await expect(registrationRow).toContainText('Inactive');
    await expect(page.getByText(receiver.url, { exact: true })).toHaveCount(0);
    await registrationRow.getByRole('button', { name: /^enable$/i }).click();
    await approvePrivilegedAction(page, /^enable webhook$/i, /^enable webhook$/i);
    await expect(registrationRow).toContainText('Active');

    await postProxyJson(page, '/api/proxy/admin/users', {
      username: `webhook_${suffix.replaceAll('-', '_')}`,
      email: `webhook-${suffix}@example.com`,
      password: 'Admin@Pass#2024!',
      role: 'user',
      is_active: true,
      is_verified: true,
    });
    await postProxyJson(page, '/api/proxy/admin/incidents', {
      title: `Private beta webhook incident ${suffix}`,
      status: 'open',
      severity: 'high',
      summary: 'Real-backend webhook delivery acceptance proof',
      tags: ['e2e'],
    });

    await expect.poll(
      () => countReceiverCaptures(receiver, (capture) => (
        ['user.created', 'incident.created'].includes(capture.headers['x-tldw-webhook-event'] ?? '')
      )),
      { timeout: 30_000 },
    ).toBe(2);
    expect(receiver.errors).toEqual([]);

    const registrations = await fetchProxyJson<{ items: WebhookRegistration[] }>(
      page,
      '/api/proxy/admin/webhooks?limit=100&offset=0',
    );
    const registration = registrations.items.find((item) => item.description === description);
    expect(registration).toBeDefined();
    if (!registration) throw new Error('Created webhook registration was not returned');

    await registrationRow.getByRole('button', { name: /show delivery history/i }).click();
    const historyRegion = page.getByRole('region', { name: /delivery history/i });
    await expect(historyRegion.getByText('user.created', { exact: true })).toBeVisible();
    await expect(historyRegion.getByText('incident.created', { exact: true })).toBeVisible();
    await expect(historyRegion.getByText(/attempt 1: succeeded/i).first()).toBeVisible();

    const automaticHistory = await fetchProxyJson<DeliveryHistory>(
      page,
      `/api/proxy/admin/webhooks/${registration.id}/deliveries?limit=100&offset=0`,
    );
    const automaticIncident = automaticHistory.items
      .map((item) => item.delivery)
      .find((delivery) => delivery.kind === 'automatic' && delivery.event_type === 'incident.created');
    expect(automaticIncident?.state).toBe('succeeded');
    if (!automaticIncident) throw new Error('Automatic incident delivery was not persisted');

    await registrationRow.getByRole('button', { name: /^run test$/i }).click();
    await approvePrivilegedAction(page, /^run webhook test$/i, /^run test$/i);
    const retryTest = page.getByRole('button', { name: /retry same test/i });
    for (let retry = 0; retry < 5; retry += 1) {
      if (await page.getByText(/test delivery succeeded/i).isVisible().catch(() => false)) break;
      if (await retryTest.isVisible().catch(() => false)) await retryTest.click();
      await page.waitForTimeout(1_000);
    }
    await expect(page.getByText(/test delivery succeeded/i)).toBeVisible();
    await expect.poll(
      () => countReceiverCaptures(
        receiver,
        (capture) => capture.headers['x-tldw-webhook-test'] === 'true',
      ),
      { timeout: 30_000 },
    ).toBe(1);
    const testCapture = receiver.captures.find(
      (capture) => capture.headers['x-tldw-webhook-test'] === 'true',
    );
    expect(testCapture?.headers['x-tldw-webhook-event']).toBe('webhook.test');

    await historyRegion.getByRole('button', { name: /redeliver incident\.created/i }).click();
    await approvePrivilegedAction(page, /^redeliver webhook event$/i, /^redeliver event$/i);
    await expect(page.getByText(/manual redelivery accepted/i)).toBeVisible();
    await expect.poll(
      () => countReceiverCaptures(receiver, (capture) => (
        capture.headers['x-tldw-webhook-event-id'] === automaticIncident.event_id
      )),
      { timeout: 30_000 },
    ).toBe(2);
    const incidentCaptures = receiver.captures.filter((capture) => (
      capture.headers['x-tldw-webhook-event-id'] === automaticIncident.event_id
    ));
    expect(incidentCaptures.map((capture) => capture.headers['x-tldw-webhook-delivery-id']))
      .toEqual(expect.arrayContaining([automaticIncident.id]));
    expect(new Set(
      incidentCaptures.map((capture) => capture.headers['x-tldw-webhook-delivery-id']),
    ).size).toBe(2);
    expect(receiver.errors).toEqual([]);

    await registrationRow.getByRole('button', { name: /^disable$/i }).click();
    await approvePrivilegedAction(page, /^disable webhook$/i, /^disable webhook$/i);
    await expect(registrationRow).toContainText('Inactive');
    await registrationRow.getByRole('button', { name: /generate a new secret/i }).click();
    await approvePrivilegedAction(
      page,
      /^generate a new signing secret$/i,
      /^generate secret$/i,
    );
    const rotatedSecret = await storeOneTimeSecret(page);
    expect(rotatedSecret).not.toBe(initialSecret);
    receiver.setSigningSecret(rotatedSecret);

    const persistedBrowserState = await page.evaluate(() => ({
      href: window.location.href,
      localStorage: JSON.stringify({ ...localStorage }),
      sessionStorage: JSON.stringify({ ...sessionStorage }),
      text: document.body.textContent ?? '',
    }));
    for (const secret of [initialSecret, rotatedSecret]) {
      expect(JSON.stringify(persistedBrowserState)).not.toContain(secret);
      expect(browserMessages.join('\n')).not.toContain(secret);
      expect(browserRequests.join('\n')).not.toContain(secret);
    }
    expect(browserMessages.join('\n')).not.toContain(receiver.url);
    expect(receiver.errors).toEqual([]);
  } finally {
    await closeServer(receiver.server);
  }
});
