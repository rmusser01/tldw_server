import { expect, test, type Route } from '@playwright/test';

import { installAdminApiRoutes, setAuthenticatedSession } from './smoke-helpers';

const INCIDENT = {
  id: 'inc_e2e_webhook',
  version: 7,
  title: 'Private beta gateway timeout',
  status: 'investigating',
  severity: 'high',
  summary: 'Internal summary not delivered to webhook receivers',
  tags: ['gateway'],
  created_at: '2026-08-31T20:00:00Z',
  updated_at: '2026-08-31T20:05:00Z',
  resolved_at: null,
  acknowledged_at: '2026-08-31T20:01:00Z',
  created_by: 'admin',
  updated_by: 'admin',
  timeline: [],
  assigned_to_user_id: null,
  assigned_to_label: null,
  root_cause: null,
  impact: null,
  runbook_url: null,
  action_items: [],
  time_to_acknowledge_seconds: 60,
  time_to_resolve_seconds: null,
};

const fulfillJson = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers: { 'x-request-id': 'request-incident-webhook-e2e' },
    body: JSON.stringify(payload),
  });
};

test('previews and queues the exact public incident webhook command', async ({ page }) => {
  await installAdminApiRoutes(page);
  await setAuthenticatedSession(page);

  let commandBody: unknown = null;
  let idempotencyKey: string | null = null;

  await page.route('**/api/proxy/admin/users**', async (route) => {
    await fulfillJson(route, { items: [], total: 0, limit: 100, offset: 0 });
  });
  await page.route('**/api/proxy/admin/incidents**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace('/api/proxy', '');

    if (request.method() === 'GET' && path === '/admin/incidents/metrics/sla') {
      await fulfillJson(route, {
        total_incidents: 1,
        resolved_count: 0,
        acknowledged_count: 1,
        avg_mtta_minutes: 1,
        avg_mttr_minutes: null,
        p95_mtta_minutes: 1,
        p95_mttr_minutes: null,
      });
      return;
    }
    if (request.method() === 'GET' && path === '/admin/incidents') {
      await fulfillJson(route, {
        items: [INCIDENT],
        total: 1,
        limit: 20,
        offset: 0,
      });
      return;
    }
    if (
      request.method() === 'POST'
      && path === `/admin/incidents/${INCIDENT.id}/notify-webhooks`
    ) {
      commandBody = request.postDataJSON();
      idempotencyKey = (await request.allHeaders())['idempotency-key'] ?? null;
      await fulfillJson(route, {
        incident_id: INCIDENT.id,
        event_id: '22222222-2222-4222-8222-222222222222',
        event_type: 'incident.notify',
        command_id: `sha256:${'3'.repeat(64)}`,
        accepted: true,
        replayed: false,
      }, 202);
      return;
    }
    await fulfillJson(route, { detail: `Unhandled incident route: ${request.method()} ${path}` }, 500);
  });

  await page.goto('/incidents');
  await expect(page.getByRole('heading', { name: new RegExp(INCIDENT.title) })).toBeVisible();
  await page.getByTestId(`incident-webhook-notify-${INCIDENT.id}`).click();

  const dialog = page.getByRole('dialog', { name: /notify webhook receivers/i });
  const preview = dialog.getByRole('region', { name: /outbound event preview/i });
  await expect(preview).toContainText('incident.notify');
  await expect(preview).toContainText(INCIDENT.id);
  await expect(preview).toContainText('resource_version');
  await expect(preview).toContainText('7');
  await expect(preview).toContainText(INCIDENT.updated_at);
  await expect(preview).not.toContainText(INCIDENT.title);
  await expect(preview).not.toContainText(INCIDENT.summary);

  const narrative = dialog.getByLabel(/receiver narrative/i);
  await narrative.fill('  Customer imports are delayed; no data loss is known.  ');
  await expect(preview).toContainText('"Customer imports are delayed; no data loss is known."');
  const send = dialog.getByRole('button', { name: /send webhook notification/i });
  await expect(send).toBeDisabled();
  await dialog.getByLabel(/i reviewed this narrative/i).check();
  await send.click();

  await expect(dialog.getByText(/command accepted for durable delivery/i)).toBeVisible();
  expect(commandBody).toEqual({
    narrative: 'Customer imports are delayed; no data loss is known.',
    expected_resource_version: 7,
  });
  expect(idempotencyKey).toMatch(/^[0-9a-f]{32}$/);

  const persistence = await page.evaluate(() => ({
    localStorage: JSON.stringify({ ...localStorage }),
    sessionStorage: JSON.stringify({ ...sessionStorage }),
    href: window.location.href,
  }));
  expect(JSON.stringify(persistence)).not.toContain(idempotencyKey ?? 'missing-key');
});
