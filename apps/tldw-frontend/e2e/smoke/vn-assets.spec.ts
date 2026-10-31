import type { Route } from '@playwright/test';
import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from './smoke.setup';
import { waitForAppShell, waitForVisualSettle } from '../utils/helpers';

const fulfillJson = async (route: Route, status: number, data: unknown) => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(data),
  });
};

test.describe('VN asset packs smoke', () => {
  test('creates a pack, applies a matrix, and reviews mocked variants', async ({ page }) => {
    await seedAuth(page);
    await page.addInitScript(() => {
      localStorage.setItem('assistant_setup_dismissed', 'true');
    });

    let packs: Array<Record<string, unknown>> = [];
    let slots: Array<Record<string, unknown>> = [];
    let items: Array<Record<string, unknown>> = [];
    let lastExportBody: Record<string, unknown> | null = null;
    let lastMatrixBody: Record<string, unknown> | null = null;
    let lastReviewBody: Record<string, unknown> | null = null;

    await page.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
      await fulfillJson(route, 200, {
        status: 'ok',
        auth_mode: 'single_user',
        test_api_key: 'THIS-IS-A-SECURE-KEY-123-LOCAL-TEST',
      });
    });

    await page.route(/\/api\/v1\/persona\/profiles(?:\?.*)?$/, async (route) => {
      await fulfillJson(route, 200, [{ id: 'smoke-profile', name: 'Smoke profile' }]);
    });

    await page.route(/\/api\/v1\/vn\/vn-assets(?:\/.*)?$/, async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const method = request.method().toUpperCase();
      const path = url.pathname.replace('/api/v1/vn/vn-assets', '');

      if (method === 'GET' && path === '/starter-matrices') {
        await fulfillJson(route, 200, {
          matrices: [
            {
              key: 'starter',
              title: 'Starter',
              slot_count: 8,
              planned_output_count: 24,
              asset_types: ['background', 'sprite', 'cg'],
            },
          ],
        });
        return;
      }

      if (method === 'GET' && path === '/packs') {
        await fulfillJson(route, 200, packs);
        return;
      }

      if (method === 'POST' && path === '/packs') {
        const body = request.postDataJSON() as Record<string, unknown>;
        const created = {
          id: 1,
          title: body.title,
          primary_character_id: body.primary_character_id,
          planned_output_count: 0,
          status: 'draft',
        };
        packs = [created];
        items = [
          {
            id: 11,
            pack_id: 1,
            slot_id: 4,
            variant_index: 0,
            generated_file_id: 101,
            mime_type: 'image/png',
            width: 512,
            height: 768,
            review_status: 'draft',
            preferred: false,
            source: 'generated',
          },
        ];
        await fulfillJson(route, 201, created);
        return;
      }

      if (method === 'GET' && path === '/packs/1/slots') {
        await fulfillJson(route, 200, slots);
        return;
      }

      if (method === 'GET' && path === '/packs/1/items') {
        await fulfillJson(route, 200, items);
        return;
      }

      if (method === 'GET' && path === '/packs/1/generation') {
        await fulfillJson(route, 200, {
          status: 'idle',
          planned_count: slots.length,
          completed_count: 0,
          failed_count: 0,
        });
        return;
      }

      if (method === 'GET' && path === '/packs/1/generation/preflight') {
        await fulfillJson(route, 200, {
          scope: 'api_process_configuration', worker_health: 'unknown',
          local_workers_enabled: true, warnings: [], slots: [],
        });
        return;
      }

      if (method === 'GET' && path === '/packs/1/readiness') {
        await fulfillJson(route, 200, {
          ready: false,
          status: 'not_ready',
          warnings: [],
          errors: [],
        });
        return;
      }

      if (method === 'POST' && path === '/packs/1/matrix/apply') {
        lastMatrixBody = request.postDataJSON() as Record<string, unknown>;
        slots = [
          {
            id: 4,
            pack_id: 1,
            asset_type: 'sprite',
            slot_key: 'sprite_neutral',
            labels: { expression: 'neutral' },
            variant_count: 1,
            width: 512,
            height: 768,
            status: 'reviewing',
          },
        ];
        await fulfillJson(route, 200, slots);
        return;
      }

      if (method === 'POST' && path === '/packs/1/export') {
        lastExportBody = request.postDataJSON() as Record<string, unknown>;
        await fulfillJson(route, 202, {
          job_id: '700',
          portability_job_id: 8,
          operation: 'export',
          pack_id: 1,
          status: 'queued',
          stage: 'queued',
          download_url: null,
        });
        return;
      }

      if (method === 'POST' && path === '/packs/1/items/bulk-review') {
        lastReviewBody = request.postDataJSON() as Record<string, unknown>;
        items = items.map((item) =>
          item.id === 11 ? { ...item, review_status: 'approved' } : item
        );
        await fulfillJson(route, 200, items);
        return;
      }

      await fulfillJson(route, 404, { detail: `unhandled vn-assets mock route: ${method} ${path}` });
    });

    await page.goto('/vn-assets');
    await waitForAppShell(page, SMOKE_LOAD_TIMEOUT);

    await expect(page.getByRole('heading', { name: 'VN asset packs' })).toBeVisible();
    await expect(page.getByText('No asset packs yet.')).toBeVisible();

    await page.getByLabel('Pack title').fill('Orbital Library');
    await page.getByLabel('Primary character ID').fill('42');
    await page.getByRole('button', { name: 'Create pack' }).click();

    await expect(page.getByText('Selected pack: Orbital Library')).toBeVisible();
    await expect(page.getByText('Character 42').first()).toBeVisible();

    await page.getByRole('button', { name: 'Apply starter matrix' }).click();
    await expect.poll(() => lastMatrixBody).toMatchObject({
      matrix_key: 'starter',
      overrides: { variant_count: 1 },
    });

    await expect(page.getByLabel('Select item 11')).toBeVisible();
    await page.getByLabel('Select item 11').check();
    await page.getByRole('button', { name: 'Approve selected' }).click();

    await expect.poll(() => lastReviewBody).toMatchObject({
      item_ids: [11],
      review_status: 'approved',
    });
    await expect(page.getByText('approved').first()).toBeVisible();

    await page.getByLabel('Include character payload').check();
    await page.getByLabel('Include full provenance').check();
    await page.getByRole('button', { name: 'Export backup bundle' }).click();

    await expect.poll(() => lastExportBody).toMatchObject({
      include_character_payload: true,
      include_full_provenance: true,
      include_world_book_payloads: false,
      strict: false,
      warn_for_sharing: true,
    });
    await expect(page.getByText('Export job: 700')).toBeVisible();
  });
});

for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
  test(`VN generation retries a failed slot and refreshes progress at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await seedAuth(page);
    const requests: Array<Record<string, unknown>> = [];
    let retried = false;
    const slot = {
      id: 12, pack_id: 7, slot_key: 'sprite_neutral', asset_type: 'sprite',
      variant_count: 1, status: 'failed', last_error: 'image_backend_unavailable',
    };
    await page.route(/\/api\/v1\/vn\/vn-assets(?:\/.*)?$/, async (route) => {
      const path = new URL(route.request().url()).pathname.replace('/api/v1/vn/vn-assets', '');
      if (path === '/starter-matrices') return fulfillJson(route, 200, { matrices: [] });
      if (path === '/packs') return fulfillJson(route, 200, [
        { id: 7, title: 'Recovery pack', primary_character_id: 42, status: 'draft' },
      ]);
      if (path === '/packs/7/slots') return fulfillJson(route, 200, [
        retried ? { ...slot, status: 'reviewing', last_error: null } : slot,
      ]);
      if (path === '/packs/7/items') return fulfillJson(route, 200, []);
      if (path === '/packs/7/readiness') return fulfillJson(route, 200, {
        ready: false, status: 'not_ready', warnings: [], errors: [],
      });
      if (path === '/packs/7/generation/preflight') return fulfillJson(route, 200, {
        scope: 'api_process_configuration', worker_health: 'unknown',
        local_workers_enabled: false,
        warnings: ['Local generation workers are not both enabled. Enable them or confirm that separate workers are running.'],
        slots: [{ slot_id: 12, backend: null, model: null, status: 'unavailable',
          message: 'Enable the selected image backend before retrying.' }],
      });
      if (path === '/packs/7/generation') return fulfillJson(route, 200, {
        status: retried ? 'completed' : 'failed',
        failed_count: retried ? 0 : 1, completed_count: retried ? 1 : 0,
      });
      if (path === '/packs/7/slots/12/retry') {
        requests.push(route.request().postDataJSON());
        if (requests.length === 1) return route.abort('failed');
        retried = true;
        return fulfillJson(route, 202, { status: 'queued', batch_id: 2 });
      }
      return fulfillJson(route, 404, { detail: 'Unexpected VN test request' });
    });
    await page.goto('/vn-assets');
    await waitForAppShell(page, SMOKE_LOAD_TIMEOUT);
    const retry = page.getByRole('button', { name: 'Retry sprite_neutral' });
    await expect(retry).toBeEnabled();
    await waitForVisualSettle(page, SMOKE_LOAD_TIMEOUT);
    await page.screenshot({ path: testInfo.outputPath('generation-failure.png'), fullPage: true });
    await retry.click();
    await expect.poll(() => requests.length).toBe(1);
    await expect(retry).toBeEnabled();
    await retry.click();
    await expect.poll(() => requests.length).toBe(2);
    expect(requests[0].idempotency_key).toEqual(expect.any(String));
    expect(requests[1]).toEqual(requests[0]);
    await expect(page.getByRole('status', { name: 'Generation status' })).toHaveText('completed');
    await expect(retry).toHaveCount(0);
    const overflowingElements = await page.evaluate(() => Array.from(document.querySelectorAll('main *'))
      .filter((element) => element.getBoundingClientRect().right > window.innerWidth)
      .map((element) => ({ tag: element.tagName, className: element.className,
        width: element.getBoundingClientRect().width, text: element.textContent?.slice(0, 60) })));
    expect(overflowingElements).toEqual([]);
    await waitForVisualSettle(page, SMOKE_LOAD_TIMEOUT);
    await page.screenshot({ path: testInfo.outputPath('generation-recovered.png'), fullPage: true });
  });
}
