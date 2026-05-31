import type { Route } from '@playwright/test';
import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from './smoke.setup';
import { waitForAppShell } from '../utils/helpers';

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
