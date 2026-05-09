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

test.describe('VN play smoke', () => {
  test('creates a story session and submits a mocked choice', async ({ page }) => {
    await seedAuth(page);
    await page.addInitScript(() => {
      localStorage.setItem('assistant_setup_dismissed', 'true');
    });

    const events: Array<Record<string, unknown>> = [];
    const branches: Array<Record<string, unknown>> = [
      {
        id: 1,
        session_id: 1,
        owner_user_id: 42,
        branch_label: 'Smoke path',
        status: 'active',
      },
    ];
    const checkpoints: Array<Record<string, unknown>> = [];
    let session: Record<string, unknown> | null = null;

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

    await page.route(/\/api\/v1\/vn-play(?:\/.*)?$/, async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const method = request.method().toUpperCase();
      const path = url.pathname.replace('/api/v1/vn-play', '');

      if (method === 'GET' && path === '/sessions') {
        await fulfillJson(route, 200, session ? [session] : []);
        return;
      }

      if (method === 'POST' && path === '/sessions') {
        const body = request.postDataJSON() as Record<string, unknown>;
        session = {
          id: 1,
          owner_user_id: 42,
          mode: body.mode,
          title: body.title,
          status: 'active',
          primary_character_id: body.primary_character_id,
          vn_asset_pack_id: body.vn_asset_pack_id,
          scene_version: 0,
          scene_state: {
            scene_version: 0,
            visible_choices: [{ id: 'c1', text: 'Open the door' }],
          },
        };
        await fulfillJson(route, 201, session);
        return;
      }

      if (method === 'GET' && path === '/sessions/1') {
        await fulfillJson(route, 200, session);
        return;
      }

      if (method === 'GET' && path === '/sessions/1/events') {
        await fulfillJson(route, 200, events);
        return;
      }

      if (method === 'GET' && path === '/sessions/1/branches') {
        await fulfillJson(route, 200, branches);
        return;
      }

      if (method === 'GET' && path === '/sessions/1/checkpoints') {
        await fulfillJson(route, 200, checkpoints);
        return;
      }

      if (method === 'POST' && path === '/sessions/1/checkpoint') {
        const body = request.postDataJSON() as Record<string, unknown>;
        const checkpoint = {
          id: checkpoints.length + 1,
          session_id: 1,
          owner_user_id: 42,
          label: body.label,
          scene_version: body.scene_version ?? (session?.scene_version as number | undefined) ?? 0,
          scene_state_snapshot: session?.scene_state ?? { scene_version: 0 },
        };
        checkpoints.push(checkpoint);
        await fulfillJson(route, 201, checkpoint);
        return;
      }

      if (method === 'POST' && path === '/sessions/1/turn') {
        const modelTurn = {
          id: 2,
          session_id: 1,
          owner_user_id: 42,
          sequence_number: 2,
          event_type: 'model_turn',
          event_payload: {
            dialogue: [{ speaker: 'Mira', text: 'The door opens onto the archive.' }],
          },
          source: 'model',
        };
        events.push(modelTurn);
        session = {
          ...(session ?? {}),
          scene_version: 1,
          scene_state: {
            scene_version: 1,
            location_key: 'archive',
            visible_choices: [{ id: 'c2', text: 'Step inside' }],
          },
        };
        await fulfillJson(route, 200, {
          turn_request_id: 4,
          status: 'completed',
          scene_version: 1,
          scene_state: session.scene_state,
          events: [modelTurn],
        });
        return;
      }

      await fulfillJson(route, 404, { detail: `unhandled vn-play mock route: ${method} ${path}` });
    });

    await page.goto('/vn-play');
    await waitForAppShell(page, SMOKE_LOAD_TIMEOUT);

    await expect(page.getByRole('heading', { name: 'VN play' })).toBeVisible();
    await page.getByRole('button', { name: 'New Story' }).click();
    await page.getByLabel('Title').fill('Smoke Story');
    await page.getByLabel('Primary character ID').fill('7');
    await page.getByLabel('VN asset pack ID').fill('12');
    await page.getByRole('button', { name: 'Create session' }).click();

    await expect(page.getByText('Selected session: Smoke Story')).toBeVisible();
    await expect(page.getByText('Smoke path')).toBeVisible();
    await page.getByLabel('Checkpoint label').fill('Before door');
    await page.getByRole('button', { name: 'Create checkpoint' }).click();
    await expect(page.getByRole('button', { name: 'Restore checkpoint Before door' })).toBeVisible();

    await page.getByRole('button', { name: 'Open the door' }).click();

    await expect(page.getByText('The door opens onto the archive.')).toBeVisible();
    await expect(page.getByText('Scene version')).toBeVisible();
    await expect(page.getByText('1').first()).toBeVisible();
  });
});
