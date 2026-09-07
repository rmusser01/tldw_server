import type { APIRequestContext, Page, Route } from '@playwright/test';
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from '../../utils/fixtures';
import { expectApiCall } from '../../utils/api-assertions';
import { TEST_CONFIG, waitForConnection } from '../../utils/helpers';
import { NotificationsPage } from '../../utils/page-objects';

type WatchlistJourneyState = {
  token: string;
  watchlistId?: number;
  sourceId?: number;
  jobId?: number;
  initialRunId?: number;
  initialItemTitle: string;
  uiRunItemTitle: string;
  jobName: string;
};

type MockNotification = {
  id: number;
  kind: string;
  title: string;
  message: string;
  severity: 'info';
  created_at: string;
  read_at: string | null;
  dismissed_at: string | null;
};

const apiUrl = (path: string): string => `${TEST_CONFIG.serverUrl}${path}`;

const apiHeaders = (): Record<string, string> => ({
  'X-API-Key': TEST_CONFIG.apiKey,
});

const ownedId = (value: number | undefined, label: string): number => {
  expect(value, `${label} must be recorded before use`).toEqual(expect.any(Number));
  return value as number;
};

const newWatchlistJourneyState = (): WatchlistJourneyState => {
  const token = `task2d-watchlist-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  return {
    token,
    initialItemTitle: 'Task 2D persisted briefing article',
    uiRunItemTitle: 'Task 2D UI Run Now article',
    jobName: `Task 2D Morning Brief ${token}`,
  };
};

const requireOk = async (
  response: Awaited<ReturnType<APIRequestContext['get']>>,
  label: string
) => {
  expect(response.ok(), `${label}: ${response.status()} ${await response.text()}`).toBe(true);
  return response;
};

const runIdFromResponse = async (
  request: APIRequestContext,
  jobId: number,
  label: string
): Promise<number> => {
  const response = await requireOk(
    await request.post(apiUrl(`/api/v1/watchlists/jobs/${jobId}/run`), {
      headers: apiHeaders(),
    }),
    label
  );
  const body = (await response.json()) as { id?: unknown; job_id?: unknown };
  expect(body.job_id).toBe(jobId);
  expect(body.id).toEqual(expect.any(Number));
  return body.id as number;
};

const waitForPersistedBriefing = async (
  request: APIRequestContext,
  state: Pick<WatchlistJourneyState, 'watchlistId' | 'jobId'>,
  runId: number
) => {
  const jobId = ownedId(state.jobId, 'job ID');
  const watchlistId = ownedId(state.watchlistId, 'Watchlist ID');
  let projection: Record<string, unknown> | null = null;
  await expect
    .poll(async () => {
      const response = await request.get(apiUrl(`/api/v1/watchlists/runs/${runId}/briefing`), {
        headers: apiHeaders(),
      });
      if (!response.ok()) return response.status();
      projection = (await response.json()) as Record<string, unknown>;
      return response.status();
    })
    .toBe(200);

  expect(projection).not.toBeNull();
  expect(projection).toMatchObject({ run_id: runId, job_id: jobId });
  expect(projection?.occurrence_id).toEqual(expect.any(Number));
  expect(projection?.output).toMatchObject({ id: expect.any(Number) });

  const latest = await requireOk(
    await request.get(apiUrl(`/api/v1/watchlists/briefings/latest?watchlist_id=${watchlistId}`), {
      headers: apiHeaders(),
    }),
    'load latest persisted briefing'
  );
  expect(await latest.json()).toMatchObject({
    occurrence_id: projection?.occurrence_id,
    run_id: runId,
    job_id: jobId,
  });
};

const provisionPersistedWatchlistJourneyState = async (
  request: APIRequestContext,
  state: WatchlistJourneyState
): Promise<void> => {
  const watchlistResponse = await requireOk(
    await request.post(apiUrl('/api/v1/watchlists'), {
      headers: apiHeaders(),
      data: {
        name: `Task 2D Watchlist ${state.token}`,
        description: 'Owned real-run fixture for the Watchlist briefing journey.',
        objective: 'Prove a persisted briefing occurrence reaches repeat controls.',
      },
    }),
    'create owned Watchlist'
  );
  const watchlist = (await watchlistResponse.json()) as { id: number };
  state.watchlistId = watchlist.id;

  const sourceResponse = await requireOk(
    await request.post(apiUrl('/api/v1/watchlists/sources'), {
      headers: apiHeaders(),
      data: {
        name: `Task 2D Feed ${state.token}`,
        url: `${TEST_CONFIG.webUrl}/e2e/task2d-watchlist-feed.xml?token=${state.token}`,
        source_type: 'rss',
        watchlist_id: watchlist.id,
        settings: { rss: { use_feed_content_if_available: true, feed_content_min_chars: 1 } },
        tags: [state.token],
      },
    }),
    'create owned RSS source'
  );
  const source = (await sourceResponse.json()) as { id: number };
  state.sourceId = source.id;
  await requireOk(
    await request.delete(apiUrl(`/api/v1/watchlists/sources/${source.id}/seen`), {
      headers: apiHeaders(),
    }),
    'clear reused source seen state before the owned run'
  );

  const jobResponse = await requireOk(
    await request.post(apiUrl('/api/v1/watchlists/jobs'), {
      headers: apiHeaders(),
      data: {
        name: state.jobName,
        description: 'Owned real-run monitor for the Watchlist briefing journey.',
        scope: { sources: [source.id] },
        active: true,
        watchlist_id: watchlist.id,
        output_prefs: {
          briefing_pipeline: {
            version: 1,
            editorial: { show_name: state.jobName },
            text: { enabled: true, type: 'briefing_markdown', format: 'md', show_notes: true },
            audio: { enabled: false },
            delivery: {
              reports: { enabled: true },
              email: { enabled: false },
              chatbook: { enabled: false },
            },
          },
        },
      },
    }),
    'create owned briefing job'
  );
  const job = (await jobResponse.json()) as { id: number; watchlist_id: number };
  expect(job.watchlist_id).toBe(watchlist.id);
  state.jobId = job.id;

  state.initialRunId = await runIdFromResponse(request, job.id, 'run owned briefing job');
  await waitForPersistedBriefing(request, state, state.initialRunId);
};

const updateSourceForUiRun = async (
  request: APIRequestContext,
  state: WatchlistJourneyState
): Promise<void> => {
  const sourceId = ownedId(state.sourceId, 'source ID');
  const uiRunFeedUrl = `${TEST_CONFIG.webUrl}/e2e/task2d-watchlist-ui-run-feed.xml?token=${state.token}`;
  const response = await requireOk(
    await request.patch(apiUrl(`/api/v1/watchlists/sources/${sourceId}`), {
      headers: apiHeaders(),
      data: { url: uiRunFeedUrl },
    }),
    'switch owned source to the UI Run Now feed'
  );
  const source = (await response.json()) as { id?: unknown; url?: unknown };
  expect(source.id).toBe(sourceId);
  expect(source.url).toBe(uiRunFeedUrl);
};

const waitForOwnedRunItem = async (
  request: APIRequestContext,
  state: WatchlistJourneyState,
  runId: number
) => {
  const watchlistId = ownedId(state.watchlistId, 'Watchlist ID');
  const jobId = ownedId(state.jobId, 'job ID');
  const sourceId = ownedId(state.sourceId, 'source ID');
  let ownedItem: Record<string, unknown> | null = null;
  await expect
    .poll(async () => {
      const response = await request.get(
        apiUrl(
          `/api/v1/watchlists/items?run_id=${runId}&watchlist_id=${watchlistId}&status=ingested&size=50`
        ),
        { headers: apiHeaders() }
      );
      if (!response.ok()) return response.status();
      const payload = (await response.json()) as { items?: Array<Record<string, unknown>> };
      ownedItem =
        payload.items?.find(
          (item) =>
            item.run_id === runId &&
            item.job_id === jobId &&
            item.source_id === sourceId &&
            item.status === 'ingested' &&
            item.title === state.uiRunItemTitle
        ) ?? null;
      return ownedItem?.run_id ?? null;
    })
    .toBe(runId);

  expect(ownedItem).toMatchObject({
    run_id: runId,
    job_id: jobId,
    source_id: sourceId,
    status: 'ingested',
    title: state.uiRunItemTitle,
    content: expect.stringContaining('UI Run Now'),
  });
  return ownedItem;
};

const cleanupWatchlistJourneyState = async (
  request: APIRequestContext,
  state: WatchlistJourneyState
): Promise<void> => {
  const cleanupFailures: string[] = [];
  if (state.sourceId !== undefined) {
    try {
      const response = await request.delete(
        apiUrl(`/api/v1/watchlists/sources/${state.sourceId}/seen`),
        { headers: apiHeaders() }
      );
      if (!response.ok()) {
        cleanupFailures.push(`source seen state: ${response.status()} ${await response.text()}`);
      }
    } catch (error) {
      cleanupFailures.push(
        `source seen state: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
  const resources = [
    ['job', state.jobId, (id: number) => `/api/v1/watchlists/jobs/${id}`],
    ['source', state.sourceId, (id: number) => `/api/v1/watchlists/sources/${id}`],
    ['Watchlist', state.watchlistId, (id: number) => `/api/v1/watchlists/${id}`],
  ] as const;
  for (const [label, id, pathForId] of resources) {
    if (id === undefined) continue;
    try {
      const response = await request.delete(apiUrl(pathForId(id)), { headers: apiHeaders() });
      if (!response.ok()) {
        cleanupFailures.push(`${label}: ${response.status()} ${await response.text()}`);
      }
    } catch (error) {
      cleanupFailures.push(`${label}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  expect(cleanupFailures, `cleanup failures:\n${cleanupFailures.join('\n')}`).toEqual([]);
};

const setupNotificationsRoute = async (page: Page) => {
  const state = { notification: null as MockNotification | null };
  await page.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route: Route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (request.method() === 'GET' && pathname === '/api/v1/notifications') {
      const items = state.notification ? [state.notification] : [];
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ items, total: items.length }),
      });
      return;
    }
    if (request.method() === 'GET' && pathname === '/api/v1/notifications/unread-count') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ unread_count: state.notification ? 1 : 0 }),
      });
      return;
    }
    if (request.method() === 'GET' && pathname === '/api/v1/notifications/stream') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: '' });
      return;
    }
    await route.continue();
  });
  return {
    recordCompletedRun: (jobName: string, runId: number) => {
      state.notification = {
        id: runId,
        kind: 'watchlist_run',
        title: 'Run completed',
        message: `${jobName} completed run ${runId}.`,
        severity: 'info',
        created_at: new Date().toISOString(),
        read_at: null,
        dismissed_at: null,
      };
    },
    notification: () => state.notification,
  };
};

test.describe('Watchlist -> Ingest -> Notify journey', () => {
  test('runs a persisted briefing monitor, surfaces the new article, and shows the inbox notification', async ({
    authedPage: page,
    request,
    serverInfo,
    diagnostics,
  }) => {
    skipIfServerUnavailable(serverInfo);
    const notifications = await setupNotificationsRoute(page);
    const state = newWatchlistJourneyState();
    let completedRunId: number | null = null;

    try {
      await provisionPersistedWatchlistJourneyState(request, state);
      await updateSourceForUiRun(request, state);

      await test.step('Open the owned monitor with its persisted briefing', async () => {
        await page.goto('/watchlists', { waitUntil: 'domcontentloaded' });
        await waitForConnection(page);

        await expect(page.getByTestId('watchlists-container-shell')).toContainText(
          `Task 2D Watchlist ${state.token}`
        );

        await expect(page.getByTestId('watchlists-health-bar')).toBeVisible();
        await expect(page.getByTestId('watchlists-repeat-actions')).toBeVisible();
        await expect(page.getByTestId('watchlists-repeat-open-runs')).toBeVisible();

        await page.getByTestId('watchlists-help-icon').click();
        await page.getByTestId('watchlists-open-command-palette').click();
        await expect(page.getByTestId('watchlists-command-palette-input')).toBeVisible();
        await page.getByTestId('watchlists-command-nav-monitors').click();

        const monitorTable = page.getByLabel(/Monitors table/i);
        await expect(monitorTable).toBeVisible();
        await expect(monitorTable.getByText(state.jobName)).toBeVisible();
      });

      await test.step('Trigger the real Run Now action and persist its exact briefing', async () => {
        const jobId = ownedId(state.jobId, 'job ID');
        const runRequest = expectApiCall(page, {
          method: 'POST',
          url: new RegExp(`/api/v1/watchlists/jobs/${jobId}/run$`),
        });
        const monitorRow = page.getByRole('row').filter({ hasText: state.jobName });
        await monitorRow.getByRole('button', { name: /^Run Now$/i }).click();

        const { response } = await runRequest;
        expect(response.status()).toBe(200);
        const body = (await response.json()) as { id?: unknown; job_id?: unknown };
        expect(body.job_id).toBe(jobId);
        expect(body.id).toEqual(expect.any(Number));
        completedRunId = body.id as number;
        await waitForPersistedBriefing(request, state, completedRunId);
        await waitForOwnedRunItem(request, state, completedRunId);
        notifications.recordCompletedRun(state.jobName, completedRunId);
      });

      await test.step('Verify the completed run appears in Activity', async () => {
        expect(completedRunId).toEqual(expect.any(Number));
        await page.getByRole('tab', { name: /^Overview$/ }).click();
        const inspectRun = page.getByTestId('watchlists-repeat-open-runs');
        await expect(inspectRun).toHaveAccessibleName(
          new RegExp(`^Inspect run ${completedRunId}:`)
        );
        await inspectRun.click();
        const activitySection = page.getByTestId('watchlists-secondary-activity');
        const exactRunOutputs = activitySection.getByTestId(
          `watchlists-run-open-outputs-${completedRunId}`
        );
        const exactRunRow = activitySection.getByRole('row').filter({
          has: page.getByTestId(`watchlists-run-open-outputs-${completedRunId}`),
        });
        await expect(page.getByLabel(/Activity runs table/i)).toBeVisible();
        await expect(exactRunRow).toHaveCount(1);
        await expect(exactRunRow).toContainText(state.jobName);
        await expect(exactRunOutputs).toBeVisible();
        const runDetails = page.getByRole('dialog', { name: 'Run Details' });
        await expect(runDetails).toBeVisible();
        await runDetails.getByRole('button', { name: 'Close' }).click();
        await expect(runDetails).toBeHidden();
      });

      await test.step('Verify the ingested article appears in Updates', async () => {
        await expect(page.getByRole('tab', { name: 'Updates' })).toHaveAttribute(
          'aria-selected',
          'true'
        );
        const row = page
          .getByTestId(/watchlists-item-row-/)
          .filter({ hasText: state.uiRunItemTitle })
          .first();
        await expect(row).toBeVisible();
        await row.click();
        await expect(page.getByTestId('watchlists-item-reader')).toContainText(
          state.uiRunItemTitle
        );
        await expect(page.getByTestId('watchlists-item-reader')).toContainText(
          'real Watchlist UI Run Now fetched this deterministic RSS article'
        );
      });

      await test.step('Verify the notification inbox reflects the completed run', async () => {
        const notification = notifications.notification();
        expect(notification).not.toBeNull();
        const notificationsPage = new NotificationsPage(page);
        await page.goto('/notifications', { waitUntil: 'domcontentloaded' });
        await notificationsPage.assertPageReady();
        await notificationsPage.waitForLoaded();

        await expect(notificationsPage.notificationsList).toBeVisible();
        await expect(notificationsPage.unreadLabel).toContainText('Unread: 1');
        await expect(page.getByText('Run completed')).toBeVisible();
        await expect(page.getByText(notification?.message || '')).toBeVisible();
      });

      await assertNoCriticalErrors(diagnostics);
    } finally {
      await cleanupWatchlistJourneyState(request, state);
    }
  });
});
