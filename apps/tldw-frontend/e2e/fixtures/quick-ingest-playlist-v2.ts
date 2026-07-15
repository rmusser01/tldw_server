import type { Page, Route } from '@playwright/test';

export const CONFERENCE_PLAYLIST_URL =
  'https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B';

const PREFLIGHT_ID = 'preflight-conference-v2';
const MATERIALIZATION_ID = 'materialization-conference-v2';
const RUN_ID = 'run-conference-v2';
const BATCH_ID = 'batch-conference-v2';
const EXPIRES_AT = '2099-07-14T23:59:59Z';

const fulfillJson = async (route: Route, body: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });
};

const conferenceItems = Array.from({ length: 34 }, (_, index) => {
  const ordinal = index + 1;
  const videoId =
    ordinal === 18 ? 'conference-talk-08' : `conference-talk-${String(ordinal).padStart(2, '0')}`;
  const duplicateStatus =
    ordinal === 8 ? 'duplicate_existing' : ordinal === 18 ? 'duplicate_in_batch' : 'new';
  return {
    occurrence_id: `conference-occurrence-${ordinal}`,
    ordinal,
    occurrence_index_for_source: ordinal === 18 ? 2 : 1,
    source_url: `https://www.youtube.com/watch?v=${videoId}`,
    normalized_source_id: `youtube:video:${videoId}`,
    source_kind: 'youtube_video',
    availability: 'available',
    duplicate_status: duplicateStatus,
    duplicate_of_occurrence_id: ordinal === 18 ? 'conference-occurrence-8' : null,
    selected_by_default: duplicateStatus === 'new',
    display_metadata: {
      title: `Talk ${ordinal}`,
      channel_or_uploader: `Speaker ${ordinal}`,
      duration_seconds: 1_800 + ordinal,
      published_at: `2010-09-${String(Math.min(ordinal, 28)).padStart(2, '0')}`,
      thumbnail_url:
        ordinal === 1 ? 'https://i.ytimg.com/vi/conference-talk-01/hqdefault.jpg' : null,
      playlist_id: 'PL0065D9B288E6804B',
      playlist_title: 'Conference 2010',
    },
  };
});

const openApiPaths = Object.fromEntries(
  [
    '/api/v1/media',
    '/api/v1/media/playlist-preflights',
    '/api/v1/media/playlist-preflights/{preflight_id}',
    '/api/v1/media/playlist-preflights/{preflight_id}/items',
    '/api/v1/media/playlist-preflights/{preflight_id}/materializations',
    '/api/v1/media/ingest/runs',
    '/api/v1/media/ingest/runs/{run_id}',
    '/api/v1/media/ingest/runs/{run_id}/items',
    '/api/v1/media/ingest/runs/{run_id}/events/stream',
    '/api/v1/media/ingest/runs/{run_id}/cancel',
    '/api/v1/media/ingest/runs/{run_id}/retry',
    '/api/v1/media/ingest/jobs',
  ].map((path) => [path, { get: {}, post: {}, delete: {} }])
);

export type QuickIngestPlaylistV2Fixture = {
  completeRun: () => void;
  getDurableFirstProgress: () => Promise<Record<string, unknown> | null>;
  getDurableTracking: () => Promise<Record<string, unknown> | null>;
  getMaterializedOccurrenceIds: () => string[];
  getRunInputs: () => Array<Record<string, unknown>>;
  getRunItemsRequestCount: () => number;
  getRunSummaryRequestCount: () => number;
  getPreflightItemsRequestCount: () => number;
  getSubmissionPostCount: () => number;
  getPerItemPollCount: () => number;
  getRemoteThumbnailRequestCount: () => number;
};

export const installQuickIngestPlaylistV2Routes = async (
  page: Page
): Promise<QuickIngestPlaylistV2Fixture> => {
  let terminal = false;
  let preflightItemsRequestCount = 0;
  let submissionPostCount = 0;
  let perItemPollCount = 0;
  let remoteThumbnailRequestCount = 0;
  let runItemsRequestCount = 0;
  let runSummaryRequestCount = 0;
  let materializedOccurrenceIds: string[] = [];
  let runInputs: Array<Record<string, unknown>> = [];

  await page.route('https://i.ytimg.com/**', async (route) => {
    remoteThumbnailRequestCount += 1;
    await route.abort();
  });
  await page.route('**/openapi.json', (route) =>
    fulfillJson(route, {
      openapi: '3.1.0',
      info: { title: 'tldw playlist v2 e2e', version: 'e2e' },
      paths: openApiPaths,
    })
  );
  await page.route('**/api/v1/config/docs-info', (route) =>
    fulfillJson(route, {
      capabilities: {
        mediaPlaylistIngestContractVersion: 2,
        hasMediaPlaylistPreflight: true,
        hasMediaIngestJobs: true,
        hasMediaIngestJobEvents: true,
        hasMediaIngestWorker: true,
      },
    })
  );
  await page.route('**/api/v1/health', (route) =>
    fulfillJson(route, { status: 'ok', version: 'e2e' })
  );
  await page.route(/\/api\/v1\/media\/?(?:\?.*)?$/, async (route, request) => {
    if (request.method() !== 'GET') return route.continue();
    await fulfillJson(route, {
      items: [],
      pagination: {
        page: 1,
        results_per_page: 20,
        total_items: 0,
        total_pages: 1,
      },
    });
  });
  await page.route(/\/api\/v1\/media\/search(?:\?.*)?$/, async (route, request) => {
    if (request.method() !== 'GET' && request.method() !== 'POST') {
      return route.continue();
    }
    await fulfillJson(route, {
      items: [],
      pagination: {
        page: 1,
        results_per_page: 20,
        total_items: 0,
        total_pages: 1,
      },
    });
  });
  await page.route('**/api/v1/media/playlist-preflights', async (route, request) => {
    if (request.method() !== 'POST') return route.continue();
    await fulfillJson(
      route,
      {
        contract_version: 2,
        preflight_id: PREFLIGHT_ID,
        status: 'pending',
        status_url: `/api/v1/media/playlist-preflights/${PREFLIGHT_ID}`,
        items_url: `/api/v1/media/playlist-preflights/${PREFLIGHT_ID}/items`,
        expires_at: EXPIRES_AT,
        limits: { max_items: 500, global_capacity: 10, owner_capacity: 2 },
      },
      202
    );
  });
  await page.route(
    `**/api/v1/media/playlist-preflights/${PREFLIGHT_ID}`,
    async (route, request) => {
      if (request.method() === 'DELETE') {
        await route.fulfill({ status: 204, body: '' });
        return;
      }
      if (request.method() !== 'GET') return route.continue();
      await fulfillJson(route, {
        contract_version: 2,
        preflight_id: PREFLIGHT_ID,
        status: 'ready',
        source_url: CONFERENCE_PLAYLIST_URL,
        source_kind: 'youtube_watch_playlist',
        playlist_id: 'PL0065D9B288E6804B',
        summary: {
          playlist_title: 'Conference 2010',
          total_count: 34,
          loaded_count: 34,
          ingestible_count: 34,
          unavailable_count: 0,
          duplicate_count: 2,
          selected_count: 32,
          warnings: [],
        },
        error: null,
        created_at: '2026-07-14T20:00:00Z',
        updated_at: '2026-07-14T20:00:01Z',
        expires_at: EXPIRES_AT,
      });
    }
  );
  await page.route(
    `**/api/v1/media/playlist-preflights/${PREFLIGHT_ID}/items*`,
    async (route, request) => {
      if (request.method() !== 'GET') return route.continue();
      preflightItemsRequestCount += 1;
      const cursor = new URL(request.url()).searchParams.get('cursor');
      await fulfillJson(route, {
        contract_version: 2,
        preflight_id: PREFLIGHT_ID,
        items: cursor === 'page-2' ? conferenceItems.slice(20) : conferenceItems.slice(0, 20),
        next_cursor: cursor === 'page-2' ? null : 'page-2',
      });
    }
  );
  await page.route(
    `**/api/v1/media/playlist-preflights/${PREFLIGHT_ID}/materializations`,
    async (route, request) => {
      if (request.method() !== 'POST') return route.continue();
      const body = request.postDataJSON() as { occurrence_ids?: string[] };
      materializedOccurrenceIds = [...(body.occurrence_ids ?? [])];
      const selected = materializedOccurrenceIds
        .map((occurrenceId) => conferenceItems.find((item) => item.occurrence_id === occurrenceId))
        .filter((item): item is (typeof conferenceItems)[number] => Boolean(item));
      await fulfillJson(route, {
        contract_version: 2,
        materialization_id: MATERIALIZATION_ID,
        preflight_id: PREFLIGHT_ID,
        status: 'ready',
        items: selected.map((item) => ({
          occurrence_id: item.occurrence_id,
          ordinal: item.ordinal,
          source_url: item.source_url,
          normalized_source_id: item.normalized_source_id,
          source_kind: item.source_kind,
          display_metadata: item.display_metadata,
        })),
        expires_at: EXPIRES_AT,
      });
    }
  );
  await page.route('**/api/v1/media/ingest/runs', async (route, request) => {
    if (request.method() !== 'POST') return route.continue();
    const body = request.postDataJSON() as { inputs?: Array<Record<string, unknown>> };
    runInputs = [...(body.inputs ?? [])];
    const processingOccurrences = runInputs.map((input, index) => {
      const occurrenceId = String(input.occurrence_id);
      const item = conferenceItems.find((candidate) => candidate.occurrence_id === occurrenceId);
      return {
        occurrence_id: occurrenceId,
        ordinal: item?.ordinal ?? index + 1,
        input_kind: 'materialized_playlist_item',
        source_url: item?.source_url ?? null,
        source_kind: item?.source_kind ?? 'youtube_video',
        display_metadata: item?.display_metadata ?? {},
        state: 'staged',
        outcome: null,
        job_id: null,
        batch_id: null,
        attempt: 1,
        planned_collection_item_id: null,
      };
    });
    await fulfillJson(route, {
      contract_version: 2,
      run_id: RUN_ID,
      status: 'staged',
      version: 1,
      status_url: `/api/v1/media/ingest/runs/${RUN_ID}`,
      items_url: `/api/v1/media/ingest/runs/${RUN_ID}/items`,
      events_url: `/api/v1/media/ingest/runs/${RUN_ID}/events/stream`,
      processing_occurrences: processingOccurrences,
    });
  });
  await page.route('**/api/v1/media/ingest/jobs', async (route, request) => {
    if (request.method() !== 'POST') return route.continue();
    submissionPostCount += 1;
    const submissions = runInputs.map((input, index) => ({
      occurrence_id: String(input.occurrence_id),
      status: 'queued',
      accepted: true,
      job_id: 2_001 + index,
      batch_id: BATCH_ID,
      error_code: null,
      message: null,
      retryable: false,
      attempt: 1,
    }));
    await fulfillJson(route, {
      batch_id: BATCH_ID,
      jobs: submissions.map((submission) => ({
        id: submission.job_id,
        status: submission.status,
      })),
      errors: [],
      submissions,
    });
  });
  await page.route(/\/api\/v1\/media\/ingest\/jobs\/\d+(?:\?.*)?$/, async (route) => {
    perItemPollCount += 1;
    await fulfillJson(route, { error: 'Per-item polling is forbidden in v2.' }, 500);
  });
  await page.route(`**/api/v1/media/ingest/runs/${RUN_ID}/**`, async (route, request) => {
    const pathname = new URL(request.url()).pathname;
    if (pathname.endsWith('/events/stream')) {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: '' });
      return;
    }
    if (pathname.endsWith('/cancel') || pathname.endsWith('/retry')) {
      await fulfillJson(route, buildRunSummary(terminal));
      return;
    }
    if (!pathname.endsWith('/items') || request.method() !== 'GET') {
      return route.continue();
    }
    runItemsRequestCount += 1;
    await fulfillJson(route, {
      contract_version: 2,
      run_id: RUN_ID,
      version: terminal ? 4 : 3,
      items: runInputs.map((input, index) => {
        const occurrenceId = String(input.occurrence_id);
        const item = conferenceItems.find((candidate) => candidate.occurrence_id === occurrenceId);
        const failed = terminal && index === runInputs.length - 1;
        return {
          occurrence_id: occurrenceId,
          ordinal: item?.ordinal ?? index + 1,
          input_kind: 'materialized_playlist_item',
          source_url: item?.source_url ?? null,
          normalized_source_id: item?.normalized_source_id ?? null,
          source_kind: item?.source_kind ?? 'youtube_video',
          display_metadata: item?.display_metadata ?? {},
          action: 'process',
          state: terminal ? 'terminal' : 'running',
          outcome: terminal ? (failed ? 'processing_failed' : 'completed') : null,
          progress_percent: terminal ? 100 : 45,
          progress_message: terminal
            ? failed
              ? 'Transcription failed'
              : 'Complete'
            : `Processing ${item?.display_metadata.title ?? occurrenceId}`,
          job_id: 2_001 + index,
          batch_id: BATCH_ID,
          media_id: terminal && !failed ? 7_001 + index : null,
          planned_collection_item_id: null,
          attempt: 1,
          retryable: failed,
        };
      }),
      next_cursor: null,
    });
  });
  await page.route(`**/api/v1/media/ingest/runs/${RUN_ID}`, async (route, request) => {
    if (request.method() !== 'GET') return route.continue();
    runSummaryRequestCount += 1;
    await fulfillJson(route, buildRunSummary(terminal));
  });

  function buildRunSummary(isTerminal: boolean) {
    return {
      contract_version: 2,
      run_id: RUN_ID,
      status: isTerminal ? 'partial_failure' : 'running',
      counts: isTerminal
        ? {
            total: runInputs.length,
            completed: Math.max(0, runInputs.length - 1),
            processing_failed: 1,
          }
        : { total: runInputs.length, running: runInputs.length },
      version: isTerminal ? 4 : 3,
      collection_id: null,
      batch_ids: submissionPostCount > 0 ? [BATCH_ID] : [],
      created_at: '2026-07-14T20:01:00Z',
      updated_at: isTerminal ? '2026-07-14T20:02:00Z' : '2026-07-14T20:01:30Z',
      expires_at: EXPIRES_AT,
    };
  }

  const readLatestDurableSession = () =>
    page.evaluate(
      () =>
        new Promise<Record<string, unknown> | null>((resolve, reject) => {
          const openRequest = indexedDB.open('PageAssistDatabase');
          openRequest.onerror = () => reject(openRequest.error);
          openRequest.onsuccess = () => {
            const database = openRequest.result;
            if (!database.objectStoreNames.contains('quickIngestSessions')) {
              database.close();
              resolve(null);
              return;
            }
            const transaction = database.transaction('quickIngestSessions', 'readonly');
            const rowsRequest = transaction.objectStore('quickIngestSessions').getAll();
            rowsRequest.onerror = () => reject(rowsRequest.error);
            rowsRequest.onsuccess = () => {
              const rows = (
                rowsRequest.result as Array<{
                  updatedAt?: number;
                  value?: string;
                }>
              ).sort((left, right) => Number(right.updatedAt || 0) - Number(left.updatedAt || 0));
              database.close();
              try {
                resolve(
                  rows[0]?.value ? (JSON.parse(rows[0].value)?.state?.session ?? null) : null
                );
              } catch (error) {
                reject(error);
              }
            };
          };
        })
    );

  return {
    completeRun: () => {
      terminal = true;
    },
    getDurableFirstProgress: async () => {
      const session = await readLatestDurableSession();
      const processingState = session?.processingState;
      if (!processingState || typeof processingState !== 'object') return null;
      const perItemProgress = (processingState as { perItemProgress?: unknown }).perItemProgress;
      const first = Array.isArray(perItemProgress) ? perItemProgress[0] : null;
      return first && typeof first === 'object' ? (first as Record<string, unknown>) : null;
    },
    getDurableTracking: async () => {
      const session = await readLatestDurableSession();
      const tracking = session?.tracking;
      return tracking && typeof tracking === 'object'
        ? (tracking as Record<string, unknown>)
        : null;
    },
    getMaterializedOccurrenceIds: () => [...materializedOccurrenceIds],
    getRunInputs: () => [...runInputs],
    getRunItemsRequestCount: () => runItemsRequestCount,
    getRunSummaryRequestCount: () => runSummaryRequestCount,
    getPreflightItemsRequestCount: () => preflightItemsRequestCount,
    getSubmissionPostCount: () => submissionPostCount,
    getPerItemPollCount: () => perItemPollCount,
    getRemoteThumbnailRequestCount: () => remoteThumbnailRequestCount,
  };
};
