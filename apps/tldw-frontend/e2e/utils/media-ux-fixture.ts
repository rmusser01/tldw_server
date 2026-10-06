import { seedAuth } from './helpers';

export const FRONTEND_URL = 'http://127.0.0.1:18881';
export const DUMMY_API_URL = 'http://127.0.0.1:18882';

const titles = [
  'Understanding retrieval: a field guide',
  'Research meeting — September transcript',
  'Designing a personal knowledge archive',
  'An unusually long document title about reviewing research evidence across formats and preserving useful context for future work',
];

export function createMediaItems(count = 40) {
  return Array.from({ length: count }, (_, index) => {
    const id = index + 1;
    const type = ['document', 'video', 'audio', 'pdf'][index % 4];
    const title =
      index < titles.length ? titles[index] : `Research source ${String(id).padStart(2, '0')}`;
    const content =
      `# ${title}\n\nThis demonstration source compares retrieval quality, useful evidence, and the work required to maintain a personal knowledge archive.\n\n## Main observations\n\nA research assistant should preserve source context and make it easy to return to the original evidence. A media library combines documents, audio, video, and meeting notes.\n\n## Next steps\n\nReview the transcript, verify quoted claims, and add research tags.\n\n` +
      Array.from(
        { length: 8 },
        (_, n) =>
          `Paragraph ${n + 1}: retrieval quality improves when source context is preserved and the researcher can inspect the complete content.`
      ).join('\n\n');
    return {
      id,
      title,
      type,
      media_type: type,
      author: 'Demo researcher',
      created_at: '2026-09-20T10:30:00Z',
      updated_at: '2026-09-21T09:00:00Z',
      url: `https://example.com/research/${id}`,
      source: 'example.com',
      snippet:
        'A practical source about retrieval quality, evidence, and maintaining a research archive.',
      content,
      analysis:
        '## Summary\n\nPreserve evidence and source context.\n\n- Review original content\n- Verify claims\n- Tag related research',
      keywords: ['research', index % 2 ? 'transcript' : 'retrieval'],
      version: 1,
      latest_version: { version_number: 1, content },
      versions: [{ version_number: 1, content }],
      is_deleted: false,
    };
  });
}

export async function installBrowserFixture(
  page: any,
  options: {
    empty?: boolean;
    count?: number;
    processingPolls?: number;
    ingestFailure?: boolean;
    errorList?: boolean;
  } = {}
) {
  const state = {
    items: options.empty ? [] : createMediaItems(options.count ?? 40),
    processingPolls: options.processingPolls ?? 1,
    ingestFailure: options.ingestFailure ?? false,
    errorList: options.errorList ?? false,
    requests: [] as Array<{ method: string; path: string; body?: any }>,
    jobs: new Map<number, any>(),
    nextJob: 7100,
    nextMedia: 40,
    attempts: new Map<string, number>(),
    trash: [] as any[],
  };
  await seedAuth(page, {
    serverUrl: DUMMY_API_URL,
    apiKey: 'ux-review-fixture-key',
    webUrl: FRONTEND_URL,
    allowOffline: false,
  });
  await page.route('**/*', async (route: any) => {
    const req = route.request();
    const url = new URL(req.url());
    if (url.origin !== DUMMY_API_URL) {
      if (
        url.origin === FRONTEND_URL ||
        url.hostname === 'localhost' ||
        url.hostname === '127.0.0.1' ||
        url.protocol === 'data:' ||
        url.protocol === 'blob:'
      )
        return route.continue();
      return route.abort();
    }
    const method = req.method().toUpperCase();
    const path = url.pathname.replace(/\/+$/, '') || '/';
    let body: any = null;
    try {
      body = req.postDataJSON();
    } catch {
      const contentType = req.headers?.()['content-type'];
      if (contentType?.includes('multipart/form-data')) {
        const form = await new Response(req.postDataBuffer(), {
          headers: { 'Content-Type': contentType },
        }).formData();
        body = {};
        for (const [key, value] of form.entries()) {
          if (typeof value !== 'string') body[key] = { name: value.name, size: value.size };
          else {
            try {
              body[key] = JSON.parse(value);
            } catch {
              body[key] = value;
            }
          }
        }
      }
    }
    state.requests.push({ method, path, body });
    const fulfill = (data: any, status = 200, contentType = 'application/json') =>
      route.fulfill({
        status,
        contentType,
        headers: {
          'access-control-allow-origin': '*',
          'access-control-allow-headers': '*',
          'access-control-allow-methods': 'GET,POST,PATCH,PUT,DELETE,OPTIONS',
        },
        body: contentType === 'application/json' ? JSON.stringify(data) : data,
      });
    if (method === 'OPTIONS')
      return route.fulfill({
        status: 204,
        headers: {
          'access-control-allow-origin': '*',
          'access-control-allow-headers': '*',
          'access-control-allow-methods': 'GET,POST,PATCH,PUT,DELETE,OPTIONS',
        },
      });
    if (path === '/health' || path === '/api/v1/health')
      return fulfill({ status: 'ok', version: 'ux-review-fixture', auth_mode: 'single_user' });
    if (path === '/openapi.json')
      return fulfill({
        openapi: '3.1.0',
        info: { title: 'UX review fixture', version: '1' },
        paths: {
          '/api/v1/media': { get: {} },
          '/api/v1/media/search': { post: {} },
          '/api/v1/media/{media_id}': { get: {} },
          '/api/v1/media/ingest/jobs': { post: {} },
          '/api/v1/media/collections': { get: {}, post: {} },
          '/api/v1/media/collections/{collection_id}': { get: {} },
          '/api/v1/rag/search': { post: {} },
          '/api/v1/media/process-web-scraping': { post: {} },
          '/api/v1/media/process-documents': { post: {} },
        },
      });
    if (path === '/api/v1/config/docs-info')
      return fulfill({
        capabilities: {
          hasMedia: true,
          hasMediaIngestJobs: true,
          hasMediaIngestJobEvents: false,
          hasMediaIngestWorker: true,
          hasMediaPlaylistPreflight: false,
          hasDurableMediaCollections: true,
          hasKnowledgeQaMediaScope: true,
        },
      });
    if (path.includes('/setup/first-run/state'))
      return fulfill({
        status: 'completed',
        current_step: null,
        completed_steps: ['first_chat'],
        skipped_steps: [],
        acknowledged_steps: ['first_chat'],
        step_data: {},
        first_chat: { completed: true },
        setup_completed: true,
      });
    if (path.includes('/setup/first-run/metadata'))
      return fulfill({
        auth_mode: 'single_user',
        manual_auth_required: false,
        setup_required: false,
        setup_completed: true,
        remote_setup_enabled: false,
        connection: {
          frontend_origin: FRONTEND_URL,
          api_origin: DUMMY_API_URL,
          browser_access: 'local',
        },
        setup_paths: [],
      });
    if (path === '/api/v1/auth/me' || path === '/api/v1/users/me')
      return fulfill({
        id: 1,
        user_id: 1,
        username: 'ux-review-demo',
        email: 'demo@example.com',
        is_active: true,
        is_admin: true,
        role: 'admin',
      });
    if (path === '/api/v1/media/capabilities') return fulfill({ can_delete: true });
    if (path === '/api/v1/media' || path === '/api/v1/media/search') {
      if (state.errorList)
        return fulfill({ detail: 'Fixture service temporarily unavailable' }, 503);
      let items = [...state.items].filter((item) => !item.is_deleted);
      const q = String(body?.query ?? url.searchParams.get('query') ?? '').toLowerCase();
      if (q)
        items = items.filter((item) =>
          `${item.title} ${item.snippet} ${item.content}`.toLowerCase().includes(q)
        );
      if (body?.media_types?.length)
        items = items.filter((item) => body.media_types.includes(item.type));
      if (body?.must_have?.length)
        items = items.filter((item) =>
          body.must_have.every((tag: string) => item.keywords.includes(tag))
        );
      const pageNumber = Number(body?.page || url.searchParams.get('page') || 1);
      const pageSize = Number(
        body?.results_per_page || url.searchParams.get('results_per_page') || 20
      );
      return fulfill({
        items: items.slice((pageNumber - 1) * pageSize, pageNumber * pageSize),
        pagination: {
          page: pageNumber,
          results_per_page: pageSize,
          total_items: items.length,
          total_pages: Math.max(1, Math.ceil(items.length / pageSize)),
        },
      });
    }
    if (path === '/api/v1/media/statistics')
      return fulfill({
        total_media: state.items.length,
        total_items: state.items.length,
        media_by_type: { document: 10, video: 10, audio: 10, pdf: 10 },
      });
    if (path === '/api/v1/media/keywords')
      return fulfill({ keywords: ['research', 'retrieval', 'transcript'] });
    if (path === '/api/v1/media/collections')
      return fulfill({ items: [], collections: [], total: 0 });
    const finishJob = (job: any) => {
      if (job.status === 'queued' || job.status === 'processing') {
        if (job.polls-- > 0) job.status = 'processing';
        else if (job.fail) {
          job.status = 'failed';
          job.error_message = 'Network error: simulated source temporarily unavailable';
        } else {
          job.status = 'completed';
          const id = ++state.nextMedia;
          const title = job.source_kind === 'file' ? job.source : `Imported source ${id}`;
          job.result = {
            status: 'Success',
            media_id: id,
            title,
            source_url: job.source_url,
            content: 'Simulated extracted evidence for UI verification.',
          };
          state.items.push({
            ...createMediaItems(1)[0],
            id,
            title,
            content: job.result.content,
            url: job.source_url,
          });
        }
      }
      return {
        ...job,
        job_id: job.id,
        progress_percent: job.status === 'processing' ? 40 : 100,
        progress_message: 'Simulated processing outcome',
      };
    };
    if (path === '/api/v1/media/process-web-scraping' && method === 'POST') {
      const source = String(body?.url_input || '');
      const attempts = (state.attempts.get(source) || 0) + 1;
      state.attempts.set(source, attempts);
      if (state.ingestFailure || (source.includes('fail-once') && attempts === 1)) {
        return fulfill({ detail: 'Network error: simulated source temporarily unavailable' }, 503);
      }
      const id = ++state.nextMedia;
      const item = { ...createMediaItems(1)[0], id, title: `Imported source ${id}`, url: source };
      state.items.push(item);
      return fulfill({
        status: 'persist-ok',
        media_ids: [id],
        title: item.title,
        content: 'Simulated extracted evidence for UI verification.',
      });
    }
    if (path === '/api/v1/media/ingest/jobs' && method === 'POST') {
      const id = ++state.nextJob;
      const source =
        body?.urls?.[0] || body?.url || body?.file?.name || body?.files?.name || 'source.txt';
      const attempts = (state.attempts.get(source) || 0) + 1;
      state.attempts.set(source, attempts);
      const job = {
        id,
        batch_id: `batch-${id}`,
        polls: state.processingPolls,
        source,
        source_url: /^https?:/.test(source) ? source : undefined,
        source_kind: /^https?:/.test(source) ? 'url' : 'file',
        status: 'queued',
        fail: state.ingestFailure || (source.includes('fail-once') && attempts === 1),
      };
      state.jobs.set(id, job);
      return fulfill({ batch_id: job.batch_id, job_ids: [id], jobs: [{ id, status: 'queued' }] });
    }
    if (path === '/api/v1/media/ingest/jobs' && method === 'GET') {
      const batch = url.searchParams.get('batch_id');
      if (!batch) return fulfill({ detail: 'batch_id required' }, 422);
      const all = [...state.jobs.values()].filter((job) => job.batch_id === batch).map(finishJob);
      const offset = Number(url.searchParams.get('offset') || 0),
        limit = Number(url.searchParams.get('limit') || 50);
      const jobs = all.slice(offset, offset + limit),
        has_more = offset + limit < all.length;
      return fulfill({
        jobs,
        offset,
        limit,
        has_more,
        next_offset: has_more ? offset + limit : null,
      });
    }
    const jobMatch = path.match(/^\/api\/v1\/media\/ingest\/jobs\/(\d+)$/);
    if (jobMatch) return fulfill(finishJob(state.jobs.get(Number(jobMatch[1]))));
    if (path === '/api/v1/media/ingest/jobs/cancel') {
      for (const id of body?.job_ids || []) {
        const job = state.jobs.get(id);
        if (job) job.status = 'cancelled';
      }
      return fulfill({ cancelled: body?.job_ids || [] });
    }
    if (path === '/api/v1/media/trash')
      return fulfill({
        items: state.items.filter((item) => item.is_deleted),
        total: state.items.filter((item) => item.is_deleted).length,
      });
    const restore = path.match(/^\/api\/v1\/media\/(\d+)\/restore$/);
    if (restore) {
      const item = state.items.find((item) => item.id === Number(restore[1]));
      if (item) item.is_deleted = false;
      return fulfill({ status: 'success', id: Number(restore[1]) });
    }
    if (path.includes('/notifications/stream')) return fulfill('', 200, 'text/event-stream');
    if (path === '/api/v1/notifications/unread-count') return fulfill({ unread_count: 0 });
    if (path === '/api/v1/notifications') return fulfill({ items: [], total: 0 });
    if (path.includes('/notes') || path.includes('/prompts'))
      return fulfill({
        items: [],
        notes: [],
        prompts: [],
        pagination: { total_items: 0, total_pages: 1 },
      });
    if (path.includes('/llm/providers'))
      return fulfill({ providers: [{ name: 'openai', configured: true, models: ['demo-model'] }] });
    if (path.includes('/models'))
      return fulfill({
        models: [{ id: 'demo-model', name: 'Demo model', provider: 'openai' }],
        data: [{ id: 'demo-model' }],
      });
    const detailMatch = path.match(/^\/api\/v1\/media\/(\d+)$/);
    if (detailMatch && method === 'DELETE') {
      const item = state.items.find((item) => item.id === Number(detailMatch[1]));
      if (item) item.is_deleted = true;
      return fulfill({ status: 'success' });
    }
    if (detailMatch)
      return fulfill(
        state.items.find((item) => item.id === Number(detailMatch[1])) || {
          ...createMediaItems(1)[0],
          id: Number(detailMatch[1]),
          title: 'New demonstration source',
        }
      );
    if (path.includes('/outline')) return fulfill({ items: [] });
    if (path.includes('/annotations')) return fulfill({ annotations: [], total: 0 });
    if (path.includes('/insights')) return fulfill({ insights: [] });
    if (path.includes('/references')) return fulfill({ references: [], total: 0 });
    if (path.includes('/figures')) return fulfill({ figures: [] });
    if (path.includes('/progress')) return fulfill({ progress: 0 });
    return fulfill({ items: [], data: [], status: 'ok' });
  });
  return state;
}
