import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
  },
  buildAuthHeaders: vi.fn(() => ({ Authorization: 'Bearer token' })),
  getApiBaseUrl: vi.fn(() => 'http://127.0.0.1:8000/api/v1'),
  streamStructuredSSE: vi.fn(),
}));

vi.mock('@web/lib/api', () => ({
  apiClient: mocks.apiClient,
  buildAuthHeaders: mocks.buildAuthHeaders,
  getApiBaseUrl: mocks.getApiBaseUrl,
}));

vi.mock('@web/lib/sse', () => ({
  streamStructuredSSE: mocks.streamStructuredSSE,
}));

import {
  approveResearchCheckpoint,
  cancelResearchRun,
  createResearchRun,
  getResearchArtifact,
  getResearchBundle,
  getResearchRun,
  hasActiveResearchRun,
  listResearchRuns,
  pauseResearchRun,
  RESEARCH_RUNS_ACTIVE_REFETCH_INTERVAL_MS,
  researchRunsRefetchInterval,
  resumeResearchRun,
  subscribeResearchRunEvents,
} from '@web/lib/api/researchRuns';

describe('researchRuns api client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.buildAuthHeaders.mockReturnValue({ Authorization: 'Bearer token' });
    mocks.getApiBaseUrl.mockReturnValue('http://127.0.0.1:8000/api/v1');
  });

  it('calls the research run CRUD endpoints with the expected payloads', async () => {
    mocks.apiClient.get.mockResolvedValueOnce([]);
    mocks.apiClient.post.mockResolvedValue({});
    mocks.apiClient.get.mockResolvedValue({});

    await listResearchRuns();
    await createResearchRun({
      query: 'Investigate local and external evidence',
      source_policy: 'balanced',
      autonomy_mode: 'checkpointed',
    });
    await getResearchRun('rs_1');
    await pauseResearchRun('rs_1');
    await resumeResearchRun('rs_1');
    await cancelResearchRun('rs_1');
    await approveResearchCheckpoint('rs_1', 'chk_1', {});
    await getResearchArtifact('rs_1', 'plan.json');
    await getResearchBundle('rs_1');

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/research/runs');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/research/runs', {
      query: 'Investigate local and external evidence',
      source_policy: 'balanced',
      autonomy_mode: 'checkpointed',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/research/runs/rs_1');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/research/runs/rs_1/pause');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/research/runs/rs_1/resume');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/research/runs/rs_1/cancel');
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/research/runs/rs_1/checkpoints/chk_1/patch-and-approve',
      { patch_payload: {} }
    );
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/research/runs/rs_1/artifacts/plan.json');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/research/runs/rs_1/bundle');
  });

  it('reconnects research event streaming with the latest after_id cursor', async () => {
    const seenUrls: string[] = [];
    let unsubscribe: (() => void) | undefined;

    mocks.streamStructuredSSE.mockImplementation(async (url: string, _options: unknown, onEvent: (event: { event: string; id?: number; payload?: unknown }) => void) => {
      seenUrls.push(url);
      if (seenUrls.length === 1) {
        onEvent({ event: 'snapshot', id: 4, payload: { latest_event_id: 4 } });
        onEvent({ event: 'progress', id: 5, payload: { event_id: 5 } });
        return;
      }
      unsubscribe?.();
    });

    unsubscribe = subscribeResearchRunEvents({
      sessionId: 'rs_1',
      onEvent: vi.fn(),
    });

    await vi.waitFor(() => {
      expect(mocks.streamStructuredSSE).toHaveBeenCalledTimes(2);
    });

    expect(seenUrls).toEqual([
      'http://127.0.0.1:8000/api/v1/research/runs/rs_1/events/stream?after_id=0',
      'http://127.0.0.1:8000/api/v1/research/runs/rs_1/events/stream?after_id=5',
    ]);
    expect(mocks.buildAuthHeaders).toHaveBeenCalledWith('GET');
  });

  it('builds relative research event SSE URLs when the api base is relative', async () => {
    mocks.getApiBaseUrl.mockReturnValue('/api/v1');
    mocks.streamStructuredSSE.mockImplementationOnce(async (url: string, _options: unknown, _onEvent: unknown) => {
      expect(url).toBe('/api/v1/research/runs/rs_relative/events/stream?after_id=7');
    });

    const unsubscribe = subscribeResearchRunEvents({
      sessionId: 'rs_relative',
      afterId: 7,
      onEvent: vi.fn(),
      onError: vi.fn(),
    });

    await vi.waitFor(() => {
      expect(mocks.streamStructuredSSE).toHaveBeenCalledTimes(1);
    });

    unsubscribe();
  });

  it('refetch interval is false unless a run is still active (WP-40)', () => {
    const run = (status: string) => ({ id: 'rs_1', status }) as never;

    expect(researchRunsRefetchInterval(undefined)).toBe(false);
    expect(researchRunsRefetchInterval([])).toBe(false);
    expect(researchRunsRefetchInterval([run('completed'), run('failed'), run('cancelled')])).toBe(false);
    expect(hasActiveResearchRun(undefined)).toBe(false);

    for (const status of ['running', 'waiting_human', 'pending', 'queued']) {
      expect(researchRunsRefetchInterval([run('completed'), run(status)])).toBe(
        RESEARCH_RUNS_ACTIVE_REFETCH_INTERVAL_MS
      );
    }
    expect(RESEARCH_RUNS_ACTIVE_REFETCH_INTERVAL_MS).toBe(5000);
  });
});
