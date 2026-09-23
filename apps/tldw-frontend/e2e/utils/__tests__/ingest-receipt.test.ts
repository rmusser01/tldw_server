import { describe, expect, it } from 'vitest';
import type { Page, Response } from '@playwright/test';
import { requireIngestMediaId, waitForCompletedIngestJob } from '../journey-helpers';

describe('ingestion receipts require a canonical Media ID', () => {
  it.each([
    { media_id: 17 },
    { result: { media_id: 17 } },
    { data: { media_id: 17 } },
    { job: { result: { media_id: 17 } } },
    { media_ids: [17] },
  ])('accepts a documented saved-media receipt: %j', payload => {
    expect(requireIngestMediaId(payload)).toBe('17');
  });
  it('accepts the completed job result over its submission envelope', () => {
    expect(requireIngestMediaId({ job_id: 90 }, '17')).toBe('17');
  });
  it.each([
    {},
    { id: 90 },
    { job_id: 90 },
    { batch_id: 'batch-90' },
    { detail: 'Scraping service failed' },
    { media_id: 'unknown' },
    { media_id: 'job-90' },
    { media_id: 0 },
    { media_id: -17 },
  ])('rejects a failure, placeholder or unrelated identifier: %j', payload => {
    expect(() => requireIngestMediaId(payload)).toThrow(/canonical Media ID/);
  });
});

// Substitute only the external response stream; execute the actual status and ID checks.
const ingestResponses = (responses: Array<{ id: number; status: number; body: unknown }>): Page => ({
  waitForResponse: async (predicate: (response: Response) => Promise<boolean>) => {
    for (const item of responses) {
      const response = {
        request: () => ({ method: () => 'GET' }),
        url: () => `http://localhost/api/v1/media/ingest/jobs/${item.id}`,
        ok: () => item.status >= 200 && item.status < 300,
        status: () => item.status,
        json: async () => item.body,
      } as Response;
      if (await predicate(response)) return response;
    }
    throw new Error('Timed out waiting for the ingest job');
  },
} as unknown as Page);

describe('ingestion completion requires a successful terminal job', () => {
  it('waits past another job and a running job that already has a Media ID', async () => {
    const page = ingestResponses([
      { id: 10, status: 200, body: { status: 'completed', result: { media_id: 90 } } },
      { id: 1, status: 200, body: { status: 'running', result: { media_id: 18 } } },
      { id: 1, status: 200, body: { status: 'completed', result: { media_id: 17 } } },
    ]);
    expect(await waitForCompletedIngestJob(page, [1], 100)).toBe('17');
  });
  it.each(['failed', 'cancelled'])('rejects a %s job even if it contains a Media ID', async status => {
    const page = ingestResponses([{ id: 1, status: 200, body: { status, result: { media_id: 17 } } }]);
    await expect(waitForCompletedIngestJob(page, [1], 100)).rejects.toThrow();
  });
  it('rejects HTTP failure rather than accepting a success-shaped error body', async () => {
    const page = ingestResponses([{ id: 1, status: 500, body: { status: 'completed', result: { media_id: 17 } } }]);
    await expect(waitForCompletedIngestJob(page, [1], 100)).rejects.toThrow();
  });
  it('propagates a missing terminal response', async () => {
    await expect(waitForCompletedIngestJob(ingestResponses([]), [1], 100)).rejects.toThrow('Timed out');
  });
});
