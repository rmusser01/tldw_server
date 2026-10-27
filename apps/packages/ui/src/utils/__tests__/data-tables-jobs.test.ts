import { afterEach, describe, expect, it, vi } from 'vitest';
import { pollDataTableJob } from '../data-tables-jobs';
afterEach(() => vi.useRealTimers());
describe('Data Tables polling cancellation', () => {
  it('refuses a late completed fetch after abort', async () => {
    const controller = new AbortController();
    let finish!: (value: { id: number; status: string }) => void;
    const fetchJob = vi.fn(
      () =>
        new Promise<{ id: number; status: string }>((resolve) => {
          finish = resolve;
        })
    );
    const pending = pollDataTableJob({ jobId: 71, fetchJob, signal: controller.signal });
    const rejected = expect(pending).rejects.toThrow('Polling cancelled');
    controller.abort();
    finish({ id: 71, status: 'completed' });
    await rejected;
  });
  it('releases every sleep listener before the next poll', async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const remove = vi.spyOn(controller.signal, 'removeEventListener');
    const fetchJob = vi
      .fn()
      .mockResolvedValueOnce({ id: 71, status: 'processing' })
      .mockResolvedValueOnce({ id: 71, status: 'completed' });
    const pending = pollDataTableJob({
      jobId: 71,
      fetchJob,
      signal: controller.signal,
      intervalMs: 10,
    });
    await vi.advanceTimersByTimeAsync(10);
    expect((await pending).status).toBe('completed');
    expect(remove).toHaveBeenCalledOnce();
  });
});
