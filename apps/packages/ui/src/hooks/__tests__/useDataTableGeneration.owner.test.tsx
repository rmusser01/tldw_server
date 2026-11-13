import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDataTableGeneration } from '../useDataTableGeneration';

const api = vi.hoisted(() => ({
  getDataTable: vi.fn(),
  getDataTableJob: vi.fn(),
  cancelDataTableJob: vi.fn(),
}));
const polling = vi.hoisted(() => vi.fn());
vi.mock('@/services/tldw/TldwApiClient', () => ({ tldwClient: api }));
vi.mock('@/utils/data-tables-jobs', () => ({ pollDataTableJob: polling }));
vi.mock('@/services/service-prompts', async (original) => ({
  ...(await original<typeof import('@/services/service-prompts')>()),
  resolveServicePromptScope: vi.fn(async () => ({
    config: { serverUrl: 'http://original-s03.test', authMode: 'multi-user' },
    userId: 42,
    scopeKey: 'original-s03-owner',
    clientPrincipalVerified: true,
  })),
}));
beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);
describe('Data Tables real account-boundary lease', () => {
  it.each(['tldw:auth-principal-changed', 'tldw:auth-credentials-changed', 'tldw:config-updated'])(
    'discards a late result on %s',
    async (event) => {
      let complete!: (value: { status: string; table_uuid: string }) => void;
      polling.mockImplementation(
        () =>
          new Promise((resolve) => {
            complete = resolve;
          })
      );
      const { result } = renderHook(() => useDataTableGeneration());
      const onComplete = vi.fn(),
        onError = vi.fn();
      const start = vi
        .fn()
        .mockResolvedValue({ job_id: 71, table: { uuid: '83569797-3613-42e0-b2a2-498794fc1cd0' } });
      let pending!: Promise<void>;
      act(() => {
        pending = result.current.run(start, 3, { onComplete, onError });
      });
      await waitFor(() => expect(polling).toHaveBeenCalledOnce());
      await act(async () => {
        window.dispatchEvent(new CustomEvent(event, { detail: { authorityChanged: true } }));
        complete({ status: 'completed', table_uuid: '83569797-3613-42e0-b2a2-498794fc1cd0' });
        await pending;
      });
      expect(onComplete).not.toHaveBeenCalled();
      expect(api.getDataTable).not.toHaveBeenCalled();
      expect(onError).not.toHaveBeenCalled();
      expect(result.current.isGenerating).toBe(false);
      expect(start.mock.calls[0][0].requestScope.userId).toBe(42);
    }
  );
});

it('retires a retained polling failure when account authority changes', async () => {
  polling.mockRejectedValue(new Error('Status unavailable'));
  api.cancelDataTableJob.mockResolvedValue({ success: true, status: 'cancelled' });
  const { result } = renderHook(() => useDataTableGeneration());
  const onComplete = vi.fn(),
    onError = vi.fn();
  await act(async () => {
    await result.current.run(
      async () => ({
        job_id: 71,
        status: 'queued',
        table: {
          uuid: '83569797-3613-42e0-b2a2-498794fc1cd0',
          name: 'Rowan',
          prompt: 'Extract facts',
          status: 'queued',
          row_count: 0,
        },
      }),
      3,
      { onComplete, onError }
    );
  });
  expect(onError).toHaveBeenCalledOnce();
  expect(result.current.isGenerating).toBe(true);
  await act(async () => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')));
  expect(result.current.isGenerating).toBe(false);
  expect(onComplete).not.toHaveBeenCalled();
});

it('retires a retained failure during pending Cancel even if DELETE later rejects', async () => {
  polling.mockRejectedValue(new Error('Status unavailable'));
  let rejectDelete!: (error: Error) => void;
  api.cancelDataTableJob.mockImplementation(
    () =>
      new Promise((_resolve, reject) => {
        rejectDelete = reject;
      })
  );
  const { result } = renderHook(() => useDataTableGeneration());
  const onComplete = vi.fn(),
    onError = vi.fn();
  await act(async () => {
    await result.current.run(
      async () => ({
        job_id: 71,
        status: 'queued',
        table: {
          uuid: '83569797-3613-42e0-b2a2-498794fc1cd0',
          name: 'Rowan',
          prompt: 'Extract facts',
          status: 'queued',
          row_count: 0,
        },
      }),
      3,
      { onComplete, onError }
    );
  });
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.cancel(vi.fn());
  });
  await waitFor(() => expect(api.cancelDataTableJob).toHaveBeenCalledOnce());
  await act(async () => {
    window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed'));
    rejectDelete(new Error('Captured account changed'));
    await pending;
  });
  expect(result.current.isGenerating).toBe(false);
  expect(result.current.isCancelling).toBe(false);
  expect(onComplete).not.toHaveBeenCalled();
});

it('aborts the real polling lease on acknowledged Cancel before releasing it', async () => {
  let finishPoll!: (value: { status: string }) => void;
  polling.mockImplementation(
    () =>
      new Promise((resolve) => {
        finishPoll = resolve;
      })
  );
  api.cancelDataTableJob.mockResolvedValue({ success: true, status: 'cancelled' });
  const { result } = renderHook(() => useDataTableGeneration());
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.run(
      async () => ({
        job_id: 71,
        status: 'queued',
        table: {
          uuid: '83569797-3613-42e0-b2a2-498794fc1cd0',
          name: 'Rowan',
          prompt: 'Extract facts',
          status: 'queued',
          row_count: 0,
        },
      }),
      3,
      { onComplete: vi.fn(), onError: vi.fn() }
    );
  });
  await waitFor(() => expect(polling).toHaveBeenCalledOnce());
  const signal: AbortSignal = polling.mock.calls[0][0].signal;
  await act(async () => result.current.cancel(vi.fn()));
  const abortedAtAcknowledgement = signal.aborted;
  await act(async () => {
    finishPoll({ status: 'cancelled' });
    await pending;
  });
  expect(abortedAtAcknowledgement).toBe(true);
  expect(result.current.isGenerating).toBe(false);
});
