import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.hoisted(() => {
  vi.stubEnv('NEXT_PUBLIC_API_URL', 'http://127.0.0.1:8000');
});

const mocks = vi.hoisted(() => ({
  bulkReviewVNAssetItems: vi.fn(),
  getVNAssetGenerationPreflight: vi.fn(),
  retryVNAssetSlot: vi.fn(),
  startVNAssetGeneration: vi.fn(),
  cancelVNAssetGeneration: vi.fn(),
  applyVNAssetMatrix: vi.fn(),
  commitVNPackImport: vi.fn(),
  createVNAssetPack: vi.fn(),
  createVNPackImportPreview: vi.fn(),
  exportVNAssetPack: vi.fn(),
  getStarterMatrices: vi.fn(),
  getVNAssetGeneration: vi.fn(),
  getVNAssetReadiness: vi.fn(),
  getVNPackImportPreview: vi.fn(),
  listVNAssetItems: vi.fn(),
  listVNAssetPacks: vi.fn(),
  listVNAssetSlots: vi.fn(),
}));

vi.mock('@web/lib/api/vnAssets', () => ({
  bulkReviewVNAssetItems: (...args: unknown[]) => mocks.bulkReviewVNAssetItems(...args),
  getVNAssetGenerationPreflight: (...args: unknown[]) => mocks.getVNAssetGenerationPreflight(...args),
  retryVNAssetSlot: (...args: unknown[]) => mocks.retryVNAssetSlot(...args),
  startVNAssetGeneration: (...args: unknown[]) => mocks.startVNAssetGeneration(...args),
  cancelVNAssetGeneration: (...args: unknown[]) => mocks.cancelVNAssetGeneration(...args),
  applyVNAssetMatrix: (...args: unknown[]) => mocks.applyVNAssetMatrix(...args),
  commitVNPackImport: (...args: unknown[]) => mocks.commitVNPackImport(...args),
  createVNAssetPack: (...args: unknown[]) => mocks.createVNAssetPack(...args),
  createVNPackImportPreview: (...args: unknown[]) => mocks.createVNPackImportPreview(...args),
  exportVNAssetPack: (...args: unknown[]) => mocks.exportVNAssetPack(...args),
  getStarterMatrices: (...args: unknown[]) => mocks.getStarterMatrices(...args),
  getVNAssetGeneration: (...args: unknown[]) => mocks.getVNAssetGeneration(...args),
  getVNAssetReadiness: (...args: unknown[]) => mocks.getVNAssetReadiness(...args),
  getVNPackImportPreview: (...args: unknown[]) => mocks.getVNPackImportPreview(...args),
  listVNAssetItems: (...args: unknown[]) => mocks.listVNAssetItems(...args),
  listVNAssetPacks: (...args: unknown[]) => mocks.listVNAssetPacks(...args),
  listVNAssetSlots: (...args: unknown[]) => mocks.listVNAssetSlots(...args),
}));

import VNAssetsWorkbench from '@web/components/vn-assets/VNAssetsWorkbench';
import { ApiError } from '@web/lib/api';
import { readPendingVNAssetGeneration, writePendingVNAssetGeneration } from '@web/lib/vnAssetIdempotency';

describe('VNAssetsWorkbench', () => {
  afterEach(() => vi.restoreAllMocks());

  beforeEach(() => {
    vi.resetAllMocks();
    window.sessionStorage.clear();
    mocks.getVNAssetGenerationPreflight.mockResolvedValue({
      scope: 'api_process_configuration', worker_health: 'unknown',
      local_workers_enabled: true, warnings: [], slots: [],
    });
    mocks.startVNAssetGeneration.mockResolvedValue({ status: 'queued' });
    mocks.retryVNAssetSlot.mockResolvedValue({ status: 'queued' });
    mocks.applyVNAssetMatrix.mockResolvedValue([]);
    mocks.commitVNPackImport.mockResolvedValue({});
    mocks.createVNAssetPack.mockResolvedValue({
      id: 7,
      title: 'Orbital Library',
      primary_character_id: 42,
      planned_output_count: 0,
      status: 'draft',
    });
    mocks.createVNPackImportPreview.mockResolvedValue({});
    mocks.exportVNAssetPack.mockResolvedValue({});
    mocks.getStarterMatrices.mockResolvedValue({
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
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'idle' });
    mocks.getVNAssetReadiness.mockResolvedValue({ ready: false, status: 'not_ready', warnings: [], errors: [] });
    mocks.getVNPackImportPreview.mockResolvedValue({});
    mocks.listVNAssetItems.mockResolvedValue([]);
    mocks.listVNAssetPacks.mockResolvedValue([]);
    mocks.listVNAssetSlots.mockResolvedValue([]);
  });

  function existingFailedPack(): void {
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    mocks.listVNAssetSlots.mockResolvedValue([
      { id: 12, pack_id: 7, asset_type: 'sprite', slot_key: 'sprite_neutral',
        variant_count: 1, status: 'failed', last_error: 'image_backend_unavailable' },
    ]);
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'failed', failed_count: 1 });
  }

  /** Keep an ambiguous operation pending while authoritative status permits cancellation. */
  async function ambiguousCancelableStart(): Promise<{
    user: ReturnType<typeof userEvent.setup>;
    view: ReturnType<typeof render>;
    key: string;
  }> {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
      { id: 8, owner_user_id: 2, title: 'Moon Archive', primary_character_id: 43, status: 'draft' },
    ]);
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Ambiguous start'))
      .mockRejectedValueOnce(new Error('Reconciliation offline'));
    const user = userEvent.setup();
    const view = render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Start generation' }));
    await screen.findByText('Ambiguous start');
    const key = readPendingVNAssetGeneration(1, 7)!.key;
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'queued' });
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await screen.findByText('Reconciliation offline');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled());
    return { user, view, key };
  }

  it.each([false, true])('successful cancellation abandons ambiguous key before next start (reload=%s)', async (reload) => {
    const { user, view, key } = await ambiguousCancelableStart();
    mocks.cancelVNAssetGeneration.mockResolvedValue({ status: 'cancelled' });
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'cancelled' });
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('cancelled'));
    expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
    if (reload) {
      view.unmount();
      render(<VNAssetsWorkbench />);
      await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
      expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2);
    }
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(3));
    expect(mocks.startVNAssetGeneration.mock.calls[2][1].idempotency_key).not.toBe(key);
  });

  it('failed cancellation retains the ambiguous key for reload reconciliation', async () => {
    const { user, view, key } = await ambiguousCancelableStart();
    mocks.cancelVNAssetGeneration.mockRejectedValue(new Error('Cancel offline'));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await screen.findByText('Cancel offline');
    expect(readPendingVNAssetGeneration(1, 7)?.key).toBe(key);
    view.unmount();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(3));
    expect(mocks.startVNAssetGeneration.mock.calls[2]).toEqual([7, { idempotency_key: key }]);
  });

  it('delayed cancellation cannot erase a newer receipt or another owner/pack operation', async () => {
    const { user } = await ambiguousCancelableStart();
    let finishCancel!: (value: unknown) => void;
    mocks.cancelVNAssetGeneration.mockImplementation(() => new Promise((resolve) => { finishCancel = resolve; }));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    writePendingVNAssetGeneration(1, 7, { kind: 'retry', slotId: 12, key: 'newer-receipt' });
    writePendingVNAssetGeneration(2, 7, { kind: 'start', key: 'other-owner' });
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'failed' });
    await user.click(screen.getByText('Moon Archive'));
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('New pack offline'));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('New pack offline');
    const otherPack = readPendingVNAssetGeneration(2, 8);
    await act(async () => { finishCancel({ status: 'cancelled' }); });
    expect(readPendingVNAssetGeneration(1, 7)?.key).toBe('newer-receipt');
    expect(readPendingVNAssetGeneration(2, 7)?.key).toBe('other-owner');
    expect(readPendingVNAssetGeneration(2, 8)).toEqual(otherPack);
    expect(screen.getByRole('status')).toHaveTextContent('failed');
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(4));
    expect(mocks.startVNAssetGeneration.mock.calls[3]).toEqual([8, { idempotency_key: otherPack!.key }]);
  });

  it('cancellation clears the matching memory key even when session storage is unavailable', async () => {
    existingFailedPack();
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Storage disabled'); });
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Memory-only start'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Start generation' }));
    await screen.findByText('Memory-only start');
    const key = mocks.startVNAssetGeneration.mock.calls[0][1].idempotency_key;
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'queued' });
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled());
    mocks.cancelVNAssetGeneration.mockResolvedValue({ status: 'cancelled' });
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2));
    expect(mocks.startVNAssetGeneration.mock.calls[1][1].idempotency_key).not.toBe(key);
  });

  it.each([
    { label: 'zero', slotId: 0 },
    { label: 'negative', slotId: -1 },
    { label: 'negative slot', slotId: -12 },
    { label: 'unsafe positive', slotId: Number.MAX_SAFE_INTEGER + 1 },
    { label: 'unsafe negative', slotId: Number.MIN_SAFE_INTEGER - 1 },
    { label: 'fractional', slotId: 1.5 },
    { label: 'numeric string', slotId: '12' },
    { label: 'null', slotId: null },
    { label: 'missing', slotId: undefined },
    { label: 'boolean', slotId: true },
    { label: 'object', slotId: {} },
    { label: 'array', slotId: [] },
  ])('does not call the retry API on reload for a persisted $label slot ID', async ({ slotId }) => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    const storageKey = 'vn-assets:pending-generation:v1:1:7';
    window.sessionStorage.setItem(storageKey, JSON.stringify({ kind: 'retry', slotId, key: 'invalid-retry-key' }));

    const first = render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem(storageKey)).toBeNull();
    first.unmount();

    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
  });

  it('keeps the retry key in memory after connection loss when storage is disabled', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('Storage disabled', 'SecurityError');
    });
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));
    await screen.findByText('Connection lost');
    const originalRequest = mocks.retryVNAssetSlot.mock.calls[0][2];

    await user.click(screen.getByRole('button', { name: 'Retry sprite_neutral' }));
    await waitFor(() => expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(2));
    expect(originalRequest.idempotency_key).toEqual(expect.any(String));
    expect(mocks.retryVNAssetSlot.mock.calls[1]).toEqual([7, 12, originalRequest]);
  });

  it.each(['Start', 'other Retry'])('abandons a missing-slot retry so %s works and reload does not replay it', async (nextAction) => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    mocks.listVNAssetSlots.mockResolvedValue([
      { id: 12, pack_id: 7, asset_type: 'sprite', slot_key: 'sprite_neutral', variant_count: 1, status: 'failed' },
      { id: 13, pack_id: 7, asset_type: 'sprite', slot_key: 'sprite_happy', variant_count: 1, status: 'failed' },
    ]);
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new ApiError('slot_not_found', { status: 404, detail: 'slot_not_found' }));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));
    await screen.findByText('slot_not_found');
    expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(1);
    first.unmount();

    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole('button', { name: nextAction === 'Start' ? 'Start generation' : 'Retry sprite_happy' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('queued'));
    if (nextAction === 'Start') expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
    else expect(mocks.retryVNAssetSlot).toHaveBeenLastCalledWith(7, 13, expect.objectContaining({ idempotency_key: expect.any(String) }));
  });

  it('abandons a restored missing-slot receipt after one reload reconciliation', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    writePendingVNAssetGeneration(1, 7, { kind: 'retry', slotId: 12, key: 'missing-slot-receipt' });
    mocks.retryVNAssetSlot.mockRejectedValue(new ApiError('slot_not_found', { status: 404, detail: 'slot_not_found' }));
    const first = render(<VNAssetsWorkbench />);
    await screen.findByText('slot_not_found');
    await waitFor(() => expect(readPendingVNAssetGeneration(1, 7)).toBeNull());
    first.unmount();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(1);
  });

  it.each([408, 429, 500, 503])('preserves a retry receipt after ambiguous HTTP %s', async (status) => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new ApiError('Try again', { status }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));
    await screen.findByText('Try again');
    const pending = readPendingVNAssetGeneration(1, 7);
    expect(pending).not.toBeNull();
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await waitFor(() => expect(readPendingVNAssetGeneration(1, 7)).toBeNull());
    expect(mocks.retryVNAssetSlot).toHaveBeenLastCalledWith(7, 12, { idempotency_key: pending!.key });
  });

  it('reuses the required start key after an ambiguous transport failure', async () => {
    existingFailedPack();
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2));
    const request = mocks.startVNAssetGeneration.mock.calls[0][1];
    expect(request.idempotency_key).toEqual(expect.any(String));
    expect(request.idempotency_key.length).toBeGreaterThan(0);
    expect(mocks.startVNAssetGeneration.mock.calls[1]).toEqual([7, request]);
  });

  it('replays an ambiguous start with the same key after reload', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    mocks.getVNAssetGeneration.mockImplementation(async () => ({
      status: mocks.startVNAssetGeneration.mock.calls.length > 1 ? 'queued' : 'failed',
    }));
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    const originalRequest = mocks.startVNAssetGeneration.mock.calls[0][1];
    first.unmount();

    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2));
    expect(mocks.startVNAssetGeneration.mock.calls[1]).toEqual([7, originalRequest]);
    await waitFor(() => expect(readPendingVNAssetGeneration(1, 7)).toBeNull());
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('queued'));
  });

  it('replays an ambiguous slot retry after reload', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));
    await screen.findByText('Connection lost');
    const originalRequest = mocks.retryVNAssetSlot.mock.calls[0][2];
    first.unmount();

    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(2));
    expect(mocks.retryVNAssetSlot.mock.calls[1]).toEqual([7, 12, originalRequest]);
    await waitFor(() => expect(readPendingVNAssetGeneration(1, 7)).toBeNull());
  });

  it("does not replay another owner's pending generation after reload", async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
    ]);
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    first.unmount();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 2, title: 'Other Library', primary_character_id: 42, status: 'draft' },
    ]);

    render(<VNAssetsWorkbench />);
    await screen.findByText('Other Library');
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
  });

  it('restores the selected pack and replays its pending generation after reload', async () => {
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, owner_user_id: 1, title: 'First Pack', primary_character_id: 42, status: 'draft' },
      { id: 8, owner_user_id: 1, title: 'Second Pack', primary_character_id: 43, status: 'draft' },
    ]);
    mocks.listVNAssetSlots.mockImplementation(async (packId: number) => packId === 8
      ? [{ id: 12, pack_id: 8, asset_type: 'sprite', slot_key: 'sprite_neutral', variant_count: 1, status: 'failed' }]
      : []);
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'failed' });
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await user.click(await screen.findByText('Second Pack'));
    const start = await screen.findByRole('button', { name: 'Start generation' });
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);
    await screen.findByText('Connection lost');
    const originalRequest = mocks.startVNAssetGeneration.mock.calls[0][1];
    first.unmount();

    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2));
    expect(mocks.startVNAssetGeneration.mock.calls[1]).toEqual([8, originalRequest]);
  });

  it('retries only the failed slot and reuses its key after connection loss', async () => {
    existingFailedPack();
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new Error('Retry connection lost'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));
    await screen.findByText('Retry connection lost');
    await user.click(screen.getByRole('button', { name: 'Retry sprite_neutral' }));
    await waitFor(() => expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(2));
    const request = mocks.retryVNAssetSlot.mock.calls[0][2];
    expect(request.idempotency_key).toEqual(expect.any(String));
    expect(mocks.retryVNAssetSlot.mock.calls[1]).toEqual([7, 12, request]);
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Retry sprite_neutral' })).toBeDisabled();
  });

  it('recovers an initial status failure through Refresh without starting work', async () => {
    existingFailedPack();
    mocks.getVNAssetGeneration.mockRejectedValueOnce(new Error('Offline'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await screen.findByText('Could not load generation status. Refresh to try again.');
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await screen.findByRole('button', { name: 'Retry sprite_neutral' });
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
  });

  it('blocks generation commands while a start request is unresolved', async () => {
    existingFailedPack();
    let resolveStart!: (value: unknown) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { resolveStart = resolve; }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    const start = screen.getByRole('button', { name: 'Start generation' });
    await waitFor(() => expect(start).toBeEnabled());
    await user.dblClick(start);
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Retry sprite_neutral' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Refresh generation status' })).toBeDisabled();
    await act(async () => { resolveStart({ status: 'queued' }); });
  });

  it('allows another pack to start without an old pending command clearing its lock', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
      { id: 8, title: 'Moon Archive', primary_character_id: 43, status: 'draft' },
    ]);
    let resolveFirst!: (value: unknown) => void;
    let resolveSecond!: (value: unknown) => void;
    mocks.startVNAssetGeneration
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveSecond = resolve; }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    const start = screen.getByRole('button', { name: 'Start generation' });
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);
    await user.click(screen.getByText('Moon Archive'));
    await waitFor(() => expect(start).toBeEnabled());
    await user.click(start);
    await act(async () => { resolveFirst({ status: 'queued' }); });
    expect(start).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Refresh generation status' })).toBeDisabled();
    expect(mocks.startVNAssetGeneration.mock.calls.map(([id]) => id)).toEqual([7, 8]);
    await act(async () => { resolveSecond({ status: 'queued' }); });
  });

  it('refreshes preflight after applying a matrix to the selected pack', async () => {
    existingFailedPack();
    mocks.getVNAssetGenerationPreflight
      .mockResolvedValueOnce({ local_workers_enabled: false, warnings: ['Old configuration'], slots: [] })
      .mockResolvedValueOnce({ local_workers_enabled: false, warnings: ['Fresh configuration'], slots: [] });
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await screen.findByText('Old configuration');
    await user.click(screen.getByRole('button', { name: 'Apply starter matrix' }));
    await screen.findByText('Fresh configuration');
    expect(screen.queryByText('Old configuration')).not.toBeInTheDocument();
    expect(mocks.getVNAssetGenerationPreflight).toHaveBeenCalledTimes(2);
  });

  it('loads final items after observing terminal generation status', async () => {
    existingFailedPack();
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await screen.findByRole('button', { name: 'Retry sprite_neutral' });
    let committed = false;
    mocks.getVNAssetGeneration.mockImplementationOnce(async () => {
      committed = true;
      return { status: 'completed' };
    });
    mocks.listVNAssetSlots.mockImplementationOnce(async () => committed ? [] : [
      { id: 12, slot_key: 'sprite_neutral', status: 'failed' },
    ]);
    mocks.listVNAssetItems.mockImplementationOnce(async () => committed ? [
      { id: 11, slot_id: 12, variant_index: 0, review_status: 'draft', source: 'generated' },
    ] : []);
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('completed'));
    expect(screen.queryByRole('button', { name: 'Retry sprite_neutral' })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Select item 11' })).toBeInTheDocument();
  });

  it('does not overlap refreshes while another detail request is still pending', async () => {
    existingFailedPack();
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await screen.findByRole('button', { name: 'Retry sprite_neutral' });
    mocks.listVNAssetSlots.mockRejectedValueOnce(new Error('Offline'));
    let resolveItems!: (value: unknown[]) => void;
    mocks.listVNAssetItems.mockImplementationOnce(() => new Promise((resolve) => { resolveItems = resolve; }));
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await waitFor(() => expect(mocks.listVNAssetItems).toHaveBeenCalledTimes(2));
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    expect(mocks.listVNAssetItems).toHaveBeenCalledTimes(2);
    await act(async () => { resolveItems([]); });
    await screen.findByText('Could not refresh generation progress. Refresh to try again.');
  });

  it('does not let an old review refresh invalidate the selected pack load', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
      { id: 8, title: 'Moon Archive', primary_character_id: 43, status: 'draft' },
    ]);
    mocks.listVNAssetItems.mockResolvedValueOnce([
      { id: 11, pack_id: 7, slot_id: 12, variant_index: 0, review_status: 'draft', source: 'generated' },
    ]);
    let resolveReview!: (value: unknown[]) => void;
    let resolveGeneration!: (value: unknown) => void;
    mocks.bulkReviewVNAssetItems.mockImplementationOnce(() => new Promise((resolve) => { resolveReview = resolve; }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('checkbox', { name: 'Select item 11' }));
    await user.click(screen.getByRole('button', { name: 'Approve selected' }));
    mocks.getVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { resolveGeneration = resolve; }));
    await user.click(screen.getByText('Moon Archive'));
    await waitFor(() => expect(mocks.getVNAssetGeneration).toHaveBeenCalledWith(8));
    await act(async () => { resolveReview([]); });
    await act(async () => { resolveGeneration({ status: 'idle' }); });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
  });

  it('keeps a successful review when an earlier refresh finishes afterward', async () => {
    existingFailedPack();
    const draft = { id: 11, pack_id: 7, slot_id: 12, variant_index: 0, review_status: 'draft', source: 'generated' };
    const approved = { ...draft, review_status: 'approved' };
    mocks.listVNAssetItems.mockResolvedValue([draft]);
    mocks.bulkReviewVNAssetItems.mockResolvedValue([approved]);
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('checkbox', { name: 'Select item 11' }));
    let resolveReadiness!: (value: unknown) => void;
    mocks.getVNAssetReadiness.mockImplementationOnce(() => new Promise((resolve) => { resolveReadiness = resolve; }));
    await user.click(screen.getByRole('button', { name: 'Refresh generation status' }));
    await waitFor(() => expect(mocks.listVNAssetItems).toHaveBeenCalledTimes(2));
    mocks.listVNAssetItems.mockResolvedValue([approved]);
    await user.click(screen.getByRole('button', { name: 'Approve selected' }));
    await screen.findByText('approved');
    await act(async () => { resolveReadiness({ ready: false, status: 'not_ready', warnings: [], errors: [] }); });
    await waitFor(() => expect(mocks.listVNAssetItems).toHaveBeenCalledTimes(3));
    expect(screen.getByText('approved')).toBeInTheDocument();
  });

  it('clears old items and review selection when the next pack cannot load', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
      { id: 8, title: 'Moon Archive', primary_character_id: 43, status: 'draft' },
    ]);
    mocks.listVNAssetItems.mockResolvedValueOnce([
      { id: 11, pack_id: 7, slot_id: 12, variant_index: 0, review_status: 'draft', source: 'generated' },
    ]).mockRejectedValueOnce(new Error('Offline'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('checkbox', { name: 'Select item 11' }));
    await user.click(screen.getByText('Moon Archive'));
    await screen.findByText('Could not load generation status. Refresh to try again.');
    expect(screen.queryByRole('checkbox', { name: 'Select item 11' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Approve selected' })).toBeDisabled();
  });

  it('ignores a late generation response after switching to another pack', async () => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
      { id: 8, title: 'Moon Archive', primary_character_id: 43, status: 'draft' },
    ]);
    let resolveStart!: (value: unknown) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { resolveStart = resolve; }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await user.click(screen.getByText('Moon Archive'));
    await waitFor(() => expect(mocks.getVNAssetGeneration).toHaveBeenCalledWith(8));
    await act(async () => { resolveStart({ status: 'queued' }); });
    expect(screen.getByRole('status')).toHaveTextContent('failed');
  });

  it('renders loading, empty, setup, matrix preview, and placeholders', async () => {
    render(<VNAssetsWorkbench />);

    expect(screen.getByText('Loading VN asset packs...')).toBeInTheDocument();
    expect(await screen.findByText('No asset packs yet.')).toBeInTheDocument();
    expect(screen.getByLabelText('Pack title')).toBeInTheDocument();
    expect(screen.getByLabelText('Primary character ID')).toBeInTheDocument();
    expect(screen.getByText('Starter matrix')).toBeInTheDocument();
    expect(screen.getByText('24 planned assets')).toBeInTheDocument();
    expect(screen.getByText('Generation monitor')).toBeInTheDocument();
    expect(screen.getByText('Review board')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Portability' })).toBeInTheDocument();
  });

  it('creates a pack and selects it for the workbench summary', async () => {
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);

    await screen.findByText('No asset packs yet.');
    await user.clear(screen.getByLabelText('Pack title'));
    await user.type(screen.getByLabelText('Pack title'), 'Orbital Library');
    await user.clear(screen.getByLabelText('Primary character ID'));
    await user.type(screen.getByLabelText('Primary character ID'), '42');
    await user.click(screen.getByRole('button', { name: 'Create pack' }));

    await waitFor(() => {
      expect(mocks.createVNAssetPack).toHaveBeenCalledWith({
        title: 'Orbital Library',
        primary_character_id: 42,
        apply_starter_matrix: false,
      });
    });
    expect(await screen.findByText('Orbital Library')).toBeInTheDocument();
    expect(screen.getAllByText('Character 42').length).toBeGreaterThan(0);
    expect(screen.getByText('0 planned assets')).toBeInTheDocument();
  });

  it('uses returned slot variants for the planned asset count after matrix apply', async () => {
    const user = userEvent.setup();
    mocks.applyVNAssetMatrix.mockResolvedValue([
      {
        id: 1,
        pack_id: 7,
        asset_type: 'sprite',
        slot_key: 'sprite.primary',
        variant_count: 2,
        status: 'planned',
      },
      {
        id: 2,
        pack_id: 7,
        asset_type: 'background',
        slot_key: 'background.interior',
        variant_count: 3,
        status: 'planned',
      },
    ]);
    render(<VNAssetsWorkbench />);

    await screen.findByText('No asset packs yet.');
    await user.click(screen.getByRole('button', { name: 'Create pack' }));
    await user.click(await screen.findByRole('button', { name: 'Apply starter matrix' }));

    await waitFor(() => {
      expect(mocks.applyVNAssetMatrix).toHaveBeenCalledWith(7, 'starter', { variant_count: 1 });
    });
    expect(await screen.findByText('5 planned assets')).toBeInTheDocument();
  });
});
