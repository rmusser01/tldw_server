import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, configure, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mocks = vi.hoisted(() => ({
  profile: vi.fn(),
  apiBaseUrl: vi.fn(),
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

vi.mock('@web/lib/api', () => ({
  apiClient: { get: (...args: unknown[]) => mocks.profile(...args) },
  getApiBaseUrl: () => mocks.apiBaseUrl(),
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
configure({ asyncUtilTimeout: 5000 });

describe('VNAssetsWorkbench', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.resetAllMocks();
    sessionStorage.clear();
    mocks.profile.mockResolvedValue({ user: { id: 1, is_active: true } });
    mocks.apiBaseUrl.mockReturnValue('http://localhost:8000/api/v1');
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
    mocks.getVNAssetGeneration.mockResolvedValue({
      batch_id: 41, status: 'failed', failed_count: 1,
      selected_slot_ids: [12], failed_slot_batch_ids: { 12: 41 },
    });
  }

  it('restores an ambiguous Start after remount without automatically posting', async () => {
    existingFailedPack();
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    const request = { ...mocks.startVNAssetGeneration.mock.calls[0][1] };
    first.unmount();
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'processing', batch_id: 41 });
    render(<VNAssetsWorkbench />);
    const recover = await screen.findByRole('button', { name: 'Recover pending request' });
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
    await user.click(recover);
    await waitFor(() => expect(mocks.startVNAssetGeneration.mock.calls[1]).toEqual([7, request]));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Recover pending request' })).not.toBeInTheDocument());
    expect(sessionStorage.length).toBe(0);
  });

  it('recovers the original Retry source after reload advertises a newer failure', async () => {
    existingFailedPack();
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new Error('Retry connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    const retry = await screen.findByRole('button', { name: 'Retry sprite_neutral' });
    await waitFor(() => expect(retry).toBeEnabled());
    await user.click(retry);
    await screen.findByText('Retry connection lost');
    const request = { ...mocks.retryVNAssetSlot.mock.calls[0][2] };
    first.unmount();
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'failed', batch_id: 42, failed_slot_batch_ids: { 12: 42 } });
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(mocks.retryVNAssetSlot.mock.calls[1]).toEqual([7, 12, request]));
    expect(request.source_batch_id).toBe(41);
  });

  it('reuses the required start key after an ambiguous transport failure', async () => {
    existingFailedPack();
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2));
    const request = mocks.startVNAssetGeneration.mock.calls[0][1];
    expect(request.idempotency_key).toEqual(expect.any(String));
    expect(request.idempotency_key.length).toBeGreaterThan(0);
    expect(mocks.startVNAssetGeneration.mock.calls[1]).toEqual([7, request]);
  });

  it('does not replay an old command when pre-send verification discovers a different account', async () => {
    existingFailedPack();
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    mocks.profile.mockResolvedValue({ user: { id: 2, is_active: true } });
    await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Recover pending request' })).not.toBeInTheDocument());
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
    expect(sessionStorage.length).toBe(0);
  });

  it.each(['success', 'error'])('reports an invalid server setting when an in-flight request returns %s', async (outcome) => {
    existingFailedPack();
    let resolve!: (value: unknown) => void;
    let reject!: (error: Error) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((accept, fail) => {
      resolve = accept;
      reject = fail;
    }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1));
    const saved = sessionStorage.getItem('tldw:vn-generation:pending:v1');
    const request = mocks.startVNAssetGeneration.mock.calls[0][1];
    mocks.apiBaseUrl.mockReturnValue('ftp://invalid-server.example/api/v1');
    await act(async () => {
      if (outcome === 'success') resolve({ status: 'queued', batch_id: 41 });
      else reject(new Error('Response lost'));
    });
    expect(screen.getByText('Current server and account could not be verified.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(screen.getByLabelText('Generation status')).not.toHaveTextContent('queued');
    expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBe(saved);
    mocks.apiBaseUrl.mockReturnValue('http://localhost:8000/api/v1');
    await user.click(screen.getByRole('button', { name: 'Retry recovery check' }));
    await user.click(await screen.findByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration.mock.calls[1]).toEqual([7, request]));
    await waitFor(() => expect(sessionStorage.length).toBe(0));
  });

  it.each(['start', 'retry'])('retains a new %s command after an unclassified HTTP 400 admission failure', async (kind) => {
    existingFailedPack();
    const send = kind === 'start' ? mocks.startVNAssetGeneration : mocks.retryVNAssetSlot;
    send.mockRejectedValueOnce(Object.assign(new Error('Job admission failed'), { status: 400 }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    const button = await screen.findByRole('button', { name: kind === 'start' ? 'Start generation' : 'Retry sprite_neutral' });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    await screen.findByText('Job admission failed');
    expect(sessionStorage.length).toBe(1);
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    const original = send.mock.calls[0];
    await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(send.mock.calls[1]).toEqual(original));
    await waitFor(() => expect(sessionStorage.length).toBe(0));
  });

  it.each(['success', 'error'])('fences concurrent pack responses after restoring an invalid server setting: %s', async (outcome) => {
    existingFailedPack();
    mocks.listVNAssetPacks.mockResolvedValue([
      { id: 7, title: 'Orbital Library', primary_character_id: 42, status: 'draft' },
      { id: 8, title: 'Moon Archive', primary_character_id: 43, status: 'draft' },
    ]);
    let firstResponse!: (value: unknown) => void;
    let secondResponse!: (value: unknown) => void;
    let secondError!: (error: Error) => void;
    mocks.startVNAssetGeneration
      .mockImplementationOnce(() => new Promise((resolve) => { firstResponse = resolve; }))
      .mockImplementationOnce(() => new Promise((resolve, reject) => {
        secondResponse = resolve;
        secondError = reject;
      }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1));
    await user.click(screen.getByText('Moon Archive'));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2));
    const saved = sessionStorage.getItem('tldw:vn-generation:pending:v1');
    const originalRequests = mocks.startVNAssetGeneration.mock.calls.map((call) => [...call]);
    mocks.apiBaseUrl.mockReturnValue('ftp://invalid-server.example/api/v1');
    await act(async () => firstResponse({ status: 'queued', batch_id: 41 }));
    expect(screen.getByText('Current server and account could not be verified.')).toBeInTheDocument();
    mocks.apiBaseUrl.mockReturnValue('http://localhost:8000/api/v1');
    await user.click(screen.getByRole('button', { name: 'Retry recovery check' }));
    await waitFor(() => expect(screen.queryByText('Current server and account could not be verified.')).not.toBeInTheDocument());
    await act(async () => {
      if (outcome === 'success') secondResponse({ status: 'queued', batch_id: 42 });
      else secondError(Object.assign(new Error('Old command rejected'), { status: 422 }));
    });
    expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBe(saved);
    expect(screen.getByLabelText('Generation status')).not.toHaveTextContent('queued');
    expect(screen.queryByText('Old command rejected')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(2);
    await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration.mock.calls[2]).toEqual(originalRequests[1]));
    await user.click(screen.getByText('Orbital Library'));
    const recover = await screen.findByRole('button', { name: 'Recover pending request' });
    await waitFor(() => expect(recover).toBeEnabled());
    await user.click(recover);
    await waitFor(() => expect(mocks.startVNAssetGeneration.mock.calls[3]).toEqual(originalRequests[0]));
    await waitFor(() => expect(sessionStorage.length).toBe(0));
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
    expect(mocks.cancelVNAssetGeneration).not.toHaveBeenCalled();
  });

  it('does not report an invalid server setting from an already stale account response', async () => {
    existingFailedPack();
    let response!: (value: unknown) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { response = resolve; }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1));
    mocks.profile.mockResolvedValue({ user: { id: 2, is_active: true } });
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed', { detail: { kind: 'switch' } })));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    mocks.apiBaseUrl.mockReturnValue('ftp://invalid-server.example/api/v1');
    await act(async () => response({ status: 'queued', batch_id: 41 }));
    expect(screen.queryByText('Current server and account could not be verified.')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Generation status')).not.toHaveTextContent('queued');
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
  });

  it.each(['start', 'retry'].flatMap((kind) => [400, 403, 404, 409, 422].map((status) => ({ kind, status }))))(
    'retains an ambiguous $kind request when replay returns HTTP $status', async ({ kind, status }) => {
      existingFailedPack();
      const send = kind === 'start' ? mocks.startVNAssetGeneration : mocks.retryVNAssetSlot;
      const message = status === 403 ? 'CSRF validation failed. Refresh the page and try again.' : 'Replay could not be admitted';
      send.mockRejectedValueOnce(new Error('Original response lost'))
        .mockRejectedValueOnce(Object.assign(new Error(message), {
          name: 'ApiError', status, detail: message,
          ...(status === 409 ? { errorCode: 'vn_asset_retry_source_unavailable' } : {}),
        }));
      const user = userEvent.setup();
      render(<VNAssetsWorkbench />);
      const button = await screen.findByRole('button', { name: kind === 'start' ? 'Start generation' : 'Retry sprite_neutral' });
      await waitFor(() => expect(button).toBeEnabled());
      await user.click(button);
      await screen.findByText('Original response lost');
      const original = send.mock.calls[0];
      const saved = sessionStorage.getItem('tldw:vn-generation:pending:v1');
      await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
      await screen.findByText(message);
      expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBe(saved);
      expect(send.mock.calls[1]).toEqual(original);
      expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Retry sprite_neutral' })).toBeDisabled();
      await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
      await waitFor(() => expect(send.mock.calls[2]).toEqual(original));
      await waitFor(() => expect(sessionStorage.length).toBe(0));
    },
  );

  it('keeps a newer account command locked when an old profile check completes', async () => {
    existingFailedPack();
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    let oldProfile!: (value: unknown) => void;
    mocks.profile.mockImplementationOnce(() => new Promise((resolve) => { oldProfile = resolve; }));
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    mocks.profile.mockResolvedValue({ user: { id: 2, is_active: true } });
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed', { detail: { kind: 'switch' } })));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    let newResponse!: (value: unknown) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { newResponse = resolve; }));
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1));
    await act(async () => oldProfile({ user: { id: 1, is_active: true } }));
    expect(screen.getByRole('button', { name: 'Recover pending request' })).toBeDisabled();
    await act(async () => newResponse({ status: 'queued' }));
  });

  it('writes the complete Retry request before the network can accept it', async () => {
    existingFailedPack();
    const user = userEvent.setup();
    mocks.retryVNAssetSlot.mockImplementationOnce(async (packId, slotId, request) => {
      const saved = JSON.parse(sessionStorage.getItem('tldw:vn-generation:pending:v1')!);
      expect(saved.scope).toEqual({ server: 'http://localhost:8000/api/v1', principal: '1' });
      expect(saved.commands).toEqual([{ packId, slotId, request }]);
      return { status: 'queued' };
    });
    render(<VNAssetsWorkbench />);
    const retry = await screen.findByRole('button', { name: 'Retry sprite_neutral' });
    await waitFor(() => expect(retry).toBeEnabled());
    await user.click(retry);
    await waitFor(() => expect(screen.getByLabelText('Generation status')).toHaveTextContent('queued'));
    expect(sessionStorage.length).toBe(0);
  });

  it.each(['account', 'server'])('does not restore an unresolved command on another %s', async (change) => {
    existingFailedPack();
    mocks.startVNAssetGeneration.mockRejectedValueOnce(new Error('Connection lost'));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('Connection lost');
    first.unmount();
    if (change === 'account') mocks.profile.mockResolvedValue({ user: { id: 2, is_active: true } });
    else mocks.apiBaseUrl.mockReturnValue('http://other-server:8000/api/v1');
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(screen.queryByRole('button', { name: 'Recover pending request' })).not.toBeInTheDocument();
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
    expect(sessionStorage.length).toBe(0);
  });

  it('clears on logout and ignores a late accepted response', async () => {
    existingFailedPack();
    let response!: (value: unknown) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { response = resolve; }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1));
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed', { detail: { kind: 'logout' } })));
    await screen.findByText('Sign in again to send generation requests.');
    await act(async () => response({ status: 'queued', batch_id: 41 }));
    expect(screen.getByLabelText('Generation status')).not.toHaveTextContent('queued');
    expect(screen.queryByRole('button', { name: 'Recover pending request' })).not.toBeInTheDocument();
    expect(sessionStorage.length).toBe(0);
  });

  it('leaves a request recoverable when its response arrives after unmount', async () => {
    existingFailedPack();
    let response!: (value: unknown) => void;
    mocks.startVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { response = resolve; }));
    const user = userEvent.setup();
    const first = render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1));
    const before = sessionStorage.getItem('tldw:vn-generation:pending:v1');
    first.unmount();
    await act(async () => response({ status: 'queued' }));
    expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBe(before);
    render(<VNAssetsWorkbench />);
    await screen.findByRole('button', { name: 'Recover pending request' });
    expect(mocks.startVNAssetGeneration).toHaveBeenCalledTimes(1);
  });

  it('does not send when session storage refuses the pre-send write', async () => {
    existingFailedPack();
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    const spy = vi.spyOn(Object.getPrototypeOf(sessionStorage), 'setItem').mockImplementation(() => { throw new Error('quota'); });
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText(/Recovery storage is unavailable/);
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    spy.mockRestore();
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(screen.getByLabelText('Generation status')).toHaveTextContent('queued'));
  });

  it.each(['{', ''])('guards unreadable recovery until explicit warned discard: %j', async (raw) => {
    existingFailedPack();
    sessionStorage.setItem('tldw:vn-generation:pending:v1', raw);
    const user = userEvent.setup();
    await act(async () => { render(<VNAssetsWorkbench />); });
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(screen.getByText(/Saved generation requests could not be read/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Discard unreadable requests' })).toBeDisabled();
    expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBe(raw);
    await user.click(screen.getByRole('checkbox', { name: /I checked server status/ }));
    await user.click(screen.getByRole('button', { name: 'Discard unreadable requests' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
  });

  it.each([
    { server: 'http://[', principal: '1' },
    { server: 'not-a-url', principal: '1' },
    { server: 'http://localhost:8000/api/v1', principal: 'x' },
    { server: 'http://localhost:8000/api/v1/', principal: '1' },
    { server: 'http://LOCALHOST:8000/api/v1', principal: '1' },
    { server: 'http://localhost:8000/api/v1?token=secret', principal: '1' },
    { server: 'http://localhost:8000/api/v1', principal: '01' },
    { server: 'http://localhost:8000/api/v1', principal: '1.0' },
  ])('locks a corrupt saved scope until explicit warned discard: %j', async (scope) => {
    existingFailedPack();
    const key = 'tldw:vn-generation:pending:v1';
    const raw = JSON.stringify({ version: 1, scope, commands: [
      { packId: 7, request: { idempotency_key: 'vn-generation-saved-key' } },
    ] });
    sessionStorage.setItem(key, raw);
    const user = userEvent.setup();
    await act(async () => { render(<VNAssetsWorkbench />); });
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Retry sprite_neutral' })).not.toBeInTheDocument();
    expect(screen.getByText(/Saved generation requests could not be read/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Discard unreadable requests' })).toBeDisabled();
    expect(sessionStorage.getItem(key)).toBe(raw);
    await user.click(screen.getByRole('checkbox', { name: /I checked server status/ }));
    await user.click(screen.getByRole('button', { name: 'Discard unreadable requests' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(sessionStorage.getItem(key)).toBeNull();
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
    expect(mocks.cancelVNAssetGeneration).not.toHaveBeenCalled();
  });

  it('keeps warned discard available after storage refuses to remove unreadable requests', async () => {
    existingFailedPack();
    sessionStorage.setItem('tldw:vn-generation:pending:v1', '{');
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await screen.findByText(/Saved generation requests could not be read/);
    const spy = vi.spyOn(Object.getPrototypeOf(sessionStorage), 'removeItem').mockImplementation(() => { throw new Error('denied'); });
    await user.click(screen.getByRole('checkbox', { name: /I checked server status/ }));
    await user.click(screen.getByRole('button', { name: 'Discard unreadable requests' }));
    await screen.findByText(/Recovery storage is unavailable/);
    expect(screen.getByRole('checkbox', { name: /I checked server status/ })).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Discard unreadable requests' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBe('{');
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
    expect(mocks.cancelVNAssetGeneration).not.toHaveBeenCalled();
    spy.mockRestore();
    await user.click(screen.getByRole('checkbox', { name: /I checked server status/ }));
    await user.click(screen.getByRole('button', { name: 'Discard unreadable requests' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(sessionStorage.getItem('tldw:vn-generation:pending:v1')).toBeNull();
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
    expect(mocks.cancelVNAssetGeneration).not.toHaveBeenCalled();
  });

  it('does not trust cached identity when profile verification fails', async () => {
    existingFailedPack();
    mocks.profile.mockRejectedValue(new Error('Current server and account could not be verified.'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await screen.findByText('Current server and account could not be verified.');
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    mocks.profile.mockResolvedValue({ user: { id: 1, is_active: true } });
    await user.click(screen.getByRole('button', { name: 'Retry recovery check' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
  });

  it.each([404, 410])('verifies a legacy single-user principal after profile HTTP %s', async (status) => {
    existingFailedPack();
    mocks.profile.mockImplementation(async (path) => {
      if (path === '/users/me/profile') throw Object.assign(new Error('User not found'), { status });
      if (path === '/auth/me') return { id: 1, is_active: true };
      throw new Error('Unexpected identity endpoint');
    });
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await waitFor(() => expect(screen.getByLabelText('Generation status')).toHaveTextContent('queued'));
    expect(mocks.profile.mock.calls.map(([path]) => path)).toEqual(['/users/me/profile', '/auth/me', '/users/me/profile', '/auth/me']);
  });

  it.each(['profile', 'legacy'].flatMap((endpoint) =>
    ['false', 1, {}, [], false, undefined].map((isActive) => ({ endpoint, isActive })),
  ))('requires boolean active status from $endpoint identity: $isActive', async ({ endpoint, isActive }) => {
    existingFailedPack();
    mocks.profile.mockImplementation(async (path) => {
      if (path === '/users/me/profile') {
        if (endpoint === 'legacy') throw Object.assign(new Error('User not found'), { status: 404 });
        return { user: { id: 1, is_active: isActive } };
      }
      return { id: 1, is_active: isActive };
    });
    const user = userEvent.setup();
    await act(async () => { render(<VNAssetsWorkbench />); });
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(screen.getByText('Current server and account could not be verified.')).toBeInTheDocument();
    mocks.profile.mockResolvedValue({ user: { id: 1, is_active: true } });
    await user.click(screen.getByRole('button', { name: 'Retry recovery check' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(mocks.retryVNAssetSlot).not.toHaveBeenCalled();
    expect(mocks.cancelVNAssetGeneration).not.toHaveBeenCalled();
  });

  it.each([401, 403, 429, 500])('does not bypass a profile HTTP %s failure through the legacy identity endpoint', async (status) => {
    existingFailedPack();
    mocks.profile.mockImplementation(async (path) => {
      if (path === '/users/me/profile') throw Object.assign(new Error('Identity verification failed'), { status });
      if (path === '/auth/me') return { id: 1, is_active: true };
      throw new Error('Unexpected identity endpoint');
    });
    await act(async () => { render(<VNAssetsWorkbench />); });
    expect(screen.getByText('Identity verification failed')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    expect(mocks.profile.mock.calls.map(([path]) => path)).toEqual(['/users/me/profile']);
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
  });

  it.each([403, 409, 500])('handles an HTTP %s response without losing an ambiguous operation', async (status) => {
    existingFailedPack();
    mocks.startVNAssetGeneration.mockRejectedValueOnce(Object.assign(new Error('Request failed'), { status }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText('Request failed');
    expect(sessionStorage.length).toBe(status === 403 ? 0 : 1);
    if (status !== 403) expect(screen.getByRole('button', { name: 'Recover pending request' })).toBeEnabled();
  });

  it.each(['vn_asset_execution_recipe_invalid', 'vn_asset_retry_source_unavailable'])('unlocks after explicit pre-admission rejection %s', async (errorCode) => {
    existingFailedPack();
    mocks.retryVNAssetSlot.mockRejectedValueOnce(Object.assign(new Error('Original settings unavailable'), { status: 409, errorCode }));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    const retry = await screen.findByRole('button', { name: 'Retry sprite_neutral' });
    await waitFor(() => expect(retry).toBeEnabled());
    await user.click(retry);
    await screen.findByText('Original settings unavailable');
    expect(sessionStorage.length).toBe(0);
    expect(screen.queryByRole('button', { name: 'Recover pending request' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled();
  });

  it.each(['focus', 'pageshow'])('restarts interrupted initial details after same-account %s verification', async (event) => {
    existingFailedPack();
    let oldStatus!: (value: unknown) => void;
    mocks.getVNAssetGeneration.mockImplementationOnce(() => new Promise((resolve) => { oldStatus = resolve; }));
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(mocks.getVNAssetGeneration).toHaveBeenCalledTimes(1));
    act(() => window.dispatchEvent(new Event(event)));
    await waitFor(() => expect(mocks.profile).toHaveBeenCalledTimes(2));
    await act(async () => oldStatus({ status: 'processing', batch_id: 99 }));
    expect(mocks.getVNAssetGeneration).toHaveBeenCalledTimes(2);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    expect(screen.getByLabelText('Generation status')).toHaveTextContent('failed');
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
  });

  it('retains the matching key if acknowledgement cleanup fails', async () => {
    existingFailedPack();
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled());
    const spy = vi.spyOn(Object.getPrototypeOf(sessionStorage), 'removeItem').mockImplementation(() => { throw new Error('denied'); });
    await user.click(screen.getByRole('button', { name: 'Start generation' }));
    await screen.findByText(/request result was received/);
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();
    const original = mocks.startVNAssetGeneration.mock.calls[0][1];
    spy.mockRestore();
    await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(sessionStorage.length).toBe(0));
    expect(mocks.startVNAssetGeneration.mock.calls[1][1]).toEqual(original);
  });

  it('retries only the failed slot and reuses its key after connection loss', async () => {
    existingFailedPack();
    mocks.retryVNAssetSlot.mockRejectedValueOnce(new Error('Retry connection lost'));
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));
    await screen.findByText('Retry connection lost');
    expect(screen.getByRole('button', { name: 'Retry sprite_neutral' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Recover pending request' }));
    await waitFor(() => expect(mocks.retryVNAssetSlot).toHaveBeenCalledTimes(2));
    const request = mocks.retryVNAssetSlot.mock.calls[0][2];
    expect(request.idempotency_key).toEqual(expect.any(String));
    expect(request.source_batch_id).toBe(41);
    expect(mocks.retryVNAssetSlot.mock.calls[1]).toEqual([7, 12, request]);
    expect(mocks.startVNAssetGeneration).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Retry sprite_neutral' })).toBeDisabled();
  });

  it('does not bind an older failed slot to the latest unrelated batch', async () => {
    existingFailedPack();
    mocks.getVNAssetGeneration.mockResolvedValue({
      batch_id: 42, status: 'failed', failed_count: 1,
      selected_slot_ids: [12], failed_slot_batch_ids: { 12: 41 },
    });
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);
    await user.click(await screen.findByRole('button', { name: 'Retry sprite_neutral' }));

    const request = mocks.retryVNAssetSlot.mock.calls[0][2];
    expect(request.source_batch_id).toBe(41);
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
