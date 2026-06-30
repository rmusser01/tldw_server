import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiClient: {
    delete: vi.fn(),
    get: vi.fn(),
    patch: vi.fn(),
    post: vi.fn(),
  },
}));

vi.mock('@web/lib/api', () => ({
  apiClient: mocks.apiClient,
}));

import {
  activateVNPlayGenerationRevision,
  cancelVNPlayGenerationRequest,
  confirmVNPlayGenerationRequest,
  createVNPlayCheckpoint,
  createVNPlaySession,
  deleteVNPlaySession,
  getVNPlayBranchNavigation,
  getVNPlayGenerationRevision,
  getVNPlayGenerationRevisionDebug,
  getVNPlaySession,
  listVNPlayGenerationRevisions,
  listVNPlayGenerations,
  listVNPlayBranches,
  listVNPlayCheckpoints,
  listVNPlayEvents,
  listVNPlaySessions,
  listVNPlaySetupOptions,
  regenerateVNPlayGeneration,
  restoreVNPlayBranch,
  restoreVNPlaySession,
  retryLastVNPlayTurn,
  submitVNPlayTurn,
  updateVNPlaySession,
} from '@web/lib/api/vnPlay';

describe('vnPlay api client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.apiClient.delete.mockResolvedValue(undefined);
    mocks.apiClient.get.mockResolvedValue({});
    mocks.apiClient.patch.mockResolvedValue({});
    mocks.apiClient.post.mockResolvedValue({});
  });

  it('creates a VN play session', async () => {
    mocks.apiClient.post.mockResolvedValueOnce({
      id: 1,
      mode: 'freeform',
      title: 'Library',
      scene_state: { scene_version: 0 },
    });

    const session = await createVNPlaySession({
      mode: 'freeform',
      title: 'Library',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
    });

    expect(session.id).toBe(1);
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions', {
      mode: 'freeform',
      title: 'Library',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
    });
  });

  it('creates a scripted-story VN play session with script identifiers and policy acknowledgements', async () => {
    await createVNPlaySession({
      mode: 'scripted_story',
      title: 'Published route',
      primary_character_id: 7,
      vn_asset_pack_id: 12,
      content_rating: 'teen',
      script_id: 44,
      script_version_id: 5,
      acknowledgements: ['script_policy_review'],
    });

    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions', {
      mode: 'scripted_story',
      title: 'Published route',
      primary_character_id: 7,
      vn_asset_pack_id: 12,
      content_rating: 'teen',
      script_id: 44,
      script_version_id: 5,
      acknowledgements: ['script_policy_review'],
    });
  });

  it('submits a VN play turn with idempotency key and scene version', async () => {
    mocks.apiClient.post.mockResolvedValueOnce({
      events: [],
      scene_state: { scene_version: 1 },
    });

    await submitVNPlayTurn(1, {
      input_text: 'Hello',
      client_scene_version: 0,
      idempotency_key: 'turn-1',
    });

    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions/1/turn', {
      input_text: 'Hello',
      client_scene_version: 0,
      idempotency_key: 'turn-1',
    });
  });

  it('calls session, checkpoint, restore, and branch endpoints', async () => {
    await listVNPlaySessions();
    await getVNPlaySession(1);
    await updateVNPlaySession(1, { title: 'Updated' });
    await deleteVNPlaySession(1);
    await retryLastVNPlayTurn(1, { client_scene_version: 2, idempotency_key: 'retry-1' });
    await listVNPlayEvents(1);
    await createVNPlayCheckpoint(1, { label: 'Before choice' });
    await listVNPlayCheckpoints(1);
    await restoreVNPlaySession(1, {
      checkpoint_id: 5,
      client_scene_version: 2,
      idempotency_key: 'restore-1',
    });
    await listVNPlayBranches(1);

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn/vn-play/sessions/1', { title: 'Updated' });
    expect(mocks.apiClient.delete).toHaveBeenCalledWith('/vn/vn-play/sessions/1');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions/1/retry-last-turn', {
      client_scene_version: 2,
      idempotency_key: 'retry-1',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1/events');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions/1/checkpoint', {
      label: 'Before choice',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1/checkpoints');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions/1/restore', {
      checkpoint_id: 5,
      client_scene_version: 2,
      idempotency_key: 'restore-1',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1/branches');
  });

  it('calls branch navigation and guarded branch restore endpoints', async () => {
    await getVNPlayBranchNavigation(1);
    await restoreVNPlayBranch(1, 12, {
      client_scene_version: 6,
      idempotency_key: 'restore-branch-12',
      target: 'choice_point',
    });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1/branch-navigation');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-play/sessions/1/branches/12/restore', {
      client_scene_version: 6,
      idempotency_key: 'restore-branch-12',
      target: 'choice_point',
    });
  });

  it('loads VN play setup options with server-side selector parameters', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      characters: [],
      asset_packs: [],
      defaults: {},
      empty_states: [],
      generated_at: '2026-05-09T15:00:00Z',
      pagination: {
        characters: { limit: 25, offset: 0, has_more: false, total: 0 },
        asset_packs: { limit: 25, offset: 0, has_more: false, total: 0 },
      },
    });

    await listVNPlaySetupOptions({
      mode: 'scripted_story',
      selected_character_id: 7,
      content_rating: 'mature',
      character_query: 'mira',
      pack_query: 'archive',
    });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/setup-options', {
      params: {
        mode: 'scripted_story',
        selected_character_id: 7,
        content_rating: 'mature',
        character_query: 'mira',
        pack_query: 'archive',
      },
    });
  });

  it('calls scripted generation history and command endpoints', async () => {
    await listVNPlayGenerations(1, { limit: 10, offset: 20, status: 'succeeded', active: true });
    await listVNPlayGenerationRevisions(1, 12, { limit: 5, offset: 0 });
    await getVNPlayGenerationRevision(1, 12, 31);
    await getVNPlayGenerationRevisionDebug(1, 12, 31);
    await getVNPlayGenerationRevisionDebug(1, 12, 31, {
      include_blocked_raw: true,
      confirm: 'REVEAL_MODERATION_BLOCKED',
    });
    await confirmVNPlayGenerationRequest(1, 91, {
      client_scene_version: 4,
      idempotency_key: 'confirm-1',
    });
    await cancelVNPlayGenerationRequest(1, 91, {
      client_scene_version: 4,
      idempotency_key: 'cancel-1',
    });
    await regenerateVNPlayGeneration(1, 12, {
      client_scene_version: 7,
      idempotency_key: 'regen-1',
    });
    await activateVNPlayGenerationRevision(1, 12, 31, {
      client_scene_version: 9,
      idempotency_key: 'activate-1',
    });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-play/sessions/1/script/generations', {
      params: { limit: 10, offset: 20, status: 'succeeded', active: true },
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generations/12/revisions',
      { params: { limit: 5, offset: 0 } }
    );
    expect(mocks.apiClient.get).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generations/12/revisions/31'
    );
    expect(mocks.apiClient.get).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generations/12/revisions/31/debug',
      { params: {} }
    );
    expect(mocks.apiClient.get).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generations/12/revisions/31/debug',
      { params: { include_blocked_raw: true, confirm: 'REVEAL_MODERATION_BLOCKED' } }
    );
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generation-requests/91/confirm',
      { client_scene_version: 4, idempotency_key: 'confirm-1' }
    );
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generation-requests/91/cancel',
      { client_scene_version: 4, idempotency_key: 'cancel-1' }
    );
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generations/12/regenerate',
      { client_scene_version: 7, idempotency_key: 'regen-1' }
    );
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-play/sessions/1/script/generations/12/revisions/31/activate',
      { client_scene_version: 9, idempotency_key: 'activate-1' }
    );
  });
});
