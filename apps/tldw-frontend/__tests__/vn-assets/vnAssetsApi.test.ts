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
  applyVNAssetMatrix,
  bulkReviewVNAssetItems,
  cancelVNAssetGeneration,
  createVNAssetPack,
  deleteVNAssetPack,
  getStarterMatrices,
  getVNAssetGeneration,
  getVNAssetManifest,
  getVNAssetPack,
  getVNAssetReadiness,
  listVNAssetItems,
  listVNAssetPacks,
  listVNAssetSlots,
  previewVNAssetPrompt,
  reviewVNAssetItem,
  setPreferredVNAssetItem,
  startVNAssetGeneration,
  updateVNAssetPack,
  updateVNAssetSlot,
} from '@web/lib/api/vnAssets';

describe('vnAssets api client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.apiClient.delete.mockResolvedValue(undefined);
    mocks.apiClient.get.mockResolvedValue({});
    mocks.apiClient.patch.mockResolvedValue({});
    mocks.apiClient.post.mockResolvedValue({});
  });

  it('calls pack and matrix endpoints with expected paths and payloads', async () => {
    mocks.apiClient.post.mockResolvedValueOnce({ id: 1, title: 'Starter', primary_character_id: 7 });

    const created = await createVNAssetPack({ title: 'Starter', primary_character_id: 7 });
    await listVNAssetPacks();
    await getVNAssetPack(1);
    await updateVNAssetPack(1, { title: 'Updated' });
    await deleteVNAssetPack(1);
    await getStarterMatrices();
    await applyVNAssetMatrix(1, 'starter', { variant_count: 2 });

    expect(created.id).toBe(1);
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs', {
      title: 'Starter',
      primary_character_id: 7,
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs/1');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn-assets/packs/1', { title: 'Updated' });
    expect(mocks.apiClient.delete).toHaveBeenCalledWith('/vn-assets/packs/1');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/starter-matrices');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs/1/matrix/apply', {
      matrix_key: 'starter',
      overrides: { variant_count: 2 },
    });
  });

  it('calls generation, item review, readiness, manifest, and prompt endpoints', async () => {
    await listVNAssetSlots(1);
    await updateVNAssetSlot(1, 2, { variant_count: 3 });
    await startVNAssetGeneration(1, { slot_ids: [2], variant_count: 1 });
    await getVNAssetGeneration(1);
    await cancelVNAssetGeneration(1);
    await listVNAssetItems(1);
    await reviewVNAssetItem(1, 5, { review_status: 'approved', preferred: true });
    await bulkReviewVNAssetItems(1, { item_ids: [5, 6], review_status: 'hidden' });
    await setPreferredVNAssetItem(1, 5);
    await getVNAssetReadiness(1);
    await getVNAssetManifest(1);
    await previewVNAssetPrompt(1, { slot_id: 2, variant_index: 0 });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs/1/slots');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn-assets/packs/1/slots/2', { variant_count: 3 });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs/1/generate', {
      slot_ids: [2],
      variant_count: 1,
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs/1/generation');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs/1/generation/cancel');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs/1/items');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn-assets/packs/1/items/5/review', {
      review_status: 'approved',
      preferred: true,
    });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs/1/items/bulk-review', {
      item_ids: [5, 6],
      review_status: 'hidden',
    });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs/1/items/5/preferred');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs/1/readiness');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn-assets/packs/1/manifest');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn-assets/packs/1/prompt-preview', {
      slot_id: 2,
      variant_index: 0,
    });
  });
});
