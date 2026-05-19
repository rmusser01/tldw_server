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
  commitVNPackImport,
  cancelVNAssetGeneration,
  createVNAssetPack,
  createVNPackImportPreview,
  deleteVNAssetPack,
  exportVNAssetPack,
  getStarterMatrices,
  getVNAssetGeneration,
  getVNAssetManifest,
  getVNAssetPack,
  getVNAssetReadiness,
  getVNPackExportJob,
  getVNPackImportJob,
  getVNPackImportPreview,
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
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs', {
      title: 'Starter',
      primary_character_id: 7,
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs/1');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn/vn-assets/packs/1', { title: 'Updated' });
    expect(mocks.apiClient.delete).toHaveBeenCalledWith('/vn/vn-assets/packs/1');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/starter-matrices');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/1/matrix/apply', {
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

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs/1/slots');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn/vn-assets/packs/1/slots/2', { variant_count: 3 });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/1/generate', {
      slot_ids: [2],
      variant_count: 1,
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs/1/generation');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/1/generation/cancel');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs/1/items');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn/vn-assets/packs/1/items/5/review', {
      review_status: 'approved',
      preferred: true,
    });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/1/items/bulk-review', {
      item_ids: [5, 6],
      review_status: 'hidden',
    });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/1/items/5/preferred');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs/1/readiness');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/packs/1/manifest');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/1/prompt-preview', {
      slot_id: 2,
      variant_index: 0,
    });
  });

  it('calls portability export and import endpoints with expected paths and payloads', async () => {
    const archive = new File(['vnpack'], 'pack.tldw-vnpack', { type: 'application/zip' });

    await exportVNAssetPack(9, {
      include_character_payload: true,
      include_world_book_payloads: false,
      include_full_provenance: true,
      strict: true,
      warn_for_sharing: true,
      idempotency_key: 'export-1',
    });
    await getVNPackExportJob('job-export');
    await createVNPackImportPreview(archive, 'preview-1');
    await getVNPackImportPreview(33);
    await commitVNPackImport({
      preview_id: 33,
      trust_mode: 'trusted_restore',
      target_mode: 'update_existing',
      character_action: 'link_existing_character',
      target_character_id: 42,
      target_pack_id: 9,
      conflict_decisions: { confirm_all_risky_diffs: true },
      idempotency_key: 'commit-1',
    });
    await getVNPackImportJob('job-import');

    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/packs/9/export', {
      include_character_payload: true,
      include_world_book_payloads: false,
      include_full_provenance: true,
      strict: true,
      warn_for_sharing: true,
      idempotency_key: 'export-1',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/portability/exports/job-export');
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-assets/import/previews',
      expect.any(FormData),
      expect.objectContaining({ headers: expect.objectContaining({ 'Content-Type': 'multipart/form-data' }) })
    );
    const formData = mocks.apiClient.post.mock.calls.find(
      ([url]) => url === '/vn/vn-assets/import/previews'
    )?.[1] as FormData;
    expect(formData.get('archive')).toBe(archive);
    expect(formData.get('idempotency_key')).toBe('preview-1');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/import/previews/33');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-assets/import/commit', {
      preview_id: 33,
      trust_mode: 'trusted_restore',
      target_mode: 'update_existing',
      character_action: 'link_existing_character',
      target_character_id: 42,
      target_pack_id: 9,
      conflict_decisions: { confirm_all_risky_diffs: true },
      idempotency_key: 'commit-1',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-assets/portability/imports/job-import');
  });
});
