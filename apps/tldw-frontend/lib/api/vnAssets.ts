import { apiClient } from '@web/lib/api';
import type {
  VNAssetBulkReviewRequest,
  VNAssetGenerationRequest,
  VNAssetGenerationStatus,
  VNAssetItem,
  VNAssetManifest,
  VNAssetPack,
  VNAssetPackCreate,
  VNAssetPackUpdate,
  VNAssetPromptPreview,
  VNAssetPromptPreviewRequest,
  VNAssetReadiness,
  VNAssetReviewRequest,
  VNAssetSlot,
  VNAssetSlotUpdate,
  VNAssetStarterMatricesResponse,
  VNPackExportRequest,
  VNPackExportResponse,
  VNPackImportCommitRequest,
  VNPackImportCommitStartResponse,
  VNPackImportJob,
  VNPackImportPreview,
  VNPackImportPreviewStartResponse,
  VNPackPortabilityJob,
} from '@web/types/vn-assets';

const VN_ASSETS_BASE = '/vn/vn-assets';

export function listVNAssetPacks(): Promise<VNAssetPack[]> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs`);
}

export function createVNAssetPack(request: VNAssetPackCreate): Promise<VNAssetPack> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs`, request);
}

export function getVNAssetPack(packId: number): Promise<VNAssetPack> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs/${packId}`);
}

export function updateVNAssetPack(packId: number, request: VNAssetPackUpdate): Promise<VNAssetPack> {
  return apiClient.patch(`${VN_ASSETS_BASE}/packs/${packId}`, request);
}

export function deleteVNAssetPack(packId: number): Promise<void> {
  return apiClient.delete(`${VN_ASSETS_BASE}/packs/${packId}`);
}

export function getStarterMatrices(): Promise<VNAssetStarterMatricesResponse> {
  return apiClient.get(`${VN_ASSETS_BASE}/starter-matrices`);
}

export function applyVNAssetMatrix(
  packId: number,
  matrixKey: string,
  overrides: Record<string, unknown> = {}
): Promise<VNAssetSlot[]> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/matrix/apply`, {
    matrix_key: matrixKey,
    overrides,
  });
}

export function listVNAssetSlots(packId: number): Promise<VNAssetSlot[]> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs/${packId}/slots`);
}

export function updateVNAssetSlot(
  packId: number,
  slotId: number,
  request: VNAssetSlotUpdate
): Promise<VNAssetSlot> {
  return apiClient.patch(`${VN_ASSETS_BASE}/packs/${packId}/slots/${slotId}`, request);
}

export function startVNAssetGeneration(
  packId: number,
  request: VNAssetGenerationRequest = {}
): Promise<VNAssetGenerationStatus> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/generate`, request);
}

export function getVNAssetGeneration(packId: number): Promise<VNAssetGenerationStatus> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs/${packId}/generation`);
}

export function cancelVNAssetGeneration(packId: number): Promise<VNAssetGenerationStatus> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/generation/cancel`);
}

export function listVNAssetItems(packId: number): Promise<VNAssetItem[]> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs/${packId}/items`);
}

export function reviewVNAssetItem(
  packId: number,
  itemId: number,
  request: VNAssetReviewRequest
): Promise<VNAssetItem> {
  return apiClient.patch(`${VN_ASSETS_BASE}/packs/${packId}/items/${itemId}/review`, request);
}

export function bulkReviewVNAssetItems(
  packId: number,
  request: VNAssetBulkReviewRequest
): Promise<VNAssetItem[]> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/items/bulk-review`, request);
}

export function setPreferredVNAssetItem(packId: number, itemId: number): Promise<VNAssetItem> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/items/${itemId}/preferred`);
}

export function getVNAssetReadiness(packId: number): Promise<VNAssetReadiness> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs/${packId}/readiness`);
}

export function getVNAssetManifest(packId: number): Promise<VNAssetManifest> {
  return apiClient.get(`${VN_ASSETS_BASE}/packs/${packId}/manifest`);
}

export function previewVNAssetPrompt(
  packId: number,
  request: VNAssetPromptPreviewRequest
): Promise<VNAssetPromptPreview> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/prompt-preview`, request);
}

export function exportVNAssetPack(
  packId: number,
  request: VNPackExportRequest
): Promise<VNPackExportResponse> {
  return apiClient.post(`${VN_ASSETS_BASE}/packs/${packId}/export`, request);
}

export function getVNPackExportJob(jobId: string): Promise<VNPackPortabilityJob> {
  return apiClient.get(`${VN_ASSETS_BASE}/portability/exports/${jobId}`);
}

export function createVNPackImportPreview(
  archive: File,
  idempotencyKey: string
): Promise<VNPackImportPreviewStartResponse> {
  const formData = new FormData();
  formData.append('archive', archive);
  formData.append('idempotency_key', idempotencyKey);
  return apiClient.post(`${VN_ASSETS_BASE}/import/previews`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
}

export function getVNPackImportPreview(previewId: number): Promise<VNPackImportPreview> {
  return apiClient.get(`${VN_ASSETS_BASE}/import/previews/${previewId}`);
}

export function commitVNPackImport(
  request: VNPackImportCommitRequest
): Promise<VNPackImportCommitStartResponse> {
  return apiClient.post(`${VN_ASSETS_BASE}/import/commit`, request);
}

export function getVNPackImportJob(jobId: string): Promise<VNPackImportJob> {
  return apiClient.get(`${VN_ASSETS_BASE}/portability/imports/${jobId}`);
}
