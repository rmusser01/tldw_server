import { apiClient } from '@web/lib/api';
import type {
  VNPlayBranch,
  VNPlayCheckpoint,
  VNPlayCheckpointCreate,
  VNPlayEvent,
  VNPlayGenerationActionRequest,
  VNPlayGenerationDebugQuery,
  VNPlayGenerationHistoryItem,
  VNPlayGenerationHistoryResponse,
  VNPlayGenerationListQuery,
  VNPlayGenerationRevisionDebugResponse,
  VNPlayGenerationRevisionListResponse,
  VNPlayRestoreRequest,
  VNPlayRetryTurnRequest,
  VNPlaySession,
  VNPlaySessionCreate,
  VNPlaySessionUpdate,
  VNPlaySetupOptionsQuery,
  VNPlaySetupOptionsResponse,
  VNPlayTurnRequest,
  VNPlayTurnResponse,
} from '@web/types/vn-play';

const VN_PLAY_BASE = '/vn/vn-play';

type QueryParams = Record<string, string | number | boolean | null | undefined>;

function toQueryParams<T extends object>(query: T): QueryParams {
  return { ...query } as QueryParams;
}

export function createVNPlaySession(request: VNPlaySessionCreate): Promise<VNPlaySession> {
  return apiClient.post(`${VN_PLAY_BASE}/sessions`, request);
}

export function listVNPlaySetupOptions(
  query: VNPlaySetupOptionsQuery = {}
): Promise<VNPlaySetupOptionsResponse> {
  return apiClient.get(`${VN_PLAY_BASE}/setup-options`, { params: toQueryParams(query) });
}

export function listVNPlaySessions(): Promise<VNPlaySession[]> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions`);
}

export function getVNPlaySession(sessionId: number): Promise<VNPlaySession> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions/${sessionId}`);
}

export function updateVNPlaySession(
  sessionId: number,
  request: VNPlaySessionUpdate
): Promise<VNPlaySession> {
  return apiClient.patch(`${VN_PLAY_BASE}/sessions/${sessionId}`, request);
}

export function deleteVNPlaySession(sessionId: number): Promise<void> {
  return apiClient.delete(`${VN_PLAY_BASE}/sessions/${sessionId}`);
}

export function submitVNPlayTurn(
  sessionId: number,
  request: VNPlayTurnRequest
): Promise<VNPlayTurnResponse> {
  return apiClient.post(`${VN_PLAY_BASE}/sessions/${sessionId}/turn`, request);
}

export function retryLastVNPlayTurn(
  sessionId: number,
  request: VNPlayRetryTurnRequest
): Promise<VNPlayTurnResponse> {
  return apiClient.post(`${VN_PLAY_BASE}/sessions/${sessionId}/retry-last-turn`, request);
}

export function listVNPlayEvents(sessionId: number): Promise<VNPlayEvent[]> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions/${sessionId}/events`);
}

export function createVNPlayCheckpoint(
  sessionId: number,
  request: VNPlayCheckpointCreate
): Promise<VNPlayCheckpoint> {
  return apiClient.post(`${VN_PLAY_BASE}/sessions/${sessionId}/checkpoint`, request);
}

export function listVNPlayCheckpoints(sessionId: number): Promise<VNPlayCheckpoint[]> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions/${sessionId}/checkpoints`);
}

export function restoreVNPlaySession(
  sessionId: number,
  request: VNPlayRestoreRequest
): Promise<VNPlaySession> {
  return apiClient.post(`${VN_PLAY_BASE}/sessions/${sessionId}/restore`, request);
}

export function listVNPlayBranches(sessionId: number): Promise<VNPlayBranch[]> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions/${sessionId}/branches`);
}

export function listVNPlayGenerations(
  sessionId: number,
  query: VNPlayGenerationListQuery = {}
): Promise<VNPlayGenerationHistoryResponse> {
  return apiClient.get(`${VN_PLAY_BASE}/sessions/${sessionId}/script/generations`, {
    params: toQueryParams(query),
  });
}

export function listVNPlayGenerationRevisions(
  sessionId: number,
  generationId: number,
  query: Pick<VNPlayGenerationListQuery, 'limit' | 'offset' | 'status' | 'source'> = {}
): Promise<VNPlayGenerationRevisionListResponse> {
  return apiClient.get(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generations/${generationId}/revisions`,
    { params: toQueryParams(query) }
  );
}

export function getVNPlayGenerationRevision(
  sessionId: number,
  generationId: number,
  revisionId: number
): Promise<VNPlayGenerationHistoryItem> {
  return apiClient.get(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generations/${generationId}/revisions/${revisionId}`
  );
}

export function getVNPlayGenerationRevisionDebug(
  sessionId: number,
  generationId: number,
  revisionId: number,
  query: VNPlayGenerationDebugQuery = {}
): Promise<VNPlayGenerationRevisionDebugResponse> {
  return apiClient.get(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generations/${generationId}/revisions/${revisionId}/debug`,
    { params: toQueryParams(query) }
  );
}

export function confirmVNPlayGenerationRequest(
  sessionId: number,
  generationRequestId: number,
  request: VNPlayGenerationActionRequest
): Promise<VNPlayTurnResponse> {
  return apiClient.post(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generation-requests/${generationRequestId}/confirm`,
    request
  );
}

export function cancelVNPlayGenerationRequest(
  sessionId: number,
  generationRequestId: number,
  request: VNPlayGenerationActionRequest
): Promise<VNPlayTurnResponse> {
  return apiClient.post(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generation-requests/${generationRequestId}/cancel`,
    request
  );
}

export function regenerateVNPlayGeneration(
  sessionId: number,
  generationId: number,
  request: VNPlayGenerationActionRequest
): Promise<VNPlayTurnResponse> {
  return apiClient.post(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generations/${generationId}/regenerate`,
    request
  );
}

export function activateVNPlayGenerationRevision(
  sessionId: number,
  generationId: number,
  revisionId: number,
  request: VNPlayGenerationActionRequest
): Promise<VNPlayTurnResponse> {
  return apiClient.post(
    `${VN_PLAY_BASE}/sessions/${sessionId}/script/generations/${generationId}/revisions/${revisionId}/activate`,
    request
  );
}
