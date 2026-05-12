import { apiClient } from '@web/lib/api';
import type {
  VNScriptCreate,
  VNScriptCreateFromTemplateRequest,
  VNScriptCreateFromTemplateResponse,
  VNScriptDiagnosticsResponse,
  VNScriptDraftPutRequest,
  VNScriptDraftResponse,
  VNScriptListQuery,
  VNScriptListResponse,
  VNScriptManifestSnapshotResponse,
  VNScriptPatch,
  VNScriptPublishRequest,
  VNScriptPublishResponse,
  VNScriptResponse,
  VNScriptTemplateListResponse,
  VNScriptValidateRequest,
  VNScriptValidationResponse,
  VNScriptVersionListQuery,
  VNScriptVersionListResponse,
  VNScriptVersionPolicyEvaluateRequest,
  VNScriptVersionPolicyEvaluateResponse,
  VNScriptVersionResponse,
} from '@web/types/vn-scripts';

const VN_SCRIPTS_BASE = '/vn/vn-scripts';

type QueryParams = Record<string, string | number | boolean | null | undefined>;

function toQueryParams<T extends object>(query: T): QueryParams {
  return Object.fromEntries(
    Object.entries(query).filter(([, value]) => value !== undefined && value !== null)
  ) as QueryParams;
}

export function createVNScript(request: VNScriptCreate): Promise<VNScriptResponse> {
  return apiClient.post(`${VN_SCRIPTS_BASE}/scripts`, request);
}

export function listVNScriptTemplates(): Promise<VNScriptTemplateListResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/templates`);
}

export function createVNScriptFromTemplate(
  templateId: string,
  request: VNScriptCreateFromTemplateRequest
): Promise<VNScriptCreateFromTemplateResponse> {
  return apiClient.post(`${VN_SCRIPTS_BASE}/templates/${templateId}/scripts`, request);
}

export function listVNScripts(query: VNScriptListQuery = {}): Promise<VNScriptListResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts`, { params: toQueryParams(query) });
}

export function getVNScript(scriptId: number): Promise<VNScriptResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts/${scriptId}`);
}

export function patchVNScript(
  scriptId: number,
  request: VNScriptPatch
): Promise<VNScriptResponse> {
  return apiClient.patch(`${VN_SCRIPTS_BASE}/scripts/${scriptId}`, request);
}

export function deleteVNScript(scriptId: number): Promise<void> {
  return apiClient.delete(`${VN_SCRIPTS_BASE}/scripts/${scriptId}`);
}

export function getVNScriptDraft(scriptId: number): Promise<VNScriptDraftResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/draft`);
}

export function putVNScriptDraft(
  scriptId: number,
  request: VNScriptDraftPutRequest
): Promise<VNScriptDraftResponse> {
  return apiClient.put(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/draft`, request);
}

export function validateVNScriptDraft(
  scriptId: number,
  request: VNScriptValidateRequest = {}
): Promise<VNScriptValidationResponse> {
  return apiClient.post(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/draft/validate`, request);
}

export function getVNScriptDiagnostics(scriptId: number): Promise<VNScriptDiagnosticsResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/draft/diagnostics`);
}

export function publishVNScript(
  scriptId: number,
  request: VNScriptPublishRequest
): Promise<VNScriptPublishResponse> {
  return apiClient.post(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/publish`, request);
}

export function listVNScriptVersions(
  scriptId: number,
  query: VNScriptVersionListQuery = {}
): Promise<VNScriptVersionListResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/versions`, {
    params: toQueryParams(query),
  });
}

export function getVNScriptVersion(
  scriptId: number,
  versionId: number
): Promise<VNScriptVersionResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/versions/${versionId}`);
}

export function getVNScriptManifestSnapshot(
  scriptId: number,
  versionId: number
): Promise<VNScriptManifestSnapshotResponse> {
  return apiClient.get(
    `${VN_SCRIPTS_BASE}/scripts/${scriptId}/versions/${versionId}/manifest-snapshot`
  );
}

export function evaluateVNScriptVersionPolicy(
  scriptId: number,
  versionId: number,
  request: VNScriptVersionPolicyEvaluateRequest = {}
): Promise<VNScriptVersionPolicyEvaluateResponse> {
  return apiClient.post(
    `${VN_SCRIPTS_BASE}/scripts/${scriptId}/versions/${versionId}/policy/evaluate`,
    request
  );
}
