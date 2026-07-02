import { bgRequest, bgUpload } from "../../background-proxy"
import { buildQuery } from "../client-utils"
import type { AllowedPath } from "../openapi-guard"
import type {
  VisualIdentityAssetResponse,
  VisualIdentityAssetUploadRequest,
  VisualIdentityBindingRequest,
  VisualIdentityBindingResponse,
  VisualIdentityCapabilitiesResponse,
  VisualIdentityDraftActivateRequest,
  VisualIdentityDraftResponse,
  VisualIdentityDraftSlotUpdate,
  VisualIdentityExpressionSlotResponse,
  VisualIdentityGeneratedFileAssetRequest,
  VisualIdentityImportZipStartResponse,
  VisualIdentityPackCreate,
  VisualIdentityPackResponse,
  VisualIdentityPackUpdate,
  VisualIdentityResolveRequest,
  VisualIdentityResolveResponse,
  VisualIdentityZipImportRequest
} from "../../../types/visual-identities"

type TldwApiClientCore = object

const VISUAL_IDENTITIES_BASE_PATH = "/api/v1/visual-identities"

const apiPath = (path: string): AllowedPath => path as AllowedPath

const encodePathSegment = (value: number | string): string =>
  encodeURIComponent(String(value))

const compactMultipartFields = (
  fields: Record<string, unknown>
): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(fields).filter(([, value]) => value !== undefined && value !== null)
  )

export const buildVisualIdentityAssetContentPath = (
  packId: number,
  assetId: number
): string =>
  `${VISUAL_IDENTITIES_BASE_PATH}/packs/${encodePathSegment(packId)}/assets/${encodePathSegment(assetId)}/content`

export const visualIdentityMethods = {
  async getVisualIdentityCapabilities(
    this: TldwApiClientCore
  ): Promise<VisualIdentityCapabilitiesResponse> {
    return await bgRequest<VisualIdentityCapabilitiesResponse>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/capabilities`),
      method: "GET"
    })
  },

  async listVisualIdentityExpressionSlots(
    this: TldwApiClientCore
  ): Promise<VisualIdentityExpressionSlotResponse[]> {
    return await bgRequest<VisualIdentityExpressionSlotResponse[]>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/expression-slots`),
      method: "GET"
    })
  },

  async listVisualIdentityPacks(
    this: TldwApiClientCore,
    params?: { status?: string | null }
  ): Promise<VisualIdentityPackResponse[]> {
    const query = buildQuery(params as Record<string, unknown> | undefined)
    return await bgRequest<VisualIdentityPackResponse[]>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/packs${query}`),
      method: "GET"
    })
  },

  async createVisualIdentityPack(
    this: TldwApiClientCore,
    request: VisualIdentityPackCreate
  ): Promise<VisualIdentityPackResponse> {
    return await bgRequest<VisualIdentityPackResponse>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/packs`),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: request
    })
  },

  async getVisualIdentityPack(
    this: TldwApiClientCore,
    packId: number
  ): Promise<VisualIdentityPackResponse> {
    return await bgRequest<VisualIdentityPackResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/packs/${encodePathSegment(packId)}`
      ),
      method: "GET"
    })
  },

  async updateVisualIdentityPack(
    this: TldwApiClientCore,
    packId: number,
    request: VisualIdentityPackUpdate
  ): Promise<VisualIdentityPackResponse> {
    return await bgRequest<VisualIdentityPackResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/packs/${encodePathSegment(packId)}`
      ),
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: request
    })
  },

  async deleteVisualIdentityPack(
    this: TldwApiClientCore,
    packId: number
  ): Promise<void> {
    await bgRequest<void>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/packs/${encodePathSegment(packId)}`
      ),
      method: "DELETE"
    })
  },

  async uploadVisualIdentityPackAsset(
    this: TldwApiClientCore,
    packId: number,
    request: VisualIdentityAssetUploadRequest
  ): Promise<VisualIdentityAssetResponse> {
    return await bgUpload<VisualIdentityAssetResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/packs/${encodePathSegment(packId)}/assets`
      ),
      method: "POST",
      fields: compactMultipartFields({
        expression_key: request.expression_key,
        draft_id: request.draft_id
      }),
      file: request.file,
      fileFieldName: "file",
      timeoutMs: request.timeoutMs
    })
  },

  async createVisualIdentityAssetFromGeneratedFile(
    this: TldwApiClientCore,
    packId: number,
    request: VisualIdentityGeneratedFileAssetRequest
  ): Promise<VisualIdentityAssetResponse> {
    return await bgRequest<VisualIdentityAssetResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/packs/${encodePathSegment(packId)}/assets/from-generated-file`
      ),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: request
    })
  },

  getVisualIdentityAssetContentPath(
    this: TldwApiClientCore,
    packId: number,
    assetId: number
  ): string {
    return buildVisualIdentityAssetContentPath(packId, assetId)
  },

  async startVisualIdentityZipImport(
    this: TldwApiClientCore,
    request: VisualIdentityZipImportRequest
  ): Promise<VisualIdentityImportZipStartResponse> {
    return await bgUpload<VisualIdentityImportZipStartResponse>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/imports/zip`),
      method: "POST",
      fields: compactMultipartFields({
        title: request.title,
        pack_id: request.pack_id,
        idempotency_key: request.idempotency_key
      }),
      file: request.archive,
      fileFieldName: "archive",
      timeoutMs: request.timeoutMs
    })
  },

  async getVisualIdentityDraft(
    this: TldwApiClientCore,
    draftId: number
  ): Promise<VisualIdentityDraftResponse> {
    return await bgRequest<VisualIdentityDraftResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/drafts/${encodePathSegment(draftId)}`
      ),
      method: "GET"
    })
  },

  async updateVisualIdentityDraftSlot(
    this: TldwApiClientCore,
    draftId: number,
    slotKey: string,
    request: VisualIdentityDraftSlotUpdate
  ): Promise<VisualIdentityDraftResponse> {
    return await bgRequest<VisualIdentityDraftResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/drafts/${encodePathSegment(draftId)}/slots/${encodePathSegment(slotKey)}`
      ),
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: request
    })
  },

  async activateVisualIdentityDraft(
    this: TldwApiClientCore,
    draftId: number,
    request: VisualIdentityDraftActivateRequest = {}
  ): Promise<VisualIdentityDraftResponse> {
    return await bgRequest<VisualIdentityDraftResponse>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/drafts/${encodePathSegment(draftId)}/activate`
      ),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: request
    })
  },

  async upsertVisualIdentityBinding(
    this: TldwApiClientCore,
    request: VisualIdentityBindingRequest
  ): Promise<VisualIdentityBindingResponse> {
    return await bgRequest<VisualIdentityBindingResponse>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/bindings`),
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: request
    })
  },

  async deleteVisualIdentityBinding(
    this: TldwApiClientCore,
    bindingId: number
  ): Promise<void> {
    await bgRequest<void>({
      path: apiPath(
        `${VISUAL_IDENTITIES_BASE_PATH}/bindings/${encodePathSegment(bindingId)}`
      ),
      method: "DELETE"
    })
  },

  async resolveVisualIdentityBinding(
    this: TldwApiClientCore,
    request: VisualIdentityResolveRequest
  ): Promise<VisualIdentityResolveResponse> {
    const query = buildQuery({
      actor_kind: request.actor_kind,
      actor_id: request.actor_id,
      expression_key: request.expression_key,
      manual_override_expression_key: request.manual_override_expression_key,
      mood_expression_key: request.mood_expression_key
    })
    return await bgRequest<VisualIdentityResolveResponse>({
      path: apiPath(`${VISUAL_IDENTITIES_BASE_PATH}/bindings/resolve${query}`),
      method: "GET"
    })
  }
}

export type VisualIdentityMethods = typeof visualIdentityMethods
