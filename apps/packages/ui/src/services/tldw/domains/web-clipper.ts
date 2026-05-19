import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import type {
  WebClipperEnrichmentPayload,
  WebClipperEnrichmentResponse,
  WebClipperSaveRequest,
  WebClipperSaveResponse,
  WebClipperStatusResponse
} from "@/services/web-clipper/types"

export interface TldwWebClipperApiClientCore {
  resolveApiPath(key: string, candidates: string[]): Promise<AllowedPath>
  fillPathParams(template: AllowedPath, values: string | string[]): AllowedPath
}

export const webClipperMethods = {
  async saveWebClip(
    this: TldwWebClipperApiClientCore,
    payload: WebClipperSaveRequest
  ): Promise<WebClipperSaveResponse> {
    const path = await this.resolveApiPath("webClipper.save", [
      "/api/v1/web-clipper/save",
      "/api/v1/web-clipper/save/"
    ])
    return await bgRequest<WebClipperSaveResponse>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async getWebClipStatus(
    this: TldwWebClipperApiClientCore,
    clipId: string
  ): Promise<WebClipperStatusResponse> {
    const template = await this.resolveApiPath("webClipper.status", [
      "/api/v1/web-clipper/{clip_id}",
      "/api/v1/web-clipper/{clip_id}/"
    ])
    const path = this.fillPathParams(template, clipId)
    return await bgRequest<WebClipperStatusResponse>({
      path,
      method: "GET"
    })
  },

  async persistWebClipEnrichment(
    this: TldwWebClipperApiClientCore,
    clipId: string,
    payload: WebClipperEnrichmentPayload
  ): Promise<WebClipperEnrichmentResponse> {
    const template = await this.resolveApiPath("webClipper.enrichment", [
      "/api/v1/web-clipper/{clip_id}/enrichments",
      "/api/v1/web-clipper/{clip_id}/enrichments/"
    ])
    const path = this.fillPathParams(template, clipId)
    return await bgRequest<WebClipperEnrichmentResponse>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }
}

export type WebClipperMethods = typeof webClipperMethods
