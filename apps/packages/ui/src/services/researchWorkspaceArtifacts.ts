import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type ResearchWorkspaceArtifactType = "audio_overview" | "data_table" | "mindmap"

export type ResearchWorkspaceArtifactGenerateRequest = {
  artifact_type: ResearchWorkspaceArtifactType
  media_ids: number[]
  model: string
  api_provider?: string
  claims_verification_provider?: string | null
  claims_verification_model?: string | null
  temperature?: number
  top_p?: number
  max_tokens?: number
}

export type ResearchWorkspaceArtifactGenerateResponse = {
  artifact_type: ResearchWorkspaceArtifactType
  content: string
  data?: Record<string, unknown>
  claim_verification?: Record<string, unknown> | null
}

export async function generateResearchWorkspaceArtifact(
  request: ResearchWorkspaceArtifactGenerateRequest,
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<ResearchWorkspaceArtifactGenerateResponse> {
  return await bgRequest<ResearchWorkspaceArtifactGenerateResponse, AllowedPath, "POST">({
    path: "/api/v1/research-workspace/artifacts/generate",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: request,
    abortSignal: options?.signal,
    timeoutMs: options?.timeoutMs ?? 180000
  })
}
