import { bgRequest } from "@/services/background-proxy"
import { requestScopeFields } from "./domains/service-prompts"
import type { AllowedPath } from "./openapi-guard"
import type { QuickIngestOperation } from "./quick-ingest-authority"
import { createServicePromptScopeChangedError } from "./service-prompt-scope-error"

/** Read one existing document navigation route with its captured ownership lease. */
export async function readOwnedOutline<T extends { media_id: number }>(
  mediaId: number, suffix: "outline" | "navigation?include_generated_fallback=false" | `navigation/${string}/content?format=auto&include_alternates=false`, operation: QuickIngestOperation, signal: AbortSignal
): Promise<T> {
  operation.assertCurrent()
  signal.throwIfAborted()
  const response = await bgRequest<T>({
    path: `/api/v1/media/${mediaId}/${suffix}` as AllowedPath,
    method: "GET",
    ...requestScopeFields(operation.requestScope),
    abortSignal: AbortSignal.any([operation.signal, signal])
  })
  operation.assertCurrent()
  signal.throwIfAborted()
  if (response.media_id !== mediaId) throw createServicePromptScopeChangedError()
  return response
}
