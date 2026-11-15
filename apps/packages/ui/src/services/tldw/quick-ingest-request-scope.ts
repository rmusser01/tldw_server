import { bgRequest, bgUpload, type BgRequestInit, type BgUploadInit } from "@/services/background-proxy"
import { requestScopeFields, type ServicePromptRequestScope } from "./domains/service-prompts"
import { createServicePromptScopeChangedError, isRequestConfigScopeChangedError } from "./service-prompt-scope-error"

export type QuickIngestRequestContext = {
  requestScope?: ServicePromptRequestScope
  signal?: AbortSignal
  assertCurrent?: () => void
}

export const assertQuickIngestRequestCurrent = (context: QuickIngestRequestContext, error?: unknown): void => {
  if (isRequestConfigScopeChangedError(error)) throw error
  if (!context.requestScope) throw createServicePromptScopeChangedError()
  context.signal?.throwIfAborted()
  context.assertCurrent?.()
}

export const scopedQuickIngestRequest = async <T>(context: QuickIngestRequestContext, request: BgRequestInit): Promise<T> => {
  assertQuickIngestRequestCurrent(context)
  const fields = requestScopeFields(context.requestScope)
  const result = await bgRequest<T>({ ...request, ...fields, headers: { ...request.headers, ...fields.headers }, abortSignal: context.signal })
  assertQuickIngestRequestCurrent(context)
  return result
}

export const scopedQuickIngestUpload = async <T>(context: QuickIngestRequestContext, request: BgUploadInit): Promise<T> => {
  assertQuickIngestRequestCurrent(context)
  const fields = requestScopeFields(context.requestScope)
  const result = await bgUpload<T>({ ...request, ...fields, headers: { ...request.headers, ...fields.headers }, abortSignal: context.signal })
  assertQuickIngestRequestCurrent(context)
  return result
}

export const requestQuickIngestJobCancellation = async (
  context: QuickIngestRequestContext,
  jobId: number
): Promise<boolean> => {
  const response = await scopedQuickIngestRequest<{
    ok: boolean
    data?: { success?: boolean; status?: string }
  }>(context, {
    path: `/api/v1/media/ingest/jobs/${jobId}?reason=user_cancelled`,
    method: "DELETE",
    timeoutMs: 10_000,
    returnResponse: true,
    preferDirect: true,
  })
  return response?.ok === true && response.data?.success === true && response.data.status === "cancelled"
}
