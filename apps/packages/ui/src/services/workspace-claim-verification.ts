import { bgRequest } from "./background-proxy"
import { tldwClient } from "./tldw/TldwApiClient"
import { requestScopeFields } from "./tldw/domains/service-prompts"
import type { ServicePromptSnapshot } from "./service-prompts"
import { claimTextHash, readClaimVerification, type ClaimVerification } from "@/utils/claim-verification"

const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value)
const offset = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0
export const uncheckedAnswerVerification = (text: string, reason: string): ClaimVerification => ({
  version: 1, text_sha256: claimTextHash(text), verdict: "unverified", provider: null, model: null,
  coverage: { total: 0, evaluated: 0, supported: 0, uncovered: 0, complete: false }, findings: [], evidence: {}, limits: {},
  context: { unavailable_reason: reason }
})

/** Client quotes locate evidence; only the server may authorize and check its source text. */
export async function verifyWorkspaceAnswer({ workspaceId, text, sources, provider, model, scope, signal }: {
  workspaceId: string; text: string; sources: unknown[]; provider?: string; model?: string
  scope?: ServicePromptSnapshot; signal: AbortSignal
}): Promise<ClaimVerification> {
  const unchecked = () => uncheckedAnswerVerification(text, "Citation checks could not finish. The original answer is retained; review its sources before relying on it.")
  if (!scope || !provider || !model || !text.trim()) return unchecked()
  const assertOwner = () => { scope.scopeInvalidatedSignal.throwIfAborted(); scope.scopeSignal.throwIfAborted() }
  try {
    assertOwner(); signal.throwIfAborted()
    const abortSignal = AbortSignal.any([signal, scope.scopeSignal, scope.scopeInvalidatedSignal, AbortSignal.timeout(130_000)])
    const enrolled = await tldwClient.getWorkspaceSources(workspaceId, { requestScope: scope.requestScope, signal: abortSignal })
    assertOwner(); abortSignal.throwIfAborted()
    const evidence = sources.flatMap((source, index) => {
      if (!record(source)) return []
      const meta = record(source.metadata) ? source.metadata : source
      const mediaId = Number(source.mediaId ?? source.media_id ?? meta.mediaId ?? meta.media_id)
      const candidates = enrolled.filter(item => item.workspace_id === workspaceId && item.media_id === mediaId)
      const quote = source.pageContent ?? source.content ?? source.text
      const citation = meta.citation_index ?? meta.citationIndex ?? index + 1
      if (candidates.length !== 1 || typeof quote !== "string" || !quote.trim() || !Number.isSafeInteger(citation) || Number(citation) < 1) return []
      const start = meta.start_char, end = meta.end_char
      return [{ id: String(citation), source_id: candidates[0].id, quote,
        ...(offset(start) && offset(end) && end > start ? { start_char: start, end_char: end } : {}),
        ...(typeof meta.navigation_version === "string" && meta.navigation_version ? { navigation_version: meta.navigation_version } : {}) }]
    })
    const response = await bgRequest({ path: "/api/v1/research-workspace/citations/verify", method: "POST",
      body: { workspace_id: workspaceId, text, api_provider: provider, model, evidence },
      ...requestScopeFields(scope.requestScope), abortSignal, timeoutMs: 130_000 })
    assertOwner(); abortSignal.throwIfAborted()
    const report = readClaimVerification(response, text)
    return report?.provider === provider && report.model === model ? report : unchecked()
  } catch {
    assertOwner()
    return unchecked()
  }
}
