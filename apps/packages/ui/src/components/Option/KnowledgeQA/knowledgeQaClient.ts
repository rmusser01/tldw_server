import { tldwClient, type ScopedRequestOptions } from "@/services/tldw/TldwApiClient"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { createServicePromptScopeChangedError, isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { submitExplicitFeedback, type ExplicitFeedbackRequest } from "@/services/feedback"

/** Keep every step of a QA operation on the owner that started it. */
export function createKnowledgeQaClient(snapshot: ServicePromptSnapshot | null, isCurrent: () => boolean) {
  const assertCurrent = (publicRequest = false) => {
    if (!isCurrent() || (!publicRequest && !snapshot) || snapshot?.scopeSignal.aborted) {
      throw createServicePromptScopeChangedError()
    }
  }
  const options = (signal?: AbortSignal): ScopedRequestOptions => {
    assertCurrent()
    return {
      requestScope: snapshot!.requestScope,
      signal: signal ? AbortSignal.any([signal, snapshot!.scopeSignal]) : snapshot!.scopeSignal,
    }
  }
  const run = async <T,>(request: () => Promise<T>, publicRequest = false): Promise<T> => {
    assertCurrent(publicRequest)
    const result = await request()
    assertCurrent(publicRequest)
    return result
  }
  return {
    initialize: () => run(async () => undefined),
    fetchWithAuth: (path: Parameters<typeof tldwClient.fetchWithAuth>[0], init?: Parameters<typeof tldwClient.fetchWithAuth>[1]) =>
      run(async () => {
        const response = await tldwClient.fetchWithAuth(path, { ...init, ...options(init?.signal) })
        // This API returns non-OK responses; scope denial must still cancel the
        // operation before optional metadata fallbacks can continue it.
        if (!response.ok && isRequestConfigScopeChangedError({ status: response.status, details: response.data })) {
          throw createServicePromptScopeChangedError()
        }
        return response
      }),
    searchCharacters: (query: string, params?: Record<string, unknown>) => run(() => tldwClient.searchCharacters(query, params, options())),
    listCharacters: (params?: Record<string, unknown>) => run(() => tldwClient.listCharacters(params, options())),
    createChat: (payload: Record<string, unknown>) => run(() => tldwClient.createChat(payload, options())),
    getChat: (id: string) => run(() => tldwClient.getChat(id, options())),
    deleteChat: (id: string) => run(() => tldwClient.deleteChat(id, options())),
    addChatMessage: (id: string, payload: Record<string, unknown>) => run(() => tldwClient.addChatMessage(id, payload, options())),
    ragSourceHealth: () => run(() => tldwClient.ragSourceHealth(options())),
    ragSearch: (query: string, settings?: Record<string, unknown> & { signal?: AbortSignal }) => run(() => tldwClient.ragSearch(query, { ...settings, ...options(settings?.signal) })),
    ragSearchStream: typeof tldwClient.ragSearchStream === "function" ? async function* (query: string, settings?: Record<string, unknown> & { signal?: AbortSignal }) {
      for await (const event of tldwClient.ragSearchStream(query, { ...settings, ...options(settings?.signal) })) {
        assertCurrent()
        yield event
      }
      assertCurrent()
    } : undefined,
    // Share tokens intentionally keep their existing unauthenticated read contract.
    resolveConversationShareLink: (token: string) => run(() => tldwClient.resolveConversationShareLink(token), true),
    createConversationShareLink: (id: string, payload?: Parameters<typeof tldwClient.createConversationShareLink>[1]) =>
      run(() => tldwClient.createConversationShareLink(id, payload, options())),
    revokeConversationShareLink: (id: string, shareId: string) => run(() => tldwClient.revokeConversationShareLink(id, shareId, options())),
    exportChatbook: (payload: Parameters<typeof tldwClient.exportChatbook>[0]) => run(() => tldwClient.exportChatbook(payload, options())),
    downloadChatbookExport: (id: string) => run(() => tldwClient.downloadChatbookExport(id, options())),
    createNote: (content: string, metadata?: Record<string, unknown>) => run(() => tldwClient.createNote(content, metadata, options())),
    submitSourceFeedback: (payload: ExplicitFeedbackRequest) => run(() => submitExplicitFeedback(payload, options())),
  }
}

export type KnowledgeQaClient = ReturnType<typeof createKnowledgeQaClient>
