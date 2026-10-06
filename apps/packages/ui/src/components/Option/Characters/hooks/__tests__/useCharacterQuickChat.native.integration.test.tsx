import { writeFileSync } from "node:fs"
import { act, renderHook, waitFor } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import type { BgRequestInit } from "@/services/background-proxy"
import type { HistorySelectionCaptureV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"

// Opt-in owned local fixture only. The server calls its local mock provider, never a paid provider.
const fixture = vi.hoisted(() => ({ url: "http://127.0.0.1:19765", apiKey: "chat-ux-first-save-review-19765",
  hits: [] as Array<{ method: string; path: string; body: Record<string, unknown> }>, navigate: vi.fn() }))
const scope = { userId: "1", config: { serverUrl: fixture.url, authMode: "single-user" as const, apiKey: fixture.apiKey,
  expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope("single-user", fixture.apiKey) } }
vi.mock("react-router-dom", () => ({ useNavigate: () => fixture.navigate }))
vi.mock("@/db/dexie/schema", async () => ({ db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory }))
vi.mock("@/services/service-prompts", async original => ({
  ...await original<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => scope,
  subscribeToServicePromptConfigChanges: () => () => {},
  loadServicePromptSnapshot: async () => ({ requestScope: scope, definitions: {}, scopeKey: "owned-quick-fixture",
    capability: "unchecked", scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: () => {} })
}))
vi.mock("@/services/background-proxy", async original => {
  const actual = await original<typeof import("@/services/background-proxy")>()
  const request = async (init: BgRequestInit) => {
    expect(init.servicePromptConfig?.serverUrl).toBe(fixture.url)
    expect(init.servicePromptConfig?.expectedUserId).toBe("1")
    expect(init.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
    fixture.hits.push({ method: init.method ?? "GET", path: init.path, body: structuredClone(init.body ?? {}) })
    const response = await fetch(fixture.url + init.path, { method: init.method ?? "GET", signal: init.abortSignal,
      headers: { "Content-Type": "application/json", ...init.headers, "X-API-KEY": fixture.apiKey },
      ...(init.body ? { body: JSON.stringify(init.body) } : {}) })
    if (!response.ok) throw new Error(`Owned fixture ${init.path}: ${response.status}: ${(await response.text()).slice(0, 300)}`)
    return response
  }
  return { ...actual,
    bgRequest: async (init: BgRequestInit) => {
      const response = await request(init)
      return init.returnResponse ? response : response.json()
    },
    // Protocol IO adapter: deliver the real server SSE data frames; progressive browser timing is a separate gate.
    bgStream: async function* (init: BgRequestInit) {
      const response = await request(init)
      for (const line of (await response.text()).split(/\r?\n/)) {
        if (line.startsWith("data:") && line.slice(5).trim() !== "[DONE]") yield line.slice(5).trim()
      }
    }
  }
})

it.skipIf(process.env.TLDW_QUICK_CHAT_NATIVE_FIXTURE !== "1")("saves two private Quick turns as one canonical native path and cold retains the next input", async () => {
  const { useCharacterQuickChat } = await import("../useCharacterQuickChat")
  const { useStoreMessageOption } = await import("@/store/option")
  const { useStoreChatModelSettings } = await import("@/store/model")
  const { useMcpToolsStore } = await import("@/store/mcp-tools")
  const { tldwClient } = await import("@/services/tldw/TldwApiClient")
  const { useComposerText } = await import("@/components/Chat/composer/hooks/useComposerText")
  const { conversationDraftKey } = await import("@/utils/chat-draft-address")
  localStorage.clear(); sessionStorage.clear(); fixture.hits.length = 0
  const tails = new Map<string, Promise<unknown>>()
  Object.defineProperty(navigator, "locks", { configurable: true, value: { request: (key: string, work: () => Promise<unknown>) => {
    const pending = (tails.get(key) ?? Promise.resolve()).then(work); tails.set(key, pending.catch(() => {})); return pending
  } } })
  useStoreChatModelSettings.getState().reset(); useMcpToolsStore.getState().setTools([])
  useStoreMessageOption.getState().startNewChatDraft()
  const source = useStoreMessageOption.getState()
  const t = ((key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key) as TFunction
  const popup = renderHook(() => useCharacterQuickChat({ t, activeQuickChatModel: "custom-openai-api:gpt-4o-mini", isServerConnected: true,
    availableModels: [{ model: "gpt-4o-mini", provider: "custom_openai_api", is_configured: true }] }))
  await waitFor(() => expect(popup.result.current.quickChatOwnerReady).toBe(true))
  await act(async () => { await popup.result.current.openQuickChat({ id: 4 }) })
  await waitFor(() => expect(popup.result.current.quickChatReadiness.canStart).toBe(true))
  for (const input of ["  FIRST PRIVATE TURN\nExact spacing  ", "SECOND PRIVATE TURN  "]) {
    act(() => popup.result.current.setQuickChatDraft(input))
    await act(async () => { await popup.result.current.sendQuickChatMessage() })
    expect(popup.result.current.quickChatError).toBeNull()
    expect(popup.result.current.quickChatDraft).toBe("")
    expect(popup.result.current.quickChatNeedsReview).toBe(false)
  }
  const id = popup.result.current.quickChatSessionId!
  expect(id).toBeTruthy()
  const rows = await tldwClient.listChatMessages(id, {}, { requestScope: scope, fresh: true })
  expect(rows.map(row => row.sender)).toEqual(["user", "assistant", "user", "assistant"])
  const selectedView = fixture.hits.filter(hit => hit.path.includes("/history/selection")).at(-1)!.body.view as HistoryViewSelectionV1
  const base = await tldwClient.captureHistorySelection(id, { purpose: "send", view: {
    ...selectedView, view_session_id: crypto.randomUUID(), cursor: { kind: "empty" }, selection_revision: 0
  } }, { requestScope: scope }) as HistorySelectionCaptureV1
  expect(base.status).toBe("captured")
  const nodes = base.snapshot.nodes
  expect(nodes).toHaveLength(4)
  expect(nodes.filter(node => node.parent_id === null)).toHaveLength(1)
  const leaf = nodes.find(node => !nodes.some(child => child.parent_id === node.id))!
  const path = await tldwClient.captureHistorySelection(id, { purpose: "send", view: {
    ...base.view, cursor: { kind: "after_message", message_id: leaf.id }, selection_revision: 1
  } }, { requestScope: scope }) as HistorySelectionCaptureV1
  expect(path.status).toBe("captured")
  expect(path.rows.map(row => row.role)).toEqual(["user", "assistant", "user", "assistant"])
  // Existing shared request/server normalization trims outer whitespace; exact unsent retention is asserted below.
  expect(path.selected_content[0].message).toBe("FIRST PRIVATE TURN\nExact spacing")
  expect(path.selected_content[2].message).toBe("SECOND PRIVATE TURN")
  expect(fixture.hits.filter(hit => hit.path.split("?")[0] === "/api/v1/chat/completions")).toHaveLength(2)
  const turns = fixture.hits.filter(hit => hit.path.split("?")[0] === "/api/v1/chat/completions")
  expect((turns[1].body.tldw_history_selection_v1 as { messages: unknown[] }).messages).toHaveLength(2)
  expect(fixture.hits.some(hit => hit.path.includes("complete-v2"))).toBe(false)
  expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source.chatDraftAddress)
  expect(useStoreMessageOption.getState().chatDraftRevision).toBe(source.chatDraftRevision)
  act(() => popup.result.current.setQuickChatDraft("  NEXT UNSENT\nTrailing spaces  "))
  await act(async () => { await popup.result.current.closeQuickChat() })
  popup.unmount(); sessionStorage.clear()
  const cold = renderHook(() => useComposerText({ draftKey: conversationDraftKey("tldw:playgroundChatDraft", source.chatDraftAddress!),
    draftAddress: source.chatDraftAddress!, recoverySurface: "tldw:playgroundChatDraft", textareaRef: { current: null } }))
  await waitFor(() => expect(cold.result.current.draftReady).toBe(true))
  const retained = (await cold.result.current.listRetainedDrafts()).drafts.find(draft => draft.creationAttempt?.chatId === id)
  expect(retained).toMatchObject({ content: "  NEXT UNSENT\nTrailing spaces  ", creationAttempt: { state: "settled", chatId: id } })
  expect(fixture.hits.some(hit => hit.method === "DELETE")).toBe(false)
  writeFileSync("/private/tmp/chat-quick-native-runtime-proof.json", JSON.stringify({ chatId: id, roles: path.rows.map(row => row.role), roots: 1,
    turns: turns.length, historyMembersInSecondTurn: 2, coldDraft: retained?.content, coreSourceUnchanged: true,
    sendFormattingLimit: "Existing shared request/server normalization trims outer whitespace; retained/copied unsent input remains exact.",
    limits: "Actual Quick/native sender/API client/controller and real owned server; scoped HTTP/SSE and account-verification IO adapters, existing profile/bookmark database IO adapter. No packaged/native browser or progressive transport timing claim." }, null, 2))
  cold.unmount()
}, 60000)
