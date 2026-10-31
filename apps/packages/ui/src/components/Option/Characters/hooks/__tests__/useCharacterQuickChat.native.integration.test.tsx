import "@/hooks/chat/__tests__/native-plasmo-storage-fixture"
import { writeFileSync } from "node:fs"
import { act, renderHook, waitFor } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import type { BgRequestInit } from "@/services/background-proxy"
import type { HistorySelectionCaptureV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import { parseRoleplayResumeCompletionRequest } from "@/services/tldw/roleplay-resume"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"

// Opt-in owned local fixture only. The server calls its local mock provider, never a paid provider.
const fixture = vi.hoisted(() => ({ url: process.env.TLDW_E2E_SERVER_URL ?? "", apiKey: process.env.TLDW_E2E_API_KEY ?? "",
  characterId: Number(process.env.TLDW_E2E_CHARACTER_ID),
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

it.skipIf(process.env.TLDW_QUICK_CHAT_NATIVE_FIXTURE !== "1")("saves private Quick turns on original frozen authority after live setup clears and cold retains the next input", async () => {
  expect(new URL(fixture.url).hostname).toMatch(/^(127\.0\.0\.1|localhost)$/)
  expect(fixture.apiKey).toBeTruthy()
  expect(Number.isInteger(fixture.characterId) && fixture.characterId > 0).toBe(true)
  const actorSettings = await import("@/services/actor-settings")
  const sceneRead = vi.spyOn(actorSettings, "getActorSceneForChat")
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
  let model: string | null = "openai:gpt-4.1-mini"
  let models = [{ model: "gpt-4.1-mini", provider: "openai", is_configured: true }]
  const popup = renderHook(() => useCharacterQuickChat({ t, activeQuickChatModel: model, isServerConnected: true, availableModels: models }))
  await waitFor(() => expect(popup.result.current.quickChatOwnerReady).toBe(true))
  await act(async () => { await popup.result.current.openQuickChat({ id: fixture.characterId }) })
  await waitFor(() => expect(popup.result.current.quickChatReadiness.canStart).toBe(true))
  for (const input of ["  FIRST PRIVATE TURN\nExact spacing  ", "SECOND PRIVATE TURN  "]) {
    if (popup.result.current.quickChatSessionId) {
      act(() => { model = null; models = []; useStoreChatModelSettings.setState({ apiProvider: "unrelated-global-provider", minP: 0.1 }); popup.rerender() })
      expect(popup.result.current.quickChatReadiness.canStart).toBe(true)
    }
    act(() => popup.result.current.setQuickChatDraft(input))
    await act(async () => { await popup.result.current.sendQuickChatMessage() })
    expect(popup.result.current.quickChatError).toBeNull()
    expect(popup.result.current.quickChatDraft).toBe("")
    expect(popup.result.current.quickChatNeedsReview).toBe(false)
  }
  expect(sceneRead).toHaveBeenCalledOnce()
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
  expect(path.selected_content[0].message).toBe("  FIRST PRIVATE TURN\nExact spacing  ")
  expect(path.selected_content[2].message).toBe("SECOND PRIVATE TURN  ")
  expect(fixture.hits.filter(hit => hit.path.split("?")[0] === "/api/v1/chat/completions")).toHaveLength(0)
  const turns = fixture.hits.filter(hit => hit.path.includes("complete-v2"))
  expect(turns).toHaveLength(2)
  for (const turn of turns) {
    expect(turn.body).not.toHaveProperty("model")
    expect(turn.body).not.toHaveProperty("api_provider")
  }
  const appends = fixture.hits.filter(hit => hit.method === "POST" && hit.path.endsWith("/messages"))
  expect(appends).toHaveLength(2)
  expect(appends.map(hit => hit.body.content)).toEqual(["  FIRST PRIVATE TURN\nExact spacing  ", "SECOND PRIVATE TURN  "])
  expect(turns.map(hit => parseRoleplayResumeCompletionRequest(hit.body).tldw_roleplay_resume_v1.input_message_id)).toEqual(appends.map(hit => hit.body.message_id))
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
    legacyTurns: 0, savedFrozenTurns: turns.length, globalModelCleared: true, coldDraft: retained?.content, coreSourceUnchanged: true,
    limits: "Actual Quick/native sender/API client/controller and real owned server; scoped HTTP/SSE and account-verification IO adapters, existing profile/bookmark database IO adapter and actual storage shim; fresh Scene reads remain real. No packaged/native browser or progressive transport timing claim." }, null, 2), { mode: 0o600 })
  cold.unmount()
}, 60000)
