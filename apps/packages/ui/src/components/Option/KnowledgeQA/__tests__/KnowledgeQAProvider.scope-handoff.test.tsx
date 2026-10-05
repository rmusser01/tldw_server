import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { MemoryRouter, useNavigate } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"

const stored = vi.hoisted(() => ({
  loading: false,
  settings: {} as Record<string, unknown>,
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string) => [
    key === "ragSearchSettingsV2" ? stored.settings : undefined,
    vi.fn(),
    { isLoading: stored.loading },
  ],
}))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ open: vi.fn() }),
}))
const search = vi.hoisted(() => ({ run: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    fetchWithAuth: async () => ({ ok: false, json: async () => [] }),
    ragSearch: (...args: unknown[]) => search.run(...args),
    createChat: async () => ({ id: "thread-1" }),
    deleteChat: async () => undefined,
    addChatMessage: async () => ({ id: "message-1" }),
    searchCharacters: async () => [{ id: 1, name: "Helpful AI Assistant" }],
    getChat: async () => ({ version: 1 }),
  },
}))
let qa: ReturnType<typeof useKnowledgeQA>
let navigate: ReturnType<typeof useNavigate>
function Probe() {
  qa = useKnowledgeQA()
  navigate = useNavigate()
  return null
}
function mount(path = "/knowledge") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <KnowledgeQAProvider>
        <Probe />
      </KnowledgeQAProvider>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  stored.loading = false
  stored.settings = DEFAULT_RAG_SETTINGS
  search.run.mockResolvedValue({ results: [], answer: "Old answer" })
})
describe("Knowledge QA source intent", () => {
  it.each(["fast", "balanced", "thorough", "custom"] as const)(
    "retains exact sources and the answer model when choosing %s",
    async (preset) => {
      mount()
      await waitFor(() => expect(qa.settings.enable_web_fallback).toBe(false))
      act(() => {
        qa.updateSetting("sources", ["media_db", "notes"])
        qa.updateSetting("include_media_ids", [3, 7])
        qa.updateSetting("include_note_ids", ["note-7"])
        qa.updateSetting("collection_id", 2)
        qa.updateSetting("keyword_filter", "research")
        qa.updateSetting("generation_provider", "ollama")
        qa.updateSetting("generation_model", "chosen-model")
        qa.updateSetting("enable_web_fallback", false)
      })
      act(() => qa.setPreset(preset))
      expect(qa.settings).toMatchObject({
        sources: ["media_db", "notes"],
        include_media_ids: [3, 7],
        include_note_ids: ["note-7"],
        collection_id: 2,
        keyword_filter: "research",
        generation_provider: "ollama",
        generation_model: "chosen-model",
        enable_web_fallback: false,
      })
    },
  )
  it("applies added IDs after defaults finish loading", async () => {
    stored.settings = {
      ...DEFAULT_RAG_SETTINGS,
      corpus: "old-corpus",
      index_namespace: "old-index",
    }
    stored.loading = true
    const view = mount("/knowledge?media_ids=3%2C7")
    stored.loading = false
    view.rerender(
      <MemoryRouter initialEntries={["/knowledge?media_ids=3%2C7"]}>
        <KnowledgeQAProvider>
          <Probe />
        </KnowledgeQAProvider>
      </MemoryRouter>,
    )
    await waitFor(() =>
      expect(qa.settings).toMatchObject({
        sources: ["media_db"],
        include_media_ids: [3, 7],
        include_note_ids: [],
        collection_id: null,
        keyword_filter: "",
        corpus: "",
        index_namespace: "",
      }),
    )
  })
  it("replaces scope on same-route arrival and drops late work and old output", async () => {
    mount()
    await waitFor(() => expect(qa.settings.enable_web_fallback).toBe(false))
    await act(async () => qa.selectThread("local-test"))
    act(() => qa.setQuery("old question"))
    await act(async () => qa.search())
    expect(qa.answer).toBe("Old answer")
    let finish!: (value: unknown) => void
    let signal: AbortSignal | undefined
    search.run.mockImplementationOnce(
      (_query: string, options: { signal: AbortSignal }) => {
        signal = options.signal
        return new Promise((resolve) => {
          finish = resolve
        })
      },
    )
    act(() => {
      void qa.search()
    })
    await waitFor(() => expect(signal).toBeDefined())
    act(() => navigate("/knowledge?media_ids=7"))
    await waitFor(() => expect(qa.settings.include_media_ids).toEqual([7]))
    expect(signal!.aborted).toBe(true)
    expect(qa.answer).toBeNull()
    expect(qa.results).toEqual([])
    expect(qa.messages).toEqual([])
    expect(qa.currentThreadId).toBeNull()
    await act(async () => {
      finish({ results: [], answer: "Late answer" })
      await Promise.resolve()
    })
    expect(qa.answer).toBeNull()
  })
  it.each(["", "invalid", "1,0", "1.5", "9007199254740992"])(
    "rejects explicit invalid scope %s without querying the whole library",
    async (ids) => {
      mount(`/knowledge?media_ids=${ids}`)
      await waitFor(() => expect(qa.error).toMatch(/source selection/i))
      act(() => qa.setQuery("Do not broaden"))
      search.run.mockImplementation(() => {
        throw new Error("Whole-library search reached")
      })
      await act(async () => qa.search())
      expect(qa.error).toMatch(/source selection/i)
      expect(qa.settings.sources).toEqual([])
    },
  )
})
