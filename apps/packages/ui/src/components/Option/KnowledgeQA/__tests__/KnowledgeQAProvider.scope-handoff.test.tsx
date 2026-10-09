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
const search = vi.hoisted(() => ({
  run: vi.fn(),
  media: vi.fn(),
  notes: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    ragSourceHealth: async () => ({
      sources: ["media_db", "notes"].map((source_id) => ({
        source_id,
        available: true,
        searchable: true,
        index_status: "ready",
        item_count: null,
        indexed_count: null
      }))
    }),
    listMedia: (...args: unknown[]) => search.media(...args),
    listNotes: (...args: unknown[]) => search.notes(...args),
    fetchWithAuth: async () => new Response("[]", { status: 200 }),
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
  const context = useKnowledgeQA()
  const routeNavigate = useNavigate()
  React.useEffect(() => {
    qa = context
    navigate = routeNavigate
  }, [context, routeNavigate])
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
  search.media.mockResolvedValue({ pagination: { total: 0 } })
  search.notes.mockResolvedValue({ total: 0 })
  localStorage.clear()
  stored.loading = false
  stored.settings = DEFAULT_RAG_SETTINGS
  search.run.mockResolvedValue({ results: [], answer: "Old answer" })
})
describe("Knowledge QA source intent", () => {
  it.each([
    { setting: "sources", value: ["notes"], select: () => qa.updateSetting("sources", ["notes"]) },
    { setting: "include_media_ids", value: [7], select: () => qa.updateSetting("include_media_ids", [7]) },
    { setting: "include_note_ids", value: ["note-7"], select: () => qa.updateSetting("include_note_ids", ["note-7"]) },
  ])("keeps the initial state when selecting $setting before asking", async ({ setting, value, select }) => {
    mount()
    await waitFor(() => expect(qa.settings.enable_web_fallback).toBe(false))

    act(select)

    expect(qa.settings).toMatchObject({ [setting]: value })
    expect(qa).toMatchObject({
      query: "",
      resultQuery: null,
      hasSearched: false,
      isSearching: false,
      error: null,
      queryStage: "idle",
      currentThreadId: null,
      messages: [],
    })
    expect(search.run).not.toHaveBeenCalled()
  })

  it("retains completed zero-results after changing sources and clearing the question", async () => {
    search.run.mockResolvedValueOnce({ results: [], answer: null })
    mount()
    await waitFor(() => expect(qa.settings.enable_web_fallback).toBe(false))
    act(() => qa.setQuery("A question with no matching sources"))
    await act(async () => qa.search())
    expect(qa.queryStage).toBe("complete")

    act(() => {
      qa.setQuery("")
      qa.updateSetting("include_note_ids", ["note-7"])
    })

    expect(qa).toMatchObject({
      query: "",
      resultQuery: "A question with no matching sources",
      hasSearched: true,
      results: [],
      answer: null,
      error: null,
    })
    expect(search.run).toHaveBeenCalledTimes(1)
  })

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
  it.each(["3,7", "invalid"])(
    "blocks %s queries while an explicit arrival waits for defaults",
    async (ids) => {
      stored.loading = true
      const view = mount(`/knowledge?media_ids=${ids}`)
      search.run.mockImplementation(async (_query, options) => ({
        results: [],
        answer: options.include_media_ids?.length
          ? `Scoped ${options.include_media_ids.join(",")}`
          : "Whole-library answer",
      }))
      act(() => qa.setQuery("Ask before hydration"))
      await act(async () => qa.search())
      expect(qa.currentThreadId).toBeNull()
      expect(qa.messages).toEqual([])
      expect(qa.answer).toBeNull()
      expect(qa.isSearching).toBe(false)
      stored.loading = false
      view.rerender(
        <MemoryRouter>
          <KnowledgeQAProvider>
            <Probe />
          </KnowledgeQAProvider>
        </MemoryRouter>,
      )
      if (ids === "invalid") {
        await waitFor(() => expect(qa.error).toMatch(/source selection/i))
        act(() => qa.setQuery("Still invalid"))
        await act(async () => qa.search())
        expect(qa.answer).toBeNull()
        expect(qa.currentThreadId).toBeNull()
        act(() => {
          qa.updateSetting("sources", ["media_db"])
          qa.updateSetting("include_media_ids", [9])
        })
      } else {
        await waitFor(() =>
          expect(qa.settings.include_media_ids).toEqual([3, 7]),
        )
      }
      act(() => qa.setQuery("Ask the applied sources"))
      await act(async () => qa.search())
      expect(qa.answer).toBe(ids === "invalid" ? "Scoped 9" : "Scoped 3,7")
    },
  )
  it("does not create a new topic ahead of an explicit arrival", async () => {
    stored.loading = true
    const view = mount("/knowledge?media_ids=3,7")
    await act(async () => expect(await qa.startNewTopic()).toBeNull())
    expect(qa.currentThreadId).toBeNull()
    stored.loading = false
    view.rerender(
      <MemoryRouter>
        <KnowledgeQAProvider>
          <Probe />
        </KnowledgeQAProvider>
      </MemoryRouter>,
    )
    await waitFor(() => expect(qa.settings.include_media_ids).toEqual([3, 7]))
  })
  it.each(["7", "invalid"])(
    "does not query the previous scope while a same-route %s arrival is pending",
    async (ids) => {
      const view = mount("/knowledge?media_ids=3")
      await waitFor(() => expect(qa.settings.include_media_ids).toEqual([3]))
      stored.loading = true
      act(() => navigate(`/knowledge?media_ids=${ids}`))
      act(() => qa.setQuery("Ask the arriving selection"))
      await act(async () => qa.search())
      expect(qa.answer).toBeNull()
      expect(qa.currentThreadId).toBeNull()
      expect(qa.messages).toEqual([])
      stored.loading = false
      view.rerender(
        <MemoryRouter>
          <KnowledgeQAProvider>
            <Probe />
          </KnowledgeQAProvider>
        </MemoryRouter>,
      )
      await waitFor(() =>
        expect(qa.settings.include_media_ids).toEqual(
          ids === "invalid" ? [] : [7],
        ),
      )
      if (ids === "invalid") expect(qa.error).toMatch(/source selection/i)
    },
  )
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

it("applies captured notes after hydration and replaces them on same-route arrival", async () => {
  const noteId = "12345678-1234-4234-8234-123456789abc"
  stored.loading = true
  const view = mount(`/knowledge?note_ids=${noteId}`)
  act(() => qa.setQuery("Too early"))
  await act(async () => qa.search())
  expect(qa.answer).toBeNull()
  stored.loading = false
  view.rerender(
    <MemoryRouter>
      <KnowledgeQAProvider>
        <Probe />
      </KnowledgeQAProvider>
    </MemoryRouter>
  )
  await waitFor(() =>
    expect(qa.settings).toMatchObject({
      sources: ["notes"],
      include_media_ids: [],
      include_note_ids: [noteId]
    })
  )
  act(() => navigate(`/knowledge?note_ids=${noteId}&media_ids=`))
  await waitFor(() => expect(qa.error).toMatch(/source selection/i))
  act(() => qa.setQuery("Invalid mixed scope"))
  await act(async () => qa.search())
  expect(qa.answer).toBeNull()
})

it("refreshes empty personal counts after current-owner ingest only", async () => {
  mount()
  await waitFor(() =>
    expect(qa.sourceHealth.bySource.media_db?.itemCount).toBe(0)
  )
  search.media.mockResolvedValue({ pagination: { total: 1 } })
  act(() =>
    window.dispatchEvent(
      new CustomEvent("tldw:quick-ingest-complete", {
        detail: { isCurrent: () => false }
      })
    )
  )
  expect(qa.sourceHealth.bySource.media_db?.itemCount).toBe(0)
  act(() =>
    window.dispatchEvent(
      new CustomEvent("tldw:quick-ingest-complete", {
        detail: { isCurrent: () => true }
      })
    )
  )
  await waitFor(() =>
    expect(qa.sourceHealth.bySource.media_db?.itemCount).toBe(1)
  )
  expect(qa.sourceHealth.bySource.media_db?.embeddingStatus).toBe("unknown")
})

it("keeps service availability separate when personal counts fail", async () => {
  search.media.mockRejectedValue(new Error("Media count unavailable"))
  mount()
  await waitFor(() => expect(qa.sourceHealth.loading).toBe(false))
  await waitFor(() =>
    expect(qa.sourceHealth.bySource.media_db?.available).toBe(true)
  )
  expect(qa.sourceHealth.bySource.media_db?.itemCount).toBeNull()
  expect(qa.sourceHealth.personalContentError).toMatch(
    /counts could not be loaded/i
  )
  expect(qa.sourceHealth.error).toBeNull()
})

it('retains the answered question when the editable search input changes', async () => {
  mount()
  await waitFor(() => expect(qa.settings.enable_web_fallback).toBe(false))
  act(() => qa.setQuery('Question one'))
  await act(async () => { await qa.search() })
  expect(qa.answer).toBe('Old answer')
  act(() => qa.setQuery('Question two, not searched'))
  expect(qa.resultQuery).toBe('Question one')
  expect(qa.query).toBe('Question two, not searched')
})
