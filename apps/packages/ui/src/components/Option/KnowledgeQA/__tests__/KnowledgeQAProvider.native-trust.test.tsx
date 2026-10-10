import { TEST_HISTORY_STORAGE_KEY } from "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"
import knowledgeEn from "@/assets/locale/en/knowledge.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import { AnswerPanel } from "../AnswerPanel"
import { ExportDialog } from "../ExportDialog"
import { readKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"

const fetchWithAuthMock = vi.fn()
const ragSearchMock = vi.fn()
const createNoteMock = vi.fn()
const queuePrefillMock = vi.fn()
const noteId = "352a4786-6f58-46ac-8119-388fb34ea726"
const answer =
  "The captured note explains the research workflow. It preserves the source for later analysis."
const excerpt =
  "The research workflow captures a source as a note and preserves its evidence for later analysis."
const document = {
  id: noteId,
  content: excerpt,
  source: "notes_db",
  sourceType: "notes",
  metadata: {
    title: "Captured source",
    note_id: noteId,
    source_type: "notes",
    evidence_origin: "local_library",
  },
}
// Actual native response shape: hard spans and a bibliography do not add inline answer markers.
const response = {
  generated_answer: answer,
  documents: [document],
  metadata: {
    knowledge_trust: {
      state: "cited_answer",
      reason_codes: [],
      evidence_origin: "local_library",
    },
    hard_citations: {
      sentences: [
        {
          text: "The captured note explains the research workflow.",
          citations: [{ doc_id: noteId, start: 0, end: 77 }],
        },
        {
          text: "It preserves the source for later analysis.",
          citations: [{ doc_id: noteId, start: 0, end: 71 }],
        },
      ],
      coverage: 1,
      total: 2,
      supported: 2,
    },
    inline_citations: { "[1]": noteId },
    citation_map: { [noteId]: [noteId] },
  },
  citations: [
    { type: "academic", formatted: "Captured source. Research notes." },
    {
      type: "chunk",
      chunk_id: noteId,
      source_document_id: noteId,
      text_snippet: excerpt,
      usage_context: "Background context",
    },
  ],
}

vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: async () => ({ id: "test-owner", is_active: true }),
  },
}));
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [undefined] }))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => "test-owner",
}))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ open: vi.fn() }),
}))
vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "session",
  submitExplicitFeedback: vi.fn(),
}))
vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: vi.fn(),
}))
vi.mock("@/utils/research-workspace-prefill", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/utils/research-workspace-prefill")
  >()),
  queueResearchWorkspacePrefill: (...args: unknown[]) =>
    queuePrefillMock(...args),
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    fetchWithAuth: (...args: unknown[]) => fetchWithAuthMock(...args),
    ragSearch: (...args: unknown[]) => ragSearchMock(...args),
    createNote: (...args: unknown[]) => createNoteMock(...args),
    searchCharacters: vi
      .fn()
      .mockResolvedValue([{ id: 7, name: "Helpful AI Assistant" }]),
    listCharacters: vi.fn().mockResolvedValue([]),
    createChat: vi.fn().mockResolvedValue({ id: "thread-1", version: 1 }),
    getChat: vi.fn().mockResolvedValue({ version: 1 }),
    addChatMessage: vi.fn(async (_thread, payload) => ({
      id: `${payload.role}-remote`,
    })),
  },
}))

let current: ReturnType<typeof useKnowledgeQA>
function Probe() {
  const context = useKnowledgeQA();
  React.useEffect(() => {
    current = context;
  }, [context]);
  const [exportOpen, setExportOpen] = React.useState(false)
  return (
    <>
      <AnswerPanel />
      <button onClick={() => setExportOpen(true)}>Open export</button>
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} />
    </>
  )
}
async function mount() {
  const i18n = createInstance().use(ICUWithInterpolation)
  await i18n.init({
    lng: "en",
    fallbackLng: "en",
    defaultNS: "knowledge",
    resources: { en: { knowledge: knowledgeEn } },
  })
  render(
    <I18nextProvider i18n={i18n}>
      <MemoryRouter>
        <KnowledgeQAProvider>
          <Probe />
        </KnowledgeQAProvider>
      </MemoryRouter>
    </I18nextProvider>
  )
  await waitFor(() => expect(current.historyHydrated).toBe(true))
}
async function search() {
  act(() => {
    current.setQuery("Explain the captured source")
  })
  await act(async () => {
    await current.search()
  })
}

describe("native response trust continuity", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    ragSearchMock.mockResolvedValue(response)
    queuePrefillMock.mockResolvedValue(undefined)
    createNoteMock.mockResolvedValue({
      id: "7b147e55-f90b-4381-8703-0948959ec122",
    })
    fetchWithAuthMock.mockImplementation(async (path: string) => ({
      ok: true,
      status: 200,
      json: async () =>
        path.includes("/messages-with-context") ||
        path.includes("/conversations?")
          ? []
          : { success: true, version: 1, keywords: [] },
      text: async () => "",
    }))
  })

  it("keeps actual native metadata consistent across visible, saved, exported and Research trust", async () => {
    await mount()
    await search()
    expect(current.answerTrustState).toBe("uncited_degraded_answer")
    expect(current.answerTrustReasonCodes).toEqual(["missing_citations"])
    expect(current.citations).toEqual([])
    expect(current.results[0].id).toBe(noteId)
    expect(
      screen.queryByText("Answer status: Cited answer")
    ).not.toBeInTheDocument()
    expect(screen.getByLabelText("Answer trust summary")).toHaveTextContent(
      "0 cited"
    )
    expect(screen.getByLabelText("Answer trust summary")).toHaveTextContent(
      "Trust: Uncited answer"
    )
    expect(screen.getByLabelText("Answer trust summary")).toHaveTextContent(
      "missing citations that map to returned sources"
    )
    expect(current.searchHistory[0].trustState).toBe("uncited_degraded_answer")
    expect(
      JSON.parse(localStorage.getItem(TEST_HISTORY_STORAGE_KEY) || "[]")[0]
        .trustState
    ).toBe("uncited_degraded_answer")
    const contextWrite = fetchWithAuthMock.mock.calls.find(
      ([path, init]) => path.includes("/rag-context") && init?.method === "POST"
    )
    expect(contextWrite).toBeDefined()
    const savedContext = JSON.parse(contextWrite![1].body).rag_context
    expect(savedContext).toMatchObject({
      trust_state: "uncited_degraded_answer",
      knowledge_trust: { state: "uncited_degraded_answer" },
      retrieved_documents: [{ id: noteId, excerpt }],
    })

    fetchWithAuthMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => "",
      json: async () => [
        {
          id: "assistant-remote",
          role: "assistant",
          content: answer,
          rag_context: savedContext,
        },
      ],
    })
    await act(async () => {
      await current.restoreFromHistory(current.searchHistory[0])
    })
    expect(current.answerTrustState).toBe("uncited_degraded_answer")
    expect(current.searchHistory[0].trustState).toBe("uncited_degraded_answer")
    expect(current.results[0].id).toBe(noteId)
    expect(screen.getByLabelText("Answer trust summary")).toHaveTextContent(
      "Trust: Uncited answer"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Continue in Research Workspace" })
    )
    await waitFor(() => expect(queuePrefillMock).toHaveBeenCalledOnce())
    expect(queuePrefillMock.mock.calls[0][0]).toMatchObject({
      answerTrustState: "uncited_degraded_answer",
      citations: [],
    })

    fireEvent.click(screen.getByRole("button", { name: "Open export" }))
    expect(screen.getByRole("button", { name: "Export" })).toBeDisabled()
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: /I understand this unsupported draft/i,
      })
    )
    fireEvent.click(screen.getByRole("button", { name: "Export" }))
    await waitFor(() =>
      expect(
        screen.getByText(
          (_, el) =>
            el?.tagName === "PRE" &&
            (el.textContent || "").includes("Answer status: Uncited answer")
        )
      ).toBeInTheDocument()
    )
    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
    await waitFor(() => expect(createNoteMock).toHaveBeenCalledOnce())
    expect(
      readKnowledgeNoteProvenance(createNoteMock.mock.calls[0][0])
    ).toMatchObject({ trust_state: "uncited_degraded_answer" })
  })

  it("keeps valid inline citation support cited", async () => {
    ragSearchMock.mockResolvedValue({
      ...response,
      generated_answer: `${answer} [1]`,
    })
    await mount()
    await search()
    expect(current.answerTrustState).toBe("cited_answer")
    expect(current.citations).toEqual([
      expect.objectContaining({ index: 1, documentId: noteId }),
    ])
    expect(screen.getByLabelText("Answer trust summary")).toHaveTextContent(
      "1 cited"
    )
  })

  it("does not keep backend cited trust when a cited source is outside the selected scope", async () => {
    ragSearchMock.mockResolvedValue({
      ...response,
      generated_answer: "Hidden result [2]",
      documents: [
        document,
        {
          id: "hidden",
          content: "Other evidence",
          sourceType: "media_db",
          metadata: { source_id: "7" },
        },
      ],
    })
    await mount()
    act(() => {
      current.updateSetting("sources", ["notes"])
    })
    await search()
    expect(current.results.map((result) => result.id)).toEqual([noteId])
    expect(current.citations).toEqual([])
    expect(current.answerTrustState).toBe("uncited_degraded_answer")
  })

  it.each([
    { stored: "cited_answer", expected: "uncited_degraded_answer" },
    { stored: "unknown_trust", expected: "unknown_trust" },
    {
      stored: "no_answer_insufficient_evidence",
      expected: "no_answer_insufficient_evidence",
    },
    { stored: undefined, expected: "unknown_trust" },
  ])(
    "normalizes saved $stored without promoting older or qualified trust",
    async ({ stored, expected }) => {
      fetchWithAuthMock.mockImplementation(async (path: string) => ({
        ok: true,
        status: 200,
        text: async () => "",
        json: async () =>
          path.includes("/messages-with-context")
            ? [
                {
                  id: "old-assistant",
                  role: "assistant",
                  content: answer,
                  rag_context: {
                    search_query: "Captured source",
                    generated_answer: answer,
                    trust_state: stored,
                    trust_reason_codes:
                      stored === "no_answer_insufficient_evidence"
                        ? ["low_relevance"]
                        : [],
                    trust_evidence_origin: "local_library",
                    retrieved_documents: [
                      { id: noteId, excerpt, source_type: "notes" },
                    ],
                  },
                },
              ]
            : [],
      }))
      await mount()
      await act(async () => {
        await current.selectThread("old-thread")
      })
      expect(current.answerTrustState).toBe(expected)
      expect(current.citations).toEqual([])
      expect(current.results[0].id).toBe(noteId)
    }
  )
})
