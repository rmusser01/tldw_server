import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"

const ragSearchMock = vi.fn()

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    if (key === "ff_knowledgeQaStreaming") return [false]
    return [defaultValue]
  },
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: vi.fn(),
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    fetchWithAuth: vi.fn().mockResolvedValue({
      ok: false,
      json: async () => [],
      text: async () => "",
    }),
    ragSearch: (...args: unknown[]) => ragSearchMock(...args),
    ragSearchStream: vi.fn(),
  },
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  latestContext = useKnowledgeQA()
  return null
}

describe("KnowledgeQAProvider answer length override", () => {
  it("keeps nonstream clarification distinct from a completed empty search", async () => {
    ragSearchMock.mockResolvedValue({ results: [], generated_answer: "Which document?", metadata: { clarification: { required: true } } })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("local-clarification") })
    act(() => latestContext!.setQuery("What does this propose?"))
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.queryStage).toBe("clarification")
    expect(latestContext!.messages.at(-1)?.ragContext?.clarification_required).toBe(true)
  })
  beforeEach(() => {
    vi.clearAllMocks()
    latestContext = null
    ragSearchMock.mockResolvedValue({
      results: [{ id: "doc-1" }],
      answer: "Adjusted answer",
    })
  })

  it.each([420, 4096, 8192])("reruns the current query with per-request token limit %s", async (tokens) => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-token-override")
    })

    act(() => {
      latestContext!.setQuery("expand this topic")
    })

    await act(async () => {
      await latestContext!.rerunWithTokenLimit(tokens)
    })

    expect(ragSearchMock).toHaveBeenCalledWith(
      "expand this topic",
      expect.objectContaining({
        max_generation_tokens: tokens,
        enable_generation: true,
      })
    )
  })
})
