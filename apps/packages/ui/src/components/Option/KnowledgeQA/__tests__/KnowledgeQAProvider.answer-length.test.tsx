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
  beforeEach(() => {
    vi.clearAllMocks()
    latestContext = null
    ragSearchMock.mockResolvedValue({
      results: [{ id: "doc-1" }],
      answer: "Adjusted answer",
    })
  })

  it("reruns the current query with per-request token limit override", async () => {
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
      await latestContext!.rerunWithTokenLimit(420)
    })

    expect(ragSearchMock).toHaveBeenCalledWith(
      "expand this topic",
      expect.objectContaining({
        max_generation_tokens: 420,
        enable_generation: true,
      })
    )
  })
})
