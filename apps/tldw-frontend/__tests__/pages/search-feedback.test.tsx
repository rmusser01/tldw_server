import { render, screen, act, waitFor } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import React from "react"

import { useFeedback } from "@/hooks/useFeedback"
import { useImplicitFeedback } from "@/hooks/useImplicitFeedback"
import { useFeedbackStore } from "@/store/feedback"
import { useStoreMessageOption } from "@/store/option"

const mocks = vi.hoisted(() => ({
  submitExplicitFeedback: vi.fn().mockResolvedValue({ ok: true, feedback_id: "fb-search-1" }),
  submitImplicitFeedback: vi.fn().mockResolvedValue({ ok: true }),
  updateChatRating: vi.fn().mockResolvedValue({}),
  notificationError: vi.fn()
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "sess-search-test",
  submitExplicitFeedback: mocks.submitExplicitFeedback,
  submitImplicitFeedback: mocks.submitImplicitFeedback,
  updateChatRating: mocks.updateChatRating
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: mocks.notificationError })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

describe("Search/source feedback flows", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useFeedbackStore.setState({ entries: {} })
    useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true)
    window.localStorage.clear()
  })

  it("sends source-level explicit feedback with document/chunk identifiers", async () => {
    const source = {
      metadata: {
        document_id: "doc-search-1",
        chunk_id: "chunk-search-1",
        corpus: "media_db"
      }
    }

    const TestHarness = () => {
      const feedback = useFeedback({
        messageKey: "srv:M_search_1",
        conversationId: "C_search_1",
        messageId: "M_search_1",
        query: null
      })
      return (
        <button
          type="button"
          onClick={() => {
            void feedback.submitSourceThumb({
              sourceKey: "doc:doc-search-1",
              source,
              thumb: "up"
            })
          }}
        >
          submit-source-up
        </button>
      )
    }

    render(<TestHarness />)
    await act(async () => {
      screen.getByRole("button", { name: "submit-source-up" }).click()
    })

    await waitFor(() => {
      expect(mocks.submitExplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          feedback_type: "helpful",
          helpful: true,
          conversation_id: "C_search_1",
          message_id: "M_search_1",
          document_ids: ["doc-search-1"],
          chunk_ids: ["chunk-search-1"],
          corpus: "media_db"
        })
      )
    })
  })

  it("emits citation_used implicit event with source metadata", async () => {
    const source = {
      metadata: {
        document_id: "doc-search-2",
        chunk_id: "chunk-search-2",
        corpus: "media_db"
      }
    }

    const TestHarness = () => {
      const implicit = useImplicitFeedback({
        conversationId: "C_search_2",
        messageId: "M_search_2",
        query: "search query",
        sources: [source]
      })
      return (
        <button
          type="button"
          onClick={() => implicit.trackCitationUsed(source, 0)}
        >
          citation-used
        </button>
      )
    }

    render(<TestHarness />)
    await act(async () => {
      screen.getByRole("button", { name: "citation-used" }).click()
    })

    await waitFor(() => {
      expect(mocks.submitImplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "citation_used",
          conversation_id: "C_search_2",
          message_id: "M_search_2",
          doc_id: "doc-search-2",
          chunk_ids: ["chunk-search-2"],
          impression_list: ["chunk-search-2"],
          corpus: "media_db"
        })
      )
    })
  })
})
