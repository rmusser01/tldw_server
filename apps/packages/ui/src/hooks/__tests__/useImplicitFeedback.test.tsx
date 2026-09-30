import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useImplicitFeedback } from "../useImplicitFeedback"

const mocks = vi.hoisted(() => ({
  submitImplicitFeedback: vi.fn().mockResolvedValue({ ok: true })
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "sess-test",
  submitImplicitFeedback: mocks.submitImplicitFeedback
}))

describe("useImplicitFeedback", () => {
  beforeEach(() => {
    mocks.submitImplicitFeedback.mockClear()
  })

  it("emits citation_used with source metadata", async () => {
    const source = {
      metadata: {
        document_id: "doc-1",
        chunk_id: "chunk-1",
        corpus: "media_db"
      }
    }
    const { result } = renderHook(() =>
      useImplicitFeedback({
        conversationId: "C_1",
        messageId: "M_1",
        query: "reset auth",
        sources: [source]
      })
    )

    act(() => {
      result.current.trackCitationUsed(source, 0)
    })

    await waitFor(() => {
      expect(mocks.submitImplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "citation_used",
          query: "reset auth",
          session_id: "sess-test",
          conversation_id: "C_1",
          message_id: "M_1",
          doc_id: "doc-1",
          chunk_ids: ["chunk-1"],
          rank: 1,
          impression_list: ["chunk-1"],
          corpus: "media_db"
        })
      )
    })
  })

  it("sends nothing while disabled (temporary mode)", async () => {
    const { result } = renderHook(() =>
      useImplicitFeedback({
        conversationId: "C_1",
        messageId: "M_1",
        query: "And of Italy?",
        sources: [],
        enabled: false
      })
    )

    act(() => {
      result.current.trackDwellTime(3000)
    })

    await Promise.resolve()
    expect(mocks.submitImplicitFeedback).not.toHaveBeenCalled()
  })

  it("emits dwell_time with dwell_ms", async () => {
    const source = {
      metadata: {
        document_id: "doc-2",
        chunk_id: "chunk-2"
      }
    }
    const { result } = renderHook(() =>
      useImplicitFeedback({
        conversationId: "C_2",
        messageId: "M_2",
        query: "dwell query",
        sources: [source]
      })
    )

    act(() => {
      result.current.trackDwellTime(3456, source, 0)
    })

    await waitFor(() => {
      expect(mocks.submitImplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "dwell_time",
          dwell_ms: 3456,
          doc_id: "doc-2",
          chunk_ids: ["chunk-2"],
          rank: 1,
          conversation_id: "C_2",
          message_id: "M_2"
        })
      )
    })
  })
})
