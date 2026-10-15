import { render, screen, act, waitFor } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import React from "react"

import { useFeedback } from "@/hooks/useFeedback"
import { useImplicitFeedback } from "@/hooks/useImplicitFeedback"
import { MessageActionsBar } from "@/components/Common/Playground/MessageActionsBar"
import { useFeedbackStore } from "@/store/feedback"
import { useStoreMessageOption } from "@/store/option"

const mocks = vi.hoisted(() => ({
  submitExplicitFeedback: vi.fn().mockResolvedValue({ ok: true, feedback_id: "fb-chat-1" }),
  submitImplicitFeedback: vi.fn().mockResolvedValue({ ok: true }),
  updateChatRating: vi.fn().mockResolvedValue({}),
  notificationError: vi.fn()
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "sess-chat-test",
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

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

describe("Chat feedback flows", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useFeedbackStore.setState({ entries: {} })
    useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true)
    window.localStorage.clear()
  })

  it("submits detailed chat feedback in one payload (rating + taxonomy + notes)", async () => {
    const TestHarness = () => {
      const feedback = useFeedback({
        messageKey: "srv:M_chat_1",
        conversationId: "C_chat_1",
        messageId: "M_chat_1",
        query: null
      })

      return (
        <button
          type="button"
          onClick={() => {
            void feedback.submitDetail({
              rating: 4,
              issues: ["missing_details", "sources_unhelpful"],
              notes: "Needed stronger citations."
            })
          }}
        >
          submit-detail
        </button>
      )
    }

    render(<TestHarness />)
    await act(async () => {
      screen.getByRole("button", { name: "submit-detail" }).click()
    })

    await waitFor(() => {
      expect(mocks.submitExplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          conversation_id: "C_chat_1",
          message_id: "M_chat_1",
          feedback_type: "relevance",
          relevance_score: 4,
          issues: ["missing_details", "sources_unhelpful"],
          user_notes: "Needed stronger citations.",
          idempotency_key: expect.any(String)
        })
      )
    })
  })

  it("emits dwell_time and citation_used implicit events", async () => {
    const source = {
      metadata: {
        document_id: "doc-chat-1",
        chunk_id: "chunk-chat-1",
        corpus: "media_db"
      }
    }

    const TestHarness = () => {
      const implicit = useImplicitFeedback({
        conversationId: "C_chat_2",
        messageId: "M_chat_2",
        query: "chat query",
        sources: [source]
      })
      return (
        <>
          <button
            type="button"
            onClick={() => implicit.trackDwellTime(3000, source, 0)}
          >
            dwell
          </button>
          <button
            type="button"
            onClick={() => implicit.trackCitationUsed(source, 0)}
          >
            citation
          </button>
        </>
      )
    }

    render(<TestHarness />)
    await act(async () => {
      screen.getByRole("button", { name: "dwell" }).click()
      screen.getByRole("button", { name: "citation" }).click()
    })

    await waitFor(() => {
      expect(mocks.submitImplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "dwell_time",
          dwell_ms: 3000,
          message_id: "M_chat_2"
        })
      )
      expect(mocks.submitImplicitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: "citation_used",
          doc_id: "doc-chat-1",
          chunk_ids: ["chunk-chat-1"],
          corpus: "media_db",
          message_id: "M_chat_2"
        })
      )
    })
  })

  it("renders feedback controls for persisted non-user messages", () => {
    render(
      <MessageActionsBar
        t={(key: string, fallback?: string) => fallback || key}
        isProMode={false}
        isBot={false}
        showVariantPager={false}
        resolvedVariantIndex={0}
        variantCount={1}
        canSwipePrev={false}
        canSwipeNext={false}
        overflowChipVisibility="hidden"
        actionRowVisibility="flex"
        isSpeaking={false}
        onToggleTts={() => {}}
        copyPressed={false}
        onCopy={() => {}}
        canReply={false}
        onReply={() => {}}
        canSaveToNotes={false}
        canSaveToFlashcards={false}
        canGenerateDocument={false}
        onGenerateDocument={() => {}}
        onSaveKnowledge={() => {}}
        savingKnowledge={null}
        isLastMessage={false}
        onRegenerate={() => {}}
        onEdit={() => {}}
        editMode={false}
        showFeedbackControls
        feedbackDisabled={false}
        feedbackDisabledReason=""
        isFeedbackSubmitting={false}
        showThanks={false}
        onThumbUp={() => {}}
        onThumbDown={() => {}}
        onOpenDetails={() => {}}
      />
    )

    expect(screen.getByText("Was this helpful?")).toBeInTheDocument()
  })
})
