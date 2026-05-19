// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { StudySuggestionsPanel } from "../../StudySuggestionsPanel"
import { useStudySuggestions } from "../../hooks/useStudySuggestions"

const mocks = vi.hoisted(() => ({
  refresh: vi.fn(),
  performAction: vi.fn()
}))

vi.mock("../../hooks/useStudySuggestions", () => ({
  useStudySuggestions: vi.fn()
}))

const initialSnapshot = {
  snapshot: {
    id: 88,
    service: "quiz",
    activity_type: "quiz_attempt",
    anchor_type: "quiz_attempt",
    anchor_id: 101,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        score: 7,
        correct_count: 2,
        total_count: 4
      },
      topics: [
        {
          id: "topic-1",
          display_label: "Renal basics",
          type: "grounded",
          status: "weakness",
          selected: true,
          source_type: "note"
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["topic-1"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T18:00:00Z",
    last_modified: "2026-04-05T18:00:00Z"
  },
  live_evidence: {
    "topic-1": {
      source_available: true,
      source_type: "note",
      source_id: "note-7"
    }
  }
}

const v2Snapshot = {
  snapshot: {
    id: 91,
    service: "quiz",
    activity_type: "quiz_attempt",
    anchor_type: "quiz_attempt",
    anchor_id: 404,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        score: 8,
        correct_count: 3,
        total_count: 4
      },
      topics: [
        {
          id: "topic-local-1",
          display_label: "Cardiovascular review",
          topic_key: "cardio-001",
          canonical_label: "Cardiovascular review",
          evidence_reasons: ["topic_gap", "incorrect_answer"],
          type: "grounded",
          status: "weakness",
          selected: true
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["cardio-001"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T18:00:00Z",
    last_modified: "2026-04-05T18:00:00Z"
  },
  live_evidence: {
    "cardio-001": {
      source_available: true,
      source_type: "note",
      source_id: "note-9"
    }
  }
}

describe("StudySuggestionsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.refresh.mockReset()
    mocks.performAction.mockReset()
    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "ready",
      snapshot: initialSnapshot,
      isLoading: false,
      isRefreshing: false,
      refresh: mocks.refresh,
      performAction: mocks.performAction
    } as never)
  })

  it("lets users add, rename, remove, and reset topics while showing manual topics as exploratory", async () => {
    render(<StudySuggestionsPanel anchorType="quiz_attempt" anchorId={101} />)

    expect(screen.getByDisplayValue("Renal basics")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Add topic/i }))

    const newTopicInput = screen.getByDisplayValue("New topic")
    expect(screen.getByText("Exploratory")).toBeInTheDocument()

    fireEvent.change(newTopicInput, { target: { value: "  Acid base  " } })
    expect(screen.getByDisplayValue("Acid base")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Remove topic 2" }))
    expect(screen.queryByDisplayValue("Acid base")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Add topic/i }))
    fireEvent.click(screen.getByRole("button", { name: /Reset topics/i }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("Renal basics")).toBeInTheDocument()
      expect(screen.queryByDisplayValue("New topic")).not.toBeInTheDocument()
    })
  })

  it("renders V2 snapshot topics through the current panel and keeps edited labels tied to the snapshot-local id", async () => {
    mocks.performAction.mockResolvedValue({
      disposition: "generated",
      snapshot_id: 91,
      selection_fingerprint: "snapshot_id=91|target_service=flashcards|target_type=deck|topics=acid base,cardiovascular review|action_kind=follow_up_flashcards|generator_version=v1",
      target_service: "flashcards",
      target_type: "deck",
      target_id: "deck-12"
    })

    vi.mocked(useStudySuggestions).mockReturnValue({
      status: "ready",
      snapshot: v2Snapshot,
      isLoading: false,
      isRefreshing: false,
      refresh: mocks.refresh,
      performAction: mocks.performAction
    } as never)

    render(<StudySuggestionsPanel anchorType="quiz_attempt" anchorId={404} />)

    expect(screen.getByDisplayValue("Cardiovascular review")).toBeInTheDocument()
    expect(screen.getByText("Evidence: Topic Gap, Incorrect Answer")).toBeInTheDocument()

    const firstTopicInput = screen.getByLabelText("Topic 1")
    fireEvent.change(firstTopicInput, { target: { value: "  Cardio review  " } })
    expect(screen.getByDisplayValue("Cardio review")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Add topic/i }))
    fireEvent.change(screen.getByLabelText("Topic 2"), { target: { value: "  Acid base  " } })

    fireEvent.click(screen.getByRole("button", { name: "Create flashcards" }))

    await waitFor(() => {
      expect(mocks.performAction).toHaveBeenCalledWith(
        expect.objectContaining({
          targetService: "flashcards",
          targetType: "deck",
          actionKind: "follow_up_flashcards",
          selectedTopicIds: ["topic-local-1"],
          selectedTopicEdits: [{ id: "topic-local-1", label: "Cardio review" }],
          manualTopicLabels: ["Acid base"],
          hasExplicitSelection: true
        })
      )
    })

    const firstToggle = screen.getByRole("button", { name: "Toggle selection for topic 1" })
    fireEvent.click(firstToggle)
    expect(firstToggle).toHaveAttribute("aria-pressed", "false")

    fireEvent.click(screen.getByRole("button", { name: "Create flashcards" }))

    await waitFor(() => {
      expect(mocks.performAction).toHaveBeenLastCalledWith(
        expect.objectContaining({
          selectedTopicIds: [],
          selectedTopicEdits: [],
          manualTopicLabels: ["Acid base"],
          hasExplicitSelection: true
        })
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Toggle selection for topic 2" }))
    expect(screen.getByRole("button", { name: "Toggle selection for topic 2" })).toHaveAttribute(
      "aria-pressed",
      "false"
    )

    fireEvent.click(screen.getByRole("button", { name: "Create flashcards" }))

    await waitFor(() => {
      expect(mocks.performAction).toHaveBeenLastCalledWith(
        expect.objectContaining({
          selectedTopicIds: [],
          selectedTopicEdits: [],
          manualTopicLabels: [],
          hasExplicitSelection: true
        })
      )
    })
  })

  it("keeps legacy snapshot topics rendering unchanged", async () => {
    render(<StudySuggestionsPanel anchorType="quiz_attempt" anchorId={101} />)

    expect(screen.getByDisplayValue("Renal basics")).toBeInTheDocument()
    expect(screen.getByText("Weakness")).toBeInTheDocument()
    expect(screen.getByText("Evidence: Grounded")).toBeInTheDocument()
  })

  it("shows Open existing when the follow-up action reuses an existing result", async () => {
    mocks.performAction.mockResolvedValue({
      disposition: "opened_existing",
      snapshot_id: 88,
      selection_fingerprint: "snapshot_id=88|target_service=flashcards|target_type=deck|topics=renal basics|action_kind=follow_up_flashcards|generator_version=v1",
      target_service: "flashcards",
      target_type: "deck",
      target_id: "deck-12"
    })

    render(<StudySuggestionsPanel anchorType="quiz_attempt" anchorId={101} />)

    fireEvent.click(screen.getByRole("button", { name: "Create flashcards" }))

    expect(await screen.findByText("Open existing")).toBeInTheDocument()
    expect(mocks.performAction).toHaveBeenCalledWith(
      expect.objectContaining({
        targetService: "flashcards",
        targetType: "deck",
        actionKind: "follow_up_flashcards",
        selectedTopicIds: ["topic-1"],
        selectedTopicEdits: [{ id: "topic-1", label: "Renal basics" }],
        manualTopicLabels: [],
        hasExplicitSelection: true
      })
    )
  })
})
