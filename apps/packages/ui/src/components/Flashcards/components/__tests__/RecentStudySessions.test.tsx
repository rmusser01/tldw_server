// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { RecentStudySessions } from "../RecentStudySessions"
import { useRecentFlashcardReviewSessionsQuery } from "../../hooks"

vi.mock("../../hooks", () => ({
  useRecentFlashcardReviewSessionsQuery: vi.fn()
}))

const sessionsMock = vi.fn()

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("RecentStudySessions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionsMock.mockReset()
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [
        {
          id: 81,
          deck_id: 12,
          review_mode: "due",
          tag_filter: null,
          scope_key: "due:deck:12",
          status: "completed",
          started_at: "2026-04-05T18:00:00Z",
          last_activity_at: "2026-04-05T18:10:00Z",
          completed_at: "2026-04-05T18:12:00Z",
          client_id: "test"
        }
      ],
      isLoading: false,
      isFetching: false
    } as any)
  })

  it("lists completed sessions and reopens the selected snapshot when clicked", () => {
    render(
      <RecentStudySessions
        deckId={12}
        selectedSessionId={null}
        onOpenSession={sessionsMock}
        isActive
      />
    )

    expect(useRecentFlashcardReviewSessionsQuery).toHaveBeenCalledWith(
      { deckId: 12, status: "completed", limit: 8 },
      expect.objectContaining({ enabled: true })
    )
    expect(screen.getByText("Recent study sessions")).toBeInTheDocument()
    expect(screen.getByText("Completed")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Reopen snapshot for session 81/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Reopen snapshot for session 81/i }))

    expect(sessionsMock).toHaveBeenCalledWith(81)
  })
})
