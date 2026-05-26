// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { RecentStudySessions } from "../RecentStudySessions"
import { useRecentFlashcardReviewSessionsQuery } from "../../hooks"

vi.mock("../../hooks", () => ({
  useRecentFlashcardReviewSessionsQuery: vi.fn()
}))

type RecentSessionsQueryResult = ReturnType<typeof useRecentFlashcardReviewSessionsQuery>
type RecentSessionsQueryMock = Partial<RecentSessionsQueryResult>

const sessionsMock = vi.fn()

const mockRecentSessionsQuery = (result: RecentSessionsQueryMock) => {
  vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue(
    result as RecentSessionsQueryResult
  )
}

const createRefetchMock = () => {
  const mock = vi.fn()
  const refetch = ((...args: Parameters<RecentSessionsQueryResult["refetch"]>) => {
    mock(...args)
    return Promise.resolve(
      {} as Awaited<ReturnType<RecentSessionsQueryResult["refetch"]>>
    )
  }) as RecentSessionsQueryResult["refetch"]

  return { mock, refetch }
}

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
    mockRecentSessionsQuery({
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
    })
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
    expect(screen.getByRole("button", { name: "View completed session" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "View completed session" }))

    expect(sessionsMock).toHaveBeenCalledWith(81)
  })

  it("labels the selected row as a completed session snapshot", () => {
    render(
      <RecentStudySessions
        deckId={12}
        selectedSessionId={81}
        onOpenSession={sessionsMock}
        isActive
      />
    )

    expect(screen.getByRole("button", { name: "Viewing completed session" })).toBeInTheDocument()
  })

  it("shows a retryable error state when loading fails", () => {
    const refetchMock = createRefetchMock()
    mockRecentSessionsQuery({
      data: undefined,
      isLoading: false,
      isFetching: false,
      isError: true,
      error: new Error("Session service offline"),
      refetch: refetchMock.refetch
    })

    render(
      <RecentStudySessions
        deckId={12}
        selectedSessionId={null}
        onOpenSession={sessionsMock}
        isActive
      />
    )

    expect(screen.getByText("Failed to load recent study sessions")).toBeInTheDocument()
    expect(screen.getByText("Session service offline")).toBeInTheDocument()
    expect(
      screen
        .getByText("Session service offline")
        .closest('[data-ds-component="EmptyState"]')
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    expect(refetchMock.mock).toHaveBeenCalled()
  })

  it("renders loading and no-session product states through canonical design-system primitives", () => {
    mockRecentSessionsQuery({
      data: undefined,
      isLoading: true,
      isFetching: false,
      isError: false
    })

    const { rerender } = render(
      <RecentStudySessions
        deckId={12}
        selectedSessionId={null}
        onOpenSession={sessionsMock}
        isActive
      />
    )

    expect(screen.getByText("Loading recent study sessions...")).toBeInTheDocument()
    expect(
      screen
        .getByText("Loading recent study sessions...")
        .closest('[data-ds-component="LoadingState"]')
    ).toBeInTheDocument()

    mockRecentSessionsQuery({
      data: [],
      isLoading: false,
      isFetching: false,
      isError: false
    })

    rerender(
      <RecentStudySessions
        deckId={12}
        selectedSessionId={null}
        onOpenSession={sessionsMock}
        isActive
      />
    )

    expect(screen.getByText("No completed study sessions yet.")).toBeInTheDocument()
    expect(
      screen
        .getByText("No completed study sessions yet.")
        .closest('[data-ds-component="EmptyState"]')
    ).toBeInTheDocument()
  })
})
