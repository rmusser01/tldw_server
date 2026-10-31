import { renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useMilestoneStore } from "@/store/milestones"
import { useMissionCards } from "../hooks/useMissionCards"

const scope = vi.hoisted(() => ({ current: "server-a:alice" as string | null }))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => scope.current
}))
vi.mock("@/store/connection", () => ({
  useConnectionStore: (select: (value: unknown) => unknown) =>
    select({ state: { userPersona: "researcher" } })
}))
beforeEach(() => {
  scope.current = "server-a:alice"
  useMilestoneStore.getState().resetMilestones()
})

it("shows only the active identity's first-value progress", () => {
  const store = useMilestoneStore.getState()
  store.markMilestone("first_chat")
  store.markMilestone("first_ingest")
  store.markScopedMilestone("server-a:alice", "first_connection")
  store.markScopedMilestone("server-a:alice", "first_ingest")
  store.markScopedMilestone("server-a:alice", "first_chat")
  store.markScopedMilestone("server-a:bob", "first_connection")
  const { result, rerender } = renderHook(() => useMissionCards())
  expect(
    result.current.gettingStartedCards.find(
      (card) => card.id === "researcher-ingest"
    )?.isCompleted
  ).toBe(true)
  expect(
    result.current.gettingStartedCards.find(
      (card) => card.id === "researcher-ask"
    )?.isCompleted
  ).toBe(true)
  scope.current = "server-a:bob"
  rerender()
  expect(
    result.current.gettingStartedCards.find(
      (card) => card.id === "researcher-ingest"
    )?.isCompleted
  ).toBe(false)
  expect(
    result.current.gettingStartedCards.find(
      (card) => card.id === "researcher-ask"
    )
  ).toBeUndefined()
  scope.current = null
  rerender()
  expect(result.current.gettingStartedCards).toEqual([])
})
