import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  clearWorkspaceUndoActionsForTests,
  getWorkspaceUndoPendingCount,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "../undo-manager"

describe("workspace undo manager", () => {
  beforeEach(() => {
    clearWorkspaceUndoActionsForTests()
  })

  it("applies action immediately and restores on undo", () => {
    let value = 10
    const handle = scheduleWorkspaceUndoAction({
      apply: () => {
        value = 0
      },
      undo: () => {
        value = 10
      }
    })

    expect(value).toBe(0)
    expect(getWorkspaceUndoPendingCount()).toBe(1)
    expect(undoWorkspaceAction(handle.id)).toBe(true)
    expect(value).toBe(10)
    expect(getWorkspaceUndoPendingCount()).toBe(0)
  })

  it("purges pending action after timeout and runs finalize", () => {
    vi.useFakeTimers()
    const finalize = vi.fn()

    scheduleWorkspaceUndoAction({
      apply: vi.fn(),
      undo: vi.fn(),
      finalize,
      timeoutMs: 1000
    })

    expect(getWorkspaceUndoPendingCount()).toBe(1)

    vi.advanceTimersByTime(1000)

    expect(finalize).toHaveBeenCalledTimes(1)
    expect(getWorkspaceUndoPendingCount()).toBe(0)
    vi.useRealTimers()
  })
})
