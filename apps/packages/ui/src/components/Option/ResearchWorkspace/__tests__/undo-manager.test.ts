import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  clearWorkspaceUndoActionsForTests,
  getWorkspaceUndoPendingCount,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction,
  finalizeWorkspaceUndoAction
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

  it("does not register an action whose apply refuses persistence", () => {
    const handle = scheduleWorkspaceUndoAction({ apply: () => false, undo: () => {} })
    expect(handle).toBeNull()
    expect(getWorkspaceUndoPendingCount()).toBe(0)
  })

  it("keeps a failed undo available for retry", () => {
    let canRestore = false
    let value = "saved"
    const handle = scheduleWorkspaceUndoAction({
      apply: () => { value = "deleted" },
      undo: () => {
        if (!canRestore) return false
        value = "saved"
        return true
      }
    })
    expect(undoWorkspaceAction(handle.id)).toBe(false)
    expect(value).toBe("deleted")
    expect(getWorkspaceUndoPendingCount()).toBe(1)
    canRestore = true
    expect(undoWorkspaceAction(handle.id)).toBe(true)
    expect(value).toBe("saved")
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
  it("finalizes one exact editor removal while retaining unrelated Undo", () => {
    const finalize = vi.fn()
    const own = scheduleWorkspaceUndoAction({ apply: () => {}, undo: vi.fn(), finalize })
    const otherUndo = vi.fn()
    const other = scheduleWorkspaceUndoAction({ apply: () => {}, undo: otherUndo })
    expect(finalizeWorkspaceUndoAction(own.id)).toBe(true)
    expect(finalize).toHaveBeenCalledTimes(1)
    expect(undoWorkspaceAction(own.id)).toBe(false)
    expect(undoWorkspaceAction(other.id)).toBe(true)
    expect(otherUndo).toHaveBeenCalledTimes(1)
  })

})
