import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { useSidepanelReplyActions } from "../sidepanel-reply-actions"

const t = (_key: string, fallback: string) => fallback

const renderActions = (onSubmit: ReturnType<typeof vi.fn>) => {
  const notification = { error: vi.fn() }
  const { result } = renderHook(() =>
    useSidepanelReplyActions({
      historySelection: null,
      onSubmit: onSubmit as never,
      actionsMountedRef: { current: true },
      retrySource: { model: "model-1", selectedSystemPrompt: null },
      t,
      notification
    })
  )
  return { actions: result.current, notification }
}

describe("side-panel Continue errors (review follow-up)", () => {
  it("says a refused Continue in Continue wording instead of failing silently", async () => {
    const { actions, notification } = renderActions(vi.fn(async () => {
      throw new Error("unsupported_history_continuation")
    }))
    await expect(actions.continueLastMessage()).resolves.toEqual(
      expect.objectContaining({ status: "failed" })
    )
    expect(notification.error).toHaveBeenCalledOnce()
    expect(String(notification.error.mock.calls[0][0].description)).toMatch(/continu/i)
  })

  it("shows an ordinary error's own text", async () => {
    const { actions, notification } = renderActions(vi.fn(async () => {
      throw new Error("Model unavailable")
    }))
    await actions.continueLastMessage()
    expect(notification.error).toHaveBeenCalledWith(expect.objectContaining({ description: "Model unavailable" }))
  })

  it("stays quiet when Continue was submitted", async () => {
    const { actions, notification } = renderActions(vi.fn(async () => ({ status: "submitted" })))
    expect(await actions.continueLastMessage()).toEqual({ status: "submitted" })
    expect(notification.error).not.toHaveBeenCalled()
  })
})
