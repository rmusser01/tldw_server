import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useComposerSubmit } from "../hooks/useComposerSubmit"

describe("useComposerSubmit", () => {
  it("forwards the payload to sendMessage", async () => {
    const sendMessage = vi.fn(async () => "ok")
    const { result } = renderHook(() => useComposerSubmit({ sendMessage }))

    await act(async () => {
      await result.current.dispatch({ message: "hi" })
    })

    expect(sendMessage).toHaveBeenCalledWith({ message: "hi" })
  })

  it("runs beforeSend before sendMessage", async () => {
    const order: string[] = []
    const sendMessage = vi.fn(async () => {
      order.push("send")
    })
    const beforeSend = vi.fn(() => {
      order.push("before")
    })

    const { result } = renderHook(() => useComposerSubmit({ sendMessage }))

    await act(async () => {
      await result.current.dispatch({ message: "hi" }, { beforeSend })
    })

    expect(order).toEqual(["before", "send"])
  })

  it("runs afterSend after sendMessage resolves", async () => {
    const order: string[] = []
    const sendMessage = vi.fn(async () => {
      order.push("send")
    })
    const afterSend = vi.fn(() => {
      order.push("after")
    })

    const { result } = renderHook(() => useComposerSubmit({ sendMessage }))

    await act(async () => {
      await result.current.dispatch({ message: "hi" }, { afterSend })
    })

    expect(order).toEqual(["send", "after"])
  })

  it("runs both hooks in the expected order: before → send → after", async () => {
    const order: string[] = []
    const sendMessage = vi.fn(async () => {
      order.push("send")
    })

    const { result } = renderHook(() =>
      useComposerSubmit({ sendMessage })
    )

    await act(async () => {
      await result.current.dispatch(
        { message: "hi" },
        {
          beforeSend: () => order.push("before"),
          afterSend: () => order.push("after"),
        }
      )
    })

    expect(order).toEqual(["before", "send", "after"])
  })

  it("skips afterSend when sendMessage rejects, and propagates the error", async () => {
    const order: string[] = []
    const sendMessage = vi.fn(async () => {
      order.push("send")
      throw new Error("boom")
    })
    const afterSend = vi.fn(() => {
      order.push("after")
    })
    const beforeSend = vi.fn(() => {
      order.push("before")
    })

    const { result } = renderHook(() => useComposerSubmit({ sendMessage }))

    await expect(
      act(async () => {
        await result.current.dispatch(
          { message: "hi" },
          { beforeSend, afterSend }
        )
      })
    ).rejects.toThrow("boom")

    expect(order).toEqual(["before", "send"])
    expect(afterSend).not.toHaveBeenCalled()
  })

  it("works without hooks", async () => {
    const sendMessage = vi.fn(async () => "ok")
    const { result } = renderHook(() => useComposerSubmit({ sendMessage }))

    await act(async () => {
      await result.current.dispatch({ message: "hi" })
    })

    expect(sendMessage).toHaveBeenCalledOnce()
  })
})
