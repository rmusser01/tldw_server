// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { useDynamicUIActionBridge } from "../useDynamicUIActionBridge"

describe("useDynamicUIActionBridge", () => {
  it("submits valid OpenUI actions as normal user messages with metadata", async () => {
    const onSubmit = vi.fn(async () => ({ status: "submitted" }))
    const { result } = renderHook(() =>
      useDynamicUIActionBridge({
        messages: [
          {
            id: "assistant-1",
            isBot: true,
            name: "Assistant",
            message: "",
            sources: []
          }
        ],
        onSubmit,
        confirmSensitiveValues: vi.fn()
      })
    )

    await act(async () => {
      await result.current({
        renderer: "openui",
        sourceMessageId: "assistant-1",
        actionId: "survey",
        actionType: "submit",
        values: { answer: "yes" }
      })
    })

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("OpenUI action: submit survey"),
        image: "",
        dynamicUIRequest: expect.objectContaining({
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "survey",
          actionType: "submit",
          values: { answer: "yes" }
        }),
        userMetadataExtra: expect.objectContaining({
          dynamic_ui_action: expect.objectContaining({ actionId: "survey" })
        })
      })
    )
  })

  it("blocks sensitive-looking values without confirmation", async () => {
    const onSubmit = vi.fn()
    const { result } = renderHook(() =>
      useDynamicUIActionBridge({
        messages: [
          {
            id: "assistant-1",
            isBot: true,
            name: "Assistant",
            message: "",
            sources: []
          }
        ],
        onSubmit,
        confirmSensitiveValues: vi.fn(async () => false)
      })
    )

    await act(async () => {
      await result.current({
        renderer: "openui",
        sourceMessageId: "assistant-1",
        actionId: "login",
        actionType: "submit",
        values: { password: "secret" }
      })
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})
