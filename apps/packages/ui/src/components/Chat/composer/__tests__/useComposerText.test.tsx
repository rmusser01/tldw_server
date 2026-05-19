import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { useComposerText } from "../hooks/useComposerText"

const createRef = () => React.createRef<HTMLTextAreaElement>()

describe("useComposerText", () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it("initializes with an empty message", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:empty",
        textareaRef: createRef(),
      })
    )

    expect(result.current.form.values.message).toBe("")
  })

  it("updates the message via setMessageValue", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:update",
        textareaRef: createRef(),
      })
    )

    act(() => {
      result.current.setMessageValue("hello world")
    })

    expect(result.current.form.values.message).toBe("hello world")
  })

  it("exposes a focus helper (no-op when ref is empty)", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:focus",
        textareaRef: createRef(),
      })
    )

    expect(() => result.current.textAreaFocus()).not.toThrow()
  })

  it("computes maxHeight from isProMode flag", () => {
    const proHook = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:pro",
        textareaRef: createRef(),
        isProMode: true,
      })
    )
    const casualHook = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:casual",
        textareaRef: createRef(),
        isProMode: false,
      })
    )

    expect(proHook.result.current.textareaMaxHeight).toBe(160)
    expect(casualHook.result.current.textareaMaxHeight).toBe(120)
  })

  it("supports a caller-supplied explicit maxHeight", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:explicit",
        textareaRef: createRef(),
        maxHeight: 200,
      })
    )

    expect(result.current.textareaMaxHeight).toBe(200)
  })

  it("returns form.getInputProps compatible with textareas", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:inputprops",
        textareaRef: createRef(),
      })
    )

    const props = result.current.form.getInputProps("message")

    expect(typeof props.onChange).toBe("function")
    expect(props.value).toBe("")
  })
})
