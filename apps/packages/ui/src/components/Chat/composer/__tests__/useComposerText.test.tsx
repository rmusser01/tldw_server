import { act, renderHook, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useComposerText } from "../hooks/useComposerText"

const createRef = () => React.createRef<HTMLTextAreaElement>()

describe("useComposerText", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("initializes with an empty message", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:empty",
        textareaRef: createRef()
      })
    )

    expect(result.current.form.values.message).toBe("")
  })

  it("updates the message via setMessageValue", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:update",
        textareaRef: createRef()
      })
    )

    act(() => {
      result.current.setMessageValue("hello world")
    })

    expect(result.current.form.values.message).toBe("hello world")
  })

  it("increments one monotonic revision for every committed message mutation", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:revision",
        textareaRef: createRef()
      })
    )

    expect(result.current.messageRevision).toBe(0)

    act(() => {
      result.current.form.getInputProps("message").onChange({
        target: { value: "typed draft" }
      })
    })
    expect(result.current.messageRevision).toBe(1)

    act(() => {
      result.current.setMessageValue("/research revised draft")
    })
    expect(result.current.messageRevision).toBe(2)

    act(() => {
      result.current.form.setFieldValue("message", "improved draft")
    })
    expect(result.current.messageRevision).toBe(3)

    act(() => {
      result.current.form.setFieldValue("message", "improved draft")
    })
    expect(result.current.messageRevision).toBe(3)

    act(() => {
      result.current.form.setFieldValue("message", "/research revised draft")
    })
    expect(result.current.messageRevision).toBe(4)

    act(() => {
      result.current.form.reset()
    })
    expect(result.current.messageRevision).toBe(5)
    expect(result.current.form.values.message).toBe("")

    act(() => {
      result.current.form.reset()
    })
    expect(result.current.messageRevision).toBe(5)
  })

  it("increments exactly once when a stored draft is restored", async () => {
    const draftKey = "tldw:test:restore-revision"
    window.localStorage.setItem(draftKey, "restored draft")

    const { result } = renderHook(() =>
      useComposerText({
        draftKey,
        textareaRef: createRef()
      })
    )

    await waitFor(() => {
      expect(result.current.form.values.message).toBe("restored draft")
    })
    expect(result.current.messageRevision).toBe(1)
  })

  it("uses reset as the send invalidation signal while clearDraft only clears persistence", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:send-revision",
        textareaRef: createRef()
      })
    )

    act(() => {
      result.current.form.setFieldValue("message", "send me")
    })
    expect(result.current.messageRevision).toBe(1)

    act(() => {
      result.current.form.reset()
      result.current.clearDraft()
    })

    expect(result.current.messageRevision).toBe(2)
    expect(result.current.form.values.message).toBe("")
  })

  it.each([false, true])(
    "marks only its explicit optimistic reset with an attempt token (StrictMode=%s)",
    (strictMode) => {
      const wrapper = strictMode
        ? ({ children }: { children: React.ReactNode }) => (
            <React.StrictMode>{children}</React.StrictMode>
          )
        : undefined
      const { result } = renderHook(
        () =>
          useComposerText({
            draftKey: `tldw:test:attempt-${strictMode}`,
            textareaRef: createRef()
          }),
        { wrapper }
      )
      type PromptAssistOwner = typeof result.current & {
        beginPromptAssistReset?: () => number
        markPromptAssistAttemptSaved?: (attemptId: number) => void
        promptAssistMutation?: {
          revision: number
          source: "owner" | "optimistic_reset"
          attemptId?: number
        }
        promptAssistSavedAttemptId?: number | null
      }
      const owner = () => result.current as PromptAssistOwner

      expect(typeof owner().beginPromptAssistReset).toBe("function")
      expect(typeof owner().markPromptAssistAttemptSaved).toBe("function")
      if (
        !owner().beginPromptAssistReset ||
        !owner().markPromptAssistAttemptSaved
      ) {
        return
      }

      act(() => owner().form.setFieldValue("message", "send this draft"))
      let attemptId = 0
      act(() => {
        attemptId = owner().beginPromptAssistReset!()
      })

      expect(owner().form.values.message).toBe("")
      expect(owner().promptAssistMutation).toEqual({
        revision: 2,
        source: "optimistic_reset",
        attemptId
      })

      act(() => owner().markPromptAssistAttemptSaved!(attemptId))
      expect(owner().promptAssistSavedAttemptId).toBe(attemptId)
    }
  )

  it("classifies a manual clear as an owner edit without an attempt token", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:manual-clear-source",
        textareaRef: createRef()
      })
    )
    type PromptAssistOwner = typeof result.current & {
      promptAssistMutation?: {
        revision: number
        source: "owner" | "optimistic_reset"
        attemptId?: number
      }
    }
    const owner = () => result.current as PromptAssistOwner

    act(() => owner().form.setFieldValue("message", "improved draft"))
    act(() => owner().form.setFieldValue("message", ""))

    expect(owner().promptAssistMutation).toEqual({
      revision: 2,
      source: "owner"
    })
  })

  it("exposes a focus helper (no-op when ref is empty)", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:focus",
        textareaRef: createRef()
      })
    )

    expect(() => result.current.textAreaFocus()).not.toThrow()
  })

  it("blurs instead of opening the mobile keyboard when returning focus", () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue("iPhone")
    const textarea = document.createElement("textarea")
    document.body.append(textarea)
    const textareaRef = { current: textarea }
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:mobile-focus",
        textareaRef
      })
    )
    textarea.focus()

    act(() => result.current.textAreaFocus())

    expect(textarea).not.toHaveFocus()
    textarea.remove()
  })

  it("computes maxHeight from isProMode flag", () => {
    const proHook = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:pro",
        textareaRef: createRef(),
        isProMode: true
      })
    )
    const casualHook = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:casual",
        textareaRef: createRef(),
        isProMode: false
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
        maxHeight: 200
      })
    )

    expect(result.current.textareaMaxHeight).toBe(200)
  })

  it("returns form.getInputProps compatible with textareas", () => {
    const { result } = renderHook(() =>
      useComposerText({
        draftKey: "tldw:test:inputprops",
        textareaRef: createRef()
      })
    )

    const props = result.current.form.getInputProps("message")

    expect(typeof props.onChange).toBe("function")
    expect(props.value).toBe("")
  })
})
