import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  CHAT_ERROR_BUBBLE_ATTRIBUTE,
  useChatErrorBubbleInView
} from "../hooks/useChatErrorBubbleInView"

type Callback = (entries: Array<{ isIntersecting: boolean }>) => void
const observers: Callback[] = []
class FakeIntersectionObserver {
  constructor(callback: Callback) {
    observers.push(callback)
  }
  observe() {}
  disconnect() {}
}

const addBubble = () => {
  const bubble = document.createElement("div")
  bubble.setAttribute(CHAT_ERROR_BUBBLE_ATTRIBUTE, "")
  document.body.appendChild(bubble)
}

// CC-N2: the composer banner repeats an error only while its bubble is off screen.
describe("useChatErrorBubbleInView", () => {
  afterEach(() => {
    document.body.innerHTML = ""
    observers.length = 0
    vi.unstubAllGlobals()
  })

  it("is false when there is no error bubble in the page, so the banner shows", () => {
    const { result } = renderHook(() => useChatErrorBubbleInView("error-1"))
    expect(result.current).toBe(false)
  })

  it("follows the bubble in and out of view", () => {
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver)
    addBubble()
    const { result } = renderHook(() => useChatErrorBubbleInView("error-1"))
    expect(result.current).toBeNull()

    act(() => observers[0]([{ isIntersecting: true }]))
    expect(result.current).toBe(true)
    act(() => observers[0]([{ isIntersecting: false }]))
    expect(result.current).toBe(false)
  })

  it("assumes the bubble is visible when the browser can't tell", () => {
    vi.stubGlobal("IntersectionObserver", undefined)
    addBubble()
    const { result } = renderHook(() => useChatErrorBubbleInView("error-1"))
    expect(result.current).toBe(true)
  })

  it("has nothing to say without an error", () => {
    const { result } = renderHook(() => useChatErrorBubbleInView(null))
    expect(result.current).toBeNull()
  })
})
