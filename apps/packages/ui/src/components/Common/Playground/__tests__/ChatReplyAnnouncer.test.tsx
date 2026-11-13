// @vitest-environment jsdom
import React from "react"
import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  ANNOUNCEMENT_CLEAR_MS,
  ChatReplyAnnouncerProvider,
  useChatReplyAnnouncer
} from "../ChatReplyAnnouncer"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key
  })
}))

let announce: ReturnType<typeof useChatReplyAnnouncer> = null
const Capture = () => {
  const announcer = useChatReplyAnnouncer()
  React.useLayoutEffect(() => {
    announce = announcer
  }, [announcer])
  return null
}

afterEach(() => {
  vi.useRealTimers()
  announce = null
})

// AX-11: the transcript's one completion announcement, outside the log. The
// log itself reads the reply; this only says it finished.
describe("ChatReplyAnnouncerProvider", () => {
  it("says a reply finished once per reply, then clears", () => {
    vi.useFakeTimers()
    render(
      <ChatReplyAnnouncerProvider>
        <Capture />
      </ChatReplyAnnouncerProvider>
    )
    const region = screen.getByTestId("chat-reply-announcer")
    expect(region).toHaveAttribute("aria-live", "polite")
    act(() => announce!.announceComplete("reply-1"))
    expect(region.textContent).toBe("Response complete")
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_CLEAR_MS))
    expect(region).toBeEmptyDOMElement()
    act(() => announce!.announceComplete("reply-1"))
    expect(region).toBeEmptyDOMElement()
  })

  it("changes its text for a second reply finishing before the first clears", () => {
    render(
      <ChatReplyAnnouncerProvider>
        <Capture />
      </ChatReplyAnnouncerProvider>
    )
    const region = screen.getByTestId("chat-reply-announcer")
    act(() => announce!.announceComplete("reply-1"))
    const first = region.textContent
    act(() => announce!.announceComplete("reply-2"))
    expect(region.textContent?.trim()).toBe("Response complete")
    expect(region.textContent).not.toBe(first)
  })

  it("is absent outside a transcript", () => {
    render(<Capture />)
    expect(announce).toBeNull()
  })
})
