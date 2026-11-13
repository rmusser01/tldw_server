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

// AX-11: the transcript's one announcement, outside the log.
describe("ChatReplyAnnouncerProvider", () => {
  it("reads a reply's plain text once per key, then clears it", () => {
    vi.useFakeTimers()
    render(
      <ChatReplyAnnouncerProvider>
        <Capture />
      </ChatReplyAnnouncerProvider>
    )
    const region = screen.getByTestId("chat-reply-announcer")
    expect(region).toHaveAttribute("aria-live", "polite")
    act(() => announce!.announce("reply-1", "## Plan\n\n**Bold** step"))
    expect(region).toHaveTextContent(/^Plan\s+Bold step$/)
    act(() => vi.advanceTimersByTime(ANNOUNCEMENT_CLEAR_MS))
    expect(region).toBeEmptyDOMElement()
    act(() => announce!.announce("reply-1", "## Plan\n\n**Bold** step"))
    expect(region).toBeEmptyDOMElement()
  })

  it("says the reply finished when it has no readable text", () => {
    render(
      <ChatReplyAnnouncerProvider>
        <Capture />
      </ChatReplyAnnouncerProvider>
    )
    act(() => announce!.announce("reply-2", "```\ncode only\n```"))
    expect(screen.getByTestId("chat-reply-announcer").textContent).toBe("Response complete")
  })

  it("is absent outside a transcript", () => {
    render(<Capture />)
    expect(announce).toBeNull()
  })
})
