// @vitest-environment jsdom
/**
 * CM-11 / CO-07 (#3101): the floating scroll-to-latest control says when a
 * reply is still arriving below, marks new messages, and sits at the right
 * edge instead of over the middle of the conversation.
 */
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ScrollToLatestButton } from "../ScrollToLatestButton"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

describe("ScrollToLatestButton", () => {
  it("is a plain scroll-to-latest button by default", () => {
    const onClick = vi.fn()
    render(<ScrollToLatestButton onClick={onClick} />)

    const button = screen.getByRole("button", { name: "Scroll to latest messages" })
    fireEvent.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("Reply in progress")).not.toBeInTheDocument()
  })

  it("reads 'Reply in progress' while a reply streams below the reader", () => {
    render(<ScrollToLatestButton onClick={vi.fn()} replyInProgress />)

    expect(screen.getByText("Reply in progress")).toBeVisible()
    expect(
      screen.getByRole("button", { name: /reply in progress.*scroll to latest/i })
    ).toBeInTheDocument()
  })

  it("marks new messages that arrived below the reader", () => {
    render(<ScrollToLatestButton onClick={vi.fn()} hasUnseenMessages />)

    expect(
      screen.getByRole("button", { name: /new messages below.*scroll to latest/i })
    ).toBeInTheDocument()
    expect(screen.getByTestId("scroll-to-latest-new-dot")).toBeInTheDocument()
  })

  it("sits at the right edge rather than centred over the conversation", () => {
    render(<ScrollToLatestButton onClick={vi.fn()} />)

    const wrapper = screen.getByRole("button").parentElement as HTMLElement
    expect(wrapper.className).toMatch(/\bjustify-end\b/)
    expect(wrapper.className).not.toMatch(/\bjustify-center\b/)
  })
})
