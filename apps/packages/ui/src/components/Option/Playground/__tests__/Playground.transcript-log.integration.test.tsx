// @vitest-environment jsdom
/**
 * AX-11 (#3101): the transcript's live region was the whole scroller, so a
 * send announced the history review, the starter deck and the composer, plus
 * a nested "Response complete" inside the log. The log now holds only the
 * message rows: a polite log that is busy while a reply streams, so a screen
 * reader reads the new reply once streaming settles. Completion is said once
 * from a region outside the log, and the reply's text is not copied there.
 */
import {
  createFakeTldwServer,
  deferred,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitFor,
  waitForChatIdle
} from "./harness/playground-harness"
import { within } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const transcriptLog = () => screen.getByRole("log", { name: "Chat messages" })

describe("Playground transcript log (AX-11)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("is one polite log of the message rows, without the composer or the starter deck", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    expect(screen.getAllByRole("log")).toHaveLength(1)
    const log = transcriptLog()
    expect(log).toHaveAttribute("aria-live", "polite")
    expect(log).toHaveAttribute("aria-relevant", "additions")
    expect(log).not.toContainElement(view.composer)
    expect(log).toBeEmptyDOMElement()
  })

  it("is busy while a reply streams and says once, outside the log, that it finished", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    const resume = deferred()
    server.planCompletion({ chunks: ["Partial ", "reply text"], pauseAfterChunks: 1, resume: resume.promise })

    await view.user.click(view.composer)
    await view.user.type(view.composer, "Hello there")
    await view.user.keyboard("{Enter}")
    await waitFor(
      () =>
        expect(
          useStoreMessageOption
            .getState()
            .messages.some((message) => message.isBot && message.message.includes("Partial"))
        ).toBe(true),
      HARNESS_WAIT
    )
    await waitFor(() => expect(transcriptLog()).toHaveAttribute("aria-busy", "true"), HARNESS_WAIT)
    const announcer = screen.getByTestId("chat-reply-announcer")
    expect(announcer).toBeEmptyDOMElement()
    // Record what the announcer says; it clears itself shortly after.
    const announced: string[] = []
    const observer = new MutationObserver(() => {
      if (announcer.textContent) announced.push(announcer.textContent)
    })
    observer.observe(announcer, { childList: true, characterData: true, subtree: true })

    resume.resolve()
    await waitForChatIdle()

    const log = transcriptLog()
    await waitFor(() => expect(log).not.toHaveAttribute("aria-busy"), HARNESS_WAIT)
    expect(within(log).getAllByTestId("chat-message")).toHaveLength(2)
    expect(log).not.toContainElement(announcer)
    await waitFor(() => expect(announced).toEqual(["Response complete"]), HARNESS_WAIT)
    observer.disconnect()
    // The reply's text is on the page once (in the log), not copied into the announcer.
    await waitFor(() => expect(log).toHaveTextContent("Partial reply text"), HARNESS_WAIT)
    expect(document.body.textContent?.split("Partial reply text")).toHaveLength(2)
    expect(announcer).not.toHaveTextContent("Partial")
    // No live region nested inside the log.
    expect(log.querySelectorAll("[aria-live]")).toHaveLength(0)
    expect(within(log).queryByText("Response complete")).toBeNull()
  })
})
