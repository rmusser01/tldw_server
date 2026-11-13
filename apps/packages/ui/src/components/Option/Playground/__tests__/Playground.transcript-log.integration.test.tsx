// @vitest-environment jsdom
/**
 * AX-11 (#3101): the transcript's live region was the whole scroller, so a
 * send announced the history review, the starter deck, each message's
 * chrome and the markdown re-render after streaming, plus a nested "Response
 * complete". The log now holds only the message rows, is quiet and busy while
 * a reply streams, and the finished reply is read once from an announcer
 * outside it.
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

  it("is one quiet log of the message rows, without the composer or the starter deck", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    expect(screen.getAllByRole("log")).toHaveLength(1)
    const log = transcriptLog()
    expect(log).toHaveAttribute("aria-live", "off")
    expect(log).not.toContainElement(view.composer)
    expect(log).toBeEmptyDOMElement()
  })

  it("is busy while a reply streams and reads the finished reply once, from outside the log", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    const resume = deferred()
    server.planCompletion({ chunks: ["Partial ", "**answer**"], pauseAfterChunks: 1, resume: resume.promise })

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
    await waitFor(() => expect(announced).toEqual(["Partial answer"]), HARNESS_WAIT)
    observer.disconnect()
    // No live region inside the log: chrome and re-renders stay silent.
    expect(log.querySelectorAll("[aria-live]")).toHaveLength(0)
    expect(within(log).queryByText("Response complete")).toBeNull()
  })
})
