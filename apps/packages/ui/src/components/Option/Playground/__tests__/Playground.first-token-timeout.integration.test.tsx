// @vitest-environment jsdom
/**
 * Red-first reproduction for #3107 (UX G06, CM-N1): a slow model's first token
 * must be awaited for the 120 s startup timeout, not cut off at 30 s. Time is
 * driven with fake timers; the plain `it` control proves the same mechanics
 * complete a turn whose first token arrives inside the 30 s window, so the
 * reproduction fails only because of the timeout itself.
 */
import {
  act,
  createFakeTldwServer,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitFor,
  type PlaygroundView
} from "./harness/playground-harness"
import { fireEvent } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"

/**
 * Type with real timers (userEvent needs them), send on a fake clock, let
 * `waitMs` of model time pass, then return to real timers for DOM queries.
 */
const sendAndWaitOnFakeClock = async (view: PlaygroundView, text: string, waitMs: number) => {
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  vi.useFakeTimers({
    toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"],
    shouldAdvanceTime: true
  })
  fireEvent.keyDown(view.composer, { key: "Enter", code: "Enter", keyCode: 13, charCode: 13 })
  for (let tick = 0; tick < 200 && view.server.completionRequests().length === 0; tick += 1) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(25)
    })
  }
  expect(view.server.completionRequests()).toHaveLength(1)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(waitMs)
  })
  vi.useRealTimers()
}

const transcript = () =>
  useStoreMessageOption
    .getState()
    .messages.map((message) => `${message.isBot ? "assistant" : "user"}: ${message.message}`)

describe("Playground first-token wait (#3107)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it("completes a turn whose first token arrives within 30 s (fake-clock control)", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ reply: "Prompt model reply", firstTokenDelayMs: 20_000 })
    const view = await renderPlayground({ server })

    await sendAndWaitOnFakeClock(view, "Quick question", 21_000)

    expect(transcript()).toEqual(["user: Quick question", "assistant: Prompt model reply"])
    await waitFor(() => expect(screen.getByText("Prompt model reply")).toBeInTheDocument(), {
      timeout: 5_000
    })
  })

  // CM-N1 #3107 — the transport's byte-level idle timer is armed before the first token, so
  // TldwChat hands it max(startup, idle); TldwChat itself enforces the 120 s startup limit
  // before the first visible token and the stream-idle limit after it.
  it("CM-N1 (#3107): a model whose first token arrives after 60 s still completes (120 s startup timeout)", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ reply: "Slow model reply", firstTokenDelayMs: 60_000 })
    const view = await renderPlayground({ server })

    await sendAndWaitOnFakeClock(view, "Slow question", 61_000)

    expect(transcript()).toEqual(["user: Slow question", "assistant: Slow model reply"])
    await waitFor(() => expect(screen.getByText("Slow model reply")).toBeInTheDocument(), {
      timeout: 5_000
    })
  })
})
