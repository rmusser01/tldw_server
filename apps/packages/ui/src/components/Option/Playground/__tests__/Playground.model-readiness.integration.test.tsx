// @vitest-environment jsdom
/**
 * CC-01 (#3111): the composer model chip must describe the selected model, not
 * the tldw server connection. A model that has never answered is "Not
 * checked", a provider failure makes it "Unreachable" with a one-click switch
 * to the server's default model, and a successful answer makes it "Working".
 */
import {
  act,
  createFakeTldwServer,
  FAKE_MODEL,
  FAKE_PROVIDER,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  waitFor,
  waitForChatIdle,
  within,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { resetChatModelSendOutcomes } from "@/utils/chat-model-readiness"
import { useStoreMessageOption } from "@/store/option"

const OLLAMA_MODEL = "llama3.2:3b"

const twoProviderServer = (ollamaHealth?: Record<string, unknown>) =>
  createFakeTldwServer({
    catalog: {
      providers: [
        {
          name: "ollama",
          displayName: "Ollama",
          models: [OLLAMA_MODEL],
          ...(ollamaHealth ? { health: ollamaHealth } : {})
        },
        { name: FAKE_PROVIDER, displayName: "Custom OpenAI API", models: [FAKE_MODEL] }
      ],
      defaultProvider: FAKE_PROVIDER
    }
  })

const modelStatus = () => screen.getByTestId("model-selector-status")
/** The status as people perceive it: in the chip's accessible name. */
const modelChip = () => screen.getByTestId("model-selector")

/** Send a message whose completion fails; the composer keeps the draft. */
const sendFailingTurn = async (view: PlaygroundView, text: string) => {
  const before = view.server.completionRequests().length
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(
    () => expect(view.server.completionRequests().length).toBe(before + 1),
    HARNESS_WAIT
  )
  await waitForChatIdle()
  await view.user.clear(view.composer)
}

describe("Playground model readiness chip (CC-01, #3111)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
    resetChatModelSendOutcomes()
  })

  it("CC-01: does not call a model that has never answered 'Healthy'", async () => {
    const server = twoProviderServer({
      status: "unknown",
      success_count: 0,
      failure_count: 0,
      consecutive_failures: 0
    })
    await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })

    // Without evidence the chip names only the server connection (the
    // harness reports no knowledge service, so that reads "Degraded").
    await waitFor(
      () => expect(modelChip()).toHaveAccessibleName(/Server connected|Degraded/),
      HARNESS_WAIT
    )
    expect(modelChip()).not.toHaveAccessibleName(/Healthy|Working/)
    expect(modelStatus()).not.toHaveAttribute("data-model-readiness")
  })

  it("CC-01: marks the model Unreachable after a provider failure and switches to the server default in one click", async () => {
    const server = twoProviderServer()
    const view = await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })
    server.planCompletion({
      status: 502,
      errorBody: {
        detail: {
          error_code: "provider_unavailable",
          message: "The chat service provider is currently unavailable."
        }
      }
    })

    await sendFailingTurn(view, "Is anyone there?")

    await waitFor(() => expect(modelChip()).toHaveAccessibleName(/Unreachable/), HARNESS_WAIT)
    const switchButton = await screen.findByRole(
      "button",
      { name: `Switch to ${FAKE_MODEL} (Custom OpenAI API)` },
      HARNESS_WAIT
    )
    await view.user.click(switchButton)

    await sendFromComposer(view, "Try the default model.")

    const requests = server.completionRequests()
    expect(requests.at(-1)?.body).toMatchObject({ model: FAKE_MODEL })
    expect(String(requests.at(-1)?.body?.api_provider)).toMatch(/custom[-_]openai[-_]api/)
    await waitFor(() => expect(modelChip()).toHaveAccessibleName(/Working/), HARNESS_WAIT)
    expect(screen.queryByTestId("model-readiness-switch")).toBeNull()
  })

  it.each([
    ["a saved chat", false],
    ["a temporary chat", true]
  ])("CC-N2: a provider failure in %s names the model, not the tldw server", async (_label, temporary) => {
    const server = twoProviderServer()
    const view = await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })
    if (temporary) act(() => useStoreMessageOption.getState().startNewChatDraft(true))
    server.planCompletion({
      status: 502,
      errorBody: {
        detail: {
          error_code: "provider_unavailable",
          message: "The chat service provider is currently unavailable."
        }
      }
    })

    await sendFailingTurn(view, "Is anyone there?")

    const named = new RegExp(`Couldn't reach ${OLLAMA_MODEL.replace(/[.]/g, "\\.")} \\(Ollama\\)\\.`)
    expect((await screen.findAllByText(named, {}, HARNESS_WAIT)).length).toBeGreaterThan(0)
    expect(screen.queryByText(/Something went wrong while talking to your tldw server/)).toBeNull()
  })

  it("CC-02: a saved chat's failed send offers Switch model, which opens the model picker", async () => {
    const server = twoProviderServer()
    const view = await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })
    server.planCompletion({
      status: 502,
      errorBody: {
        detail: {
          error_code: "provider_unavailable",
          message: "The chat service provider is currently unavailable."
        }
      }
    })

    await sendFailingTurn(view, "Is anyone there?")

    const title = await screen.findByText("No reply to this message", {}, HARNESS_WAIT)
    const card = title.closest("[role='status']") as HTMLElement
    expect(within(card).getByRole("alert")).toHaveTextContent(`Couldn't reach ${OLLAMA_MODEL} (Ollama).`)
    await view.user.click(within(card).getByRole("button", { name: "Switch model" }))
    expect(await screen.findByRole("textbox", { name: "Search models" }, HARNESS_WAIT)).toBeInTheDocument()
  })

  it("CC-01: shows a provider the server reports as failing as Unreachable before any send", async () => {
    const server = twoProviderServer({
      status: "circuit_open",
      success_count: 0,
      failure_count: 6,
      consecutive_failures: 6
    })
    await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })

    await waitFor(() => expect(modelChip()).toHaveAccessibleName(/Unreachable/), HARNESS_WAIT)
    expect(
      screen.getByRole("button", { name: `Switch to ${FAKE_MODEL} (Custom OpenAI API)` })
    ).toBeInTheDocument()
  })

  // The switch button disappears once the new model is selected; focus must
  // not fall to the page, and the change must be announced.
  it("CC-01: a one-click switch moves focus to the model chip and announces the new model", async () => {
    const server = twoProviderServer({
      status: "circuit_open",
      success_count: 0,
      failure_count: 6,
      consecutive_failures: 6
    })
    const view = await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })

    const switchButton = await screen.findByRole(
      "button",
      { name: `Switch to ${FAKE_MODEL} (Custom OpenAI API)` },
      HARNESS_WAIT
    )
    await view.user.click(switchButton)

    await waitFor(
      () => expect(screen.queryByTestId("model-readiness-switch")).toBeNull(),
      HARNESS_WAIT
    )
    expect(document.activeElement).toBe(screen.getByTestId("model-selector"))
    await waitFor(() =>
      expect(screen.getByTestId("model-readiness-announcement")).toHaveTextContent(
        `Switched to ${FAKE_MODEL} (Custom OpenAI API)`
      )
    )
  })
})
