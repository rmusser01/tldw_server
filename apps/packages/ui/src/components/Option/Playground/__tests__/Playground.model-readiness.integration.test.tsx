// @vitest-environment jsdom
/**
 * CC-01 (#3111): the composer model chip must describe the selected model, not
 * the tldw server connection. A model that has never answered is "Not
 * checked", a provider failure makes it "Unreachable" with a one-click switch
 * to the server's default model, and a successful answer makes it "Working".
 */
import {
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
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { resetChatModelSendOutcomes } from "@/utils/chat-model-readiness"

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

    await waitFor(() => expect(modelStatus()).toHaveTextContent("Not checked"), HARNESS_WAIT)
    expect(modelStatus()).not.toHaveTextContent("Healthy")
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

    await waitFor(() => expect(modelStatus()).toHaveTextContent("Unreachable"), HARNESS_WAIT)
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
    await waitFor(() => expect(modelStatus()).toHaveTextContent("Working"), HARNESS_WAIT)
    expect(screen.queryByTestId("model-readiness-switch")).toBeNull()
  })

  it("CC-01: shows a provider the server reports as failing as Unreachable before any send", async () => {
    const server = twoProviderServer({
      status: "circuit_open",
      success_count: 0,
      failure_count: 6,
      consecutive_failures: 6
    })
    await renderPlayground({ server, persistedModel: `ollama:${OLLAMA_MODEL}` })

    await waitFor(() => expect(modelStatus()).toHaveTextContent("Unreachable"), HARNESS_WAIT)
    expect(
      screen.getByRole("button", { name: `Switch to ${FAKE_MODEL} (Custom OpenAI API)` })
    ).toBeInTheDocument()
  })
})
