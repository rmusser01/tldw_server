// @vitest-environment jsdom
/**
 * CC-06 (#3111): one way to name the chat model. The composer chip puts the
 * model first and the provider second, a streaming reply is not headed with a
 * raw `provider_key:model` id, the picker's "Current" row names its provider,
 * and the picker's scope control really switches between the models that are
 * ready to use and every model the server knows.
 */
import {
  createFakeTldwServer,
  deferred,
  FAKE_MODEL,
  FAKE_PROVIDER,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitFor,
  waitForChatIdle,
  within
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const RAW_MODEL_KEY = /custom[-_]openai[-_]api:local-uat-chat/

const openModelPicker = async (view: Awaited<ReturnType<typeof renderPlayground>>) => {
  await view.user.click(screen.getByTestId("model-selector"))
  return await screen.findByRole("menu", {}, HARNESS_WAIT)
}

/** Text showing a raw provider:model id on screen. */
const rawModelKeysOnScreen = () => {
  const found: string[] = []
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent ?? ""
    // Off-screen route text (screen-reader only, or a closed disclosure) is allowed.
    if (RAW_MODEL_KEY.test(text) && !node.parentElement?.closest("details:not([open]), .sr-only")) {
      found.push(text)
    }
  }
  return found
}

/** Segmented options hide their radio input; people click the option label. */
const clickScopeOption = async (
  view: Awaited<ReturnType<typeof renderPlayground>>,
  name: string
) => {
  const radio = screen.getByRole("radio", { name })
  await view.user.click(radio.closest("label") ?? radio)
  return radio
}

const pickerOption = (modelId: string) =>
  document.querySelector<HTMLElement>(
    `[data-testid="model-selector-option"][data-model-id="${modelId}"]`
  )

describe("Playground model identity (CC-06, #3111)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("CC-06: the composer chip names the model first and the provider second", async () => {
    await renderPlayground({ server: createFakeTldwServer() })

    const chip = screen.getByTestId("model-selector")
    await waitFor(() => expect(chip.textContent ?? "").toMatch(/^local-uat-chat/), HARNESS_WAIT)
    expect(chip).toHaveTextContent("Custom OpenAI API")
    expect(chip.textContent).not.toContain("/")
  })

  it("CC-06: a streaming reply is headed with the model name, not the provider key", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    const resume = deferred()
    server.planCompletion({ chunks: ["Partial ", "answer"], pauseAfterChunks: 1, resume: resume.promise })

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

    expect(rawModelKeysOnScreen()).toEqual([])
    expect(screen.getAllByText(FAKE_MODEL, { exact: true }).length).toBeGreaterThan(0)

    resume.resolve()
    await waitForChatIdle()
  })

  it("CC-06: the picker's Current row names its provider", async () => {
    const view = await renderPlayground({ server: createFakeTldwServer() })

    await openModelPicker(view)

    const option = await waitFor(() => {
      const found = pickerOption(FAKE_MODEL)
      expect(found).not.toBeNull()
      return found as HTMLElement
    }, HARNESS_WAIT)
    expect(within(option).getByText("Custom OpenAI API")).toBeInTheDocument()
  })

  it("CC-06: 'All models' adds models whose provider is not set up, and 'Ready to use' hides them", async () => {
    const server = createFakeTldwServer({
      catalog: {
        providers: [
          { name: FAKE_PROVIDER, displayName: "Custom OpenAI API", models: [FAKE_MODEL] },
          { name: "anthropic", displayName: "Anthropic", models: ["claude-catalog"], configured: false }
        ],
        defaultProvider: FAKE_PROVIDER
      }
    })
    const view = await renderPlayground({ server })

    await openModelPicker(view)
    const readyToUse = screen.getByRole("radio", { name: "Ready to use" })
    const allModels = screen.getByRole("radio", { name: "All models" })
    expect(readyToUse).toBeChecked()
    expect(pickerOption("claude-catalog")).toBeNull()

    await clickScopeOption(view, "All models")

    const catalogOption = await waitFor(() => {
      const found = pickerOption("claude-catalog")
      expect(found).not.toBeNull()
      return found as HTMLElement
    }, HARNESS_WAIT)
    expect(allModels).toBeChecked()
    expect(within(catalogOption).getByText("Not set up")).toBeInTheDocument()

    await clickScopeOption(view, "Ready to use")
    await waitFor(() => expect(pickerOption("claude-catalog")).toBeNull(), HARNESS_WAIT)
    expect(pickerOption(FAKE_MODEL)).not.toBeNull()
  })
})
