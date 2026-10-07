// @vitest-environment jsdom
/**
 * Red-first reproduction for #3111 (CC-01): on a first run with no persisted
 * model, the composer must start on the server's default provider rather than
 * whichever configured provider is listed first. The `it.fails` asserts the
 * correct behaviour on what the user observes (the request a send makes), so
 * it flips however the fix is implemented.
 */
import {
  completionMessages,
  createFakeTldwServer,
  FAKE_MODEL,
  FAKE_PROVIDER,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  sendFromComposer,
  waitFor
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

describe("Playground startup model selection (#3111)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  // CC-01 #3111 — model-startup-selection.ts:19-61 has no server-default input (favorites, then the first listed model),
  // and PlaygroundForm.tsx:1530-1540 applies it, so the catalog's first provider wins over default_provider.
  it.fails("CC-01 (#3111): a first run starts on the server's default provider model, not the first listed provider", async () => {
    const server = createFakeTldwServer({
      catalog: {
        providers: [
          { name: "ollama", displayName: "Ollama", models: ["llama3.2:3b"] },
          { name: FAKE_PROVIDER, displayName: "Custom OpenAI API", models: [FAKE_MODEL] }
        ],
        defaultProvider: FAKE_PROVIDER
      }
    })
    const view = await renderPlayground({ server, persistedModel: null })
    await waitFor(() => expect(useStoreMessageOption.getState().selectedModel).toBeTruthy(), HARNESS_WAIT)

    await sendFromComposer(view, "Which model answers?")

    const [request] = server.completionRequests()
    expect(completionMessages(request)).toEqual([{ role: "user", content: "Which model answers?" }])
    expect({ provider: request.body?.api_provider, model: request.body?.model }).toEqual({
      provider: FAKE_PROVIDER,
      model: FAKE_MODEL
    })
  })
})
