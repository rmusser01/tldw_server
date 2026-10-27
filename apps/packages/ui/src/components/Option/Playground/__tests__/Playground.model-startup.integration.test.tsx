// @vitest-environment jsdom
/**
 * Red-first reproduction for #3111 (CC-01): on a first run with no persisted
 * model, the composer must start on the server's default provider rather than
 * whichever configured provider is listed first. Assert the provider/model
 * in the actual send request, independent of the selection implementation.
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

describe(
  "Playground startup model selection (#3111)",
  { timeout: 60_000 },
  () => {
    beforeEach(async () => {
      await resetPlaygroundHarness()
    })

    it.each([FAKE_PROVIDER, "custom-openai-api-3", "poe", "minimax", "tabby"])(
      "CC-01 (#3111): a first run uses the declared default provider %s",
      async (defaultProvider) => {
        const server = createFakeTldwServer({
          catalog: {
            providers: [
              {
                name: "ollama",
                displayName: "Ollama",
                models: ["llama3.2:3b"]
              },
              {
                name: defaultProvider,
                displayName: "Custom OpenAI API",
                models: [FAKE_MODEL]
              }
            ],
            defaultProvider
          }
        })
        const view = await renderPlayground({ server, persistedModel: null })
        await waitFor(
          () =>
            expect(useStoreMessageOption.getState().selectedModel).toBeTruthy(),
          HARNESS_WAIT
        )

        await sendFromComposer(view, "Which model answers?")

        const [request] = server.completionRequests()
        expect(completionMessages(request)).toEqual([
          { role: "user", content: "Which model answers?" }
        ])
        expect({
          provider: request.body?.api_provider,
          model: request.body?.model
        }).toEqual({
          provider: defaultProvider === "tabby" ? "tabbyapi" : defaultProvider,
          model: FAKE_MODEL
        })
      }
    )

    it("keeps the default provider when another provider offers the same model ID", async () => {
      const server = createFakeTldwServer({
        catalog: {
          providers: [
            { name: "ollama", models: [FAKE_MODEL] },
            { name: FAKE_PROVIDER, models: [FAKE_MODEL] }
          ],
          defaultProvider: FAKE_PROVIDER
        }
      })
      const view = await renderPlayground({ server, persistedModel: null })
      await waitFor(
        () =>
          expect(useStoreMessageOption.getState().selectedModel).toBeTruthy(),
        HARNESS_WAIT
      )

      await sendFromComposer(view, "Which provider answers?")

      expect(server.completionRequests()[0]?.body).toMatchObject({
        api_provider: FAKE_PROVIDER,
        model: FAKE_MODEL
      })
    })

    it("preserves a saved provider-qualified choice ahead of the server default", async () => {
      const server = createFakeTldwServer({
        catalog: {
          providers: [
            { name: "ollama", models: ["llama3.2:3b"] },
            { name: FAKE_PROVIDER, models: [FAKE_MODEL] }
          ],
          defaultProvider: "ollama"
        }
      })
      const view = await renderPlayground({
        server,
        persistedModel: `${FAKE_PROVIDER}:${FAKE_MODEL}`
      })

      await sendFromComposer(view, "Keep my model choice.")

      expect(server.completionRequests()[0]?.body).toMatchObject({
        api_provider: FAKE_PROVIDER,
        model: FAKE_MODEL
      })
    })
  }
)
