import { test, expect } from "@playwright/test"
import { launchWithExtensionOrSkip } from "./utils/real-server"
import {
  waitForConnectionStore,
  forceConnected
} from "./utils/connection"
import path from "path"

const EXT_PATH = path.resolve("build/chrome-mv3")

test.describe("Sidepanel / Options page handoff", () => {
  test("Open full view from sidepanel opens WebUI chat with handoff context", async () => {
    test.setTimeout(90_000)

    const { context, openSidepanel } =
      await launchWithExtensionOrSkip(test, EXT_PATH, {
        seedConfig: {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: true,
          tldwConfig: {
            serverUrl: "http://127.0.0.1:8000",
            webUiUrl: "http://127.0.0.1:8080",
            authMode: "single-user",
            apiKey: "test-key"
          }
        }
      })

    try {
      const sidepanel = await openSidepanel()
      await waitForConnectionStore(sidepanel, "handoff:sp-store")
      await forceConnected(
        sidepanel,
        { serverUrl: "http://127.0.0.1:8000" },
        "handoff:sp-connected"
      )

      const draft = `handoff draft ${Date.now()}`
      await sidepanel.getByTestId("chat-input").fill(draft)

      const [newPage] = await Promise.all([
        context.waitForEvent("page"),
        sidepanel.getByTestId("chat-open-full-screen").click()
      ])
      await expect
        .poll(() => newPage.url(), { timeout: 10_000 })
        .toContain("http://127.0.0.1:8080/chat")

      const openedUrl = new URL(newPage.url())
      expect(openedUrl.pathname).toBe("/chat")
      expect(openedUrl.href).not.toContain("/options.html")
      expect(openedUrl.searchParams.has("handoff")).toBe(false)

      const encodedHandoff = new URLSearchParams(
        openedUrl.hash.slice(1)
      ).get("handoff")
      expect(encodedHandoff).toBeTruthy()
      const decodedHandoff = JSON.parse(
        Buffer.from(
          encodedHandoff!.replace(/-/g, "+").replace(/_/g, "/"),
          "base64"
        ).toString("utf8")
      )
      expect(decodedHandoff).toMatchObject({
        source: "sidepanel-chat",
        draft,
        chatMode: "normal",
        webSearch: false,
        toolChoice: "none"
      })

      await context.close()
    } catch (error) {
      await context.close()
      throw error
    }
  })

  test("settings changed in options page are accessible from sidepanel storage", async () => {
    test.setTimeout(90_000)

    const { context, page, openSidepanel, extensionId } =
      await launchWithExtensionOrSkip(test, EXT_PATH, {
        seedConfig: {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: true,
          tldwConfig: {
            serverUrl: "http://127.0.0.1:8000",
            authMode: "single-user",
            apiKey: "test-key"
          }
        }
      })

    try {
      // Write a value via chrome.storage from the options page
      const testValue = `handoff-test-${Date.now()}`
      await page.evaluate(
        (val) =>
          new Promise<void>((resolve) => {
            chrome.storage.local.set(
              { __e2e_handoff_test: val },
              () => resolve()
            )
          }),
        testValue
      )

      // Open sidepanel and read the value back
      const sidepanel = await openSidepanel()
      const readValue = await sidepanel.evaluate(
        () =>
          new Promise<string | null>((resolve) => {
            if (typeof chrome === "undefined" || !chrome.storage?.local) {
              resolve(null)
              return
            }
            chrome.storage.local.get("__e2e_handoff_test", (items) => {
              resolve(items?.__e2e_handoff_test ?? null)
            })
          })
      )

      expect(readValue).toBe(testValue)

      await context.close()
    } catch (error) {
      await context.close()
      throw error
    }
  })
})
