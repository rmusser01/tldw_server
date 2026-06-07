import { test, expect } from "@playwright/test"

import { launchWithBuiltExtension } from "./utils/extension-build"
import {
  forceConnected,
  forceUnconfigured,
  waitForConnectionStore,
} from "./utils/connection"

test.describe("Knowledge QA deterministic extension states", () => {
  test("extension renders setup-required state without live backend data", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {},
      allowOffline: false,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-qa-setup-required")
      await forceUnconfigured(page, "knowledge-qa-setup-required")

      await expect(page.getByText(/Setup Required/i)).toBeVisible()
      await expect(page.getByRole("button", { name: /Finish setup/i })).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("extension renders connected ready search state without live backend data", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        tldwConfig: {
          serverUrl: "http://dummy-tldw",
          authMode: "single-user",
          apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
        },
      },
      allowOffline: true,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-qa-ready")
      await forceConnected(
        page,
        { serverUrl: "http://dummy-tldw" },
        "knowledge-qa-ready"
      )

      await expect(page.getByRole("heading", { name: /Ask Your Library/i })).toBeVisible()
      await expect(page.getByLabel(/Search your knowledge base/i)).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
