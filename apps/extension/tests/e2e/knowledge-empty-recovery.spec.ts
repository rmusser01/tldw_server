import { expect, test } from "@playwright/test"

import { launchWithBuiltExtension } from "./utils/extension-build"
import { forceConnected, waitForConnectionStore } from "./utils/connection"

test.describe("Knowledge QA extension empty recovery", () => {
  test("shows add/index recovery when the connected backend reports no indexed sources", async () => {
    const serverUrl = "http://dummy-tldw"
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        tldwConfig: {
          serverUrl,
          authMode: "single-user",
          apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
        },
      },
      allowOffline: true,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-empty-recovery:no-indexed")
      await forceConnected(
        page,
        { serverUrl, knowledgeStatus: "empty" },
        "knowledge-empty-recovery:no-indexed"
      )

      await expect(page.getByText("No indexed library sources yet")).toBeVisible()
      await expect(page.getByRole("link", { name: "Add or index sources" })).toBeVisible()

      await page.getByLabel(/Search your knowledge base/i).fill("What does my library say?")
      await expect(page.getByRole("button", { name: /^Ask$/i })).toBeDisabled()
      await expect(
        page.getByText("Add or index library sources before asking Knowledge QA.")
      ).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("shows source-selection recovery when indexed sources exist but none are selected", async () => {
    const serverUrl = "http://dummy-tldw"
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        tldwConfig: {
          serverUrl,
          authMode: "single-user",
          apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
        },
      },
      allowOffline: true,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-empty-recovery:no-selected")
      await forceConnected(
        page,
        { serverUrl, knowledgeStatus: "ready" },
        "knowledge-empty-recovery:no-selected"
      )

      const webToggle = page.getByRole("button", {
        name: /Web fallback is currently/i,
      })
      await expect(webToggle).toBeVisible()
      if ((await webToggle.getAttribute("aria-pressed")) === "true") {
        await webToggle.click()
      }
      await page.getByRole("button", { name: /Open source scope and saved profiles/i }).click()
      const scopeDialog = page.getByRole("dialog", { name: "Source scope and profiles" })
      await expect(scopeDialog).toBeVisible()
      await scopeDialog.getByRole("button", { name: /Sources:/i }).click()
      for (const label of [
        "Documents & Media",
        "Notes",
        "Story Characters",
        "Conversations",
        "Task Boards",
      ]) {
        const sourceOption = scopeDialog.getByRole("menuitemcheckbox", {
          name: new RegExp(label),
        })
        if ((await sourceOption.count()) === 0) continue
        if ((await sourceOption.first().getAttribute("aria-checked")) === "true") {
          await sourceOption.first().click()
        }
      }
      await scopeDialog.getByRole("button", { name: "Close source scope" }).click()

      await expect(page.getByText("No source categories selected")).toBeVisible()
      await expect(page.getByRole("button", { name: "Select source categories" })).toBeVisible()

      await page.getByLabel(/Search your knowledge base/i).fill("What does my library say?")
      await expect(page.getByRole("button", { name: /^Ask$/i })).toBeDisabled()
      await expect(
        page.getByText("Select source categories or enable web fallback before asking Knowledge QA.")
      ).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
