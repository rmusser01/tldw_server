import type { Locator, Page } from "@playwright/test"
import { expect, seedAuth, test } from "./smoke.setup"

const section = (page: Page, testId: string): Locator => page.getByTestId(testId)

const expectNoGateBlockers = async (page: Page) => {
  await expect(page.getByTestId("server-readiness-recovery")).toHaveCount(0)
  await expect(page.getByTestId("first-run-gate-overlay")).toHaveCount(0)
}

test.describe("Mermaid chat-card browser QA harness", () => {
  test("renders assistant Mermaid and fallback fixtures without readiness gates", async ({
    page
  }) => {
    test.setTimeout(90_000)
    await seedAuth(page, {
      authMode: "single-user",
      apiKey: "test-key-not-placeholder",
      allowOffline: true
    })

    await page.goto("/__debug__/mermaid-chat-cards")

    await expect(page.getByTestId("mermaid-chat-card-harness")).toBeVisible({
      timeout: 30_000
    })
    await expectNoGateBlockers(page)

    const assistant = section(page, "mermaid-harness-assistant")
    await expect(assistant.getByText("Assistant Mermaid render")).toBeVisible()
    await expect(
      assistant.getByRole("button", { name: "Open Mermaid preview" })
    ).toBeVisible()
    await expect(
      assistant.getByRole("button", { name: "Copy Mermaid source" })
    ).toBeVisible()
    await expect(
      assistant.getByRole("img", { name: "Mermaid diagram" })
    ).toBeVisible()

    const user = section(page, "mermaid-harness-user")
    await expect(user.getByText("```mermaid")).toBeVisible()
    await expect(user.getByText("flowchart TD")).toBeVisible()
    await expect(
      user.getByRole("button", { name: "Open Mermaid preview" })
    ).toHaveCount(0)

    const disabled = section(page, "mermaid-harness-disabled")
    await expect(disabled.getByText("flowchart TD")).toBeVisible()
    await expect(
      disabled.getByRole("button", { name: "Open Mermaid preview" })
    ).toHaveCount(0)

    const invalid = section(page, "mermaid-harness-invalid")
    await expect(
      invalid.getByText("Unable to render Mermaid diagram.")
    ).toBeVisible({
      timeout: 30_000
    })
    await expect(
      invalid.locator("pre").getByText("not a valid mermaid diagram")
    ).toBeVisible()

    const graphviz = section(page, "mermaid-harness-graphviz")
    await expect(graphviz.getByText("digraph G")).toBeVisible()
    await expect(
      graphviz.getByRole("button", { name: "Open Mermaid preview" })
    ).toHaveCount(0)

    const artifact = section(page, "mermaid-harness-artifact")
    await expect(
      artifact.getByRole("img", { name: "Mermaid diagram" })
    ).toBeVisible()
    await expect(
      artifact.getByRole("button", { name: "Open Mermaid preview" })
    ).toBeVisible()
    await expect(
      artifact.getByRole("button", { name: "Copy Mermaid source" })
    ).toBeVisible()
  })
})
