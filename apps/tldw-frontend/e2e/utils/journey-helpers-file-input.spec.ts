import { writeFile } from "node:fs/promises"
import { test, expect } from "@playwright/test"
import { queueFileForQuickIngest } from "./journey-helpers"

test("queues a new upload when the reattach input precedes it", async ({ page }, testInfo) => {
  const fileName = "knowledge-source.txt"
  const filePath = testInfo.outputPath(fileName)
  await writeFile(filePath, "A source to ingest", "utf8")
  await page.setContent(`
    <div role="dialog" aria-label="Quick Ingest">
      <input type="file" aria-label="Reattach queued file" hidden />
      <button aria-pressed="false">Files</button>
      <input type="file" data-testid="qi-file-input" hidden />
      <section aria-label="Queued sources"><output></output></section>
    </div>
  `)
  await page.getByTestId("qi-file-input").evaluate((input: HTMLInputElement) => {
    input.addEventListener("change", () => {
      document.querySelector("output")!.textContent = input.files![0].name
    })
  })

  const dialog = page.getByRole("dialog", { name: "Quick Ingest" })
  await queueFileForQuickIngest(dialog, filePath, 500)
  await expect(dialog.getByRole("region", { name: "Queued sources" })).toHaveText(fileName)
  expect(await dialog.getByLabel("Reattach queued file").inputValue()).toBe("")
})
