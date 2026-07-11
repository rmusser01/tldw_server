import { type Locator, expect } from "@playwright/test"

export async function setQuickIngestSwitch(
  dialog: Locator,
  optionName: string,
  checked: boolean,
  timeout = 20_000
) {
  const escapedName = optionName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  const optionSwitch = dialog
    .getByRole("switch", {
      name: new RegExp(`^Ingestion options\\s*[–-]\\s*${escapedName}$`, "i")
    })
    .first()
  const desiredState = String(checked)

  await expect(optionSwitch).toBeVisible({ timeout })
  if ((await optionSwitch.getAttribute("aria-checked")) !== desiredState) {
    await optionSwitch.click()
  }
  await expect(optionSwitch).toHaveAttribute("aria-checked", desiredState, {
    timeout
  })
}
