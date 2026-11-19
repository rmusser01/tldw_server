import { test, expect } from "@playwright/test"
import path from "path"
import { waitForConnectionStore, forceConnected } from "./utils/connection"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value : `http://${value}`

test.describe('Options first-run and connection panel', () => {
  test('shows onboarding shell on first run', async () => {
    const extPath = path.resolve('build/chrome-mv3')
    const { context, page } = await launchWithExtensionOrSkip(test, extPath)

    // Clear storage to guarantee first-run onboarding state.
    await page.evaluate(async () => {
      await new Promise<void>((resolve) => {
        // @ts-ignore
        chrome.storage.local.clear(() => resolve())
      })
    })
    await page.reload()

    // First-run path should show onboarding shell + connect form.
    await expect(page.getByText(/Home Onboarding/i)).toBeVisible()
    await expect(
      page.getByText(/Welcome to tldw Browser Assistant.*connected/i)
    ).toBeVisible()
    await expect(page.getByText(/^Server URL$/i)).toBeVisible()
    await expect(page.getByText(/^API Key$/i)).toBeVisible()
    await expect(page.getByRole('button', { name: /Try Demo/i })).toBeVisible()

    await context.close()
  })

  test('Start chatting focuses the composer when connected', async () => {
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const serverBaseUrl = normalizeServerUrl(serverUrl)

    const extPath = path.resolve('build/chrome-mv3')
    const seed = {
      __tldw_first_run_complete: true,
      tldwConfig: {
        serverUrl: serverBaseUrl,
        authMode: 'single-user',
        apiKey
      }
    }
    const { context, page: initialPage, extensionId } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seed
    })
    let page = initialPage
    const optionsUrl = `chrome-extension://${extensionId}/options.html`

    await page.goto(optionsUrl, { waitUntil: 'domcontentloaded' })
    await page.waitForSelector('#root', { state: 'attached', timeout: 5000 })

    // Force connected state via shared helper to avoid network flakiness.
    await waitForConnectionStore(page, 'options-first-run-connected')
    await forceConnected(page, { serverUrl: serverBaseUrl }, 'options-first-run-connected')

    // Landing hub should expose Start Chatting and route to the chat composer.
    const landingDialog = page.getByRole('dialog', { name: /Welcome to tldw/i })
    const useDialogButton = await landingDialog.isVisible().catch(() => false)
    const startChattingButton = useDialogButton
      ? landingDialog.getByRole('button', { name: /Start Chatting/i }).first()
      : page.getByRole('button', { name: /Start Chatting/i }).first()
    await expect(startChattingButton).toBeVisible()
    await startChattingButton.click()
    await expect(page).toHaveURL(/options\.html#\/chat/i)
    const composer = page.locator('#textarea-message')
    await expect(composer).toBeVisible()
    await expect(composer).toBeFocused()

    await context.close()
  })
})
