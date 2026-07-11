import { expect, test, type BrowserContext } from '@playwright/test'
import path from 'path'
import { launchWithBuiltExtension } from './utils/extension-build'
import { setQuickIngestSwitch } from './utils/quick-ingest-options'

const SERVER_URL = process.env.TLDW_URL || 'http://127.0.0.1:8000'
const API_KEY = process.env.TLDW_API_KEY || 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY'

test.describe('Quick Ingest workflows and UX', () => {
  test('URLs, files, inspector intro, and help flows', async ({}, testInfo) => {
    let context: BrowserContext | null = null

    try {
      const launchResult = await launchWithBuiltExtension({
        seedConfig: {
          serverUrl: SERVER_URL,
          authMode: 'single-user',
          apiKey: API_KEY
        },
        allowOffline: false
      })
      context = launchResult.context
      const { page, optionsUrl } = launchResult

      await page.goto(optionsUrl + '#/playground', { waitUntil: 'domcontentloaded' })
      await page.waitForLoadState('networkidle')
      // Force a connection check with the seeded config
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('tldw:check-connection'))
      })

      // Click header trigger directly (online path), fallback to media route if missing
      const resolveQuickIngestTrigger = async () => {
        const byTestId = page.getByTestId('open-quick-ingest')
        if (await byTestId.count()) return byTestId.first()
        return page.getByRole('button', { name: /quick ingest/i }).first()
      }

      let trigger = await resolveQuickIngestTrigger()
      if (!(await trigger.count())) {
        await page.goto(optionsUrl + '#/media', { waitUntil: 'domcontentloaded' })
        await page.waitForLoadState('networkidle')
        trigger = await resolveQuickIngestTrigger()
      }
      await expect(trigger).toBeVisible({ timeout: 5000 })
      const tooltip = await trigger.getAttribute('title')
      if (tooltip) {
        expect(tooltip).toMatch(
          /Import URLs, documents, and media to your knowledge base|Quick ingest/i
        )
      }
      await trigger.click()

      // Wait for modal content (root may briefly stay hidden)
      await expect(page.locator('.quick-ingest-modal')).not.toHaveAttribute('hidden', 'true', { timeout: 10_000 })
      await expect(page.locator('.quick-ingest-modal [data-state="ready"]')).toBeVisible({ timeout: 10_000 })
      const notConnectedBanner = page.getByText(/Not connected to server/i).first()
      await notConnectedBanner.waitFor({ state: 'hidden', timeout: 10_000 }).catch(() => {})

      // Workflow 1: add URLs and see queue + Inspector intro
      const urlInput = page
        .getByLabel(/Paste URLs input/i)
        .or(page.getByPlaceholder(/https:\/\/example\.com/i))
        .first()
      await urlInput.click()
      await urlInput.fill('https://example.com, https://example.org')
      await page.getByRole('button', { name: /Add URLs/i }).click()

      const firstUrlRow = page.getByText('https://example.com').first()
      await expect(firstUrlRow).toBeVisible()
      await firstUrlRow.click()

      const intro = page.getByText(/How to use the Inspector/i)
      await expect(intro).toBeVisible()
      await expect(page.getByRole('button', { name: /Dismiss Inspector intro and close/i })).toBeVisible()
      await page.getByRole('button', { name: /Dismiss Inspector intro and close/i }).click()
      await expect(intro).toBeHidden()

      // Workflow 2: reopen intro via help button
      await page.getByRole('button', { name: /Open Inspector intro/i }).click()
      await expect(intro).toBeVisible()

      // Workflow 3: upload a file, open Inspector, toggle options
      const sampleFile = path.resolve(__dirname, '..', '..', 'README.md')
      await page.setInputFiles('[data-testid="qi-file-input"]', sampleFile)

      const fileRow = page.getByText('README.md').first()
      await expect(fileRow).toBeVisible()
      await fileRow.click()

      await expect(page.getByText(/File settings follow/i)).toBeVisible()

      const dialog = page.getByRole('dialog', { name: /quick ingest/i }).first()
      await setQuickIngestSwitch(dialog, 'analysis', false)
      await setQuickIngestSwitch(dialog, 'chunking', false)

      // Snapshot for UX review
      const screenshotPath = testInfo.outputPath('quick-ingest-workflows.png')
      await page.screenshot({ path: screenshotPath, fullPage: true })
    } finally {
      await context?.close()
    }
  })
})
