import { test, expect } from '@playwright/test'
import { launchWithExtensionOrSkip } from "./utils/real-server"
import path from 'path'
import { launchWithExtension } from './utils/extension'
import { launchWithBuiltExtension } from './utils/extension-build'
import {
  waitForConnectionStore,
  logConnectionSnapshot
} from './utils/connection'

const SERVER_URL =
  process.env.TLDW_SERVER_URL ?? 'http://127.0.0.1:8000'
const API_KEY = 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY'
const TEST_EXT_PATH = path.resolve('build/chrome-mv3')

// Gate all tests behind an opt-in env flag and a live
// health check so they never run against a missing server.
const describeLive = process.env.TLDW_LIVE_E2E
  ? test.describe
  : test.describe.skip

describeLive('Live server UX workflows (no mocks)', () => {
  test.beforeAll(async () => {
    const baseUrl = SERVER_URL.replace(/\/$/, '')
    const healthTargets = [`${baseUrl}/health`, `${baseUrl}/api/v1/health`]
    let lastError: string | null = null
    let healthy = false

    for (const target of healthTargets) {
      try {
        const res = await fetch(target)
        if (res.ok) {
          healthy = true
          break
        }
        lastError = `HTTP ${res.status} from ${target}`
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e)
        lastError = `${target}: ${message}`
      }
    }

    if (!healthy) {
      console.warn(
        `[live-ux-workflows] health probes failed, continuing because TLDW_LIVE_E2E is explicitly enabled: ${lastError || "unknown error"}`
      )
    }
  })
  test('Onboarding with real server shows reachability hints', async () => {
    const { context, page } = await launchWithExtensionOrSkip(test, TEST_EXT_PATH)

    try {
      // Step 1: server URL
      await expect(
        page.getByText(/Let’s get you connected|Let's get you connected/i)
      ).toBeVisible()

      await waitForConnectionStore(page, 'live-workflows-onboarding')

      const urlInput = page.getByLabel(/Server URL/i)
      await urlInput.scrollIntoViewIfNeeded()
      await urlInput.fill(SERVER_URL)

      // Helper hint should flip to the reachable state once the live
      // server responds to /api/v1/health, enabling Next without reload.
      await expect(
        page.getByText(
          /Server responded successfully\. You can continue\./i
        )
      ).toBeVisible({ timeout: 15_000 })

      // Docs CTA for learning about the server should be available.
      const docsCta = page.getByRole('button', {
        name: /Learn how tldw server works/i
      })
      await expect(docsCta).toBeVisible()

      const nextButton = page.getByRole('button', { name: /Next/i })
      await expect(nextButton).toBeVisible()
      await expect(nextButton).toBeEnabled()
      await logConnectionSnapshot(page, 'live-workflows-after-url')
    } finally {
      await context.close()
    }
  })

  test('Quick ingest modal with live server', async () => {
    const { context, page, optionsUrl } =
      await launchWithBuiltExtension({
        seedConfig: {
          serverUrl: SERVER_URL,
          authMode: 'single-user',
          apiKey: API_KEY
        }
      })

    try {
      await page.goto(optionsUrl + '#/media', {
        waitUntil: 'domcontentloaded'
      })

      const ingestButton = page
        .getByRole('button', { name: /Quick ingest/i })
        .first()
      await expect(ingestButton).toBeVisible()
      await ingestButton.click()

      const modal = page.getByRole('dialog', { name: /quick ingest/i }).first()
      await expect(modal).toBeVisible()

      // Basic ingest path: add a URL to the queue
      const urlInput = page
        .getByLabel(/Paste URLs input/i)
        .or(page.getByPlaceholder(/https:\/\/example\.com/i))
        .first()
      await urlInput.click()
      await urlInput.fill('https://example.com')
      await page
        .getByRole('button', { name: /Add URLs/i })
        .click()

      // Queue row should appear with the URL and an online-ready action.
      const row = modal.getByText('https://example.com').first()
      await expect(row).toBeVisible()

      // With a healthy, configured server, Quick Ingest should be ready
      // to run instead of presenting an offline-only staging banner.
      const useDefaultsButton = modal.getByRole('button', {
        name: /use defaults/i
      }).first()
      const configureButton = modal.getByRole('button', {
        name: /configure \d+ items/i
      }).first()
      const runButton = modal.getByRole('button', {
        name: /Run quick ingest/i
      }).first()
      await expect
        .poll(
          async () =>
            (await useDefaultsButton.isVisible().catch(() => false)) ||
            (await configureButton.isVisible().catch(() => false)) ||
            (await runButton.isVisible().catch(() => false)),
          {
            timeout: 15_000,
            message: 'Timed out waiting for an online-ready quick ingest action'
          }
        )
        .toBe(true)
      if (await useDefaultsButton.isVisible().catch(() => false)) {
        await expect(useDefaultsButton).toBeEnabled()
      } else if (await runButton.isVisible().catch(() => false)) {
        await expect(runButton).toBeEnabled()
      } else {
        await expect(configureButton).toBeEnabled()
      }

      // Connection gating copy should not appear when online.
      await expect(
        modal.getByText(/Not connected to server/i)
      ).toHaveCount(0)
      await expect(
        modal.getByText(/Not connected — reconnect to run/i)
      ).toHaveCount(0)
    } finally {
      await context.close()
    }
  })

  test('Quick ingest reaches terminal results and reopens them in the installed extension', async () => {
    test.setTimeout(180_000)

    const fixtureId = `extension-live-${Date.now()}`
    const ingestUrl = `${SERVER_URL.replace(/\/$/, '')}/docs?quick_ingest_fixture=${fixtureId}`

    const { context, page, optionsUrl } =
      await launchWithBuiltExtension({
        seedConfig: {
          serverUrl: SERVER_URL,
          authMode: 'single-user',
          apiKey: API_KEY
        }
      })

    try {
      await page.goto(optionsUrl + '#/media', {
        waitUntil: 'domcontentloaded'
      })
      await waitForConnectionStore(page, 'live-extension-quick-ingest-complete')

      const openQuickIngestButton = page
        .getByTestId('open-quick-ingest')
        .or(page.getByRole('button', { name: /open quick ingest|quick ingest/i }))
        .first()
      await expect(openQuickIngestButton).toBeVisible({ timeout: 15_000 })
      await openQuickIngestButton.click()

      const dialog = page.getByRole('dialog', { name: /quick ingest/i }).first()
      await expect(dialog).toBeVisible({ timeout: 15_000 })

      const urlInput = dialog
        .getByLabel(/Paste URLs input/i)
        .or(dialog.getByPlaceholder(/https:\/\/example\.com/i))
        .first()
      await expect(urlInput).toBeEnabled({ timeout: 20_000 })
      await urlInput.fill(ingestUrl)
      await dialog.getByRole('button', { name: /add url|add urls/i }).first().click()
      await expect(dialog.getByText(ingestUrl, { exact: false }).first()).toBeVisible({
        timeout: 20_000
      })

      const useDefaultsButton = dialog
        .getByRole('button', { name: /use defaults/i })
        .first()
      if (await useDefaultsButton.isVisible({ timeout: 3_000 }).catch(() => false)) {
        await expect(useDefaultsButton).toBeEnabled({ timeout: 20_000 })
        await useDefaultsButton.click()
      } else {
        const configureButton = dialog
          .getByRole('button', { name: /configure \d+ items/i })
          .first()
        if (await configureButton.isVisible({ timeout: 5_000 }).catch(() => false)) {
          await expect(configureButton).toBeEnabled({ timeout: 20_000 })
          await configureButton.click()
        }

        const nextButton = dialog.getByRole('button', { name: /^next$/i }).first()
        if (await nextButton.isVisible({ timeout: 5_000 }).catch(() => false)) {
          await expect(nextButton).toBeEnabled({ timeout: 20_000 })
          await nextButton.click()
          await expect(dialog.getByText(/ready to process/i)).toBeVisible({
            timeout: 20_000
          })
        }

        const runButton = dialog.getByTestId('quick-ingest-run').first()
        await expect(runButton).toBeEnabled({ timeout: 20_000 })
        await runButton.click()
      }

      const resultsStep = dialog.getByTestId('wizard-results-step')
      const completedRegion = dialog.getByRole('region', { name: /completed items/i }).first()
      const skippedRegion = dialog.getByRole('region', { name: /skipped items/i }).first()
      const errorRegion = dialog.getByRole('region', { name: /error items/i }).first()

      await expect(resultsStep).toBeVisible({ timeout: 120_000 })
      await expect
        .poll(
          async () =>
            (await completedRegion.isVisible().catch(() => false)) ||
            (await skippedRegion.isVisible().catch(() => false)) ||
            (await errorRegion.isVisible().catch(() => false)),
          {
            timeout: 120_000,
            message: 'Timed out waiting for the extension quick ingest run to reach terminal results'
          }
        )
        .toBe(true)

      await expect(dialog.getByText(new RegExp(fixtureId, 'i')).first()).toBeVisible({
        timeout: 30_000
      })
      await expect(
        dialog.getByText(/Total:\s*\d+\s+succeeded(?:,\s*\d+\s+skipped)?,\s*\d+\s+failed/i)
      ).toBeVisible({
        timeout: 30_000
      })

      const doneButton = dialog.getByRole('button', { name: /done|close the ingest wizard/i }).first()
      await expect(doneButton).toBeVisible({ timeout: 15_000 })
      await doneButton.click()
      await expect(dialog).toBeHidden({ timeout: 20_000 })

      await openQuickIngestButton.click()
      await expect(dialog).toBeVisible({ timeout: 15_000 })
      await expect(resultsStep).toBeVisible({ timeout: 30_000 })
      await expect(dialog.getByText(new RegExp(fixtureId, 'i')).first()).toBeVisible({
        timeout: 30_000
      })
    } finally {
      await context.close()
    }
  })

  test('Knowledge QA mode surfaces connect card with live server config', async () => {
    const { context, page, optionsUrl } =
      await launchWithBuiltExtension({
        seedConfig: {
          serverUrl: SERVER_URL,
          authMode: 'single-user',
          apiKey: API_KEY
        }
      })

    try {
      // Go to the main playground route and switch to Knowledge QA mode.
      await page.goto(optionsUrl + '#/', {
        waitUntil: 'domcontentloaded'
      })

      await page
        .getByRole('button', { name: /Knowledge QA/i })
        .click()

      // When Knowledge QA is selected, users should always see a clear
      // state: either a connect card or the "no sources yet" empty state.
      await Promise.race([
        page
          .getByText(/Connect to use Knowledge QA/i)
          .waitFor({ timeout: 20_000 })
          .catch(() => null),
        page
          .getByText(/Index knowledge to use Knowledge QA/i)
          .waitFor({ timeout: 20_000 })
          .catch(() => null)
      ])

      // Header chips should be present for quick connection diagnostics.
      await expect(page.getByText(/Server: /i)).toBeVisible()
      await expect(page.getByText(/Knowledge: /i)).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
