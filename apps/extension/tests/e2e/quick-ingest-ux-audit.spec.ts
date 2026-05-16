import { test, expect, type Locator, type Page } from '@playwright/test'
import { launchWithBuiltExtension } from './utils/extension-build'
import {
  waitForConnectionStore,
  forceUnconfigured,
  forceErrorUnreachable
} from './utils/connection'

const API_KEY = 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY'

const openQuickIngestDialog = async (page: Page): Promise<Locator> => {
  const ingestButton = page
    .getByRole('button', { name: /^Quick ingest$/i })
    .or(page.getByRole('button', { name: /open quick ingest/i }))
    .first()
  await expect(ingestButton).toBeVisible()
  await ingestButton.click()

  const dialog = page.getByRole('dialog', { name: /quick ingest/i }).first()
  await expect(dialog).toBeVisible()
  return dialog
}

const addUrlToQuickIngest = async (dialog: Locator, url: string) => {
  const urlInput = dialog
    .getByLabel(/url input area|paste urls input/i)
    .or(dialog.getByPlaceholder(/https:\/\/example\.com/i))
    .first()
  await urlInput.click()
  await urlInput.fill(url)
  const addButton = dialog.locator('button').filter({ hasText: /^Add$/i }).last()
  await expect(addButton).toBeEnabled()
  await addButton.click()
  await expect(
    dialog.getByRole('button', { name: /configure \d+ items/i })
  ).toBeEnabled()
}

const startQueuedQuickIngest = async (dialog: Locator) => {
  const useDefaultsButton = dialog
    .getByRole('button', { name: /use defaults & process/i })
    .first()
  if (await useDefaultsButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await useDefaultsButton.click()
    return
  }

  const configureButton = dialog
    .getByRole('button', { name: /configure \d+ items/i })
    .first()
  await expect(configureButton).toBeVisible()
  await configureButton.click()

  const nextButton = dialog.getByRole('button', { name: /^next$/i }).first()
  await expect(nextButton).toBeVisible()
  await nextButton.click()

  const startButton = dialog.getByRole('button', { name: /start processing/i }).first()
  await expect(startButton).toBeVisible()
  await startButton.click()
}

const patchQuickIngestRuntimeResults = async (
  page: Page,
  results: Array<Record<string, unknown>>
) => {
  return page.evaluate((items) => {
    try {
      const runtime =
        (globalThis as any)?.browser?.runtime ||
        (globalThis as any)?.chrome?.runtime
      const onMessage = runtime?.onMessage
      const originalSendMessage =
        typeof runtime?.sendMessage === 'function'
          ? runtime.sendMessage.bind(runtime)
          : null
      const originalGetManifest =
        typeof runtime?.getManifest === 'function'
          ? runtime.getManifest.bind(runtime)
          : null
      if (!runtime || !onMessage || !originalSendMessage) {
        return false
      }

      const originalAddListener =
        typeof onMessage.addListener === 'function'
          ? onMessage.addListener.bind(onMessage)
          : null
      const originalRemoveListener =
        typeof onMessage.removeListener === 'function'
          ? onMessage.removeListener.bind(onMessage)
          : null
      const listeners = new Set<(message: any, sender?: any, sendResponse?: any) => void>()

      const emit = (message: any) => {
        for (const listener of [...listeners]) {
          listener(message, {}, () => undefined)
        }
      }

      onMessage.addListener = (listener: any) => {
        listeners.add(listener)
      }
      onMessage.removeListener = (listener: any) => {
        listeners.delete(listener)
      }
      runtime.getManifest = () => ({
        ...(originalGetManifest?.() || {}),
        manifest_version: 2
      })

      runtime.sendMessage = async (message: any) => {
        if (message?.type === 'tldw:ping') {
          return { ok: true }
        }
        if (message?.type === 'tldw:quick-ingest/start') {
          const sessionId = 'qi-e2e-ux-audit-session'
          setTimeout(() => {
            emit({
              type: 'tldw:quick-ingest/completed',
              payload: {
                sessionId,
                results: items
              }
            })
          }, 50)
          return { ok: true, sessionId }
        }
        if (message?.type === 'tldw:quick-ingest-batch') {
          return { ok: true, results: items }
        }
        return originalSendMessage(message)
      }

      ;(window as any).__restoreQuickIngestUxAuditPatch = () => {
        runtime.sendMessage = originalSendMessage
        if (originalGetManifest) {
          runtime.getManifest = originalGetManifest
        }
        if (originalAddListener) {
          onMessage.addListener = originalAddListener
        }
        if (originalRemoveListener) {
          onMessage.removeListener = originalRemoveListener
        }
        listeners.clear()
      }
      return true
    } catch {
      return false
    }
  }, results)
}

const restoreQuickIngestRuntimePatch = async (page: Page) => {
  try {
    await page.evaluate(() => {
      try {
        const restore = (window as any).__restoreQuickIngestUxAuditPatch
        if (typeof restore === 'function') {
          restore()
        }
        delete (window as any).__restoreQuickIngestUxAuditPatch
      } catch {
        // ignore best-effort cleanup failures
      }
    })
  } catch {
    // ignore cleanup failures if page/context is already torn down
  }
}

test.describe('Quick ingest – UX audit', () => {
  test('first-time user sees purpose and supported input copy before configuration', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: 'http://127.0.0.1:8000',
        authMode: 'single-user',
        apiKey: API_KEY
      }
    })

    try {
      await page.goto(optionsUrl + '#/media', { waitUntil: 'domcontentloaded' })

      const modal = await openQuickIngestDialog(page)

      await expect(
        modal.getByText(/Add URLs or files\. Stored items appear in Media/i)
      ).toBeVisible()
      await expect(
        modal.getByText(/Supported: PDF, EPUB, DOC\/DOCX, TXT\/RTF, Markdown, HTML, XML, JSON, audio, video/i)
      ).toBeVisible()
      await expect(
        modal.getByText(/Max file size: 50 MB/i)
      ).toBeVisible()

      // Advanced options belong to the configure step, not the first add step.
      await expect(
        modal.getByText(/Advanced options/i)
      ).toHaveCount(0)

      // Inspector drawer should not be open by default.
      await expect(
        page.getByRole('dialog', { name: /Inspector/i })
      ).toHaveCount(0)
    } finally {
      await context.close()
    }
  })

  test('success results surface a Media handoff action', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: 'http://127.0.0.1:8000',
        authMode: 'single-user',
        apiKey: API_KEY
      }
    })

    try {
      await page.goto(optionsUrl + '#/media', { waitUntil: 'domcontentloaded' })

      const patched = await patchQuickIngestRuntimeResults(page, [
        {
          id: 'ok-1',
          status: 'ok',
          type: 'html',
          url: 'https://example.com',
          title: 'Example quick ingest',
          mediaId: 'qi-extension-media-1'
        }
      ])
      expect(
        patched,
        'Quick-ingest runtime patching must succeed for deterministic success handoff audit.'
      ).toBe(true)

      const modal = await openQuickIngestDialog(page)

      await addUrlToQuickIngest(modal, 'https://example.com')
      await startQueuedQuickIngest(modal)

      await expect(modal.getByTestId('wizard-results-step')).toBeVisible({
        timeout: 30_000
      })
      await expect(
        modal.getByRole('region', { name: /completed items/i })
      ).toBeVisible()

      await expect(
        modal.getByRole('button', { name: /Open .* in Media/i }).first()
      ).toBeVisible()
    } finally {
      await restoreQuickIngestRuntimePatch(page)
      await context.close()
    }
  })

  test('configure step exposes presets and keeps advanced options collapsed by default', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: 'http://127.0.0.1:8000',
        authMode: 'single-user',
        apiKey: API_KEY
      }
    })

    try {
      await page.goto(optionsUrl + '#/media', { waitUntil: 'domcontentloaded' })

      const modal = await openQuickIngestDialog(page)
      await addUrlToQuickIngest(modal, 'https://example.com/configure')
      await modal.getByRole('button', { name: /configure \d+ items/i }).click()

      await expect(
        modal.getByText(/Presets are starting points/i)
      ).toBeVisible()
      await expect(
        modal.getByRole('button', { name: /Advanced options/i })
      ).toHaveAttribute('aria-expanded', 'false')

      await modal.getByRole('button', { name: /Advanced options/i }).click()
      await expect(
        modal.getByRole('button', { name: /Hide advanced options/i })
      ).toHaveAttribute('aria-expanded', 'true')
      await expect(modal.getByText(/Audio options|quickIngest\.audioOptions/i)).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test('mixed results surface completed and error sections without legacy filters', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: 'http://127.0.0.1:8000',
        authMode: 'single-user',
        apiKey: API_KEY
      }
    })

    try {
      await page.goto(optionsUrl + '#/media', { waitUntil: 'domcontentloaded' })

      const patched = await patchQuickIngestRuntimeResults(page, [
        {
          id: 'fail-1',
          status: 'error',
          type: 'html',
          url: 'https://fail.example.com',
          error: 'Simulated failure'
        },
        {
          id: 'ok-1',
          status: 'ok',
          type: 'html',
          url: 'https://ok.example.com',
          title: 'Successful quick ingest item',
          mediaId: 'qi-extension-media-ok'
        }
      ])

      expect(
        patched,
        'Quick-ingest message patching must succeed for deterministic mixed-results UX audit.'
      ).toBe(true)

      const modal = await openQuickIngestDialog(page)
      await addUrlToQuickIngest(modal, 'https://example.com/mixed')
      await startQueuedQuickIngest(modal)

      await expect(modal.getByTestId('wizard-results-step')).toBeVisible({
        timeout: 30_000
      })
      await expect(
        modal.getByRole('region', { name: /completed items/i })
      ).toBeVisible()
      await expect(
        modal.getByRole('region', { name: /error items/i })
      ).toBeVisible()
      await expect(modal.getByText(/https:\/\/fail\.example\.com/i)).toBeVisible()
      await expect(modal.getByText(/Successful quick ingest item/i)).toBeVisible()
    } finally {
      await restoreQuickIngestRuntimePatch(page)
      await context.close()
    }
  })

  test('offline states share a clear headline and footer', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    try {
      await page.goto(optionsUrl + '#/media', { waitUntil: 'domcontentloaded' })
      await waitForConnectionStore(page, 'quick-ingest-ux-offline')

      const openQuickIngest = async () => {
        const modal = await openQuickIngestDialog(page)
        return { modal }
      }

      const assertBannerAndFooter = async () => {
        const { modal } = await openQuickIngest()

        // Headline should indicate that processing cannot start yet.
        await expect(
          modal.getByText(/Server offline/i)
        ).toBeVisible()

        // Banner body should explain how to recover without blocking queue review.
        await expect(
          modal.getByText(/Configure your tldw server|Cannot reach your tldw server|Reconnect to your tldw server/i)
        ).toBeVisible()

        await expect(
          modal.getByRole('button', { name: /retry connection/i })
        ).toBeVisible()

        // Close between states so each run starts clean.
        await modal.getByRole('button', { name: /close/i }).click()
        await expect(modal).toBeHidden()
      }

      // 1) Unconfigured state
      await forceUnconfigured(page, 'qi-ux-unconfigured')
      await assertBannerAndFooter()

      // 2) Offline unreachable state
      await forceErrorUnreachable(
        page,
        { errorKind: 'unreachable', serverUrl: 'http://127.0.0.1:8000' },
        'qi-ux-offline'
      )
      await assertBannerAndFooter()

    } finally {
      await context.close()
    }
  })
})
