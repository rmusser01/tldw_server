import { test, expect } from '@playwright/test'
import { launchWithBuiltExtension } from './utils/extension-build'
import {
  waitForConnectionStore,
  forceConnected
} from './utils/connection'

interface MessageOptionStoreState {
  chatMode?: string
}

interface MessageOptionStore {
  getState?: () => MessageOptionStoreState
}

interface MessageOptionWindow {
  __tldw_useStoreMessageOption?: MessageOptionStore
}

async function readChatMode(page: import('@playwright/test').Page): Promise<string | null> {
  return page.evaluate(() => {
    const w = window as unknown as MessageOptionWindow
    return w.__tldw_useStoreMessageOption?.getState?.().chatMode ?? null
  })
}

async function dismissWelcomeOverlay(page: import('@playwright/test').Page) {
  const welcomeHeading = page.getByText(/Welcome to tldw(?![\s_]*(browser|assistant))/i).first()
  const isTimeoutError = (err: unknown): err is Error =>
    err instanceof Error && err.name === 'TimeoutError'

  let appeared = true
  try {
    await welcomeHeading.waitFor({ state: 'visible', timeout: 3_000 })
  } catch (err) {
    if (isTimeoutError(err)) {
      appeared = false
    } else {
      throw new Error('Failed waiting for welcome overlay to appear', { cause: err })
    }
  }
  if (!appeared) return

  const dialog = page.locator('[role="dialog"]').filter({ has: welcomeHeading }).first()
  const closeButton = dialog.getByRole('button', { name: /close/i }).first()

  let closeVisible = false
  try {
    closeVisible = await closeButton.isVisible()
  } catch (err) {
    if (!isTimeoutError(err)) {
      throw new Error('Failed checking welcome overlay close button visibility', { cause: err })
    }
  }

  if (closeVisible) {
    try {
      await closeButton.click()
    } catch (err) {
      if (!isTimeoutError(err)) {
        throw new Error('Failed clicking welcome overlay close button', { cause: err })
      }
    }
  } else {
    try {
      await page.keyboard.press('Escape')
    } catch (err) {
      if (!isTimeoutError(err)) {
        throw new Error('Failed pressing Escape to dismiss welcome overlay', { cause: err })
      }
    }
  }

  try {
    await welcomeHeading.waitFor({ state: 'hidden', timeout: 8_000 })
  } catch (err) {
    if (!isTimeoutError(err)) {
      throw new Error('Failed waiting for welcome overlay to hide', { cause: err })
    }
  }
}

test.describe('Knowledge RAG workspace UX', () => {
  test('workflow hub "Do Research" opens Knowledge QA route', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    try {
      await waitForConnectionStore(page, 'knowledge-rag-workflow-entry')
      await forceConnected(
        page,
        { serverUrl: 'http://dummy-tldw' },
        'knowledge-rag-workflow-entry'
      )

      await page.goto(optionsUrl + '#/settings/manageKnowledge')
      await page.waitForLoadState('networkidle')

      const workflowDialog = page
        .locator('[role="dialog"]')
        .filter({ has: page.getByText(/What would you like to do/i).first() })
        .first()
      let dialogAlreadyOpen = false
      try {
        dialogAlreadyOpen = await workflowDialog.isVisible()
      } catch (error) {
        if (!(error instanceof Error && error.name === 'TimeoutError')) {
          throw new Error('Failed checking workflow dialog visibility', { cause: error })
        }
      }
      if (!dialogAlreadyOpen) {
        const workflowButton = page.getByTestId('workflow-button').first()
        await expect(workflowButton).toBeVisible()
        await workflowButton.click()
      }

      await expect(workflowDialog).toBeVisible()
      await workflowDialog.getByRole('button', { name: /Do Research/i }).click()

      await expect
        .poll(() => page.url(), { timeout: 15_000 })
        .toContain('#/knowledge')
      await expect(page.getByText(/Knowledge QA|Knowledge search & chat/i)).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test('shows RAG workspace and (when available) allows toggling per-reply RAG', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    try {
      // Pretend the server is connected so the Knowledge workspace renders
      await waitForConnectionStore(page, 'knowledge-rag-connected')
      await forceConnected(
        page,
        { serverUrl: 'http://dummy-tldw' },
        'knowledge-rag-connected'
      )

      await page.goto(optionsUrl + '#/settings/knowledge')
      await page.waitForLoadState('networkidle')
      await dismissWelcomeOverlay(page)

      // Knowledge workspace header should be present with de-jargoned title
      await expect(
        page.getByText(/Knowledge search & chat/i)
      ).toBeVisible()
      await expect(
        page.getByText(/Retrieval-augmented generation \(RAG\) lets the assistant ground answers/i)
      ).toBeVisible()

      // If the bundled OpenAPI spec does not advertise RAG endpoints,
      // the workspace shows a capability callout instead of full controls.
      const ragUnsupportedCallout = page.getByText(
        /RAG search is not available on this server/i
      )
      let calloutVisible = false
      try {
        calloutVisible = await ragUnsupportedCallout.isVisible()
      } catch (error) {
        if (!(error instanceof Error && error.name === 'TimeoutError')) {
          throw new Error('Failed checking RAG unsupported callout visibility', { cause: error })
        }
      }

      if (!calloutVisible) {
        // Auto-RAG toggle should be visible and wired to chatMode
        const autoRagSwitch = page.getByRole('switch', {
          name: /Use RAG for every reply/i
        })
        await expect(autoRagSwitch).toBeVisible()

        const initialMode = await readChatMode(page)
        expect(initialMode).toBe('normal')

        await autoRagSwitch.click()

        const ragMode = await readChatMode(page)
        expect(ragMode).toBe('rag')

        await autoRagSwitch.click()

        const backMode = await readChatMode(page)
        expect(backMode).toBe('normal')
      } else {
        // When RAG is unsupported, we at least show a Diagnostics CTA
        await expect(
          page.getByRole('button', { name: /Health & diagnostics/i })
        ).toBeVisible()
      }
    } finally {
      await context.close()
    }
  })

  test('fullscreen playground Search & Context exposes key-level RAG options', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    try {
      await waitForConnectionStore(page, 'knowledge-rag-playground-context')
      await forceConnected(
        page,
        { serverUrl: 'http://dummy-tldw' },
        'knowledge-rag-playground-context'
      )

      await page.goto(optionsUrl + '#/')
      await page.waitForLoadState('networkidle')
      await dismissWelcomeOverlay(page)

      const contextButton = page
        .locator(
          '[data-playground-knowledge-trigger="true"], button[aria-label*="Search & Context"], button[title*="Search & Context"]'
        )
        .first()
      let triggerAlreadyVisible = false
      try {
        triggerAlreadyVisible = await contextButton.isVisible()
      } catch (error) {
        if (!(error instanceof Error && error.name === 'TimeoutError')) {
          throw new Error('Failed checking Search & Context trigger visibility', { cause: error })
        }
      }
      if (!triggerAlreadyVisible) {
        const startChatCard = page.getByText(/Start Chatting/i).first()
        let startChatVisible = false
        try {
          startChatVisible = await startChatCard.isVisible()
        } catch (error) {
          if (!(error instanceof Error && error.name === 'TimeoutError')) {
            throw new Error('Failed checking Start Chatting card visibility', { cause: error })
          }
        }
        if (startChatVisible) {
          await startChatCard.click()
        }
      }

      await expect(contextButton).toBeVisible({ timeout: 15_000 })
      await contextButton.click()

      await expect(page.getByRole('heading', { name: /Search & Context/i })).toBeVisible()

      const settingsTab = page.getByRole('tab', { name: /Settings/i }).first()
      await settingsTab.click()

      const settingsSearch = page.getByLabel(/Search settings/i).first()
      await settingsSearch.fill('all options')

      const allOptionsSectionToggle = page
        .getByRole('button', { name: /All options/i })
        .first()
      await expect(allOptionsSectionToggle).toBeVisible()
      await allOptionsSectionToggle.click()

      const allOptionsFilter = page.getByTestId('knowledge-all-options-filter')
      await expect(allOptionsFilter).toBeVisible()
      await allOptionsFilter.fill('agentic_time_budget_sec')
      await expect(page.getByText('agentic_time_budget_sec')).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
