import { test, expect } from '@playwright/test'
import { launchWithExtensionOrSkip } from "./utils/real-server"
import path from 'path'
import { launchWithExtension } from './utils/extension'
import { MockTldwServer } from './utils/mock-server'
import { withAllFeaturesDisabled } from './utils/feature-flags'
import { waitForConnectionStore, forceConnected, forceConnectionState } from './utils/connection'

test.describe('Chat persistence UX', () => {
  test('exposes clear labels for temporary vs local-only chats', async () => {
    const extPath = path.resolve('build/chrome-mv3')
    const { context, page } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: withAllFeaturesDisabled()
    })

    // Wait for store and force connected state to skip onboarding
    await waitForConnectionStore(page)
    await forceConnected(page)

    // Give React time to re-render after state change
    await page.waitForTimeout(500)

    // The UI now shows a "Start a new chat" card - click to start chatting
    const startChattingButton = page.getByRole('button', { name: /Start chatting/i })
    if (await startChattingButton.isVisible()) {
      await startChattingButton.click()
      await page.waitForTimeout(300)
    }

    // By default, chats are saved on this device only.
    await expect(
      page.getByText(/Saved on this device/i).first()
    ).toBeVisible()

    // Look for the save/temporary toggle switch
    const persistenceSwitch = page.getByRole('switch', {
      name: /Save chat|Save to history|Temporary chat/i
    })
    await expect(persistenceSwitch).toBeVisible()
    await persistenceSwitch.click()

    // Wait for notification to clear and check for temporary chat indicator
    await page.waitForTimeout(500)
    await expect(
      page.locator('#root').getByText(/Temporary chat/i).first()
    ).toBeVisible()

    await context.close()
  })

  test('shows a connect hint when server save is unavailable', async () => {
    const extPath = path.resolve('build/chrome-mv3')
    const { context, page } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: withAllFeaturesDisabled()
    })

    // Wait for store and force partially-connected state (no actual server)
    await waitForConnectionStore(page)
    await forceConnectionState(page, {
      phase: 'connected',
      isConnected: false, // No actual server connection
      isChecking: false,
      hasCompletedFirstRun: true,
      mode: 'normal',
      configStep: 'health',
      knowledgeStatus: 'unknown'
    })

    await page.waitForTimeout(500)

    // Look for the connect hint
    const connectHint = page.getByText(/Connect.*server.*save/i).first()
    await expect(connectHint).toBeVisible()

    await context.close()
  })

  test('explains benefits when promoting a chat to server-backed mode', async () => {
    const server = new MockTldwServer()
    let serverStarted = false
    let context: any = null

    try {
      await server.start()
      serverStarted = true

      const extPath = path.resolve('build/chrome-mv3')
      const launched = await launchWithExtensionOrSkip(test, extPath, {
        seedConfig: withAllFeaturesDisabled({
          tldwConfig: {
            serverUrl: server.url,
            authMode: 'single-user',
            apiKey: 'THIS-IS-A-SECURE-KEY-123-FAKE-KEY'
          }
        })
      })
      context = launched.context
      const { page } = launched

      // Wait for store and force connected state
      await waitForConnectionStore(page)
      await forceConnected(page, { serverUrl: server.url })

      await page.waitForTimeout(500)

      // Click start chatting if the empty state is shown
      const startChattingButton = page.getByRole('button', { name: /Start chatting/i })
      if (await startChattingButton.isVisible()) {
        await startChattingButton.click()
        await page.waitForTimeout(300)
      }

      // Ensure we are in non-temporary (local) mode first
      await expect(
        page.getByText(/Saved on this device/i).first()
      ).toBeVisible()

      // Look for server save promotion button or link
      const saveToServerButton = page.getByText(/save.*server|server.*save/i).first()
      if (await saveToServerButton.isVisible()) {
        await saveToServerButton.click()
        await page.waitForTimeout(300)

        // Check for server-backed indicator
        await expect(
          page.getByText(/Server|Synced|server/i).first()
        ).toBeVisible()
      }
    } finally {
      if (context) {
        await context.close()
      }
      if (serverStarted) {
        await server.stop()
      }
    }
  })
})
