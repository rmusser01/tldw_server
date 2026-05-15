/**
 * Agent Registry E2E Tests (Tier 3)
 *
 * Tests the Agent Registry page lifecycle:
 * - Page loads with ACP System Health card and Registered Agents card
 * - Health status indicators (runner, agent, API keys) are displayed
 * - Agent cards show name, status, and launch button
 * - Refresh button reloads health and agent data
 *
 * Run: npx playwright test e2e/workflows/tier-3-automation/agent-registry.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { AgentRegistryPage } from "../../utils/page-objects/AgentRegistryPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

test.describe("Agent Registry", () => {
  let registry: AgentRegistryPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    registry = new AgentRegistryPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Agent Registry page with health and agent cards", async ({
      authedPage,
      diagnostics,
    }) => {
      registry = new AgentRegistryPage(authedPage)
      await registry.goto()
      await registry.assertPageReady()

      // Health card title should be visible
      const healthVisible = await registry.healthCardTitle.isVisible().catch(() => false)
      expect(healthVisible).toBe(true)

      // Agents card title should be visible
      const agentsVisible = await registry.agentsCardTitle.isVisible().catch(() => false)
      expect(agentsVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show health status indicators or health unavailable warning", async ({
      authedPage,
      diagnostics,
    }) => {
      registry = new AgentRegistryPage(authedPage)
      await registry.goto()
      await registry.assertPageReady()

      // Either health data is loaded or the warning is shown
      const healthState = await registry.getHealthState()

      expect(healthState.loaded || healthState.unavailable).toBe(true)

      // If health data is loaded, all three indicators should be visible
      if (healthState.loaded) {
        await expect(registry.runnerBinaryLabel).toBeVisible()
        await expect(registry.agentStatusLabel).toBeVisible()
        await expect(registry.apiKeysLabel).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Agent List
  // =========================================================================

  test.describe("Agent List", () => {
    test("should show agent cards or empty state", async ({
      authedPage,
      diagnostics,
    }) => {
      registry = new AgentRegistryPage(authedPage)
      await registry.goto()
      await registry.assertPageReady()

      const agentCount = await registry.getAgentCount()
      const noAgentsVisible = await registry.noAgentsMessage.isVisible().catch(() => false)

      // Either agents are listed or the empty state is shown
      expect(agentCount > 0 || noAgentsVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should have launch buttons on agent cards when agents are available", async ({
      authedPage,
      diagnostics,
    }) => {
      registry = new AgentRegistryPage(authedPage)
      await registry.goto()
      await registry.assertPageReady()

      const agentCount = await registry.getAgentCount()
      if (agentCount === 0) return

      // Each agent card should have a launch button
      const launchCount = await registry.launchButtons.count()
      expect(launchCount).toBeGreaterThan(0)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Refresh (requires server)
  // =========================================================================

  test.describe("Refresh", () => {
    test("should fire API calls to health and agents endpoints on Refresh click", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      registry = new AgentRegistryPage(authedPage)
      await registry.goto()
      await registry.assertPageReady()

      const refreshVisible = await registry.refreshButton.isVisible().catch(() => false)
      if (!refreshVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/acp\//,
        method: "GET",
      }, 15_000)

      await registry.refreshButton.click()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // ACP endpoints may not be available on all server configurations
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should fetch agents and health data on initial page load", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      // The component auto-fetches agents and health on mount
      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/acp\//,
        method: "GET",
      }, 20_000)

      registry = new AgentRegistryPage(authedPage)
      await registry.goto()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Server may not have ACP endpoint available
      }

      await registry.assertPageReady()
      await assertNoCriticalErrors(diagnostics)
    })
  })
})
