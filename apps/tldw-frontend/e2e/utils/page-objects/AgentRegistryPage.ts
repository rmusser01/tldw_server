/**
 * Page Object for Agent Registry workflow
 *
 * The Agent Registry page displays:
 * - ACP System Health card (runner binary, agent status, API keys)
 * - Registered Agents card with agent cards showing name, status, launch button
 * - Refresh button to reload health and agent data
 *
 * Route: /agents
 * Component: packages/ui/src/components/Option/AgentRegistry/index.tsx
 */
import { type Page, type Locator } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export interface AgentRegistryHealthState {
  loaded: boolean
  unavailable: boolean
}

export class AgentRegistryPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(): Promise<void> {
    await this.page.goto("/agents", { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    // Wait for the health card or agent list card to appear
    const healthCard = this.page.getByText("ACP System Health")
    const agentCard = this.page.getByText("Registered Agents")
    const healthWarning = this.page.getByText("Health check unavailable")
    await Promise.race([
      healthCard.first().waitFor({ state: "visible", timeout: 20_000 }),
      agentCard.first().waitFor({ state: "visible", timeout: 20_000 }),
      healthWarning.first().waitFor({ state: "visible", timeout: 20_000 }),
    ]).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  /** ACP System Health card title */
  get healthCardTitle(): Locator {
    return this.page.getByText("ACP System Health")
  }

  /** Registered Agents card title */
  get agentsCardTitle(): Locator {
    return this.page.getByText("Registered Agents")
  }

  /** Refresh button in the health card */
  get refreshButton(): Locator {
    return this.page.getByRole("button", { name: /refresh/i })
  }

  /** Runner Binary status label */
  get runnerBinaryLabel(): Locator {
    return this.page.getByText("Runner Binary")
  }

  /** Agent Status label */
  get agentStatusLabel(): Locator {
    return this.page.getByText("Agent Status")
  }

  /** API Keys status label */
  get apiKeysLabel(): Locator {
    return this.page.getByText("API Keys")
  }

  /** Health check unavailable warning */
  get healthUnavailableWarning(): Locator {
    return this.page.getByText("Health check unavailable")
  }

  /** No agents registered empty state */
  get noAgentsMessage(): Locator {
    return this.page.getByText("No agents registered")
  }

  /** Agent cards in the grid */
  get agentCards(): Locator {
    return this.page.locator(".rounded-lg.border.border-border.p-4")
  }

  /** Launch buttons on agent cards */
  get launchButtons(): Locator {
    return this.page.getByRole("button", { name: /launch/i })
  }

  // -- Helpers ----------------------------------------------------------------

  /** Wait for the health card to finish its loading state. */
  async waitForHealthSettled(timeout = 20_000): Promise<void> {
    await Promise.race([
      Promise.all([
        this.runnerBinaryLabel.waitFor({ state: "visible", timeout }),
        this.agentStatusLabel.waitFor({ state: "visible", timeout }),
        this.apiKeysLabel.waitFor({ state: "visible", timeout }),
      ]),
      this.healthUnavailableWarning.waitFor({ state: "visible", timeout }),
    ])
  }

  /** Wait once for health to settle, then read the resulting state. */
  async getHealthState(timeout = 20_000): Promise<AgentRegistryHealthState> {
    await this.waitForHealthSettled(timeout)
    const [runnerVisible, agentVisible, apiKeysVisible, unavailable] =
      await Promise.all([
        this.runnerBinaryLabel.isVisible().catch(() => false),
        this.agentStatusLabel.isVisible().catch(() => false),
        this.apiKeysLabel.isVisible().catch(() => false),
        this.healthUnavailableWarning.isVisible().catch(() => false),
      ])

    return {
      loaded: runnerVisible && agentVisible && apiKeysVisible,
      unavailable,
    }
  }

  /** Check if health data loaded successfully (status indicators are visible) */
  async isHealthDataLoaded(): Promise<boolean> {
    return (await this.getHealthState()).loaded
  }

  /** Wait for the agent list to finish its loading state. */
  async waitForAgentListSettled(timeout = 20_000): Promise<void> {
    await Promise.race([
      this.agentCards.first().waitFor({ state: "visible", timeout }),
      this.launchButtons.first().waitFor({ state: "visible", timeout }),
      this.noAgentsMessage.waitFor({ state: "visible", timeout }),
    ])
  }

  /** Get the count of visible agent cards */
  async getAgentCount(): Promise<number> {
    await this.waitForAgentListSettled()
    const cardCount = await this.agentCards.count()
    if (cardCount > 0) return cardCount
    return this.launchButtons.count()
  }

  /** Check if the page is showing health warning (server unreachable) */
  async isHealthUnavailable(): Promise<boolean> {
    return (await this.getHealthState()).unavailable
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Refresh button",
        locator: this.refreshButton,
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/acp\//,
          method: "GET",
        },
      },
    ]
  }
}
