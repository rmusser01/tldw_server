// @vitest-environment jsdom

import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  useCanonicalConnectionConfig: vi.fn(),
  getTldwConfig: vi.fn(),
  listPersonalIntegrations: vi.fn(),
  listWorkspaceIntegrations: vi.fn(),
  connectPersonalIntegration: vi.fn(),
  updatePersonalIntegration: vi.fn(),
  deletePersonalIntegration: vi.fn(),
  getWorkspaceSlackPolicy: vi.fn(),
  getWorkspaceDiscordPolicy: vi.fn(),
  getWorkspaceTelegramBot: vi.fn(),
  listWorkspaceTelegramLinkedActors: vi.fn(),
  updateWorkspaceSlackPolicy: vi.fn(),
  updateWorkspaceDiscordPolicy: vi.fn(),
  updateWorkspaceTelegramBot: vi.fn(),
  createWorkspaceTelegramPairingCode: vi.fn(),
  revokeWorkspaceTelegramLinkedActor: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) => mocks.useCanonicalConnectionConfig(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => mocks.getTldwConfig(...args)
  }
}))

vi.mock("@/services/integrations-control-plane", () => ({
  listPersonalIntegrations: (...args: unknown[]) => mocks.listPersonalIntegrations(...args),
  listWorkspaceIntegrations: (...args: unknown[]) => mocks.listWorkspaceIntegrations(...args),
  connectPersonalIntegration: (...args: unknown[]) => mocks.connectPersonalIntegration(...args),
  updatePersonalIntegration: (...args: unknown[]) => mocks.updatePersonalIntegration(...args),
  deletePersonalIntegration: (...args: unknown[]) => mocks.deletePersonalIntegration(...args),
  getWorkspaceSlackPolicy: (...args: unknown[]) => mocks.getWorkspaceSlackPolicy(...args),
  getWorkspaceDiscordPolicy: (...args: unknown[]) => mocks.getWorkspaceDiscordPolicy(...args),
  getWorkspaceTelegramBot: (...args: unknown[]) => mocks.getWorkspaceTelegramBot(...args),
  listWorkspaceTelegramLinkedActors: (...args: unknown[]) => mocks.listWorkspaceTelegramLinkedActors(...args),
  updateWorkspaceSlackPolicy: (...args: unknown[]) => mocks.updateWorkspaceSlackPolicy(...args),
  updateWorkspaceDiscordPolicy: (...args: unknown[]) => mocks.updateWorkspaceDiscordPolicy(...args),
  updateWorkspaceTelegramBot: (...args: unknown[]) => mocks.updateWorkspaceTelegramBot(...args),
  createWorkspaceTelegramPairingCode: (...args: unknown[]) => mocks.createWorkspaceTelegramPairingCode(...args),
  revokeWorkspaceTelegramLinkedActor: (...args: unknown[]) => mocks.revokeWorkspaceTelegramLinkedActor(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) => {
      if (typeof fallback === "string") {
        return fallback
      }
      return fallback?.defaultValue ?? _key
    }
  })
}))

import {
  IntegrationManagementPage,
  buildIntegrationQueryKey
} from "../IntegrationManagementPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/integrations"]}>{ui}</MemoryRouter>
    </QueryClientProvider>
  )
}

const mockWorkspaceQueries = (overrides?: {
  overviewItems?: Array<Record<string, unknown>>
  slackPolicy?: Record<string, unknown>
  discordPolicy?: Record<string, unknown>
}) => {
  mocks.listWorkspaceIntegrations.mockResolvedValue({
    scope: "workspace",
    items: overrides?.overviewItems ?? [
      {
        id: "workspace:slack",
        provider: "slack",
        scope: "workspace",
        display_name: "Slack workspace",
        status: "connected",
        enabled: true,
        metadata: {},
        actions: ["disable", "remove"]
      },
      {
        id: "workspace:discord",
        provider: "discord",
        scope: "workspace",
        display_name: "Discord workspace",
        status: "connected",
        enabled: true,
        metadata: {},
        actions: ["disable", "remove"]
      },
      {
        id: "workspace:telegram",
        provider: "telegram",
        scope: "workspace",
        display_name: "Telegram bot",
        status: "needs_config",
        enabled: false,
        metadata: {},
        actions: ["enable"]
      }
    ]
  })
  mocks.getWorkspaceSlackPolicy.mockResolvedValue({
    provider: "slack",
    scope: "workspace",
    installation_ids: ["T-1"],
    uniform: true,
    policy: {
      allowed_commands: ["help"],
      channel_allowlist: [],
      channel_denylist: [],
      default_response_mode: "thread",
      strict_user_mapping: false,
      service_user_id: null,
      user_mappings: {},
      workspace_quota_per_minute: 10,
      user_quota_per_minute: 5,
      status_scope: "workspace",
      ...overrides?.slackPolicy
    }
  })
  mocks.getWorkspaceDiscordPolicy.mockResolvedValue({
    provider: "discord",
    scope: "workspace",
    installation_ids: ["G-1"],
    uniform: true,
    policy: {
      allowed_commands: ["help"],
      channel_allowlist: [],
      channel_denylist: [],
      default_response_mode: "channel",
      strict_user_mapping: false,
      service_user_id: null,
      user_mappings: {},
      guild_quota_per_minute: 10,
      user_quota_per_minute: 5,
      status_scope: "guild",
      ...overrides?.discordPolicy
    }
  })
  mocks.getWorkspaceTelegramBot.mockResolvedValue({
    ok: true,
    provider: "telegram",
    scope_type: "org",
    scope_id: 1,
    bot_username: "examplebot",
    enabled: true
  })
  mocks.listWorkspaceTelegramLinkedActors.mockResolvedValue({
    ok: true,
    scope_type: "org",
    scope_id: 1,
    items: []
  })
  mocks.updateWorkspaceSlackPolicy.mockResolvedValue({
    provider: "slack",
    scope: "workspace",
    installation_ids: ["T-1"],
    uniform: true,
    policy: {}
  })
  mocks.updateWorkspaceDiscordPolicy.mockResolvedValue({
    provider: "discord",
    scope: "workspace",
    installation_ids: ["G-1"],
    uniform: true,
    policy: {}
  })
}

describe("IntegrationManagementPage", () => {
  beforeEach(() => {
    for (const mock of Object.values(mocks)) {
      mock.mockReset()
    }
    fetchMock.mockReset()
    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key",
        orgId: 101
      },
      loading: false
    })
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/integrations/personal": {}
        }
      })
    })
  })

  it("renders personal slack and discord cards and hides telegram", async () => {
    mocks.listPersonalIntegrations.mockResolvedValue({
      scope: "personal",
      items: [
        {
          id: "personal:slack",
          provider: "slack",
          scope: "personal",
          display_name: "Slack",
          status: "connected",
          enabled: true,
          metadata: {},
          actions: ["disable", "remove"]
        },
        {
          id: "personal:discord",
          provider: "discord",
          scope: "personal",
          display_name: "Discord",
          status: "disconnected",
          enabled: false,
          metadata: {},
          actions: ["connect"]
        },
        {
          id: "workspace:telegram",
          provider: "telegram",
          scope: "workspace",
          display_name: "Telegram",
          status: "connected",
          enabled: true,
          metadata: {},
          actions: ["disable"]
        }
      ]
    })

    renderWithQueryClient(<IntegrationManagementPage scope="personal" />)

    expect(await screen.findByText("Slack")).toBeInTheDocument()
    expect(screen.getByText("Discord")).toBeInTheDocument()
    expect(screen.queryByText("Telegram")).not.toBeInTheDocument()
  })

  it("starts the personal connect flow from the management drawer", async () => {
    const user = userEvent.setup()
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null)

    mocks.listPersonalIntegrations.mockResolvedValue({
      scope: "personal",
      items: [
        {
          id: "personal:slack",
          provider: "slack",
          scope: "personal",
          display_name: "Slack",
          status: "disconnected",
          enabled: false,
          metadata: {},
          actions: ["connect"]
        }
      ]
    })
    mocks.connectPersonalIntegration.mockResolvedValue({
      provider: "slack",
      connection_id: "personal:slack",
      status: "ready",
      auth_url: "https://slack.example.test/oauth",
      auth_session_id: "session-123",
      expires_at: "2026-03-20T22:00:00+00:00"
    })

    renderWithQueryClient(<IntegrationManagementPage scope="personal" />)

    await user.click(await screen.findByRole("button", { name: "Manage" }))
    await user.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(mocks.connectPersonalIntegration).toHaveBeenCalledWith("slack")
    })
    expect(openSpy).toHaveBeenCalledWith("https://slack.example.test/oauth", "_blank", "noopener,noreferrer")

    openSpy.mockRestore()
  })

  it("updates and removes a personal integration from the management drawer", async () => {
    const user = userEvent.setup()

    mocks.listPersonalIntegrations.mockResolvedValue({
      scope: "personal",
      items: [
        {
          id: "personal:slack",
          provider: "slack",
          scope: "personal",
          display_name: "Slack",
          status: "connected",
          enabled: true,
          metadata: {},
          actions: ["disable", "remove"]
        }
      ]
    })
    mocks.updatePersonalIntegration.mockResolvedValue({
      id: "personal:slack",
      provider: "slack",
      scope: "personal",
      display_name: "Slack",
      status: "disabled",
      enabled: false,
      metadata: {},
      actions: ["enable", "remove"]
    })
    mocks.deletePersonalIntegration.mockResolvedValue({
      deleted: true,
      provider: "slack",
      connection_id: "personal:slack"
    })

    renderWithQueryClient(<IntegrationManagementPage scope="personal" />)

    await user.click(await screen.findByRole("button", { name: "Manage" }))
    await user.click(await screen.findByRole("button", { name: "Disable" }))

    await waitFor(() => {
      expect(mocks.updatePersonalIntegration).toHaveBeenCalledWith("slack", "personal:slack", { enabled: false })
    })

    await user.click(await screen.findByRole("button", { name: "Remove" }))

    await waitFor(() => {
      expect(mocks.deletePersonalIntegration).toHaveBeenCalledWith("slack", "personal:slack")
    })
  })

  it("shows an unsupported-state message when personal integrations are unavailable on the server", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })

    renderWithQueryClient(<IntegrationManagementPage scope="personal" />)

    expect(await screen.findByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByText("Personal integrations are unavailable")).toBeInTheDocument()
    expect(
      screen.getByText("This server does not expose the personal integrations capability.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Check server setup" })).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent("/api/v1/integrations/personal")
    expect(mocks.listPersonalIntegrations).not.toHaveBeenCalled()
  })

  it("keeps integration overview endpoint failures in diagnostics", async () => {
    mockWorkspaceQueries()
    mocks.listWorkspaceIntegrations.mockRejectedValue(
      Object.assign(new Error("Internal Server Error (GET /api/v1/integrations/workspace)"), {
        status: 500
      })
    )

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    const heading = await screen.findByRole(
      "heading",
      {
        name: "Workspace integrations could not load"
      },
      { timeout: 4000 }
    )
    const primaryState = heading.closest("div")
    const diagnostics = screen.getByLabelText("Diagnostics")

    expect(primaryState).not.toHaveTextContent("/api/v1/integrations/workspace")
    expect(diagnostics).toHaveTextContent("/api/v1/integrations/workspace")
    expect(diagnostics).toHaveTextContent("500")
    expect(diagnostics).toHaveTextContent(
      "Internal Server Error (GET /api/v1/integrations/workspace)"
    )
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument()
  })

  it("retries transient integration overview failures before showing state", async () => {
    mockWorkspaceQueries()
    mocks.listWorkspaceIntegrations
      .mockRejectedValueOnce(
        Object.assign(new Error("Temporary gateway error"), { status: 502 })
      )
      .mockRejectedValueOnce(
        Object.assign(new Error("Temporary gateway error"), { status: 502 })
      )
      .mockResolvedValueOnce({
        scope: "workspace",
        items: [
          {
            id: "workspace:slack",
            provider: "slack",
            scope: "workspace",
            display_name: "Slack workspace",
            status: "connected",
            enabled: true,
            metadata: {},
            actions: ["disable", "remove"]
          }
        ]
      })

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    expect(await screen.findByText("Slack workspace", {}, { timeout: 4000 })).toBeInTheDocument()
    await waitFor(() => {
      expect(mocks.listWorkspaceIntegrations).toHaveBeenCalledTimes(3)
    })
  })

  it("keys workspace-scoped queries by the active org id", () => {
    expect(buildIntegrationQueryKey("workspace", 101, "overview")).toEqual([
      "integrations",
      "workspace",
      101,
      "overview"
    ])
    expect(buildIntegrationQueryKey("workspace", 202, "slack-policy")).toEqual([
      "integrations",
      "workspace",
      202,
      "slack-policy"
    ])
    expect(buildIntegrationQueryKey("personal", 101, "overview")).toEqual([
      "integrations",
      "personal",
      "overview"
    ])
  })

  it("refreshes workspace-scoped queries when the active org config updates", async () => {
    mockWorkspaceQueries()
    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key",
        orgId: 101
      },
      loading: false
    })
    mocks.getTldwConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key",
      orgId: 202
    })

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    expect(await screen.findByText("Slack")).toBeInTheDocument()
    expect(mocks.listWorkspaceIntegrations).toHaveBeenCalledTimes(1)

    window.dispatchEvent(new Event("tldw:config-updated"))

    await waitFor(() => {
      expect(mocks.getTldwConfig).toHaveBeenCalled()
      expect(mocks.listWorkspaceIntegrations).toHaveBeenCalledTimes(2)
    })
  })

  it("renders workspace slack, discord, and telegram controls", async () => {
    mockWorkspaceQueries()

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    expect(await screen.findByText("Slack")).toBeInTheDocument()
    expect(screen.getByText("Discord")).toBeInTheDocument()
    expect(screen.getByText("Telegram")).toBeInTheDocument()
    expect(screen.getByText("Slack policy")).toBeInTheDocument()
    expect(screen.getByText("Telegram bot")).toBeInTheDocument()
  })

  it("shows Telegram linked-actor load failures as a degraded shared state", async () => {
    mockWorkspaceQueries()
    mocks.listWorkspaceTelegramLinkedActors.mockRejectedValue(
      Object.assign(
        new Error(
          "Telegram actors unavailable (GET /api/v1/integrations/workspace/telegram/linked-actors)"
        ),
        { status: 503 }
      )
    )

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    expect(await screen.findByText("Degraded")).toBeInTheDocument()
    expect(screen.getByText("Workspace integrations are partially available")).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "/api/v1/integrations/workspace/telegram/linked-actors"
    )
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent("503")
  })

  it("preserves hidden Slack policy fields when saving visible settings", async () => {
    const user = userEvent.setup()

    mockWorkspaceQueries({
      slackPolicy: {
        allowed_commands: ["help", "status"],
        channel_allowlist: ["C-1"],
        channel_denylist: ["C-2"],
        strict_user_mapping: true,
        service_user_id: "U-service",
        user_mappings: { U123: "alice" },
        workspace_quota_per_minute: 12,
        user_quota_per_minute: 6,
        status_scope: "workspace_and_user"
      }
    })

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    await user.click(await screen.findByRole("button", { name: "Save Slack policy" }))

    await waitFor(() => {
      expect(mocks.updateWorkspaceSlackPolicy).toHaveBeenCalledWith({
        allowed_commands: ["help", "status"],
        channel_allowlist: ["C-1"],
        channel_denylist: ["C-2"],
        default_response_mode: "thread",
        strict_user_mapping: true,
        service_user_id: "U-service",
        user_mappings: { U123: "alice" },
        workspace_quota_per_minute: 12,
        user_quota_per_minute: 6,
        status_scope: "workspace_and_user"
      })
    })
  })

  it("surfaces Slack policy load failures and blocks saving defaults", async () => {
    mockWorkspaceQueries()
    mocks.getWorkspaceSlackPolicy.mockRejectedValue(new Error("Slack policy unavailable"))

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    expect(await screen.findByText("Unable to load Slack policy")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save Slack policy" })).toBeDisabled()
  })

  it("rejects zero-value Slack quotas instead of sending a no-op payload", async () => {
    mockWorkspaceQueries()

    renderWithQueryClient(<IntegrationManagementPage scope="workspace" />)

    const quotaInput = await screen.findByRole("spinbutton", { name: "Workspace quota / min" })
    fireEvent.change(quotaInput, { target: { value: "0" } })
    fireEvent.blur(quotaInput)
    await userEvent.setup().click(screen.getByRole("button", { name: "Save Slack policy" }))

    await waitFor(() => {
      expect(mocks.updateWorkspaceSlackPolicy).not.toHaveBeenCalled()
    })
  })
})
