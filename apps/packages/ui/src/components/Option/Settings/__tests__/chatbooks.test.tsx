import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { getHeaderShortcutGroups } from "@/components/Layouts/header-shortcut-items"
import { getRouteMetadata } from "@/routes/route-metadata"
import { ChatbooksSettings } from "../chatbooks"

const mocks = vi.hoisted(() => ({
  capabilities: {
    hasChatbooks: true
  },
  notificationError: vi.fn(),
  notificationSuccess: vi.fn(),
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    exportChatbook: vi.fn(async () => ({ job_id: "job-1" }))
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: mocks.capabilities
  })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    serverChatId: "server-conversation-1"
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: mocks.notificationError,
    success: mocks.notificationSuccess
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: mocks.tldwClient
}))

const renderSettings = () =>
  render(
    <MemoryRouter>
      <ChatbooksSettings />
    </MemoryRouter>
  )

describe("ChatbooksSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.capabilities.hasChatbooks = true
  })

  it("routes full account backup and import work to the dedicated screen", () => {
    renderSettings()

    const backupLink = screen.getByRole("link", {
      name: /open backup & import/i
    })

    expect(backupLink).toHaveAttribute("href", "/chatbooks")
    expect(
      screen.getAllByText(/complete account backups and archive imports/i)
        .length
    ).toBeGreaterThan(0)
    expect(screen.queryByRole("button", { name: /backup all/i })).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /import chatbook/i })
    ).not.toBeInTheDocument()
  })

  it("keeps the settings export shortcut explicitly conversation-only", async () => {
    renderSettings()

    expect(
      screen.getByRole("heading", { name: /selective conversation export/i })
    ).toBeInTheDocument()
    expect(
      screen.getByText(/not a full account backup/i)
    ).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText(/conversation ids/i), {
      target: { value: "conv-1, conv-2" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: /export selected conversations/i })
    )

    await waitFor(() => {
      expect(mocks.tldwClient.exportChatbook).toHaveBeenCalledWith(
        expect.objectContaining({
          content_selections: {
            conversation: ["conv-1", "conv-2"]
          }
        })
      )
    })
  })

  it("uses Backup & Import naming for the route and header shortcut", () => {
    const chatbookShortcut = getHeaderShortcutGroups()
      .flatMap((group) => group.items)
      .find((item) => item.id === "chatbooks-playground")

    expect(chatbookShortcut).toMatchObject({
      to: "/chatbooks",
      labelDefault: "Chatbooks Backup & Import"
    })
    expect(getRouteMetadata("/chatbooks")).toMatchObject({
      label: "Chatbooks Backup & Import",
      canonicalPath: "/chatbooks"
    })
    expect(getRouteMetadata("/chatbooks-playground")).toMatchObject({
      label: "Chatbooks Backup & Import",
      canonicalPath: "/chatbooks",
      surface: "legacy_alias",
      redirectsTo: "/chatbooks"
    })
  })
})
