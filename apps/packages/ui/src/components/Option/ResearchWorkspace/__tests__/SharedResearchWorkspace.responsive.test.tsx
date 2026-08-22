import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { I18nextProvider } from "react-i18next"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { SharedResearchWorkspace } from "../SharedResearchWorkspace"
import {
  buildBootstrap,
  createSharedWorkspaceTestI18n,
  preview,
  sourcePage
} from "./shared-research-workspace-test-utils"

let testI18n: Awaited<ReturnType<typeof createSharedWorkspaceTestI18n>>

const { api, fetchChatModels } = vi.hoisted(() => ({
  api: {
    bootstrap: vi.fn(),
    listSources: vi.fn(),
    previewSource: vi.fn(),
    listMessages: vi.fn(),
    ask: vi.fn()
  },
  fetchChatModels: vi.fn()
}))

vi.mock("@/services/tldw/domains/shared-workspaces", () => ({
  sharedWorkspacesApi: api
}))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels }))

const renderAt = (width: number) => {
  window.innerWidth = width
  return render(
    <MemoryRouter>
      <I18nextProvider i18n={testI18n}>
        <SharedResearchWorkspace shareId={42} />
      </I18nextProvider>
    </MemoryRouter>
  )
}

describe("SharedResearchWorkspace responsive layout", () => {
  beforeAll(async () => {
    testI18n = await createSharedWorkspaceTestI18n()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.bootstrap.mockResolvedValue(buildBootstrap())
    api.listSources.mockResolvedValue(sourcePage)
    api.previewSource.mockResolvedValue(preview)
    api.listMessages.mockResolvedValue({
      conversation_id: "conversation-1",
      messages: [],
      next_before: null
    })
    api.ask.mockResolvedValue(undefined)
    fetchChatModels.mockResolvedValue([])
  })

  it("uses Sources and Chat tabs in a bounded 390x844 mobile shell", async () => {
    renderAt(390)
    const shell = await screen.findByTestId("shared-workspace-shell")
    expect(shell.className).toContain("min-w-0")
    expect(shell.className).toContain("overflow-hidden")
    expect(screen.getByRole("tablist", { name: "Shared workspace panes" })).toBeInTheDocument()
    expect(screen.queryByTestId("shared-workspace-desktop-grid")).not.toBeInTheDocument()
  })

  it("uses fixed minmax Sources and Chat tracks in a bounded 1440x900 shell", async () => {
    renderAt(1440)
    const grid = await screen.findByTestId("shared-workspace-desktop-grid")
    expect(grid.className).toContain("min-w-0")
    expect(grid.className).toContain("grid-cols-[minmax(18rem,0.72fr)_minmax(0,1.28fr)]")
    expect(screen.getByTestId("shared-workspace-sources-pane").className).toContain("min-w-0")
    expect(screen.getByTestId("shared-workspace-chat-pane").className).toContain("min-w-0")
    expect(screen.queryByRole("tablist", { name: "Shared workspace panes" })).not.toBeInTheDocument()
  })
})
