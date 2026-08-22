import { fireEvent, render, screen } from "@testing-library/react"
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

const renderAt = (width: number, height: number) => {
  window.innerWidth = width
  window.innerHeight = height
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
    renderAt(390, 844)
    await screen.findByTestId("shared-workspace-shell")
    expect(screen.getByRole("tablist", { name: "Shared workspace panes" })).toBeInTheDocument()
    const panels = screen.getAllByRole("tabpanel", { hidden: true })
    expect(panels).toHaveLength(2)
    expect(panels[0]).not.toHaveAttribute("hidden")
    expect(panels[1]).toHaveAttribute("hidden")
    fireEvent.click(screen.getByRole("tab", { name: "Chat" }))
    expect(panels[0]).toHaveAttribute("hidden")
    expect(panels[1]).not.toHaveAttribute("hidden")
    expect(screen.queryByTestId("shared-workspace-desktop-grid")).not.toBeInTheDocument()
  })

  it("opens a mobile preview sheet with a loading label before evidence arrives", async () => {
    api.previewSource.mockReturnValue(new Promise(() => undefined))
    renderAt(390, 844)
    fireEvent.click(
      await screen.findByRole("button", { name: "Preview Queryable report" })
    )

    expect(
      await screen.findByRole("dialog", { name: "Loading source preview" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Loading source preview", { selector: "p" })
    ).toHaveAttribute("aria-live", "polite")
  })

  it("uses fixed minmax Sources and Chat tracks in a bounded 1440x900 shell", async () => {
    renderAt(1440, 900)
    await screen.findByTestId("shared-workspace-desktop-grid")
    expect(screen.getByTestId("shared-workspace-sources-pane")).toBeVisible()
    expect(screen.getByTestId("shared-workspace-chat-pane")).toBeVisible()
    expect(screen.queryByRole("tablist", { name: "Shared workspace panes" })).not.toBeInTheDocument()
  })
})
