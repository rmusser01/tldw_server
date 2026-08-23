import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { I18nextProvider } from "react-i18next"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { SharedResearchWorkspace } from "../SharedResearchWorkspace"
import {
  buildBootstrap,
  chatResponse,
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

vi.mock(
  "@/services/tldw/domains/shared-workspaces",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/services/tldw/domains/shared-workspaces")
      >()
    return { ...actual, sharedWorkspacesApi: api }
  }
)
vi.mock("@/services/tldw-server", () => ({ fetchChatModels }))

const renderWorkspace = () =>
  render(
    <MemoryRouter>
      <I18nextProvider i18n={testI18n}>
        <SharedResearchWorkspace shareId={42} />
      </I18nextProvider>
    </MemoryRouter>
  )

describe("SharedResearchWorkspace accessibility", () => {
  beforeAll(async () => {
    testI18n = await createSharedWorkspaceTestI18n()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    window.innerWidth = 390
    api.bootstrap.mockResolvedValue(buildBootstrap())
    api.listSources.mockResolvedValue(sourcePage)
    api.previewSource.mockResolvedValue(preview)
    api.listMessages.mockResolvedValue({
      conversation_id: "conversation-1",
      messages: [],
      next_before: null
    })
    api.ask.mockImplementation((_shareId, request) =>
      Promise.resolve({ ...chatResponse, request_id: request.request_id })
    )
    fetchChatModels.mockResolvedValue([])
  })

  it("focuses the route heading and exposes semantic mobile tabs and labels", async () => {
    renderWorkspace()
    const heading = await screen.findByRole("heading", {
      name: "Election evidence review"
    })
    await waitFor(() => expect(heading).toHaveFocus())

    const tabs = screen.getByRole("tablist", { name: "Shared workspace panes" })
    expect(tabs).toBeInTheDocument()
    const sourcesTab = screen.getByRole("tab", { name: "Sources" })
    expect(sourcesTab).toHaveAttribute(
      "aria-selected",
      "true"
    )
    expect(
      screen.getByRole("checkbox", { name: "Select Queryable report" })
    ).toBeInTheDocument()
    const panels = screen.getAllByRole("tabpanel", { hidden: true })
    expect(panels).toHaveLength(2)
    expect(panels[0]).not.toHaveAttribute("hidden")
    expect(panels[1]).toHaveAttribute("hidden")

    sourcesTab.focus()
    fireEvent.keyDown(sourcesTab, { key: "ArrowRight" })
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveFocus()
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
    expect(screen.getByLabelText("Ask about shared sources")).toBeInTheDocument()
    expect(screen.getByRole("log", { name: "Shared workspace messages" })).toBeInTheDocument()
  })

  it("opens sources and citations from the keyboard and returns preview focus", async () => {
    const user = userEvent.setup()
    renderWorkspace()
    const source = await screen.findByRole("button", {
      name: "Preview Queryable report"
    })
    source.focus()
    await user.keyboard("{Enter}")
    const dialog = await screen.findByRole("dialog", { name: "Source preview" })
    expect(dialog).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Close source preview" }))
    await waitFor(() => expect(source).toHaveFocus())

    fireEvent.click(screen.getByRole("tab", { name: "Chat" }))
    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "What does the report conclude?" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))
    const citation = await screen.findByRole("button", {
      name: "Open citation 1 from Queryable report"
    })
    citation.focus()
    await user.keyboard(" ")
    expect(await screen.findByRole("dialog", { name: "Source preview" })).toBeInTheDocument()
  })

  it("announces loading and exposes a dynamic submit label", async () => {
    let resolveAsk: (value: typeof chatResponse) => void = () => undefined
    api.ask.mockImplementation(
      (_shareId, request) =>
        new Promise((resolve) => {
          resolveAsk = (value) =>
            resolve({ ...value, request_id: request.request_id })
        })
    )
    renderWorkspace()
    await screen.findByText("Queryable report")
    fireEvent.click(screen.getByRole("tab", { name: "Chat" }))
    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "Pending question" }
    })
    const send = screen.getByRole("button", { name: "Ask shared workspace" })
    fireEvent.click(send)
    expect(screen.getByRole("status")).toHaveTextContent("Asking shared workspace")
    expect(
      screen.getByRole("button", { name: "Asking shared workspace" })
    ).toBeDisabled()

    resolveAsk(chatResponse)
    expect(await screen.findByRole("status")).toHaveTextContent("Answer added")
    expect(
      screen.getByRole("button", { name: "Ask shared workspace" })
    ).toBeInTheDocument()
  })

  it("announces and scrolls only for the assistant message completed by the submission", async () => {
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView
    let resolveAsk: (value: typeof chatResponse) => void = () => undefined
    let resolveHistory: (
      value: Awaited<ReturnType<typeof api.listMessages>>
    ) => void = () => undefined
    api.ask.mockImplementation(
      (_shareId, request) =>
        new Promise((resolve) => {
          resolveAsk = (value) =>
            resolve({ ...value, request_id: request.request_id })
        })
    )
    api.listMessages.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveHistory = resolve
        })
    )
    renderWorkspace()
    await screen.findByText("Queryable report")
    fireEvent.click(screen.getByRole("tab", { name: "Chat" }))
    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "Pending question" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))
    expect(screen.getByRole("status")).toHaveTextContent(
      "Asking shared workspace"
    )

    fireEvent.click(screen.getByRole("button", { name: "Load older messages" }))
    resolveHistory({
      conversation_id: "conversation-1",
      messages: [
        {
          message_id: "older-assistant",
          role: "assistant",
          content: "An older answer",
          created_at: "2026-08-20T10:00:00Z",
          citations: []
        }
      ],
      next_before: null
    })
    expect(await screen.findByText("An older answer")).toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent(
      "Asking shared workspace"
    )
    expect(scrollIntoView).not.toHaveBeenCalled()

    resolveAsk(chatResponse)

    expect(await screen.findByRole("status")).toHaveTextContent("Answer added")
    expect(scrollIntoView).toHaveBeenCalledTimes(1)
  })
})
