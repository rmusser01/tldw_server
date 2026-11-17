import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { App, ConfigProvider, notification } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { type Project } from "@/services/prompt-studio"
import { ProjectsTab } from "../ProjectsTab"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key
  })
}))

const existing: Project = { id: 7, name: "Existing project", description: "Existing description", status: "active" }
let projects: Project[]
let client: QueryClient
const writes = () => vi.mocked(apiSend).mock.calls.map(([payload]) => payload).filter(payload => payload.method !== "GET")

function mount() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <App><QueryClientProvider client={client}><ProjectsTab /></QueryClientProvider></App>
    </ConfigProvider>
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  usePromptStudioStore.getState().resetStore()
  projects = []
  vi.spyOn(notification, "success").mockImplementation(() => {})
  vi.spyOn(notification, "error").mockImplementation(() => {})
  vi.mocked(apiSend).mockImplementation(async payload => {
    const path = String(payload.path)
    if (payload.method === "GET" && path.includes("projects?")) return { ok: true, status: 200, data: { data: projects } }
    if (payload.method === "GET" && path.endsWith("projects/get/7")) return { ok: true, status: 200, data: { data: existing } }
    if (payload.method === "POST") {
      const created = { ...payload.body as Project, id: 8 }
      projects = [...projects, created]
      return { ok: true, status: 201, data: { data: created } }
    }
    if (payload.method === "PUT" && path.endsWith("projects/7")) return { ok: true, status: 200, data: { data: { ...existing, ...payload.body as Project } } }
    throw new Error(`Unexpected transport ${payload.method} ${path}`)
  })
})
afterEach(() => { cleanup(); client?.clear(); notification.destroy() })

describe("Project form reachability", () => {
  it.each(["empty", "populated"] as const)("creates and selects a project from the %s list", async state => {
    if (state === "populated") projects = [existing]
    usePromptStudioStore.getState().setEditingProjectId(7)
    usePromptStudioStore.getState().setSelectedPromptId(99)
    mount()
    fireEvent.click(await screen.findByRole("button", { name: "Create Project" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByLabelText("Project Name"), { target: { value: "  Rowan project  " } })
    fireEvent.change(within(dialog).getByLabelText("Description"), { target: { value: "  Fictional source  " } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Project" }))
    await waitFor(() => expect(writes()).toHaveLength(1))
    expect(writes()[0]).toMatchObject({ method: "POST", path: "/api/v1/prompt-studio/projects/", body: { name: "Rowan project", description: "Fictional source" }, headers: { "Idempotency-Key": expect.any(String) } })
    await waitFor(() => expect(usePromptStudioStore.getState().isProjectModalOpen).toBe(false))
    expect(usePromptStudioStore.getState()).toMatchObject({ selectedProjectId: 8, selectedPromptId: null, editingProjectId: null })
    expect(await screen.findByText("Rowan project")).toBeVisible()
  })

  it("cancels an empty-list draft without saving and resets it when reopened", async () => {
    mount()
    fireEvent.click(await screen.findByRole("button", { name: "Create Project" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByLabelText("Project Name"), { target: { value: "Unsaved draft" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(writes()).toHaveLength(0)
    fireEvent.click(screen.getByRole("button", { name: "Create Project" }))
    expect(within(await screen.findByRole("dialog")).getByLabelText("Project Name")).toHaveValue("")
  })

  it("still edits the selected populated row through the same form", async () => {
    projects = [existing]
    mount()
    const row = (await screen.findByText("Existing project")).closest("tr")!
    fireEvent.click(within(row).getByRole("button"))
    fireEvent.click(await screen.findByText("Edit", { exact: true }))
    expect(usePromptStudioStore.getState()).toMatchObject({ activeSubTab: "projects", selectedProjectId: null })
    const dialog = await screen.findByRole("dialog")
    const name = await within(dialog).findByLabelText("Project Name")
    await waitFor(() => expect(name).toHaveValue("Existing project"))
    fireEvent.change(name, { target: { value: "Updated project" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() => expect(writes()).toHaveLength(1))
    expect(writes()[0]).toMatchObject({ method: "PUT", path: "/api/v1/prompt-studio/projects/7", body: { name: "Updated project", description: "Existing description" } })
    await waitFor(() => expect(usePromptStudioStore.getState().isProjectModalOpen).toBe(false))
    expect(usePromptStudioStore.getState().editingProjectId).toBeNull()
  })

  it("opens Prompts when the project row itself is selected", async () => {
    projects = [existing]
    mount()
    fireEvent.click((await screen.findByText("Existing project")).closest("tr")!)
    expect(usePromptStudioStore.getState()).toMatchObject({ activeSubTab: "prompts", selectedProjectId: 7 })
    expect(writes()).toHaveLength(0)
  })
})
