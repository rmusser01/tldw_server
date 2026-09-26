import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import { WorkspaceHeader } from "../WorkspaceHeader"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn(),
  navigate: vi.fn()
}))
vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceLifecycleContext: api.context,
  createOwnedWorkspaceMetadataContext: vi.fn()
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => api.navigate }))
vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const { default: playground } =
    await import("@/assets/locale/en/playground.json")
  const i18n = createInstance()
  await i18n.init({ lng: "en", resources: { en: { playground } } })
  return { useTranslation: () => ({ t: i18n.t.bind(i18n) }) }
})
vi.mock("@/services/workspace-context", () => ({
  useActiveWorkspaceContext: () => ({
    context: {
      state: "ready",
      workspace: null,
      recovery: { reasonCode: "allowed" }
    },
    loading: false
  })
}))
vi.mock("@/components/Common/PersonaBuddy/BuddyManagementButton", () => ({
  BuddyManagementButton: () => null
}))
vi.mock("../ShareDialog", () => ({ ShareDialog: () => null }))
vi.mock("../WorkspaceAgentTaskHandoffModal", () => ({
  WorkspaceAgentTaskHandoffModal: () => null
}))
vi.mock("../WorkspaceACPHistoryModal", () => ({
  WorkspaceACPHistoryModal: () => null
}))
vi.mock("../WorkspaceSandboxDiagnosticsPanel", () => ({
  WorkspaceSandboxDiagnosticsPanel: () => null
}))
vi.mock("../WorkspaceShortcutsModal", () => ({
  WorkspaceShortcutsModal: () => null
}))

const workspace = {
  id: "owned",
  name: "Original",
  version: 3,
  archived: false,
  deleted: false,
  created_at: "2026-09-13",
  last_modified: "2026-09-13",
  study_materials_policy: "general" as const,
  workspace_profile: "research" as const,
  banner_title: null,
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null
}
const scope = {
  serverBase: "https://research.test",
  principalId: "2",
  organizationId: null
}
const state = () => useWorkspaceStore.getState()
const mount = () =>
  render(
    <WorkspaceHeader
      leftPaneOpen
      rightPaneOpen
      onToggleLeftPane={() => {}}
      onToggleRightPane={() => {}}
    />
  )
const open = async () => {
  fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
  fireEvent.click(
    await screen.findByRole("menuitem", { name: "Archive Current Workspace" })
  )
  return screen.findByRole("dialog")
}
beforeEach(() => {
  vi.resetAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  const attempt = state().beginOwnedWorkspace(scope, "owned")
  state().activateOwnedWorkspace(attempt, {
    workspace,
    sources: [],
    notes: [],
    artifacts: []
  })
  api.context.mockResolvedValue({ scope, get: api.get, setArchived: api.patch })
  api.patch.mockResolvedValue({ ...workspace, version: 4, archived: true })
  api.get.mockResolvedValue({ ...workspace, name: "Reviewed name", version: 5 })
})
afterEach(() => {
  cleanup()
  Modal.destroyAll()
  vi.restoreAllMocks()
})

describe("owned Header archive", () => {
  it("explains retained local drafts and manager restore before a canonical archive without undo", async () => {
    const legacy = vi.spyOn(state(), "archiveWorkspace")
    mount()
    expect(api.context).not.toHaveBeenCalled()
    const dialog = await open()
    expect(dialog).toHaveTextContent(/drafts.*retained locally/i)
    expect(dialog).toHaveTextContent(/restore.*Workspaces manager/i)
    fireEvent.click(within(dialog).getByRole("button", { name: "Archive" }))
    await waitFor(() =>
      expect(api.navigate).toHaveBeenCalledWith("/workspaces")
    )
    expect(api.patch).toHaveBeenCalledExactlyOnceWith(true, 3)
    expect(legacy).not.toHaveBeenCalled()
    expect(
      screen.queryByRole("button", { name: "Undo" })
    ).not.toBeInTheDocument()
  })

  it("requires a fresh archive confirmation after explicitly checking an unarchived server record", async () => {
    api.patch.mockRejectedValueOnce({ status: 409 })
    mount()
    const dialog = await open()
    fireEvent.click(within(dialog).getByRole("button", { name: "Archive" }))
    fireEvent.click(await screen.findByRole("button", { name: "Check status" }))
    expect(
      await screen.findByText(/Reviewed name.*not archived/i)
    ).toBeInTheDocument()
    expect(api.patch).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "Archive" }))
    await waitFor(() => expect(api.patch).toHaveBeenCalledTimes(2))
    expect(api.patch).toHaveBeenLastCalledWith(true, 5)
  })

  it("blocks dismissal after server success and retries only safe completion when storage recovers", async () => {
    let resolve!: (value: typeof workspace) => void
    api.patch.mockReturnValue(
      new Promise<typeof workspace>((done) => {
        resolve = done
      })
    )
    mount()
    const dialog = await open()
    expect(screen.getByTestId("workspace-header")).not.toHaveAttribute(
      "data-workspace-blocking-recovery"
    )
    fireEvent.click(within(dialog).getByRole("button", { name: "Archive" }))
    await waitFor(() => expect(api.patch).toHaveBeenCalledTimes(1))
    const storage = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("quota")
      })
    await act(async () => resolve({ ...workspace, version: 4, archived: true }))
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /editing is blocked/i
    )
    expect(screen.getByTestId("workspace-header")).toHaveAttribute(
      "data-workspace-blocking-recovery",
      "owned"
    )
    expect(
      within(dialog).queryByRole("button", { name: "Cancel" })
    ).not.toBeInTheDocument()
    expect(
      within(dialog).queryByRole("button", { name: "Close" })
    ).not.toBeInTheDocument()
    expect(api.navigate).not.toHaveBeenCalled()
    storage.mockRestore()
    fireEvent.click(screen.getByRole("button", { name: "Finish archiving" }))
    await waitFor(() =>
      expect(api.navigate).toHaveBeenCalledWith("/workspaces")
    )
    expect(screen.getByTestId("workspace-header")).not.toHaveAttribute(
      "data-workspace-blocking-recovery"
    )
    expect(api.patch).toHaveBeenCalledTimes(1)
  })

  it("does not offer legacy archived rows in an owned workspace and keeps manager access", async () => {
    useWorkspaceStore.setState({
      archivedWorkspaces: [
        {
          id: "legacy",
          name: "Local archive",
          tag: "",
          collectionId: null,
          sourceCount: 0,
          createdAt: new Date("2026-09-13"),
          lastAccessedAt: new Date("2026-09-13")
        }
      ]
    })
    mount()
    fireEvent.click(screen.getByRole("button", { name: "Workspaces" }))
    await screen.findByRole("menu")
    expect(screen.queryByText("Local archive")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Manage in Workspaces" })
    )
    expect(api.navigate).toHaveBeenCalledWith("/workspaces")
  })

  it("mirrors the localized archive recovery messages into the extension", async () => {
    const { default: source } =
      await import("@/assets/locale/en/playground.json")
    const { default: extension } =
      await import("@/public/_locales/en/playground.json")
    expect(source.workspace.ownedArchive).toBeDefined()
    const messages: Record<string, { message: string }> = extension
    for (const [key, message] of Object.entries(source.workspace.ownedArchive))
      expect(messages[`workspace_ownedArchive_${key}`], key).toEqual({
        message
      })
  })

  it.each(["context", "receipt"])(
    "ignores a late %s after cancelling and reopening the archive modal",
    async (phase) => {
      let resolve!: (value: unknown) => void
      const pending = new Promise((done) => {
        resolve = done
      })
      if (phase === "context") api.context.mockReturnValueOnce(pending)
      else api.patch.mockReturnValueOnce(pending)
      mount()
      const dialog = await open()
      fireEvent.click(within(dialog).getByRole("button", { name: "Archive" }))
      await waitFor(() =>
        expect(
          phase === "context" ? api.context : api.patch
        ).toHaveBeenCalledTimes(1)
      )
      fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
      await open()
      await act(async () =>
        resolve(
          phase === "context"
            ? { scope, get: api.get, setArchived: api.patch }
            : { ...workspace, version: 4, archived: true }
        )
      )
      expect(api.navigate).not.toHaveBeenCalled()
      expect(state().workspaceId).toBe("owned")
      expect(screen.getByRole("dialog")).toBeInTheDocument()
      if (phase === "context") expect(api.patch).not.toHaveBeenCalled()
      else {
        expect(
          screen.queryByRole("button", { name: "Archive" })
        ).not.toBeInTheDocument()
        expect(
          screen.getByRole("button", { name: "Check status" })
        ).toBeEnabled()
        expect(api.patch).toHaveBeenCalledTimes(1)
      }
    }
  )
})
