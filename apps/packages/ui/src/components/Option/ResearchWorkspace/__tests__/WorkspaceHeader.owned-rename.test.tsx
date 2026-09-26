import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import { WorkspaceHeader } from "../WorkspaceHeader"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn(),
  translate: vi.fn()
}))
vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceMetadataContext: api.context
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }))
vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const { default: playground } =
    await import("@/assets/locale/en/playground.json")
  const i18n = createInstance()
  await i18n.init({ lng: "en", resources: { en: { playground } } })
  api.translate.mockImplementation(i18n.t.bind(i18n))
  return { useTranslation: () => ({ t: api.translate }) }
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
const mount = () =>
  render(
    <WorkspaceHeader
      leftPaneOpen
      rightPaneOpen
      onToggleLeftPane={() => {}}
      onToggleRightPane={() => {}}
    />
  )
const edit = () => {
  fireEvent.click(screen.getByRole("button", { name: "Rename workspace" }))
  fireEvent.change(screen.getByRole("textbox", { name: "Workspace name" }), {
    target: { value: "Mine" }
  })
}
const save = () => fireEvent.click(screen.getByRole("button", { name: "Save" }))
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  const state = useWorkspaceStore.getState()
  const attempt = state.beginOwnedWorkspace(
    {
      serverBase: "https://research.test",
      principalId: "2",
      organizationId: null
    },
    "owned"
  )
  state.activateOwnedWorkspace(attempt, {
    workspace,
    notes: [],
    sources: [],
    artifacts: []
  })
  api.context.mockResolvedValue({ get: api.get, patch: api.patch })
  api.patch.mockResolvedValue({ ...workspace, name: "Mine", version: 4 })
  api.get.mockResolvedValue({ ...workspace, name: "Remote", version: 4 })
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  const state = useWorkspaceStore.getState()
  state.setOwnedWorkspaceRenameDraft({
    origin: state.activeWorkspaceOrigin,
    workspaceId: state.workspaceId,
    expectedDraft: state.ownedWorkspaceRenameDraft,
    draft: null
  })
})

describe("owned Header inline rename", () => {
  it("saves canonically through the existing inline editor with no mount request", async () => {
    mount()
    expect(api.context).not.toHaveBeenCalled()
    edit()
    save()
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Mine" })).toBeInTheDocument()
    )
    expect(api.patch).toHaveBeenCalledWith({ name: "Mine", version: 3 })
    expect(api.get).not.toHaveBeenCalled()
  })

  it("renders the persisted editor immediately on remount without requests", () => {
    const view = mount()
    edit()
    view.unmount()
    mount()
    expect(screen.getByRole("textbox", { name: "Workspace name" })).toHaveValue(
      "Mine"
    )
    expect(api.context).not.toHaveBeenCalled()
  })

  it.each(["Keep my name", "Use server name"])(
    "requires explicit conflict review and %s",
    async (choice) => {
      api.patch.mockRejectedValue({ status: 409 })
      mount()
      edit()
      save()
      fireEvent.click(
        await screen.findByRole("button", { name: "Review latest name" })
      )
      expect(
        await screen.findByText("Latest server name: Remote")
      ).toBeInTheDocument()
      expect(
        screen.getByRole("textbox", { name: "Workspace name" })
      ).toHaveValue("Mine")
      fireEvent.click(screen.getByRole("button", { name: choice }))
      if (choice === "Keep my name") {
        expect(
          screen.getByRole("textbox", { name: "Workspace name" })
        ).toHaveValue("Mine")
        expect(screen.getByRole("button", { name: "Save" })).toBeEnabled()
        expect(
          useWorkspaceStore.getState().ownedWorkspaceRenameDraft?.baseVersion
        ).toBe(4)
      } else {
        expect(
          screen.getByRole("heading", { name: "Remote" })
        ).toBeInTheDocument()
      }
      expect(api.patch).toHaveBeenCalledTimes(1)
    }
  )

  it.each([
    [412, "accountChanged"],
    [undefined, "saveFailed"]
  ] as const)(
    "uses localized inline failure copy for %s and retains typing",
    async (status, key) => {
      api.patch.mockRejectedValue({ status })
      const { default: locale } =
        await import("@/assets/locale/en/playground.json")
      mount()
      edit()
      save()
      expect(await screen.findByRole("alert")).toHaveTextContent(
        locale.workspace.ownedRename[key]
      )
      expect(api.translate).toHaveBeenCalledWith(
        `playground:workspace.ownedRename.${key}`
      )
      expect(
        screen.getByRole("textbox", { name: "Workspace name" })
      ).toHaveValue("Mine")
    }
  )

  it("shows storage failure inline without discarding the rename", async () => {
    const { default: locale } =
      await import("@/assets/locale/en/playground.json")
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota")
    })
    mount()
    edit()
    expect(
      screen.getByText(locale.workspace.ownedRename.storageUnavailable)
    ).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Workspace name" })).toHaveValue(
      "Mine"
    )
  })

  it("locks Save while pending but permits typing and cancelling", async () => {
    let finish!: (value: typeof workspace) => void
    api.patch.mockReturnValue(
      new Promise<typeof workspace>((resolve) => {
        finish = resolve
      })
    )
    mount()
    edit()
    save()
    await waitFor(() => expect(api.patch).toHaveBeenCalledTimes(1))
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    fireEvent.change(screen.getByRole("textbox", { name: "Workspace name" }), {
      target: { value: "Newer" }
    })
    await act(async () => {
      finish({ ...workspace, name: "Mine", version: 4 })
    })
    expect(screen.getByRole("textbox", { name: "Workspace name" })).toHaveValue(
      "Newer"
    )
  })

  it("mirrors every new English rename message into the extension", async () => {
    const { default: source } =
      await import("@/assets/locale/en/playground.json")
    const { default: extension } =
      await import("@/public/_locales/en/playground.json")
    expect(source.workspace.ownedRename).toBeDefined()
    const messages: Record<string, { message: string }> = extension
    for (const [key, message] of Object.entries(source.workspace.ownedRename)) {
      expect(messages[`workspace_ownedRename_${key}`], key).toEqual({
        message
      })
    }
  })
})
