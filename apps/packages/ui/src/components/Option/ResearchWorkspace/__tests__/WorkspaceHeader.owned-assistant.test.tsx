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
import { initialState, useWorkspaceStore } from "@/store/workspace"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { WorkspaceHeader } from "../WorkspaceHeader"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn(),
  listPersonas: vi.fn(),
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

const defaults = (
  assistantId = "one",
  personaMemoryMode: "read_only" | "read_write" = "read_only"
) => ({
  assistantKind: "persona" as const,
  assistantId,
  personaMemoryMode,
  voice: null,
  style: null,
  toolPolicyProfileId: null
})
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
  audio_speed: null,
  assistantDefaults: defaults()
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
const open = async () => {
  fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
  fireEvent.click(
    await screen.findByRole("menuitem", { name: "Default assistant" })
  )
  await screen.findByTestId("workspace-default-assistant-modal")
  await waitFor(() => expect(api.listPersonas).toHaveBeenCalled())
}
const select = (value: string) =>
  fireEvent.change(screen.getByTestId("workspace-default-assistant-select"), {
    target: { value }
  })
const save = () =>
  fireEvent.click(screen.getByRole("button", { name: "Save default" }))
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
  api.context.mockResolvedValue({
    get: api.get,
    patch: api.patch,
    listPersonas: api.listPersonas
  })
  api.get.mockResolvedValue(workspace)
  api.patch.mockResolvedValue({
    ...workspace,
    version: 4,
    assistantDefaults: defaults("two")
  })
  api.listPersonas.mockResolvedValue([
    { id: "one", name: "First Persona" },
    { id: "two", name: "Second Persona" }
  ])
  vi.spyOn(tldwClient, "getWorkspace").mockRejectedValue(
    new Error("generic owned read")
  )
  vi.spyOn(tldwClient, "listPersonaProfiles").mockRejectedValue(
    new Error("generic owned catalog")
  )
  vi.spyOn(tldwClient, "patchWorkspace").mockRejectedValue(
    new Error("generic owned write")
  )
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("owned Header default assistant modal", () => {
  it("opens and saves through the owned context with no mount request or generic client", async () => {
    mount()
    expect(api.context).not.toHaveBeenCalled()
    await open()
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save default" })).toBeEnabled()
    )
    select("two")
    save()
    await waitFor(() =>
      expect(useWorkspaceStore.getState().assistantDefaults).toEqual(
        defaults("two")
      )
    )
    expect(api.patch).toHaveBeenCalledWith({
      version: 3,
      assistantDefaults: defaults("two")
    })
    expect(tldwClient.getWorkspace).not.toHaveBeenCalled()
    expect(tldwClient.listPersonaProfiles).not.toHaveBeenCalled()
    expect(tldwClient.patchWorkspace).not.toHaveBeenCalled()
  })

  it("requires explicit fresh consent when the Persona selection changes", async () => {
    mount()
    await open()
    fireEvent.change(
      screen.getByTestId("workspace-default-assistant-memory-mode"),
      { target: { value: "read_write" } }
    )
    expect(screen.getByRole("button", { name: "Save default" })).toBeDisabled()
    fireEvent.click(screen.getByRole("checkbox"))
    expect(screen.getByRole("button", { name: "Save default" })).toBeEnabled()
    select("two")
    expect(screen.getByRole("checkbox")).not.toBeChecked()
    expect(screen.getByRole("button", { name: "Save default" })).toBeDisabled()
    fireEvent.click(screen.getByRole("checkbox"))
    save()
    await waitFor(() =>
      expect(api.patch).toHaveBeenCalledWith({
        version: 3,
        assistantDefaults: defaults("two", "read_write"),
        confirmReadWriteAssistantDefault: true
      })
    )
  })

  it.each([404, 500])(
    "allows Clear but not Save or selection after catalog failure %s",
    async (status) => {
      api.listPersonas.mockRejectedValue({ status })
      api.patch.mockResolvedValue({
        ...workspace,
        version: 4,
        assistantDefaults: null
      })
      mount()
      await open()
      expect(
        await screen.findByText(
          "Could not load Personas. You can still clear the current default."
        )
      ).toBeInTheDocument()
      expect(
        screen.getByTestId("workspace-default-assistant-select")
      ).toBeDisabled()
      expect(
        screen.getByRole("button", { name: "Save default" })
      ).toBeDisabled()
      fireEvent.click(screen.getByRole("button", { name: "Clear default" }))
      await waitFor(() =>
        expect(api.patch).toHaveBeenCalledWith({
          version: 3,
          assistantDefaults: null
        })
      )
    }
  )

  it.each(["Keep my settings", "Use server default"])(
    "shows the latest server Persona and memory mode before %s",
    async (choice) => {
      api.patch.mockRejectedValue({ status: 409 })
      mount()
      await open()
      save()
      const review = await screen.findByRole("button", {
        name: "Review latest default"
      })
      api.get.mockResolvedValue({
        ...workspace,
        name: "Unrelated name",
        version: 4,
        assistantDefaults: defaults("two", "read_write")
      })
      fireEvent.click(review)
      const latest = await screen.findByTestId("owned-assistant-review")
      expect(within(latest).getByText("Second Persona")).toBeInTheDocument()
      expect(within(latest).getByText("Read and write")).toBeInTheDocument()
      expect(
        screen.getByTestId("workspace-default-assistant-select")
      ).toHaveValue("one")
      expect(
        screen.getByRole("button", { name: /Save default/ })
      ).toBeDisabled()
      fireEvent.click(screen.getByRole("button", { name: choice }))
      expect(api.patch).toHaveBeenCalledTimes(1)
      if (choice === "Keep my settings") {
        expect(
          screen.getByRole("button", { name: /Save default/ })
        ).toBeEnabled()
        expect(
          useWorkspaceStore.getState().ownedWorkspaceAssistantDraft?.baseVersion
        ).toBe(4)
      } else {
        expect(useWorkspaceStore.getState().assistantDefaults).toEqual(
          defaults("two", "read_write")
        )
        expect(
          useWorkspaceStore.getState().ownedWorkspaceAssistantDraft
        ).toBeNull()
      }
    }
  )

  it("retains draft settings across unmount but never checkbox consent", async () => {
    const first = mount()
    await open()
    select("two")
    fireEvent.change(
      screen.getByTestId("workspace-default-assistant-memory-mode"),
      { target: { value: "read_write" } }
    )
    fireEvent.click(screen.getByRole("checkbox"))
    first.unmount()
    mount()
    await open()
    expect(
      screen.getByTestId("workspace-default-assistant-select")
    ).toHaveValue("two")
    expect(screen.getByRole("checkbox")).not.toBeChecked()
    expect(screen.getByRole("button", { name: "Save default" })).toBeDisabled()
    fireEvent.click(
      await screen.findByRole("button", { name: "Keep my settings" })
    )
    expect(screen.getByRole("button", { name: "Save default" })).toBeDisabled()
  })

  it("Cancel discards the persisted draft through the store action", async () => {
    mount()
    await open()
    select("two")
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    expect(useWorkspaceStore.getState().ownedWorkspaceAssistantDraft).toBeNull()
  })

  it("keeps a reopened modal unchanged after an already dispatched save", async () => {
    let finish!: (value: typeof workspace) => void
    api.patch.mockReturnValue(
      new Promise<typeof workspace>((resolve) => {
        finish = resolve
      })
    )
    mount()
    await open()
    save()
    await waitFor(() => expect(api.patch).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    await open()
    select("two")
    await act(async () => {
      finish({ ...workspace, version: 4 })
    })
    expect(
      screen.getByTestId("workspace-default-assistant-select")
    ).toHaveValue("two")
    expect(
      screen.getByRole("button", { name: /Save default/ })
    ).toBeInTheDocument()
  })

  it("mirrors every owned assistant message into the English extension locale", async () => {
    const { default: source } =
      await import("@/assets/locale/en/playground.json")
    const { default: extension } =
      await import("@/public/_locales/en/playground.json")
    expect(source.workspace.ownedAssistant).toBeDefined()
    const messages: Record<string, { message: string }> = extension
    for (const [key, message] of Object.entries(
      source.workspace.ownedAssistant
    )) {
      expect(messages[`workspace_ownedAssistant_${key}`], key).toEqual({
        message
      })
    }
  })
})
