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
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { WorkspaceHeader } from "../WorkspaceHeader"

const api = vi.hoisted(() => ({
  context: vi.fn(),
  get: vi.fn(),
  patch: vi.fn(),
  translate: vi.fn(),
  image: vi.fn()
}))
vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceMetadataContext: api.context
}))
vi.mock("../workspace-banner-image", () => ({
  normalizeWorkspaceBannerImage: api.image,
  WorkspaceBannerImageNormalizationError: class extends Error {}
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
  banner_title: "Original title",
  banner_subtitle: "Original subtitle",
  banner_color: "#123456",
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null,
  assistantDefaults: null
}
const state = () => useWorkspaceStore.getState()
const activate = (id = "owned", principalId = "2") => {
  const attempt = state().beginOwnedWorkspace(
    { serverBase: "https://research.test", principalId, organizationId: null },
    id
  )
  state().activateOwnedWorkspace(attempt, {
    workspace: { ...workspace, id },
    notes: [],
    sources: [],
    artifacts: []
  })
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
const open = async (owned = true) => {
  fireEvent.click(screen.getByRole("button", { name: "Workspace settings" }))
  fireEvent.click(
    await screen.findByRole("menuitem", { name: "Customize banner" })
  )
  await screen.findByLabelText("Banner title")
  if (owned)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save" })).toBeEnabled()
    )
}
const changeTitle = (value: string) =>
  fireEvent.change(screen.getByLabelText("Banner title"), {
    target: { value }
  })
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  activate()
  api.context.mockResolvedValue({ get: api.get, patch: api.patch })
  api.get.mockResolvedValue(workspace)
  api.patch.mockResolvedValue({
    ...workspace,
    version: 4,
    banner_title: "Mine"
  })
  vi.spyOn(tldwClient, "getWorkspace").mockRejectedValue(
    new Error("generic owned read")
  )
  vi.spyOn(tldwClient, "patchWorkspace").mockRejectedValue(
    new Error("generic owned write")
  )
})
afterEach(() => {
  cleanup()
  Modal.destroyAll()
  vi.restoreAllMocks()
})

describe("owned Header banner modal", () => {
  it("loads only on open and saves canonical text through the owned context", async () => {
    mount()
    expect(api.context).not.toHaveBeenCalled()
    await open()
    expect(screen.getByLabelText("Banner title")).toHaveValue("Original title")
    expect(screen.getByLabelText("Banner title")).toHaveAttribute(
      "maxlength",
      "80"
    )
    expect(screen.getByLabelText("Banner subtitle")).toHaveAttribute(
      "maxlength",
      "180"
    )
    changeTitle("Mine")
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(state().workspaceBanner.title).toBe("Mine"))
    expect(api.patch).toHaveBeenCalledWith({
      version: 3,
      banner_title: "Mine",
      banner_subtitle: "Original subtitle"
    })
    expect(state().ownedWorkspaceBundle?.workspace.banner_color).toBe("#123456")
    expect(tldwClient.getWorkspace).not.toHaveBeenCalled()
    expect(tldwClient.patchWorkspace).not.toHaveBeenCalled()
  })

  it("disables both image actions with an accessible explanation and no local upload success", async () => {
    mount()
    await open()
    for (const name of ["Upload image", "Remove image"]) {
      const control = screen.getByRole("button", { name })
      expect(control).toBeDisabled()
      expect(control).toHaveAccessibleDescription(
        "Images are not supported for server-owned banners. Existing banner color is preserved."
      )
    }
    fireEvent.change(screen.getByTestId("workspace-banner-upload-input"), {
      target: {
        files: [new File(["image"], "image.png", { type: "image/png" })]
      }
    })
    expect(api.image).not.toHaveBeenCalled()
    expect(state().workspaceBanner.image).toBeNull()
  })

  it("only resets after confirmation and sends empty text without clearing color", async () => {
    const confirmation = vi.spyOn(Modal, "confirm")
    api.patch.mockResolvedValue({
      ...workspace,
      version: 4,
      banner_title: "",
      banner_subtitle: ""
    })
    mount()
    await open()
    fireEvent.click(screen.getByRole("button", { name: "Reset banner" }))
    expect(api.patch).not.toHaveBeenCalled()
    expect(state().workspaceBanner.title).toBe("Original title")
    expect(confirmation.mock.calls[0][0].content).toBe(
      "Clear the saved banner title and subtitle? The banner color will not change."
    )
    await act(async () => {
      await confirmation.mock.calls[0][0].onOk?.()
    })
    expect(api.patch).toHaveBeenCalledWith({
      version: 3,
      banner_title: "",
      banner_subtitle: ""
    })
    expect(state().workspaceBanner.title).toBe("")
    expect(state().ownedWorkspaceBundle?.workspace.banner_color).toBe("#123456")
  })

  it.each(["reopen", "account"])(
    "does not let an old confirmation reset the %s editor",
    async (transition) => {
      const confirmation = vi
        .spyOn(Modal, "confirm")
        .mockReturnValue({ destroy: vi.fn(), update: vi.fn() })
      mount()
      await open()
      fireEvent.click(screen.getByRole("button", { name: "Reset banner" }))
      const confirm = confirmation.mock.calls[0][0].onOk!
      if (transition === "account") act(() => activate("other", "9"))
      else fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
      api.get.mockResolvedValue({
        ...workspace,
        id: transition === "account" ? "other" : "owned"
      })
      await open()
      changeTitle("New editor")
      await act(async () => {
        await confirm()
      })
      expect(api.patch).not.toHaveBeenCalled()
      expect(screen.getByLabelText("Banner title")).toHaveValue("New editor")
    }
  )

  it.each(["Keep my text", "Use server text"])(
    "shows canonical text for explicit conflict review: %s",
    async (choice) => {
      api.patch.mockRejectedValueOnce({ status: 409 })
      mount()
      await open()
      changeTitle("Mine")
      fireEvent.click(screen.getByRole("button", { name: "Save" }))
      const review = await screen.findByRole("button", {
        name: "Review latest banner"
      })
      api.get.mockResolvedValue({
        ...workspace,
        version: 4,
        banner_title: "Remote title",
        banner_subtitle: "Remote subtitle"
      })
      fireEvent.click(review)
      const latest = await screen.findByTestId("owned-banner-review")
      expect(within(latest).getByText("Remote title")).toBeInTheDocument()
      expect(within(latest).getByText("Remote subtitle")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: /Save/ })).toBeDisabled()
      expect(
        screen.getByRole("button", { name: "Reset banner" })
      ).toBeDisabled()
      fireEvent.click(screen.getByRole("button", { name: choice }))
      expect(api.patch).toHaveBeenCalledTimes(1)
      if (choice === "Keep my text") {
        expect(screen.getByLabelText("Banner title")).toHaveValue("Mine")
        expect(screen.getByRole("button", { name: /Save/ })).toBeEnabled()
        expect(state().ownedWorkspaceBannerDraft?.baseVersion).toBe(4)
      } else expect(state().workspaceBanner.title).toBe("Remote title")
    }
  )

  it("keeps legacy image completion and reset callbacks out of an owned modal", async () => {
    act(() => {
      useWorkspaceStore.setState({
        ...initialState,
        workspaceId: "legacy",
        storeHydrated: true
      })
    })
    let finish!: (image: unknown) => void
    api.image.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve
      })
    )
    const confirmation = vi
      .spyOn(Modal, "confirm")
      .mockReturnValue({ destroy: vi.fn(), update: vi.fn() })
    mount()
    await open(false)
    fireEvent.change(screen.getByTestId("workspace-banner-upload-input"), {
      target: {
        files: [new File(["image"], "image.png", { type: "image/png" })]
      }
    })
    fireEvent.click(screen.getByRole("button", { name: "Reset banner" }))
    const confirm = confirmation.mock.calls[0][0].onOk!
    act(() => activate())
    await open()
    changeTitle("Owned text")
    await act(async () => {
      finish({
        dataUrl: "data:image/png;base64,AAAA",
        mimeType: "image/png",
        width: 1,
        height: 1,
        sizeBytes: 3
      })
      await confirm()
    })
    expect(screen.getByLabelText("Banner title")).toHaveValue("Owned text")
    expect(
      screen.getByTestId("workspace-banner-preview").style.backgroundImage
    ).not.toContain("data:")
    expect(state().workspaceBanner.title).toBe("Original title")
    expect(api.patch).not.toHaveBeenCalled()
  })

  it("mirrors owned banner translations into the English extension locale", async () => {
    const { default: source } =
      await import("@/assets/locale/en/playground.json")
    const { default: extension } =
      await import("@/public/_locales/en/playground.json")
    expect(source.workspace.ownedBanner).toBeDefined()
    const messages: Record<string, { message: string }> = extension
    for (const [key, message] of Object.entries(source.workspace.ownedBanner)) {
      expect(messages[`workspace_ownedBanner_${key}`], key).toEqual({
        message
      })
    }
  })
})
