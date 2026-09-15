import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useStoreMessageOption } from "@/store/option"
import { useActorStore } from "@/store/actor"
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import type { RolePlayState } from "../role-play-state"
import { RolePlaySetupDrawer } from "../RolePlaySetupDrawer"
import { createStartupTemplateBundle } from "../startup-template-bundles"

const actorSettingsMocks = vi.hoisted(() => ({
  getActorSettingsForChatWithCharacterFallback: vi.fn(),
  saveActorSettingsForChat: vi.fn()
}))

vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChatWithCharacterFallback:
    actorSettingsMocks.getActorSettingsForChatWithCharacterFallback,
  saveActorSettingsForChat: actorSettingsMocks.saveActorSettingsForChat
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string, values?: Record<string, unknown>) =>
      (fallback || key).replace(/\{\{(\w+)\}\}/g, (_match, token) =>
        values?.[token] == null ? "" : String(values[token])
      )
  })
}))

vi.mock("@/components/Common/AssistantSelect", () => ({
  AssistantSelect: () => <button data-testid="role-play-identity-select">Mira</button>
}))

vi.mock("../playground-features", () => ({
  PRESETS: [
    {
      key: "creative",
      label: "Creative",
      description: "Creative",
      icon: null,
      settings: { temperature: 1.2, topP: 0.95 }
    },
    {
      key: "balanced",
      label: "Balanced",
      description: "Balanced",
      icon: null,
      settings: { temperature: 0.7, topP: 0.9 }
    },
    {
      key: "precise",
      label: "Precise",
      description: "Precise",
      icon: null,
      settings: { temperature: 0.2, topP: 0.8 }
    },
    {
      key: "custom",
      label: "Custom",
      description: "Custom",
      icon: null,
      settings: {}
    }
  ],
  SystemPromptTemplatesModal: ({
    open,
    onSelect
  }: {
    open: boolean
    onSelect: (template: {
      id: string
      title: string
      content: string
      category: string
    }) => void
  }) =>
    open ? (
      <div role="dialog" aria-label="System prompts">
        <button
          type="button"
          onClick={() =>
            onSelect({
              id: "detective",
              title: "Detective",
              content: "Observe everything.",
              category: "roleplay"
            })
          }>
          Use Detective
        </button>
      </div>
    ) : null
}))

vi.mock("antd", () => ({
  Drawer: ({
    open,
    title,
    children,
    footer,
    onClose
  }: {
    open?: boolean
    title?: React.ReactNode
    children: React.ReactNode
    footer?: React.ReactNode
    onClose?: () => void
  }) =>
    open ? (
      <section role="dialog" aria-label={String(title || "Role-play setup")}
        onKeyDown={(event) => { if (event.key === "Escape") onClose?.() }}>
        <button type="button" onClick={onClose}>
          Close drawer
        </button>
        {children}
        {footer}
      </section>
    ) : null,
  Button: React.forwardRef<HTMLButtonElement, {
    children: React.ReactNode
    onClick?: () => void
    htmlType?: string
    "aria-label"?: string
  }>(({
    children,
    onClick,
    htmlType,
    "aria-label": ariaLabel
  }, ref) => (
    <button
      ref={ref}
      type={htmlType === "submit" ? "submit" : "button"}
      aria-label={ariaLabel}
      onClick={onClick}>
      {children}
    </button>
  )),
  Checkbox: ({
    checked,
    onChange,
    children
  }: {
    checked?: boolean
    onChange?: (event: { target: { checked: boolean } }) => void
    children?: React.ReactNode
  }) => (
    <label>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange?.({ target: { checked: event.target.checked } })}
      />
      {children}
    </label>
  ),
  Input: Object.assign(
    ({
      value,
      onChange,
      "aria-label": ariaLabel
    }: {
      value?: string
      onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void
      "aria-label"?: string
    }) => <input aria-label={ariaLabel} value={value} onChange={onChange} />,
    {
      TextArea: ({
        value,
        onChange,
        "aria-label": ariaLabel
      }: {
        value?: string
        onChange?: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
        "aria-label"?: string
      }) => <textarea aria-label={ariaLabel} value={value} onChange={onChange} />
    }
  ),
  Skeleton: () => <div data-testid="scene-loading" />,
  Switch: ({
    checked,
    onChange
  }: {
    checked?: boolean
    onChange?: (checked: boolean) => void
  }) => (
    <input
      aria-label="Scene enabled"
      type="checkbox"
      checked={checked}
      onChange={(event) => onChange?.(event.target.checked)}
    />
  )
}))

const beforeState: RolePlayState = {
  active: true,
  identity: {
    kind: "character",
    id: "char-mira",
    name: "Mira"
  },
  behavior: {
    source: "template",
    templateId: "character-actor",
    title: "Character Actor",
    modified: false
  },
  scene: null,
  generationStyle: {
    key: "creative",
    label: "Creative"
  },
  context: {
    pinnedCount: 2,
    hasExternalContext: true
  }
}

const activeScene = (): ActorSettings => {
  const settings = createDefaultActorSettings()
  return {
    ...settings,
    isEnabled: true,
    notes: "The room smells like ozone.",
    aspects: settings.aspects.map((aspect) =>
      aspect.id === "world_location"
        ? { ...aspect, value: "Observatory" }
        : aspect
    )
  }
}

const renderDrawer = (
  overrides: Partial<React.ComponentProps<typeof RolePlaySetupDrawer>> = {}
) => {
  const returnFocusRef = React.createRef<HTMLButtonElement>()
  const view = render(
    <>
      <button ref={returnFocusRef} type="button">
        Setup trigger
      </button>
      <RolePlaySetupDrawer
        open
        beforeState={beforeState}
        historyId="history-1"
        serverChatId="server-1"
        characterId="char-mira"
        returnFocusRef={returnFocusRef}
        onClose={vi.fn()}
        onApply={vi.fn()}
        {...overrides}
      />
    </>
  )
  return { returnFocusRef, ...view }
}

describe("RolePlaySetupDrawer", () => {
  beforeEach(() => {
    useStoreMessageOption.setState({ serverChatId: "server-1", historyId: "history-1" })
    actorSettingsMocks.getActorSettingsForChatWithCharacterFallback.mockResolvedValue(
      activeScene()
    )
    actorSettingsMocks.saveActorSettingsForChat.mockResolvedValue(true)
    vi.clearAllMocks()
  })

  it("loads actor settings and renders before/after role-play state", async () => {
    renderDrawer()

    expect(
      await screen.findByRole("dialog", { name: "Role-play setup" })
    ).toBeInTheDocument()

    await waitFor(() =>
      expect(
        actorSettingsMocks.getActorSettingsForChatWithCharacterFallback
      ).toHaveBeenCalledWith({
        historyId: "history-1",
        serverChatId: "server-1",
        characterId: "char-mira"
      })
    )
    expect(screen.getAllByText("Character").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Mira").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Behavior").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Character Actor").length).toBeGreaterThan(0)
    expect(screen.getByTestId("role-play-identity-select")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Choose behavior template" })
    ).toBeInTheDocument()
    expect(screen.getAllByText("Generation style").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Creative").length).toBeGreaterThan(0)
    expect(screen.getByText(/1 detail/)).toBeInTheDocument()
    expect(screen.getByText(/2 pinned/)).toBeInTheDocument()
  })

  it("announces scene setting loading as a status", async () => {
    let resolveActor: (settings: ActorSettings) => void = () => {}
    actorSettingsMocks.getActorSettingsForChatWithCharacterFallback.mockReturnValueOnce(
      new Promise<ActorSettings>((resolve) => {
        resolveActor = resolve
      })
    )

    renderDrawer()

    expect(screen.getByRole("status")).toHaveTextContent(
      "Loading scene settings..."
    )

    resolveActor(activeScene())
    await screen.findByText(/1 detail/)
  })

  it("does not crash when stored actor settings omit aspects", async () => {
    actorSettingsMocks.getActorSettingsForChatWithCharacterFallback.mockResolvedValueOnce(
      {
        ...activeScene(),
        aspects: undefined
      } as unknown as ActorSettings
    )

    renderDrawer()

    expect(
      await screen.findByRole("dialog", { name: "Role-play setup" })
    ).toBeInTheDocument()
    expect(await screen.findByLabelText("Scene notes")).toHaveValue(
      "The room smells like ozone."
    )
  })

  it("surfaces scene load failures without applying stale scene state", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined)
    actorSettingsMocks.getActorSettingsForChatWithCharacterFallback.mockRejectedValueOnce(
      new Error("load failed")
    )

    renderDrawer()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Scene settings could not be loaded."
    )
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    consoleError.mockRestore()
  })

  it("cancel closes without applying or saving", async () => {
    const onClose = vi.fn()
    const onApply = vi.fn()
    renderDrawer({ onClose, onApply })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onApply).not.toHaveBeenCalled()
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
  })

  it("applies all staged changes with one apply callback", async () => {
    const onApply = vi.fn()
    renderDrawer({ onApply })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Clear identity" }))
    fireEvent.click(screen.getByRole("button", { name: "Clear behavior" }))
    fireEvent.click(screen.getByRole("button", { name: "Reset generation" }))
    fireEvent.click(screen.getByRole("button", { name: "Clear scene" }))
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))

    await waitFor(() => expect(onApply).toHaveBeenCalledTimes(1))
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({
        clearIdentity: true,
        clearBehavior: true,
        resetGenerationStyle: true,
        sceneSettings: expect.objectContaining({
          isEnabled: false,
          notes: ""
        })
      })
    )
    expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledWith({
      historyId: "history-1",
      serverChatId: "server-1",
      settings: expect.objectContaining({
        isEnabled: false,
        notes: ""
      })
    })
  })

  it("stages behavior templates and generation style before applying", async () => {
    const onApply = vi.fn()
    renderDrawer({ onApply })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Choose behavior template" }))
    fireEvent.click(screen.getByRole("button", { name: "Use Detective" }))
    fireEvent.click(screen.getByRole("radio", { name: "Precise" }))
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))

    await waitFor(() => expect(onApply).toHaveBeenCalledTimes(1))
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({
        behaviorTemplate: {
          id: "detective",
          title: "Detective",
          content: "Observe everything.",
          category: "roleplay"
        },
        generationPresetKey: "precise"
      })
    )
  })

  it("resets scene draft to default actor settings", async () => {
    const onApply = vi.fn()
    renderDrawer({ onApply })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Reset scene" }))
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))

    await waitFor(() => expect(onApply).toHaveBeenCalledTimes(1))
    expect(onApply.mock.calls[0][0].sceneSettings).toEqual(
      createDefaultActorSettings()
    )
  })

  it("closes on Escape and returns focus to the trigger", async () => {
    const onClose = vi.fn()
    const { returnFocusRef } = renderDrawer({ onClose })

    await screen.findByText(/1 detail/)
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Role-play setup" }), { key: "Escape" })

    expect(onClose).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(document.activeElement).toBe(returnFocusRef.current))
  })

  it("shows only role-play relevant saved setups", async () => {
    const savedSetup = createStartupTemplateBundle(
      {
        name: "Mira detective scene",
        selectedModel: "openai:gpt-4.1",
        systemPrompt: "Observe everything.",
        source: "role-play-setup",
        character: {
          id: "char-mira",
          name: "Mira"
        } as any,
        rolePlay: {
          source: "role-play-setup",
          identity: {
            kind: "character",
            id: "char-mira",
            name: "Mira"
          },
          behavior: {
            source: "template",
            templateId: "detective",
            templateTitle: "Detective",
            templateCategory: "roleplay",
            systemPrompt: "Observe everything.",
            modified: false
          },
          scene: activeScene(),
          generation: null,
          context: null
        }
      },
      { id: "saved-role-play", now: 1 }
    )
    const genericTemplate = createStartupTemplateBundle(
      {
        name: "Generic writing template",
        selectedModel: "openai:gpt-4.1",
        systemPrompt: "",
        presetKey: "creative"
      },
      { id: "generic", now: 2 }
    )
    const onPreviewSavedSetup = vi.fn()

    renderDrawer({
      savedRolePlaySetups: [savedSetup, genericTemplate],
      savedSetupDraftName: "",
      savedSetupNameFallback: "Mira setup",
      onSavedSetupDraftNameChange: vi.fn(),
      onSaveRolePlaySetup: vi.fn(),
      onPreviewSavedSetup,
      onApplySavedSetup: vi.fn(),
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup: vi.fn()
    })

    await screen.findByText(/1 detail/)
    expect(screen.getByText("Saved role-play setups")).toBeInTheDocument()
    expect(
      screen.getByRole("list", { name: "Saved role-play setup list" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("listitem", { name: "Mira detective scene role-play setup" })
    ).toBeInTheDocument()
    expect(screen.getByText("Mira detective scene")).toBeInTheDocument()
    expect(screen.queryByText("Generic writing template")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Preview Mira detective scene" }))
    expect(onPreviewSavedSetup).toHaveBeenCalledWith("saved-role-play")
  })

  it("requires confirmation before deleting a saved role-play setup", async () => {
    const savedSetup = createStartupTemplateBundle(
      {
        name: "Mira detective scene",
        selectedModel: "openai:gpt-4.1",
        systemPrompt: "Observe everything.",
        source: "role-play-setup",
        character: {
          id: "char-mira",
          name: "Mira"
        } as any,
        rolePlay: {
          source: "role-play-setup",
          identity: {
            kind: "character",
            id: "char-mira",
            name: "Mira"
          },
          behavior: null,
          scene: activeScene(),
          generation: null,
          context: null
        }
      },
      { id: "saved-role-play", now: 1 }
    )
    const onDeleteSavedSetup = vi.fn()

    renderDrawer({
      savedRolePlaySetups: [savedSetup],
      savedSetupDraftName: "",
      savedSetupNameFallback: "Mira setup",
      onSavedSetupDraftNameChange: vi.fn(),
      onSaveRolePlaySetup: vi.fn(),
      onPreviewSavedSetup: vi.fn(),
      onApplySavedSetup: vi.fn(),
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup
    })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Delete Mira detective scene" }))

    expect(onDeleteSavedSetup).not.toHaveBeenCalled()
    const alert = screen.getByRole("alert")
    expect(alert).toHaveTextContent(
      "Delete Mira detective scene?"
    )
    const confirmButton = screen.getByRole("button", {
      name: "Confirm delete Mira detective scene"
    })
    expect(confirmButton).toHaveFocus()

    fireEvent.click(confirmButton)
    expect(onDeleteSavedSetup).toHaveBeenCalledWith("saved-role-play")
  })

  it("exposes generation style choices as radios", async () => {
    renderDrawer()

    await screen.findByText(/1 detail/)
    const group = screen.getByRole("radiogroup", { name: "Generation style" })
    expect(within(group).getByRole("radio", { name: "Creative" })).toBeChecked()
    expect(within(group).getByRole("radio", { name: "Balanced" })).not.toBeChecked()
    expect(within(group).getByRole("radio", { name: "Precise" })).not.toBeChecked()
    expect(within(group).getByRole("radio", { name: "Custom" })).not.toBeChecked()
  })

  it("applies saved role-play setup scene through the actor settings save path", async () => {
    const savedSetup = createStartupTemplateBundle(
      {
        name: "Mira detective scene",
        selectedModel: "openai:gpt-4.1",
        systemPrompt: "Observe everything.",
        source: "role-play-setup",
        character: {
          id: "char-mira",
          name: "Mira"
        } as any,
        rolePlay: {
          source: "role-play-setup",
          identity: {
            kind: "character",
            id: "char-mira",
            name: "Mira"
          },
          behavior: null,
          scene: activeScene(),
          generation: null,
          context: null
        }
      },
      { id: "saved-role-play", now: 1 }
    )
    const onApplySavedSetup = vi.fn()

    renderDrawer({
      savedRolePlaySetups: [savedSetup],
      savedSetupDraftName: "",
      savedSetupNameFallback: "Mira setup",
      onSavedSetupDraftNameChange: vi.fn(),
      onSaveRolePlaySetup: vi.fn(),
      onPreviewSavedSetup: vi.fn(),
      onApplySavedSetup,
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup: vi.fn()
    })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply Mira detective scene" }))

    await waitFor(() =>
      expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledWith({
        historyId: "history-1",
        serverChatId: "server-1",
        settings: expect.objectContaining({
          isEnabled: true,
          notes: "The room smells like ozone."
        })
      })
    )
    expect(onApplySavedSetup).toHaveBeenCalledWith(savedSetup)
  })

  it("saves the staged role-play setup metadata", async () => {
    const onSaveRolePlaySetup = vi.fn()
    renderDrawer({
      currentSystemPrompt: "Observe everything.",
      ragPinnedResultIds: ["source-1", "source-2"],
      savedRolePlaySetups: [],
      savedSetupDraftName: "Mira saved",
      savedSetupNameFallback: "Mira setup",
      onSavedSetupDraftNameChange: vi.fn(),
      onSaveRolePlaySetup,
      onPreviewSavedSetup: vi.fn(),
      onApplySavedSetup: vi.fn(),
      onRenameSavedSetup: vi.fn(),
      onDeleteSavedSetup: vi.fn()
    })

    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Choose behavior template" }))
    fireEvent.click(screen.getByRole("button", { name: "Use Detective" }))
    fireEvent.click(screen.getByRole("radio", { name: "Precise" }))
    fireEvent.click(screen.getByRole("button", { name: "Save setup" }))

    expect(onSaveRolePlaySetup).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Mira saved",
        rolePlay: expect.objectContaining({
          source: "role-play-setup",
          identity: {
            kind: "character",
            id: "char-mira",
            name: "Mira"
          },
          behavior: expect.objectContaining({
            source: "template",
            templateId: "detective",
            templateTitle: "Detective",
            templateCategory: "roleplay",
            systemPrompt: "Observe everything."
          }),
          generation: expect.objectContaining({
            presetKey: "precise"
          }),
          context: {
            ragPinnedCount: 2,
            ragPinnedResultIds: ["source-1", "source-2"]
          }
        })
      })
    )
  })
  it("persists a scene only after accepted replacement and against its new destination", async () => {
    let release!: (accepted: boolean) => void
    const pending = new Promise<boolean>(resolve => { release = resolve })
    const onApply = vi.fn(() => {
      useStoreMessageOption.setState({ serverChatId: null, historyId: null })
      return pending
    })
    renderDrawer({ onApply })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    await act(async () => { release(true); await pending })
    expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledExactlyOnceWith({ historyId: null, serverChatId: null, settings: activeScene() })
  })

  it("does not persist or display a scene after identity application is cancelled", async () => {
    const onClose = vi.fn()
    renderDrawer({ onApply: async () => false, onClose })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await act(async () => { await Promise.resolve() })
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
  })

  it.each(["target", "unmount"])("suppresses held scene presentation after %s changes", async change => {
    let release!: (saved: boolean) => void
    const pending = new Promise<boolean>(resolve => { release = resolve })
    actorSettingsMocks.saveActorSettingsForChat.mockReturnValueOnce(pending)
    const onClose = vi.fn()
    const view = renderDrawer({ onApply: async () => true, onClose })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() => expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalled())
    const before = useActorStore.getState()
    if (change === "unmount") view.unmount()
    else useStoreMessageOption.setState({ serverChatId: "replacement", historyId: "other-history" })
    await act(async () => { release(true); await pending })
    expect(useActorStore.getState()).toBe(before)
    expect(onClose).not.toHaveBeenCalled()
  })

  it("keeps the scene draft and reports partial failure without rolling back the accepted identity", async () => {
    actorSettingsMocks.saveActorSettingsForChat.mockResolvedValueOnce(false)
    const onApply = vi.fn(async () => true)
    const onClose = vi.fn()
    renderDrawer({ onApply, onClose })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await screen.findByText(/Identity and behavior were applied, but scene settings could not be saved/)
    expect(onApply).toHaveBeenCalledTimes(1)
    expect(onClose).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
    expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledTimes(2)
  })

  it("does not publish a held scene after the drawer closes and reopens", async () => {
    let release!: (saved: boolean) => void
    const pending = new Promise<boolean>(resolve => { release = resolve })
    actorSettingsMocks.saveActorSettingsForChat.mockReturnValueOnce(pending)
    const onClose = vi.fn()
    const props = { beforeState, historyId: "history-1", serverChatId: "server-1", onClose, onApply: async () => true }
    const view = render(<RolePlaySetupDrawer {...props} open />)
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() => expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalled())
    view.rerender(<RolePlaySetupDrawer {...props} open={false} />)
    view.rerender(<RolePlaySetupDrawer {...props} open />)
    const before = useActorStore.getState()
    await act(async () => { release(true); await pending })
    expect(useActorStore.getState()).toBe(before)
    expect(onClose).not.toHaveBeenCalled()
  })

  it("respects an explicit owner action change while the accepted scene destination stays empty", async () => {
    useStoreMessageOption.setState({ serverChatId: null, historyId: null })
    let actionRevision = 0
    let release!: (saved: boolean) => void
    const pending = new Promise<boolean>(resolve => { release = resolve })
    actorSettingsMocks.saveActorSettingsForChat.mockReturnValueOnce(pending)
    const onClose = vi.fn()
    renderDrawer({ historyId: null, serverChatId: null, onApply: async () => true, onClose,
      captureApplyGuard: () => { const captured = actionRevision; return () => captured === actionRevision }
    })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() => expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalled())
    actionRevision++
    const before = useActorStore.getState()
    await act(async () => { release(true); await pending })
    expect(useActorStore.getState()).toBe(before)
    expect(onClose).not.toHaveBeenCalled()
  })

})


it("review: accepted detachment retains the edited scene through failed save and retry", async () => {
  useStoreMessageOption.setState({serverChatId:"server-1",historyId:"history-1"});
  actorSettingsMocks.getActorSettingsForChatWithCharacterFallback.mockReset().mockResolvedValueOnce(activeScene()).mockResolvedValue(createDefaultActorSettings());
  let release!: () => void;
  actorSettingsMocks.saveActorSettingsForChat.mockReset().mockImplementationOnce(() => new Promise<boolean>(resolve => {release=()=>resolve(false)})).mockResolvedValue(true);
  const onApply=vi.fn(async()=> {useStoreMessageOption.setState({serverChatId:null,historyId:null});return true});
  const onClose=vi.fn();
  const props={open:true,beforeState,historyId:"history-1" as string|null,serverChatId:"server-1" as string|null,onClose,onApply};
  const view=render(<RolePlaySetupDrawer {...props} />);
  await screen.findByText(/1 detail/);
  fireEvent.change(screen.getByLabelText("Scene notes"),{target:{value:"Keep my edited scene after detachment"}});
  fireEvent.click(screen.getByRole("button",{name:"Apply"}));
  await waitFor(()=>expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledTimes(1));
  view.rerender(<RolePlaySetupDrawer {...props} historyId={null} serverChatId={null} />);
  await act(async()=>{release();await new Promise(resolve=>setTimeout(resolve,30))});
  await screen.findByText(/Identity and behavior were applied, but scene settings could not be saved/);
  expect(screen.getByLabelText("Scene notes")).toHaveValue("Keep my edited scene after detachment");
  fireEvent.click(screen.getByRole("button",{name:"Apply"}));
  await waitFor(()=>expect(onClose).toHaveBeenCalledTimes(1));
  expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenLastCalledWith(expect.objectContaining({settings:expect.objectContaining({notes:"Keep my edited scene after detachment"})}));
  view.unmount();
});
