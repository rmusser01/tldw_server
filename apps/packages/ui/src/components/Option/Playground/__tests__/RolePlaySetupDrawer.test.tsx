import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import type { RolePlayState } from "../role-play-state"
import { RolePlaySetupDrawer } from "../RolePlaySetupDrawer"

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
    t: (key: string, fallback?: string) => fallback || key
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
    onClose
  }: {
    open?: boolean
    title?: React.ReactNode
    children: React.ReactNode
    onClose?: () => void
  }) =>
    open ? (
      <section role="dialog" aria-label={String(title || "Role-play setup")}>
        <button type="button" onClick={onClose}>
          Close drawer
        </button>
        {children}
      </section>
    ) : null,
  Button: ({
    children,
    onClick,
    htmlType
  }: {
    children: React.ReactNode
    onClick?: () => void
    htmlType?: string
  }) => (
    <button type={htmlType === "submit" ? "submit" : "button"} onClick={onClick}>
      {children}
    </button>
  ),
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
  render(
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
  return { returnFocusRef }
}

describe("RolePlaySetupDrawer", () => {
  beforeEach(() => {
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
    fireEvent.click(screen.getByRole("button", { name: "Precise" }))
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
    fireEvent.keyDown(document, { key: "Escape" })

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(returnFocusRef.current)
  })
})
