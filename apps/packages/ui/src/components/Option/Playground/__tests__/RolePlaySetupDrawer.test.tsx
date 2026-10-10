import { act, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useStoreMessageOption } from "@/store/option"
import { useActorStore } from "@/store/actor"
import { HistorySelectionContext, useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import type { RolePlayState } from "../role-play-state"
import {
  RolePlaySetupDrawer,
  type RolePlayApplyContext,
  type RolePlaySetupApplyPayload
} from "../RolePlaySetupDrawer"
import { summarizeRolePlayScene } from "../role-play-scene"
import type {
  ActorSceneRead,
  ActorSceneRecord
} from "@/services/actor-settings"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import {
  captureActorSettingsTarget,
  type ActorSettingsTarget
} from "@/utils/actor-settings-target"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"
import { installSettingsLocks } from "@/services/__tests__/chat-settings-fixture"
import { createStartupTemplateBundle } from "../startup-template-bundles"

const actorSettingsMocks = vi.hoisted(() => ({
  getActorSceneForChat:
    vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>(),
  saveActorSettingsForChat:
    vi.fn<
      typeof import("@/services/actor-settings").saveActorSettingsForChat
    >(),
  captureCurrentActorSceneTarget:
    vi.fn<
      typeof import("@/hooks/chat/actor-scene-source").captureCurrentActorSceneTarget
    >(),
  accountRevision: 0,
  values: new Map<string, unknown>(),
  get: vi.fn<(key: string) => Promise<unknown>>(),
  set: vi.fn<(key: string, value: unknown) => Promise<void>>(),
  retries: [] as Array<() => void>,
  closes: [] as Array<() => void>,
  discards: [] as Array<() => unknown>
}))

vi.mock("@/services/actor-settings", () => ({
  getActorSceneForChat: actorSettingsMocks.getActorSceneForChat,
  saveActorSettingsForChat: actorSettingsMocks.saveActorSettingsForChat
}))

// These legacy UI cases declare account/native-view acceptance at the qualifier
// boundary. The real typed address, source/session checks and H1 fences remain.
vi.mock("@/hooks/chat/actor-scene-source", () => ({
  captureCurrentActorSceneTarget:
    actorSettingsMocks.captureCurrentActorSceneTarget
}))
// Only the physical durable storage engine is doubled for the real service read.
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    get: actorSettingsMocks.get,
    set: actorSettingsMocks.set,
    getAll: async () => Object.fromEntries(actorSettingsMocks.values),
    watch: () => () => undefined,
    unwatch: () => undefined,
    remove: async (key: string) => {
      actorSettingsMocks.values.delete(key)
    }
  })
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

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  return {
    Modal: {
      useModal: () => {
        const [modal, holder] = actual.Modal.useModal()
        return [
          {
            ...modal,
            confirm: (config: Parameters<typeof modal.confirm>[0]) => {
              if (config.onOk) actorSettingsMocks.discards.push(config.onOk)
              return modal.confirm(config)
            }
          },
          holder
        ]
      }
    },
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
    }) => {
      if (open && onClose) actorSettingsMocks.closes.push(onClose)
      return open ? (
        <section
          role="dialog"
          aria-label={String(title || "Role-play setup")}
          onKeyDown={(event) => {
            if (event.key === "Escape") onClose?.()
          }}>
          <button type="button" onClick={onClose}>
            Close drawer
          </button>
          {children}
          {footer}
        </section>
      ) : null
    },
    Button: React.forwardRef<
      HTMLButtonElement,
      {
        children: React.ReactNode
        onClick?: () => void
        htmlType?: string
        "aria-label"?: string
      }
    >(({ children, onClick, htmlType, "aria-label": ariaLabel }, ref) => {
      if (children === "Retry" && onClick)
        actorSettingsMocks.retries.push(onClick)
      return (
        <button
          ref={ref}
          type={htmlType === "submit" ? "submit" : "button"}
          aria-label={ariaLabel}
          onClick={onClick}>
          {children}
        </button>
      )
    }),
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
          onChange={(event) =>
            onChange?.({ target: { checked: event.target.checked } })
          }
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
        }) => (
          <textarea aria-label={ariaLabel} value={value} onChange={onChange} />
        )
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
  }
})

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

const declaredSceneScope: ServicePromptRequestScope = {
  config: { serverUrl: "https://drawer-unit.example", authMode: "single-user" },
  userId: 1
}
const declaredNativeOwnerKey = "drawer-unit-native-owner"
const selectDeclaredChat = (historyId: string, serverChatId: string) => {
  useStoreMessageOption.setState({
    historyId,
    serverChatId,
    chatDraftAddress: {
      kind: "native",
      id: serverChatId,
      ownerKey: declaredNativeOwnerKey,
      scope: { type: "global" }
    },
    chatDraftRevision: useStoreMessageOption.getState().chatDraftRevision + 1
  })
}
const declaredSceneTarget = (
  commandIsCurrent: () => boolean = () => true
): ActorSettingsTarget | null => {
  const source = useStoreMessageOption.getState()
  const address = source.chatDraftAddress
  const accountRevision = actorSettingsMocks.accountRevision
  if (
    !address ||
    (address.kind === "native" && source.serverChatId !== address.id) ||
    (address.kind === "fresh" &&
      (source.serverChatId !== null ||
        (source.historyId !== null && source.historyId !== "temp")))
  )
    return null
  const accountIsCurrent = () =>
    actorSettingsMocks.accountRevision === accountRevision
  const isCurrent = () => {
    const current = useStoreMessageOption.getState()
    return (
      commandIsCurrent() &&
      accountIsCurrent() &&
      current.chatDraftAddress === address &&
      current.chatDraftRevision === source.chatDraftRevision &&
      current.historyId === source.historyId &&
      current.serverChatId === source.serverChatId
    )
  }
  return captureActorSettingsTarget({
    address,
    requestScope: declaredSceneScope,
    accountIsCurrent,
    isCurrent,
    owner:
      address.kind === "native"
        ? {
            kind: "native",
            conversation_id: address.id,
            owner_key: address.ownerKey,
            request_scope: declaredSceneScope,
            scope: address.scope,
            validate_lease: accountIsCurrent
          }
        : null,
    view:
      address.kind === "native"
        ? {
            owner_key: address.ownerKey,
            conversation_id: address.id,
            view_session_id: "declared-drawer-view",
            interpretation: { kind: "parent_graph_v1" },
            cursor: { kind: "empty" },
            selection_revision: 0
          }
        : null
  })
}
const actorRecord = (
  settings: ActorSettings = activeScene()
): ActorSceneRecord => ({
  value: structuredClone(settings),
  updatedAt: 41
})
const sceneRead = (
  target: ActorSettingsTarget,
  settings: ActorSettings = activeScene()
): ActorSceneRead => ({
  target,
  originalRecord: actorRecord(settings),
  settings: structuredClone(settings),
  origin: "scene",
  profileUnavailable: false
})
const originalSceneTarget = () => {
  const target =
    actorSettingsMocks.getActorSceneForChat.mock.calls[0]?.[0].target
  if (!target)
    throw new Error("The fixture must reach its original qualified Scene read")
  return target
}
type DeclaredDestination = Parameters<
  NonNullable<RolePlayApplyContext["onDestination"]>
>[0] & { sceneTarget: ActorSettingsTarget }
const issueFreshDestination = (
  original: ActorSettingsTarget
): DeclaredDestination => {
  const source = useStoreMessageOption.getState()
  const account = original.account
  if (!original.isCurrent() || !account?.isCurrent())
    throw new Error("Original declared owner retired")
  const address = freshChatDraftAddress(false)
  const revision = source.chatDraftRevision + 1
  // This is the real address acknowledgement, retaining exactly the issued value.
  if (
    !source.setChatDraftAddress(address, {
      sourceAddress: source.chatDraftAddress,
      expectedRevision: source.chatDraftRevision
    })
  )
    throw new Error("The fixture's own fresh address was not acknowledged")
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  const isCurrent = () => {
    const current = useStoreMessageOption.getState()
    return (
      account.isCurrent() &&
      current.chatDraftAddress === address &&
      current.chatDraftRevision === revision &&
      current.historyId === null &&
      current.serverChatId === null
    )
  }
  const sceneTarget = captureActorSettingsTarget({
    address,
    requestScope: account.requestScope,
    accountIsCurrent: account.isCurrent,
    isCurrent
  })
  if (
    !sceneTarget ||
    sceneTarget.accountKey !== original.accountKey ||
    !sceneTarget.isCurrent()
  )
    throw new Error("Missing exact acknowledged fresh Scene destination")
  return { historyId: null, serverChatId: null, isCurrent, sceneTarget }
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
    actorSettingsMocks.closes.length = 0
    actorSettingsMocks.discards.length = 0
    actorSettingsMocks.accountRevision = 0
    actorSettingsMocks.values.clear()
    actorSettingsMocks.get
      .mockReset()
      .mockImplementation(async (key) => actorSettingsMocks.values.get(key))
    actorSettingsMocks.set.mockReset().mockImplementation(async (key, value) => {
      actorSettingsMocks.values.set(key, structuredClone(value))
    })
    installSettingsLocks()
    useActorStore.getState().reset()
    selectDeclaredChat("history-1", "server-1")
    actorSettingsMocks.captureCurrentActorSceneTarget
      .mockReset()
      .mockImplementation((_controller, commandIsCurrent) =>
        declaredSceneTarget(commandIsCurrent)
      )
    actorSettingsMocks.getActorSceneForChat
      .mockReset()
      .mockImplementation(async ({ target }) => sceneRead(target))
    actorSettingsMocks.saveActorSettingsForChat
      .mockReset()
      .mockResolvedValue(true)
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("loads actor settings and renders before/after role-play state", async () => {
    renderDrawer()

    expect(
      await screen.findByRole("dialog", { name: "Role-play setup" })
    ).toBeInTheDocument()

    await waitFor(() =>
      expect(actorSettingsMocks.getActorSceneForChat).toHaveBeenCalledWith({
        target: originalSceneTarget(),
        characterId: "char-mira",
        validate_lease: expect.any(Function)
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
    let resolveActor!: (settings: ActorSceneRead) => void
    let target!: ActorSettingsTarget
    actorSettingsMocks.getActorSceneForChat.mockImplementationOnce((params) => {
      target = params.target
      return new Promise<ActorSceneRead>((resolve) => {
        resolveActor = resolve
      })
    })

    renderDrawer()

    expect(screen.getByRole("status")).toHaveTextContent(
      "Loading scene settings..."
    )

    resolveActor(sceneRead(target))
    await screen.findByText(/1 detail/)
  })

  it("retries a real shared scene read instead of replacing unread storage with defaults", async () => {
    const service = await vi.importActual<
      typeof import("@/services/actor-settings")
    >("@/services/actor-settings")
    const target = declaredSceneTarget()
    if (!target) throw new Error("Missing declared qualified storage owner")
    const ownedKey = `actorSettings:owned:${target.key}`
    actorSettingsMocks.values.set(ownedKey, actorRecord())
    actorSettingsMocks.get.mockRejectedValueOnce(new Error("read unavailable"))
    actorSettingsMocks.getActorSceneForChat.mockImplementation(
      service.getActorSceneForChat
    )
    const apply = vi.fn()
    renderDrawer({ onApply: apply })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Scene settings could not be loaded"
    )
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    expect(apply).not.toHaveBeenCalled()
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: /retry/i }))
    expect(await screen.findByLabelText("Scene notes")).toHaveValue(
      "The room smells like ozone."
    )
    expect(actorSettingsMocks.get).toHaveBeenNthCalledWith(1, ownedKey)
    expect(actorSettingsMocks.get).toHaveBeenNthCalledWith(2, ownedKey)
    expect(actorSettingsMocks.values.get(ownedKey)).toEqual(actorRecord())
    expect(actorSettingsMocks.set).not.toHaveBeenCalled()
  })

  it("reloads a held scene after actual same-ID H1 retirement", async () => {
    const selection = renderHook(() => useHistorySelection())
    let release!: (value: ActorSceneRead) => void
    let originalTarget!: ActorSettingsTarget
    actorSettingsMocks.getActorSceneForChat.mockImplementationOnce(
      ({ target }) => {
        originalTarget = target
        return new Promise<ActorSceneRead>((resolve) => {
          release = resolve
        })
      }
    )
    const captureGuard = () => selection.result.current.fence()
    const content = () => (
      <HistorySelectionContext.Provider value={selection.result.current}>
        <RolePlaySetupDrawer
          open
          beforeState={beforeState}
          historyId="history-1"
          serverChatId="server-1"
          characterId="char-mira"
          onApply={vi.fn()}
          onClose={vi.fn()}
          captureApplyGuard={captureGuard}
        />
      </HistorySelectionContext.Provider>
    )
    const view = render(content())
    await waitFor(() =>
      expect(actorSettingsMocks.getActorSceneForChat).toHaveBeenCalledTimes(1)
    )
    act(() => {
      selection.result.current.beginLoad()
    })
    view.rerender(content())
    await waitFor(() =>
      expect(actorSettingsMocks.getActorSceneForChat).toHaveBeenCalledTimes(2)
    )
    await act(async () =>
      release(
        sceneRead(originalTarget, { ...activeScene(), notes: "Retired scene." })
      )
    )
    await waitFor(() =>
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "The room smells like ozone."
      )
    )
  })

  it("ignores a failed previous scene's Retry after another chat has unsaved setup", async () => {
    actorSettingsMocks.getActorSceneForChat.mockRejectedValueOnce(
      new Error("unavailable")
    )
    const apply = vi.fn()
    const props = {
      open: true,
      beforeState,
      historyId: "history-1",
      serverChatId: "server-1",
      characterId: "char-mira",
      onApply: apply,
      onClose: vi.fn()
    }
    const view = render(<RolePlaySetupDrawer {...props} />)
    await screen.findByRole("alert")
    const oldRetry = actorSettingsMocks.retries.at(-1)!
    act(() => selectDeclaredChat("history-2", "server-2"))
    view.rerender(
      <RolePlaySetupDrawer
        {...props}
        historyId="history-2"
        serverChatId="server-2"
      />
    )
    await waitFor(() =>
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "The room smells like ozone."
      )
    )
    fireEvent.change(screen.getByLabelText("Scene notes"), {
      target: { value: "Unsaved setup." }
    })
    const reads = actorSettingsMocks.getActorSceneForChat.mock.calls.length
    await act(async () => oldRetry())
    expect(actorSettingsMocks.getActorSceneForChat).toHaveBeenCalledTimes(reads)
    expect(screen.getByLabelText("Scene notes")).toHaveValue("Unsaved setup.")
  })

  it("clears the previous Apply failure when the next owner is unavailable", async () => {
    actorSettingsMocks.saveActorSettingsForChat.mockResolvedValueOnce(false)
    const props = {
      open: true,
      beforeState,
      historyId: "history-1",
      serverChatId: "server-1",
      characterId: "char-mira",
      onApply: async () => true,
      onClose: vi.fn()
    }
    const view = render(<RolePlaySetupDrawer {...props} />)
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await screen.findByText(
      /Identity and behavior were applied, but scene settings could not be saved/
    )
    view.rerender(
      <RolePlaySetupDrawer
        {...props}
        serverChatId="other-chat"
        captureApplyGuard={() => () => false}
      />
    )
    await waitFor(() =>
      expect(
        screen.queryByText(
          /Identity and behavior were applied, but scene settings could not be saved/
        )
      ).not.toBeInTheDocument()
    )
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Scene settings could not be loaded"
    )
    expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledTimes(1)
  })

  it("reports an unavailable initiating owner without leaving the drawer loading", async () => {
    renderDrawer({ captureApplyGuard: () => () => false })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Scene settings could not be loaded"
    )
    expect(
      screen.queryByText("Loading scene settings...")
    ).not.toBeInTheDocument()
    expect(actorSettingsMocks.getActorSceneForChat).not.toHaveBeenCalled()
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    expect(
      screen.queryByRole("button", { name: "Retry" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(/check your account verification and selected chat/i)
    ).toBeInTheDocument()
  })

  it("does not crash when stored actor settings omit aspects", async () => {
    actorSettingsMocks.getActorSceneForChat.mockImplementationOnce(
      async ({ target }) =>
        sceneRead(target, {
          ...activeScene(),
          aspects: undefined
        } as unknown as ActorSettings)
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
    actorSettingsMocks.getActorSceneForChat.mockRejectedValueOnce(
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
      }),
      expect.objectContaining({
        isCurrent: expect.any(Function),
        onDestination: expect.any(Function)
      })
    )
    expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledWith({
      historyId: "history-1",
      serverChatId: "server-1",
      settings: expect.objectContaining({
        isEnabled: false,
        notes: ""
      }),
      target: originalSceneTarget(),
      expectedRecord: actorRecord(),
      validate_lease: expect.any(Function)
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
      }),
      expect.objectContaining({ isCurrent: expect.any(Function), onDestination: expect.any(Function) })
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
    fireEvent.click(
      screen.getByRole("button", { name: "Apply Mira detective scene" })
    )

    await waitFor(() =>
      expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledWith({
        historyId: "history-1",
        serverChatId: "server-1",
        settings: expect.objectContaining({
          isEnabled: true,
          notes: "The room smells like ozone."
        }),
        target: originalSceneTarget(),
        expectedRecord: actorRecord(),
        validate_lease: expect.any(Function)
      })
    )
    expect(onApplySavedSetup).toHaveBeenCalledWith(
      savedSetup,
      expect.objectContaining({
        isCurrent: expect.any(Function),
        onDestination: expect.any(Function)
      })
    )
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
    const pending = new Promise<boolean>((resolve) => {
      release = resolve
    })
    let destination!: DeclaredDestination
    const onApply = vi.fn(
      (_payload: RolePlaySetupApplyPayload, context?: RolePlayApplyContext) => {
        destination = issueFreshDestination(originalSceneTarget())
        context?.onDestination?.(destination)
        return pending
      }
    )
    renderDrawer({ onApply })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    await act(async () => {
      release(true)
      await pending
    })
    expect(
      actorSettingsMocks.saveActorSettingsForChat
    ).toHaveBeenCalledExactlyOnceWith({
      historyId: null,
      serverChatId: null,
      settings: activeScene(),
      target: destination.sceneTarget,
      expectedRecord: null,
      validate_lease: expect.any(Function)
    })
    expect(destination.sceneTarget.accountKey).toBe(
      chatDraftOwnerKey(declaredSceneScope)
    )
    expect(useActorStore.getState().applied?.target).toBe(destination.sceneTarget)
    expect(destination.sceneTarget.isCurrent()).toBe(true)
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
    expect(onApply).toHaveBeenCalledOnce()
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
    useStoreMessageOption.getState().startNewChatDraft(false)
    let actionRevision = 0
    let release!: (saved: boolean) => void
    const pending = new Promise<boolean>((resolve) => {
      release = resolve
    })
    actorSettingsMocks.saveActorSettingsForChat.mockReturnValueOnce(pending)
    const onClose = vi.fn()
    renderDrawer({
      historyId: null,
      serverChatId: null,
      onApply: async () => true,
      onClose,
      captureApplyGuard: () => {
        const captured = actionRevision
        return () => captured === actionRevision
      }
    })
    await screen.findByText(/1 detail/)
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() =>
      expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalled()
    )
    actionRevision++
    const before = useActorStore.getState()
    await act(async () => {
      release(true)
      await pending
    })
    expect(useActorStore.getState()).toBe(before)
    expect(onClose).not.toHaveBeenCalled()
  })

  it("accepted detachment retains the edited scene through failed save and retry", async () => {
    useStoreMessageOption.setState({
      serverChatId: "server-1",
      historyId: "history-1"
    })
    actorSettingsMocks.getActorSceneForChat
      .mockReset()
      .mockImplementationOnce(async ({ target }) => sceneRead(target))
      .mockImplementation(async ({ target }) =>
        sceneRead(target, createDefaultActorSettings())
      )
    let release!: () => void
    actorSettingsMocks.saveActorSettingsForChat
      .mockReset()
      .mockImplementationOnce(
        () =>
          new Promise<boolean>((resolve) => {
            release = () => resolve(false)
          })
      )
      .mockResolvedValue(true)
    let destination!: DeclaredDestination
    const onApply = vi.fn(
      async (
        _payload: RolePlaySetupApplyPayload,
        context?: RolePlayApplyContext
      ) => {
        destination = issueFreshDestination(originalSceneTarget())
        context?.onDestination?.(destination)
        return true
      }
    )
    const onClose = vi.fn()
    const props = {
      open: true,
      beforeState,
      historyId: "history-1" as string | null,
      serverChatId: "server-1" as string | null,
      onClose,
      onApply
    }
    const view = render(<RolePlaySetupDrawer {...props} />)
    await screen.findByText(/1 detail/)
    fireEvent.change(screen.getByLabelText("Scene notes"), {
      target: { value: "Keep my edited scene after detachment" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() =>
      expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledTimes(1)
    )
    view.rerender(
      <RolePlaySetupDrawer {...props} historyId={null} serverChatId={null} />
    )
    await act(async () => {
      release()
      await new Promise((resolve) => setTimeout(resolve, 30))
    })
    await screen.findByText(
      /Identity and behavior were applied, but scene settings could not be saved/
    )
    expect(screen.getByLabelText("Scene notes")).toHaveValue(
      "Keep my edited scene after detachment"
    )
    fireEvent.click(screen.getByRole("button", { name: "Apply" }))
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
    expect(onApply).toHaveBeenCalledOnce()
    expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenLastCalledWith(
      expect.objectContaining({
        settings: expect.objectContaining({
          notes: "Keep my edited scene after detachment"
        }),
        target: destination.sceneTarget,
        expectedRecord: null,
        validate_lease: expect.any(Function)
      })
    )
    expect(
      actorSettingsMocks.saveActorSettingsForChat.mock.calls[0][0].target
    ).toBe(destination.sceneTarget)
    expect(
      actorSettingsMocks.saveActorSettingsForChat.mock.calls[0][0].expectedRecord
    ).toBeNull()
    view.unmount()
  })

  it.each(["replacement", "reopen"])(
    "loads the new scene normally after %s during an own-detach save",
    async (change) => {
      actorSettingsMocks.getActorSceneForChat
        .mockReset()
        .mockImplementationOnce(async ({ target }) => sceneRead(target))
        .mockImplementation(async ({ target }) =>
          sceneRead(target, {
            ...createDefaultActorSettings(),
            notes: "New context scene"
          })
        )
      let release!: (saved: boolean) => void
      const pending = new Promise<boolean>((resolve) => {
        release = resolve
      })
      actorSettingsMocks.saveActorSettingsForChat.mockReturnValueOnce(pending)
      const onApply = async (
        _payload: RolePlaySetupApplyPayload,
        context?: RolePlayApplyContext
      ) => {
        context?.onDestination?.(issueFreshDestination(originalSceneTarget()))
        return true
      }
      const props = { beforeState, onApply, onClose: vi.fn() }
      const view = render(
        <RolePlaySetupDrawer
          {...props}
          open
          historyId="history-1"
          serverChatId="server-1"
        />
      )
      await screen.findByText(/1 detail/)
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Pending old scene" }
      })
      fireEvent.click(screen.getByRole("button", { name: "Apply" }))
      await waitFor(() =>
        expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalled()
      )
      view.rerender(
        <RolePlaySetupDrawer
          {...props}
          open
          historyId={null}
          serverChatId={null}
        />
      )
      if (change === "replacement") {
        selectDeclaredChat("other-history", "other-server")
        view.rerender(
          <RolePlaySetupDrawer
            {...props}
            open
            historyId="other-history"
            serverChatId="other-server"
          />
        )
      } else {
        view.rerender(
          <RolePlaySetupDrawer
            {...props}
            open={false}
            historyId={null}
            serverChatId={null}
          />
        )
        view.rerender(
          <RolePlaySetupDrawer
            {...props}
            open
            historyId={null}
            serverChatId={null}
          />
        )
      }
      await waitFor(() =>
        expect(screen.getByLabelText("Scene notes")).toHaveValue(
          "New context scene"
        )
      )
      expect(actorSettingsMocks.getActorSceneForChat).toHaveBeenCalledTimes(2)
      await act(async () => {
        release(false)
        await pending
      })
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "New context scene"
      )
      expect(props.onClose).not.toHaveBeenCalled()
    }
  )

  it("stops the accepted scene publication tail when its actor write retires the owner", async () => {
    let current = true
    const onClose = vi.fn()
    const guard = () => () => current
    renderDrawer({ onApply: async () => true, captureApplyGuard: guard, onClose })
    await screen.findByText(/1 detail/)
    const publications: ReturnType<typeof useActorStore.getState>[] = []
    const unsubscribe = useActorStore.subscribe((state) => {
      publications.push(state)
      if (state.settings) current = false
    })
    try {
      fireEvent.click(screen.getByRole("button", { name: "Apply" }))
      await waitFor(() =>
        expect(actorSettingsMocks.saveActorSettingsForChat).toHaveBeenCalledOnce()
      )
      await act(async () => {
        await Promise.resolve()
      })
      const preview = summarizeRolePlayScene(activeScene())
      const published = useActorStore.getState()
      expect(onClose).not.toHaveBeenCalled()
      expect(publications).toHaveLength(1)
      expect(published.applied).toEqual({
        target: originalSceneTarget(),
        settings: activeScene(),
        preview: preview.prompt,
        tokenCount: preview.tokenCount
      })
      expect(published.settings).toBe(published.applied?.settings)
      expect(published.preview).toBe(published.applied?.preview)
      expect(published.tokenCount).toBe(published.applied?.tokenCount)
      expect(published.applied?.target.isCurrent()).toBe(true)
    } finally {
      unsubscribe()
    }
  })

  it("ignores a discard confirmation admitted for an older scene", async () => {
    const onClose = vi.fn()
    const props = {
      open: true,
      beforeState,
      historyId: "history-1",
      serverChatId: "server-1",
      onClose,
      onApply: async () => true
    }
    const view = render(<RolePlaySetupDrawer {...props} />)
    await screen.findByText(/1 detail/)
    fireEvent.change(screen.getByLabelText("Scene notes"), {
      target: { value: "Older staged edit" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Close drawer" }))
    const discard = await screen.findByRole("button", {
      name: "Discard",
      exact: true
    })
    act(() => selectDeclaredChat("history-2", "server-2"))
    view.rerender(
      <RolePlaySetupDrawer
        {...props}
        historyId="history-2"
        serverChatId="server-2"
      />
    )
    await waitFor(() =>
      expect(screen.getByLabelText("Scene notes")).toHaveValue(
        "The room smells like ozone."
      )
    )
    fireEvent.click(discard)
    expect(onClose).not.toHaveBeenCalled()
  })

  it.each(["new draft", "same-ID H1", "owner"])(
    "rejects retained dismissal before React commits a %s change",
    async (change) => {
      const selection = renderHook(() => useHistorySelection())
      let ownerRevision = 0
      const onClose = vi.fn()
      render(
        <HistorySelectionContext.Provider value={selection.result.current}>
          <RolePlaySetupDrawer
            open
            beforeState={beforeState}
            historyId="history-1"
            serverChatId="server-1"
            onClose={onClose}
            onApply={async () => true}
            captureApplyGuard={() => {
              const originalOwner = ownerRevision
              return () => originalOwner === ownerRevision
            }}
          />
        </HistorySelectionContext.Provider>
      )
      await screen.findByText(/1 detail/)
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Keep this source's staged scene" }
      })
      fireEvent.click(screen.getByRole("button", { name: "Close drawer" }))
      await screen.findByRole("button", { name: "Discard", exact: true })
      const discard = actorSettingsMocks.discards.at(-1)!
      act(() => {
        if (change === "new draft")
          useStoreMessageOption.getState().startNewChatDraft()
        else if (change === "same-ID H1") selection.result.current.reset()
        else ownerRevision++
        discard()
      })
      expect(onClose).not.toHaveBeenCalled()
    }
  )

  it.each(["new draft", "same-ID H1", "owner"])(
    "rejects retained header close before React commits a %s change",
    async (change) => {
      const selection = renderHook(() => useHistorySelection())
      let ownerRevision = 0
      const onClose = vi.fn()
      render(
        <HistorySelectionContext.Provider value={selection.result.current}>
          <RolePlaySetupDrawer
            open
            beforeState={beforeState}
            historyId="history-1"
            serverChatId="server-1"
            onClose={onClose}
            onApply={async () => true}
            captureApplyGuard={() => {
              const originalOwner = ownerRevision
              return () => originalOwner === ownerRevision
            }}
          />
        </HistorySelectionContext.Provider>
      )
      await screen.findByText(/1 detail/)
      fireEvent.change(screen.getByLabelText("Scene notes"), {
        target: { value: "Keep this source's staged scene" }
      })
      const close = actorSettingsMocks.closes.at(-1)!
      act(() => {
        if (change === "new draft")
          useStoreMessageOption.getState().startNewChatDraft()
        else if (change === "same-ID H1") selection.result.current.reset()
        else ownerRevision++
        close()
      })
      expect(actorSettingsMocks.discards).toHaveLength(0)
      expect(onClose).not.toHaveBeenCalled()
    }
  )

  it.each(["Close", "Escape"])(
    "allows %s on the current unavailable-owner error panel",
    async (action) => {
      const onClose = vi.fn()
      renderDrawer({ captureApplyGuard: () => () => false, onClose })
      await screen.findByRole("alert")
      if (action === "Close")
        fireEvent.click(screen.getByRole("button", { name: "Close drawer" }))
      else
        fireEvent.keyDown(
          screen.getByRole("dialog", { name: "Role-play setup" }),
          { key: "Escape" }
        )
      expect(onClose).toHaveBeenCalledOnce()
      expect(actorSettingsMocks.getActorSceneForChat).not.toHaveBeenCalled()
      expect(actorSettingsMocks.saveActorSettingsForChat).not.toHaveBeenCalled()
    }
  )

  it.each(["new draft", "same-ID H1"])(
    "refuses a retired unavailable-owner panel before %s renders",
    async (change) => {
      const selection = renderHook(() => useHistorySelection())
      const onClose = vi.fn()
      render(
        <HistorySelectionContext.Provider value={selection.result.current}>
          <RolePlaySetupDrawer
            open
            beforeState={beforeState}
            historyId="history-1"
            serverChatId="server-1"
            onClose={onClose}
            onApply={async () => true}
            captureApplyGuard={() => () => false}
          />
        </HistorySelectionContext.Provider>
      )
      await screen.findByRole("alert")
      const close = actorSettingsMocks.closes.at(-1)!
      act(() => {
        if (change === "new draft")
          useStoreMessageOption.getState().startNewChatDraft()
        else selection.result.current.reset()
        close()
      })
      expect(onClose).not.toHaveBeenCalled()
    }
  )

})
