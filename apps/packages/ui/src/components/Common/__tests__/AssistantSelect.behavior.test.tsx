import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  historySelection: null as Record<string, unknown> | null,
  realSelection: false,
  ownerReady: null as Promise<void> | null,
  ownerFailed: false,
  assistantMeta: { isLoading: false, assistantKey: "alice", isCurrent: () => true, setRenderValue: vi.fn() },
  listAllCharacters: vi.fn(async () => []),
  listPersonaProfiles: vi.fn(async () => []),
  getPersonaProfile: vi.fn(async () => null),
  getCharacter: vi.fn(async () => null),
  selectedAssistant: {
    value: null as null | {
      kind: "character" | "persona"
      id: string
      name: string
    }
  },
  setSelectedAssistant: vi.fn(async () => undefined),
  updateSettings: vi.fn(async () => null)
}))

const state = vi.hoisted(() => ({
  option: {
    historyId: "history-overlay-1",
    serverChatId: "chat-overlay-1",
    serverChatAssistantKind: null as string | null,
    serverChatAssistantId: null as string | null,
    serverChatCharacterId: null as string | null,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatAssistantKind: vi.fn(),
    setServerChatAssistantId: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
    setServerChatMetaLoaded: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock(
  "@plasmohq/storage",
  () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock(
  "@plasmohq/storage/hook",
  () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook")
)
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: async () => {
    await mocks.ownerReady
    if (mocks.ownerFailed) throw new Error("Account unavailable")
    return {
      scopeKey: "alice",
      userId: "alice",
      config: {
        serverUrl: "http://localhost:8000",
        authMode: "multi-user",
        authSource: "manual",
        orgId: 2
      }
    }
  }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => null),
    listAllCharacters: mocks.listAllCharacters,
    listPersonaProfiles: mocks.listPersonaProfiles,
    getPersonaProfile: mocks.getPersonaProfile,
    getCharacter: mocks.getCharacter
  }
}))

vi.mock("@/hooks/useSelectedAssistant", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/hooks/useSelectedAssistant")>()
  return {
    useSelectedAssistant: () =>
      mocks.realSelection
        ? actual.useSelectedAssistant()
        : [
            mocks.selectedAssistant.value,
            mocks.setSelectedAssistant,
            mocks.assistantMeta
          ]
  }
})

vi.mock("@/store/option", () => ({
  useStoreMessageOption: Object.assign(
    (selector: (state: Record<string, unknown>) => unknown) =>
      selector(state.option),
    { getState: () => state.option }
  )
}))

vi.mock("@/hooks/chat/useHistorySelection", async importOriginal => ({
  ...await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>(),
  useHistorySelectionContext: () => mocks.historySelection
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: null,
    updateSettings: mocks.updateSettings,
    chatKey: "chat-overlay-1"
  })
}))

vi.mock("@/hooks/useServerChatHistory", () => ({ useServerChatHistory: () => ({ data: [] }) }))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({ useSelectServerChat: () => vi.fn() }))

vi.mock("antd", async () => {
  const React = await import("react")

  const Input = React.forwardRef<HTMLInputElement, any>((props, ref) => (
    <input
      ref={ref}
      aria-label={props["aria-label"] ?? props.placeholder}
      value={props.value}
      defaultValue={props.defaultValue}
      onChange={props.onChange}
      onKeyDown={props.onKeyDown}
    />
  ))

  const Tooltip = ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  )

  const Dropdown = ({ open, onOpenChange, popupRender, children }: any) => {
    const containerRef = React.useRef<HTMLDivElement | null>(null)

    React.useEffect(() => {
      if (open) {
        onOpenChange?.(true)
      }
    }, [open, onOpenChange])

    React.useEffect(() => {
      if (!open) return
      const onMouseDown = (event: MouseEvent) => {
        if (!containerRef.current?.contains(event.target as Node)) {
          onOpenChange?.(false)
        }
      }
      const onKeyDown = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          onOpenChange?.(false)
        }
      }
      document.addEventListener("mousedown", onMouseDown)
      document.addEventListener("keydown", onKeyDown)
      return () => {
        document.removeEventListener("mousedown", onMouseDown)
        document.removeEventListener("keydown", onKeyDown)
      }
    }, [open, onOpenChange])

    return (
      <div ref={containerRef}>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? popupRender?.(null) : null}
      </div>
    )
  }

  return {
    Dropdown,
    Input,
    Tooltip
  }
})

import { AssistantSelect } from "../AssistantSelect"
import { CharacterControlsSheet } from "@/components/Sidepanel/Chat/CharacterControlsSheet"

const renderAssistantSelect = (
  props: React.ComponentProps<typeof AssistantSelect> = {}
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  const renderTree = (
    nextProps: React.ComponentProps<typeof AssistantSelect> = props
  ) => (
    <QueryClientProvider client={queryClient}>
      <button id="assistant-rail-trigger" type="button">
        Runtime rail trigger
      </button>
      <AssistantSelect variant="dropdown" {...nextProps} />
    </QueryClientProvider>
  )

  const result = render(renderTree())
  return {
    ...result,
    rerenderAssistantSelect: (
      nextProps: React.ComponentProps<typeof AssistantSelect>
    ) => result.rerender(renderTree(nextProps))
  }
}

describe("AssistantSelect behavior", () => {
  it("uses the actual compact sheet and picker without fallback writes after refusal or outer close", async () => {
    let decide!: (accepted: boolean) => void
    let current!: () => boolean
    const onTrackedSelection = vi.fn((_selection, isCurrent: () => boolean) => { current = isCurrent; return new Promise<boolean>(resolve => { decide = resolve }) })
    const close = vi.fn()
    const user = userEvent.setup()
    const mounted = render(<CharacterControlsSheet onTrackedSelection={onTrackedSelection} onRequestClose={close} />)
    await user.click(screen.getByRole("button", { name: "Start tracked character chat" }))
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    expect(onTrackedSelection).toHaveBeenCalledOnce()
    await act(async () => { decide(false) })
    expect(screen.getByTestId("tracked-assistant-picker")).toBeInTheDocument()
    expect(close).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Alpha" }))
    mounted.unmount()
    expect(current()).toBe(false)
    await act(async () => { decide(true) })
    expect(close).not.toHaveBeenCalled()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(state.option.setServerChatId).not.toHaveBeenCalled()
  })

  it.each(["persona", "overlay", "character"])("retires held compact selection when a newer %s intent opens", async intent => {
    let current!: () => boolean
    let decide!: (accepted: boolean) => void
    const onTrackedSelection = vi.fn((_selection, isCurrent: () => boolean) => { current = isCurrent; return new Promise<boolean>(resolve => { decide = resolve }) })
    const close = vi.fn()
    const user = userEvent.setup()
    render(<CharacterControlsSheet onTrackedSelection={onTrackedSelection} onRequestClose={close} />)
    await user.click(screen.getByRole("button", { name: "Start tracked character chat" }))
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    await user.click(screen.getByRole("button", { name: intent === "persona" ? "Start tracked persona chat" : intent === "character" ? "Start tracked character chat" : "Apply overlay" }))
    expect(current()).toBe(false)
    await act(async () => { decide(true) })
    expect(close).not.toHaveBeenCalled()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
  })

  it("allows the current account to choose while retired work is still settling", async () => {
    let aliceCurrent = true
    let resolveAlice!: (accepted: boolean) => void
    let resolveBob!: (accepted: boolean) => void
    const onTrackedSelection = vi.fn()
      .mockImplementationOnce(() => new Promise<boolean>(resolve => { resolveAlice = resolve }))
      .mockImplementationOnce(() => new Promise<boolean>(resolve => { resolveBob = resolve }))
    mocks.assistantMeta = { isLoading: false, assistantKey: "alice", isCurrent: () => aliceCurrent, setRenderValue: vi.fn() }
    const user = userEvent.setup()
    const mounted = renderAssistantSelect({ variant: "inline", onTrackedSelection })
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    aliceCurrent = false
    mocks.assistantMeta = { isLoading: false, assistantKey: "bob", isCurrent: () => true, setRenderValue: vi.fn() }
    mounted.rerenderAssistantSelect({ variant: "inline", onTrackedSelection })
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    expect(onTrackedSelection).toHaveBeenCalledTimes(2)
    await act(async () => { resolveAlice(true) })
    expect(screen.getByRole("button", { name: "Alpha" })).toBeDisabled()
    await act(async () => { resolveBob(false) })
    expect(screen.getByRole("button", { name: "Alpha" })).toBeEnabled()
  })

  it("shows account verification failure and retries verification before fetching private catalogs", async () => {
    mocks.realSelection = true
    mocks.ownerFailed = true
    const user = userEvent.setup()
    renderAssistantSelect()
    await user.click(screen.getByRole("button", { name: "Select character or persona" }))
    const retry = await screen.findByRole("button", { name: "Retry account verification" })
    expect(screen.queryByRole("status", { name: /loading character and persona catalogs/i })).toBeNull()
    expect(mocks.listAllCharacters).not.toHaveBeenCalled()
    mocks.ownerFailed = false
    await user.click(retry)
    expect(await screen.findByRole("button", { name: "Alpha" })).toBeEnabled()
  })

  it("retires an old account catalog and rejects its held response", async () => {
    let aliceCurrent = true
    let resolveAlice!: (rows: unknown[]) => void
    mocks.assistantMeta = { isLoading: false, assistantKey: "alice", isCurrent: () => aliceCurrent, setRenderValue: vi.fn() }
    mocks.listAllCharacters.mockImplementationOnce(() => new Promise(resolve => { resolveAlice = resolve }))
    const user = userEvent.setup()
    const mounted = renderAssistantSelect()
    await user.click(screen.getByRole("button", { name: "Select character or persona" }))
    await waitFor(() => expect(resolveAlice).toBeTypeOf("function"))
    aliceCurrent = false
    mocks.assistantMeta = { isLoading: false, assistantKey: "bob", isCurrent: () => true, setRenderValue: vi.fn() }
    mocks.listAllCharacters.mockResolvedValueOnce([{ id: "bob-card", name: "Bob card" }] as never)
    mounted.rerenderAssistantSelect({})
    expect(await screen.findByRole("button", { name: "Bob card" })).toBeInTheDocument()
    await act(async () => { resolveAlice([{ id: "alice-private", name: "Alice private card" }]) })
    expect(screen.queryByRole("button", { name: "Alice private card" })).toBeNull()
    expect(screen.getByRole("button", { name: "Bob card" })).toBeInTheDocument()
    mocks.assistantMeta = { isLoading: false, assistantKey: "alice", isCurrent: () => true, setRenderValue: vi.fn() }
  })

  it("owns Escape throughout the real popup before an enclosing disclosure can see it", async () => {
    const parentEscape = vi.fn()
    const user = userEvent.setup()
    render(
      <div
        onKeyDown={(event) => {
          if (event.key === "Escape") parentEscape()
        }}>
        <AssistantSelect variant="dropdown" />
      </div>
    )
    const trigger = screen.getByRole("button", {
      name: "Select character or persona"
    })
    await user.click(trigger)
    const tab = await screen.findByRole("tab", { name: "Characters" })
    tab.focus()
    await user.keyboard("{Escape}")
    expect(parentEscape).not.toHaveBeenCalled()
    expect(
      screen.queryByTestId("assistant-select-panel")
    ).not.toBeInTheDocument()
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("waits for a root tracked-selection decision before closing or replacing chat", async () => {
    let decide!: (accepted: boolean) => void
    const onTrackedSelection = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          decide = resolve
        })
    )
    const user = userEvent.setup()
    renderAssistantSelect({ onTrackedSelection })
    await user.click(
      screen.getByRole("button", { name: "Select character or persona" })
    )
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    expect(screen.getByTestId("assistant-select-menu")).toBeInTheDocument()
    expect(onTrackedSelection).toHaveBeenCalledOnce()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(state.option.setServerChatId).not.toHaveBeenCalled()
    await act(async () => {
      decide(false)
    })
    expect(screen.getByTestId("assistant-select-menu")).toBeInTheDocument()
  })
  it("does not close a reopened picker after an obsolete tracked decision", async () => {
    let decide!: (accepted: boolean) => void
    let callerIsCurrent!: () => boolean
    const onTrackedSelection = vi.fn((_selection, isCurrent) => {
      callerIsCurrent = isCurrent!
      return new Promise<boolean>((resolve) => {
        decide = resolve
      })
    })
    const user = userEvent.setup()
    renderAssistantSelect({ onTrackedSelection })
    const trigger = screen.getByRole("button", {
      name: "Select character or persona"
    })
    await user.click(trigger)
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    await user.click(trigger)
    await user.click(trigger)
    expect(callerIsCurrent()).toBe(false)
    await act(async () => {
      decide(true)
    })
    expect(screen.getByTestId("assistant-select-menu")).toBeInTheDocument()
  })
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    mocks.realSelection = false
    mocks.historySelection = null
    mocks.assistantMeta = { isLoading: false, assistantKey: "alice", isCurrent: () => true, setRenderValue: vi.fn() }
    mocks.ownerReady = null
    mocks.ownerFailed = false
    mocks.selectedAssistant.value = null
    state.option = {
      historyId: "history-overlay-1",
      serverChatId: "chat-overlay-1",
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatCharacterId: null,
      setHistoryId: vi.fn(),
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatCharacterId: vi.fn(),
      setServerChatAssistantKind: vi.fn(),
      setServerChatAssistantId: vi.fn(),
      setServerChatPersonaMemoryMode: vi.fn(),
      setServerChatMetaLoaded: vi.fn()
    }
    mocks.listAllCharacters.mockResolvedValue([
      { id: "char-1", name: "Alpha", system_prompt: "Summary prompt" },
      { id: "char-2", name: "Beta" }
    ])
    mocks.listPersonaProfiles.mockResolvedValue([
      { id: "persona-1", name: "Guide Persona" }
    ])
    mocks.getPersonaProfile.mockResolvedValue({
      id: "persona-1",
      name: "Guide Persona",
      avatar_url: "https://example.com/guide-full.png",
      system_prompt: "Persona full prompt"
    })
    mocks.getCharacter.mockResolvedValue({
      id: "char-1",
      name: "Alpha",
      avatar_url: "https://example.com/alpha-full.png",
      system_prompt: "Character full prompt"
    })
  })

  it("disables compact tracked persona launch with guidance while keeping tracked character selection", async () => {
    mocks.historySelection = { getCurrent: () => ({ owner: null }) }
    const onTrackedSelection = vi.fn().mockResolvedValue(true)
    const user = userEvent.setup()
    render(<CharacterControlsSheet onTrackedSelection={onTrackedSelection} />)
    const persona = screen.getByRole("button", { name: "Start tracked persona chat" })
    expect(persona).toBeDisabled()
    expect(persona).toHaveAccessibleDescription(/tracked persona chat.*unavailable.*overlay/i)
    await user.click(persona)
    expect(screen.queryByTestId("tracked-assistant-picker")).toBeNull()
    expect(onTrackedSelection).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Start tracked character chat" }))
    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    await waitFor(() => expect(onTrackedSelection).toHaveBeenCalledWith(expect.objectContaining({ kind: "character" }), expect.any(Function)))
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it.each(["character", "persona"])("does not offer the ordinary-chat overlay action for a tracked %s conversation", async kind => {
    mocks.historySelection = { getCurrent: () => ({ owner: { kind: "native" } }) }
    Object.assign(state.option, { serverChatAssistantKind: kind, serverChatAssistantId: "persona-1",
      serverChatCharacterId: kind === "character" ? "char-1" : null })
    const user = userEvent.setup()
    renderAssistantSelect()
    await user.click(screen.getByTestId("character-select"))
    await user.click(screen.getByRole("tab", { name: "Personas" }))
    expect(await screen.findByRole("button", { name: "Guide Persona" })).toBeDisabled()
    expect(screen.queryByRole("button", { name: "Use a persona prompt overlay" })).toBeNull()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it("offers an explicit public persona overlay choice without silently adopting tracked identity", async () => {
    mocks.historySelection = { getCurrent: () => ({ owner: null }) }
    const user = userEvent.setup()
    renderAssistantSelect()
    await user.click(screen.getByTestId("character-select"))
    await user.click(screen.getByRole("tab", { name: "Personas" }))
    const choice = await screen.findByRole("button", { name: "Guide Persona" })
    expect(choice).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Use a persona prompt overlay" }))
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(choice).toBeEnabled()
    await user.click(choice)
    await waitFor(() => expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(expect.objectContaining({
      kind: "persona", id: "persona-1", metadata: expect.objectContaining({ selectionMode: "overlay" })
    })))
    expect(mocks.updateSettings).toHaveBeenCalledWith({ assistantOverlay: expect.objectContaining({ kind: "persona", id: "persona-1" }) })
  })

  it.each(["idle", "local", "native"])("explains unavailable tracked persona in protected %s history without writes", async source => {
    mocks.historySelection = { getCurrent: () => ({ owner: source === "idle" ? null : { kind: source } }) }
    if (source === "idle") Object.assign(state.option, { historyId: null, serverChatId: null })
    if (source === "local") Object.assign(state.option, { historyId: "local-history", serverChatId: null })
    const onTrackedSelection = vi.fn()
    const user = userEvent.setup()
    renderAssistantSelect({ initialTab: "persona", onTrackedSelection })
    await user.click(screen.getByRole("button", { name: "Select character or persona" }))
    const choice = await screen.findByRole("button", { name: "Guide Persona" })
    expect(choice).toBeDisabled()
    expect(choice).toHaveAccessibleDescription(/tracked persona chat.*unavailable.*overlay/i)
    await user.click(choice)
    expect(onTrackedSelection).not.toHaveBeenCalled()
    expect(mocks.getPersonaProfile).not.toHaveBeenCalled()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
    expect(state.option.setHistoryId).not.toHaveBeenCalled()
    expect(state.option.setServerChatId).not.toHaveBeenCalled()
  })

  it.each(["event", "preference"])("retires a held tracked character when an overlay intent opens the same protected picker by %s", async boundary => {
    mocks.historySelection = { getCurrent: () => ({ owner: null }) }
    let release!: (accepted: boolean) => void
    const pending = new Promise<boolean>(resolve => { release = resolve })
    let isCurrent!: () => boolean
    const tracked = vi.fn((_selection, guard: () => boolean) => { isCurrent = guard; return pending })
    const complete = vi.fn()
    const user = userEvent.setup()
    const view = renderAssistantSelect({ onTrackedSelection: tracked, onSelectionComplete: complete })
    try {
      await user.click(screen.getByRole("button", { name: "Select character or persona" }))
      await user.click(await screen.findByRole("button", { name: "Alpha" }))
      expect(isCurrent()).toBe(true)
      if (boundary === "event") act(() => window.dispatchEvent(new CustomEvent("tldw:open-assistant-select", {
        detail: { tab: "persona", applyAs: "overlay" }
      })))
      else view.rerenderAssistantSelect({ initialTab: "persona", selectionModePreference: "overlay", onTrackedSelection: tracked, onSelectionComplete: complete })
      expect(isCurrent()).toBe(false)
      await act(async () => { release(true); await pending })
      expect(complete).not.toHaveBeenCalled()
      const persona = await screen.findByRole("button", { name: "Guide Persona" })
      expect(persona).toBeEnabled()
      expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
      expect(mocks.updateSettings).not.toHaveBeenCalled()
      await user.click(persona)
      await waitFor(() => expect(complete).toHaveBeenCalledWith(expect.objectContaining({
        kind: "persona", metadata: expect.objectContaining({ selectionMode: "overlay" })
      })))
      expect(tracked).toHaveBeenCalledOnce()
      expect(state.option.setServerChatId).not.toHaveBeenCalled()
    } finally {
      await act(async () => { release(true); await pending })
      view.unmount()
    }
  })

  it("blocks protected persona staging and updates overlay intent while the same picker is open", async () => {
    mocks.historySelection = { getCurrent: () => ({ owner: null }) }
    const stage = vi.fn()
    const user = userEvent.setup()
    renderAssistantSelect({ initialTab: "persona", selection: null, onSelectionChange: stage })
    await user.click(screen.getByRole("button", { name: "Select character or persona" }))
    const choice = await screen.findByRole("button", { name: "Guide Persona" })
    expect(choice).toBeDisabled()
    await user.click(choice)
    expect(stage).not.toHaveBeenCalled()
    const requestMode = (applyAs: "tracked" | "overlay") => act(() => {
      window.dispatchEvent(new CustomEvent("tldw:open-assistant-select", { detail: { tab: "persona", applyAs } }))
    })
    requestMode("overlay")
    await waitFor(() => expect(choice).toBeEnabled())
    requestMode("tracked")
    await waitFor(() => expect(choice).toBeDisabled())
    requestMode("overlay")
    await waitFor(() => expect(choice).toBeEnabled())
    await user.click(choice)
    await waitFor(() => expect(stage).toHaveBeenCalledWith(expect.objectContaining({
      kind: "persona", metadata: expect.objectContaining({ selectionMode: "overlay" })
    })))
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
    expect(state.option.setServerChatId).not.toHaveBeenCalled()
  })

  it("waits for verified ownership before accepting a picker choice or replacing Chat", async () => {
    localStorage.clear()
    mocks.realSelection = true
    let release!: () => void
    mocks.ownerReady = new Promise<void>((resolve) => {
      release = resolve
    })
    const complete = vi.fn()
    const user = userEvent.setup()
    renderAssistantSelect({ onSelectionComplete: complete })
    await user.click(
      screen.getByRole("button", { name: "Select character or persona" })
    )
    expect(screen.queryByRole("button", { name: "Alpha" })).toBeNull()
    expect(mocks.listAllCharacters).not.toHaveBeenCalled()
    expect(state.option.setServerChatId).not.toHaveBeenCalled()
    expect(complete).not.toHaveBeenCalled()
    await act(async () => {
      release()
    })
    const choice = await screen.findByRole("button", { name: "Alpha" })
    await waitFor(() => expect(choice).toBeEnabled())
    await user.click(choice)
    await waitFor(() =>
      expect(complete).toHaveBeenCalledWith(
        expect.objectContaining({ id: "char-1" })
      )
    )
  })

  it.each([
    "javascript:alert(1)",
    "data:image/svg+xml,<svg onload='alert(1)'/>"
  ])("does not render an unsafe assistant avatar: %s", async (avatar_url) => {
    mocks.listAllCharacters.mockResolvedValue([
      { id: "char-1", name: "Alpha", avatar_url }
    ])
    renderAssistantSelect({ variant: "inline" })

    await screen.findByRole("button", { name: "Alpha" })
    expect(screen.queryByRole("img", { name: "Alpha" })).not.toBeInTheDocument()
  })

  it("renders an allowed assistant avatar", async () => {
    mocks.listAllCharacters.mockResolvedValue([
      {
        id: "char-1",
        name: "Alpha",
        avatar_url: "https://example.com/alpha.png"
      }
    ])
    renderAssistantSelect({ variant: "inline" })

    expect(await screen.findByRole("img", { name: "Alpha" })).toHaveAttribute(
      "src",
      "https://example.com/alpha.png"
    )
  })

  it("stages a controlled selection without changing the active chat or stored assistant", async () => {
    const user = userEvent.setup()
    const onSelectionChange = vi.fn()
    state.option.serverChatAssistantKind = "character"
    state.option.serverChatCharacterId = "char-1"
    mocks.selectedAssistant.value = {
      kind: "character",
      id: "char-1",
      name: "Alpha"
    }
    const { rerenderAssistantSelect } = renderAssistantSelect({
      selection: null,
      onSelectionChange
    })

    await user.click(
      screen.getByRole("button", { name: "Select character or persona" })
    )
    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    expect(onSelectionChange).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        metadata: { selectionMode: "tracked" }
      })
    )
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
    expect(state.option.setHistoryId).not.toHaveBeenCalled()
    expect(state.option.setMessages).not.toHaveBeenCalled()
    rerenderAssistantSelect({
      selection: { kind: "persona", id: "persona-1", name: "Guide Persona" },
      onSelectionChange
    })
    expect(screen.getByTestId("character-select")).toHaveAccessibleName(
      "Guide Persona"
    )
  })

  it("does not load dropdown catalogs until the selector opens", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await act(async () => {
      await Promise.resolve()
    })

    expect(mocks.listAllCharacters).not.toHaveBeenCalled()
    expect(mocks.listPersonaProfiles).not.toHaveBeenCalled()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await waitFor(() => {
      expect(mocks.listAllCharacters).toHaveBeenCalledTimes(1)
      expect(mocks.listPersonaProfiles).toHaveBeenCalledTimes(1)
    })
  })

  it("loads catalogs immediately for an inline selector", async () => {
    renderAssistantSelect({ variant: "inline" })

    await waitFor(() => {
      expect(mocks.listAllCharacters).toHaveBeenCalledTimes(1)
      expect(mocks.listPersonaProfiles).toHaveBeenCalledTimes(1)
    })
  })

  it("opens a searchable menu and filters visible characters", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.type(
      await screen.findByRole("textbox", {
        name: /search characters and personas/i
      }),
      "beta"
    )

    expect(
      await screen.findByRole("button", { name: "Beta" })
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Alpha" })).toBeNull()
  })

  it("announces character and persona catalog loading in the selector", async () => {
    mocks.listAllCharacters.mockReturnValue(new Promise(() => {}))
    mocks.listPersonaProfiles.mockReturnValue(new Promise(() => {}))

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    const status = await screen.findByRole("status", {
      name: /loading character and persona catalogs/i
    })
    expect(status).toHaveTextContent("Loading characters and personas")
  })

  it("shows a retryable character catalog failure instead of an empty list", async () => {
    mocks.listAllCharacters.mockRejectedValueOnce(
      new Error("characters failed")
    )

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(
      await screen.findByText(/could not load characters/i)
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /retry characters/i }))

    expect(mocks.listAllCharacters).toHaveBeenCalledTimes(2)
  })

  it("keeps loaded characters usable when persona catalog loading fails", async () => {
    mocks.listPersonaProfiles.mockRejectedValueOnce(
      new Error("personas failed")
    )

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(
      await screen.findByRole("button", { name: "Alpha" })
    ).toBeInTheDocument()
    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    expect(
      await screen.findByText(/could not load personas/i)
    ).toBeInTheDocument()
  })

  it("does not select a character when its favorite star is clicked", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(
      await screen.findByRole("button", { name: /add beta to favorites/i })
    )

    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
  })

  it("opens the character tab when character chat requests character selection", async () => {
    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: { tab: "character", source: "chat-header" }
      })
    )

    expect(
      await screen.findByRole("button", { name: "Alpha" })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Characters" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
  })

  it("finishes the explicit selector return in its qualified frame without stealing later task focus", async () => {
    renderAssistantSelect()
    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )
    const tab = await screen.findByRole("tab", { name: "Characters" })
    tab.focus()
    const callbacks: FrameRequestCallback[] = []
    const frame = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback) => {
        callbacks.push(callback)
        return callbacks.length
      })
    try {
      fireEvent.keyDown(tab, { key: "Escape" })
      act(() => {
        callbacks.shift()?.(0)
      })
      const other = document.createElement("button")
      document.body.appendChild(other)
      try {
        other.focus()
        act(() => {
          while (callbacks.length) callbacks.shift()?.(0)
        })
        expect(other).toHaveFocus()
      } finally {
        other.remove()
      }
    } finally {
      frame.mockRestore()
    }
  })

  it.each(["unmount", "reopen"] as const)(
    "retires an explicit selector return after %s",
    async (transition) => {
      const view = renderAssistantSelect()
      const intent = () =>
        window.dispatchEvent(
          new CustomEvent("tldw:open-assistant-select", {
            detail: {
              tab: "character",
              returnFocusSelector: "#assistant-rail-trigger"
            }
          })
        )
      intent()
      const tab = await screen.findByRole("tab", { name: "Characters" })
      tab.focus()
      const callbacks: FrameRequestCallback[] = []
      const frame = vi
        .spyOn(window, "requestAnimationFrame")
        .mockImplementation((callback) => {
          callbacks.push(callback)
          return callbacks.length
        })
      const cancel = vi
        .spyOn(window, "cancelAnimationFrame")
        .mockImplementation(() => {})
      try {
        fireEvent.keyDown(tab, { key: "Escape" })
        if (transition === "unmount") view.unmount()
        else
          act(() => {
            intent()
          })
        const replacement = document.createElement("button")
        replacement.id = "assistant-rail-trigger"
        if (transition === "unmount") document.body.appendChild(replacement)
        if (document.activeElement instanceof HTMLElement) {
          document.activeElement.blur()
        }
        try {
          act(() => {
            while (callbacks.length) callbacks.shift()?.(0)
          })
          if (transition === "unmount") expect(document.body).toHaveFocus()
          else
            expect(
              screen.getByRole("textbox", {
                name: "Search characters and personas"
              })
            ).toHaveFocus()
        } finally {
          replacement.remove()
        }
      } finally {
        frame.mockRestore()
        cancel.mockRestore()
      }
    }
  )

  it("returns focus to the requested rail trigger after selection", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()
    const railTrigger = screen.getByRole("button", {
      name: "Runtime rail trigger"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "persona",
          source: "playground-cockpit",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )

    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(document.activeElement).toBe(railTrigger)
    })
  })

  it("closes the menu and restores focus without waiting for selection persistence", async () => {
    const user = userEvent.setup()
    let resolveSelection: (() => void) | undefined
    mocks.setSelectedAssistant.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          resolveSelection = resolve
        })
    )

    renderAssistantSelect()
    const railTrigger = screen.getByRole("button", {
      name: "Runtime rail trigger"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          source: "playground-cockpit",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )

    await user.click(await screen.findByRole("button", { name: "Alpha" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-select-menu")).toBeNull()
      expect(document.activeElement).toBe(railTrigger)
    })
    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "character",
        id: "char-1",
        name: "Alpha"
      })
    )

    resolveSelection?.()
  })

  it("returns focus to the requested rail trigger after Escape closes the menu", async () => {
    renderAssistantSelect()
    const railTrigger = screen.getByRole("button", {
      name: "Runtime rail trigger"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          source: "playground-cockpit",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )

    expect(
      await screen.findByRole("button", { name: "Alpha" })
    ).toBeInTheDocument()

    fireEvent.keyDown(document, { key: "Escape" })

    await waitFor(() => {
      expect(document.activeElement).toBe(railTrigger)
    })
  })

  it("labels identity and optional scene choices without mixing character and persona concepts", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(
      screen.getByRole("tablist", { name: "Character or persona" })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Characters" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Personas" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Optional scene context" })
    ).toBeInTheDocument()
  })

  it("keeps personas accessible and dispatches the actor footer action", async () => {
    const user = userEvent.setup()
    const actorListener = vi.fn()
    window.addEventListener("tldw:open-actor-settings", actorListener)

    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    expect(
      await screen.findByRole("button", { name: "Guide Persona" })
    ).toBeInTheDocument()

    await user.click(
      await screen.findByRole("button", { name: "Optional scene context" })
    )

    expect(actorListener).toHaveBeenCalledTimes(1)
    window.removeEventListener("tldw:open-actor-settings", actorListener)
  })

  it("lists and selects personas with canonical buddy summary payloads", async () => {
    const user = userEvent.setup()
    mocks.listPersonaProfiles.mockResolvedValue([
      {
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: "https://example.com/guide.png",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Guide Persona",
          role_summary: "Keeps the chat on course",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "dawn"
          }
        }
      }
    ])

    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(await screen.findByRole("tab", { name: "Personas" }))

    const personaButton = await screen.findByRole("button", {
      name: "Guide Persona"
    })
    expect(
      within(personaButton).getByRole("img", { name: "Guide Persona" })
    ).toHaveAttribute("src", "https://example.com/guide.png")

    await user.click(personaButton)

    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: "https://example.com/guide.png",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Guide Persona",
          role_summary: "Keeps the chat on course",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "dawn"
          }
        }
      })
    )
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it.each([
    {
      label: "character to another character",
      source: { kind: "character" as const, id: "char-1", name: "Alpha" },
      targetTab: "Characters",
      targetName: "Beta",
      expected: { kind: "character", id: "char-2", name: "Beta" }
    },
    {
      label: "character to persona",
      source: { kind: "character" as const, id: "char-1", name: "Alpha" },
      targetTab: "Personas",
      targetName: "Guide Persona",
      expected: { kind: "persona", id: "persona-1", name: "Guide Persona" }
    },
    {
      label: "persona to character",
      source: {
        kind: "persona" as const,
        id: "persona-1",
        name: "Guide Persona"
      },
      targetTab: "Characters",
      targetName: "Alpha",
      expected: { kind: "character", id: "char-1", name: "Alpha" }
    },
    {
      label: "none to persona",
      source: null,
      targetTab: "Personas",
      targetName: "Guide Persona",
      expected: { kind: "persona", id: "persona-1", name: "Guide Persona" }
    },
    {
      label: "none to character",
      source: null,
      targetTab: "Characters",
      targetName: "Alpha",
      expected: { kind: "character", id: "char-1", name: "Alpha" }
    }
  ])(
    "supports $label from the assistant selector",
    async ({ source, targetTab, targetName, expected }) => {
      const user = userEvent.setup()
      mocks.selectedAssistant.value = source
      renderAssistantSelect()

      await user.click(
        await screen.findByRole("button", {
          name: source?.name ?? "Select character or persona"
        })
      )

      await user.click(await screen.findByRole("tab", { name: targetTab }))
      mocks.setSelectedAssistant.mockClear()
      await user.click(await screen.findByRole("button", { name: targetName }))

      expect(mocks.setSelectedAssistant).toHaveBeenCalledTimes(1)
      expect(mocks.setSelectedAssistant).toHaveBeenLastCalledWith(
        expect.objectContaining(expected)
      )
    }
  )

  it("moves a favorited character ahead of other characters and closes on escape", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(
      await screen.findByRole("button", { name: /add beta to favorites/i })
    )

    const menu = screen.getByTestId("assistant-select-menu")
    const characterButtons = within(menu).getAllByRole("button")
    expect(characterButtons[0]).toHaveTextContent("Beta")

    fireEvent.keyDown(document, { key: "Escape" })

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-select-menu")).toBeNull()
    })
  })

  it("resolves persona overlay snapshots from full persona detail on apply", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "persona",
          applyAs: "overlay",
          source: "playground-cockpit"
        }
      })
    )
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "persona",
          id: "persona-1",
          name: "Guide Persona",
          metadata: expect.objectContaining({
            selectionMode: "overlay"
          })
        })
      )
    })
    await waitFor(() => {
      expect(mocks.getPersonaProfile).toHaveBeenCalledWith("persona-1")
    })
    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: "https://example.com/guide-full.png",
        system_prompt_snapshot: "Persona full prompt"
      })
    })
  })

  it("refreshes character detail for overlay snapshots when summary prompt material is missing", async () => {
    const user = userEvent.setup()
    mocks.listAllCharacters.mockResolvedValue([{ id: "char-2", name: "Beta" }])
    mocks.getCharacter.mockResolvedValue({
      id: "char-2",
      name: "Beta",
      avatar_url: "https://example.com/beta-full.png",
      system_prompt: "Character fetched prompt"
    })

    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          applyAs: "overlay",
          source: "playground-cockpit"
        }
      })
    )
    await user.click(await screen.findByRole("button", { name: "Beta" }))

    await waitFor(() => {
      expect(mocks.getCharacter).toHaveBeenCalledWith("char-2", {
        forceRefresh: true
      })
    })
    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: expect.objectContaining({
        kind: "character",
        id: "char-2",
        name: "Beta",
        avatar_url: "https://example.com/beta-full.png",
        system_prompt_snapshot: "Character fetched prompt"
      })
    })
  })

  it("still applies persona selection when persona detail lookup fails", async () => {
    const user = userEvent.setup()
    mocks.getPersonaProfile.mockRejectedValueOnce(
      new Error("persona detail unavailable")
    )

    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "persona",
          applyAs: "overlay",
          source: "playground-cockpit"
        }
      })
    )
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "persona",
          id: "persona-1",
          name: "Guide Persona"
        })
      )
    })
    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        system_prompt_snapshot: null
      })
    })
  })

  it("still applies local selection when overlay settings persistence fails", async () => {
    const user = userEvent.setup()
    mocks.updateSettings.mockRejectedValueOnce(
      new Error("settings write failed")
    )

    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "persona",
          applyAs: "overlay",
          source: "playground-cockpit"
        }
      })
    )
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "persona",
          id: "persona-1",
          name: "Guide Persona"
        })
      )
    })
  })

  it("keeps event-supplied tracked mode when dropdown open change fires", async () => {
    const user = userEvent.setup()
    renderAssistantSelect({
      selectionModePreference: "overlay",
      labelOverride: "Apply assistant"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          applyAs: "tracked",
          source: "playground-cockpit"
        }
      })
    )

    await user.click(await screen.findByRole("button", { name: "Alpha" }))

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "character",
          id: "char-1",
          name: "Alpha",
          metadata: expect.objectContaining({
            selectionMode: "tracked"
          })
        })
      )
    })
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it("uses the latest prop-driven selection mode when choosing after rerender", async () => {
    const user = userEvent.setup()
    const { rerenderAssistantSelect } = renderAssistantSelect({
      selectionModePreference: "overlay",
      labelOverride: "Apply assistant"
    })

    await user.click(
      await screen.findByRole("button", { name: "Apply assistant" })
    )
    expect(
      await screen.findByRole("button", { name: "Alpha" })
    ).toBeInTheDocument()

    rerenderAssistantSelect({
      selectionModePreference: "tracked",
      labelOverride: "Apply assistant"
    })
    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "persona",
          id: "persona-1",
          name: "Guide Persona",
          metadata: expect.objectContaining({
            selectionMode: "tracked"
          })
        })
      )
    })
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it("switches the mounted picker to the latest requested initial tab", async () => {
    const user = userEvent.setup()
    const { rerenderAssistantSelect } = renderAssistantSelect({
      initialTab: "character",
      labelOverride: "Apply assistant"
    })

    await user.click(
      await screen.findByRole("button", { name: "Apply assistant" })
    )
    expect(
      await screen.findByRole("tab", { name: "Characters" })
    ).toHaveAttribute("aria-selected", "true")

    rerenderAssistantSelect({
      initialTab: "persona",
      labelOverride: "Apply assistant"
    })

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Personas" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })
  })

  it("uses the prop-driven overlay selection path and custom label without requiring an open event", async () => {
    const user = userEvent.setup()
    renderAssistantSelect({
      selectionModePreference: "overlay",
      labelOverride: "Apply overlay"
    })

    await user.click(
      await screen.findByRole("button", { name: "Apply overlay" })
    )
    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "persona",
          id: "persona-1",
          name: "Guide Persona"
        })
      )
    })
    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: "https://example.com/guide-full.png",
        system_prompt_snapshot: "Persona full prompt"
      })
    })
  })

  it("refuses overlay writes when the current chat is already tracked", async () => {
    const user = userEvent.setup()
    state.option = {
      historyId: "history-overlay-1",
      serverChatId: "chat-overlay-1",
      serverChatAssistantKind: "character",
      serverChatAssistantId: null,
      serverChatCharacterId: "char-9",
      setHistoryId: vi.fn(),
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatCharacterId: vi.fn(),
      setServerChatAssistantKind: vi.fn(),
      setServerChatAssistantId: vi.fn(),
      setServerChatPersonaMemoryMode: vi.fn(),
      setServerChatMetaLoaded: vi.fn()
    }

    renderAssistantSelect({
      selectionModePreference: "overlay",
      labelOverride: "Apply overlay"
    })

    await user.click(
      await screen.findByRole("button", { name: "Apply overlay" })
    )
    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    await user.click(
      await screen.findByRole("button", { name: "Guide Persona" })
    )

    await waitFor(() => {
      expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    })
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it("clears the active server chat when selecting a different tracked character", async () => {
    const user = userEvent.setup()
    state.option = {
      historyId: "history-miku",
      serverChatId: "chat-miku",
      serverChatAssistantKind: "character",
      serverChatAssistantId: "char-1",
      serverChatCharacterId: "char-1",
      setHistoryId: vi.fn(),
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatCharacterId: vi.fn(),
      setServerChatAssistantKind: vi.fn(),
      setServerChatAssistantId: vi.fn(),
      setServerChatPersonaMemoryMode: vi.fn(),
      setServerChatMetaLoaded: vi.fn()
    }
    mocks.selectedAssistant.value = {
      kind: "character",
      id: "char-1",
      name: "Alpha"
    }

    renderAssistantSelect()

    await user.click(await screen.findByRole("button", { name: "Alpha" }))
    await user.click(await screen.findByRole("button", { name: "Beta" }))

    expect(state.option.setHistoryId).toHaveBeenCalledWith(null, {
      preserveServerChatId: false
    })
    expect(state.option.setHistory).toHaveBeenCalledWith([])
    expect(state.option.setMessages).toHaveBeenCalledWith([])
    expect(state.option.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(state.option.setServerChatAssistantKind).toHaveBeenCalledWith(null)
    expect(state.option.setServerChatAssistantId).toHaveBeenCalledWith(null)
    expect(state.option.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(
      null
    )
    expect(state.option.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(state.option.setServerChatId).toHaveBeenCalledWith(null)
    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "character",
        id: "char-2",
        name: "Beta",
        metadata: expect.objectContaining({ selectionMode: "tracked" })
      })
    )
  })
})
