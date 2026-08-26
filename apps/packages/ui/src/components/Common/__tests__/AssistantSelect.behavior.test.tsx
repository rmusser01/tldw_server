import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  listAllCharacters: vi.fn(async () => []),
  listPersonaProfiles: vi.fn(async () => []),
  getPersonaProfile: vi.fn(async () => null),
  getCharacter: vi.fn(async () => null),
  selectedAssistant: {
    value: null as
      | null
      | {
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

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
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

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [
    mocks.selectedAssistant.value,
    mocks.setSelectedAssistant,
    { isLoading: false, setRenderValue: vi.fn() }
  ]
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector(state.option)
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: null,
    updateSettings: mocks.updateSettings,
    chatKey: "chat-overlay-1"
  })
}))

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

  const Tooltip = ({ children }: { children: React.ReactNode }) => <>{children}</>

  const Dropdown = ({
    open,
    onOpenChange,
    popupRender,
    children
  }: any) => {
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
  beforeEach(() => {
    vi.clearAllMocks()
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
      await screen.findByRole("textbox", { name: /search characters and personas/i }),
      "beta"
    )

    expect(await screen.findByRole("button", { name: "Beta" })).toBeInTheDocument()
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
    mocks.listAllCharacters.mockRejectedValueOnce(new Error("characters failed"))

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
    mocks.listPersonaProfiles.mockRejectedValueOnce(new Error("personas failed"))

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(await screen.findByRole("button", { name: "Alpha" })).toBeInTheDocument()
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

    await user.click(await screen.findByRole("button", { name: "Guide Persona" }))

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
      expect.objectContaining({ kind: "character", id: "char-1", name: "Alpha" })
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

    expect(await screen.findByRole("button", { name: "Alpha" })).toBeInTheDocument()

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
    expect(within(personaButton).getByRole("img", { name: "Guide Persona" })).toHaveAttribute(
      "src",
      "https://example.com/guide.png"
    )

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
      source: { kind: "persona" as const, id: "persona-1", name: "Guide Persona" },
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
  ])("supports $label from the assistant selector", async ({ source, targetTab, targetName, expected }) => {
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
  })

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
    mocks.listAllCharacters.mockResolvedValue([
      { id: "char-2", name: "Beta" }
    ])
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
    await user.click(
      await screen.findByRole("button", { name: "Beta" })
    )

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
    mocks.updateSettings.mockRejectedValueOnce(new Error("settings write failed"))

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
    expect(await screen.findByRole("button", { name: "Alpha" })).toBeInTheDocument()

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
    expect(await screen.findByRole("tab", { name: "Characters" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

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
    await user.click(
      await screen.findByRole("tab", { name: "Personas" })
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
    await user.click(
      await screen.findByRole("tab", { name: "Personas" })
    )
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
    expect(state.option.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(null)
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
