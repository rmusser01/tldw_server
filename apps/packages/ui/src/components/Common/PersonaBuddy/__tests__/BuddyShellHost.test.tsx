import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import {
  BuddyShellRenderContextProvider
} from "../BuddyShellRenderContext"
import { PERSONA_BUDDY_SHELL_ENABLED_SETTING } from "@/services/settings/ui-settings"
import {
  DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS,
  usePersonaBuddyShellStore
} from "@/store/persona-buddy-shell"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import type { PersonaBuddyRenderContext } from "@/types/persona-buddy"
import {
  asPersonaVisualCustomStateId,
  PERSONA_VISUAL_PACK_ACTIVATED_EVENT
} from "@/types/persona-visuals"
import { BuddyShellHost } from "../BuddyShellHost"

const mocks = vi.hoisted(() => ({
  isDesktop: true,
  selectedAssistant: null as Record<string, unknown> | null,
  buddyShellEnabled: true
}))

const capabilityMocks = vi.hoisted(() => ({
  state: {
    capabilities: {
      hasPersonaLiveControl: true
    },
    loading: false,
    error: null
  }
}))

const visualMocks = vi.hoisted(() => ({
  listPersonaVisualPacks: vi.fn(),
  getPersonaVisualPack: vi.fn()
}))

const liveControlMocks = vi.hoisted(() => ({
  calls: [] as unknown[],
  state: {
    sessions: [],
    focusedSessionId: null,
    focusedSession: null,
    loading: false,
    error: null,
    lastSendError: null,
    streamState: "closed",
    pendingFocusSessionId: null,
    canSendText: false,
    voiceAvailable: false,
    reload: vi.fn(),
    focusSession: vi.fn(),
    startTextSession: vi.fn(),
    stopSession: vi.fn(),
    sendText: vi.fn()
  } as Record<string, unknown>
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => mocks.isDesktop
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [
    mocks.selectedAssistant,
    vi.fn(),
    {
      isLoading: false,
      setRenderValue: vi.fn()
    }
  ]
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { key?: string; defaultValue: unknown }) => {
    if (setting?.key === PERSONA_BUDDY_SHELL_ENABLED_SETTING.key) {
      return [mocks.buddyShellEnabled, vi.fn(), { isLoading: false }]
    }
    return [setting.defaultValue, vi.fn(), { isLoading: false }]
  }
}))

vi.mock("@/services/persona-visuals", () => visualMocks)

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.state
}))

vi.mock("@/hooks/usePersonaLiveControl", () => ({
  usePersonaLiveControl: (options?: unknown) => {
    liveControlMocks.calls.push(options)
    return liveControlMocks.state
  }
}))

const buildPersonaSelection = ({
  id = "persona-1",
  hasBuddy = true
}: {
  id?: string
  hasBuddy?: boolean
} = {}) => ({
  kind: "persona",
  id,
  name: `Persona ${id}`,
  buddy_summary: hasBuddy
    ? {
        has_buddy: true,
        persona_name: `Persona ${id}`,
        role_summary: "Keeps the route on track",
        visual: {
          species_id: "owl",
          silhouette_id: "perch",
          palette_id: "dawn"
        }
      }
    : null
})

const buildBuddySummary = (id: string, hasBuddy = true) => ({
  has_buddy: hasBuddy,
  persona_name: `Persona ${id}`,
  role_summary: hasBuddy ? "Keeps the route on track" : null,
  visual: hasBuddy
    ? {
        species_id: "owl",
        silhouette_id: "perch",
        palette_id: "dawn"
      }
    : null
})

const buildVisualPack = (personaId = "persona-1") => ({
  id: "pack-1",
  persona_id: personaId,
  title: "Animated buddy",
  renderer_type: "sprite_frames" as const,
  status: "active" as const,
  manifest: {
    manifest_version: 1 as const,
    renderer_type: "sprite_frames" as const,
    states: {
      idle: { animation_id: "idle" },
      tool_running: { animation_id: "tool" }
    },
    animations: {
      idle: {
        frames: [{ asset_id: "idle-asset", duration_ms: 100 }]
      },
      tool: {
        frames: [{ asset_id: "tool-asset", duration_ms: 100 }]
      }
    }
  },
  assets_by_id: {
    "idle-asset": {
      id: "idle-asset",
      url: "/assets/idle.png",
      mime_type: "image/png",
      asset_role: "frame",
      width: 24,
      height: 24
    },
    "tool-asset": {
      id: "tool-asset",
      url: "/assets/tool.png",
      mime_type: "image/png",
      asset_role: "frame",
      width: 24,
      height: 24
    }
  }
})

const buildLiveSession = (overrides: Record<string, unknown> = {}) => ({
  sessionId: "live-session-1",
  personaId: "persona-1",
  personaName: "Live Research Buddy",
  lifecycle: "connected",
  status: "active",
  isFocused: true,
  focusedAt: "2026-05-20T12:00:00Z",
  focusGeneration: 1,
  lastActivityAt: "2026-05-20T12:01:00Z",
  pendingApprovalCount: 0,
  activeToolName: null,
  errorState: null,
  recoveryHint: null,
  suggestedVisualState: null,
  allowedActions: ["send_text_ws", "focus", "stop"],
  capabilities: {
    text: true,
    voice: false,
    browserMicrophoneRequired: false
  },
  ...overrides
})

const buildMovementVisualPack = (
  personaId = "persona-1",
  movementStates: Array<"moving_left" | "moving_right"> = [
    "moving_left",
    "moving_right"
  ]
) => {
  const basePack = buildVisualPack(personaId)
  const movementEntries = Object.fromEntries(
    movementStates.map((state) => [
      asPersonaVisualCustomStateId(state),
      { animation_id: `${state}-animation` }
    ])
  )
  const movementAnimations = Object.fromEntries(
    movementStates.map((state) => [
      `${state}-animation`,
      {
        frames: [{ asset_id: `${state}-asset`, duration_ms: 100 }]
      }
    ])
  )
  const movementAssets = Object.fromEntries(
    movementStates.map((state) => [
      `${state}-asset`,
      {
        id: `${state}-asset`,
        url: `/assets/${state}.png`,
        mime_type: "image/png",
        asset_role: "frame" as const,
        width: 24,
        height: 24
      }
    ])
  )
  const movementCatalog = Object.fromEntries(
    movementStates.map((state) => [
      asPersonaVisualCustomStateId(state),
      {
        label: state === "moving_left" ? "Moving left" : "Moving right",
        kind: "live_variant" as const
      }
    ])
  )

  return {
    ...basePack,
    manifest: {
      ...basePack.manifest,
      state_catalog: movementCatalog,
      states: {
        ...basePack.manifest.states,
        ...movementEntries
      },
      animations: {
        ...basePack.manifest.animations,
        ...movementAnimations
      }
    },
    assets_by_id: {
      ...basePack.assets_by_id,
      ...movementAssets
    }
  }
}

const mockDockRect = () =>
  vi.spyOn(HTMLDivElement.prototype, "getBoundingClientRect").mockReturnValue({
    x: 0,
    y: 0,
    left: 100,
    top: 100,
    right: 300,
    bottom: 220,
    width: 200,
    height: 120,
    toJSON: () => ({})
  } as DOMRect)

const dragBuddyBy = async (deltaX: number) => {
  const dragHandle = await screen.findByTestId("persona-buddy-drag-handle")
  fireEvent.pointerDown(dragHandle, {
    button: 0,
    pointerId: 1,
    clientX: 140,
    clientY: 130
  })
  fireEvent.pointerMove(window, {
    pointerId: 1,
    clientX: 140 + deltaX,
    clientY: 130
  })
}

const renderHost = ({
  root = "web",
  context,
  selectedAssistant = buildPersonaSelection(),
  isDesktop = true
}: {
  root?: "web" | "sidepanel"
  context?: {
    [K in keyof PersonaBuddyRenderContext]: PersonaBuddyRenderContext[K]
  }
  selectedAssistant?: Record<string, unknown> | null
  isDesktop?: boolean
} = {}) => {
  mocks.isDesktop = isDesktop
  mocks.selectedAssistant = selectedAssistant

  return render(
    <MemoryRouter>
      <BuddyShellRenderContextProvider initialContext={context ?? null}>
        <BuddyShellHost root={root} />
      </BuddyShellRenderContextProvider>
    </MemoryRouter>
  )
}

describe("BuddyShellHost", () => {
  beforeEach(() => {
    mocks.isDesktop = true
    mocks.selectedAssistant = null
    mocks.buddyShellEnabled = true
    capabilityMocks.state = {
      capabilities: {
        hasPersonaLiveControl: true
      },
      loading: false,
      error: null
    }
    visualMocks.listPersonaVisualPacks.mockReset()
    visualMocks.getPersonaVisualPack.mockReset()
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [],
      active_pack: null
    })
    visualMocks.getPersonaVisualPack.mockResolvedValue(null)
    liveControlMocks.state = {
      sessions: [],
      focusedSessionId: null,
      focusedSession: null,
      loading: false,
      error: null,
      lastSendError: null,
      streamState: "closed",
      pendingFocusSessionId: null,
      canSendText: false,
      voiceAvailable: false,
      reload: vi.fn(),
      focusSession: vi.fn(),
      startTextSession: vi.fn(),
      stopSession: vi.fn(),
      sendText: vi.fn()
    }
    liveControlMocks.calls = []
    document.body.innerHTML = ""
    const portalRoot = document.createElement("div")
    portalRoot.id = "tldw-portal-root"
    document.body.appendChild(portalRoot)
    localStorage.clear()
    usePersonaBuddyShellStore.setState({
      isOpen: false,
      positions: {
        "web-desktop": {
          ...DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["web-desktop"]
        },
        "sidepanel-desktop": {
          ...DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["sidepanel-desktop"]
        }
      }
    })
    usePersonaVisualRuntimeStore.setState({
      override: null,
      runtimeDiagnostics: null
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("stays dormant until the current surface explicitly activates buddy rendering", () => {
    renderHost({
      context: {
        surface_id: "chat",
        surface_active: false,
        active_persona_id: "persona-1",
        position_bucket: "web-desktop",
        persona_source: "route-local"
      }
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()
  })

  it("unmounts the shell when the global buddy setting is disabled", () => {
    mocks.buddyShellEnabled = false
    usePersonaBuddyShellStore.setState((state) => ({
      ...state,
      isOpen: true
    }))

    renderHost({
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "web-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      }
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(true)
  })

  it("suppresses the web host below the desktop breakpoint", () => {
    renderHost({
      isDesktop: false,
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "web-desktop",
        persona_source: "route-local"
      }
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()
  })

  it("allows the sidepanel host even when the viewport is narrow", () => {
    renderHost({
      root: "sidepanel",
      isDesktop: false,
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      }
    })

    expect(screen.getByTestId("persona-buddy-dock")).toBeInTheDocument()
  })

  it("requires the render-context persona match before using selected-assistant fallback", () => {
    const fallbackPersona = buildPersonaSelection({ id: "persona-1" })

    const firstRender = renderHost({
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: "persona-2",
        position_bucket: "web-desktop",
        persona_source: "route-local"
      },
      selectedAssistant: fallbackPersona
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()

    firstRender.unmount()
    mocks.selectedAssistant = fallbackPersona
    renderHost({
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: null,
        position_bucket: "web-desktop",
        persona_source: "selected-assistant-fallback"
      },
      selectedAssistant: fallbackPersona
    })

    expect(screen.getByTestId("persona-buddy-dock")).toBeInTheDocument()
  })

  it("does not use selected-assistant fallback for route-local surfaces without an explicit fallback source", () => {
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: null,
        position_bucket: "web-desktop",
        persona_source: "route-local"
      },
      selectedAssistant: buildPersonaSelection({ id: "persona-1" })
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()
  })

  it("ignores malformed persona selections that do not include a name", () => {
    renderHost({
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: null,
        position_bucket: "web-desktop",
        persona_source: "selected-assistant-fallback"
      },
      selectedAssistant: {
        kind: "persona",
        id: "persona-1"
      }
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()
  })

  it("prefers route-local buddy summary over stale selected-assistant persona data", () => {
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-2",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-2")
      },
      root: "sidepanel",
      selectedAssistant: buildPersonaSelection({ id: "persona-1" })
    })

    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent(
      "Persona persona-2"
    )
    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent("owl")
  })

  it("treats an explicit null surface summary as authoritative over cached assistant buddy data", () => {
    renderHost({
      context: {
        surface_id: "sidepanel-chat",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "catalog",
        buddy_summary: null
      },
      root: "sidepanel",
      selectedAssistant: buildPersonaSelection({ id: "persona-1" })
    })

    expect(screen.getByTestId("persona-buddy-dock")).toHaveAttribute(
      "data-dormant",
      "true"
    )
    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent(
      "buddy unavailable"
    )
    expect(screen.getByTestId("persona-buddy-dock")).not.toHaveTextContent("owl")
  })

  it("renders a dormant shell when the resolved persona has no buddy summary", () => {
    renderHost({
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "web-desktop",
        persona_source: "route-local",
        buddy_summary: null
      }
    })

    expect(screen.getByTestId("persona-buddy-dock")).toHaveAttribute(
      "data-dormant",
      "true"
    )
    expect(
      screen.queryByTestId("persona-buddy-popover")
    ).not.toBeInTheDocument()
  })

  it("clamps persisted positions back into the viewport after mount", async () => {
    const rectSpy = vi
      .spyOn(HTMLDivElement.prototype, "getBoundingClientRect")
      .mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 200,
        bottom: 120,
        width: 200,
        height: 120,
        toJSON: () => ({})
      } as DOMRect)

    const originalWidth = window.innerWidth
    const originalHeight = window.innerHeight

    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 320
    })
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      value: 240
    })

    usePersonaBuddyShellStore.setState((state) => ({
      ...state,
      positions: {
        ...state.positions,
        "web-desktop": {
          x: 9999,
          y: 9999
        }
      }
    }))

    renderHost({
      context: {
        surface_id: "chat",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "web-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      }
    })

    await waitFor(() => {
      expect(
        usePersonaBuddyShellStore.getState().positions["web-desktop"]
      ).toEqual({
        x: 104,
        y: 104
      })
    })

    rectSpy.mockRestore()
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: originalWidth
    })
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      value: originalHeight
    })
  })

  it("stays dormant when an active chat surface does not have a persona selected", () => {
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "sidepanel-chat",
        surface_active: true,
        active_persona_id: null,
        position_bucket: "sidepanel-desktop",
        persona_source: null
      },
      selectedAssistant: {
        kind: "character",
        id: "character-1",
        name: "Narrator"
      }
    })

    expect(screen.queryByTestId("persona-buddy-dock")).not.toBeInTheDocument()
  })

  it("loads and renders the active visual pack for the active persona", async () => {
    const visualPack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(visualMocks.listPersonaVisualPacks).toHaveBeenCalledWith("persona-1")
    })
    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "src",
        expect.stringContaining("/assets/idle.png")
      )
    })
  })

  it("reloads the active visual pack when the persona editor activates one", async () => {
    const visualPack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks
      .mockResolvedValue({
        packs: [visualPack],
        active_pack: visualPack
      })
      .mockResolvedValueOnce({
        packs: [],
        active_pack: null
      })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(visualMocks.listPersonaVisualPacks).toHaveBeenCalledWith("persona-1")
    })
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()

    fireEvent(
      window,
      new CustomEvent(PERSONA_VISUAL_PACK_ACTIVATED_EVENT, {
        detail: {
          personaId: "persona-1",
          packId: visualPack.id
        }
      })
    )

    await waitFor(() => {
      expect(visualMocks.listPersonaVisualPacks.mock.calls.length).toBeGreaterThanOrEqual(2)
    })
    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "src",
        expect.stringContaining("/assets/idle.png")
      )
    })
  })

  it("falls back and reports diagnostics for unsupported active renderer packs", async () => {
    const basePack = buildVisualPack("persona-1")
    const visualPack = {
      ...basePack,
      renderer_type: "live2d" as const,
      manifest: {
        ...basePack.manifest,
        renderer_type: "live2d" as const
      }
    }
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
        "Visual renderer is not supported here"
      )
    })
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "The Buddy runtime cannot render live2d packs yet."
    )
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent("Persona persona-1")
  })

  it("falls back when sprite-frame packs have assets but no resolvable frame asset", async () => {
    const visualPack = buildVisualPack("persona-1")
    const malformedPack = {
      ...visualPack,
      assets_by_id: {
        "other-asset": {
          id: "other-asset",
          url: "/assets/other.png",
          mime_type: "image/png",
          asset_role: "frame" as const,
          width: 24,
          height: 24
        }
      }
    }
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [malformedPack],
      active_pack: malformedPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
        "Visual asset is missing"
      )
    })
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "idle-asset"
    )
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent("Persona persona-1")
  })

  it("clears published visual diagnostics when the host unmounts", async () => {
    const visualPack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    const view = renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(
        usePersonaVisualRuntimeStore.getState().runtimeDiagnostics
      ).toEqual(
        expect.objectContaining({
          personaId: "persona-1",
          packId: "pack-1",
          packLoadStatus: "loaded",
          visualState: "idle"
        })
      )
    })

    view.unmount()

    expect(usePersonaVisualRuntimeStore.getState().runtimeDiagnostics).toBeNull()
  })

  it("links the active buddy popover to the persona Visuals workflow", () => {
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel"
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Toggle buddy for Persona persona-1" })
    )

    expect(
      screen.getByRole("link", { name: "Choose/Change Buddy" })
    ).toHaveAttribute("href", "/persona?persona_id=persona-1&tab=visuals")
  })

  it("shows a focused live session status in the dock", async () => {
    liveControlMocks.state = {
      ...liveControlMocks.state,
      focusedSessionId: "live-session-1",
      focusedSession: buildLiveSession(),
      sessions: [buildLiveSession()],
      streamState: "open",
      canSendText: true
    }

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel"
    })

    expect(screen.getByTestId("persona-buddy-live-status")).toHaveTextContent(
      "Connected"
    )
  })

  it("hides live controls when the server capability is unavailable", () => {
    capabilityMocks.state = {
      capabilities: {
        hasPersonaLiveControl: false
      },
      loading: false,
      error: null
    }
    liveControlMocks.state = {
      ...liveControlMocks.state,
      focusedSessionId: "live-session-1",
      focusedSession: buildLiveSession(),
      sessions: [buildLiveSession()],
      streamState: "open",
      canSendText: true
    }
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel"
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Toggle buddy for Persona persona-1" })
    )

    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument()
    expect(screen.queryByTestId("persona-buddy-text-input")).not.toBeInTheDocument()
    expect(liveControlMocks.calls.at(-1)).toMatchObject({ autoLoad: false })
  })

  it("passes route voice state into the Buddy live controls", () => {
    liveControlMocks.state = {
      ...liveControlMocks.state,
      focusedSessionId: "live-session-1",
      focusedSession: buildLiveSession({
        capabilities: {
          text: true,
          voice: true,
          browserMicrophoneRequired: true
        }
      }),
      sessions: [
        buildLiveSession({
          capabilities: {
            text: true,
            voice: true,
            browserMicrophoneRequired: true
          }
        })
      ],
      streamState: "open",
      canSendText: true,
      voiceAvailable: true
    }
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "listening"
      },
      root: "sidepanel"
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Toggle buddy for Persona persona-1" })
    )

    expect(screen.getByRole("link", { name: "Stop listening" })).toHaveAttribute(
      "href",
      "/persona?persona_id=persona-1&tab=live"
    )
  })

  it("keeps urgent badge visible while drag movement override is active", async () => {
    const rectSpy = mockDockRect()
    const visualPack = buildMovementVisualPack("persona-1", ["moving_right"])
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })
    liveControlMocks.state = {
      ...liveControlMocks.state,
      focusedSessionId: "live-session-1",
      focusedSession: buildLiveSession({ pendingApprovalCount: 3 }),
      sessions: [buildLiveSession({ pendingApprovalCount: 3 })],
      streamState: "open",
      canSendText: true
    }

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        live_session_id: "live-session-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "idle"
      )
    })
    await dragBuddyBy(48)

    expect(screen.getByTestId("persona-buddy-urgent-badge")).toHaveTextContent("3")
    expect(usePersonaVisualRuntimeStore.getState().override?.state).toBe(
      "moving_right"
    )
    rectSpy.mockRestore()
  })

  it("hides the Visuals workflow action when the buddy context has no persona id", () => {
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: null,
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel",
      selectedAssistant: null
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Toggle buddy for Persona persona-1" })
    )

    expect(
      screen.queryByRole("link", { name: "Choose/Change Buddy" })
    ).not.toBeInTheDocument()
  })

  it("maps active tool status into the tool_running visual state", async () => {
    const visualPack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "thinking",
        active_tool_status: "Running notes.search"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "tool_running"
      )
    })
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      expect.stringContaining("/assets/tool.png")
    )
  })

  it("renders custom visual states from exact active tool names", async () => {
    const basePack = buildVisualPack("persona-1")
    const customState = asPersonaVisualCustomStateId("tool.notes_search")
    const visualPack = {
      ...basePack,
      manifest: {
        ...basePack.manifest,
        state_catalog: {
          [customState]: {
            label: "Searching notes",
            kind: "tool_variant"
          }
        },
        states: {
          ...basePack.manifest.states,
          [customState]: { animation_id: "tool-notes-search" }
        },
        animations: {
          ...basePack.manifest.animations,
          "tool-notes-search": {
            frames: [{ asset_id: "tool-notes-search-asset", duration_ms: 100 }]
          }
        },
        authored_triggers: [
          {
            id: "notes-search",
            source: "tool_name",
            match: "notes.search",
            state: customState,
            duration_ms: 500,
            priority: 90
          }
        ]
      },
      assets_by_id: {
        ...basePack.assets_by_id,
        "tool-notes-search-asset": {
          id: "tool-notes-search-asset",
          url: "/assets/tool-notes-search.png",
          mime_type: "image/png",
          asset_role: "frame",
          width: 24,
          height: 24
        }
      }
    }
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })
    const context = {
      surface_id: "persona-garden",
      surface_active: true,
      active_persona_id: "persona-1",
      position_bucket: "sidepanel-desktop",
      persona_source: "route-local",
      buddy_summary: buildBuddySummary("persona-1"),
      live_voice_state: "thinking",
      active_tool_name: "notes.search",
      active_tool_status: "Searching notes"
    } as PersonaBuddyRenderContext

    renderHost({
      context,
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "tool.notes_search"
      )
    })
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      expect.stringContaining("/assets/tool-notes-search.png")
    )
  })

  it("renders custom visual states from active-pack runtime overrides", async () => {
    const basePack = buildVisualPack("persona-1")
    const customState = asPersonaVisualCustomStateId("tool.notes_search")
    const visualPack = {
      ...basePack,
      manifest: {
        ...basePack.manifest,
        state_catalog: {
          [customState]: {
            label: "Searching notes",
            kind: "tool_variant"
          }
        },
        states: {
          ...basePack.manifest.states,
          [customState]: { animation_id: "tool-notes-search" }
        },
        animations: {
          ...basePack.manifest.animations,
          "tool-notes-search": {
            frames: [{ asset_id: "tool-notes-search-asset", duration_ms: 100 }]
          }
        }
      },
      assets_by_id: {
        ...basePack.assets_by_id,
        "tool-notes-search-asset": {
          id: "tool-notes-search-asset",
          url: "/assets/tool-notes-search.png",
          mime_type: "image/png",
          asset_role: "frame",
          width: 24,
          height: 24
        }
      }
    }
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })
    usePersonaVisualRuntimeStore.getState().setOverride({
      personaId: "persona-1",
      sessionId: "session-1",
      state: customState,
      reason: "mcp_runtime.notes.search",
      expiresAt: Date.now() + 10_000
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        live_session_id: "session-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "thinking"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "tool.notes_search"
      )
    })
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "src",
      expect.stringContaining("/assets/tool-notes-search.png")
    )
  })

  it("sets a moving_right runtime override while dragging right when the pack declares it", async () => {
    const rectSpy = mockDockRect()
    const visualPack = buildMovementVisualPack("persona-1", ["moving_right"])
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        live_session_id: "session-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "idle"
      )
    })
    await dragBuddyBy(48)

    expect(usePersonaVisualRuntimeStore.getState().override).toEqual(
      expect.objectContaining({
        personaId: "persona-1",
        sessionId: "session-1",
        state: "moving_right",
        reason: "buddy_drag"
      })
    )
    rectSpy.mockRestore()
  })

  it("sets a moving_left runtime override while dragging left when the pack declares it", async () => {
    const rectSpy = mockDockRect()
    const visualPack = buildMovementVisualPack("persona-1", ["moving_left"])
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        live_session_id: null,
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "idle"
      )
    })
    await dragBuddyBy(-48)

    expect(usePersonaVisualRuntimeStore.getState().override).toEqual(
      expect.objectContaining({
        personaId: "persona-1",
        sessionId: null,
        state: "moving_left",
        reason: "buddy_drag"
      })
    )
    rectSpy.mockRestore()
  })

  it("clears the Buddy drag movement override on pointer release", async () => {
    const rectSpy = mockDockRect()
    const visualPack = buildMovementVisualPack("persona-1", ["moving_right"])
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "idle"
      )
    })
    await dragBuddyBy(48)
    expect(usePersonaVisualRuntimeStore.getState().override?.state).toBe(
      "moving_right"
    )

    fireEvent.pointerUp(window, { pointerId: 1 })

    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
    rectSpy.mockRestore()
  })

  it("keeps dock dragging without setting a movement override for packs without movement states", async () => {
    const rectSpy = mockDockRect()
    const visualPack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: visualPack
    })
    const initialPosition =
      usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "idle"
      )
    })
    await dragBuddyBy(48)

    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
    expect(
      usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    ).not.toEqual(initialPosition)
    rectSpy.mockRestore()
  })

  it("keeps derived buddy text when active visual pack loading fails", async () => {
    visualMocks.listPersonaVisualPacks.mockRejectedValue(new Error("offline"))

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(visualMocks.listPersonaVisualPacks).toHaveBeenCalledWith("persona-1")
    })
    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent("Persona persona-1")
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "Visual pack did not load"
    )
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "offline"
    )
    expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
  })

  it("reports missing visual assets while preserving the buddy text fallback", async () => {
    const visualPack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [visualPack],
      active_pack: {
        ...visualPack,
        assets_by_id: {
          "idle-asset": visualPack.assets_by_id["idle-asset"]
        }
      }
    })

    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        active_tool_status: "Running notes.search"
      },
      root: "sidepanel"
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
        "Visual asset is missing"
      )
    })
    expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent("Persona persona-1")
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "tool-asset"
    )
  })
})
