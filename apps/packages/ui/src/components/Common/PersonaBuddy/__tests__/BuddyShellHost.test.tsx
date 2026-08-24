import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import {
  BuddyShellRenderContextProvider,
  useSetBuddyShellRenderContext
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
  reducedMotion: false,
  selectedAssistant: null as Record<string, unknown> | null,
  buddyShellEnabled: true
}))

const companionMocks = vi.hoisted(() => ({
  calls: [] as unknown[],
  snapshot: {
    generation: 1,
    phase: "idle",
    actionToken: null as number | null,
    requestedState: null as string | null,
    facing: "right",
    transientOffsetX: 0,
    suspension: "none"
  },
  react: vi.fn(() => true),
  completeAction: vi.fn()
}))

const preferenceMocks = vi.hoisted(() => ({
  getBuddyPreferences: vi.fn(),
  getPersonaBuddyPreferences: vi.fn(),
  updateBuddyPreferences: vi.fn(),
  updatePersonaBuddyPreferences: vi.fn()
}))

const assetMocks = vi.hoisted(() => ({
  acquirePersonaVisualAsset: vi.fn()
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
  useDesktop: () => mocks.isDesktop,
  useMediaQuery: () => mocks.reducedMotion
}))

vi.mock("../usePersonaCompanion", () => ({
  usePersonaCompanion: (input: { semanticState?: string }) => {
    companionMocks.calls.push(input)
    return {
      snapshot: {
        ...companionMocks.snapshot,
        requestedState:
          companionMocks.snapshot.requestedState ?? input.semanticState ?? "idle"
      },
      react: companionMocks.react,
      completeAction: companionMocks.completeAction
    }
  }
}))

vi.mock("@/services/persona-buddy", () => preferenceMocks)

vi.mock("@/services/persona-visual-assets", () => assetMocks)

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
  const buddy = await screen.findByRole("button", { name: /Buddy for/i })
  fireEvent.pointerDown(buddy, {
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

const ContextDrivenHost: React.FC<{
  root: "web" | "sidepanel"
  context: PersonaBuddyRenderContext
}> = ({ root, context }) => {
  const setRenderContext = useSetBuddyShellRenderContext()
  React.useEffect(() => setRenderContext(context), [context, setRenderContext])
  return <BuddyShellHost root={root} />
}

const renderSwitchableHost = (
  context: PersonaBuddyRenderContext,
  root: "web" | "sidepanel" = "sidepanel"
) =>
  render(
    <MemoryRouter>
      <BuddyShellRenderContextProvider>
        <ContextDrivenHost root={root} context={context} />
      </BuddyShellRenderContextProvider>
    </MemoryRouter>
  )

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

const personaContext = (personaId: string): PersonaBuddyRenderContext => ({
  surface_id: "persona-garden",
  surface_active: true,
  active_persona_id: personaId,
  position_bucket: "sidepanel-desktop",
  persona_source: "route-local",
  buddy_summary: buildBuddySummary(personaId),
  live_voice_state: "idle"
})

describe("BuddyShellHost", () => {
  beforeEach(() => {
    mocks.isDesktop = true
    mocks.selectedAssistant = null
    mocks.buddyShellEnabled = true
    mocks.reducedMotion = false
    companionMocks.calls = []
    companionMocks.snapshot = {
      generation: 1,
      phase: "idle",
      actionToken: null,
      requestedState: null,
      facing: "right",
      transientOffsetX: 0,
      suspension: "none"
    }
    companionMocks.react.mockReset().mockReturnValue(true)
    companionMocks.completeAction.mockReset()
    preferenceMocks.getBuddyPreferences.mockReset().mockResolvedValue({
      ambient_mode: "expressive",
      version: null,
      stored: false
    })
    preferenceMocks.getPersonaBuddyPreferences.mockReset().mockResolvedValue({
      ambient_mode: null,
      version: 1,
      stored: false
    })
    preferenceMocks.updateBuddyPreferences.mockReset()
    preferenceMocks.updatePersonaBuddyPreferences.mockReset()
    assetMocks.acquirePersonaVisualAsset.mockReset().mockImplementation(
      async (asset: { id: string; mime_type: string }) => ({
        url: `blob:${asset.id}`,
        mimeType: asset.mime_type,
        release: vi.fn()
      })
    )
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
      firstUseHintDismissed: false,
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

    expect(screen.getByRole("button", {
      name: "Buddy for Persona persona-2"
    })).toBeInTheDocument()
    expect(screen.getByTestId("persona-buddy-dock")).not.toHaveTextContent("owl")
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
    expect(screen.getByRole("button", {
      name: "Buddy for Persona Buddy"
    })).toBeDisabled()
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
        "blob:idle-asset"
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
        "blob:idle-asset"
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
    expect(screen.getByRole("button", {
      name: "Buddy for Persona persona-1"
    })).toBeInTheDocument()
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
    expect(screen.getByRole("button", {
      name: "Buddy for Persona persona-1"
    })).toBeInTheDocument()
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
      screen.getByRole("button", { name: "Open Buddy controls" })
    )

    expect(
      screen.getByRole("link", { name: "Choose/Change Buddy" })
    ).toHaveAttribute("href", "/persona?persona_id=persona-1&tab=visuals")
  })

  it("closes controls through the named button and Escape", () => {
    renderHost({
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "web-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1")
      }
    })

    fireEvent.click(screen.getByRole("button", { name: "Open Buddy controls" }))
    fireEvent.click(screen.getByRole("button", { name: "Close Buddy controls" }))
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(false)
    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ controlsOpen: false })
    )

    fireEvent.click(screen.getByRole("button", { name: "Open Buddy controls" }))
    fireEvent.keyDown(window, { key: "Escape" })
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(false)
  })

  it("keeps connected resting chrome quiet", async () => {
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

    expect(screen.queryByTestId("persona-buddy-live-status")).not.toBeInTheDocument()
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
      screen.getByRole("button", { name: "Open Buddy controls" })
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
        live_voice_state: "idle",
        live_voice_is_listening: true
      },
      root: "sidepanel"
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Open Buddy controls" })
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
    await dragBuddyBy(-48)

    expect(screen.getByTestId("persona-buddy-urgent-badge")).toHaveTextContent("3")
    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ dragging: true })
    )
    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
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
      screen.getByRole("button", { name: "Open Buddy controls" })
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
      "blob:tool-asset"
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
      "blob:tool-notes-search-asset"
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
      "blob:tool-notes-search-asset"
    )
  })

  it("keeps declared movement states inside the hook-owned drag flow", async () => {
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

    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ dragging: true })
    )
    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
    rectSpy.mockRestore()
  })

  it("does not create a second runtime override while dragging left", async () => {
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

    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ dragging: true })
    )
    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
    rectSpy.mockRestore()
  })

  it("ends hook-owned dragging and sends the drag reaction on pointer release", async () => {
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
    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ dragging: true })
    )

    fireEvent.pointerUp(window, { pointerId: 1 })

    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ dragging: false })
    )
    expect(companionMocks.react).toHaveBeenCalledWith("drag")
    rectSpy.mockRestore()
  })

  it("keeps dock dragging without setting a movement override for packs without movement states", async () => {
    const rectSpy = mockDockRect()
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
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "idle"
      )
    })
    const initialPosition =
      usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    await dragBuddyBy(-48)

    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
    expect(
      usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    ).toEqual(initialPosition)
    fireEvent.pointerUp(window, { pointerId: 1, clientX: 92, clientY: 130 })
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
    expect(screen.getByRole("button", {
      name: "Buddy for Persona persona-1"
    })).toBeInTheDocument()
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
    expect(screen.getByRole("button", {
      name: "Buddy for Persona persona-1"
    })).toBeInTheDocument()
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "tool-asset"
    )
  })

  it("drives the single companion hook with fail-closed layered settings and sidepanel coercion", async () => {
    preferenceMocks.getBuddyPreferences.mockResolvedValue({
      ambient_mode: "roaming",
      version: 4,
      stored: true
    })

    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })

    await waitFor(() => {
      expect(companionMocks.calls.at(-1)).toEqual(
        expect.objectContaining({
          personaId: "persona-1",
          mode: "expressive",
          surface: "sidepanel",
          semanticState: "idle"
        })
      )
    })
    expect(preferenceMocks.getBuddyPreferences).toHaveBeenCalledTimes(1)
    expect(preferenceMocks.getPersonaBuddyPreferences).toHaveBeenCalledWith(
      "persona-1"
    )
    fireEvent.click(screen.getByRole("button", { name: "Open Buddy controls" }))
    expect(screen.getByTestId("persona-buddy-effective-mode")).toHaveTextContent(
      "Effective: Expressive · Roaming is limited to Expressive in the sidepanel."
    )
  })

  it("refetches layered settings and reports a stale per-Persona update", async () => {
    preferenceMocks.getPersonaBuddyPreferences
      .mockResolvedValueOnce({
        ambient_mode: null,
        version: 2,
        stored: false
      })
      .mockResolvedValueOnce({
        ambient_mode: "off",
        version: 3,
        stored: true
      })
    preferenceMocks.updatePersonaBuddyPreferences.mockRejectedValue(
      Object.assign(new Error("stale"), { status: 409 })
    )

    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })

    fireEvent.click(await screen.findByRole("button", { name: "Open Buddy controls" }))
    const personaModes = screen.getByRole("group", { name: "For this Persona" })
    fireEvent.click(within(personaModes).getByRole("radio", { name: "Roaming" }))

    await waitFor(() => {
      expect(preferenceMocks.updatePersonaBuddyPreferences).toHaveBeenCalledWith(
        "persona-1",
        { ambient_mode: "roaming", expected_version: 2 }
      )
      expect(preferenceMocks.getPersonaBuddyPreferences).toHaveBeenCalledTimes(2)
    })
    expect(screen.getByRole("status")).toHaveTextContent(
      "Settings changed elsewhere. Latest values were loaded."
    )
    expect(within(personaModes).getByRole("radio", { name: "Off" })).toBeChecked()
  })

  it("lets the only layered Persona read settle after a global save completes", async () => {
    const initialGlobalRead = deferred<{
      ambient_mode: "expressive"
      version: number
      stored: boolean
    }>()
    const initialPersonaRead = deferred<{
      ambient_mode: null
      version: number
      stored: boolean
    }>()
    preferenceMocks.getBuddyPreferences.mockReturnValue(initialGlobalRead.promise)
    preferenceMocks.getPersonaBuddyPreferences.mockReturnValue(initialPersonaRead.promise)
    preferenceMocks.updateBuddyPreferences.mockResolvedValue({
      ambient_mode: "roaming",
      version: 2,
      stored: true
    })

    renderSwitchableHost(personaContext("persona-1"), "web")
    fireEvent.click(await screen.findByRole("button", { name: "Open Buddy controls" }))
    fireEvent.click(
      within(screen.getByRole("group", { name: "Buddy behavior" }))
        .getByRole("radio", { name: "Roaming" })
    )
    await waitFor(() => {
      expect(preferenceMocks.updateBuddyPreferences).toHaveBeenCalledWith({
        ambient_mode: "roaming",
        expected_version: null
      })
    })

    await act(async () => {
      initialGlobalRead.resolve({ ambient_mode: "expressive", version: 1, stored: true })
      initialPersonaRead.resolve({ ambient_mode: null, version: 1, stored: false })
      await Promise.all([initialGlobalRead.promise, initialPersonaRead.promise])
    })

    expect(screen.getByTestId("persona-buddy-effective-mode")).toHaveTextContent(
      "Effective: Roaming"
    )
    expect(
      within(screen.getByRole("group", { name: "For this Persona" }))
        .getByRole("radio", { name: "Use global" })
    ).toBeChecked()
  })

  it("keeps a completed global save across a Persona switch and a stale layered global read", async () => {
    const globalUpdate = deferred<{
      ambient_mode: "roaming"
      version: number
      stored: boolean
    }>()
    const personaBGlobalRead = deferred<{
      ambient_mode: "expressive"
      version: number
      stored: boolean
    }>()
    const personaBRead = deferred<{
      ambient_mode: null
      version: number
      stored: boolean
    }>()
    preferenceMocks.getBuddyPreferences
      .mockResolvedValueOnce({ ambient_mode: "expressive", version: 1, stored: true })
      .mockReturnValueOnce(personaBGlobalRead.promise)
    preferenceMocks.getPersonaBuddyPreferences.mockImplementation(
      (personaId: string) => personaId === "persona-1"
        ? Promise.resolve({ ambient_mode: null, version: 1, stored: false })
        : personaBRead.promise
    )
    preferenceMocks.updateBuddyPreferences.mockReturnValue(globalUpdate.promise)

    const view = renderSwitchableHost(personaContext("persona-1"), "web")
    fireEvent.click(await screen.findByRole("button", { name: "Open Buddy controls" }))
    await waitFor(() => {
      expect(
        within(screen.getByRole("group", { name: "For this Persona" }))
          .getByRole("radio", { name: "Use global" })
      ).toBeChecked()
    })
    fireEvent.click(
      within(screen.getByRole("group", { name: "Buddy behavior" }))
        .getByRole("radio", { name: "Roaming" })
    )
    await waitFor(() => {
      expect(preferenceMocks.updateBuddyPreferences).toHaveBeenCalledTimes(1)
    })

    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost root="web" context={personaContext("persona-2")} />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    await waitFor(() => {
      expect(preferenceMocks.getPersonaBuddyPreferences).toHaveBeenCalledWith("persona-2")
    })

    await act(async () => {
      globalUpdate.resolve({ ambient_mode: "roaming", version: 2, stored: true })
      await globalUpdate.promise
      personaBGlobalRead.resolve({ ambient_mode: "expressive", version: 1, stored: true })
      personaBRead.resolve({ ambient_mode: null, version: 1, stored: false })
      await Promise.all([personaBGlobalRead.promise, personaBRead.promise])
    })

    expect(screen.getByTestId("persona-buddy-effective-mode")).toHaveTextContent(
      "Effective: Roaming"
    )
    expect(
      within(screen.getByRole("group", { name: "For this Persona" }))
        .getByRole("radio", { name: "Use global" })
    ).toBeChecked()
  })

  it("ignores a delayed Persona preference read after focus moves to another Persona", async () => {
    const personaA = deferred<{
      ambient_mode: "roaming"
      version: number
      stored: boolean
    }>()
    const personaB = deferred<{
      ambient_mode: "off"
      version: number
      stored: boolean
    }>()
    preferenceMocks.getPersonaBuddyPreferences.mockImplementation(
      (personaId: string) => personaId === "persona-1" ? personaA.promise : personaB.promise
    )

    const view = renderSwitchableHost(personaContext("persona-1"))
    await waitFor(() => {
      expect(preferenceMocks.getPersonaBuddyPreferences).toHaveBeenCalledWith("persona-1")
    })
    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost root="sidepanel" context={personaContext("persona-2")} />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    await waitFor(() => {
      expect(preferenceMocks.getPersonaBuddyPreferences).toHaveBeenCalledWith("persona-2")
    })
    await act(async () => {
      personaB.resolve({ ambient_mode: "off", version: 7, stored: true })
      await personaB.promise
    })
    await act(async () => {
      personaA.resolve({ ambient_mode: "roaming", version: 3, stored: true })
      await personaA.promise
    })

    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ personaId: "persona-2", mode: "off" })
    )
    fireEvent.click(screen.getByRole("button", { name: "Open Buddy controls" }))
    expect(
      within(screen.getByRole("group", { name: "For this Persona" }))
        .getByRole("radio", { name: "Off" })
    ).toBeChecked()
  })

  it.each(["success", "error", "conflict"] as const)(
    "ignores a delayed Persona A preference write %s after focus moves to Persona B",
    async (outcome) => {
      preferenceMocks.getPersonaBuddyPreferences.mockImplementation(
        async (personaId: string) => personaId === "persona-1"
          ? { ambient_mode: null, version: 1, stored: false }
          : { ambient_mode: "off", version: 4, stored: true }
      )
      const update = deferred<{
        ambient_mode: "roaming"
        version: number
        stored: boolean
      }>()
      preferenceMocks.updatePersonaBuddyPreferences.mockReturnValue(update.promise)
      const view = renderSwitchableHost(personaContext("persona-1"))
      fireEvent.click(await screen.findByRole("button", { name: "Open Buddy controls" }))
      await waitFor(() => {
        expect(
          within(screen.getByRole("group", { name: "For this Persona" }))
            .getByRole("radio", { name: "Use global" })
        ).toBeChecked()
      })
      fireEvent.click(
        within(screen.getByRole("group", { name: "For this Persona" }))
          .getByRole("radio", { name: "Roaming" })
      )
      await waitFor(() => {
        expect(preferenceMocks.updatePersonaBuddyPreferences).toHaveBeenCalledWith(
          "persona-1",
          { ambient_mode: "roaming", expected_version: 1 }
        )
      })

      view.rerender(
        <MemoryRouter>
          <BuddyShellRenderContextProvider>
            <ContextDrivenHost root="sidepanel" context={personaContext("persona-2")} />
          </BuddyShellRenderContextProvider>
        </MemoryRouter>
      )
      await waitFor(() => {
        expect(companionMocks.calls.at(-1)).toEqual(
          expect.objectContaining({ personaId: "persona-2", mode: "off" })
        )
      })

      await act(async () => {
        if (outcome === "success") {
          update.resolve({ ambient_mode: "roaming", version: 2, stored: true })
        } else {
          update.reject(
            Object.assign(new Error(outcome), {
              status: outcome === "conflict" ? 409 : 500
            })
          )
        }
        try {
          await update.promise
        } catch {
          // The component owns the rejection; the test only releases it.
        }
      })

      expect(companionMocks.calls.at(-1)).toEqual(
        expect.objectContaining({ personaId: "persona-2", mode: "off" })
      )
      expect(screen.queryByText("Buddy settings could not be saved.")).not.toBeInTheDocument()
      expect(
        screen.queryByText("Settings changed elsewhere. Latest values were loaded.")
      ).not.toBeInTheDocument()
      expect(
        within(screen.getByRole("group", { name: "For this Persona" }))
          .getByRole("radio", { name: "Off" })
      ).toBeChecked()
    }
  )

  it.each(["error", "success"] as const)(
    "ignores a delayed Persona A preference write %s after an A to B to A focus cycle",
    async (outcome) => {
      preferenceMocks.getPersonaBuddyPreferences
        .mockResolvedValueOnce({ ambient_mode: null, version: 1, stored: false })
        .mockResolvedValueOnce({ ambient_mode: "off", version: 4, stored: true })
        .mockResolvedValueOnce({ ambient_mode: "off", version: 9, stored: true })
      const staleUpdate = deferred<{
        ambient_mode: "roaming"
        version: number
        stored: boolean
      }>()
      preferenceMocks.updatePersonaBuddyPreferences
        .mockReturnValueOnce(staleUpdate.promise)
        .mockResolvedValue({ ambient_mode: "expressive", version: 10, stored: true })

      const view = renderSwitchableHost(personaContext("persona-1"))
      fireEvent.click(await screen.findByRole("button", { name: "Open Buddy controls" }))
      const personaModes = screen.getByRole("group", { name: "For this Persona" })
      await waitFor(() => {
        expect(
          within(personaModes).getByRole("radio", { name: "Use global" })
        ).toBeChecked()
      })
      fireEvent.click(within(personaModes).getByRole("radio", { name: "Roaming" }))
      await waitFor(() => {
        expect(preferenceMocks.updatePersonaBuddyPreferences).toHaveBeenCalledWith(
          "persona-1",
          { ambient_mode: "roaming", expected_version: 1 }
        )
      })

      view.rerender(
        <MemoryRouter>
          <BuddyShellRenderContextProvider>
            <ContextDrivenHost root="sidepanel" context={personaContext("persona-2")} />
          </BuddyShellRenderContextProvider>
        </MemoryRouter>
      )
      await waitFor(() => {
        expect(
          within(screen.getByRole("group", { name: "For this Persona" }))
            .getByRole("radio", { name: "Off" })
        ).toBeChecked()
      })

      view.rerender(
        <MemoryRouter>
          <BuddyShellRenderContextProvider>
            <ContextDrivenHost root="sidepanel" context={personaContext("persona-1")} />
          </BuddyShellRenderContextProvider>
        </MemoryRouter>
      )
      await waitFor(() => {
        expect(preferenceMocks.getPersonaBuddyPreferences).toHaveBeenCalledTimes(3)
        expect(
          within(screen.getByRole("group", { name: "For this Persona" }))
            .getByRole("radio", { name: "Off" })
        ).toBeChecked()
      })

      await act(async () => {
        if (outcome === "success") {
          staleUpdate.resolve({ ambient_mode: "roaming", version: 2, stored: true })
        } else {
          staleUpdate.reject(Object.assign(new Error("offline"), { status: 500 }))
        }
        try {
          await staleUpdate.promise
        } catch {
          // The component owns the rejection; the test only releases it.
        }
      })

      const currentPersonaModes = screen.getByRole("group", { name: "For this Persona" })
      expect(within(currentPersonaModes).getByRole("radio", { name: "Off" })).toBeChecked()
      expect(screen.queryByText("Buddy settings could not be saved.")).not.toBeInTheDocument()
      expect(
        screen.queryByText("Settings changed elsewhere. Latest values were loaded.")
      ).not.toBeInTheDocument()

      fireEvent.click(
        within(currentPersonaModes).getByRole("radio", { name: "Expressive" })
      )
      await waitFor(() => {
        expect(preferenceMocks.updatePersonaBuddyPreferences).toHaveBeenLastCalledWith(
          "persona-1",
          { ambient_mode: "expressive", expected_version: 9 }
        )
      })
    }
  )

  it("renders grounded transient movement without persisting it as an anchor", async () => {
    companionMocks.snapshot = {
      ...companionMocks.snapshot,
      transientOffsetX: 48
    }
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })

    const dock = await screen.findByTestId("persona-buddy-dock")
    const groundedAnchor =
      usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    expect(dock).toHaveStyle({
      left: `${groundedAnchor.x + 48}px`,
      top: `${groundedAnchor.y}px`
    })
    expect(
      usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    ).toEqual(groundedAnchor)
  })

  it("offers a focusable controls button and persists first-use hint dismissal", async () => {
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })

    const controls = await screen.findByRole("button", { name: "Open Buddy controls" })
    controls.focus()
    expect(controls).toHaveFocus()
    fireEvent.click(controls)
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(true)
    expect(screen.getByTestId("persona-buddy-first-use-hint")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Got it" }))
    expect(screen.queryByTestId("persona-buddy-first-use-hint")).not.toBeInTheDocument()
    expect(usePersonaBuddyShellStore.getState().firstUseHintDismissed).toBe(true)
  })

  it("treats a touch tap as one deferred reaction without opening controls", async () => {
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    vi.useFakeTimers()
    fireEvent.pointerDown(buddy, {
      button: 0,
      pointerId: 12,
      pointerType: "touch",
      clientX: 100,
      clientY: 100
    })
    fireEvent.pointerUp(window, {
      pointerId: 12,
      pointerType: "touch",
      clientX: 100,
      clientY: 100
    })
    act(() => vi.advanceTimersByTime(299))
    expect(companionMocks.react).not.toHaveBeenCalled()
    act(() => vi.advanceTimersByTime(1))
    expect(companionMocks.react).toHaveBeenCalledWith("click")
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(false)
    vi.useRealTimers()
  })

  it("defers one pointer click but lets a second click open controls without reacting", async () => {
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    vi.useFakeTimers()
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerUp(window, { pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 2, clientX: 100, clientY: 100 })
    fireEvent.pointerUp(window, { pointerId: 2, clientX: 100, clientY: 100 })
    act(() => vi.advanceTimersByTime(500))

    expect(companionMocks.react).not.toHaveBeenCalled()
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(true)
    vi.useRealTimers()
  })

  it("drops a deferred click when the focused Persona changes", async () => {
    const view = renderSwitchableHost(personaContext("persona-1"))
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    vi.useFakeTimers()
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 21, clientX: 100, clientY: 100 })
    fireEvent.pointerUp(window, { pointerId: 21, clientX: 100, clientY: 100 })

    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost root="sidepanel" context={personaContext("persona-2")} />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    act(() => vi.advanceTimersByTime(300))

    expect(companionMocks.react).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  it("drops a deferred click when the companion engine generation changes", async () => {
    const context = personaContext("persona-1")
    const view = renderSwitchableHost(context)
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    vi.useFakeTimers()
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 22, clientX: 100, clientY: 100 })
    fireEvent.pointerUp(window, { pointerId: 22, clientX: 100, clientY: 100 })
    companionMocks.snapshot = { ...companionMocks.snapshot, generation: 2 }
    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost root="sidepanel" context={context} />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    act(() => vi.advanceTimersByTime(300))

    expect(companionMocks.react).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  it("drops a deferred click when the active visual pack changes", async () => {
    const packA = buildVisualPack("persona-1")
    const packB = { ...buildVisualPack("persona-1"), id: "pack-2" }
    visualMocks.listPersonaVisualPacks
      .mockResolvedValueOnce({ packs: [packA], active_pack: packA })
      .mockResolvedValue({ packs: [packB], active_pack: packB })
    renderSwitchableHost(personaContext("persona-1"))
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    await waitFor(() => {
      expect(companionMocks.calls.at(-1)).toEqual(
        expect.objectContaining({ packId: "pack-1" })
      )
    })
    vi.useFakeTimers()
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 24, clientX: 100, clientY: 100 })
    fireEvent.pointerUp(window, { pointerId: 24, clientX: 100, clientY: 100 })
    await act(async () => {
      fireEvent(
        window,
        new CustomEvent(PERSONA_VISUAL_PACK_ACTIVATED_EVENT, {
          detail: { personaId: "persona-1", packId: "pack-2" }
        })
      )
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(companionMocks.calls.at(-1)).toEqual(
      expect.objectContaining({ packId: "pack-2" })
    )
    act(() => vi.advanceTimersByTime(300))

    expect(companionMocks.react).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  it("removes a fallback nudge immediately on semantic or reduced-motion transitions", async () => {
    const pack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({
      packs: [pack],
      active_pack: pack
    })
    companionMocks.react.mockReturnValue(false)
    const idleContext = personaContext("persona-1")
    const view = renderSwitchableHost(idleContext)
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    await screen.findByTestId("persona-buddy-visual-wrapper")
    vi.useFakeTimers()
    fireEvent.keyDown(buddy, { key: " " })
    expect(screen.getByTestId("persona-buddy-visual-wrapper")).toHaveStyle({
      transform: "scaleX(1) translateX(4px)"
    })

    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost
            root="sidepanel"
            context={{ ...idleContext, visual_state: "thinking" }}
          />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    expect(screen.getByTestId("persona-buddy-visual-wrapper")).toHaveStyle({
      transform: "scaleX(1) translateX(0)"
    })

    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost root="sidepanel" context={idleContext} />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    fireEvent.keyDown(buddy, { key: " " })
    expect(screen.getByTestId("persona-buddy-visual-wrapper")).toHaveStyle({
      transform: "scaleX(1) translateX(4px)"
    })
    mocks.reducedMotion = true
    view.rerender(
      <MemoryRouter>
        <BuddyShellRenderContextProvider>
          <ContextDrivenHost root="sidepanel" context={idleContext} />
        </BuddyShellRenderContextProvider>
      </MemoryRouter>
    )
    expect(screen.getByTestId("persona-buddy-visual-wrapper")).toHaveStyle({
      transform: "scaleX(1) translateX(0)"
    })
    act(() => vi.advanceTimersByTime(500))
    expect(screen.getByTestId("persona-buddy-visual-wrapper")).toHaveStyle({
      transform: "scaleX(1) translateX(0)"
    })
    vi.useRealTimers()
  })

  it("persists no drag position before the 8px threshold and stores only the final anchor", async () => {
    const rectSpy = mockDockRect()
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    const initial = usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 9, clientX: 140, clientY: 130 })
    fireEvent.pointerMove(window, { pointerId: 9, clientX: 147, clientY: 130 })
    expect(usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]).toEqual(initial)
    fireEvent.pointerMove(window, { pointerId: 9, clientX: 92, clientY: 130 })
    expect(usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]).toEqual(initial)
    fireEvent.pointerUp(window, { pointerId: 9, clientX: 92, clientY: 130 })
    expect(usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]).not.toEqual(initial)
    rectSpy.mockRestore()
  })

  it("cancels a moved pointer without persisting a partial drag or reacting", async () => {
    const rectSpy = mockDockRect()
    renderHost({ root: "sidepanel", context: personaContext("persona-1") })
    const buddy = await screen.findByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    const initial = usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]
    fireEvent.pointerDown(buddy, { button: 0, pointerId: 23, clientX: 140, clientY: 130 })
    fireEvent.pointerMove(window, { pointerId: 23, clientX: 92, clientY: 130 })
    fireEvent.pointerCancel(window, { pointerId: 23, clientX: 92, clientY: 130 })

    expect(usePersonaBuddyShellStore.getState().positions["sidepanel-desktop"]).toEqual(initial)
    expect(companionMocks.react).not.toHaveBeenCalled()
    rectSpy.mockRestore()
  })

  it("opens on Enter, reacts on Space without scrolling, and forwards the exact renderer action token", async () => {
    companionMocks.snapshot = {
      ...companionMocks.snapshot,
      phase: "action",
      actionToken: 42,
      generation: 7
    }
    const pack = buildVisualPack("persona-1")
    visualMocks.listPersonaVisualPacks.mockResolvedValue({ packs: [pack], active_pack: pack })
    vi.useFakeTimers()
    renderHost({
      root: "sidepanel",
      context: {
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "persona-1",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: buildBuddySummary("persona-1"),
        live_voice_state: "idle"
      }
    })
    await act(async () => {})
    const buddy = screen.getByRole("button", {
      name: "Buddy for Persona persona-1"
    })
    fireEvent.keyDown(buddy, { key: "Enter" })
    expect(usePersonaBuddyShellStore.getState().isOpen).toBe(true)
    const spaceAllowed = fireEvent.keyDown(buddy, { key: " " })
    expect(spaceAllowed).toBe(false)
    expect(companionMocks.react).toHaveBeenCalledWith("space")
    act(() => vi.advanceTimersByTime(100))
    expect(companionMocks.completeAction).toHaveBeenCalledWith(42, true)
    vi.useRealTimers()
  })
})
