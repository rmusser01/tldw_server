import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import {
  DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS,
  usePersonaBuddyShellStore
} from "@/store/persona-buddy-shell"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import { asPersonaVisualCustomStateId } from "@/types/persona-visuals"
import { BuddyShellRenderContextProvider } from "../BuddyShellRenderContext"
import { BuddyShellHost } from "../BuddyShellHost"

const mocks = vi.hoisted(() => ({
  reducedMotion: false,
  visualPack: null as Record<string, unknown> | null
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => true,
  useMediaQuery: () => mocks.reducedMotion
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (_setting: unknown) => [true, vi.fn(), { isLoading: false }]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn(), { isLoading: false, setRenderValue: vi.fn() }]
}))

vi.mock("@/services/persona-buddy", () => ({
  getBuddyPreferences: vi.fn(async () => ({
    ambient_mode: "off",
    version: 1,
    stored: true
  })),
  getPersonaBuddyPreferences: vi.fn(async () => ({
    ambient_mode: null,
    version: 1,
    stored: false
  })),
  updateBuddyPreferences: vi.fn(),
  updatePersonaBuddyPreferences: vi.fn()
}))

vi.mock("@/services/persona-visual-assets", () => ({
  acquirePersonaVisualAsset: vi.fn(async (asset: { id: string; mime_type: string }) => ({
    url: `blob:${asset.id}`,
    mimeType: asset.mime_type,
    release: vi.fn()
  }))
}))

vi.mock("@/services/persona-visuals", () => ({
  listPersonaVisualPacks: vi.fn(async () => ({
    packs: mocks.visualPack ? [mocks.visualPack] : [],
    active_pack: mocks.visualPack
  })),
  getPersonaVisualPack: vi.fn(async () => mocks.visualPack)
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasPersonaLiveControl: false },
    loading: false,
    error: null
  })
}))

vi.mock("@/hooks/usePersonaLiveControl", () => ({
  usePersonaLiveControl: () => ({
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
  })
}))

const clickState = asPersonaVisualCustomStateId("reaction.click")
const dragState = asPersonaVisualCustomStateId("reaction.drag")
const walkState = asPersonaVisualCustomStateId("reaction.walk")

const buildPack = (clickMoves = false) => {
  const behaviorEntries = clickMoves
    ? [{
        state: walkState,
        trigger: "click" as const,
        category: "move" as const,
        movement: {
          direction: "horizontal" as const,
          motion_start_ratio: 0.1,
          motion_end_ratio: 0.9
        }
      }]
    : [
        {
          state: clickState,
          trigger: "click" as const,
          category: "reaction" as const
        },
        {
          state: dragState,
          trigger: "drag" as const,
          category: "reaction" as const
        }
      ]
  const states = clickMoves
    ? { idle: { animation_id: "idle" }, [walkState]: { animation_id: "walk" } }
    : {
        idle: { animation_id: "idle" },
        [clickState]: { animation_id: "click" },
        [dragState]: { animation_id: "drag" }
      }
  const animationIds = clickMoves ? ["idle", "walk"] : ["idle", "click", "drag"]
  const animations = Object.fromEntries(
    animationIds.map((id) => [
      id,
      { frames: [{ asset_id: `${id}-asset`, duration_ms: 100 }] }
    ])
  )
  const assets = Object.fromEntries(
    animationIds.map((id) => [
      `${id}-asset`,
      {
        id: `${id}-asset`,
        url: `/assets/${id}.png`,
        mime_type: "image/png",
        asset_role: "frame",
        width: 24,
        height: 24
      }
    ])
  )

  return {
    id: clickMoves ? "pack-move" : "pack-reactions",
    persona_id: "persona-1",
    title: "Integration buddy",
    renderer_type: "sprite_frames",
    status: "active",
    version: 4,
    manifest: {
      manifest_version: 1,
      renderer_type: "sprite_frames",
      states,
      animations
    },
    assets_by_id: assets,
    companion_behavior: {
      schema_version: 1,
      entries: behaviorEntries
    }
  }
}

const context = {
  surface_id: "persona-garden",
  surface_active: true,
  active_persona_id: "persona-1",
  position_bucket: "web-desktop" as const,
  persona_source: "route-local" as const,
  buddy_summary: {
    has_buddy: true,
    persona_name: "Persona One",
    role_summary: "Keeps the route on track",
    visual: {
      species_id: "owl",
      silhouette_id: "perch",
      palette_id: "dawn"
    }
  },
  live_voice_state: "idle" as const
}

const renderRealHost = () =>
  render(
    <MemoryRouter>
      <BuddyShellRenderContextProvider initialContext={context}>
        <BuddyShellHost root="web" />
      </BuddyShellRenderContextProvider>
    </MemoryRouter>
  )

describe("BuddyShellHost real companion integration", () => {
  let rectSpy: ReturnType<typeof vi.spyOn>
  let originalWidth: number

  beforeEach(() => {
    originalWidth = window.innerWidth
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 300 })
    mocks.reducedMotion = false
    mocks.visualPack = buildPack()
    document.body.innerHTML = '<div id="tldw-portal-root"></div>'
    usePersonaBuddyShellStore.setState({
      isOpen: false,
      firstUseHintDismissed: true,
      positions: {
        "web-desktop": { x: 20, y: 100 },
        "sidepanel-desktop": {
          ...DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["sidepanel-desktop"]
        }
      }
    })
    usePersonaVisualRuntimeStore.setState({
      override: null,
      runtimeDiagnostics: null
    })
    rectSpy = vi.spyOn(HTMLDivElement.prototype, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 20,
      top: 100,
      right: 84,
      bottom: 164,
      width: 64,
      height: 64,
      toJSON: () => ({})
    } as DOMRect)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: originalWidth
    })
  })

  it("selects an authored Space reaction after real focus suspension is active", async () => {
    renderRealHost()
    const buddy = await screen.findByRole("button", { name: "Toggle buddy for Persona One" })
    fireEvent.focus(buddy)
    fireEvent.keyDown(buddy, { key: " " })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "reaction.click"
      )
    })
  })

  it("selects the authored static PNG reaction through the real engine in reduced motion", async () => {
    mocks.reducedMotion = true
    renderRealHost()
    const buddy = await screen.findByRole("button", { name: "Toggle buddy for Persona One" })
    fireEvent.focus(buddy)
    fireEvent.keyDown(buddy, { key: " " })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "src",
        "blob:click-asset"
      )
    })
    expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
      "data-visual-state",
      "reaction.click"
    )
  })

  it("runs the completed-drag reaction only after the real engine leaves drag suspension", async () => {
    renderRealHost()
    const buddy = await screen.findByRole("button", { name: "Toggle buddy for Persona One" })
    fireEvent.pointerDown(buddy, {
      button: 0,
      pointerId: 31,
      clientX: 20,
      clientY: 100
    })
    fireEvent.pointerMove(window, {
      pointerId: 31,
      clientX: 40,
      clientY: 100
    })
    fireEvent.pointerUp(window, {
      pointerId: 31,
      clientX: 40,
      clientY: 100
    })

    await waitFor(() => {
      expect(screen.getByTestId("persona-visual-frame")).toHaveAttribute(
        "data-visual-state",
        "reaction.drag"
      )
    })
  })

  it("reactively re-clamps transient movement on resize without persisting it", async () => {
    mocks.visualPack = buildPack(true)
    vi.spyOn(Math, "random").mockReturnValue(0.75)
    renderRealHost()
    const buddy = await screen.findByRole("button", { name: "Toggle buddy for Persona One" })
    vi.useFakeTimers()
    fireEvent.pointerDown(buddy, {
      button: 0,
      pointerId: 32,
      clientX: 20,
      clientY: 100
    })
    fireEvent.pointerUp(window, {
      pointerId: 32,
      clientX: 20,
      clientY: 100
    })
    act(() => vi.advanceTimersByTime(300))
    expect(screen.getByTestId("persona-buddy-dock")).toHaveStyle({ left: "68px" })
    expect(usePersonaBuddyShellStore.getState().positions["web-desktop"]).toEqual({
      x: 20,
      y: 100
    })

    Object.defineProperty(window, "innerWidth", { configurable: true, value: 100 })
    fireEvent(window, new Event("resize"))
    await act(async () => {})

    expect(screen.getByTestId("persona-buddy-dock")).toHaveStyle({ left: "20px" })
    expect(usePersonaBuddyShellStore.getState().positions["web-desktop"]).toEqual({
      x: 20,
      y: 100
    })
  })
})
