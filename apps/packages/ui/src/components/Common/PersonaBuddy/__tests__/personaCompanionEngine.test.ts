import { act, renderHook } from "@testing-library/react"
import { createElement, StrictMode, type PropsWithChildren } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { asPersonaVisualCustomStateId } from "@/types/persona-visuals"
import type {
  PersonaCompanionDiagnosticEvent,
  PersonaCompanionEngine,
  PersonaCompanionRuntime,
  PersonaCompanionTimer
} from "../personaCompanionEngine"
import { createPersonaCompanionEngine } from "../personaCompanionEngine"
import { usePersonaCompanion } from "../usePersonaCompanion"

type FakeTimer = {
  id: number
  due: number
  callback: () => void
  canceled: boolean
}

const createFakeCompanionRuntime = (randomValues: number[] = [0]) => {
  let now = 0
  let nextId = 1
  let randomIndex = 0
  const timers: FakeTimer[] = []
  const diagnostics: PersonaCompanionDiagnosticEvent[] = []

  const runtime: PersonaCompanionRuntime = {
    now: () => now,
    random: () => randomValues[randomIndex++] ?? randomValues.at(-1) ?? 0,
    setTimer: (callback, delayMs) => {
      const timer = {
        id: nextId++,
        due: now + delayMs,
        callback,
        canceled: false
      }
      timers.push(timer)
      return timer as PersonaCompanionTimer
    },
    clearTimer: (handle) => {
      ;(handle as FakeTimer).canceled = true
    },
    diagnose: (event) => diagnostics.push(event)
  }

  const advanceBy = (milliseconds: number) => {
    const target = now + milliseconds
    while (true) {
      const timer = timers
        .filter((candidate) => !candidate.canceled && candidate.due <= target)
        .sort((a, b) => a.due - b.due || a.id - b.id)[0]
      if (!timer) break
      now = timer.due
      timer.canceled = true
      timer.callback()
    }
    now = target
  }

  return {
    runtime,
    diagnostics,
    advanceBy,
    activeTimerCount: () => timers.filter((timer) => !timer.canceled).length,
    latestTimer: () => timers.at(-1),
    fireCanceled: (timer: FakeTimer) => timer.callback()
  }
}

const look = asPersonaVisualCustomStateId("ambient.look")
const wave = asPersonaVisualCustomStateId("ambient.wave")
const walk = asPersonaVisualCustomStateId("ambient.walk")
const turnLeft = asPersonaVisualCustomStateId("ambient.turn.left")
const click = asPersonaVisualCustomStateId("reaction.click")
const badWeight = asPersonaVisualCustomStateId("ambient.bad_weight")
const badCooldown = asPersonaVisualCustomStateId("ambient.bad_cooldown")
const badNegative = asPersonaVisualCustomStateId("ambient.bad_negative")
const badNegativeCooldown = asPersonaVisualCustomStateId(
  "ambient.bad_negative_cooldown"
)
const badMovement = asPersonaVisualCustomStateId("ambient.bad_movement")
const badOrder = asPersonaVisualCustomStateId("ambient.bad_order")
const badNullMovement = asPersonaVisualCustomStateId(
  "ambient.bad_null_movement"
)
const badMovementShape = asPersonaVisualCustomStateId(
  "ambient.bad_movement_shape"
)
const badDirection = asPersonaVisualCustomStateId("ambient.bad_direction")
const badMissingMovement = asPersonaVisualCustomStateId(
  "ambient.bad_missing_movement"
)
const badMovementCategory = asPersonaVisualCustomStateId(
  "ambient.bad_movement_category"
)

const horizontalMovement = {
  direction: "horizontal" as const,
  motion_start_ratio: 0.1,
  motion_end_ratio: 0.9
}
const walkEntry = {
  state: walk,
  trigger: "ambient" as const,
  category: "move" as const,
  movement: horizontalMovement
}

const entries = (...states: Array<typeof look | typeof wave>) =>
  states.map((state) => ({
    state,
    trigger: "ambient" as const,
    category: "idle_variant" as const,
    suggested_weight: state === wave ? 3 : 1
  }))

const idleInput = (overrides: Record<string, unknown> = {}) => ({
  personaId: "persona-1",
  packId: "pack-1",
  packRevision: 1,
  semanticState: "idle" as const,
  mode: "expressive" as const,
  surface: "web" as const,
  visibility: "visible" as const,
  controlsOpen: false,
  focusWithin: false,
  dragging: false,
  reducedMotion: false,
  behavior: {
    schema_version: 1 as const,
    entries: entries(look)
  },
  availableStates: [look],
  mirrorSafeStates: [],
  horizontalBounds: { min: -100, max: 100 },
  timing: {
    ambientMinMs: 30_000,
    ambientMaxMs: 30_000,
    actionDurationMs: 500,
    movementDistancePx: 48
  },
  ...overrides
})

const StrictModeWrapper = ({ children }: PropsWithChildren) =>
  createElement(StrictMode, null, children)

const PROHIBITED_ENGINE_IMPORTS = [
  "@/services/tldw",
  "@/services/tldw/TldwApiClient",
  "@/services/persona-visual-assets",
  "@/services/persona-visuals",
  "@/store/model",
  "../personaVisualAssets",
  "../personaVisualDiagnostics",
  "../personaVisualRenderers",
  "../SpriteFrameRenderer"
] as const

const getActionToken = (engine: PersonaCompanionEngine): number => {
  const token = engine.getSnapshot().actionToken
  if (typeof token !== "number") throw new Error("expected active action token")
  return token
}

const completeActionToken = (
  engine: PersonaCompanionEngine,
  token: number,
  succeeded: boolean
) =>
  engine.completeAction(token, succeeded)

const completeCurrentAction = (
  engine: PersonaCompanionEngine,
  succeeded = true
) => engine.completeAction(getActionToken(engine), succeeded)

afterEach(() => {
  vi.unstubAllGlobals()
  for (const moduleId of PROHIBITED_ENGINE_IMPORTS) vi.doUnmock(moduleId)
  vi.resetModules()
})

describe("createPersonaCompanionEngine", () => {
  it("starts a fresh full interval after hidden-tab resume", () => {
    const fake = createFakeCompanionRuntime([0, 0.5])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    fake.advanceBy(29_999)
    expect(engine.getSnapshot().phase).toBe("idle")
    engine.update(idleInput({ visibility: "hidden" }))
    fake.advanceBy(120_000)
    engine.update(idleInput())
    fake.advanceBy(29_999)
    expect(engine.getSnapshot().phase).toBe("idle")
    fake.advanceBy(1)
    expect(engine.getSnapshot().requestedState).toBe("ambient.look")
  })

  it("uses relative weights and avoids an immediate repeat when an alternative exists", () => {
    const fake = createFakeCompanionRuntime([0, 0.9, 0, 0.9])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        behavior: { schema_version: 1, entries: entries(look, wave) },
        availableStates: [look, wave]
      })
    )

    fake.advanceBy(30_000)
    expect(engine.getSnapshot().requestedState).toBe("ambient.wave")
    completeCurrentAction(engine)
    fake.advanceBy(30_000)
    expect(engine.getSnapshot().requestedState).toBe("ambient.look")
  })

  it("stays idle for empty or unresolved ambient sets", () => {
    const fake = createFakeCompanionRuntime()
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput({ availableStates: [] }))
    fake.advanceBy(30_000)

    expect(engine.getSnapshot().requestedState).toBe("idle")
    expect(fake.diagnostics.at(-1)?.failureClass).toBe("empty_set")
  })

  it("fresh-imports and runs behind renderer-neutral dependency guards", async () => {
    vi.resetModules()
    const fetch = vi.fn(() => {
      throw new Error("network access is forbidden")
    })
    vi.stubGlobal("fetch", fetch)
    for (const moduleId of PROHIBITED_ENGINE_IMPORTS) {
      vi.doMock(moduleId, () => {
        throw new Error(`companion engine imported ${moduleId}`)
      })
    }

    const { createPersonaCompanionEngine: createGuardedEngine } = await import(
      "../personaCompanionEngine"
    )
    const fake = createFakeCompanionRuntime([0, 0, 0])
    const engine = createGuardedEngine(fake.runtime)
    engine.update(
      idleInput({
        behavior: {
          schema_version: 1,
          entries: [
            ...entries(look),
            { state: click, trigger: "click", category: "reaction" }
          ]
        },
        availableStates: [look, click]
      })
    )
    fake.advanceBy(30_000)
    completeCurrentAction(engine)
    expect(engine.react("click")).toBe(true)
    completeCurrentAction(engine)

    expect(fetch).not.toHaveBeenCalled()
  })

  it("clamps engine timing and movement distance with allowed metadata defaults", () => {
    const fake = createFakeCompanionRuntime([0, 0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: {
          schema_version: 1,
          entries: [walkEntry]
        },
        availableStates: [walk],
        horizontalBounds: { min: -20, max: 30 },
        timing: {
          ambientMinMs: -1,
          ambientMaxMs: Number.POSITIVE_INFINITY,
          actionDurationMs: 1,
          movementDistancePx: 5_000
        }
      })
    )

    fake.advanceBy(29_999)
    expect(engine.getSnapshot().phase).toBe("idle")
    fake.advanceBy(1)
    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({ phase: "action", transientOffsetX: -20 })
    )
    fake.advanceBy(149)
    expect(engine.getSnapshot().phase).toBe("action")
    fake.advanceBy(1)
    expect(engine.getSnapshot().phase).toBe("idle")
    expect(Object.values(engine.getSnapshot()).every((value) => {
      return typeof value !== "number" || Number.isFinite(value)
    })).toBe(true)
  })

  it("fails closed when every declared entry has invalid present metadata", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: {
          schema_version: 1,
          entries: [
            {
              state: badWeight,
              trigger: "ambient",
              category: "idle_variant",
              suggested_weight: Number.POSITIVE_INFINITY
            },
            {
              state: badCooldown,
              trigger: "ambient",
              category: "idle_variant",
              suggested_cooldown_ms: Number.NaN
            },
            {
              state: badNegative,
              trigger: "ambient",
              category: "idle_variant",
              suggested_weight: -1
            },
            {
              state: badNegativeCooldown,
              trigger: "ambient",
              category: "idle_variant",
              suggested_cooldown_ms: -1
            },
            {
              state: badMovement,
              trigger: "ambient",
              category: "move",
              movement: {
                direction: "horizontal",
                motion_start_ratio: Number.NaN,
                motion_end_ratio: 0.9
              }
            },
            {
              state: badOrder,
              trigger: "ambient",
              category: "move",
              movement: {
                direction: "horizontal",
                motion_start_ratio: 0.9,
                motion_end_ratio: 0.1
              }
            }
          ]
        },
        availableStates: [
          badWeight,
          badCooldown,
          badNegative,
          badNegativeCooldown,
          badMovement,
          badOrder
        ]
      })
    )
    fake.advanceBy(30_000)

    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({
        phase: "idle",
        actionToken: null,
        requestedState: "idle",
        transientOffsetX: 0
      })
    )
    expect(fake.diagnostics.at(-1)).toEqual(
      expect.objectContaining({
        event: "ambient_skipped",
        failureClass: "empty_set"
      })
    )
  })

  it("keeps deterministic valid selection when invalid entries are mixed in", () => {
    const fake = createFakeCompanionRuntime([0, 0.9])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        behavior: {
          schema_version: 1,
          entries: [
            ...entries(look),
            {
              state: wave,
              trigger: "ambient",
              category: "idle_variant",
              suggested_weight: Number.POSITIVE_INFINITY
            }
          ]
        },
        availableStates: [look, wave]
      })
    )
    fake.advanceBy(30_000)

    expect(engine.getSnapshot().requestedState).toBe("ambient.look")
  })

  it("excludes malformed movement declarations by presence, shape, direction, and category", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: {
          schema_version: 1,
          entries: [
            {
              state: badNullMovement,
              trigger: "ambient",
              category: "move",
              movement: null
            },
            {
              state: badMovementShape,
              trigger: "ambient",
              category: "move",
              movement: "horizontal"
            },
            {
              state: badDirection,
              trigger: "ambient",
              category: "move",
              movement: { ...horizontalMovement, direction: "vertical" }
            },
            {
              state: badMissingMovement,
              trigger: "ambient",
              category: "move"
            },
            {
              state: badMovementCategory,
              trigger: "ambient",
              category: "idle_variant",
              movement: horizontalMovement
            }
          ]
        },
        availableStates: [
          badNullMovement,
          badMovementShape,
          badDirection,
          badMissingMovement,
          badMovementCategory
        ]
      })
    )
    fake.advanceBy(30_000)

    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({
        phase: "idle",
        actionToken: null,
        requestedState: "idle",
        transientOffsetX: 0
      })
    )
    expect(fake.diagnostics.at(-1)).toEqual(
      expect.objectContaining({
        event: "ambient_skipped",
        failureClass: "empty_set"
      })
    )
  })

  it("still schedules valid omitted suggestions beside malformed movement", () => {
    const fake = createFakeCompanionRuntime([0, 0.9])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: {
          schema_version: 1,
          entries: [
            ...entries(look),
            {
              state: badDirection,
              trigger: "ambient",
              category: "move",
              movement: { ...horizontalMovement, direction: "vertical" },
              suggested_weight: 1_000
            }
          ]
        },
        availableStates: [look, badDirection]
      })
    )
    fake.advanceBy(30_000)

    expect(engine.getSnapshot().requestedState).toBe("ambient.look")
  })

  it("clamps cadence and duration at their upper bounds", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        timing: {
          ambientMinMs: 999_999,
          ambientMaxMs: 999_999,
          actionDurationMs: 999_999,
          movementDistancePx: 48
        }
      })
    )
    fake.advanceBy(89_999)
    expect(engine.getSnapshot().phase).toBe("idle")
    fake.advanceBy(1)
    expect(engine.getSnapshot().phase).toBe("action")
    fake.advanceBy(7_999)
    expect(engine.getSnapshot().phase).toBe("action")
    fake.advanceBy(1)
    expect(engine.getSnapshot().phase).toBe("idle")
  })

  it("honors cooldowns clamped to one day", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        behavior: {
          schema_version: 1,
          entries: [
            {
              ...entries(look)[0],
              suggested_cooldown_ms: 999_999_999
            }
          ]
        }
      })
    )
    fake.advanceBy(30_000)
    completeCurrentAction(engine)
    fake.advanceBy(86_399_999)
    expect(engine.getSnapshot().requestedState).toBe("idle")
    fake.advanceBy(1)
    expect(engine.getSnapshot().requestedState).toBe("ambient.look")
  })

  it("uses source-scoped, idempotent, expiring leases", () => {
    const fake = createFakeCompanionRuntime()
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput({ mode: "off" }))
    const oldLease = engine.acquireLease("voice", "speaking", 1_000)
    const currentLease = engine.acquireLease("voice", "error", 1_000)

    oldLease.release()
    oldLease.release()
    expect(engine.getSnapshot().requestedState).toBe("error")
    currentLease.release()
    expect(engine.getSnapshot().requestedState).toBe("idle")

    engine.acquireLease("voice", "speaking", 1_000)
    fake.advanceBy(999)
    expect(engine.getSnapshot().requestedState).toBe("speaking")
    fake.advanceBy(1)
    expect(engine.getSnapshot().requestedState).toBe("idle")
  })

  it("resolves lease priority and preempts ambient actions", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    fake.advanceBy(30_000)
    engine.acquireLease("approval", "approval_needed", 1_000)
    engine.acquireLease("offline", "offline", 1_000)
    engine.acquireLease("error", "error", 1_000)

    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({ phase: "idle", requestedState: "error" })
    )
    expect(fake.diagnostics).toContainEqual(
      expect.objectContaining({
        event: "ambient_preempted",
        state: "ambient.look"
      })
    )
  })

  it("fences an action completion callback after semantic preemption", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    fake.advanceBy(30_000)
    const staleCompletion = fake.latestTimer()
    if (!staleCompletion) throw new Error("expected action timer")
    engine.acquireLease("voice", "speaking", 1_000)
    const preempted = engine.getSnapshot()
    fake.fireCanceled(staleCompletion)

    expect(engine.getSnapshot()).toBe(preempted)
    expect(fake.diagnostics.at(-1)).toEqual(
      expect.objectContaining({
        event: "stale_generation",
        failureClass: "stale_timer"
      })
    )
  })

  it("ignores stale public completions through preemption and turn-to-move rotation", () => {
    const fake = createFakeCompanionRuntime([0, 0, 0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    fake.advanceBy(30_000)
    const actionAToken = getActionToken(engine)

    engine.update(idleInput({ focusWithin: true }))
    const preempted = engine.getSnapshot()
    completeActionToken(engine, actionAToken, true)
    expect(engine.getSnapshot()).toBe(preempted)

    const movement = {
      mode: "roaming" as const,
      packRevision: 2,
      behavior: {
        schema_version: 1 as const,
        entries: [
          walkEntry,
          {
            state: turnLeft,
            trigger: "ambient" as const,
            category: "idle_variant" as const
          }
        ]
      },
      availableStates: [walk, turnLeft]
    }
    engine.update(idleInput(movement))
    fake.advanceBy(30_000)
    const turnToken = getActionToken(engine)
    const turnSnapshot = engine.getSnapshot()
    completeActionToken(engine, actionAToken, true)
    expect(engine.getSnapshot()).toBe(turnSnapshot)

    completeActionToken(engine, turnToken, true)
    const moveToken = getActionToken(engine)
    const moveSnapshot = engine.getSnapshot()
    expect(moveToken).not.toBe(turnToken)
    expect(moveSnapshot).toEqual(
      expect.objectContaining({
        requestedState: "ambient.walk",
        facing: "left",
        transientOffsetX: -48
      })
    )

    completeActionToken(engine, turnToken, false)
    completeActionToken(engine, turnToken, true)
    expect(engine.getSnapshot()).toBe(moveSnapshot)
    expect(engine.getSnapshot().generation).toBe(moveSnapshot.generation)
    expect(fake.diagnostics.at(-1)).toEqual(
      expect.objectContaining({
        event: "stale_generation",
        failureClass: "stale_action"
      })
    )
  })

  it("ignores public completion after disposal", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    fake.advanceBy(30_000)
    const token = getActionToken(engine)
    const disposedSnapshot = engine.getSnapshot()
    engine.dispose()
    completeActionToken(engine, token, true)

    expect(engine.getSnapshot()).toBe(disposedSnapshot)
    expect(engine.getSnapshot().generation).toBe(disposedSnapshot.generation)
    expect(fake.diagnostics.at(-1)).toEqual(
      expect.objectContaining({
        event: "stale_generation",
        failureClass: "stale_action"
      })
    )
  })

  it.each([
    ["semantic", { semanticState: "thinking" }],
    ["hidden", { visibility: "hidden" }],
    ["controls", { controlsOpen: true }],
    ["focus", { focusWithin: true }],
    ["drag", { dragging: true }],
    ["reduced_motion", { reducedMotion: true }],
    ["surface", { surface: "popup" }]
  ] as const)("suspends with stable non-moving intent for %s", (suspension, patch) => {
    const fake = createFakeCompanionRuntime()
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput(patch))
    fake.advanceBy(120_000)
    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({
        phase: "idle",
        suspension,
        transientOffsetX: 0
      })
    )
  })

  it("cancels an active ambient action when reduced motion becomes active", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    fake.advanceBy(30_000)
    expect(engine.getSnapshot().phase).toBe("action")
    engine.update(idleInput({ reducedMotion: true }))
    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({
        phase: "idle",
        requestedState: "idle",
        suspension: "reduced_motion"
      })
    )
  })

  it("reclamps transient roaming offset when viewport bounds change", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    const movement = {
      schema_version: 1 as const,
      entries: [walkEntry]
    }
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: movement,
        availableStates: [walk],
        timing: {
          ambientMinMs: 30_000,
          ambientMaxMs: 30_000,
          actionDurationMs: 500,
          movementDistancePx: 80
        }
      })
    )
    fake.advanceBy(30_000)
    expect(engine.getSnapshot().transientOffsetX).toBe(-80)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: movement,
        availableStates: [walk],
        horizontalBounds: { min: -10, max: 10 }
      })
    )
    expect(engine.getSnapshot().transientOffsetX).toBe(-10)
  })

  it.each([
    ["persona", { personaId: "persona-2" }],
    ["pack", { packId: "pack-2", packRevision: 2 }],
    ["focus", { focusWithin: true }]
  ] as const)("fences a stale timer across %s changes", (_name, patch) => {
    const fake = createFakeCompanionRuntime()
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput())
    const staleTimer = fake.latestTimer()
    if (!staleTimer) throw new Error("expected ambient timer")
    engine.update(idleInput(patch))
    const current = engine.getSnapshot()
    fake.fireCanceled(staleTimer)

    expect(engine.getSnapshot()).toBe(current)
    expect(fake.diagnostics.at(-1)).toEqual(
      expect.objectContaining({ event: "stale_generation" })
    )
  })

  it("never calls a persisted-position hook during roaming", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const persistPosition = vi.fn()
    const runtime = Object.assign(fake.runtime, { persistPosition })
    const engine = createPersonaCompanionEngine(runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: {
          schema_version: 1,
          entries: [walkEntry]
        },
        availableStates: [walk]
      })
    )
    fake.advanceBy(31_000)
    expect(persistPosition).not.toHaveBeenCalled()
  })

  it("keeps direct click and Space reactions available while mode is Off", () => {
    const fake = createFakeCompanionRuntime([0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "off",
        behavior: {
          schema_version: 1,
          entries: [
            { state: click, trigger: "click", category: "reaction" }
          ]
        },
        availableStates: [click]
      })
    )
    fake.advanceBy(120_000)
    expect(engine.getSnapshot().requestedState).toBe("idle")
    expect(engine.react("click")).toBe(true)
    completeCurrentAction(engine)
    expect(engine.react("space")).toBe(true)
    expect(engine.getSnapshot().requestedState).toBe("reaction.click")
  })

  it("coerces sidepanel roaming before ambient selection", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        surface: "sidepanel",
        behavior: {
          schema_version: 1,
          entries: [walkEntry]
        },
        availableStates: [walk]
      })
    )
    fake.advanceBy(30_000)
    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({
        requestedState: "idle",
        transientOffsetX: 0
      })
    )
  })

  it("plays a declared turn before committing a facing change", () => {
    const fake = createFakeCompanionRuntime([0, 0])
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(
      idleInput({
        mode: "roaming",
        behavior: {
          schema_version: 1,
          entries: [
            walkEntry,
            {
              state: turnLeft,
              trigger: "ambient",
              category: "idle_variant"
            }
          ]
        },
        availableStates: [walk, turnLeft]
      })
    )
    fake.advanceBy(30_000)
    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({ requestedState: "ambient.turn.left", facing: "right" })
    )
    completeCurrentAction(engine)
    expect(engine.getSnapshot()).toEqual(
      expect.objectContaining({ requestedState: "ambient.walk", facing: "left" })
    )
  })

  it("preserves facing without a turn unless the target is mirror-safe", () => {
    const moveInput = {
      mode: "roaming" as const,
      behavior: {
        schema_version: 1 as const,
        entries: [walkEntry]
      },
      availableStates: [walk]
    }
    const unsafeFake = createFakeCompanionRuntime([0, 0])
    const unsafe = createPersonaCompanionEngine(unsafeFake.runtime)
    unsafe.update(idleInput(moveInput))
    unsafeFake.advanceBy(30_000)
    expect(unsafe.getSnapshot().facing).toBe("right")

    const safeFake = createFakeCompanionRuntime([0, 0])
    const safe = createPersonaCompanionEngine(safeFake.runtime)
    safe.update(idleInput({ ...moveInput, mirrorSafeStates: [walk] }))
    safeFake.advanceBy(30_000)
    expect(safe.getSnapshot().facing).toBe("left")
  })

  it("commits facing after a failed turn only for a mirror-safe target", () => {
    const movement = {
      mode: "roaming" as const,
      behavior: {
        schema_version: 1 as const,
        entries: [
          walkEntry,
          {
            state: turnLeft,
            trigger: "ambient" as const,
            category: "idle_variant" as const
          }
        ]
      },
      availableStates: [walk, turnLeft]
    }
    const unsafeFake = createFakeCompanionRuntime([0, 0])
    const unsafe = createPersonaCompanionEngine(unsafeFake.runtime)
    unsafe.update(idleInput(movement))
    unsafeFake.advanceBy(30_000)
    completeCurrentAction(unsafe, false)
    expect(unsafe.getSnapshot().facing).toBe("right")

    const safeFake = createFakeCompanionRuntime([0, 0])
    const safe = createPersonaCompanionEngine(safeFake.runtime)
    safe.update(idleInput({ ...movement, mirrorSafeStates: [walk] }))
    safeFake.advanceBy(30_000)
    completeCurrentAction(safe, false)
    expect(safe.getSnapshot().facing).toBe("left")
  })

  it("keeps snapshots referentially stable for no-op updates", () => {
    const fake = createFakeCompanionRuntime()
    const engine = createPersonaCompanionEngine(fake.runtime)
    const input = idleInput()
    engine.update(input)
    const snapshot = engine.getSnapshot()
    engine.update(input)
    expect(engine.getSnapshot()).toBe(snapshot)
  })

  it("survives StrictMode replay with one live timer chain and final cleanup", () => {
    const fake = createFakeCompanionRuntime()
    const initialProps = idleInput()
    const { result, rerender, unmount } = renderHook(
      (input) => usePersonaCompanion({ ...input, runtime: fake.runtime }),
      { initialProps, wrapper: StrictModeWrapper }
    )
    expect(fake.activeTimerCount()).toBe(1)
    act(() => fake.advanceBy(30_000))
    expect(result.current.snapshot.requestedState).toBe("ambient.look")
    rerender({ ...initialProps, packRevision: 2 })
    expect(fake.activeTimerCount()).toBe(1)
    const updated = result.current
    rerender({ ...initialProps, packRevision: 2 })
    expect(result.current).toBe(updated)
    act(() => unmount())
    expect(fake.activeTimerCount()).toBe(0)
  })

  it("exposes stable interaction and token-scoped completion commands", () => {
    const fake = createFakeCompanionRuntime([0])
    const initialProps = idleInput({
      mode: "off",
      behavior: {
        schema_version: 1,
        entries: [{ state: click, trigger: "click", category: "reaction" }]
      },
      availableStates: [click]
    })
    const { result, rerender, unmount } = renderHook(
      (input) => usePersonaCompanion({ ...input, runtime: fake.runtime }),
      { initialProps, wrapper: StrictModeWrapper }
    )
    const controller = () => result.current
    const react = controller().react
    const completeAction = controller().completeAction

    act(() => expect(controller().react("click")).toBe(true))
    const clickToken = controller().snapshot.actionToken
    expect(controller().snapshot.requestedState).toBe("reaction.click")
    if (clickToken === null) throw new Error("expected click action token")
    act(() => controller().completeAction(clickToken, true))
    act(() => expect(controller().react("space")).toBe(true))
    rerender(initialProps)

    expect(controller().react).toBe(react)
    expect(controller().completeAction).toBe(completeAction)
    act(() => unmount())
    expect(fake.activeTimerCount()).toBe(0)
  })

  it("rejects an old renderer callback after the hook recreates its engine", () => {
    const first = createFakeCompanionRuntime([0])
    const second = createFakeCompanionRuntime([0])
    const companionInput = idleInput({
      mode: "off",
      behavior: {
        schema_version: 1,
        entries: [{ state: click, trigger: "click", category: "reaction" }]
      },
      availableStates: [click]
    })
    const initialProps = { runtime: first.runtime, packRevision: 1 }
    const { result, rerender, unmount } = renderHook(
      ({ runtime, packRevision }) =>
        usePersonaCompanion({ ...companionInput, packRevision, runtime }),
      { initialProps, wrapper: StrictModeWrapper }
    )
    const controller = () => result.current

    act(() => expect(controller().react("click")).toBe(true))
    const oldToken = controller().snapshot.actionToken
    const staleRendererCompletion = controller().completeAction
    if (oldToken === null) throw new Error("expected old action token")
    expect(first.activeTimerCount()).toBe(1)

    rerender({ runtime: second.runtime, packRevision: 2 })
    expect(first.activeTimerCount()).toBe(0)
    expect(second.activeTimerCount()).toBe(0)
    act(() => expect(controller().react("click")).toBe(true))
    const currentToken = controller().snapshot.actionToken
    if (currentToken === null) throw new Error("expected current action token")
    expect(currentToken).not.toBe(oldToken)
    const currentSnapshot = controller().snapshot
    expect(currentSnapshot).toEqual(
      expect.objectContaining({
        actionToken: currentToken,
        requestedState: "reaction.click",
        facing: "right",
        transientOffsetX: 0
      })
    )
    expect(second.activeTimerCount()).toBe(1)

    act(() => staleRendererCompletion(oldToken, true))
    expect(controller().snapshot).toBe(currentSnapshot)
    expect(controller().snapshot.generation).toBe(currentSnapshot.generation)
    expect(second.activeTimerCount()).toBe(1)

    act(() => unmount())
    expect(second.activeTimerCount()).toBe(0)
  })

  it("disposes the sole timer chain with live leases and actions", () => {
    const fake = createFakeCompanionRuntime()
    const engine = createPersonaCompanionEngine(fake.runtime)
    engine.update(idleInput({ mode: "off" }))
    const lease = engine.acquireLease("voice", "speaking", 10_000)
    expect(fake.activeTimerCount()).toBe(1)
    engine.dispose()
    lease.release()
    expect(fake.activeTimerCount()).toBe(0)
  })
})
