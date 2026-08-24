import type {
  PersonaAmbientMode,
  PersonaCompanionBehavior,
  PersonaCompanionBehaviorEntry,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import {
  createPersonaCompanionDiagnostic,
  type PersonaCompanionDiagnosticEvent
} from "./personaVisualDiagnostics"
import { resolveEffectiveAmbientMode } from "./personaCompanionPolicy"

export type { PersonaCompanionDiagnosticEvent }

export type PersonaCompanionTimer = unknown

export type PersonaCompanionRuntime = {
  now: () => number
  random: () => number
  setTimer: (callback: () => void, delayMs: number) => PersonaCompanionTimer
  clearTimer: (timer: PersonaCompanionTimer) => void
  diagnose?: (event: PersonaCompanionDiagnosticEvent) => void
}

export type PersonaCompanionTiming = {
  ambientMinMs?: number
  ambientMaxMs?: number
  actionDurationMs?: number
  movementDistancePx?: number
}

export type PersonaCompanionInput = {
  personaId?: string | null
  packId?: string | null
  packRevision?: string | number | null
  semanticState?: PersonaVisualStateId
  mode: PersonaAmbientMode
  surface: string
  visibility: "visible" | "hidden"
  controlsOpen: boolean
  focusWithin: boolean
  dragging: boolean
  reducedMotion: boolean
  behavior?: PersonaCompanionBehavior | null
  availableStates?: Iterable<PersonaVisualStateId>
  mirrorSafeStates?: Iterable<PersonaVisualStateId>
  horizontalBounds?: { min: number; max: number }
  timing?: PersonaCompanionTiming
}

export type PersonaCompanionSnapshot = {
  generation: number
  phase: "idle" | "action"
  requestedState: PersonaVisualStateId
  facing: "left" | "right"
  transientOffsetX: number
  suspension:
    | "none"
    | "semantic"
    | "hidden"
    | "controls"
    | "focus"
    | "drag"
    | "reduced_motion"
    | "surface"
}

export type PersonaVisualLeaseSource = string

export type PersonaVisualStateLease = {
  source: PersonaVisualLeaseSource
  state: PersonaVisualStateId
  release: () => void
}

export type PersonaCompanionReactionTrigger = "click" | "space" | "drag"

export interface PersonaCompanionEngine {
  update(input: PersonaCompanionInput): void
  react(trigger: PersonaCompanionReactionTrigger): boolean
  acquireLease(
    source: PersonaVisualLeaseSource,
    state: PersonaVisualStateId,
    ttlMs: number
  ): PersonaVisualStateLease
  completeAction(succeeded: boolean): void
  getSnapshot(): PersonaCompanionSnapshot
  subscribe(listener: () => void): () => void
  dispose(): void
}

type NormalizedEntry = PersonaCompanionBehaviorEntry & {
  suggested_weight: number
  suggested_cooldown_ms: number
}

type NormalizedInput = Omit<
  PersonaCompanionInput,
  | "behavior"
  | "availableStates"
  | "mirrorSafeStates"
  | "horizontalBounds"
  | "timing"
> & {
  personaId: string | null
  packId: string | null
  packRevision: string | number | null
  semanticState: PersonaVisualStateId
  entries: NormalizedEntry[]
  availableStates: Set<PersonaVisualStateId>
  mirrorSafeStates: Set<PersonaVisualStateId>
  horizontalBounds: { min: number; max: number }
  timing: Required<PersonaCompanionTiming>
  signature: string
}

type LeaseRecord = {
  token: number
  state: PersonaVisualStateId
  expiresAt: number
}

type PreparedMove = {
  entry: NormalizedEntry
  desiredFacing: "left" | "right"
  targetOffsetX: number
}

type ActiveAction = {
  entry: NormalizedEntry
  trigger: "ambient" | "click" | "drag"
  pendingMove?: PreparedMove
}

const AMBIENT_MIN_MS = 30_000
const AMBIENT_MAX_MS = 90_000
const ACTION_MIN_MS = 150
const ACTION_MAX_MS = 8_000
const COOLDOWN_MAX_MS = 86_400_000
const LEASE_MIN_MS = 150
const LEASE_MAX_MS = 86_400_000
const DEFAULT_ACTION_MS = 1_000
const DEFAULT_MOVEMENT_PX = 48

const clampNumber = (
  value: unknown,
  minimum: number,
  maximum: number,
  fallback: number
): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, value))
    : fallback

const randomUnit = (random: () => number): number => {
  const value = random()
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(1 - Number.EPSILON, Math.max(0, value))
    : 0
}

const normalizeInput = (input: PersonaCompanionInput): NormalizedInput => {
  const mode = resolveEffectiveAmbientMode({
    persona: input.mode,
    readFailed: false,
    surface: input.surface
  })
  const rawMin = clampNumber(
    input.timing?.ambientMinMs,
    AMBIENT_MIN_MS,
    AMBIENT_MAX_MS,
    AMBIENT_MIN_MS
  )
  const rawMax = clampNumber(
    input.timing?.ambientMaxMs,
    AMBIENT_MIN_MS,
    AMBIENT_MAX_MS,
    AMBIENT_MAX_MS
  )
  const ambientMinMs = Math.min(rawMin, rawMax)
  const ambientMaxMs = Math.max(rawMin, rawMax)
  const rawBounds = input.horizontalBounds
  const firstBound = clampNumber(rawBounds?.min, -1_000_000, 1_000_000, 0)
  const secondBound = clampNumber(rawBounds?.max, -1_000_000, 1_000_000, 0)
  const horizontalBounds = {
    min: Math.min(firstBound, secondBound),
    max: Math.max(firstBound, secondBound)
  }
  const movementLimit = horizontalBounds.max - horizontalBounds.min
  const timing = {
    ambientMinMs,
    ambientMaxMs,
    actionDurationMs: clampNumber(
      input.timing?.actionDurationMs,
      ACTION_MIN_MS,
      ACTION_MAX_MS,
      DEFAULT_ACTION_MS
    ),
    movementDistancePx: clampNumber(
      input.timing?.movementDistancePx,
      0,
      movementLimit,
      Math.min(DEFAULT_MOVEMENT_PX, movementLimit)
    )
  }
  const entries = (input.behavior?.entries ?? []).map((entry) => ({
    ...entry,
    suggested_weight: clampNumber(entry.suggested_weight, 0, 1_000_000, 1),
    suggested_cooldown_ms: clampNumber(
      entry.suggested_cooldown_ms,
      0,
      COOLDOWN_MAX_MS,
      0
    ),
    ...(entry.movement
      ? {
          movement: {
            direction: "horizontal" as const,
            motion_start_ratio: clampNumber(
              entry.movement.motion_start_ratio,
              0,
              1,
              0
            ),
            motion_end_ratio: clampNumber(
              entry.movement.motion_end_ratio,
              0,
              1,
              1
            )
          }
        }
      : {})
  }))
  for (const entry of entries) {
    if (entry.movement) {
      entry.movement.motion_end_ratio = Math.max(
        entry.movement.motion_start_ratio,
        entry.movement.motion_end_ratio
      )
    }
  }
  const availableStates = new Set(
    input.availableStates ?? entries.map((entry) => entry.state)
  )
  const mirrorSafeStates = new Set(input.mirrorSafeStates ?? [])
  const serializable = {
    ...input,
    mode,
    behavior: entries,
    availableStates: [...availableStates].sort(),
    mirrorSafeStates: [...mirrorSafeStates].sort(),
    horizontalBounds,
    timing
  }

  return {
    ...input,
    mode,
    personaId: input.personaId ?? null,
    packId: input.packId ?? null,
    packRevision: input.packRevision ?? null,
    semanticState: input.semanticState ?? "idle",
    entries,
    availableStates,
    mirrorSafeStates,
    horizontalBounds,
    timing,
    signature: JSON.stringify(serializable)
  }
}

const semanticPriority = (state: PersonaVisualStateId): number => {
  if (state === "error") return 90
  if (state === "approval_needed") return 80
  if (state === "offline") return 70
  if (state === "wake_armed") return 60
  if (state === "listening") return 50
  if (state === "thinking") return 40
  if (state === "speaking") return 30
  if (state === "tool_running") return 20
  return state === "idle" ? 0 : 20
}

const createDefaultRuntime = (): PersonaCompanionRuntime => ({
  now: () => performance.now(),
  random: Math.random,
  setTimer: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimer: (timer) => clearTimeout(timer as ReturnType<typeof setTimeout>)
})

export const createPersonaCompanionEngine = (
  runtime: PersonaCompanionRuntime = createDefaultRuntime()
): PersonaCompanionEngine => {
  let input = normalizeInput({
    mode: "off",
    surface: "web",
    visibility: "visible",
    controlsOpen: false,
    focusWithin: false,
    dragging: false,
    reducedMotion: false
  })
  let hasInput = false
  let disposed = false
  let generation = 0
  let leaseToken = 0
  let scheduledTimer: PersonaCompanionTimer | null = null
  let ambientDueAt: number | null = null
  let actionDueAt: number | null = null
  let currentAction: ActiveAction | null = null
  let lastAmbientState: PersonaVisualStateId | null = null
  let transientOffsetX = 0
  let facing: "left" | "right" = "right"
  const listeners = new Set<() => void>()
  const leases = new Map<PersonaVisualLeaseSource, LeaseRecord>()
  const lastSelectedAt = new Map<PersonaVisualStateId, number>()
  let snapshot: PersonaCompanionSnapshot = {
    generation,
    phase: "idle",
    requestedState: "idle",
    facing,
    transientOffsetX,
    suspension: "none"
  }

  const diagnose = (event: PersonaCompanionDiagnosticEvent) => {
    runtime.diagnose?.(
      createPersonaCompanionDiagnostic({
        ...event,
        personaId: input.personaId ?? undefined,
        packId: input.packId ?? undefined
      })
    )
  }

  const clearScheduledTimer = () => {
    if (scheduledTimer !== null) runtime.clearTimer(scheduledTimer)
    scheduledTimer = null
  }

  const advanceGeneration = () => {
    generation += 1
    clearScheduledTimer()
  }

  const notifySnapshot = (
    next: Omit<PersonaCompanionSnapshot, "generation">
  ) => {
    const changed =
      snapshot.generation !== generation ||
      snapshot.phase !== next.phase ||
      snapshot.requestedState !== next.requestedState ||
      snapshot.facing !== next.facing ||
      snapshot.transientOffsetX !== next.transientOffsetX ||
      snapshot.suspension !== next.suspension
    if (!changed) return
    snapshot = { generation, ...next }
    listeners.forEach((listener) => listener())
  }

  const expireLeases = (): boolean => {
    const now = runtime.now()
    let expired = false
    for (const [source, lease] of leases) {
      if (lease.expiresAt <= now) {
        leases.delete(source)
        expired = true
      }
    }
    return expired
  }

  const winningSemanticState = (): PersonaVisualStateId => {
    let winning = input.semanticState
    for (const lease of leases.values()) {
      if (semanticPriority(lease.state) > semanticPriority(winning)) {
        winning = lease.state
      }
    }
    return winning
  }

  const suspensionFor = (semanticState: PersonaVisualStateId) => {
    if (semanticState !== "idle") return "semantic" as const
    if (input.visibility === "hidden") return "hidden" as const
    if (input.controlsOpen) return "controls" as const
    if (input.focusWithin) return "focus" as const
    if (input.dragging) return "drag" as const
    if (input.reducedMotion) return "reduced_motion" as const
    if (input.surface !== "web" && input.surface !== "sidepanel") {
      return "surface" as const
    }
    return "none" as const
  }

  const nextAmbientInterval = (): number => {
    const range = input.timing.ambientMaxMs - input.timing.ambientMinMs
    return input.timing.ambientMinMs + Math.floor(randomUnit(runtime.random) * range)
  }

  const ensureAmbientDue = () => {
    if (ambientDueAt === null) ambientDueAt = runtime.now() + nextAmbientInterval()
  }

  const chooseWeighted = (candidates: NormalizedEntry[]): NormalizedEntry => {
    const total = candidates.reduce(
      (sum, candidate) => sum + candidate.suggested_weight,
      0
    )
    if (total <= 0) return candidates[0]
    let target = randomUnit(runtime.random) * total
    for (const candidate of candidates) {
      target -= candidate.suggested_weight
      if (target < 0) return candidate
    }
    return candidates[candidates.length - 1]
  }

  const findTurnEntry = (
    desiredFacing: "left" | "right"
  ): NormalizedEntry | null =>
    input.entries.find(
      (entry) =>
        entry.trigger === "ambient" &&
        entry.state === `ambient.turn.${desiredFacing}` &&
        input.availableStates.has(entry.state)
    ) ?? null

  const startPreparedMove = (
    prepared: PreparedMove,
    trigger: "ambient" | "click" | "drag"
  ) => {
    if (
      prepared.desiredFacing === facing ||
      input.mirrorSafeStates.has(prepared.entry.state)
    ) {
      facing = prepared.desiredFacing
    }
    transientOffsetX = prepared.targetOffsetX
    currentAction = { entry: prepared.entry, trigger }
    actionDueAt = runtime.now() + input.timing.actionDurationMs
    advanceGeneration()
  }

  const startEntry = (
    entry: NormalizedEntry,
    trigger: "ambient" | "click" | "drag"
  ) => {
    if (entry.category !== "move") {
      currentAction = { entry, trigger }
      actionDueAt = runtime.now() + input.timing.actionDurationMs
      advanceGeneration()
      return
    }

    const leftRoom = transientOffsetX - input.horizontalBounds.min
    const rightRoom = input.horizontalBounds.max - transientOffsetX
    const desiredFacing: "left" | "right" =
      leftRoom > 0 && rightRoom > 0
        ? randomUnit(runtime.random) < 0.5
          ? "left"
          : "right"
        : leftRoom > 0
          ? "left"
          : "right"
    const signedDistance = desiredFacing === "left" ? -1 : 1
    const targetOffsetX = clampNumber(
      transientOffsetX + signedDistance * input.timing.movementDistancePx,
      input.horizontalBounds.min,
      input.horizontalBounds.max,
      transientOffsetX
    )
    const prepared = { entry, desiredFacing, targetOffsetX }
    const turn = desiredFacing !== facing ? findTurnEntry(desiredFacing) : null
    if (turn) {
      currentAction = { entry: turn, trigger, pendingMove: prepared }
      actionDueAt = runtime.now() + input.timing.actionDurationMs
      advanceGeneration()
      return
    }
    startPreparedMove(prepared, trigger)
  }

  const ambientCandidates = () =>
    input.entries.filter(
      (entry) =>
        entry.trigger === "ambient" &&
        !String(entry.state).startsWith("ambient.turn.") &&
        input.availableStates.has(entry.state) &&
        entry.suggested_weight > 0 &&
        (input.mode === "roaming" || entry.category !== "move")
    )

  const startAmbient = () => {
    const all = ambientCandidates()
    if (all.length === 0) {
      diagnose({ event: "ambient_skipped", failureClass: "empty_set" })
      ensureAmbientDue()
      return
    }
    const now = runtime.now()
    let eligible = all.filter(
      (entry) =>
        now - (lastSelectedAt.get(entry.state) ?? Number.NEGATIVE_INFINITY) >=
        entry.suggested_cooldown_ms
    )
    if (eligible.length === 0) {
      diagnose({ event: "ambient_skipped", failureClass: "cooldown" })
      ensureAmbientDue()
      return
    }
    if (
      lastAmbientState &&
      eligible.length > 1 &&
      eligible.some((entry) => entry.state !== lastAmbientState)
    ) {
      eligible = eligible.filter((entry) => entry.state !== lastAmbientState)
    }
    const selected = chooseWeighted(eligible)
    lastAmbientState = selected.state
    lastSelectedAt.set(selected.state, now)
    diagnose({ event: "ambient_selected", state: selected.state })
    startEntry(selected, "ambient")
  }

  const finishAction = (succeeded: boolean) => {
    if (!currentAction) return
    const completed = currentAction
    currentAction = null
    actionDueAt = null
    advanceGeneration()
    if (completed.pendingMove) {
      if (
        succeeded ||
        input.mirrorSafeStates.has(completed.pendingMove.entry.state)
      ) {
        facing = completed.pendingMove.desiredFacing
      }
      startPreparedMove(completed.pendingMove, completed.trigger)
      return
    }
    ambientDueAt = null
  }

  const scheduleNext = () => {
    clearScheduledTimer()
    if (disposed) return
    const dueTimes = [
      actionDueAt,
      ambientDueAt,
      ...[...leases.values()].map((lease) => lease.expiresAt)
    ].filter((due): due is number => due !== null && Number.isFinite(due))
    if (dueTimes.length === 0) return
    const due = Math.min(...dueTimes)
    const scheduledGeneration = generation
    scheduledTimer = runtime.setTimer(() => {
      scheduledTimer = null
      if (disposed || scheduledGeneration !== generation) {
        diagnose({
          event: "stale_generation",
          failureClass: "stale_timer"
        })
        return
      }
      const now = runtime.now()
      if (expireLeases()) advanceGeneration()
      if (actionDueAt !== null && actionDueAt <= now) {
        finishAction(!currentAction?.pendingMove)
      }
      if (ambientDueAt !== null && ambientDueAt <= now && !currentAction) {
        ambientDueAt = null
        startAmbient()
      }
      settle()
    }, Math.max(0, due - runtime.now()))
  }

  const settle = () => {
    const semanticState = winningSemanticState()
    const suspension = suspensionFor(semanticState)
    if (suspension !== "none") {
      ambientDueAt = null
      notifySnapshot({
        phase: "idle",
        requestedState: semanticState,
        facing,
        transientOffsetX,
        suspension
      })
      scheduleNext()
      return
    }
    if (currentAction) {
      notifySnapshot({
        phase: "action",
        requestedState: currentAction.entry.state,
        facing,
        transientOffsetX,
        suspension: "none"
      })
      scheduleNext()
      return
    }
    if (input.mode !== "off") ensureAmbientDue()
    notifySnapshot({
      phase: "idle",
      requestedState: "idle",
      facing,
      transientOffsetX,
      suspension: "none"
    })
    scheduleNext()
  }

  const cancelActionForInputChange = () => {
    if (currentAction?.trigger === "ambient") {
      diagnose({
        event: "ambient_preempted",
        state: currentAction.pendingMove?.entry.state ?? currentAction.entry.state,
        failureClass: "preempted"
      })
    }
    currentAction = null
    actionDueAt = null
    ambientDueAt = null
  }

  return {
    update(nextInput) {
      if (disposed) return
      const normalized = normalizeInput(nextInput)
      if (hasInput && normalized.signature === input.signature) return
      const identityChanged =
        hasInput &&
        (normalized.personaId !== input.personaId ||
          normalized.packId !== input.packId ||
          normalized.packRevision !== input.packRevision ||
          normalized.surface !== input.surface)
      if (currentAction) cancelActionForInputChange()
      if (identityChanged) transientOffsetX = 0
      input = normalized
      hasInput = true
      transientOffsetX = clampNumber(
        transientOffsetX,
        input.horizontalBounds.min,
        input.horizontalBounds.max,
        0
      )
      advanceGeneration()
      settle()
    },
    react(trigger) {
      if (disposed) return false
      if (expireLeases()) advanceGeneration()
      const semantic = winningSemanticState()
      if (
        semantic !== "idle" ||
        suspensionFor(semantic) !== "none" ||
        currentAction
      ) {
        settle()
        return false
      }
      const authoredTrigger = trigger === "space" ? "click" : trigger
      const candidates = input.entries.filter(
        (entry) =>
          entry.trigger === authoredTrigger &&
          input.availableStates.has(entry.state) &&
          entry.suggested_weight > 0
      )
      if (candidates.length === 0) return false
      ambientDueAt = null
      startEntry(chooseWeighted(candidates), authoredTrigger)
      settle()
      return true
    },
    acquireLease(source, state, ttlMs) {
      const token = ++leaseToken
      if (!disposed) {
        leases.set(source, {
          token,
          state,
          expiresAt:
            runtime.now() +
            clampNumber(ttlMs, LEASE_MIN_MS, LEASE_MAX_MS, LEASE_MIN_MS)
        })
        if (currentAction) cancelActionForInputChange()
        advanceGeneration()
        settle()
      }
      let released = false
      return {
        source,
        state,
        release() {
          if (released || disposed) return
          released = true
          if (leases.get(source)?.token !== token) return
          leases.delete(source)
          advanceGeneration()
          settle()
        }
      }
    },
    completeAction(succeeded) {
      if (disposed || !currentAction) return
      finishAction(succeeded)
      settle()
    },
    getSnapshot: () => snapshot,
    subscribe(listener) {
      if (disposed) return () => undefined
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    dispose() {
      if (disposed) return
      disposed = true
      clearScheduledTimer()
      leases.clear()
      listeners.clear()
      currentAction = null
      ambientDueAt = null
      actionDueAt = null
    }
  }
}
