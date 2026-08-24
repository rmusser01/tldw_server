import { useEffect, useState, useSyncExternalStore } from "react"

import {
  createPersonaCompanionEngine,
  type PersonaCompanionEngine,
  type PersonaCompanionInput,
  type PersonaCompanionRuntime,
  type PersonaCompanionSnapshot
} from "./personaCompanionEngine"

export type UsePersonaCompanionInput = PersonaCompanionInput & {
  runtime?: PersonaCompanionRuntime
}

const INITIAL_SNAPSHOT: PersonaCompanionSnapshot = Object.freeze({
  generation: 0,
  phase: "idle",
  actionToken: null,
  requestedState: "idle",
  facing: "right",
  transientOffsetX: 0,
  suspension: "none"
})

const createHookStore = () => {
  let engine: PersonaCompanionEngine | null = null
  let snapshot = INITIAL_SNAPSHOT
  let unsubscribeEngine: (() => void) | null = null
  const listeners = new Set<() => void>()
  const publish = (next: PersonaCompanionSnapshot) => {
    if (next === snapshot) return
    snapshot = next
    listeners.forEach((listener) => listener())
  }

  return {
    attach(nextEngine: PersonaCompanionEngine) {
      unsubscribeEngine?.()
      engine = nextEngine
      unsubscribeEngine = nextEngine.subscribe(() =>
        publish(nextEngine.getSnapshot())
      )
    },
    detach(expectedEngine: PersonaCompanionEngine) {
      if (engine !== expectedEngine) return
      unsubscribeEngine?.()
      unsubscribeEngine = null
      engine = null
    },
    update(input: PersonaCompanionInput) {
      engine?.update(input)
    },
    subscribe(listener: () => void) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    getSnapshot: () => snapshot
  }
}

export const usePersonaCompanion = ({
  runtime,
  ...input
}: UsePersonaCompanionInput): PersonaCompanionSnapshot => {
  const [store] = useState(createHookStore)

  useEffect(() => {
    const engine = createPersonaCompanionEngine(runtime)
    store.attach(engine)
    return () => {
      store.detach(engine)
      engine.dispose()
    }
  }, [runtime, store])

  useEffect(() => store.update(input), [input, store])

  return useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    store.getSnapshot
  )
}
