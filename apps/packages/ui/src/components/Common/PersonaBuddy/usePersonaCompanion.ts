import { useEffect, useState, useSyncExternalStore } from "react"

import {
  createPersonaCompanionEngine,
  type PersonaCompanionInput,
  type PersonaCompanionRuntime,
  type PersonaCompanionSnapshot
} from "./personaCompanionEngine"

export type UsePersonaCompanionInput = PersonaCompanionInput & {
  runtime?: PersonaCompanionRuntime
}

export const usePersonaCompanion = ({
  runtime,
  ...input
}: UsePersonaCompanionInput): PersonaCompanionSnapshot => {
  const [engine] = useState(() => createPersonaCompanionEngine(runtime))

  useEffect(() => {
    engine.update(input)
  }, [engine, input])

  useEffect(() => () => engine.dispose(), [engine])

  return useSyncExternalStore(
    engine.subscribe,
    engine.getSnapshot,
    engine.getSnapshot
  )
}
