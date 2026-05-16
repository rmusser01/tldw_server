import { create } from "zustand"

import type { PersonaVisualDiagnostic } from "@/components/Common/PersonaBuddy/personaVisualDiagnostics"
import type {
  PersonaVisualStateId
} from "@/types/persona-visuals"

export interface PersonaVisualRuntimeOverride {
  personaId: string
  sessionId: string | null
  state: PersonaVisualStateId
  reason: string | null
  expiresAt: number
}

export interface PersonaVisualRuntimeDiagnostics {
  sourceId?: string
  personaId: string
  sessionId: string | null
  packId: string | null
  packTitle: string | null
  packLoadStatus: "idle" | "loading" | "loaded" | "error"
  visualState: PersonaVisualStateId
  diagnostic: PersonaVisualDiagnostic | null
  updatedAt: number
}

type PersonaVisualRuntimeStore = {
  override: PersonaVisualRuntimeOverride | null
  runtimeDiagnostics: PersonaVisualRuntimeDiagnostics | null
  setOverride: (override: PersonaVisualRuntimeOverride) => void
  setRuntimeDiagnostics: (
    diagnostics: PersonaVisualRuntimeDiagnostics | null
  ) => void
  clearRuntimeDiagnostics: (sourceId?: string) => void
  clearExpired: (now?: number) => void
  clearForSession: (sessionId: string | null) => void
}

export const usePersonaVisualRuntimeStore = create<PersonaVisualRuntimeStore>(
  (set, get) => ({
    override: null,
    runtimeDiagnostics: null,
    setOverride: (override) => set({ override }),
    setRuntimeDiagnostics: (runtimeDiagnostics) => set({ runtimeDiagnostics }),
    clearRuntimeDiagnostics: (sourceId) => {
      const diagnostics = get().runtimeDiagnostics
      if (
        diagnostics &&
        (!sourceId || !diagnostics.sourceId || diagnostics.sourceId === sourceId)
      ) {
        set({ runtimeDiagnostics: null })
      }
    },
    clearExpired: (now = Date.now()) => {
      const current = get().override
      if (current && current.expiresAt <= now) {
        set({ override: null })
      }
    },
    clearForSession: (sessionId) => {
      const current = get().override
      if (current && current.sessionId === sessionId) {
        set({ override: null })
      }
      const diagnostics = get().runtimeDiagnostics
      if (diagnostics && diagnostics.sessionId === sessionId) {
        set({ runtimeDiagnostics: null })
      }
    }
  })
)
