import { create } from "zustand"

import type { PersonaVisualStateId } from "@/types/persona-visuals"

export interface PersonaVisualRuntimeOverride {
  personaId: string
  sessionId: string | null
  state: PersonaVisualStateId
  reason: string | null
  expiresAt: number
}

type PersonaVisualRuntimeStore = {
  override: PersonaVisualRuntimeOverride | null
  setOverride: (override: PersonaVisualRuntimeOverride) => void
  clearExpired: (now?: number) => void
  clearForSession: (sessionId: string | null) => void
}

export const usePersonaVisualRuntimeStore = create<PersonaVisualRuntimeStore>(
  (set, get) => ({
    override: null,
    setOverride: (override) => set({ override }),
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
    }
  })
)
