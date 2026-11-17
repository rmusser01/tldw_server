import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist } from "zustand/middleware"

import type { ActorSettingsTarget } from "@/utils/actor-settings-target"
import type { ActorEditorMode, ActorSettings } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"

export type ActorAppliedScene = {
  target: ActorSettingsTarget
  settings: ActorSettings
  preview: string
  tokenCount: number
}

export const actorSceneForTarget = (
  applied: ActorAppliedScene | null,
  target: ActorSettingsTarget | null
): ActorAppliedScene | null =>
  applied &&
  target &&
  applied.target.key === target.key &&
  applied.target.isCurrent() &&
  target.isCurrent()
    ? applied
    : null

type ActorUiStoreState = {
  applied: ActorAppliedScene | null
  applyScene: (scene: ActorAppliedScene) => boolean
  /**
   * Current chat's Actor settings in memory.
   * Persisted per-chat via services/actor-settings.
   */
  settings: ActorSettings | null
  /**
   * Cached preview text built from settings.
   */
  preview: string
  /**
   * Rough token estimate for the preview.
   */
  tokenCount: number
  /**
   * Replace current settings.
   */
  setSettings: (next: ActorSettings | null) => void
  /**
   * Update preview + token count together.
   */
  setPreviewAndTokens: (preview: string, tokenCount: number) => void
  /**
   * Reset in-memory Actor UI state.
   */
  reset: () => void
}

export const useActorStore = createWithEqualityFn<ActorUiStoreState>(
  (set, get) => ({
    applied: null,
    applyScene: (scene) => {
      if (!scene.target.isCurrent()) return false
      const applied = { ...scene, settings: structuredClone(scene.settings) }
      set({
        applied,
        settings: applied.settings,
        preview: applied.preview,
        tokenCount: applied.tokenCount
      })
      return scene.target.isCurrent() && get().applied === applied
    },
    settings: null,
    preview: "",
    tokenCount: 0,
    setSettings: (next: ActorSettings | null) => {
      set((state) =>
        state.settings === next && !state.applied
          ? state
          : { settings: next, applied: null }
      )
    },
    setPreviewAndTokens: (preview: string, tokenCount: number) =>
      set((state) =>
        state.preview === preview &&
        state.tokenCount === tokenCount &&
        !state.applied
          ? state
          : { preview, tokenCount, applied: null }
      ),
    reset: () =>
      set({
        applied: null,
        settings: createDefaultActorSettings(),
        preview: "",
        tokenCount: 0
      })
  })
)

/**
 * Persisted editor preferences for the Actor drawer.
 * Kept separate from runtime state to avoid persisting preview/tokens.
 */
type ActorEditorPrefsState = {
  editorMode: ActorEditorMode
  setEditorMode: (mode: ActorEditorMode) => void
}

export const useActorEditorPrefs = createWithEqualityFn<ActorEditorPrefsState>()(
  persist(
    (set) => ({
      editorMode: "simple", // Default to simple for new users
      setEditorMode: (mode) => set({ editorMode: mode })
    }),
    {
      name: "tldw-actor-editor-prefs",
      // Baseline version so future shape changes can migrate instead of discarding
      // persisted state (see apps/FRONTEND_AUDIT.md §6 / TASK-12102).
      version: 1,
      migrate: (persisted) => persisted as any,
      storage: createJSONStorage(() => localStorage)
    }
  )
)
