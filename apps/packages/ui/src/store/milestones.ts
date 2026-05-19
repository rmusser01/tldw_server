/**
 * Milestone Tracking Store
 *
 * Tracks key user milestones (first connection, first ingest, first chat,
 * first quiz) with timestamps. Persists to localStorage and supports
 * bootstrapping from pre-existing client-side state so returning users
 * get credit for actions they already performed.
 */

import { create } from "zustand"
import { FAMILY_GUARDRAILS_WIZARD_TELEMETRY_STORAGE_KEY } from "@/utils/family-guardrails-wizard-telemetry"

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

export const MILESTONES_STORAGE_KEY = "tldw:milestones"

/**
 * localStorage key used by onboarding-ingestion-telemetry.
 * We read this during bootstrap to infer first_ingest / first_chat.
 */
const ONBOARDING_TELEMETRY_KEY = "tldw:onboarding:ingestion:telemetry"

/**
 * localStorage key set after the initial setup/connection flow completes.
 */
const FIRST_RUN_COMPLETE_KEY = "__tldw_first_run_complete"
const MODERATION_ONBOARDING_KEY = "moderation-playground-onboarded"
const QUIZ_ATTEMPT_SCAN_DONE_KEY = "tldw:milestones:quiz-attempt-scan-done"

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type MilestoneId =
  | "first_connection"
  | "first_ingest"
  | "first_chat"
  | "first_quiz_taken"
  | "family_profiles_created"
  | "content_rules_reviewed"
  | "content_rules_tested"

type MilestoneState = {
  completedMilestones: Partial<Record<MilestoneId, number>>
  markMilestone: (id: MilestoneId) => void
  isMilestoneCompleted: (id: MilestoneId) => boolean
  getCompletedCount: () => number
  bootstrapFromExistingUsage: () => void
  resetMilestones: () => void
}

// ─────────────────────────────────────────────────────────────────────────────
// localStorage helpers (SSR-safe)
// ─────────────────────────────────────────────────────────────────────────────

const loadPersistedMilestones = (): Partial<Record<MilestoneId, number>> => {
  if (typeof window === "undefined") return {}
  try {
    const raw = localStorage.getItem(MILESTONES_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Partial<Record<MilestoneId, number>>
    }
    return {}
  } catch {
    return {}
  }
}

const persistMilestones = (
  milestones: Partial<Record<MilestoneId, number>>
): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(MILESTONES_STORAGE_KEY, JSON.stringify(milestones))
  } catch {
    // localStorage may be full or unavailable; silently ignore
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useMilestoneStore = create<MilestoneState>()((set, get) => ({
  completedMilestones: loadPersistedMilestones(),

  markMilestone: (id: MilestoneId) => {
    const current = get().completedMilestones
    if (current[id] != null) return // already completed — no-op

    const updated = { ...current, [id]: Date.now() }
    persistMilestones(updated)
    set({ completedMilestones: updated })
  },

  isMilestoneCompleted: (id: MilestoneId) => {
    return get().completedMilestones[id] != null
  },

  getCompletedCount: () => {
    return Object.keys(get().completedMilestones).length
  },

  bootstrapFromExistingUsage: () => {
    const current = get().completedMilestones
    const updates: Partial<Record<MilestoneId, number>> = {}
    const now = Date.now()

    // ── first_connection ──────────────────────────────────────────────
    if (current.first_connection == null) {
      try {
        if (localStorage.getItem(FIRST_RUN_COMPLETE_KEY) === "true") {
          updates.first_connection = now
        }
      } catch {
        // ignore
      }
    }

    // ── first_ingest & first_chat (from onboarding telemetry) ────────
    try {
      const raw = localStorage.getItem(ONBOARDING_TELEMETRY_KEY)
      if (raw) {
        const telemetry = JSON.parse(raw)

        if (current.first_ingest == null && telemetry?.current_session?.first_ingest_at != null) {
          updates.first_ingest = telemetry.current_session.first_ingest_at
        }
        // Also check the counter as a fallback — the session may have been
        // reset but the counter persists across sessions.
        if (
          current.first_ingest == null &&
          updates.first_ingest == null &&
          typeof telemetry?.counters?.onboarding_first_ingest_success === "number" &&
          telemetry.counters.onboarding_first_ingest_success > 0
        ) {
          updates.first_ingest = now
        }

        if (
          current.first_chat == null &&
          telemetry?.current_session?.first_chat_after_ingest_at != null
        ) {
          updates.first_chat = telemetry.current_session.first_chat_after_ingest_at
        }
        if (
          current.first_chat == null &&
          updates.first_chat == null &&
          typeof telemetry?.counters?.onboarding_first_chat_after_ingest === "number" &&
          telemetry.counters.onboarding_first_chat_after_ingest > 0
        ) {
          updates.first_chat = now
        }
      }
    } catch {
      // ignore malformed telemetry
    }

    // ── family mission milestones ────────────────────────────────────
    if (current.family_profiles_created == null) {
      try {
        const raw = localStorage.getItem(FAMILY_GUARDRAILS_WIZARD_TELEMETRY_STORAGE_KEY)
        if (raw) {
          const telemetry = JSON.parse(raw)
          if (
            typeof telemetry?.counters?.setup_completed === "number" &&
            telemetry.counters.setup_completed > 0
          ) {
            updates.family_profiles_created =
              typeof telemetry?.last_event_at === "number" ? telemetry.last_event_at : now
          }
        }
      } catch {
        // ignore malformed family telemetry
      }
    }

    if (current.content_rules_reviewed == null) {
      try {
        if (localStorage.getItem(MODERATION_ONBOARDING_KEY) === "true") {
          updates.content_rules_reviewed = now
        }
      } catch {
        // ignore
      }
    }

    // ── first_quiz_taken ─────────────────────────────────────────────
    if (current.first_quiz_taken == null) {
      try {
        if (localStorage.getItem(QUIZ_ATTEMPT_SCAN_DONE_KEY) !== "1") {
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i)
            if (key && key.startsWith("quiz-attempt-")) {
              updates.first_quiz_taken = now
              break
            }
          }
          localStorage.setItem(QUIZ_ATTEMPT_SCAN_DONE_KEY, "1")
        }
      } catch {
        // ignore
      }
    }

    // Only mutate if we found new milestones
    if (Object.keys(updates).length > 0) {
      const merged = { ...current, ...updates }
      persistMilestones(merged)
      set({ completedMilestones: merged })
    }
  },

  resetMilestones: () => {
    try {
      localStorage.removeItem(QUIZ_ATTEMPT_SCAN_DONE_KEY)
    } catch {
      // ignore
    }
    persistMilestones({})
    set({ completedMilestones: {} })
  }
}))

// Expose for debugging in non-production builds
if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useMilestoneStore = useMilestoneStore
}
