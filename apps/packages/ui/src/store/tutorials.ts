/**
 * Tutorials Zustand Store
 * Manages state for the per-page tutorial system with React Joyride
 */

import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

// ─────────────────────────────────────────────────────────────────────────────
// Storage Configuration
// ─────────────────────────────────────────────────────────────────────────────

const STORAGE_KEY = "tldw-tutorials"

/**
 * Creates a memory storage fallback for SSR environments
 */
const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

/**
 * Creates localStorage-backed storage with SSR safety
 */
const createTutorialStorage = (): StateStorage => {
  if (typeof window === "undefined") {
    return createMemoryStorage()
  }

  return {
    getItem: (name: string): string | null => {
      return localStorage.getItem(name)
    },
    setItem: (name: string, value: string): void => {
      localStorage.setItem(name, value)
    },
    removeItem: (name: string): void => {
      localStorage.removeItem(name)
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// State Types
// ─────────────────────────────────────────────────────────────────────────────

interface TutorialPersistedState {
  /** IDs of tutorials that the user has completed */
  completedTutorials: string[]
  /** Page paths where the user has seen the first-visit tutorial prompt */
  seenPromptPages: string[]
}

export type TutorialFocusReturn = {
  target: HTMLElement | null
  route: string
  isCurrent: () => boolean
  lastFocusedControl?: () => HTMLElement | null
  dispose?: () => void
}
export type TutorialRun = { id: number; focusReturn?: TutorialFocusReturn }
let nextTutorialRunId = 0

interface TutorialRuntimeState {
  /** Currently active tutorial ID, null if no tutorial is running */
  activeTutorialId: string | null
  activeRun: TutorialRun | null
  /** Current step index in the active tutorial */
  activeStepIndex: number
  /** Whether the help modal is open */
  isHelpModalOpen: boolean
}

interface TutorialActions {
  // Completion tracking (persisted)
  /** Mark a tutorial as completed */
  markComplete: (tutorialId: string, expectedRun?: TutorialRun) => void
  /** Check if a tutorial has been completed */
  isCompleted: (tutorialId: string) => boolean

  // First-visit prompt tracking (persisted)
  /** Check if the user has seen the tutorial prompt for a page */
  hasSeenPromptForPage: (pageKey: string) => boolean
  /** Mark the tutorial prompt as seen for a page */
  markPromptSeen: (pageKey: string) => void

  // Active tutorial state (not persisted)
  /** Start a tutorial by its ID */
  startTutorial: (tutorialId: string, focusReturn?: TutorialFocusReturn) => void
  /** End the current tutorial (completed or skipped) */
  endTutorial: (expectedRun?: TutorialRun) => void
  /** Set the current step index */
  setStepIndex: (index: number, expectedRun?: TutorialRun) => void

  // Help modal state (not persisted)
  /** Open the help modal */
  openHelpModal: () => void
  /** Close the help modal */
  closeHelpModal: () => void
  /** Toggle the help modal */
  toggleHelpModal: () => void

  // Reset
  /** Reset all tutorial progress (completion and seen prompts) */
  resetProgress: () => void
}

export type TutorialState = TutorialPersistedState &
  TutorialRuntimeState &
  TutorialActions

// ─────────────────────────────────────────────────────────────────────────────
// Initial State
// ─────────────────────────────────────────────────────────────────────────────

const initialPersistedState: TutorialPersistedState = {
  completedTutorials: [],
  seenPromptPages: []
}

const initialRuntimeState: TutorialRuntimeState = {
  activeTutorialId: null,
  activeRun: null,
  activeStepIndex: 0,
  isHelpModalOpen: false
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useTutorialStore = createWithEqualityFn<TutorialState>()(
  persist<TutorialState, [], [], TutorialPersistedState>(
    (set, get) => ({
      ...initialPersistedState,
      ...initialRuntimeState,

      // ─────────────────────────────────────────────────────────────────────
      // Completion Tracking Actions
      // ─────────────────────────────────────────────────────────────────────

      markComplete: (tutorialId, expectedRun) =>
        set((state) => {
          if (
            (expectedRun && state.activeRun !== expectedRun) ||
            state.completedTutorials.includes(tutorialId)
          ) {
            return state
          }
          return {
            completedTutorials: [...state.completedTutorials, tutorialId]
          }
        }),

      isCompleted: (tutorialId) => {
        return get().completedTutorials.includes(tutorialId)
      },

      // ─────────────────────────────────────────────────────────────────────
      // First-Visit Prompt Actions
      // ─────────────────────────────────────────────────────────────────────

      hasSeenPromptForPage: (pageKey) => {
        return get().seenPromptPages.includes(pageKey)
      },

      markPromptSeen: (pageKey) =>
        set((state) => {
          if (state.seenPromptPages.includes(pageKey)) {
            return state
          }
          return {
            seenPromptPages: [...state.seenPromptPages, pageKey]
          }
        }),

      // ─────────────────────────────────────────────────────────────────────
      // Active Tutorial Actions
      // ─────────────────────────────────────────────────────────────────────

      startTutorial: (tutorialId, focusReturn) => {
        get().activeRun?.focusReturn?.dispose?.()
        set({
          activeTutorialId: tutorialId,
          activeRun: { id: ++nextTutorialRunId, focusReturn },
          activeStepIndex: 0,
          isHelpModalOpen: false // Close help modal when starting a tutorial
        })
      },

      endTutorial: (expectedRun) => {
        if (expectedRun && get().activeRun !== expectedRun) return
        get().activeRun?.focusReturn?.dispose?.()
        set({ activeTutorialId: null, activeRun: null, activeStepIndex: 0 })
      },

      setStepIndex: (index, expectedRun) =>
        set((state) =>
          expectedRun && state.activeRun !== expectedRun
            ? state
            : {
                activeStepIndex: index
              }
        ),

      // ─────────────────────────────────────────────────────────────────────
      // Help Modal Actions
      // ─────────────────────────────────────────────────────────────────────

      openHelpModal: () =>
        set({
          isHelpModalOpen: true
        }),

      closeHelpModal: () =>
        set({
          isHelpModalOpen: false
        }),

      toggleHelpModal: () =>
        set((state) => ({
          isHelpModalOpen: !state.isHelpModalOpen
        })),

      // ─────────────────────────────────────────────────────────────────────
      // Reset Actions
      // ─────────────────────────────────────────────────────────────────────

      resetProgress: () => {
        get().activeRun?.focusReturn?.dispose?.()
        set({ ...initialPersistedState, ...initialRuntimeState })
      }
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => createTutorialStorage()),
      // Only persist completion and seen prompt state, not runtime state
      partialize: (state): TutorialPersistedState => ({
        completedTutorials: state.completedTutorials,
        seenPromptPages: state.seenPromptPages
      })
    }
  )
)

// ─────────────────────────────────────────────────────────────────────────────
// Hooks for common operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hook to get the active tutorial state
 */
export const useActiveTutorial = () => {
  return useTutorialStore((state) => ({
    activeTutorialId: state.activeTutorialId,
    activeRun: state.activeRun,
    activeStepIndex: state.activeStepIndex,
    startTutorial: state.startTutorial,
    endTutorial: state.endTutorial,
    setStepIndex: state.setStepIndex,
    markComplete: state.markComplete
  }))
}

/**
 * Hook for the help modal state
 */
export const useHelpModal = () => {
  return useTutorialStore((state) => ({
    isOpen: state.isHelpModalOpen,
    open: state.openHelpModal,
    close: state.closeHelpModal,
    toggle: state.toggleHelpModal
  }))
}

/**
 * Hook to check tutorial completion status
 */
export const useTutorialCompletion = () => {
  return useTutorialStore((state) => ({
    completedTutorials: state.completedTutorials,
    isCompleted: state.isCompleted,
    markComplete: state.markComplete,
    resetProgress: state.resetProgress
  }))
}

// Expose for debugging in non-production builds
if (
  typeof window !== "undefined" &&
  process.env.NODE_ENV !== "production"
) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useTutorialStore = useTutorialStore
}
