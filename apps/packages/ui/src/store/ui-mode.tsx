import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

export type UiMode = "casual" | "pro"

type UiModeState = {
  mode: UiMode
  setMode: (mode: UiMode) => void
  toggleMode: () => void
}

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

export const useUiModeStore = createWithEqualityFn<UiModeState>()(
  persist(
    (set, get) => ({
      mode: "casual",
      setMode: (mode) => set({ mode }),
      toggleMode: () =>
        set({ mode: get().mode === "pro" ? "casual" : "pro" })
    }),
    {
      name: "tldw-ui-mode",
      // Baseline version so a future shape change can migrate instead of silently
      // discarding persisted state (see apps/FRONTEND_AUDIT.md §6 / TASK-12102).
      version: 1,
      migrate: (persisted) => persisted as any,
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? localStorage : createMemoryStorage()
      )
    }
  )
)
