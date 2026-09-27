import { create } from "zustand"

export type BuddyManagementTarget = {
  scope_type: "conversation" | "workspace"
  scope_id: string
}
type BuddyManagementState = {
  open: boolean
  target: BuddyManagementTarget | null
  attached: boolean
  conversationSettings: (() => void) | null
  show: (
    target?: BuddyManagementTarget | null,
    conversationSettings?: (() => void) | null
  ) => void
  close: () => void
  setAttached: (attached: boolean) => void
}
// Sensitive conversation/profile data stays in the authenticated shell lifetime.
// Only the server owns the saved attachment; no global browser identity cache.
export const useBuddyManagementStore = create<BuddyManagementState>((set) => ({
  open: false,
  target: null,
  attached: false,
  conversationSettings: null,
  show: (target = null, conversationSettings = null) =>
    set({ open: true, target, conversationSettings }),
  close: () => set({ open: false, target: null, conversationSettings: null }),
  setAttached: (attached) => set({ attached })
}))
