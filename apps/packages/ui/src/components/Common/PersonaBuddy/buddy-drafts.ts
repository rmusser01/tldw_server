import type React from "react"

/** Reply drafts and ambiguous request identities exist only in one authenticated shell. */
export type BuddyDraftState = {
  drafts: Record<string, string>
  setDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>
  requestKeys: React.MutableRefObject<
    Map<string, { text: string; key: string }>
  >
}
