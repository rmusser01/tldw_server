import { describe, expect, it } from "vitest"
import { isProtectedTrackedPersonaUnavailable } from "../assistant-selection"
import type { AssistantKind, AssistantSelectionMode } from "../assistant-selection"

describe("protected history persona choices", () => {
  it.each([
    { label: "tracked persona", protectedHistory: true, kind: "persona", mode: "tracked", unavailable: true },
    { label: "saved persona without an explicit mode", protectedHistory: true, kind: "persona", mode: null, unavailable: true },
    { label: "explicit persona overlay", protectedHistory: true, kind: "persona", mode: "overlay", unavailable: false },
    { label: "tracked character", protectedHistory: true, kind: "character", mode: "tracked", unavailable: false },
    { label: "character overlay", protectedHistory: true, kind: "character", mode: "overlay", unavailable: false },
    { label: "cleared identity", protectedHistory: true, kind: null, mode: null, unavailable: false },
    { label: "legacy tracked persona without H1", protectedHistory: false, kind: "persona", mode: "tracked", unavailable: false },
    { label: "legacy saved persona without H1", protectedHistory: false, kind: "persona", mode: null, unavailable: false },
  ])("$label", ({ protectedHistory, kind, mode, unavailable }) => {
    expect(isProtectedTrackedPersonaUnavailable(protectedHistory, kind as AssistantKind | null, mode as AssistantSelectionMode | null)).toBe(unavailable)
  })
})
