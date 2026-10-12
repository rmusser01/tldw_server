import { describe, expect, it } from "vitest"
import {
  CAPTURED_NOTE_KEYWORD,
  withCapturedNoteKeyword
} from "@/services/notes-capture"

describe("notes capture helpers", () => {
  it("uses a stable reserved tag marker for captured notes", () => {
    expect(CAPTURED_NOTE_KEYWORD).toBe("captured")
  })

  it("adds the captured marker without duplicating user-entered tags", () => {
    expect(withCapturedNoteKeyword(["research", "planning"])).toEqual([
      "research",
      "planning",
      "captured"
    ])

    expect(withCapturedNoteKeyword(["research", "Captured", "research"])).toEqual([
      "research",
      "captured"
    ])
  })

})
