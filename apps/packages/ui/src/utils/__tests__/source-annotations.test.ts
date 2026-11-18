import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  persistSourceAnnotations,
  readPersistedSourceAnnotations,
  validateSourceAnnotations
} from "../source-annotations"

const annotation = {
  id: "a",
  quote: "Evidence",
  note: "Check source",
  createdAt: 1,
  updatedAt: 2
}
describe("portable source annotations", () => {
  beforeEach(() => {
    const data = new Map<string, string>()
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => data.get(key) ?? null,
        setItem: (key: string, value: string) => data.set(key, value)
      }
    })
  })
  it("keeps workspace identities separate and excludes removed sources from export", () => {
    persistSourceAnnotations("first", {
      source: [annotation],
      removed: [annotation]
    })
    expect(readPersistedSourceAnnotations("first", ["source"])).toEqual({
      source: [annotation]
    })
    expect(readPersistedSourceAnnotations("second")).toEqual({})
  })
  it.each([
    null,
    [],
    { source: "wrong" },
    { source: [{ ...annotation, updatedAt: null }] },
    { source: [{ ...annotation, quote: "x".repeat(100001) }] }
  ])("rejects malformed or oversized imported notes", (value) => {
    expect(validateSourceAnnotations(value, ["source"])).toBeNull()
  })
  it("projects only annotation content, never arbitrary imported metadata", () => {
    expect(
      validateSourceAnnotations(
        { source: [{ ...annotation, credentials: "not-note-data" }] },
        ["source"]
      )
    ).toEqual({ source: [annotation] })
  })
  it("retains valid local annotations when a legacy entry is malformed", () => {
    window.localStorage.setItem(
      "tldw:research-workspace:source-annotations:v1:first",
      JSON.stringify({ source: [annotation, { id: 1 }] })
    )
    expect(readPersistedSourceAnnotations("first")).toEqual({
      source: [annotation]
    })
    const stored = window.localStorage.getItem("tldw:research-workspace:source-annotations:v1:first")
    expect(persistSourceAnnotations("first", { source: [annotation] })).toBe(false)
    expect(window.localStorage.getItem("tldw:research-workspace:source-annotations:v1:first")).toBe(stored)
  })
  it("preserves oversized valid legacy annotations when reading local storage", () => {
    const legacy = { source: [{ ...annotation, note: "x".repeat(100001) }] }
    window.localStorage.setItem(
      "tldw:research-workspace:source-annotations:v1:first",
      JSON.stringify(legacy)
    )
    const restored = readPersistedSourceAnnotations("first")
    expect(restored).toEqual(legacy)
    expect(persistSourceAnnotations("first", { ...restored, other: [annotation] })).toBe(true)
    expect(JSON.parse(window.localStorage.getItem(
      "tldw:research-workspace:source-annotations:v1:first"
    )!)).toEqual({ ...legacy, other: [annotation] })
  })
  it("preserves local annotations beyond the portable aggregate limit", () => {
    const legacy = Object.fromEntries(Array.from({ length: 1001 }, (_, index) => [`s${index}`, [annotation]]))
    window.localStorage.setItem(
      "tldw:research-workspace:source-annotations:v1:first",
      JSON.stringify(legacy)
    )
    expect(readPersistedSourceAnnotations("first")).toEqual(legacy)
  })
  it("reports browser persistence failure without claiming a saved annotation", () => {
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("Quota exceeded")
    })
    expect(persistSourceAnnotations("first", { source: [annotation] })).toBe(
      false
    )
  })
})
