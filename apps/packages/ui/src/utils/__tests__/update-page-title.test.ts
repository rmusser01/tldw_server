import { afterEach, describe, expect, it, vi } from "vitest"
import { updatePageTitle } from "../update-page-title"

describe("updatePageTitle", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    window.history.replaceState({}, "", "/")
    document.head.innerHTML = "<title>Reset</title>"
  })

  it("updates document.title without requiring an existing title element", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    document.head.innerHTML = ""

    updatePageTitle("Chat Ready")

    expect(document.title).toBe("Chat Ready")
    expect(warnSpy).not.toHaveBeenCalled()

    warnSpy.mockRestore()
  })

  it("preserves imperative titles in extension documents", () => {
    vi.stubGlobal("chrome", { runtime: { id: "extension-id" } })
    window.history.replaceState({}, "", "/options.html#/chat")

    updatePageTitle("Extension conversation")

    expect(document.title).toBe("Extension conversation")
  })

  it("does not require a browser document during server rendering", () => {
    vi.stubGlobal("document", undefined)
    expect(() => updatePageTitle("Server render")).not.toThrow()
  })
})
