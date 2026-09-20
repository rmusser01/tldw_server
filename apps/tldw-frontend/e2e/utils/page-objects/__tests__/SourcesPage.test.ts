import { describe, expect, it, vi } from "vitest"
import { SourcesPage } from "../SourcesPage"

describe("SourcesPage", () => {
  it("recognizes an online empty workspace when two New source CTAs are rendered", async () => {
    const visibleLocator = {
      isVisible: vi.fn(async () => true),
    }
    const hiddenLocator = {
      isVisible: vi.fn(async () => false),
    }
    const duplicateNewSourceLocators = {
      first: vi.fn(() => visibleLocator),
      isVisible: vi.fn(async () => {
        throw new Error("strict mode violation: locator resolved to 2 elements")
      }),
    }
    const page = {
      getByRole: vi.fn((_role: string, options?: { name?: unknown }) => {
        const name = String(options?.name)
        if (name === "/new source/i") return duplicateNewSourceLocators
        if (name === "/^sources$/i") return { first: () => visibleLocator }
        return hiddenLocator
      }),
      getByText: vi.fn(() => hiddenLocator),
    }
    const sources = new SourcesPage(page as never)

    expect(await sources.isOnlineWorkspace()).toBe(true)
    expect(duplicateNewSourceLocators.first).toHaveBeenCalledOnce()
    expect(sources.newSourceButton).toBe(visibleLocator)
  })
})
