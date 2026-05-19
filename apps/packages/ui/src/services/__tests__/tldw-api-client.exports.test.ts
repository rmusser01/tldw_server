import { describe, expect, it, vi } from "vitest"

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

describe("TldwApiClient module exports", () => {
  it("loads helper exports used by domain mixins", async () => {
    const module = await import("@/services/tldw/TldwApiClient")

    expect(module.TldwApiClientBase).toBeTypeOf("function")
    expect(module.normalizeReadingDigestSchedule).toBeTypeOf("function")
    expect(module.toFiniteNumber).toBeTypeOf("function")
    expect(module.toOptionalString).toBeTypeOf("function")
    expect(module.toRecord).toBeTypeOf("function")
    expect(module.normalizeIngestionSourceSyncSummary).toBeTypeOf("function")
    expect(module.normalizeIngestionSourceType).toBeTypeOf("function")
    expect(module.normalizeIngestionSource).toBeTypeOf("function")
    expect(module.normalizeIngestionSourceListResponse).toBeTypeOf("function")
    expect(module.normalizeIngestionSourceItem).toBeTypeOf("function")
    expect(module.normalizeIngestionSourceItemsListResponse).toBeTypeOf("function")
    expect(module.normalizeIngestionSourceSyncTrigger).toBeTypeOf("function")
    expect(module.clonePresentationVisualStyleSnapshot).toBeTypeOf("function")
    expect(module.buildPresentationVisualStyleSnapshot).toBeTypeOf("function")
    expect(module.normalizePersonaProfile).toBeTypeOf("function")
    expect(module.normalizePersonaExemplar).toBeTypeOf("function")
  })
})
