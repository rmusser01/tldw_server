import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
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

import {
  presentationsMethods,
  type TldwApiClientCore
} from "@/services/tldw/domains/presentations"

describe("TldwApiClient presentations normalization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("trims and filters array-based visual style metadata fields", async () => {
    const client: TldwApiClientCore = {
      ensureConfigForRequest: vi.fn(async () => ({})),
      request: vi.fn(async () => ({
        styles: [
          {
            id: "notebooklm-blueprint",
            name: "Blueprint",
            scope: "builtin",
            tags: [" technical ", " ", "technical_grid"],
            best_for: [" systems explanation ", "", "architecture walkthrough "],
            artifact_preferences: [" timeline ", "comparison_matrix", ""],
            appearance_defaults: { theme: "night" },
            generation_rules: {},
            fallback_policy: {}
          }
        ],
        total_count: 1
      })) as unknown as TldwApiClientCore["request"],
      resolveApiPath: vi.fn(),
      fillPathParams: vi.fn()
    }

    const styles = await presentationsMethods.listVisualStyles.call(client)

    expect(styles).toHaveLength(1)
    expect(styles[0]?.tags).toEqual(["technical", "technical_grid"])
    expect(styles[0]?.best_for).toEqual(["systems explanation", "architecture walkthrough"])
    expect(styles[0]?.artifact_preferences).toEqual(["timeline", "comparison_matrix"])
  })

  it("uses canonical pagination.total when total_count is absent", async () => {
    const client: TldwApiClientCore = {
      ensureConfigForRequest: vi.fn(async () => ({})),
      request: vi
        .fn()
        .mockResolvedValueOnce({
          styles: [
            {
              id: "page-1-style",
              name: "Page 1",
              scope: "builtin",
              generation_rules: {},
              artifact_preferences: [],
              appearance_defaults: {},
              fallback_policy: {}
            }
          ],
          pagination: {
            mode: "offset",
            limit: 1,
            offset: 0,
            total: 2,
            has_more: true,
            next_offset: 1
          }
        })
        .mockResolvedValueOnce({
          styles: [
            {
              id: "page-2-style",
              name: "Page 2",
              scope: "user",
              generation_rules: {},
              artifact_preferences: [],
              appearance_defaults: {},
              fallback_policy: {}
            }
          ],
          pagination: {
            mode: "offset",
            limit: 1,
            offset: 1,
            total: 2,
            has_more: false,
            next_offset: null
          }
        }),
      resolveApiPath: vi.fn(),
      fillPathParams: vi.fn()
    }

    const styles = await presentationsMethods.listVisualStyles.call(client)

    expect(styles.map((style) => style.id)).toEqual(["page-1-style", "page-2-style"])
    expect(client.request).toHaveBeenCalledTimes(2)
  })
})
