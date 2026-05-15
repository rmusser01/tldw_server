import watchlistsLocale from "../../../../assets/locale/en/watchlists.json"
import { describe, expect, it } from "vitest"

type JsonObject = Record<string, unknown>

const getNestedValue = (source: JsonObject, keyPath: string): unknown =>
  keyPath.split(".").reduce<unknown>((acc, segment) => {
    if (!acc || typeof acc !== "object") return undefined
    return (acc as JsonObject)[segment]
  }, source)

describe("Watchlists Stage 4 Updates copy contract", () => {
  it("frames the selected Watchlist review surface as Updates, not generic articles", () => {
    const labels = watchlistsLocale as JsonObject
    const copyPaths = [
      "items.description",
      "items.searchPlaceholder",
      "items.empty",
      "items.batch.confirmDescriptionScoped",
      "items.batch.scope.page",
      "items.batch.scope.allFiltered",
      "items.shortcuts.nextPrevious",
      "items.live.selectionCleared"
    ]

    for (const keyPath of copyPaths) {
      const value = String(getNestedValue(labels, keyPath) || "")
      expect(value.trim().length, `Missing Updates copy key: ${keyPath}`).toBeGreaterThan(0)
      expect(value.toLowerCase(), `Copy should use update language at ${keyPath}`).toContain(
        "update"
      )
      expect(value.toLowerCase(), `Copy should avoid article language at ${keyPath}`).not.toContain(
        "article"
      )
    }

    expect(String(getNestedValue(labels, "items.alertMatches") || "")).toBe("Alert matches")
    expect(String(getNestedValue(labels, "items.savedViews.importLocal") || "")).toBe(
      "Import local views"
    )
  })
})
