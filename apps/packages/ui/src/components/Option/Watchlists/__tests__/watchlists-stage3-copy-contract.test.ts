import { describe, expect, it } from "vitest"

import watchlistsLocale from "../../../../assets/locale/en/watchlists.json"

describe("Watchlists Stage 3 content alert copy contract", () => {
  it("labels content alerts separately from pipeline health issues", () => {
    const locale = watchlistsLocale as unknown as {
      tabs?: Record<string, string>
      alerts?: Record<string, unknown>
      orientation?: Record<string, Record<string, string>>
    }

    expect(locale.tabs?.alerts).toBe("Alerts")
    expect(locale.alerts?.rulesTitle).toBe("Content alert rules")
    expect(locale.alerts?.inboxTitle).toBe("Alert inbox")
    expect(locale.alerts?.healthBoundary).toBe(
      "Run failures and source problems are health issues, not content alerts."
    )
  })

  it("keeps alert setup copy useful for CTI and news workflows", () => {
    const serialized = JSON.stringify((watchlistsLocale as { alerts?: unknown }).alerts || {})

    expect(serialized).toContain("descriptor")
    expect(serialized).toContain("keyword")
    expect(serialized).toContain("classification")
    expect(serialized).toContain("entity")
    expect(serialized).toContain("source")
    expect(serialized).not.toMatch(/run failures are alerts/i)
    expect(serialized).not.toMatch(/pipeline alerts/i)
  })
})
