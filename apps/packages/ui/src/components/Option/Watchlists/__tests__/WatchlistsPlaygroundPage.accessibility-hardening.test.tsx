// @vitest-environment jsdom

import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = dirname(fileURLToPath(import.meta.url))
const pageSource = readFileSync(
  resolve(testDir, "../WatchlistsPlaygroundPage.tsx"),
  "utf8"
)

describe("WatchlistsPlaygroundPage accessibility hardening", () => {
  it("keeps layout controls named and consequence-linked", () => {
    expect(pageSource).toContain('aria-labelledby="watchlists-show-all-views-label"')
    expect(pageSource).toContain('aria-describedby="watchlists-show-all-views-description"')
    expect(pageSource).toContain('id="watchlists-show-all-views-label"')
    expect(pageSource).toContain('id="watchlists-show-all-views-description"')
  })

  it("keeps first-viewport help and repeated guidance behind disclosure", () => {
    expect(pageSource).not.toContain('data-testid="watchlists-repeat-actions"')
    expect(pageSource).not.toContain('data-testid="watchlists-orientation-alert"')
    expect(pageSource).toContain('data-testid="watchlists-help-panel"')
    expect(pageSource.indexOf('data-testid="watchlists-main-docs-link"'))
      .toBeGreaterThan(pageSource.indexOf('data-testid="watchlists-help-panel"'))
    expect(pageSource.lastIndexOf('data-testid="watchlists-create-container"'))
      .toBeGreaterThan(pageSource.indexOf('data-testid="watchlists-help-panel"'))
    expect(pageSource.lastIndexOf('data-testid="watchlists-edit-container"'))
      .toBeGreaterThan(pageSource.indexOf('data-testid="watchlists-help-panel"'))
  })

  it("uses the existing caption scale for attention badges", () => {
    expect(pageSource).not.toContain("text-[10px]")
    expect(pageSource).toContain("rounded-full bg-red-500 px-1.5 text-xs")
  })
})
