import { readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"

const watchlistsRoot = join(process.cwd(), "src/components/Option/Watchlists")

const readWatchlistsFile = (relativePath: string): string =>
  readFileSync(join(watchlistsRoot, relativePath), "utf8")

describe("Watchlists selected container scoping contract", () => {
  it.each([
    ["OverviewTab/OverviewTab.tsx", "fetchWatchlistsOverviewData"],
    ["SourcesTab/SourcesTab.tsx", "fetchWatchlistSources"],
    ["JobsTab/JobsTab.tsx", "fetchWatchlistJobs"],
    ["RunsTab/RunsTab.tsx", "fetchWatchlistRuns"],
    ["ItemsTab/ItemsTab.tsx", "fetchScrapedItems"],
    ["OutputsTab/OutputsTab.tsx", "fetchWatchlistOutputs"]
  ])("%s threads selectedWatchlistId into its primary fetch path", (relativePath, fetchName) => {
    const source = readWatchlistsFile(relativePath)

    expect(source).toContain("selectedWatchlistId")
    expect(source).toContain(fetchName)
    expect(source).toContain("watchlist_id: selectedWatchlistId ?? undefined")
  })

  it("creation forms attach newly created feeds and monitors to the selected Watchlist", () => {
    const sourcesTab = readWatchlistsFile("SourcesTab/SourcesTab.tsx")
    const jobForm = readWatchlistsFile("JobsTab/JobFormModal.tsx")

    expect(sourcesTab).toContain("createWatchlistSource({")
    expect(sourcesTab).toContain("watchlist_id: selectedWatchlistId ?? undefined")
    expect(jobForm).toContain("watchlistId?: number | null")
    expect(jobForm).toContain("watchlist_id: watchlistId ?? undefined")
  })

  it("contains child tab width inside the Watchlists page shell for constrained viewports", () => {
    const page = readWatchlistsFile("WatchlistsPlaygroundPage.tsx")

    expect(page).toContain('data-testid="watchlists-tab-content-shell"')
    expect(page).toContain("overflow-x-auto")
  })
})
