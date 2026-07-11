import { describe, expect, it } from "vitest"

import i18n, { ensureI18nNamespaces } from "@/i18n"
import option from "@/assets/locale/en/option.json"
import watchlists from "@/assets/locale/en/watchlists.json"

describe("sources locale wiring", () => {
  it("loads shared and route-local english namespaces on demand", async () => {
    i18n.removeResourceBundle("en", "common")
    i18n.removeResourceBundle("en", "sources")

    expect(i18n.hasResourceBundle("en", "common")).toBe(false)
    expect(i18n.hasResourceBundle("en", "sources")).toBe(false)

    await ensureI18nNamespaces(["common"], "en")
    expect(i18n.hasResourceBundle("en", "common")).toBe(true)
    const common = i18n.getResourceBundle("en", "common") as { noData: string }
    expect(common.noData).toBe("No data")

    await ensureI18nNamespaces(["sources"], "en")
    expect(i18n.hasResourceBundle("en", "sources")).toBe(true)
    const sources = i18n.getResourceBundle("en", "sources") as { title: string }
    expect(sources.title).toBe("Sources")
    expect(option.header.sources).toBe("Sources")
  })

  it("loads the Watchlists namespace on demand", async () => {
    i18n.removeResourceBundle("en", "watchlists")

    expect(i18n.hasResourceBundle("en", "watchlists")).toBe(false)

    await ensureI18nNamespaces(["watchlists"], "en")

    expect(i18n.hasResourceBundle("en", "watchlists")).toBe(true)
    const loadedWatchlists = i18n.getResourceBundle("en", "watchlists") as {
      overview: { pipelineSetup: { title: string } }
    }
    expect(loadedWatchlists.overview.pipelineSetup.title).toBe(
      watchlists.overview.pipelineSetup.title
    )
  })
})
