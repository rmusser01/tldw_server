export type WatchlistsHelpTopic =
  | "opml"
  | "cron"
  | "ttl"
  | "jinja2"
  | "claimClusters"

export type WatchlistsTabHelpKey =
  | "overview"
  | "sources"
  | "jobs"
  | "runs"
  | "items"
  | "alerts"
  | "outputs"
  | "templates"
  | "settings"

export const WATCHLISTS_HELP_DOCS: Record<WatchlistsHelpTopic, string> = {
  opml: "https://en.wikipedia.org/wiki/OPML",
  cron: "https://crontab.guru/",
  ttl: "https://en.wikipedia.org/wiki/Time_to_live",
  jinja2: "https://jinja.palletsprojects.com/en/stable/templates/",
  claimClusters: "https://github.com/rmusser01/tldw_server#documentation--resources"
}

export const WATCHLISTS_MAIN_DOCS_URL =
  "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md"

export const WATCHLISTS_ISSUE_REPORT_URL =
  "https://github.com/rmusser01/tldw_server/issues/new"

export const WATCHLISTS_TAB_HELP_DOCS: Record<WatchlistsTabHelpKey, string> = {
  overview: "https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Watchlists/Watchlist_PRD.md",
  sources: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API/Watchlists_Filters_OPML.md",
  jobs: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#jobs",
  runs: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#runs",
  items: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#updates-triage",
  alerts: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#content-alert-rules",
  outputs: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#outputs",
  templates: "https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#templates",
  settings: "https://github.com/rmusser01/tldw_server/blob/main/Docs/Monitoring/WATCHLISTS_ERROR_PREVENTION_POLICY_2026_02_18.md"
}

export const isValidWatchlistsHelpDocUrl = (value: string): boolean => {
  try {
    const parsed = new URL(value)
    return parsed.protocol === "https:"
  } catch {
    return false
  }
}
