import React, { useEffect } from "react"
import { useSearchParams } from "react-router-dom"
import OptionLayout from "@/components/Layouts/Layout"
import { WatchlistsPlaygroundPage } from "@/components/Option/Watchlists/WatchlistsPlaygroundPage"
import { useWatchlistsStore } from "@/store/watchlists"
import type { WatchlistTab } from "@/types/watchlists"

const WATCHLIST_TABS: Set<WatchlistTab> = new Set([
  "overview",
  "sources",
  "jobs",
  "runs",
  "items",
  "alerts",
  "outputs",
  "templates",
  "settings"
])

/** Backward-compat: map new user-facing tab names to internal keys */
const TAB_ALIASES: Record<string, WatchlistTab> = {
  feeds: "sources",
  monitors: "jobs",
  activity: "runs",
  articles: "items",
  reports: "outputs"
}

const parsePositiveIntegerParam = (
  params: URLSearchParams,
  keys: string[]
): number | undefined => {
  for (const key of keys) {
    const raw = params.get(key)
    if (!raw) continue
    const parsed = Number(raw)
    if (Number.isInteger(parsed) && parsed > 0) {
      return parsed
    }
  }
  return undefined
}

const parseStringParam = (
  params: URLSearchParams,
  keys: string[]
): string | undefined => {
  for (const key of keys) {
    if (!params.has(key)) continue
    return params.get(key) ?? ""
  }
  return undefined
}

const parseTabParam = (params: URLSearchParams): WatchlistTab | undefined => {
  const rawTab = parseStringParam(params, ["tab"])
  if (!rawTab) return undefined
  // Check canonical tab names first
  if (WATCHLIST_TABS.has(rawTab as WatchlistTab)) return rawTab as WatchlistTab
  // Check aliases (feeds→sources, monitors→jobs, etc.)
  const aliased = TAB_ALIASES[rawTab.toLowerCase()]
  if (aliased) return aliased
  return undefined
}

const OptionWatchlists = () => {
  const [searchParams] = useSearchParams()
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const setItemsSelectedSourceId = useWatchlistsStore((s) => s.setItemsSelectedSourceId)
  const setItemsSmartFilter = useWatchlistsStore((s) => s.setItemsSmartFilter)
  const setItemsStatusFilter = useWatchlistsStore((s) => s.setItemsStatusFilter)
  const setItemsSearchQuery = useWatchlistsStore((s) => s.setItemsSearchQuery)
  const setRunsJobFilter = useWatchlistsStore((s) => s.setRunsJobFilter)
  const setRunsStatusFilter = useWatchlistsStore((s) => s.setRunsStatusFilter)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const setOutputsJobFilter = useWatchlistsStore((s) => s.setOutputsJobFilter)
  const setOutputsRunFilter = useWatchlistsStore((s) => s.setOutputsRunFilter)
  const openOutputPreview = useWatchlistsStore((s) => s.openOutputPreview)
  const deepLinkSignature = searchParams.toString()

  useEffect(() => {
    const params = new URLSearchParams(deepLinkSignature)
    const tab = parseTabParam(params)
    const shouldOpenRun = params.get("open_run") === "1"
    const shouldOpenOutput = params.get("open_output") === "1"

    if (tab) setActiveTab(tab)

    const sourceId = parsePositiveIntegerParam(params, ["source_id", "sourceId"])
    if (sourceId !== undefined) {
      setItemsSelectedSourceId(sourceId)
    }

    const itemSmartFilter = parseStringParam(params, ["item_smart", "itemSmart"])
    if (itemSmartFilter !== undefined) {
      setItemsSmartFilter(itemSmartFilter)
    }

    const itemStatusFilter = parseStringParam(params, ["item_status", "itemStatus"])
    if (itemStatusFilter !== undefined) {
      setItemsStatusFilter(itemStatusFilter)
    }

    const itemSearchQuery = parseStringParam(params, ["item_q", "itemQuery", "q"])
    if (itemSearchQuery !== undefined) {
      setItemsSearchQuery(itemSearchQuery)
    }

    const jobId = parsePositiveIntegerParam(params, ["job_id", "jobId"])
    if (jobId !== undefined) {
      setRunsJobFilter(jobId)
      setOutputsJobFilter(jobId)
    }

    const runStatus = parseStringParam(params, ["run_status", "runStatus"])
    if (runStatus !== undefined) {
      setRunsStatusFilter(runStatus || null)
    }

    const runId = parsePositiveIntegerParam(params, ["run_id", "runId"])
    if (runId !== undefined) {
      setOutputsRunFilter(runId)
      if (tab === "runs" || shouldOpenRun) {
        openRunDetail(runId)
      }
    }

    const outputId = parsePositiveIntegerParam(params, ["output_id", "outputId"])
    if (outputId !== undefined && (tab === "outputs" || shouldOpenOutput)) {
      openOutputPreview(outputId)
    }
  }, [
    deepLinkSignature,
    openOutputPreview,
    openRunDetail,
    setActiveTab,
    setItemsSearchQuery,
    setItemsSelectedSourceId,
    setItemsSmartFilter,
    setItemsStatusFilter,
    setOutputsJobFilter,
    setOutputsRunFilter,
    setRunsJobFilter,
    setRunsStatusFilter
  ])

  return (
    <OptionLayout>
      <WatchlistsPlaygroundPage />
    </OptionLayout>
  )
}

export default OptionWatchlists
