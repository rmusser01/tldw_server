import React, { useEffect, useState } from "react"
import { Select, Tabs, Tag } from "antd"
import { Folder, Rss, Tags } from "lucide-react"
import { useTranslation } from "react-i18next"
import {
  fetchWatchlistGroups,
  fetchWatchlistSources,
  fetchWatchlistTags
} from "@/services/watchlists"
import type { JobScope, WatchlistGroup, WatchlistSource, WatchlistTag } from "@/types/watchlists"

interface ScopeSelectorProps {
  value: JobScope
  onChange: (scope: JobScope) => void
  watchlistId?: number | null
}

type ScopeMode = "sources" | "groups" | "tags"
const SCOPE_SELECTOR_PAGE_SIZE = 200
const SCOPE_SELECTOR_SOURCE_LIMIT = 500

export const ScopeSelector: React.FC<ScopeSelectorProps> = ({
  value,
  onChange,
  watchlistId
}) => {
  const { t } = useTranslation(["watchlists"])

  const [sources, setSources] = useState<WatchlistSource[]>([])
  const [groups, setGroups] = useState<WatchlistGroup[]>([])
  const [tags, setTags] = useState<WatchlistTag[]>([])
  const [loading, setLoading] = useState(false)

  // Determine active mode based on current value
  const getActiveMode = (): ScopeMode => {
    if (value.sources?.length) return "sources"
    if (value.groups?.length) return "groups"
    if (value.tags?.length) return "tags"
    return "sources"
  }

  const [activeMode, setActiveMode] = useState<ScopeMode>(getActiveMode)

  // Load data on mount
  useEffect(() => {
    let active = true

    const fetchSourceOptions = async (): Promise<WatchlistSource[]> => {
      const allSources: WatchlistSource[] = []
      let page = 1

      while (allSources.length < SCOPE_SELECTOR_SOURCE_LIMIT) {
        const result = await fetchWatchlistSources({
          watchlist_id: watchlistId ?? undefined,
          page,
          size: SCOPE_SELECTOR_PAGE_SIZE
        })
        const batch = Array.isArray(result.items) ? result.items : []
        allSources.push(...batch)
        const total = typeof result.total === "number" ? result.total : undefined

        if (batch.length < SCOPE_SELECTOR_PAGE_SIZE) break
        if (total != null && allSources.length >= total) break
        page += 1
      }

      return allSources.slice(0, SCOPE_SELECTOR_SOURCE_LIMIT)
    }

    const loadData = async () => {
      setLoading(true)
      try {
        const [sourcesRes, groupsRes, tagsRes] = await Promise.all([
          fetchSourceOptions(),
          fetchWatchlistGroups({ page: 1, size: 200 }),
          fetchWatchlistTags({ page: 1, size: 200 })
        ])
        if (!active) return
        setSources(sourcesRes)
        setGroups(groupsRes.items || [])
        setTags(tagsRes.items || [])
      } catch (err) {
        if (!active) return
        console.error("Failed to load scope data:", err)
      } finally {
        if (active) setLoading(false)
      }
    }
    loadData()

    return () => {
      active = false
    }
  }, [watchlistId])

  // Sync active tab when value changes
  useEffect(() => {
    setActiveMode(getActiveMode())
  }, [value])

  // Handle mode change
  const handleModeChange = (mode: string) => {
    setActiveMode(mode as ScopeMode)
  }

  // Handle source selection
  const handleSourcesChange = (sourceIds: number[]) => {
    onChange({
      ...value,
      sources: sourceIds.length > 0 ? sourceIds : undefined
    })
  }

  // Handle group selection
  const handleGroupsChange = (groupIds: number[]) => {
    onChange({
      ...value,
      groups: groupIds.length > 0 ? groupIds : undefined
    })
  }

  // Handle tag selection
  const handleTagsChange = (tagNames: string[]) => {
    onChange({
      ...value,
      tags: tagNames.length > 0 ? tagNames : undefined
    })
  }

  const tabItems = [
    {
      key: "sources",
      label: (
        <span className="flex items-center gap-1.5">
          <Rss className="h-3.5 w-3.5" />
          {t("watchlists:jobs.scope.sources", "Feeds")}
        </span>
      ),
      children: (
        <Select
          mode="multiple"
          placeholder={t("watchlists:jobs.scope.selectSources", "Select feeds to include")}
          value={value.sources || []}
          onChange={handleSourcesChange}
          loading={loading}
          className="w-full"
          optionFilterProp="label"
          options={sources.map((s) => ({
            label: s.name,
            value: s.id,
            disabled: !s.active
          }))}
          tagRender={(props) => {
            const source = sources.find((s) => s.id === props.value)
            return (
              <Tag closable={props.closable} onClose={props.onClose} className="mr-1">
                {source?.name || props.value}
              </Tag>
            )
          }}
        />
      )
    },
    {
      key: "groups",
      label: (
        <span className="flex items-center gap-1.5">
          <Folder className="h-3.5 w-3.5" />
          {t("watchlists:jobs.scope.groups", "Groups")}
        </span>
      ),
      children: (
        <Select
          mode="multiple"
          placeholder={t("watchlists:jobs.scope.selectGroups", "Select groups to include")}
          value={value.groups || []}
          onChange={handleGroupsChange}
          loading={loading}
          className="w-full"
          optionFilterProp="label"
          options={groups.map((g) => ({
            label: g.name,
            value: g.id
          }))}
          tagRender={(props) => {
            const group = groups.find((g) => g.id === props.value)
            return (
              <Tag closable={props.closable} onClose={props.onClose} className="mr-1">
                {group?.name || props.value}
              </Tag>
            )
          }}
        />
      )
    },
    {
      key: "tags",
      label: (
        <span className="flex items-center gap-1.5">
          <Tags className="h-3.5 w-3.5" />
          {t("watchlists:jobs.scope.tags", "Tags")}
        </span>
      ),
      children: (
        <Select
          mode="multiple"
          placeholder={t("watchlists:jobs.scope.selectTags", "Select tags to include")}
          value={value.tags || []}
          onChange={handleTagsChange}
          loading={loading}
          className="w-full"
          optionFilterProp="label"
          options={tags.map((t) => ({
            label: t.name,
            value: t.name
          }))}
        />
      )
    }
  ]

  return (
    <div className="border border-border rounded-lg p-3">
      <Tabs
        activeKey={activeMode}
        onChange={handleModeChange}
        items={tabItems}
        size="small"
      />
      <div className="mt-2 text-xs text-text-muted">
        {activeMode === "sources" &&
          t("watchlists:jobs.scope.sourcesHelp", "Monitor will fetch directly from selected feeds")}
        {activeMode === "groups" &&
          t("watchlists:jobs.scope.groupsHelp", "Monitor will fetch from all feeds in selected groups")}
        {activeMode === "tags" &&
          t("watchlists:jobs.scope.tagsHelp", "Monitor will fetch from all feeds with selected tags")}
      </div>
    </div>
  )
}
