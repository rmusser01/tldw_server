import React, { useEffect, useState } from "react"
import { Select, Tabs, Tag } from "antd"
import { Folder, Rss, Tags } from "lucide-react"
import { useTranslation } from "react-i18next"
import {
  fetchWatchlistGroups,
  fetchWatchlistTags
} from "@/services/watchlists"
import type { JobScope } from "@/types/watchlists"
import { useWatchlistsQuery } from "@/hooks/useWatchlistsQuery"
import { useScopeSources } from "./useScopeSources"

interface ScopeSelectorProps {
  value: JobScope
  onChange: (scope: JobScope) => void
  watchlistId?: number | null
}

type ScopeMode = "sources" | "groups" | "tags"
const CATALOG_PARAMS = { page: 1, size: 200 }

export const ScopeSelector: React.FC<ScopeSelectorProps> = ({
  value,
  onChange,
  watchlistId
}) => {
  const { t } = useTranslation(["watchlists"])

  const { sources, loading: sourcesLoading } = useScopeSources(watchlistId)
  const groupsQuery = useWatchlistsQuery("groups", CATALOG_PARAMS, ({ signal }) => fetchWatchlistGroups(CATALOG_PARAMS, signal))
  const tagsQuery = useWatchlistsQuery("tags", CATALOG_PARAMS, ({ signal }) => fetchWatchlistTags(CATALOG_PARAMS, signal))

  // Determine active mode based on current value
  const getActiveMode = (): ScopeMode => {
    if (value.sources?.length) return "sources"
    if (value.groups?.length) return "groups"
    if (value.tags?.length) return "tags"
    return "sources"
  }

  const [activeMode, setActiveMode] = useState<ScopeMode>(getActiveMode)

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
          loading={sourcesLoading}
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
          loading={groupsQuery.isFetching}
          className="w-full"
          optionFilterProp="label"
          options={(groupsQuery.data?.items ?? []).map((g) => ({
            label: g.name,
            value: g.id
          }))}
          tagRender={(props) => {
            const group = groupsQuery.data?.items.find((g) => g.id === props.value)
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
          loading={tagsQuery.isFetching}
          className="w-full"
          optionFilterProp="label"
          options={(tagsQuery.data?.items ?? []).map((t) => ({
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
