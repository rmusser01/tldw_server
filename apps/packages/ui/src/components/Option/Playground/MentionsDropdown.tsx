import React from "react"
import { TabInfo, MentionPosition } from "~/hooks/useTabMentions"
import { Globe, X, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"

interface MentionsDropdownProps {
  show: boolean
  tabs: TabInfo[]
  mentionPosition: MentionPosition | null
  onSelectTab: (tab: TabInfo) => void
  onClose: () => void
  textareaRef: React.RefObject<HTMLTextAreaElement>
  refetchTabs: () => Promise<void>
  onMentionsOpen: () => Promise<void>
}

export const MentionsDropdown: React.FC<MentionsDropdownProps> = ({
  show,
  tabs,
  mentionPosition,
  onSelectTab,
  onClose,
  refetchTabs,
  onMentionsOpen
}) => {
  const { t } = useTranslation(["common"])
  const activeOptionRef = React.useRef<HTMLButtonElement>(null)
  const [selectedIndex, setSelectedIndex] = React.useState(0)
  const [isRefreshing, setIsRefreshing] = React.useState(false)
  const toText = React.useCallback(
    (value: unknown): string =>
      typeof value === "string" ? value : String(value),
    []
  )

  const groupedTabs = React.useMemo(() => {
    if (tabs.length === 0) return []
    const groups = new Map<string, TabInfo[]>()
    for (const tab of tabs) {
      let category = "Other"
      try {
        const host = new URL(tab.url).hostname.replace(/^www\./i, "").trim()
        category = host || "Other"
      } catch {
        category = "Other"
      }
      const existing = groups.get(category)
      if (existing) {
        existing.push(tab)
      } else {
        groups.set(category, [tab])
      }
    }
    return Array.from(groups.entries()).map(([category, items]) => ({
      category,
      items
    }))
  }, [tabs])

  const orderedTabs = React.useMemo(
    () => groupedTabs.flatMap((group) => group.items),
    [groupedTabs]
  )

  React.useEffect(() => {
    setSelectedIndex(0)
  }, [orderedTabs])

  React.useEffect(() => {
    if (show) {
      onMentionsOpen()
    }
  }, [show, onMentionsOpen])

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!show) return

      switch (e.key) {
        case "ArrowDown":
          if (orderedTabs.length === 0) return
          e.preventDefault()
          setSelectedIndex((prev) => (prev + 1) % orderedTabs.length)
          break
        case "ArrowUp":
          if (orderedTabs.length === 0) return
          e.preventDefault()
          setSelectedIndex(
            (prev) => (prev - 1 + orderedTabs.length) % orderedTabs.length
          )
          break
        case "Enter":
          e.preventDefault()
          if (orderedTabs[selectedIndex]) {
            onSelectTab(orderedTabs[selectedIndex])
          }
          break
        case "Escape":
          e.preventDefault()
          onClose()
          break
      }
    }

    if (show) {
      document.addEventListener("keydown", handleKeyDown)
      return () => document.removeEventListener("keydown", handleKeyDown)
    }
  }, [show, orderedTabs, selectedIndex, onSelectTab, onClose])

  React.useEffect(() => {
    if (show)
      activeOptionRef.current?.scrollIntoView?.({
        block: "nearest",
        inline: "nearest"
      })
  }, [show, selectedIndex, orderedTabs])

  const handleRefreshTabs = async () => {
    if (isRefreshing) return
    
    setIsRefreshing(true)
    try {
      await refetchTabs()
    } catch (error) {
      console.error("Failed to refresh tabs:", error)
    } finally {
      setIsRefreshing(false)
    }
  }

  if (!show) return null

  return (
    <div
      className="relative z-50 my-2 min-w-0 w-full max-w-full bg-surface border border-border rounded-lg shadow-lg max-h-80 overflow-y-auto"
      role="listbox"
      aria-label={
        t(
          "playground:mentions.dropdownLabel",
          "Tab mention suggestions"
        ) as string
      }
    >
      <div className="p-2 border-b border-border flex items-center justify-between">
        <span className="text-sm font-medium text-text">
          {t("playground:mentions.selectTab", "Select tab")}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefreshTabs}
            disabled={isRefreshing}
            type="button"
            className="text-text-subtle hover:text-text disabled:opacity-50 disabled:cursor-not-allowed"
            title={t("playground:mentions.refreshTabs", "Refresh tabs") as string}
            aria-label={t("playground:mentions.refreshTabs", "Refresh tabs") as string}>
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={onClose}
            className="text-text-subtle hover:text-text"
            aria-label={t("common:close", "Close") as string}
            title={t("common:close", "Close") as string}>
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="max-h-56 overflow-y-auto">
        {(() => {
          let globalIndex = 0
          return groupedTabs.map(({ category, items }) => (
            <div key={category}>
              <div
                className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-text-subtle"
                data-testid={`mentions-category-${category}`}
              >
                {category}
              </div>
              {items.map((tab) => {
                const index = globalIndex
                globalIndex += 1
                return (
                  <button
                    key={tab.id}
                    ref={index === selectedIndex ? activeOptionRef : undefined}
                    type="button"
                    role="option"
                    aria-selected={index === selectedIndex}
                    onClick={() => onSelectTab(tab)}
                    title={tab.title}
                    className={`w-full text-left p-3 hover:bg-surface2 flex items-center gap-3 transition-colors ${
                      index === selectedIndex
                        ? "bg-surface2 border-r-2 border-primary"
                        : ""
                    }`}>
                    <div className="flex-shrink-0">
                      {tab.favIconUrl ? (
                        <img
                          src={tab.favIconUrl}
                          alt=""
                          className="w-4 h-4 rounded"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement
                            target.style.display = "none"
                            target.nextElementSibling?.classList.remove("hidden")
                          }}
                        />
                      ) : null}
                      <Globe
                        className={`w-4 h-4 text-text-subtle ${tab.favIconUrl ? "hidden" : ""}`}
                      />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-text truncate">
                        {tab.title}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          ))
        })()}
      </div>

      {orderedTabs.length === 0 && (
        <div className="p-4 text-center text-text-subtle text-sm">
          {mentionPosition?.query ? (
            <p>
              {toText(
                t(
                  "playground:mentions.noTabsMatching",
                  `No tabs found matching "{{query}}"`,
                  { query: mentionPosition.query } as any
                )
              )}
            </p>
          ) : (
            <p>
              {toText(
                t(
                  "playground:mentions.noTabsHint",
                  "Type to search open tabs, or refresh to load recent tabs."
                )
              )}
            </p>
          )}
          <button
            onClick={handleRefreshTabs}
            disabled={isRefreshing}
            title={
              isRefreshing
                ? toText(t("playground:mentions.refreshing", "Refreshing..."))
                : toText(t("playground:mentions.refreshTabs", "Refresh tabs"))
            }
            className="mt-2 text-primary hover:text-primaryStrong disabled:opacity-50">
            {isRefreshing
              ? toText(t("playground:mentions.refreshing", "Refreshing..."))
              : toText(t("playground:mentions.refreshTabs", "Refresh tabs"))}
          </button>
        </div>
      )}
    </div>
  )
}
