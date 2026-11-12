import React, { useCallback, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Loader2 } from "lucide-react"
import type { QuickIngestTab, TabBadgeState } from "./types"
import { isEditableTarget } from "@/utils/editable-target"

type QuickIngestTabsProps = {
  activeTab: QuickIngestTab
  onTabChange: (tab: QuickIngestTab) => void
  badges: TabBadgeState
}

const TABS: QuickIngestTab[] = ["queue", "options", "results"]

export const QuickIngestTabs: React.FC<QuickIngestTabsProps> = ({
  activeTab,
  onTabChange,
  badges
}) => {
  const { t } = useTranslation(["option"])
  const tabListRef = useRef<HTMLDivElement>(null)
  const tabRefs = useRef<Map<QuickIngestTab, HTMLButtonElement>>(new Map())

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // Keyboard navigation: 1/2/3 keys switch tabs (except in text inputs)
  const handleTabListKeyDown = (e: React.KeyboardEvent) => {
    if (isEditableTarget(e.target)) return

    if (e.key === "1") {
      e.preventDefault()
      onTabChange("queue")
    } else if (e.key === "2") {
      e.preventDefault()
      onTabChange("options")
    } else if (e.key === "3") {
      e.preventDefault()
      onTabChange("results")
    }
  }

  // Arrow key navigation within tablist
  const handleTabKeyDown = (e: React.KeyboardEvent, currentTab: QuickIngestTab) => {
    const currentIndex = TABS.indexOf(currentTab)
    let nextIndex = currentIndex

    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault()
      nextIndex = (currentIndex + 1) % TABS.length
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault()
      nextIndex = (currentIndex - 1 + TABS.length) % TABS.length
    } else if (e.key === "Home") {
      e.preventDefault()
      nextIndex = 0
    } else if (e.key === "End") {
      e.preventDefault()
      nextIndex = TABS.length - 1
    }

    if (nextIndex !== currentIndex) {
      const nextTab = TABS[nextIndex]
      onTabChange(nextTab)
      tabRefs.current.get(nextTab)?.focus()
    }
  }

  const getTabLabel = (tab: QuickIngestTab): string => {
    switch (tab) {
      case "queue":
        return qi("tabs.queue", "Queue")
      case "options":
        return qi("tabs.options", "Options")
      case "results":
        return qi("tabs.results", "Results")
    }
  }

  const getTabAriaLabel = (tab: QuickIngestTab): string => {
    switch (tab) {
      case "queue":
        if (badges.queueCount > 0) {
          return qi("tabs.queueAriaWithCount", "Queue tab, {{count}} items", {
            count: badges.queueCount
          })
        }
        return qi("tabs.queueAria", "Queue tab")
      case "options":
        if (badges.optionsModified) {
          return qi("tabs.optionsAriaModified", "Options tab, settings modified")
        }
        return qi("tabs.optionsAria", "Options tab")
      case "results":
        if (badges.isProcessing) {
          return qi("tabs.resultsAriaProcessing", "Results tab, processing in progress")
        }
        if (badges.hasFailure) {
          return qi("tabs.resultsAriaFailed", "Results tab, last run failed")
        }
        return qi("tabs.resultsAria", "Results tab")
    }
  }

  const renderBadge = (tab: QuickIngestTab) => {
    switch (tab) {
      case "queue":
        if (badges.queueCount > 0) {
          return (
            <span className="ml-1.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-primary px-1.5 text-[10px] font-medium text-white">
              {badges.queueCount > 99 ? "99+" : badges.queueCount}
            </span>
          )
        }
        return null
      case "options":
        if (badges.optionsModified) {
          return (
            <span
              className="ml-1.5 inline-flex h-2 w-2 rounded-full bg-warn"
              aria-hidden="true"
            />
          )
        }
        return null
      case "results":
        if (badges.isProcessing) {
          return (
            <Loader2
              className="ml-1.5 h-3.5 w-3.5 animate-spin text-primary"
              aria-hidden="true"
            />
          )
        }
        if (badges.hasFailure) {
          return (
            <span
              className="ml-1.5 inline-flex h-2 w-2 rounded-full bg-danger"
              aria-hidden="true"
            />
          )
        }
        return null
    }
  }

  return (
    <div
      ref={tabListRef}
      role="tablist"
      aria-label={qi("tabs.ariaLabel", "Quick Ingest sections")}
      className="flex border-b border-border"
      onKeyDown={handleTabListKeyDown}
    >
      {TABS.map((tab) => {
        const isActive = activeTab === tab
        return (
          <button
            key={tab}
            ref={(el) => {
              if (el) tabRefs.current.set(tab, el)
            }}
            role="tab"
            id={`quick-ingest-tab-${tab}`}
            aria-selected={isActive}
            aria-controls={`quick-ingest-panel-${tab}`}
            aria-label={getTabAriaLabel(tab)}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onTabChange(tab)}
            onKeyDown={(e) => handleTabKeyDown(e, tab)}
            className={`relative flex items-center px-4 py-2.5 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus ${
              isActive
                ? "text-primary"
                : "text-text-muted hover:text-text"
            }`}
          >
            {getTabLabel(tab)}
            {renderBadge(tab)}
            {isActive && (
              <span
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary"
                aria-hidden="true"
              />
            )}
          </button>
        )
      })}
    </div>
  )
}

export default QuickIngestTabs
