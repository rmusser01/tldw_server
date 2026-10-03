import React from "react"
import { useTranslation } from "react-i18next"
import type { KnowledgeTab } from "./KnowledgePanel"
import { isEditableTarget } from "@/utils/editable-target"

type KnowledgeTabsProps = {
  activeTab: KnowledgeTab
  onTabChange: (tab: KnowledgeTab) => void
  contextCount?: number
  className?: string
}

/** Canonical ordered tab IDs (excludes the "search" backward-compat alias). */
const TAB_IDS: KnowledgeTab[] = [
  "qa-search",
  "file-search",
  "settings",
  "context"
]

/**
 * Tab navigation for the Knowledge panel
 *
 * Features:
 * - 4 tabs: QA Search, File Search, Settings, Context
 * - Badge on Context tab showing attached item count
 * - Keyboard navigation (1/2/3/4 when focused, disabled in text inputs)
 * - ARIA roles for accessibility
 */
export const KnowledgeTabs: React.FC<KnowledgeTabsProps> = ({
  activeTab,
  onTabChange,
  contextCount = 0,
  className = ""
}) => {
  const { t } = useTranslation(["sidepanel"])

  const tabs: { id: KnowledgeTab; label: string; badge?: number }[] = [
    {
      id: "qa-search",
      label: t("sidepanel:knowledge.tabs.qaSearch", "QA Search")
    },
    {
      id: "file-search",
      label: t("sidepanel:knowledge.tabs.fileSearch", "File Search")
    },
    {
      id: "settings",
      label: t("sidepanel:knowledge.tabs.settings", "Settings")
    },
    {
      id: "context",
      label: t("sidepanel:knowledge.tabs.context", "Context"),
      badge: contextCount > 0 ? contextCount : undefined
    }
  ]

  const focusTab = (tabId: KnowledgeTab) => {
    const element = document.getElementById(`knowledge-tab-${tabId}`)
    element?.focus()
  }

  // Handle keyboard navigation (arrow keys and 1/2/3/4 to switch tabs)
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Only handle when focus is on tablist, not inside text inputs
    if (isEditableTarget(e.target)) {
      return
    }

    const currentIndex = Math.max(TAB_IDS.indexOf(activeTab), 0)

    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault()
      const nextIndex = (currentIndex + 1) % TAB_IDS.length
      const nextTab = TAB_IDS[nextIndex]
      onTabChange(nextTab)
      focusTab(nextTab)
      return
    }

    if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault()
      const prevIndex = (currentIndex - 1 + TAB_IDS.length) % TAB_IDS.length
      const prevTab = TAB_IDS[prevIndex]
      onTabChange(prevTab)
      focusTab(prevTab)
      return
    }

    const keyMap: Record<string, KnowledgeTab> = {
      "1": "qa-search",
      "2": "file-search",
      "3": "settings",
      "4": "context"
    }
    const mapped = keyMap[e.key]
    if (mapped) {
      onTabChange(mapped)
      focusTab(mapped)
    }
  }

  return (
    <div
      role="tablist"
      aria-label={t(
        "sidepanel:knowledge.tabs.label",
        "Knowledge panel sections"
      )}
      className={`flex border-b border-border ${className}`}
      onKeyDown={handleKeyDown}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={activeTab === tab.id}
          aria-controls={`knowledge-tabpanel-${tab.id}`}
          id={`knowledge-tab-${tab.id}`}
          tabIndex={activeTab === tab.id ? 0 : -1}
          onClick={() => onTabChange(tab.id)}
          className={`
            relative flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors
            focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2
            ${
              activeTab === tab.id
                ? "text-accent border-b-2 border-accent -mb-[1px]"
                : "text-text-muted hover:text-text"
            }
          `}
        >
          {tab.label}
          {tab.badge !== undefined && (
            <span
              className="ml-1 inline-flex h-5 min-w-[20px] items-center justify-center rounded-full bg-accent/20 px-1.5 text-xs font-semibold text-accent"
              aria-label={t(
                "sidepanel:knowledge.tabs.contextBadge",
                "{{count}} items attached",
                { count: tab.badge }
              )}
            >
              {tab.badge}
            </span>
          )}
        </button>
      ))}
    </div>
  )
}
