import React, { useMemo, useState } from "react"
import { Button, Drawer, Tag } from "antd"
import { Menu } from "lucide-react"
import type { WatchlistTab } from "@/types/watchlists"

export interface WatchlistsMobileNavigationItem {
  key: WatchlistTab
  label: string
  description?: string
  count?: number
}

export interface WatchlistsMobileNavigationGroup {
  key: string
  label: string
  items: WatchlistsMobileNavigationItem[]
}

interface WatchlistsMobileNavigationProps {
  activeKey: WatchlistTab
  fallbackLabel: string
  groups: WatchlistsMobileNavigationGroup[]
  navigationLabel: string
  onNavigate: (tab: WatchlistTab) => void
  title: string
}

export const WatchlistsMobileNavigation: React.FC<WatchlistsMobileNavigationProps> = ({
  activeKey,
  fallbackLabel,
  groups,
  navigationLabel,
  onNavigate,
  title
}) => {
  const [open, setOpen] = useState(false)
  const activeItem = useMemo(
    () => groups.flatMap((group) => group.items).find((item) => item.key === activeKey),
    [activeKey, groups]
  )

  const handleNavigate = (tab: WatchlistTab) => {
    onNavigate(tab)
    setOpen(false)
  }

  return (
    <div className="mb-4">
      <Button
        block
        icon={<Menu className="h-4 w-4" />}
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        data-testid="watchlists-constrained-nav-trigger"
      >
        {activeItem?.label || fallbackLabel}
      </Button>
      <Drawer
        title={title}
        open={open}
        onClose={() => setOpen(false)}
        placement="bottom"
        size="min(86vh, 680px)"
        data-testid="watchlists-constrained-nav-drawer"
      >
        <nav
          className="space-y-5"
          aria-label={navigationLabel}
        >
          {groups.map((group) => (
            <section key={group.key} className="space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
                {group.label}
              </h3>
              <div className="space-y-2">
                {group.items.map((item) => {
                  const selected = item.key === activeKey
                  return (
                    <button
                      key={item.key}
                      type="button"
                      className={`flex w-full items-start justify-between gap-3 rounded-lg border px-3 py-2.5 text-left transition ${
                        selected
                          ? "border-primary bg-primary/10 text-text"
                          : "border-border bg-surface text-text-muted hover:bg-surface-hover hover:text-text"
                      }`}
                      aria-label={item.label}
                      aria-current={selected ? "page" : undefined}
                      onClick={() => handleNavigate(item.key)}
                    >
                      <span className="min-w-0">
                        <span className="block font-medium">{item.label}</span>
                        {item.description ? (
                          <span className="mt-0.5 block text-xs text-text-subtle">
                            {item.description}
                          </span>
                        ) : null}
                      </span>
                      {typeof item.count === "number" && item.count > 0 ? (
                        <Tag color="blue" className="shrink-0">
                          {item.count}
                        </Tag>
                      ) : null}
                    </button>
                  )
                })}
              </div>
            </section>
          ))}
        </nav>
      </Drawer>
    </div>
  )
}
