import React from "react"
import { useTranslation } from "react-i18next"

export type PersonaGardenTabItem = {
  key: string
  label: string
  content: React.ReactNode
}

type PersonaGardenTabsProps = {
  activeKey: string
  items: PersonaGardenTabItem[]
  onChange: (key: string) => void
}

export const PersonaGardenTabs: React.FC<PersonaGardenTabsProps> = ({
  activeKey,
  items,
  onChange
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const tabRefs = React.useRef<Record<string, HTMLButtonElement | null>>({})
  React.useEffect(() => {
    tabRefs.current[activeKey]?.scrollIntoView?.({ block: "nearest", inline: "nearest" })
  }, [activeKey])

  return (
    <div className="flex min-w-0 w-full flex-1 flex-col gap-3">
      <div
        role="tablist"
        aria-label={t("sidepanel:personaGarden.tabs.ariaLabel", {
          defaultValue: "Persona Garden sections"
        })}
        className="flex min-w-0 max-w-full gap-1 overflow-x-auto border-b border-border p-1 pb-2 sm:flex-wrap"
      >
        {items.map((item) => {
          const isActive = item.key === activeKey
          const tabId = `persona-garden-tab-${item.key}`
          const panelId = `persona-garden-panel-${item.key}`
          return (
            <button
              key={item.key}
              id={tabId}
              type="button"
              role="tab"
              ref={(node) => { tabRefs.current[item.key] = node }}
              tabIndex={isActive ? 0 : -1}
              aria-selected={isActive}
              aria-controls={panelId}
              className={`min-h-10 shrink-0 whitespace-nowrap rounded-md border px-3 py-2 text-sm font-medium transition-colors motion-reduce:transition-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${
                isActive
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-surface text-text-muted hover:bg-surface2 hover:text-text"
              }`}
              onClick={() => onChange(item.key)}
              onKeyDown={(event) => {
                const index = items.findIndex((candidate) => candidate.key === item.key)
                let nextIndex: number
                switch (event.key) {
                  case "ArrowRight": nextIndex = (index + 1) % items.length; break
                  case "ArrowLeft": nextIndex = (index - 1 + items.length) % items.length; break
                  case "Home": nextIndex = 0; break
                  case "End": nextIndex = items.length - 1; break
                  default: return
                }
                event.preventDefault()
                const nextKey = items[nextIndex].key
                onChange(nextKey)
                tabRefs.current[nextKey]?.focus()
              }}
            >
              {item.label}
            </button>
          )
        })}
      </div>
      <div className="flex flex-1 flex-col">
        {items.map((item) => {
          const isActive = item.key === activeKey
          const tabId = `persona-garden-tab-${item.key}`
          const panelId = `persona-garden-panel-${item.key}`
          return (
            <section
              key={item.key}
              id={panelId}
              role="tabpanel"
              aria-labelledby={tabId}
              hidden={!isActive}
              className={isActive ? "flex flex-1 flex-col gap-3" : undefined}
            >
              {item.content}
            </section>
          )
        })}
      </div>
    </div>
  )
}
