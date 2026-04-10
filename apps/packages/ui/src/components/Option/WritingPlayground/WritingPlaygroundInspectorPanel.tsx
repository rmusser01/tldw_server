import { useRef, type FC, type ReactNode } from "react"
import type { InspectorTabKey } from "./WritingPlayground.types"

type TabDefinition = {
  key: InspectorTabKey
  label: string
  testId: string
}

type WritingPlaygroundInspectorPanelProps = {
  activeTab: InspectorTabKey
  onTabChange: (tab: InspectorTabKey) => void
  essentialsStrip?: ReactNode
  sampling: ReactNode
  context: ReactNode
  setup: ReactNode
  inspect: ReactNode
  characters: ReactNode
  research: ReactNode
  agent: ReactNode
  feedback?: ReactNode
  tabLabels?: Partial<Record<InspectorTabKey, string>>
  tabBadges?: Partial<Record<InspectorTabKey, ReactNode>>
}

const TAB_DEFINITIONS: TabDefinition[] = [
  {
    key: "sampling",
    label: "Sampling",
    testId: "writing-inspector-tab-sampling"
  },
  {
    key: "context",
    label: "Context",
    testId: "writing-inspector-tab-context"
  },
  {
    key: "setup",
    label: "Setup",
    testId: "writing-inspector-tab-setup"
  },
  {
    key: "inspect",
    label: "Analysis",
    testId: "writing-inspector-tab-inspect"
  },
  {
    key: "characters",
    label: "Characters",
    testId: "writing-inspector-tab-characters"
  },
  {
    key: "research",
    label: "Research",
    testId: "writing-inspector-tab-research"
  },
  {
    key: "agent",
    label: "Agent",
    testId: "writing-inspector-tab-agent"
  },
  {
    key: "feedback",
    label: "Feedback",
    testId: "writing-inspector-tab-feedback"
  }
]

export const WritingPlaygroundInspectorPanel: FC<
  WritingPlaygroundInspectorPanelProps
> = ({
  activeTab,
  onTabChange,
  essentialsStrip,
  sampling,
  context,
  setup,
  inspect,
  characters,
  research,
  agent,
  feedback,
  tabLabels,
  tabBadges
}) => {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])

  const selectTabByIndex = (index: number, focus = false) => {
    const normalizedIndex =
      ((index % TAB_DEFINITIONS.length) + TAB_DEFINITIONS.length) %
      TAB_DEFINITIONS.length
    onTabChange(TAB_DEFINITIONS[normalizedIndex].key)
    if (focus) {
      tabRefs.current[normalizedIndex]?.focus()
    }
  }

  const panelMap: Record<InspectorTabKey, ReactNode> = {
    sampling,
    context,
    setup,
    inspect,
    characters,
    research,
    agent,
    feedback: feedback ?? null
  }

  return (
    <div data-testid="writing-playground-inspector-panel" className="flex flex-col gap-3">
      {essentialsStrip ? (
        <div data-testid="writing-essentials-strip">{essentialsStrip}</div>
      ) : null}

      <div
        role="tablist"
        aria-label="Writing inspector tabs"
        aria-orientation="horizontal"
        className="inline-flex w-full items-center rounded-md border border-border p-[2px]">
        {TAB_DEFINITIONS.map((tab) => {
          const selected = activeTab === tab.key
          const tabIndex = TAB_DEFINITIONS.findIndex(
            (definition) => definition.key === tab.key
          )
          return (
            <button
              key={tab.key}
              ref={(element) => {
                tabRefs.current[tabIndex] = element
              }}
              id={`writing-inspector-tab-${tab.key}`}
              data-testid={tab.testId}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`writing-inspector-panel-${tab.key}`}
              tabIndex={selected ? 0 : -1}
              className={`flex-1 rounded px-2 py-1 text-xs transition-colors ${
                selected
                  ? "bg-primary text-primary-foreground"
                  : "text-text-secondary hover:bg-surface-hover"
              }`}
              onClick={() => onTabChange(tab.key)}
              onKeyDown={(event) => {
                const currentIndex = TAB_DEFINITIONS.findIndex(
                  (definition) => definition.key === tab.key
                )
                if (event.key === "ArrowRight") {
                  event.preventDefault()
                  selectTabByIndex(currentIndex + 1, true)
                } else if (event.key === "ArrowLeft") {
                  event.preventDefault()
                  selectTabByIndex(currentIndex - 1, true)
                } else if (event.key === "Home") {
                  event.preventDefault()
                  selectTabByIndex(0, true)
                } else if (event.key === "End") {
                  event.preventDefault()
                  selectTabByIndex(TAB_DEFINITIONS.length - 1, true)
                }
              }}>
              {tabLabels?.[tab.key] || tab.label}
              {tabBadges?.[tab.key] ? (
                <span className="ml-1" aria-hidden="true">{tabBadges[tab.key]}</span>
              ) : null}
            </button>
          )
        })}
      </div>

      {TAB_DEFINITIONS.map((tab) => {
        const selected = activeTab === tab.key
        return (
          <section
            key={tab.key}
            id={`writing-inspector-panel-${tab.key}`}
            role="tabpanel"
            aria-labelledby={`writing-inspector-tab-${tab.key}`}
            hidden={!selected}>
            {selected ? panelMap[tab.key] : null}
          </section>
        )
      })}
    </div>
  )
}
