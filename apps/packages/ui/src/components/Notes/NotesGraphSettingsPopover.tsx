import { Settings2 } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import type { NotesGraphLayout } from "./hooks/useNotesGraphWorkspace"

const SETTINGS_PANEL_ID = "notes-graph-settings-panel"

const LAYOUT_OPTIONS: Array<{ value: NotesGraphLayout; defaultLabel: string }> = [
  { value: "dagre", defaultLabel: "Tree" },
  { value: "circle", defaultLabel: "Circle" },
  { value: "grid", defaultLabel: "Grid" },
  { value: "concentric", defaultLabel: "Rings" }
]

const fieldClassName = "flex flex-col gap-1 text-xs text-text-muted"
const controlClassName =
  "h-9 w-full border border-border bg-bg px-2 text-sm text-text focus:ring-2 focus:ring-focus"

type NotesGraphSettingsPopoverProps = {
  radius: 1 | 2
  maxNodes: number
  maxNodeCap: number
  layout: NotesGraphLayout
  onRadiusChange: (radius: 1 | 2) => void
  onMaxNodesChange: (maxNodes: number) => void
  onLayoutChange: (layout: NotesGraphLayout) => void
}

/**
 * The graph's tuning controls, out of the toolbar and in plain words (NL-14):
 * how far from the focus note to reach, how many items to show at most, and
 * the layout (Tree, Circle, Grid or Rings). Escape, a click outside, or focus
 * moving to a control outside closes it.
 */
const NotesGraphSettingsPopover: React.FC<NotesGraphSettingsPopoverProps> = ({
  radius,
  maxNodes,
  maxNodeCap,
  layout,
  onRadiusChange,
  onMaxNodesChange,
  onLayoutChange
}) => {
  const { t } = useTranslation(["option", "common"])
  const [open, setOpen] = React.useState(false)
  const rootRef = React.useRef<HTMLDivElement | null>(null)
  const buttonRef = React.useRef<HTMLButtonElement | null>(null)
  const title = t("option:notesSearch.graphSettings", {
    defaultValue: "Graph settings"
  })

  React.useEffect(() => {
    if (!open) return
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener("pointerdown", closeOnOutsidePointer)
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer)
  }, [open])

  const closeOnEscape = (event: React.KeyboardEvent) => {
    if (event.key !== "Escape" || !open) return
    event.stopPropagation()
    setOpen(false)
    buttonRef.current?.focus()
  }

  const closeOnFocusOut = (event: React.FocusEvent) => {
    // No next target (a click on the panel itself) is not leaving it.
    const next = event.relatedTarget as Node | null
    if (!open || !next || rootRef.current?.contains(next)) return
    setOpen(false)
  }

  return (
    <div ref={rootRef} className="relative" onKeyDown={closeOnEscape} onBlur={closeOnFocusOut}>
      <button
        ref={buttonRef}
        type="button"
        className="inline-flex min-h-[44px] items-center gap-2 border border-border bg-surface px-3 text-sm text-text hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
        aria-expanded={open}
        aria-controls={SETTINGS_PANEL_ID}
        onClick={() => setOpen((current) => !current)}>
        <Settings2 size={16} aria-hidden="true" />
        {title}
      </button>
      {open ? (
        <div
          id={SETTINGS_PANEL_ID}
          role="group"
          aria-label={title}
          className="absolute left-0 top-12 z-20 flex w-[240px] flex-col gap-3 border border-border bg-elevated p-3 shadow-lg">
          <label className={fieldClassName}>
            {t("option:notesSearch.graphRadiusLabel", {
              defaultValue: "Distance from focus"
            })}
            <select
              className={controlClassName}
              aria-label={t("option:notesSearch.graphRadiusAria", {
                defaultValue: "Distance from focus note"
              })}
              value={radius}
              onChange={(event) => onRadiusChange(Number(event.target.value) as 1 | 2)}>
              <option value={1}>
                {t("option:notesSearch.graphRadiusOption.one", {
                  defaultValue: "1 link away"
                })}
              </option>
              <option value={2}>
                {t("option:notesSearch.graphRadiusOption.two", {
                  defaultValue: "2 links away"
                })}
              </option>
            </select>
          </label>
          <label className={fieldClassName}>
            {t("option:notesSearch.graphMaxNodesLabel", {
              defaultValue: "Most items shown"
            })}
            <input
              type="number"
              className={controlClassName}
              aria-label={t("option:notesSearch.graphMaxNodesAria", {
                defaultValue: "Most items shown in the graph"
              })}
              min={20}
              max={maxNodeCap}
              value={maxNodes}
              onChange={(event) => onMaxNodesChange(Number(event.target.value))}
            />
          </label>
          <label className={fieldClassName}>
            {t("option:notesSearch.graphLayoutLabel", { defaultValue: "Layout" })}
            <select
              className={controlClassName}
              aria-label={t("option:notesSearch.graphLayoutAria", {
                defaultValue: "Graph layout"
              })}
              value={layout}
              onChange={(event) => onLayoutChange(event.target.value as NotesGraphLayout)}>
              {LAYOUT_OPTIONS.map(({ value, defaultLabel }) => (
                <option key={value} value={value}>
                  {t(`option:notesSearch.graphLayoutOption.${value}`, { defaultValue: defaultLabel })}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
    </div>
  )
}

export default NotesGraphSettingsPopover
