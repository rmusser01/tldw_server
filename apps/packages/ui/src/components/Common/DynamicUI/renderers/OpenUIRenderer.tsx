import { Renderer, createLibrary, type ActionEvent } from "@openuidev/react-lang"
import "@openuidev/react-ui/components.css"
import "@openuidev/react-ui/defaults.css"
import { openuiChatLibrary } from "@openuidev/react-ui/genui-lib"
import type { CSSProperties } from "react"
import type { DynamicUIRendererProps } from "../registry"

const BLOCKED_OPENUI_COMPONENT_NAMES = new Set([
  "BarChart",
  "LineChart",
  "AreaChart",
  "RadarChart",
  "HorizontalBarChart",
  "Series",
  "PieChart",
  "RadialChart",
  "SingleStackedBarChart",
  "Slice",
  "ScatterChart",
  "ScatterSeries",
  "Point"
])

const BLOCKED_OPENUI_GROUP_PREFIX = "Charts"

const openUIThemeStyle = {
  "--openui-background": "rgb(var(--color-surface))",
  "--openui-foreground": "rgb(var(--color-surface))",
  "--openui-popover-background": "rgb(var(--color-elevated))",
  "--openui-text-neutral-primary": "rgb(var(--color-text))",
  "--openui-text-neutral-secondary": "rgb(var(--color-text-muted))",
  "--openui-text-neutral-tertiary": "rgb(var(--color-text-subtle))",
  "--openui-text-neutral-link": "rgb(var(--color-primary))",
  "--openui-text-brand": "rgb(var(--color-primary))",
  "--openui-interactive-accent-default": "rgb(var(--color-primary))",
  "--openui-interactive-accent-hover": "rgb(var(--color-primary-strong))",
  "--openui-interactive-accent-pressed": "rgb(var(--color-primary-strong))",
  "--openui-interactive-accent-disabled": "rgb(var(--color-primary) / 0.45)",
  "--openui-border-default": "rgb(var(--color-border))",
  "--openui-border-interactive": "rgb(var(--color-border-strong))",
  "--openui-border-interactive-selected": "rgb(var(--color-primary))",
  "--openui-border-accent": "rgb(var(--color-border))",
  "--openui-border-accent-emphasis": "rgb(var(--color-primary))",
  "--openui-font-body": "var(--font-family)",
  "--openui-font-heading": "var(--font-family)",
  "--openui-font-label": "var(--font-family)",
  "--openui-font-code": "var(--font-family-mono)"
} as CSSProperties

const safeOpenUIChatLibrary = createLibrary({
  root: openuiChatLibrary.root,
  components: Object.values(openuiChatLibrary.components).filter(
    (component) => !BLOCKED_OPENUI_COMPONENT_NAMES.has(component.name)
  ),
  componentGroups: (openuiChatLibrary.componentGroups ?? [])
    .filter((group) => !group.name.startsWith(BLOCKED_OPENUI_GROUP_PREFIX))
    .map((group) => ({
      ...group,
      components: group.components.filter(
        (componentName) => !BLOCKED_OPENUI_COMPONENT_NAMES.has(componentName)
      )
    }))
    .filter((group) => group.components.length > 0)
})

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const unwrapOpenUIFieldValue = (value: unknown): unknown => {
  if (isRecord(value) && Object.prototype.hasOwnProperty.call(value, "value")) {
    return value.value
  }
  return value
}

const normalizeOpenUIFormValues = (
  formState: ActionEvent["formState"],
  formName: ActionEvent["formName"]
): Record<string, unknown> => {
  if (!isRecord(formState)) return {}
  const formValues =
    typeof formName === "string" && isRecord(formState[formName])
      ? formState[formName]
      : formState

  return Object.fromEntries(
    Object.entries(formValues).map(([key, value]) => [key, unwrapOpenUIFieldValue(value)])
  )
}

const stripActionParams = (params: ActionEvent["params"]): Record<string, unknown> => {
  if (!isRecord(params)) return {}
  return Object.fromEntries(
    Object.entries(params).filter(([key]) => key !== "actionId" && key !== "actionType")
  )
}

const getOpenUIActionId = (event: ActionEvent): string => {
  const params = isRecord(event.params) ? event.params : null
  if (typeof params?.actionId === "string" && params.actionId.trim()) {
    return params.actionId.trim()
  }
  if (typeof event.formName === "string" && event.formName.trim()) {
    return event.formName.trim()
  }
  if (typeof event.type === "string" && event.type.trim()) {
    return event.type.trim()
  }
  return "submit"
}

const normalizeOpenUIAction = (event: ActionEvent): Record<string, unknown> => {
  const formValues = normalizeOpenUIFormValues(event.formState, event.formName)
  const fallbackValues = stripActionParams(event.params)

  return {
    actionId: getOpenUIActionId(event),
    actionType: "submit",
    values: Object.keys(formValues).length > 0 ? formValues : fallbackValues
  }
}

const OpenUIRenderer = ({ envelope, source, onAction }: DynamicUIRendererProps) => (
  <div
    data-testid="dynamic-ui-openui-shell"
    className="dynamic-ui-openui max-w-full overflow-x-auto rounded-md border border-border bg-surface p-3 text-text"
    style={openUIThemeStyle}>
    <Renderer
      response={source}
      library={safeOpenUIChatLibrary}
      isStreaming={false}
      initialState={envelope.state}
      onAction={(event) => onAction?.(normalizeOpenUIAction(event))}
      toolProvider={null}
    />
  </div>
)

export default OpenUIRenderer
