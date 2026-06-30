/**
 * NodePalette Component
 *
 * Displays available step types organized by category.
 * Nodes can be dragged onto the canvas to add them.
 */

import { useState, useMemo, useEffect } from "react"
import { Input, Collapse, Tooltip, Spin, Button } from "antd"
import { Search, GripVertical } from "lucide-react"
import type { WorkflowStepType } from "@/types/workflow-editor"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { Alert } from "@/components/ui/primitives"
import { getCategorizedSteps, type StepTypeMetadata } from "./step-registry"
import { STEP_ICON_COMPONENTS, DEFAULT_STEP_ICON } from "./step-icons"

// Icon mapping for the palette
const STEP_ICONS = STEP_ICON_COMPONENTS

const PALETTE_ACTIVE_KEYS_STORAGE_KEY =
  "tldw:workflow-editor:palette:active-categories"
const DEFAULT_ACTIVE_CATEGORY_KEYS = [
  "ai",
  "search",
  "media",
  "control"
]

const STEP_SEARCH_ALIASES: Record<string, string[]> = {
  media_ingest: ["youtube", "yt", "download", "ingestion", "url"],
  rag_search: ["retrieval", "knowledge base", "kb", "vector", "semantic search"],
  web_search: ["internet", "google", "browser", "web lookup"],
  tts: ["text to speech", "voice", "speak", "audio output"],
  stt_transcribe: ["speech to text", "transcription", "voice to text", "whisper"],
  embed: ["embedding", "vectorize", "vector"],
  rerank: ["re-rank", "ranking", "sort relevance"],
  webhook: ["http", "api", "request", "post", "callback"],
  branch: ["if", "condition", "route", "decision"],
  map: ["for each", "loop", "iterate", "fan out"],
  llm: ["model", "chat", "completion", "generate text"],
  summarize: ["summary", "tl;dr", "condense"],
  image_gen: ["image generation", "picture", "art"],
  subtitle_generate: ["captions", "srt", "subtitles"],
  video_trim: ["cut video", "clip", "trim"]
}

const loadPaletteActiveKeys = (): string[] => {
  try {
    const raw = localStorage.getItem(PALETTE_ACTIVE_KEYS_STORAGE_KEY)
    if (!raw) return [...DEFAULT_ACTIVE_CATEGORY_KEYS]
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return [...DEFAULT_ACTIVE_CATEGORY_KEYS]
    return parsed
      .filter((key): key is string => typeof key === "string")
      .slice(0, 10)
  } catch {
    return [...DEFAULT_ACTIVE_CATEGORY_KEYS]
  }
}

const persistPaletteActiveKeys = (keys: string[]) => {
  try {
    localStorage.setItem(PALETTE_ACTIVE_KEYS_STORAGE_KEY, JSON.stringify(keys))
  } catch {
    // Best-effort persistence only.
  }
}

const normalizeActiveKeys = (
  keys: string[],
  categoryKeys: string[]
): string[] => {
  const unique = Array.from(new Set(keys))
  const normalized = unique.filter((key) => categoryKeys.includes(key))
  if (normalized.length > 0) return normalized
  return categoryKeys.slice(0, Math.min(6, categoryKeys.length))
}

const buildStepSearchText = (
  step: StepTypeMetadata,
  categoryLabel: string
): string => {
  const aliases = STEP_SEARCH_ALIASES[step.type] || []
  return [
    step.label,
    step.description,
    step.type,
    categoryLabel,
    ...aliases
  ]
    .join(" ")
    .toLowerCase()
}

// theme-exempt: workflow category colors
const CATEGORY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  purple: {
    bg: "bg-primary/10",
    text: "text-primary",
    border: "border-primary/30"
  },
  blue: {
    bg: "bg-primary/10",
    text: "text-primary",
    border: "border-primary/30"
  },
  indigo: {
    bg: "bg-indigo-100 dark:bg-indigo-900/30",
    text: "text-indigo-600 dark:text-indigo-400",
    border: "border-indigo-300 dark:border-indigo-700"
  },
  cyan: {
    bg: "bg-cyan-100 dark:bg-cyan-900/30",
    text: "text-cyan-600 dark:text-cyan-400",
    border: "border-cyan-300 dark:border-cyan-700"
  },
  violet: {
    bg: "bg-violet-100 dark:bg-violet-900/30",
    text: "text-violet-600 dark:text-violet-400",
    border: "border-violet-300 dark:border-violet-700"
  },
  teal: {
    bg: "bg-teal-100 dark:bg-teal-900/30",
    text: "text-teal-600 dark:text-teal-400",
    border: "border-teal-300 dark:border-teal-700"
  },
  emerald: {
    bg: "bg-emerald-100 dark:bg-emerald-900/30",
    text: "text-emerald-600 dark:text-emerald-400",
    border: "border-emerald-300 dark:border-emerald-700"
  },
  orange: {
    bg: "bg-indigo-100 dark:bg-indigo-900/30",
    text: "text-indigo-600 dark:text-indigo-400",
    border: "border-indigo-300 dark:border-indigo-700"
  },
  green: {
    bg: "bg-success/10",
    text: "text-success",
    border: "border-success/30"
  },
  gray: {
    bg: "bg-surface",
    text: "text-text-muted",
    border: "border-border"
  }
}

interface PaletteItemProps {
  step: StepTypeMetadata
  categoryColor: string
  onDragStart: (e: React.DragEvent, stepType: WorkflowStepType) => void
  onAddStep: (stepType: WorkflowStepType) => void
}

const PaletteItem = ({
  step,
  categoryColor,
  onDragStart,
  onAddStep
}: PaletteItemProps) => {
  const Icon = STEP_ICONS[step.icon] || DEFAULT_STEP_ICON
  const colors = CATEGORY_COLORS[categoryColor] || CATEGORY_COLORS.gray

  return (
    <Tooltip
      title={step.description}
      placement="right"
      mouseEnterDelay={0.5}
    >
      <div
        draggable
        onDragStart={(e) => onDragStart(e, step.type)}
        onClick={() => onAddStep(step.type)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            onAddStep(step.type)
          }
        }}
        role="button"
        tabIndex={0}
        aria-label={`Add ${step.label}`}
        className={`
          flex items-center gap-2 p-2 rounded-md cursor-grab
          border ${colors.border}
          ${colors.bg}
          hover:shadow-md hover:scale-[1.02]
          active:cursor-grabbing active:scale-100
          transition-all duration-150
        `}
      >
        <GripVertical className="w-3 h-3 text-text-subtle shrink-0" />
        <div className={`p-1.5 rounded ${colors.bg}`}>
          <Icon className={`w-4 h-4 ${colors.text}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-text truncate">
            {step.label}
          </div>
        </div>
      </div>
    </Tooltip>
  )
}

interface NodePaletteProps {
  className?: string
}

export const NodePalette = ({ className = "" }: NodePaletteProps) => {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeKeys, setActiveKeys] = useState<string[]>(() =>
    loadPaletteActiveKeys()
  )

  const stepRegistry = useWorkflowEditorStore((s) => s.stepRegistry)
  const stepTypesStatus = useWorkflowEditorStore((s) => s.stepTypesStatus)
  const stepTypesError = useWorkflowEditorStore((s) => s.stepTypesError)
  const loadStepTypes = useWorkflowEditorStore((s) => s.loadStepTypes)
  const addNode = useWorkflowEditorStore((s) => s.addNode)
  const nodes = useWorkflowEditorStore((s) => s.nodes)
  const zoom = useWorkflowEditorStore((s) => s.zoom)
  const panPosition = useWorkflowEditorStore((s) => s.panPosition)

  const categories = useMemo(
    () => getCategorizedSteps(stepRegistry),
    [stepRegistry]
  )

  const categoryKeys = useMemo(
    () => categories.map((cat) => cat.category),
    [categories]
  )

  useEffect(() => {
    setActiveKeys((current) => normalizeActiveKeys(current, categoryKeys))
  }, [categoryKeys])

  useEffect(() => {
    if (activeKeys.length === 0) return
    persistPaletteActiveKeys(activeKeys)
  }, [activeKeys])

  // Filter steps by search query
  const filteredCategories = useMemo(() => {
    if (!searchQuery.trim()) return categories

    const queryTokens = searchQuery
      .toLowerCase()
      .trim()
      .split(/\s+/)
      .filter(Boolean)

    return categories
      .map((cat) => ({
        ...cat,
        steps: cat.steps.filter(
          (step) => {
            const searchable = buildStepSearchText(step, cat.label)
            return queryTokens.every((token) => searchable.includes(token))
          }
        )
      }))
      .filter((cat) => cat.steps.length > 0)
  }, [categories, searchQuery])

  const handleDragStart = (e: React.DragEvent, stepType: WorkflowStepType) => {
    e.dataTransfer.setData("application/workflow-step", stepType)
    e.dataTransfer.effectAllowed = "copy"
  }

  const handleRetryStepTypes = () => {
    void loadStepTypes(true)
  }

  const getPaletteAddPosition = () => {
    const offsetIndex = nodes.length % 8
    const offsetX = offsetIndex * 28
    const offsetY = offsetIndex * 18
    const safeZoom = Math.max(zoom || 1, 0.001)
    const flowRoot = document.querySelector(".react-flow")

    if (flowRoot instanceof HTMLElement) {
      const rect = flowRoot.getBoundingClientRect()
      const centerClientX = rect.left + rect.width / 2
      const centerClientY = rect.top + rect.height / 2
      return {
        x: (centerClientX - rect.left - panPosition.x) / safeZoom + offsetX,
        y: (centerClientY - rect.top - panPosition.y) / safeZoom + offsetY
      }
    }

    // Fallback when canvas bounds are unavailable (tests or early mount)
    return {
      x: 260 + offsetX,
      y: 220 + offsetY
    }
  }

  const handleAddStep = (stepType: WorkflowStepType) => {
    addNode({
      type: stepType,
      position: getPaletteAddPosition()
    })
  }

  const collapseItems = filteredCategories.map((cat) => ({
    key: cat.category,
    label: (
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full bg-${cat.color}-500`}
          style={{ backgroundColor: `var(--${cat.color}-500, #888)` }}
        />
        <span className="font-medium">{cat.label}</span>
        <span className="text-xs text-text-subtle">({cat.steps.length})</span>
      </div>
    ),
    children: (
      <div className="flex flex-col gap-2 pb-2">
        {cat.steps.map((step) => (
          <PaletteItem
            key={step.type}
            step={step}
            categoryColor={cat.color}
            onDragStart={handleDragStart}
            onAddStep={handleAddStep}
          />
        ))}
      </div>
    )
  }))

  const effectiveActiveKeys = searchQuery.trim()
    ? filteredCategories.map((cat) => cat.category)
    : activeKeys

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-3 border-b border-border">
        <h3 className="text-sm font-semibold text-text-muted mb-2">
          Node Library
        </h3>
        <Input
          prefix={<Search className="w-4 h-4 text-text-subtle" />}
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          allowClear
          size="small"
        />
      </div>

      {stepTypesStatus === "error" && (
        <div className="px-3 pt-2">
          <Alert
            variant="warning"
            title="Limited node library"
            action={{
              label: "Retry",
              onClick: handleRetryStepTypes
            }}
          >
            <span className="text-xs">
              Could not load server step types. Showing fallback steps.
            </span>
          </Alert>
        </div>
      )}

      {/* Node Categories */}
      <div className="flex-1 overflow-y-auto p-2">
        {stepTypesStatus === "loading" ? (
          <div className="flex items-center justify-center py-10 text-text-subtle">
            <Spin size="small" />
            <span className="ml-2 text-sm">Loading steps…</span>
          </div>
        ) : filteredCategories.length === 0 ? (
          <div className="text-center text-text-subtle py-8">
            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">
              {stepTypesError ? "Unable to load step types" : "No nodes found"}
            </p>
            {stepTypesError && (
              <Button
                type="link"
                size="small"
                onClick={handleRetryStepTypes}
                className="mt-2"
              >
                Retry loading step types
              </Button>
            )}
          </div>
        ) : (
          <Collapse
            activeKey={effectiveActiveKeys}
            onChange={(keys) => {
              const nextKeys = Array.isArray(keys)
                ? (keys as string[])
                : keys
                ? [keys as string]
                : []
              setActiveKeys(normalizeActiveKeys(nextKeys, categoryKeys))
            }}
            ghost
            expandIconPlacement="end"
            items={collapseItems}
            className="workflow-node-palette"
          />
        )}
      </div>

      {/* Help text */}
      <div className="p-3 border-t border-border">
        <p className="text-xs text-text-subtle text-center">
          Drag, click, or press Enter to add nodes
        </p>
      </div>
    </div>
  )
}

export default NodePalette
