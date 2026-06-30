import React from "react"
import { Popover } from "antd"
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  CircleDashed,
  Languages,
  Layers,
  Loader2,
  User2
} from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  CharacterSelect,
  type SidepanelAssistantSelectOpenDetail,
  type SidepanelAssistantSelectTab
} from "./CharacterSelect"
import {
  formatContextSourceLabel,
  resolveContextReadiness,
  summarizeConversationContextPieces
} from "./conversation-context-utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { normalizeConversationContextIdList } from "@/services/conversation-context/conversationContextSettings"
import type {
  ConversationContextComposition,
  ConversationContextSelection
} from "@/types/conversation-context"
import type { ConversationContextCompositionStatus } from "@/hooks/chat/useConversationContextComposition"

export type ConversationContextAssetOption = {
  id: number
  name: string
  disabled?: boolean
}

type ConversationContextPopoverProps = {
  chatId?: string | null
  selectedCharacterId: string | null
  setSelectedCharacterId: (id: string | null) => void
  composition?: ConversationContextComposition | null
  compositionStatus: ConversationContextCompositionStatus
  saveSelection?: (
    selection: Pick<
      ConversationContextSelection,
      "worldBookIds" | "dictionaryIds"
    >
  ) => Promise<unknown> | unknown
  worldBookOptions?: ConversationContextAssetOption[]
  dictionaryOptions?: ConversationContextAssetOption[]
  disabled?: boolean
  className?: string
  iconClassName?: string
}

const triggerBaseClass =
  "flex min-w-[44px] sm:min-w-[104px] min-h-[44px] sm:min-h-0 h-9 items-center justify-center gap-2 rounded px-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"

const toneClass = {
  ready: "text-text-muted hover:bg-surface2 hover:text-text",
  partial: "bg-warning/10 text-warning hover:bg-warning/20",
  blocked: "bg-error/10 text-error hover:bg-error/20",
  loading: "text-text-muted hover:bg-surface2 hover:text-text"
} as const

const readinessIcon = (tone: keyof typeof toneClass) => {
  if (tone === "loading") return <Loader2 className="size-3.5 animate-spin" />
  if (tone === "blocked") return <AlertTriangle className="size-3.5" />
  if (tone === "partial") return <CircleDashed className="size-3.5" />
  return <CheckCircle2 className="size-3.5" />
}

const toAssetOption = (
  value: unknown,
  kind: "worldbook" | "dictionary"
): ConversationContextAssetOption | null => {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const rawId =
    kind === "worldbook" ? record.world_book_id ?? record.id : record.id
  const id = Number(rawId)
  if (!Number.isFinite(id) || id <= 0) return null
  const rawName =
    kind === "worldbook"
      ? record.world_book_name ?? record.name
      : record.name
  const name =
    typeof rawName === "string" && rawName.trim()
      ? rawName.trim()
      : kind === "worldbook"
        ? `Worldbook ${Math.trunc(id)}`
        : `Dictionary ${Math.trunc(id)}`
  return {
    id: Math.trunc(id),
    name,
    disabled:
      kind === "dictionary" && typeof record.is_active === "boolean"
        ? !record.is_active
        : false
  }
}

const normalizeAssetOptions = (
  values: unknown,
  kind: "worldbook" | "dictionary"
): ConversationContextAssetOption[] => {
  const list = Array.isArray(values) ? values : []
  const seen = new Set<number>()
  const options: ConversationContextAssetOption[] = []
  for (const value of list) {
    const option = toAssetOption(value, kind)
    if (!option || seen.has(option.id)) continue
    seen.add(option.id)
    options.push(option)
  }
  return options.sort((a, b) => a.name.localeCompare(b.name))
}

const toggleId = (ids: number[], id: number, checked: boolean): number[] => {
  const next = new Set(ids)
  if (checked) {
    next.add(id)
  } else {
    next.delete(id)
  }
  return Array.from(next).sort((a, b) => a - b)
}

const Section = ({
  icon,
  title,
  value,
  detail
}: {
  icon: React.ReactNode
  title: string
  value: string
  detail?: React.ReactNode
}) => (
  <div className="border-t border-border py-2 first:border-t-0">
    <div className="flex items-center justify-between gap-3">
      <div className="flex min-w-0 items-center gap-2">
        <span className="text-text-subtle">{icon}</span>
        <span className="truncate text-sm font-medium text-text">{title}</span>
      </div>
      <span className="shrink-0 text-xs text-text-subtle">{value}</span>
    </div>
    {detail ? <div className="mt-2 text-xs text-text-subtle">{detail}</div> : null}
  </div>
)

export const ConversationContextPopover: React.FC<
  ConversationContextPopoverProps
> = ({
  chatId,
  selectedCharacterId,
  setSelectedCharacterId,
  composition,
  compositionStatus,
  saveSelection,
  worldBookOptions,
  dictionaryOptions,
  disabled = false,
  className = "",
  iconClassName = "size-4"
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [open, setOpen] = React.useState(false)
  const [loadedWorldBookOptions, setLoadedWorldBookOptions] = React.useState<
    ConversationContextAssetOption[]
  >([])
  const [loadedDictionaryOptions, setLoadedDictionaryOptions] = React.useState<
    ConversationContextAssetOption[]
  >([])
  const [assetOptionsLoading, setAssetOptionsLoading] = React.useState(false)
  const [assetOptionsError, setAssetOptionsError] =
    React.useState<string | null>(null)
  const [saving, setSaving] = React.useState(false)
  const [assistantSelectOpenRequest, setAssistantSelectOpenRequest] =
    React.useState<{
      id: number
      tab?: SidepanelAssistantSelectTab
    } | null>(null)
  const summary = React.useMemo(
    () => summarizeConversationContextPieces(composition),
    [composition]
  )
  const readiness = React.useMemo(
    () => resolveContextReadiness({ composition, status: compositionStatus }),
    [composition, compositionStatus]
  )
  const hasOptionalContext =
    summary.worldbooks.total > 0 || summary.dictionaries.total > 0
  const worldbookValue =
    summary.worldbooks.total > 0
      ? `${summary.worldbooks.matched} ${t("sidepanel:conversationContext.matched", "matched")} / ${summary.worldbooks.total} ${t("sidepanel:conversationContext.configured", "configured")}`
      : t("sidepanel:conversationContext.none", "None")
  const dictionaryValue =
    summary.dictionaries.total > 0
      ? `${summary.dictionaries.active} ${t("sidepanel:conversationContext.active", "active")} / ${summary.dictionaries.total} ${t("sidepanel:conversationContext.configured", "configured")}`
      : t("sidepanel:conversationContext.none", "None")

  const previewSections = composition?.previewSections ?? []
  const selectedWorldBookIds = React.useMemo(
    () =>
      normalizeConversationContextIdList(composition?.selection.worldBookIds),
    [composition]
  )
  const selectedDictionaryIds = React.useMemo(
    () =>
      normalizeConversationContextIdList(composition?.selection.dictionaryIds),
    [composition]
  )
  const resolvedWorldBookOptions = worldBookOptions ?? loadedWorldBookOptions
  const resolvedDictionaryOptions =
    dictionaryOptions ?? loadedDictionaryOptions
  const canEditAssets = Boolean(chatId && saveSelection && !disabled)
  const sources = React.useMemo(() => {
    const labels = new Set<string>()
    for (const piece of composition?.pieces ?? []) {
      labels.add(formatContextSourceLabel(piece.source))
    }
    return Array.from(labels)
  }, [composition])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handleOpenAssistantSelect = (event: Event) => {
      const detail = (event as CustomEvent<SidepanelAssistantSelectOpenDetail>)
        .detail
      setOpen(true)
      setAssistantSelectOpenRequest((previous) => ({
        id: (previous?.id ?? 0) + 1,
        tab: detail?.tab
      }))
    }
    window.addEventListener(
      "tldw:open-sidepanel-assistant-select",
      handleOpenAssistantSelect
    )
    return () => {
      window.removeEventListener(
        "tldw:open-sidepanel-assistant-select",
        handleOpenAssistantSelect
      )
    }
  }, [])

  React.useEffect(() => {
    if (!open) return
    if (worldBookOptions && dictionaryOptions) return
    let cancelled = false
    setAssetOptionsLoading(true)
    setAssetOptionsError(null)

    const load = async () => {
      try {
        await tldwClient.initialize()
        const [worldBookResponse, dictionaryResponse] = await Promise.all([
          worldBookOptions
            ? Promise.resolve(null)
            : tldwClient.listWorldBooks(false),
          dictionaryOptions
            ? Promise.resolve(null)
            : tldwClient.listDictionaries(false, true)
        ])
        if (cancelled) return
        if (!worldBookOptions) {
          setLoadedWorldBookOptions(
            normalizeAssetOptions(
              worldBookResponse?.world_books ?? worldBookResponse,
              "worldbook"
            )
          )
        }
        if (!dictionaryOptions) {
          setLoadedDictionaryOptions(
            normalizeAssetOptions(
              dictionaryResponse?.dictionaries ?? dictionaryResponse,
              "dictionary"
            )
          )
        }
      } catch (error) {
        if (!cancelled) {
          setAssetOptionsError(
            error instanceof Error ? error.message : String(error)
          )
        }
      } finally {
        if (!cancelled) setAssetOptionsLoading(false)
      }
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [dictionaryOptions, open, worldBookOptions])

  const saveAssetSelection = React.useCallback(
    async (nextSelection: Pick<ConversationContextSelection, "worldBookIds" | "dictionaryIds">) => {
      if (!canEditAssets || !saveSelection) return
      setSaving(true)
      try {
        await saveSelection(nextSelection)
      } catch (error) {
        setAssetOptionsError(
          error instanceof Error ? error.message : String(error)
        )
      } finally {
        setSaving(false)
      }
    },
    [canEditAssets, saveSelection]
  )

  const handleWorldBookToggle = React.useCallback(
    (id: number, checked: boolean) => {
      void saveAssetSelection({
        worldBookIds: toggleId(selectedWorldBookIds, id, checked),
        dictionaryIds: selectedDictionaryIds
      })
    },
    [saveAssetSelection, selectedDictionaryIds, selectedWorldBookIds]
  )

  const handleDictionaryToggle = React.useCallback(
    (id: number, checked: boolean) => {
      void saveAssetSelection({
        worldBookIds: selectedWorldBookIds,
        dictionaryIds: toggleId(selectedDictionaryIds, id, checked)
      })
    },
    [saveAssetSelection, selectedDictionaryIds, selectedWorldBookIds]
  )

  const renderAssetOptions = (
    options: ConversationContextAssetOption[],
    selectedIds: number[],
    onToggle: (id: number, checked: boolean) => void,
    emptyLabel: string
  ) => {
    if (options.length === 0) {
      return (
        <div className="text-xs text-text-subtle">
          {assetOptionsLoading
            ? t("sidepanel:conversationContext.loadingAssets", "Loading...")
            : emptyLabel}
        </div>
      )
    }

    return (
      <div className="grid max-h-28 gap-1 overflow-auto pr-1">
        {options.map((option) => {
          const checked = selectedIds.includes(option.id)
          return (
            <label
              key={option.id}
              className="flex min-h-7 items-center gap-2 text-xs text-text"
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={!canEditAssets || saving || option.disabled}
                onChange={(event) => onToggle(option.id, event.target.checked)}
              />
              <span className="min-w-0 truncate">{option.name}</span>
            </label>
          )
        })}
      </div>
    )
  }

  const content = (
    <div
      className="w-[320px] max-w-[calc(100vw-32px)] p-3"
      onKeyDown={(event) => {
        if (event.key === "Escape") setOpen(false)
      }}
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <Layers className="size-4 text-text-subtle" />
          <span className="truncate text-sm font-semibold text-text">
            {t("sidepanel:conversationContext.title", "Conversation context")}
          </span>
        </div>
        <span
          className={`shrink-0 rounded px-2 py-0.5 text-xs ${
            readiness.tone === "ready"
              ? "bg-success/10 text-success"
              : readiness.tone === "partial" || readiness.tone === "loading"
                ? "bg-warning/10 text-warning"
                : "bg-error/10 text-error"
          }`}
        >
          {readiness.label}
        </span>
      </div>

      <Section
        icon={<User2 className="size-3.5" />}
        title={t("sidepanel:conversationContext.character", "Character")}
        value={
          selectedCharacterId
            ? t("sidepanel:conversationContext.selected", "Selected")
            : t("sidepanel:conversationContext.none", "None")
        }
        detail={
          <CharacterSelect
            selectedCharacterId={selectedCharacterId}
            setSelectedCharacterId={setSelectedCharacterId}
            iconClassName="size-4"
            className="px-2 text-text-muted hover:text-text"
            openRequest={assistantSelectOpenRequest ?? undefined}
          />
        }
      />
      <Section
        icon={<BookOpen className="size-3.5" />}
        title={t("sidepanel:conversationContext.worldbooks", "Worldbooks")}
        value={worldbookValue}
        detail={
          summary.worldbooks.total > 0 ? (
            <div className="space-y-2">
              <span>
                {summary.worldbooks.explicit}{" "}
                {t("sidepanel:conversationContext.chatScoped", "chat-scoped")}
                {summary.worldbooks.inherited > 0
                  ? `, ${summary.worldbooks.inherited} ${t(
                      "sidepanel:conversationContext.inherited",
                      "inherited"
                    )}`
                  : ""}
              </span>
              {renderAssetOptions(
                resolvedWorldBookOptions,
                selectedWorldBookIds,
                handleWorldBookToggle,
                t("sidepanel:conversationContext.noWorldbooks", "No worldbooks")
              )}
            </div>
          ) : (
            renderAssetOptions(
              resolvedWorldBookOptions,
              selectedWorldBookIds,
              handleWorldBookToggle,
              t("sidepanel:conversationContext.noWorldbooks", "No worldbooks")
            )
          )
        }
      />
      <Section
        icon={<Languages className="size-3.5" />}
        title={t("sidepanel:conversationContext.dictionaries", "Dictionaries")}
        value={dictionaryValue}
        detail={
          summary.dictionaries.total > 0 ? (
            <div className="space-y-2">
              <span>
                {summary.dictionaries.explicit}{" "}
                {t("sidepanel:conversationContext.chatScoped", "chat-scoped")}
              </span>
              {renderAssetOptions(
                resolvedDictionaryOptions,
                selectedDictionaryIds,
                handleDictionaryToggle,
                t(
                  "sidepanel:conversationContext.noDictionaries",
                  "No dictionaries"
                )
              )}
            </div>
          ) : (
            renderAssetOptions(
              resolvedDictionaryOptions,
              selectedDictionaryIds,
              handleDictionaryToggle,
              t("sidepanel:conversationContext.noDictionaries", "No dictionaries")
            )
          )
        }
      />

      {assetOptionsError ? (
        <div className="border-t border-border py-2 text-xs text-error">
          {assetOptionsError}
        </div>
      ) : null}

      {!hasOptionalContext && !selectedCharacterId ? (
        <div className="border-t border-border py-2 text-xs text-text-subtle">
          {t(
            "sidepanel:conversationContext.noOptionalContext",
            "No optional context"
          )}
        </div>
      ) : null}

      {previewSections.length > 0 ? (
        <div className="border-t border-border pt-2">
          <div className="mb-1 text-xs font-medium text-text-subtle">
            {t("sidepanel:conversationContext.preview", "Preview")}
          </div>
          <div className="max-h-32 space-y-2 overflow-auto pr-1">
            {previewSections.map((section, index) => (
              <div key={`${section.name}-${index}`}>
                <div className="text-xs font-medium text-text">
                  {section.name}
                </div>
                <div className="line-clamp-3 whitespace-pre-wrap text-xs text-text-subtle">
                  {section.content}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {sources.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1 border-t border-border pt-2">
          {sources.map((source) => (
            <span
              key={source}
              className="rounded bg-surface2 px-1.5 py-0.5 text-[11px] text-text-subtle"
            >
              {source}
            </span>
          ))}
        </div>
      ) : null}

      {!chatId ? (
        <div className="mt-2 border-t border-border pt-2 text-xs text-text-subtle">
          {t(
            "sidepanel:conversationContext.unsavedChat",
            "Edits are available after the chat is created."
          )}
        </div>
      ) : null}
    </div>
  )

  return (
    <Popover
      trigger="click"
      open={open}
      onOpenChange={setOpen}
      placement="topLeft"
      content={content}
    >
      <button
        type="button"
        data-testid="conversation-context-trigger"
        className={`${triggerBaseClass} ${toneClass[readiness.tone]} ${className}`}
        disabled={disabled}
        aria-disabled={disabled}
        aria-label={t(
          "sidepanel:conversationContext.tooltip",
          "Conversation context"
        )}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={t(
          "sidepanel:conversationContext.tooltip",
          "Conversation context"
        )}
      >
        <Layers className={iconClassName} />
        <span className="hidden truncate sm:inline">
          {t("sidepanel:conversationContext.shortLabel", "Context")}
        </span>
        <span className="flex h-4 w-4 shrink-0 items-center justify-center">
          {readinessIcon(readiness.tone)}
        </span>
      </button>
    </Popover>
  )
}

export default ConversationContextPopover
