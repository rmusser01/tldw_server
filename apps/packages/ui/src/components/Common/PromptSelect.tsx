import { useQuery } from "@tanstack/react-query"
import { Dropdown, Empty, Input, Modal, Tooltip } from "antd"
import type { InputRef } from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"
import type { ItemType, MenuItemType } from "antd/es/menu/interface"
import { BookIcon, ComputerIcon, ZapIcon, Search } from "lucide-react"
import React, { useState, useMemo, useRef, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { getAllPrompts } from "@/db/dexie/helpers"
import type { Prompt } from "@/db/dexie/types"
import { getDesignSystemState } from "@/design-system"
import { useStorage } from "@plasmohq/storage/hook"
import {
  OPEN_PROMPT_SELECT_EVENT,
  type PromptSelectOpenDetail
} from "@/utils/prompt-select-events"
import {
  normalizeFocusSelector,
  scheduleFocusFirstVisibleElement
} from "@/utils/focus-return"
import { IconButton } from "./IconButton"
import { PromptAssistMenu } from "./PromptAssist/PromptAssistMenu"
import { PromptAssistPanel } from "./PromptAssist/PromptAssistPanel"
import {
  usePromptAssist,
  type PromptTargetAdapter
} from "./PromptAssist/usePromptAssist"
import type { PromptImproveModelSelection } from "@/services/prompt-improvement"
import { fetchPromptCapabilities } from "@/services/prompts-api"
import {
  captureSystemPromptOverrideSnapshot,
  normalizeSystemPromptOverrideValue,
  resolveEffectiveSystemPromptState,
  resolveSelectedSystemPromptContent,
  restoreSystemPromptOverrideSnapshot,
  type SystemPromptOverrideSnapshot
} from "./system-prompt-utils"

type Props = {
  setSelectedSystemPrompt: (promptId: string | undefined) => void
  setSelectedQuickPrompt: (prompt: string | undefined) => void
  selectedSystemPrompt: string | undefined
  systemPrompt: string | undefined
  setSystemPrompt: (prompt: string | undefined) => void
  selectedModel?: string | null
  currentProvider?: string | null
  promptAssistContextKey?: string
  promptAssistBackendKey?: string | null
  onSelectModel?: () => void
  className?: string
  iconClassName?: string
}

type SystemPromptUndoSnapshot = {
  override: SystemPromptOverrideSnapshot
  editorDraft: string
}

export const PromptSelect: React.FC<Props> = ({
  setSelectedQuickPrompt,
  setSelectedSystemPrompt,
  selectedSystemPrompt,
  systemPrompt,
  setSystemPrompt,
  selectedModel,
  currentProvider,
  promptAssistContextKey = "system-prompt",
  promptAssistBackendKey = null,
  onSelectModel,
  className = "text-text-muted",
  iconClassName = "size-5"
}) => {
  const { t } = useTranslation("option")
  const [menuDensity] = useStorage("menuDensity", "comfortable")
  const [searchText, setSearchText] = useState("")
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [editorOpen, setEditorOpen] = useState(false)
  const [editorLoading, setEditorLoading] = useState(false)
  const [editorDraft, setEditorDraft] = useState("")
  const [editorTemplateContent, setEditorTemplateContent] = useState("")
  const [editorOverrideActive, setEditorOverrideActive] = useState(false)
  const searchInputRef = useRef<InputRef | null>(null)
  const editorInputRef = useRef<TextAreaRef | null>(null)
  const editorFocusRequestedRef = useRef(false)
  const returnFocusSelectorRef = useRef<string | null>(null)
  const editorDraftRef = useRef("")
  const editorTemplateContentRef = useRef("")
  const editorRevisionRef = useRef(0)
  const editorLookupEpochRef = useRef(0)
  const editorMountedRef = useRef(true)
  const editorOpenRef = useRef(false)
  const rawSystemPromptRef = useRef(systemPrompt)
  const assistEntryDraftRef = useRef("")

  rawSystemPromptRef.current = systemPrompt

  const modelSelection = useMemo<PromptImproveModelSelection | null>(() => {
    const normalizedModel = selectedModel?.trim()
    if (!normalizedModel) return null
    const normalizedProvider = currentProvider?.trim()
    return {
      selected_model: normalizedModel,
      ...(normalizedProvider ? { provider_hint: normalizedProvider } : {})
    }
  }, [currentProvider, selectedModel])
  const modelSelectionRef = useRef(modelSelection)
  modelSelectionRef.current = modelSelection

  const normalizedPromptAssistBackendKey =
    promptAssistBackendKey?.trim() || null
  const editorLifecycleKey = JSON.stringify([
    promptAssistContextKey,
    selectedSystemPrompt ?? "",
    currentProvider?.trim() ?? "",
    selectedModel?.trim() ?? "",
    normalizedPromptAssistBackendKey ?? ""
  ])
  const editorLifecycleKeyRef = useRef(editorLifecycleKey)
  if (editorLifecycleKeyRef.current !== editorLifecycleKey) {
    editorLifecycleKeyRef.current = editorLifecycleKey
    editorLookupEpochRef.current += 1
  }

  const { data: promptCapabilities } = useQuery({
    queryKey: ["promptCapabilities", normalizedPromptAssistBackendKey],
    queryFn: fetchPromptCapabilities,
    enabled: Boolean(normalizedPromptAssistBackendKey),
    retry: false
  })
  const promptAssistCapability = !promptCapabilities
    ? "unknown"
    : promptCapabilities.availability === "available" &&
        promptCapabilities.prompt_improvement_v1.supported
      ? "supported"
      : "unsupported"

  const updateEditorDraft = React.useCallback((nextDraft: string) => {
    editorLookupEpochRef.current += 1
    setEditorLoading(false)
    editorDraftRef.current = nextDraft
    editorRevisionRef.current += 1
    setEditorDraft(nextDraft)
  }, [])

  const applySystemPromptCandidate = React.useCallback(
    (candidate: string) => {
      const normalized = normalizeSystemPromptOverrideValue({
        draft: candidate,
        templateContent: editorTemplateContentRef.current
      })
      updateEditorDraft(candidate)
      setSystemPrompt(normalized)
      setEditorOverrideActive(
        Boolean(selectedSystemPrompt) && normalized.trim().length > 0
      )
    },
    [selectedSystemPrompt, setSystemPrompt, updateEditorDraft]
  )

  const promptTargetAdapter = useMemo<PromptTargetAdapter>(
    () => ({
      target: "system",
      read: () => editorDraftRef.current,
      readRevision: () => String(editorRevisionRef.current),
      apply: applySystemPromptCandidate,
      captureUndo: (): SystemPromptUndoSnapshot => ({
        override: captureSystemPromptOverrideSnapshot(
          rawSystemPromptRef.current
        ),
        editorDraft: editorDraftRef.current
      }),
      restoreUndo: (snapshot) => {
        const restored = snapshot as SystemPromptUndoSnapshot
        const rawOverride = restoreSystemPromptOverrideSnapshot(
          restored.override
        )
        updateEditorDraft(restored.editorDraft)
        setSystemPrompt(rawOverride)
        setEditorOverrideActive(
          Boolean(selectedSystemPrompt) &&
            typeof rawOverride === "string" &&
            rawOverride.trim().length > 0 &&
            rawOverride !== editorTemplateContentRef.current
        )
      }
    }),
    [
      applySystemPromptCandidate,
      selectedSystemPrompt,
      setSystemPrompt,
      updateEditorDraft
    ]
  )

  const promptAssist = usePromptAssist({
    adapter: promptTargetAdapter,
    readActiveRoute: () =>
      modelSelectionRef.current ?? { selected_model: "" },
    limits:
      promptCapabilities?.prompt_improvement_v1.supported === true
        ? promptCapabilities.prompt_improvement_v1.limits
        : null,
    contextKey: editorLifecycleKey,
    surfaceOpen: editorOpen
  })

  useEffect(() => {
    editorMountedRef.current = true
    return () => {
      editorMountedRef.current = false
      editorLookupEpochRef.current += 1
    }
  }, [])

  useEffect(() => {
    setEditorLoading(false)
  }, [editorLifecycleKey])

  const requestEditorFocus = React.useCallback(() => {
    editorFocusRequestedRef.current = true
    if (editorInputRef.current) {
      editorFocusRequestedRef.current = false
      editorInputRef.current.focus()
    }
  }, [])

  useEffect(() => {
    if (
      !editorOpen ||
      !editorFocusRequestedRef.current ||
      (promptAssist.state.status !== "idle" &&
        promptAssist.state.status !== "applied")
    ) {
      return
    }
    editorFocusRequestedRef.current = false
    editorInputRef.current?.focus()
  }, [editorOpen, promptAssist.state.status])

  const restorePromptSelectFocus = React.useCallback(() => {
    const returnFocusSelector =
      returnFocusSelectorRef.current ?? "[data-testid='chat-prompt-select']"
    returnFocusSelectorRef.current = null

    scheduleFocusFirstVisibleElement(returnFocusSelector)
  }, [])

  const {
    data,
    isLoading: promptsLoading,
    isError: promptsError,
    refetch: refetchPrompts
  } = useQuery({
    queryKey: ["getAllPromptsForSelect"],
    queryFn: getAllPrompts
  })

  // Filter prompts based on search text
  const filteredData = useMemo<Prompt[]>(() => {
    if (!data) return []
    if (!searchText.trim()) return data
    const q = searchText.toLowerCase()
    return data.filter(
      (prompt) =>
        prompt.title?.toLowerCase().includes(q) ||
        prompt.content?.toLowerCase().includes(q)
    )
  }, [data, searchText])

  const handlePromptChange = React.useCallback((value?: string) => {
    if (!value) {
      setSelectedSystemPrompt(undefined)
      setSelectedQuickPrompt(undefined)
      return
    }
    const prompt = data?.find((prompt) => prompt.id === value)
    if (!prompt) return
    if (prompt?.is_system) {
      setSelectedSystemPrompt(prompt.id)
    } else {
      setSelectedSystemPrompt(undefined)
      setSelectedQuickPrompt(prompt.content)
    }
  }, [data, setSelectedSystemPrompt, setSelectedQuickPrompt])

  const selectedPromptLabel = useMemo(() => {
    if (!selectedSystemPrompt || !data) return null
    const prompt = data.find((item) => item.id === selectedSystemPrompt)
    return prompt?.title || null
  }, [data, selectedSystemPrompt])

  const promptLoadingLabel = t(
    "promptSelect.loadingPrompts",
    "Loading prompts"
  )
  const promptUnavailableLabel = t(
    "promptSelect.libraryUnavailable",
    "Prompt library unavailable"
  )
  const promptRetryLabel = t(
    "promptSelect.retryPromptLibrary",
    "Retry prompt library"
  )

  const openSystemPromptEditor = React.useCallback(async () => {
    setDropdownOpen(false)
    promptAssist.dismiss()
    editorOpenRef.current = true
    setEditorOpen(true)
    const lookupEpoch = ++editorLookupEpochRef.current
    const lifecycleKey = editorLifecycleKeyRef.current
    setEditorLoading(true)
    const resolved = await resolveEffectiveSystemPromptState({
      selectedSystemPrompt,
      systemPrompt
    })
    if (
      !editorMountedRef.current ||
      !editorOpenRef.current ||
      editorLookupEpochRef.current !== lookupEpoch ||
      editorLifecycleKeyRef.current !== lifecycleKey
    ) {
      return
    }
    editorTemplateContentRef.current = resolved.templateContent
    editorDraftRef.current = resolved.effectiveContent
    editorRevisionRef.current += 1
    setEditorTemplateContent(resolved.templateContent)
    setEditorDraft(resolved.effectiveContent)
    setEditorOverrideActive(resolved.overrideActive)
    setEditorLoading(false)
  }, [promptAssist, selectedSystemPrompt, systemPrompt])

  const handleEditorSave = React.useCallback(() => {
    const nextValue = normalizeSystemPromptOverrideValue({
      draft: editorDraft,
      templateContent: editorTemplateContent
    })
    setSystemPrompt(nextValue)
    promptAssist.notifySendOrSave()
    editorOpenRef.current = false
    editorLookupEpochRef.current += 1
    setEditorLoading(false)
    setEditorOpen(false)
  }, [editorDraft, editorTemplateContent, promptAssist, setSystemPrompt])

  const handleEditorReset = React.useCallback(async () => {
    const lookupEpoch = ++editorLookupEpochRef.current
    const lifecycleKey = editorLifecycleKeyRef.current
    setEditorLoading(true)
    const nextValue = await resolveSelectedSystemPromptContent(
      selectedSystemPrompt
    )
    if (
      !editorMountedRef.current ||
      !editorOpenRef.current ||
      editorLookupEpochRef.current !== lookupEpoch ||
      editorLifecycleKeyRef.current !== lifecycleKey
    ) {
      return
    }
    editorTemplateContentRef.current = nextValue
    editorDraftRef.current = nextValue
    editorRevisionRef.current += 1
    setEditorTemplateContent(nextValue)
    setEditorDraft(nextValue)
    setEditorOverrideActive(false)
    setSystemPrompt(nextValue)
    promptAssist.notifySendOrSave()
    setEditorLoading(false)
  }, [promptAssist, selectedSystemPrompt, setSystemPrompt])

  const closeEditor = React.useCallback(() => {
    promptAssist.dismiss()
    editorOpenRef.current = false
    editorLookupEpochRef.current += 1
    setEditorLoading(false)
    setEditorOpen(false)
  }, [promptAssist])

  const handleSelectModel = React.useCallback(() => {
    if (!onSelectModel) return
    promptAssist.dismiss()
    editorOpenRef.current = false
    editorLookupEpochRef.current += 1
    setEditorLoading(false)
    setEditorOpen(false)
    window.setTimeout(onSelectModel, 0)
  }, [onSelectModel, promptAssist])

  const enterPromptAssist = React.useCallback(
    (action: () => void) => {
      assistEntryDraftRef.current = editorDraftRef.current
      action()
    },
    []
  )

  const cancelPromptAssist = React.useCallback(() => {
    editorLookupEpochRef.current += 1
    setEditorLoading(false)
    editorDraftRef.current = assistEntryDraftRef.current
    editorRevisionRef.current += 1
    setEditorDraft(assistEntryDraftRef.current)
    promptAssist.dismiss()
  }, [promptAssist])

  // Group prompts by category: Favorites, System, Quick
  const groupedMenuItems = useMemo<ItemType[]>(() => {
    const hasCurrentSystemPrompt =
      typeof systemPrompt === "string" && systemPrompt.trim().length > 0
    const currentSystemPromptRecoveryItems: ItemType[] = hasCurrentSystemPrompt
      ? [
          {
            key: "__edit_current_system_prompt__",
            label: t(
              "promptSelect.editCurrentSystemPrompt",
              "Edit current system prompt"
            ),
            onClick: () => {
              void openSystemPromptEditor()
            }
          },
          {
            key: "__clear_current_system_prompt__",
            label: t(
              "promptSelect.clearCurrentSystemPrompt",
              "Clear current system prompt"
            ),
            onClick: () => {
              setSystemPrompt("")
              setDropdownOpen(false)
              restorePromptSelectFocus()
            }
          }
        ]
      : []

    if (promptsLoading) {
      return [
        {
          key: "__prompts_loading__",
          label: (
            <span role="status" aria-label={promptLoadingLabel}>
              {promptLoadingLabel}
            </span>
          )
        }
      ]
    }

    if (promptsError) {
      return [
        {
          key: "__prompts_error__",
          label: promptUnavailableLabel
        },
        {
          key: "__prompts_retry__",
          label: promptRetryLabel,
          onClick: () => {
            void refetchPrompts()
          }
        },
        ...(currentSystemPromptRecoveryItems.length > 0
          ? [
              {
                key: "__current_system_prompt_divider__",
                type: "divider" as const
              },
              ...currentSystemPromptRecoveryItems
            ]
          : [])
      ]
    }

    if (filteredData.length === 0) {
      return [
        {
          key: "empty",
          label: (
            <Empty
              description={
                searchText
                  ? t("noMatchingPrompts", "No matching prompts")
                  : t("promptSelect.noSavedPrompts", "No saved prompts")
              }
            />
          )
        },
        ...(currentSystemPromptRecoveryItems.length > 0
          ? [
              {
                key: "__current_system_prompt_divider__",
                type: "divider" as const
              },
              ...currentSystemPromptRecoveryItems
            ]
          : [])
      ]
    }

    const favorites = filteredData.filter((prompt) => prompt.favorite)
    const systemPrompts = filteredData.filter(
      (prompt) => !prompt.favorite && prompt.is_system
    )
    const quickPrompts = filteredData.filter(
      (prompt) => !prompt.favorite && !prompt.is_system
    )

    const createPromptItem = (prompt: Prompt): MenuItemType => ({
      key: prompt.id,
      label: (
        <div className="w-56 py-0.5">
          <div className="flex items-center gap-2">
            {prompt.is_system ? (
              <ComputerIcon className="w-4 h-4 flex-shrink-0" />
            ) : (
              <ZapIcon className="w-4 h-4 flex-shrink-0" />
            )}
            {prompt?.favorite && (
              <span className="text-warn flex-shrink-0" title="Favorite">★</span>
            )}
            <span className="truncate font-medium">{prompt.title}</span>
          </div>
          {prompt.content && (
            <p className="text-xs text-text-subtle line-clamp-1 mt-0.5 ml-6">
              {prompt.content}
            </p>
          )}
        </div>
      ),
      onClick: () => {
        if (selectedSystemPrompt === prompt.id) {
          setSelectedSystemPrompt(undefined)
        } else {
          handlePromptChange(prompt.id)
        }
        setDropdownOpen(false)
        restorePromptSelectFocus()
      }
    })

    const items: ItemType[] = []

    if (favorites.length > 0) {
      items.push({
        type: 'group',
        label: t("promptSelect.favorites", "Favorites"),
        children: favorites.map(createPromptItem)
      })
    }

    if (systemPrompts.length > 0) {
      items.push({
        type: 'group',
        label: t("promptSelect.system", "System prompts"),
        children: systemPrompts.map(createPromptItem)
      })
    }

    if (quickPrompts.length > 0) {
      items.push({
        type: 'group',
        label: t("promptSelect.quick", "Quick prompts"),
        children: quickPrompts.map(createPromptItem)
      })
    }

    if (items.length > 0) {
      items.push({
        key: "__prompt_actions_divider__",
        type: "divider"
      })
    }

    items.push({
      key: "__edit_system_prompt__",
      label: t("promptSelect.editSystemPrompt", "Edit system prompt"),
      onClick: () => {
        void openSystemPromptEditor()
      }
    })

    if (currentSystemPromptRecoveryItems.length > 0) {
      items.push(...currentSystemPromptRecoveryItems)
    }

    // If no groups (shouldn't happen, but fallback)
    if (items.length === 0) {
      return filteredData.map(createPromptItem)
    }

    return items
  }, [
    filteredData,
    searchText,
    selectedSystemPrompt,
    systemPrompt,
    promptsLoading,
    promptsError,
    promptLoadingLabel,
    promptUnavailableLabel,
    promptRetryLabel,
    refetchPrompts,
    t,
    handlePromptChange,
    openSystemPromptEditor,
    restorePromptSelectFocus,
    setDropdownOpen,
    setSelectedSystemPrompt,
    setSystemPrompt
  ])

  // Focus search input when dropdown opens
  useEffect(() => {
    if (typeof window === "undefined") return

    const handleOpenPromptSelect = (event: Event) => {
      const detail = (event as CustomEvent<PromptSelectOpenDetail>).detail
      returnFocusSelectorRef.current = normalizeFocusSelector(
        detail?.returnFocusSelector
      )
      setDropdownOpen(true)
    }

    window.addEventListener(OPEN_PROMPT_SELECT_EVENT, handleOpenPromptSelect)
    return () => {
      window.removeEventListener(OPEN_PROMPT_SELECT_EVENT, handleOpenPromptSelect)
    }
  }, [])

  useEffect(() => {
    if (!dropdownOpen) {
      setSearchText("") // Clear search when closed
      return
    }

    let frameId: number | null = null
    let attempts = 0
    let canceled = false
    const focusWhenReady = () => {
      if (canceled) return
      if (searchInputRef.current) {
        searchInputRef.current.focus()
        return
      }
      if (attempts < 10) {
        attempts += 1
        frameId = window.requestAnimationFrame(focusWhenReady)
      }
    }

    frameId = window.requestAnimationFrame(focusWhenReady)

    return () => {
      canceled = true
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId)
      }
    }
  }, [dropdownOpen])

  useEffect(() => {
    if (!dropdownOpen) return

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return
      setDropdownOpen(false)
      restorePromptSelectFocus()
    }

    window.addEventListener("keydown", handleEscape)
    window.addEventListener("keyup", handleEscape)
    return () => {
      window.removeEventListener("keydown", handleEscape)
      window.removeEventListener("keyup", handleEscape)
    }
  }, [dropdownOpen, restorePromptSelectFocus])

  const triggerLabel = promptsLoading
    ? promptLoadingLabel
    : promptsError
      ? promptUnavailableLabel
      : selectedPromptLabel || t("promptSelect.label", "Prompt")
  const triggerAriaLabel = promptsLoading
    ? promptLoadingLabel
    : promptsError
      ? promptUnavailableLabel
      : (t("selectAPrompt") as string)

  return (
    <>
      <Dropdown
        open={dropdownOpen}
        onOpenChange={(nextOpen) => {
          setDropdownOpen(nextOpen)
          if (!nextOpen) {
            restorePromptSelectFocus()
          }
        }}
        menu={{
          items: groupedMenuItems,
          style: {
            maxHeight: 400,
            overflowY: "auto"
          },
          className: `no-scrollbar ${menuDensity === 'compact' ? 'menu-density-compact' : 'menu-density-comfortable'}`,
          activeKey: selectedSystemPrompt
        }}
        popupRender={(menu) => (
          <div
            className="bg-surface rounded-lg shadow-lg border border-border"
            onKeyDown={(e) => {
              if (e.key !== "Escape") return
              setDropdownOpen(false)
              restorePromptSelectFocus()
              e.stopPropagation()
            }}
          >
            <div
              className="p-2 border-b border-border"
              onKeyDownCapture={(e) => {
                if (e.key !== "Escape") return
                e.preventDefault()
                setDropdownOpen(false)
                restorePromptSelectFocus()
                e.stopPropagation()
              }}
            >
              <Input
                ref={searchInputRef}
                placeholder={t("searchPrompts", "Search prompts...")}
                prefix={<Search className="size-4 text-text-subtle" />}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
                size="small"
                onKeyDown={(e) => {
                  e.stopPropagation()
                }}
              />
            </div>
            {menu}
          </div>
        )}
        placement={"topLeft"}
        trigger={["click"]}>
        <Tooltip title={triggerAriaLabel}>
          <IconButton
            ariaLabel={triggerAriaLabel}
            hasPopup="menu"
            dataTestId="chat-prompt-select"
            className={className}>
            <BookIcon className={iconClassName} />
            <span className="ml-1 hidden max-w-[120px] truncate text-xs font-medium text-text sm:inline">
              {triggerLabel}
            </span>
          </IconButton>
        </Tooltip>
      </Dropdown>
      <Modal
        open={editorOpen}
        title={t("promptSelect.editSystemPrompt", "Edit system prompt")}
        onCancel={closeEditor}
        footer={
          promptAssist.state.status === "idle" ||
          promptAssist.state.status === "applied" ? (
            <div className="flex items-center justify-end gap-2">
              <button type="button" onClick={closeEditor}>
                {t("common:cancel", "Cancel")}
              </button>
              <button
                type="button"
                onClick={() => {
                  void handleEditorReset()
                }}
              >
                {t("common:reset", "Reset")}
              </button>
              {promptAssist.state.status === "idle" ? (
                <PromptAssistMenu
                  draft={editorDraft}
                  capability={promptAssistCapability}
                  modelSelection={modelSelection}
                  onImproveNow={() =>
                    enterPromptAssist(promptAssist.improveNow)
                  }
                  onReviewChanges={() =>
                    enterPromptAssist(promptAssist.reviewChanges)
                  }
                  onSelectModel={onSelectModel ? handleSelectModel : undefined}
                />
              ) : null}
              <button type="button" onClick={handleEditorSave}>
                {t("common:save", "Save")}
              </button>
            </div>
          ) : null
        }>
        <div className="space-y-3">
          {editorOverrideActive ? (
            <div className="text-xs text-text-subtle">
              {t(
                "promptSelect.overrideActive",
                "Override active: this conversation is currently using a custom system prompt instead of the selected template."
              )}
            </div>
          ) : null}
          <Input.TextArea
            ref={editorInputRef}
            rows={6}
            placeholder={t(
              "promptSelect.systemPromptPlaceholder",
              "Enter system prompt"
            )}
            value={editorDraft}
            onChange={(event) => {
              updateEditorDraft(event.target.value)
              promptAssist.notifyTargetEdited()
            }}
          />
          {editorLoading ? (
            <div className="text-xs text-text-subtle">
              {t("common:loading.title", getDesignSystemState("loading")?.label)}
            </div>
          ) : null}
        </div>
        {promptAssist.state.status !== "idle" ||
        promptAssist.state.notice === "no_change" ? (
          <PromptAssistPanel
            state={promptAssist.state}
            onCancel={cancelPromptAssist}
            onRetry={promptAssist.retry}
            onSelectModel={onSelectModel ? handleSelectModel : undefined}
            onCandidateChange={promptAssist.editCandidate}
            onApply={promptAssist.applyCandidate}
            onConfirmReplace={promptAssist.confirmReplaceCurrent}
            onUndo={promptAssist.undo}
            onRequestReturnFocus={requestEditorFocus}
          />
        ) : null}
      </Modal>
    </>
  )
}
