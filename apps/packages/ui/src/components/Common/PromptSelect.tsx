import { useQuery } from "@tanstack/react-query"
import { Dropdown, Empty, Input, Modal, Tooltip } from "antd"
import type { InputRef } from "antd"
import type { ItemType, MenuItemType } from "antd/es/menu/interface"
import { BookIcon, ComputerIcon, ZapIcon, Search } from "lucide-react"
import React, { useState, useMemo, useRef, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { getAllPrompts } from "@/db/dexie/helpers"
import type { Prompt } from "@/db/dexie/types"
import { getDesignSystemState } from "@/design-system"
import { useStorage } from "@plasmohq/storage/hook"
import { IconButton } from "./IconButton"
import {
  normalizeSystemPromptOverrideValue,
  resolveEffectiveSystemPromptState,
  resolveSelectedSystemPromptContent
} from "./system-prompt-utils"

type Props = {
  setSelectedSystemPrompt: (promptId: string | undefined) => void
  setSelectedQuickPrompt: (prompt: string | undefined) => void
  selectedSystemPrompt: string | undefined
  systemPrompt: string | undefined
  setSystemPrompt: (prompt: string) => void
  className?: string
  iconClassName?: string
}

export const PromptSelect: React.FC<Props> = ({
  setSelectedQuickPrompt,
  setSelectedSystemPrompt,
  selectedSystemPrompt,
  systemPrompt,
  setSystemPrompt,
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

  const { data } = useQuery({
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

  const openSystemPromptEditor = React.useCallback(async () => {
    setDropdownOpen(false)
    setEditorOpen(true)
    setEditorLoading(true)
    const resolved = await resolveEffectiveSystemPromptState({
      selectedSystemPrompt,
      systemPrompt
    })
    setEditorTemplateContent(resolved.templateContent)
    setEditorDraft(resolved.effectiveContent)
    setEditorOverrideActive(
      Boolean(selectedSystemPrompt) &&
        typeof systemPrompt === "string" &&
        systemPrompt.trim().length > 0 &&
        systemPrompt !== resolved.templateContent
    )
    setEditorLoading(false)
  }, [selectedSystemPrompt, systemPrompt])

  const handleEditorSave = React.useCallback(() => {
    const nextValue = normalizeSystemPromptOverrideValue({
      draft: editorDraft,
      templateContent: editorTemplateContent
    })
    setSystemPrompt(nextValue)
    setEditorOpen(false)
  }, [editorDraft, editorTemplateContent, setSystemPrompt])

  const handleEditorReset = React.useCallback(async () => {
    setEditorLoading(true)
    const nextValue = await resolveSelectedSystemPromptContent(
      selectedSystemPrompt
    )
    setEditorTemplateContent(nextValue)
    setEditorDraft(nextValue)
    setEditorOverrideActive(false)
    setSystemPrompt(nextValue)
    setEditorLoading(false)
  }, [selectedSystemPrompt, setSystemPrompt])

  // Group prompts by category: Favorites, System, Quick
  const groupedMenuItems = useMemo<ItemType[]>(() => {
    if (filteredData.length === 0) {
      return [
        {
        key: "empty",
        label: <Empty description={searchText ? t("noMatchingPrompts", "No matching prompts") : undefined} />
      }
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

    // If no groups (shouldn't happen, but fallback)
    if (items.length === 0) {
      return filteredData.map(createPromptItem)
    }

    return items
  }, [
    filteredData,
    searchText,
    selectedSystemPrompt,
    t,
    handlePromptChange,
    openSystemPromptEditor,
    setDropdownOpen,
    setSelectedSystemPrompt
  ])

  // Focus search input when dropdown opens
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

  return (
    <>
      {data && (
        <>
          <Dropdown
            open={dropdownOpen}
            onOpenChange={setDropdownOpen}
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
              <div className="bg-surface rounded-lg shadow-lg border border-border">
                <div className="p-2 border-b border-border">
                  <Input
                    ref={searchInputRef}
                    placeholder={t("searchPrompts", "Search prompts...")}
                    prefix={<Search className="size-4 text-text-subtle" />}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    allowClear
                    size="small"
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                </div>
                {menu}
              </div>
            )}
            placement={"topLeft"}
            trigger={["click"]}>
            <Tooltip title={t("selectAPrompt")}>
              <IconButton
                ariaLabel={t("selectAPrompt") as string}
                hasPopup="menu"
                dataTestId="chat-prompt-select"
                className={className}>
                <BookIcon className={iconClassName} />
                <span className="ml-1 hidden max-w-[120px] truncate text-xs font-medium text-text sm:inline">
                  {selectedPromptLabel ||
                    t("promptSelect.label", "Prompt")}
                </span>
              </IconButton>
            </Tooltip>
          </Dropdown>
          <Modal
            open={editorOpen}
            title={t("promptSelect.editSystemPrompt", "Edit system prompt")}
            onCancel={() => setEditorOpen(false)}
            footer={
              <div className="flex items-center justify-end gap-2">
                <button type="button" onClick={() => setEditorOpen(false)}>
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
                <button type="button" onClick={handleEditorSave}>
                  {t("common:save", "Save")}
                </button>
              </div>
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
                rows={6}
                placeholder={t(
                  "promptSelect.systemPromptPlaceholder",
                  "Enter system prompt"
                )}
                value={editorDraft}
                onChange={(event) => setEditorDraft(event.target.value)}
              />
              {editorLoading ? (
                <div className="text-xs text-text-subtle">
                  {t("common:loading", getDesignSystemState("loading")?.label)}
                </div>
              ) : null}
            </div>
          </Modal>
        </>
      )}
    </>
  )
}
