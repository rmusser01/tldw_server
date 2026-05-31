import React from "react"
import { useQuery } from "@tanstack/react-query"
import { Empty, Input, List, Modal, Select, Tag } from "antd"
import { useTranslation } from "react-i18next"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export type PromptInsertItem = {
  id: string
  title: string
  systemPrompt?: string
  userPrompt?: string
  tags: string[]
}

type Props = {
  open: boolean
  onClose: () => void
  onInsertPrompt: (prompt: PromptInsertItem) => void
}

const normalizeTags = (raw: unknown) => {
  if (Array.isArray(raw)) {
    return raw
      .map((tag) => String(tag).trim())
      .filter((tag) => tag.length > 0)
  }
  if (typeof raw === "string") {
    return raw
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0)
  }
  return []
}

const normalizePromptText = (value: unknown) =>
  typeof value === "string" ? value : ""

const normalizeServerPrompt = (prompt: any): PromptInsertItem | null => {
  if (!prompt) return null
  const title = String(prompt.name || prompt.title || "Untitled")
  const systemPrompt = normalizePromptText(prompt.system_prompt)
  const userPrompt = normalizePromptText(prompt.user_prompt)
  const fallbackPrompt = normalizePromptText(
    prompt.content || prompt.prompt || ""
  )
  const isSystemFlag =
    typeof prompt.is_system === "boolean" ? prompt.is_system : undefined
  const resolvedSystem =
    systemPrompt ||
    (!userPrompt && isSystemFlag ? fallbackPrompt : "")
  const resolvedUser =
    userPrompt ||
    (!systemPrompt && (isSystemFlag === false || !isSystemFlag)
      ? fallbackPrompt
      : "")
  const tags = normalizeTags(prompt.keywords ?? prompt.tags)
  const idSource = prompt.id ?? prompt.uuid ?? prompt.slug ?? title
  return {
    id: String(idSource),
    title,
    systemPrompt: resolvedSystem || undefined,
    userPrompt: resolvedUser || undefined,
    tags
  }
}

export const PromptInsertModal: React.FC<Props> = ({
  open,
  onClose,
  onInsertPrompt
}) => {
  const { t } = useTranslation(["option", "settings"])
  const [query, setQuery] = React.useState("")
  const [tagFilter, setTagFilter] = React.useState<string[]>([])

  const {
    data: prompts = [],
    isLoading,
    isError,
    error,
    refetch
  } = useQuery({
    queryKey: ["promptInsertPrompts"],
    queryFn: async () => {
      await tldwClient.initialize()
      const response = await tldwClient.getPrompts()
      const list = Array.isArray(response)
        ? response
        : response?.results || response?.prompts || []
      return Array.isArray(list) ? list : []
    },
    enabled: open
  })

  React.useEffect(() => {
    if (!open) {
      setQuery("")
      setTagFilter([])
    }
  }, [open])

  const normalizedPrompts = React.useMemo<PromptInsertItem[]>(() => {
    return (prompts || [])
      .map((prompt) => normalizeServerPrompt(prompt))
      .filter((prompt): prompt is PromptInsertItem => Boolean(prompt))
      .filter((prompt) => {
        const hasSystem = Boolean(prompt.systemPrompt?.trim())
        const hasUser = Boolean(prompt.userPrompt?.trim())
        return hasSystem || hasUser
      })
  }, [prompts])

  const tagOptions = React.useMemo(() => {
    const set = new Set<string>()
    for (const prompt of normalizedPrompts) {
      for (const tag of prompt.tags) {
        set.add(tag)
      }
    }
    return Array.from(set)
      .sort((a, b) => a.localeCompare(b))
      .map((tag) => ({ label: tag, value: tag }))
  }, [normalizedPrompts])

  const filteredPrompts = React.useMemo(() => {
    const trimmedQuery = query.trim().toLowerCase()
    return normalizedPrompts.filter((prompt) => {
      const matchesQuery =
        !trimmedQuery ||
        prompt.title.toLowerCase().includes(trimmedQuery) ||
        (prompt.systemPrompt || "")
          .toLowerCase()
          .includes(trimmedQuery) ||
        (prompt.userPrompt || "").toLowerCase().includes(trimmedQuery)
      const matchesTags =
        tagFilter.length === 0 ||
        prompt.tags.some((tag) => tagFilter.includes(tag))
      return matchesQuery && matchesTags
    })
  }, [normalizedPrompts, query, tagFilter])

  const systemLabel = t("settings:managePrompts.systemPrompt", {
    defaultValue: "System prompt"
  })
  const quickLabel = t("settings:managePrompts.quickPrompt", {
    defaultValue: "Quick prompt"
  })
  const noMatches = t("option:noMatchingPrompts", "No matching prompts")
  const errorTitle = t("option:error", "Error")
  const retryLabel = t("option:retry", "Retry")
  const errorFallback = t("option:somethingWentWrong", "Something went wrong")
  const errorDescription =
    error instanceof Error && error.message ? error.message : errorFallback
  const showLoadError = isError && prompts.length === 0

  return (
    <Modal
      title={t("option:promptInsert.useInChat", "Insert prompt")}
      open={open}
      onCancel={onClose}
      footer={null}
      destroyOnHidden
      centered
    >
      <div className="space-y-3">
        <Input
          value={query}
          allowClear
          autoFocus
          placeholder={t("settings:managePrompts.search", {
            defaultValue: "Search prompts..."
          })}
          onChange={(event) => setQuery(event.target.value)}
        />
        <Select
          mode="multiple"
          allowClear
          value={tagFilter}
          placeholder={t("settings:managePrompts.tags.placeholder", {
            defaultValue: "Filter keywords"
          })}
          options={tagOptions}
          onChange={(value) => setTagFilter(value)}
          disabled={tagOptions.length === 0}
        />
        <div className="max-h-80 overflow-auto rounded-md border border-border">
          {showLoadError ? (
            <div className="p-3">
              <DesignSystemAlert
                variant="error"
                title={errorTitle}
                action={{
                  label: retryLabel,
                  onClick: () => {
                    void refetch()
                  }
                }}
              >
                {errorDescription}
              </DesignSystemAlert>
            </div>
          ) : filteredPrompts.length === 0 ? (
            <div className="py-8">
              <Empty description={isLoading ? undefined : noMatches} />
            </div>
          ) : (
            <List
              dataSource={filteredPrompts}
              renderItem={(item) => {
                const hasSystem = Boolean(item.systemPrompt?.trim())
                const hasUser = Boolean(item.userPrompt?.trim())
                const preview = item.userPrompt || item.systemPrompt || ""
                return (
                  <List.Item className="!p-0">
                    <button
                      type="button"
                      onClick={() => onInsertPrompt(item)}
                      className="w-full px-3 py-2 text-left transition hover:bg-surface2"
                    >
                      <div className="flex items-center gap-2">
                        <span className="truncate font-medium">{item.title}</span>
                        {hasSystem && (
                          <Tag className="m-0" color="geekblue">
                            {systemLabel}
                          </Tag>
                        )}
                        {hasUser && (
                          <Tag className="m-0" color="default">
                            {quickLabel}
                          </Tag>
                        )}
                      </div>
                      <div className="text-xs text-text-subtle line-clamp-2">
                        {preview}
                      </div>
                      {item.tags.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {item.tags.slice(0, 6).map((tag) => (
                            <Tag key={`${item.id}-${tag}`} className="m-0">
                              {tag}
                            </Tag>
                          ))}
                        </div>
                      )}
                    </button>
                  </List.Item>
                )
              }}
            />
          )}
        </div>
      </div>
    </Modal>
  )
}

export default PromptInsertModal
