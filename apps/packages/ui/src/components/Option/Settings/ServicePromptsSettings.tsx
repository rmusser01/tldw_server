import React from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Button, Form, Input, Skeleton } from "antd"
import { useTranslation } from "react-i18next"
import { useSearchParams } from "react-router-dom"

import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { Alert, Badge } from "@/components/ui/primitives"
import { RecoveryCallout } from "@/components/ui/state"
import {
  clearLegacyServicePromptCandidate,
  importLegacyServicePromptCandidate,
  readLegacyServicePromptCandidates,
  renderServicePromptPart,
  resolveServicePromptScope,
  validateServicePromptParts,
  type LegacyServicePromptCandidate,
  type ServicePromptScope
} from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  ServicePromptApiError,
  type KnownServicePromptId,
  type ServicePromptCatalogItem,
  type ServicePromptDetail
} from "@/services/tldw/domains/service-prompts"

type Draft = {
  scopeKey: string
  definitionId: string
  parts: Record<string, string>
  revision: string | null
}

type MigrationItem = LegacyServicePromptCandidate & {
  value: string
  error?: string
}

const KNOWN_DEFINITIONS = {
  "chat.rag.answer": {
    key: "chatRagAnswer",
    label: "RAG answer",
    description:
      "Controls how retrieved context and the current question are presented to the model."
  },
  "chat.rag.question_rewrite": {
    key: "chatRagQuestionRewrite",
    label: "RAG follow-up rewrite",
    description:
      "Controls how a conversational follow-up is rewritten into a standalone retrieval query."
  },
  "chat.web_search.answer": {
    key: "chatWebSearchAnswer",
    label: "Web-search answer",
    description:
      "Controls how normalized web-search results are presented for the final answer."
  },
  "media.text.translation": {
    key: "mediaTextTranslation",
    label: "Text translation",
    description:
      "Controls the visible instructions used by synchronous text translation."
  }
} as const

const KNOWN_WORKFLOWS: Record<string, { key: string; label: string }> = {
  "chat.main.rag": { key: "mainChatRag", label: "Main chat RAG" },
  "chat.tab.rag": { key: "tabChatRag", label: "Tab chat RAG" },
  "chat.document.rag": {
    key: "documentChatRag",
    label: "Document chat RAG"
  },
  "chat.sidepanel.rag": { key: "sidepanelRag", label: "Sidepanel RAG" },
  "chat.main.web_search": {
    key: "mainChatWebSearch",
    label: "Main chat web search"
  },
  "chat.compare.web_search": {
    key: "compareWebSearch",
    label: "Compare web search"
  },
  "media.text.translation": {
    key: "textTranslation",
    label: "Text translation"
  }
}

const KNOWN_PARTS: Record<string, { key: string; label: string }> = {
  template: { key: "template", label: "Template" },
  system: { key: "system", label: "System instructions" },
  user_template: { key: "userTemplate", label: "User template" }
}

const isKnownDefinitionId = (id: string): id is KnownServicePromptId =>
  Object.prototype.hasOwnProperty.call(KNOWN_DEFINITIONS, id)

const isAbortError = (error: unknown): boolean =>
  (error as { name?: unknown } | null)?.name === "AbortError"

const getDefinitionText = (
  definition: ServicePromptCatalogItem,
  field: "label" | "description",
  t: ReturnType<typeof useTranslation>["t"]
): string => {
  if (!isKnownDefinitionId(definition.id)) return definition[field]
  const known = KNOWN_DEFINITIONS[definition.id]
  return t(`servicePrompts.definitions.${known.key}.${field}`, {
    defaultValue: known[field]
  })
}

const getWorkflowLabel = (
  id: string,
  fallback: string,
  t: ReturnType<typeof useTranslation>["t"]
): string => {
  const known = KNOWN_WORKFLOWS[id]
  return known
    ? t(`servicePrompts.workflows.${known.key}`, { defaultValue: known.label })
    : fallback
}

const getPartLabel = (
  key: string,
  fallback: string,
  t: ReturnType<typeof useTranslation>["t"]
): string => {
  const known = KNOWN_PARTS[key]
  return known
    ? t(`servicePrompts.parts.${known.key}`, { defaultValue: known.label })
    : fallback
}

const getPromptSearchId = (searchParams: URLSearchParams): string | null => {
  const value = searchParams.get("prompt")?.trim()
  return value || null
}

export const ServicePromptsSettings = () => {
  const { t } = useTranslation("settings")
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedId = getPromptSearchId(searchParams)

  const [scope, setScope] = React.useState<ServicePromptScope | null>(null)
  const [scopeError, setScopeError] = React.useState<unknown>(null)
  const [scopeLoading, setScopeLoading] = React.useState(true)
  const [scopeGeneration, setScopeGeneration] = React.useState(0)
  const [scopeChanged, setScopeChanged] = React.useState(false)
  const [draft, setDraft] = React.useState<Draft | null>(null)
  const [dirty, setDirty] = React.useState(false)
  const [fieldErrors, setFieldErrors] = React.useState<Record<string, string>>({})
  const [preview, setPreview] = React.useState<Record<string, string> | null>(null)
  const [conflict, setConflict] = React.useState(false)
  const [operationError, setOperationError] = React.useState<string | null>(null)
  const [isSaving, setIsSaving] = React.useState(false)
  const [isResetting, setIsResetting] = React.useState(false)
  const [migrationItems, setMigrationItems] = React.useState<MigrationItem[]>([])
  const [migrationLoading, setMigrationLoading] = React.useState(false)
  const [migrationMessage, setMigrationMessage] = React.useState<string | null>(null)

  const scopeRef = React.useRef<ServicePromptScope | null>(null)
  const dirtyRef = React.useRef(false)
  const editedUrlRef = React.useRef(window.location.href)
  const historyIndexRef = React.useRef(0)
  const suppressPopstateRef = React.useRef(false)
  const mutationControllerRef = React.useRef<AbortController | null>(null)
  const migrationProbedScopeRef = React.useRef<string | null>(null)

  React.useEffect(() => {
    scopeRef.current = scope
  }, [scope])

  React.useEffect(() => {
    dirtyRef.current = dirty
    if (dirty) editedUrlRef.current = window.location.href
  }, [dirty])

  React.useEffect(() => {
    const existing = window.history.state ?? {}
    const existingIndex = existing.servicePromptHistoryIndex
    if (typeof existingIndex === "number") {
      historyIndexRef.current = existingIndex
      return
    }
    window.history.replaceState(
      { ...existing, servicePromptHistoryIndex: historyIndexRef.current },
      "",
      window.location.href
    )
  }, [])

  React.useEffect(() => {
    const controller = new AbortController()
    setScopeLoading(true)
    setScopeError(null)
    void resolveServicePromptScope({ signal: controller.signal })
      .then((resolved) => {
        if (controller.signal.aborted) return
        setScope(resolved)
        setScopeLoading(false)
      })
      .catch((error) => {
        if (controller.signal.aborted || isAbortError(error)) return
        setScopeError(error)
        setScopeLoading(false)
      })
    return () => controller.abort()
  }, [scopeGeneration])

  React.useEffect(() => {
    const handleScopeChange = () => {
      const oldScope = scopeRef.current
      mutationControllerRef.current?.abort()
      if (oldScope) {
        const queryKey = ["service-prompts", oldScope.scopeKey]
        void queryClient.cancelQueries({ queryKey })
        void queryClient.invalidateQueries({ queryKey, refetchType: "none" })
      }
      migrationProbedScopeRef.current = null
      setMigrationItems([])
      setMigrationMessage(null)
      setScope(null)
      setScopeError(null)
      setScopeLoading(true)
      setScopeChanged(true)
      setDirty(false)
      setFieldErrors({})
      setPreview(null)
      setConflict(false)
      setOperationError(null)
      const next = new URLSearchParams(searchParams)
      next.delete("prompt")
      setSearchParams(next, { replace: true })
      setScopeGeneration((value) => value + 1)
    }

    window.addEventListener("tldw:config-updated", handleScopeChange)
    window.addEventListener("tldw:auth-credentials-changed", handleScopeChange)
    return () => {
      window.removeEventListener("tldw:config-updated", handleScopeChange)
      window.removeEventListener("tldw:auth-credentials-changed", handleScopeChange)
    }
  }, [queryClient, searchParams, setSearchParams])

  const catalogKey = [
    "service-prompts",
    scope?.scopeKey ?? "unresolved",
    "catalog"
  ] as const
  const catalogQuery = useQuery({
    queryKey: catalogKey,
    enabled: Boolean(scope),
    queryFn: ({ signal }) => tldwClient.listServicePrompts({ signal })
  })

  React.useEffect(() => {
    if (!scope || !catalogQuery.data || migrationProbedScopeRef.current === scope.scopeKey) {
      return
    }
    migrationProbedScopeRef.current = scope.scopeKey
    const controller = new AbortController()
    void readLegacyServicePromptCandidates({ signal: controller.signal })
      .then((candidates) => {
        if (controller.signal.aborted || scopeRef.current?.scopeKey !== scope.scopeKey) {
          return
        }
        setMigrationItems(candidates.map((candidate) => ({
          ...candidate,
          value: candidate.value
        })))
      })
      .catch((error) => {
        if (!isAbortError(error)) {
          setMigrationMessage(t("servicePrompts.migration.readFailed", {
            defaultValue: "Unable to read browser-local workflow prompts."
          }))
        }
      })
    return () => controller.abort()
  }, [catalogQuery.data, scope, t])

  const selectedDefinition = React.useMemo(
    () => catalogQuery.data?.find((item) => item.id === selectedId) ?? null,
    [catalogQuery.data, selectedId]
  )
  const detailKey = [
    "service-prompts",
    scope?.scopeKey ?? "unresolved",
    "detail",
    selectedId ?? "unselected"
  ] as const
  const detailQuery = useQuery({
    queryKey: detailKey,
    enabled: Boolean(scope && selectedDefinition),
    queryFn: ({ signal }) =>
      tldwClient.getServicePrompt(selectedDefinition!.id, { signal })
  })

  React.useEffect(() => {
    const detail = detailQuery.data
    if (!detail || !scope || dirty) return
    setDraft({
      scopeKey: scope.scopeKey,
      definitionId: detail.id,
      parts: { ...detail.effective_parts },
      revision: detail.revision
    })
    setFieldErrors({})
    setPreview(null)
    setConflict(false)
    setOperationError(null)
  }, [detailQuery.data, dirty, scope])

  React.useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirtyRef.current) return
      event.preventDefault()
      event.returnValue = ""
    }
    const anchorClick = (event: MouseEvent) => {
      if (!dirtyRef.current || event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return
      }
      const element = event.target instanceof Element
        ? event.target.closest("a")
        : null
      if (!(element instanceof HTMLAnchorElement) || element.target ||
        element.hasAttribute("download")) {
        return
      }
      const destination = new URL(element.href, window.location.href)
      if (destination.origin !== window.location.origin) return
      const leave = window.confirm(t("servicePrompts.unsaved.leave", {
        defaultValue: "Discard unsaved workflow prompt changes?"
      }))
      if (!leave) {
        event.preventDefault()
        return
      }
      dirtyRef.current = false
      setDirty(false)
    }
    const popstate = (event: PopStateEvent) => {
      if (suppressPopstateRef.current) {
        suppressPopstateRef.current = false
        return
      }
      if (!dirtyRef.current) {
        const index = event.state?.servicePromptHistoryIndex
        if (typeof index === "number") historyIndexRef.current = index
        return
      }
      const leave = window.confirm(t("servicePrompts.unsaved.leave", {
        defaultValue: "Discard unsaved workflow prompt changes?"
      }))
      if (leave) {
        dirtyRef.current = false
        setDirty(false)
        const index = event.state?.servicePromptHistoryIndex
        if (typeof index === "number") historyIndexRef.current = index
        return
      }
      const targetIndex = event.state?.servicePromptHistoryIndex
      if (typeof targetIndex === "number") {
        const delta = historyIndexRef.current - targetIndex
        suppressPopstateRef.current = true
        window.history.go(delta || 1)
      } else {
        window.history.pushState(
          { servicePromptHistoryIndex: historyIndexRef.current },
          "",
          editedUrlRef.current
        )
      }
    }

    window.addEventListener("beforeunload", beforeUnload)
    window.addEventListener("popstate", popstate)
    document.addEventListener("click", anchorClick, true)
    return () => {
      window.removeEventListener("beforeunload", beforeUnload)
      window.removeEventListener("popstate", popstate)
      document.removeEventListener("click", anchorClick, true)
    }
  }, [t])

  React.useEffect(() => () => mutationControllerRef.current?.abort(), [])

  const definitionLabel = React.useCallback(
    (definition: ServicePromptCatalogItem) =>
      getDefinitionText(definition, "label", t),
    [t]
  )

  const selectDefinition = (id: string) => {
    if (id === selectedId) return
    if (dirty && !window.confirm(t("servicePrompts.unsaved.changePrompt", {
      defaultValue: "Discard unsaved changes and open another workflow prompt?"
    }))) {
      return
    }
    setDirty(false)
    setScopeChanged(false)
    setFieldErrors({})
    setPreview(null)
    setConflict(false)
    setOperationError(null)
    const next = new URLSearchParams(searchParams)
    next.set("prompt", id)
    historyIndexRef.current += 1
    setSearchParams(next)
    queueMicrotask(() => {
      window.history.replaceState(
        {
          ...(window.history.state ?? {}),
          servicePromptHistoryIndex: historyIndexRef.current
        },
        "",
        window.location.href
      )
    })
  }

  const updatePart = (key: string, value: string) => {
    setDraft((current) => current ? {
      ...current,
      parts: { ...current.parts, [key]: value }
    } : current)
    setDirty(true)
    setFieldErrors((current) => {
      const next = { ...current }
      delete next[key]
      return next
    })
    setPreview(null)
    setConflict(false)
    setOperationError(null)
  }

  const draftIsCurrent = Boolean(
    draft && scope && selectedDefinition &&
    draft.scopeKey === scope.scopeKey &&
    draft.definitionId === selectedDefinition.id
  )

  const previewDraft = () => {
    if (!selectedDefinition || !draftIsCurrent || !draft) return
    const errors = validateServicePromptParts(selectedDefinition, draft.parts)
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) return
    const rendered: Record<string, string> = {}
    for (const part of selectedDefinition.parts) {
      const values = Object.fromEntries(
        part.required_variables.map((name) => [name, `[${name}]`])
      )
      rendered[part.key] = renderServicePromptPart(
        selectedDefinition,
        part.key,
        draft.parts[part.key],
        values
      )
    }
    setPreview(rendered)
  }

  const saveDraft = async () => {
    if (!selectedDefinition || !scope || !draftIsCurrent || !draft) return
    const errors = validateServicePromptParts(selectedDefinition, draft.parts)
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) return
    const operationScope = scope.scopeKey
    const controller = new AbortController()
    mutationControllerRef.current?.abort()
    mutationControllerRef.current = controller
    setIsSaving(true)
    setConflict(false)
    setOperationError(null)
    try {
      const saved = await tldwClient.saveServicePrompt(
        selectedDefinition.id,
        {
          parts: { ...draft.parts },
          expected_revision: draft.revision
        },
        { signal: controller.signal }
      )
      if (controller.signal.aborted || scopeRef.current?.scopeKey !== operationScope) {
        return
      }
      queryClient.setQueryData(detailKey, saved)
      setDraft({
        scopeKey: operationScope,
        definitionId: saved.id,
        parts: { ...saved.effective_parts },
        revision: saved.revision
      })
      setDirty(false)
      setPreview(null)
      await queryClient.invalidateQueries({ queryKey: detailKey, refetchType: "none" })
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) return
      if (error instanceof ServicePromptApiError && error.status === 422) {
        setFieldErrors(error.fieldErrors ?? {})
      } else if (error instanceof ServicePromptApiError && error.status === 409) {
        setConflict(true)
      } else {
        setOperationError(t("servicePrompts.errors.saveFailed", {
          defaultValue: "Unable to save this workflow prompt."
        }))
      }
    } finally {
      if (mutationControllerRef.current === controller) {
        mutationControllerRef.current = null
        setIsSaving(false)
      }
    }
  }

  const reloadServerValue = async () => {
    setFieldErrors({})
    const result = await detailQuery.refetch()
    if (result.isError || !result.data || !scope) {
      setDirty(true)
      setConflict(true)
      setOperationError(t("servicePrompts.errors.reloadFailed", {
        defaultValue: "Unable to reload the server value."
      }))
      return
    }
    setDraft({
      scopeKey: scope.scopeKey,
      definitionId: result.data.id,
      parts: { ...result.data.effective_parts },
      revision: result.data.revision
    })
    setDirty(false)
    setConflict(false)
    setOperationError(null)
  }

  const resetPrompt = async (revision: string | null, corrupt = false) => {
    if (!selectedDefinition || !scope || (!corrupt && !draftIsCurrent)) return
    const label = definitionLabel(selectedDefinition)
    const confirmed = await confirmDanger({
      title: t("servicePrompts.reset.title", {
        defaultValue: "Reset {{name}}?",
        name: label
      }),
      content: t("servicePrompts.reset.content", {
        defaultValue:
          "This will permanently remove the saved customization. V1 has no history or undo.",
        name: label
      }),
      okText: t("servicePrompts.actions.resetConfirm", { defaultValue: "Reset" })
    })
    if (!confirmed || scopeRef.current?.scopeKey !== scope.scopeKey) return
    const controller = new AbortController()
    mutationControllerRef.current?.abort()
    mutationControllerRef.current = controller
    setIsResetting(true)
    setOperationError(null)
    try {
      const reset = await tldwClient.resetServicePrompt(
        selectedDefinition.id,
        revision,
        { signal: controller.signal }
      )
      if (controller.signal.aborted || scopeRef.current?.scopeKey !== scope.scopeKey) {
        return
      }
      queryClient.setQueryData(detailKey, reset)
      setDraft({
        scopeKey: scope.scopeKey,
        definitionId: reset.id,
        parts: { ...reset.effective_parts },
        revision: null
      })
      setDirty(false)
      setFieldErrors({})
      setPreview(null)
      await queryClient.invalidateQueries({ queryKey: detailKey, refetchType: "none" })
    } catch (error) {
      if (!controller.signal.aborted && !isAbortError(error)) {
        if (error instanceof ServicePromptApiError && error.status === 409) {
          setConflict(true)
        } else {
          setOperationError(t("servicePrompts.errors.resetFailed", {
            defaultValue: "Unable to reset this workflow prompt."
          }))
        }
      }
    } finally {
      if (mutationControllerRef.current === controller) {
        mutationControllerRef.current = null
        setIsResetting(false)
      }
    }
  }

  const remainingMigrationMessage = (count: number) =>
    t("servicePrompts.migration.remaining", {
      defaultValue: "{{count}} browser-local prompt still needs attention.",
      count
    })

  const importMigration = async () => {
    if (!scope || migrationItems.length === 0 || migrationLoading) return
    const operationScope = scope.scopeKey
    const controller = new AbortController()
    mutationControllerRef.current?.abort()
    mutationControllerRef.current = controller
    setMigrationLoading(true)
    setMigrationMessage(null)
    const details = new Map<string, ServicePromptDetail>()
    const nextItems = migrationItems.map((item) => ({ ...item, error: undefined }))
    try {
      for (const item of nextItems) {
        const definition = catalogQuery.data?.find(
          (candidate) => candidate.id === item.definitionId
        )
        if (!definition) continue
        const errors = validateServicePromptParts(definition, {
          template: item.value
        })
        if (errors.template) item.error = errors.template
      }
      if (nextItems.some((item) => item.error)) {
        setMigrationItems(nextItems)
        return
      }

      for (const item of nextItems) {
        const detail = await tldwClient.getServicePrompt(item.definitionId, {
          signal: controller.signal
        })
        if (scopeRef.current?.scopeKey !== operationScope) return
        details.set(item.definitionId, detail)
      }

      const replacements = nextItems.filter((item) => {
        const detail = details.get(item.definitionId)
        return detail?.source === "user" &&
          detail.effective_parts[item.partKey] !== item.value
      })
      if (replacements.length > 0) {
        const names = replacements.map((item) => {
          const definition = catalogQuery.data?.find(
            (candidate) => candidate.id === item.definitionId
          )
          return definition ? definitionLabel(definition) : item.definitionId
        }).join(", ")
        const confirmed = await confirmDanger({
          title: t("servicePrompts.migration.replaceTitle", {
            defaultValue: "Replace saved customizations?"
          }),
          content: t("servicePrompts.migration.replaceContent", {
            defaultValue: "Importing will replace saved customizations for: {{names}}.",
            names
          }),
          okText: t("servicePrompts.migration.replaceAction", {
            defaultValue: "Replace and import"
          })
        })
        if (!confirmed) return
      }

      let remaining = [...nextItems]
      for (const item of nextItems) {
        try {
          await importLegacyServicePromptCandidate(
            item,
            details.get(item.definitionId)!,
            { signal: controller.signal }
          )
          if (scopeRef.current?.scopeKey !== operationScope) return
          remaining = remaining.filter(
            (candidate) => candidate.definitionId !== item.definitionId
          )
          setMigrationItems(remaining)
          await queryClient.invalidateQueries({
            queryKey: ["service-prompts", operationScope]
          })
        } catch (error) {
          if (controller.signal.aborted || isAbortError(error)) return
          remaining = remaining.map((candidate) =>
            candidate.definitionId === item.definitionId
              ? {
                  ...candidate,
                  error: t("servicePrompts.migration.importFailed", {
                    defaultValue: "Import failed. The browser-local value was preserved."
                  })
                }
              : candidate
          )
          setMigrationItems(remaining)
        }
      }
      if (remaining.length > 0) {
        setMigrationMessage(remainingMigrationMessage(remaining.length))
      }
    } finally {
      if (mutationControllerRef.current === controller) {
        mutationControllerRef.current = null
        setMigrationLoading(false)
      }
    }
  }

  const discardMigration = async () => {
    if (!scope || migrationItems.length === 0 || migrationLoading) return
    const confirmed = await confirmDanger({
      title: t("servicePrompts.migration.discardTitle", {
        defaultValue: "Discard browser-local workflow prompts?"
      }),
      content: t("servicePrompts.migration.discardContent", {
        defaultValue:
          "This permanently removes only the three mapped browser-local values."
      }),
      okText: t("servicePrompts.migration.discardAction", {
        defaultValue: "Discard"
      })
    })
    if (!confirmed || scopeRef.current?.scopeKey !== scope.scopeKey) return
    const operationScope = scope.scopeKey
    const controller = new AbortController()
    mutationControllerRef.current?.abort()
    mutationControllerRef.current = controller
    setMigrationLoading(true)
    setMigrationMessage(null)
    let remaining = [...migrationItems]
    for (const item of migrationItems) {
      if (controller.signal.aborted || scopeRef.current?.scopeKey !== operationScope) break
      try {
        await clearLegacyServicePromptCandidate(item.definitionId)
        remaining = remaining.filter(
          (candidate) => candidate.definitionId !== item.definitionId
        )
        setMigrationItems(remaining)
      } catch {
        remaining = remaining.map((candidate) =>
          candidate.definitionId === item.definitionId
            ? {
                ...candidate,
                error: t("servicePrompts.migration.discardFailed", {
                  defaultValue: "Discard failed. The browser-local value was preserved."
                })
              }
            : candidate
        )
        setMigrationItems(remaining)
      }
    }
    if (remaining.length > 0) {
      setMigrationMessage(remainingMigrationMessage(remaining.length))
    }
    if (mutationControllerRef.current === controller) {
      mutationControllerRef.current = null
      setMigrationLoading(false)
    }
  }

  const retryScope = () => setScopeGeneration((value) => value + 1)
  const catalogError = catalogQuery.error
  const unsupported = catalogError instanceof ServicePromptApiError &&
    catalogError.status === 404
  const corruptError = detailQuery.error instanceof ServicePromptApiError &&
    detailQuery.error.code === "service_prompt_corrupt_override" &&
    detailQuery.error.canReset === true &&
    typeof detailQuery.error.revision === "string"

  return (
    <div className="flex min-w-0 flex-col gap-5">
      <header className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-[70ch]">
          <h1 className="text-xl font-semibold text-text">
            {t("servicePrompts.title", { defaultValue: "Workflow prompts" })}
          </h1>
          <p className="mt-1 text-sm text-text-muted">
            {t("servicePrompts.description", {
              defaultValue:
                "Review and customize the instructions used by supported content workflows."
            })}
          </p>
        </div>
        <a
          href="/prompts"
          className="w-fit text-sm font-medium text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          {t("servicePrompts.actions.openLibrary", {
            defaultValue: "Open reusable Prompts workspace"
          })}
        </a>
      </header>

      {scopeChanged ? (
        <Alert variant="warning" title={t("servicePrompts.scope.changedTitle", {
          defaultValue: "Server or account changed"
        })}>
          {t("servicePrompts.scope.changedDescription", {
            defaultValue:
              "Choose a workflow prompt again after the new server and account scope loads. The previous draft cannot be saved."
          })}
        </Alert>
      ) : null}

      {scopeLoading ? (
        <div aria-label={t("servicePrompts.states.loadingScope", {
          defaultValue: "Loading server and account scope…"
        })}>
          <p className="mb-2 text-sm text-text-muted">
            {t("servicePrompts.states.loadingScope", {
              defaultValue: "Loading server and account scope…"
            })}
          </p>
          <Skeleton active paragraph={{ rows: 5 }} />
        </div>
      ) : scopeError ? (
        <RecoveryCallout
          state="unavailable"
          role="alert"
          title={t("servicePrompts.errors.scopeTitle", {
            defaultValue: "Unable to resolve the connected server and account."
          })}
          message={t("servicePrompts.errors.scopeDescription", {
            defaultValue: "Check the connection and credentials, then try again."
          })}
          primaryAction={{
            label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
            onClick: retryScope
          }}
        />
      ) : catalogQuery.isPending ? (
        <div aria-label={t("servicePrompts.states.loadingCatalog", {
          defaultValue: "Loading workflow prompts…"
        })}>
          <Skeleton active paragraph={{ rows: 6 }} />
        </div>
      ) : unsupported ? (
        <RecoveryCallout
          state="unavailable"
          title={t("servicePrompts.unsupported.title", {
            defaultValue: "Workflow prompts require a server update"
          })}
          message={t("servicePrompts.unsupported.description", {
            defaultValue:
              "This older server does not support server-synced editing. Existing browser-local prompt behavior remains active until the server is updated."
          })}
        />
      ) : catalogQuery.isError ? (
        <RecoveryCallout
          state="error"
          role="alert"
          title={t("servicePrompts.errors.catalogTitle", {
            defaultValue: "Unable to load workflow prompts"
          })}
          message={t("servicePrompts.errors.catalogDescription", {
            defaultValue: "The connected server returned an error. No local values were changed."
          })}
          primaryAction={{
            label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
            onClick: () => void catalogQuery.refetch()
          }}
        />
      ) : (
        <>
          {scope && migrationItems.length > 0 ? (
            <RecoveryCallout
              state="blocked"
              title={t("servicePrompts.migration.title", {
                defaultValue: "Browser-local workflow prompts found"
              })}
              message={t("servicePrompts.migration.description", {
                defaultValue:
                  "Review these values before Chat uses this server. Imported overrides belong to the connected server and account and are not included in Backup supported account data."
              })}
            >
              <dl className="grid gap-1 text-sm sm:grid-cols-[8rem_minmax(0,1fr)]">
                <dt className="font-medium text-text-muted">
                  {t("servicePrompts.scope.server", { defaultValue: "Server" })}
                </dt>
                <dd className="break-all text-text">{scope.config.serverUrl}</dd>
                <dt className="font-medium text-text-muted">
                  {t("servicePrompts.scope.account", { defaultValue: "Account scope" })}
                </dt>
                <dd><code className="break-all text-xs text-text">{scope.scopeKey}</code></dd>
              </dl>
              <div className="mt-3 divide-y divide-border rounded-md border border-border">
                {migrationItems.map((item) => {
                  const definition = catalogQuery.data?.find(
                    (candidate) => candidate.id === item.definitionId
                  )
                  const label = definition
                    ? definitionLabel(definition)
                    : item.definitionId
                  return (
                    <div key={item.definitionId} className="p-3">
                      <label
                        htmlFor={`migration-${item.definitionId}`}
                        className="text-sm font-semibold text-text"
                      >
                        {label}
                      </label>
                      <Input.TextArea
                        id={`migration-${item.definitionId}`}
                        aria-label={t("servicePrompts.migration.repairLabel", {
                          defaultValue: "Repair {{name}}",
                          name: label
                        })}
                        className="mt-2 font-mono"
                        autoSize={{ minRows: 3, maxRows: 10 }}
                        value={item.value}
                        onChange={(event) => setMigrationItems((items) =>
                          items.map((candidate) =>
                            candidate.definitionId === item.definitionId
                              ? { ...candidate, value: event.target.value, error: undefined }
                              : candidate
                          )
                        )}
                      />
                      {item.error ? (
                        <p className="mt-1 text-sm text-danger" role="alert">
                          {item.error}
                        </p>
                      ) : null}
                    </div>
                  )
                })}
              </div>
              {migrationMessage ? (
                <p className="mt-3 text-sm text-warn" role="status">{migrationMessage}</p>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  type="primary"
                  loading={migrationLoading}
                  onClick={() => void importMigration()}
                >
                  {t("servicePrompts.migration.importAction", {
                    defaultValue: "Import to this server"
                  })}
                </Button>
                <Button
                  danger
                  disabled={migrationLoading}
                  onClick={() => void discardMigration()}
                >
                  {t("servicePrompts.migration.discardButton", {
                    defaultValue: "Discard local values"
                  })}
                </Button>
              </div>
            </RecoveryCallout>
          ) : null}

          <div className="grid min-w-0 grid-cols-1 gap-5 lg:grid-cols-[minmax(14rem,18rem)_minmax(0,1fr)]">
            <nav aria-label={t("servicePrompts.listLabel", {
              defaultValue: "Workflow prompt definitions"
            })}>
              <ul className="divide-y divide-border rounded-lg border border-border bg-surface">
                {(catalogQuery.data ?? []).map((definition) => {
                  const label = definitionLabel(definition)
                  const selected = definition.id === selectedId
                  return (
                    <li key={definition.id} data-testid="service-prompt-list-item">
                      <button
                        type="button"
                        aria-label={label}
                        aria-current={selected ? "page" : undefined}
                        onClick={() => selectDefinition(definition.id)}
                        className={`w-full px-3 py-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus ${
                          selected
                            ? "bg-primary/10 font-semibold text-primary"
                            : "text-text hover:bg-surface2"
                        }`}
                      >
                        <span className="block text-sm">{label}</span>
                        <span className="mt-0.5 block line-clamp-2 text-xs font-normal text-text-muted">
                          {getDefinitionText(definition, "description", t)}
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </nav>

            {!selectedDefinition ? (
              <section className="min-w-0 rounded-lg border border-border bg-surface p-5">
                <h2 className="text-base font-semibold text-text">
                  {t("servicePrompts.states.chooseTitle", {
                    defaultValue: "Choose a workflow prompt"
                  })}
                </h2>
                <p className="mt-1 text-sm text-text-muted">
                  {t("servicePrompts.states.chooseDescription", {
                    defaultValue: "Select a definition to review its affected workflows and instructions."
                  })}
                </p>
              </section>
            ) : detailQuery.isPending ? (
              <section className="min-w-0" aria-label={t("servicePrompts.states.loadingDetail", {
                defaultValue: "Loading workflow prompt…"
              })}>
                <Skeleton active paragraph={{ rows: 7 }} />
              </section>
            ) : corruptError ? (
              <RecoveryCallout
                state="error"
                title={t("servicePrompts.corrupt.title", {
                  defaultValue: "Saved customization is unavailable"
                })}
                message={t("servicePrompts.corrupt.description", {
                  defaultValue:
                    "The saved value cannot be read safely. Reset it conditionally to restore the server default."
                })}
                primaryAction={{
                  label: t("servicePrompts.corrupt.resetAction", {
                    defaultValue: "Reset corrupt customization"
                  }),
                  onClick: () => void resetPrompt(detailQuery.error.revision!, true),
                  loading: isResetting
                }}
              />
            ) : detailQuery.isError && !(draft && draftIsCurrent && dirty) ? (
              <RecoveryCallout
                state="error"
                title={t("servicePrompts.errors.detailTitle", {
                  defaultValue: "Unable to load this workflow prompt"
                })}
                primaryAction={{
                  label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
                  onClick: () => void detailQuery.refetch()
                }}
              />
            ) : draft && draftIsCurrent && detailQuery.data ? (
              <section
                className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5"
                role="region"
                aria-label={t("servicePrompts.editorLabel", {
                  defaultValue: "{{name}} editor",
                  name: definitionLabel(selectedDefinition)
                })}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h2 className="text-lg font-semibold text-text">
                      {definitionLabel(selectedDefinition)}
                    </h2>
                    <p className="mt-1 max-w-[70ch] text-sm text-text-muted">
                      {getDefinitionText(selectedDefinition, "description", t)}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge
                      variant={detailQuery.data.source === "user" ? "primary" : "secondary"}
                    >
                      {detailQuery.data.source === "user"
                        ? t("servicePrompts.states.customized", { defaultValue: "Customized" })
                        : t("servicePrompts.states.packaged", { defaultValue: "Server default" })}
                    </Badge>
                    {dirty ? (
                      <Badge variant="warning">
                        {t("servicePrompts.states.dirty", { defaultValue: "Unsaved" })}
                      </Badge>
                    ) : null}
                  </div>
                </div>

                {scope ? (
                  <dl className="mt-4 grid gap-x-4 gap-y-1 border-y border-border py-3 text-sm sm:grid-cols-[8rem_minmax(0,1fr)]">
                    <dt className="font-medium text-text-muted">
                      {t("servicePrompts.scope.server", { defaultValue: "Server" })}
                    </dt>
                    <dd className="break-all text-text">{scope.config.serverUrl}</dd>
                    <dt className="font-medium text-text-muted">
                      {t("servicePrompts.scope.account", { defaultValue: "Account scope" })}
                    </dt>
                    <dd><code className="break-all text-xs text-text">{scope.scopeKey}</code></dd>
                    <dt className="font-medium text-text-muted">
                      {t("servicePrompts.scope.workflows", { defaultValue: "Affected workflows" })}
                    </dt>
                    <dd>
                      <ul className="flex flex-wrap gap-x-3 gap-y-1">
                        {selectedDefinition.affected_workflows.map((workflow) => (
                          <li key={workflow.id}>
                            {getWorkflowLabel(workflow.id, workflow.label, t)}
                          </li>
                        ))}
                      </ul>
                    </dd>
                  </dl>
                ) : null}

                {conflict ? (
                  <Alert
                    className="mt-4"
                    variant="warning"
                    title={t("servicePrompts.conflict.title", {
                      defaultValue: "This prompt changed on the server."
                    })}
                    action={{
                      label: t("servicePrompts.conflict.reload", {
                        defaultValue: "Reload server value"
                      }),
                      onClick: () => void reloadServerValue()
                    }}
                  >
                    {t("servicePrompts.conflict.description", {
                      defaultValue:
                        "Your complete local draft is preserved. Reload only when you are ready to replace it."
                    })}
                  </Alert>
                ) : null}
                {operationError ? (
                  <Alert className="mt-4" variant="error" title={operationError} />
                ) : null}

                <Form className="mt-5" layout="vertical" onFinish={() => void saveDraft()}>
                  <div className="flex flex-col gap-5">
                    {selectedDefinition.parts.map((part) => {
                      const partLabel = getPartLabel(part.key, part.label, t)
                      return (
                        <section key={part.key} className="min-w-0">
                          <Form.Item
                            label={partLabel}
                            validateStatus={fieldErrors[part.key] ? "error" : undefined}
                            help={fieldErrors[part.key]}
                          >
                            <Input.TextArea
                              aria-label={partLabel}
                              className="font-mono"
                              autoSize={{ minRows: 6, maxRows: 18 }}
                              value={draft.parts[part.key]}
                              onChange={(event) => updatePart(part.key, event.target.value)}
                            />
                          </Form.Item>
                          {part.mode === "template" ? (
                            <div className="flex flex-wrap items-center gap-2" aria-label={t(
                              "servicePrompts.variables.label",
                              { defaultValue: "Required variables" }
                            )}>
                              <span className="text-xs font-medium text-text-muted">
                                {t("servicePrompts.variables.label", {
                                  defaultValue: "Required variables"
                                })}
                              </span>
                              {part.required_variables.map((variable) => (
                                <Badge key={variable} size="sm" outline>
                                  {variable}
                                </Badge>
                              ))}
                            </div>
                          ) : null}
                        </section>
                      )
                    })}
                  </div>

                  {preview ? (
                    <section
                      className="mt-5 min-w-0 rounded-lg border border-border bg-surface2 p-3"
                      aria-label={t("servicePrompts.preview.label", {
                        defaultValue: "Prompt preview"
                      })}
                    >
                      <h3 className="text-sm font-semibold text-text">
                        {t("servicePrompts.preview.title", { defaultValue: "Local preview" })}
                      </h3>
                      <div className="mt-3 flex flex-col gap-3">
                        {selectedDefinition.parts.map((part) => (
                          <div key={part.key} className="min-w-0">
                            <p className="text-xs font-medium text-text-muted">
                              {getPartLabel(part.key, part.label, t)}
                            </p>
                            <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface p-3 text-sm text-text">
                              <code role="code">{preview[part.key]}</code>
                            </pre>
                          </div>
                        ))}
                      </div>
                    </section>
                  ) : null}

                  <div className="mt-5 flex flex-wrap items-center gap-2">
                    <Button onClick={previewDraft}>
                      {t("servicePrompts.actions.preview", { defaultValue: "Preview" })}
                    </Button>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={isSaving}
                      disabled={!dirty || !draftIsCurrent}
                    >
                      {t("servicePrompts.actions.save", { defaultValue: "Save changes" })}
                    </Button>
                    <Button
                      danger
                      loading={isResetting}
                      disabled={detailQuery.data.source !== "user" || !draftIsCurrent}
                      onClick={() => void resetPrompt(draft.revision)}
                    >
                      {t("servicePrompts.actions.reset", {
                        defaultValue: "Reset to default"
                      })}
                    </Button>
                  </div>
                </Form>
              </section>
            ) : null}
          </div>
        </>
      )}
    </div>
  )
}

export default ServicePromptsSettings
