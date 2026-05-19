import React from "react"
import { Alert, Button, Empty, Input, Spin, Tag, Typography } from "antd"
import { Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import type {
  FlashcardTemplate,
  FlashcardTemplateCreate
} from "@/services/flashcards"
import {
  useCreateFlashcardTemplateMutation,
  useDeleteFlashcardTemplateMutation,
  useFlashcardTemplatesQuery,
  useUpdateFlashcardTemplateMutation
} from "../hooks"
import { FlashcardTemplateForm } from "../components"
import { getModelTypeLabel } from "../utils/model-type-labels"

const { Title, Text } = Typography

const filterTemplates = (templates: FlashcardTemplate[], query: string): FlashcardTemplate[] => {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return templates
  }

  return templates.filter((template) => {
    if (template.name.toLowerCase().includes(normalizedQuery)) {
      return true
    }
    return template.placeholder_definitions.some((definition) =>
      definition.label.toLowerCase().includes(normalizedQuery) ||
      definition.key.toLowerCase().includes(normalizedQuery)
    )
  })
}

export const TemplatesTab: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const templatesQuery = useFlashcardTemplatesQuery()
  const createMutation = useCreateFlashcardTemplateMutation()
  const updateMutation = useUpdateFlashcardTemplateMutation()
  const deleteMutation = useDeleteFlashcardTemplateMutation()
  const templates = templatesQuery.data?.items ?? []
  const isMutating =
    createMutation.isPending || updateMutation.isPending || deleteMutation.isPending
  const [searchValue, setSearchValue] = React.useState("")
  const [selectedTemplateId, setSelectedTemplateId] = React.useState<number | null>(() => templates[0]?.id ?? null)
  const [isCreating, setIsCreating] = React.useState(false)
  const [isTemplateDirty, setIsTemplateDirty] = React.useState(false)
  const [optimisticTemplate, setOptimisticTemplate] = React.useState<FlashcardTemplate | null>(null)

  const visibleTemplates = React.useMemo(() => {
    if (!optimisticTemplate) {
      return templates
    }
    return [optimisticTemplate, ...templates.filter((template) => template.id !== optimisticTemplate.id)]
  }, [optimisticTemplate, templates])

  const filteredTemplates = React.useMemo(
    () => filterTemplates(visibleTemplates, searchValue),
    [searchValue, visibleTemplates]
  )

  const activeTemplate = React.useMemo(
    () => visibleTemplates.find((template) => template.id === selectedTemplateId) ?? null,
    [selectedTemplateId, visibleTemplates]
  )

  React.useEffect(() => {
    if (!optimisticTemplate) {
      return
    }

    const syncedTemplate = templates.find((template) => template.id === optimisticTemplate.id)
    if (syncedTemplate && syncedTemplate.version >= optimisticTemplate.version) {
      setOptimisticTemplate(null)
    }
  }, [optimisticTemplate, templates])

  React.useEffect(() => {
    if (isCreating || isTemplateDirty) {
      return
    }
    if (filteredTemplates.length === 0) {
      setSelectedTemplateId(null)
      return
    }
    if (
      selectedTemplateId == null ||
      !filteredTemplates.some((template) => template.id === selectedTemplateId)
    ) {
      setSelectedTemplateId(filteredTemplates[0].id)
    }
  }, [filteredTemplates, isCreating, isTemplateDirty, selectedTemplateId])

  const confirmDiscardTemplateChanges = React.useCallback(() => {
    if (!isTemplateDirty) {
      return true
    }
    const shouldDiscard = window.confirm(
      t("option:flashcards.templatesDiscardChangesConfirm", {
        defaultValue: "Discard unsaved template changes?"
      })
    )
    if (shouldDiscard) {
      setIsTemplateDirty(false)
    }
    return shouldDiscard
  }, [isTemplateDirty, t])

  const handleStartCreate = React.useCallback(() => {
    if (isMutating) {
      return
    }
    if (!confirmDiscardTemplateChanges()) {
      return
    }
    setIsCreating(true)
    setSelectedTemplateId(null)
  }, [confirmDiscardTemplateChanges, isMutating])

  const handleSelectTemplate = React.useCallback((templateId: number) => {
    if (isMutating) {
      return
    }
    if (!isCreating && templateId === selectedTemplateId) {
      return
    }
    if (!confirmDiscardTemplateChanges()) {
      return
    }
    setSelectedTemplateId(templateId)
    setIsCreating(false)
  }, [confirmDiscardTemplateChanges, isCreating, isMutating, selectedTemplateId])

  const handleCancelCreate = React.useCallback(() => {
    if (isMutating) {
      return
    }
    if (!confirmDiscardTemplateChanges()) {
      return
    }
    setIsCreating(false)
    setIsTemplateDirty(false)
    setSelectedTemplateId(filteredTemplates[0]?.id ?? templates[0]?.id ?? null)
  }, [confirmDiscardTemplateChanges, filteredTemplates, isMutating, templates])

  const handleCreate = React.useCallback(
    async (values: FlashcardTemplateCreate) => {
      try {
        const template = await createMutation.mutateAsync(values)
        setOptimisticTemplate(template)
        setSearchValue("")
        message.success(
          t("common:created", {
            defaultValue: "Created"
          })
        )
        setIsTemplateDirty(false)
        setSelectedTemplateId(template.id)
        setIsCreating(false)
      } catch (error: unknown) {
        message.error(error instanceof Error ? error.message : "Failed to create template")
      }
    },
    [createMutation, message, t]
  )

  const handleUpdate = React.useCallback(
    async (values: FlashcardTemplateCreate) => {
      if (!activeTemplate) {
        return
      }

      try {
        const template = await updateMutation.mutateAsync({
          templateId: activeTemplate.id,
          update: {
            ...values,
            expected_version: activeTemplate.version
          }
        })
        setOptimisticTemplate(template)
        if (filterTemplates([template], searchValue).length === 0) {
          setSearchValue("")
        }
        message.success(
          t("common:saved", {
            defaultValue: "Saved"
          })
        )
        setIsTemplateDirty(false)
        setSelectedTemplateId(template.id)
        setIsCreating(false)
      } catch (error: unknown) {
        message.error(error instanceof Error ? error.message : "Failed to update template")
      }
    },
    [activeTemplate, message, searchValue, t, updateMutation]
  )

  const handleDelete = React.useCallback(async () => {
    if (!activeTemplate) {
      return
    }

    const shouldDelete = window.confirm(
      t("option:flashcards.templatesDeleteConfirm", {
        defaultValue: "Delete this template?"
      })
    )
    if (!shouldDelete) {
      return
    }

    try {
      await deleteMutation.mutateAsync({
        templateId: activeTemplate.id,
        expectedVersion: activeTemplate.version
      })
      if (optimisticTemplate?.id === activeTemplate.id) {
        setOptimisticTemplate(null)
      }
      message.success(
        t("common:deleted", {
          defaultValue: "Deleted"
        })
      )
      const remainingVisibleTemplates = visibleTemplates.filter((template) => template.id !== activeTemplate.id)
      const remainingFilteredTemplates = filterTemplates(remainingVisibleTemplates, searchValue)
      if (remainingFilteredTemplates.length > 0) {
        setSelectedTemplateId(remainingFilteredTemplates[0].id)
      } else if (remainingVisibleTemplates.length > 0) {
        setSearchValue("")
        setSelectedTemplateId(remainingVisibleTemplates[0].id)
      } else {
        setSelectedTemplateId(null)
      }
      setIsTemplateDirty(false)
      setIsCreating(false)
    } catch (error: unknown) {
      message.error(error instanceof Error ? error.message : "Failed to delete template")
    }
  }, [activeTemplate, deleteMutation, message, optimisticTemplate?.id, searchValue, t, visibleTemplates])

  if (templatesQuery.isLoading) {
    return (
      <div className="flex min-h-[320px] items-center justify-center">
        <Spin size="large" />
      </div>
    )
  }

  if (templatesQuery.error) {
    return (
      <Alert
        type="error"
        message={t("option:flashcards.templatesLoadError", {
          defaultValue: "Could not load templates."
        })}
        description={
          templatesQuery.error instanceof Error
            ? templatesQuery.error.message
            : t("option:flashcards.templatesLoadErrorFallback", {
                defaultValue: "Try refreshing the page."
              })
        }
      />
    )
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[320px,minmax(0,1fr)]">
      <div className="space-y-4 rounded border border-border bg-surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <Title level={4} className="!mb-1">
              {t("option:flashcards.templatesLibraryTitle", {
                defaultValue: "Templates"
              })}
            </Title>
            <Text type="secondary">
              {t("option:flashcards.templatesLibraryCount", {
                defaultValue: "{{count}} saved templates",
                count: templates.length
              })}
            </Text>
          </div>
          <Button
            type="primary"
            icon={<Plus className="size-4" />}
            onClick={handleStartCreate}
            disabled={isMutating}
          >
            {t("option:flashcards.templatesCreateCta", {
              defaultValue: "Create template"
            })}
          </Button>
        </div>

        <Input
          value={searchValue}
          onChange={(event) => setSearchValue(event.target.value)}
          disabled={isMutating}
          placeholder={t("option:flashcards.templatesSearchPlaceholder", {
            defaultValue: "Search templates"
          })}
        />

        {filteredTemplates.length > 0 ? (
          <div className="space-y-2">
            {filteredTemplates.map((template) => {
              const isSelected = !isCreating && template.id === selectedTemplateId

              return (
                <button
                  key={template.id}
                  type="button"
                  disabled={isMutating}
                  className={`w-full rounded border p-3 text-left transition ${
                    isSelected
                      ? "border-primary bg-surface2"
                      : "border-border bg-background hover:border-primary/50"
                  }`}
                  onClick={() => handleSelectTemplate(template.id)}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <Text strong className="block">
                        {template.name}
                      </Text>
                      <Text type="secondary" className="block text-xs">
                        {t("option:flashcards.templatesPlaceholderCount", {
                          defaultValue: "{{count}} placeholders",
                          count: template.placeholder_definitions.length
                        })}
                      </Text>
                    </div>
                    <Tag>{getModelTypeLabel(template.model_type, t)}</Tag>
                  </div>
                </button>
              )
            })}
          </div>
        ) : templates.length > 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={t("option:flashcards.templatesNoMatches", {
              defaultValue: "No templates match this search."
            })}
          />
        ) : null}
      </div>

      <div className="min-h-[320px]">
        {isCreating ? (
          <FlashcardTemplateForm
            mode="create"
            submitting={createMutation.isPending}
            onSubmit={handleCreate}
            onCancel={handleCancelCreate}
            cancelDisabled={isMutating}
            onDirtyChange={setIsTemplateDirty}
          />
        ) : activeTemplate ? (
          <FlashcardTemplateForm
            mode="edit"
            template={activeTemplate}
            submitting={updateMutation.isPending}
            onSubmit={handleUpdate}
            onDelete={handleDelete}
            onCancel={handleCancelCreate}
            cancelDisabled={isMutating}
            deleteDisabled={isMutating}
            onDirtyChange={setIsTemplateDirty}
          />
        ) : templates.length > 0 && filteredTemplates.length === 0 ? (
          <div className="flex h-full min-h-[320px] items-center justify-center rounded border border-dashed border-border bg-surface p-6">
            <Empty
              description={t("option:flashcards.templatesNoMatches", {
                defaultValue: "No templates match this search."
              })}
            >
              <Button onClick={() => setSearchValue("")}>
                {t("option:flashcards.templatesClearSearch", {
                  defaultValue: "Clear search"
                })}
              </Button>
            </Empty>
          </div>
        ) : (
          <div className="flex h-full min-h-[320px] items-center justify-center rounded border border-dashed border-border bg-surface p-6">
            <Empty
              description={t("option:flashcards.templatesEmpty", {
                defaultValue: "No templates yet"
              })}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default TemplatesTab
