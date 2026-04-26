import { useMutation } from "@tanstack/react-query"
import { Form } from "antd"
import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  isDictionaryVersionConflictError,
  normalizeDictionaryTags,
} from "../listUtils"
import {
  normalizeCreateDictionaryPayload,
  normalizeDictionaryFormPayload,
} from "./dictionaryFormPayloadUtils"
import { getDictionaryStarterTemplate } from "./dictionaryStarterTemplates"

type UseDictionaryFormManagementParams = {
  queryClient: {
    invalidateQueries: (input: { queryKey: readonly unknown[] }) => Promise<unknown>
  }
  notification: {
    error: (config: { message: string; description?: string }) => void
    info: (config: { message: string; description?: string }) => void
    success: (config: { message: string; description?: string }) => void
  }
  confirmDanger: (config: {
    title: string
    content: string
    okText: string
    cancelText: string
  }) => Promise<boolean>
  t: (key: string, fallbackOrOptions?: any) => string
  dictionariesById: Map<number, any>
  confirmDeactivationIfNeeded: (dictionary: any, nextIsActive: boolean) => Promise<boolean>
}

type UseDictionaryFormManagementResult = {
  openCreate: boolean
  openEdit: boolean
  editId: number | null
  createForm: any
  editForm: any
  creating: boolean
  updating: boolean
  openCreateModal: () => void
  closeCreateModal: () => void
  openEditModal: (record: any) => void
  closeEditModal: () => void
  handleCreateSubmit: (values: any) => Promise<void>
  handleEditSubmit: (values: any) => Promise<void>
}

export function useDictionaryFormManagement({
  queryClient,
  notification,
  confirmDanger,
  t,
  dictionariesById,
  confirmDeactivationIfNeeded,
}: UseDictionaryFormManagementParams): UseDictionaryFormManagementResult {
  const [openCreate, setOpenCreate] = React.useState(false)
  const [openEdit, setOpenEdit] = React.useState(false)
  const [editId, setEditId] = React.useState<number | null>(null)
  const [createForm] = Form.useForm()
  const [editForm] = Form.useForm()

  const { mutateAsync: createDict, isPending: creating } = useMutation({
    mutationFn: (values: any) => tldwClient.createDictionary(values),
  })

  const { mutateAsync: updateDict, isPending: updating } = useMutation({
    mutationFn: (values: any) =>
      editId != null ? tldwClient.updateDictionary(editId, values) : Promise.resolve(null),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["tldw:listDictionaries"] })
      notification.success({
        message: t("option:dictionaries.updatedTitle", "Dictionary updated")
      })
      setOpenEdit(false)
      editForm.resetFields()
      setEditId(null)
    },
  })

  const openCreateModal = React.useCallback(() => {
    setOpenCreate(true)
  }, [])

  const closeCreateModal = React.useCallback(() => {
    setOpenCreate(false)
  }, [])

  const openEditModal = React.useCallback(
    (record: any) => {
      setEditId(record.id)
      editForm.setFieldsValue({
        ...record,
        category:
          typeof record?.category === "string" ? record.category : undefined,
        tags: normalizeDictionaryTags(record?.tags),
      })
      setOpenEdit(true)
    },
    [editForm]
  )

  const closeEditModal = React.useCallback(() => {
    setOpenEdit(false)
    editForm.resetFields()
    setEditId(null)
  }, [editForm])

  const handleCreateSubmit = React.useCallback(
    async (values: any) => {
      const starterTemplate = getDictionaryStarterTemplate(values?.starter_template)
      try {
        const createdDictionary = await createDict(normalizeCreateDictionaryPayload(values))
        let starterEntriesApplied = 0
        const starterEntriesTotal = starterTemplate?.entries.length ?? 0

        if (starterTemplate) {
          const dictionaryId = Number(
            createdDictionary?.id ??
              createdDictionary?.dictionary_id ??
              createdDictionary?.dictionary?.id
          )

          if (!Number.isFinite(dictionaryId) || dictionaryId <= 0) {
            notification.error({
              message: t(
                "option:dictionaries.templateNotAppliedTitle",
                "Template not applied"
              ),
              description: t(
                "option:dictionaries.templateNotAppliedDescription",
                "Dictionary was created, but starter entries could not be added automatically."
              ),
            })
          } else {
            const starterEntryResults = await Promise.allSettled(
              starterTemplate.entries.map((entry) =>
                tldwClient.addDictionaryEntry(dictionaryId, entry)
              )
            )
            starterEntriesApplied = starterEntryResults.filter(
              (result) => result.status === "fulfilled"
            ).length

            if (starterEntriesApplied < starterEntriesTotal) {
              notification.error({
                message: t(
                  "option:dictionaries.templatePartiallyAppliedTitle",
                  "Template only partially applied"
                ),
                description:
                  starterEntriesApplied > 0
                    ? t("option:dictionaries.templatePartiallyAppliedDescription", {
                        defaultValue:
                          "Dictionary was created, but only {{applied}} of {{total}} starter entries were added automatically.",
                        applied: starterEntriesApplied,
                        total: starterEntriesTotal
                      })
                    : t(
                        "option:dictionaries.templateNotAppliedDescription",
                        "Dictionary was created, but starter entries could not be added automatically."
                      ),
              })
            }
          }
        }

        await queryClient.invalidateQueries({ queryKey: ["tldw:listDictionaries"] })
        notification.success({
          message: t("option:dictionaries.createdTitle", "Dictionary created"),
          description:
            starterEntriesApplied > 0 && starterEntriesApplied === starterEntriesTotal
              ? t("option:dictionaries.createdWithTemplateDescription", {
                  defaultValue:
                    '"{{name}}" created with {{count}} starter entr{{suffix}}.',
                  name: values?.name || t("option:dictionaries.dictionaryLabel", "Dictionary"),
                  count: starterEntriesApplied,
                  suffix: starterEntriesApplied === 1 ? "y" : "ies"
                })
              : starterEntriesApplied > 0
                ? t("option:dictionaries.createdWithPartialTemplateDescription", {
                    defaultValue:
                      '"{{name}}" created with {{applied}} of {{total}} starter entries.',
                    name: values?.name || t("option:dictionaries.dictionaryLabel", "Dictionary"),
                    applied: starterEntriesApplied,
                    total: starterEntriesTotal
                  })
                : t("option:dictionaries.createdDescription", {
                    defaultValue: '"{{name}}" created.',
                    name: values?.name || t("option:dictionaries.dictionaryLabel", "Dictionary")
                  }),
        })
        setOpenCreate(false)
        createForm.resetFields()
      } catch (error: any) {
        notification.error({
          message: t("common:error", { defaultValue: "Error" }),
          description:
            error?.message ||
            t(
              "option:dictionaries.createFailedDescription",
              "Failed to create dictionary"
            ),
        })
      }
    },
    [createDict, createForm, notification, queryClient, t]
  )

  const handleEditSubmit = React.useCallback(
    async (values: any) => {
      if (editId == null) {
        try {
          await updateDict(values)
        } catch (error: any) {
          notification.error({
            message: t("common:error", { defaultValue: "Error" }),
            description:
              error?.message ||
              t(
                "option:dictionaries.updateFailedDescription",
                "Failed to update dictionary"
              ),
          })
        }
        return
      }
      const dictionary = dictionariesById.get(editId)
      const nextIsActive =
        typeof values?.is_active === "boolean"
          ? values.is_active
          : Boolean(dictionary?.is_active)
      const confirmed = await confirmDeactivationIfNeeded(dictionary, nextIsActive)
      if (!confirmed) return

      const version = Number(dictionary?.version)
      const payload = {
        ...normalizeDictionaryFormPayload(values, {
          allowNullDefaultTokenBudget: true,
          allowNullCategory: true,
          includeEmptyTags: true,
        }),
        ...(Number.isFinite(version) && version > 0 ? { version } : {}),
      }

      try {
        await updateDict(payload)
      } catch (error: any) {
        if (!isDictionaryVersionConflictError(error)) {
          notification.error({
            message: t("common:error", { defaultValue: "Error" }),
            description:
              error?.message ||
              t(
                "option:dictionaries.updateFailedDescription",
                "Failed to update dictionary"
              ),
          })
          return
        }

        const shouldReload = await confirmDanger({
          title: t(
            "option:dictionaries.versionConflictTitle",
            "Dictionary changed in another session"
          ),
          content: t(
            "option:dictionaries.versionConflictDescription",
            "This dictionary was updated elsewhere. Reload the latest version while keeping your current edits?"
          ),
          okText: t(
            "option:dictionaries.reloadLatest",
            "Reload latest"
          ),
          cancelText: t("common:cancel", { defaultValue: "Cancel" }),
        })
        if (!shouldReload) return

        try {
          const latest = await tldwClient.getDictionary(editId)
          editForm.setFieldsValue({
            ...latest,
            ...values,
            version: latest?.version,
          })
          notification.info({
            message: t(
              "option:dictionaries.latestVersionLoadedTitle",
              "Latest version loaded"
            ),
            description: t(
              "option:dictionaries.latestVersionLoadedDescription",
              "Your edits were preserved. Save again to retry."
            ),
          })
          await queryClient.invalidateQueries({ queryKey: ["tldw:listDictionaries"] })
        } catch (reloadError: any) {
          notification.error({
            message: t("option:dictionaries.reloadFailedTitle", "Reload failed"),
            description:
              reloadError?.message ||
              t(
                "option:dictionaries.reloadFailedDescription",
                "Could not load the latest dictionary version. Please retry."
              ),
          })
        }
      }
    },
    [
      confirmDanger,
      confirmDeactivationIfNeeded,
      dictionariesById,
      editForm,
      editId,
      notification,
      queryClient,
      t,
      updateDict,
    ]
  )

  return {
    openCreate,
    openEdit,
    editId,
    createForm,
    editForm,
    creating,
    updating,
    openCreateModal,
    closeCreateModal,
    openEditModal,
    closeEditModal,
    handleCreateSubmit,
    handleEditSubmit,
  }
}
