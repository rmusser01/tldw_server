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
      notification.success({ message: "Dictionary updated" })
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
              message: "Template not applied",
              description:
                "Dictionary was created, but starter entries could not be added automatically.",
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
                message: "Template only partially applied",
                description:
                  starterEntriesApplied > 0
                    ? `Dictionary was created, but only ${starterEntriesApplied} of ${starterEntriesTotal} starter entries were added automatically.`
                    : "Dictionary was created, but starter entries could not be added automatically.",
              })
            }
          }
        }

        await queryClient.invalidateQueries({ queryKey: ["tldw:listDictionaries"] })
        notification.success({
          message: "Dictionary created",
          description:
            starterEntriesApplied > 0 && starterEntriesApplied === starterEntriesTotal
              ? `"${values?.name || "Dictionary"}" created with ${starterEntriesApplied} starter entr${starterEntriesApplied === 1 ? "y" : "ies"}.`
              : starterEntriesApplied > 0
                ? `"${values?.name || "Dictionary"}" created with ${starterEntriesApplied} of ${starterEntriesTotal} starter entries.`
                : `"${values?.name || "Dictionary"}" created.`,
        })
        setOpenCreate(false)
        createForm.resetFields()
      } catch (error: any) {
        notification.error({
          message: "Error",
          description: error?.message || "Failed to create dictionary",
        })
      }
    },
    [createDict, createForm, notification, queryClient]
  )

  const handleEditSubmit = React.useCallback(
    async (values: any) => {
      if (editId == null) {
        try {
          await updateDict(values)
        } catch (error: any) {
          notification.error({
            message: "Error",
            description: error?.message || "Failed to update dictionary",
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
            message: "Error",
            description: error?.message || "Failed to update dictionary",
          })
          return
        }

        const shouldReload = await confirmDanger({
          title: "Dictionary changed in another session",
          content:
            "This dictionary was updated elsewhere. Reload the latest version while keeping your current edits?",
          okText: "Reload latest",
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
            message: "Latest version loaded",
            description: "Your edits were preserved. Save again to retry.",
          })
          await queryClient.invalidateQueries({ queryKey: ["tldw:listDictionaries"] })
        } catch (reloadError: any) {
          notification.error({
            message: "Reload failed",
            description:
              reloadError?.message ||
              "Could not load the latest dictionary version. Please retry.",
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
