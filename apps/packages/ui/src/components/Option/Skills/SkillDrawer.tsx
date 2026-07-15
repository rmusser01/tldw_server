import React from "react"
import { useMutation } from "@tanstack/react-query"
import {
  Alert,
  Button,
  Collapse,
  Drawer,
  Form,
  Input,
  Modal,
  Radio,
  Segmented,
  Space,
  Switch
} from "antd"
import type { RadioChangeEvent } from "antd"
import { useTranslation } from "react-i18next"
import { Plus, Trash2 } from "lucide-react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"
import type { SkillCreate, SkillResponse, SkillUpdate } from "@/types/skill"
import {
  buildGuidedDraftFromSkill,
  buildGuidedDraftFromTemplate,
  buildDuplicateSkillContent,
  buildInitialSkillContent,
  buildSupportingFilesForCreate,
  buildSupportingFilesForUpdate,
  serializeGuidedSkillContent,
  SKILL_NAME_REGEX,
  SKILL_TEMPLATE_OPTIONS,
  validateGuidedSkillDraft,
  validateRawSkillContent,
  type SkillGuidedDraft,
  type SkillTemplateId,
  type SupportingFileFormEntry
} from "./skill-form-utils"

const SUPPORTING_FILE_NAME_REGEX = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/
const DEFAULT_TEMPLATE_ID: SkillTemplateId = "summarizer"
const DRAFT_STORAGE_PREFIX = "tldw:skills:authoring-draft:v1:"

type AuthoringMode = "guided" | "source"

interface SkillDrawerFormValues extends SkillGuidedDraft {
  content: string
  supportingFiles?: SupportingFileFormEntry[]
}

interface StoredSkillDraft {
  mode: AuthoringMode
  templateId: SkillTemplateId
  values: SkillDrawerFormValues
  baseVersion?: number | null
}

interface SkillDrawerProps {
  open: boolean
  skill: SkillResponse | null
  duplicateFrom?: SkillResponse | null
  draftScope: string | null
  requestSignal?: AbortSignal
  onClose: () => void
  onAfterClose?: () => void
  onSaved: (skillName?: string) => void
}

const getDraftStorageKey = (
  draftScope: string | null,
  skill: SkillResponse | null,
  duplicateFrom: SkillResponse | null | undefined
): string | null => draftScope
  ? `${DRAFT_STORAGE_PREFIX}${draftScope}:${skill?.name ?? (duplicateFrom ? `copy:${duplicateFrom.name}` : "new")}`
  : null

const getSupportingFileRows = (skill: SkillResponse | null): SupportingFileFormEntry[] =>
  Object.entries(skill?.supporting_files ?? {}).map(([filename, content]) => ({
    filename,
    content,
    originalFilename: filename
  }))

const buildDuplicateName = (name: string): string =>
  `${name.slice(0, 59).replace(/-+$/g, "")}-copy`

const buildBaseFormValues = (
  skill: SkillResponse | null,
  duplicateFrom: SkillResponse | null | undefined
): SkillDrawerFormValues => {
  const source = skill ?? duplicateFrom ?? null
  const guided = source
    ? {
        ...buildGuidedDraftFromSkill(source),
        ...(duplicateFrom ? { name: buildDuplicateName(duplicateFrom.name) } : {})
      }
    : buildGuidedDraftFromTemplate(DEFAULT_TEMPLATE_ID, "")
  return {
    ...guided,
    content: skill
      ? buildInitialSkillContent(skill)
      : duplicateFrom
        ? buildDuplicateSkillContent(duplicateFrom, guided.name)
        : serializeGuidedSkillContent(guided),
    supportingFiles: getSupportingFileRows(source)
  }
}

const normalizeSkillSource = (content: string): string =>
  content.replace(/\r\n/g, "\n").trim()

const snapshotFormValues = (values: SkillDrawerFormValues): string =>
  JSON.stringify({
    name: values.name ?? "",
    description: values.description ?? "",
    argumentHint: values.argumentHint ?? "",
    instructions: values.instructions ?? "",
    context: values.context ?? "inline",
    userInvocable: values.userInvocable ?? true,
    allowModelInvocation: values.allowModelInvocation ?? true,
    model: values.model ?? "",
    allowedTools: values.allowedTools ?? "",
    content: values.content ?? "",
    supportingFiles: values.supportingFiles ?? []
  })

const readStoredDraft = (key: string | null): StoredSkillDraft | null => {
  if (typeof window === "undefined" || !key) return null
  try {
    const parsed = JSON.parse(window.sessionStorage.getItem(key) ?? "null") as
      | Partial<StoredSkillDraft>
      | null
    if (!parsed || (parsed.mode !== "guided" && parsed.mode !== "source")) return null
    if (!parsed.values || typeof parsed.values !== "object") return null
    if (!SKILL_TEMPLATE_OPTIONS.some((template) => template.id === parsed.templateId)) {
      return null
    }
    if (
      parsed.baseVersion !== undefined
      && parsed.baseVersion !== null
      && (!Number.isInteger(parsed.baseVersion) || parsed.baseVersion < 1)
    ) {
      return null
    }
    return parsed as StoredSkillDraft
  } catch {
    return null
  }
}

const writeStoredDraft = (key: string | null, draft: StoredSkillDraft | null): void => {
  if (typeof window === "undefined" || !key) return
  try {
    if (draft) window.sessionStorage.setItem(key, JSON.stringify(draft))
    else window.sessionStorage.removeItem(key)
  } catch {
    // Draft persistence is best effort and must not block authoring.
  }
}

const isConflictError = (error: unknown): boolean => {
  const candidate = error as {
    status?: unknown
    statusCode?: unknown
    response?: { status?: unknown }
    message?: unknown
  } | null
  const message = typeof candidate?.message === "string" ? candidate.message : ""
  return candidate?.status === 409
    || candidate?.statusCode === 409
    || candidate?.response?.status === 409
    || /\b409\b/.test(message)
}

const getErrorDescription = (error: unknown): string | undefined =>
  sanitizeServerErrorMessage(error, "") || undefined

const isAbortError = (error: unknown): boolean =>
  Boolean(error && typeof error === "object" && (error as { name?: unknown }).name === "AbortError")

const throwIfAborted = (signal?: AbortSignal): void => {
  if (!signal?.aborted) return
  const error = new Error("Skills request was cancelled")
  error.name = "AbortError"
  throw error
}

export const SkillDrawer: React.FC<SkillDrawerProps> = ({
  open,
  skill,
  duplicateFrom,
  draftScope,
  requestSignal,
  onClose,
  onAfterClose,
  onSaved
}) => {
  const { t } = useTranslation(["option", "common"])
  const notification = useAntdNotification()
  const [form] = Form.useForm<SkillDrawerFormValues>()
  const [modal, contextHolder] = Modal.useModal()
  const drawerConfirmationsRef = React.useRef(
    new Set<ReturnType<typeof modal.confirm>>()
  )
  const requestSignalRef = React.useRef(requestSignal)
  requestSignalRef.current = requestSignal
  const isEdit = Boolean(skill)
  const storageKey = getDraftStorageKey(draftScope, skill, duplicateFrom)
  const storageKeyRef = React.useRef(storageKey)
  const [drawerName, setDrawerName] = React.useState("")
  const [mode, setMode] = React.useState<AuthoringMode>("guided")
  const [selectedTemplateId, setSelectedTemplateId] =
    React.useState<SkillTemplateId>(DEFAULT_TEMPLATE_ID)
  const [isDirty, setIsDirty] = React.useState(false)
  const [recoveredDraft, setRecoveredDraft] = React.useState(false)
  const [hasVersionConflict, setHasVersionConflict] = React.useState(false)
  const [activeVersion, setActiveVersion] = React.useState(skill?.version ?? 1)
  const baselineRef = React.useRef("")
  const baseValuesRef = React.useRef<SkillDrawerFormValues | null>(null)
  const baseModeRef = React.useRef<AuthoringMode>("guided")
  const lastGeneratedContentRef = React.useRef("")
  const dirtyRef = React.useRef(false)

  const runScopedRequest = React.useCallback(async <T,>(
    request: (signal?: AbortSignal) => Promise<T>
  ): Promise<T> => {
    const signal = requestSignal
    throwIfAborted(signal)
    const result = await request(signal)
    throwIfAborted(signal)
    return result
  }, [requestSignal])

  const destroyDrawerConfirmations = React.useCallback(() => {
    const confirmations = Array.from(drawerConfirmationsRef.current)
    drawerConfirmationsRef.current.clear()
    confirmations.forEach((confirmation) => confirmation?.destroy?.())
  }, [])

  const showDrawerConfirmation = React.useCallback((
    config: Parameters<typeof modal.confirm>[0]
  ) => {
    const signal = requestSignal
    let confirmation: ReturnType<typeof modal.confirm>
    const isCurrent = () => (
      signal === requestSignalRef.current
      && !signal?.aborted
    )
    confirmation = modal.confirm({
      ...config,
      onOk: () => (isCurrent() ? config.onOk?.() : undefined),
      onCancel: () => (isCurrent() ? config.onCancel?.() : undefined),
      afterClose: () => {
        drawerConfirmationsRef.current.delete(confirmation)
        config.afterClose?.()
      }
    })
    if (confirmation) drawerConfirmationsRef.current.add(confirmation)
    return confirmation
  }, [modal, requestSignal])

  React.useEffect(() => {
    if (!requestSignal) return
    const closeConfirmations = () => destroyDrawerConfirmations()
    if (requestSignal.aborted) {
      closeConfirmations()
      return
    }
    requestSignal.addEventListener("abort", closeConfirmations, { once: true })
    return () => requestSignal.removeEventListener("abort", closeConfirmations)
  }, [destroyDrawerConfirmations, requestSignal])

  React.useEffect(() => destroyDrawerConfirmations, [destroyDrawerConfirmations])

  React.useEffect(() => {
    storageKeyRef.current = storageKey
  }, [storageKey])

  const updateDirtyState = React.useCallback((dirty: boolean) => {
    dirtyRef.current = dirty
    setIsDirty(dirty)
  }, [])

  const persistValues = React.useCallback(
    (
      values: SkillDrawerFormValues,
      nextMode: AuthoringMode = mode,
      nextTemplateId: SkillTemplateId = selectedTemplateId,
      nextBaseVersion: number = activeVersion
    ) => {
      const dirty = snapshotFormValues(values) !== baselineRef.current
      updateDirtyState(dirty)
      writeStoredDraft(
        storageKeyRef.current,
        dirty
          ? {
              mode: nextMode,
              templateId: nextTemplateId,
              values,
              baseVersion: isEdit ? nextBaseVersion : null
            }
          : null
      )
    },
    [activeVersion, isEdit, mode, selectedTemplateId, updateDirtyState]
  )

  React.useEffect(() => {
    if (!open) return

    const baseValues = buildBaseFormValues(skill, duplicateFrom)
    const guidedContent = serializeGuidedSkillContent(
      skill
        ? buildGuidedDraftFromSkill(skill)
        : duplicateFrom
          ? { ...buildGuidedDraftFromSkill(duplicateFrom), name: buildDuplicateName(duplicateFrom.name) }
          : buildGuidedDraftFromTemplate(DEFAULT_TEMPLATE_ID, "")
    )
    const source = skill ?? duplicateFrom
    const baseMode: AuthoringMode = source?.raw_content?.trim()
      && normalizeSkillSource(baseValues.content) !== normalizeSkillSource(guidedContent)
      ? "source"
      : "guided"
    baseValuesRef.current = baseValues
    baseModeRef.current = baseMode
    baselineRef.current = snapshotFormValues(baseValues)
    lastGeneratedContentRef.current = guidedContent
    setActiveVersion(skill?.version ?? 1)
    setHasVersionConflict(false)
    setSelectedTemplateId(DEFAULT_TEMPLATE_ID)

    const stored = readStoredDraft(storageKeyRef.current)
    if (stored) {
      const storedBaseVersion = stored.baseVersion ?? null
      form.setFieldsValue({ ...baseValues, ...stored.values })
      setDrawerName(stored.values.name ?? baseValues.name)
      setMode(stored.mode)
      setSelectedTemplateId(stored.templateId)
      setRecoveredDraft(true)
      if (skill) {
        setActiveVersion(storedBaseVersion ?? skill.version)
        setHasVersionConflict(
          storedBaseVersion === null || storedBaseVersion !== skill.version
        )
      }
      updateDirtyState(true)
      return
    }

    form.resetFields()
    form.setFieldsValue(baseValues)
    setDrawerName(baseValues.name)
    setMode(baseMode)
    setRecoveredDraft(false)
    updateDirtyState(false)
  }, [duplicateFrom, form, open, skill, updateDirtyState])

  const handleValuesChange = (
    _changedValues: Partial<SkillDrawerFormValues>,
    allValues: SkillDrawerFormValues
  ) => {
    const nextValues = { ...allValues }
    setDrawerName(nextValues.name ?? "")
    if (mode === "guided") {
      const generated = serializeGuidedSkillContent(nextValues)
      nextValues.content = generated
      lastGeneratedContentRef.current = generated
      form.setFieldValue("content", generated)
    }
    persistValues(nextValues)
  }

  const applyTemplate = (templateId: SkillTemplateId) => {
    const currentName = form.getFieldValue("name") ?? ""
    const currentTemplate = buildGuidedDraftFromTemplate(selectedTemplateId, "")
    const preserveName = currentName !== currentTemplate.name ? currentName : ""
    const guided = buildGuidedDraftFromTemplate(templateId, preserveName)
    const content = serializeGuidedSkillContent(guided)
    const nextValues: SkillDrawerFormValues = {
      ...form.getFieldsValue(true),
      ...guided,
      content
    }
    setSelectedTemplateId(templateId)
    setDrawerName(guided.name)
    lastGeneratedContentRef.current = content
    form.setFieldsValue(nextValues)
    persistValues(nextValues, "guided", templateId)
  }

  const handleTemplateChange = (event: RadioChangeEvent) => {
    const nextTemplateId = event.target.value as SkillTemplateId
    if (nextTemplateId === selectedTemplateId) return

    const currentValues = form.getFieldsValue(true)
    const expected = buildGuidedDraftFromTemplate(selectedTemplateId, currentValues.name ?? "")
    const hasCustomizedTemplate = currentValues.description !== expected.description
      || currentValues.argumentHint !== expected.argumentHint
      || currentValues.instructions !== expected.instructions
      || currentValues.context !== expected.context
      || currentValues.userInvocable !== expected.userInvocable
      || currentValues.allowModelInvocation !== expected.allowModelInvocation
      || currentValues.model !== expected.model
      || currentValues.allowedTools !== expected.allowedTools

    if (!hasCustomizedTemplate) {
      applyTemplate(nextTemplateId)
      return
    }

    showDrawerConfirmation({
      title: t("option:skills.replaceTemplateTitle", {
        defaultValue: "Replace guided draft with template?"
      }),
      content: t("option:skills.replaceTemplateDescription", {
        defaultValue: "This replaces the current guided fields with the selected template."
      }),
      okText: t("option:skills.replaceTemplateConfirm", { defaultValue: "Replace draft" }),
      cancelText: t("option:skills.replaceTemplateCancel", { defaultValue: "Keep current draft" }),
      onOk: () => applyTemplate(nextTemplateId)
    })
  }

  const changeMode = (nextMode: AuthoringMode) => {
    if (nextMode === mode) return

    if (nextMode === "source") {
      setMode("source")
      persistValues(form.getFieldsValue(true), "source")
      return
    }

    const currentContent = form.getFieldValue("content") ?? ""
    if (currentContent === lastGeneratedContentRef.current) {
      setMode("guided")
      persistValues(form.getFieldsValue(true), "guided")
      return
    }

    showDrawerConfirmation({
      title: t("option:skills.replaceAdvancedTitle", {
        defaultValue: "Replace advanced source?"
      }),
      content: t("option:skills.replaceAdvancedDescription", {
        defaultValue:
          "Guided fields cannot preserve custom frontmatter. Continue only if you want to replace the advanced source."
      }),
      okText: t("option:skills.replaceAdvancedConfirm", {
        defaultValue: "Use guided fields"
      }),
      cancelText: t("option:skills.replaceAdvancedCancel", {
        defaultValue: "Keep advanced source"
      }),
      onOk: () => {
        const values = form.getFieldsValue(true)
        const content = serializeGuidedSkillContent(values)
        const nextValues = { ...values, content }
        form.setFieldValue("content", content)
        lastGeneratedContentRef.current = content
        setMode("guided")
        persistValues(nextValues, "guided")
      }
    })
  }

  const clearDraft = React.useCallback(() => {
    writeStoredDraft(storageKeyRef.current, null)
    updateDirtyState(false)
    setRecoveredDraft(false)
  }, [updateDirtyState])

  const discardRecoveredDraft = () => {
    const baseValues = baseValuesRef.current
    if (!baseValues) return
    form.setFieldsValue(baseValues)
    setDrawerName(baseValues.name)
    setMode(baseModeRef.current)
    setActiveVersion(skill?.version ?? 1)
    setHasVersionConflict(false)
    clearDraft()
  }

  const requestClose = () => {
    if (!dirtyRef.current) {
      onClose()
      return
    }

    showDrawerConfirmation({
      title: t("option:skills.discardDraftTitle", {
        defaultValue: "Discard unsaved skill draft?"
      }),
      content: t("option:skills.discardDraftDescription", {
        defaultValue: "Your unsaved changes will be removed from this browser session."
      }),
      okText: t("option:skills.discardDraftConfirm", { defaultValue: "Discard draft" }),
      okButtonProps: { danger: true },
      cancelText: t("option:skills.discardDraftCancel", { defaultValue: "Keep editing" }),
      onOk: () => {
        clearDraft()
        onClose()
      }
    })
  }

  const createMutation = useMutation({
    mutationFn: (values: SkillCreate) => runScopedRequest((signal) =>
      signal
        ? tldwClient.createSkill(values, { signal })
        : tldwClient.createSkill(values)
    ),
    onSuccess: (result, values) => {
      clearDraft()
      notification.success({
        message: t("option:skills.createSuccess", { defaultValue: "Skill created" })
      })
      const responseName = typeof result?.name === "string" ? result.name.trim() : ""
      onSaved(SKILL_NAME_REGEX.test(responseName) ? responseName : values.name)
    },
    onError: (error: unknown) => {
      if (isAbortError(error)) return
      notification.error({
        message: t("option:skills.createError", { defaultValue: "Failed to create skill" }),
        description: isConflictError(error)
          ? t("option:skills.duplicateError", {
              defaultValue: "A skill with this name already exists."
            })
          : getErrorDescription(error)
      })
    }
  })

  const updateMutation = useMutation({
    mutationFn: (values: SkillUpdate) =>
      runScopedRequest((signal) => signal
        ? tldwClient.updateSkill(skill!.name, values, activeVersion, { signal })
        : tldwClient.updateSkill(skill!.name, values, activeVersion)
      ),
    onSuccess: () => {
      clearDraft()
      notification.success({
        message: t("option:skills.updateSuccess", { defaultValue: "Skill updated" })
      })
      onSaved()
    },
    onError: (error: unknown) => {
      if (isAbortError(error)) return
      if (isConflictError(error)) {
        setHasVersionConflict(true)
        return
      }
      notification.error({
        message: t("option:skills.updateError", { defaultValue: "Failed to update skill" }),
        description: getErrorDescription(error)
      })
    }
  })

  const reloadVersionMutation = useMutation({
    mutationFn: () => runScopedRequest((signal) => signal
      ? tldwClient.getSkill(skill!.name, { signal })
      : tldwClient.getSkill(skill!.name)
    ),
    onSuccess: (latest) => {
      const latestSource = latest.raw_content?.trim()
        ? latest.raw_content
        : buildInitialSkillContent(latest)
      showDrawerConfirmation({
        title: t("option:skills.conflictReviewTitle", {
          defaultValue: "Overwrite the latest server version?"
        }),
        content: (
          <div className="grid gap-3">
            <p className="m-0">
              {t("option:skills.conflictReviewDescription", {
                defaultValue:
                  "Review the latest source below. Continuing keeps your local draft and uses the latest version token when you save."
              })}
            </p>
            <pre
              role="region"
              tabIndex={0}
              aria-label={t("option:skills.latestServerSource", {
                defaultValue: "Latest server source"
              })}
              className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-surface-muted p-3 text-xs"
            >
              {latestSource}
            </pre>
          </div>
        ),
        okText: t("option:skills.keepDraftAndOverwrite", {
          defaultValue: "Keep draft and overwrite"
        }),
        cancelText: t("option:skills.keepReviewing", { defaultValue: "Keep reviewing" }),
        okButtonProps: { danger: true },
        onOk: () => {
          setActiveVersion(latest.version)
          setHasVersionConflict(false)
          persistValues(
            form.getFieldsValue(true),
            mode,
            selectedTemplateId,
            latest.version
          )
          notification.success({
            message: t("option:skills.conflictOverwriteReady", {
              defaultValue: "Draft ready to overwrite latest"
            }),
            description: t("option:skills.conflictOverwriteReadyDescription", {
              defaultValue: "Your local draft is unchanged. Save when you are ready to overwrite version {{version}}.",
              version: latest.version
            })
          })
        }
      })
    },
    onError: (error: unknown) => {
      if (isAbortError(error)) return
      notification.error({
        message: t("option:skills.versionReloadError", {
          defaultValue: "Failed to reload latest version"
        }),
        description: getErrorDescription(error)
      })
    }
  })

  const setValidationErrors = (errors: string[]) => {
    const fields: Array<{ name: keyof SkillDrawerFormValues; errors: string[] }> = []
    const nameError = errors.find((error) => error.startsWith("Name "))
    const descriptionError = errors.find((error) => error.startsWith("Description "))
    const instructionsError = errors.find((error) => error.startsWith("Instructions "))
    if (nameError) fields.push({ name: "name", errors: [nameError] })
    if (descriptionError) fields.push({ name: "description", errors: [descriptionError] })
    if (instructionsError) fields.push({ name: "instructions", errors: [instructionsError] })
    form.setFields(fields)
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      const content = mode === "guided"
        ? serializeGuidedSkillContent(values)
        : values.content
      const contentErrors = mode === "guided"
        ? validateGuidedSkillDraft(values)
        : validateRawSkillContent(content, values.name.trim())
      if (contentErrors.length > 0) {
        if (mode === "guided") setValidationErrors(contentErrors)
        else form.setFields([{ name: "content", errors: contentErrors }])
        return
      }

      if (isEdit) {
        let supportingFiles: Record<string, string | null> | undefined
        try {
          supportingFiles = buildSupportingFilesForUpdate(
            skill!.supporting_files,
            values.supportingFiles ?? baseValuesRef.current?.supportingFiles
          )
        } catch (error: unknown) {
          notification.error({
            message: t("option:skills.supportingFilesInvalid", {
              defaultValue: "Invalid supporting files"
            }),
            description: getErrorDescription(error)
          })
          return
        }
        const payload: SkillUpdate = { content }
        if (supportingFiles) payload.supporting_files = supportingFiles
        updateMutation.mutate(payload)
        return
      }

      let supportingFiles: Record<string, string> | undefined
      try {
        supportingFiles = buildSupportingFilesForCreate(
          values.supportingFiles ?? baseValuesRef.current?.supportingFiles
        )
      } catch (error: unknown) {
        notification.error({
          message: t("option:skills.supportingFilesInvalid", {
            defaultValue: "Invalid supporting files"
          }),
          description: getErrorDescription(error)
        })
        return
      }
      const payload: SkillCreate = { name: values.name.trim(), content }
      if (supportingFiles) payload.supporting_files = supportingFiles
      createMutation.mutate(payload)
    } catch {
      // Ant Design renders field-level validation errors.
    }
  }

  const isSaving = createMutation.isPending || updateMutation.isPending
  const titleName = drawerName.trim() || skill?.name || t("option:skills.untitled", {
    defaultValue: "untitled"
  })

  return (
    <Drawer
      title={isEdit
        ? t("option:skills.editTitleNamed", {
            defaultValue: "Edit Skill: {{name}}",
            name: titleName
          })
        : t("option:skills.newTitleNamed", {
            defaultValue: "New Skill: {{name}}",
            name: titleName
          })}
      open={open}
      onClose={requestClose}
      afterOpenChange={(isOpen) => {
        if (!isOpen) onAfterClose?.()
      }}
      size={720}
      styles={{ wrapper: { maxWidth: "100vw" } }}
      destroyOnHidden
      extra={(
        <Space>
          <Button onClick={requestClose}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            onClick={handleSubmit}
            loading={isSaving}
            disabled={hasVersionConflict}
            aria-label={t("common:save", { defaultValue: "Save" })}
          >
            {t("common:save", { defaultValue: "Save" })}
          </Button>
        </Space>
      )}
    >
      {contextHolder}

      {recoveredDraft && (
        <Alert
          className="mb-4"
          type="info"
          showIcon
          title={t("option:skills.draftRecovered", {
            defaultValue: "Recovered your unsaved draft from this session."
          })}
          action={(
            <Button size="small" onClick={discardRecoveredDraft}>
              {t("option:skills.discardRecovered", { defaultValue: "Discard recovered draft" })}
            </Button>
          )}
        />
      )}

      {hasVersionConflict && (
        <Alert
          className="mb-4"
          type="warning"
          showIcon
          title={t("option:skills.versionConflict", {
            defaultValue:
              "This skill changed elsewhere. Review the latest version before choosing whether to overwrite it."
          })}
          action={(
            <Button
              size="small"
              loading={reloadVersionMutation.isPending}
              onClick={() => reloadVersionMutation.mutate()}
            >
              {t("option:skills.reviewLatest", { defaultValue: "Review latest" })}
            </Button>
          )}
        />
      )}

      <Segmented<AuthoringMode>
        block
        className="mb-4"
        value={mode}
        onChange={changeMode}
        options={[
          { label: t("option:skills.guidedMode", { defaultValue: "Guided" }), value: "guided" },
          {
            label: t("option:skills.sourceMode", { defaultValue: "Advanced source" }),
            value: "source"
          }
        ]}
      />

      <Form
        form={form}
        layout="vertical"
        autoComplete="off"
        onValuesChange={handleValuesChange}
      >
        {mode === "guided" && !isEdit && (
          <Form.Item
            label={t("option:skills.templateLabel", { defaultValue: "Start from template" })}
            extra={t("option:skills.templateHelpGuided", {
              defaultValue: "Choose a starting point, then edit the fields below."
            })}
          >
            <Radio.Group
              className="flex flex-wrap"
              optionType="button"
              buttonStyle="solid"
              value={selectedTemplateId}
              onChange={handleTemplateChange}
            >
              {SKILL_TEMPLATE_OPTIONS.map((template) => (
                <Radio.Button key={template.id} value={template.id}>
                  {t(`option:skills.templates.${template.id}.label`, {
                    defaultValue: template.label
                  })}
                </Radio.Button>
              ))}
            </Radio.Group>
            <p className="mt-2 mb-0 text-sm text-text-subtle">
              {(() => {
                const template = SKILL_TEMPLATE_OPTIONS.find(
                  (option) => option.id === selectedTemplateId
                ) ?? SKILL_TEMPLATE_OPTIONS[0]
                return t(`option:skills.templates.${template.id}.description`, {
                  defaultValue: template.description
                })
              })()}
            </p>
          </Form.Item>
        )}

        {mode === "guided" ? (
          <>
            {isEdit && skill?.raw_content
              && skill.raw_content !== lastGeneratedContentRef.current && (
                <p className="mb-4 text-sm text-text-subtle">
                  {t("option:skills.guidedRewriteNotice", {
                    defaultValue:
                      "Guided changes rewrite supported metadata. Use Advanced source to preserve custom frontmatter."
                  })}
                </p>
              )}

            <Form.Item name="name" label={t("option:skills.nameLabel", { defaultValue: "Name" })}>
              <Input
                placeholder="my-skill-name"
                disabled={isEdit}
                maxLength={64}
                className="font-mono"
              />
            </Form.Item>

            <Form.Item
              name="description"
              label={t("option:skills.descriptionLabel", { defaultValue: "Description" })}
            >
              <Input placeholder="What this skill helps the user do" maxLength={500} />
            </Form.Item>

            <Form.Item
              name="argumentHint"
              label={t("option:skills.argumentHintLabel", { defaultValue: "Argument hint" })}
              extra={t("option:skills.argumentHintHelp", {
                defaultValue: "Shown to users as a short example of the expected input."
              })}
            >
              <Input className="font-mono" placeholder="[topic] [audience]" />
            </Form.Item>

            <Form.Item
              name="instructions"
              label={t("option:skills.instructionsLabel", { defaultValue: "Instructions" })}
              extra={t("option:skills.instructionsHelp", {
                defaultValue: "Use $ARGUMENTS where supplied input should be inserted."
              })}
            >
              <Input.TextArea rows={12} placeholder="Process $ARGUMENTS and return..." />
            </Form.Item>

            <Form.Item
              name="context"
              label={t("option:skills.executionModeLabel", { defaultValue: "Execution mode" })}
            >
              <Radio.Group>
                <Radio value="inline">
                  {t("option:skills.executionInline", { defaultValue: "Inline" })}
                </Radio>
                <Radio value="fork">
                  {t("option:skills.executionFork", { defaultValue: "Forked model run" })}
                </Radio>
              </Radio.Group>
            </Form.Item>

            <div className="grid grid-cols-1 gap-x-4 sm:grid-cols-2">
              <Form.Item
                name="userInvocable"
                valuePropName="checked"
                label={t("option:skills.userInvocableLabel", { defaultValue: "Visible in chat" })}
              >
                <Switch
                  aria-label={t("option:skills.userInvocableLabel", {
                    defaultValue: "Visible in chat"
                  })}
                />
              </Form.Item>
              <Form.Item
                name="allowModelInvocation"
                valuePropName="checked"
                label={t("option:skills.modelInvocationLabel", {
                  defaultValue: "Allow model invocation"
                })}
              >
                <Switch
                  aria-label={t("option:skills.modelInvocationLabel", {
                    defaultValue: "Allow model invocation"
                  })}
                />
              </Form.Item>
            </div>

            <Form.Item
              name="model"
              label={t("option:skills.modelOverrideLabel", { defaultValue: "Model override" })}
              extra={t("option:skills.optionalField", { defaultValue: "Optional" })}
            >
              <Input placeholder="Provider model name" />
            </Form.Item>

            <Form.Item
              name="allowedTools"
              label={t("option:skills.allowedToolsLabel", { defaultValue: "Declared tools" })}
              extra={t("option:skills.allowedToolsHelp", {
                defaultValue: "Separate tool names with commas or new lines."
              })}
            >
              <Input.TextArea rows={3} className="font-mono text-xs" placeholder="Read, Grep" />
            </Form.Item>
          </>
        ) : (
          <>
            <Form.Item
              name="name"
              label={t("option:skills.nameLabel", { defaultValue: "Name" })}
              rules={[
                {
                  required: true,
                  whitespace: true,
                  message: t("option:skills.nameRequired", {
                    defaultValue: "Name is required"
                  })
                },
                {
                  validator: async (_, value: string | undefined) => {
                    const trimmed = (value ?? "").trim()
                    if (!trimmed || SKILL_NAME_REGEX.test(trimmed)) return
                    throw new Error(t("option:skills.nameInvalid", {
                      defaultValue:
                        "Must start with a letter, use only lowercase letters, numbers, and hyphens (max 64 chars)"
                    }))
                  }
                }
              ]}
            >
              <Input
                placeholder="my-skill-name"
                disabled={isEdit}
                maxLength={64}
                className="font-mono"
              />
            </Form.Item>
            <Form.Item
              name="content"
              label={t("option:skills.contentLabel", { defaultValue: "SKILL.md Content" })}
              extra={t("option:skills.contentHelp", {
                defaultValue:
                  "Edit YAML frontmatter and instructions directly. Use $ARGUMENTS for input substitution."
              })}
            >
              <Input.TextArea rows={24} className="font-mono text-xs" />
            </Form.Item>
          </>
        )}

        <Collapse
          className="mb-4"
          items={[{
            key: "supporting-files",
            label: t("option:skills.supportingFilesLabel", { defaultValue: "Supporting Files" }),
            children: (
              <Form.List name="supportingFiles">
                {(fields, { add, remove }) => (
                  <div className="flex flex-col gap-3">
                    {fields.map((field) => (
                      <div key={field.key} className="rounded border p-3">
                        <Form.Item name={[field.name, "originalFilename"]} hidden>
                          <Input />
                        </Form.Item>
                        <div className="mb-2 flex items-start gap-2">
                          <Form.Item
                            className="mb-0 flex-1"
                            name={[field.name, "filename"]}
                            label={t("option:skills.supportingFileName", { defaultValue: "Filename" })}
                            rules={[
                              {
                                required: true,
                                whitespace: true,
                                message: t("option:skills.supportingFileNameRequired", {
                                  defaultValue: "Filename is required"
                                })
                              },
                              {
                                validator: async (_, value: string | undefined) => {
                                  const trimmed = (value ?? "").trim()
                                  if (!trimmed) return
                                  if (!SUPPORTING_FILE_NAME_REGEX.test(trimmed)) {
                                    throw new Error(t("option:skills.supportingFileNameInvalid", {
                                      defaultValue:
                                        "Use letters, numbers, dot, underscore, or hyphen (max 100 chars)"
                                    }))
                                  }
                                  if (trimmed.toLowerCase() === "skill.md") {
                                    throw new Error(t("option:skills.supportingFileNameReserved", {
                                      defaultValue: "SKILL.md is reserved"
                                    }))
                                  }
                                }
                              }
                            ]}
                          >
                            <Input className="font-mono text-xs" placeholder="reference.md" />
                          </Form.Item>
                          <Button
                            danger
                            type="text"
                            className="min-h-11 min-w-11"
                            icon={<Trash2 size={14} />}
                            aria-label={t("option:skills.removeSupportingFile", {
                              defaultValue: "Remove supporting file"
                            })}
                            onClick={() => remove(field.name)}
                          />
                        </div>
                        <Form.Item
                          name={[field.name, "content"]}
                          label={t("option:skills.supportingFileContent", { defaultValue: "Content" })}
                          className="mb-0"
                        >
                          <Input.TextArea rows={5} className="font-mono text-xs" />
                        </Form.Item>
                      </div>
                    ))}
                    <Button
                      type="dashed"
                      icon={<Plus size={14} />}
                      onClick={() => add({ filename: "", content: "" })}
                    >
                      {t("option:skills.addSupportingFile", { defaultValue: "Add Supporting File" })}
                    </Button>
                  </div>
                )}
              </Form.List>
            )
          }]}
        />
      </Form>
      <span className="sr-only" aria-live="polite">
        {isDirty ? t("option:skills.unsavedChanges", { defaultValue: "Unsaved changes" }) : ""}
      </span>
    </Drawer>
  )
}
