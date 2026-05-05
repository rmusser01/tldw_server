import React from "react"
import {
  Button,
  Drawer,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Switch,
  Tag,
  Tooltip,
  message
} from "antd"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { fetchChatModels } from "@/services/tldw-server"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useUiModeStore } from "@/store/ui-mode"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { Alert } from "@/components/ui/primitives"
import { ModalFooter } from "@/components/ui/layout"
import { RefreshCw, Trash2 } from "lucide-react"

const Markdown = React.lazy(() => import("../Markdown"))

type DocumentGeneratorDrawerProps = {
  open: boolean
  onClose: () => void
  conversationId?: string | null
  defaultModel?: string | null
  seedMessage?: string | null
  seedMessageId?: string | null
}

type DocumentJob = {
  job_id: string
  status?: string
  document_type?: string
  created_at?: string
  completed_at?: string
  progress_percentage?: number | null
  error_message?: string | null
  result_content?: string | null
}

type GeneratedDocument = {
  id: number
  title?: string
  content?: string
  document_type?: string
  created_at?: string
  provider?: string
  model?: string
  generation_time_ms?: number
  token_count?: number
}

type ChatModel = {
  model: string
  nickname?: string
  provider?: string
}

const DOCUMENT_TYPES = [
  "summary",
  "timeline",
  "briefing",
  "study_guide",
  "q_and_a",
  "meeting_notes"
] as const

type DocumentType = (typeof DOCUMENT_TYPES)[number]

type GeneratorFormState = {
  documentType: DocumentType
  selectedModel: string
  manualModel: string
  manualProvider: string
  customPrompt: string
  useSpecificMessage: boolean
  specificMessage: string
  asyncGeneration: boolean
}

type GeneratorFormAction =
  | Partial<GeneratorFormState>
  | ((prev: GeneratorFormState) => Partial<GeneratorFormState>)

const INITIAL_FORM_STATE: GeneratorFormState = {
  documentType: "summary",
  selectedModel: "",
  manualModel: "",
  manualProvider: "",
  customPrompt: "",
  useSpecificMessage: false,
  specificMessage: "",
  asyncGeneration: true
}

const formatTimestamp = (value?: string) => {
  if (!value) return ""
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const extractDocumentList = (res: unknown): GeneratedDocument[] => {
  if (!res || typeof res !== "object") return []
  const data = res as Record<string, unknown>
  const list = data.documents || data.items || data.results
  return Array.isArray(list) ? (list as GeneratedDocument[]) : []
}

const getErrorMessage = (err: unknown, fallback: string) => {
  if (err instanceof Error) return err.message
  if (!err || typeof err !== "object" || !("message" in err)) return fallback
  const messageValue = (err as { message?: unknown }).message
  return typeof messageValue === "string" && messageValue ? messageValue : fallback
}

const hasValidationErrorFields = (err: unknown) =>
  Boolean(err && typeof err === "object" && "errorFields" in err)

export const DocumentGeneratorDrawer: React.FC<DocumentGeneratorDrawerProps> = ({
  open,
  onClose,
  conversationId,
  defaultModel,
  seedMessage,
  seedMessageId
}) => {
  const { t } = useTranslation(["playground", "common"])
  const { capabilities } = useServerCapabilities()
  const uiMode = useUiModeStore((state) => state.mode)
  const [promptForm] = Form.useForm()
  const [formState, setFormState] = React.useReducer(
    (prev: GeneratorFormState, action: GeneratorFormAction) => ({
      ...prev,
      ...(typeof action === "function" ? action(prev) : action)
    }),
    INITIAL_FORM_STATE
  )
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [documents, setDocuments] = React.useState<GeneratedDocument[]>([])
  const [jobs, setJobs] = React.useState<DocumentJob[]>([])
  const [docsLoading, setDocsLoading] = React.useState(false)
  const [promptLoading, setPromptLoading] = React.useState(false)
  const [activeDoc, setActiveDoc] = React.useState<GeneratedDocument | null>(null)
  const initializedRef = React.useRef(false)

  const {
    documentType,
    selectedModel,
    manualModel,
    manualProvider,
    customPrompt,
    useSpecificMessage,
    specificMessage,
    asyncGeneration
  } = formState

  const { data: composerModels = [] } = useQuery<ChatModel[]>({
    queryKey: ["document-generator:models"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: open
  })

  const modelOptions = React.useMemo(() => {
    return composerModels
      .filter((model) => model?.model)
      .map((model) => ({
        value: model.model,
        label: model.nickname || model.model,
        provider: String(model.provider || "custom").toLowerCase()
      }))
  }, [composerModels])

  const modelMeta = React.useMemo(() => {
    const map = new Map<string, { label: string; provider: string }>()
    modelOptions.forEach((model) => {
      map.set(model.value, {
        label: model.label,
        provider: model.provider
      })
    })
    return map
  }, [modelOptions])

  const activeProvider = React.useMemo(() => {
    if (modelOptions.length === 0) return manualProvider
    return modelMeta.get(selectedModel)?.provider || ""
  }, [manualProvider, modelMeta, modelOptions.length, selectedModel])

  const activeModel = React.useMemo(() => {
    if (modelOptions.length === 0) return manualModel
    return selectedModel
  }, [manualModel, modelOptions.length, selectedModel])

  const docTypeOptions = React.useMemo(
    () =>
      DOCUMENT_TYPES.map((type) => ({
        value: type,
        label: t(
          `playground:documentGenerator.types.${type}`,
          type.replace(/_/g, " ")
        )
      })),
    [t]
  )

  const resetDefaults = React.useCallback(() => {
    const fallbackModel =
      (defaultModel && modelMeta.has(defaultModel) && defaultModel) ||
      modelOptions[0]?.value ||
      ""
    if (modelOptions.length > 0) {
      setFormState({
        selectedModel: fallbackModel,
        manualModel: INITIAL_FORM_STATE.manualModel,
        manualProvider: INITIAL_FORM_STATE.manualProvider
      })
    } else {
      setFormState({
        selectedModel: INITIAL_FORM_STATE.selectedModel,
        manualModel: defaultModel || INITIAL_FORM_STATE.manualModel,
        manualProvider: INITIAL_FORM_STATE.manualProvider
      })
    }
    setFormState({
      customPrompt: INITIAL_FORM_STATE.customPrompt,
      useSpecificMessage: Boolean(seedMessage),
      specificMessage: seedMessage || INITIAL_FORM_STATE.specificMessage
    })
  }, [defaultModel, modelMeta, modelOptions, seedMessage])

  React.useEffect(() => {
    if (open && !initializedRef.current) {
      initializedRef.current = true
      resetDefaults()
    }
    if (!open) {
      initializedRef.current = false
    }
  }, [open, resetDefaults])

  React.useEffect(() => {
    if (!open || modelOptions.length === 0) return
    if (!selectedModel) {
      const fallback =
        (defaultModel && modelMeta.has(defaultModel) && defaultModel) ||
        modelOptions[0]?.value ||
        ""
      if (fallback) {
        setFormState({ selectedModel: fallback })
      }
    }
  }, [defaultModel, modelMeta, modelOptions, open, selectedModel])

  React.useEffect(() => {
    if (!open) return
    if (seedMessage && seedMessage !== specificMessage) {
      setFormState({
        useSpecificMessage: true,
        specificMessage: seedMessage
      })
    }
  }, [open, seedMessage, specificMessage])

  const ensureTldwClient = React.useCallback(
    async (context: string, options: { notify?: boolean } = {}) => {
      try {
        await tldwClient.initialize()
        return true
      } catch (err) {
        console.error(`tldwClient.initialize failed (${context})`, err)
        if (options.notify) {
          message.error(
            t("common:somethingWentWrong", { defaultValue: "Something went wrong" })
          )
        }
        return false
      }
    },
    [t]
  )

  const refreshDocuments = React.useCallback(async () => {
    if (!conversationId) return
    setDocsLoading(true)
    try {
      if (!(await ensureTldwClient("refreshDocuments"))) return
      const res = await tldwClient.listChatDocuments({
        conversation_id: conversationId,
        limit: 50
      })
      const list = extractDocumentList(res)
      setDocuments(list)
    } catch (err: unknown) {
      message.error(getErrorMessage(err, t("common:somethingWentWrong")))
    } finally {
      setDocsLoading(false)
    }
  }, [conversationId, ensureTldwClient, t])

  React.useEffect(() => {
    if (!open || !conversationId) return
    void refreshDocuments()
  }, [open, conversationId, refreshDocuments])

  const loadPromptConfig = React.useCallback(async (options: { notify?: boolean } = {}) => {
    setPromptLoading(true)
    try {
      if (!(await ensureTldwClient("loadPromptConfig", options))) return
      const res = await tldwClient.getChatDocumentPrompt(documentType)
      promptForm.setFieldsValue({
        system_prompt: res?.system_prompt ?? "",
        user_prompt: res?.user_prompt ?? "",
        temperature: res?.temperature ?? 0.7,
        max_tokens: res?.max_tokens ?? 2000
      })
    } catch (err: unknown) {
      message.error(getErrorMessage(err, t("common:somethingWentWrong")))
    } finally {
      setPromptLoading(false)
    }
  }, [documentType, ensureTldwClient, promptForm, t])

  React.useEffect(() => {
    if (!open || uiMode !== "pro") return
    void loadPromptConfig()
  }, [open, uiMode, documentType, loadPromptConfig])

  const refreshJobs = React.useCallback(async () => {
    if (jobs.length === 0) return
    try {
      if (!(await ensureTldwClient("refreshJobs"))) return
      const updates = await Promise.all(
        jobs.map(async (job) => {
          try {
            const res = await tldwClient.getChatDocumentJob(job.job_id)
            return {
              ...job,
              ...(res || {})
            }
          } catch (err) {
            console.debug(`Failed to refresh job ${job.job_id}:`, err)
            return job
          }
        })
      )
      setJobs(updates)
      if (
        updates.some((job) =>
          ["completed", "failed", "cancelled"].includes(String(job.status))
        )
      ) {
        void refreshDocuments()
      }
    } catch (err) {
      console.error("refreshJobs polling error", err)
    }
  }, [ensureTldwClient, jobs, refreshDocuments])

  React.useEffect(() => {
    if (!open) return
    const hasActiveJob = jobs.some((job) =>
      ["pending", "in_progress"].includes(String(job.status))
    )
    if (!hasActiveJob) return
    const id = window.setInterval(() => {
      void refreshJobs()
    }, 5000)
    return () => window.clearInterval(id)
  }, [jobs, open, refreshJobs])

  const handleGenerate = async () => {
    if (!conversationId) return
    if (!activeModel || !activeProvider) {
      message.error(
        t(
          "playground:documentGenerator.errors.missingModel",
          "Select a model and provider first."
        )
      )
      return
    }
    setIsGenerating(true)
    try {
      if (!(await ensureTldwClient("handleGenerate", { notify: true }))) return
      const payload = {
        conversation_id: conversationId,
        document_type: documentType,
        provider: activeProvider,
        model: activeModel,
        custom_prompt: customPrompt.trim() || undefined,
        specific_message:
          useSpecificMessage && specificMessage.trim()
            ? specificMessage.trim()
            : undefined,
        async_generation: asyncGeneration,
        stream: false
      }
      const res = await tldwClient.generateChatDocument(payload)
      if (res?.job_id) {
        setJobs((prev) => [
          {
            job_id: res.job_id,
            status: res.status,
            document_type: res.document_type,
            created_at: res.created_at
          },
          ...prev
        ])
        message.success(
          t(
            "playground:documentGenerator.jobQueued",
            "Document generation queued."
          )
        )
      } else if (res?.document_id || res?.id) {
        message.success(
          t(
            "playground:documentGenerator.generated",
            "Document generated."
          )
        )
        void refreshDocuments()
      } else {
        message.success(
          t(
            "playground:documentGenerator.generated",
            "Document generated."
          )
        )
        void refreshDocuments()
      }
    } catch (err: unknown) {
      message.error(getErrorMessage(err, t("common:somethingWentWrong")))
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDeleteDocument = async (documentId: number) => {
    Modal.confirm({
      title: t("playground:documentGenerator.deleteTitle", "Delete document?"),
      content: t(
        "playground:documentGenerator.deleteBody",
        "This will permanently remove the generated document."
      ),
      okText: t("common:delete", "Delete"),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", "Cancel"),
      onOk: async () => {
        try {
          if (!(await ensureTldwClient("handleDeleteDocument", { notify: true }))) return
          await tldwClient.deleteChatDocument(documentId)
          setDocuments((prev) => prev.filter((doc) => doc.id !== documentId))
          message.success(t("common:deleted", "Deleted"))
        } catch (err: unknown) {
          message.error(getErrorMessage(err, t("common:somethingWentWrong")))
        }
      }
    })
  }

  const handleCancelJob = async (jobId: string) => {
    try {
      if (!(await ensureTldwClient("handleCancelJob", { notify: true }))) return
      await tldwClient.cancelChatDocumentJob(jobId)
      setJobs((prev) =>
        prev.map((job) =>
          job.job_id === jobId ? { ...job, status: "cancelled" } : job
        )
      )
      message.success(
        t(
          "playground:documentGenerator.jobCancelled",
          "Job cancelled."
        )
      )
    } catch (err: unknown) {
      message.error(getErrorMessage(err, t("common:somethingWentWrong")))
    }
  }

  const savePromptConfig = async () => {
    try {
      const values = await promptForm.validateFields()
      if (!(await ensureTldwClient("savePromptConfig", { notify: true }))) return
      await tldwClient.saveChatDocumentPrompt({
        document_type: documentType,
        system_prompt: values.system_prompt,
        user_prompt: values.user_prompt,
        temperature: values.temperature,
        max_tokens: values.max_tokens
      })
      message.success(
        t(
          "playground:documentGenerator.promptSaved",
          "Prompt preset saved."
        )
      )
    } catch (err: unknown) {
      if (hasValidationErrorFields(err)) return
      message.error(getErrorMessage(err, t("common:somethingWentWrong")))
    }
  }

  const hasChatDocuments = Boolean(capabilities?.hasChatDocuments)
  const canGenerate = React.useMemo(
    () =>
      Boolean(conversationId) &&
      Boolean(activeModel) &&
      Boolean(activeProvider) &&
      hasChatDocuments,
    [conversationId, activeModel, activeProvider, hasChatDocuments]
  )

  return (
    <Drawer
      open={open}
      onClose={onClose}
      styles={{ wrapper: { width: 480 } }}
      title={t("playground:documentGenerator.title", "Document generator")}
      destroyOnHidden={false}
    >
      {capabilities && !capabilities.hasChatDocuments && (
        <Alert
          variant="warning"
          role="status"
          aria-live="polite"
          data-testid="document-generator-capability-alert"
          className="mb-4"
        >
          {t(
            "playground:documentGenerator.unavailable",
            "Document generation is not available on this server."
          )}
        </Alert>
      )}
      {!conversationId && (
        <Alert
          variant="info"
          role="status"
          aria-live="polite"
          data-testid="document-generator-conversation-alert"
          className="mb-4"
        >
          {t(
            "playground:documentGenerator.noConversation",
            "Start a server-backed chat to generate documents."
          )}
        </Alert>
      )}

      <div className="space-y-4">
        <div className="space-y-1">
          <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
            {t("playground:documentGenerator.source", "Source")}
          </div>
          <div className="text-xs text-text-muted">
            {conversationId
              ? t(
                  "playground:documentGenerator.sourceId",
                  "Conversation {{id}}",
                  { id: conversationId }
                )
              : t(
                  "playground:documentGenerator.sourceMissing",
                  "No active conversation yet."
                )}
          </div>
          {seedMessageId && (
            <div className="text-[10px] text-text-subtle">
              {t(
                "playground:documentGenerator.sourceMessageId",
                "Message {{id}}",
                { id: seedMessageId }
              )}
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div>
            <div className="text-xs font-medium text-text">
              {t("playground:documentGenerator.documentType", "Document type")}
            </div>
            <Select<DocumentType>
              value={documentType}
              onChange={(value) => setFormState({ documentType: value })}
              options={docTypeOptions}
              className="w-full"
            />
          </div>

          <div>
            <div className="text-xs font-medium text-text">
              {t("playground:documentGenerator.model", "Model")}
            </div>
            {modelOptions.length > 0 ? (
              <Select
                value={selectedModel}
                onChange={(value) => setFormState({ selectedModel: value })}
                options={modelOptions}
                className="w-full"
                showSearch
                optionFilterProp="label"
              />
            ) : (
              <div className="grid grid-cols-1 gap-2">
                <Input
                  value={manualModel}
                  onChange={(e) => setFormState({ manualModel: e.target.value })}
                  placeholder={t(
                    "playground:documentGenerator.modelPlaceholder",
                    "Model ID"
                  )}
                />
                <Input
                  value={manualProvider}
                  onChange={(e) => setFormState({ manualProvider: e.target.value })}
                  placeholder={t(
                    "playground:documentGenerator.providerPlaceholder",
                    "Provider name"
                  )}
                />
              </div>
            )}
            {activeProvider && (
              <div className="mt-1 text-[10px] text-text-subtle">
                {t(
                  "playground:documentGenerator.providerLabel",
                  "Provider: {{provider}}",
                  { provider: activeProvider }
                )}
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium text-text">
                {t(
                  "playground:documentGenerator.specificMessage",
                  "Focus on a specific message"
                )}
              </div>
              <Switch
                checked={useSpecificMessage}
                onChange={(value) => setFormState({ useSpecificMessage: value })}
              />
            </div>
            <Input.TextArea
              value={specificMessage}
              onChange={(e) =>
                setFormState({ specificMessage: e.target.value })
              }
              rows={3}
              disabled={!useSpecificMessage}
              placeholder={t(
                "playground:documentGenerator.specificMessagePlaceholder",
                "Optional: paste a message excerpt to focus on."
              )}
              className="mt-2"
            />
          </div>

          <div>
            <div className="text-xs font-medium text-text">
              {t("playground:documentGenerator.customPrompt", "Custom prompt")}
            </div>
            <Input.TextArea
              value={customPrompt}
              onChange={(e) => setFormState({ customPrompt: e.target.value })}
              rows={3}
              placeholder={t(
                "playground:documentGenerator.customPromptPlaceholder",
                "Optional instructions for the generated document."
              )}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-text">
              {t(
                "playground:documentGenerator.asyncGeneration",
                "Generate asynchronously"
              )}
            </div>
            <Switch
              checked={asyncGeneration}
              onChange={(value) => setFormState({ asyncGeneration: value })}
            />
          </div>
        </div>

        {uiMode === "pro" && (
          <div className="rounded-md border border-border bg-surface2 p-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-text">
                  {t(
                    "playground:documentGenerator.promptPresetTitle",
                    "Prompt preset"
                  )}
                </div>
                <div className="text-[10px] text-text-muted">
                  {t(
                    "playground:documentGenerator.promptPresetHint",
                    "Customize the system/user prompts used for this document type."
                  )}
                </div>
              </div>
              <Tooltip
                title={t(
                  "playground:documentGenerator.promptReload",
                  "Reload from server"
                ) as string}
              >
                <Button
                  size="small"
                  onClick={() => void loadPromptConfig({ notify: true })}
                  loading={promptLoading}
                  aria-label={t(
                    "playground:documentGenerator.promptReload",
                    "Reload from server"
                  ) as string}
                  title={t(
                    "playground:documentGenerator.promptReload",
                    "Reload from server"
                  ) as string}
                >
                  <RefreshCw className="h-3 w-3" />
                </Button>
              </Tooltip>
            </div>
            <div className="mt-3">
              <Form form={promptForm} layout="vertical">
                <Form.Item
                  name="system_prompt"
                  label={t(
                    "playground:documentGenerator.systemPrompt",
                    "System prompt"
                  )}
                  rules={[{ required: true }]}
                >
                  <Input.TextArea rows={3} />
                </Form.Item>
                <Form.Item
                  name="user_prompt"
                  label={t(
                    "playground:documentGenerator.userPrompt",
                    "User prompt"
                  )}
                  rules={[{ required: true }]}
                >
                  <Input.TextArea rows={3} />
                </Form.Item>
                <div className="grid grid-cols-2 gap-3">
                  <Form.Item
                    name="temperature"
                    label={t(
                      "playground:documentGenerator.temperature",
                      "Temperature"
                    )}
                  >
                    <InputNumber min={0} max={2} step={0.1} className="w-full" />
                  </Form.Item>
                  <Form.Item
                    name="max_tokens"
                    label={t(
                      "playground:documentGenerator.maxTokens",
                      "Max tokens"
                    )}
                  >
                    <InputNumber min={100} max={10000} step={50} className="w-full" />
                  </Form.Item>
                </div>
                <div className="flex justify-end">
                  <Button
                    onClick={savePromptConfig}
                    type="primary"
                    title={t("common:save", "Save") as string}
                  >
                    {t("common:save", "Save")}
                  </Button>
                </div>
              </Form>
            </div>
          </div>
        )}

        <ModalFooter
          align="left"
          hideCancel
          data-testid="document-generator-drawer-footer"
          actions={[
            {
              label: t("common:generate", "Generate"),
              onClick: () => {
                void handleGenerate()
              },
              loading: isGenerating,
              disabled: !canGenerate,
              title: t("common:generate", "Generate") as string,
              variant: "primary"
            },
            {
              label: t("common:refresh", "Refresh"),
              onClick: () => {
                void refreshDocuments()
              },
              disabled: !conversationId,
              title: t("common:refresh", "Refresh") as string
            }
          ]}
        />

        <div className="space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
            {t(
              "playground:documentGenerator.documentsTitle",
              "Generated documents"
            )}
          </div>
          {docsLoading ? (
            <div className="text-xs text-text-muted">
              {t("playground:documentGenerator.loading", "Loading…")}
            </div>
          ) : documents.length === 0 ? (
            <div className="text-xs text-text-muted">
              {t(
                "playground:documentGenerator.noDocuments",
                "No documents yet."
              )}
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="rounded-md border border-border bg-surface p-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="text-sm font-medium text-text">
                        {doc.title || doc.document_type || "Document"}
                      </div>
                      <div className="text-[10px] text-text-subtle">
                        {doc.document_type && (
                          <span className="mr-2">
                            {t(
                              `playground:documentGenerator.types.${doc.document_type}`,
                              doc.document_type?.replace(/_/g, " ")
                            )}
                          </span>
                        )}
                        {formatTimestamp(doc.created_at)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        size="small"
                        onClick={() => setActiveDoc(doc)}
                        title={t("common:view", "View") as string}
                      >
                        {t("common:view", "View")}
                      </Button>
                      <Button
                        size="small"
                        danger
                        onClick={() => handleDeleteDocument(doc.id)}
                        aria-label={t("common:delete", "Delete") as string}
                        title={t("common:delete", "Delete") as string}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {jobs.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
              {t(
                "playground:documentGenerator.jobsTitle",
                "Generation jobs"
              )}
            </div>
            <div className="space-y-2">
              {jobs.map((job) => (
                <div
                  key={job.job_id}
                  className="rounded-md border border-border bg-surface2 p-2 text-xs"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <Tag>{job.document_type || "doc"}</Tag>
                        <span className="text-text-subtle">
                          {job.status || "pending"}
                        </span>
                      </div>
                      <div className="text-[10px] text-text-subtle">
                        {job.job_id}
                      </div>
                      {job.progress_percentage != null && (
                        <div className="text-[10px] text-text-subtle">
                          {t(
                            "playground:documentGenerator.progress",
                            "{{percent}}% complete",
                            { percent: job.progress_percentage }
                          )}
                        </div>
                      )}
                      {job.error_message && (
                        <div className="text-[10px] text-danger">
                          {job.error_message}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        size="small"
                        onClick={refreshJobs}
                        title={t("common:refresh", "Refresh") as string}
                      >
                        {t("common:refresh", "Refresh")}
                      </Button>
                      {!["completed", "failed", "cancelled"].includes(
                        String(job.status)
                      ) && (
                        <Button
                          size="small"
                          danger
                          onClick={() => handleCancelJob(job.job_id)}
                          title={t("common:cancel", "Cancel") as string}
                        >
                          {t("common:cancel", "Cancel")}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <Modal
        open={Boolean(activeDoc)}
        onCancel={() => setActiveDoc(null)}
        footer={null}
        width={720}
        title={activeDoc?.title || activeDoc?.document_type || "Document"}
      >
        {activeDoc?.content ? (
          <React.Suspense
            fallback={
              <div className="text-sm text-text-muted">
                {t("playground:documentGenerator.loading", "Loading…")}
              </div>
            }
          >
            <Markdown
              message={activeDoc.content}
              className="prose max-w-none dark:prose-invert"
            />
          </React.Suspense>
        ) : (
          <div className="text-sm text-text-muted">
            {t(
              "playground:documentGenerator.noContent",
              "No content available."
            )}
          </div>
        )}
      </Modal>
    </Drawer>
  )
}

export default DocumentGeneratorDrawer
