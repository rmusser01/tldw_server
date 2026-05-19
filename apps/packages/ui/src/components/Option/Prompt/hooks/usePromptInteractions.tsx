import React, { useEffect, useMemo, useRef, useState } from "react"
import { useMutation, useQuery, type QueryClient } from "@tanstack/react-query"
import { notification, Form } from "antd"
import { useNavigate } from "react-router-dom"
import {
  getAllCopilotPrompts,
  upsertCopilotPrompts
} from "@/services/application"
import { useMessageOption } from "@/hooks/useMessageOption"
import {
  hasPromptStudio,
  getPrompt as getStudioPromptById,
  getLlmProviders
} from "@/services/prompt-studio"
import {
  getExecuteDefaultModel,
  getExecuteDefaultProvider,
  normalizeExecuteProvidersCatalog
} from "../Studio/Prompts/execute-playground-provider-utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { filterCopilotPrompts } from "../copilot-prompts-utils"
import type { PromptRowVM } from "../prompt-workspace-types"

type LocalQuickTestPrompt = {
  id: string
  name: string
  systemText?: string
  userText?: string
}

type PromptSegment = "custom" | "copilot" | "studio" | "trash"

const PROMPT_SEGMENTS: PromptSegment[] = ["custom", "copilot", "studio", "trash"]

const normalizePromptSegment = (value: string | undefined): PromptSegment =>
  PROMPT_SEGMENTS.includes(value as PromptSegment)
    ? (value as PromptSegment)
    : "custom"

export interface UsePromptInteractionsDeps {
  queryClient: QueryClient
  isOnline: boolean
  initialSegment?: string
  t: (key: string, opts?: Record<string, any>) => string
  getPromptTexts: (prompt: any) => { systemText: string | undefined; userText: string | undefined }
  getPromptKeywords: (prompt: any) => string[]
  getPromptRecordById: (promptId: string) => any
  getPromptModifiedAt: (prompt: any) => number
  getPromptUsageCount: (prompt: any) => number
  getPromptLastUsedAt: (prompt: any) => number | null
  editorMarkPromptAsUsed: (promptId: string) => Promise<void>
}

export function usePromptInteractions(deps: UsePromptInteractionsDeps) {
  const {
    queryClient,
    isOnline,
    initialSegment = "custom",
    t,
    getPromptTexts,
    getPromptKeywords,
    getPromptRecordById,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    editorMarkPromptAsUsed
  } = deps

  const navigate = useNavigate()
  const { setSelectedQuickPrompt, setSelectedSystemPrompt } = useMessageOption()
  const setStudioActiveSubTab = usePromptStudioStore((s) => s.setActiveSubTab)
  const setStudioSelectedProjectId = usePromptStudioStore((s) => s.setSelectedProjectId)
  const setStudioSelectedPromptId = usePromptStudioStore((s) => s.setSelectedPromptId)
  const setStudioExecutePlaygroundOpen = usePromptStudioStore((s) => s.setExecutePlaygroundOpen)

  // Copilot state
  const [openCopilotEdit, setOpenCopilotEdit] = useState(false)
  const [editCopilotId, setEditCopilotId] = useState("")
  const [editCopilotForm] = Form.useForm()
  const [copilotSearchText, setCopilotSearchText] = useState("")
  const [copilotKeyFilter, setCopilotKeyFilter] = useState<string>("all")

  // Insert prompt state
  const [insertPrompt, setInsertPrompt] = useState<{
    id: string
    systemText?: string
    userText?: string
  } | null>(null)

  // Local quick test state
  const [localQuickTestPrompt, setLocalQuickTestPrompt] =
    useState<LocalQuickTestPrompt | null>(null)
  const [localQuickTestInput, setLocalQuickTestInput] = useState("")
  const [localQuickTestOutput, setLocalQuickTestOutput] = useState<string | null>(null)
  const [isRunningLocalQuickTest, setIsRunningLocalQuickTest] = useState(false)
  const [localQuickTestRunInfo, setLocalQuickTestRunInfo] = useState<{
    provider?: string
    model: string
  } | null>(null)

  // Inspector state
  const [inspectorPromptId, setInspectorPromptId] = useState<string | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)

  // Shortcuts
  const [shortcutsHelpOpen, setShortcutsHelpOpen] = useState(false)

  // Segment state
  const [selectedSegment, setSelectedSegment] = useState<PromptSegment>(
    normalizePromptSegment(initialSegment)
  )
  const previousInitialSegmentRef = useRef(initialSegment)

  // Copilot queries
  const { data: copilotData, status: copilotStatus } = useQuery({
    queryKey: ["fetchCopilotPrompts"],
    queryFn: getAllCopilotPrompts,
    enabled: isOnline
  })

  const { data: hasStudio } = useQuery({
    queryKey: ["prompt-studio", "capability"],
    queryFn: hasPromptStudio,
    enabled: isOnline
  })

  const copilotEditPromptValue = Form.useWatch("prompt", editCopilotForm)

  const copilotPromptIncludesTextPlaceholder =
    typeof copilotEditPromptValue === "string" &&
    copilotEditPromptValue.includes("{text}")

  const copilotPromptKeyOptions = useMemo(() => {
    if (!Array.isArray(copilotData)) return []
    const keys = Array.from(
      new Set(
        copilotData
          .map((item) => (typeof item?.key === "string" ? item.key : ""))
          .filter((key) => key.length > 0)
      )
    )
    return keys.map((key) => ({
      value: key,
      label: t(`common:copilot.${key}`, { defaultValue: key })
    }))
  }, [copilotData, t])

  const filteredCopilotData = useMemo(() => {
    if (!Array.isArray(copilotData)) return []
    return filterCopilotPrompts(copilotData, {
      keyFilter: copilotKeyFilter,
      queryLower: copilotSearchText.trim().toLowerCase(),
      resolveKeyLabel: (key) => t(`common:copilot.${key}`, { defaultValue: key })
    })
  }, [copilotData, copilotKeyFilter, copilotSearchText, t])

  const { mutate: updateCopilotPrompt, isPending: isUpdatingCopilotPrompt } =
    useMutation({
      mutationFn: async (data: any) => {
        return await upsertCopilotPrompts([
          {
            key: data.key,
            prompt: data.prompt
          }
        ])
      },
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: ["fetchCopilotPrompts"]
        })
        setOpenCopilotEdit(false)
        editCopilotForm.resetFields()
        notification.success({
          message: t("managePrompts.notification.updatedSuccess"),
          description: t("managePrompts.notification.updatedSuccessDesc")
        })
      },
      onError: (error) => {
        notification.error({
          message: t("managePrompts.notification.error"),
          description:
            error?.message || t("managePrompts.notification.someError")
        })
      }
    })

  // Inspector
  const inspectorPrompt = useMemo<PromptRowVM | null>(() => {
    if (!inspectorPromptId) return null
    const promptRecord = getPromptRecordById(inspectorPromptId)
    if (!promptRecord) return null
    const { systemText, userText } = getPromptTexts(promptRecord)
    return {
      id: promptRecord.id,
      title:
        promptRecord?.name || promptRecord?.title || t("common:untitled", { defaultValue: "Untitled" }),
      author: promptRecord?.author,
      details: promptRecord?.details,
      previewSystem: systemText || undefined,
      previewUser: userText || undefined,
      keywords: getPromptKeywords(promptRecord) || [],
      favorite: !!promptRecord?.favorite,
      syncStatus: promptRecord?.syncStatus || "local",
      sourceSystem: promptRecord?.sourceSystem || "workspace",
      serverId: promptRecord?.serverId,
      updatedAt: getPromptModifiedAt(promptRecord),
      createdAt:
        typeof promptRecord?.createdAt === "number"
          ? promptRecord.createdAt
          : Date.now(),
      usageCount: getPromptUsageCount(promptRecord),
      lastUsedAt: getPromptLastUsedAt(promptRecord)
    }
  }, [
    inspectorPromptId,
    getPromptRecordById,
    getPromptTexts,
    getPromptKeywords,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    t
  ])

  useEffect(() => {
    if (!inspectorOpen) return
    if (inspectorPrompt) return
    setInspectorOpen(false)
    setInspectorPromptId(null)
  }, [inspectorOpen, inspectorPrompt])

  useEffect(() => {
    if (previousInitialSegmentRef.current === initialSegment) return
    previousInitialSegmentRef.current = initialSegment
    setSelectedSegment(normalizePromptSegment(initialSegment))
  }, [initialSegment])

  const closeInspector = React.useCallback(() => {
    setInspectorOpen(false)
    setInspectorPromptId(null)
  }, [])

  const openPromptInspector = React.useCallback((promptId: string) => {
    setInspectorPromptId(promptId)
    setInspectorOpen(true)
  }, [])

  // Insert handlers
  const handleInsertChoice = React.useCallback(async (choice: "system" | "quick" | "both") => {
    if (!insertPrompt) return
    await editorMarkPromptAsUsed(insertPrompt.id)
    if (choice === "system") {
      setSelectedSystemPrompt(insertPrompt.id)
      setSelectedQuickPrompt(undefined)
      setInsertPrompt(null)
      navigate("/chat")
      return
    }
    if (choice === "both") {
      setSelectedSystemPrompt(insertPrompt.id)
      if (insertPrompt.userText) {
        setSelectedQuickPrompt(insertPrompt.userText)
      }
      setInsertPrompt(null)
      navigate("/chat")
      return
    }
    const quickContent = insertPrompt.userText ?? insertPrompt.systemText
    if (quickContent) {
      setSelectedQuickPrompt(quickContent)
      setSelectedSystemPrompt(undefined)
      setInsertPrompt(null)
      navigate("/chat")
    }
  }, [editorMarkPromptAsUsed, insertPrompt, navigate, setSelectedQuickPrompt, setSelectedSystemPrompt])

  const handleUsePromptInChat = React.useCallback(
    async (record: any) => {
      const { systemText, userText } = getPromptTexts(record)
      const hasSystem =
        typeof systemText === "string" && systemText.trim().length > 0
      const hasUser =
        typeof userText === "string" && userText.trim().length > 0

      if (hasSystem) {
        setInsertPrompt({
          id: record.id,
          systemText,
          userText: hasUser ? userText : undefined
        })
        return
      }

      const quickContent = userText ?? record?.content
      if (quickContent) {
        await editorMarkPromptAsUsed(record.id)
        setSelectedQuickPrompt(quickContent)
        setSelectedSystemPrompt(undefined)
        navigate("/chat")
      }
    },
    [
      getPromptTexts,
      editorMarkPromptAsUsed,
      navigate,
      setSelectedQuickPrompt,
      setSelectedSystemPrompt
    ]
  )

  // Copilot handlers
  const copyCopilotToCustom = React.useCallback(
    (record: { key?: string; prompt?: string }, openCreateDrawer: (vals: any) => void) => {
      const promptText = typeof record?.prompt === "string" ? record.prompt : ""
      const labelKey = typeof record?.key === "string" ? record.key : "custom"
      const label = t(`common:copilot.${labelKey}`, { defaultValue: labelKey })
      const namePrefix = t("managePrompts.copilot.copyToCustom.namePrefix", {
        defaultValue: "Copilot"
      })
      setSelectedSegment("custom")
      openCreateDrawer({
        name: `${namePrefix}: ${label}`,
        user_prompt: promptText
      })
    },
    [t]
  )

  const copyCopilotPromptToClipboard = React.useCallback(
    async (record: { prompt?: string }) => {
      const promptText = typeof record?.prompt === "string" ? record.prompt : ""
      if (!promptText) {
        notification.warning({
          message: t("managePrompts.copilot.clipboard.emptyTitle", {
            defaultValue: "Nothing to copy"
          }),
          description: t("managePrompts.copilot.clipboard.emptyDesc", {
            defaultValue: "This copilot prompt is empty."
          })
        })
        return
      }

      try {
        if (
          typeof navigator === "undefined" ||
          !navigator.clipboard ||
          typeof navigator.clipboard.writeText !== "function"
        ) {
          throw new Error(
            t("managePrompts.copilot.clipboard.notSupported", {
              defaultValue: "Clipboard is not available in this environment."
            })
          )
        }
        await navigator.clipboard.writeText(promptText)
        notification.success({
          message: t("managePrompts.copilot.clipboard.successTitle", {
            defaultValue: "Copied to clipboard"
          }),
          description: t("managePrompts.copilot.clipboard.successDesc", {
            defaultValue: "Copilot prompt text was copied."
          })
        })
      } catch (error: any) {
        notification.error({
          message: t("managePrompts.copilot.clipboard.errorTitle", {
            defaultValue: "Copy failed"
          }),
          description:
            error?.message ||
            t("managePrompts.copilot.clipboard.errorDesc", {
              defaultValue: "Could not copy prompt text to clipboard."
            })
        })
      }
    },
    [t]
  )

  const copyPromptShareLink = React.useCallback(
    async (record: { serverId?: number | null }) => {
      const serverId = record?.serverId
      if (typeof serverId !== "number" || serverId <= 0) {
        notification.warning({
          message: t("managePrompts.share.missingServerIdTitle", {
            defaultValue: "Share link unavailable"
          }),
          description: t("managePrompts.share.missingServerIdDesc", {
            defaultValue:
              "Only prompts synced to the server can generate a share link."
          })
        })
        return
      }
      if (
        typeof navigator === "undefined" ||
        !navigator.clipboard ||
        typeof navigator.clipboard.writeText !== "function"
      ) {
        notification.error({
          message: t("managePrompts.share.copyUnavailableTitle", {
            defaultValue: "Clipboard unavailable"
          }),
          description: t("managePrompts.share.copyUnavailableDesc", {
            defaultValue:
              "Your browser does not allow copying links automatically."
          })
        })
        return
      }
      const url = new URL(window.location.href)
      url.searchParams.set("prompt", String(serverId))
      url.searchParams.set("source", "studio")
      const shareUrl = `${url.origin}${url.pathname}?${url.searchParams.toString()}`
      try {
        await navigator.clipboard.writeText(shareUrl)
        notification.success({
          message: t("managePrompts.share.copySuccessTitle", {
            defaultValue: "Share link copied"
          }),
          description: t("managePrompts.share.copySuccessDesc", {
            defaultValue:
              "Send this link to another user with access to your prompt server."
          })
        })
      } catch {
        notification.error({
          message: t("managePrompts.share.copyFailedTitle", {
            defaultValue: "Could not copy link"
          }),
          description: t("managePrompts.share.copyFailedDesc", {
            defaultValue:
              "Copy failed. Please try again."
          })
        })
      }
    },
    [t]
  )

  // Quick test
  const closeLocalQuickTestModal = React.useCallback(() => {
    setLocalQuickTestPrompt(null)
    setLocalQuickTestInput("")
    setLocalQuickTestOutput(null)
    setLocalQuickTestRunInfo(null)
  }, [])

  const openLocalQuickTestModal = React.useCallback(
    (record: any) => {
      const { systemText, userText } = getPromptTexts(record)
      setLocalQuickTestPrompt({
        id: String(record?.id || ""),
        name: record?.name || record?.title || "Prompt",
        systemText: systemText?.trim() || undefined,
        userText: userText?.trim() || undefined
      })
      setLocalQuickTestInput("")
      setLocalQuickTestOutput(null)
      setLocalQuickTestRunInfo(null)
    },
    [getPromptTexts]
  )

  const openStudioQuickTest = React.useCallback(
    async (record: any) => {
      const serverPromptId = Number(record?.serverId)
      if (!Number.isInteger(serverPromptId) || serverPromptId <= 0) {
        openLocalQuickTestModal(record)
        return
      }

      if (!isOnline || hasStudio === false) {
        notification.warning({
          message: t("managePrompts.quickTest.studioUnavailableTitle", {
            defaultValue: "Studio quick test unavailable"
          }),
          description: t("managePrompts.quickTest.studioUnavailableDesc", {
            defaultValue:
              "Prompt Studio is unavailable right now, so local quick test is opening instead."
          })
        })
        openLocalQuickTestModal(record)
        return
      }

      let projectId =
        typeof record?.studioProjectId === "number" && record.studioProjectId > 0
          ? record.studioProjectId
          : null

      if (!projectId) {
        try {
          const response = await getStudioPromptById(serverPromptId)
          const serverPrompt = (response as any)?.data?.data
          if (
            typeof serverPrompt?.project_id === "number" &&
            serverPrompt.project_id > 0
          ) {
            projectId = serverPrompt.project_id
          }
        } catch {
          projectId = null
        }
      }

      if (!projectId) {
        notification.warning({
          message: t("managePrompts.quickTest.missingProjectTitle", {
            defaultValue: "Quick test unavailable"
          }),
          description: t("managePrompts.quickTest.missingProjectDesc", {
            defaultValue:
              "This prompt is synced, but its Prompt Studio project could not be resolved."
          })
        })
        return
      }

      setStudioSelectedProjectId(projectId)
      setStudioSelectedPromptId(serverPromptId)
      setStudioActiveSubTab("prompts")
      setStudioExecutePlaygroundOpen(true)
      setSelectedSegment("studio")
    },
    [
      hasStudio,
      isOnline,
      openLocalQuickTestModal,
      setStudioActiveSubTab,
      setStudioExecutePlaygroundOpen,
      setStudioSelectedProjectId,
      setStudioSelectedPromptId,
      t
    ]
  )

  const handleQuickTest = React.useCallback(
    async (record: any) => {
      const hasServerPromptId =
        typeof record?.serverId === "number" && record.serverId > 0
      if (hasServerPromptId) {
        await openStudioQuickTest(record)
        return
      }
      openLocalQuickTestModal(record)
    },
    [openLocalQuickTestModal, openStudioQuickTest]
  )

  const runLocalQuickTest = React.useCallback(async () => {
    if (!localQuickTestPrompt || isRunningLocalQuickTest) return

    const userTemplate = localQuickTestPrompt.userText || ""
    const systemTemplate = localQuickTestPrompt.systemText || ""
    const normalizedInput = localQuickTestInput.trim()
    const textTemplateRegex = /\{\{\s*text\s*\}\}/gi
    const hasTextTemplateVar = textTemplateRegex.test(userTemplate)

    if (hasTextTemplateVar && normalizedInput.length === 0) {
      notification.warning({
        message: t("managePrompts.quickTest.inputRequiredTitle", {
          defaultValue: "Input required"
        }),
        description: t("managePrompts.quickTest.inputRequiredDesc", {
          defaultValue:
            "This prompt uses {{text}}. Enter sample input before running quick test."
        })
      })
      return
    }

    let userMessage = userTemplate
    if (hasTextTemplateVar) {
      userMessage = userTemplate.replace(/\{\{\s*text\s*\}\}/gi, normalizedInput)
    } else if (userTemplate && normalizedInput) {
      userMessage = `${userTemplate}\n\n${normalizedInput}`
    } else if (!userTemplate) {
      userMessage = normalizedInput
    }

    if (!userMessage.trim()) {
      notification.warning({
        message: t("managePrompts.quickTest.noMessageTitle", {
          defaultValue: "Nothing to test"
        }),
        description: t("managePrompts.quickTest.noMessageDesc", {
          defaultValue:
            "Add prompt content or input text before running quick test."
        })
      })
      return
    }

    setIsRunningLocalQuickTest(true)
    setLocalQuickTestOutput(null)
    setLocalQuickTestRunInfo(null)

    let provider: string | undefined
    let model = "gpt-4o-mini"

    try {
      const llmProvidersResponse = await getLlmProviders()
      const providersPayload =
        (llmProvidersResponse as any)?.data ?? llmProvidersResponse
      const providersCatalog = normalizeExecuteProvidersCatalog(providersPayload)
      const resolvedProvider = getExecuteDefaultProvider(providersCatalog) || undefined
      const resolvedModel =
        getExecuteDefaultModel(providersCatalog, resolvedProvider) || null
      provider = resolvedProvider
      if (resolvedModel) {
        model = resolvedModel
      }
    } catch {
      // Keep fallback defaults when provider lookup fails.
    }

    try {
      await tldwClient.initialize().catch(() => null)
      const messages: Array<{ role: "system" | "user"; content: string }> = []

      if (systemTemplate.trim()) {
        messages.push({
          role: "system",
          content: systemTemplate.trim()
        })
      }

      messages.push({
        role: "user",
        content: userMessage
      })

      const completionResponse = await tldwClient.createChatCompletion({
        model,
        api_provider: provider,
        messages,
        temperature: 0.2
      })
      const completionPayload = await completionResponse.json()
      const outputCandidate =
        completionPayload?.choices?.[0]?.message?.content ??
        completionPayload?.output ??
        completionPayload?.content
      const output =
        typeof outputCandidate === "string" ? outputCandidate.trim() : ""

      if (!output) {
        throw new Error(
          t("managePrompts.quickTest.emptyResultError", {
            defaultValue: "The model returned an empty result."
          })
        )
      }

      setLocalQuickTestOutput(output)
      setLocalQuickTestRunInfo({ provider, model })
    } catch (error: any) {
      notification.error({
        message: t("managePrompts.quickTest.runFailedTitle", {
          defaultValue: "Quick test failed"
        }),
        description:
          error?.message ||
          t("managePrompts.quickTest.runFailedDesc", {
            defaultValue:
              "The quick test request could not be completed."
          })
      })
    } finally {
      setIsRunningLocalQuickTest(false)
    }
  }, [isRunningLocalQuickTest, localQuickTestInput, localQuickTestPrompt, t])

  return {
    // Copilot
    openCopilotEdit,
    setOpenCopilotEdit,
    editCopilotId,
    setEditCopilotId,
    editCopilotForm,
    copilotSearchText,
    setCopilotSearchText,
    copilotKeyFilter,
    setCopilotKeyFilter,
    copilotData,
    copilotStatus,
    copilotEditPromptValue,
    copilotPromptIncludesTextPlaceholder,
    copilotPromptKeyOptions,
    filteredCopilotData,
    updateCopilotPrompt,
    isUpdatingCopilotPrompt,
    copyCopilotToCustom,
    copyCopilotPromptToClipboard,
    copyPromptShareLink,
    // Insert
    insertPrompt,
    setInsertPrompt,
    handleInsertChoice,
    handleUsePromptInChat,
    // Quick test
    localQuickTestPrompt,
    localQuickTestInput,
    setLocalQuickTestInput,
    localQuickTestOutput,
    isRunningLocalQuickTest,
    localQuickTestRunInfo,
    closeLocalQuickTestModal,
    handleQuickTest,
    runLocalQuickTest,
    // Inspector
    inspectorPromptId,
    inspectorOpen,
    inspectorPrompt,
    closeInspector,
    openPromptInspector,
    // Shortcuts
    shortcutsHelpOpen,
    setShortcutsHelpOpen,
    // Segment
    selectedSegment,
    setSelectedSegment,
    // Studio
    hasStudio,
    setStudioActiveSubTab,
    setStudioSelectedProjectId,
    setStudioSelectedPromptId,
    setStudioExecutePlaygroundOpen
  }
}
