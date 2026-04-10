import React, { Suspense, useState, useEffect, useRef } from "react"
import { useTranslation } from "react-i18next"
import {
  Headphones,
  FileText,
  GitBranch,
  FileSpreadsheet,
  Layers,
  HelpCircle,
  Scale,
  Calendar,
  Presentation,
  Table as TableIconLucide,
  Loader2,
  CheckCircle,
  XCircle,
  Eye,
  Download,
  RefreshCw,
  Square,
  MessageCircle,
  StickyNote,
  Pencil,
  Trash2,
  ChevronDown,
  ChevronUp,
  Settings2,
  PanelRightClose
} from "lucide-react"
import {
  Button,
  Empty,
  Tooltip,
  Input,
  Modal,
  message,
  Slider,
  Select,
  Dropdown,
  Switch
} from "antd"
import { useMobile } from "@/hooks/useMediaQuery"
import { useWorkspaceStore } from "@/store/workspace"
import type { AudioTtsProvider } from "@/types/workspace"
import { OUTPUT_TYPES } from "@/types/workspace"
import type {
  ArtifactType,
  GeneratedArtifact
} from "@/types/workspace"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { getWorkspaceStudioNoSourcesHint } from "../source-location-copy"
import {
  useArtifactGeneration,
  useAudioTtsSettings,
  useQuizParsing,
  useArtifactExport,
  useStudioDerivedState,
} from "./hooks"
import {
  estimateGenerationSeconds,
  encodeSlidesVisualStyleValue,
} from "./hooks/useArtifactGeneration"
import {
  TTS_PROVIDERS,
  AUDIO_FORMATS,
} from "./hooks/useAudioTtsSettings"
import {
  getResponsiveArtifactModalProps,
  SLIDES_EXPORT_FORMATS,
} from "./hooks/useArtifactExport"

// Re-export for external consumers
export { estimateGenerationSeconds, estimateGenerationTokens, estimateGenerationCostUsd } from "./hooks/useArtifactGeneration"

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Icon mapping for artifact types
const ARTIFACT_TYPE_ICONS: Record<ArtifactType, React.ElementType> = {
  audio_overview: Headphones,
  summary: FileText,
  mindmap: GitBranch,
  report: FileSpreadsheet,
  compare_sources: Scale,
  flashcards: Layers,
  quiz: HelpCircle,
  timeline: Calendar,
  slides: Presentation,
  data_table: TableIconLucide
}

// Output type button configuration
const OUTPUT_BUTTONS: {
  type: ArtifactType
  label: string
  description: string
  icon: React.ElementType
}[] = [
  {
    type: "audio_overview",
    label: "Audio Summary",
    description:
      OUTPUT_TYPES.find((config) => config.type === "audio_overview")
        ?.description || "Generate a spoken summary of your sources",
    icon: Headphones
  },
  {
    type: "summary",
    label: "Summary",
    description:
      OUTPUT_TYPES.find((config) => config.type === "summary")?.description ||
      "Create a concise summary of key points",
    icon: FileText
  },
  {
    type: "mindmap",
    label: "Mind Map",
    description:
      OUTPUT_TYPES.find((config) => config.type === "mindmap")?.description ||
      "Visualize concepts and relationships",
    icon: GitBranch
  },
  {
    type: "report",
    label: "Report",
    description:
      OUTPUT_TYPES.find((config) => config.type === "report")?.description ||
      "Generate a detailed report document",
    icon: FileSpreadsheet
  },
  {
    type: "compare_sources",
    label: "Compare Sources",
    description:
      OUTPUT_TYPES.find((config) => config.type === "compare_sources")
        ?.description ||
      "Compare claims, evidence, and disagreements across selected sources",
    icon: Scale
  },
  {
    type: "flashcards",
    label: "Flashcards",
    description:
      OUTPUT_TYPES.find((config) => config.type === "flashcards")?.description ||
      "Create study flashcards for review",
    icon: Layers
  },
  {
    type: "quiz",
    label: "Quiz",
    description:
      OUTPUT_TYPES.find((config) => config.type === "quiz")?.description ||
      "Generate a quiz to test understanding",
    icon: HelpCircle
  },
  {
    type: "timeline",
    label: "Timeline",
    description:
      OUTPUT_TYPES.find((config) => config.type === "timeline")?.description ||
      "Create a chronological timeline",
    icon: Calendar
  },
  {
    type: "slides",
    label: "Slides",
    description:
      OUTPUT_TYPES.find((config) => config.type === "slides")?.description ||
      "Generate presentation slides",
    icon: Presentation
  },
  {
    type: "data_table",
    label: "Data Table",
    description:
      OUTPUT_TYPES.find((config) => config.type === "data_table")
        ?.description || "Extract structured data into a table",
    icon: TableIconLucide
  }
]

const OUTPUT_GROUPS: Array<{
  id: string
  label: string
  types: ArtifactType[]
}> = [
  {
    id: "study-aids",
    label: "Study Aids",
    types: ["quiz", "flashcards"]
  },
  {
    id: "analysis",
    label: "Analysis",
    types: ["summary", "report", "compare_sources", "timeline", "data_table"]
  },
  {
    id: "creative",
    label: "Creative",
    types: ["mindmap", "slides", "audio_overview"]
  }
]

// Primary output types shown by default; remaining are collapsed behind an expander
const PRIMARY_OUTPUT_TYPES = new Set<ArtifactType>(["summary", "flashcards", "quiz", "report"])

// Status icons for artifacts
const STATUS_ICONS: Record<
  GeneratedArtifact["status"],
  { icon: React.ElementType; className: string }
> = {
  pending: { icon: Loader2, className: "text-text-muted animate-spin" },
  generating: { icon: Loader2, className: "text-primary animate-spin" },
  completed: { icon: CheckCircle, className: "text-success" },
  failed: { icon: XCircle, className: "text-error" }
}

import {
  STUDIO_DEFAULT_RAG_TOP_K,
  STUDIO_DEFAULT_RAG_MIN_SCORE,
  STUDIO_DEFAULT_ENABLE_RERANKING,
  STUDIO_DEFAULT_MAX_TOKENS,
  STUDIO_DEFAULT_SUMMARY_INSTRUCTION,
  OUTPUT_VIRTUALIZATION_THRESHOLD,
  OUTPUT_VIRTUAL_ROW_HEIGHT,
  OUTPUT_VIRTUAL_OVERSCAN
} from "./hooks/useStudioDerivedState"

const MindMapArtifactViewer = React.lazy(() =>
  import("./ArtifactModalContent").then((module) => ({
    default: module.MindMapArtifactViewer
  }))
)

const DataTableArtifactViewer = React.lazy(() =>
  import("./ArtifactModalContent").then((module) => ({
    default: module.DataTableArtifactViewer
  }))
)

const FlashcardArtifactEditor = React.lazy(() =>
  import("./ArtifactModalContent").then((module) => ({
    default: module.FlashcardArtifactEditor
  }))
)

const QuizArtifactEditor = React.lazy(() =>
  import("./ArtifactModalContent").then((module) => ({
    default: module.QuizArtifactEditor
  }))
)

const QuickNotesSection = React.lazy(() =>
  import("./QuickNotesSection").then((module) => ({
    default: module.QuickNotesSection
  }))
)

const renderArtifactModalContent = (node: React.ReactNode) => (
  <Suspense
    fallback={
      <div className="flex min-h-[200px] items-center justify-center text-sm text-text-muted">
        Loading output viewer...
      </div>
    }
  >
    {node}
  </Suspense>
)

type BrowserSpeechArtifactViewerProps = {
  content: string
  playbackRate?: number
}

const BrowserSpeechArtifactViewer: React.FC<BrowserSpeechArtifactViewerProps> = ({
  content,
  playbackRate = 1
}) => {
  const { t } = useTranslation(["playground", "common"])
  const [speechState, setSpeechState] = useState<
    "idle" | "speaking" | "paused" | "unavailable"
  >(
    typeof window === "undefined" || !("speechSynthesis" in window)
      ? "unavailable"
      : "idle"
  )
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)

  useEffect(() => {
    return () => {
      if (
        typeof window !== "undefined" &&
        "speechSynthesis" in window &&
        utteranceRef.current
      ) {
        window.speechSynthesis.cancel()
      }
      utteranceRef.current = null
    }
  }, [])

  const handlePlay = () => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setSpeechState("unavailable")
      return
    }

    const synthesis = window.speechSynthesis
    if (speechState === "paused" && synthesis.paused) {
      synthesis.resume()
      setSpeechState("speaking")
      return
    }

    synthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(content)
    utterance.rate = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1
    utterance.onstart = () => setSpeechState("speaking")
    utterance.onpause = () => setSpeechState("paused")
    utterance.onresume = () => setSpeechState("speaking")
    utterance.onend = () => setSpeechState("idle")
    utterance.onerror = () => setSpeechState("idle")
    utteranceRef.current = utterance
    synthesis.speak(utterance)
    setSpeechState("speaking")
  }

  const handlePauseToggle = () => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setSpeechState("unavailable")
      return
    }

    const synthesis = window.speechSynthesis
    if (synthesis.paused) {
      synthesis.resume()
      setSpeechState("speaking")
      return
    }

    if (synthesis.speaking) {
      synthesis.pause()
      setSpeechState("paused")
    }
  }

  const handleStop = () => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setSpeechState("unavailable")
      return
    }

    window.speechSynthesis.cancel()
    utteranceRef.current = null
    setSpeechState("idle")
  }

  const statusText =
    speechState === "speaking"
      ? t("playground:studio.browserAudioSpeaking", "Speaking in your browser.")
      : speechState === "paused"
        ? t("playground:studio.browserAudioPaused", "Browser speech is paused.")
        : speechState === "unavailable"
          ? t(
              "playground:studio.browserAudioUnavailable",
              "Browser speech playback is unavailable in this environment."
            )
          : t(
              "playground:studio.browserAudioReady",
              "Use your browser to play this audio summary."
            )

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button type="primary" onClick={handlePlay}>
          {speechState === "paused"
            ? t("common:resume", "Resume")
            : t("common:play", "Play")}
        </Button>
        <Button
          onClick={handlePauseToggle}
          disabled={speechState !== "speaking" && speechState !== "paused"}
        >
          {speechState === "paused"
            ? t("common:resume", "Resume")
            : t("common:pause", "Pause")}
        </Button>
        <Button onClick={handleStop} disabled={speechState === "idle"}>
          {t("common:stop", "Stop")}
        </Button>
      </div>
      <p className="text-sm text-text-muted">{statusText}</p>
      <div className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded bg-surface2 p-3 text-sm">
        {content}
      </div>
    </div>
  )
}

const renderQuickNotesSection = (onCollapse: () => void) => (
  <Suspense
    fallback={
      <div className="flex min-h-[220px] flex-1 items-center justify-center border-t border-border px-4 py-3 text-sm text-text-muted">
        Loading notes...
      </div>
    }
  >
    <QuickNotesSection onCollapse={onCollapse} />
  </Suspense>
)

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

interface StudioPaneProps {
  /** Callback to hide/collapse the pane */
  onHide?: () => void
}

/**
 * StudioPane - Right pane for generating outputs
 */
export const StudioPane: React.FC<StudioPaneProps> = ({ onHide }) => {
  const { t } = useTranslation(["playground", "common"])
  const isMobile = useMobile()
  const [messageApi, contextHolder] = message.useMessage()

  // Store state
  const selectedSourceIds = useWorkspaceStore((s) => s.selectedSourceIds)
  const selectedSourceFolderIds = useWorkspaceStore(
    (s) => s.selectedSourceFolderIds
  ) || []
  const getSelectedMediaIds = useWorkspaceStore((s) => s.getSelectedMediaIds)
  const getEffectiveSelectedMediaIds = useWorkspaceStore(
    (s) => s.getEffectiveSelectedMediaIds
  )
  const getEffectiveSelectedSources = useWorkspaceStore(
    (s) => s.getEffectiveSelectedSources
  )
  const generatedArtifacts = useWorkspaceStore((s) => s.generatedArtifacts)
  const isGeneratingOutput = useWorkspaceStore((s) => s.isGeneratingOutput)
  const generatingOutputType = useWorkspaceStore((s) => s.generatingOutputType)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const workspaceName = useWorkspaceStore((s) => s.workspaceName)
  const workspaceTag = useWorkspaceStore((s) => s.workspaceTag)
  const studyMaterialsPolicy = useWorkspaceStore((s) => s.studyMaterialsPolicy)
  const audioSettings = useWorkspaceStore((s) => s.audioSettings)
  const noteFocusTarget = useWorkspaceStore((s) => s.noteFocusTarget)

  // Store actions
  const addArtifact = useWorkspaceStore((s) => s.addArtifact)
  const updateArtifactStatus = useWorkspaceStore((s) => s.updateArtifactStatus)
  const removeArtifact = useWorkspaceStore((s) => s.removeArtifact)
  const restoreArtifact = useWorkspaceStore((s) => s.restoreArtifact)
  const setIsGeneratingOutput = useWorkspaceStore((s) => s.setIsGeneratingOutput)
  const setAudioSettings = useWorkspaceStore((s) => s.setAudioSettings)
  const captureToCurrentNote = useWorkspaceStore((s) => s.captureToCurrentNote)

  // Workspace chat/store options
  const selectedModel = useStoreMessageOption((s) => s.selectedModel)
  const setSelectedModel = useStoreMessageOption((s) => s.setSelectedModel)
  const ragSearchMode = useStoreMessageOption((s) => s.ragSearchMode)
  const setRagSearchMode = useStoreMessageOption((s) => s.setRagSearchMode)
  const ragTopK = useStoreMessageOption((s) => s.ragTopK)
  const setRagTopK = useStoreMessageOption((s) => s.setRagTopK)
  const ragEnableGeneration = useStoreMessageOption((s) => s.ragEnableGeneration)
  const setRagEnableGeneration = useStoreMessageOption((s) => s.setRagEnableGeneration)
  const ragEnableCitations = useStoreMessageOption((s) => s.ragEnableCitations)
  const setRagEnableCitations = useStoreMessageOption((s) => s.setRagEnableCitations)
  const ragAdvancedOptions = useStoreMessageOption((s) => s.ragAdvancedOptions)
  const setRagAdvancedOptions = useStoreMessageOption((s) => s.setRagAdvancedOptions)

  const apiProvider = useStoreChatModelSettings((s) => s.apiProvider)
  const temperature = useStoreChatModelSettings((s) => s.temperature)
  const topP = useStoreChatModelSettings((s) => s.topP)
  const numPredict = useStoreChatModelSettings((s) => s.numPredict)
  const setApiProvider = useStoreChatModelSettings((s) => s.setApiProvider)
  const setTemperature = useStoreChatModelSettings((s) => s.setTemperature)
  const setTopP = useStoreChatModelSettings((s) => s.setTopP)
  const setNumPredict = useStoreChatModelSettings((s) => s.setNumPredict)
  const updateModelSetting = useStoreChatModelSettings((s) => s.updateSetting)

  // Local UI state
  const [slidesVisualStyleValue, setSlidesVisualStyleValue] = useState("")
  const [selectedFlashcardDeck, setSelectedFlashcardDeck] = useState<"auto" | number>("auto")
  const [activeOutputType, setActiveOutputType] = useState<ArtifactType | null>(null)
  const [moreOutputsExpanded, setMoreOutputsExpanded] = useState(false)
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)

  // Collapsible sections
  const [studioOptionsExpanded, setStudioOptionsExpanded] = useState(false)
  const [studioExpanded, setStudioExpanded] = useState(true)
  const [outputsExpanded, setOutputsExpanded] = useState(true)
  const [notesExpanded, setNotesExpanded] = useState(false)
  const outputListContainerRef = useRef<HTMLDivElement | null>(null)
  const [outputListScrollTop, setOutputListScrollTop] = useState(0)
  const [outputListViewportHeight, setOutputListViewportHeight] = useState(320)

  // ── Derived values ──

  const selectedMediaIds = React.useMemo(
    () =>
      typeof getEffectiveSelectedMediaIds === "function"
        ? getEffectiveSelectedMediaIds()
        : getSelectedMediaIds(),
    [
      getEffectiveSelectedMediaIds,
      getSelectedMediaIds,
      selectedSourceFolderIds,
      selectedSourceIds
    ]
  )
  const selectedSources = React.useMemo(
    () =>
      typeof getEffectiveSelectedSources === "function"
        ? getEffectiveSelectedSources().filter((source) =>
            selectedMediaIds.includes(source.mediaId)
          )
        : [],
    [
      getEffectiveSelectedSources,
      selectedMediaIds,
      selectedSourceFolderIds,
      selectedSourceIds
    ]
  )
  const hasSelectedSources = selectedMediaIds.length > 0
  const selectedMediaCount = selectedMediaIds.length
  const normalizedApiProvider =
    typeof apiProvider === "string" && apiProvider.trim().length > 0
      ? apiProvider.trim().toLowerCase()
      : "__auto__"

  const studioDerived = useStudioDerivedState({
    ragTopK,
    ragAdvancedOptions,
    temperature,
    topP,
    numPredict,
    setRagTopK,
    setRagAdvancedOptions,
    setTemperature,
    setTopP,
    generatedArtifacts,
    outputListScrollTop,
    outputListViewportHeight
  })
  const {
    normalizedRagAdvancedOptions,
    resolvedSummaryInstruction,
    resolvedStudioTopK,
    studioSimilarityThreshold,
    studioRerankingEnabled,
    resolvedTemperature,
    resolvedTopP,
    resolvedNumPredict,
    patchRagAdvancedOptions,
    handleStudioTopKChange,
    handleStudioSimilarityThresholdChange,
    handleStudioTemperatureChange,
    handleStudioTopPChange,
    useVirtualizedOutputs,
    visibleArtifacts,
    virtualOutputTopPadding,
    virtualOutputBottomPadding
  } = studioDerived

  const contextualAudioSettingsVisible =
    activeOutputType === "audio_overview" ||
    generatingOutputType === "audio_overview"
  const studioControlSize = isMobile ? "large" : "small"
  const summaryUsesDirectSourceGeneration =
    activeOutputType === "summary" || generatingOutputType === "summary"
  const mobileSliderClassName = isMobile
    ? "[&_.ant-slider-rail]:!h-2 [&_.ant-slider-track]:!h-2 [&_.ant-slider-handle]:!h-5 [&_.ant-slider-handle]:!w-5"
    : undefined

  // ── Hooks ──

  const artifactGeneration = useArtifactGeneration({
    messageApi,
    selectedMediaIds,
    selectedSources,
    selectedMediaCount,
    hasSelectedSources,
    audioSettings,
    workspaceTag,
    outputButtons: OUTPUT_BUTTONS,
    generatedArtifacts,
    isGeneratingOutput,
    generatingOutputType,
    addArtifact,
    updateArtifactStatus,
    setIsGeneratingOutput,
    selectedModel,
    normalizedApiProvider,
    resolvedTemperature,
    resolvedTopP,
    resolvedNumPredict,
    resolvedSummaryInstruction,
    slidesVisualStyleValue,
    selectedFlashcardDeck,
    workspaceId,
    workspaceName,
    studyMaterialsPolicy,
    ragAdvancedOptions: normalizedRagAdvancedOptions,
    t,
  })

  const {
    generationPhase,
    chatModels: _chatModels,
    loadingChatModels,
    recentOutputTypes: _recentOutputTypes,
    slidesVisualStyles: _slidesVisualStyles,
    slidesVisualStylesLoading,
    slidesVisualStyleValueLocal: _slidesVisualStyleValueLocal,
    setSlidesVisualStyleValueLocal: _setSlidesVisualStyleValueLocal,
    availableDecks,
    loadingDecks,
    providerOptions,
    modelOptions,
    selectedSlidesVisualStyle,
    groupedSlidesVisualStyles,
    etaSeconds,
    cumulativeUsage,
    handleGenerateOutput,
    handleCancelGeneration,
    loadFlashcardDecks,
  } = artifactGeneration

  // Sync local slides style value with hook's local default when empty
  useEffect(() => {
    if (!slidesVisualStyleValue && _slidesVisualStyleValueLocal) {
      setSlidesVisualStyleValue(_slidesVisualStyleValueLocal)
    }
  }, [slidesVisualStyleValue, _slidesVisualStyleValueLocal])

  // Reset flashcard deck selection if the deck no longer exists
  useEffect(() => {
    if (
      selectedFlashcardDeck !== "auto" &&
      availableDecks.length > 0 &&
      !availableDecks.some((deck) => deck.id === selectedFlashcardDeck)
    ) {
      setSelectedFlashcardDeck("auto")
    }
  }, [availableDecks, selectedFlashcardDeck])

  // Elapsed-time counter while generating an artifact
  useEffect(() => {
    if (!isGeneratingOutput) {
      setGenerationElapsedSeconds(0)
      return
    }
    setGenerationElapsedSeconds(0)
    const id = setInterval(() => {
      setGenerationElapsedSeconds((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(id)
  }, [isGeneratingOutput])

  const audioTts = useAudioTtsSettings({
    audioSettings,
    setAudioSettings,
    messageApi,
    t,
  })

  const {
    showTtsSettings,
    setShowTtsSettings,
    loadingVoices,
    previewingVoice,
    getVoiceOptions,
    getModelOptions: getTtsModelOptions,
    handlePreviewVoice,
  } = audioTts

  const showAudioSettingsPanel = showTtsSettings || contextualAudioSettingsVisible

  const quizParsing = useQuizParsing()

  const {
    getArtifactFlashcards,
    formatFlashcardsContent,
    getArtifactQuizQuestions,
    formatQuizQuestionsContent: formatQuizContent,
  } = quizParsing

  const artifactExport = useArtifactExport({
    messageApi,
    isMobile,
    generatedArtifacts,
    removeArtifact,
    restoreArtifact,
    captureToCurrentNote,
    t,
  })

  const {
    handleDeleteArtifact,
    handleDiscussArtifact,
    handleSaveArtifactToNotes,
    handleDownloadArtifact,
    handleSlidesDownload,
    handleIconButtonKeyDown,
  } = artifactExport

  // ── Side effects ──

  useEffect(() => {
    if (!noteFocusTarget) return
    if (!notesExpanded) {
      setNotesExpanded(true)
    }
  }, [noteFocusTarget, notesExpanded])

  useEffect(() => {
    const container = outputListContainerRef.current
    if (!container) return

    const syncViewportHeight = () => {
      setOutputListViewportHeight(container.clientHeight || 320)
    }

    syncViewportHeight()

    if (typeof ResizeObserver === "undefined") {
      return
    }

    const observer = new ResizeObserver(() => {
      syncViewportHeight()
    })
    observer.observe(container)

    return () => {
      observer.disconnect()
    }
  }, [outputsExpanded])

  useEffect(() => {
    if (!useVirtualizedOutputs) {
      setOutputListScrollTop(0)
      return
    }

    const container = outputListContainerRef.current
    if (!container) return

    const maxScrollTop = Math.max(
      0,
      generatedArtifacts.length * OUTPUT_VIRTUAL_ROW_HEIGHT - outputListViewportHeight
    )

    if (container.scrollTop > maxScrollTop) {
      container.scrollTop = maxScrollTop
      setOutputListScrollTop(maxScrollTop)
    }
  }, [generatedArtifacts.length, outputListViewportHeight, useVirtualizedOutputs])

  // ── View artifact handler ──

  const handleViewArtifact = (artifact: GeneratedArtifact) => {
    const responsiveModalProps = (desktopWidth: number) =>
      getResponsiveArtifactModalProps(isMobile, desktopWidth)

    if (artifact.type === "audio_overview" && artifact.audioUrl) {
      Modal.info({
        title: artifact.title,
        content: (
          <div className="flex flex-col gap-4">
            <audio
              controls
              className="w-full"
              src={artifact.audioUrl}
            >
              Your browser does not support the audio element.
            </audio>
            {artifact.content && (
              <details className="mt-2">
                <summary className="cursor-pointer text-sm text-text-muted">
                  View Script
                </summary>
                <div className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap rounded bg-surface2 p-2 text-sm">
                  {artifact.content}
                </div>
              </details>
            )}
          </div>
        ),
        ...responsiveModalProps(500)
      })
      return
    }

    if (
      artifact.type === "audio_overview" &&
      artifact.audioFormat === "browser" &&
      artifact.content
    ) {
      Modal.info({
        title: artifact.title,
        content: (
          <BrowserSpeechArtifactViewer
            content={artifact.content}
            playbackRate={audioSettings.speed}
          />
        ),
        ...responsiveModalProps(560)
      })
      return
    }

    if (artifact.type === "mindmap" && artifact.content) {
      Modal.info({
        title: artifact.title,
        content: renderArtifactModalContent(
          <MindMapArtifactViewer title={artifact.title} content={artifact.content} />
        ),
        ...responsiveModalProps(960),
        footer: null,
        icon: null
      })
      return
    }

    if (artifact.type === "data_table" && artifact.content) {
      Modal.info({
        title: artifact.title,
        content: renderArtifactModalContent(
          <DataTableArtifactViewer title={artifact.title} content={artifact.content} />
        ),
        ...responsiveModalProps(980),
        footer: null,
        icon: null
      })
      return
    }

    if (artifact.type === "flashcards") {
      const initialCards = getArtifactFlashcards(artifact)
      const modal = Modal.info({
        title: artifact.title,
        content: renderArtifactModalContent(
          <FlashcardArtifactEditor
            cards={initialCards}
            onSave={(cards) => {
              const nextContent = formatFlashcardsContent(cards)
              updateArtifactStatus(artifact.id, artifact.status, {
                content: nextContent,
                data: {
                  ...(artifact.data || {}),
                  flashcards: cards
                }
              })
              messageApi.success(
                t(
                  "playground:studio.flashcardsSaved",
                  "Flashcards updated"
                )
              )
              modal.destroy()
            }}
          />
        ),
        ...responsiveModalProps(820),
        footer: null,
        icon: null
      })
      return
    }

    if (artifact.type === "quiz") {
      const initialQuestions = getArtifactQuizQuestions(artifact)
      const modal = Modal.info({
        title: artifact.title,
        content: renderArtifactModalContent(
          <QuizArtifactEditor
            questions={initialQuestions}
            onSave={(questions) => {
              const nextContent = formatQuizContent(
                questions,
                artifact.title
              )
              updateArtifactStatus(artifact.id, artifact.status, {
                content: nextContent,
                data: {
                  ...(artifact.data || {}),
                  questions
                }
              })
              messageApi.success(
                t("playground:studio.quizSaved", "Quiz updated")
              )
              modal.destroy()
            }}
          />
        ),
        ...responsiveModalProps(860),
        footer: null,
        icon: null
      })
      return
    }

    if (artifact.content) {
      Modal.info({
        title: artifact.title,
        content: (
          <div className="max-h-96 overflow-y-auto whitespace-pre-wrap">
            {artifact.content}
          </div>
        ),
        ...responsiveModalProps(600)
      })
    }
  }

  // ── Render ──

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      {contextHolder}

      {/* Header */}
      <div className="flex items-start justify-between border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-text">
            {t("playground:studio.title", "Studio")}
          </h2>
          <p className="mt-0.5 text-xs text-text-muted">
            {t("playground:studio.subtitle", "Generate outputs from your sources")}
          </p>
        </div>
        {onHide && (
          <Tooltip title={t("playground:workspace.hideStudio", "Hide studio")}>
            <button
              type="button"
              onClick={onHide}
              className="hidden h-9 w-9 items-center justify-center rounded text-text-muted transition hover:bg-surface2 hover:text-text lg:flex"
              aria-label={t("playground:workspace.hideStudio", "Hide studio")}
            >
              <PanelRightClose className="h-4 w-4" />
            </button>
          </Tooltip>
        )}
      </div>

      {/* Studio Options Section - Collapsible */}
      <div className="border-b border-border">
        <button
          type="button"
          onClick={() => setStudioOptionsExpanded(!studioOptionsExpanded)}
          aria-expanded={studioOptionsExpanded}
          aria-controls="studio-options-section"
          className="flex w-full items-center justify-between px-4 py-3 text-left transition hover:bg-surface2/50"
        >
          <h3 className="text-xs font-semibold uppercase text-text-muted">
            {t("playground:studio.studioOptions", "Studio Options")}
          </h3>
          {studioOptionsExpanded ? (
            <ChevronUp className="h-4 w-4 text-text-muted" />
          ) : (
            <ChevronDown className="h-4 w-4 text-text-muted" />
          )}
        </button>
        <div
          id="studio-options-section"
          hidden={!studioOptionsExpanded}
          className="px-4 pb-4"
        >
          <div
            data-testid="studio-options-accordion"
            className="space-y-4 rounded border border-border bg-surface2/30 p-3"
          >
            <section aria-label={t("playground:studio.modelRuntime", "Model Runtime")}>
              <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
                {t("playground:studio.modelRuntime", "Model Runtime")}
              </h4>
              <div className="space-y-3">
                <div>
                  <label className="mb-1 block text-xs font-medium text-text-muted">
                    {t("playground:studio.apiProvider", "API Provider")}
                  </label>
                  <Select
                    size={studioControlSize}
                    className="w-full"
                    value={normalizedApiProvider}
                    onChange={(value) => {
                      if (value === "__auto__") {
                        updateModelSetting("apiProvider", undefined)
                        return
                      }
                      setApiProvider(String(value))
                    }}
                    options={[
                      {
                        value: "__auto__",
                        label: t(
                          "playground:studio.apiProviderAuto",
                          "Auto (from selected model)"
                        )
                      },
                      ...providerOptions
                    ]}
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-text-muted">
                    {t("playground:studio.modelSelection", "Model")}
                  </label>
                  <Select
                    size={studioControlSize}
                    className="w-full"
                    value={selectedModel ?? undefined}
                    onChange={(value) => setSelectedModel(String(value))}
                    options={modelOptions}
                    loading={loadingChatModels}
                    placeholder={t(
                      "playground:studio.modelSelectionPlaceholder",
                      "Select a model"
                    )}
                  />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                    <label className="font-medium">
                      {t("playground:studio.temperature", "Temperature")}
                    </label>
                    <span className="font-medium text-text">
                      {resolvedTemperature.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={2}
                    step={0.01}
                    value={resolvedTemperature}
                    onChange={handleStudioTemperatureChange}
                  />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                    <label className="font-medium">
                      {t("playground:studio.topP", "Top P")}
                    </label>
                    <span className="font-medium text-text">
                      {resolvedTopP.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={resolvedTopP}
                    onChange={handleStudioTopPChange}
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-text-muted">
                    {t("playground:studio.maxTokens", "Max Tokens")}
                  </label>
                  <Input
                    size={studioControlSize}
                    type="number"
                    min={1}
                    max={32768}
                    value={resolvedNumPredict}
                    onChange={(event) => {
                      const raw = event.target.value.trim()
                      if (!raw) {
                        setNumPredict(undefined)
                        return
                      }
                      const parsed = Number(raw)
                      if (!Number.isFinite(parsed)) return
                      setNumPredict(Math.max(1, Math.min(32768, Math.round(parsed))))
                    }}
                  />
                </div>
              </div>
            </section>

            <section aria-label={t("playground:studio.ragSettings", "RAG Settings")}>
              <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
                {t("playground:studio.ragSettings", "RAG Settings")}
              </h4>
              <div className="space-y-3">
                {summaryUsesDirectSourceGeneration && (
                  <p className="rounded border border-border bg-surface px-2 py-1 text-[11px] text-text-muted">
                    {t(
                      "playground:studio.summaryDirectGenerationNote",
                      "Summary uses the workspace summary prompt and selected source content directly. Retrieval settings below do not apply."
                    )}
                  </p>
                )}
                <div>
                  <label className="mb-1 block text-xs font-medium text-text-muted">
                    {t("playground:studio.ragSearchMode", "Search Mode")}
                  </label>
                  <Select
                    size={studioControlSize}
                    className="w-full"
                    value={ragSearchMode}
                    disabled={summaryUsesDirectSourceGeneration}
                    onChange={(value) =>
                      setRagSearchMode(value as "hybrid" | "vector" | "fts")
                    }
                    options={[
                      { value: "hybrid", label: t("playground:studio.ragHybrid", "Hybrid") },
                      { value: "vector", label: t("playground:studio.ragVector", "Vector") },
                      { value: "fts", label: t("playground:studio.ragFts", "FTS") }
                    ]}
                  />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                    <span>{t("playground:studio.ragTopK", "Top K")}</span>
                    <span className="font-medium text-text">{resolvedStudioTopK}</span>
                  </div>
                  <Slider
                    min={1}
                    max={50}
                    step={1}
                    value={resolvedStudioTopK}
                    disabled={summaryUsesDirectSourceGeneration}
                    onChange={handleStudioTopKChange}
                  />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                    <span>
                      {t(
                        "playground:studio.ragSimilarityThreshold",
                        "Similarity threshold"
                      )}
                    </span>
                    <span className="font-medium text-text">
                      {studioSimilarityThreshold.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={studioSimilarityThreshold}
                    disabled={summaryUsesDirectSourceGeneration}
                    onChange={handleStudioSimilarityThresholdChange}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-muted">
                    {t("playground:studio.ragEnableGeneration", "Enable generation")}
                  </span>
                  <Switch
                    size="small"
                    checked={ragEnableGeneration}
                    disabled={summaryUsesDirectSourceGeneration}
                    onChange={(checked) => setRagEnableGeneration(checked)}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-muted">
                    {t("playground:studio.ragEnableCitations", "Enable citations")}
                  </span>
                  <Switch
                    size="small"
                    checked={ragEnableCitations}
                    disabled={summaryUsesDirectSourceGeneration}
                    onChange={(checked) => setRagEnableCitations(checked)}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-muted">
                    {t("playground:studio.ragEnableReranking", "Enable reranking")}
                  </span>
                  <Switch
                    size="small"
                    checked={studioRerankingEnabled}
                    disabled={summaryUsesDirectSourceGeneration}
                    onChange={(checked) =>
                      patchRagAdvancedOptions({ enable_reranking: checked })
                    }
                  />
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>

      {/* Studio Section - Collapsible */}
      <div className="border-b border-border">
        <button
          type="button"
          onClick={() => setStudioExpanded(!studioExpanded)}
          aria-expanded={studioExpanded}
          aria-controls="studio-output-types-section"
          className="flex w-full items-center justify-between px-4 py-3 text-left transition hover:bg-surface2/50"
        >
          <h3 className="text-xs font-semibold uppercase text-text-muted">
            {t("playground:studio.outputTypes", "Output Types")}
          </h3>
          {studioExpanded ? (
            <ChevronUp className="h-4 w-4 text-text-muted" />
          ) : (
            <ChevronDown className="h-4 w-4 text-text-muted" />
          )}
        </button>
        <div
          id="studio-output-types-section"
          hidden={!studioExpanded}
          className="px-4 pb-4"
        >
        {isGeneratingOutput && (
          <div className="mb-3 space-y-2">
            {/* Multi-phase progress indicator */}
            <div className="flex items-center gap-2">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              <p className="text-xs font-medium text-text">
                {generationPhase === "preparing"
                  ? t("playground:studio.phasePreparing", "Preparing...")
                  : generationPhase === "retrieving"
                    ? t("playground:studio.phaseRetrieving", "Retrieving relevant content...")
                    : generationPhase === "finalizing"
                      ? t("playground:studio.phaseFinalizing", "Finalizing...")
                      : t(
                          "playground:studio.phaseGenerating",
                          "Generating {{type}}...",
                          {
                            type:
                              OUTPUT_BUTTONS.find(
                                (button) => button.type === generatingOutputType
                              )?.label || generatingOutputType || "output"
                          }
                        )}
                {generationElapsedSeconds > 0 && (
                  <span className="ml-1 text-text-muted">
                    ({generationElapsedSeconds}s)
                  </span>
                )}
              </p>
            </div>
            {/* Phase progress bar */}
            <div className="flex gap-1">
              {(["preparing", "retrieving", "generating", "finalizing"] as const).map((phase) => {
                const phaseOrder = ["preparing", "retrieving", "generating", "finalizing"]
                const currentIdx = generationPhase ? phaseOrder.indexOf(generationPhase) : -1
                const thisIdx = phaseOrder.indexOf(phase)
                const isComplete = thisIdx < currentIdx
                const isActive = thisIdx === currentIdx
                return (
                  <div
                    key={phase}
                    className={`h-1 flex-1 rounded-full transition-colors ${
                      isComplete
                        ? "bg-primary"
                        : isActive
                          ? "bg-primary/60 animate-pulse"
                          : "bg-border"
                    }`}
                  />
                )
              })}
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="text-[11px] text-text-muted">
                {t(
                  "playground:studio.generatingWithEta",
                  "~{{seconds}}s for {{count}} source{{suffix}}",
                  {
                    seconds: etaSeconds ?? 15,
                    count: Math.max(1, selectedMediaCount),
                    suffix: Math.max(1, selectedMediaCount) === 1 ? "" : "s"
                  }
                )}
              </p>
              <p className="text-[10px] text-text-muted/70 mt-0.5">
                {t(
                  "playground:studio.generatingUsualDuration",
                  "Usually takes 10\u201330 seconds"
                )}
              </p>
            </div>
            <Button
              size="small"
              danger
              icon={<Square className="h-3.5 w-3.5 fill-current" />}
              onClick={handleCancelGeneration}
            >
              {t("common:cancel", "Cancel")}
            </Button>
            </div>
          </div>
        )}
        {(() => {
          const primaryButtons = OUTPUT_BUTTONS.filter((b) => PRIMARY_OUTPUT_TYPES.has(b.type))
          const secondaryButtons = OUTPUT_BUTTONS.filter((b) => !PRIMARY_OUTPUT_TYPES.has(b.type))

          const artifactStatusForType = (type: ArtifactType): "completed" | "failed" | null => {
            const match = generatedArtifacts.find((a) => a.type === type)
            if (!match) return null
            if (match.status === "completed") return "completed"
            if (match.status === "failed") return "failed"
            return null
          }

          const renderOutputButton = (type: ArtifactType) => {
            const button = OUTPUT_BUTTONS.find((entry) => entry.type === type)
            if (!button) return null
            const { label, icon: Icon, description } = button
            const isGenerating =
              isGeneratingOutput && generatingOutputType === type
            const requiresMultipleSources =
              type === "compare_sources" && selectedMediaCount < 2
            const isDisabled =
              !hasSelectedSources || isGeneratingOutput || requiresMultipleSources
            const artifactStatus = artifactStatusForType(type)

            return (
              <Tooltip
                key={type}
                title={
                  requiresMultipleSources
                    ? t(
                        "playground:studio.compareRequiresMultipleSourcesHint",
                        "Compare Sources requires at least two selected sources."
                      )
                    : !hasSelectedSources
                    ? t(
                        "playground:studio.selectSourcesFirst",
                        "Select sources first"
                      )
                    : description
                }
              >
                <button
                  type="button"
                  disabled={isDisabled}
                  onFocus={() => setActiveOutputType(type)}
                  onMouseEnter={() => setActiveOutputType(type)}
                  onClick={() => {
                    setActiveOutputType(type)
                    if (type === "audio_overview") {
                      setShowTtsSettings(true)
                    }
                    void handleGenerateOutput(type)
                  }}
                  className={`relative flex flex-col items-center justify-center rounded-lg border p-3 transition-colors ${
                    isDisabled
                      ? "cursor-not-allowed border-border bg-surface2/50 text-text-muted"
                      : "border-border bg-surface hover:border-primary/50 hover:bg-primary/5"
                  }`}
                >
                  {isGenerating ? (
                    <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                  <span className="mt-1.5 flex flex-col items-center text-center">
                    <span className="text-xs font-medium">{label}</span>
                    <span className="text-[10px] leading-tight text-text-muted font-normal line-clamp-2 mt-0.5">
                      {description}
                    </span>
                  </span>
                  {artifactStatus === "completed" && (
                    <CheckCircle className="absolute right-1.5 top-1.5 h-3.5 w-3.5 text-success" />
                  )}
                  {artifactStatus === "failed" && (
                    <XCircle className="absolute right-1.5 top-1.5 h-3.5 w-3.5 text-error" />
                  )}
                </button>
              </Tooltip>
            )
          }

          return (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                {primaryButtons.map((btn) => renderOutputButton(btn.type))}
              </div>

              {secondaryButtons.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => setMoreOutputsExpanded((prev) => !prev)}
                    className="mt-2 flex w-full items-center justify-center gap-1 rounded-md border border-border px-3 py-1.5 text-xs text-text-muted hover:bg-surface2 transition-colors"
                    aria-expanded={moreOutputsExpanded}
                  >
                    {moreOutputsExpanded
                      ? t("playground:studio.lessOutputs", "Show fewer")
                      : t("playground:studio.moreOutputs", { count: secondaryButtons.length, defaultValue: "More outputs ({{count}})" })}
                    <ChevronDown
                      className={`h-3.5 w-3.5 transition-transform ${moreOutputsExpanded ? "rotate-180" : ""}`}
                    />
                  </button>

                  {moreOutputsExpanded && (
                    <div className="mt-2 grid grid-cols-2 gap-2">
                      {secondaryButtons.map((btn) => renderOutputButton(btn.type))}
                    </div>
                  )}
                </>
              )}
            </div>
          )
        })()}
        {!hasSelectedSources && (
          <p className="mt-2 text-center text-xs text-text-muted">
            {t(
              "playground:studio.selectSourcesHint",
              getWorkspaceStudioNoSourcesHint(isMobile)
            )}
          </p>
        )}

        <div className="mt-4 rounded border border-border bg-surface2/30 p-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                {t("playground:studio.slidesSettings", "Slides Settings")}
              </p>
              <p className="mt-1 text-xs text-text-muted">
                {t(
                  "playground:studio.slidesStyleHint",
                  "Choose the presentation strategy used when generating Slides output."
                )}
              </p>
            </div>
          </div>
          <div className="mt-3 space-y-2">
            <label
              className="block text-xs font-medium text-text-muted"
              htmlFor="workspace-slides-visual-style"
            >
              {t("playground:studio.slidesVisualStyle", "Slides visual style")}
            </label>
            <select
              id="workspace-slides-visual-style"
              aria-label="Slides visual style"
              value={slidesVisualStyleValue}
              onChange={(event) => setSlidesVisualStyleValue(event.target.value)}
              disabled={slidesVisualStylesLoading}
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text outline-none transition focus:border-primary/50 disabled:cursor-not-allowed disabled:opacity-70"
            >
              <option value="">
                {t(
                  "playground:studio.noSlidesVisualStyle",
                  "No visual style preset"
                )}
              </option>
              {groupedSlidesVisualStyles.builtin.length > 0 && (
                <optgroup label={t("playground:studio.builtinStyles", "Built-in styles")}>
                  {groupedSlidesVisualStyles.builtin.map((style) => (
                    <option
                      key={`${style.scope}:${style.id}`}
                      value={encodeSlidesVisualStyleValue(style.id, style.scope)}
                    >
                      {style.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {groupedSlidesVisualStyles.user.length > 0 && (
                <optgroup label={t("playground:studio.customStyles", "Custom styles")}>
                  {groupedSlidesVisualStyles.user.map((style) => (
                    <option
                      key={`${style.scope}:${style.id}`}
                      value={encodeSlidesVisualStyleValue(style.id, style.scope)}
                    >
                      {style.name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <p className="text-xs text-text-muted">
              {selectedSlidesVisualStyle?.description ||
                t(
                  "playground:studio.slidesStyleFallback",
                  "Slides fall back to the default presentation generator when no preset is selected."
                )}
            </p>
          </div>
        </div>

        {/* TTS Settings Panel */}
        <div className="mt-4">
          {!contextualAudioSettingsVisible && !showTtsSettings && (
            <p className="mb-2 rounded border border-border bg-surface2/30 px-3 py-2 text-xs text-text-muted">
              {t(
                "playground:studio.audioSettingsHint",
                "Select Audio Summary to configure TTS voice and speed."
              )}
            </p>
          )}
          <div
            data-testid="studio-audio-settings-accordion"
            className="overflow-hidden rounded border border-border bg-surface2/30"
          >
            <button
              type="button"
              onClick={() => setShowTtsSettings(!showTtsSettings)}
              aria-expanded={showAudioSettingsPanel}
              aria-controls="studio-audio-settings-panel"
              className={`flex w-full items-center justify-between px-3 py-2 text-xs text-text-muted transition-colors ${
                showAudioSettingsPanel
                  ? "border-b border-border bg-surface2/60"
                  : "bg-surface2/50 hover:bg-surface2"
              }`}
            >
              <span className="flex items-center gap-2">
                <Settings2 className="h-3.5 w-3.5" />
                {t("playground:studio.audioSettings", "Audio Settings")}
              </span>
              {showAudioSettingsPanel ? (
                <ChevronUp className="h-3.5 w-3.5" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5" />
              )}
            </button>

            {showAudioSettingsPanel && (
              <div
                id="studio-audio-settings-panel"
                className={`p-3 ${
                  isMobile ? "space-y-4" : "space-y-3"
                }`}
              >
              {/* Provider */}
              <div>
                <label className="mb-1 block text-xs font-medium text-text-muted">
                  {t("playground:studio.ttsProvider", "TTS Provider")}
                </label>
                <Select
                  size={studioControlSize}
                  className="w-full"
                  value={audioSettings.provider}
                  onChange={(value) => setAudioSettings({ provider: value as AudioTtsProvider })}
                  options={TTS_PROVIDERS}
                />
              </div>

              {/* Model (for tldw and openai) */}
              {audioSettings.provider !== "browser" && (
                <div>
                  <label className="mb-1 block text-xs font-medium text-text-muted">
                    {t("playground:studio.ttsModel", "Model")}
                  </label>
                  <Select
                    size={studioControlSize}
                    className="w-full"
                    value={audioSettings.model}
                    onChange={(value) => setAudioSettings({ model: value })}
                    options={getTtsModelOptions()}
                  />
                </div>
              )}

              {/* Voice */}
              {audioSettings.provider !== "browser" && (
                <div>
                  <label className="mb-1 block text-xs font-medium text-text-muted">
                    {t("playground:studio.ttsVoice", "Voice")}
                  </label>
                  <Select
                    size={studioControlSize}
                    className="w-full"
                    value={audioSettings.voice}
                    onChange={(value) => setAudioSettings({ voice: value })}
                    options={getVoiceOptions()}
                    loading={loadingVoices}
                    showSearch
                    optionFilterProp="label"
                  />
                  <div className="mt-2 flex justify-end">
                    <Button
                      size={studioControlSize}
                      onClick={() => void handlePreviewVoice()}
                      loading={previewingVoice}
                    >
                      {t("playground:studio.previewVoice", "Preview")}
                    </Button>
                  </div>
                </div>
              )}

              {/* Speed */}
              <div>
                <label className="mb-1 block text-xs font-medium text-text-muted">
                  {t("playground:studio.ttsSpeed", "Speed")}: {audioSettings.speed.toFixed(1)}x
                </label>
                <Slider
                  data-testid="studio-tts-speed-slider"
                  className={mobileSliderClassName}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  value={audioSettings.speed}
                  onChange={(value) => setAudioSettings({ speed: value })}
                  tooltip={{ formatter: (v) => `${v}x` }}
                />
              </div>

              {/* Format */}
              <div>
                <label className="mb-1 block text-xs font-medium text-text-muted">
                  {t("playground:studio.ttsFormat", "Output Format")}
                </label>
                <Select
                  size={studioControlSize}
                  className="w-full"
                  value={audioSettings.format}
                  onChange={(value) => setAudioSettings({ format: value as any })}
                  options={AUDIO_FORMATS}
                />
              </div>

              <div className="rounded border border-border bg-surface/40 p-2">
                <div className="mb-1 flex items-center justify-between">
                  <label className="block text-xs font-medium text-text-muted">
                    {t("playground:studio.flashcardDeck", "Flashcard Deck")}
                  </label>
                  <Button
                    size={studioControlSize}
                    onClick={() => void loadFlashcardDecks()}
                    loading={loadingDecks}
                  >
                    {t("common:refresh", "Refresh")}
                  </Button>
                </div>
                <Select
                  size={studioControlSize}
                  className="w-full"
                  value={selectedFlashcardDeck}
                  onChange={(value) =>
                    setSelectedFlashcardDeck(
                      value === "auto" ? "auto" : Number(value)
                    )
                  }
                  options={[
                    {
                      value: "auto",
                      label: t(
                        "playground:studio.flashcardDeckAuto",
                        "Auto (create new deck)"
                      )
                    },
                    ...availableDecks.map((deck) => ({
                      value: deck.id,
                      label: deck.name
                    }))
                  ]}
                />
              </div>
              </div>
            )}
          </div>
        </div>
        </div>
      </div>

      {/* Generated Outputs Section - Collapsible */}
      <div className="border-b border-border">
        <button
          type="button"
          onClick={() => setOutputsExpanded(!outputsExpanded)}
          aria-expanded={outputsExpanded}
          aria-controls="studio-generated-outputs-section"
          className="flex w-full items-center justify-between px-4 py-3 text-left transition hover:bg-surface2/50"
        >
          <h3 className="text-xs font-semibold uppercase text-text-muted">
            {t("playground:studio.generatedOutputs", "Generated Outputs")}
            {generatedArtifacts.length > 0 && (
              <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                {generatedArtifacts.length}
              </span>
            )}
          </h3>
          {outputsExpanded ? (
            <ChevronUp className="h-4 w-4 text-text-muted" />
          ) : (
            <ChevronDown className="h-4 w-4 text-text-muted" />
          )}
        </button>
        <div
          id="studio-generated-outputs-section"
          ref={outputListContainerRef}
          hidden={!outputsExpanded}
          className="custom-scrollbar min-h-[10rem] overflow-y-auto px-4 pb-4"
          style={{ maxHeight: "40vh" }}
          onScroll={(event) => {
            if (!useVirtualizedOutputs) return
            setOutputListScrollTop(event.currentTarget.scrollTop)
          }}
          data-virtualized={useVirtualizedOutputs ? "true" : "false"}
          data-testid={
            useVirtualizedOutputs
              ? "generated-outputs-virtualized"
              : "generated-outputs-standard"
          }
        >
          {generatedArtifacts.length > 0 && (
            <div className="mb-2 rounded border border-border bg-surface/60 px-2.5 py-2 text-[11px] text-text-muted">
              {t(
                "playground:studio.usageSummary",
                "Estimated workspace usage: {{tokens}} tokens • ${{cost}}",
                {
                  tokens: Math.round(cumulativeUsage.tokens).toLocaleString(),
                  cost: cumulativeUsage.costUsd.toFixed(3)
                }
              )}
            </div>
          )}
          {generatedArtifacts.length === 0 ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <span className="text-xs text-text-muted">
                  {t("playground:studio.noOutputs", "No outputs generated yet")}
                </span>
              }
            />
          ) : (
            <div
              className="space-y-2"
              style={
                useVirtualizedOutputs
                  ? {
                      paddingTop: virtualOutputTopPadding,
                      paddingBottom: virtualOutputBottomPadding
                    }
                  : undefined
              }
            >
              {visibleArtifacts.map((artifact) => {
                const Icon = ARTIFACT_TYPE_ICONS[artifact.type] || FileText
                const StatusConfig = STATUS_ICONS[artifact.status]
                const StatusIcon = StatusConfig.icon
                const failedStatusDeleteLabel = t(
                  "playground:studio.deleteFailedOutput",
                  "Delete failed output"
                )
                const failedRetryLabel = t(
                  "playground:studio.retryFailedOutput",
                  "Retry"
                )

                return (
                  <div
                    key={artifact.id}
                    data-testid={`studio-artifact-card-${artifact.id}`}
                    className="group rounded-lg border border-border bg-surface2/50 p-3"
                  >
                    <div className="flex items-start gap-2">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded bg-surface text-text-muted">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="truncate text-sm font-medium text-text">
                            {artifact.title}
                          </p>
                          {artifact.status === "failed" ? (
                            <>
                              <Tooltip title={failedRetryLabel}>
                                <button
                                  type="button"
                                  disabled={!hasSelectedSources || isGeneratingOutput}
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    if (!hasSelectedSources || isGeneratingOutput) return
                                    void handleGenerateOutput(artifact.type, {
                                      mode: "replace",
                                      targetArtifactId: artifact.id
                                    })
                                  }}
                                  onKeyDown={handleIconButtonKeyDown}
                                  className="rounded p-0.5 hover:bg-primary/10 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
                                  aria-label={failedRetryLabel}
                                  data-testid={`studio-artifact-retry-${artifact.id}`}
                                >
                                  <RefreshCw className="h-3.5 w-3.5 shrink-0 text-text-muted hover:text-primary" />
                                </button>
                              </Tooltip>
                              <Tooltip title={failedStatusDeleteLabel}>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    handleDeleteArtifact(artifact)
                                  }}
                                  onKeyDown={handleIconButtonKeyDown}
                                  className="rounded p-0.5 hover:bg-error/10"
                                  aria-label={failedStatusDeleteLabel}
                                >
                                  <StatusIcon
                                    className={`h-4 w-4 shrink-0 ${StatusConfig.className}`}
                                  />
                                </button>
                              </Tooltip>
                            </>
                          ) : (
                            <StatusIcon
                              className={`h-4 w-4 shrink-0 ${StatusConfig.className}`}
                            />
                          )}
                        </div>
                        <p className="text-xs text-text-muted">
                          {artifact.createdAt.toLocaleString()}
                        </p>
                        {(artifact.totalTokens ||
                          artifact.estimatedTokens ||
                          artifact.totalCostUsd ||
                          artifact.estimatedCostUsd) && (
                          <p className="text-[11px] text-text-muted">
                            {t(
                              "playground:studio.usagePerOutput",
                              "Tokens: {{tokens}} • Cost: ${{cost}}",
                              {
                                tokens: Math.round(
                                  artifact.totalTokens || artifact.estimatedTokens || 0
                                ).toLocaleString(),
                                cost: (
                                  artifact.totalCostUsd || artifact.estimatedCostUsd || 0
                                ).toFixed(3)
                              }
                            )}
                          </p>
                        )}
                        {artifact.status === "failed" && artifact.errorMessage && (
                          <p className="mt-1 text-xs text-error">
                            {artifact.errorMessage}
                          </p>
                        )}
                      </div>
                    </div>
                    {artifact.status === "completed" && (
                      <div className="mt-2 space-y-1.5">
                        <div
                          data-testid={`studio-artifact-primary-actions-${artifact.id}`}
                          role="group"
                          aria-label={t(
                            "playground:studio.primaryActions",
                            "Primary output actions"
                          )}
                          className="flex flex-wrap items-center gap-1 rounded border border-border/70 bg-surface/70 px-1.5 py-1"
                        >
                          {(artifact.content || artifact.audioUrl) && (
                            <Tooltip title={t("common:view", "View")}>
                              <button
                                type="button"
                                onClick={() => handleViewArtifact(artifact)}
                                onKeyDown={handleIconButtonKeyDown}
                                className="rounded p-1 text-text-muted hover:bg-surface2 hover:text-text"
                                aria-label={t("common:view", "View")}
                              >
                                <Eye className="h-4 w-4" />
                              </button>
                            </Tooltip>
                          )}
                          {(artifact.type === "flashcards" || artifact.type === "quiz") && (
                            <Tooltip title={t("common:edit", "Edit")}>
                              <button
                                type="button"
                                onClick={() => handleViewArtifact(artifact)}
                                onKeyDown={handleIconButtonKeyDown}
                                className="rounded p-1 text-text-muted hover:bg-surface2 hover:text-text"
                                aria-label={t("common:edit", "Edit")}
                                data-testid={`studio-artifact-edit-${artifact.id}`}
                              >
                                <Pencil className="h-4 w-4" />
                              </button>
                            </Tooltip>
                          )}
                          <Tooltip title={t("common:download", "Download")}>
                            <button
                              type="button"
                              onClick={() =>
                                artifact.type === "slides" && artifact.presentationId
                                  ? handleSlidesDownload(artifact)
                                  : handleDownloadArtifact(artifact)
                              }
                              onKeyDown={handleIconButtonKeyDown}
                              className="rounded p-1 text-text-muted hover:bg-surface2 hover:text-text"
                              aria-label={t("common:download", "Download")}
                            >
                              <Download className="h-4 w-4" />
                            </button>
                          </Tooltip>
                        </div>
                        <div
                          data-testid={`studio-artifact-secondary-actions-${artifact.id}`}
                          role="group"
                          aria-label={t(
                            "playground:studio.secondaryActions",
                            "Secondary output actions"
                          )}
                          className="flex flex-wrap items-center gap-1 rounded border border-border/60 bg-surface2/50 px-1.5 py-1"
                        >
                          <Dropdown
                            trigger={["click"]}
                            menu={{
                              items: [
                                {
                                  key: "replace",
                                  label: t(
                                    "playground:studio.regenerateReplace",
                                    "Replace existing"
                                  )
                                },
                                {
                                  key: "new_version",
                                  label: t(
                                    "playground:studio.regenerateNewVersion",
                                    "Create new version"
                                  )
                                }
                              ],
                              onClick: ({ key }) => {
                                const mode =
                                  key === "replace" ? "replace" : "new_version"
                                handleGenerateOutput(artifact.type, {
                                  mode,
                                  targetArtifactId: artifact.id
                                })
                              }
                            }}
                          >
                            <Tooltip
                              title={t(
                                "playground:studio.regenerateOptions",
                                "Regenerate options"
                              )}
                            >
                              <button
                                type="button"
                                onKeyDown={handleIconButtonKeyDown}
                                className="rounded p-1 text-text-muted hover:bg-surface hover:text-text"
                                aria-label={t(
                                  "playground:studio.regenerateOptions",
                                  "Regenerate options"
                                )}
                              >
                                <RefreshCw className="h-4 w-4" />
                              </button>
                            </Tooltip>
                          </Dropdown>
                          <Tooltip
                            title={t(
                              "playground:studio.discussAction",
                              "Discuss in chat"
                            )}
                          >
                            <button
                              type="button"
                              onClick={() => handleDiscussArtifact(artifact)}
                              onKeyDown={handleIconButtonKeyDown}
                              className="rounded p-1 text-text-muted hover:bg-surface hover:text-text"
                              aria-label={t(
                                "playground:studio.discussAction",
                                "Discuss in chat"
                              )}
                            >
                              <MessageCircle className="h-4 w-4" />
                            </button>
                          </Tooltip>
                          {artifact.content && (
                            <Dropdown
                              trigger={["click"]}
                              menu={{
                                items: [
                                  {
                                    key: "append",
                                    label: t(
                                      "playground:studio.saveToNotesAppend",
                                      "Append to notes"
                                    )
                                  },
                                  {
                                    key: "replace",
                                    label: t(
                                      "playground:studio.saveToNotesReplace",
                                      "Replace note draft"
                                    )
                                  }
                                ],
                                onClick: ({ key }) => {
                                  handleSaveArtifactToNotes(
                                    artifact,
                                    key === "replace" ? "replace" : "append"
                                  )
                                }
                              }}
                            >
                              <Tooltip
                                title={t(
                                  "playground:studio.saveToNotesAction",
                                  "Save to notes"
                                )}
                              >
                                <button
                                  type="button"
                                  onKeyDown={handleIconButtonKeyDown}
                                  className="rounded p-1 text-text-muted hover:bg-surface hover:text-text"
                                  aria-label={t(
                                    "playground:studio.saveToNotesAction",
                                    "Save to notes"
                                  )}
                                >
                                  <StickyNote className="h-4 w-4" />
                                </button>
                              </Tooltip>
                            </Dropdown>
                          )}
                          <Tooltip title={t("common:delete", "Delete")}>
                            <button
                              type="button"
                              onClick={() => handleDeleteArtifact(artifact)}
                              onKeyDown={handleIconButtonKeyDown}
                              className="rounded p-1 text-text-muted hover:bg-error/10 hover:text-error"
                              aria-label={t("common:delete", "Delete")}
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </Tooltip>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
          </div>
      </div>

      {/* Quick Notes Section - Collapsible, fills remaining height */}
      <div className="flex min-h-[220px] flex-1 flex-col">
        {notesExpanded ? (
          renderQuickNotesSection(() => setNotesExpanded(false))
        ) : (
          <button
            type="button"
            onClick={() => setNotesExpanded(true)}
            className="flex w-full items-center justify-between border-t border-border px-4 py-3 text-left transition hover:bg-surface2/50"
          >
            <h3 className="text-xs font-semibold uppercase text-text-muted">
              {t("playground:studio.quickNotes", "Quick Notes")}
            </h3>
            <ChevronDown className="h-4 w-4 text-text-muted" />
          </button>
        )}
      </div>
    </div>
  )
}

export default StudioPane
