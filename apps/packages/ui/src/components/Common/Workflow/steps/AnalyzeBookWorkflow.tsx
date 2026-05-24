import React, { useState, useCallback, useEffect, useRef } from "react"
import {
  Input,
  Radio,
  Button,
  Space,
  Spin,
  message,
  Upload,
  Card,
  Progress,
  Collapse,
  Tooltip,
  Select
} from "antd"
import {
  Copy,
  Download,
  MessageSquare,
  CheckCircle,
  BookOpen,
  Upload as UploadIcon,
  AlertTriangle,
  RefreshCw,
  Save,
  ChevronDown,
  Edit2,
  FileText,
  Layers
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { bgRequest, bgUpload } from "@/services/background-proxy"
import { getProcessPathForType, inferUploadMediaTypeFromFile } from "@/services/tldw/media-routing"
import { useWorkflowsStore } from "@/store/workflows"
import { Alert } from "@/components/ui/primitives/Alert"
import { WizardShell } from "../WizardShell"
import { ANALYZE_BOOK_WORKFLOW } from "../workflow-definitions"

const { TextArea } = Input
const { Dragger } = Upload
const { Panel } = Collapse

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface BookInfo {
  file: File | null
  mediaId: string | null
  title: string
  content: string
  fileType: string
}

interface Chapter {
  id: string
  number: number
  title: string
  content: string
  wordCount: number
  charCount: number
  status: "clean" | "warning" | "error"
  preview: string
}

type AnalysisPreset =
  | "comprehensive"
  | "chapterSummaries"
  | "characterAnalysis"
  | "keyConcepts"
  | "custom"

type AnalysisScope = "whole" | "per-chapter"

interface AnalysisConfig {
  preset: AnalysisPreset
  customPrompt: string
  selectedModel: string
  scope: AnalysisScope
}

interface AnalysisResult {
  wholeBook: string | null
  perChapter: Record<number, string>
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const SUPPORTED_FILE_TYPES = [
  ".pdf",
  ".epub",
  ".txt",
  ".md",
  ".doc",
  ".docx"
]

const DEFAULT_CHAPTER_PATTERN = /^(?:chapter|part|section)\s*\d*[:\s]*(.*)/im
const MAX_ANALYSIS_CHARS = 30000

const ANALYSIS_PRESETS: Record<
  Exclude<AnalysisPreset, "custom">,
  { labelKey: string; prompt: string }
> = {
  comprehensive: {
    labelKey: "workflows:analyzeBook.presets.comprehensive",
    prompt: `Analyze this book content comprehensively:
1. Main themes and arguments
2. Key insights and takeaways
3. Notable quotes or passages
4. Strengths and areas of interest
5. Recommended applications or follow-up reading`
  },
  chapterSummaries: {
    labelKey: "workflows:analyzeBook.presets.chapterSummaries",
    prompt: `Provide a detailed summary of this chapter including:
- Main topic and purpose
- Key points covered
- Important concepts introduced
- Connection to broader themes`
  },
  characterAnalysis: {
    labelKey: "workflows:analyzeBook.presets.characterAnalysis",
    prompt: `Analyze the characters in this content:
- Main characters and their roles
- Character development and arcs
- Relationships between characters
- Motivations and conflicts`
  },
  keyConcepts: {
    labelKey: "workflows:analyzeBook.presets.keyConcepts",
    prompt: `Extract and explain the key concepts:
- Core ideas and theories
- Technical terms with definitions
- Frameworks or models presented
- Practical applications`
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AnalyzeBookWorkflow
 *
 * A complete workflow for analyzing books:
 * 1. Select - Upload or select a book
 * 2. Chunking - Review chapter detection
 * 3. Configure - Choose analysis options
 * 4. Process - Run analysis
 * 5. Review - View and act on results
 */
export const AnalyzeBookWorkflow: React.FC = () => {
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)

  if (!activeWorkflow || activeWorkflow.workflowId !== "analyze-book") {
    return null
  }

  return (
    <WizardShell workflow={ANALYZE_BOOK_WORKFLOW}>
      <AnalyzeBookStepContent />
    </WizardShell>
  )
}

const AnalyzeBookStepContent: React.FC = () => {
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const stepIndex = activeWorkflow?.currentStepIndex ?? 0

  switch (stepIndex) {
    case 0:
      return <SelectStep />
    case 1:
      return <ChunkingStep />
    case 2:
      return <ConfigureStep />
    case 3:
      return <ProcessStep />
    case 4:
      return <ReviewStep />
    default:
      return null
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Select
// ─────────────────────────────────────────────────────────────────────────────

const SelectStep: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)
  const setWorkflowStep = useWorkflowsStore((s) => s.setWorkflowStep)
  const setProcessing = useWorkflowsStore((s) => s.setProcessing)
  const setWorkflowError = useWorkflowsStore((s) => s.setWorkflowError)
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)

  const advanceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [bookInfo, setBookInfo] = useState<BookInfo | null>(
    (activeWorkflow?.data?.book as BookInfo) || null
  )
  const [isProcessing, setIsProcessingLocal] = useState(false)

  const handleFileUpload = async (file: File) => {
    try {
      setIsProcessingLocal(true)
      setProcessing(
        true,
        t("workflows:analyzeBook.reading", "Reading book file...")
      )

      // Read file content
      const content = await readFileContent(file)

      const info: BookInfo = {
        file,
        mediaId: null,
        title: file.name.replace(/\.[^/.]+$/, ""),
        content,
        fileType: file.name.split(".").pop()?.toLowerCase() || "txt"
      }

      setBookInfo(info)
      updateWorkflowData({ book: info })
      setProcessing(false)
      setIsProcessingLocal(false)

      // Auto-advance after a brief delay
      if (advanceTimeoutRef.current) {
        clearTimeout(advanceTimeoutRef.current)
      }
      advanceTimeoutRef.current = setTimeout(() => {
        const currentId = useWorkflowsStore.getState().activeWorkflow?.workflowId
        if (currentId !== "analyze-book") return
        setWorkflowStep(1)
      }, 500)
    } catch (error) {
      console.error("File read error:", error)
      setProcessing(false)
      setIsProcessingLocal(false)
      setWorkflowError(
        error instanceof Error
          ? error.message
          : t("workflows:analyzeBook.errors.readFailed", "Failed to read file")
      )
    }

    return false // Prevent default upload behavior
  }

  useEffect(() => {
    return () => {
      if (advanceTimeoutRef.current) {
        clearTimeout(advanceTimeoutRef.current)
      }
    }
  }, [])

  if (isProcessing) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spin size="large" />
        <p className="mt-4 text-text-muted">
          {t("workflows:analyzeBook.reading", "Reading book file...")}
        </p>
      </div>
    )
  }

  if (bookInfo) {
    return (
      <div className="space-y-4">
        <Alert
          variant="success"
          icon={<CheckCircle className="h-4 w-4" />}
          title={t("workflows:analyzeBook.bookSelected", "Book selected")}
        >
          <div className="mt-2">
            <p className="font-medium">{bookInfo.title}</p>
            <p className="text-xs text-text-muted flex items-center gap-1 mt-1">
              <FileText className="h-3 w-3" />
              {bookInfo.fileType.toUpperCase()} -{" "}
              {bookInfo.content.length.toLocaleString()} characters
            </p>
          </div>
        </Alert>
        <Button
          onClick={() => {
            setBookInfo(null)
            updateWorkflowData({ book: null })
          }}
        >
          {t("workflows:analyzeBook.selectDifferent", "Select a different book")}
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-text mb-3">
          {t("workflows:analyzeBook.uploadBook", "Upload a book")}
        </h3>
        <Dragger
          accept={SUPPORTED_FILE_TYPES.join(",")}
          beforeUpload={handleFileUpload}
          showUploadList={false}
          className="border-dashed"
        >
          <p className="ant-upload-drag-icon">
            <UploadIcon className="h-12 w-12 mx-auto text-text-muted" />
          </p>
          <p className="ant-upload-text">
            {t(
              "workflows:analyzeBook.dropzone",
              "Click or drag file to upload"
            )}
          </p>
          <p className="ant-upload-hint text-xs text-text-muted">
            {t(
              "workflows:analyzeBook.supportedFormats",
              "Supports PDF, EPUB, TXT, MD, DOC, DOCX"
            )}
          </p>
        </Dragger>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Chunking
// ─────────────────────────────────────────────────────────────────────────────

const ChunkingStep: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)
  const setProcessing = useWorkflowsStore((s) => s.setProcessing)

  const bookInfo = activeWorkflow?.data?.book as BookInfo | undefined
  const savedChapters = activeWorkflow?.data?.chapters as Chapter[] | undefined

  const [chapters, setChapters] = useState<Chapter[]>(savedChapters || [])
  const [chapterPattern, setChapterPattern] = useState(
    (activeWorkflow?.data?.chapterPattern as string) ||
      "^(?:chapter|part|section)\\s*\\d*[:\\s]*(.*)"
  )
  const [isChunking, setIsChunking] = useState(!savedChapters)

  const chunkContent = useCallback(
    (content: string, pattern: string) => {
      setIsChunking(true)
      setProcessing(
        true,
        t("workflows:analyzeBook.detecting", "Detecting chapters...")
      )

      try {
        const regex = new RegExp(pattern, "gim")
        const matches: { index: number; title: string }[] = []
        let match

        while ((match = regex.exec(content)) !== null) {
          matches.push({
            index: match.index,
            title: match[1]?.trim() || `Chapter ${matches.length + 1}`
          })
        }

        const detectedChapters: Chapter[] = []

        if (matches.length === 0) {
          // No chapters detected - treat entire content as one chunk
          const words = content.split(/\s+/).length
          detectedChapters.push({
            id: "chapter-1",
            number: 1,
            title: t("workflows:analyzeBook.entireBook", "Entire Book"),
            content: content,
            wordCount: words,
            charCount: content.length,
            status: words > 50000 ? "warning" : "clean",
            preview: content.slice(0, 200) + "..."
          })
        } else {
          for (let i = 0; i < matches.length; i++) {
            const start = matches[i].index
            const end = matches[i + 1]?.index || content.length
            const chapterContent = content.slice(start, end).trim()
            const words = chapterContent.split(/\s+/).length

            let status: "clean" | "warning" | "error" = "clean"
            if (words < 100) status = "warning"
            else if (words > 50000) status = "warning"

            detectedChapters.push({
              id: `chapter-${i + 1}`,
              number: i + 1,
              title: matches[i].title || `Chapter ${i + 1}`,
              content: chapterContent,
              wordCount: words,
              charCount: chapterContent.length,
              status,
              preview: chapterContent.slice(0, 200) + "..."
            })
          }
        }

        setChapters(detectedChapters)
        updateWorkflowData({
          chapters: detectedChapters,
          chapterPattern: pattern
        })
      } catch (error) {
        console.error("Chunking error:", error)
        message.error(
          t(
            "workflows:analyzeBook.errors.invalidPattern",
            "Invalid chapter pattern"
          )
        )
      } finally {
        setIsChunking(false)
        setProcessing(false)
      }
    },
    [t, updateWorkflowData, setProcessing]
  )

  useEffect(() => {
    if (savedChapters || !bookInfo?.content) return

    chunkContent(bookInfo.content, chapterPattern)
  }, [savedChapters, bookInfo?.content, chapterPattern, chunkContent])

  const handleReChunk = () => {
    if (bookInfo?.content) {
      chunkContent(bookInfo.content, chapterPattern)
    }
  }

  if (isChunking) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spin size="large" />
        <p className="mt-4 text-text-muted">
          {t("workflows:analyzeBook.detecting", "Detecting chapters...")}
        </p>
      </div>
    )
  }

  const cleanCount = chapters.filter((c) => c.status === "clean").length
  const warningCount = chapters.filter((c) => c.status === "warning").length

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-text">
          {t("workflows:analyzeBook.chapterDetection", "Chapter Detection")}
        </h3>
        <span className="text-xs text-text-muted">
          {t("workflows:analyzeBook.foundChapters", "Found {{count}} chapters", {
            count: chapters.length
          })}
        </span>
      </div>

      {warningCount > 0 && (
        <Alert
          variant="warning"
          icon={<AlertTriangle className="h-4 w-4" />}
          title={t(
            "workflows:analyzeBook.chunkingWarning",
            "{{count}} chapters may need review",
            { count: warningCount }
          )}
        >
          {t(
            "workflows:analyzeBook.chunkingWarningDesc",
            "Some chapters appear too short or too long. You can adjust the chapter pattern below."
          )}
        </Alert>
      )}

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {chapters.map((chapter) => (
          <ChapterCard key={chapter.id} chapter={chapter} />
        ))}
      </div>

      <div className="border-t border-border pt-4">
        <label className="text-xs font-medium text-text-muted block mb-2">
          {t("workflows:analyzeBook.chapterPattern", "Chapter Pattern (RegEx)")}
        </label>
        <div className="flex gap-2">
          <Input
            value={chapterPattern}
            onChange={(e) => setChapterPattern(e.target.value)}
            placeholder="^(?:chapter|part|section)\s*\d*[:\s]*(.*)"
            className="font-mono text-sm"
          />
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={handleReChunk}
          >
            {t("workflows:analyzeBook.reChunk", "Re-chunk")}
          </Button>
        </div>
      </div>
    </div>
  )
}

const ChapterCard: React.FC<{ chapter: Chapter }> = ({ chapter }) => {
  const { t } = useTranslation(["workflows"])

  const statusColors = {
    clean: "border-l-success",
    warning: "border-l-warn",
    error: "border-l-danger"
  }

  const statusIcons = {
    clean: <CheckCircle className="h-4 w-4 text-success" />,
    warning: <AlertTriangle className="h-4 w-4 text-warn" />,
    error: <AlertTriangle className="h-4 w-4 text-danger" />
  }

  return (
    <Card
      size="small"
      className={`border-l-4 ${statusColors[chapter.status]}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          {statusIcons[chapter.status]}
          <span className="font-medium">
            Chapter {chapter.number}: {chapter.title}
          </span>
        </div>
        <span className="text-xs text-text-muted">
          {chapter.wordCount.toLocaleString()}w
        </span>
      </div>
      <p className="text-xs text-text-muted mt-2 line-clamp-2">
        {chapter.preview}
      </p>
      {chapter.status === "warning" && (
        <p className="text-xs text-warn mt-1">
          {chapter.wordCount < 100
            ? t("workflows:analyzeBook.tooShort", "Too short - may be incomplete")
            : t("workflows:analyzeBook.tooLong", "Very long - consider splitting")}
        </p>
      )}
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Configure
// ─────────────────────────────────────────────────────────────────────────────

const ConfigureStep: React.FC = () => {
  const { t } = useTranslation(["workflows"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)

  const savedConfig = activeWorkflow?.data?.analysisConfig as
    | AnalysisConfig
    | undefined

  const [preset, setPreset] = useState<AnalysisPreset>(
    savedConfig?.preset || "comprehensive"
  )
  const [customPrompt, setCustomPrompt] = useState(
    savedConfig?.customPrompt || ""
  )
  const [scope, setScope] = useState<AnalysisScope>(
    savedConfig?.scope || "whole"
  )

  useEffect(() => {
    const config: AnalysisConfig = {
      preset,
      customPrompt,
      selectedModel: "",
      scope
    }
    updateWorkflowData({ analysisConfig: config })
  }, [preset, customPrompt, scope, updateWorkflowData])

  const presetOptions = [
    {
      value: "comprehensive" as const,
      label: t(
        "workflows:analyzeBook.presets.comprehensive",
        "Comprehensive Analysis"
      ),
      description: t(
        "workflows:analyzeBook.presets.comprehensiveDesc",
        "Full themes, insights, and takeaways"
      )
    },
    {
      value: "chapterSummaries" as const,
      label: t(
        "workflows:analyzeBook.presets.chapterSummaries",
        "Chapter Summaries"
      ),
      description: t(
        "workflows:analyzeBook.presets.chapterSummariesDesc",
        "Summary per chapter"
      )
    },
    {
      value: "characterAnalysis" as const,
      label: t(
        "workflows:analyzeBook.presets.characterAnalysis",
        "Character Analysis"
      ),
      description: t(
        "workflows:analyzeBook.presets.characterAnalysisDesc",
        "For fiction - character arcs and relationships"
      )
    },
    {
      value: "keyConcepts" as const,
      label: t("workflows:analyzeBook.presets.keyConcepts", "Key Concepts"),
      description: t(
        "workflows:analyzeBook.presets.keyConceptsDesc",
        "For non-fiction - ideas and frameworks"
      )
    },
    {
      value: "custom" as const,
      label: t("workflows:analyzeBook.presets.custom", "Custom"),
      description: t(
        "workflows:analyzeBook.presets.customDesc",
        "Use your own analysis prompt"
      )
    }
  ]

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-text mb-3">
          {t("workflows:analyzeBook.choosePreset", "Choose analysis type")}
        </h3>
        <Radio.Group
          value={preset}
          onChange={(e) => setPreset(e.target.value)}
          className="w-full"
        >
          <Space orientation="vertical" className="w-full">
            {presetOptions.map((option) => (
              <Radio
                key={option.value}
                value={option.value}
                className="w-full p-3 border border-border rounded-lg hover:border-primary transition-colors"
              >
                <div>
                  <span className="font-medium">{option.label}</span>
                  <p className="text-xs text-text-muted mt-0.5">
                    {option.description}
                  </p>
                </div>
              </Radio>
            ))}
          </Space>
        </Radio.Group>
      </div>

      {preset === "custom" && (
        <div>
          <label className="text-xs font-medium text-text-muted block mb-2">
            {t("workflows:analyzeBook.customPrompt", "Custom Analysis Prompt")}
          </label>
          <TextArea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            placeholder={t(
              "workflows:analyzeBook.customPromptPlaceholder",
              "Enter your analysis prompt..."
            )}
            autoSize={{ minRows: 4, maxRows: 8 }}
          />
        </div>
      )}

      <div className="border-t border-border pt-4">
        <h3 className="text-sm font-medium text-text mb-3">
          {t("workflows:analyzeBook.analysisScope", "Analysis Scope")}
        </h3>
        <Radio.Group
          value={scope}
          onChange={(e) => setScope(e.target.value)}
          optionType="button"
          buttonStyle="solid"
        >
          <Radio.Button value="whole">
            {t("workflows:analyzeBook.wholeBook", "Whole Book")}
          </Radio.Button>
          <Radio.Button value="per-chapter">
            {t("workflows:analyzeBook.perChapter", "Per Chapter")}
          </Radio.Button>
        </Radio.Group>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Process
// ─────────────────────────────────────────────────────────────────────────────

const ProcessStep: React.FC = () => {
  const { t } = useTranslation(["workflows"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)
  const setWorkflowStep = useWorkflowsStore((s) => s.setWorkflowStep)
  const setProcessing = useWorkflowsStore((s) => s.setProcessing)
  const setProcessingProgress = useWorkflowsStore((s) => s.setProcessingProgress)
  const setWorkflowError = useWorkflowsStore((s) => s.setWorkflowError)

  const [currentChapter, setCurrentChapter] = useState(0)
  const [totalChapters, setTotalChapters] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(true)
  const abortControllerRef = useRef<AbortController | null>(null)
  const advanceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const chapters = activeWorkflow?.data?.chapters as Chapter[] | undefined
  const config = activeWorkflow?.data?.analysisConfig as
    | AnalysisConfig
    | undefined
  const savedResult = activeWorkflow?.data?.analysis as
    | AnalysisResult
    | undefined

  useEffect(() => {
    if (savedResult || !chapters || !config) {
      if (savedResult) {
        setIsAnalyzing(false)
        if (advanceTimeoutRef.current) {
          clearTimeout(advanceTimeoutRef.current)
        }
        advanceTimeoutRef.current = setTimeout(() => {
          const currentId = useWorkflowsStore.getState().activeWorkflow?.workflowId
          if (currentId !== "analyze-book") return
          setWorkflowStep(4)
        }, 500)
      }
      return
    }

    const runAnalysis = async () => {
      abortControllerRef.current = new AbortController()

      try {
        setIsAnalyzing(true)
        setProcessing(
          true,
          t("workflows:analyzeBook.analyzing", "Analyzing book...")
        )

        const result: AnalysisResult = {
          wholeBook: null,
          perChapter: {}
        }

        const warnTruncation = () => {
          message.warning(
            t(
              "workflows:analyzeBook.truncateWarning",
              "Long content is truncated to the first {{limit}} characters during analysis.",
              { limit: MAX_ANALYSIS_CHARS }
            )
          )
        }

        if (config.scope === "whole") {
          // Analyze entire book at once
          setTotalChapters(1)
          setCurrentChapter(1)

          const content = chapters.map((c) => c.content).join("\n\n")
          const { truncatedContent, truncated } = truncateForAnalysis(content)
          if (truncated) {
            warnTruncation()
          }
          const analysis = await analyzeContent(
            truncatedContent,
            config,
            abortControllerRef.current.signal
          )

          result.wholeBook = analysis
          setProcessingProgress(100)
        } else {
          // Analyze per chapter
          setTotalChapters(chapters.length)
          const hasTruncatedChapter = chapters.some(
            (chapter) => chapter.content.length > MAX_ANALYSIS_CHARS
          )
          if (hasTruncatedChapter) {
            warnTruncation()
          }

          for (let i = 0; i < chapters.length; i++) {
            if (abortControllerRef.current.signal.aborted) break

            setCurrentChapter(i + 1)
            setProcessingProgress(Math.round(((i + 1) / chapters.length) * 100))

            const { truncatedContent } = truncateForAnalysis(chapters[i].content)
            const analysis = await analyzeContent(
              truncatedContent,
              config,
              abortControllerRef.current.signal
            )

            result.perChapter[chapters[i].number] = analysis
          }
        }

        updateWorkflowData({ analysis: result })
        setProcessing(false)
        setIsAnalyzing(false)

        // Auto-advance to review
        if (advanceTimeoutRef.current) {
          clearTimeout(advanceTimeoutRef.current)
        }
        advanceTimeoutRef.current = setTimeout(() => {
          const currentId = useWorkflowsStore.getState().activeWorkflow?.workflowId
          if (currentId !== "analyze-book") return
          setWorkflowStep(4)
        }, 500)
      } catch (error) {
        if ((error as Error).name === "AbortError") return

        console.error("Analysis error:", error)
        setProcessing(false)
        setIsAnalyzing(false)
        setWorkflowError(
          error instanceof Error
            ? error.message
            : t(
                "workflows:analyzeBook.errors.analysisFailed",
                "Analysis failed"
              )
        )
      }
    }

    runAnalysis()

    return () => {
      abortControllerRef.current?.abort()
      if (advanceTimeoutRef.current) {
        clearTimeout(advanceTimeoutRef.current)
      }
    }
  }, [
    chapters,
    config,
    savedResult,
    t,
    updateWorkflowData,
    setWorkflowStep,
    setProcessing,
    setProcessingProgress,
    setWorkflowError
  ])

  const progress = useWorkflowsStore((s) => s.processingProgress)

  return (
    <div className="flex flex-col items-center justify-center py-12">
      <Spin size="large" />
      <Progress
        percent={progress}
        className="w-64 mt-6"
        status="active"
        strokeColor={{ from: "rgb(var(--color-primary))", to: "rgb(var(--color-success))" }}
      />
      <p className="mt-4 text-text-muted">
        {config?.scope === "per-chapter"
          ? t(
              "workflows:analyzeBook.analyzingChapter",
              "Analyzing chapter {{current}} of {{total}}...",
              { current: currentChapter, total: totalChapters }
            )
          : t("workflows:analyzeBook.analyzing", "Analyzing book...")}
      </p>
      <Button
        className="mt-4"
        onClick={() => abortControllerRef.current?.abort()}
        danger
      >
        {t("common:cancel", "Cancel")}
      </Button>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5: Review
// ─────────────────────────────────────────────────────────────────────────────

const ReviewStep: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)
  const completeWorkflow = useWorkflowsStore((s) => s.completeWorkflow)

  const analysis = activeWorkflow?.data?.analysis as AnalysisResult | undefined
  const chapters = activeWorkflow?.data?.chapters as Chapter[] | undefined
  const config = activeWorkflow?.data?.analysisConfig as
    | AnalysisConfig
    | undefined
  const bookInfo = activeWorkflow?.data?.book as BookInfo | undefined

  const [editingSection, setEditingSection] = useState<string | null>(null)
  const [editContent, setEditContent] = useState("")

  const handleCopy = useCallback(async () => {
    const fullContent = getFullAnalysisText(analysis, chapters, config)
    try {
      await navigator.clipboard.writeText(fullContent)
      message.success(t("common:copied", "Copied to clipboard"))
    } catch {
      message.error(t("common:copyFailed", "Failed to copy"))
    }
  }, [analysis, chapters, config, t])

  const handleDownload = useCallback(() => {
    const fullContent = getFullAnalysisText(analysis, chapters, config)
    const blob = new Blob([fullContent], { type: "text/markdown" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `analysis-${bookInfo?.title || "book"}.md`
    a.click()
    URL.revokeObjectURL(url)
  }, [analysis, chapters, config, bookInfo])

  const handleSaveEdit = (section: string) => {
    if (!analysis) return

    if (section === "whole") {
      updateWorkflowData({
        analysis: { ...analysis, wholeBook: editContent }
      })
    } else {
      const chapterNum = parseInt(section.replace("chapter-", ""))
      updateWorkflowData({
        analysis: {
          ...analysis,
          perChapter: { ...analysis.perChapter, [chapterNum]: editContent }
        }
      })
    }

    setEditingSection(null)
    setEditContent("")
  }

  const startEdit = (section: string, content: string) => {
    setEditingSection(section)
    setEditContent(content)
  }

  if (!analysis) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-text-muted">
          {t("workflows:analyzeBook.noAnalysis", "No analysis available")}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Alert
        variant="success"
        icon={<CheckCircle className="h-4 w-4" />}
        title={t(
          "workflows:analyzeBook.analysisComplete",
          "Analysis Complete"
        )}
      >
        {t(
          "workflows:analyzeBook.analysisCompleteDesc",
          "Your book analysis is ready. You can edit, save, or export the results below."
        )}
      </Alert>

      {config?.scope === "whole" && analysis.wholeBook && (
        <AnalysisSection
          title={t("workflows:analyzeBook.overallAnalysis", "Overall Analysis")}
          content={analysis.wholeBook}
          sectionKey="whole"
          isEditing={editingSection === "whole"}
          editContent={editContent}
          onStartEdit={() => startEdit("whole", analysis.wholeBook || "")}
          onSaveEdit={() => handleSaveEdit("whole")}
          onCancelEdit={() => setEditingSection(null)}
          onEditChange={setEditContent}
        />
      )}

      {config?.scope === "per-chapter" && (
        <Collapse defaultActiveKey={["1"]} ghost>
          {chapters?.map((chapter) => (
            <Panel
              key={chapter.number}
              header={
                <span className="font-medium">
                  Chapter {chapter.number}: {chapter.title}
                </span>
              }
            >
              <AnalysisSection
                title=""
                content={analysis.perChapter[chapter.number] || ""}
                sectionKey={`chapter-${chapter.number}`}
                isEditing={editingSection === `chapter-${chapter.number}`}
                editContent={editContent}
                onStartEdit={() =>
                  startEdit(
                    `chapter-${chapter.number}`,
                    analysis.perChapter[chapter.number] || ""
                  )
                }
                onSaveEdit={() => handleSaveEdit(`chapter-${chapter.number}`)}
                onCancelEdit={() => setEditingSection(null)}
                onEditChange={setEditContent}
              />
            </Panel>
          ))}
        </Collapse>
      )}

      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-4">
        <Button icon={<Save className="h-4 w-4" />} type="primary">
          {t("workflows:analyzeBook.saveToKB", "Save to Knowledge Base")}
        </Button>
        <Button icon={<Copy className="h-4 w-4" />} onClick={handleCopy}>
          {t("common:copy", "Copy")}
        </Button>
        <Button icon={<Download className="h-4 w-4" />} onClick={handleDownload}>
          {t("workflows:analyzeBook.exportMD", "Export as Markdown")}
        </Button>
        <Button icon={<MessageSquare className="h-4 w-4" />}>
          {t("workflows:analyzeBook.sendToChat", "Send to Chat")}
        </Button>
      </div>

      <div className="border-t border-border pt-4">
        <h4 className="text-sm font-medium text-text mb-2">
          {t("workflows:analyzeBook.whatsNext", "What's next?")}
        </h4>
        <div className="flex flex-wrap gap-2">
          <Button size="small">
            {t("workflows:analyzeBook.createQuiz", "Create a quiz")}
          </Button>
          <Button size="small">
            {t("workflows:analyzeBook.makeFlashcards", "Make flashcards")}
          </Button>
          <Button size="small">
            {t("workflows:analyzeBook.askQuestions", "Ask questions")}
          </Button>
        </div>
      </div>

      <div className="flex justify-end pt-4">
        <Button type="primary" onClick={completeWorkflow}>
          {t("common:done", "Done")}
        </Button>
      </div>
    </div>
  )
}

interface AnalysisSectionProps {
  title: string
  content: string
  sectionKey: string
  isEditing: boolean
  editContent: string
  onStartEdit: () => void
  onSaveEdit: () => void
  onCancelEdit: () => void
  onEditChange: (content: string) => void
}

const AnalysisSection: React.FC<AnalysisSectionProps> = ({
  title,
  content,
  sectionKey,
  isEditing,
  editContent,
  onStartEdit,
  onSaveEdit,
  onCancelEdit,
  onEditChange
}) => {
  const { t } = useTranslation(["common"])

  if (isEditing) {
    return (
      <div className="space-y-2">
        {title && (
          <h4 className="text-sm font-medium text-text">{title}</h4>
        )}
        <TextArea
          value={editContent}
          onChange={(e) => onEditChange(e.target.value)}
          autoSize={{ minRows: 6, maxRows: 20 }}
          className="font-mono text-sm"
        />
        <div className="flex gap-2">
          <Button size="small" type="primary" onClick={onSaveEdit}>
            {t("common:save", "Save")}
          </Button>
          <Button size="small" onClick={onCancelEdit}>
            {t("common:cancel", "Cancel")}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {title && (
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium text-text">{title}</h4>
          <Button
            size="small"
            icon={<Edit2 className="h-3 w-3" />}
            onClick={onStartEdit}
          >
            {t("common:edit", "Edit")}
          </Button>
        </div>
      )}
      {!title && (
        <div className="flex justify-end">
          <Button
            size="small"
            icon={<Edit2 className="h-3 w-3" />}
            onClick={onStartEdit}
          >
            {t("common:edit", "Edit")}
          </Button>
        </div>
      )}
      <div className="prose prose-sm dark:prose-invert max-w-none bg-surface p-4 rounded-lg">
        <pre className="whitespace-pre-wrap text-sm font-sans">{content}</pre>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const TEXT_FILE_EXTENSIONS = new Set([".txt", ".md"])

const getFileExtension = (file: File) => {
  const name = file.name.toLowerCase()
  const dotIndex = name.lastIndexOf(".")
  return dotIndex >= 0 ? name.slice(dotIndex) : ""
}

const readTextFile = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })

const extractParsedText = (data: any): string | null => {
  if (!data) return null
  if (typeof data === "string") return data
  if (typeof data?.content?.text === "string") return data.content.text
  if (typeof data?.content === "string") return data.content
  if (typeof data?.text === "string") return data.text
  if (typeof data?.result?.content?.text === "string") return data.result.content.text
  if (typeof data?.result?.text === "string") return data.result.text
  if (typeof data?.result?.content === "string") return data.result.content
  if (typeof data?.media?.content?.text === "string") return data.media.content.text
  if (typeof data?.media?.text === "string") return data.media.text
  return null
}

const parseFileRemotely = async (file: File): Promise<string> => {
  const uploadType = inferUploadMediaTypeFromFile(file.name, file.type)
  if (!["pdf", "ebook", "document"].includes(uploadType)) {
    console.error("Unsupported file type for remote parsing:", file.name, file.type)
    throw new Error("Unsupported file type.")
  }
  const path = getProcessPathForType(uploadType)

  try {
    const buffer = await file.arrayBuffer()
    const data = new Uint8Array(buffer)
    const response = await bgUpload<any>({
      path,
      method: "POST",
      file: {
        name: file.name,
        type: file.type || "application/octet-stream",
        data
      }
    })
    const parsed = extractParsedText(response)
    if (!parsed) {
      console.error("No parsed content returned for file:", file.name, response)
      throw new Error("No text content returned from the file parser.")
    }
    return parsed
  } catch (error) {
    console.error("Failed to parse file with backend:", file.name, error)
    throw error
  }
}

async function readFileContent(file: File): Promise<string> {
  const extension = getFileExtension(file)
  if (extension && !SUPPORTED_FILE_TYPES.includes(extension)) {
    console.error("Unsupported file type:", file.name, file.type)
    throw new Error("Unsupported file type.")
  }

  if (TEXT_FILE_EXTENSIONS.has(extension) || file.type.startsWith("text/")) {
    return await readTextFile(file)
  }

  return await parseFileRemotely(file)
}

const truncateForAnalysis = (content: string) => {
  if (content.length <= MAX_ANALYSIS_CHARS) {
    return { truncatedContent: content, truncated: false }
  }

  return {
    truncatedContent: content.slice(0, MAX_ANALYSIS_CHARS),
    truncated: true
  }
}

async function analyzeContent(
  content: string,
  config: AnalysisConfig,
  signal: AbortSignal
): Promise<string> {
  const prompt =
    config.preset === "custom"
      ? config.customPrompt
      : ANALYSIS_PRESETS[config.preset]?.prompt || ""

  try {
    const data = await bgRequest<any>({
      path: "/api/v1/chat/completions",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        messages: [
          {
            role: "system",
            content: `You are a helpful assistant that analyzes books and documents. ${prompt}`
          },
          {
            role: "user",
            content: `Please analyze the following content:\n\n${content}`
          }
        ]
      },
      abortSignal: signal
    })
    return (
      data.choices?.[0]?.message?.content ||
      "Analysis could not be generated. Please ensure your tldw server is running."
    )
  } catch (error) {
    const message = error instanceof Error ? error.message : ""
    if (signal.aborted || message === "Aborted" || (error as Error).name === "AbortError") {
      throw new DOMException("Aborted", "AbortError")
    }
    console.error("Analysis API error:", error)
    throw error
  }
}

function getFullAnalysisText(
  analysis: AnalysisResult | undefined,
  chapters: Chapter[] | undefined,
  config: AnalysisConfig | undefined
): string {
  if (!analysis) return ""

  let text = "# Book Analysis\n\n"

  if (config?.scope === "whole" && analysis.wholeBook) {
    text += analysis.wholeBook
  } else if (chapters) {
    for (const chapter of chapters) {
      const chapterAnalysis = analysis.perChapter[chapter.number]
      if (chapterAnalysis) {
        text += `## Chapter ${chapter.number}: ${chapter.title}\n\n`
        text += chapterAnalysis + "\n\n"
      }
    }
  }

  return text
}
