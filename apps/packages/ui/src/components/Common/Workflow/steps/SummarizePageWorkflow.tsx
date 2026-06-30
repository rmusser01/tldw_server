import React, { useState, useCallback, useEffect, useRef } from "react"
import { Input, Radio, Button, Space, Spin, message } from "antd"
import { Copy, Download, MessageSquare, Globe } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import { useWorkflowsStore } from "@/store/workflows"
import { WizardShell } from "../WizardShell"
import { SUMMARIZE_PAGE_WORKFLOW } from "../workflow-definitions"

const { TextArea } = Input

type SummaryStyle = "brief" | "detailed" | "bullets"

/**
 * SummarizePageWorkflow
 *
 * A complete workflow for summarizing the current webpage.
 * Steps:
 * 1. Capture - Auto-capture page content
 * 2. Options - Choose summary style (optional)
 * 3. Result - View and act on summary
 */
export const SummarizePageWorkflow: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)

  if (!activeWorkflow || activeWorkflow.workflowId !== "summarize-page") {
    return null
  }

  return (
    <WizardShell workflow={SUMMARIZE_PAGE_WORKFLOW}>
      <SummarizePageStepContent />
    </WizardShell>
  )
}

const SummarizePageStepContent: React.FC = () => {
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const stepIndex = activeWorkflow?.currentStepIndex ?? 0

  switch (stepIndex) {
    case 0:
      return <CaptureStep />
    case 1:
      return <OptionsStep />
    case 2:
      return <ResultStep />
    default:
      return null
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Capture
// ─────────────────────────────────────────────────────────────────────────────

const CaptureStep: React.FC = () => {
  const { t } = useTranslation(["workflows"])
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)
  const setWorkflowStep = useWorkflowsStore((s) => s.setWorkflowStep)
  const setProcessing = useWorkflowsStore((s) => s.setProcessing)
  const setWorkflowError = useWorkflowsStore((s) => s.setWorkflowError)

  const advanceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [pageInfo, setPageInfo] = useState<{
    title: string
    url: string
    content: string
  } | null>(null)
  const [isCapturing, setIsCapturing] = useState(true)

  useEffect(() => {
    const capturePageContent = async () => {
      try {
        setProcessing(true, t("workflows:summarizePage.capturing", "Capturing page content..."))

        // Get current tab info
        const tabs = await chrome.tabs.query({
          active: true,
          currentWindow: true
        })
        const tab = tabs[0]

        if (!tab?.id || !tab.url) {
          throw new Error(
            t("workflows:summarizePage.errors.noTab", "No active tab found")
          )
        }

        // Execute script to get page content
        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            // Get main content, preferring article or main elements
            const article =
              document.querySelector("article") ||
              document.querySelector("main") ||
              document.body

            // Remove script and style elements for cleaner text
            const clone = article.cloneNode(true) as HTMLElement
            clone
              .querySelectorAll("script, style, nav, footer, header")
              .forEach((el) => el.remove())

            return {
              title: document.title,
              content: clone.innerText?.trim() || ""
            }
          }
        })

        const result = results[0]?.result
        if (!result?.content) {
          throw new Error(
            t(
              "workflows:summarizePage.errors.noContent",
              "Could not extract page content"
            )
          )
        }

        const captured = {
          title: result.title || tab.title || "Untitled",
          url: tab.url,
          content: result.content.slice(0, 50000) // Limit content size
        }

        setPageInfo(captured)
        updateWorkflowData({ pageInfo: captured })
        setProcessing(false)
        setIsCapturing(false)

        // Auto-advance to next step after a brief delay
        if (advanceTimeoutRef.current) {
          clearTimeout(advanceTimeoutRef.current)
        }
        advanceTimeoutRef.current = setTimeout(() => {
          const currentId = useWorkflowsStore.getState().activeWorkflow?.workflowId
          if (currentId !== "summarize-page") return
          setWorkflowStep(1)
        }, 500)
      } catch (error) {
        console.error("Capture error:", error)
        setProcessing(false)
        setIsCapturing(false)
        setWorkflowError(
          error instanceof Error
            ? error.message
            : t("workflows:summarizePage.errors.captureFailed", "Failed to capture page")
        )
      }
    }

    capturePageContent()

    return () => {
      if (advanceTimeoutRef.current) {
        clearTimeout(advanceTimeoutRef.current)
      }
    }
  }, [t, updateWorkflowData, setWorkflowStep, setProcessing, setWorkflowError])

  if (isCapturing) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spin size="large" />
        <p className="mt-4 text-text-muted">
          {t("workflows:summarizePage.capturing", "Capturing page content...")}
        </p>
      </div>
    )
  }

  if (pageInfo) {
    return (
      <div className="space-y-4">
        <Alert
          variant="success"
          title={t("workflows:summarizePage.captured", "Page captured")}
        >
          <div className="mt-2">
            <p className="font-medium">{pageInfo.title}</p>
            <p className="text-xs text-text-muted flex items-center gap-1 mt-1">
              <Globe className="h-3 w-3" />
              {pageInfo.url}
            </p>
          </div>
        </Alert>
        <p className="text-sm text-text-muted">
          {t(
            "workflows:summarizePage.contentPreview",
            "Content length: {{length}} characters",
            { length: pageInfo.content.length.toLocaleString() }
          )}
        </p>
      </div>
    )
  }

  return null
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Options
// ─────────────────────────────────────────────────────────────────────────────

const OptionsStep: React.FC = () => {
  const { t } = useTranslation(["workflows"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)

  const currentStyle = (activeWorkflow?.data?.summaryStyle as SummaryStyle) || "brief"

  const handleStyleChange = (style: SummaryStyle) => {
    updateWorkflowData({ summaryStyle: style })
  }

  const styleOptions = [
    {
      value: "brief" as const,
      label: t("workflows:summarizePage.styles.brief", "Brief"),
      description: t(
        "workflows:summarizePage.styles.briefDesc",
        "2-3 sentence overview"
      )
    },
    {
      value: "detailed" as const,
      label: t("workflows:summarizePage.styles.detailed", "Detailed"),
      description: t(
        "workflows:summarizePage.styles.detailedDesc",
        "Comprehensive summary with key points"
      )
    },
    {
      value: "bullets" as const,
      label: t("workflows:summarizePage.styles.bullets", "Bullet Points"),
      description: t(
        "workflows:summarizePage.styles.bulletsDesc",
        "Key points as a list"
      )
    }
  ]

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-text mb-3">
          {t("workflows:summarizePage.chooseStyle", "Choose summary style")}
        </h3>
        <Radio.Group
          value={currentStyle}
          onChange={(e) => handleStyleChange(e.target.value)}
          className="w-full"
        >
          <Space orientation="vertical" className="w-full">
            {styleOptions.map((option) => (
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
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Result
// ─────────────────────────────────────────────────────────────────────────────

const ResultStep: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)
  const setProcessing = useWorkflowsStore((s) => s.setProcessing)
  const setProcessingProgress = useWorkflowsStore((s) => s.setProcessingProgress)
  const setWorkflowError = useWorkflowsStore((s) => s.setWorkflowError)
  const isProcessing = useWorkflowsStore((s) => s.isProcessing)

  const [summary, setSummary] = useState<string>(
    (activeWorkflow?.data?.summary as string) || ""
  )
  const [isGenerating, setIsGenerating] = useState(!summary)

  useEffect(() => {
    if (summary || !activeWorkflow?.data?.pageInfo) return

    const generateSummary = async () => {
      try {
        setIsGenerating(true)
        setProcessing(
          true,
          t("workflows:summarizePage.generating", "Generating summary...")
        )

        const pageInfo = activeWorkflow.data.pageInfo as {
          title: string
          url: string
          content: string
        }
        const style = (activeWorkflow.data.summaryStyle as SummaryStyle) || "brief"

        // Simulate progress for better UX
        let progress = 0
        const progressInterval = setInterval(() => {
          progress = Math.min(progress + 10, 90)
          setProcessingProgress(progress)
        }, 500)

        // Build prompt based on style
        let stylePrompt = ""
        switch (style) {
          case "brief":
            stylePrompt = "Provide a brief 2-3 sentence summary."
            break
          case "detailed":
            stylePrompt =
              "Provide a detailed summary covering all key points and main arguments."
            break
          case "bullets":
            stylePrompt =
              "Provide a summary as a bulleted list of key points (5-10 points)."
            break
        }

        // Call the tldw server for summarization
        // This is a placeholder - integrate with actual API
        const response = await fetch("/api/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [
              {
                role: "system",
                content: `You are a helpful assistant that summarizes web content. ${stylePrompt}`
              },
              {
                role: "user",
                content: `Please summarize this webpage:\n\nTitle: ${pageInfo.title}\nURL: ${pageInfo.url}\n\nContent:\n${pageInfo.content.slice(0, 10000)}`
              }
            ]
          })
        })

        clearInterval(progressInterval)

        if (!response.ok) {
          throw new Error("Failed to generate summary")
        }

        const data = await response.json()
        const generatedSummary =
          data.choices?.[0]?.message?.content ||
          t(
            "workflows:summarizePage.mockSummary",
            "This is a placeholder summary. The actual summary would be generated by your AI backend."
          )

        setSummary(generatedSummary)
        updateWorkflowData({ summary: generatedSummary })
        setProcessingProgress(100)
        setProcessing(false)
        setIsGenerating(false)
      } catch (error) {
        console.error("Summary generation error:", error)
        // Provide a fallback summary for demo purposes
        const fallbackSummary = t(
          "workflows:summarizePage.demoSummary",
          "Unable to connect to the summarization service. Please ensure your tldw server is running and configured correctly."
        )
        setSummary(fallbackSummary)
        updateWorkflowData({ summary: fallbackSummary })
        setProcessing(false)
        setIsGenerating(false)
      }
    }

    generateSummary()
  }, [
    summary,
    activeWorkflow,
    t,
    updateWorkflowData,
    setProcessing,
    setProcessingProgress,
    setWorkflowError
  ])

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(summary)
      message.success(t("common:copied", "Copied to clipboard"))
    } catch {
      message.error(t("common:copyFailed", "Failed to copy"))
    }
  }, [summary, t])

  const handleDownload = useCallback(() => {
    const pageInfo = activeWorkflow?.data?.pageInfo as { title: string } | undefined
    const blob = new Blob([summary], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `summary-${pageInfo?.title || "page"}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [summary, activeWorkflow])

  if (isGenerating) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spin size="large" />
        <p className="mt-4 text-text-muted">
          {t("workflows:summarizePage.generating", "Generating summary...")}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-text mb-2">
          {t("workflows:summarizePage.yourSummary", "Your Summary")}
        </h3>
        <TextArea
          value={summary}
          onChange={(e) => {
            setSummary(e.target.value)
            updateWorkflowData({ summary: e.target.value })
          }}
          autoSize={{ minRows: 6, maxRows: 12 }}
          className="font-mono text-sm"
        />
      </div>

      <div className="flex items-center gap-2">
        <Button icon={<Copy className="h-4 w-4" />} onClick={handleCopy}>
          {t("common:copy", "Copy")}
        </Button>
        <Button icon={<Download className="h-4 w-4" />} onClick={handleDownload}>
          {t("common:download", "Download")}
        </Button>
        <Button
          icon={<MessageSquare className="h-4 w-4" />}
          onClick={() => {
            // TODO: Open chat with summary as context
            message.info(
              t(
                "workflows:summarizePage.chatComingSoon",
                "Chat feature coming soon"
              )
            )
          }}
        >
          {t("workflows:summarizePage.askFollowUp", "Ask Follow-up")}
        </Button>
      </div>
    </div>
  )
}
