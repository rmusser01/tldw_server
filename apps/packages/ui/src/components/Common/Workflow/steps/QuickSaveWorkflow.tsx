import React, { useState, useCallback, useEffect, useRef } from "react"
import { Input, Button, Select, Tag, Spin, message } from "antd"
import { Save, CheckCircle, FolderOpen, Tag as TagIcon, Globe } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import { useWorkflowsStore } from "@/store/workflows"
import { WizardShell } from "../WizardShell"
import { QUICK_SAVE_WORKFLOW } from "../workflow-definitions"

const { TextArea } = Input

/**
 * QuickSaveWorkflow
 *
 * A streamlined workflow for saving content (selected text or full page) to notes.
 * Steps:
 * 1. Capture - Auto-capture selected text or page content
 * 2. Details - Add title, tags, folder (optional)
 * 3. Confirm - Save and show confirmation
 */
export const QuickSaveWorkflow: React.FC = () => {
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)

  if (!activeWorkflow || activeWorkflow.workflowId !== "quick-save") {
    return null
  }

  return (
    <WizardShell workflow={QUICK_SAVE_WORKFLOW}>
      <QuickSaveStepContent />
    </WizardShell>
  )
}

const QuickSaveStepContent: React.FC = () => {
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const stepIndex = activeWorkflow?.currentStepIndex ?? 0

  switch (stepIndex) {
    case 0:
      return <CaptureStep />
    case 1:
      return <DetailsStep />
    case 2:
      return <ConfirmStep />
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

  const [capturedContent, setCapturedContent] = useState<{
    type: "selection" | "page"
    title: string
    url: string
    content: string
  } | null>(null)
  const [isCapturing, setIsCapturing] = useState(true)

  useEffect(() => {
    const captureContent = async () => {
      try {
        setProcessing(
          true,
          t("workflows:quickSave.capturing", "Capturing content...")
        )

        // Get current tab info
        const tabs = await chrome.tabs.query({
          active: true,
          currentWindow: true
        })
        const tab = tabs[0]

        if (!tab?.id || !tab.url) {
          throw new Error(
            t("workflows:quickSave.errors.noTab", "No active tab found")
          )
        }

        // Try to get selected text first, fall back to page content
        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            const selection = window.getSelection()?.toString()?.trim()
            if (selection && selection.length > 10) {
              return {
                type: "selection" as const,
                content: selection
              }
            }

            // Fall back to page content
            const article =
              document.querySelector("article") ||
              document.querySelector("main") ||
              document.body

            const clone = article.cloneNode(true) as HTMLElement
            clone
              .querySelectorAll("script, style, nav, footer, header")
              .forEach((el) => el.remove())

            return {
              type: "page" as const,
              content: clone.innerText?.trim() || ""
            }
          }
        })

        const result = results[0]?.result
        if (!result?.content) {
          throw new Error(
            t(
              "workflows:quickSave.errors.noContent",
              "Could not capture content"
            )
          )
        }

        const captured = {
          type: result.type,
          title: tab.title || "Untitled",
          url: tab.url,
          content: result.content.slice(0, 50000)
        }

        setCapturedContent(captured)
        updateWorkflowData({
          capturedContent: captured,
          noteTitle: captured.title,
          noteTags: [],
          noteFolder: "default"
        })
        setProcessing(false)
        setIsCapturing(false)

        // Auto-advance to next step
        if (advanceTimeoutRef.current) {
          clearTimeout(advanceTimeoutRef.current)
        }
        advanceTimeoutRef.current = setTimeout(() => {
          const currentId = useWorkflowsStore.getState().activeWorkflow?.workflowId
          if (currentId !== "quick-save") return
          setWorkflowStep(1)
        }, 500)
      } catch (error) {
        console.error("Capture error:", error)
        setProcessing(false)
        setIsCapturing(false)
        setWorkflowError(
          error instanceof Error
            ? error.message
            : t("workflows:quickSave.errors.captureFailed", "Failed to capture content")
        )
      }
    }

    captureContent()

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
          {t("workflows:quickSave.capturing", "Capturing content...")}
        </p>
      </div>
    )
  }

  if (capturedContent) {
    return (
      <div className="space-y-4">
        <Alert
          variant="success"
          title={
            capturedContent.type === "selection"
              ? t("workflows:quickSave.selectionCaptured", "Selection captured")
              : t("workflows:quickSave.pageCaptured", "Page captured")
          }
        >
          <div className="mt-2">
            <p className="font-medium">{capturedContent.title}</p>
            <p className="text-xs text-text-muted flex items-center gap-1 mt-1">
              <Globe className="h-3 w-3" />
              {capturedContent.url}
            </p>
          </div>
        </Alert>
        <div className="bg-surface rounded-lg p-3">
          <p className="text-sm text-text-muted line-clamp-4">
            {capturedContent.content.slice(0, 300)}
            {capturedContent.content.length > 300 && "..."}
          </p>
        </div>
      </div>
    )
  }

  return null
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Details
// ─────────────────────────────────────────────────────────────────────────────

const DetailsStep: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const updateWorkflowData = useWorkflowsStore((s) => s.updateWorkflowData)

  const title = (activeWorkflow?.data?.noteTitle as string) || ""
  const tags = (activeWorkflow?.data?.noteTags as string[]) || []
  const folder = (activeWorkflow?.data?.noteFolder as string) || "default"

  const [newTag, setNewTag] = useState("")

  // Mock folders - in real implementation, fetch from notes service
  const folderOptions = [
    { value: "default", label: t("workflows:quickSave.folders.default", "Default") },
    { value: "research", label: t("workflows:quickSave.folders.research", "Research") },
    { value: "reading", label: t("workflows:quickSave.folders.reading", "Reading List") },
    { value: "work", label: t("workflows:quickSave.folders.work", "Work") }
  ]

  const handleAddTag = () => {
    if (newTag.trim() && !tags.includes(newTag.trim())) {
      updateWorkflowData({ noteTags: [...tags, newTag.trim()] })
      setNewTag("")
    }
  }

  const handleRemoveTag = (tagToRemove: string) => {
    updateWorkflowData({ noteTags: tags.filter((t) => t !== tagToRemove) })
  }

  return (
    <div className="space-y-4">
      {/* Title */}
      <div>
        <label className="text-sm font-medium text-text block mb-2">
          {t("workflows:quickSave.titleLabel", "Title")}
        </label>
        <Input
          value={title}
          onChange={(e) => updateWorkflowData({ noteTitle: e.target.value })}
          placeholder={t("workflows:quickSave.titlePlaceholder", "Enter a title")}
        />
      </div>

      {/* Folder */}
      <div>
        <label className="text-sm font-medium text-text block mb-2">
          <FolderOpen className="h-4 w-4 inline mr-1" />
          {t("workflows:quickSave.folder", "Folder")}
        </label>
        <Select
          value={folder}
          onChange={(value) => updateWorkflowData({ noteFolder: value })}
          options={folderOptions}
          className="w-full"
        />
      </div>

      {/* Tags */}
      <div>
        <label className="text-sm font-medium text-text block mb-2">
          <TagIcon className="h-4 w-4 inline mr-1" />
          {t("workflows:quickSave.tags", "Tags")}
        </label>
        <div className="flex gap-2 mb-2 flex-wrap">
          {tags.map((tag) => (
            <Tag
              key={tag}
              closable
              onClose={() => handleRemoveTag(tag)}
              className="mb-1"
            >
              {tag}
            </Tag>
          ))}
        </div>
        <Input
          value={newTag}
          onChange={(e) => setNewTag(e.target.value)}
          onPressEnter={handleAddTag}
          placeholder={t("workflows:quickSave.addTag", "Add a tag and press Enter")}
          suffix={
            <Button type="text" size="small" onClick={handleAddTag}>
              {t("common:add", "Add")}
            </Button>
          }
        />
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Confirm
// ─────────────────────────────────────────────────────────────────────────────

const ConfirmStep: React.FC = () => {
  const { t } = useTranslation(["workflows", "common"])
  const activeWorkflow = useWorkflowsStore((s) => s.activeWorkflow)
  const setProcessing = useWorkflowsStore((s) => s.setProcessing)
  const setProcessingProgress = useWorkflowsStore((s) => s.setProcessingProgress)
  const completeWorkflow = useWorkflowsStore((s) => s.completeWorkflow)

  const [isSaving, setIsSaving] = useState(true)
  const [isSaved, setIsSaved] = useState(false)

  useEffect(() => {
    const saveNote = async () => {
      try {
        setProcessing(true, t("workflows:quickSave.saving", "Saving note..."))
        setProcessingProgress(30)

        // Simulate save operation
        // In real implementation, call notes service API
        await new Promise((resolve) => setTimeout(resolve, 1000))
        setProcessingProgress(70)

        // TODO: Actual save to notes service
        // const noteData = {
        //   title: activeWorkflow?.data?.noteTitle,
        //   content: activeWorkflow?.data?.capturedContent?.content,
        //   url: activeWorkflow?.data?.capturedContent?.url,
        //   tags: activeWorkflow?.data?.noteTags,
        //   folder: activeWorkflow?.data?.noteFolder,
        // }
        // await notesService.createNote(noteData)

        await new Promise((resolve) => setTimeout(resolve, 500))
        setProcessingProgress(100)
        setProcessing(false)
        setIsSaving(false)
        setIsSaved(true)

        message.success(t("workflows:quickSave.saved", "Note saved!"))
      } catch (error) {
        console.error("Save error:", error)
        setProcessing(false)
        setIsSaving(false)
        message.error(t("workflows:quickSave.saveFailed", "Failed to save note"))
      }
    }

    if (!isSaved) {
      saveNote()
    }
  }, [t, activeWorkflow, setProcessing, setProcessingProgress, isSaved])

  const handleViewNotes = useCallback(() => {
    // Open notes page
    chrome.tabs.create({ url: chrome.runtime.getURL("options.html#/notes") })
    completeWorkflow()
  }, [completeWorkflow])

  const handleDone = useCallback(() => {
    completeWorkflow()
  }, [completeWorkflow])

  if (isSaving) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Spin size="large" />
        <p className="mt-4 text-text-muted">
          {t("workflows:quickSave.saving", "Saving note...")}
        </p>
      </div>
    )
  }

  if (isSaved) {
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <div className="w-16 h-16 rounded-full bg-success/20 flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="h-8 w-8 text-success" />
          </div>
          <h3 className="text-lg font-medium text-text">
            {t("workflows:quickSave.savedTitle", "Note Saved!")}
          </h3>
          <p className="text-sm text-text-muted mt-2">
            {t(
              "workflows:quickSave.savedDescription",
              "Your content has been saved to your notes."
            )}
          </p>
        </div>

        <div className="bg-surface rounded-lg p-4">
          <div className="text-sm">
            <p className="font-medium text-text">
              {activeWorkflow?.data?.noteTitle as string}
            </p>
            <p className="text-text-muted mt-1">
              {t("workflows:quickSave.folder", "Folder")}:{" "}
              {activeWorkflow?.data?.noteFolder as string}
            </p>
            {((activeWorkflow?.data?.noteTags as string[]) || []).length > 0 && (
              <div className="mt-2 flex gap-1 flex-wrap">
                {((activeWorkflow?.data?.noteTags as string[]) || []).map(
                  (tag) => (
                    <Tag key={tag} className="text-xs">
                      {tag}
                    </Tag>
                  )
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex justify-center gap-3">
          <Button onClick={handleViewNotes}>
            {t("workflows:quickSave.viewNotes", "View Notes")}
          </Button>
          <Button type="primary" onClick={handleDone}>
            {t("common:done", "Done")}
          </Button>
        </div>
      </div>
    )
  }

  return null
}
