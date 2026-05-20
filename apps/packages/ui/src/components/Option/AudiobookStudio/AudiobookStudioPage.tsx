import React, { useEffect, useState, useCallback, useMemo, useRef } from "react"
import { Tabs, Typography, Input, Space, Tag, Button, message, Tooltip } from "antd"
import { DismissibleBetaAlert } from "@/components/Common/DismissibleBetaAlert"
import { useTranslation } from "react-i18next"
import {
  FileText,
  ListOrdered,
  Play,
  Download,
  Headphones,
  Save,
  FolderOpen,
  Plus,
  Settings,
  Check
} from "lucide-react"
import { PageShell } from "@/components/Common/PageShell"
import { useAudiobookStudioStore } from "@/store/audiobook-studio"
import { useCurrentProject } from "@/hooks/useAudiobookProjects"
import { TextEditor } from "./ContentInput/TextEditor"
import { ChapterList } from "./ChapterEditor/ChapterList"
import { GenerationPanel } from "./Generation/GenerationPanel"
import { OutputPanel } from "./Output/OutputPanel"
import { ProjectListView } from "./ProjectManagement/ProjectListView"
import { ProjectMetadataForm } from "./ProjectManagement/ProjectMetadataForm"

const { Title, Text } = Typography

// Auto-save interval in milliseconds
const AUTO_SAVE_INTERVAL = 30000 // 30 seconds
const DEBOUNCE_SAVE_DELAY = 5000 // 5 seconds after changes
const SAVED_JUST_NOW_WINDOW = 60000 // 1 minute

export const AudiobookStudioPage: React.FC = () => {
  const { t } = useTranslation(["audiobook", "common"])
  const [activeTab, setActiveTab] = useState("content")
  const [showProjectList, setShowProjectList] = useState(false)
  const [showMetadataForm, setShowMetadataForm] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const [hasUnsaved, setHasUnsaved] = useState(false)
  const [saveStatusNow, setSaveStatusNow] = useState(() => Date.now())

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null)
  const previousProjectIdRef = useRef<string | null | undefined>(undefined)

  const chapters = useAudiobookStudioStore((s) => s.chapters)
  const rawContent = useAudiobookStudioStore((s) => s.rawContent)
  const isGenerating = useAudiobookStudioStore((s) => s.isGenerating)
  const projectTitle = useAudiobookStudioStore((s) => s.projectTitle)
  const projectAuthor = useAudiobookStudioStore((s) => s.projectAuthor)
  const projectDescription = useAudiobookStudioStore(
    (s) => s.projectDescription
  )
  const projectCoverImageUrl = useAudiobookStudioStore(
    (s) => s.projectCoverImageUrl
  )
  const projectId = useAudiobookStudioStore((s) => s.currentProjectId)
  const setProjectTitle = useAudiobookStudioStore((s) => s.setProjectTitle)
  const revokeAllAudioUrls = useAudiobookStudioStore((s) => s.revokeAllAudioUrls)

  const { saveProject, createNewProject } = useCurrentProject()

  const completedCount = chapters.filter((ch) => ch.status === "completed").length
  const pendingCount = chapters.filter(
    (ch) => ch.status === "pending" || ch.status === "error"
  ).length
  const isRecentlySaved = useMemo(
    () =>
      lastSaved
        ? saveStatusNow - lastSaved.getTime() < SAVED_JUST_NOW_WINDOW
        : false,
    [lastSaved, saveStatusNow]
  )
  const saveStatusLabel = useMemo(() => {
    if (isSaving) return t("audiobook:saveStatus.saving", "Saving...")
    if (!projectId && !lastSaved) {
      return t("audiobook:saveStatus.draftNotSaved", "Draft not saved")
    }
    if (hasUnsaved) return t("audiobook:saveStatus.unsaved", "Unsaved changes")
    if (isRecentlySaved) {
      return t("audiobook:saveStatus.savedJustNow", "Saved just now")
    }
    return t("audiobook:saveStatus.saved", "Saved")
  }, [isSaving, projectId, lastSaved, hasUnsaved, isRecentlySaved, t])
  const saveStatusType = useMemo<"warning" | "secondary">(
    () => (!projectId && !lastSaved) || hasUnsaved ? "warning" : "secondary",
    [projectId, lastSaved, hasUnsaved]
  )

  // Manual save
  const handleSave = useCallback(async () => {
    setIsSaving(true)
    try {
      await saveProject()
      const savedAt = new Date()
      setLastSaved(savedAt)
      setSaveStatusNow(savedAt.getTime())
      setHasUnsaved(false)
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
      message.success(t("audiobook:saved", "Project saved"))
    } catch (err) {
      console.error("Save failed:", err)
      message.error(t("audiobook:saveError", "Failed to save project"))
    } finally {
      setIsSaving(false)
    }
  }, [saveProject, t])

  useEffect(() => {
    if (!lastSaved) return

    const elapsed = Date.now() - lastSaved.getTime()
    const timeoutDelay = Math.max(SAVED_JUST_NOW_WINDOW - elapsed, 0)

    if (timeoutDelay === 0) {
      setSaveStatusNow(Date.now())
      return
    }

    const timeout = setTimeout(() => {
      setSaveStatusNow(Date.now())
    }, timeoutDelay)

    return () => clearTimeout(timeout)
  }, [lastSaved])

  useEffect(() => {
    if (previousProjectIdRef.current === undefined) {
      previousProjectIdRef.current = projectId
      return
    }

    if (previousProjectIdRef.current === projectId) return

    previousProjectIdRef.current = projectId
    setLastSaved(null)
    setSaveStatusNow(Date.now())
    setHasUnsaved(false)
    setIsSaving(false)

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
      debounceTimerRef.current = null
    }
  }, [projectId])

  // Auto-save when content changes (debounced)
  useEffect(() => {
    // Mark as having unsaved changes
    if (
      rawContent ||
      chapters.length > 0 ||
      projectTitle ||
      projectAuthor ||
      projectDescription ||
      projectCoverImageUrl
    ) {
      setHasUnsaved(true)
    }

    // Clear existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    // Set up debounced save (5 seconds after last change)
    if (projectId && (rawContent || chapters.length > 0)) {
      debounceTimerRef.current = setTimeout(async () => {
        try {
          await saveProject()
          const savedAt = new Date()
          setLastSaved(savedAt)
          setSaveStatusNow(savedAt.getTime())
          setHasUnsaved(false)
        } catch (err) {
          console.error("Auto-save failed:", err)
        }
      }, DEBOUNCE_SAVE_DELAY)
    }

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [
    rawContent,
    chapters,
    projectTitle,
    projectAuthor,
    projectDescription,
    projectCoverImageUrl,
    projectId,
    saveProject
  ])

  // Periodic auto-save
  useEffect(() => {
    if (!projectId) return

    autoSaveTimerRef.current = setInterval(async () => {
      if (hasUnsaved) {
        try {
          await saveProject()
          const savedAt = new Date()
          setLastSaved(savedAt)
          setSaveStatusNow(savedAt.getTime())
          setHasUnsaved(false)
        } catch (err) {
          console.error("Periodic auto-save failed:", err)
        }
      }
    }, AUTO_SAVE_INTERVAL)

    return () => {
      if (autoSaveTimerRef.current) {
        clearInterval(autoSaveTimerRef.current)
      }
    }
  }, [projectId, hasUnsaved, saveProject])

  // Cleanup audio URLs on unmount
  useEffect(() => {
    return () => {
      revokeAllAudioUrls()
    }
  }, [revokeAllAudioUrls])

  // Handle new project creation
  const handleNewProject = useCallback(async () => {
    await createNewProject()
    setShowProjectList(false)
  }, [createNewProject])

  // Handle project opened from list
  const handleProjectOpened = useCallback(() => {
    setShowProjectList(false)
  }, [])

  const tabItems = [
    {
      key: "content",
      label: (
        <span className="flex items-center gap-2">
          <FileText className="h-4 w-4" />
          {t("audiobook:tabs.content", "Content")}
        </span>
      ),
      children: <TextEditor onSplitComplete={() => setActiveTab("chapters")} />
    },
    {
      key: "chapters",
      label: (
        <span className="flex items-center gap-2">
          <ListOrdered className="h-4 w-4" />
          {t("audiobook:tabs.chapters", "Chapters")}
          {chapters.length > 0 && (
            <Tag color="blue" className="ml-1">
              {chapters.length}
            </Tag>
          )}
        </span>
      ),
      children: <ChapterList />
    },
    {
      key: "generate",
      label: (
        <span className="flex items-center gap-2">
          <Play className="h-4 w-4" />
          {t("audiobook:tabs.generate", "Generate")}
          {isGenerating && (
            <Tag color="processing" className="ml-1">
              {t("audiobook:generating", "Generating...")}
            </Tag>
          )}
        </span>
      ),
      children: <GenerationPanel />
    },
    {
      key: "output",
      label: (
        <span className="flex items-center gap-2">
          <Download className="h-4 w-4" />
          {t("audiobook:tabs.output", "Output")}
          {completedCount > 0 && (
            <Tag color="green" className="ml-1">
              {completedCount}/{chapters.length}
            </Tag>
          )}
        </span>
      ),
      children: <OutputPanel />
    }
  ]

  // If showing project list, render it instead of the main editor
  if (showProjectList) {
    return (
      <PageShell maxWidthClassName="max-w-5xl" className="py-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <Headphones className="h-6 w-6 text-primary" />
              <Title level={3} className="!mb-0">
                {t("audiobook:title", "Audiobook Studio")}
              </Title>
            </div>
            <Text type="secondary">
              {t(
                "audiobook:subtitle",
                "Convert text into professional audiobooks with AI-powered text-to-speech."
              )}
            </Text>
          </div>
          <Button onClick={() => setShowProjectList(false)}>
            {t("audiobook:backToEditor", "Back to Editor")}
          </Button>
        </div>

        <DismissibleBetaAlert
          storageKey="beta-dismissed:audiobookStudio"
          message={t("audiobook:betaNotice", "Beta Feature")}
          description={t(
            "audiobook:betaDescription",
            "Audiobook Studio is currently in beta. Some features may be incomplete or change."
          )}
          className="mb-4"
        />

        <ProjectListView
          onOpenProject={handleProjectOpened}
          onCreateNew={handleNewProject}
        />
      </PageShell>
    )
  }

  return (
    <PageShell maxWidthClassName="max-w-5xl" className="py-6">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <Headphones className="h-6 w-6 text-primary" />
            <Title level={3} className="!mb-0">
              {t("audiobook:title", "Audiobook Studio")}
            </Title>
          </div>
          <Text type="secondary">
            {t(
              "audiobook:subtitle",
              "Convert text into professional audiobooks with AI-powered text-to-speech."
            )}
          </Text>
        </div>
        <Space wrap>
          <Button
            icon={<FolderOpen className="h-4 w-4" />}
            onClick={() => setShowProjectList(true)}
          >
            {t("audiobook:projects.myProjects", "My Projects")}
          </Button>
          <Button
            icon={<Plus className="h-4 w-4" />}
            onClick={handleNewProject}
          >
            {t("audiobook:projects.new", "New")}
          </Button>
          <Tooltip
            title={
              lastSaved
                ? t("audiobook:lastSaved", "Last saved: {{time}}", {
                    time: lastSaved.toLocaleTimeString()
                  })
                : saveStatusLabel
            }
          >
            <Button
              type={hasUnsaved || !projectId ? "primary" : "default"}
              aria-label={t("audiobook:saveProject", "Save project")}
              icon={
                hasUnsaved || !projectId ? (
                  <Save className="h-4 w-4" />
                ) : (
                  <Check className="h-4 w-4" />
                )
              }
              onClick={handleSave}
              loading={isSaving}
            >
              {t("audiobook:save", "Save")}
            </Button>
          </Tooltip>
        </Space>
      </div>

      <span
        role="status"
        aria-live="polite"
        aria-label={t(
          "audiobook:saveStatus.ariaLabel",
          "Project save status"
        )}
        className={`mb-3 block text-sm ${
          saveStatusType === "warning" ? "text-warning" : "text-text-subtle"
        }`}
      >
        {saveStatusLabel}
      </span>

      <DismissibleBetaAlert
        storageKey="beta-dismissed:audiobookStudio"
        message={t("audiobook:betaNotice", "Beta Feature")}
        description={t(
          "audiobook:betaDescription",
          "Audiobook Studio is currently in beta. Some features may be incomplete or change."
        )}
        className="mb-4"
      />

      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <label className="block text-sm font-medium">
            {t("audiobook:projectTitle", "Project Title")}
          </label>
          <Button
            type="text"
            size="small"
            icon={<Settings className="h-3 w-3" />}
            onClick={() => setShowMetadataForm(true)}
          />
        </div>
        <Input
          value={projectTitle}
          onChange={(e) => setProjectTitle(e.target.value)}
          placeholder={t("audiobook:projectTitlePlaceholder", "My Audiobook")}
          className="max-w-md"
        />
      </div>

      <div className="flex flex-wrap items-center gap-4 mb-4 text-sm text-text-muted">
        <span>
          {t("audiobook:stats.chapters", "{{count}} chapters", {
            count: chapters.length
          })}
        </span>
        {completedCount > 0 && (
          <span className="text-success">
            {t("audiobook:stats.completed", "{{count}} completed", {
              count: completedCount
            })}
          </span>
        )}
        {pendingCount > 0 && (
          <span>
            {t("audiobook:stats.pending", "{{count}} pending", {
              count: pendingCount
            })}
          </span>
        )}
        {lastSaved && (
          <span className="text-xs">
            {t("audiobook:autoSaved", "Auto-saved {{time}}", {
              time: lastSaved.toLocaleTimeString()
            })}
          </span>
        )}
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        className="audiobook-tabs"
      />

      <ProjectMetadataForm
        open={showMetadataForm}
        onClose={() => setShowMetadataForm(false)}
        onSave={handleSave}
      />
    </PageShell>
  )
}

export default AudiobookStudioPage
