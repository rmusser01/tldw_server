import React from "react"
import { Alert, Button, Checkbox, List, Space, Tag, Typography } from "antd"
import type { AudiobookProject } from "@/db/dexie/types"
import type { AudiobookMigrationCounts } from "@/services/audio-studio"

type MigrationBannerProps = {
  projects?: AudiobookProject[]
  selectedProjectIds?: string[]
  previewCounts?: AudiobookMigrationCounts
  errorMessage?: string | null
  isPreviewing?: boolean
  isCommitting?: boolean
  onSelectionChange?: (projectIds: string[]) => void
  onPreview?: () => void
  onCommit?: () => void
}

const pluralize = (count: number, singular: string, plural: string) =>
  `${count} ${count === 1 ? singular : plural}`

const EMPTY_PROJECTS: AudiobookProject[] = []
const EMPTY_PROJECT_IDS: string[] = []

const toCount = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value) ? value : 0

const formatPreviewCounts = (counts: AudiobookMigrationCounts): string => {
  const projects = toCount(counts.projects)
  const chapters = toCount(counts.chapters)
  const audioAssets = toCount(counts.audio_assets ?? counts.audioAssets)

  return [
    pluralize(projects, "project", "projects"),
    pluralize(chapters, "chapter", "chapters"),
    pluralize(audioAssets, "audio asset", "audio assets")
  ].join(", ")
}

export const MigrationBanner: React.FC<MigrationBannerProps> = ({
  projects,
  selectedProjectIds,
  previewCounts,
  errorMessage,
  isPreviewing = false,
  isCommitting = false,
  onSelectionChange,
  onPreview,
  onCommit
}) => {
  const normalizedProjects = projects ?? EMPTY_PROJECTS
  const normalizedSelectedProjectIds = selectedProjectIds ?? EMPTY_PROJECT_IDS
  const selectedProjectIdSet = React.useMemo(
    () => new Set(normalizedSelectedProjectIds),
    [normalizedSelectedProjectIds]
  )

  if (!projects || !selectedProjectIds || !onSelectionChange || !onPreview || !onCommit) {
    return (
      <Alert
        type="info"
        showIcon
        className="rounded-md"
        title="Audiobook projects can move into Audio Studio Narration"
        description="Open the legacy Audiobook Studio route to check local projects and migrate them without deleting Dexie data."
      />
    )
  }

  const allSelected =
    normalizedProjects.length > 0 &&
    normalizedSelectedProjectIds.length === normalizedProjects.length
  const isBusy = isPreviewing || isCommitting
  const hasSelection = normalizedSelectedProjectIds.length > 0
  const previewSummary = previewCounts
    ? formatPreviewCounts(previewCounts)
    : null

  const toggleProject = (projectId: string, checked: boolean) => {
    if (checked) {
      onSelectionChange([...selectedProjectIdSet, projectId])
      return
    }
    onSelectionChange(normalizedSelectedProjectIds.filter((id) => id !== projectId))
  }

  const toggleAll = (checked: boolean) => {
    onSelectionChange(checked ? normalizedProjects.map((project) => project.id) : [])
  }

  return (
    <section
      aria-labelledby="audio-studio-migration-heading"
      className="rounded-md border border-blue-200 bg-blue-50 p-4"
    >
      <Space orientation="vertical" size="middle" className="w-full">
        <div>
          <Typography.Title
            id="audio-studio-migration-heading"
            level={3}
            className="!mb-2 !text-lg"
          >
            Move local Audiobook projects into Audio Studio
          </Typography.Title>
          <Typography.Paragraph className="!mb-0">
            Preview the structure migration before creating server-backed
            Narration projects. Local Audiobook rows stay in Dexie after
            migration.
          </Typography.Paragraph>
        </div>

        {errorMessage ? (
          <Alert type="error" showIcon title={errorMessage} />
        ) : null}

        {previewSummary ? (
          <Alert
            type="success"
            showIcon
            title="Migration preview"
            description={previewSummary}
          />
        ) : null}

        <div className="flex items-center justify-between gap-3">
          <Checkbox
            checked={allSelected}
            indeterminate={hasSelection && !allSelected}
            disabled={isBusy}
            onChange={(event) => toggleAll(event.target.checked)}
          >
            Select all
          </Checkbox>
          <Tag color="blue">
            {pluralize(normalizedProjects.length, "project", "projects")}
          </Tag>
        </div>

        <List
          size="small"
          bordered
          dataSource={normalizedProjects}
          renderItem={(project) => (
            <List.Item>
              <Checkbox
                checked={selectedProjectIdSet.has(project.id)}
                disabled={isBusy}
                onChange={(event) =>
                  toggleProject(project.id, event.target.checked)
                }
              >
                <span className="font-medium">{project.title}</span>
                <span className="ml-2 text-sm text-gray-600">
                  {pluralize(project.chapters?.length ?? 0, "chapter", "chapters")}
                </span>
              </Checkbox>
            </List.Item>
          )}
        />

        <Space wrap>
          <Button
            type="default"
            disabled={!hasSelection || isBusy}
            loading={isPreviewing}
            onClick={onPreview}
          >
            Preview migration
          </Button>
          <Button
            type="primary"
            disabled={!hasSelection || isBusy || !previewCounts}
            loading={isCommitting}
            onClick={onCommit}
          >
            Migrate selected
          </Button>
        </Space>
      </Space>
    </section>
  )
}

export { formatPreviewCounts }
