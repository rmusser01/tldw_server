import React from "react"
import { Button, Tag } from "antd"
import { BookOpen, Copy, FileSearch, Plus } from "lucide-react"

export type VisualPackReusePanelProps = {
  selectedPersonaName: string
  hasSelectedPack: boolean
  canImport?: boolean
  libraryItemCount: number
  hasDuplicateTargets: boolean
  duplicateTargetsLoading: boolean
  onCreateDraft: () => void
  onOpenLibrary: () => void
  onOpenImport: () => void
  onOpenDuplicate: () => void
}

const formatLibraryCount = (count: number): string => {
  if (count <= 0) return "No saved visual packs yet"
  if (count === 1) return "1 saved visual pack"
  return `${count} saved visual packs`
}

const getDuplicateStatus = ({
  duplicateTargetsLoading,
  hasDuplicateTargets,
  hasSelectedPack
}: Pick<
  VisualPackReusePanelProps,
  "duplicateTargetsLoading" | "hasDuplicateTargets" | "hasSelectedPack"
>): string => {
  if (duplicateTargetsLoading) return "Loading persona targets."
  if (!hasSelectedPack) return "Select a pack before duplicating."
  if (!hasDuplicateTargets) return "Add another persona before duplicating."
  return "Copy the selected pack as a draft for another persona."
}

export const VisualPackReusePanel: React.FC<VisualPackReusePanelProps> = ({
  selectedPersonaName,
  hasSelectedPack,
  canImport = true,
  libraryItemCount,
  hasDuplicateTargets,
  duplicateTargetsLoading,
  onCreateDraft,
  onOpenLibrary,
  onOpenImport,
  onOpenDuplicate
}) => {
  const duplicateDisabled =
    duplicateTargetsLoading || !hasSelectedPack || !hasDuplicateTargets

  return (
    <div
      data-testid="persona-visual-reuse-panel"
      className="rounded-lg border border-border bg-surface p-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            Reuse visual packs
          </div>
          <div className="mt-1 text-sm font-medium text-text">
            {selectedPersonaName}
          </div>
          <div className="mt-1 max-w-3xl text-xs leading-5 text-text-muted">
            Assets are user-owned and attached to this persona by default. Each
            path creates or opens a draft for review; activate it only after the
            pack is ready.
          </div>
        </div>
        <Tag color="blue">draft first</Tag>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Start fresh</span>
            <Tag>draft</Tag>
          </div>
          <div className="mt-1 min-h-[2.5rem] text-xs leading-5 text-text-muted">
            Create an empty draft pack for new poses, animations, and state
            mappings.
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<Plus className="h-3.5 w-3.5" />}
            onClick={onCreateDraft}
          >
            Create draft
          </Button>
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Personal library</span>
            <Tag>reference</Tag>
          </div>
          <div className="mt-1 min-h-[2.5rem] text-xs leading-5 text-text-muted">
            {formatLibraryCount(libraryItemCount)}. Use one to create a reviewed
            draft for the target persona.
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<BookOpen className="h-3.5 w-3.5" />}
            onClick={onOpenLibrary}
          >
            Use personal library
          </Button>
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Portable archive</span>
            <Tag>preview</Tag>
          </div>
          <div className="mt-1 min-h-[2.5rem] text-xs leading-5 text-text-muted">
            {canImport
              ? "Import an archive through preview, choose conflicts, then commit as a draft."
              : "Select or create a draft before using import preview controls."}
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<FileSearch className="h-3.5 w-3.5" />}
            disabled={!canImport}
            onClick={onOpenImport}
          >
            Import archive
          </Button>
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Another persona</span>
            <Tag>copy</Tag>
          </div>
          <div className="mt-1 min-h-[2.5rem] text-xs leading-5 text-text-muted">
            {getDuplicateStatus({
              duplicateTargetsLoading,
              hasDuplicateTargets,
              hasSelectedPack
            })}
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<Copy className="h-3.5 w-3.5" />}
            disabled={duplicateDisabled}
            loading={duplicateTargetsLoading}
            onClick={onOpenDuplicate}
          >
            Duplicate to persona
          </Button>
        </div>
      </div>
    </div>
  )
}

export default VisualPackReusePanel
