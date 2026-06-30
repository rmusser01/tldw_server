import React from "react"
import { Checkbox } from "antd"
import type { FolderSelectionState } from "@/store/workspace-organization"

export interface SourceFolderTreeNode {
  id: string
  name: string
  sourceCount: number
  children: SourceFolderTreeNode[]
}

interface SourceFolderTreeProps {
  nodes: SourceFolderTreeNode[]
  activeFolderId: string | null
  selectionStateByFolderId: Record<string, FolderSelectionState>
  onClearFocus: () => void
  onCreateFolder: () => void
  editingFolderId?: string | null
  editingFolderName?: string
  onEditingFolderNameChange?: (name: string) => void
  onCommitFolderRename?: (folderId: string) => void
  onCancelFolderRename?: () => void
  onFocusFolder: (folderId: string) => void
  onToggleFolderSelection: (folderId: string) => void
}

const renderNode = (
  node: SourceFolderTreeNode,
  depth: number,
  activeFolderId: string | null,
  selectionStateByFolderId: Record<string, FolderSelectionState>,
  editingFolderId: string | null | undefined,
  editingFolderName: string | undefined,
  onEditingFolderNameChange: ((name: string) => void) | undefined,
  onCommitFolderRename: ((folderId: string) => void) | undefined,
  onFolderRenameInputBlur: ((folderId: string) => void) | undefined,
  onCancelFolderRename: (() => void) | undefined,
  onFocusFolder: (folderId: string) => void,
  onToggleFolderSelection: (folderId: string) => void
): React.ReactElement => {
  const selectionState = selectionStateByFolderId[node.id] || "unchecked"
  const isActive = activeFolderId === node.id
  const isEditing = editingFolderId === node.id

  return (
    <div key={node.id} className="space-y-1">
      <div
        className={`flex items-center gap-2 rounded-md px-2 py-1 ${
          isActive ? "bg-primary/10" : "hover:bg-surface2"
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        <Checkbox
          aria-label={`Select folder ${node.name}`}
          checked={selectionState === "checked"}
          indeterminate={selectionState === "indeterminate"}
          onChange={() => onToggleFolderSelection(node.id)}
        />
        {isEditing ? (
          <input
            autoFocus
            aria-label={`Rename folder ${node.name}`}
            className="min-w-0 flex-1 rounded border border-primary/40 bg-surface px-1.5 py-0.5 text-sm text-text outline-none focus:ring-2 focus:ring-focus"
            value={editingFolderName ?? node.name}
            onChange={(event) => onEditingFolderNameChange?.(event.target.value)}
            onBlur={() => onFolderRenameInputBlur?.(node.id)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault()
                onCommitFolderRename?.(node.id)
              }
              if (event.key === "Escape") {
                event.preventDefault()
                onCancelFolderRename?.()
              }
            }}
          />
        ) : (
          <button
            type="button"
            aria-label={`Focus folder ${node.name}`}
            className={`min-w-0 flex-1 truncate text-left text-sm ${
              isActive ? "font-medium text-primary" : "text-text"
            }`}
            onClick={() => onFocusFolder(node.id)}
          >
            {node.name}
          </button>
        )}
        <span className="shrink-0 text-[11px] text-text-muted">
          {node.sourceCount}
        </span>
      </div>
      {node.children.map((child) =>
        renderNode(
          child,
          depth + 1,
          activeFolderId,
          selectionStateByFolderId,
          editingFolderId,
          editingFolderName,
          onEditingFolderNameChange,
          onCommitFolderRename,
          onFolderRenameInputBlur,
          onCancelFolderRename,
          onFocusFolder,
          onToggleFolderSelection
        )
      )}
    </div>
  )
}

export const SourceFolderTree: React.FC<SourceFolderTreeProps> = ({
  nodes,
  activeFolderId,
  selectionStateByFolderId,
  onClearFocus,
  onCreateFolder,
  editingFolderId,
  editingFolderName,
  onEditingFolderNameChange,
  onCommitFolderRename,
  onCancelFolderRename,
  onFocusFolder,
  onToggleFolderSelection
}) => {
  const cancelingRenameRef = React.useRef(false)
  const cancelingRenameResetRef = React.useRef<number | null>(null)

  React.useEffect(
    () => () => {
      if (cancelingRenameResetRef.current !== null) {
        window.clearTimeout(cancelingRenameResetRef.current)
      }
    },
    []
  )

  const handleCancelFolderRename = React.useCallback(() => {
    cancelingRenameRef.current = true
    if (cancelingRenameResetRef.current !== null) {
      window.clearTimeout(cancelingRenameResetRef.current)
    }
    cancelingRenameResetRef.current = window.setTimeout(() => {
      cancelingRenameRef.current = false
      cancelingRenameResetRef.current = null
    }, 0)
    onCancelFolderRename?.()
  }, [onCancelFolderRename])

  const handleFolderRenameInputBlur = React.useCallback(
    (folderId: string) => {
      if (cancelingRenameRef.current) {
        cancelingRenameRef.current = false
        if (cancelingRenameResetRef.current !== null) {
          window.clearTimeout(cancelingRenameResetRef.current)
          cancelingRenameResetRef.current = null
        }
        return
      }
      onCommitFolderRename?.(folderId)
    },
    [onCommitFolderRename]
  )

  return (
    <div
      className="rounded-lg border border-border/70 bg-surface/60 p-2"
      aria-label="Source folders"
    >
      <div className="mb-2 flex items-center justify-between gap-2 px-2">
        <span className="text-xs font-semibold uppercase tracking-[0.08em] text-text-muted">
          Folders
        </span>
        <div className="flex items-center gap-2">
          {activeFolderId && (
            <button
              type="button"
              className="rounded border border-border px-2 py-0.5 text-[11px] font-medium text-text-muted transition hover:bg-surface2 hover:text-text"
              onClick={onClearFocus}
            >
              All sources
            </button>
          )}
          <button
            type="button"
            className="rounded border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary transition hover:bg-primary/15"
            onClick={onCreateFolder}
          >
            New folder
          </button>
        </div>
      </div>
      {nodes.length === 0 ? (
        <div className="rounded-md border border-dashed border-border/70 px-3 py-2 text-xs text-text-muted">
          <p className="font-medium text-text">No folders yet</p>
          <p className="mt-1">Create a folder to organize related sources.</p>
        </div>
      ) : (
        <div className="space-y-1">
          {nodes.map((node) =>
            renderNode(
              node,
              0,
              activeFolderId,
              selectionStateByFolderId,
              editingFolderId,
              editingFolderName,
              onEditingFolderNameChange,
              onCommitFolderRename,
              handleFolderRenameInputBlur,
              handleCancelFolderRename,
              onFocusFolder,
              onToggleFolderSelection
            )
          )}
        </div>
      )}
    </div>
  )
}
