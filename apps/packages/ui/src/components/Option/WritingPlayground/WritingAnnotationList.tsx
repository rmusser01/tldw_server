import React from "react"
import { Button, Empty, Input, Tag } from "antd"
import type { ManuscriptAnnotationResponse } from "@/services/writing-playground"

const { TextArea } = Input

export type WritingAnnotationListProps = {
  annotations: ManuscriptAnnotationResponse[]
  onUpdate: (
    annotationId: string,
    input: { status?: "open" | "resolved"; body?: string },
    version: number
  ) => Promise<unknown>
  onDelete: (annotationId: string, version: number) => Promise<unknown>
  disabled?: boolean
}

export function WritingAnnotationList({
  annotations,
  onUpdate,
  onDelete,
  disabled = false
}: WritingAnnotationListProps) {
  const [editingId, setEditingId] = React.useState<string | null>(null)
  const [draftBody, setDraftBody] = React.useState("")

  if (annotations.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No annotations" />
  }

  return (
    <div className="flex flex-col gap-2" data-testid="writing-annotation-list">
      {annotations.map((annotation) => {
        const isEditing = editingId === annotation.id
        return (
          <div
            id={`writing-annotation-inspector-row-${annotation.id}`}
            key={annotation.id}
            data-testid="writing-annotation-inspector-row"
            aria-labelledby={`writing-annotation-inspector-row-${annotation.id}-summary`}
            className="rounded border border-border bg-surface p-2 text-xs">
            <span
              id={`writing-annotation-inspector-row-${annotation.id}-summary`}
              className="sr-only"
            >
              {annotation.category} annotation {annotation.id}
            </span>
            <div className="mb-2 flex flex-wrap items-center gap-1">
              <Tag className="!m-0">{annotation.status}</Tag>
              <Tag className="!m-0">{annotation.category}</Tag>
              <Tag className="!m-0">{annotation.source}</Tag>
              <Tag className="!m-0">{annotation.anchor_status}</Tag>
            </div>
            {annotation.selected_text ? (
              <blockquote className="mb-2 border-l-2 border-border pl-2 text-text-muted">
                {annotation.selected_text}
              </blockquote>
            ) : null}
            {isEditing ? (
              <div className="flex flex-col gap-2">
                <TextArea
                  aria-label="Edit annotation body"
                  size="small"
                  value={draftBody}
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  onChange={(event) => setDraftBody(event.target.value)}
                />
                <div className="flex gap-1">
                  <Button
                    size="small"
                    type="primary"
                    aria-label={`Save ${annotation.id}`}
                    disabled={disabled || !draftBody.trim()}
                    onClick={() => {
                      void onUpdate(
                        annotation.id,
                        { body: draftBody.trim() },
                        annotation.version
                      ).then(() => setEditingId(null))
                    }}>
                    Save
                  </Button>
                  <Button
                    size="small"
                    onClick={() => setEditingId(null)}
                    disabled={disabled}>
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <p className="mb-2 whitespace-pre-wrap text-text">{annotation.body}</p>
            )}
            <div className="flex flex-wrap gap-1">
              {annotation.status === "open" ? (
                <Button
                  size="small"
                  aria-label={`Resolve ${annotation.id}`}
                  disabled={disabled}
                  onClick={() => {
                    void onUpdate(
                      annotation.id,
                      { status: "resolved" },
                      annotation.version
                    )
                  }}>
                  Resolve
                </Button>
              ) : (
                <Button
                  size="small"
                  aria-label={`Reopen ${annotation.id}`}
                  disabled={disabled}
                  onClick={() => {
                    void onUpdate(
                      annotation.id,
                      { status: "open" },
                      annotation.version
                    )
                  }}>
                  Reopen
                </Button>
              )}
              <Button
                size="small"
                aria-label={`Edit ${annotation.id}`}
                disabled={disabled}
                onClick={() => {
                  setEditingId(annotation.id)
                  setDraftBody(annotation.body)
                }}>
                Edit
              </Button>
              <Button
                size="small"
                danger
                aria-label={`Delete ${annotation.id}`}
                disabled={disabled}
                onClick={() => {
                  void onDelete(annotation.id, annotation.version)
                }}>
                Delete
              </Button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
