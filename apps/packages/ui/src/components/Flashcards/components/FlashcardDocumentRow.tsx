import React from "react"
import { Alert, Button, Checkbox, Input, Select, Tag, Typography, type InputRef } from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"

import type { Deck, Flashcard, FlashcardBulkUpdateItem, FlashcardBulkUpdateResponse } from "@/services/flashcards"
import { getFlashcardSourceMeta } from "../utils/source-reference"
import { FlashcardQueueStateBadge } from "../utils/queue-state-badges"
import { formatDeckDisplayName } from "../utils/deck-display"
import { useFlashcardDocumentRowState } from "../hooks/useFlashcardDocumentRowState"
import type { DocumentQueryFilterContext } from "../utils/document-cache-policy"
import { FlashcardImageInsertButton } from "./FlashcardImageInsertButton"
import { MarkdownWithBoundary } from "./MarkdownWithBoundary"
import {
  getSelectionFromElement,
  insertTextAtSelection,
  restoreSelection,
  type TextSelection
} from "../utils/text-selection"

const { Text } = Typography
const { TextArea } = Input
type EditableTextField = "front" | "back" | "notes"

export interface FlashcardDocumentRowProps {
  card: Flashcard
  decks: Deck[]
  selected: boolean
  selectAllAcross: boolean
  filterContext: DocumentQueryFilterContext
  queryKey: readonly unknown[]
  onToggleSelect: (uuid: string, checked: boolean) => void
  onOpenDrawer?: (card: Flashcard) => void
  bulkUpdate: (items: FlashcardBulkUpdateItem[]) => Promise<FlashcardBulkUpdateResponse>
  loadLatestCard?: (uuid: string) => Promise<Flashcard>
}

export const FlashcardDocumentRow: React.FC<FlashcardDocumentRowProps> = ({
  card,
  decks,
  selected,
  selectAllAcross,
  filterContext,
  queryKey,
  onToggleSelect,
  onOpenDrawer,
  bulkUpdate,
  loadLatestCard
}) => {
  const [uploadError, setUploadError] = React.useState<string | null>(null)
  const {
    card: savedCard,
    draft,
    isEditing,
    status,
    errorMessage,
    validationFields,
    undoSnapshot,
    isSaving,
    enterEditMode,
    cancelEdit,
    setField,
    commit,
    undo,
    reloadRow,
    reapplyConflict
  } = useFlashcardDocumentRowState({
    card,
    filterContext,
    queryKey,
    bulkUpdate,
    loadLatestCard
  })

  const rowRef = React.useRef<HTMLDivElement | null>(null)
  const frontInputRef = React.useRef<InputRef | null>(null)
  const backInputRef = React.useRef<TextAreaRef | null>(null)
  const notesInputRef = React.useRef<TextAreaRef | null>(null)
  const selectionRef = React.useRef<Record<EditableTextField, TextSelection>>({
    front: { start: 0, end: 0 },
    back: { start: 0, end: 0 },
    notes: { start: 0, end: 0 }
  })
  const deckLabel =
    savedCard.deck_id != null
      ? formatDeckDisplayName(
          decks.find((deck) => deck.id === savedCard.deck_id),
          `Deck ${savedCard.deck_id}`
        )
      : "No deck"
  const sourceMeta = getFlashcardSourceMeta(savedCard)

  const getFieldElement = React.useCallback(
    (field: EditableTextField): HTMLInputElement | HTMLTextAreaElement | null => {
      if (field === "front") {
        return frontInputRef.current?.input ?? null
      }
      if (field === "back") {
        return backInputRef.current?.resizableTextArea?.textArea ?? null
      }
      return notesInputRef.current?.resizableTextArea?.textArea ?? null
    },
    []
  )

  const updateSelection = React.useCallback(
    (
      field: EditableTextField,
      element: HTMLInputElement | HTMLTextAreaElement | null | undefined
    ) => {
      const currentValue = String(draft[field] ?? "")
      selectionRef.current[field] = getSelectionFromElement(element, currentValue)
    },
    [draft]
  )

  const handleInsertImage = React.useCallback(
    async (field: EditableTextField, markdownSnippet: string) => {
      const currentValue = String(draft[field] ?? "")
      const element = getFieldElement(field)
      const selection =
        selectionRef.current[field] ?? getSelectionFromElement(element, currentValue)
      const { nextValue, cursor } = insertTextAtSelection(
        currentValue,
        selection,
        markdownSnippet
      )
      setUploadError(null)
      setField(field, nextValue)
      restoreSelection(element, cursor)
    },
    [draft, getFieldElement, setField]
  )

  const handleRowBlur = React.useCallback(
    (event: React.FocusEvent<HTMLDivElement>) => {
      const nextTarget = event.relatedTarget as Node | null
      if (nextTarget && event.currentTarget.contains(nextTarget)) return
      if (!isEditing) return
      void commit()
    },
    [commit, isEditing]
  )

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault()
        void commit()
      }
      if (event.key === "Escape") {
        event.preventDefault()
        cancelEdit()
      }
    },
    [cancelEdit, commit]
  )

  const statusTone =
    status === "conflict"
      ? "border-warning/60 bg-warning/5"
      : status === "validation_error" || status === "not_found"
        ? "border-danger/60 bg-danger/5"
        : status === "saved"
          ? "border-success/60 bg-success/5"
          : "border-border bg-surface"

  return (
    <div
      ref={rowRef}
    className={`grid gap-3 border-b border-border p-4 transition-colors md:grid-cols-[auto_minmax(0,1.2fr)_minmax(0,1.2fr)_minmax(240px,0.9fr)] ${statusTone}`}
    onClick={enterEditMode}
    onFocusCapture={enterEditMode}
      onBlurCapture={handleRowBlur}
      onKeyDown={handleKeyDown}
      data-testid={`flashcards-document-row-${savedCard.uuid}`}
      tabIndex={-1}
    >
      <div className="pt-1">
        <Checkbox
          checked={selectAllAcross ? true : selected}
          disabled={selectAllAcross}
          onChange={(event) => onToggleSelect(savedCard.uuid, event.target.checked)}
          data-testid={`flashcards-document-row-select-${savedCard.uuid}`}
          aria-label={`Select document row for ${savedCard.front.slice(0, 80)}`}
        />
      </div>

      <div className="min-w-0 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <Text strong className="text-xs uppercase tracking-wide text-text-muted">
            Question
          </Text>
          <div className="flex items-center gap-2">
            {isEditing && (
              <FlashcardImageInsertButton
                ariaLabel={`Upload image for Question ${savedCard.uuid}`}
                onInsert={(markdownSnippet) => handleInsertImage("front", markdownSnippet)}
                onError={(error) => setUploadError(error.message)}
              />
            )}
            {status === "saving" && <Tag color="processing">Saving</Tag>}
            {status === "saved" && undoSnapshot && (
              <Button
                size="small"
                type="link"
                onClick={(event) => {
                  event.stopPropagation()
                  void undo()
                }}
              >
                Undo
              </Button>
            )}
            <Button
              size="small"
              type="link"
              onClick={(event) => {
                event.stopPropagation()
                onOpenDrawer?.(savedCard)
              }}
              disabled={!onOpenDrawer}
            >
              Open drawer
            </Button>
          </div>
        </div>
        {isEditing ? (
          <Input
            ref={frontInputRef}
            value={draft.front}
            onChange={(event) => {
              setUploadError(null)
              setField("front", event.target.value)
            }}
            onSelect={(event) => updateSelection("front", event.currentTarget)}
            onClick={(event) => updateSelection("front", event.currentTarget)}
            onKeyUp={(event) => updateSelection("front", event.currentTarget)}
            data-testid={`flashcards-document-row-front-input-${savedCard.uuid}`}
          />
        ) : (
          <div
            className="cursor-text whitespace-pre-wrap break-words"
            data-testid={`flashcards-document-row-front-display-${savedCard.uuid}`}
          >
            <MarkdownWithBoundary content={savedCard.front} size="sm" />
          </div>
        )}
        <div className="flex flex-wrap gap-1.5">
          <Tag>{savedCard.model_type === "cloze" ? "Fill-in-blank" : savedCard.reverse ? "Reversible" : "Standard"}</Tag>
          <FlashcardQueueStateBadge
            card={savedCard}
            testId={`flashcards-document-row-queue-state-${savedCard.uuid}`}
          />
          <Tag color="blue">{deckLabel}</Tag>
          {(savedCard.tags || []).map((tag) => (
            <Tag key={`${savedCard.uuid}-${tag}`}>{tag}</Tag>
          ))}
          {sourceMeta && (
            <Tag color={sourceMeta.unavailable ? "default" : "green"}>
              {sourceMeta.label}
            </Tag>
          )}
        </div>
      </div>

      <div className="min-w-0 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <Text strong className="block text-xs uppercase tracking-wide text-text-muted">
            Answer
          </Text>
          {isEditing && (
            <FlashcardImageInsertButton
              ariaLabel={`Upload image for Answer ${savedCard.uuid}`}
              onInsert={(markdownSnippet) => handleInsertImage("back", markdownSnippet)}
              onError={(error) => setUploadError(error.message)}
            />
          )}
        </div>
        {isEditing ? (
          <TextArea
            ref={backInputRef}
            value={draft.back}
            onChange={(event) => {
              setUploadError(null)
              setField("back", event.target.value)
            }}
            onSelect={(event) => updateSelection("back", event.currentTarget)}
            onClick={(event) => updateSelection("back", event.currentTarget)}
            onKeyUp={(event) => updateSelection("back", event.currentTarget)}
            autoSize={{ minRows: 3, maxRows: 8 }}
            data-testid={`flashcards-document-row-back-input-${savedCard.uuid}`}
          />
        ) : (
          <div className="cursor-text whitespace-pre-wrap break-words">
            <MarkdownWithBoundary content={savedCard.back} size="sm" />
          </div>
        )}
      </div>

      <div className="min-w-0 space-y-3">
        <Text strong className="block text-xs uppercase tracking-wide text-text-muted">
          Organization
        </Text>

        {isEditing ? (
          <div className="space-y-3">
            <Select<number>
              allowClear
              value={draft.deck_id ?? undefined}
              options={decks.map((deck) => ({
                label: formatDeckDisplayName(deck, `Deck ${deck.id}`),
                value: deck.id
              }))}
              placeholder="Select deck"
              onChange={(value) => {
                setField("deck_id", value ?? null)
                void commit({
                  ...draft,
                  deck_id: value ?? null
                })
              }}
              data-testid={`flashcards-document-row-deck-select-${savedCard.uuid}`}
            />

            <Input
              value={draft.tags_text}
              placeholder="comma,separated,tags"
              onChange={(event) => setField("tags_text", event.target.value)}
              data-testid={`flashcards-document-row-tags-input-${savedCard.uuid}`}
            />

            <Select
              value={draft.template}
              options={[
                { value: "basic", label: "Standard" },
                { value: "basic_reverse", label: "Reversible" },
                { value: "cloze", label: "Fill-in-blank" }
              ]}
              onChange={(value) => {
                setField("template", value as Flashcard["model_type"])
                void commit({
                  ...draft,
                  template: value as Flashcard["model_type"]
                })
              }}
              data-testid={`flashcards-document-row-template-select-${savedCard.uuid}`}
            />

            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2">
                <Text className="text-[11px] uppercase tracking-wide text-text-muted">
                  Notes
                </Text>
                <FlashcardImageInsertButton
                  ariaLabel={`Upload image for Notes ${savedCard.uuid}`}
                  onInsert={(markdownSnippet) => handleInsertImage("notes", markdownSnippet)}
                  onError={(error) => setUploadError(error.message)}
                />
              </div>
              <TextArea
                ref={notesInputRef}
                value={draft.notes}
                placeholder="Notes"
                onChange={(event) => {
                  setUploadError(null)
                  setField("notes", event.target.value)
                }}
                onSelect={(event) => updateSelection("notes", event.currentTarget)}
                onClick={(event) => updateSelection("notes", event.currentTarget)}
                onKeyUp={(event) => updateSelection("notes", event.currentTarget)}
                autoSize={{ minRows: 2, maxRows: 6 }}
                data-testid={`flashcards-document-row-notes-input-${savedCard.uuid}`}
              />
            </div>

            <div className="flex items-center justify-end gap-2">
              <Button
                size="small"
                onClick={(event) => {
                  event.stopPropagation()
                  cancelEdit()
                }}
              >
                Cancel
              </Button>
              <Button
                size="small"
                type="primary"
                loading={isSaving}
                onClick={(event) => {
                  event.stopPropagation()
                  void commit()
                }}
              >
                Save
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-2 text-sm text-text-muted">
            <div>{deckLabel}</div>
            <div>{savedCard.tags?.length ? savedCard.tags.join(", ") : "No tags"}</div>
            <div>
              {savedCard.notes?.trim() ? (
                <MarkdownWithBoundary content={savedCard.notes} size="sm" />
              ) : (
                "No notes"
              )}
            </div>
          </div>
        )}

        {uploadError && (
          <Alert
            type="error"
            showIcon
            title={uploadError}
            data-testid={`flashcards-document-row-upload-error-${savedCard.uuid}`}
          />
        )}

        {status === "conflict" && (
          <Alert
            type="warning"
            showIcon
            title={errorMessage || "This row changed elsewhere."}
            action={
              <div className="flex gap-2">
                <Button
                  size="small"
                  onClick={(event) => {
                    event.stopPropagation()
                    void reloadRow()
                  }}
                >
                  Reload row
                </Button>
                <Button
                  size="small"
                  type="primary"
                  onClick={(event) => {
                    event.stopPropagation()
                    void reapplyConflict()
                  }}
                >
                  Reapply my edit
                </Button>
              </div>
            }
            data-testid={`flashcards-document-row-conflict-${savedCard.uuid}`}
          />
        )}

        {(status === "validation_error" || status === "not_found") && (
          <Alert
            type="error"
            showIcon
            title={errorMessage || "This row could not be saved."}
            description={
              validationFields.length > 0 ? `Check: ${validationFields.join(", ")}` : undefined
            }
            data-testid={`flashcards-document-row-error-${savedCard.uuid}`}
          />
        )}
      </div>
    </div>
  )
}

export default FlashcardDocumentRow
