import React from "react"
import {
  Badge,
  Button,
  Collapse,
  Drawer,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Tooltip,
  Typography
} from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"
import dayjs from "dayjs"
import relativeTime from "dayjs/plugin/relativeTime"
import { useTranslation } from "react-i18next"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useDebouncedFormField } from "../hooks"
import { FLASHCARDS_DRAWER_WIDTH_PX } from "../constants"
import { normalizeFlashcardTemplateFields } from "../utils/template-helpers"
import { normalizeOptionalFlashcardTags } from "../utils/tag-normalization"
import {
  FLASHCARD_FIELD_MAX_BYTES,
  getFlashcardFieldLimitState,
  getUtf8ByteLength
} from "../utils/field-byte-limit"
import { getFlashcardSourceMeta } from "../utils/source-reference"
import {
  getSelectionFromElement,
  insertTextAtSelection,
  restoreSelection,
  type TextSelection
} from "../utils/text-selection"
import { formatDeckDisplayName } from "../utils/deck-display"
import { FlashcardImageInsertButton } from "./FlashcardImageInsertButton"
import { FlashcardTagPicker } from "./FlashcardTagPicker"
import { MarkdownWithBoundary } from "./MarkdownWithBoundary"
import type { Flashcard, FlashcardUpdate, Deck } from "@/services/flashcards"

const { Text } = Typography
dayjs.extend(relativeTime)
const CLOZE_PATTERN = /\{\{c\d+::[\s\S]+?\}\}/

type FlashcardModelType = Flashcard["model_type"]
type EditableTextField = "front" | "back" | "extra" | "notes"

interface FlashcardEditDrawerProps {
  open: boolean
  onClose: () => void
  card: Flashcard | null
  onSave: (values: FlashcardUpdate) => Promise<void>
  onDelete: () => void
  onResetScheduling?: () => Promise<void>
  isLoading?: boolean
  decks: Deck[]
  decksLoading?: boolean
}

export const FlashcardEditDrawer: React.FC<FlashcardEditDrawerProps> = ({
  open,
  onClose,
  card,
  onSave,
  onDelete,
  onResetScheduling,
  isLoading = false,
  decks,
  decksLoading = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const [form] = Form.useForm<FlashcardUpdate & { tags_text?: string[] }>()
  const selectedModelType = Form.useWatch("model_type", form) as
    | FlashcardModelType
    | undefined
  const [showPreview, setShowPreview] = React.useState(false)
  const [isDirty, setIsDirty] = React.useState(false)
  const [confirmCloseOpen, setConfirmCloseOpen] = React.useState(false)
  const [confirmResetOpen, setConfirmResetOpen] = React.useState(false)

  const frontPreview = useDebouncedFormField(form, "front")
  const backPreview = useDebouncedFormField(form, "back")
  const extraPreview = useDebouncedFormField(form, "extra")
  const notesPreview = useDebouncedFormField(form, "notes")
  const frontValue = Form.useWatch("front", form) as string | undefined
  const backValue = Form.useWatch("back", form) as string | undefined
  const textAreaRefs = React.useRef<Record<EditableTextField, TextAreaRef | null>>({
    front: null,
    back: null,
    extra: null,
    notes: null
  })
  const selectionRef = React.useRef<Record<EditableTextField, TextSelection>>({
    front: { start: 0, end: 0 },
    back: { start: 0, end: 0 },
    extra: { start: 0, end: 0 },
    notes: { start: 0, end: 0 }
  })
  const frontByteLength = getUtf8ByteLength(frontValue)
  const backByteLength = getUtf8ByteLength(backValue)
  const frontLimitState = getFlashcardFieldLimitState(frontByteLength)
  const backLimitState = getFlashcardFieldLimitState(backByteLength)

  const previewLabel = t("option:flashcards.preview", { defaultValue: "Preview" })
  const markdownSupportHint = t("option:flashcards.markdownSupportHint", {
    defaultValue: "Supports Markdown and LaTeX."
  })
  const isClozeTemplate = selectedModelType === "cloze"
  const sourceMeta = React.useMemo(
    () => (card ? getFlashcardSourceMeta(card) : null),
    [card]
  )

  const templateHelperText = React.useMemo(() => {
    if (selectedModelType === "basic_reverse") {
      return t("option:flashcards.templateReverseHelp", {
        defaultValue:
          "Choose Basic + Reverse when you want both directions (term -> meaning and meaning -> term)."
      })
    }
    if (selectedModelType === "cloze") {
      return t("option:flashcards.templateClozeHelp", {
        defaultValue:
          "Choose Cloze when you want to hide key words inside a sentence or paragraph."
      })
    }
    return t("option:flashcards.templateBasicHelp", {
      defaultValue:
        "Choose Basic for direct question and answer cards (facts, definitions, short prompts)."
    })
  }, [selectedModelType, t])

  const renderByteUsageHint = React.useCallback(
    (field: "front" | "back", byteLength: number, state: "normal" | "warning" | "over") => {
      const fieldLabel = t(`option:flashcards.${field}`, {
        defaultValue: field === "front" ? "Front" : "Back"
      })
      const usageText = t("option:flashcards.fieldByteUsage", {
        defaultValue: "{{field}}: {{used}} / {{max}} bytes",
        field: fieldLabel,
        used: byteLength,
        max: FLASHCARD_FIELD_MAX_BYTES
      })
      if (state === "over") {
        return t("option:flashcards.fieldByteOverLimit", {
          defaultValue: "{{usage}}. Exceeds limit by {{over}} bytes.",
          usage: usageText,
          over: byteLength - FLASHCARD_FIELD_MAX_BYTES
        })
      }
      if (state === "warning") {
        return t("option:flashcards.fieldByteNearLimit", {
          defaultValue: "{{usage}}. Approaching the {{max}}-byte limit.",
          usage: usageText,
          max: FLASHCARD_FIELD_MAX_BYTES
        })
      }
      return usageText
    },
    [t]
  )

  // Count how many additional fields have values for the badge indicator
  const additionalFieldCount = React.useMemo(() => {
    let count = 0
    if (extraPreview && extraPreview.trim()) count++
    if (notesPreview && notesPreview.trim()) count++
    return count
  }, [extraPreview, notesPreview])

  // Sync form with card data when card changes
  React.useEffect(() => {
    if (card && open) {
      form.setFieldsValue({
        deck_id: card.deck_id ?? undefined,
        front: card.front,
        back: card.back,
        notes: card.notes || undefined,
        extra: card.extra || undefined,
        tags: card.tags || undefined,
        model_type: card.model_type,
        expected_version: card.version
      })
      setIsDirty(false)
    }
  }, [card, open, form])

  // Track form changes
  const handleFormChange = React.useCallback(() => {
    setIsDirty(true)
  }, [])

  const syncTemplateFields = React.useCallback(
    (partial: Partial<Pick<FlashcardUpdate, "model_type" | "reverse" | "is_cloze">>) => {
      const normalized = normalizeFlashcardTemplateFields(partial)
      form.setFieldsValue({
        model_type: normalized.model_type,
        reverse: normalized.reverse,
        is_cloze: normalized.is_cloze
      })
    },
    [form]
  )

  const handleSave = async () => {
    try {
      const rawValues = (await form.validateFields()) as FlashcardUpdate
      const values = normalizeFlashcardTemplateFields({
        ...rawValues,
        tags: normalizeOptionalFlashcardTags(rawValues.tags)
      })
      await onSave(values)
    } catch (e: any) {
      // Validation errors handled by form
      if (!e?.errorFields) {
        console.error("Save error:", e)
      }
    }
  }

  const handleClose = React.useCallback(() => {
    form.resetFields()
    setIsDirty(false)
    setConfirmCloseOpen(false)
    setConfirmResetOpen(false)
    onClose()
  }, [form, onClose])

  const handleAttemptClose = React.useCallback(() => {
    if (isDirty) {
      setConfirmCloseOpen(true)
    } else {
      handleClose()
    }
  }, [handleClose, isDirty])

  const handleConfirmResetScheduling = React.useCallback(async () => {
    if (!onResetScheduling) return
    try {
      await onResetScheduling()
      setConfirmResetOpen(false)
      setIsDirty(false)
    } catch (e: unknown) {
      console.error("Reset scheduling error:", e)
    }
  }, [onResetScheduling])

  const updateSelection = React.useCallback(
    (
      field: EditableTextField,
      element: HTMLTextAreaElement | null | undefined
    ) => {
      const currentValue = String(form.getFieldValue(field) ?? "")
      selectionRef.current[field] = getSelectionFromElement(element, currentValue)
    },
    [form]
  )

  const handleInsertImage = React.useCallback(
    async (field: EditableTextField, markdownSnippet: string) => {
      const currentValue = String(form.getFieldValue(field) ?? "")
      const textArea =
        textAreaRefs.current[field]?.resizableTextArea?.textArea ?? null
      const selection =
        selectionRef.current[field] ?? getSelectionFromElement(textArea, currentValue)
      const { nextValue, cursor } = insertTextAtSelection(
        currentValue,
        selection,
        markdownSnippet
      )
      form.setFieldsValue({ [field]: nextValue })
      setIsDirty(true)
      restoreSelection(textArea, cursor)
    },
    [form]
  )

  const renderFieldLabel = React.useCallback(
    (field: EditableTextField, label: string) => (
      <div className="flex items-center justify-between gap-3">
        <span>{label}</span>
        <FlashcardImageInsertButton
          ariaLabel={`Upload image for ${label}`}
          buttonLabel={t("option:flashcards.insertImage", {
            defaultValue: "Insert image"
          })}
          onInsert={(markdownSnippet) => handleInsertImage(field, markdownSnippet)}
          onError={(error) => message.error(error.message)}
        />
      </div>
    ),
    [handleInsertImage, message, t]
  )

  return (
    <>
    <Drawer
      placement="right"
      styles={{ wrapper: { width: FLASHCARDS_DRAWER_WIDTH_PX } }}
      open={open}
      onClose={handleAttemptClose}
      title={t("option:flashcards.editCard", { defaultValue: "Edit Flashcard" })}
      footer={
        <div className="flex justify-end">
          <Space>
            <Button onClick={handleAttemptClose}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button danger onClick={onDelete}>
              {t("common:delete", { defaultValue: "Delete" })}
            </Button>
            <Button type="primary" loading={isLoading} onClick={handleSave}>
              {t("common:save", { defaultValue: "Save" })}
            </Button>
          </Space>
        </div>
      }
    >
      <Form form={form} layout="vertical" onValuesChange={handleFormChange}>
        {/* Section: Organization */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-text-muted mb-3">
            {t("option:flashcards.organization", { defaultValue: "Organization" })}
          </h3>
          <Form.Item
            name="deck_id"
            label={t("option:flashcards.deck", { defaultValue: "Deck" })}
          >
            <Select
              allowClear
              loading={decksLoading}
              placeholder={t("option:flashcards.selectDeck", {
                defaultValue: "Select deck"
              })}
              options={decks.map((d) => ({
                label: formatDeckDisplayName(d, `Deck ${d.id}`),
                value: d.id
              }))}
            />
          </Form.Item>
          <Form.Item
            name="model_type"
            label={t("option:flashcards.modelType", {
              defaultValue: "Card model"
            })}
          >
            <Select
              options={[
                {
                  label: t("option:flashcards.templateBasic", {
                    defaultValue: "Basic (Question - Answer)"
                  }),
                  value: "basic"
                },
                {
                  label: t("option:flashcards.templateReverse", {
                    defaultValue: "Basic + Reverse (Both directions)"
                  }),
                  value: "basic_reverse"
                },
                {
                  label: t("option:flashcards.templateCloze", {
                    defaultValue: "Cloze (Fill in the blank)"
                  }),
                  value: "cloze"
                }
              ]}
              onChange={(value: FlashcardModelType) => {
                syncTemplateFields({ model_type: value })
              }}
            />
          </Form.Item>
          <Text type="secondary" className="block text-[11px] -mt-4 mb-3">
            {templateHelperText}
          </Text>
          {isClozeTemplate && (
            <Text type="secondary" className="block text-[11px] -mt-2 mb-3">
              {t("option:flashcards.clozeSyntaxHelp", {
                defaultValue:
                  "Cloze syntax: add at least one deletion like {{syntax}} in Front text.",
                syntax: "{{c1::answer}}"
              })}
            </Text>
          )}
          <Form.Item
            name="tags"
            label={t("option:flashcards.tags", { defaultValue: "Tags" })}
          >
            <FlashcardTagPicker
              active={open}
              dataTestId="flashcards-edit-tag-picker"
              placeholder={t("option:flashcards.tagsPlaceholder", {
                defaultValue: "Add tags..."
              })}
            />
          </Form.Item>
          {sourceMeta && (
            <div className="-mt-2 mb-2">
              <Text type="secondary" className="block text-[11px]">
                {t("option:flashcards.source", { defaultValue: "Source" })}
              </Text>
              {sourceMeta.href ? (
                <a href={sourceMeta.href} className="text-sm text-primary hover:underline">
                  {sourceMeta.label}
                </a>
              ) : (
                <Text className="text-sm">{sourceMeta.label}</Text>
              )}
            </div>
          )}
        </div>

        {/* Section: Scheduling (read-only metadata) */}
        {card && (
          <div className="mb-6 rounded border border-border bg-surface p-3">
            <h3 className="text-sm font-medium text-text-muted mb-3">
              {t("option:flashcards.scheduling", { defaultValue: "Scheduling" })}
            </h3>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <Text type="secondary" className="block text-[11px]">
                  <Tooltip
                    title={t("option:flashcards.schedulingMemoryStrengthHelp", {
                      defaultValue: "SM-2 ease factor (how fast review gaps grow)."
                    })}
                  >
                    <span>
                      {t("option:flashcards.memoryStrength", {
                        defaultValue: "Memory strength"
                      })}
                    </span>
                  </Tooltip>
                </Text>
                <Text>{card.ef.toFixed(2)}</Text>
              </div>
              <div>
                <Text type="secondary" className="block text-[11px]">
                  <Tooltip
                    title={t("option:flashcards.schedulingNextGapHelp", {
                      defaultValue: "SM-2 interval (days until next review)."
                    })}
                  >
                    <span>
                      {t("option:flashcards.nextReviewGap", {
                        defaultValue: "Next review gap"
                      })}
                    </span>
                  </Tooltip>
                </Text>
                <Text>
                  {t("option:flashcards.intervalDaysShort", {
                    defaultValue: "{{count}}d",
                    count: Math.max(0, card.interval_days)
                  })}
                </Text>
              </div>
              <div>
                <Text type="secondary" className="block text-[11px]">
                  <Tooltip
                    title={t("option:flashcards.schedulingRecallRunsHelp", {
                      defaultValue: "SM-2 repetitions (successful recalls)."
                    })}
                  >
                    <span>
                      {t("option:flashcards.recallRuns", {
                        defaultValue: "Recall runs"
                      })}
                    </span>
                  </Tooltip>
                </Text>
                <Text>{Math.max(0, card.repetitions)}</Text>
              </div>
              <div>
                <Text type="secondary" className="block text-[11px]">
                  <Tooltip
                    title={t("option:flashcards.schedulingRelearnsHelp", {
                      defaultValue: "SM-2 lapses (times forgotten)."
                    })}
                  >
                    <span>
                      {t("option:flashcards.relearns", {
                        defaultValue: "Relearns"
                      })}
                    </span>
                  </Tooltip>
                </Text>
                <Text>{Math.max(0, card.lapses)}</Text>
              </div>
            </div>

            <div className="mt-3 space-y-2 text-xs">
              <div>
                <Text type="secondary" className="block text-[11px]">
                  {t("option:flashcards.dueAt", { defaultValue: "Due at" })}
                </Text>
                <Text>
                  {card.due_at
                    ? t("option:flashcards.timestampWithRelative", {
                        defaultValue: "{{absolute}} ({{relative}})",
                        absolute: dayjs(card.due_at).format("YYYY-MM-DD HH:mm"),
                        relative: dayjs(card.due_at).fromNow()
                      })
                    : t("option:flashcards.notScheduled", {
                        defaultValue: "Not scheduled"
                      })}
                </Text>
              </div>
              <div>
                <Text type="secondary" className="block text-[11px]">
                  {t("option:flashcards.lastReviewed", { defaultValue: "Last reviewed" })}
                </Text>
                <Text>
                  {card.last_reviewed_at
                    ? t("option:flashcards.timestampWithRelative", {
                        defaultValue: "{{absolute}} ({{relative}})",
                        absolute: dayjs(card.last_reviewed_at).format("YYYY-MM-DD HH:mm"),
                        relative: dayjs(card.last_reviewed_at).fromNow()
                      })
                    : t("option:flashcards.neverReviewed", {
                        defaultValue: "Never"
                      })}
                </Text>
              </div>
            </div>

            {onResetScheduling && (
              <div className="mt-3 border-t border-border pt-3">
                <Text type="secondary" className="block text-[11px] mb-2">
                  {t("option:flashcards.resetSchedulingHelp", {
                    defaultValue:
                      "Reset scheduling when this card's timing feels wrong. This keeps card content but starts scheduling from new-card defaults."
                  })}
                </Text>
                <Button
                  danger
                  size="small"
                  onClick={() => setConfirmResetOpen(true)}
                  disabled={isLoading}
                >
                  {t("option:flashcards.resetScheduling", {
                    defaultValue: "Reset scheduling"
                  })}
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Section: Content */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-text-muted">
              {t("option:flashcards.content", { defaultValue: "Content" })}
            </h3>
            <button
              type="button"
              className="text-xs text-primary hover:underline"
              onClick={() => setShowPreview(!showPreview)}
            >
              {showPreview
                ? t("option:flashcards.hidePreview", { defaultValue: "Hide preview" })
                : t("option:flashcards.showPreview", { defaultValue: "Show preview" })}
            </button>
          </div>
          <Form.Item
            name="front"
            label={renderFieldLabel(
              "front",
              t("option:flashcards.front", { defaultValue: "Front" })
            )}
            rules={[
              {
                required: true,
                message: t("option:flashcards.frontRequired", {
                  defaultValue: "Front is required."
                })
              },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (getFieldValue("model_type") !== "cloze") {
                    return Promise.resolve()
                  }
                  const frontText = String(value ?? "")
                  if (CLOZE_PATTERN.test(frontText)) {
                    return Promise.resolve()
                  }
                  return Promise.reject(
                    new Error(
                      t("option:flashcards.clozeValidationMessage", {
                        defaultValue:
                          "For Cloze cards, include at least one deletion like {{syntax}}.",
                        syntax: "{{c1::answer}}"
                      })
                    )
                  )
                }
              }),
              {
                validator(_, value) {
                  const byteLength = getUtf8ByteLength(String(value ?? ""))
                  if (byteLength <= FLASHCARD_FIELD_MAX_BYTES) {
                    return Promise.resolve()
                  }
                  return Promise.reject(
                    new Error(
                      t("option:flashcards.fieldByteValidation", {
                        defaultValue: "{{field}} must be {{max}} bytes or fewer.",
                        field: t("option:flashcards.front", { defaultValue: "Front" }),
                        max: FLASHCARD_FIELD_MAX_BYTES
                      })
                    )
                  )
                }
              }
            ]}
          >
            <Input.TextArea
              ref={(instance) => {
                textAreaRefs.current.front = instance
              }}
              rows={3}
              onSelect={(event) => updateSelection("front", event.currentTarget)}
              onClick={(event) => updateSelection("front", event.currentTarget)}
              onKeyUp={(event) => updateSelection("front", event.currentTarget)}
            />
          </Form.Item>
          <Text
            type={frontLimitState === "over" ? "danger" : frontLimitState === "warning" ? "warning" : "secondary"}
            className="block text-[11px] -mt-4 mb-2"
          >
            {renderByteUsageHint("front", frontByteLength, frontLimitState)}
          </Text>
          <Text type="secondary" className="block text-[11px] mb-3">
            {markdownSupportHint}
          </Text>
          {showPreview && frontPreview && (
            <div className="mb-4 border rounded p-2 text-xs bg-surface">
              <Text type="secondary" className="block text-[11px] mb-1">
                {previewLabel}
              </Text>
              <MarkdownWithBoundary content={frontPreview || ""} size="xs" />
            </div>
          )}

          <Form.Item
            name="back"
            label={renderFieldLabel(
              "back",
              t("option:flashcards.back", { defaultValue: "Back" })
            )}
            rules={[
              {
                required: true,
                message: t("option:flashcards.backRequired", {
                  defaultValue: "Back is required."
                })
              },
              {
                validator(_, value) {
                  const byteLength = getUtf8ByteLength(String(value ?? ""))
                  if (byteLength <= FLASHCARD_FIELD_MAX_BYTES) {
                    return Promise.resolve()
                  }
                  return Promise.reject(
                    new Error(
                      t("option:flashcards.fieldByteValidation", {
                        defaultValue: "{{field}} must be {{max}} bytes or fewer.",
                        field: t("option:flashcards.back", { defaultValue: "Back" }),
                        max: FLASHCARD_FIELD_MAX_BYTES
                      })
                    )
                  )
                }
              }
            ]}
          >
            <Input.TextArea
              ref={(instance) => {
                textAreaRefs.current.back = instance
              }}
              rows={5}
              onSelect={(event) => updateSelection("back", event.currentTarget)}
              onClick={(event) => updateSelection("back", event.currentTarget)}
              onKeyUp={(event) => updateSelection("back", event.currentTarget)}
            />
          </Form.Item>
          <Text
            type={backLimitState === "over" ? "danger" : backLimitState === "warning" ? "warning" : "secondary"}
            className="block text-[11px] -mt-4 mb-2"
          >
            {renderByteUsageHint("back", backByteLength, backLimitState)}
          </Text>
          <Text type="secondary" className="block text-[11px] mb-3">
            {markdownSupportHint}
          </Text>
          {showPreview && backPreview && (
            <div className="mb-4 border rounded p-2 text-xs bg-surface">
              <Text type="secondary" className="block text-[11px] mb-1">
                {previewLabel}
              </Text>
              <MarkdownWithBoundary content={backPreview || ""} size="xs" />
            </div>
          )}
        </div>

        {/* Section: Additional (collapsed) */}
        <Collapse
          ghost
          items={[
            {
              key: "additional",
              label: (
                <span className="inline-flex items-center gap-2">
                  {t("option:flashcards.additionalFields", {
                    defaultValue: "Additional fields"
                  })}
                  {additionalFieldCount > 0 && (
                    <Badge
                      count={additionalFieldCount}
                      size="small"
                      title={t("option:flashcards.additionalFieldsSet", {
                        defaultValue: "{{count}} field(s) set",
                        count: additionalFieldCount
                      })}
                    />
                  )}
                </span>
              ),
              children: (
                <>
                  <Form.Item
                    name="extra"
                    label={renderFieldLabel(
                      "extra",
                      t("option:flashcards.extra", { defaultValue: "Extra" })
                    )}
                  >
                    <Input.TextArea
                      ref={(instance) => {
                        textAreaRefs.current.extra = instance
                      }}
                      rows={2}
                      onSelect={(event) => updateSelection("extra", event.currentTarget)}
                      onClick={(event) => updateSelection("extra", event.currentTarget)}
                      onKeyUp={(event) => updateSelection("extra", event.currentTarget)}
                    />
                  </Form.Item>
                  {showPreview && extraPreview && (
                    <div className="mb-4 border rounded p-2 text-xs bg-surface">
                      <MarkdownWithBoundary content={extraPreview || ""} size="xs" />
                    </div>
                  )}
                  <Form.Item
                    name="notes"
                    label={renderFieldLabel(
                      "notes",
                      t("option:flashcards.notes", { defaultValue: "Notes" })
                    )}
                  >
                    <Input.TextArea
                      ref={(instance) => {
                        textAreaRefs.current.notes = instance
                      }}
                      rows={2}
                      onSelect={(event) => updateSelection("notes", event.currentTarget)}
                      onClick={(event) => updateSelection("notes", event.currentTarget)}
                      onKeyUp={(event) => updateSelection("notes", event.currentTarget)}
                    />
                  </Form.Item>
                  {showPreview && notesPreview && (
                    <div className="mb-4 border rounded p-2 text-xs bg-surface">
                      <MarkdownWithBoundary content={notesPreview || ""} size="xs" />
                    </div>
                  )}
                </>
              )
            }
          ]}
        />

        {/* Hidden fields */}
        <Form.Item name="expected_version" hidden>
          <Input type="number" />
        </Form.Item>
        <Form.Item name="reverse" hidden>
          <Input />
        </Form.Item>
        <Form.Item name="is_cloze" hidden>
          <Input />
        </Form.Item>
      </Form>
    </Drawer>

    {/* Confirmation modal for unsaved changes */}
    <Modal
      open={confirmCloseOpen}
      title={t("option:flashcards.unsavedChangesTitle", {
        defaultValue: "Unsaved changes"
      })}
      onCancel={() => setConfirmCloseOpen(false)}
      footer={[
        <Button key="cancel" onClick={() => setConfirmCloseOpen(false)}>
          {t("common:cancel", { defaultValue: "Cancel" })}
        </Button>,
        <Button key="discard" danger onClick={handleClose}>
          {t("common:discard", { defaultValue: "Discard" })}
        </Button>
      ]}
    >
      <p>
        {t("option:flashcards.unsavedChangesDescription", {
          defaultValue: "You have unsaved changes. Are you sure you want to close?"
        })}
      </p>
    </Modal>

    <Modal
      open={confirmResetOpen}
      title={t("option:flashcards.resetSchedulingConfirmTitle", {
        defaultValue: "Reset scheduling for this card?"
      })}
      onCancel={() => setConfirmResetOpen(false)}
      footer={[
        <Button key="cancel" onClick={() => setConfirmResetOpen(false)}>
          {t("common:cancel", { defaultValue: "Cancel" })}
        </Button>,
        <Button
          key="reset"
          danger
          loading={isLoading}
          onClick={handleConfirmResetScheduling}
        >
          {t("option:flashcards.resetScheduling", {
            defaultValue: "Reset scheduling"
          })}
        </Button>
      ]}
    >
      <div className="space-y-2">
        {card && (
          <div className="space-y-1 text-xs">
            <Text type="secondary" className="block">
              {t("option:flashcards.resetSchedulingCurrentSummary", {
                defaultValue: "Current scheduling state:"
              })}
            </Text>
            <Text className="block">
              {t("option:flashcards.resetSchedulingCurrentEf", {
                defaultValue: "Memory strength: {{value}}",
                value: card.ef.toFixed(2)
              })}
            </Text>
            <Text className="block">
              {t("option:flashcards.resetSchedulingCurrentInterval", {
                defaultValue: "Next review gap: {{count}} day(s)",
                count: Math.max(0, card.interval_days)
              })}
            </Text>
            <Text className="block">
              {t("option:flashcards.resetSchedulingCurrentRepetitions", {
                defaultValue: "Recall runs: {{count}}",
                count: Math.max(0, card.repetitions)
              })}
            </Text>
            <Text className="block">
              {t("option:flashcards.resetSchedulingCurrentLapses", {
                defaultValue: "Relearns: {{count}}",
                count: Math.max(0, card.lapses)
              })}
            </Text>
          </div>
        )}
        <p>
          {t("option:flashcards.resetSchedulingConfirmDescription", {
            defaultValue:
              "This will reset memory strength, next review gap, recall runs, and relearns to new-card defaults. Card content and tags will not change."
          })}
        </p>
      </div>
    </Modal>
    </>
  )
}

export default FlashcardEditDrawer
