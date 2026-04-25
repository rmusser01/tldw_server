import React from "react"
import {
  Badge,
  Button,
  Collapse,
  Divider,
  Drawer,
  Form,
  Input,
  Select,
  Space,
  Typography
} from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"
import { Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import {
  useDecksQuery,
  useCreateFlashcardMutation,
  useCreateFlashcardTemplateMutation,
  useCreateDeckMutation,
  useDebouncedFormField,
  type UseFlashcardQueriesOptions
} from "../hooks"
import { FLASHCARDS_DRAWER_WIDTH_PX } from "../constants"
import { MarkdownWithBoundary } from "./MarkdownWithBoundary"
import { FlashcardImageInsertButton } from "./FlashcardImageInsertButton"
import { FlashcardDeckReferenceSection } from "./FlashcardDeckReferenceSection"
import { FlashcardTagPicker } from "./FlashcardTagPicker"
import { DeckSchedulerSettingsEditor } from "./DeckSchedulerSettingsEditor"
import { FlashcardTemplateValueModal } from "./FlashcardTemplateValueModal"
import { FlashcardSaveTemplateModal } from "./FlashcardSaveTemplateModal"
import { normalizeFlashcardTemplateFields } from "../utils/template-helpers"
import { formatDeckDisplayName } from "../utils/deck-display"
import { normalizeOptionalFlashcardTags } from "../utils/tag-normalization"
import {
  getSelectionFromElement,
  insertTextAtSelection,
  restoreSelection,
  type TextSelection
} from "../utils/text-selection"
import {
  FLASHCARD_FIELD_MAX_BYTES,
  getFlashcardFieldLimitState,
  getUtf8ByteLength
} from "../utils/field-byte-limit"
import type { FlashcardCreate, Deck, FlashcardTemplateCreate } from "@/services/flashcards"
import { useDeckSchedulerDraft } from "../hooks/useDeckSchedulerDraft"
import { formatSchedulerSummary } from "../utils/scheduler-settings"

const { Text } = Typography
const CLOZE_PATTERN = /\{\{c\d+::[\s\S]+?\}\}/
type FlashcardModelType = NonNullable<FlashcardCreate["model_type"]>
type EditableTextField = "front" | "back" | "extra" | "notes"

interface PreviewProps {
  content?: string
  showPreview: boolean
}

const Preview: React.FC<PreviewProps> = ({ content, showPreview }) => {
  const { t } = useTranslation(["option"])
  if (!showPreview || !content) return null
  return (
    <div className="mt-2 rounded border border-border bg-surface p-2 text-xs">
      <Text type="secondary" className="block text-[11px] mb-1">
        {t("flashcards.preview", { defaultValue: "Preview" })}
      </Text>
      <MarkdownWithBoundary content={content} size="xs" />
    </div>
  )
}

interface FlashcardCreateDrawerProps {
  open: boolean
  onClose: () => void
  decks?: Deck[]
  decksLoading?: boolean
  onSuccess?: () => void
}

type FlashcardCreateDrawerVisibilityProps = Pick<
  UseFlashcardQueriesOptions,
  "includeWorkspaceItems" | "workspaceId"
>

export const FlashcardCreateDrawer: React.FC<
  FlashcardCreateDrawerProps & FlashcardCreateDrawerVisibilityProps
> = ({
  open,
  onClose,
  decks: propDecks,
  decksLoading: propDecksLoading,
  onSuccess,
  includeWorkspaceItems,
  workspaceId
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()

  // Form and state
  const [form] = Form.useForm<FlashcardCreate>()
  const selectedModelType = Form.useWatch("model_type", form) as
    | FlashcardModelType
    | undefined
  const selectedDeckId = Form.useWatch("deck_id", form) as number | undefined
  const [showPreview, setShowPreview] = React.useState(false)
  const frontPreview = useDebouncedFormField(form, "front")
  const backPreview = useDebouncedFormField(form, "back")
  const extraPreview = useDebouncedFormField(form, "extra")
  const notesPreview = useDebouncedFormField(form, "notes")
  const tagsValue = useDebouncedFormField(form, "tags")
  const selectedTags = Form.useWatch("tags", form) as string[] | undefined
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

  // Count how many advanced fields have values for the badge indicator
  const advancedFieldCount = React.useMemo(() => {
    let count = 0
    if (Array.isArray(tagsValue) && tagsValue.length > 0) count++
    if (extraPreview && extraPreview.trim()) count++
    if (notesPreview && notesPreview.trim()) count++
    return count
  }, [tagsValue, extraPreview, notesPreview])

  // Inline deck creation state
  const [showInlineCreate, setShowInlineCreate] = React.useState(false)
  const [inlineDeckName, setInlineDeckName] = React.useState("")
  const [templateValueModalOpen, setTemplateValueModalOpen] = React.useState(false)
  const [saveTemplateModalOpen, setSaveTemplateModalOpen] = React.useState(false)
  const [saveTemplateInitialValues, setSaveTemplateInitialValues] = React.useState<Partial<FlashcardTemplateCreate> | null>(null)
  const inlineSchedulerDraft = useDeckSchedulerDraft()

  // Queries and mutations - use props if provided, otherwise fetch
  const decksQuery = useDecksQuery({
    enabled: !propDecks,
    includeWorkspaceItems,
    workspaceId
  })
  const decks = propDecks ?? decksQuery.data ?? []
  const decksLoading = propDecksLoading ?? decksQuery.isLoading
  const selectedDeck = React.useMemo(
    () => decks.find((deck) => deck.id === selectedDeckId) ?? null,
    [decks, selectedDeckId]
  )

  const createMutation = useCreateFlashcardMutation()
  const createTemplateMutation = useCreateFlashcardTemplateMutation()
  const createDeckMutation = useCreateDeckMutation()
  const isClozeTemplate = selectedModelType === "cloze"
  const frontByteLength = getUtf8ByteLength(frontValue)
  const backByteLength = getUtf8ByteLength(backValue)
  const frontLimitState = getFlashcardFieldLimitState(frontByteLength)
  const backLimitState = getFlashcardFieldLimitState(backByteLength)

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

  // Reset form when drawer opens
  React.useEffect(() => {
    if (open) {
      form.resetFields()
      setShowPreview(false)
      setShowInlineCreate(false)
      setTemplateValueModalOpen(false)
      setSaveTemplateModalOpen(false)
      setSaveTemplateInitialValues(null)
      setInlineDeckName("")
      inlineSchedulerDraft.resetToDefaults()
    }
  }, [form, inlineSchedulerDraft.resetToDefaults, open])

  // Create new deck (inline)
  const handleInlineCreateDeck = async () => {
    try {
      if (!inlineDeckName.trim()) {
        message.error(
          t("option:flashcards.newDeckNameRequired", {
            defaultValue: "Enter a deck name."
          })
        )
        return
      }
      const schedulerSettings = inlineSchedulerDraft.getValidatedSettings()
      if (!schedulerSettings) return
      const deck = await createDeckMutation.mutateAsync({
        name: inlineDeckName.trim(),
        scheduler_type: schedulerSettings.scheduler_type,
        scheduler_settings: schedulerSettings.scheduler_settings
      })
      message.success(t("common:created", { defaultValue: "Created" }))
      setShowInlineCreate(false)
      setInlineDeckName("")
      inlineSchedulerDraft.resetToDefaults()
      form.setFieldsValue({ deck_id: deck.id })
    } catch (e: unknown) {
      const errorMessage =
        e instanceof Error ? e.message : "Failed to create deck"
      message.error(errorMessage)
    }
  }

  // Create flashcard
  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      await createMutation.mutateAsync(
        normalizeFlashcardTemplateFields({
          ...values,
          tags: normalizeOptionalFlashcardTags(values.tags)
        })
      )
      message.success(t("common:created", { defaultValue: "Created" }))
      form.resetFields()
      onSuccess?.()
      onClose()
    } catch (e: unknown) {
      if (e && typeof e === "object" && "errorFields" in e) return // form validation
      const errorMessage = e instanceof Error ? e.message : "Create failed"
      message.error(errorMessage)
    }
  }

  // Create and add another
  const handleCreateAndAddAnother = async () => {
    try {
      const values = await form.validateFields()
      await createMutation.mutateAsync(
        normalizeFlashcardTemplateFields({
          ...values,
          tags: normalizeOptionalFlashcardTags(values.tags)
        })
      )
      message.success(t("common:created", { defaultValue: "Created" }))
      form.resetFields(["front", "back", "extra", "notes", "tags"])
      onSuccess?.()
    } catch (e: unknown) {
      if (e && typeof e === "object" && "errorFields" in e) return
      const errorMessage = e instanceof Error ? e.message : "Create failed"
      message.error(errorMessage)
    }
  }

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

  const handleApplyTemplateDraft = React.useCallback(
    (
      draft: Pick<FlashcardCreate, "deck_id" | "tags" | "model_type" | "front" | "back" | "notes" | "extra">
    ) => {
      form.setFieldsValue(
        normalizeFlashcardTemplateFields({
          deck_id: draft.deck_id ?? undefined,
          tags: draft.tags ?? undefined,
          model_type: draft.model_type,
          front: draft.front ?? "",
          back: draft.back ?? "",
          notes: draft.notes ?? "",
          extra: draft.extra ?? ""
        })
      )
      setTemplateValueModalOpen(false)
    },
    [form]
  )

  const handleOpenSaveTemplate = React.useCallback(() => {
    const currentValues = form.getFieldsValue(["model_type", "front", "back", "notes", "extra"])
    setSaveTemplateInitialValues({
      model_type: (currentValues.model_type as FlashcardTemplateCreate["model_type"] | undefined) ?? "basic",
      front_template: String(currentValues.front ?? ""),
      back_template: currentValues.back == null ? null : String(currentValues.back),
      notes_template: currentValues.notes == null ? null : String(currentValues.notes),
      extra_template: currentValues.extra == null ? null : String(currentValues.extra),
      placeholder_definitions: []
    })
    setSaveTemplateModalOpen(true)
  }, [form])

  const handleSaveTemplate = React.useCallback(
    async (values: FlashcardTemplateCreate) => {
      try {
        await createTemplateMutation.mutateAsync(values)
        message.success(
          t("common:created", {
            defaultValue: "Created"
          })
        )
      } catch (error: unknown) {
        message.error(error instanceof Error ? error.message : "Failed to save template")
        throw error
      }
    },
    [createTemplateMutation, message, t]
  )

  return (
    <>
    <Drawer
      placement="right"
      styles={{ wrapper: { width: FLASHCARDS_DRAWER_WIDTH_PX } }}
      open={open}
      onClose={onClose}
      title={t("option:flashcards.createCard", { defaultValue: "Create Flashcard" })}
      footer={
        <div className="flex justify-end">
          <Space>
            <Button onClick={onClose}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button
              onClick={handleCreateAndAddAnother}
              loading={createMutation.isPending}
            >
              {t("option:flashcards.createAndAddAnother", {
                defaultValue: "Create & Add Another"
              })}
            </Button>
            <Button
              type="primary"
              onClick={handleCreate}
              loading={createMutation.isPending}
            >
              {t("common:create", { defaultValue: "Create" })}
            </Button>
          </Space>
        </div>
      }
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          is_cloze: false,
          model_type: "basic",
          reverse: false
        }}
      >
        {/* Section: Organization */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-text-muted mb-3">
            {t("option:flashcards.organization", { defaultValue: "Organization" })}
          </h3>
          {!showInlineCreate ? (
            <>
              <Form.Item
                name="deck_id"
                label={t("option:flashcards.deck", { defaultValue: "Deck" })}
                className="!mb-0"
              >
                <Select
                  placeholder={t("option:flashcards.selectDeck", {
                    defaultValue: "Select deck"
                  })}
                  allowClear
                  loading={decksLoading}
                  className="w-full"
                  options={decks.map((d) => ({
                    label: formatDeckDisplayName(d, `Deck ${d.id}`),
                    value: d.id
                  }))}
                  popupRender={(menu) => (
                    <>
                      {menu}
                      <Divider className="!my-2" />
                      <button
                        type="button"
                        className="w-full text-left px-3 py-2 text-primary hover:bg-primary/5 flex items-center gap-2"
                        onClick={(e) => {
                          e.preventDefault()
                          setShowInlineCreate(true)
                        }}
                      >
                        <Plus className="size-4" />
                        {t("option:flashcards.createNewDeck", {
                          defaultValue: "Create new deck"
                        })}
                      </button>
                    </>
                  )}
                />
              </Form.Item>
              <button
                type="button"
                className="text-xs text-primary hover:text-primaryStrong -mt-2 mb-2 block"
                onClick={() => setShowInlineCreate(true)}
                data-testid="flashcards-create-new-deck-link"
              >
                {t("option:flashcards.orCreateNewDeck", {
                  defaultValue: "or create a new deck"
                })}
              </button>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <Input
                  placeholder={t("option:flashcards.newDeckNamePlaceholder", {
                    defaultValue: "New deck name"
                  })}
                  value={inlineDeckName}
                  onChange={(e) => setInlineDeckName(e.target.value)}
                  className="flex-1"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleInlineCreateDeck()
                    if (e.key === "Escape") {
                      setShowInlineCreate(false)
                      setInlineDeckName("")
                      inlineSchedulerDraft.resetToDefaults()
                    }
                  }}
                />
              </div>
              <DeckSchedulerSettingsEditor
                schedulerDraft={inlineSchedulerDraft}
              />
              <div className="flex items-center gap-2">
                <Button
                  type="primary"
                  size="small"
                  onClick={handleInlineCreateDeck}
                  loading={createDeckMutation.isPending}
                  data-testid="flashcards-inline-create-deck-submit"
                >
                  {t("common:create", { defaultValue: "Create" })}
                </Button>
                <Button
                  size="small"
                  data-testid="flashcards-inline-create-deck-cancel"
                  onClick={() => {
                    setShowInlineCreate(false)
                    setInlineDeckName("")
                    inlineSchedulerDraft.resetToDefaults()
                  }}
                >
                  {t("common:cancel", { defaultValue: "Cancel" })}
                </Button>
              </div>
              <Text type="secondary" className="block text-xs">
                {t("option:flashcards.schedulerTabRedirectHint", {
                  defaultValue: "Use the Scheduler tab later if you want to refine this deck further."
                })}
              </Text>
            </>
          )}

          {!showInlineCreate && selectedDeck && (
            <Text type="secondary" className="block text-xs -mt-2 mb-3">
              {formatSchedulerSummary(selectedDeck.scheduler_type, selectedDeck.scheduler_settings)}
            </Text>
          )}

          <div className="mb-3 flex flex-wrap gap-2">
            <Button size="small" onClick={() => setTemplateValueModalOpen(true)}>
              {t("option:flashcards.applyTemplate", {
                defaultValue: "Apply template"
              })}
            </Button>
            <Button size="small" onClick={handleOpenSaveTemplate}>
              {t("option:flashcards.saveAsTemplate", {
                defaultValue: "Save as template"
              })}
            </Button>
          </div>

          {/* Card model */}
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
            />
          </Form.Item>
          <Text type="secondary" className="block text-xs -mt-4 mb-3">
            {templateHelperText}
          </Text>
          {isClozeTemplate && (
            <Text type="secondary" className="block text-xs -mt-2 mb-3">
              {t("option:flashcards.clozeSyntaxHelp", {
                defaultValue:
                  "Cloze syntax: add at least one deletion like {{syntax}} in Front text.",
                syntax: "{{c1::answer}}"
              })}
            </Text>
          )}
        </div>

        <FlashcardDeckReferenceSection
          open={open}
          deckId={selectedDeckId ?? null}
          deckName={selectedDeck?.name ?? null}
          includeWorkspaceItems={includeWorkspaceItems}
          workspaceId={workspaceId}
        />

        {/* Hidden fields for API compatibility */}
        <Form.Item name="reverse" hidden>
          <Input />
        </Form.Item>
        <Form.Item name="is_cloze" hidden>
          <Input />
        </Form.Item>

        {/* Section: Content */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-text-muted mb-3">
            {t("option:flashcards.content", { defaultValue: "Content" })}
          </h3>

          <Collapse
            ghost
            size="small"
            className="mb-3"
            defaultActiveKey={isClozeTemplate ? ["tips"] : undefined}
            key={isClozeTemplate ? "cloze-tips" : "default-tips"}
            items={[
              {
                key: "tips",
                label: t("option:flashcards.writingTipsHeader", {
                  defaultValue: "Tips for effective flashcards"
                }),
                children: (
                  <ul className="list-disc pl-4 text-xs text-text-muted space-y-1">
                    <li>{t("option:flashcards.writingTip1", { defaultValue: "Keep each card focused on one concept" })}</li>
                    <li>{t("option:flashcards.writingTip2", { defaultValue: "Use simple, clear language on the front" })}</li>
                    <li>{t("option:flashcards.writingTip3", { defaultValue: "Include context clues but avoid giving away the answer" })}</li>
                    <li>{t("option:flashcards.writingTip4", { defaultValue: "Use images or diagrams when they help understanding" })}</li>
                    <li>{t("option:flashcards.writingTip5", { defaultValue: "For cloze deletions, use {{syntax}} syntax", syntax: "{{c1::answer}}" })}</li>
                  </ul>
                )
              }
            ]}
          />

          {/* Front - required */}
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
              placeholder={t("option:flashcards.frontPlaceholder", {
                defaultValue: "Question or prompt..."
              })}
              onSelect={(event) => updateSelection("front", event.currentTarget)}
              onClick={(event) => updateSelection("front", event.currentTarget)}
              onKeyUp={(event) => updateSelection("front", event.currentTarget)}
            />
          </Form.Item>
          <Text
            type={frontLimitState === "over" ? "danger" : frontLimitState === "warning" ? "warning" : "secondary"}
            className="block text-[11px] -mt-4 mb-3"
          >
            {renderByteUsageHint("front", frontByteLength, frontLimitState)}
          </Text>
          <Preview content={frontPreview} showPreview={showPreview} />

          {/* Back - required */}
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
              placeholder={t("option:flashcards.backPlaceholder", {
                defaultValue: "Answer..."
              })}
              onSelect={(event) => updateSelection("back", event.currentTarget)}
              onClick={(event) => updateSelection("back", event.currentTarget)}
              onKeyUp={(event) => updateSelection("back", event.currentTarget)}
            />
          </Form.Item>
          <Text
            type={backLimitState === "over" ? "danger" : backLimitState === "warning" ? "warning" : "secondary"}
            className="block text-[11px] -mt-4 mb-3"
          >
            {renderByteUsageHint("back", backByteLength, backLimitState)}
          </Text>
          <Preview content={backPreview} showPreview={showPreview} />

          {/* Preview toggle and help text */}
          <div className="flex items-center gap-4">
            <button
              type="button"
              className="text-xs text-primary hover:text-primaryStrong"
              onClick={() => setShowPreview((v) => !v)}
            >
              {showPreview
                ? t("option:flashcards.hidePreview", { defaultValue: "Hide preview" })
                : t("option:flashcards.showPreview", { defaultValue: "Show preview" })}
            </button>
            <Text type="secondary" className="text-xs">
              {t("option:flashcards.markdownHint", {
                defaultValue: "Supports Markdown and LaTeX"
              })}
            </Text>
          </div>
        </div>

        {/* Advanced options - collapsed by default */}
        <Collapse
          ghost
          className="-mx-4"
          items={[
            {
              key: "advanced",
              label: (
                <span className="inline-flex items-center gap-2">
                  <Text type="secondary">
                    {t("option:flashcards.advancedOptions", {
                      defaultValue: "Advanced options (tags, extra, notes)"
                    })}
                  </Text>
                  {advancedFieldCount > 0 && (
                    <Badge
                      count={advancedFieldCount}
                      size="small"
                      title={t("option:flashcards.advancedFieldsSet", {
                        defaultValue: "{{count}} field(s) set",
                        count: advancedFieldCount
                      })}
                    />
                  )}
                </span>
              ),
              children: (
                <div className="space-y-4">
                  <Form.Item
                    name="tags"
                    label={t("option:flashcards.tags", { defaultValue: "Tags" })}
                    className="!mb-0"
                  >
                    <FlashcardTagPicker
                      active={open}
                      dataTestId="flashcards-create-tag-picker"
                      placeholder={t("option:flashcards.tagsPlaceholder", {
                        defaultValue: "tag1, tag2"
                      })}
                    />
                  </Form.Item>

                  <Form.Item
                    name="extra"
                    label={renderFieldLabel(
                      "extra",
                      t("option:flashcards.extra", { defaultValue: "Extra" })
                    )}
                    className="!mb-0"
                  >
                    <Input.TextArea
                      ref={(instance) => {
                        textAreaRefs.current.extra = instance
                      }}
                      rows={2}
                      placeholder={t("option:flashcards.extraPlaceholder", {
                        defaultValue: "Optional hints or explanations..."
                      })}
                      onSelect={(event) => updateSelection("extra", event.currentTarget)}
                      onClick={(event) => updateSelection("extra", event.currentTarget)}
                      onKeyUp={(event) => updateSelection("extra", event.currentTarget)}
                    />
                  </Form.Item>
                  <Preview content={extraPreview} showPreview={showPreview} />

                  <Form.Item
                    name="notes"
                    label={renderFieldLabel(
                      "notes",
                      t("option:flashcards.notes", { defaultValue: "Notes" })
                    )}
                    className="!mb-0"
                  >
                    <Input.TextArea
                      ref={(instance) => {
                        textAreaRefs.current.notes = instance
                      }}
                      rows={2}
                      placeholder={t("option:flashcards.notesPlaceholder", {
                        defaultValue: "Internal notes (not shown during review)..."
                      })}
                      onSelect={(event) => updateSelection("notes", event.currentTarget)}
                      onClick={(event) => updateSelection("notes", event.currentTarget)}
                      onKeyUp={(event) => updateSelection("notes", event.currentTarget)}
                    />
                  </Form.Item>
                  <Preview content={notesPreview} showPreview={showPreview} />
                </div>
              )
            }
          ]}
        />
      </Form>
    </Drawer>
    {templateValueModalOpen ? (
      <FlashcardTemplateValueModal
        open={templateValueModalOpen}
        onClose={() => setTemplateValueModalOpen(false)}
        onApply={handleApplyTemplateDraft}
        draftDefaults={{
          deck_id: selectedDeckId,
          tags: normalizeOptionalFlashcardTags(selectedTags)
        }}
      />
    ) : null}
    {saveTemplateModalOpen ? (
      <FlashcardSaveTemplateModal
        open={saveTemplateModalOpen}
        onClose={() => setSaveTemplateModalOpen(false)}
        initialValues={saveTemplateInitialValues}
        onSave={handleSaveTemplate}
        isSaving={createTemplateMutation.isPending}
      />
    ) : null}
    </>
  )
}

export default FlashcardCreateDrawer
