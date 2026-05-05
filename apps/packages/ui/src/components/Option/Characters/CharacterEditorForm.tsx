import React from "react"
import { Button, Form, Input, InputNumber, Modal, Select } from "antd"
import type { FormInstance, InputRef } from "antd"
import { ChevronDown, ChevronUp } from "lucide-react"
import type { TFunction } from "i18next"
import type { CharacterField } from "@/services/character-generation"
import { CHARACTER_PROMPT_PRESETS } from "@/data/character-prompt-presets"
import { AvatarField, extractAvatarValues } from "./AvatarField"
import { CharacterPreview } from "./CharacterPreview"
import { GenerateFieldButton } from "./GenerateFieldButton"
import {
  MAX_NAME_LENGTH,
  SYSTEM_PROMPT_EXAMPLE,
  getCharacterVisibleTags,
  type AdvancedSectionKey,
  type CharacterWorldBookOption,
  type CharacterFolderOption,
} from "./utils"

type CharacterFormMode = "create" | "edit"
type AdvancedSectionState = Record<AdvancedSectionKey, boolean>

export type CharacterEditorFormProps = {
  t: TFunction
  form: FormInstance
  mode: CharacterFormMode
  initialValues?: Record<string, any>
  worldBookFieldContext: {
    options: CharacterWorldBookOption[]
    loading: boolean
    editCharacterNumericId: number | null
  }
  isSubmitting: boolean
  submitButtonClassName?: string
  submitPendingLabel: string
  submitIdleLabel: string
  showPreview: boolean
  onTogglePreview: () => void
  onValuesChange: (allValues: Record<string, any>) => void
  onFinish: (values: Record<string, any>) => void
  generatingField: string | null
  isGenerating: boolean
  handleGenerateField: (
    field: CharacterField,
    form: FormInstance,
    targetForm: CharacterFormMode
  ) => Promise<void>
  showSystemPromptExample: boolean
  setShowSystemPromptExample: React.Dispatch<React.SetStateAction<boolean>>
  markModeDirty: (mode: CharacterFormMode) => void
  popularTags: Array<{ tag: string; count: number }>
  tagOptionsWithCounts: Array<{ value: string; label: React.ReactNode }>
  characterFolderOptions: CharacterFolderOption[]
  characterFolderOptionsLoading: boolean
  showAdvanced: boolean
  setShowAdvanced: React.Dispatch<React.SetStateAction<boolean>>
  advancedSections: AdvancedSectionState
  setAdvancedSections: React.Dispatch<React.SetStateAction<AdvancedSectionState>>
  createNameRef: React.RefObject<InputRef | null>
  editNameRef: React.RefObject<InputRef | null>
}

export const CharacterEditorForm: React.FC<CharacterEditorFormProps> = ({
  t,
  form,
  mode,
  initialValues,
  worldBookFieldContext,
  isSubmitting,
  submitButtonClassName,
  submitPendingLabel,
  submitIdleLabel,
  showPreview,
  onTogglePreview,
  onValuesChange,
  onFinish,
  generatingField,
  isGenerating,
  handleGenerateField,
  showSystemPromptExample,
  setShowSystemPromptExample,
  markModeDirty,
  popularTags,
  tagOptionsWithCounts,
  characterFolderOptions,
  characterFolderOptionsLoading,
  showAdvanced,
  setShowAdvanced,
  advancedSections,
  setAdvancedSections,
  createNameRef,
  editNameRef,
}) => {
  const promptPresetOptions = React.useMemo(
    () =>
      CHARACTER_PROMPT_PRESETS.map((preset) => ({
        value: preset.id,
        label: t(
          `settings:manageCharacters.promptPresets.${preset.id}.label`,
          { defaultValue: preset.label },
        ),
      })),
    [t],
  )

  const applySystemPromptExample = React.useCallback(() => {
    const currentValue = String(form.getFieldValue("system_prompt") ?? "").trim()
    const shouldConfirmOverwrite =
      currentValue.length > 0 && currentValue !== SYSTEM_PROMPT_EXAMPLE

    const apply = () => {
      form.setFieldValue("system_prompt", SYSTEM_PROMPT_EXAMPLE)
      markModeDirty(mode)
      setShowSystemPromptExample(false)
    }

    if (!shouldConfirmOverwrite) {
      apply()
      return
    }

    Modal.confirm({
      title: t("settings:manageCharacters.form.systemPrompt.exampleOverwrite.title", {
        defaultValue: "Replace current system prompt?",
      }),
      content: t("settings:manageCharacters.form.systemPrompt.exampleOverwrite.content", {
        defaultValue:
          "This will replace your current system prompt with the Writing Assistant example.",
      }),
      okText: t("settings:manageCharacters.form.systemPrompt.exampleOverwrite.confirm", {
        defaultValue: "Replace",
      }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: apply,
    })
  }, [form, markModeDirty, mode, setShowSystemPromptExample, t])

  const renderSystemPromptField = React.useCallback(
    () => (
      <>
        <Form.Item
          name="system_prompt"
          label={
            <span>
              {t("settings:manageCharacters.form.systemPrompt.label", {
                defaultValue: "Behavior / instructions",
              })}
              <span className="text-danger ml-0.5" aria-hidden="true">*</span>
              <span className="sr-only">
                {" "}
                ({t("common:required", { defaultValue: "required" })})
              </span>
              <GenerateFieldButton
                isGenerating={generatingField === "system_prompt"}
                disabled={isGenerating}
                onClick={() => handleGenerateField("system_prompt", form, mode)}
              />
            </span>
          }
          help={t("settings:manageCharacters.form.systemPrompt.help", {
            defaultValue:
              "System prompt: full behavioral instructions sent to the model, including role, tone, and constraints. (max 2000 characters)",
          })}
          extra={
            <div className="space-y-2">
              <button
                type="button"
                className="text-xs font-medium text-primary underline-offset-2 hover:underline"
                onClick={() => setShowSystemPromptExample((value) => !value)}
              >
                {showSystemPromptExample
                  ? t("settings:manageCharacters.form.systemPrompt.hideExample", {
                      defaultValue: "Hide example",
                    })
                  : t("settings:manageCharacters.form.systemPrompt.showExample", {
                      defaultValue: "Show example",
                    })}
              </button>
              {showSystemPromptExample ? (
                <div className="rounded border border-border bg-surface2 p-2">
                  <p className="mb-2 text-xs font-medium text-text">
                    {t("settings:manageCharacters.form.systemPrompt.exampleLabel", {
                      defaultValue: "Writing Assistant example",
                    })}
                  </p>
                  <p className="whitespace-pre-wrap text-xs text-text-muted">
                    {SYSTEM_PROMPT_EXAMPLE}
                  </p>
                  <Button
                    type="link"
                    size="small"
                    className="mt-2 p-0"
                    onClick={applySystemPromptExample}
                  >
                    {t("settings:manageCharacters.form.systemPrompt.useExample", {
                      defaultValue: "Use this example",
                    })}
                  </Button>
                </div>
              ) : null}
            </div>
          }
          rules={[
            {
              required: true,
              message: t("settings:manageCharacters.form.systemPrompt.required", {
                defaultValue:
                  "Please add instructions for how the character should respond.",
              }),
            },
            {
              min: 10,
              message: t("settings:manageCharacters.form.systemPrompt.min", {
                defaultValue:
                  "Add a short description so the character knows how to respond.",
              }),
            },
            {
              max: 2000,
              message: t("settings:manageCharacters.form.systemPrompt.max", {
                defaultValue: "System prompt must be 2000 characters or less.",
              }),
            },
          ]}
        >
          <Input.TextArea
            autoSize={{ minRows: 3, maxRows: 8 }}
            showCount
            maxLength={2000}
            placeholder={t("settings:manageCharacters.form.systemPrompt.placeholder", {
              defaultValue:
                "E.g., You are a patient math teacher who explains concepts step by step and checks understanding with short examples.",
            })}
          />
        </Form.Item>

        <Form.Item
          name="prompt_preset"
          label={t("settings:manageCharacters.form.promptPreset.label", {
            defaultValue: "Prompt preset",
          })}
          help={t("settings:manageCharacters.form.promptPreset.help", {
            defaultValue:
              "Controls how character fields are formatted in system prompts for character chats.",
          })}
        >
          <Select options={promptPresetOptions} />
        </Form.Item>
      </>
    ),
    [
      applySystemPromptExample,
      form,
      generatingField,
      handleGenerateField,
      isGenerating,
      mode,
      promptPresetOptions,
      setShowSystemPromptExample,
      showSystemPromptExample,
      t,
    ]
  )

  const renderAlternateGreetingsField = React.useCallback(() => {
    const markDirty = () => markModeDirty(mode)

    return (
      <Form.Item
        label={
          <span>
            {t("settings:manageCharacters.form.alternateGreetings.label", {
              defaultValue: "Alternate greetings",
            })}
            <GenerateFieldButton
              isGenerating={generatingField === "alternate_greetings"}
              disabled={isGenerating}
              onClick={() => handleGenerateField("alternate_greetings", form, mode)}
            />
          </span>
        }
        help={t("settings:manageCharacters.form.alternateGreetings.help", {
          defaultValue:
            "Optional alternate greetings to rotate between when starting chats.",
        })}
      >
        <Form.List name="alternate_greetings">
          {(fields, { add, remove, move }) => (
            <div className="space-y-2">
              {fields.length === 0 ? (
                <p className="text-xs text-text-subtle">
                  {t("settings:manageCharacters.form.alternateGreetings.empty", {
                    defaultValue:
                      "No alternate greetings yet. Add one to vary how chats start.",
                  })}
                </p>
              ) : null}
              {fields.map((field, index) => {
                const { key, ...fieldProps } = field
                return (
                  <div key={key} className="rounded-md border border-border bg-surface2 p-2">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="text-xs font-medium text-text-muted">
                        {t("settings:manageCharacters.form.alternateGreetings.itemLabel", {
                          defaultValue: "Greeting {{index}}",
                          index: index + 1,
                        })}
                      </span>
                      <div className="flex items-center gap-1">
                        <Button
                          type="text"
                          size="small"
                          icon={<ChevronUp className="h-4 w-4" />}
                          aria-label={t(
                            "settings:manageCharacters.form.alternateGreetings.moveUp",
                            { defaultValue: "Move greeting up" }
                          )}
                          disabled={index === 0}
                          onClick={() => {
                            move(index, index - 1)
                            markDirty()
                          }}
                        />
                        <Button
                          type="text"
                          size="small"
                          icon={<ChevronDown className="h-4 w-4" />}
                          aria-label={t(
                            "settings:manageCharacters.form.alternateGreetings.moveDown",
                            { defaultValue: "Move greeting down" }
                          )}
                          disabled={index === fields.length - 1}
                          onClick={() => {
                            move(index, index + 1)
                            markDirty()
                          }}
                        />
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<span className="text-xs font-medium">X</span>}
                          aria-label={t(
                            "settings:manageCharacters.form.alternateGreetings.remove",
                            { defaultValue: "Remove greeting" }
                          )}
                          onClick={() => {
                            remove(field.name)
                            markDirty()
                          }}
                        />
                      </div>
                    </div>
                    <Form.Item
                      {...fieldProps}
                      className="mb-0"
                      rules={[
                        {
                          validator: async (_rule, value) => {
                            if (!value || String(value).trim().length === 0) {
                              return Promise.resolve()
                            }
                            if (String(value).trim().length > 1000) {
                              return Promise.reject(
                                new Error(
                                  t("settings:manageCharacters.form.alternateGreetings.max", {
                                    defaultValue:
                                      "Alternate greeting must be 1000 characters or less.",
                                  })
                                )
                              )
                            }
                            return Promise.resolve()
                          },
                        },
                      ]}
                    >
                      <Input.TextArea
                        autoSize={{ minRows: 2, maxRows: 6 }}
                        showCount
                        maxLength={1000}
                        placeholder={t(
                          "settings:manageCharacters.form.alternateGreetings.itemPlaceholder",
                          {
                            defaultValue: "Enter an alternate greeting message",
                          }
                        )}
                        onChange={() => markDirty()}
                      />
                    </Form.Item>
                  </div>
                )
              })}
              <Button
                type="dashed"
                size="small"
                onClick={() => {
                  add("")
                  markDirty()
                }}
              >
                {t("settings:manageCharacters.form.alternateGreetings.add", {
                  defaultValue: "Add alternate greeting",
                })}
              </Button>
            </div>
          )}
        </Form.List>
      </Form.Item>
    )
  }, [form, generatingField, handleGenerateField, isGenerating, markModeDirty, mode, t])

  const renderNameField = React.useCallback(
    () => (
      <Form.Item
        name="name"
        label={
          <span>
            {t("settings:manageCharacters.form.name.label", {
              defaultValue: "Name",
            })}
            <span className="text-danger ml-0.5">*</span>
            <GenerateFieldButton
              isGenerating={generatingField === "name"}
              disabled={isGenerating}
              onClick={() => handleGenerateField("name", form, mode)}
            />
          </span>
        }
        rules={[
          {
            required: true,
            message: t("settings:manageCharacters.form.name.required", {
              defaultValue: "Please enter a name",
            }),
          },
          {
            max: MAX_NAME_LENGTH,
            message: t("settings:manageCharacters.form.name.maxLength", {
              defaultValue: `Name must be ${MAX_NAME_LENGTH} characters or fewer`,
            }),
          },
        ]}
      >
        <Input
          ref={mode === "create" ? createNameRef : editNameRef}
          placeholder={t("settings:manageCharacters.form.name.placeholder", {
            defaultValue: "e.g. Writing coach",
          })}
          maxLength={MAX_NAME_LENGTH}
          showCount
        />
      </Form.Item>
    ),
    [
      createNameRef,
      editNameRef,
      form,
      generatingField,
      handleGenerateField,
      isGenerating,
      mode,
      t,
    ]
  )

  const renderGreetingField = React.useCallback(
    () => (
      <Form.Item
        name="greeting"
        label={
          <span>
            {t("settings:manageCharacters.form.greeting.label", {
              defaultValue: "Greeting message (optional)",
            })}
            <GenerateFieldButton
              isGenerating={generatingField === "first_message"}
              disabled={isGenerating}
              onClick={() => handleGenerateField("first_message", form, mode)}
            />
          </span>
        }
        help={t("settings:manageCharacters.form.greeting.help", {
          defaultValue:
            "Optional first message the character will send when you start a chat.",
        })}
      >
        <Input.TextArea
          autoSize={{ minRows: 2, maxRows: 6 }}
          placeholder={t("settings:manageCharacters.form.greeting.placeholder", {
            defaultValue:
              "Hi there! I'm your writing coach. Paste your draft and I'll help you tighten it up.",
          })}
          showCount
          maxLength={1000}
        />
      </Form.Item>
    ),
    [form, generatingField, handleGenerateField, isGenerating, mode, t]
  )

  const renderDescriptionField = React.useCallback(
    () => (
      <Form.Item
        name="description"
        label={
          <span>
            {t("settings:manageCharacters.form.description.label", {
              defaultValue: "Description",
            })}
            <GenerateFieldButton
              isGenerating={generatingField === "description"}
              disabled={isGenerating}
              onClick={() => handleGenerateField("description", form, mode)}
            />
          </span>
        }
        help={t("settings:manageCharacters.form.description.help", {
          defaultValue: "Description: brief blurb shown in character lists and cards.",
        })}
      >
        <Input
          placeholder={t("settings:manageCharacters.form.description.placeholder", {
            defaultValue: "Short description",
          })}
        />
      </Form.Item>
    ),
    [form, generatingField, handleGenerateField, isGenerating, mode, t]
  )

  const renderTagsField = React.useCallback(
    () => (
      <Form.Item
        name="tags"
        label={
          <span>
            {t("settings:manageCharacters.tags.label", {
              defaultValue: "Tags",
            })}
            <GenerateFieldButton
              isGenerating={generatingField === "tags"}
              disabled={isGenerating}
              onClick={() => handleGenerateField("tags", form, mode)}
            />
          </span>
        }
        help={t("settings:manageCharacters.tags.help", {
          defaultValue:
            "Use tags to group characters by use case (e.g., 'writing', 'teaching').",
        })}
      >
        <div className="space-y-2">
          {popularTags.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              <span className="text-xs text-text-subtle mr-1">
                {t("settings:manageCharacters.tags.popular", { defaultValue: "Popular:" })}
              </span>
              {popularTags.map(({ tag, count }) => {
                const currentTags = form.getFieldValue("tags") || []
                const isSelected = currentTags.includes(tag)
                return (
                  <button
                    key={tag}
                    type="button"
                    className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full border transition-colors motion-reduce:transition-none ${
                      isSelected
                        ? "bg-primary/10 border-primary text-primary"
                        : "bg-surface border-border text-text-muted hover:border-primary/50 hover:text-primary"
                    }`}
                    onClick={() => {
                      const current = form.getFieldValue("tags") || []
                      if (isSelected) {
                        form.setFieldValue(
                          "tags",
                          current.filter((value: string) => value !== tag)
                        )
                      } else {
                        form.setFieldValue("tags", [...current, tag])
                      }
                      markModeDirty(mode)
                    }}
                  >
                    {tag}
                    <span className="text-text-subtle">({count})</span>
                  </button>
                )
              })}
            </div>
          ) : null}
          <Form.Item name="tags" noStyle>
            <Select
              mode="tags"
              allowClear
              placeholder={t("settings:manageCharacters.tags.placeholder", {
                defaultValue: "Add tags",
              })}
              options={tagOptionsWithCounts}
              onChange={(value) => {
                form.setFieldValue("tags", getCharacterVisibleTags(value))
                markModeDirty(mode)
              }}
              filterOption={(input, option) =>
                option?.value?.toString().toLowerCase().includes(input.toLowerCase()) ?? false
              }
            />
          </Form.Item>
        </div>
      </Form.Item>
    ),
    [
      form,
      generatingField,
      handleGenerateField,
      isGenerating,
      markModeDirty,
      mode,
      popularTags,
      t,
      tagOptionsWithCounts,
    ]
  )

  const renderAvatarField = React.useCallback(
    () => (
      <Form.Item
        noStyle
        shouldUpdate={(prev, cur) =>
          prev?.name !== cur?.name || prev?.description !== cur?.description
        }
      >
        {({ getFieldValue }) => (
          <Form.Item
            name="avatar"
            label={t("settings:manageCharacters.avatar.label", {
              defaultValue: "Avatar (optional)",
            })}
          >
            <AvatarField
              characterName={getFieldValue("name")}
              characterDescription={getFieldValue("description")}
            />
          </Form.Item>
        )}
      </Form.Item>
    ),
    [t]
  )

  const renderAdvancedFields = React.useCallback(() => {
    const worldBookOptions = worldBookFieldContext.options
    const worldBookOptionsLoading = worldBookFieldContext.loading
    const worldBookEditCharacterNumericId = worldBookFieldContext.editCharacterNumericId
    const toggleSection = (section: AdvancedSectionKey) => {
      setAdvancedSections((current) => ({ ...current, [section]: !current[section] }))
    }
    const renderSection = (
      section: AdvancedSectionKey,
      title: string,
      children: React.ReactNode
    ) => (
      <div className="rounded-md border border-border/70 bg-bg/40">
        <button
          type="button"
          className="flex w-full items-center justify-between px-3 py-2 text-left"
          aria-expanded={advancedSections[section]}
          onClick={() => toggleSection(section)}
        >
          <span className="text-sm font-medium text-text">{title}</span>
          {advancedSections[section] ? (
            <ChevronUp className="h-4 w-4 text-text-subtle" />
          ) : (
            <ChevronDown className="h-4 w-4 text-text-subtle" />
          )}
        </button>
        {advancedSections[section] ? (
          <div className="space-y-3 border-t border-border/60 p-3">{children}</div>
        ) : null}
      </div>
    )

    return (
      <>
        <button
          type="button"
          className="mb-2 text-xs font-medium text-primary underline-offset-2 hover:underline"
          onClick={() => setShowAdvanced((value) => !value)}
        >
          {showAdvanced
            ? t("settings:manageCharacters.advanced.hide", {
                defaultValue: "Hide advanced fields",
              })
            : t("settings:manageCharacters.advanced.show", {
                defaultValue: "Show advanced fields",
              })}
        </button>
        {showAdvanced ? (
          <div className="space-y-3 rounded-md border border-dashed border-border p-3">
            {renderSection(
              "promptControl",
              t("settings:manageCharacters.advanced.section.promptControl", {
                defaultValue: "Prompt control",
              }),
              <>
                <Form.Item
                  name="personality"
                  label={
                    <span>
                      {t("settings:manageCharacters.form.personality.label", {
                        defaultValue: "Personality",
                      })}
                      <GenerateFieldButton
                        isGenerating={generatingField === "personality"}
                        disabled={isGenerating}
                        onClick={() => handleGenerateField("personality", form, mode)}
                      />
                    </span>
                  }
                  help={t("settings:manageCharacters.form.personality.help", {
                    defaultValue:
                      "Personality: adjectives and traits injected into context to shape voice and behavior.",
                  })}
                >
                  <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
                </Form.Item>
                <Form.Item
                  name="scenario"
                  label={
                    <span>
                      {t("settings:manageCharacters.form.scenario.label", {
                        defaultValue: "Scenario",
                      })}
                      <GenerateFieldButton
                        isGenerating={generatingField === "scenario"}
                        disabled={isGenerating}
                        onClick={() => handleGenerateField("scenario", form, mode)}
                      />
                    </span>
                  }
                >
                  <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
                </Form.Item>
                <Form.Item
                  name="post_history_instructions"
                  label={t("settings:manageCharacters.form.postHistory.label", {
                    defaultValue: "Post-history instructions",
                  })}
                >
                  <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
                </Form.Item>
                <Form.Item
                  name="message_example"
                  label={
                    <span>
                      {t("settings:manageCharacters.form.messageExample.label", {
                        defaultValue: "Message example",
                      })}
                      <GenerateFieldButton
                        isGenerating={generatingField === "message_example"}
                        disabled={isGenerating}
                        onClick={() => handleGenerateField("message_example", form, mode)}
                      />
                    </span>
                  }
                >
                  <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
                </Form.Item>
                <Form.Item
                  name="creator_notes"
                  label={
                    <span>
                      {t("settings:manageCharacters.form.creatorNotes.label", {
                        defaultValue: "Creator notes",
                      })}
                      <GenerateFieldButton
                        isGenerating={generatingField === "creator_notes"}
                        disabled={isGenerating}
                        onClick={() => handleGenerateField("creator_notes", form, mode)}
                      />
                    </span>
                  }
                >
                  <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
                </Form.Item>
                {renderAlternateGreetingsField()}
                <Form.Item
                  name="default_author_note"
                  label={t("settings:manageCharacters.form.defaultAuthorNote.label", {
                    defaultValue: "Default author note",
                  })}
                  help={t("settings:manageCharacters.form.defaultAuthorNote.help", {
                    defaultValue:
                      "Optional default note used by character chats when the chat-level author note is empty.",
                  })}
                >
                  <Input.TextArea
                    autoSize={{ minRows: 2, maxRows: 6 }}
                    showCount
                    maxLength={2000}
                    placeholder={t(
                      "settings:manageCharacters.form.defaultAuthorNote.placeholder",
                      {
                        defaultValue:
                          "E.g., Keep replies concise, grounded, and in first-person voice.",
                      }
                    )}
                  />
                </Form.Item>
              </>
            )}

            {renderSection(
              "generationSettings",
              t("settings:manageCharacters.advanced.section.generationSettings", {
                defaultValue: "Generation settings",
              }),
              <>
                <Form.Item
                  name="generation_temperature"
                  label={t("settings:manageCharacters.form.generationTemperature.label", {
                    defaultValue: "Generation temperature",
                  })}
                  help={t("settings:manageCharacters.form.generationTemperature.help", {
                    defaultValue:
                      "Optional per-character sampling temperature for character chat completions.",
                  })}
                >
                  <InputNumber min={0} max={2} step={0.01} className="w-full" />
                </Form.Item>
                <Form.Item
                  name="generation_top_p"
                  label={t("settings:manageCharacters.form.generationTopP.label", {
                    defaultValue: "Generation top_p",
                  })}
                  help={t("settings:manageCharacters.form.generationTopP.help", {
                    defaultValue:
                      "Optional per-character nucleus sampling value (0.0 to 1.0).",
                  })}
                >
                  <InputNumber min={0} max={1} step={0.01} className="w-full" />
                </Form.Item>
                <Form.Item
                  name="generation_repetition_penalty"
                  label={t(
                    "settings:manageCharacters.form.generationRepetitionPenalty.label",
                    { defaultValue: "Repetition penalty" }
                  )}
                  help={t(
                    "settings:manageCharacters.form.generationRepetitionPenalty.help",
                    {
                      defaultValue:
                        "Optional per-character repetition penalty used for character chat completions.",
                    }
                  )}
                >
                  <InputNumber min={0} max={3} step={0.01} className="w-full" />
                </Form.Item>
                <Form.Item
                  name="generation_stop_strings"
                  label={t("settings:manageCharacters.form.generationStopStrings.label", {
                    defaultValue: "Stop strings",
                  })}
                  help={t("settings:manageCharacters.form.generationStopStrings.help", {
                    defaultValue:
                      "Optional stop sequences for this character. Use one per line.",
                  })}
                >
                  <Input.TextArea
                    autoSize={{ minRows: 2, maxRows: 6 }}
                    placeholder={t(
                      "settings:manageCharacters.form.generationStopStrings.placeholder",
                      {
                        defaultValue: "Example:\n###\nEND",
                      }
                    )}
                  />
                </Form.Item>
              </>
            )}

            {renderSection(
              "metadata",
              t("settings:manageCharacters.advanced.section.metadata", {
                defaultValue: "Metadata",
              }),
              <>
                <Form.Item
                  name="folder_id"
                  label={t("settings:manageCharacters.folder.label", {
                    defaultValue: "Folder",
                  })}
                  help={t("settings:manageCharacters.folder.help", {
                    defaultValue:
                      "Assign a single folder for organization. This does not change your visible tags.",
                  })}
                >
                  <Select
                    allowClear
                    showSearch
                    optionFilterProp="label"
                    placeholder={t("settings:manageCharacters.folder.placeholder", {
                      defaultValue: "Select folder",
                    })}
                    options={characterFolderOptions.map((folder) => ({
                      value: String(folder.id),
                      label: folder.name,
                    }))}
                    loading={characterFolderOptionsLoading}
                  />
                </Form.Item>
                <Form.Item
                  name="creator"
                  label={t("settings:manageCharacters.form.creator.label", {
                    defaultValue: "Creator",
                  })}
                >
                  <Input />
                </Form.Item>
                <Form.Item
                  name="character_version"
                  label={t("settings:manageCharacters.form.characterVersion.label", {
                    defaultValue: "Character version",
                  })}
                  help={t("settings:manageCharacters.form.characterVersion.help", {
                    defaultValue: 'Free text, e.g. "1.0" or "2024-01"',
                  })}
                >
                  <Input />
                </Form.Item>
                <Form.Item
                  name="extensions"
                  label={t("settings:manageCharacters.form.extensions.label", {
                    defaultValue: "Extensions (JSON)",
                  })}
                  help={t("settings:manageCharacters.form.extensions.help", {
                    defaultValue:
                      "Optional JSON object with additional metadata; invalid JSON will be sent as raw text.",
                  })}
                >
                  <Input.TextArea autoSize={{ minRows: 2, maxRows: 8 }} />
                </Form.Item>
                <Form.Item
                  name="world_book_ids"
                  label={t("settings:manageCharacters.worldBooks.editorTitle", {
                    defaultValue: "World book attachments",
                  })}
                  help={t("settings:manageCharacters.worldBooks.editorDescription", {
                    defaultValue:
                      "Attach or detach world books used for character context injection.",
                  })}
                >
                  <Select
                    mode="multiple"
                    allowClear
                    optionFilterProp="label"
                    placeholder={t("settings:manageCharacters.worldBooks.attachPlaceholder", {
                      defaultValue: "Select world book to attach",
                    })}
                    options={worldBookOptions.map((worldBook) => ({
                      value: worldBook.id,
                      label: worldBook.name,
                    }))}
                    loading={worldBookOptionsLoading}
                  />
                </Form.Item>
                {mode === "edit" && worldBookEditCharacterNumericId == null ? (
                  <div className="-mt-2 text-xs text-text-muted">
                    {t("settings:manageCharacters.worldBooks.unsyncedCharacter", {
                      defaultValue:
                        "Save this character to the server before attaching world books.",
                    })}
                  </div>
                ) : null}
                <div className="rounded-md border border-dashed border-border px-3 py-2">
                  <p className="text-xs font-medium text-text">
                    {t("settings:manageCharacters.form.moodImages.placeholderTitle", {
                      defaultValue: "Mood images (coming soon)",
                    })}
                  </p>
                  <p className="mt-1 text-xs text-text-muted">
                    {t("settings:manageCharacters.form.moodImages.placeholderBody", {
                      defaultValue:
                        "Per-mood image variants are planned but not yet available in the character editor.",
                    })}
                  </p>
                </div>
              </>
            )}
          </div>
        ) : null}
      </>
    )
  }, [
    advancedSections,
    characterFolderOptions,
    characterFolderOptionsLoading,
    form,
    generatingField,
    handleGenerateField,
    isGenerating,
    mode,
    renderAlternateGreetingsField,
    setAdvancedSections,
    setShowAdvanced,
    showAdvanced,
    t,
    worldBookFieldContext,
  ])

  return (
    <Form
      layout="vertical"
      form={form}
      initialValues={initialValues}
      className="space-y-3"
      onValuesChange={(_, allValues) => {
        onValuesChange(allValues)
      }}
      onFinish={onFinish}
    >
      {renderNameField()}
      {renderSystemPromptField()}
      {renderGreetingField()}
      {renderDescriptionField()}
      {renderTagsField()}
      {renderAvatarField()}
      {renderAdvancedFields()}

      <button
        type="button"
        className="mt-4 mb-2 flex items-center gap-1 text-xs font-medium text-text-muted hover:text-text"
        onClick={onTogglePreview}
      >
        {showPreview ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
        {showPreview
          ? t("settings:manageCharacters.preview.hide", {
              defaultValue: "Hide preview",
            })
          : t("settings:manageCharacters.preview.show", {
              defaultValue: "Show preview",
            })}
      </button>

      {showPreview ? (
        <Form.Item noStyle shouldUpdate>
          {() => {
            const avatar = form.getFieldValue("avatar")
            const avatarValues = avatar ? extractAvatarValues(avatar) : {}
            return (
              <CharacterPreview
                name={form.getFieldValue("name")}
                description={form.getFieldValue("description")}
                avatar_url={avatarValues.avatar_url}
                image_base64={avatarValues.image_base64}
                system_prompt={form.getFieldValue("system_prompt")}
                greeting={form.getFieldValue("greeting")}
                tags={form.getFieldValue("tags")}
              />
            )
          }}
        </Form.Item>
      ) : null}

      <Button
        type="primary"
        htmlType="submit"
        loading={isSubmitting}
        className={submitButtonClassName}
      >
        {isSubmitting ? submitPendingLabel : submitIdleLabel}
      </Button>
    </Form>
  )
}

export default CharacterEditorForm
