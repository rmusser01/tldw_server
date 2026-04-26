import React from "react"
import {
  Button,
  Card,
  Checkbox,
  Divider,
  Form,
  Input,
  Select,
  Space,
  Typography
} from "antd"
import { Plus, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import type {
  FlashcardTemplate,
  FlashcardTemplateCreate,
  FlashcardTemplateFieldTarget,
  FlashcardTemplatePlaceholderDefinition
} from "@/services/flashcards"

const { Text } = Typography

type FlashcardTemplateFormMode = "create" | "edit"

interface FlashcardTemplateFormProps {
  mode: FlashcardTemplateFormMode
  template?: FlashcardTemplate | null
  initialValues?: Partial<FlashcardTemplateCreate> | null
  submitting?: boolean
  onSubmit: (values: FlashcardTemplateCreate) => Promise<void> | void
  onCancel?: () => void
  cancelDisabled?: boolean
  onDelete?: () => void
  deleteDisabled?: boolean
  onDirtyChange?: (dirty: boolean) => void
}

type FlashcardTemplateFormValues = FlashcardTemplateCreate

const TARGET_OPTIONS: Array<{ label: string; value: FlashcardTemplateFieldTarget }> = [
  { label: "Front template", value: "front_template" },
  { label: "Back template", value: "back_template" },
  { label: "Notes template", value: "notes_template" },
  { label: "Extra template", value: "extra_template" }
]

const coerceTemplateFormValues = (
  values?: Partial<FlashcardTemplateCreate> | FlashcardTemplate | null
): FlashcardTemplateFormValues => ({
  name: values?.name ?? "",
  model_type: values?.model_type ?? "basic",
  front_template: values?.front_template ?? "",
  back_template: values?.back_template ?? "",
  notes_template: values?.notes_template ?? "",
  extra_template: values?.extra_template ?? "",
  placeholder_definitions: (values?.placeholder_definitions ?? []).map((definition) => ({
    ...definition,
    help_text: definition.help_text ?? "",
    default_value: definition.default_value ?? "",
    required: definition.required ?? false
  }))
})

const buildDefaultValues = (): FlashcardTemplateFormValues => coerceTemplateFormValues(null)

const buildInitialValues = (
  template?: FlashcardTemplate | null,
  initialValues?: Partial<FlashcardTemplateCreate> | null
): FlashcardTemplateFormValues => coerceTemplateFormValues(template ?? initialValues)

const normalizeOptionalText = (value: string | null | undefined): string | null => {
  const trimmed = value?.trim() ?? ""
  return trimmed.length > 0 ? trimmed : null
}

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

const normalizePlaceholderDefinitions = (
  definitions?: FlashcardTemplatePlaceholderDefinition[]
): FlashcardTemplatePlaceholderDefinition[] => {
  return (definitions ?? []).map((definition) => ({
    key: definition.key.trim(),
    label: definition.label.trim(),
    help_text: normalizeOptionalText(definition.help_text),
    default_value: normalizeOptionalText(definition.default_value),
    required: definition.required ?? false,
    targets: definition.targets
  }))
}

const collectPlaceholderFieldErrors = (
  values: FlashcardTemplateFormValues,
  t: ReturnType<typeof useTranslation>["t"]
) => {
  const placeholderDefinitions = normalizePlaceholderDefinitions(values.placeholder_definitions)
  const keyCounts = new Map<string, number>()

  placeholderDefinitions.forEach((definition) => {
    if (!definition.key) {
      return
    }
    keyCounts.set(definition.key, (keyCounts.get(definition.key) ?? 0) + 1)
  })

  const templateFieldContent: Record<FlashcardTemplateFieldTarget, string> = {
    front_template: values.front_template ?? "",
    back_template: values.back_template ?? "",
    notes_template: values.notes_template ?? "",
    extra_template: values.extra_template ?? ""
  }

  return placeholderDefinitions.flatMap((definition, index) => {
    const errors: Array<{ name: (string | number)[]; errors: string[] }> = []

    if (definition.key && (keyCounts.get(definition.key) ?? 0) > 1) {
      errors.push({
        name: ["placeholder_definitions", index, "key"],
        errors: [
          t("option:flashcards.templatesPlaceholderKeyUnique", {
            defaultValue: "Placeholder keys must be unique."
          })
        ]
      })
    }

    if (definition.key && definition.targets.length > 0) {
      const placeholderToken = `{{${definition.key}}}`
      const placeholderPattern = new RegExp(`\\{\\{\\s*${escapeRegExp(definition.key)}\\s*\\}\\}`)
      const appearsInTarget = definition.targets.some((target) =>
        placeholderPattern.test(templateFieldContent[target])
      )

      if (!appearsInTarget) {
        errors.push({
          name: ["placeholder_definitions", index, "targets"],
          errors: [
            t("option:flashcards.templatesPlaceholderReferenceRequired", {
              defaultValue: "Add {{token}} to at least one targeted template field.",
              token: placeholderToken
            })
          ]
        })
      }
    }

    return errors
  })
}

export const FlashcardTemplateForm: React.FC<FlashcardTemplateFormProps> = ({
  mode,
  template,
  initialValues,
  submitting = false,
  onSubmit,
  onCancel,
  cancelDisabled = false,
  onDelete,
  deleteDisabled = false,
  onDirtyChange
}) => {
  const { t } = useTranslation(["option", "common"])
  const [form] = Form.useForm<FlashcardTemplateFormValues>()
  const selectedModelType = Form.useWatch("model_type", form) ?? "basic"
  const initialFormValues = React.useMemo(
    () => buildInitialValues(template, initialValues),
    [initialValues, template]
  )

  React.useEffect(() => {
    form.resetFields()
    form.setFieldsValue(initialFormValues)
    onDirtyChange?.(false)
  }, [form, initialFormValues, mode, onDirtyChange])

  const handleValuesChange = React.useCallback(
    (_changedValues: Partial<FlashcardTemplateFormValues>, allValues: FlashcardTemplateFormValues) => {
      if (!onDirtyChange) {
        return
      }
      const dirty =
        JSON.stringify(coerceTemplateFormValues(allValues)) !== JSON.stringify(initialFormValues)
      onDirtyChange(dirty)
    },
    [initialFormValues, onDirtyChange]
  )

  const handleFinish = React.useCallback(
    async (values: FlashcardTemplateFormValues) => {
      const placeholderFieldErrors = collectPlaceholderFieldErrors(values, t)
      if (placeholderFieldErrors.length > 0) {
        form.setFields(placeholderFieldErrors as Parameters<typeof form.setFields>[0])
        return
      }

      await onSubmit({
        name: values.name.trim(),
        model_type: values.model_type,
        front_template: values.front_template.trim(),
        back_template: normalizeOptionalText(values.back_template),
        notes_template: normalizeOptionalText(values.notes_template),
        extra_template: normalizeOptionalText(values.extra_template),
        placeholder_definitions: normalizePlaceholderDefinitions(values.placeholder_definitions)
      })
      onDirtyChange?.(false)
    },
    [form, onDirtyChange, onSubmit, t]
  )

  return (
    <Card
      title={
        mode === "create"
          ? t("option:flashcards.templatesCreateTitle", {
              defaultValue: "Create template"
            })
          : t("option:flashcards.templatesEditTitle", {
              defaultValue: "Edit template"
            })
      }
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={buildDefaultValues()}
        onFinish={handleFinish}
        onValuesChange={handleValuesChange}
      >
        <Form.Item
          label={t("option:flashcards.templatesName", { defaultValue: "Template name" })}
          name="name"
          rules={[
            {
              validator: async (_rule, value: string | undefined) => {
                if ((value ?? "").trim().length === 0) {
                  throw new Error(
                    t("option:flashcards.templatesNameRequired", {
                      defaultValue: "Enter a template name."
                    })
                  )
                }
              }
            }
          ]}
        >
          <Input
            placeholder={t("option:flashcards.templatesNamePlaceholder", {
              defaultValue: "Medical terminology"
            })}
          />
        </Form.Item>

        <Form.Item
          label={t("option:flashcards.templatesModelType", { defaultValue: "Card type" })}
          name="model_type"
        >
          <Select
            options={[
              {
                label: t("option:flashcards.modelTypeBasic", {
                  defaultValue: "Standard"
                }),
                value: "basic"
              },
              {
                label: t("option:flashcards.modelTypeBasicReverse", {
                  defaultValue: "Reversible"
                }),
                value: "basic_reverse"
              },
              {
                label: t("option:flashcards.modelTypeCloze", {
                  defaultValue: "Fill-in-blank"
                }),
                value: "cloze"
              }
            ]}
          />
        </Form.Item>

        <Form.Item
          label={t("option:flashcards.templatesFrontTemplate", { defaultValue: "Front template" })}
          name="front_template"
          rules={[
            {
              validator: async (_rule, value: string | undefined) => {
                if ((value ?? "").trim().length === 0) {
                  throw new Error(
                    t("option:flashcards.templatesFrontTemplateRequired", {
                      defaultValue: "Enter a front template."
                    })
                  )
                }
              }
            }
          ]}
        >
          <Input.TextArea
            rows={5}
            placeholder={t("option:flashcards.templatesFrontTemplatePlaceholder", {
              defaultValue: "Question: {{prompt}}"
            })}
          />
        </Form.Item>

        <Form.Item
          dependencies={["model_type"]}
          label={t("option:flashcards.templatesBackTemplate", { defaultValue: "Back template" })}
          name="back_template"
          rules={[
            ({ getFieldValue }) => ({
              validator: async (_rule, value: string | undefined) => {
                if (getFieldValue("model_type") === "cloze") {
                  return
                }
                if ((value ?? "").trim().length === 0) {
                  throw new Error(
                    t("option:flashcards.templatesBackTemplateRequired", {
                      defaultValue: "Enter a back template."
                    })
                  )
                }
              }
            })
          ]}
          extra={
            selectedModelType === "cloze"
              ? t("option:flashcards.templatesBackTemplateOptional", {
                  defaultValue: "Optional for cloze templates."
                })
              : undefined
          }
        >
          <Input.TextArea
            rows={5}
            placeholder={t("option:flashcards.templatesBackTemplatePlaceholder", {
              defaultValue: "Answer: {{answer}}"
            })}
          />
        </Form.Item>

        <div className="grid gap-4 lg:grid-cols-2">
          <Form.Item
            label={t("option:flashcards.templatesNotesTemplate", { defaultValue: "Notes template" })}
            name="notes_template"
          >
            <Input.TextArea
              rows={4}
              placeholder={t("option:flashcards.templatesNotesTemplatePlaceholder", {
                defaultValue: "Optional notes or hints."
              })}
            />
          </Form.Item>

          <Form.Item
            label={t("option:flashcards.templatesExtraTemplate", { defaultValue: "Extra template" })}
            name="extra_template"
          >
            <Input.TextArea
              rows={4}
              placeholder={t("option:flashcards.templatesExtraTemplatePlaceholder", {
                defaultValue: "Optional supporting content."
              })}
            />
          </Form.Item>
        </div>

        <Divider className="!mt-0" />

        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <Text strong>
              {t("option:flashcards.templatesPlaceholdersTitle", {
                defaultValue: "Placeholders"
              })}
            </Text>
            <Text type="secondary" className="block">
              {t("option:flashcards.templatesPlaceholdersHelp", {
                defaultValue: "Define reusable fields and where they can be inserted."
              })}
            </Text>
          </div>
        </div>

        <Form.List name="placeholder_definitions">
          {(fields, { add, remove }) => (
            <div className="space-y-4">
              {fields.map((field, index) => (
                <div
                  key={field.key}
                  className="rounded border border-border bg-surface p-4"
                >
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <Text strong>
                      {t("option:flashcards.templatesPlaceholderLabel", {
                        defaultValue: "Placeholder {{index}}",
                        index: index + 1
                      })}
                    </Text>
                    <Button
                      danger
                      type="text"
                      icon={<Trash2 className="size-4" />}
                      onClick={() => remove(field.name)}
                    >
                      {t("common:remove", { defaultValue: "Remove" })}
                    </Button>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-2">
                    <Form.Item
                      label={t("option:flashcards.templatesPlaceholderKey", {
                        defaultValue: "Key"
                      })}
                      name={[field.name, "key"]}
                      rules={[
                        {
                          validator: async (_rule, value: string | undefined) => {
                            if ((value ?? "").trim().length === 0) {
                              throw new Error(
                                t("option:flashcards.templatesPlaceholderKeyRequired", {
                                  defaultValue: "Enter a placeholder key."
                                })
                              )
                            }
                          }
                        }
                      ]}
                    >
                      <Input placeholder="prompt" />
                    </Form.Item>

                    <Form.Item
                      label={t("option:flashcards.templatesPlaceholderDisplayName", {
                        defaultValue: "Label"
                      })}
                      name={[field.name, "label"]}
                      rules={[
                        {
                          validator: async (_rule, value: string | undefined) => {
                            if ((value ?? "").trim().length === 0) {
                              throw new Error(
                                t("option:flashcards.templatesPlaceholderLabelRequired", {
                                  defaultValue: "Enter a placeholder label."
                                })
                              )
                            }
                          }
                        }
                      ]}
                    >
                      <Input
                        placeholder={t("option:flashcards.templatesPlaceholderDisplayNamePlaceholder", {
                          defaultValue: "Prompt"
                        })}
                      />
                    </Form.Item>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-2">
                    <Form.Item
                      label={t("option:flashcards.templatesPlaceholderTargets", {
                        defaultValue: "Targets"
                      })}
                      name={[field.name, "targets"]}
                      rules={[
                        {
                          validator: async (_rule, value: FlashcardTemplateFieldTarget[] | undefined) => {
                            if (!Array.isArray(value) || value.length === 0) {
                              throw new Error(
                                t("option:flashcards.templatesPlaceholderTargetsRequired", {
                                  defaultValue: "Select at least one target field."
                                })
                              )
                            }
                          }
                        }
                      ]}
                    >
                      <Select
                        mode="multiple"
                        options={TARGET_OPTIONS}
                        placeholder={t("option:flashcards.templatesPlaceholderTargetsPlaceholder", {
                          defaultValue: "Choose where this placeholder can be used"
                        })}
                      />
                    </Form.Item>

                    <Form.Item
                      label={t("option:flashcards.templatesPlaceholderDefaultValue", {
                        defaultValue: "Default value"
                      })}
                      name={[field.name, "default_value"]}
                    >
                      <Input
                        placeholder={t("option:flashcards.templatesPlaceholderDefaultValuePlaceholder", {
                          defaultValue: "Optional default value"
                        })}
                      />
                    </Form.Item>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-[1fr,200px]">
                    <Form.Item
                      label={t("option:flashcards.templatesPlaceholderHelpText", {
                        defaultValue: "Help text"
                      })}
                      name={[field.name, "help_text"]}
                    >
                      <Input
                        placeholder={t("option:flashcards.templatesPlaceholderHelpTextPlaceholder", {
                          defaultValue: "Optional guidance for the person creating cards"
                        })}
                      />
                    </Form.Item>

                    <Form.Item
                      label=" "
                      name={[field.name, "required"]}
                      valuePropName="checked"
                    >
                      <Checkbox>
                        {t("option:flashcards.templatesPlaceholderRequired", {
                          defaultValue: "Required placeholder"
                        })}
                      </Checkbox>
                    </Form.Item>
                  </div>
                </div>
              ))}

              <Button
                type="dashed"
                onClick={() =>
                  add({
                    key: "",
                    label: "",
                    help_text: "",
                    default_value: "",
                    required: false,
                    targets: ["front_template"]
                  })
                }
                icon={<Plus className="size-4" />}
              >
                {t("option:flashcards.templatesAddPlaceholder", {
                  defaultValue: "Add placeholder"
                })}
              </Button>
            </div>
          )}
        </Form.List>

        <Divider />

        <Space wrap>
          <Button type="primary" htmlType="submit" loading={submitting}>
            {mode === "create"
              ? t("option:flashcards.templatesSaveTemplate", {
                  defaultValue: "Save template"
                })
              : t("option:flashcards.templatesSaveChanges", {
                  defaultValue: "Save changes"
                })}
          </Button>
          {onCancel ? (
            <Button onClick={onCancel} disabled={cancelDisabled}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
          ) : null}
          {mode === "edit" && onDelete ? (
            <Button
              danger
              onClick={onDelete}
              disabled={deleteDisabled}
            >
              {t("option:flashcards.templatesDelete", {
                defaultValue: "Delete template"
              })}
            </Button>
          ) : null}
        </Space>
      </Form>
    </Card>
  )
}

export default FlashcardTemplateForm
