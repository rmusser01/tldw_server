import React from "react"
import { Alert, Empty, Form, Input, Modal, Select, Spin, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import type { FlashcardCreate } from "@/services/flashcards"
import { useFlashcardTemplatesQuery } from "../hooks"
import {
  getFlashcardTemplatePlaceholderDefaults,
  materializeFlashcardTemplateDraft
} from "../utils/flashcard-template-resolution"

const { Text } = Typography

interface FlashcardTemplateValueModalProps {
  open: boolean
  onClose: () => void
  onApply: (
    draft: Pick<FlashcardCreate, "deck_id" | "tags" | "model_type" | "front" | "back" | "notes" | "extra">
  ) => void
  draftDefaults?: Pick<FlashcardCreate, "deck_id" | "tags">
}

type TemplateValueModalFormValues = {
  template_id?: number
  placeholder_values?: Record<string, string>
}

export const FlashcardTemplateValueModal: React.FC<FlashcardTemplateValueModalProps> = ({
  open,
  onClose,
  onApply,
  draftDefaults
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const [form] = Form.useForm<TemplateValueModalFormValues>()
  const initializedRef = React.useRef(false)
  const templatesQuery = useFlashcardTemplatesQuery({
    enabled: open
  })
  const templates = templatesQuery.data?.items ?? []
  const selectedTemplateId = Form.useWatch("template_id", form)
  const placeholderValues = Form.useWatch("placeholder_values", form)
  const selectedTemplate = React.useMemo(
    () => templates.find((template) => template.id === selectedTemplateId) ?? null,
    [selectedTemplateId, templates]
  )
  const isSubmitDisabled = React.useMemo(() => {
    if (templatesQuery.isLoading || templatesQuery.error || templates.length === 0 || !selectedTemplate) {
      return true
    }

    return selectedTemplate.placeholder_definitions.some((definition) => {
      if (!definition.required) {
        return false
      }
      if ((definition.default_value ?? "").trim().length > 0) {
        return false
      }
      return String(placeholderValues?.[definition.key] ?? "").trim().length === 0
    })
  }, [
    placeholderValues,
    selectedTemplate,
    templates.length,
    templatesQuery.error,
    templatesQuery.isLoading
  ])

  React.useEffect(() => {
    if (!open) {
      initializedRef.current = false
      form.resetFields()
      return
    }

    if (initializedRef.current) {
      return
    }

    const initialTemplate = templates[0]
    if (!initialTemplate) {
      return
    }
    form.setFieldsValue({
      template_id: initialTemplate.id,
      placeholder_values: getFlashcardTemplatePlaceholderDefaults(initialTemplate)
    })
    initializedRef.current = true
  }, [form, open, templates])

  React.useEffect(() => {
    if (!open || !selectedTemplate) {
      return
    }
    form.setFieldValue("placeholder_values", getFlashcardTemplatePlaceholderDefaults(selectedTemplate))
  }, [form, open, selectedTemplate?.id])

  const handleFinish = React.useCallback(
    async (values: TemplateValueModalFormValues) => {
      if (!selectedTemplate) {
        return
      }

      try {
        const draft = materializeFlashcardTemplateDraft(
          selectedTemplate,
          values.placeholder_values ?? {},
          draftDefaults
        )
        onApply(draft)
      } catch (error: unknown) {
        message.error(error instanceof Error ? error.message : "Failed to apply template")
      }
    },
    [draftDefaults, message, onApply, selectedTemplate]
  )

  return (
    <Modal
      open={open}
      onCancel={onClose}
      destroyOnHidden
      title={t("option:flashcards.applyTemplate", {
        defaultValue: "Apply template"
      })}
      okText={t("option:flashcards.applyTemplateAction", {
        defaultValue: "Apply"
      })}
      cancelText={t("common:cancel", {
        defaultValue: "Cancel"
      })}
      okButtonProps={{ disabled: isSubmitDisabled }}
      onOk={() => {
        if (!isSubmitDisabled) {
          form.submit()
        }
      }}
    >
      {templatesQuery.isLoading ? (
        <div className="flex min-h-[160px] items-center justify-center">
          <Spin />
        </div>
      ) : templatesQuery.error ? (
        <Alert
          type="error"
          message={t("option:flashcards.templatesLoadError", {
            defaultValue: "Could not load templates."
          })}
          description={
            templatesQuery.error instanceof Error ? templatesQuery.error.message : undefined
          }
        />
      ) : templates.length === 0 ? (
        <Empty
          description={t("option:flashcards.templatesEmpty", {
            defaultValue: "No templates yet"
          })}
        />
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleFinish}
        >
          <Form.Item
            label={t("option:flashcards.template", {
              defaultValue: "Template"
            })}
            name="template_id"
            rules={[
              {
                required: true,
                message: t("option:flashcards.templateRequired", {
                  defaultValue: "Select a template."
                })
              }
            ]}
          >
            <Select
              options={templates.map((template) => ({
                label: template.name,
                value: template.id
              }))}
            />
          </Form.Item>

          {selectedTemplate?.placeholder_definitions.length ? (
            <div className="space-y-4">
              {selectedTemplate.placeholder_definitions.map((definition) => (
                <Form.Item
                  key={definition.key}
                  label={definition.label}
                  name={["placeholder_values", definition.key]}
                  extra={definition.help_text ? (
                    <Text type="secondary">{definition.help_text}</Text>
                  ) : undefined}
                  rules={[
                    {
                      required: definition.required && !(definition.default_value ?? "").trim(),
                      message: t("option:flashcards.placeholderRequired", {
                        defaultValue: "{{label}} is required.",
                        label: definition.label
                      })
                    }
                  ]}
                >
                  <Input />
                </Form.Item>
              ))}
            </div>
          ) : (
            <Text type="secondary">
              {t("option:flashcards.templateHasNoPlaceholders", {
                defaultValue: "This template does not require any placeholder values."
              })}
            </Text>
          )}
        </Form>
      )}
    </Modal>
  )
}

export default FlashcardTemplateValueModal
