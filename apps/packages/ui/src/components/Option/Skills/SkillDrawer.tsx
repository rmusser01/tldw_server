import React from "react"
import { useMutation } from "@tanstack/react-query"
import { Button, Collapse, Drawer, Form, Input, Modal, Radio, Space } from "antd"
import type { RadioChangeEvent } from "antd"
import { useTranslation } from "react-i18next"
import { Plus, Trash2 } from "lucide-react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import type { SkillCreate, SkillResponse, SkillUpdate } from "@/types/skill"
import {
  buildInitialSkillContent,
  buildSkillTemplateContent,
  buildSupportingFilesForCreate,
  buildSupportingFilesForUpdate,
  SKILL_TEMPLATE_OPTIONS,
  type SkillTemplateId,
  type SupportingFileFormEntry
} from "./skill-form-utils"

const SKILL_NAME_REGEX = /^[a-z][a-z0-9-]{0,63}$/
const SUPPORTING_FILE_NAME_REGEX = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/
const DEFAULT_TEMPLATE_ID: SkillTemplateId = "summarizer"

interface SkillDrawerFormValues {
  name: string
  content: string
  supportingFiles?: SupportingFileFormEntry[]
}

interface SkillDrawerProps {
  open: boolean
  skill: SkillResponse | null
  onClose: () => void
  onSaved: () => void
}

export const SkillDrawer: React.FC<SkillDrawerProps> = ({
  open,
  skill,
  onClose,
  onSaved
}) => {
  const { t } = useTranslation(["option", "common"])
  const notification = useAntdNotification()
  const [form] = Form.useForm<SkillDrawerFormValues>()
  const isEdit = Boolean(skill)
  const [modal, contextHolder] = Modal.useModal()
  const [selectedTemplateId, setSelectedTemplateId] =
    React.useState<SkillTemplateId>(DEFAULT_TEMPLATE_ID)
  const lastGeneratedContentRef = React.useRef("")

  const getTemplateNameForContent = React.useCallback(() => {
    const name = form.getFieldValue("name") ?? ""
    return !name || SKILL_NAME_REGEX.test(name) ? name : ""
  }, [form])

  const generateTemplateContent = React.useCallback(
    (templateId: SkillTemplateId) =>
      buildSkillTemplateContent(templateId, getTemplateNameForContent()),
    [getTemplateNameForContent]
  )

  const applyTemplate = React.useCallback(
    (templateId: SkillTemplateId) => {
      const content = generateTemplateContent(templateId)
      lastGeneratedContentRef.current = content
      setSelectedTemplateId(templateId)
      form.setFieldsValue({ content })
    },
    [form, generateTemplateContent]
  )

  React.useEffect(() => {
    if (open) {
      if (skill) {
        lastGeneratedContentRef.current = ""
        const supportingFiles = Object.entries(skill.supporting_files ?? {}).map(
          ([filename, content]) => ({
            filename,
            content,
            originalFilename: filename
          })
        )
        form.setFieldsValue({
          name: skill.name,
          content: buildInitialSkillContent(skill),
          supportingFiles
        })
      } else {
        const content = buildSkillTemplateContent(DEFAULT_TEMPLATE_ID, "")
        lastGeneratedContentRef.current = content
        setSelectedTemplateId(DEFAULT_TEMPLATE_ID)
        form.resetFields()
        form.setFieldsValue({
          content,
          supportingFiles: []
        })
      }
    }
  }, [open, skill, form])

  const handleTemplateChange = (event: RadioChangeEvent) => {
    const nextTemplateId = event.target.value as SkillTemplateId
    if (nextTemplateId === selectedTemplateId) return

    const currentContent = form.getFieldValue("content") ?? ""
    const canReplaceWithoutPrompt =
      !currentContent || currentContent === lastGeneratedContentRef.current

    if (canReplaceWithoutPrompt) {
      applyTemplate(nextTemplateId)
      return
    }

    modal.confirm({
      title: t("option:skills.replaceTemplateTitle", {
        defaultValue: "Replace draft with template?"
      }),
      content: t("option:skills.replaceTemplateDescription", {
        defaultValue:
          "This will replace your current SKILL.md draft with the selected template."
      }),
      okText: t("option:skills.replaceTemplateConfirm", {
        defaultValue: "Replace draft"
      }),
      cancelText: t("option:skills.replaceTemplateCancel", {
        defaultValue: "Keep current draft"
      }),
      onOk: () => applyTemplate(nextTemplateId)
    })
  }

  const handleFormValuesChange = (changedValues: Partial<SkillDrawerFormValues>) => {
    if (isEdit || !Object.prototype.hasOwnProperty.call(changedValues, "name")) {
      return
    }

    const nextName = changedValues.name ?? ""
    if (nextName && !SKILL_NAME_REGEX.test(nextName)) {
      return
    }

    const currentContent = form.getFieldValue("content") ?? ""
    if (currentContent !== lastGeneratedContentRef.current) {
      return
    }

    const content = generateTemplateContent(selectedTemplateId)
    lastGeneratedContentRef.current = content
    form.setFieldsValue({ content })
  }

  const createMutation = useMutation({
    mutationFn: (values: SkillCreate) =>
      tldwClient.createSkill(values),
    onSuccess: () => {
      notification.success({
        message: t("option:skills.createSuccess", { defaultValue: "Skill created" })
      })
      onSaved()
    },
    onError: (err: any) => {
      const desc =
        err?.message?.includes("409") || err?.status === 409
          ? t("option:skills.duplicateError", {
              defaultValue: "A skill with this name already exists."
            })
          : err?.message
      notification.error({
        message: t("option:skills.createError", { defaultValue: "Failed to create skill" }),
        description: desc
      })
    }
  })

  const updateMutation = useMutation({
    mutationFn: (values: SkillUpdate) =>
      tldwClient.updateSkill(skill!.name, values, skill!.version),
    onSuccess: () => {
      notification.success({
        message: t("option:skills.updateSuccess", { defaultValue: "Skill updated" })
      })
      onSaved()
    },
    onError: (err: any) => {
      const desc =
        err?.message?.includes("409") || err?.status === 409
          ? t("option:skills.versionConflict", {
              defaultValue:
                "Version conflict: the skill was modified elsewhere. Please reload and try again."
            })
          : err?.message
      notification.error({
        message: t("option:skills.updateError", { defaultValue: "Failed to update skill" }),
        description: desc
      })
    }
  })

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      if (isEdit) {
        let supportingFilesUpdate: Record<string, string | null> | undefined
        try {
          supportingFilesUpdate = buildSupportingFilesForUpdate(
            skill!.supporting_files,
            values.supportingFiles
          )
        } catch (err: any) {
          notification.error({
            message: t("option:skills.supportingFilesInvalid", {
              defaultValue: "Invalid supporting files"
            }),
            description: err?.message
          })
          return
        }

        const payload: SkillUpdate = { content: values.content }
        if (supportingFilesUpdate) {
          payload.supporting_files = supportingFilesUpdate
        }
        updateMutation.mutate(payload)
      } else {
        let supportingFiles: Record<string, string> | undefined
        try {
          supportingFiles = buildSupportingFilesForCreate(values.supportingFiles)
        } catch (err: any) {
          notification.error({
            message: t("option:skills.supportingFilesInvalid", {
              defaultValue: "Invalid supporting files"
            }),
            description: err?.message
          })
          return
        }

        const payload: SkillCreate = {
          name: values.name,
          content: values.content
        }
        if (supportingFiles) {
          payload.supporting_files = supportingFiles
        }
        createMutation.mutate(payload)
      }
    } catch {
      // validation errors handled by antd
    }
  }

  const isSaving = createMutation.isPending || updateMutation.isPending

  return (
    <Drawer
      title={
        isEdit
          ? t("option:skills.editTitle", { defaultValue: "Edit Skill" })
          : t("option:skills.newTitle", { defaultValue: "New Skill" })
      }
      open={open}
      onClose={onClose}
      size={640}
      destroyOnClose
      extra={
        <Space>
          <Button onClick={onClose}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button type="primary" onClick={handleSubmit} loading={isSaving}>
            {t("common:save", { defaultValue: "Save" })}
          </Button>
        </Space>
      }
    >
      {contextHolder}
      <Form
        form={form}
        layout="vertical"
        autoComplete="off"
        onValuesChange={handleFormValuesChange}
      >
        <Form.Item
          name="name"
          label={t("option:skills.nameLabel", { defaultValue: "Name" })}
          rules={[
            {
              required: true,
              message: t("option:skills.nameRequired", {
                defaultValue: "Skill name is required"
              })
            },
            {
              pattern: SKILL_NAME_REGEX,
              message: t("option:skills.nameInvalid", {
                defaultValue:
                  "Must start with a letter, use only lowercase letters, numbers, and hyphens (max 64 chars)"
              })
            }
          ]}
        >
          <Input
            placeholder="my-skill-name"
            disabled={isEdit}
            maxLength={64}
            className="font-mono"
          />
        </Form.Item>

        {!isEdit && (
          <Form.Item
            label={t("option:skills.templateLabel", {
              defaultValue: "Start from template"
            })}
            extra={t("option:skills.templateHelp", {
              defaultValue:
                "Templates generate valid SKILL.md frontmatter and a starter instruction body. You can still edit everything below."
            })}
          >
            <Radio.Group
              optionType="button"
              buttonStyle="solid"
              value={selectedTemplateId}
              onChange={handleTemplateChange}
            >
              {SKILL_TEMPLATE_OPTIONS.map((template) => (
                <Radio.Button key={template.id} value={template.id}>
                  {t(`option:skills.templates.${template.id}.label`, {
                    defaultValue: template.label
                  })}
                </Radio.Button>
              ))}
            </Radio.Group>
          </Form.Item>
        )}

        <Form.Item
          name="content"
          label={t("option:skills.contentLabel", {
            defaultValue: "SKILL.md Content"
          })}
          rules={[
            {
              required: true,
              message: t("option:skills.contentRequired", {
                defaultValue: "Content is required"
              })
            }
          ]}
          extra={t("option:skills.contentHelp", {
            defaultValue:
              "Advanced: edit the generated YAML frontmatter (---) and instructions directly. Use $ARGUMENTS for argument substitution."
          })}
        >
          <Input.TextArea
            rows={20}
            className="font-mono text-xs"
            placeholder={`---
description: What this skill does
argument-hint: "[text]"
context: inline
---

Process the following: $ARGUMENTS`}
          />
        </Form.Item>

        <Collapse
          className="mb-4"
          items={[
            {
              key: "supporting-files",
              label: t("option:skills.supportingFilesLabel", {
                defaultValue: "Supporting Files"
              }),
              children: (
                <Form.List name="supportingFiles">
                  {(fields, { add, remove }) => (
                    <div className="flex flex-col gap-3">
                      {fields.map((field) => (
                        <div key={field.key} className="rounded border p-3">
                          <Form.Item
                            name={[field.name, "originalFilename"]}
                            hidden
                          >
                            <Input />
                          </Form.Item>
                          <div className="mb-2 flex items-start gap-2">
                            <Form.Item
                              className="mb-0 flex-1"
                              name={[field.name, "filename"]}
                              label={t("option:skills.supportingFileName", {
                                defaultValue: "Filename"
                              })}
                              rules={[
                                {
                                  required: true,
                                  whitespace: true,
                                  message: t("option:skills.supportingFileNameRequired", {
                                    defaultValue: "Filename is required"
                                  })
                                },
                                {
                                  validator: async (_, value: string | undefined) => {
                                    const trimmed = (value ?? "").trim()
                                    if (!trimmed) return
                                    if (!SUPPORTING_FILE_NAME_REGEX.test(trimmed)) {
                                      throw new Error(
                                        t("option:skills.supportingFileNameInvalid", {
                                          defaultValue:
                                            "Use letters, numbers, dot, underscore, or hyphen (max 100 chars)"
                                        })
                                      )
                                    }
                                    if (trimmed.toLowerCase() === "skill.md") {
                                      throw new Error(
                                        t("option:skills.supportingFileNameReserved", {
                                          defaultValue: "SKILL.md is reserved"
                                        })
                                      )
                                    }
                                  }
                                }
                              ]}
                            >
                              <Input
                                className="font-mono text-xs"
                                placeholder="reference.md"
                              />
                            </Form.Item>
                            <Button
                              danger
                              type="text"
                              icon={<Trash2 size={14} />}
                              aria-label={t("option:skills.removeSupportingFile", {
                                defaultValue: "Remove supporting file"
                              })}
                              onClick={() => remove(field.name)}
                            />
                          </div>
                          <Form.Item
                            name={[field.name, "content"]}
                            label={t("option:skills.supportingFileContent", {
                              defaultValue: "Content"
                            })}
                            className="mb-0"
                          >
                            <Input.TextArea rows={5} className="font-mono text-xs" />
                          </Form.Item>
                        </div>
                      ))}
                      <Button
                        type="dashed"
                        icon={<Plus size={14} />}
                        onClick={() => add({ filename: "", content: "" })}
                      >
                        {t("option:skills.addSupportingFile", {
                          defaultValue: "Add Supporting File"
                        })}
                      </Button>
                    </div>
                  )}
                </Form.List>
              )
            }
          ]}
        />
      </Form>
    </Drawer>
  )
}
