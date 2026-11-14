import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Modal, Form, Input, notification, Skeleton } from "antd"
import React, { useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  createProject,
  updateProject,
  getProject,
  type ProjectCreatePayload
} from "@/services/prompt-studio"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { Button } from "@/components/Common/Button"

type ProjectFormModalProps = {
  open: boolean
  projectId: number | null
  onClose: () => void
}

type FormValues = {
  name: string
  description?: string
}

export const ProjectFormModal: React.FC<ProjectFormModalProps> = ({
  open,
  projectId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const queryClient = useQueryClient()

  const setSelectedProjectId = usePromptStudioStore((s) => s.setSelectedProjectId)

  const isEditing = projectId !== null

  // Fetch existing project for editing
  const { data: projectResponse, isLoading: isLoadingProject } = useQuery({
    queryKey: ["prompt-studio", "project", projectId],
    queryFn: () => getProject(projectId!),
    enabled: open && isEditing
  })

  const existingProject = (projectResponse as any)?.data?.data

  // Set form values when editing
  useEffect(() => {
    if (open && isEditing && existingProject) {
      form.setFieldsValue({
        name: existingProject.name,
        description: existingProject.description || ""
      })
    } else if (open && !isEditing) {
      form.resetFields()
    }
  }, [open, isEditing, existingProject, form])

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (values: ProjectCreatePayload) =>
      createProject(values, crypto.randomUUID()),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ["prompt-studio", "projects"] })
      const newProject = (response as any)?.data?.data
      if (newProject?.id) {
        setSelectedProjectId(newProject.id)
      }
      notification.success({
        message: t("managePrompts.studio.projects.createSuccess", {
          defaultValue: "Project created"
        }),
        description: t("managePrompts.studio.projects.createSuccessDesc", {
          defaultValue: "Your new project is ready to use."
        })
      })
      onClose()
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error.label", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (values: ProjectCreatePayload) =>
      updateProject(projectId!, values),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-studio", "projects"] })
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "project", projectId]
      })
      notification.success({
        message: t("managePrompts.studio.projects.updateSuccess", {
          defaultValue: "Project updated"
        })
      })
      onClose()
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error.label", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleSubmit = (values: FormValues) => {
    const payload: ProjectCreatePayload = {
      name: values.name.trim(),
      description: values.description?.trim() || null
    }

    if (isEditing) {
      updateMutation.mutate(payload)
    } else {
      createMutation.mutate(payload)
    }
  }

  const isPending = createMutation.isPending || updateMutation.isPending

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={
        isEditing
          ? t("managePrompts.studio.projects.editTitle", {
              defaultValue: "Edit Project"
            })
          : t("managePrompts.studio.projects.createTitle", {
              defaultValue: "Create Project"
            })
      }
      footer={null}
      destroyOnHidden
    >
      {isEditing && isLoadingProject ? (
        <Skeleton paragraph={{ rows: 3 }} />
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          className="mt-4"
        >
          <Form.Item
            name="name"
            label={t("managePrompts.studio.projects.form.name", {
              defaultValue: "Project Name"
            })}
            rules={[
              {
                required: true,
                message: t("managePrompts.studio.projects.form.nameRequired", {
                  defaultValue: "Please enter a project name"
                })
              },
              {
                max: 100,
                message: t("managePrompts.studio.projects.form.nameTooLong", {
                  defaultValue: "Name must be 100 characters or less"
                })
              }
            ]}
          >
            <Input
              placeholder={t(
                "managePrompts.studio.projects.form.namePlaceholder",
                {
                  defaultValue: "e.g., Customer Support Prompts"
                }
              )}
              autoFocus
            />
          </Form.Item>

          <Form.Item
            name="description"
            label={t("managePrompts.studio.projects.form.description", {
              defaultValue: "Description"
            })}
            rules={[
              {
                max: 500,
                message: t(
                  "managePrompts.studio.projects.form.descriptionTooLong",
                  {
                    defaultValue: "Description must be 500 characters or less"
                  }
                )
              }
            ]}
          >
            <Input.TextArea
              placeholder={t(
                "managePrompts.studio.projects.form.descriptionPlaceholder",
                {
                  defaultValue: "Describe what this project is for (optional)"
                }
              )}
              rows={3}
            />
          </Form.Item>

          <div className="flex justify-end gap-2 mt-6">
            <Button type="secondary" onClick={onClose} disabled={isPending}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button type="primary" htmlType="submit" loading={isPending}>
              {isEditing
                ? t("common:save", { defaultValue: "Save" })
                : t("managePrompts.studio.projects.createBtn", {
                    defaultValue: "Create Project"
                  })}
            </Button>
          </div>
        </Form>
      )}
    </Modal>
  )
}
