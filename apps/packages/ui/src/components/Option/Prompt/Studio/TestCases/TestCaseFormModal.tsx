import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Modal, Form, Input, Select, Switch, notification, Skeleton } from "antd"
import React, { useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  createTestCase,
  updateTestCase,
  getTestCase,
  type TestCaseCreatePayload,
  type TestCaseUpdatePayload
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"

type TestCaseFormModalProps = {
  open: boolean
  testCaseId: number | null
  projectId: number
  onClose: () => void
}

type FormValues = {
  name?: string
  description?: string
  inputs: string // JSON string
  expected_outputs?: string // JSON string
  tags?: string[]
  is_golden?: boolean
}

export const TestCaseFormModal: React.FC<TestCaseFormModalProps> = ({
  open,
  testCaseId,
  projectId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const queryClient = useQueryClient()

  const isEditing = testCaseId !== null

  // Fetch existing test case for editing
  const { data: testCaseResponse, isLoading: isLoadingTestCase } = useQuery({
    queryKey: ["prompt-studio", "test-case", testCaseId],
    queryFn: () => getTestCase(testCaseId!),
    enabled: open && isEditing
  })

  const existingTestCase = (testCaseResponse as any)?.data?.data

  // Set form values when editing
  useEffect(() => {
    if (open && isEditing && existingTestCase) {
      form.setFieldsValue({
        name: existingTestCase.name || "",
        description: existingTestCase.description || "",
        inputs: JSON.stringify(existingTestCase.inputs, null, 2),
        expected_outputs: existingTestCase.expected_outputs
          ? JSON.stringify(existingTestCase.expected_outputs, null, 2)
          : "",
        tags: existingTestCase.tags || [],
        is_golden: existingTestCase.is_golden || false
      })
    } else if (open && !isEditing) {
      form.resetFields()
      form.setFieldsValue({
        inputs: "{}",
        is_golden: false
      })
    }
  }, [open, isEditing, existingTestCase, form])

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (payload: TestCaseCreatePayload) => createTestCase(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-cases", projectId]
      })
      notification.success({
        message: t("managePrompts.studio.testCases.createSuccess", {
          defaultValue: "Test case created"
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
    mutationFn: (payload: TestCaseUpdatePayload) =>
      updateTestCase(testCaseId!, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-cases", projectId]
      })
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-case", testCaseId]
      })
      notification.success({
        message: t("managePrompts.studio.testCases.updateSuccess", {
          defaultValue: "Test case updated"
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

  const parseJson = (value: string | undefined): Record<string, any> | null => {
    if (!value?.trim()) return null
    try {
      return JSON.parse(value)
    } catch {
      return null
    }
  }

  const handleSubmit = (values: FormValues) => {
    const inputs = parseJson(values.inputs)
    if (!inputs) {
      notification.error({
        message: t("managePrompts.studio.testCases.invalidInputs", {
          defaultValue: "Invalid inputs JSON"
        })
      })
      return
    }

    const expectedOutputs = parseJson(values.expected_outputs)

    if (isEditing) {
      const payload: TestCaseUpdatePayload = {
        name: values.name?.trim() || null,
        description: values.description?.trim() || null,
        inputs,
        expected_outputs: expectedOutputs,
        tags: values.tags && values.tags.length > 0 ? values.tags : null,
        is_golden: values.is_golden
      }
      updateMutation.mutate(payload)
    } else {
      const payload: TestCaseCreatePayload = {
        project_id: projectId,
        name: values.name?.trim() || null,
        description: values.description?.trim() || null,
        inputs,
        expected_outputs: expectedOutputs,
        tags: values.tags && values.tags.length > 0 ? values.tags : null,
        is_golden: values.is_golden
      }
      createMutation.mutate(payload)
    }
  }

  const validateJson = (_: any, value: string) => {
    if (!value?.trim()) return Promise.resolve()
    try {
      JSON.parse(value)
      return Promise.resolve()
    } catch {
      return Promise.reject(
        new Error(
          t("managePrompts.studio.testCases.invalidJson", {
            defaultValue: "Invalid JSON format"
          })
        )
      )
    }
  }

  const isPending = createMutation.isPending || updateMutation.isPending

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={
        isEditing
          ? t("managePrompts.studio.testCases.editTitle", {
              defaultValue: "Edit Test Case"
            })
          : t("managePrompts.studio.testCases.createTitle", {
              defaultValue: "Create Test Case"
            })
      }
      width={600}
      footer={null}
      destroyOnClose
    >
      {isEditing && isLoadingTestCase ? (
        <Skeleton paragraph={{ rows: 6 }} />
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          className="mt-4"
        >
          <Form.Item
            name="name"
            label={t("managePrompts.studio.testCases.form.name", {
              defaultValue: "Name (optional)"
            })}
          >
            <Input
              placeholder={t(
                "managePrompts.studio.testCases.form.namePlaceholder",
                {
                  defaultValue: "e.g., Happy path - simple query"
                }
              )}
            />
          </Form.Item>

          <Form.Item
            name="description"
            label={t("managePrompts.studio.testCases.form.description", {
              defaultValue: "Description (optional)"
            })}
          >
            <Input.TextArea
              placeholder={t(
                "managePrompts.studio.testCases.form.descriptionPlaceholder",
                {
                  defaultValue: "What does this test case verify?"
                }
              )}
              rows={2}
            />
          </Form.Item>

          <Form.Item
            name="inputs"
            label={t("managePrompts.studio.testCases.form.inputs", {
              defaultValue: "Inputs (JSON)"
            })}
            rules={[
              {
                required: true,
                message: t(
                  "managePrompts.studio.testCases.form.inputsRequired",
                  {
                    defaultValue: "Please provide inputs"
                  }
                )
              },
              { validator: validateJson }
            ]}
          >
            <Input.TextArea
              rows={4}
              className="font-mono text-sm"
              placeholder={`{\n  "customer_query": "How do I reset my password?"\n}`}
            />
          </Form.Item>

          <Form.Item
            name="expected_outputs"
            label={t("managePrompts.studio.testCases.form.expectedOutputs", {
              defaultValue: "Expected Outputs (JSON, optional)"
            })}
            rules={[{ validator: validateJson }]}
          >
            <Input.TextArea
              rows={4}
              className="font-mono text-sm"
              placeholder={`{\n  "response": "To reset your password..."\n}`}
            />
          </Form.Item>

          <Form.Item
            name="tags"
            label={t("managePrompts.studio.testCases.form.tags", {
              defaultValue: "Tags"
            })}
          >
            <Select
              mode="tags"
              placeholder={t(
                "managePrompts.studio.testCases.form.tagsPlaceholder",
                {
                  defaultValue: "Add tags..."
                }
              )}
              tokenSeparators={[","]}
            />
          </Form.Item>

          <Form.Item
            name="is_golden"
            label={t("managePrompts.studio.testCases.form.isGolden", {
              defaultValue: "Golden Test Case"
            })}
            valuePropName="checked"
            tooltip={t("managePrompts.studio.testCases.form.isGoldenHelp", {
              defaultValue:
                "Golden test cases are important examples that should always pass."
            })}
          >
            <Switch />
          </Form.Item>

          <div className="flex justify-end gap-2 mt-6">
            <Button type="secondary" onClick={onClose} disabled={isPending}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button type="primary" htmlType="submit" loading={isPending}>
              {isEditing
                ? t("common:save", { defaultValue: "Save" })
                : t("managePrompts.studio.testCases.createBtn", {
                    defaultValue: "Create"
                  })}
            </Button>
          </div>
        </Form>
      )}
    </Modal>
  )
}
