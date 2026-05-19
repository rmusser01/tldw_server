import React from "react"
import { Button, Form, Input, InputNumber, Modal, Select, Switch } from "antd"
import { LabelWithHelp } from "@/components/Common/LabelWithHelp"
import { DICTIONARY_STARTER_TEMPLATES } from "./dictionaryStarterTemplates"

type DictionaryFormModalProps = {
  title: string
  open: boolean
  onCancel: () => void
  form: any
  onFinish: (values: any) => void
  submitLabel: string
  submitLoading: boolean
  includeActiveField?: boolean
}

export const DictionaryFormModal: React.FC<DictionaryFormModalProps> = ({
  title,
  open,
  onCancel,
  form,
  onFinish,
  submitLabel,
  submitLoading,
  includeActiveField = false
}) => {
  return (
    <Modal title={title} open={open} onCancel={onCancel} footer={null}>
      <Form layout="vertical" form={form} onFinish={onFinish}>
        <Form.Item name="name" label="Name" rules={[{ required: true }]}>
          <Input />
        </Form.Item>
        <Form.Item name="description" label="Description">
          <Input />
        </Form.Item>
        <Form.Item
          name="category"
          label="Category"
          extra="Optional. Use a broad grouping such as Medical, Roleplay, or Product."
        >
          <Input placeholder="Optional category" />
        </Form.Item>
        <Form.Item
          name="tags"
          label="Tags"
          extra="Optional. Add searchable tags to organize dictionaries."
        >
          <Select
            mode="tags"
            tokenSeparators={[","]}
            placeholder="Add tags"
            maxTagCount="responsive"
          />
        </Form.Item>
        {!includeActiveField ? (
          <Form.Item
            name="starter_template"
            label="Starter Template"
            extra="Optional. Adds prebuilt sample entries after the dictionary is created."
          >
            <Select
              allowClear
              placeholder="Start from a blank dictionary"
              options={DICTIONARY_STARTER_TEMPLATES.map((template) => ({
                label: `${template.label} - ${template.description}`,
                value: template.id,
              }))}
            />
          </Form.Item>
        ) : null}
        {includeActiveField ? (
          <Form.Item name="is_active" label="Active" valuePropName="checked">
            <Switch checkedChildren="On" unCheckedChildren="Off" />
          </Form.Item>
        ) : null}
        <Form.Item
          name="default_token_budget"
          label={(
            <LabelWithHelp
              label="Processing limit"
              help="Maximum amount of text processed per message. Leave empty for no limit. Only relevant for large dictionaries."
            />
          )}
          rules={[{ type: "number", min: 1, message: "Must be at least 1 token." }]}
        >
          <InputNumber min={1} style={{ width: "100%" }} placeholder="Optional" />
        </Form.Item>
        <Button type="primary" htmlType="submit" loading={submitLoading} className="w-full">
          {submitLabel}
        </Button>
      </Form>
    </Modal>
  )
}
