/**
 * NodeConfigPanel Component
 *
 * Configuration sidebar for editing node properties.
 * Dynamically renders form fields based on the step's config schema.
 */

import { useMemo, useCallback, useEffect, useState } from "react"
import {
  Input,
  InputNumber,
  Select,
  Switch,
  Form,
  Button,
  Tooltip,
  Empty,
  Divider
} from "antd"
import {
  Settings,
  Trash2,
  Copy,
  Info
} from "lucide-react"
import type {
  WorkflowNodeData,
  WorkflowNode
} from "@/types/workflow-editor"
import type { ConfigFieldSchema } from "./step-registry"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { Alert } from "@/components/ui/primitives"
import { getStepMetadata } from "./step-registry"
import { schemaHasProperties, schemaToConfigFields } from "./schema-utils"
import { useWorkflowDynamicOptions } from "./dynamic-options"

const { TextArea } = Input

interface NodeConfigPanelProps {
  className?: string
}

export const NodeConfigPanel = ({ className = "" }: NodeConfigPanelProps) => {
  const nodes = useWorkflowEditorStore((s) => s.nodes)
  const selectedNodeIds = useWorkflowEditorStore((s) => s.selectedNodeIds)
  const updateNode = useWorkflowEditorStore((s) => s.updateNode)
  const deleteNodes = useWorkflowEditorStore((s) => s.deleteNodes)
  const duplicateNodes = useWorkflowEditorStore((s) => s.duplicateNodes)
  const stepRegistry = useWorkflowEditorStore((s) => s.stepRegistry)
  const stepSchemas = useWorkflowEditorStore((s) => s.stepSchemas)

  const selectedNode = useMemo(
    () =>
      selectedNodeIds.length === 1
        ? nodes.find((n) => n.id === selectedNodeIds[0])
        : null,
    [nodes, selectedNodeIds]
  )

  const metadata = useMemo(
    () =>
      selectedNode
        ? getStepMetadata(selectedNode.data.stepType, stepRegistry)
        : null,
    [selectedNode, stepRegistry]
  )

  const serverSchema = selectedNode
    ? stepSchemas[selectedNode.data.stepType]
    : undefined

  const serverFields = useMemo(
    () => schemaToConfigFields(serverSchema),
    [serverSchema]
  )

  const mergedServerFields = useMemo(() => {
    if (!metadata || serverFields.length === 0) return serverFields
    const overrides = new Map(metadata.configSchema.map((field) => [field.key, field]))
    return serverFields.map((field) => {
      const override = overrides.get(field.key)
      if (!override) return field
      return {
        ...field,
        ...override,
        description: override.description ?? field.description,
        required: field.required ?? override.required,
        default: field.default ?? override.default,
        options: override.options ?? field.options
      }
    })
  }, [metadata, serverFields])

  const configFields = useMemo(() => {
    if (!metadata) return []
    if (schemaHasProperties(serverSchema)) {
      return mergedServerFields
    }
    if (metadata.configSchema.length > 0) {
      return metadata.configSchema
    }
    return serverFields
  }, [metadata, serverSchema, serverFields, mergedServerFields])

  const { optionsByKey, loadingByKey } = useWorkflowDynamicOptions({
    fields: configFields,
    stepType: selectedNode?.data.stepType,
    config: selectedNode?.data.config
  })

  const resolvedFields = useMemo(
    () =>
      configFields.map((field) => {
        if (field.options && field.options.length > 0) return field
        const dynamicOptions = optionsByKey[field.key]
        if (dynamicOptions && dynamicOptions.length > 0) {
          return { ...field, options: dynamicOptions }
        }
        return field
      }),
    [configFields, optionsByKey]
  )

  const [rawConfigText, setRawConfigText] = useState<string>("")
  const [rawConfigError, setRawConfigError] = useState<string | null>(null)

  useEffect(() => {
    if (!selectedNode) return
    setRawConfigText(
      JSON.stringify(selectedNode.data.config ?? {}, null, 2)
    )
    setRawConfigError(null)
  }, [selectedNode?.id, selectedNode?.data?.config])

  const handleConfigChange = useCallback(
    (key: string, value: unknown) => {
      if (!selectedNode) return
      const currentConfig = selectedNode.data.config as Record<string, unknown>
      updateNode(selectedNode.id, {
        config: {
          ...currentConfig,
          [key]: value
        }
      } as Partial<WorkflowNodeData>)
    },
    [selectedNode, updateNode]
  )

  const handleLabelChange = useCallback(
    (label: string) => {
      if (!selectedNode) return
      updateNode(selectedNode.id, { label })
    },
    [selectedNode, updateNode]
  )

  const handleDelete = useCallback(() => {
    if (!selectedNode) return
    deleteNodes([selectedNode.id])
  }, [selectedNode, deleteNodes])

  const handleDuplicate = useCallback(() => {
    if (!selectedNode) return
    duplicateNodes([selectedNode.id])
  }, [selectedNode, duplicateNodes])

  const handleRawConfigChange = useCallback(
    (value: string) => {
      if (!selectedNode) return
      setRawConfigText(value)
      if (!value.trim()) {
        updateNode(selectedNode.id, { config: {} } as Partial<WorkflowNodeData>)
        setRawConfigError(null)
        return
      }
      try {
        const parsed = JSON.parse(value)
        if (parsed === null || typeof parsed !== "object") {
          throw new Error("Config must be a JSON object")
        }
        updateNode(selectedNode.id, {
          config: parsed as Record<string, unknown>
        } as Partial<WorkflowNodeData>)
        setRawConfigError(null)
      } catch (error) {
        setRawConfigError(
          error instanceof Error ? error.message : "Invalid JSON"
        )
      }
    },
    [selectedNode, updateNode]
  )

  // No node selected
  if (!selectedNode || !metadata) {
    return (
      <div className={`flex flex-col h-full ${className}`}>
        <div className="p-3 border-b border-border">
          <h3 className="text-sm font-semibold text-text-muted flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Node Configuration
          </h3>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              selectedNodeIds.length > 1
                ? "Multiple nodes selected"
                : "Select a node to configure"
            }
          />
        </div>
      </div>
    )
  }

  const shouldShowField = (field: ConfigFieldSchema): boolean => {
    if (!field.showWhen) return true
    const dependentValue = selectedNode.data.config[field.showWhen.field]
    return dependentValue === field.showWhen.value
  }

  const renderField = (field: ConfigFieldSchema) => {
    if (!shouldShowField(field)) return null

    const value = selectedNode.data.config[field.key] ?? field.default
    const isLoading = loadingByKey[field.key] === true

    const label = (
      <span className="flex items-center gap-1">
        {field.label}
        {field.required && <span className="text-danger">*</span>}
        {field.description && (
          <Tooltip title={field.description}>
            <Info className="w-3 h-3 text-text-subtle cursor-help" />
          </Tooltip>
        )}
      </span>
    )

    switch (field.type) {
      case "text":
      case "url":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <Input
              value={value as string}
              onChange={(e) => handleConfigChange(field.key, e.target.value)}
              placeholder={field.description}
              type={field.type === "url" ? "url" : "text"}
            />
          </Form.Item>
        )

      case "textarea":
      case "template-editor":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <TextArea
              value={value as string}
              onChange={(e) => handleConfigChange(field.key, e.target.value)}
              placeholder={field.description}
              rows={field.type === "template-editor" ? 4 : 3}
              className="font-mono text-sm"
            />
            {field.type === "template-editor" && (
              <div className="text-xs text-text-subtle mt-1">
                Use {"{{variable}}"} for template placeholders
              </div>
            )}
          </Form.Item>
        )

      case "number":
      case "duration":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <InputNumber
              value={value as number}
              onChange={(val) => handleConfigChange(field.key, val)}
              min={field.validation?.min}
              max={field.validation?.max}
              step={field.type === "duration" ? 1 : 0.1}
              className="w-full"
              addonAfter={field.type === "duration" ? "sec" : undefined}
            />
          </Form.Item>
        )

      case "select":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <Select
              value={value as string}
              onChange={(val) => handleConfigChange(field.key, val)}
              options={field.options ?? []}
              className="w-full"
              loading={isLoading}
              showSearch
              optionFilterProp="label"
              allowClear
              placeholder={field.description || "Select an option"}
              notFoundContent={isLoading ? "Loading options..." : undefined}
            />
          </Form.Item>
        )

      case "multiselect":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <Select
              mode="multiple"
              value={value as string[]}
              onChange={(val) => handleConfigChange(field.key, val)}
              options={field.options ?? []}
              className="w-full"
              loading={isLoading}
              showSearch
              optionFilterProp="label"
              allowClear
              placeholder={field.description || "Select options"}
              notFoundContent={isLoading ? "Loading options..." : undefined}
            />
          </Form.Item>
        )

      case "checkbox":
        return (
          <Form.Item key={field.key} className="mb-3">
            <div className="flex items-center justify-between">
              {label}
              <Switch
                checked={value as boolean}
                onChange={(checked) => handleConfigChange(field.key, checked)}
              />
            </div>
          </Form.Item>
        )

      case "json-editor":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <TextArea
              value={
                typeof value === "string"
                  ? value
                  : JSON.stringify(value, null, 2)
              }
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value)
                  handleConfigChange(field.key, parsed)
                } catch {
                  // Keep as string if not valid JSON
                  handleConfigChange(field.key, e.target.value)
                }
              }}
              rows={4}
              className="font-mono text-xs"
            />
          </Form.Item>
        )

      case "model-picker":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <Select
              value={value as string}
              onChange={(val) => handleConfigChange(field.key, val)}
              className="w-full"
              options={field.options ?? []}
              loading={isLoading}
              showSearch
              optionFilterProp="label"
              allowClear
              placeholder={field.description || "Select a model"}
              notFoundContent={isLoading ? "Loading options..." : undefined}
            />
          </Form.Item>
        )

      case "collection-picker":
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <Select
              value={value as string}
              onChange={(val) => handleConfigChange(field.key, val)}
              className="w-full"
              options={field.options ?? []}
              loading={isLoading}
              showSearch
              optionFilterProp="label"
              allowClear
              placeholder={field.description || "Select a collection"}
              notFoundContent={isLoading ? "Loading options..." : undefined}
            />
          </Form.Item>
        )

      default:
        return (
          <Form.Item key={field.key} label={label} className="mb-3">
            <Input
              value={value as string}
              onChange={(e) => handleConfigChange(field.key, e.target.value)}
            />
          </Form.Item>
        )
    }
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-text-muted flex items-center gap-2">
            <Settings className="w-4 h-4" />
            {metadata.label}
          </h3>
          <div className="flex items-center gap-1">
            <Tooltip title="Duplicate">
              <Button
                type="text"
                size="small"
                aria-label="Duplicate node"
                title="Duplicate node"
                icon={<Copy className="w-4 h-4" />}
                onClick={handleDuplicate}
              />
            </Tooltip>
            <Tooltip title="Delete">
              <Button
                type="text"
                size="small"
                danger
                aria-label="Delete node"
                title="Delete node"
                icon={<Trash2 className="w-4 h-4" />}
                onClick={handleDelete}
              />
            </Tooltip>
          </div>
        </div>
        <p className="text-xs text-text-muted">
          {metadata.description}
        </p>
      </div>

      {/* Configuration Form */}
      <div className="flex-1 overflow-y-auto p-3">
        <Form layout="vertical" size="small">
          {/* Node Label */}
          <Form.Item label="Node Label" className="mb-3">
            <Input
              value={selectedNode.data.label}
              onChange={(e) => handleLabelChange(e.target.value)}
              placeholder="Enter node label"
            />
          </Form.Item>

          <Divider className="my-3" />

          {/* Dynamic Config Fields */}
          {resolvedFields.map(renderField)}

          {resolvedFields.length === 0 && (
            <Form.Item label="Raw Config" className="mb-3">
              <TextArea
                value={rawConfigText}
                onChange={(e) => handleRawConfigChange(e.target.value)}
                rows={6}
                className="font-mono text-xs"
                placeholder='{"key": "value"}'
              />
              {rawConfigError && (
                <Alert
                  variant="error"
                  title={rawConfigError}
                  className="mt-2"
                />
              )}
            </Form.Item>
          )}
        </Form>
      </div>

      {/* Port Info */}
      <div className="p-3 border-t border-border bg-surface">
        <div className="text-xs">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-text-muted">
              Inputs:
            </span>
            <span className="text-text-subtle">
              {metadata.inputs.length > 0
                ? metadata.inputs.map((i) => i.label).join(", ")
                : "None"}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-text-muted">
              Outputs:
            </span>
            <span className="text-text-subtle">
              {metadata.outputs.length > 0
                ? metadata.outputs.map((o) => o.label).join(", ")
                : "None"}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default NodeConfigPanel
