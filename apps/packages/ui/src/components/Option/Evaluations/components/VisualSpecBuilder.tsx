/**
 * VisualSpecBuilder component
 * Form-based spec editor for common evaluation types with JSON toggle.
 */

import React, { useMemo, useState, useEffect } from "react"
import { Checkbox, Divider, InputNumber, Slider, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { Alert as DsAlert } from "@/components/ui/primitives"
import { JsonEditor } from "./JsonEditor"
import {
  getEvalSpecSchema,
  type EvalSpecSchema
} from "../utils/evalSpecSchemas"

interface VisualSpecBuilderProps {
  evalType: string
  specText: string
  onSpecChange: (text: string) => void
  onValidationError?: (error: string | null) => void
}

const getValueAtPath = (obj: Record<string, any>, path: string) => {
  const parts = path.split(".")
  let cursor: any = obj
  for (const part of parts) {
    if (!cursor || typeof cursor !== "object") return undefined
    cursor = cursor[part]
  }
  return cursor
}

const setValueAtPath = (obj: Record<string, any>, path: string, value: any) => {
  const parts = path.split(".")
  const next = JSON.parse(JSON.stringify(obj)) as Record<string, any>
  let cursor: any = next
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i]
    if (!cursor[key] || typeof cursor[key] !== "object") {
      cursor[key] = {}
    }
    cursor = cursor[key]
  }
  cursor[parts[parts.length - 1]] = value
  return next
}

export const VisualSpecBuilder: React.FC<VisualSpecBuilderProps> = ({
  evalType,
  specText,
  onSpecChange,
  onValidationError
}) => {
  const { t } = useTranslation(["evaluations", "common"])
  const schema = useMemo<EvalSpecSchema | null>(
    () => getEvalSpecSchema(evalType),
    [evalType]
  )
  const [showJsonEditor, setShowJsonEditor] = useState(false)

  useEffect(() => {
    setShowJsonEditor(schema?.builder === "json")
  }, [schema?.builder])

  const { specObject, parseError } = useMemo(() => {
    try {
      const parsed = JSON.parse(specText || "{}")
      return { specObject: parsed, parseError: null }
    } catch (e: any) {
      return {
        specObject: schema?.defaultSpec || {},
        parseError: e?.message || "Invalid JSON"
      }
    }
  }, [specText, schema])

  const updateSpec = (next: Record<string, any>) => {
    onSpecChange(JSON.stringify(next, null, 2))
    onValidationError?.(null)
  }

  const updateSpecAtPath = (path: string, value: any) => {
    const next = setValueAtPath(specObject, path, value)
    updateSpec(next)
  }

  const metricsValue = Array.isArray(specObject?.metrics)
    ? specObject.metrics
    : []

  const renderBuilder = () => {
    if (!schema || schema.builder === "json") {
      return (
        <DsAlert
          variant="info"
          title={t("evaluations:specBuilderJsonOnly", {
            defaultValue:
              "This evaluation type uses JSON configuration. Use the editor below."
          })}
        />
      )
    }

    return (
      <div className="space-y-4">
        {schema.metricOptions && schema.metricOptions.length > 0 && (
          <div>
            <div className="mb-2 text-xs text-text-subtle">
              {t("evaluations:specMetricsLabel", {
                defaultValue: "Metrics"
              })}
            </div>
            <Checkbox.Group
              options={schema.metricOptions}
              value={metricsValue}
              onChange={(values) => updateSpecAtPath("metrics", values)}
            />
          </div>
        )}

        {schema.thresholdFields?.map((field) => {
          const current = Number(getValueAtPath(specObject, field.path) ?? 0)
          return (
            <div key={field.key} className="space-y-2">
              <div className="text-xs text-text-subtle">{field.label}</div>
              <div className="flex items-center gap-3">
                <Slider
                  min={field.min ?? 0}
                  max={field.max ?? 1}
                  step={field.step ?? 0.05}
                  className="flex-1"
                  value={Number.isFinite(current) ? current : field.min ?? 0}
                  onChange={(value) => updateSpecAtPath(field.path, value)}
                />
                <InputNumber
                  min={field.min ?? 0}
                  max={field.max ?? 1}
                  step={field.step ?? 0.05}
                  value={Number.isFinite(current) ? current : field.min ?? 0}
                  onChange={(value) =>
                    updateSpecAtPath(field.path, Number(value ?? 0))
                  }
                />
              </div>
            </div>
          )
        })}

        {schema.booleanFields?.map((field) => (
          <div key={field.key} className="flex items-center justify-between">
            <span className="text-xs text-text-subtle">{field.label}</span>
            <Switch
              checked={Boolean(getValueAtPath(specObject, field.path))}
              onChange={(value) => updateSpecAtPath(field.path, value)}
            />
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {parseError && (
        <DsAlert
          variant="warning"
          title={t("evaluations:specBuilderParseWarning", {
            defaultValue:
              "Spec JSON is invalid. Fix the JSON to use the visual builder."
          })}
        >
          {parseError}
        </DsAlert>
      )}

      {renderBuilder()}

      <Divider className="my-2" />

      <div className="flex items-center justify-between">
        <span className="text-xs text-text-subtle">
          {t("evaluations:specAdvancedLabel", {
            defaultValue: "Advanced: Edit JSON"
          })}
        </span>
        <Switch
          checked={showJsonEditor || schema?.builder === "json"}
          disabled={schema?.builder === "json"}
          onChange={(checked) => setShowJsonEditor(checked)}
        />
      </div>

      {(showJsonEditor || schema?.builder === "json") && (
        <JsonEditor
          rows={6}
          value={specText}
          onChange={onSpecChange}
          onValidationError={onValidationError}
        />
      )}
    </div>
  )
}

export default VisualSpecBuilder
