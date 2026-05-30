import React from "react"
import { Input, Button, Space, Typography, Switch } from "antd"
import { Plus, Trash2 } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"

const { Text } = Typography
const { TextArea } = Input

interface KeyValuePair {
  key: string
  value: string
}

interface ServerArgsEditorProps {
  /** Current server arguments as key-value object */
  value: Record<string, any>
  /** Callback when arguments change */
  onChange: (args: Record<string, any>) => void
  /** Placeholder text for empty state */
  placeholder?: string
  className?: string
}

/**
 * Editor for custom llama.cpp server arguments.
 * Supports both form-based key-value editing and raw JSON mode.
 */
export const ServerArgsEditor: React.FC<ServerArgsEditorProps> = ({
  value,
  onChange,
  placeholder = "Add custom server arguments...",
  className
}) => {
  const [jsonMode, setJsonMode] = React.useState(false)
  const [jsonText, setJsonText] = React.useState("")
  const [jsonError, setJsonError] = React.useState<string | null>(null)

  // Convert object to key-value pairs for form mode
  const pairs: KeyValuePair[] = React.useMemo(() => {
    return Object.entries(value).map(([key, val]) => ({
      key,
      value: typeof val === "string" ? val : JSON.stringify(val)
    }))
  }, [value])

  // Sync JSON text when switching to JSON mode or when value changes
  React.useEffect(() => {
    if (jsonMode) {
      setJsonText(JSON.stringify(value, null, 2))
      setJsonError(null)
    }
  }, [jsonMode, value])

  const handleAddPair = () => {
    onChange({ ...value, "": "" })
  }

  const handleRemovePair = (keyToRemove: string) => {
    const newValue = { ...value }
    delete newValue[keyToRemove]
    onChange(newValue)
  }

  const handlePairChange = (oldKey: string, newKey: string, newValue: string) => {
    const entries = Object.entries(value)
    const newObj: Record<string, any> = {}

    for (const [k, v] of entries) {
      if (k === oldKey) {
        if (newKey) {
          // Try to parse the value as JSON (for numbers, booleans, etc.)
          let parsedValue: any = newValue
          try {
            if (newValue.trim() !== "") {
              const parsed = JSON.parse(newValue)
              if (typeof parsed !== "object" || parsed === null) {
                parsedValue = parsed
              }
            }
          } catch {
            // Keep as string if not valid JSON
          }
          newObj[newKey] = parsedValue
        }
      } else {
        newObj[k] = v
      }
    }

    onChange(newObj)
  }

  const handleJsonChange = (text: string) => {
    setJsonText(text)
    try {
      const parsed = JSON.parse(text)
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        setJsonError(null)
        onChange(parsed)
      } else {
        setJsonError("Must be a JSON object")
      }
    } catch (e) {
      setJsonError("Invalid JSON")
    }
  }

  const handleModeSwitch = (checked: boolean) => {
    if (!checked && jsonError) {
      // Don't switch if there's a JSON error
      return
    }
    setJsonMode(checked)
  }

  return (
    <div className={className}>
      <div className="mb-2 flex items-center justify-between">
        <Text type="secondary" className="text-xs">
          {jsonMode ? "JSON mode" : "Key-value mode"}
        </Text>
        <Space size="small">
          <Text type="secondary" className="text-xs">
            JSON
          </Text>
          <Switch
            size="small"
            checked={jsonMode}
            onChange={handleModeSwitch}
          />
        </Space>
      </div>

      {jsonMode ? (
        <div>
          <TextArea
            value={jsonText}
            onChange={(e) => handleJsonChange(e.target.value)}
            placeholder='{"key": "value"}'
            autoSize={{ minRows: 3, maxRows: 10 }}
            status={jsonError ? "error" : undefined}
            className="font-mono text-xs"
          />
          {jsonError && (
            <DesignSystemAlert
              variant="error"
              title={jsonError}
              className="mt-2"
            />
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {pairs.length === 0 ? (
            <Text type="secondary" className="text-sm">
              {placeholder}
            </Text>
          ) : (
            pairs.map((pair, index) => (
              <Space key={index} className="w-full" align="start">
                <Input
                  size="small"
                  placeholder="key"
                  value={pair.key}
                  onChange={(e) => handlePairChange(pair.key, e.target.value, pair.value)}
                  style={{ width: 120 }}
                  className="font-mono"
                />
                <Input
                  size="small"
                  placeholder="value"
                  value={pair.value}
                  onChange={(e) => handlePairChange(pair.key, pair.key, e.target.value)}
                  style={{ width: 160 }}
                  className="font-mono"
                />
                <Button
                  size="small"
                  type="text"
                  danger
                  icon={<Trash2 size={14} />}
                  onClick={() => handleRemovePair(pair.key)}
                />
              </Space>
            ))
          )}
          <Button
            size="small"
            type="dashed"
            icon={<Plus size={14} />}
            onClick={handleAddPair}
          >
            Add argument
          </Button>
        </div>
      )}
    </div>
  )
}

export default ServerArgsEditor
