import React from "react"
import { Input, Button, Space, Typography, Switch } from "antd"
import { Plus, Trash2 } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"

const { Text } = Typography
const { TextArea } = Input

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
 * One editable argument row. `raw` is the value as accepted/emitted (NOT the
 * display string), so values the user never touches round-trip unchanged.
 */
interface EditorRow {
  /** Row-owned id: assigned at pair creation, kept across renames, reorders,
   *  and sibling deletion — never derived from the (editable) arg key. */
  id: string
  key: string
  raw: unknown
}

const displayValue = (raw: unknown): string =>
  typeof raw === "string" ? raw : JSON.stringify(raw)

// The editor's long-standing value coercion: strings that parse as JSON
// scalars are emitted as those scalars; objects, arrays, and unparsable
// text stay strings.
const coerceValue = (text: string): unknown => {
  if (text.trim() === "") return text
  try {
    const parsed = JSON.parse(text)
    if (typeof parsed !== "object" || parsed === null) return parsed
  } catch {
    // Keep as string if not valid JSON
  }
  return text
}

const rowsFromValue = (
  value: Record<string, any>,
  reuseIds: Map<string, string>,
  allocateId: () => string
): EditorRow[] =>
  Object.entries(value ?? {}).map(([key, raw]) => ({
    id: reuseIds.get(key) ?? allocateId(),
    key,
    raw
  }))

// Structural equality for row arrays: parents commonly pass
// `settings.customArgs || {}` (a fresh {} per render) or clone the emitted
// object, so the re-sync effect must not churn rows — or re-render the
// editor — when the incoming value is content-identical.
const rowsEqual = (a: EditorRow[], b: EditorRow[]): boolean =>
  a.length === b.length &&
  a.every(
    (row, index) =>
      row.id === b[index].id &&
      row.key === b[index].key &&
      row.raw === b[index].raw
  )

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

  // Rows are component state with row-owned ids (C-S5 fix1): an earlier
  // scheme derived ids from the argument key, so every keystroke in a key
  // input changed the row's React key and remounted it, dropping focus.
  // Renames now update the row in place; the value prop stays the source of
  // truth via the re-sync effect below.
  const nextRowIdRef = React.useRef(0)
  const allocateId = () => {
    nextRowIdRef.current += 1
    return `arg-${nextRowIdRef.current}`
  }
  // The last object this editor emitted through onChange: when the parent
  // echoes it straight back (the normal controlled flow), re-syncing would
  // pointlessly rebuild every row.
  const lastEmittedRef = React.useRef<Record<string, any>>(value)
  const [rows, setRows] = React.useState<EditorRow[]>(() =>
    rowsFromValue(value, new Map(), allocateId)
  )

  // Prop -> rows sync for changes that originate outside this editor (a
  // different profile loaded, JSON-mode edits, a cloning parent). Keys that
  // already exist keep their row ids, and content-identical values keep the
  // existing rows array, so external refreshes do not remount the rows and
  // unstable-but-equal props do not re-render the editor.
  React.useEffect(() => {
    if (value === lastEmittedRef.current) return
    setRows((prev) => {
      const next = rowsFromValue(
        value,
        new Map(prev.map((row) => [row.key, row.id])),
        allocateId
      )
      return rowsEqual(prev, next) ? prev : next
    })
  }, [value])

  // Sync JSON text when switching to JSON mode or when value changes
  React.useEffect(() => {
    if (jsonMode) {
      setJsonText(JSON.stringify(value, null, 2))
      setJsonError(null)
    }
  }, [jsonMode, value])

  const emit = (nextRows: EditorRow[]) => {
    const emitted: Record<string, any> = {}
    for (const row of nextRows) {
      emitted[row.key] = row.raw
    }
    // Renaming onto an existing key collapses the duplicate, exactly like
    // the previous value-derived rendering did; mirror that in the rows.
    if (Object.keys(emitted).length !== nextRows.length) {
      const collapsed = rowsFromValue(
        emitted,
        new Map(nextRows.map((row) => [row.key, row.id])),
        allocateId
      )
      setRows(collapsed)
    } else {
      setRows(nextRows)
    }
    lastEmittedRef.current = emitted
    onChange(emitted)
  }

  const handleAddPair = () => {
    // Object keys are unique, so {"": ""} can only exist once — don't stack
    // a second unnamed row on top of an existing one.
    if (rows.some((row) => row.key === "")) return
    emit([...rows, { id: allocateId(), key: "", raw: "" }])
  }

  const handleRemovePair = (row: EditorRow) => {
    emit(rows.filter((candidate) => candidate.id !== row.id))
  }

  const handleKeyChange = (row: EditorRow, nextKey: string) => {
    if (nextKey === "") {
      // Clearing a key removes the pair (previous behavior: the emitted
      // object dropped it and the row list re-derived from the value).
      emit(rows.filter((candidate) => candidate.id !== row.id))
      return
    }
    // In-place rename: same id, so the row — and the focused input's DOM
    // node — survive every keystroke.
    emit(
      rows.map((candidate) =>
        candidate.id === row.id
          ? {
              id: candidate.id,
              key: nextKey,
              raw: coerceValue(displayValue(candidate.raw))
            }
          : candidate
      )
    )
  }

  const handleValueChange = (row: EditorRow, nextValue: string) => {
    emit(
      rows.map((candidate) =>
        candidate.id === row.id
          ? { id: candidate.id, key: candidate.key, raw: coerceValue(nextValue) }
          : candidate
      )
    )
  }

  const handleJsonChange = (text: string) => {
    setJsonText(text)
    try {
      const parsed = JSON.parse(text)
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        setJsonError(null)
        lastEmittedRef.current = parsed
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
    // Leaving JSON mode: rows may be stale if the JSON text rewrote args,
    // so re-materialize them from the current value.
    if (!checked) {
      setRows((prev) =>
        rowsFromValue(
          value,
          new Map(prev.map((row) => [row.key, row.id])),
          allocateId
        )
      )
    }
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
            aria-label="Toggle JSON mode"
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
          {rows.length === 0 ? (
            <Text type="secondary" className="text-sm">
              {placeholder}
            </Text>
          ) : (
            rows.map((row) => (
              <Space key={row.id} className="w-full" align="start">
                <Input
                  size="small"
                  placeholder="key"
                  value={row.key}
                  onChange={(e) => handleKeyChange(row, e.target.value)}
                  style={{ width: 120 }}
                  className="font-mono"
                />
                <Input
                  size="small"
                  placeholder="value"
                  value={displayValue(row.raw)}
                  onChange={(e) => handleValueChange(row, e.target.value)}
                  style={{ width: 160 }}
                  className="font-mono"
                />
                <Button
                  size="small"
                  type="text"
                  danger
                  icon={<Trash2 size={14} />}
                  onClick={() => handleRemovePair(row)}
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
