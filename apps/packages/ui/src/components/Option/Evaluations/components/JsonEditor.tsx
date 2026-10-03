/**
 * JsonEditor component
 * TextArea with JSON validation on blur and syntax error display
 */

import React, { useState, useCallback } from "react"
import { Input } from "antd"
import type { TextAreaProps } from "antd/es/input/TextArea"
import { Highlight } from "prism-react-renderer"
import { resolveTheme, safeLanguage } from "@/utils/code-theme"

interface JsonEditorProps extends Omit<TextAreaProps, "onChange"> {
  value: string
  onChange: (value: string) => void
  onValidationError?: (error: string | null) => void
  showError?: boolean
  errorClassName?: string
}

export const JsonEditor: React.FC<JsonEditorProps> = ({
  value,
  onChange,
  onValidationError,
  showError = true,
  errorClassName = "mt-1 text-xs text-danger",
  className,
  ...textAreaProps
}) => {
  const [error, setError] = useState<string | null>(null)

  const handleBlur = useCallback(() => {
    if (!value || !value.trim()) {
      setError(null)
      onValidationError?.(null)
      return
    }
    try {
      JSON.parse(value)
      setError(null)
      onValidationError?.(null)
    } catch (e: any) {
      const errorMessage = e?.message || "Invalid JSON"
      setError(errorMessage)
      onValidationError?.(errorMessage)
    }
  }, [value, onValidationError])

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value)
      // Clear error on change to allow user to fix
      if (error) {
        setError(null)
        onValidationError?.(null)
      }
    },
    [onChange, error, onValidationError]
  )

  return (
    <div className="w-full">
      <Input.TextArea
        {...textAreaProps}
        value={value}
        onChange={handleChange}
        onBlur={handleBlur}
        className={`font-mono text-sm ${className || ""}`}
        status={error ? "error" : undefined}
      />
      {showError && error && <div className={errorClassName}>{error}</div>}
      {value?.trim() && (
        <div className="mt-2 rounded border border-border bg-surface2 text-xs">
          <Highlight code={value} language={safeLanguage("json")} theme={resolveTheme("auto")}>
            {({
              className: highlightClassName,
              style,
              tokens,
              getLineProps,
              getTokenProps
            }) => (
              <pre
                className={`${highlightClassName} m-0 overflow-auto px-3 py-2`}
                style={{
                  ...style,
                  background: "transparent",
                  fontFamily: "var(--font-mono)"
                }}
              >
                {tokens.map((line, i) => (
                  <div key={i} {...getLineProps({ line })}>
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token })} />
                    ))}
                  </div>
                ))}
              </pre>
            )}
          </Highlight>
        </div>
      )}
    </div>
  )
}

export default JsonEditor
