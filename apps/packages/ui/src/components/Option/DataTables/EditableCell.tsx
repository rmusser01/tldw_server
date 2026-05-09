import React, { useEffect, useRef, useState } from "react"
import { Checkbox, Input, InputNumber, Tag } from "antd"
import type { ColumnType } from "@/types/data-tables"

interface EditableCellProps {
  value: any
  columnType: ColumnType
  columnName: string
  rowIndex: number
  isEditing: boolean
  isModified?: boolean
  onStartEdit: () => void
  onFinishEdit: (value: any) => void
  onCancelEdit: () => void
}

const DATE_ONLY_PATTERN = /^(\d{4})-(\d{2})-(\d{2})/

const hasDateValue = (value: unknown): boolean =>
  value !== null && value !== undefined && value !== ""

const isLeapYear = (year: number): boolean =>
  year % 400 === 0 || (year % 4 === 0 && year % 100 !== 0)

const isValidDateParts = (year: string, month: string, day: string): boolean => {
  const numericYear = Number(year)
  const numericMonth = Number(month)
  const numericDay = Number(day)
  const daysInMonth = [
    31,
    isLeapYear(numericYear) ? 29 : 28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31
  ]

  return (
    numericMonth >= 1 &&
    numericMonth <= 12 &&
    numericDay >= 1 &&
    numericDay <= daysInMonth[numericMonth - 1]
  )
}

const toDateOnlyString = (value: unknown): string | null => {
  if (!hasDateValue(value)) return null

  if (typeof value === "string") {
    const dateOnlyMatch = value.match(DATE_ONLY_PATTERN)
    if (dateOnlyMatch) {
      const [, year, month, day] = dateOnlyMatch
      if (!isValidDateParts(year, month, day)) return null
      return `${year}-${month}-${day}`
    }
  }

  const date =
    value instanceof Date
      ? value
      : typeof value === "number"
        ? new Date(value)
        : new Date(String(value))
  if (Number.isNaN(date.getTime())) return null

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

/**
 * EditableCell
 *
 * A cell component that can be clicked to enter edit mode.
 * Renders appropriate input based on column type.
 */
export const EditableCell: React.FC<EditableCellProps> = ({
  value,
  columnType,
  columnName,
  rowIndex,
  isEditing,
  isModified,
  onStartEdit,
  onFinishEdit,
  onCancelEdit
}) => {
  const [editValue, setEditValue] = useState(value)
  const inputRef = useRef<any>(null)

  // Reset edit value when value changes or when entering edit mode
  useEffect(() => {
    setEditValue(value)
  }, [value, isEditing])

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      // Small delay to ensure the input is rendered
      const timeoutId = window.setTimeout(() => {
        inputRef.current?.focus?.()
        inputRef.current?.select?.()
      }, 0)
      return () => clearTimeout(timeoutId)
    }
  }, [isEditing])

  // Handle keyboard events
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onFinishEdit(editValue)
    } else if (e.key === "Escape") {
      e.preventDefault()
      onCancelEdit()
    }
  }

  // Handle blur (finish editing)
  const handleBlur = () => {
    onFinishEdit(editValue)
  }

  // Render display value (non-editing mode)
  const renderDisplayValue = () => {
    if (value === null || value === undefined) {
      return <span className="text-text-subtle italic">-</span>
    }

    switch (columnType) {
      case "url":
        if (typeof value === "string" && value) {
          return (
            <a
              href={value}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:text-primaryStrong"
              onClick={(e) => e.stopPropagation()}
            >
              {value.length > 40 ? `${value.slice(0, 40)}...` : value}
            </a>
          )
        }
        return String(value)

      case "boolean":
        return value ? (
          <Tag color="green">Yes</Tag>
        ) : (
          <Tag color="red">No</Tag>
        )

      case "date":
        if (hasDateValue(value)) {
          return toDateOnlyString(value) ?? String(value)
        }
        return String(value)

      case "currency":
        if (typeof value === "number") {
          return `$${value.toFixed(2)}`
        }
        return String(value)

      case "number":
        return typeof value === "number" ? value.toLocaleString() : String(value)

      default:
        return String(value)
    }
  }

  // Render edit input based on column type
  const renderEditInput = () => {
    switch (columnType) {
      case "number":
        return (
          <InputNumber
            ref={inputRef}
            value={editValue}
            onChange={(v) => setEditValue(v)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            className="w-full"
            size="small"
          />
        )

      case "currency":
        return (
          <InputNumber
            ref={inputRef}
            value={editValue}
            onChange={(v) => setEditValue(v)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            prefix="$"
            precision={2}
            className="w-full"
            size="small"
          />
        )

      case "date":
        return (
          <input
            ref={inputRef}
            type="date"
            aria-label={columnName}
            value={toDateOnlyString(editValue) ?? ""}
            onChange={(event) => {
              setEditValue(event.target.value || null)
            }}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            className="w-full rounded border border-border bg-background px-2 py-1 text-sm text-text focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
        )

      case "boolean":
        return (
          <Checkbox
            ref={inputRef}
            checked={!!editValue}
            onChange={(e) => {
              const newValue = e.target.checked
              setEditValue(newValue)
              onFinishEdit(newValue)
            }}
            autoFocus
          />
        )

      case "url":
        return (
          <Input
            ref={inputRef}
            value={editValue || ""}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            placeholder="https://"
            size="small"
          />
        )

      default:
        return (
          <Input
            ref={inputRef}
            value={editValue || ""}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            size="small"
          />
        )
    }
  }

  // Editing mode
  if (isEditing) {
    return (
      <div className="editable-cell-editing" onClick={(e) => e.stopPropagation()}>
        {renderEditInput()}
      </div>
    )
  }

  // Display mode
  return (
    <div
      className={`editable-cell cursor-pointer hover:bg-surface px-1 py-0.5 rounded min-h-[24px] ${
        isModified ? "bg-warn/10 border-l-2 border-warn" : ""
      }`}
      onClick={onStartEdit}
      title="Click to edit"
    >
      {renderDisplayValue()}
    </div>
  )
}
