const DATE_INPUT_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/

const formatDateParts = (date: Date) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

const parseDateInputParts = (value: string): [number, number, number] | null => {
  const match = DATE_INPUT_PATTERN.exec(value)
  if (!match) return null

  const [, yearText, monthText, dayText] = match
  const year = Number(yearText)
  const month = Number(monthText)
  const day = Number(dayText)
  const date = new Date(year, month - 1, day)

  if (
    !Number.isFinite(date.getTime()) ||
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day
  ) {
    return null
  }

  return [year, month, day]
}

export const formatDateInputValue = (value?: string | null): string => {
  if (!value) return ""
  if (parseDateInputParts(value)) return value

  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return ""

  return formatDateParts(date)
}

export const parseDateInputValue = (
  value: string,
  boundary: "start" | "end"
): string | null => {
  if (!value) return null

  const parts = parseDateInputParts(value)
  if (!parts) return null

  const [year, month, day] = parts
  const date =
    boundary === "start"
      ? new Date(year, month - 1, day, 0, 0, 0, 0)
      : new Date(year, month - 1, day, 23, 59, 59, 999)

  return date.toISOString()
}

export const formatDateOnlyLabel = (value?: string | null): string | null => {
  if (!value) return null
  if (parseDateInputParts(value)) return value

  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return value

  return formatDateParts(date)
}
