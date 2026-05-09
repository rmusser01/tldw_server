const padDateTimePart = (value: number): string => String(value).padStart(2, "0")

export const formatKanbanDateTimeLocalValue = (
  value: string | null | undefined
): string => {
  if (!value) return ""

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""

  const year = date.getFullYear()
  const month = padDateTimePart(date.getMonth() + 1)
  const day = padDateTimePart(date.getDate())
  const hour = padDateTimePart(date.getHours())
  const minute = padDateTimePart(date.getMinutes())

  return `${year}-${month}-${day}T${hour}:${minute}`
}

export const parseKanbanDateTimeLocalValue = (
  value: string | null | undefined
): string | null => {
  if (!value) return null

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null

  return date.toISOString()
}
