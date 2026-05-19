export const toUnreadCount = (value: unknown): number => {
  const next = typeof value === "number" ? value : Number(value)
  return Number.isFinite(next) ? next : 0
}
