export const normalizeRoutePath = (path: string): string => {
  const trimmed = path.trim()
  if (!trimmed) {
    return "/"
  }

  const withoutHash = trimmed.split("#", 1)[0]
  const withoutQuery = withoutHash.split("?", 1)[0] || "/"
  if (withoutQuery === "/") {
    return withoutQuery
  }

  return withoutQuery.replace(/\/+$/, "")
}
