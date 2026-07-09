import type { SettingsNavItem } from "./settings-nav"

const normalizePathname = (value: string): string => {
  const raw = String(value || "").trim()
  if (!raw) return "/"

  const queryIndex = raw.indexOf("?")
  const hashIndex = raw.indexOf("#")
  const suffixCandidates = [queryIndex, hashIndex].filter((index) => index >= 0)
  const end = suffixCandidates.length > 0 ? Math.min(...suffixCandidates) : raw.length
  const pathOnly = raw.slice(0, end) || "/"

  if (pathOnly.length > 1 && pathOnly.endsWith("/")) {
    return pathOnly.replace(/\/+$/, "") || "/"
  }
  return pathOnly
}

export const isSettingsNavItemActive = (
  currentPathname: string,
  navPath: string
): boolean => {
  const current = normalizePathname(currentPathname)
  const target = normalizePathname(navPath)

  if (current === target) return true
  if (target === "/settings") return false
  return current.startsWith(`${target}/`)
}

export const resolveCurrentSettingsNavItem = (
  pathname: string,
  groups: Array<{ items: SettingsNavItem[] }>
): SettingsNavItem | null => {
  let matchedItem: SettingsNavItem | null = null

  for (const group of groups) {
    for (const item of group.items) {
      if (!isSettingsNavItemActive(pathname, item.to)) continue
      if (!matchedItem || item.to.length > matchedItem.to.length) {
        matchedItem = item
      }
    }
  }

  return matchedItem
}
