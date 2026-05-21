export type HeaderActionPolicy = {
  showChatSessionActions: boolean
  showChatTitle: boolean
  showSessionModeBadge: boolean
  showShareConversation: boolean
}

const normalizePathname = (pathname: string): string => {
  const trimmed = pathname.trim()
  if (!trimmed || trimmed === "/") return "/"
  return trimmed.endsWith("/") && trimmed.length > 1
    ? trimmed.slice(0, -1)
    : trimmed
}

export const isMainChatRoute = (pathname: string): boolean =>
  normalizePathname(pathname) === "/chat"

export const getHeaderActionPolicy = (pathname: string): HeaderActionPolicy => {
  const chatRoute = isMainChatRoute(pathname)
  return {
    showChatSessionActions: chatRoute,
    showChatTitle: chatRoute,
    showSessionModeBadge: chatRoute,
    showShareConversation: chatRoute,
  }
}
