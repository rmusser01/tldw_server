export const HOSTED_VISIBLE_OPTION_PATHS = new Set([
  "/",
  "/chat",
  "/media",
  "/knowledge",
  "/collections",
  "/stt",
  "/tts"
])

export const isHostedVisibleOptionPath = (path: string) =>
  HOSTED_VISIBLE_OPTION_PATHS.has(path)
