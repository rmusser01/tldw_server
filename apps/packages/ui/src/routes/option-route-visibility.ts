import { CHAT_WORKSPACE_PATH } from "./route-paths"

export const HOSTED_VISIBLE_OPTION_PATHS = new Set([
  "/",
  "/chat",
  CHAT_WORKSPACE_PATH,
  "/media",
  "/knowledge",
  "/collections",
  "/stt",
  "/tts"
])

export const isHostedVisibleOptionPath = (path: string) =>
  HOSTED_VISIBLE_OPTION_PATHS.has(path)
