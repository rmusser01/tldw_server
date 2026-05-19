import { CHAT_WORKSPACE_PATH, RESEARCH_STUDIO_PATH } from "./route-paths"

export const HOSTED_VISIBLE_OPTION_PATHS = new Set([
  "/",
  "/chat",
  CHAT_WORKSPACE_PATH,
  "/media",
  "/knowledge",
  "/collections",
  RESEARCH_STUDIO_PATH,
  "/stt",
  "/tts"
])

export const isHostedVisibleOptionPath = (path: string) =>
  HOSTED_VISIBLE_OPTION_PATHS.has(path)
