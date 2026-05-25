import {
  CHAT_WORKSPACE_PATH,
  RESEARCH_WORKSPACE_PATH
} from "./route-paths"
import { normalizeRoutePath } from "./route-path-normalization"

export const HOSTED_VISIBLE_OPTION_PATHS_LIST = [
  "/",
  "/chat",
  CHAT_WORKSPACE_PATH,
  "/media",
  "/knowledge",
  "/collections",
  RESEARCH_WORKSPACE_PATH,
  "/stt",
  "/tts"
] as const

export const HOSTED_VISIBLE_OPTION_PATHS = new Set(
  HOSTED_VISIBLE_OPTION_PATHS_LIST.map((path) => normalizeRoutePath(path))
)
