import {
  HOSTED_VISIBLE_OPTION_PATHS,
  HOSTED_VISIBLE_OPTION_PATHS_LIST
} from "./route-hosted-visibility"
import { normalizeRoutePath } from "./route-path-normalization"

export { HOSTED_VISIBLE_OPTION_PATHS, HOSTED_VISIBLE_OPTION_PATHS_LIST }

export const isHostedVisibleOptionPath = (path: string) =>
  HOSTED_VISIBLE_OPTION_PATHS.has(normalizeRoutePath(path))
