import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import type { PrivateWorkspaceScope } from "@/hooks/usePrivateWorkspaceScope"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

/** Explicit owner for component-only tests; live verification is tested separately. */
export function testKanbanScope(config: TldwConfig): PrivateWorkspaceScope {
  const controller = new AbortController()
  return {
    key: deriveConnectionAuthorityId(config), config,
    requestScope: { config, userId: null },
    signal: controller.signal, isCurrent: () => !controller.signal.aborted
  }
}
