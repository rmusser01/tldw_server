import type { LucideIcon } from "lucide-react"
import { UploadCloud } from "lucide-react"
import type { SidebarShortcutId } from "@/services/settings/ui-settings"
import { SIDEBAR_SHORTCUT_IDS } from "@/services/settings/ui-settings"
import { getHeaderShortcutItems } from "@/components/Layouts/header-shortcut-items"

export type SidebarShortcutAction =
  | {
      id: SidebarShortcutId
      icon: LucideIcon
      labelKey: string
      labelDefault: string
      kind: "route"
      path: string
    }
  | {
      id: SidebarShortcutId
      icon: LucideIcon
      labelKey: string
      labelDefault: string
      kind: "event"
      eventName: string
    }

const headerShortcutEntries = Object.fromEntries(
  getHeaderShortcutItems().map((item) => [
    item.id,
    {
      id: item.id,
      icon: item.icon,
      labelKey: item.labelKey,
      labelDefault: item.labelDefault,
      kind: "route",
      path: item.to
    } satisfies SidebarShortcutAction
  ])
) as Record<SidebarShortcutId, SidebarShortcutAction>

const SIDEBAR_SHORTCUT_CONFIG: Record<SidebarShortcutId, SidebarShortcutAction> = {
  "quick-ingest": {
    id: "quick-ingest",
    icon: UploadCloud,
    labelKey: "common:chatSidebar.ingest",
    labelDefault: "Quick Ingest",
    kind: "event",
    eventName: "tldw:open-quick-ingest"
  },
  ...headerShortcutEntries
}

const isShortcutAction = (
  value: SidebarShortcutAction | undefined
): value is SidebarShortcutAction => Boolean(value)

export const SIDEBAR_SHORTCUT_ACTIONS: SidebarShortcutAction[] =
  SIDEBAR_SHORTCUT_IDS
    .map((id) => SIDEBAR_SHORTCUT_CONFIG[id])
    .filter(isShortcutAction)

export const normalizeSidebarShortcutSelection = (
  selection: SidebarShortcutId[]
): SidebarShortcutAction[] => {
  return selection
    .map((id) => SIDEBAR_SHORTCUT_CONFIG[id])
    .filter(isShortcutAction)
}
