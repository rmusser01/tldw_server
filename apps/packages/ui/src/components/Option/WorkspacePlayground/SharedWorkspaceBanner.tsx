/**
 * Banner shown at the top of a workspace when accessed via sharing.
 * Displays owner info, access level, and restrictions.
 */
import React from "react"
import { Tag } from "antd"
import { Users, Eye, PenLine, Edit3 } from "lucide-react"
import { useSharedWorkspace } from "./SharedWorkspaceContext"
import {
  ACCESS_LEVEL_LABELS,
  ACCESS_LEVEL_COLORS,
  type AccessLevel,
} from "@/types/sharing"

const ACCESS_LEVEL_ICONS: Record<string, React.ReactNode> = {
  view_chat: <Eye className="h-3.5 w-3.5" />,
  view_chat_add: <PenLine className="h-3.5 w-3.5" />,
  full_edit: <Edit3 className="h-3.5 w-3.5" />,
}

export const SharedWorkspaceBanner: React.FC = () => {
  const { isShared, ownerUserId, accessLevel } = useSharedWorkspace()

  if (!isShared || !accessLevel) return null

  return (
    <div
      className="flex items-center gap-3 border-b border-border bg-blue-50 px-4 py-2 text-sm dark:bg-blue-950/30"
      data-testid="shared-workspace-banner"
    >
      <Users className="h-4 w-4 text-blue-600 dark:text-blue-400" />
      <span className="text-text-muted">
        Shared by user #{ownerUserId}
      </span>
      <Tag
        color={ACCESS_LEVEL_COLORS[accessLevel] || "default"}
        icon={ACCESS_LEVEL_ICONS[accessLevel]}
      >
        {ACCESS_LEVEL_LABELS[accessLevel as AccessLevel] || accessLevel}
      </Tag>
      {accessLevel === "view_chat" && (
        <span className="text-xs text-text-muted">
          You can view and chat, but cannot modify sources or outputs
        </span>
      )}
      {accessLevel === "view_chat_add" && (
        <span className="text-xs text-text-muted">
          You can add new sources but cannot edit existing ones
        </span>
      )}
    </div>
  )
}
