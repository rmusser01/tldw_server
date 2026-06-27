import type { ReactNode } from "react"

export const renderWorkspaceMessageActionContent = (
  message: ReactNode,
  action: ReactNode
) => (
  <div className="flex flex-wrap items-center gap-2">
    <span>{message}</span>
    {action}
  </div>
)
