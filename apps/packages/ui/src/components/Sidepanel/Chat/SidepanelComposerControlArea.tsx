import type { ReactNode } from "react"

type SidepanelComposerControlAreaProps = {
  promptAssistAction: ReactNode
  children: ReactNode
}

export function SidepanelComposerControlArea({
  promptAssistAction,
  children
}: SidepanelComposerControlAreaProps) {
  return (
    <div
      data-testid="sidepanel-send-action-cluster"
      className="flex shrink-0 items-center gap-2">
      {promptAssistAction}
      {children}
    </div>
  )
}
