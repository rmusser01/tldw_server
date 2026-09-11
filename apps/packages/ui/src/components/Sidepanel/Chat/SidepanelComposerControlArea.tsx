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
    <>
      {promptAssistAction}
      {children}
    </>
  )
}
