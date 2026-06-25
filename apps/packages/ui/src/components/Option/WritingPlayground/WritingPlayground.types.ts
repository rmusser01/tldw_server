import type { ReactNode } from "react"

export type InspectorTabKey = "sampling" | "context" | "setup" | "inspect" | "annotations" | "characters" | "research" | "agent" | "feedback"

export interface EssentialsStripProps {
  children: ReactNode
}

export interface WritingPlaygroundPanelProps {
  title?: ReactNode
  extra?: ReactNode
  children: ReactNode
  testId?: string
}
