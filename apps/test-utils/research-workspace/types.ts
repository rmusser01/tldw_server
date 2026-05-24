import type { Page } from "@playwright/test"

export type ResearchWorkspacePlatform = "web" | "extension"

export interface ResearchWorkspaceParityContext {
  platform: ResearchWorkspacePlatform
  page: Page
  optionsUrl?: string
}
