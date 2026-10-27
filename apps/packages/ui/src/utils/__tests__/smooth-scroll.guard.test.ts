import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

// AX-19 (#3124): a behaviour passed from script ("smooth") wins over the CSS
// reduced-motion block, so scripted scrolls must ask smoothScrollBehavior().
// Chat, the side panel and quick chat are done; the files below still pass a
// literal "smooth" and are the remaining work. Remove a file from this list
// when you convert it; never add one.
const REMAINING = new Set([
  "components/Common/TutorialRunner.tsx",
  "components/DocumentWorkspace/DocumentTabBar.tsx",
  "components/DocumentWorkspace/RightPanel/DocumentChat.tsx",
  "components/Flashcards/tabs/ImportExport/ImportPanel.tsx",
  "components/Media/DiffViewModal.tsx",
  "components/Media/hooks/useReadingProgress.tsx",
  "components/Media/hooks/useTranscriptDisplay.tsx",
  "components/Option/ACPPlayground/ACPChatPanel.tsx",
  "components/Option/ChunkingPlayground/ChunkCardView.tsx",
  "components/Option/Dictionaries/components/useValidationRowHighlight.ts",
  "components/Option/KnowledgeQA/KnowledgeQAProvider.tsx",
  "components/Option/KnowledgeQA/SourceList.tsx",
  "components/Option/ResearchWorkspace/ChatPane/index.tsx",
  "components/Option/ResearchWorkspace/SourcesPane/index.tsx",
  "components/Option/Settings/FamilyGuardrailsWizard.tsx",
  "components/Option/Settings/tldw-settings-tabs.tsx",
  "components/Option/Settings/TldwBillingSettings.tsx",
  "components/Option/Watchlists/JobsTab/FilterBuilder.tsx",
  "components/Option/WritingPlayground/AIAgentTab.tsx",
  "components/PersonaGarden/AssistantDefaultsPanel.tsx",
  "components/PersonaGarden/CommandsPanel.tsx",
  "components/PersonaGarden/ConnectionsPanel.tsx",
  "components/PersonaGarden/TestLabPanel.tsx",
  "components/PersonaGarden/VisualPackEditor.tsx",
  "components/Quiz/tabs/TakeQuizTab.tsx",
  "components/Review/reading-pane-section-navigation.ts",
  "hooks/document-workspace/usePdfSearch.ts",
  "routes/hooks/usePersonaGovernanceContext.tsx"
])

const LITERAL_SMOOTH = /behavior:\s*["']smooth["']/

const srcRoot = path.resolve(__dirname, "../..")

function collect(dir: string, out: string[]) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "__tests__" || entry.name === "node_modules") continue
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) collect(full, out)
    else if (/\.(tsx?|jsx?)$/.test(entry.name) && !/\.(test|spec)\./.test(entry.name)) out.push(full)
  }
}

describe("scripted smooth scrolling", () => {
  it("asks smoothScrollBehavior() instead of passing a literal smooth behaviour", () => {
    const files: string[] = []
    collect(srcRoot, files)
    const offenders = files
      .map((file) => path.relative(srcRoot, file).split(path.sep).join("/"))
      .filter((relative) => !REMAINING.has(relative))
      .filter((relative) => LITERAL_SMOOTH.test(fs.readFileSync(path.join(srcRoot, relative), "utf8")))
    expect(offenders).toEqual([])
  })

  it("keeps the remaining list current", () => {
    const stale = [...REMAINING].filter((relative) => {
      const file = path.join(srcRoot, relative)
      return !fs.existsSync(file) || !LITERAL_SMOOTH.test(fs.readFileSync(file, "utf8"))
    })
    expect(stale).toEqual([])
  })
})
