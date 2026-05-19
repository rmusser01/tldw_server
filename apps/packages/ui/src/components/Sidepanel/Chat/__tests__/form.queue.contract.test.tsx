import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const resolveSidepanelFormPath = () => {
  const candidates = [
    path.resolve(__dirname, "../form.tsx"),
    path.resolve(process.cwd(), "src/components/Sidepanel/Chat/form.tsx"),
    path.resolve(
      process.cwd(),
      "../packages/ui/src/components/Sidepanel/Chat/form.tsx"
    ),
    path.resolve(
      process.cwd(),
      "apps/packages/ui/src/components/Sidepanel/Chat/form.tsx"
    )
  ]

  const resolved = candidates.find((candidate) => existsSync(candidate))
  if (!resolved) {
    throw new Error("Unable to locate Sidepanel chat form source")
  }

  return resolved
}

const sidepanelFormSource = readFileSync(resolveSidepanelFormPath(), "utf8")

const resolveControlRowPath = () => {
  const candidates = [
    path.resolve(__dirname, "../ControlRow.tsx"),
    path.resolve(process.cwd(), "src/components/Sidepanel/Chat/ControlRow.tsx"),
    path.resolve(
      process.cwd(),
      "../packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx"
    ),
    path.resolve(
      process.cwd(),
      "apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx"
    )
  ]

  const resolved = candidates.find((candidate) => existsSync(candidate))
  if (!resolved) {
    throw new Error("Unable to locate Sidepanel chat ControlRow source")
  }

  return resolved
}

const controlRowSource = readFileSync(resolveControlRowPath(), "utf8")

describe("sidepanel queued request contract", () => {
  it("uses the shared queued request panel and orchestration hook", () => {
    expect(sidepanelFormSource).toContain('from "@/components/Common/ChatQueuePanel"')
    // Sidepanel now consumes the shared composer queue primitive, which
    // wraps the lower-level `useQueuedRequests` internally (see
    // Chat/composer/hooks/useComposerQueue.ts).
    expect(sidepanelFormSource).toContain(
      'from "@/components/Chat/composer/hooks/useComposerQueue"'
    )
    expect(sidepanelFormSource).not.toContain("QueuedMessagesBanner")
  })

  it("treats busy and disconnected submits as queue actions", () => {
    expect(sidepanelFormSource).toContain(
      "const shouldQueueInsteadOfSend = isSending || !isConnectionReady"
    )
    expect(sidepanelFormSource).toContain('t("common:queue", "Queue")')
    expect(sidepanelFormSource).toContain(
      't("playground:composer.queue.primaryAria", "Queue request")'
    )
  })

  it("disables destructive cancel-and-run for server-backed turns", () => {
    expect(sidepanelFormSource).toContain(
      "playground:composer.queue.cancelAndRunDisabled"
    )
    expect(sidepanelFormSource).toContain("forceRunDisabledReason")
  })

  it("passes saved request settings into queued replay dispatches", () => {
    expect(sidepanelFormSource).toContain("requestOverrides:")
    expect(sidepanelFormSource).toContain("chatMode:")
    expect(sidepanelFormSource).toContain("selectedSystemPrompt:")
    expect(sidepanelFormSource).toContain("toolChoice:")
    expect(sidepanelFormSource).toContain("webSearch:")
    expect(sidepanelFormSource).toContain("useOCR:")
  })

  it("clears persisted drafts after send/queue success and resets file inputs with the DOM-event handler", () => {
    expect(sidepanelFormSource).toContain("const { form, draftSaved, clearDraft } = useComposerText")
    expect(sidepanelFormSource).toContain("clearDraft()")
    expect(sidepanelFormSource).toContain("onChange={attachmentHandler.onFileInputChange}")
  })

  it("passes extension playlist handoff detail into the Quick Ingest session", () => {
    expect(sidepanelFormSource).toContain(
      "createQuickIngestSessionSeedFromOpenDetail(detail)"
    )
    expect(sidepanelFormSource).toContain("upsertQuickIngestSession(seed)")
    expect(controlRowSource).toContain("buildQuickIngestOpenDetailFromUrl")
    expect(controlRowSource).toContain("Import playlist to tldw")
  })
})
