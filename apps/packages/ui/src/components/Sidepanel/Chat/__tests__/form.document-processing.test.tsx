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

describe("sidepanel document processing contract", () => {
  it("uses the shared document processing choice surface for context files", () => {
    expect(sidepanelFormSource).toContain(
      'from "@/components/Option/Playground/DocumentProcessingChoices"'
    )
    expect(sidepanelFormSource).toContain("<DocumentProcessingChoices")
    expect(sidepanelFormSource).toContain("files={contextFiles}")
    expect(sidepanelFormSource).toContain(
      "onChangeFiles={setContextFiles}"
    )
  })

  it("tags sidepanel document uploads with the shared default decision", () => {
    expect(sidepanelFormSource).toContain("withDefaultDocumentDecision")
  })

  it("uses backend preflight to resolve sidepanel document capabilities", () => {
    expect(sidepanelFormSource).toContain("normalizeDocumentPreflightResponse")
    expect(sidepanelFormSource).toContain("preflightDocumentUpload")
  })

  it("creates a server-backed draft when continuing attached documents in WebUI", () => {
    expect(sidepanelFormSource).toContain("openChatInWebUiWithDocumentHandoff")
    expect(sidepanelFormSource).toContain("createDocumentUploadDraft")
    expect(sidepanelFormSource).toContain("chatDocumentDraftId")
  })

  it("prepares document attachments on send before dispatching chat", () => {
    expect(sidepanelFormSource).toContain(
      "prepareChatDocumentAttachmentsForSend"
    )
    expect(sidepanelFormSource).toContain("preparedDocumentAttachments")
    expect(sidepanelFormSource).toContain("documentUploadedFiles")
    expect(sidepanelFormSource).toContain("documentRequestOverrides")
    expect(sidepanelFormSource).toContain(
      "uploadedFiles: intent.isImageCommand ? [] : documentUploadedFiles"
    )
  })
})
