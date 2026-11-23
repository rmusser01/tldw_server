import { beforeEach, describe, expect, it } from "vitest"
import { useWorkspaceStore } from "../workspace"
import { serverWorkspaceMetadata } from "./workspace-activation.fixtures"
import { retainKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"
import type { WorkspaceNote } from "@/types/workspace"

const history = { origin: "reviewed_sources" as const, question: "Retained question" }
const capturedHistory = { ...history, question: "New capture" }

const note = (): WorkspaceNote => ({
  id: "12345678-1234-4123-8123-123456789abc",
  title: "Retained note",
  content: retainKnowledgeNoteProvenance("Retained body", history),
  keywords: ["retained"],
  version: 40,
  knowledge_provenance_state: "active",
  knowledge_provenance_version: 3,
  knowledge_provenance: history,
  isDirty: false
})

describe("canonical capture guard", () => {
  beforeEach(() => {
    localStorage.clear()
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.getState().initializeWorkspace("Local capture")
  })

  it.each(["append", "replace"] as const)("refuses %s through every canonical binding", mode => {
    for (const binding of ["workspace", "note"] as const) {
      for (const state of ["active", "deleted", "unsupported"] as const) {
        const currentNote = {
          ...note(), knowledge_provenance_state: state,
          ...(binding === "note" ? { serverWorkspaceId: "server-research", serverScopeKey: "owner-a" } : {})
        }
        useWorkspaceStore.setState({
          currentNote,
          serverWorkspace: binding === "workspace" ? {
            scopeKey: "owner-a", sourceSignature: "", selectedSourceSignature: "",
            metadata: serverWorkspaceMetadata, notes: []
          } : null
        })
        const before = useWorkspaceStore.getState()
        before.captureToCurrentNote({ title: "New title", content: "New body", mode, provenance: capturedHistory })
        expect(useWorkspaceStore.getState()).toBe(before)
        expect(useWorkspaceStore.getState().currentNote).toBe(currentNote)
        expect(currentNote).toEqual({ ...note(), knowledge_provenance_state: state,
          ...(binding === "note" ? { serverWorkspaceId: "server-research", serverScopeKey: "owner-a" } : {}) })
      }
    }
  })

  it.each(["append", "replace"] as const)("retains ordinary local %s capture and pending history", mode => {
    useWorkspaceStore.getState().setCurrentNote(note())
    useWorkspaceStore.getState().captureToCurrentNote({ title: "New title", content: "New body", mode, provenance: capturedHistory })
    const after = useWorkspaceStore.getState().currentNote
    expect(after.content).toContain("New body")
    expect(after.content.includes("Retained body")).toBe(mode === "append")
    expect(after).toMatchObject({ id: note().id, title: "Retained note", version: 40,
      knowledge_provenance: history, pendingKnowledgeProvenance: capturedHistory, isDirty: true })
  })

  it("preserves unbound draft seeding for the captured-owner Knowledge import", () => {
    useWorkspaceStore.setState({ serverWorkspace: {
      scopeKey: "owner-a", sourceSignature: "", selectedSourceSignature: "",
      metadata: serverWorkspaceMetadata, notes: []
    } })
    for (const body of ["", "Retained unsent local body"]) {
      useWorkspaceStore.getState().setCurrentNote({ title: "", content: body, keywords: [], isDirty: Boolean(body) })
      useWorkspaceStore.getState().captureToCurrentNote({ title: "Knowledge QA", content: "Import reference: owned-import", mode: "append" })
      const after = useWorkspaceStore.getState().currentNote
      expect(after.id).toBeUndefined()
      expect(after.serverWorkspaceId).toBeUndefined()
      expect(after.content).toContain("Import reference: owned-import")
      if (body) expect(after.content).toContain(body)
      expect(after.isDirty).toBe(true)
    }
  })
})
