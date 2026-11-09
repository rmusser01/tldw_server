import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { useDocumentWorkspaceStore as store } from "../document-workspace"
import type { Annotation } from "@/components/DocumentWorkspace/types"

const saved = (documentId: number): Annotation => ({
  id: `saved-${documentId}`,
  documentId,
  location: 1,
  text: "Saved",
  color: "yellow",
  createdAt: new Date(0),
  updatedAt: new Date(0)
})
beforeEach(() => store.getState().reset())
afterEach(() => store.getState().reset())

describe("document annotation reconciliation", () => {
  it("keeps a new pending annotation when a canonical GET finishes", () => {
    store.setState({ activeDocumentId: 1 })
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "New local note",
        color: "blue"
      })
    const pending = store.getState().pendingAnnotations
    store.getState().setAnnotations([saved(1)])
    expect({
      pending: store.getState().pendingAnnotations,
      texts: store.getState().annotations.map((a) => a.text)
    }).toEqual({ pending, texts: ["Saved", "New local note"] })
  })
  it("retains another document's pending creation during a read", () => {
    store
      .getState()
      .addAnnotation({
        documentId: 2,
        location: 1,
        text: "Other document",
        color: "blue"
      })
    const pending = store.getState().pendingAnnotations
    store.setState({ activeDocumentId: 1 })
    store.getState().setAnnotations([saved(1)])
    expect(store.getState().pendingAnnotations).toEqual(pending)
  })
  it("never treats a general status change as an acknowledgement", () => {
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "Unacknowledged",
        color: "blue"
      })
    const pending = store.getState().pendingAnnotations
    store.getState().setAnnotationSyncStatus("synced")
    expect(store.getState().pendingAnnotations).toEqual(pending)
  })
  it("retires a removed local creation before it can be sent", () => {
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "Remove before save",
        color: "blue"
      })
    const id = store.getState().pendingAnnotations[0].id
    store.getState().removeAnnotation(id)
    expect(store.getState().pendingAnnotations).toEqual([])
  })

  it("merges a GET-before-ACK without duplicating the committed creation", () => {
    store.setState({ activeDocumentId: 1 })
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 1,
        text: "First",
        color: "blue"
      })
    const submitted = store.getState().pendingAnnotations[0]
    store
      .getState()
      .addAnnotation({
        documentId: 1,
        location: 2,
        text: "Later",
        color: "blue"
      })
    const later = store.getState().pendingAnnotations[1]
    const saved = { ...submitted, id: "canonical-first" }
    store.getState().setAnnotations([saved], 1)
    store
      .getState()
      .acknowledgeAnnotations([submitted], [saved], {
        [submitted.id]: saved.id
      })
    expect(store.getState().annotations.map((a) => a.id)).toEqual([
      saved.id,
      later.id
    ])
    expect(store.getState().pendingAnnotations).toEqual([later])
  })
})
