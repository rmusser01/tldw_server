import { describe, expect, it } from "vitest"
import type { Message } from "@/store/option"
import { excludeLocalRagDiagnostics, getLocalRagDiagnosticUser, isLocalRagDiagnosticInfo } from "../local-rag-diagnostic"

const info = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
const pair = (): Message[] => [
  { id: "user", isBot: false, name: "You", message: "Same question", sources: [], generationInfo: info },
  { id: "diagnostic", isBot: true, name: "Assistant", message: "Same prose", sources: [], generationInfo: info, parentMessageId: "user" }
]

describe("local selected-source diagnostic eligibility", () => {
  it.each([null, {}, { reason: info.reason }, { ...info, mode: "normal" }, { ...info, grounded: true }, { ...info, reason: "other" }])("does not classify other metadata: %j", value => {
    expect(isLocalRagDiagnosticInfo(value)).toBe(false)
  })

  it("omits only the exact local pair, retaining equal ordinary prose, images and a detached draft", () => {
    const diagnostic = pair()
    const ordinary = diagnostic.map(row => ({ ...row, id: `ordinary-${row.id}`, parentMessageId: "ordinary-user", generationInfo: undefined, images: ["data:image/png;base64,keep"] }))
    const draft = { ...diagnostic[0], id: "draft" }
    expect(excludeLocalRagDiagnostics([...diagnostic, ...ordinary, draft])).toEqual([...ordinary, draft])
  })

  it("retains canonical rows even when their metadata happens to match", () => {
    const canonical = pair().map(row => ({ ...row, serverMessageId: `server-${row.id}` }))
    expect(excludeLocalRagDiagnostics(canonical)).toEqual(canonical)
  })

  it("retains the acknowledged user while a separate local diagnostic stays out of context", () => {
    const rows = pair()
    rows[0].serverMessageId = "server-user"
    expect(excludeLocalRagDiagnostics(rows)).toEqual([rows[0]])
  })

  it("does not infer an undispatched user from ambiguous IDs or missing parent provenance", () => {
    const rows = pair()
    expect(getLocalRagDiagnosticUser([...rows, { ...rows[0] }], rows[1])).toBeUndefined()
    expect(getLocalRagDiagnosticUser([{ ...rows[0], generationInfo: undefined }, rows[1]], rows[1])).toBeUndefined()
  })

  it("restores a temporary user's eligibility after a real response without an ACK", () => {
    const rows = pair()
    rows[1] = { ...rows[1], id: "real-answer", message: "Real response", generationInfo: undefined,
      variants: [{ id: "diagnostic", message: "Same prose", generationInfo: info }, { id: "real-answer", message: "Real response" }] }
    expect(excludeLocalRagDiagnostics(rows)).toEqual(rows)
  })

  it("does not suppress an already answered user when an older diagnostic variant is selected", () => {
    const rows = pair()
    rows[1].variants = [{ id: "diagnostic", message: "Same prose", generationInfo: info }, { id: "real-answer", message: "Real response" }]
    expect(excludeLocalRagDiagnostics(rows)).toEqual([rows[0]])
  })

  it("resolves repeated failure variants to the original user, without text matching", () => {
    const rows = pair()
    rows[1].variants = [{ id: "first", message: "Different diagnostic", generationInfo: info }, { id: "second", message: "Same prose", generationInfo: info }]
    expect(getLocalRagDiagnosticUser(rows, rows[1])).toBe(rows[0])
  })
})
