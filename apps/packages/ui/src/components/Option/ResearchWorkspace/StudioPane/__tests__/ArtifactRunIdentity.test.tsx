import React from "react"
import { render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it } from "vitest"
import type { GeneratedArtifact } from "@/types/workspace"
import { ArtifactRunIdentity, LongDocumentWaitingGuidance } from "../ArtifactRunIdentity"
import { getArtifactRunIdentity } from "../output-identity"

const selection = (node: string) => ({ source_id: "source", node_id: node, navigation_version: "v1" })
const section = (node: string, title: string, status = "completed") => ({ ...selection(node), title, status })
const artifact = (overrides: Partial<GeneratedArtifact> = {}): GeneratedArtifact => ({
  id: "output", type: "summary", title: "Summary", status: "completed",
  createdAt: new Date("2026-11-16T10:15:00Z"),
  producerMetadata: { origin: "research_workspace_output_job", settings: {
    model: "gpt-6-luna", provider: "openai", sections: [selection("one"), selection("two")]
  } },
  data: { sectionAnalysis: { schemaVersion: 1, sections: [
    section("one", "First chapter"), section("two", "Second chapter", "failed")
  ] } },
  ...overrides
})

const i18n = createInstance()
beforeEach(async () => {
  await i18n.init({ lng: "en", fallbackLng: "en", resources: { en: { playground: {} } }, interpolation: { escapeValue: false } })
})
const show = (element: React.ReactNode) => render(<I18nextProvider i18n={i18n}>{element}</I18nextProvider>)

describe("saved output identity", () => {
  it("uses saved Jobs settings, counts selected rather than completed sections, and leaves the artifact unchanged", () => {
    const saved = artifact()
    const before = structuredClone(saved)
    const identity = getArtifactRunIdentity(saved)
    expect(identity).toMatchObject({ model: "gpt-6-luna", provider: "openai", sectionCount: 2,
      sectionTitles: ["First chapter", "Second chapter"], imported: false })
    expect(identity.createdAt?.toISOString()).toBe("2026-11-16T10:15:00.000Z")
    expect(saved).toEqual(before)
  })

  it("does not fill missing Jobs settings from an unrelated producer model", () => {
    expect(getArtifactRunIdentity(artifact({ producerMetadata: {
      origin: "research_workspace_output_job", model: "unrelated", provider: "other", settings: { sections: [] }
    }, data: undefined }))).toMatchObject({ model: undefined, provider: undefined, sectionCount: undefined })
  })

  it("reads direct saved model/provider metadata from other producers", () => {
    expect(getArtifactRunIdentity(artifact({ producerMetadata: { model: "saved-model", provider: "saved-provider" } })))
      .toMatchObject({ model: "saved-model", provider: "saved-provider" })
  })

  it("preserves original generation identity through repeated snapshot imports", () => {
    const original = artifact()
    const imported = artifact({ producerMetadata: { origin: "workspace_snapshot_import", importedProducer: {
      origin: "workspace_snapshot_import", importedProducer: original.producerMetadata
    } } })
    expect(getArtifactRunIdentity(imported)).toMatchObject({ imported: true, model: "gpt-6-luna", provider: "openai", sectionCount: 2 })
  })

  it("reads complete-pack generation settings without changing the saved artifact or date", () => {
    const saved = artifact({ producerMetadata: { origin: "research_pack_import", importedProducer: artifact().producerMetadata } })
    const before = structuredClone(saved)
    const identity = getArtifactRunIdentity(saved)
    expect(identity).toMatchObject({ imported: true, model: "gpt-6-luna", provider: "openai", sectionCount: 2 })
    identity.createdAt?.setUTCFullYear(2000)
    expect(saved).toEqual(before)
  })

  it("retains generation settings through mixed complete-pack and snapshot imports", () => {
    const saved = artifact({ producerMetadata: { origin: "research_pack_import", importedProducer: {
      origin: "workspace_snapshot_import", importedProducer: {
        origin: "research_pack_import", importedProducer: artifact().producerMetadata
      }
    } } })
    expect(getArtifactRunIdentity(saved)).toMatchObject({ imported: true, model: "gpt-6-luna", provider: "openai", sectionCount: 2 })
  })

  it.each(["workspace_snapshot_import", "research_pack_import"])("fails closed for cyclic or missing %s metadata", origin => {
    const cyclic: Record<string, unknown> = { origin }
    cyclic.importedProducer = cyclic
    for (const producerMetadata of [cyclic, { origin }]) {
      expect(getArtifactRunIdentity(artifact({ producerMetadata, data: undefined })))
        .toMatchObject({ imported: true, model: undefined, provider: undefined, sectionCount: undefined })
    }
  })

  it("bounds mixed import traversal at five wrappers", () => {
    let producerMetadata = artifact().producerMetadata
    for (let index = 0; index < 5; index++) producerMetadata = {
      origin: index % 2 ? "workspace_snapshot_import" : "research_pack_import", importedProducer: producerMetadata
    }
    expect(getArtifactRunIdentity(artifact({ producerMetadata }))).toMatchObject({ imported: true, model: "gpt-6-luna" })
    expect(getArtifactRunIdentity(artifact({ producerMetadata: {
      origin: "research_pack_import", importedProducer: producerMetadata
    }, data: undefined }))).toMatchObject({ imported: true, model: undefined, provider: undefined, sectionCount: undefined })
  })

  it("does not treat an unknown wrapper as verified imported generation metadata", () => {
    expect(getArtifactRunIdentity(artifact({ producerMetadata: {
      origin: "unknown_import", importedProducer: artifact().producerMetadata
    }, data: undefined }))).toMatchObject({ imported: false, model: undefined, provider: undefined, sectionCount: undefined })
  })

  it("reads canonical saved section rows when producer settings are unavailable", () => {
    expect(getArtifactRunIdentity(artifact({ producerMetadata: undefined, data: undefined,
      versionMetadata: { workspaceClient: { schemaVersion: 1, data: { sectionAnalysis: {
        schemaVersion: 1, sections: [section("exim", "Export–Import Bank")]
      } } } }
    }))).toMatchObject({ sectionCount: 1, sectionTitles: ["Export–Import Bank"] })
  })

  it("uses only titles matching the saved selection and bounds their preview", () => {
    const rows = [1, 2, 3, 4].map(n => section(String(n), `Chapter ${n}`))
    const saved = artifact({ producerMetadata: { origin: "research_workspace_output_job", settings: {
      sections: rows.map(row => selection(row.node_id))
    } }, data: { sectionAnalysis: { schemaVersion: 1, sections: [section("other", "Unselected"), ...rows] } } })
    expect(getArtifactRunIdentity(saved)).toMatchObject({ sectionCount: 4, sectionTitles: ["Chapter 1", "Chapter 2", "Chapter 3"] })
  })

  it.each([null, [{ node_id: "missing-source" }], [selection("one"), selection("one")]])(
    "does not invent a section scope from malformed saved selections: %j", sections => {
      expect(getArtifactRunIdentity(artifact({ producerMetadata: { origin: "research_workspace_output_job", settings: { sections } },
        data: undefined })).sectionCount).toBeUndefined()
    }
  )

  it("accepts saved serialized dates but never invents an unavailable creation time", () => {
    expect(getArtifactRunIdentity(artifact({ createdAt: "2026-11-16T10:15:00Z" as unknown as Date })).createdAt?.toISOString())
      .toBe("2026-11-16T10:15:00.000Z")
    for (const value of [undefined, null, "", "invalid", "0", new Date(NaN), 0]) {
      expect(getArtifactRunIdentity(artifact({ createdAt: value as Date })).createdAt).toBeUndefined()
    }
  })

  it("renders recorded model, time and scope with an explicit imported label", () => {
    const saved = artifact()
    const { container } = show(<ArtifactRunIdentity artifact={artifact({ producerMetadata: {
      origin: "workspace_snapshot_import", importedProducer: saved.producerMetadata
    } })} />)
    expect(screen.getByText(/gpt-6-luna/)).toHaveTextContent("openai")
    expect(screen.getByText("Imported output")).toBeInTheDocument()
    expect(screen.getByText(/2 selected sections/)).toHaveTextContent("First chapter")
    expect(container.querySelector("time")).toHaveAttribute("datetime", "2026-11-16T10:15:00.000Z")
    expect(container.querySelector("time")?.parentElement).toHaveTextContent(/^Imported /)
    expect(container.querySelector("time")?.parentElement).not.toHaveTextContent("Created")
  })

  it("labels the complete-pack timestamp as import time while retaining its recorded model", () => {
    const { container } = show(<ArtifactRunIdentity artifact={artifact({ producerMetadata: {
      origin: "research_pack_import", importedProducer: artifact().producerMetadata
    } })} />)
    expect(screen.getByText(/gpt-6-luna/)).toHaveTextContent("openai")
    expect(screen.getByText("Imported output")).toBeInTheDocument()
    expect(container.querySelector("time")?.parentElement).toHaveTextContent(/^Imported /)
  })

  it("does not invent an import time when the canonical timestamp is unavailable", () => {
    show(<ArtifactRunIdentity artifact={artifact({ producerMetadata: {
      origin: "research_pack_import", importedProducer: artifact().producerMetadata
    }, createdAt: undefined as unknown as Date })} />)
    expect(screen.getByText("Import time not recorded")).toBeInTheDocument()
    expect(screen.queryByText("Creation time not recorded")).not.toBeInTheDocument()
  })

  it("renders honest missing metadata and escapes saved text", () => {
    const { container } = show(<ArtifactRunIdentity artifact={artifact({ producerMetadata: { model: "<script>saved</script>" },
      data: undefined, createdAt: undefined as unknown as Date })} />)
    expect(screen.getByText(/<script>saved<\/script>/)).toBeInTheDocument()
    expect(container.querySelector("script")).toBeNull()
    expect(screen.getByText(/Provider not recorded/)).toBeInTheDocument()
    expect(screen.getByText(/Creation time not recorded/)).toBeInTheDocument()
    expect(screen.getByText(/Section scope not recorded/)).toBeInTheDocument()
  })

  it("renders the singular selected-section label", () => {
    show(<ArtifactRunIdentity artifact={artifact({ producerMetadata: undefined,
      data: { sectionAnalysis: { schemaVersion: 1, sections: [section("one", "First chapter")] } } })} />)
    expect(screen.getByText(/1 selected section:/)).toBeInTheDocument()
  })
})

describe("long-document waiting guidance", () => {
  it("qualifies the measured example without presenting an ETA or cost estimate", () => {
    show(<LongDocumentWaitingGuidance />)
    expect(screen.getByText(/922-page.*30-chapter.*52–62 minutes/)).toHaveTextContent("In observed 922-page, 30-chapter runs before citation checks were added, generation took 52–62 minutes.")
    expect(screen.getByText(/not an estimate for this run/)).toHaveTextContent("model")
    expect(screen.getByText(/Progress is saved after each portion/)).toBeInTheDocument()
    expect(screen.getByText(/Read completed chapters while the remaining sections/)).toBeInTheDocument()
    expect(screen.getByText(/Token usage and cost are unavailable/)).toBeInTheDocument()
    expect(screen.queryByText(/\$|minutes remaining/)).not.toBeInTheDocument()
  })
})
