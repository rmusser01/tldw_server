import { describe, expect, it } from "vitest"
import type { GeneratedArtifact } from "@/types/workspace"
import { getSectionArtifactErrorMessage, INCOMPLETE_SECTIONS_MESSAGE } from "../section-artifact-status"

describe("overview recovery warnings", () => {
  it.each([
    ["overview_sections_missing", "The overview omits a cited point from Financial. Retry the overview from saved chapters."],
    ["overview_input_uncited", "Financial has no usable citation in its saved chapter summary. Generate a new output; overview retry cannot repair it."],
    ["overview_coverage_too_large", "The selected chapter checklist exceeds the overview context limit. Generate an overview for fewer sections; saved chapters remain available."]
  ])("preserves the specific %s recovery guidance after restore", (code, message) => {
    const artifact: GeneratedArtifact = {
      id: "restored", type: "summary", title: "Chapters", status: "failed", createdAt: new Date(0),
      errorMessage: INCOMPLETE_SECTIONS_MESSAGE,
      versionMetadata: { workspaceClient: { schemaVersion: 1, data: { sectionAnalysis: {
        schemaVersion: 1, synthesis: "Retained overview [E1.1.1].", synthesis_error_code: code,
        synthesis_error: message, sections: [{ title: "Financial", status: "completed" }]
      } } } }
    }
    const original = JSON.stringify(artifact)
    expect(getSectionArtifactErrorMessage(artifact)).toBe(message)
    expect(JSON.stringify(artifact)).toBe(original)
  })
})
