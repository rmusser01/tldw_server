import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readWritingPlaygroundSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", filename), "utf8")

const readWritingPlaygroundModalSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", "modals", filename), "utf8")

const readWritingPlaygroundRootSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", "..", "..", "..", "services", filename), "utf8")

const extractExportTypeSource = (source: string, typeName: string) => {
  const start = source.indexOf(`export type ${typeName} =`)
  if (start < 0) return ""
  const candidates = [
    "\n\nexport type ",
    "\n\nexport const ",
    "\n\nexport async function ",
    "\n\nexport function ",
    "\n\nconst ",
  ]
    .map((marker) => source.indexOf(marker, start + 1))
    .filter((index) => index > start)
  const end = candidates.length ? Math.min(...candidates) : source.length
  return source.slice(start, end)
}

const extractExportFunctionSignature = (source: string, functionName: string) => {
  const start = source.indexOf(`export async function ${functionName}(`)
  if (start < 0) return ""
  const returnTypeStart = source.indexOf("): Promise<", start)
  if (returnTypeStart < 0) return ""
  const bodyStart = source.indexOf(" {", returnTypeStart)
  if (bodyStart < 0) return ""
  return source.slice(start, bodyStart)
}

describe("Writing playground manuscript API shape guards", () => {
  it("uses typed wrapper response fields for agent and connection-web manuscript lookups", () => {
    const aiAgentSource = readWritingPlaygroundSource("AIAgentTab.tsx")
    const connectionWebSource = readWritingPlaygroundModalSource("ConnectionWebModal.tsx")

    expect(aiAgentSource).toContain("const chars = charsResp.characters || []")
    expect(aiAgentSource).toContain("const items = worldResp.items || []")
    expect(connectionWebSource).toContain("const characters = charsData?.characters || []")
    expect(connectionWebSource).toContain("const relationships = relsData?.relationships || []")
    expect(connectionWebSource).toContain("const worldItems = worldData?.items || []")
    expect(connectionWebSource).not.toContain("charsData as any")
    expect(connectionWebSource).not.toContain("relsData as any")
    expect(connectionWebSource).not.toContain("worldData as any")
  })

  it("avoids manuscript any-casts in character and research tabs", () => {
    const characterWorldSource = readWritingPlaygroundSource("CharacterWorldTab.tsx")
    const researchSource = readWritingPlaygroundSource("ResearchTab.tsx")

    expect(characterWorldSource).not.toContain("as any[]")
    expect(researchSource).not.toContain("resp as any")
    expect(researchSource).toContain("resp.results")
  })

  it("types manuscript service return values with explicit wrapper contracts", () => {
    const serviceSource = readWritingPlaygroundRootSource("writing-playground.ts")

    expect(serviceSource).toContain("export type ManuscriptCharacterResponse")
    expect(serviceSource).toContain("export type ManuscriptCharacterListResponse")
    expect(serviceSource).toContain("export type ManuscriptWorldInfoListResponse")
    expect(serviceSource).toContain("export type ManuscriptRelationshipListResponse")
    expect(serviceSource).toContain("): Promise<ManuscriptCharacterListResponse>")
    expect(serviceSource).toContain("): Promise<ManuscriptWorldInfoListResponse>")
    expect(serviceSource).toContain("): Promise<ManuscriptRelationshipListResponse>")
    expect(serviceSource).toContain("): Promise<ManuscriptResearchResponse>")
  })

  it("types manuscript annotation service contracts and review provider payloads explicitly", () => {
    const serviceSource = readWritingPlaygroundRootSource("writing-playground.ts")
    const responseTypeSource = extractExportTypeSource(serviceSource, "ManuscriptAnnotationResponse")
    const listTypeSource = extractExportTypeSource(serviceSource, "ManuscriptAnnotationListResponse")
    const createTypeSource = extractExportTypeSource(serviceSource, "ManuscriptAnnotationCreateInput")
    const updateTypeSource = extractExportTypeSource(serviceSource, "ManuscriptAnnotationUpdateInput")
    const selectedReviewTypeSource = extractExportTypeSource(
      serviceSource,
      "ManuscriptSelectedTextAnnotationReviewRequest"
    )
    const sceneReviewTypeSource = extractExportTypeSource(
      serviceSource,
      "ManuscriptSceneAnnotationReviewRequest"
    )
    const jobTypeSource = extractExportTypeSource(
      serviceSource,
      "ManuscriptSceneAnnotationReviewJobResponse"
    )
    const listAnnotationsSignature = extractExportFunctionSignature(
      serviceSource,
      "listManuscriptAnnotations"
    )
    const createAnnotationSignature = extractExportFunctionSignature(
      serviceSource,
      "createManuscriptAnnotation"
    )
    const getAnnotationSignature = extractExportFunctionSignature(
      serviceSource,
      "getManuscriptAnnotation"
    )
    const updateAnnotationSignature = extractExportFunctionSignature(
      serviceSource,
      "updateManuscriptAnnotation"
    )
    const reviewSelectionSignature = extractExportFunctionSignature(
      serviceSource,
      "reviewManuscriptSelection"
    )
    const reviewSceneSignature = extractExportFunctionSignature(
      serviceSource,
      "reviewManuscriptScene"
    )

    expect(responseTypeSource).toContain("export type ManuscriptAnnotationResponse")
    expect(listTypeSource).toContain("export type ManuscriptAnnotationListResponse")
    expect(createTypeSource).toContain("export type ManuscriptAnnotationCreateInput")
    expect(updateTypeSource).toContain("export type ManuscriptAnnotationUpdateInput")
    expect(selectedReviewTypeSource).toContain("export type ManuscriptSelectedTextAnnotationReviewRequest")
    expect(sceneReviewTypeSource).toContain("export type ManuscriptSceneAnnotationReviewRequest")
    expect(jobTypeSource).toContain("export type ManuscriptSceneAnnotationReviewJobResponse")
    expect(responseTypeSource).not.toContain("Record<string, unknown>")
    expect(listTypeSource).not.toContain("Record<string, unknown>")
    expect(createTypeSource).not.toContain("Record<string, unknown>")
    expect(updateTypeSource).not.toContain("Record<string, unknown>")
    expect(selectedReviewTypeSource).toContain("provider: string")
    expect(selectedReviewTypeSource).toContain("model: string")
    expect(sceneReviewTypeSource).toContain("provider: string")
    expect(sceneReviewTypeSource).toContain("model: string")
    expect(selectedReviewTypeSource).not.toContain("api_provider")
    expect(selectedReviewTypeSource).not.toContain("apiProvider")
    expect(selectedReviewTypeSource).not.toContain("llm_provider")
    expect(selectedReviewTypeSource).not.toContain("llmProvider")
    expect(sceneReviewTypeSource).not.toContain("api_provider")
    expect(sceneReviewTypeSource).not.toContain("apiProvider")
    expect(sceneReviewTypeSource).not.toContain("llm_provider")
    expect(sceneReviewTypeSource).not.toContain("llmProvider")
    expect(updateTypeSource).toContain("status?: ManuscriptAnnotationStatus\n")
    expect(updateTypeSource).toContain("category?: ManuscriptAnnotationCategory\n")
    expect(updateTypeSource).toContain("body?: string\n")
    expect(updateTypeSource).not.toContain("status?: ManuscriptAnnotationStatus | null")
    expect(updateTypeSource).not.toContain("category?: ManuscriptAnnotationCategory | null")
    expect(updateTypeSource).not.toContain("body?: string | null")
    expect(listAnnotationsSignature).toContain("): Promise<ManuscriptAnnotationListResponse>")
    expect(createAnnotationSignature).toContain("): Promise<ManuscriptAnnotationResponse>")
    expect(getAnnotationSignature).toContain("): Promise<ManuscriptAnnotationResponse>")
    expect(updateAnnotationSignature).toContain("): Promise<ManuscriptAnnotationResponse>")
    expect(reviewSelectionSignature).toContain("): Promise<ManuscriptAnnotationResponse>")
    expect(reviewSceneSignature).toContain("): Promise<ManuscriptSceneAnnotationReviewJobResponse>")
  })

  it("exports manuscript scene and research response types only once", () => {
    const serviceSource = readWritingPlaygroundRootSource("writing-playground.ts")

    expect(serviceSource.match(/export type ManuscriptSceneResponse =/g)?.length ?? 0).toBe(1)
    expect(serviceSource.match(/export type ManuscriptResearchResponse =/g)?.length ?? 0).toBe(1)
  })

  it("reuses the shared mood color mapping in the status bar", () => {
    const indexSource = readWritingPlaygroundSource("index.tsx")

    expect(indexSource).toContain("MOOD_COLORS[feedback.currentMood]")
    expect(indexSource).not.toContain('color={{tense:"#ff4d4f"')
  })
})
