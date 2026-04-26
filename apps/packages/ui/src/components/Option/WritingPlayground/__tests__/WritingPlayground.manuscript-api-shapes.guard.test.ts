import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readWritingPlaygroundSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", filename), "utf8")

const readWritingPlaygroundModalSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", "modals", filename), "utf8")

const readWritingPlaygroundRootSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", "..", "..", "..", "services", filename), "utf8")

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
