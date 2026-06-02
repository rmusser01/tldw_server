import { describe, expect, it } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(testDir, "../..")
const mockOpenAiRoot = path.join(frontendRoot, "e2e/onboarding-uat/mock-openai")

const configFiles = [
  "hosted-success.json",
  "local-success.json",
  "chat-fail-once.json",
  "model-unavailable.json",
]

const responseFiles = [
  "chat/default.json",
  "chat/source-summary.json",
  "embeddings/default.json",
]

const secretMarkers = [
  "sk-",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "ghp_",
  "xoxb-",
  "AKIA",
  "BEGIN PRIVATE KEY",
]

function readJson(relativePath: string) {
  return JSON.parse(readFileSync(path.join(mockOpenAiRoot, relativePath), "utf8"))
}

function readText(relativePath: string) {
  return readFileSync(path.join(frontendRoot, relativePath), "utf8")
}

function modelIds(config: { models?: Array<{ id?: string }> }) {
  return (config.models ?? []).map((model) => model.id)
}

describe("onboarding UAT static fixtures", () => {
  it("provides mock OpenAI config files with the expected scenario shapes", () => {
    const configs = Object.fromEntries(
      configFiles.map((file) => [file, readJson(`configs/${file}`)])
    )

    expect(modelIds(configs["hosted-success.json"])).toEqual(
      expect.arrayContaining(["gpt-4.1-mini", "text-embedding-3-small"])
    )

    expect(modelIds(configs["local-success.json"])).toEqual(
      expect.arrayContaining(["llama3.2:3b", "local-uat-chat"])
    )

    expect(
      configs["chat-fail-once.json"].scenario_failures?.chat_completions?.[0]
    ).toMatchObject({
      status_code: 503,
      code: "uat_fail_once",
      times: 1,
    })

    const unavailable = configs["model-unavailable.json"]
    const omitsSelectedModel = !modelIds(unavailable).includes("missing-uat-model")
    const failsSelectedChatModel = (
      unavailable.scenario_failures?.chat_completions ?? []
    ).some(
      (failure: {
        match?: { model?: string }
        status_code?: number
        code?: string
      }) =>
        failure.match?.model === "missing-uat-model" &&
        failure.status_code === 404 &&
        failure.code === "model_not_found"
    )

    expect(omitsSelectedModel || failsSelectedChatModel).toBe(true)
  })

  it("keeps JSON fixtures static, synthetic, and free of obvious real-secret markers", () => {
    for (const file of [
      ...configFiles.map((name) => `configs/${name}`),
      ...responseFiles.map((name) => `responses/${name}`),
    ]) {
      const content = readFileSync(path.join(mockOpenAiRoot, file), "utf8")

      expect(() => JSON.parse(content)).not.toThrow()
      for (const marker of secretMarkers) {
        expect(content).not.toContain(marker)
      }
    }
  })

  it("provides stable chat and embedding responses for future UAT assertions", () => {
    const defaultChat = readJson("responses/chat/default.json")
    const sourceSummary = readJson("responses/chat/source-summary.json")
    const embeddings = readJson("responses/embeddings/default.json")

    expect(defaultChat.choices?.[0]?.message?.content).toContain(
      "onboarding UAT ready"
    )
    expect(sourceSummary.choices?.[0]?.message?.content).toContain(
      "short first-run wizard"
    )
    expect(sourceSummary.choices?.[0]?.message?.content).toContain(
      "Deterministic evidence"
    )

    expect(embeddings.object).toBe("list")
    expect(embeddings.data?.[0]).toMatchObject({
      object: "embedding",
      index: 0,
    })
    expect(embeddings.data?.[0]?.embedding.length).toBeGreaterThan(0)
    expect(
      embeddings.data[0].embedding.every((value: unknown) => typeof value === "number")
    ).toBe(true)
  })

  it("provides structured markdown and HTML source fixtures", () => {
    const markdownPath = "e2e/fixtures/media/onboarding-uat-note.md"
    const htmlPath = "public/e2e/onboarding-uat-research-note.html"

    expect(existsSync(path.join(frontendRoot, markdownPath))).toBe(true)
    expect(existsSync(path.join(frontendRoot, htmlPath))).toBe(true)

    const markdown = readText(markdownPath)
    const html = readText(htmlPath)

    for (const fixture of [markdown, html]) {
      expect(fixture).toContain("Onboarding UAT Research Note")
      expect(fixture).toContain("2026-06-02")
      expect(fixture).toContain("Claims")
      expect(fixture).toContain("short first-run wizard")
      expect(fixture).toContain("Deterministic evidence")
      expect(fixture).toContain("Action Items")
      expect(fixture).toContain("Verify first chat")
      expect(fixture).toContain("Add one source")
      expect(fixture).toContain("Ask for a summary")
    }
  })
})
