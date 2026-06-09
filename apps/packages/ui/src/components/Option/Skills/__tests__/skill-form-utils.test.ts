import { describe, expect, it } from "vitest"
import type { SkillResponse } from "@/types/skill"
import {
  buildInitialSkillContent,
  buildSkillTemplateContent,
  buildSupportingFilesForCreate,
  buildSupportingFilesForUpdate,
  type SkillTemplateId
} from "../skill-form-utils"

const makeSkill = (overrides?: Partial<SkillResponse>): SkillResponse => ({
  id: "skill-1",
  name: "test-skill",
  description: "Skill description",
  argument_hint: "[arg]",
  disable_model_invocation: false,
  user_invocable: true,
  allowed_tools: ["Read", "Grep"],
  model: "gpt-4o-mini",
  context: "fork",
  content: "Body content",
  raw_content: null,
  supporting_files: { "notes.md": "hello" },
  directory_path: "/tmp/test-skill",
  created_at: "2026-02-16T00:00:00Z",
  last_modified: "2026-02-16T00:00:00Z",
  version: 1,
  ...overrides
})

describe("skill-form-utils", () => {
  it("prefers raw content when available to preserve frontmatter", () => {
    const rawContent = "---\nname: test-skill\ncustom-key: true\n---\n\nBody content"
    const skill = makeSkill({ raw_content: rawContent, content: "Body only" })

    expect(buildInitialSkillContent(skill)).toBe(rawContent)
  })

  it("builds editable content with serialized frontmatter when raw content is absent", () => {
    const skill = makeSkill({ raw_content: null })
    const result = buildInitialSkillContent(skill)

    expect(result).toContain('name: "test-skill"')
    expect(result).toContain('description: "Skill description"')
    expect(result).toContain('argument-hint: "[arg]"')
    expect(result).toContain('allowed-tools: "Read, Grep"')
    expect(result).toContain("context: fork")
    expect(result).toContain("\n\nBody content")
  })

  it.each<SkillTemplateId>(["summarizer", "explainer", "extractor", "blank"])(
    "builds a valid %s starter template with normalized skill name",
    (templateId) => {
      const result = buildSkillTemplateContent(templateId, "Research Summary!!")

      expect(result).toMatch(/^---\n/)
      expect(result).toContain('name: "research-summary"')
      expect(result).toContain("description:")
      expect(result).toContain("argument-hint:")
      expect(result).toContain("context: inline")
      expect(result).toContain("---\n\n")
      expect(result).toContain("$ARGUMENTS")
    }
  )

  it("uses a deterministic fallback name when no skill name has been entered", () => {
    const result = buildSkillTemplateContent("explainer", "")

    expect(result).toContain('name: "explainer-skill"')
    expect(result).toContain("Explain the following concept")
  })

  it("builds update payload with add/edit/remove operations", () => {
    const result = buildSupportingFilesForUpdate(
      { "a.md": "A", "b.md": "B" },
      [
        { filename: "a.md", content: "A2", originalFilename: "a.md" },
        { filename: "c.md", content: "C" }
      ]
    )

    expect(result).toEqual({
      "a.md": "A2",
      "b.md": null,
      "c.md": "C"
    })
  })

  it("rejects duplicate supporting file names in create payload", () => {
    expect(() =>
      buildSupportingFilesForCreate([
        { filename: "notes.md", content: "A" },
        { filename: "notes.md", content: "B" }
      ])
    ).toThrow("Duplicate supporting file name: notes.md")
  })
})
