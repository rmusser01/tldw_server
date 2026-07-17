import { describe, expect, it } from "vitest"
import type { SkillResponse } from "@/types/skill"
import {
  buildGuidedDraftFromSkill,
  buildGuidedDraftFromTemplate,
  buildDuplicateSkillContent,
  buildInitialSkillContent,
  buildSkillTemplateContent,
  buildSupportingFilesForCreate,
  buildSupportingFilesForUpdate,
  limitSkillSelection,
  parseAllowedTools,
  serializeGuidedSkillContent,
  validateGuidedSkillDraft,
  validateRawSkillContent,
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

  it("renames a duplicate without dropping custom frontmatter", () => {
    const result = buildDuplicateSkillContent(
      makeSkill({
        raw_content: "---\nname: test-skill\ncustom-key: true\n---\n\nBody content"
      }),
      "test-skill-copy"
    )

    expect(result).toContain('name: "test-skill-copy"')
    expect(result).toContain("custom-key: true")
    expect(result).not.toContain("name: test-skill\n")
  })

  it("inserts a root duplicate name without replacing a nested name", () => {
    const result = buildDuplicateSkillContent(
      makeSkill({
        raw_content: "---\nmetadata:\n  name: nested-name\ncustom-key: true\n---\n\nBody content"
      }),
      "test-skill-copy"
    )

    expect(result).toContain('---\nname: "test-skill-copy"\nmetadata:')
    expect(result).toContain("  name: nested-name")
    expect(result).toContain("custom-key: true")
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

  it("builds guided template fields without making users edit YAML", () => {
    expect(buildGuidedDraftFromTemplate("explainer", "concept-coach")).toEqual({
      name: "concept-coach",
      description: "Teach a concept step by step for a specific audience.",
      argumentHint: "[concept] [audience or current understanding]",
      instructions: expect.stringContaining("Explain the following concept"),
      context: "inline",
      userInvocable: true,
      allowModelInvocation: true,
      model: "",
      allowedTools: ""
    })
  })

  it("builds guided fields from the structured skill response", () => {
    expect(buildGuidedDraftFromSkill(makeSkill())).toEqual({
      name: "test-skill",
      description: "Skill description",
      argumentHint: "[arg]",
      instructions: "Body content",
      context: "fork",
      userInvocable: true,
      allowModelInvocation: true,
      model: "gpt-4o-mini",
      allowedTools: "Read, Grep"
    })
  })

  it("serializes guided fields into canonical SKILL.md content", () => {
    const result = serializeGuidedSkillContent({
      name: "research-helper",
      description: "Research a topic",
      argumentHint: "[topic]",
      instructions: "Research $ARGUMENTS and cite sources.",
      context: "fork",
      userInvocable: false,
      allowModelInvocation: false,
      model: "gpt-4o-mini",
      allowedTools: "Read, Grep\nRead, WebSearch"
    })

    expect(result).toContain('name: "research-helper"')
    expect(result).toContain('description: "Research a topic"')
    expect(result).toContain('argument-hint: "[topic]"')
    expect(result).toContain("disable-model-invocation: true")
    expect(result).toContain("user-invocable: false")
    expect(result).toContain('allowed-tools: "Read, Grep, WebSearch"')
    expect(result).toContain('model: "gpt-4o-mini"')
    expect(result).toContain("context: fork")
    expect(result.endsWith("Research $ARGUMENTS and cite sources.")).toBe(true)
  })

  it("normalizes declared tools from comma and newline separated input", () => {
    expect(parseAllowedTools(" Read, Grep\nRead, WebSearch ,, ")).toEqual([
      "Read",
      "Grep",
      "WebSearch"
    ])
  })

  it("deduplicates and caps bulk selection at the API contract", () => {
    const requestedNames = [
      ...Array.from({ length: 101 }, (_, index) => `skill-${index + 1}`),
      "skill-1"
    ]

    expect(limitSkillSelection(requestedNames)).toEqual({
      names: Array.from({ length: 100 }, (_, index) => `skill-${index + 1}`),
      limited: true
    })
  })

  it("returns actionable guided-field validation errors", () => {
    expect(
      validateGuidedSkillDraft({
        name: "Bad Name",
        description: " ",
        argumentHint: "",
        instructions: "",
        context: "inline",
        userInvocable: true,
        allowModelInvocation: true,
        model: "",
        allowedTools: ""
      })
    ).toEqual([
      "Name must start with a lowercase letter and use only lowercase letters, numbers, and hyphens (max 64 characters).",
      "Description is required.",
      "Instructions are required."
    ])
  })

  it("validates advanced source without rejecting body-only skills", () => {
    expect(validateRawSkillContent("Process $ARGUMENTS")).toEqual([])
    expect(validateRawSkillContent("---\nname: test\nProcess $ARGUMENTS")).toEqual([
      "Frontmatter starts with --- but has no closing --- delimiter."
    ])
    expect(validateRawSkillContent("---\nname: test\n---\n\n ")).toEqual([
      "Skill instructions are required after frontmatter."
    ])
    expect(validateRawSkillContent("  ")).toEqual(["Skill content is required."])
  })

  it("validates top-level frontmatter names against the canonical skill name", () => {
    expect(
      validateRawSkillContent('---\nname: "canonical-skill" # identifier\n---\n\nBody', "canonical-skill")
    ).toEqual([])
    expect(
      validateRawSkillContent("---\n'name': 'canonical-skill'\n---\n\nBody", "canonical-skill")
    ).toEqual([])
    expect(
      validateRawSkillContent("---\nmetadata:\n  name: nested-name\n---\n\nBody", "canonical-skill")
    ).toEqual([])
    expect(
      validateRawSkillContent("---\nname: other-skill\n---\n\nBody", "canonical-skill")
    ).toEqual([
      'Frontmatter name "other-skill" must match canonical name "canonical-skill".'
    ])
    expect(
      validateRawSkillContent(
        "---\n{name: other-skill, description: Mismatch}\n---\n\nBody",
        "canonical-skill"
      )
    ).toEqual([
      'Frontmatter name "other-skill" must match canonical name "canonical-skill".'
    ])
    expect(
      validateRawSkillContent("---\nname: yes\n---\n\nBody", "yes")
    ).toEqual(["Frontmatter name must be a string."])
    expect(
      validateRawSkillContent('---\nname: "yes"\n---\n\nBody', "yes")
    ).toEqual([])
    expect(
      validateRawSkillContent(
        '---\nname: "Canonical-Skill"\n---\n\nBody',
        "canonical-skill"
      )
    ).toEqual([])
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
