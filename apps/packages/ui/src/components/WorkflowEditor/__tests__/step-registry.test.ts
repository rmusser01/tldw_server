import { describe, expect, it } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import {
  buildStepRegistry,
  humanizeStepType,
  resolveStepCategory,
  resolveStepIcon,
  resolveStepPorts,
  STEP_CATEGORIES,
  BASE_STEP_REGISTRY,
  CATEGORY_OVERRIDES,
  ICON_OVERRIDES
} from "../step-registry"
import { schemaToConfigFields } from "../schema-utils"
import { STEP_ICON_COMPONENTS } from "../step-icons"
import type { WorkflowStepSchema, StepCategory } from "@/types/workflow-editor"

const BACKEND_REGISTRY_CANDIDATE_PATHS = [
  path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "../../../../../../../tldw_Server_API/app/core/Workflows/registry.py"
  )
]

const loadBackendStepTypes = (): string[] => {
  const backendPath = BACKEND_REGISTRY_CANDIDATE_PATHS.find((candidate) =>
    existsSync(candidate)
  )
  expect(
    backendPath,
    `Unable to locate backend workflow registry. Tried: ${BACKEND_REGISTRY_CANDIDATE_PATHS.join(", ")}`
  ).toBeDefined()

  const source = readFileSync(backendPath as string, "utf8")
  const stepRegex = /"([a-z0-9_]+)":\s*StepType\("([a-z0-9_]+)"/g
  const stepTypes = new Set<string>()

  for (const match of source.matchAll(stepRegex)) {
    expect(
      match[1],
      `Backend step key/name mismatch in ${backendPath}: ${match[1]} != ${match[2]}`
    ).toBe(match[2])
    stepTypes.add(match[1])
  }

  expect(
    stepTypes.size,
    `No backend workflow step types were parsed from ${backendPath}`
  ).toBeGreaterThan(0)
  return Array.from(stepTypes).sort()
}

const BACKEND_STEP_TYPES = loadBackendStepTypes()

// Expected frontend categorization map (must track backend registry)
const ALL_STEP_TYPES: Record<StepCategory, string[]> = {
  ai: [
    "prompt",
    "llm",
    "llm_with_tools",
    "llm_compare",
    "llm_critique",
    "summarize",
    "image_gen",
    "image_describe",
    "translate",
    "voice_intent",
    "flashcard_generate",
    "quiz_generate",
    "quiz_evaluate",
    "outline_generate",
    "glossary_extract",
    "mindmap_generate",
    "report_generate",
    "newsletter_generate",
    "slides_generate",
    "diagram_generate",
    "literature_review",
    "moderation"
  ],
  search: [
    "rag_search",
    "web_search",
    "query_expand",
    "query_rewrite",
    "hyde_generate",
    "semantic_cache_check",
    "search_aggregate",
    "rerank",
    "embed",
    "rss_fetch",
    "atom_fetch"
  ],
  media: [
    "media_ingest",
    "process_media",
    "pdf_extract",
    "ocr",
    "document_table_extract",
    "chunking",
    "claims_extract",
    "citations",
    "bibliography_generate",
    "document_merge",
    "document_diff"
  ],
  text: [
    "json_transform",
    "json_validate",
    "csv_to_json",
    "json_to_csv",
    "regex_extract",
    "text_clean",
    "xml_transform",
    "template_render",
    "markdown_to_html",
    "html_to_markdown",
    "keyword_extract",
    "sentiment_analyze",
    "language_detect",
    "topic_model",
    "entity_extract",
    "context_build"
  ],
  research: [
    "deep_research",
    "deep_research_wait",
    "deep_research_load_bundle",
    "deep_research_select_bundle_fields",
    "arxiv_search",
    "arxiv_download",
    "pubmed_search",
    "semantic_scholar_search",
    "google_scholar_search",
    "patent_search",
    "doi_resolve",
    "reference_parse",
    "bibtex_generate"
  ],
  audio: [
    "tts",
    "multi_voice_tts",
    "stt_transcribe",
    "audio_normalize",
    "audio_concat",
    "audio_trim",
    "audio_convert",
    "audio_extract",
    "audio_mix",
    "audio_diarize",
    "audio_briefing_compose"
  ],
  video: [
    "video_trim",
    "video_concat",
    "video_convert",
    "video_thumbnail",
    "video_extract_frames",
    "subtitle_generate",
    "subtitle_translate",
    "subtitle_burn"
  ],
  control: [
    "acp_stage",
    "branch",
    "map",
    "batch",
    "parallel",
    "retry",
    "cache_result",
    "checkpoint",
    "workflow_call",
    "wait_for_human",
    "wait_for_approval"
  ],
  io: [
    "webhook",
    "notify",
    "kanban",
    "s3_upload",
    "s3_download",
    "github_create_issue",
    "email_send",
    "screenshot_capture",
    "podcast_rss_publish",
    "mcp_tool",
    "chatbooks",
    "character_chat",
    "notes",
    "prompts",
    "collections",
    "evaluations",
    "schedule_workflow"
  ],
  utility: [
    "delay",
    "log",
    "token_count",
    "context_window_check",
    "policy_check",
    "diff_change_detector",
    "sandbox_exec",
    "timing_start",
    "timing_stop",
    "eval_readability"
  ]
}

const FLAT_STEP_TYPES = Object.values(ALL_STEP_TYPES).flat()
const EXPECTED_CATEGORY_BY_STEP = Object.entries(ALL_STEP_TYPES).reduce<
  Record<string, StepCategory>
>((acc, [category, stepTypes]) => {
  for (const stepType of stepTypes) {
    acc[stepType] = category as StepCategory
  }
  return acc
}, {})

const sampleSchema: WorkflowStepSchema = {
  type: "object",
  properties: {
    url: { type: "string", description: "Target URL" },
    count: { type: "integer", default: 2 },
    enabled: { type: "boolean", default: true },
    mode: { type: "string", enum: ["fast", "safe"] },
    tags: { type: "array", items: { type: "string", enum: ["a", "b"] } }
  },
  required: ["url"]
}

describe("workflow step registry", () => {
  it("frontend expected step list stays in sync with backend registry", () => {
    expect([...FLAT_STEP_TYPES].sort()).toEqual(BACKEND_STEP_TYPES)
  })

  it("builds registry entries for server-provided steps", () => {
    const registry = buildStepRegistry([
      { name: "foo_step", description: "Foo step", schema: sampleSchema }
    ])

    expect(registry.foo_step).toBeDefined()
    expect(registry.foo_step.label).toBe(humanizeStepType("foo_step"))
    expect(registry.foo_step.description).toBe("Foo step")
  })

  it("preserves acronym casing in fallback labels", () => {
    expect(humanizeStepType("llm")).toBe("LLM")
    expect(humanizeStepType("llm_compare")).toBe("LLM Compare")

    const registry = buildStepRegistry([{ name: "llm", description: "LLM step" }])
    expect(registry.llm.label).toBe("LLM")
  })

  it("every backend step type has an explicit CATEGORY_OVERRIDES entry", () => {
    for (const stepType of BACKEND_STEP_TYPES) {
      const expectedCategory = EXPECTED_CATEGORY_BY_STEP[stepType]
      expect(
        expectedCategory,
        `Missing expected category mapping for backend step "${stepType}"`
      ).toBeDefined()
      expect(
        CATEGORY_OVERRIDES[stepType],
        `Missing explicit CATEGORY_OVERRIDES entry for backend step "${stepType}"`
      ).toBe(expectedCategory)
      expect(resolveStepCategory(stepType)).toBe(expectedCategory)
    }
  })

  it("every backend step type has an explicit icon override and valid icon", () => {
    for (const stepType of BACKEND_STEP_TYPES) {
      const iconName = ICON_OVERRIDES[stepType]
      expect(
        iconName,
        `Missing ICON_OVERRIDES entry for backend step "${stepType}"`
      ).toBeDefined()
      expect(
        STEP_ICON_COMPONENTS[iconName],
        `Icon "${iconName}" for step "${stepType}" not in STEP_ICON_COMPONENTS`
      ).toBeDefined()
      expect(resolveStepIcon(stepType)).toBe(iconName)
    }
    // Also check start/end
    for (const special of ["start", "end"]) {
      const iconName = resolveStepIcon(special)
      expect(
        STEP_ICON_COMPONENTS[iconName],
        `Icon "${iconName}" for step "${special}" not in STEP_ICON_COMPONENTS`
      ).toBeDefined()
    }
  })

  it("buildStepRegistry assigns correct categories for all backend step types", () => {
    const serverSteps = BACKEND_STEP_TYPES.map((name) => ({
      name,
      description: `${name} description`
    }))
    const registry = buildStepRegistry(serverSteps)

    for (const stepType of BACKEND_STEP_TYPES) {
      const expectedCategory = EXPECTED_CATEGORY_BY_STEP[stepType]
      expect(registry[stepType]).toBeDefined()
      expect(
        registry[stepType].category,
        `${stepType} in built registry should be ${expectedCategory}`
      ).toBe(expectedCategory)
    }
  })

  it("uses category-aware fallback ports for non-overridden steps", () => {
    const researchPorts = resolveStepPorts("arxiv_search")
    expect(researchPorts.inputs[0].dataType).toBe("string")
    expect(researchPorts.outputs[0].dataType).toBe("array")

    const integrationPorts = resolveStepPorts("podcast_rss_publish")
    expect(integrationPorts.inputs[0].dataType).toBe("object")
    expect(integrationPorts.outputs[0].dataType).toBe("object")
  })

  it("no regressions on existing BASE_STEP_REGISTRY entries", () => {
    const expectedBaseEntries = [
      "prompt",
      "rag_search",
      "media_ingest",
      "branch",
      "map",
      "wait_for_human",
      "webhook",
      "tts",
      "stt_transcribe",
      "deep_research",
      "deep_research_wait",
      "deep_research_load_bundle",
      "deep_research_select_bundle_fields",
      "delay",
      "log",
      "start",
      "end"
    ]
    for (const type of expectedBaseEntries) {
      expect(
        BASE_STEP_REGISTRY[type],
        `${type} should be in BASE_STEP_REGISTRY`
      ).toBeDefined()
      expect(BASE_STEP_REGISTRY[type].label).toBeTruthy()
      expect(BASE_STEP_REGISTRY[type].configSchema.length).toBeGreaterThan(0)
    }
  })

  it("STEP_CATEGORIES has all 10 categories", () => {
    const expected: StepCategory[] = [
      "ai",
      "search",
      "media",
      "text",
      "research",
      "audio",
      "video",
      "control",
      "io",
      "utility"
    ]
    for (const cat of expected) {
      expect(STEP_CATEGORIES[cat], `Category ${cat} should exist`).toBeDefined()
      expect(STEP_CATEGORIES[cat].label).toBeTruthy()
      expect(STEP_CATEGORIES[cat].color).toBeTruthy()
    }
    expect(STEP_CATEGORIES.ai.color).toBe("blue")
    expect(STEP_CATEGORIES.control.color).toBe("indigo")
  })

  it("all backend step types have meaningful icons", () => {
    const stepsWithDefaultIcon: string[] = []
    for (const stepType of BACKEND_STEP_TYPES) {
      const icon = resolveStepIcon(stepType)
      if (
        icon === "MessageSquare" &&
        stepType !== "prompt" &&
        stepType !== "llm" &&
        stepType !== "prompts"
      ) {
        stepsWithDefaultIcon.push(stepType)
      }
    }
    expect(
      stepsWithDefaultIcon,
      `These step types still use the default MessageSquare icon: ${stepsWithDefaultIcon.join(", ")}`
    ).toEqual([])
  })

  it("frontend step count matches backend registry count", () => {
    expect(FLAT_STEP_TYPES.length).toBe(BACKEND_STEP_TYPES.length)
  })

  it("categories are ordered correctly", () => {
    const entries = Object.entries(STEP_CATEGORIES).sort(
      (a, b) => a[1].order - b[1].order
    )
    expect(entries.map(([k]) => k)).toEqual([
      "ai",
      "search",
      "media",
      "text",
      "research",
      "audio",
      "video",
      "control",
      "io",
      "utility"
    ])
  })
})

describe("schemaToConfigFields", () => {
  it("maps JSON schema properties to config fields", () => {
    const fields = schemaToConfigFields(sampleSchema)
    const byKey = Object.fromEntries(fields.map((field) => [field.key, field]))

    expect(byKey.url.type).toBe("url")
    expect(byKey.url.required).toBe(true)
    expect(byKey.count.type).toBe("number")
    expect(byKey.enabled.type).toBe("checkbox")
    expect(byKey.mode.type).toBe("select")
    expect(byKey.tags.type).toBe("multiselect")
  })

  it("assigns dynamic field types for common resource references", () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        model: { type: "string" },
        prompt_id: { type: "string" },
        collection_id: { type: "string" },
        provider: { type: "string" },
        voice: { type: "string" },
        dataset_id: { type: "string" },
        run_id: { type: "string" },
        item_id: { type: "string" },
        output_id: { type: "string" },
        run_ids: { type: "array", items: { type: "string" } },
        item_ids: { type: "array", items: { type: "string" } }
      }
    }
    const fields = schemaToConfigFields(schema)
    const byKey = Object.fromEntries(fields.map((field) => [field.key, field]))

    expect(byKey.model.type).toBe("model-picker")
    expect(byKey.prompt_id.type).toBe("select")
    expect(byKey.collection_id.type).toBe("collection-picker")
    expect(byKey.provider.type).toBe("select")
    expect(byKey.voice.type).toBe("select")
    expect(byKey.dataset_id.type).toBe("select")
    expect(byKey.run_id.type).toBe("select")
    expect(byKey.item_id.type).toBe("select")
    expect(byKey.output_id.type).toBe("select")
    expect(byKey.run_ids.type).toBe("multiselect")
    expect(byKey.item_ids.type).toBe("multiselect")
  })
})
