import { describe, expect, it, vi } from "vitest"
import type { ChatMacroSettings } from "@/services/chat-macros"
import {
  createBlankMacroDraft,
  outputProfilesToSettings,
  parseMacroSource,
  readMacroImport,
  serializeGuidedMacro,
  type GuidedMacroDraft
} from "../chat-macro-editor-utils"

const draft: GuidedMacroDraft = {
  name: "handoff",
  command: "handoff",
  description: "Prepare a concise team handoff.",
  outputProfile: "brief",
  maxBranches: 4,
  maxConcurrency: 2,
  timeoutSeconds: 90,
  branches: [
    {
      id: "summary",
      label: "Summary",
      output: "summary",
      prompt: "Summarize the current work."
    },
    {
      id: "risks",
      label: "Risks",
      output: "risks",
      prompt: "Identify remaining risks."
    }
  ],
  merge: {
    id: "merge",
    output: "handoff_result",
    prompt: "Combine the branch outputs into a handoff."
  }
}

describe("chat macro editor utils", () => {
  it("serializes a guided draft as a v1 macro YAML definition", () => {
    const raw = serializeGuidedMacro(draft)

    expect(raw).toContain("schema_version: 1")
    expect(raw).toContain("type: branch_prompt")
    expect(raw).toContain("type: post_result")
  })

  it("round-trips supported guided topology without changing authoring fields", () => {
    const result = parseMacroSource(serializeGuidedMacro(draft))

    expect(result).toEqual({ mode: "guided", draft })
  })

  it("keeps unsupported definitions in source mode without rewriting YAML", () => {
    const raw = `schema_version: 1
name: handoff
command: handoff
enabled: true
args: {}
context:
  surfaces: [chat]
  include_chat_history: true
  include_workspace_context: auto
  retrieval: auto
  snapshot_at_dispatch: true
execution:
  mode_default: background
  branch_strategy: auto
  max_branches: 4
  max_concurrency: 2
  timeout_seconds: 90
  retries_per_branch: 1
  merge_retries: 1
  partial_failure: best_effort
  retain_scratch_branches: false
steps:
  - id: prepare
    type: prompt
    output: preparation
    prompt: Prepare the context.
  - id: summary
    type: branch_prompt
    label: Summary
    output: summary
    prompt: Summarize the current work.
  - id: merge
    type: merge
    consumes: [summary]
    output: handoff_result
    prompt: Combine the branch outputs into a handoff.
  - id: post
    type: post_result
    consumes: [handoff_result]
output_profile: brief
permissions:
  tool_calls: []
  skills: []
`

    expect(parseMacroSource(raw)).toEqual({ mode: "source", raw })
  })

  it("keeps out-of-bounds guided caps in source mode without rewriting YAML", () => {
    const raw = serializeGuidedMacro(draft).replace("max_branches: 4", "max_branches: 7")

    expect(parseMacroSource(raw)).toEqual({ mode: "source", raw })
  })

  it("returns bounded errors for malformed and non-mapping YAML", () => {
    expect(parseMacroSource("name: [")).toEqual({
      mode: "source",
      raw: "name: [",
      error: "Macro YAML is invalid."
    })
    expect(parseMacroSource("- handoff\n- wrapup\n")).toEqual({
      mode: "source",
      raw: "- handoff\n- wrapup\n",
      error: "Macro YAML must be a mapping."
    })
  })

  it("rejects oversized YAML imports before reading the file", async () => {
    const text = vi.fn()
    const file = { name: "handoff.yaml", size: 500_001, text } as unknown as File

    await expect(readMacroImport(file)).rejects.toThrow("500,000 bytes")
    expect(text).not.toHaveBeenCalled()
  })

  it("reads supported YAML imports as text", async () => {
    const yaml = { name: "handoff.yaml", size: 20, text: vi.fn().mockResolvedValue("name: handoff") }
    const yml = { name: "handoff.yml", size: 20, text: vi.fn().mockResolvedValue("name: handoff") }

    await expect(readMacroImport(yaml as unknown as File)).resolves.toBe("name: handoff")
    await expect(readMacroImport(yml as unknown as File)).resolves.toBe("name: handoff")
  })

  it("retains non-profile macro settings when replacing output profiles", () => {
    const settings: ChatMacroSettings = {
      disabled_builtins: ["wrapup"],
      user_macro_enabled: { handoff: false },
      output_profiles: {},
      future_setting: { enabled: true }
    }
    const profiles = {
      brief: {
        format: "structured_sections" as const,
        sections: ["summary"],
        section_titles: { summary: "Executive brief" },
        include_branch_outputs: false
      }
    }

    expect(outputProfilesToSettings(settings, profiles)).toEqual({
      disabled_builtins: ["wrapup"],
      user_macro_enabled: { handoff: false },
      output_profiles: profiles,
      future_setting: { enabled: true }
    })
  })

  it("starts blank guided drafts with v1 execution defaults", () => {
    expect(createBlankMacroDraft()).toMatchObject({
      outputProfile: "default",
      maxBranches: 6,
      maxConcurrency: 3,
      timeoutSeconds: 180
    })
  })
})
