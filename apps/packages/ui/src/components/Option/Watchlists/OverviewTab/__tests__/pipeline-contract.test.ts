import { describe, expect, it, vi } from "vitest"
import {
  buildPipelineReviewSummary,
  toPipelineJobCreatePayload,
  toPipelineOutputCreatePayload,
  validateBriefingPipelineDraft,
  type BriefingPipelineDraft
} from "../pipeline-contract"

describe("watchlists pipeline contract", () => {
  const baseDraft: BriefingPipelineDraft = {
    monitorName: "Morning Brief",
    sourceIds: [10, 11],
    schedulePreset: "daily",
    templateName: "briefing_md",
    templateFormat: "md",
    templateVersion: 2,
    includeAudio: true,
    audioVoice: "alloy",
    targetAudioMinutes: 8,
    emailRecipients: [" analyst@example.com ", ""],
    createChatbook: true,
    chatbookTitle: "Morning Intel"
  }

  it("validates required fields for pipeline setup", () => {
    expect(validateBriefingPipelineDraft(baseDraft)).toEqual({
      valid: true,
      errors: []
    })

    expect(
      validateBriefingPipelineDraft({
        ...baseDraft,
        monitorName: " ",
        sourceIds: [],
        templateName: "",
        audioVoice: "",
        targetAudioMinutes: 0
      })
    ).toEqual({
      valid: false,
      errors: [
        "monitorName",
        "sourceIds",
        "templateName",
        "audioVoice",
        "targetAudioMinutes"
      ]
    })
  })

  it("maps pipeline draft into job and output payloads", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "UTC" })
          }) as Intl.DateTimeFormat
      )

    expect(toPipelineJobCreatePayload(baseDraft)).toEqual(
      expect.objectContaining({
        name: "Morning Brief",
        scope: { sources: [10, 11] },
        schedule_expr: "0 8 * * *",
        timezone: "UTC",
        output_prefs: expect.objectContaining({
          auto_output: {
            enabled: true,
            type: "briefing_markdown",
            format: "md",
            template_name: "briefing_md",
            template_version: 2
          },
          template_name: "briefing_md",
          generate_audio: true,
          audio_voice: "alloy",
          target_audio_minutes: 8,
          deliveries: {
            email: { enabled: true, recipients: ["analyst@example.com"] },
            chatbook: { enabled: true, title: "Morning Intel" }
          }
        })
      })
    )

    expect(toPipelineOutputCreatePayload(9001, baseDraft, [1, 2])).toEqual({
      run_id: 9001,
      item_ids: [1, 2],
      type: "briefing_markdown",
      format: "md",
      template_name: "briefing_md",
      template_version: 2,
      metadata: {
        audio: {
          enabled: true,
          voice: "alloy",
          target_minutes: 8
        }
      },
      deliveries: {
        email: { recipients: ["analyst@example.com"] },
        chatbook: { enabled: true, title: "Morning Intel" }
      }
    })

    timezoneSpy.mockRestore()
  })

  it("does not auto-generate scheduled output for manual-only monitor drafts", () => {
    expect(
      toPipelineJobCreatePayload({
        ...baseDraft,
        schedulePreset: "none",
        includeAudio: false,
        emailRecipients: [],
        createChatbook: false
      }).output_prefs
    ).not.toHaveProperty("auto_output")
  })

  it("propagates html template format into job and output payloads", () => {
    const htmlDraft: BriefingPipelineDraft = {
      ...baseDraft,
      templateFormat: "html"
    }

    expect(toPipelineJobCreatePayload(htmlDraft)).toEqual(
      expect.objectContaining({
        output_prefs: expect.objectContaining({
          template: expect.objectContaining({
            default_format: "html"
          })
        })
      })
    )

    expect(toPipelineOutputCreatePayload(9002, htmlDraft)).toEqual(
      expect.objectContaining({
        run_id: 9002,
        format: "html"
      })
    )
  })

  it("builds review summary with expected schedule, artifacts, and delivery channels", () => {
    expect(buildPipelineReviewSummary(baseDraft)).toEqual({
      scheduleLabel: "Daily at 08:00",
      artifacts: ["Text briefing", "Audio briefing"],
      deliveries: ["Email", "Chatbook"]
    })

    expect(
      buildPipelineReviewSummary({
        ...baseDraft,
        schedulePreset: "none",
        includeAudio: false,
        emailRecipients: [],
        createChatbook: false
      })
    ).toEqual({
      scheduleLabel: "Manual only",
      artifacts: ["Text briefing"],
      deliveries: ["In-app reports"]
    })
  })
})
