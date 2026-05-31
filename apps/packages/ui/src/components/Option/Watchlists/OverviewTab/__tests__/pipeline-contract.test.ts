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

    const jobPayload = toPipelineJobCreatePayload(baseDraft)
    expect(jobPayload).toEqual(
      expect.objectContaining({
        name: "Morning Brief",
        scope: { sources: [10, 11] },
        schedule_expr: "0 8 * * *",
        timezone: "UTC",
        output_prefs: expect.objectContaining({
          template_name: "briefing_markdown",
          template: expect.objectContaining({
            default_name: "briefing_markdown"
          }),
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
    expect(jobPayload.output_prefs).toMatchObject({
      template_name: "briefing_markdown",
      template: { default_name: "briefing_markdown" }
    })

    const outputPayload = toPipelineOutputCreatePayload(9001, baseDraft, [1, 2])
    expect(outputPayload).toEqual({
      run_id: 9001,
      item_ids: [1, 2],
      type: "briefing_markdown",
      format: "md",
      template_name: "briefing_markdown",
      template_version: 2,
      generate_audio: true,
      audio_voice: "alloy",
      target_audio_minutes: 8,
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
    expect(outputPayload.template_name).toBe("briefing_markdown")

    timezoneSpy.mockRestore()
  })

  it("does not auto-generate scheduled output unless requested", () => {
    expect(toPipelineJobCreatePayload(baseDraft).output_prefs).not.toHaveProperty("auto_output")
  })

  it("enables scheduled output when a pipeline draft explicitly requests scheduled reports", () => {
    expect(
      toPipelineJobCreatePayload({
        ...baseDraft,
        createScheduledOutput: true
      }).output_prefs
    ).toMatchObject({
      auto_output: {
        enabled: true,
        type: "briefing_markdown",
        format: "md",
        template_name: "briefing_markdown",
        template_version: 2
      }
    })
  })

  it("does not auto-generate scheduled output for manual-only monitor drafts", () => {
    expect(
      toPipelineJobCreatePayload({
        ...baseDraft,
        schedulePreset: "none",
        createScheduledOutput: true,
        includeAudio: false,
        emailRecipients: [],
        createChatbook: false
      }).output_prefs
    ).not.toHaveProperty("auto_output")
  })

  it("serializes variable cadence drafts through the existing schedule contract", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "UTC" })
          }) as Intl.DateTimeFormat
      )

    expect(
      toPipelineJobCreatePayload({
        ...baseDraft,
        schedulePreset: "none",
        scheduleCadence: { kind: "interval", every: 30, unit: "minute" }
      })
    ).toMatchObject({
      schedule_expr: "*/30 * * * *",
      timezone: "UTC"
    })

    expect(
      toPipelineJobCreatePayload({
        ...baseDraft,
        schedulePreset: "none",
        scheduleCadence: { kind: "weekly", weekday: "fri", time: "09:15" }
      })
    ).toMatchObject({
      schedule_expr: "15 9 * * FRI",
      timezone: "UTC"
    })

    expect(
      toPipelineJobCreatePayload({
        ...baseDraft,
        schedulePreset: "none",
        scheduleCadence: { kind: "advanced", cron: "20 6 * * TUE" }
      })
    ).toMatchObject({
      schedule_expr: "20 6 * * TUE",
      timezone: "UTC"
    })

    timezoneSpy.mockRestore()
  })

  it("uses the same schedule precedence for payload and review summary", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "UTC" })
          }) as Intl.DateTimeFormat
      )
    const draft: BriefingPipelineDraft = {
      ...baseDraft,
      schedulePreset: "none",
      scheduleExpr: "0 8 * * *",
      scheduleCadence: { kind: "interval", every: 30, unit: "minute" }
    }

    expect(toPipelineJobCreatePayload(draft)).toMatchObject({
      schedule_expr: "*/30 * * * *",
      timezone: "UTC"
    })
    expect(buildPipelineReviewSummary(draft).scheduleLabel).toBe("Every 30 minutes")

    timezoneSpy.mockRestore()
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

  it("labels variable cadence drafts in the review summary", () => {
    expect(
      buildPipelineReviewSummary({
        ...baseDraft,
        schedulePreset: "none",
        scheduleCadence: { kind: "interval", every: 30, unit: "minute" }
      }).scheduleLabel
    ).toBe("Every 30 minutes")

    expect(
      buildPipelineReviewSummary({
        ...baseDraft,
        schedulePreset: "none",
        scheduleCadence: { kind: "weekly", weekday: "fri", time: "09:15" }
      }).scheduleLabel
    ).toBe("Weekly on Friday at 09:15")

    expect(
      buildPipelineReviewSummary({
        ...baseDraft,
        schedulePreset: "none",
        scheduleCadence: { kind: "advanced", cron: "20 6 * * TUE" }
      }).scheduleLabel
    ).toBe("Custom cron: 20 6 * * TUE")
  })

  it("supports localized cadence label copy in the review summary", () => {
    expect(
      buildPipelineReviewSummary(
        {
          ...baseDraft,
          schedulePreset: "none",
          scheduleCadence: { kind: "interval", every: 1, unit: "hour" }
        },
        {
          schedule: {
            interval: (value, unit) => `localized ${value} ${unit}`
          }
        }
      ).scheduleLabel
    ).toBe("localized 1 hours")
  })
})
