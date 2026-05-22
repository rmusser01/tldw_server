import { describe, expect, it, vi } from "vitest"
import {
  buildPipelineWizardReviewSummary,
  createDefaultPipelineWizardDraft,
  toBriefingPipelineDraft,
  toPipelineWizardSourcePayload,
  validatePipelineWizardCron,
  validatePipelineWizardDraft
} from "../pipeline-wizard-state"

describe("watchlists pipeline wizard state", () => {
  it("validates source, monitor, digest, delivery, and optional audio requirements", () => {
    const base = createDefaultPipelineWizardDraft()

    expect(validatePipelineWizardDraft(base)).toEqual({
      valid: false,
      errors: expect.arrayContaining(["sourceIds", "monitorName", "templateName"])
    })

    expect(
      validatePipelineWizardDraft({
        ...base,
        sourceMode: "new",
        sourceName: "Security News",
        sourceUrl: "https://example.com/rss.xml",
        sourceType: "rss",
        monitorName: "Morning Brief",
        templateName: "briefing_md",
        scheduleMode: "interval",
        scheduleIntervalUnit: "hours",
        scheduleIntervalValue: 100,
        emailDeliveryEnabled: true,
        emailRecipients: ["bad-email"],
        audioEnabled: true,
        audioSpeakers: [
          { id: "host", label: "Host", role: "host", voice: "alloy" },
          { id: "host", label: "Duplicate", role: "analyst", voice: "" },
          { id: "guest", label: "Guest", role: "guest", voice: "nova" },
          { id: "editor", label: "Editor", role: "editor", voice: "echo" },
          { id: "extra", label: "Extra", role: "extra", voice: "fable" }
        ],
        targetAudioMinutes: 0
      })
    ).toEqual({
      valid: false,
      errors: expect.arrayContaining([
        "scheduleIntervalValue",
        "emailRecipients",
        "audioSpeakers",
        "audioSpeakerIds",
        "audioSpeakerVoices",
        "targetAudioMinutes"
      ])
    })

    expect(
      validatePipelineWizardDraft({
        ...base,
        sourceMode: "existing",
        sourceIds: [10],
        monitorName: "Morning Brief",
        templateName: "briefing_md",
        audioEnabled: false,
        audioSpeakers: []
      })
    ).toEqual({ valid: true, errors: [] })

    expect(
      validatePipelineWizardDraft({
        ...base,
        sourceMode: "existing",
        sourceIds: [10],
        monitorName: "Morning Brief",
        templateName: "briefing_md",
        scheduleMode: "daily",
        scheduleHour: 24,
        scheduleMinute: 60,
        audioEnabled: false,
        audioSpeakers: []
      })
    ).toEqual({
      valid: false,
      errors: expect.arrayContaining(["scheduleHour", "scheduleMinute"])
    })
  })

  it("builds source payloads and variable cadence contract drafts", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "UTC" })
          }) as Intl.DateTimeFormat
      )

    const draft = {
      ...createDefaultPipelineWizardDraft(),
      sourceMode: "new" as const,
      sourceName: "AI News",
      sourceUrl: "https://example.com/feed.xml",
      sourceType: "rss" as const,
      monitorName: "Five Hour Brief",
      scheduleMode: "interval" as const,
      scheduleIntervalValue: 5,
      scheduleIntervalUnit: "hours" as const,
      scheduleMinute: 15,
      templateName: "newsletter_markdown",
      templateFormat: "md" as const,
      audioEnabled: true,
      audioSpeakers: [
        { id: "host", label: "Host", role: "host", voice: "alloy" },
        { id: "analyst", label: "Analyst", role: "analyst", voice: "nova" }
      ],
      targetAudioMinutes: 9
    }

    expect(toPipelineWizardSourcePayload(draft, 42)).toEqual({
      name: "AI News",
      url: "https://example.com/feed.xml",
      source_type: "rss",
      active: true,
      watchlist_id: 42
    })

    expect(toBriefingPipelineDraft(draft, [88])).toEqual(
      expect.objectContaining({
        monitorName: "Five Hour Brief",
        sourceIds: [88],
        scheduleExpr: "15 */5 * * *",
        timezone: "UTC",
        templateName: "newsletter_markdown",
        includeAudio: true,
        audioCast: {
          speaker_count: 2,
          speakers: [
            { id: "host", label: "Host", role: "host", voice: "alloy" },
            { id: "analyst", label: "Analyst", role: "analyst", voice: "nova" }
          ]
        },
        voiceMap: {
          host: "alloy",
          analyst: "nova"
        }
      })
    )

    timezoneSpy.mockRestore()
  })

  it("does not force an unknown template format into payload drafts", () => {
    const draft = {
      ...createDefaultPipelineWizardDraft(),
      sourceMode: "existing" as const,
      sourceIds: [10],
      monitorName: "HTML Brief",
      templateName: "html_newsletter",
      audioEnabled: false,
      audioSpeakers: []
    }

    expect(toBriefingPipelineDraft(draft)).toEqual(
      expect.not.objectContaining({
        templateFormat: expect.any(String)
      })
    )
  })

  it("summarizes source, cadence, filters, output, delivery, and audio expectations", () => {
    const summary = buildPipelineWizardReviewSummary(
      {
        ...createDefaultPipelineWizardDraft(),
        sourceMode: "existing",
        sourceIds: [7, 8],
        monitorName: "Weekly Policy Brief",
        scheduleMode: "weekly",
        scheduleHour: 7,
        scheduleMinute: 30,
        scheduleWeekday: "MON",
        templateName: "briefing_md",
        emailDeliveryEnabled: true,
        emailRecipients: ["team@example.com"],
        audioEnabled: true,
        audioSpeakers: [
          { id: "host", label: "Host", role: "host", voice: "alloy" }
        ]
      },
      [
        { id: 7, name: "AI Feed" },
        { id: 8, name: "Security Feed" }
      ]
    )

    expect(summary).toEqual({
      sources: "AI Feed, Security Feed",
      cadence: "Weekly on Monday at 07:30",
      filters: "Monitor filters can be refined after creation",
      output: "briefing_md digest",
      delivery: "Email",
      audio: "1 speaker audio briefing"
    })
  })

  it("serializes weekdays and advanced cadence without changing backend schedule fields", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "UTC" })
          }) as Intl.DateTimeFormat
      )
    const base = {
      ...createDefaultPipelineWizardDraft(),
      sourceMode: "existing" as const,
      sourceIds: [10],
      monitorName: "Cadence Brief",
      templateName: "briefing_md",
      audioEnabled: false,
      audioSpeakers: []
    }

    expect(
      toBriefingPipelineDraft({
        ...base,
        scheduleMode: "weekdays",
        scheduleHour: 8,
        scheduleMinute: 15
      })
    ).toEqual(
      expect.objectContaining({
        scheduleExpr: "15 8 * * MON-FRI",
        timezone: "UTC"
      })
    )

    expect(
      toBriefingPipelineDraft({
        ...base,
        scheduleMode: "advanced",
        scheduleAdvancedCron: "20 6 * * TUE"
      })
    ).toEqual(
      expect.objectContaining({
        scheduleExpr: "20 6 * * TUE",
        timezone: "UTC"
      })
    )

    timezoneSpy.mockRestore()
  })

  it("rejects malformed and too-frequent advanced cron cadence", () => {
    const base = {
      ...createDefaultPipelineWizardDraft(),
      sourceMode: "existing" as const,
      sourceIds: [10],
      monitorName: "Cadence Brief",
      templateName: "briefing_md",
      scheduleMode: "advanced" as const,
      audioEnabled: false,
      audioSpeakers: []
    }

    expect(
      validatePipelineWizardDraft({
        ...base,
        scheduleAdvancedCron: "15 6 *"
      })
    ).toEqual({
      valid: false,
      errors: ["scheduleAdvancedCron"]
    })

    expect(
      validatePipelineWizardDraft({
        ...base,
        scheduleAdvancedCron: "15 6 * * WED;rm"
      })
    ).toEqual({
      valid: false,
      errors: ["scheduleAdvancedCron"]
    })

    expect(
      validatePipelineWizardDraft({
        ...base,
        scheduleAdvancedCron: "61 6 * * WED"
      })
    ).toEqual({
      valid: false,
      errors: ["scheduleAdvancedCron"]
    })

    expect(
      validatePipelineWizardDraft({
        ...base,
        scheduleAdvancedCron: "? 6 * * WED"
      })
    ).toEqual({
      valid: false,
      errors: ["scheduleAdvancedCron"]
    })

    expect(
      validatePipelineWizardDraft({
        ...base,
        scheduleAdvancedCron: "*/1 * * * *"
      })
    ).toEqual({
      valid: false,
      errors: ["scheduleAdvancedCronTooFrequent"]
    })
    expect(validatePipelineWizardCron("*/1 * * * *")).toBe("too_frequent")
    expect(
      toBriefingPipelineDraft({
        ...base,
        scheduleAdvancedCron: "*/1 * * * *"
      })
    ).toEqual(expect.not.objectContaining({ scheduleExpr: expect.any(String) }))
  })

  it("summarizes one-source and audio-off review states without podcast assumptions", () => {
    const summary = buildPipelineWizardReviewSummary(
      {
        ...createDefaultPipelineWizardDraft(),
        sourceMode: "existing",
        sourceIds: [7],
        monitorName: "Manual Brief",
        scheduleMode: "manual",
        templateName: "briefing_md",
        audioEnabled: false,
        audioSpeakers: []
      },
      [{ id: 7, name: "AI Feed" }]
    )

    expect(summary).toEqual(
      expect.objectContaining({
        sources: "AI Feed",
        cadence: "Manual only",
        audio: "Audio disabled"
      })
    )
  })
})
