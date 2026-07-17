import { describe, expect, it, vi } from "vitest"
import {
  buildPipelineWizardReviewSummary,
  createDefaultPipelineWizardDraft,
  getPipelineWizardBriefingOutcome,
  getPipelineWizardBriefingStatus,
  getPipelineWizardSourceSignature,
  normalizePipelineWizardSpeakers,
  projectPipelineWizardOccurrences,
  toBriefingPipelineDraft,
  toPipelineWizardSourcePayload,
  validatePipelineWizardCron,
  validatePipelineWizardDraft,
  waitForPipelineWizardBriefing
} from "../pipeline-wizard-state"
import { toPipelineJobCreatePayload } from "../pipeline-contract"

describe("watchlists pipeline wizard state", () => {
  it("binds persisted sources to normalized draft identity", () => {
    const base = createDefaultPipelineWizardDraft()
    expect(getPipelineWizardSourceSignature({
      ...base,
      sourceMode: "new",
      sourceName: " Feed ",
      sourceUrl: " https://example.com/feed.xml "
    })).toBe(getPipelineWizardSourceSignature({
      ...base,
      sourceMode: "new",
      sourceName: "Feed",
      sourceUrl: "https://example.com/feed.xml"
    }))
    expect(getPipelineWizardSourceSignature({
      ...base,
      sourceIds: [3, 1, 3]
    })).toBe(getPipelineWizardSourceSignature({
      ...base,
      sourceIds: [1, 3]
    }))
  })

  it("polls the exact run and stops on an observed stage failure", async () => {
    const projection = (stageStatus: "running" | "failed") => ({
      occurrence_id: 1,
      run_id: 44,
      job_id: 7,
      artifact_status: "running" as const,
      delivery_status: "waiting_for_artifacts" as const,
      stages: { select: { status: stageStatus, retryable: stageStatus === "failed" } },
      output: null,
      audio: null,
      editorial: {},
      selection: {},
      next_run_at: null,
      recovery: {}
    })
    const getBriefing = vi.fn()
      .mockResolvedValueOnce(projection("running"))
      .mockResolvedValueOnce(projection("failed"))
    const onProgress = vi.fn()

    const result = await waitForPipelineWizardBriefing(44, getBriefing, onProgress, {
      intervalMs: 0,
      maxAttempts: 3,
      waitForDelivery: true
    })

    expect(getBriefing).toHaveBeenCalledTimes(2)
    expect(onProgress).toHaveBeenCalledTimes(2)
    expect(result.stages.select?.status).toBe("failed")
  })

  it("aborts the current sleep and makes no further polling requests", async () => {
    const controller = new AbortController()
    const projection = {
      occurrence_id: 1,
      run_id: 44,
      job_id: 7,
      artifact_status: "running" as const,
      delivery_status: "waiting_for_artifacts" as const,
      stages: { select: { status: "running" as const } },
      output: null,
      audio: null,
      editorial: {},
      selection: {},
      next_run_at: null,
      recovery: {}
    }
    const getBriefing = vi.fn().mockResolvedValue(projection)
    const polling = waitForPipelineWizardBriefing(
      44,
      getBriefing,
      () => controller.abort(),
      { intervalMs: 10_000, maxAttempts: 3, signal: controller.signal }
    )

    await expect(polling).rejects.toMatchObject({ name: "AbortError" })
    expect(getBriefing).toHaveBeenCalledTimes(1)
    expect(getBriefing).toHaveBeenCalledWith(44, controller.signal)
  })

  it("returns a still-running projection at the foreground polling bound", async () => {
    const projection = {
      occurrence_id: 19,
      run_id: 44,
      job_id: 7,
      artifact_status: "running" as const,
      delivery_status: "waiting_for_artifacts" as const,
      stages: { generate_audio: { status: "running" as const } },
      output: { id: 90 },
      audio: null,
      editorial: {},
      selection: {},
      next_run_at: null,
      recovery: {}
    }
    const getBriefing = vi.fn().mockResolvedValue(projection)

    await expect(waitForPipelineWizardBriefing(44, getBriefing, vi.fn(), {
      intervalMs: 0,
      maxAttempts: 2
    })).resolves.toBe(projection)
    expect(getPipelineWizardBriefingStatus(projection)).toBe("running")
    expect(getBriefing).toHaveBeenCalledTimes(2)
  })

  it("derives terminal status from aggregate, stages, and blocking delivery", () => {
    const projection = {
      occurrence_id: 1,
      run_id: 44,
      job_id: 7,
      artifact_status: "ready" as const,
      delivery_status: "failed" as const,
      stages: { persist_text: { status: "ready" as const, code: "persisted" } },
      output: { id: 90 },
      audio: null,
      editorial: {},
      selection: {},
      next_run_at: null,
      recovery: {}
    }

    expect(getPipelineWizardBriefingStatus(projection)).toBe("ready")
    expect(getPipelineWizardBriefingStatus(projection, true)).toBe("failed")
    expect(getPipelineWizardBriefingStatus({
      ...projection,
      artifact_status: "running",
      delivery_status: "waiting_for_artifacts",
      stages: { generate_audio: { status: "failed", code: "tts_unavailable" } }
    })).toBe("failed")
    expect(getPipelineWizardBriefingStatus({
      ...projection,
      artifact_status: "cancelled",
      delivery_status: "waiting_for_artifacts",
      stages: { generate_audio: { status: "cancelled", code: "user_cancelled" } }
    })).toBe("cancelled")
  })

  it("leaves advanced cron occurrence projection to the backend", () => {
    const base = {
      ...createDefaultPipelineWizardDraft(),
      scheduleMode: "advanced" as const,
      scheduleAdvancedCron: "0 8 * * MON-FRI",
      timezone: "UTC"
    }

    expect(projectPipelineWizardOccurrences(base, new Date("2026-07-12T07:59:00Z"))).toEqual({})
  })

  it("classifies briefing outcomes without embedding user-facing prose", () => {
    const projection = {
      occurrence_id: 1,
      run_id: 44,
      job_id: 7,
      artifact_status: "running" as const,
      delivery_status: "waiting_for_artifacts" as const,
      stages: { generate_audio: { status: "failed" as const, code: "tts_unavailable" } },
      output: null,
      audio: null,
      editorial: {},
      selection: {},
      next_run_at: null,
      recovery: {}
    }

    expect(getPipelineWizardBriefingOutcome(projection)).toEqual({
      status: "failed",
      stage: "generate_audio",
      code: "tts_unavailable",
      runId: 44
    })
    expect(JSON.stringify(getPipelineWizardBriefingOutcome(projection))).not.toContain("failed (")
  })

  it("normalizes internal speaker ids so duplicate ids never become user errors", () => {
    const speakers = normalizePipelineWizardSpeakers([
      { id: "host", label: "Host", voice: "alloy" },
      { id: "host", label: "Analyst", voice: "nova" },
      { id: "", label: "Guest", voice: "echo" }
    ])

    expect(speakers.map((speaker) => speaker.id)).toEqual(["host", "speaker_2", "speaker_3"])
    expect(new Set(speakers.map((speaker) => speaker.id))).toHaveProperty("size", 3)
  })
  it("validates source, monitor, digest, delivery, and optional audio requirements", () => {
    const base = createDefaultPipelineWizardDraft()

    expect(validatePipelineWizardDraft(base)).toEqual({
      valid: false,
      errors: expect.arrayContaining(["sourceIds", "monitorName"])
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
        createScheduledOutput: true,
        includeAudio: true,
        audioCast: {
          speaker_count: 2,
          speakers: [
            expect.objectContaining({ id: "host", label: "Host", role: "host", voice: "alloy" }),
            expect.objectContaining({ id: "analyst", label: "Analyst", role: "analyst", voice: "nova" })
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

  it("keeps required Reports output canonical for scheduled wizard monitors", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "UTC" })
          }) as Intl.DateTimeFormat
      )

    const scheduledDraft = {
      ...createDefaultPipelineWizardDraft(),
      sourceMode: "existing" as const,
      sourceIds: [10],
      monitorName: "Daily Brief",
      scheduleMode: "daily" as const,
      templateName: "briefing_md",
      audioEnabled: false,
      audioSpeakers: []
    }

    const expectedContract = toPipelineJobCreatePayload(
      toBriefingPipelineDraft(scheduledDraft)
    ).output_prefs?.briefing_pipeline
    expect(expectedContract).toMatchObject({
      text: { enabled: true, type: "briefing_markdown" },
      delivery: { reports: { enabled: true } }
    })
    expect(
      toPipelineJobCreatePayload(
        toBriefingPipelineDraft({
          ...scheduledDraft,
          createScheduledOutput: true
        })
      ).output_prefs?.briefing_pipeline
    ).toEqual(expectedContract)

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

    expect(
      toBriefingPipelineDraft({
        ...createDefaultPipelineWizardDraft(),
        sourceMode: "existing",
        sourceIds: [7],
        monitorName: "Manual Brief",
        scheduleMode: "manual",
        templateName: "briefing_md",
        audioEnabled: false,
        audioSpeakers: []
      })
    ).toEqual(
      expect.objectContaining({
        createScheduledOutput: false
      })
    )
  })
})
