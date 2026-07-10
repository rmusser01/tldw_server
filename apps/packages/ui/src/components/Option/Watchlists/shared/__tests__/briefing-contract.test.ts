import { describe, expect, it, vi } from "vitest"
import {
  buildBriefingPipelineContract,
  normalizeLegacyBriefingContract,
  toCanonicalWatchlistJobPayload,
  type BriefingSetupDraft,
  type WatchlistProgramFormat
} from "../briefing-contract"
import { toPipelineJobCreatePayload, type BriefingPipelineDraft } from "../../OverviewTab/pipeline-contract"
import { toQuickSetupJobPayload } from "../../OverviewTab/quick-setup"
import { buildWatchlistSetupJobPayload } from "../../SetupWizard/watchlist-setup-model"

const canonicalDraft: BriefingSetupDraft = {
  monitorName: "Morning Brief",
  scope: { sources: [12] },
  active: true,
  scheduleExpr: "0 8 * * *",
  timezone: "America/Los_Angeles",
  templateName: "briefing_markdown",
  templateFormat: "md",
  audioEnabled: true,
  audioVoice: "alloy",
  targetAudioMinutes: 8
}

describe("shared watchlists briefing contract", () => {
  it("produces the same contract from every setup adapter", () => {
    const timezoneSpy = vi
      .spyOn(Intl, "DateTimeFormat")
      .mockImplementation(
        () =>
          ({
            resolvedOptions: () => ({ timeZone: "America/Los_Angeles" })
          }) as Intl.DateTimeFormat
      )

    const expected = buildBriefingPipelineContract(canonicalDraft)
    const quickSetup = toQuickSetupJobPayload(
      {
        monitorName: "Morning Brief",
        schedulePreset: "daily",
        setupGoal: "briefing",
        includeAudioBriefing: true
      },
      [12]
    )
    const watchlistSetup = buildWatchlistSetupJobPayload(
      {
        preset: "general",
        startMode: "report_goal",
        name: "Morning Brief",
        objective: "",
        trackedScopeText: "",
        sourceUrlsText: "",
        monitorName: "Morning Brief",
        reportGoal: "",
        includeAudioBriefing: true,
        schedulePreset: "daily"
      },
      [12]
    )
    const pipelineDraft: BriefingPipelineDraft = {
      monitorName: "Morning Brief",
      sourceIds: [12],
      schedulePreset: "daily",
      templateName: "briefing_markdown",
      templateFormat: "md",
      includeAudio: true,
      audioVoice: "alloy",
      audioCast: {
        speaker_count: 1,
        speakers: [
          { id: "speaker_1", label: "Speaker 1", role: "host", voice: "alloy" }
        ]
      },
      voiceMap: { speaker_1: "alloy" },
      targetAudioMinutes: 8
    }
    const pipelineWizard = toPipelineJobCreatePayload(pipelineDraft)
    const jobEditor = toCanonicalWatchlistJobPayload(canonicalDraft)

    expect(quickSetup.output_prefs?.briefing_pipeline).toEqual(expected)
    expect(watchlistSetup.output_prefs?.briefing_pipeline).toEqual(expected)
    expect(pipelineWizard.output_prefs?.briefing_pipeline).toEqual(expected)
    expect(jobEditor.output_prefs?.briefing_pipeline).toEqual(expected)
    expect([
      quickSetup,
      watchlistSetup,
      pipelineWizard,
      jobEditor
    ].map((payload) => JSON.stringify(payload.output_prefs?.briefing_pipeline))).toEqual([
      JSON.stringify(expected),
      JSON.stringify(expected),
      JSON.stringify(expected),
      JSON.stringify(expected)
    ])

    timezoneSpy.mockRestore()
  })

  it("normalizes legacy preferences and preserves unrelated fields", () => {
    const normalized = normalizeLegacyBriefingContract(
      {
        auto_output: {
          enabled: true,
          type: "briefing_markdown",
          format: "html",
          template_name: "weekly_html"
        },
        generate_audio: true,
        target_audio_minutes: 20,
        audio_cast: {
          speaker_count: 2,
          speakers: [
            { id: "host", label: "Host", voice: "alloy" },
            { id: "analyst", label: "Analyst", voice: "nova" }
          ]
        },
        deliveries: {
          email: { enabled: true, recipients: ["analyst@example.com"] }
        },
        retention: { default_seconds: 604800 },
        custom_future_key: { keep: true }
      },
      { scheduled: true }
    )

    expect(normalized.contract).toMatchObject({
      version: 1,
      text: {
        enabled: true,
        format: "html",
        template_name: "weekly_html"
      },
      audio: {
        enabled: true,
        target_minutes: 20,
        cast: { speaker_count: 2 }
      },
      delivery: {
        reports: { enabled: true },
        email: { enabled: true, recipients: ["analyst@example.com"] },
        chatbook: { enabled: false }
      },
      test: { external_delivery: false, audio_sample_seconds: 60 }
    })
    expect(normalized.outputPrefs).toMatchObject({
      retention: { default_seconds: 604800 },
      custom_future_key: { keep: true },
      briefing_pipeline: normalized.contract
    })
    expect(normalized.outputPrefs).not.toHaveProperty("auto_output")
    expect(normalized.outputPrefs).not.toHaveProperty("generate_audio")
    expect(normalized.outputPrefs).not.toHaveProperty("deliveries")
    expect(normalized.warnings).toEqual(["legacy_briefing_preferences_normalized"])
  })

  it.each<WatchlistProgramFormat>([
    "concise_briefing",
    "solo_update",
    "host_discussion",
    "sportscast",
    "culture_roundtable",
    "custom"
  ])("supports the %s program format", (programFormat) => {
    expect(
      buildBriefingPipelineContract({
        ...canonicalDraft,
        programFormat,
        outcomeNoun: programFormat === "concise_briefing" ? "briefing" : "episode"
      }).editorial
    ).toMatchObject({
      program_format: programFormat,
      outcome_noun: programFormat === "concise_briefing" ? "briefing" : "episode"
    })
  })

  it("enforces canonical selection, audio, report, and test bounds", () => {
    const contract = buildBriefingPipelineContract({
      ...canonicalDraft,
      maxItems: 5000,
      audioEnabled: true,
      targetAudioMinutes: 0,
      emailEnabled: false,
      emailRecipients: ["ignored@example.com"],
      chatbookEnabled: false
    })

    expect(contract.selection.max_items).toBe(1000)
    expect(contract.audio.target_minutes).toBe(1)
    expect(contract.delivery).toEqual({
      reports: { enabled: true },
      email: { enabled: false, recipients: [] },
      chatbook: { enabled: false }
    })
    expect(contract.test).toEqual({
      external_delivery: false,
      audio_sample_seconds: 60
    })
  })

  it("sanitizes invalid canonical optional fields at the compatibility boundary", () => {
    const normalized = normalizeLegacyBriefingContract(
      {
        briefing_pipeline: {
          version: 1,
          selection: { mode: "automatic", max_items: 100 },
          editorial: { program_format: "concise_briefing", outcome_noun: "briefing" },
          text: {
            enabled: true,
            type: "briefing_markdown",
            format: "md",
            template_name: "briefing_markdown",
            template_version: -4,
            show_notes: false
          },
          audio: {
            enabled: false,
            language: "en",
            voice: " ",
            cast: { speaker_count: 1, speakers: [] },
            voice_map: { HOST: "" }
          },
          delivery: {
            reports: { enabled: true },
            email: { enabled: false, recipients: [] },
            chatbook: { enabled: false, title: " " }
          },
          test: { external_delivery: false, audio_sample_seconds: 60 }
        }
      },
      { scheduled: true }
    )

    expect(normalized.contract.text).not.toHaveProperty("template_version")
    expect(normalized.contract.audio).not.toHaveProperty("voice")
    expect(normalized.contract.audio).not.toHaveProperty("cast")
    expect(normalized.contract.audio).not.toHaveProperty("voice_map")
    expect(normalized.contract.delivery.chatbook).not.toHaveProperty("title")
  })
})
