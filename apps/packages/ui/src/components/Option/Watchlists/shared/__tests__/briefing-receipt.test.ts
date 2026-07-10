import { describe, expect, it } from "vitest"
import { buildBriefingPipelineContract } from "../briefing-contract"
import { buildBriefingReceiptModel } from "../briefing-receipt"

describe("watchlists briefing receipt", () => {
  it("describes a two-host sportscast with target duration and timezone", () => {
    const contract = buildBriefingPipelineContract({
      monitorName: "Purple and Gold Weekly",
      scope: { sources: [1, 2, 3, 4, 5, 6, 7, 8] },
      active: true,
      scheduleExpr: "0 18 * * SUN",
      timezone: "America/Los_Angeles",
      programFormat: "sportscast",
      outcomeNoun: "episode",
      showName: "Purple and Gold Weekly",
      templateName: "briefing_markdown",
      templateFormat: "md",
      showNotes: true,
      audioEnabled: true,
      targetAudioMinutes: 20,
      audioCast: {
        speaker_count: 2,
        speakers: [
          { id: "host", label: "Host", role: "host", voice: "alloy" },
          { id: "analyst", label: "Analyst", role: "analyst", voice: "nova" }
        ]
      }
    })

    const receipt = buildBriefingReceiptModel({
      contract,
      sourceCount: 8,
      nextRunAt: "2026-07-12T18:00:00-07:00",
      timezone: "America/Los_Angeles"
    })

    expect(receipt).toMatchObject({
      outcomeNoun: "episode",
      programFormat: "sportscast",
      speakerCount: 2,
      targetMinutes: 20,
      sourceCount: 8,
      nextRunAt: "2026-07-12T18:00:00-07:00",
      timezone: "America/Los_Angeles",
      timezoneAbbreviation: "PDT"
    })
    expect(receipt.sentence).toContain(
      "Sunday, July 12 at 6:00 PM PDT (America/Los_Angeles)"
    )
    expect(receipt.sentence).toContain("8 sources")
    expect(receipt.sentence).toContain("two-host sportscast")
    expect(receipt.sentence).toContain("targeting 20 minutes")
    expect(receipt.sentence).toContain("save both in Reports")
  })

  it("reports the DST offset change between consecutive occurrences", () => {
    const contract = buildBriefingPipelineContract({
      monitorName: "Daily Brief",
      scope: { sources: [1] },
      active: true,
      scheduleExpr: "0 1 * * *",
      timezone: "America/Los_Angeles",
      templateName: "briefing_markdown",
      audioEnabled: false
    })

    const receipt = buildBriefingReceiptModel({
      contract,
      sourceCount: 1,
      nextRunAt: "2026-10-31T01:00:00-07:00",
      followingRunAt: "2026-11-02T01:00:00-08:00",
      timezone: "America/Los_Angeles"
    })

    expect(receipt.timezoneAbbreviation).toBe("PDT")
    expect(receipt.dstNote).toContain("PST")
    expect(receipt.dstNote).toContain("America/Los_Angeles")
  })

  it("names deterministic reviewed email and Chatbook destinations", () => {
    const contract = buildBriefingPipelineContract({
      monitorName: "Delivery Brief",
      scope: { sources: [1, 2] },
      active: true,
      scheduleExpr: "0 9 * * *",
      timezone: "UTC",
      templateName: "briefing_markdown",
      audioEnabled: true,
      targetAudioMinutes: 8,
      emailEnabled: true,
      emailRecipients: ["zeta@example.com", "alpha@example.com"],
      chatbookEnabled: true,
      chatbookTitle: "Morning Review"
    })

    const receipt = buildBriefingReceiptModel({
      contract,
      sourceCount: 2,
      nextRunAt: "2026-07-12T09:00:00Z",
      timezone: "UTC"
    })

    expect(receipt.emailRecipients).toEqual([
      "alpha@example.com",
      "zeta@example.com"
    ])
    expect(receipt.chatbookTitle).toBe("Morning Review")
    expect(receipt.sentence).toContain(
      "email the outcome to alpha@example.com and zeta@example.com"
    )
    expect(receipt.sentence).toContain('save it to Chatbook “Morning Review”')
    expect(receipt.sentence).toContain("targeting 8 minutes")
    expect(receipt.sentence).toContain("Reports")
  })

  it("formats 24-hour locales without an undefined day period", () => {
    const contract = buildBriefingPipelineContract({
      monitorName: "UK Brief",
      scope: { sources: [1] },
      active: true,
      templateName: "briefing_markdown",
      audioEnabled: false
    })

    const receipt = buildBriefingReceiptModel({
      contract,
      sourceCount: 1,
      nextRunAt: "2026-07-12T18:00:00+01:00",
      timezone: "Europe/London",
      locale: "en-GB"
    })

    expect(receipt.nextRunLabel).not.toContain("undefined")
    expect(receipt.sentence).not.toContain("undefined")
    expect(receipt.nextRunLabel).toContain("18:00")
  })
})
