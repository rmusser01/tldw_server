import option from "@/assets/locale/en/option.json"
import i18n, { ensureI18nNamespaces } from "@/i18n"
import { describe, expect, it } from "vitest"

const requiredInspectionKeys = [
  "title",
  "regionAria",
  "readyLabel",
  "readyMessage",
  "unavailableLabel",
  "unavailableMessage",
  "failedLabel",
  "failedMessage",
  "blockedLabel",
  "blockedMessage",
  "expiredLabel",
  "expiredMessage",
  "cancelledLabel",
  "cancelledMessage",
  "inspectingLabel",
  "inspectingMessage",
  "cancel",
  "cancelAria",
  "retry",
  "retryAria",
  "remove",
  "removeAria",
  "moreNotLoaded",
  "sessionDuplicates"
] as const

const requiredPreflightKeys = [
  "detected",
  "details",
  "refresh",
  "refreshAria",
  "itemCount",
  "selectedCount",
  "duplicateCount",
  "selectionWarning",
  "selectAll",
  "selectNone",
  "selectNew",
  "filterAria",
  "filterAll",
  "filterNew",
  "filterDuplicates",
  "filterUnavailable",
  "listAria",
  "untitled",
  "duplicate",
  "duplicateUnknown",
  "availabilityAvailable",
  "availabilityDeleted",
  "availabilityNeedsAuth",
  "availabilityPremiumOnly",
  "availabilityPrivate",
  "availabilitySubscriberOnly",
  "availabilityUnavailable",
  "availabilityUnknown",
  "itemSelectionAria",
  "itemDetails",
  "selectionStatus",
  "addVideos",
  "loadThumbnailAria",
  "loadThumbnail",
  "thumbnailAlt",
  "thumbnailUnavailable"
] as const

describe("English Quick Ingest playlist copy", () => {
  it("defines every typed inspection status and user action used by the playlist UI", () => {
    const quickIngest = option.quickIngest as Record<string, unknown>
    const inspection = quickIngest.playlistInspection as Record<string, unknown>
    const preflight = quickIngest.playlistPreflight as Record<string, unknown>

    expect(inspection).toBeTypeOf("object")
    expect(preflight).toBeTypeOf("object")
    for (const key of requiredInspectionKeys) {
      expect(
        inspection[key],
        `quickIngest.playlistInspection.${key}`
      ).toBeTypeOf("string")
    }
    for (const key of requiredPreflightKeys) {
      expect(preflight[key], `quickIngest.playlistPreflight.${key}`).toBeTypeOf(
        "string"
      )
    }
  })

  it("formats repeated row names and changing counts from each call instead of the first cached value", async () => {
    await ensureI18nNamespaces(["option"], "en")

    expect(
      i18n.t("quickIngest.playlistPreflight.itemSelectionAria", {
        ns: "option",
        ordinal: 1,
        title: "Talk 1"
      })
    ).toBe("Select playlist item 1: Talk 1")
    expect(
      i18n.t("quickIngest.playlistPreflight.itemSelectionAria", {
        ns: "option",
        ordinal: 3,
        title: "Talk 3"
      })
    ).toBe("Select playlist item 3: Talk 3")
    expect(
      i18n.t("quickIngest.playlistPreflight.selectedCount", {
        ns: "option",
        count: 32
      })
    ).toBe("32 selected")
    expect(
      i18n.t("quickIngest.playlistPreflight.selectedCount", {
        ns: "option",
        count: 31
      })
    ).toBe("31 selected")
  })
})
