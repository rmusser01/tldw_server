import { createInstance, type TFunction } from "i18next";
import { beforeAll, describe, expect, it, vi } from "vitest";
import ICU from "@/i18n/icu-format";
import option from "@/assets/locale/en/option.json";
import { formatFlashcardReviewGap } from "../date-display";

const reviewed = "2026-09-17T06:26:35.074Z";
const tenMinutesLater = "2026-09-17T06:36:35.074Z";
const learning = {
  queue_state: "learning",
  interval_days: 0,
  last_reviewed_at: reviewed,
  due_at: tenMinutesLater,
};

describe.each(["resource", "fallback"] as const)(
  "saved review gap with %s ICU messages",
  (mode) => {
    let t: TFunction;
    beforeAll(async () => {
      const instance = createInstance().use(ICU);
      await instance.init({
        lng: "en",
        fallbackLng: false,
        resources: mode === "resource" ? { en: { option } } : {},
      });
      t = instance.t.bind(instance);
    });

    it.each([
      {
        label: "one minute",
        change: { due_at: "2026-09-17T06:27:35.074Z" },
        full: "1 minute",
        compact: "1 min",
      },
      {
        label: "ten minutes",
        change: {},
        full: "10 minutes",
        compact: "10 min",
      },
      {
        label: "one hour",
        change: { due_at: "2026-09-17T07:26:35.074Z" },
        full: "1 hour",
        compact: "1 hr",
      },
      {
        label: "two hours",
        change: { due_at: "2026-09-17T08:26:35.074Z" },
        full: "2 hours",
        compact: "2 hr",
      },
      {
        label: "ninety minutes retain minute precision",
        change: { due_at: "2026-09-17T07:56:35.074Z" },
        full: "90 minutes",
        compact: "90 min",
      },
      {
        label: "one day learning step",
        change: { due_at: "2026-09-18T06:26:35.074Z" },
        full: "1 day",
        compact: "1d",
      },
      {
        label: "partial minute is never zero",
        change: { due_at: "2026-09-17T06:26:45.074Z" },
        full: "1 minute",
        compact: "1 min",
      },
      {
        label: "relearning overrides former days",
        change: { queue_state: "relearning", interval_days: 7 },
        full: "10 minutes",
        compact: "10 min",
      },
      {
        label: "learning overrides former days",
        change: { interval_days: 7 },
        full: "10 minutes",
        compact: "10 min",
      },
      {
        label: "zero-day legacy review",
        change: { queue_state: "review" },
        full: "10 minutes",
        compact: "10 min",
      },
      {
        label: "legacy response without state",
        change: { queue_state: undefined },
        full: "10 minutes",
        compact: "10 min",
      },
      {
        label: "review keeps one stored day",
        change: { queue_state: "review", interval_days: 1 },
        full: "1 day",
        compact: "1d",
      },
      {
        label: "review keeps stored days",
        change: { queue_state: "review", interval_days: 7 },
        full: "7 days",
        compact: "7d",
      },
      {
        label: "review days survive missing timestamps",
        change: {
          queue_state: "review",
          interval_days: 5,
          due_at: null,
          last_reviewed_at: null,
        },
        full: "5 days",
        compact: "5d",
      },
      {
        label: "time zone offsets describe the same gap",
        change: { last_reviewed_at: "2026-09-16T23:26:35.074-07:00" },
        full: "10 minutes",
        compact: "10 min",
      },
      {
        label: "epoch seconds and milliseconds",
        change: {
          last_reviewed_at: Date.parse(reviewed) / 1000,
          due_at: Date.parse(tenMinutesLater),
        },
        full: "10 minutes",
        compact: "10 min",
      },
    ])("formats $label", ({ change, full, compact }) => {
      expect(formatFlashcardReviewGap({ ...learning, ...change }, t)).toBe(
        full,
      );
      expect(
        formatFlashcardReviewGap({ ...learning, ...change }, t, {
          compact: true,
        }),
      ).toBe(compact);
    });

    it.each([
      { queue_state: "new" },
      { queue_state: "suspended", due_at: null },
      { due_at: null },
      { last_reviewed_at: null },
      { due_at: "invalid" },
      { last_reviewed_at: "invalid" },
      { due_at: reviewed },
      { due_at: "2026-09-17T06:25:35.074Z" },
      { queue_state: "relearning", interval_days: 7, due_at: null },
      { queue_state: "review", interval_days: Number.NaN },
      { queue_state: "review", interval_days: -1 },
    ])("does not invent an unavailable gap from %j", (change) => {
      expect(formatFlashcardReviewGap({ ...learning, ...change }, t)).toBe(
        "not available",
      );
      expect(
        formatFlashcardReviewGap({ ...learning, ...change }, t, {
          compact: true,
        }),
      ).toBe("—");
    });

    it("does not read the wall clock or mutate saved schedule fields", () => {
      const schedule = Object.freeze({ ...learning });
      const clock = vi.spyOn(Date, "now").mockImplementation(() => {
        throw Error("Display gap must not depend on now");
      });
      try {
        expect(formatFlashcardReviewGap(schedule, t)).toBe("10 minutes");
      } finally {
        clock.mockRestore();
      }
      expect(schedule).toEqual(learning);
    });

    it("updates singular and plural counts using the same translator instance", () => {
      expect(
        formatFlashcardReviewGap(
          { ...learning, due_at: "2026-09-17T06:27:35.074Z" },
          t,
        ),
      ).toBe("1 minute");
      expect(formatFlashcardReviewGap(learning, t)).toBe("10 minutes");
      expect(
        formatFlashcardReviewGap(
          { ...learning, queue_state: "review", interval_days: 1 },
          t,
        ),
      ).toBe("1 day");
      expect(
        formatFlashcardReviewGap(
          { ...learning, queue_state: "review", interval_days: 7 },
          t,
        ),
      ).toBe("7 days");
    });
  },
);
