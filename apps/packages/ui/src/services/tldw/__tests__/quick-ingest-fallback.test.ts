import { describe, expect, it } from "vitest";

import {
  normalizePersistentAddResponse,
  shouldFallbackToPersistentAdd,
} from "@/services/tldw/quick-ingest-fallback";

describe("quick ingest fallback helpers", () => {
  it("falls back when the queue endpoint returns a recognized concurrent-limit 429", () => {
    const error = Object.assign(
      new Error("Concurrent job limit reached: queue is full."),
      { status: 429 },
    );

    expect(shouldFallbackToPersistentAdd(error)).toBe(true);
  });

  it("does not fall back for unrelated 429 responses", () => {
    const error = Object.assign(new Error("Rate limited"), { status: 429 });

    expect(shouldFallbackToPersistentAdd(error)).toBe(false);
  });

  it("does not fall back for non-429 responses even if the text looks similar", () => {
    const error = Object.assign(
      new Error("Concurrent job limit reached: queue is full."),
      { status: 500 },
    );

    expect(shouldFallbackToPersistentAdd(error)).toBe(false);
  });

  it("normalizes legacy /media/add db_id fields into media_id", () => {
    const response = normalizePersistentAddResponse({
      results: [
        {
          status: "Success",
          db_id: 321,
        },
      ],
    });

    expect(response).toEqual({
      results: [
        {
          status: "Success",
          db_id: 321,
          media_id: 321,
        },
      ],
    });
  });
});
