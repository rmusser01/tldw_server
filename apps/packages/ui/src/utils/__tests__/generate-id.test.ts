import { afterEach, expect, it, vi } from "vitest";
import { generateID as legacyGenerateID } from "@/db/dexie/helpers";
import { generateID } from "@/utils/generate-id";

afterEach(() => vi.restoreAllMocks());

it.each([
  [0, "pa_0000-0000-000-0000"],
  [0.5, "pa_8888-8888-888-8888"],
  [0.999999999999, "pa_ffff-ffff-fff-ffff"],
] as const)(
  "preserves the existing ID projection for random value %s",
  (random, expected) => {
    vi.spyOn(Math, "random").mockReturnValue(random);
    expect(generateID()).toBe(expected);
  },
);

it("preserves the legacy database helper export", () => {
  expect(legacyGenerateID).toBe(generateID);
});
