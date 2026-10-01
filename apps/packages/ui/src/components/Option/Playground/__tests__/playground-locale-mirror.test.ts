import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

type NestedLocaleJson = Record<string, unknown>;
type ExtensionLocaleJson = Record<string, { message?: unknown }>;

const testDir = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(testDir, "../../../../");

const playgroundLocale = JSON.parse(
  readFileSync(
    path.resolve(srcRoot, "assets/locale/en/playground.json"),
    "utf8",
  ),
) as NestedLocaleJson;

const extensionPlaygroundLocale = JSON.parse(
  readFileSync(
    path.resolve(srcRoot, "public/_locales/en/playground.json"),
    "utf8",
  ),
) as ExtensionLocaleJson;

const flattenNested = (
  value: unknown,
  prefix: string[] = [],
): Record<string, string> => {
  if (typeof value === "string") {
    return { [prefix.join("_")]: value };
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  // Chrome message names allow only [A-Za-z0-9_], so apps/extension/scripts/
  // sync-public-locales.js writes "preset.sliceOfLife" as "preset_sliceOfLife".
  return Object.entries(value as Record<string, unknown>).reduce(
    (acc, [key, nested]) => ({
      ...acc,
      ...flattenNested(nested, [...prefix, key.replace(/[^A-Za-z0-9_]/g, "_")]),
    }),
    {} as Record<string, string>,
  );
};

describe("playground locale mirror parity", () => {
  it("localizes all media handoff conflict choices in supported locales", () => {
    const localesRoot = path.resolve(srcRoot, "assets/locale");
    const keys = ["conflictLabel", "conflict", "insert", "replace", "cancel"];
    for (const locale of readdirSync(localesRoot)) {
      const copy = JSON.parse(
        readFileSync(path.join(localesRoot, locale, "playground.json"), "utf8"),
      );
      for (const key of keys) {
        expect(
          copy.mediaHandoff?.[key],
          `${locale}:mediaHandoff.${key}`,
        ).toEqual(expect.any(String));
        expect(copy.mediaHandoff[key].trim()).not.toBe("");
        if (locale !== "en") {
          expect(copy.mediaHandoff[key]).not.toBe(
            (playgroundLocale.mediaHandoff as Record<string, string>)[key],
          );
        }
      }
    }
  });

  it("mirrors nested English playground strings into extension locale messages", () => {
    const flattenedNested = flattenNested(playgroundLocale);
    const extensionMessages = Object.fromEntries(
      Object.entries(extensionPlaygroundLocale).map(([key, value]) => [
        key,
        String(value?.message ?? ""),
      ]),
    );

    for (const [key, value] of Object.entries(flattenedNested)) {
      expect(extensionMessages[key]).toBe(value);
    }
  });

  it("keeps all shared workspace English copy mirrored exactly", () => {
    const sharedWorkspace = playgroundLocale.sharedWorkspace;
    expect(sharedWorkspace).toBeTruthy();

    const flattenedSharedWorkspace = flattenNested(sharedWorkspace, [
      "sharedWorkspace",
    ]);
    expect(Object.keys(flattenedSharedWorkspace).length).toBeGreaterThan(20);

    for (const [key, value] of Object.entries(flattenedSharedWorkspace)) {
      expect(extensionPlaygroundLocale[key]?.message).toBe(value);
    }
  });
});
