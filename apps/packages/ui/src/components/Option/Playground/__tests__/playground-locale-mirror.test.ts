import { readFileSync } from "node:fs";
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

  return Object.entries(value as Record<string, unknown>).reduce(
    (acc, [key, nested]) => ({
      ...acc,
      ...flattenNested(nested, [...prefix, key]),
    }),
    {} as Record<string, string>,
  );
};

describe("playground locale mirror parity", () => {
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
});
