import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const playgroundPath = path.resolve(testDir, "../Playground.tsx");
const cockpitShellPath = path.resolve(
  testDir,
  "../PlaygroundCockpitShell.tsx",
);

describe("Playground cockpit regression guard", () => {
  it("keeps the main /chat cockpit shell and rails wired into Playground", () => {
    const source = readFileSync(playgroundPath, "utf8");

    expect(source).toContain("PlaygroundCockpitShell");
    expect(source).toContain("PlaygroundContextRail");
    expect(source).toContain("PlaygroundRuntimeInspector");
    expect(source).toContain("CharacterControlRail");
    expect(source).toContain("<PlaygroundCockpitShell");
    expect(source).toContain("<PlaygroundContextRail");
    expect(source).toContain("<PlaygroundRuntimeInspector");
    expect(source).toContain("<CharacterControlRail");
  });

  it("keeps cockpit shell test ids and mobile rail state available", () => {
    const source = readFileSync(cockpitShellPath, "utf8");

    expect(source).toContain("playground-cockpit-shell");
    expect(source).toContain("playground-cockpit-left-rail");
    expect(source).toContain("playground-cockpit-right-rail");
    expect(source).toContain("playground-cockpit-mobile-rails");
    expect(source).toContain("Enter focus chat");
    expect(source).toContain("Show cockpit panels");
  });

  it("keeps focus mode as a reversible state rather than a separate route", () => {
    const source = readFileSync(playgroundPath, "utf8");

    expect(source).toContain("playgroundChatLayoutMode");
    expect(source).toContain('"cockpit"');
    expect(source).toContain('"focus"');
  });
});
