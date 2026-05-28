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
const playgroundFormPath = path.resolve(testDir, "../PlaygroundForm.tsx");

describe("Playground cockpit regression guard", () => {
  it("keeps the main /chat cockpit shell and core rails wired into Playground", () => {
    const source = readFileSync(playgroundPath, "utf8");

    expect(source).toContain("PlaygroundCockpitShell");
    expect(source).toContain("PlaygroundContextRail");
    expect(source).toContain("PlaygroundRuntimeInspector");
    expect(source).toContain("<PlaygroundCockpitShell");
    expect(source).toContain("<PlaygroundContextRail");
    expect(source).toContain("<PlaygroundRuntimeInspector");
  });

  it("does not render the standalone desktop character controls rail in /chat", () => {
    const source = readFileSync(playgroundPath, "utf8");

    expect(source).not.toContain("CharacterControlRail");
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

  it("keeps mobile cockpit mode from letting an expanded composer overlap the rails", () => {
    const source = readFileSync(playgroundPath, "utf8");
    const formSource = readFileSync(playgroundFormPath, "utf8");

    expect(source).toContain("mobileCockpitComposerConstrained");
    expect(source).toContain("mobileCockpitModeActive={mobileCockpitComposerConstrained}");
    expect(source).toContain("min-h-0 shrink overflow-y-auto overscroll-contain");
    expect(formSource).toContain("suppressComposerToolbarForMobileCockpit");
    expect(formSource).toContain('<div className="hidden">{composerToolbarNode}</div>');
    expect(formSource).toContain("facetsSlot={composerToolbarSlot}");
    expect(formSource).toContain("bottomBarSlot={composerToolbarSlot}");
  });

  it("keeps configured and catalog model scope controls wired into the selector", () => {
    const source = readFileSync(playgroundFormPath, "utf8");

    expect(source).toContain("<PlaygroundModelCatalogControls");
    expect(source).toContain("catalogControls={modelCatalogControls}");
  });
});
