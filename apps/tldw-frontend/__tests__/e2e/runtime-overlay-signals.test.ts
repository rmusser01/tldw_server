import { describe, expect, it } from "vitest";
import {
  hasRuntimeOverlayBodySignal,
  hasRuntimeOverlayConsoleSignal,
  hasTransientRuntimeOverlaySignal,
} from "../../e2e/smoke/runtime-overlay";

describe("runtime overlay smoke signals", () => {
  it("does not treat chat cockpit runtime rail copy as a framework overlay", () => {
    const chatRailBodyText = [
      "Chat cockpit",
      "Context and runtime rails visible.",
      "Runtime",
      "Runtime",
      "Error",
      "Choose model",
      "No provider selected",
      "No model selected",
    ].join(" ");

    expect(hasRuntimeOverlayBodySignal(chatRailBodyText)).toBe(false);
  });

  it("still detects framework overlay signals in body text and console text", () => {
    const nextRuntimeOverlayText =
      "Unhandled Runtime Error TypeError: message.error is not a function";

    expect(
      hasRuntimeOverlayBodySignal(nextRuntimeOverlayText),
    ).toBe(true);
    expect(hasRuntimeOverlayBodySignal("Build Error failed to compile")).toBe(
      true,
    );
    expect(hasRuntimeOverlayConsoleSignal("Unhandled Runtime Error")).toBe(true);
    expect(hasTransientRuntimeOverlaySignal("Runtime SyntaxError")).toBe(true);
  });
});
