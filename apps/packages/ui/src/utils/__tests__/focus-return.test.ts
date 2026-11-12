// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import {
  focusFirstVisibleElement,
  normalizeFocusSelector,
  scheduleFocusFirstVisibleElement,
} from "../focus-return";

describe("focus-return", () => {
  it("prefers the first visible enabled selector match", () => {
    document.body.innerHTML = `
      <button data-return-target style="display: none">Hidden target</button>
      <button data-return-target disabled>Disabled target</button>
      <button data-return-target>Visible target</button>
    `;

    expect(focusFirstVisibleElement("[data-return-target]")).toBe(true);

    expect(document.activeElement?.textContent).toBe("Visible target");
  });

  it("falls back to the first match when no visible candidate is available", () => {
    document.body.innerHTML = `
      <button data-return-target style="display: none">Hidden target</button>
    `;

    expect(focusFirstVisibleElement("[data-return-target]")).toBe(true);

    expect(document.activeElement?.textContent).toBe("Hidden target");
  });

  it("ignores malformed selectors instead of throwing", () => {
    document.body.innerHTML = `
      <button data-return-target>Visible target</button>
    `;

    expect(() => focusFirstVisibleElement("[")).not.toThrow();
    expect(focusFirstVisibleElement("[")).toBe(false);
    expect(document.activeElement).toBe(document.body);
  });

  it("normalizes event-provided focus selectors before use", () => {
    expect(normalizeFocusSelector("  [data-return-target]  ")).toBe(
      "[data-return-target]",
    );
    expect(normalizeFocusSelector("   ")).toBeNull();
    expect(normalizeFocusSelector(null)).toBeNull();
  });
});

it.each([true, false])(
  "scheduled focus respects its original owner (current %s)",
  (current) => {
    document.body.innerHTML =
      "<button data-return-target>Prompt picker</button><input data-new-focus />";
    let runFrame!: () => void;
    const frame = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback) => {
        runFrame = () => callback(0);
        return 1;
      });
    try {
      const input =
        document.querySelector<HTMLInputElement>("[data-new-focus]")!;
      input.focus();
      scheduleFocusFirstVisibleElement(
        "[data-return-target]",
        undefined,
        () => current,
      );
      runFrame();
      expect(document.activeElement).toBe(
        current ? document.querySelector("[data-return-target]") : input,
      );
    } finally {
      frame.mockRestore();
    }
  },
);
