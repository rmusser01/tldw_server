// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { focusFirstVisibleElement } from "../focus-return";

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
});
