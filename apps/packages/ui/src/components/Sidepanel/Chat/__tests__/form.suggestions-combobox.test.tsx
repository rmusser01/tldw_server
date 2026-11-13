// @vitest-environment jsdom
/**
 * AX-12: the side panel's slash menu was option buttons with no listbox and
 * no ids, and the textarea did not point into it, so a screen reader stayed
 * silent while the user arrowed through commands.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { mountReady } from "./form.test-harness";

describe("Sidepanel composer suggestions (AX-12)", () => {
  it("makes the textarea a combobox that controls the slash command listbox", async () => {
    const fixture = await mountReady();
    try {
      const textarea = screen.getByTestId("chat-input");
      expect(textarea).toHaveAttribute("role", "combobox");
      expect(textarea).toHaveAttribute("aria-expanded", "false");
      expect(textarea).not.toHaveAttribute("aria-controls");

      fireEvent.change(textarea, { target: { value: "/" } });
      const listbox = await screen.findByRole("listbox", { name: "Commands" });
      expect(textarea).toHaveAttribute("aria-expanded", "true");
      expect(textarea).toHaveAttribute("aria-controls", listbox.id);
      const options = within(listbox).getAllByRole("option");
      expect(options.length).toBeGreaterThan(1);
      expect(textarea).toHaveAttribute("aria-activedescendant", options[0].id);

      fireEvent.keyDown(textarea, { key: "ArrowDown" });
      await waitFor(() =>
        expect(textarea).toHaveAttribute("aria-activedescendant", options[1].id),
      );
      expect(options[1]).toHaveAttribute("aria-selected", "true");
    } finally {
      await fixture.cleanup();
    }
  });
});
