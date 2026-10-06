import "./dialogTestSetup";
import React from "react";
import { createPortal } from "react-dom";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { containDialogTab } from "../dialogKeyboard";

describe("dialog Tab boundaries", () => {
  it("skips hidden, disabled and negative-tabindex controls at both boundaries", async () => {
    const user = userEvent.setup();
    render(
      <div role="dialog" onKeyDown={containDialogTab}>
        <fieldset disabled>
          <button>Disabled group</button>
        </fieldset>
        <button disabled>Disabled first</button>
        <button hidden>Hidden first</button>
        <button>First</button>
        <input aria-label="Unavailable" tabIndex={-1} />
        <button>Last</button>
        <button style={{ display: "none" }}>Hidden last</button>
        <button disabled>Disabled last</button>
        <button style={{ visibility: "hidden" }}>Invisible last</button>
      </div>,
    );
    screen.getByRole("button", { name: "Last" }).focus();
    await user.tab();
    expect(screen.getByRole("button", { name: "First" })).toHaveFocus();
    await user.tab({ shift: true });
    expect(screen.getByRole("button", { name: "Last" })).toHaveFocus();
  });
  it("leaves nested portal controls to their owner", async () => {
    const user = userEvent.setup();
    render(
      <div role="dialog" onKeyDown={containDialogTab}>
        <button>Scope control</button>
        {createPortal(
          <div>
            <button>Portal first</button>
            <button>Portal last</button>
          </div>,
          document.body,
        )}
      </div>,
    );
    screen.getByRole("button", { name: "Portal first" }).focus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Portal last" })).toHaveFocus();
  });
});
