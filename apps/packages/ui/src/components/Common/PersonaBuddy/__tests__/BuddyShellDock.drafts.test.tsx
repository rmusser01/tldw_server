import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { BuddyShellDock } from "../BuddyShellDock";

const buildProps = () => ({
  buddySummary: {
    has_buddy: true,
    persona_name: "Migu",
    role_summary: null,
    visual: null,
  },
  personaId: "persona-1",
  isOpen: true,
  position: { x: 16, y: 16 },
  onOpenControls: vi.fn(),
  onCloseControls: vi.fn(),
  onBuddyPointerDown: vi.fn(),
  onBuddyKeyDown: vi.fn(),
  dockRef: React.createRef<HTMLDivElement>(),
  liveControl: {
    focusedSession: {
      sessionId: "session-1",
      personaId: "persona-1",
      personaName: "Migu",
      lifecycle: "connected" as const,
      pendingApprovalCount: 0,
    },
    sessions: [],
    focusedSessionId: "session-1",
    pendingFocusSessionId: null,
    streamState: "open" as const,
    canSendText: true,
    startTextSession: vi.fn(),
    stopSession: vi.fn(),
    focusSession: vi.fn(),
    sendText: vi.fn(),
  },
});

const dock = (props: React.ComponentProps<typeof BuddyShellDock>) => (
  <MemoryRouter>
    <BuddyShellDock {...props} />
  </MemoryRouter>
);

describe("Buddy draft lifetime", () => {
  it.each([false, true])(
    "retains a closed draft with dormant=%s without exposing controls",
    (isDormant) => {
      const props = buildProps();
      const { rerender } = render(dock(props));
      fireEvent.change(
        screen.getByRole("textbox", { name: "Message your Buddy" }),
        {
          target: { value: "Keep this unsent draft" },
        },
      );

      rerender(dock({ ...props, isOpen: false, isDormant }));
      expect(
        screen.queryByRole("textbox", { name: "Message your Buddy" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Send" }),
      ).not.toBeInTheDocument();
      rerender(dock(props));

      expect(
        screen.getByRole("textbox", { name: "Message your Buddy" }),
      ).toHaveValue("Keep this unsent draft");
    },
  );

  it("retains the in-flight draft and retry identity across closing", async () => {
    const props = buildProps();
    let finish!: (result: {
      ok: boolean;
      clientMessageId: string;
      error: string;
    }) => void;
    props.liveControl.sendText.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const { rerender } = render(dock(props));
    fireEvent.change(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
      {
        target: { value: "Retry this message" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() =>
      expect(props.liveControl.sendText).toHaveBeenCalledTimes(1),
    );
    const requestIdentity = props.liveControl.sendText.mock.calls[0][1];

    rerender(dock({ ...props, isOpen: false }));
    rerender(dock(props));
    expect(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
    ).toHaveValue("Retry this message");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await act(async () =>
      finish({
        ok: false,
        clientMessageId: requestIdentity.clientMessageId,
        error: "Connection interrupted",
      }),
    );
    expect(screen.getByText("Connection interrupted")).toBeVisible();

    props.liveControl.sendText.mockResolvedValue({
      ok: true,
      clientMessageId: requestIdentity.clientMessageId,
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() =>
      expect(props.liveControl.sendText).toHaveBeenCalledTimes(2),
    );
    expect(props.liveControl.sendText).toHaveBeenLastCalledWith(
      "Retry this message",
      requestIdentity,
    );
    await waitFor(() =>
      expect(
        screen.getByRole("textbox", { name: "Message your Buddy" }),
      ).toHaveValue(""),
    );
  });

  it("does not expose one Persona's draft after another Persona is selected", () => {
    const props = buildProps();
    const { rerender } = render(dock(props));
    fireEvent.change(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
      {
        target: { value: "Persona one's draft" },
      },
    );
    rerender(dock({ ...props, personaId: "persona-2" }));
    expect(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
    ).toHaveValue("");
    rerender(dock(props));
    expect(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
    ).toHaveValue("");
  });

  it("drops the draft when its Buddy lifetime ends", () => {
    const props = buildProps();
    const { unmount } = render(dock(props));
    fireEvent.change(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
      {
        target: { value: "Previous owner draft" },
      },
    );
    unmount();
    render(dock(props));
    expect(
      screen.getByRole("textbox", { name: "Message your Buddy" }),
    ).toHaveValue("");
  });
});
