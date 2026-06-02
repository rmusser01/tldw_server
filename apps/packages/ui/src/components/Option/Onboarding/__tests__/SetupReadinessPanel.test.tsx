// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { SetupReadinessStatusResponse } from "@/services/tldw/setup-readiness";
import { SetupReadinessPanel } from "../SetupReadinessPanel";

const readinessStatus: SetupReadinessStatusResponse = {
  readiness_status: "ready_with_warnings",
  lane_ids: ["chat", "embeddings_rag", "speech"],
  supported_statuses: [],
  supported_overlays: [],
  active_overlays: ["restart_required", "downloads_disabled"],
  overlays: ["network_unavailable"],
  lanes: [
    {
      lane_id: "chat",
      label: "Chat",
      status: "ready_with_warnings",
      warnings: ["Hosted provider key will be verified on first chat."],
      blockers: [],
      consequences: [],
    },
    {
      lane_id: "embeddings_rag",
      label: "Embeddings/RAG",
      status: "not_configured",
      warnings: [],
      blockers: [],
      consequences: [
        "RAG search will be limited until embeddings are configured.",
      ],
    },
    {
      lane_id: "speech",
      label: "Speech",
      status: "blocked",
      warnings: ["Speech bundle needs provisioning."],
      blockers: ["Package installs are disabled."],
      consequences: ["Transcription can be configured later."],
    },
  ],
};

describe("SetupReadinessPanel", () => {
  it("renders backend lane labels and statuses", () => {
    render(<SetupReadinessPanel status={readinessStatus} />);

    expect(screen.getByText("Chat")).toBeInTheDocument();
    expect(screen.getByText("Embeddings/RAG")).toBeInTheDocument();
    expect(screen.getByText("Speech")).toBeInTheDocument();
    expect(screen.getByText("ready with warnings")).toBeInTheDocument();
    expect(screen.getByText("not configured")).toBeInTheDocument();
    expect(screen.getByText("blocked")).toBeInTheDocument();
  });

  it("shows lane warnings, blockers, and consequences inside compact details", () => {
    render(<SetupReadinessPanel status={readinessStatus} />);

    const details = screen
      .getByText(/speech details/i)
      .closest("details");
    expect(details).not.toHaveAttribute("open");

    fireEvent.click(screen.getByText(/speech details/i));

    expect(
      screen.getByText("Speech bundle needs provisioning."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Package installs are disabled."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Transcription can be configured later."),
    ).toBeInTheDocument();
  });

  it("renders overlays with readable labels", () => {
    render(<SetupReadinessPanel status={readinessStatus} />);

    expect(screen.getByText("Restart required")).toBeInTheDocument();
    expect(screen.getByText("Downloads disabled")).toBeInTheDocument();
    expect(screen.getByText("Network unavailable")).toBeInTheDocument();
  });

  it("marks embeddings and speech as deferrable when optional lanes are unavailable", () => {
    render(<SetupReadinessPanel status={readinessStatus} />);

    const embeddingsLane = screen.getByTestId(
      "setup-readiness-lane-embeddings_rag",
    );
    const speechLane = screen.getByTestId("setup-readiness-lane-speech");

    expect(within(embeddingsLane).getByText(/deferrable/i)).toBeInTheDocument();
    expect(within(speechLane).getByText(/deferrable/i)).toBeInTheDocument();
  });

  it("marks non-ready chat states as blocking first chat", () => {
    render(
      <SetupReadinessPanel
        status={{
          ...readinessStatus,
          lanes: [
            {
              lane_id: "chat",
              label: "Chat",
              status: "not_configured",
            },
          ],
        }}
      />,
    );

    const chatLane = screen.getByTestId("setup-readiness-lane-chat");

    expect(within(chatLane).getByText(/blocks first chat/i)).toBeInTheDocument();
  });

  it("renders duplicate detail messages without React key warnings", () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);

    render(
      <SetupReadinessPanel
        status={{
          ...readinessStatus,
          lanes: [
            {
              lane_id: "chat",
              label: "Chat",
              status: "failed",
              warnings: [
                "Retry provider validation.",
                "Retry provider validation.",
              ],
            },
          ],
        }}
      />,
    );

    expect(
      consoleError.mock.calls.some((call) =>
        String(call[0]).includes("Encountered two children with the same key"),
      ),
    ).toBe(false);
    consoleError.mockRestore();
  });

  it("calls retry handler from the retry button", () => {
    const onRetry = vi.fn();

    render(
      <SetupReadinessPanel
        status={readinessStatus}
        error="Setup readiness could not be loaded."
        onRetry={onRetry}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("handles loading and error states without a readiness payload", () => {
    render(
      <SetupReadinessPanel
        status={null}
        loading
        error="Setup readiness could not be loaded."
      />,
    );

    expect(screen.getByText(/checking setup readiness/i)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      /setup readiness could not be loaded/i,
    );
  });

  it("does not render an empty panel when readiness is unavailable", () => {
    const { container } = render(<SetupReadinessPanel status={null} />);

    expect(container).toBeEmptyDOMElement();
  });
});
