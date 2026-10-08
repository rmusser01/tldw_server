// @vitest-environment jsdom
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  deferred,
  mountDisconnected,
  mountReady,
  persistedReadyDraft,
  ready,
} from "./form.test-harness";
import { useStoreMessageOption } from "@/store/option";

describe("Sidepanel Form draft ownership", () => {
  it("mounts the real composer, message, metadata and queue hooks", async () => {
    const fixture = await mountDisconnected();
    try {
      expect(screen.getByTestId("chat-input")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Queue request" }),
      ).toBeInTheDocument();
    } finally {
      await fixture.cleanup();
    }
  });

  it("optimistically resets the captured text before a held provider response", async () => {
    const providerRelease = deferred<void>();
    const fixture = await mountReady();
    try {
      fixture.provider.mockImplementation(async function* () {
        await providerRelease.promise;
        yield {
          choices: [
            { delta: { content: "synthetic answer" }, finish_reason: "stop" },
          ],
        };
      });
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "send this question" },
      });
      await waitFor(() =>
        expect(persistedReadyDraft()).toBe("send this question"),
      );
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));
      expect(screen.getByTestId("chat-input")).toHaveValue("");
      let outcomes: PromiseSettledResult<void>[] = [];
      await act(async () => {
        providerRelease.resolve();
        outcomes = await Promise.allSettled(ready.submissions);
      });
      expect(outcomes.map((outcome) => outcome.status)).toEqual(["fulfilled"]);
    } finally {
      providerRelease.resolve();
      await fixture.cleanup();
    }
  });

  it("dispatches the original question after a held embedding lookup without a draft change", async () => {
    const embedding = deferred<string | null>();
    const providerRelease = deferred<void>();
    const fixture = await mountReady();
    try {
      ready.embedding.mockReturnValue(embedding.promise);
      fixture.provider.mockImplementation(async function* () {
        await providerRelease.promise;
        yield {
          choices: [
            { delta: { content: "synthetic answer" }, finish_reason: "stop" },
          ],
        };
      });
      act(() => useStoreMessageOption.getState().setWebSearch(true));
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "original web question" },
      });
      await waitFor(() =>
        expect(persistedReadyDraft()).toBe("original web question"),
      );
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(ready.embedding).toHaveBeenCalledTimes(1));
      await act(async () => {
        embedding.resolve("embedding-model");
      });
      await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));
      expect(JSON.stringify(fixture.provider.mock.calls[0][0])).toContain(
        "original web question",
      );
      expect(screen.getByTestId("chat-input")).toHaveValue("");
      let outcomes: PromiseSettledResult<void>[] = [];
      await act(async () => {
        providerRelease.resolve();
        outcomes = await Promise.allSettled(ready.submissions);
      });
      expect(outcomes.map((outcome) => outcome.status)).toEqual(["fulfilled"]);
    } finally {
      embedding.resolve("embedding-model");
      providerRelease.resolve();
      await fixture.cleanup();
    }
  });

  it("rejects a held preparation after the composer draft changes", async () => {
    const embedding = deferred<string | null>();
    const fixture = await mountReady();
    try {
      ready.embedding.mockReturnValue(embedding.promise);
      act(() => useStoreMessageOption.getState().setWebSearch(true));
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "first question" },
      });
      await waitFor(() => expect(persistedReadyDraft()).toBe("first question"));
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(ready.embedding).toHaveBeenCalledTimes(1));
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "new question" },
      });
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(ready.submissions);
      });
      expect(fixture.provider).not.toHaveBeenCalled();
      expect(screen.getByTestId("chat-input")).toHaveValue("new question");
    } finally {
      embedding.resolve("embedding-model");
      await fixture.cleanup();
    }
  });

  it.each(["completed", "rejected"])(
    "retains a newer saved draft after the earlier provider transport has %s",
    async (providerOutcome) => {
      const providerRelease = deferred<void>();
      const fixture = await mountReady();
      try {
        fixture.provider.mockImplementation(async function* () {
          await providerRelease.promise;
          if (providerOutcome === "rejected")
            throw new Error("Synthetic provider failure");
          yield {
            choices: [
              { delta: { content: "synthetic answer" }, finish_reason: "stop" },
            ],
          };
        });
        const input = screen.getByTestId("chat-input");
        fireEvent.change(input, { target: { value: "first question" } });
        await waitFor(() =>
          expect(persistedReadyDraft()).toBe("first question"),
        );
        fireEvent.click(screen.getByRole("button", { name: "Send message" }));
        await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));
        expect(input).toHaveValue("");
        fireEvent.change(input, { target: { value: "next unsent question" } });
        await waitFor(() =>
          expect(persistedReadyDraft()).toBe("next unsent question"),
        );
        let outcomes: PromiseSettledResult<void>[] = [];
        await act(async () => {
          providerRelease.resolve();
          outcomes = await Promise.allSettled(ready.submissions);
        });
        expect(outcomes).toHaveLength(1);
        if (providerOutcome === "rejected") {
          const outcome = outcomes[0];
          expect(outcome.status).toBe("rejected");
          if (outcome.status === "rejected") {
            expect(outcome.reason).toBeInstanceOf(Error);
            expect((outcome.reason as Error).message).toBe(
              "Stream completion failed",
            );
            expect(
              (outcome.reason as Error & { cause?: unknown }).cause,
            ).toEqual(new Error("Synthetic provider failure"));
          }
        } else {
          expect(outcomes[0].status).toBe("fulfilled");
        }
        expect(input).toHaveValue("next unsent question");
        expect(persistedReadyDraft()).toBe("next unsent question");
      } finally {
        providerRelease.resolve();
        await fixture.cleanup();
      }
    },
  );

  it("keeps the draft when conversation selection changes during preparation", async () => {
    const embedding = deferred<string | null>();
    const fixture = await mountReady();
    try {
      ready.embedding.mockReturnValue(embedding.promise);
      act(() => useStoreMessageOption.getState().setWebSearch(true));
      const input = screen.getByTestId("chat-input");
      fireEvent.change(input, { target: { value: "question for first chat" } });
      await waitFor(() =>
        expect(persistedReadyDraft()).toBe("question for first chat"),
      );
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(ready.embedding).toHaveBeenCalledTimes(1));
      act(() =>
        useStoreMessageOption.getState().setHistoryId("other-local-history"),
      );
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(ready.submissions);
      });
      expect(fixture.provider).not.toHaveBeenCalled();
      expect(input).toHaveValue("question for first chat");
      expect(persistedReadyDraft()).toBe("question for first chat");
    } finally {
      embedding.resolve("embedding-model");
      await fixture.cleanup();
    }
  });

  it("rejects a held preparation after edit and restore to the same text", async () => {
    const embedding = deferred<string | null>();
    const fixture = await mountReady();
    try {
      ready.embedding.mockReturnValue(embedding.promise);
      act(() => useStoreMessageOption.getState().setWebSearch(true));
      const input = screen.getByTestId("chat-input");
      fireEvent.change(input, { target: { value: "same question" } });
      await waitFor(() => expect(persistedReadyDraft()).toBe("same question"));
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(ready.embedding).toHaveBeenCalledTimes(1));
      fireEvent.change(input, { target: { value: "changed question" } });
      fireEvent.change(input, { target: { value: "same question" } });
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(ready.submissions);
      });
      expect(fixture.provider).not.toHaveBeenCalled();
      expect(input).toHaveValue("same question");
    } finally {
      embedding.resolve("embedding-model");
      await fixture.cleanup();
    }
  });

  it("allows one held preparation and prevents a duplicate Send", async () => {
    const embedding = deferred<string | null>();
    const fixture = await mountReady();
    try {
      ready.embedding.mockReturnValue(embedding.promise);
      act(() => useStoreMessageOption.getState().setWebSearch(true));
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "one question" },
      });
      await waitFor(() => expect(persistedReadyDraft()).toBe("one question"));
      const send = screen.getByRole("button", { name: "Send message" });
      fireEvent.click(send);
      fireEvent.click(send);
      await waitFor(() => expect(ready.embedding).toHaveBeenCalledTimes(1));
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "new question" },
      });
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(ready.submissions);
      });
      expect(ready.embedding).toHaveBeenCalledTimes(1);
      expect(fixture.provider).not.toHaveBeenCalled();
    } finally {
      embedding.resolve("embedding-model");
      await fixture.cleanup();
    }
  });

  it("rejects Alice's held preparation after the account changes", async () => {
    const embedding = deferred<string | null>();
    const fixture = await mountReady();
    try {
      ready.embedding.mockReturnValue(embedding.promise);
      act(() => useStoreMessageOption.getState().setWebSearch(true));
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "Alice private" },
      });
      await waitFor(() => expect(persistedReadyDraft()).toBe("Alice private"));
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(ready.embedding).toHaveBeenCalledTimes(1));
      act(() => {
        ready.account = "bob";
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"));
      });
      await waitFor(() =>
        expect(screen.getByTestId("chat-input")).toHaveValue(""),
      );
      fireEvent.change(screen.getByTestId("chat-input"), {
        target: { value: "Bob private" },
      });
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(ready.submissions);
      });
      expect(fixture.provider).not.toHaveBeenCalled();
      expect(screen.getByTestId("chat-input")).toHaveValue("Bob private");
    } finally {
      embedding.resolve("embedding-model");
      await fixture.cleanup();
    }
  });
});
