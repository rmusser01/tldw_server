// @vitest-environment jsdom
/**
 * G19 follow-up to XS-10: while a reply is pending, Send queues the next
 * message, so Stop sits next to attach and the voice controls instead of
 * replacing them; the queued message can still get an image or dictation.
 */
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { deferred, mountReady, persistedReadyDraft, ready } from "./form.test-harness";
// After the harness, which must load the Form's module graph first.
import { useStoreMessageOption } from "@/store/option";

// Voice chat is available, so only the pending reply can disable it.
vi.mock("@/services/tldw/voice-conversation", async (original) => ({
  ...(await original<typeof import("@/services/tldw/voice-conversation")>()),
  resolveVoiceConversationAvailability: () => ({ available: true, reason: null, message: "" }),
}));

describe("Sidepanel composer controls while a reply is pending (G19)", () => {
  it("keeps attach and the voice controls next to Stop in the Pro composer", async () => {
    const providerRelease = deferred<void>();
    const fixture = await mountReady({ mode: "pro" });
    try {
      fixture.provider.mockImplementation(async function* () {
        await providerRelease.promise;
        yield { choices: [{ delta: { content: "synthetic answer" }, finish_reason: "stop" }] };
      });
      fireEvent.change(screen.getByTestId("chat-input"), { target: { value: "send this question" } });
      await waitFor(() => expect(persistedReadyDraft()).toBe("send this question"));
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));

      expect(screen.getByTestId("chat-stop-streaming")).toBeInTheDocument();
      expect(screen.getByTestId("chat-upload-image-inline")).toBeEnabled();
      expect(screen.getByRole("button", { name: "Voice chat settings" })).toBeInTheDocument();
      expect(screen.getByTestId("chat-character-controls-trigger")).toBeInTheDocument();
      // Voice chat stays off for the whole pending request, not only while
      // tokens stream: here the request is still pending but not streaming.
      act(() => {
        useStoreMessageOption.setState({ streaming: false, isProcessing: false });
      });
      expect(screen.getByTestId("chat-stop-streaming")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Voice chat" })).toBeDisabled();

      await act(async () => {
        providerRelease.resolve();
        await Promise.allSettled(ready.submissions);
      });
    } finally {
      providerRelease.resolve();
      await fixture.cleanup();
    }
  });
});
