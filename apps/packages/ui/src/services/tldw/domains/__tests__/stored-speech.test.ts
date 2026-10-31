import { beforeEach, describe, expect, it, vi } from "vitest";
import { modelsAudioMethods } from "../models-audio";

const wire = vi.hoisted(() => vi.fn());
vi.mock("@/services/background-proxy", () => ({
  bgRequest: wire,
  sanitizeResponseData: (value: unknown) => value,
}));
const scope = {
  config: { serverUrl: "https://owned.test", authMode: "multi-user" as const },
  userId: "alice",
};
const bytes = () => Uint8Array.from([1, 2, 3]).buffer;
const unscoped = vi.fn(() => {
  throw new Error("Unscoped speech request");
});
const client = {
  ensureConfigForRequest: unscoped,
  requestWithCurrentConfig: unscoped,
};
const stored = (options: Record<string, unknown> = {}) =>
  modelsAudioMethods.synthesizeSpeechDetailed.call(
    client,
    "Selene tours start Friday at 18:00.",
    {
      requestScope: scope,
      responseFormat: "wav",
      returnDownloadLink: true,
      ...options,
    },
  );
beforeEach(() => {
  wire.mockReset();
  unscoped.mockClear();
  wire.mockResolvedValue({
    ok: true,
    data: bytes(),
    headers: { "X-Generated-File-Id": "47", "Content-Type": "audio/wav" },
  });
});
describe("stored Studio speech acknowledgement", () => {
  it("requests stored speech without streaming on the captured account", async () => {
    await stored();
    expect(wire).toHaveBeenCalledTimes(1);
    expect(wire.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/audio/speech",
      method: "POST",
      body: { return_download_link: true, stream: false },
      servicePromptConfig: {
        serverUrl: "https://owned.test",
        expectedUserId: "alice",
      },
      headers: { Accept: "audio/wav", "X-TLDW-Expected-User-ID": "alice" },
    });
    expect(unscoped).not.toHaveBeenCalled();
  });
  it("returns the exact generated-file identity with the audio", async () => {
    const result = await stored();
    expect(result.generatedFileId).toBe(47);
    expect(new Uint8Array(result.buffer)).toEqual(Uint8Array.from([1, 2, 3]));
  });
  it.each([undefined, "0", "-1", "1.5", "9007199254740992", "/other/file"])(
    "rejects an absent or invalid stored-file identity %s",
    async (id) => {
      wire.mockResolvedValue({
        ok: true,
        data: bytes(),
        headers: id === undefined ? {} : { "x-generated-file-id": id },
      });
      await expect(stored()).rejects.toThrow(/saved|file/i);
    },
  );
  it("refuses a streaming storage request before dispatch", async () => {
    await expect(stored({ stream: true })).rejects.toThrow(/stream/i);
    expect(wire).not.toHaveBeenCalled();
  });

  it.each([
    {
      label: "empty bytes",
      data: new ArrayBuffer(0),
      contentType: "audio/wav",
    },
    { label: "non-audio response", data: bytes(), contentType: "text/plain" },
  ])(
    "rejects unusable stored speech: $label",
    async ({ data, contentType }) => {
      wire.mockResolvedValue({
        ok: true,
        data,
        headers: { "X-Generated-File-Id": "47", "Content-Type": contentType },
      });
      await expect(stored()).rejects.toThrow(/audio/i);
    },
  );
  it("keeps ordinary speech opt-in behavior unchanged", async () => {
    const result = await stored({ returnDownloadLink: false });
    expect(result).not.toHaveProperty("generatedFileId");
    expect(wire.mock.calls[0][0].body).not.toHaveProperty(
      "return_download_link",
    );
    expect(wire.mock.calls[0][0].body).not.toHaveProperty("stream");
  });
});
