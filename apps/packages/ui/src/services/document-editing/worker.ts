/** Pure document worker. Never import application, storage or authentication code. */
import { checkDocument, formatDocument } from "./engine";
import { createDocumentRequest } from "./types";

const MAX_FRAME = 6 * 1024 * 1024 + 65536;
self.onmessage = (event: MessageEvent<unknown>) => {
  try {
    if (
      !(event.data instanceof ArrayBuffer) ||
      event.data.byteLength > MAX_FRAME
    )
      throw new Error("Invalid frame");
    const value = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(event.data),
    );
    if (value.operation !== "check" && value.operation !== "format")
      throw new Error("Invalid operation");
    const request = createDocumentRequest(value.request);
    const result =
      value.operation === "check"
        ? checkDocument(request)
        : formatDocument(request);
    const frame = new TextEncoder().encode(JSON.stringify(result)).buffer;
    if (frame.byteLength > MAX_FRAME) throw new Error("Reply exceeds limit");
    self.postMessage(frame, { transfer: [frame] });
  } catch {
    // Fixed malformed frame: the parent returns its source-free failure result.
    self.postMessage(null);
  }
};
