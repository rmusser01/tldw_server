import { describe, expect, it } from "vitest";
import { assertNoCriticalDiagnostics } from "../e2e/onboarding-uat/helpers";

describe("onboarding diagnostics", () => {
  it("fails on chat global settings 404 instead of exempting ephemeral-link regressions", () => {
    expect(() => assertNoCriticalDiagnostics({
      console: [{ type: "error", text: "Failed to load resource: the server responded with a status of 404 (Not Found)", location: { url: "http://server/api/v1/chats/ephemeral/settings?scope_type=global", lineNumber: 0 } }],
      pageErrors: [], requestFailures: []
    })).toThrow("consoleErrors=1");
  });
  it("preserves the existing model metadata availability allowance", () => {
    expect(() => assertNoCriticalDiagnostics({
      console: [{ type: "error", text: "Failed to fetch chat models: Error: Failed to fetch (GET /api/v1/llm/models/metadata)" }],
      pageErrors: [], requestFailures: []
    })).not.toThrow();
  });
});
