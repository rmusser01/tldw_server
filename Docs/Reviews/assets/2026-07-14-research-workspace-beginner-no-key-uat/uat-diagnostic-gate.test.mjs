import assert from "node:assert/strict";
import test from "node:test";
import { assertCleanContextDiagnostics } from "./uat-diagnostic-gate.mjs";

const makeCleanContext = () => ({
  pageErrors: [],
  requestFailures: [],
  httpErrors: [],
});

test("accepts contexts without runtime or network failures", () => {
  assert.doesNotThrow(() =>
    assertCleanContextDiagnostics({ desktop: makeCleanContext() }),
  );
});

for (const bucketName of ["pageErrors", "requestFailures", "httpErrors"]) {
  test(`rejects a non-empty ${bucketName} bucket`, () => {
    const context = makeCleanContext();
    context[bucketName].push({ message: "diagnostic failure" });

    assert.throws(
      () => assertCleanContextDiagnostics({ desktop: context }),
      new RegExp(`desktop.*${bucketName}`),
    );
  });
}
