import assert from "node:assert/strict";

const FAILURE_BUCKETS = ["pageErrors", "requestFailures", "httpErrors"];

export const assertCleanContextDiagnostics = (contexts) => {
  for (const [contextName, diagnostics] of Object.entries(contexts)) {
    for (const bucketName of FAILURE_BUCKETS) {
      const failures = diagnostics[bucketName];
      assert.ok(
        Array.isArray(failures),
        `${contextName} context is missing the ${bucketName} diagnostics bucket`,
      );
      assert.deepEqual(
        failures,
        [],
        `${contextName} context recorded ${bucketName}: ${JSON.stringify(failures)}`,
      );
    }
  }
};
