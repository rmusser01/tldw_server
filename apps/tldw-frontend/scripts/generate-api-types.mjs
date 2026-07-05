#!/usr/bin/env node
/**
 * Regenerate the typed OpenAPI view for the frontend.
 *
 * 1. Runs the backend exporter (Helper_Scripts/export_openapi_schema.py) to
 *    produce a canonical openapi.json AND refresh the checked-in fingerprint.
 * 2. Runs openapi-typescript over that JSON to emit lib/api/generated/schema.d.ts.
 *
 * The full openapi.json (~5MB) and schema.d.ts (~6MB) are build artifacts and
 * are gitignored (lib/api/generated/). Only the tiny openapi.fingerprint.json is
 * committed — the CI drift gate keys off that. Run this whenever the backend
 * API contract changes and the drift gate fails.
 *
 * Requires a Python environment with the server deps importable (run from repo
 * root or with the venv active).
 */
import { execFileSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendRoot = resolve(here, "..");
const repoRoot = resolve(frontendRoot, "..", "..");

const generatedDir = resolve(frontendRoot, "lib/api/generated");
const openapiJson = resolve(generatedDir, "openapi.json");
const schemaDts = resolve(generatedDir, "schema.d.ts");
const fingerprint = resolve(frontendRoot, "lib/api/openapi.fingerprint.json");

mkdirSync(generatedDir, { recursive: true });

const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");

console.log("[generate-api-types] exporting canonical OpenAPI schema + fingerprint…");
execFileSync(
  python,
  [
    "Helper_Scripts/export_openapi_schema.py",
    "--out",
    openapiJson,
    "--fingerprint",
    fingerprint,
  ],
  { cwd: repoRoot, stdio: "inherit", env: { ...process.env, PYTHONPATH: repoRoot } },
);

console.log("[generate-api-types] generating schema.d.ts via openapi-typescript…");
// `bun x` (not `bunx`) — the standalone `bunx` shim can ENOENT on Windows.
execFileSync("bun", ["x", "openapi-typescript", openapiJson, "-o", schemaDts], {
  cwd: frontendRoot,
  stdio: "inherit",
});

console.log(`[generate-api-types] done -> ${schemaDts}`);
console.log("[generate-api-types] committed artifact: lib/api/openapi.fingerprint.json");
