#!/usr/bin/env node
/**
 * Regenerate the shared typed OpenAPI view for ALL frontend shells
 * (ADR-069: this package is the single contract source).
 *
 * 1. Runs the backend exporter (Helper_Scripts/export_openapi_schema.py) to
 *    produce a canonical openapi.json AND refresh the checked-in fingerprint.
 * 2. Runs openapi-typescript over that JSON to emit schema.d.ts (rich types;
 *    gitignored — regenerated, never committed).
 * 3. The same backend exporter renders paths.d.ts: a sorted, deduped
 *    string-literal union that IS committed, so frontend-only checkouts and
 *    CI jobs (no Python needed) can typecheck against the contract.
 *
 * Committed artifacts: openapi.fingerprint.json, paths.d.ts.
 * Gitignored artifacts: openapi.json, schema.d.ts.
 *
 * Requires a Python environment with the server deps importable (run from repo
 * root or with the venv active).
 */
import { execFileSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "..", "..", "..");

const openapiJson = resolve(pkgRoot, "openapi.json");
const schemaDts = resolve(pkgRoot, "schema.d.ts");
const pathsDts = resolve(pkgRoot, "paths.d.ts");
const fingerprint = resolve(pkgRoot, "openapi.fingerprint.json");

mkdirSync(pkgRoot, { recursive: true });

const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");

console.log("[api-types] exporting canonical OpenAPI schema + fingerprint…");
execFileSync(
  python,
  [
    "Helper_Scripts/export_openapi_schema.py",
    "--out",
    openapiJson,
    "--fingerprint",
    fingerprint,
    "--paths",
    pathsDts,
  ],
  { cwd: repoRoot, stdio: "inherit", env: { ...process.env, PYTHONPATH: repoRoot } },
);

console.log("[api-types] generating schema.d.ts via openapi-typescript…");
// `bun x` (not `bunx`) — the standalone `bunx` shim can ENOENT on Windows.
execFileSync("bun", ["x", "openapi-typescript", openapiJson, "-o", schemaDts], {
  cwd: pkgRoot,
  stdio: "inherit",
});

console.log(`[api-types] committed path union -> ${pathsDts}`);
console.log("[api-types] committed artifacts: openapi.fingerprint.json, paths.d.ts");
