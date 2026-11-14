import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { summarizeBunInstall } from "../ci/summarize_bun_install.mjs";

const helper = fileURLToPath(new URL("../ci/summarize_bun_install.mjs", import.meta.url));
const lock = { packages: {
  "parent/alias": ["@scope/optional@1.2.3", "", {}],
  "optional": ["optional@2.0.0", "", {}],
  "other/optional": ["optional@1.0.0", "", {}],
  "unsafe": ["bad\nname@1.0.0", "", {}],
} };
const warning = (name, script = "postinstall") =>
  `warn: deleting optional dependency '${name}' due to failed '${script}' script`;
const unidentified = [{ classification: "unidentified_install_failure" }];

test("reports only canonical locked identities and fixed lifecycle stages, once", () => {
  assert.deepEqual(summarizeBunInstall([
    warning("@scope/optional"), warning("@scope/optional"), warning("optional", "install"),
  ].join("\n"), lock), [
    { classification: "optional_lifecycle_warning", package: "@scope/optional", script: "postinstall" },
    { classification: "optional_lifecycle_warning", package: "optional", script: "install" },
  ]);
});

test("accepts SGR color and CRLF without accepting arbitrary terminal controls", () => {
  assert.equal(summarizeBunInstall(`\x1b[33m${warning("optional")}\x1b[0m\r\n`, lock)[0].package, "optional");
  for (const control of ["\x1b[2J", "\x1b]0;secret\x07", "\t", "\x00", "\x7f"]) {
    assert.deepEqual(summarizeBunInstall(control + warning("optional"), lock), unidentified);
  }
});

test("does not disclose headers, script output, unlisted names or malformed records", () => {
  for (const line of [
    `> Authorization: ${warning("optional")}`, `< Set-Cookie: ${warning("optional")}`,
    `prefix ${warning("optional")}`, warning("parent/alias"), warning("not-locked"),
    warning("optional", "secret-value"), warning("optional") + " trailing",
    warning("bad\nname"), warning("x".repeat(1100)), "error: arbitrary script secret",
  ]) assert.deepEqual(summarizeBunInstall(line, lock), unidentified);
  assert.deepEqual(summarizeBunInstall("", lock), unidentified);
});

test("rejects an invalid lock map and does not accept unsafe package names", () => {
  for (const invalid of [null, {}, { packages: [] }, { packages: null }]) {
    assert.throws(() => summarizeBunInstall(warning("optional"), invalid));
  }
  for (const name of ["../escape", "bad:name", "a".repeat(215)]) {
    assert.deepEqual(summarizeBunInstall(warning(name), { packages: { key: [`${name}@1.0.0`] } }), unidentified);
  }
});

test("exact installed Bun reads JSONC snapshots; malformed input never prints raw data", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "bun-summary-test-"));
  try {
    const snapshot = path.join(directory, "snapshot.lock");
    const log = path.join(directory, "private.log");
    fs.writeFileSync(snapshot, '{ // JSONC input\n"packages": {"nested/alias": ["optional@1.0.0",],},}\n');
    fs.writeFileSync(log, `> Authorization: private-marker\n${warning("optional")}\n`);
    const run = () => spawnSync(process.env.BUN_BINARY || "bun", ["--no-install", helper, snapshot, log], { encoding: "utf8" });
    const result = run();
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), { classification: "optional_lifecycle_warning", package: "optional", script: "postinstall" });
    assert.equal(result.stderr, "");
    fs.writeFileSync(snapshot, "malformed private-marker");
    const malformed = run();
    assert.equal(malformed.status, 1);
    assert.equal(malformed.stdout + malformed.stderr, "");
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
