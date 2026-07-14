import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const validator = path.join(repoRoot, "Helper_Scripts/validate_service_prompt_inventory.mjs");
const inventoryRelative = "Docs/Design/service-prompt-inventory.md";
const planRelatives = [
  "Docs/superpowers/plans/2026-07-13-service-prompts-domain-summarization-media-audio.md",
  "Docs/superpowers/plans/2026-07-13-service-prompts-domain-documents-web.md",
  "Docs/superpowers/plans/2026-07-13-service-prompts-domain-rag-generation.md",
  "Docs/superpowers/plans/2026-07-13-service-prompts-domain-reports-digests-watchlists-outputs.md",
  "Docs/superpowers/plans/2026-07-13-service-prompts-domain-extraction-chunking.md",
];

function copyFile(relativePath, fixtureRoot) {
  const destination = path.join(fixtureRoot, relativePath);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(path.join(repoRoot, relativePath), destination);
}

function makeFixture() {
  const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), "service-prompt-inventory-"));
  copyFile(inventoryRelative, fixtureRoot);
  for (const relativePath of planRelatives) copyFile(relativePath, fixtureRoot);

  const trackedFiles = spawnSync("git", ["-C", repoRoot, "ls-files"], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  })
    .stdout.trim()
    .split("\n");
  const inventory = fs.readFileSync(path.join(repoRoot, inventoryRelative), "utf8");
  const referenced = new Set();
  for (const match of inventory.matchAll(/`([^`\n]+)`/g)) {
    const source = match[1].match(
      /^([^:]+\.(?:py|ts|tsx|yaml|yml|json|md)):\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*/,
    );
    if (!source) continue;
    const exact = trackedFiles.includes(source[1])
      ? source[1]
      : trackedFiles.find((candidate) => candidate.endsWith(`/${source[1]}`));
    if (exact) referenced.add(exact);
  }
  for (const relativePath of referenced) {
    if (relativePath === inventoryRelative || planRelatives.includes(relativePath)) continue;
    const destination = path.join(fixtureRoot, relativePath);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.symlinkSync(path.join(repoRoot, relativePath), destination);
  }
  return fixtureRoot;
}

function runValidator(fixtureRoot) {
  return spawnSync(process.execPath, [validator, fixtureRoot], { encoding: "utf8" });
}

function assertBaselinePasses(fixtureRoot) {
  const result = runValidator(fixtureRoot);
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
}

function rewrite(relativePath, fixtureRoot, transform) {
  const target = path.join(fixtureRoot, relativePath);
  fs.writeFileSync(target, transform(fs.readFileSync(target, "utf8")));
}

function replaceInSection(text, startMarker, endMarker, before, after) {
  const start = text.indexOf(startMarker);
  const end = text.indexOf(endMarker, start + startMarker.length);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const section = text.slice(start, end);
  assert.ok(section.includes(before));
  return `${text.slice(0, start)}${section.replace(before, after)}${text.slice(end)}`;
}

test("rejects coordinated decision and subset-count drift", (t) => {
  const fixtureRoot = makeFixture();
  t.after(() => fs.rmSync(fixtureRoot, { recursive: true, force: true }));
  assertBaselinePasses(fixtureRoot);

  rewrite(inventoryRelative, fixtureRoot, (text) =>
    text
      .replace(
        "| excluded | The approved boundary keeps generic chat system prompts",
        "| deferred | The approved boundary keeps generic chat system prompts",
      )
      .replace(
        "232 rows = 73 eligible, 75 deferred, 84 excluded**. The file-backed subset is **36 = 5 eligible, 14 deferred, 17 excluded",
        "232 rows = 73 eligible, 76 deferred, 83 excluded**. The file-backed subset is **36 = 5 eligible, 15 deferred, 16 excluded",
      ),
  );

  const result = runValidator(fixtureRoot);
  assert.notEqual(result.status, 0, result.stdout);
});

test("rejects an ID-to-domain swap coordinated across matrix and plans", (t) => {
  const fixtureRoot = makeFixture();
  t.after(() => fs.rmSync(fixtureRoot, { recursive: true, force: true }));
  assertBaselinePasses(fixtureRoot);

  rewrite(inventoryRelative, fixtureRoot, (text) => {
    const lines = text.split("\n");
    for (let index = 0; index < lines.length; index += 1) {
      if (lines[index].includes("`rag.client.answer`")) {
        lines[index] = lines[index].replace(/\| RAG generation \|$/, "| extraction/chunking |");
      } else if (lines[index].includes("`chunking.rolling.summary`")) {
        lines[index] = lines[index].replace(/\| extraction\/chunking \|$/, "| RAG generation |");
      }
    }
    return lines.join("\n");
  });
  rewrite(planRelatives[2], fixtureRoot, (text) =>
    replaceInSection(
      text,
      "## Approved scope and exact contracts",
      "For Explainer, keep",
      "rag.client.answer",
      "chunking.rolling.summary",
    ),
  );
  rewrite(planRelatives[4], fixtureRoot, (text) =>
    replaceInSection(
      text,
      "## Approved scope and contracts",
      "OCR, proposition extraction",
      "chunking.rolling.summary",
      "rag.client.answer",
    ),
  );

  const result = runValidator(fixtureRoot);
  assert.notEqual(result.status, 0, result.stdout);
});

test("rejects duplicate candidate identities", (t) => {
  const fixtureRoot = makeFixture();
  t.after(() => fs.rmSync(fixtureRoot, { recursive: true, force: true }));
  assertBaselinePasses(fixtureRoot);

  rewrite(inventoryRelative, fixtureRoot, (text) =>
    text.replace("| Legacy chat default assistant |", "| Audio transcript analysis summary |"),
  );

  const result = runValidator(fixtureRoot);
  assert.notEqual(result.status, 0, result.stdout);
});

test("rejects blank matrix cells", (t) => {
  const fixtureRoot = makeFixture();
  t.after(() => fs.rmSync(fixtureRoot, { recursive: true, force: true }));
  assertBaselinePasses(fixtureRoot);

  rewrite(inventoryRelative, fixtureRoot, (text) =>
    text.replace("| Requesting chat user if adopted |", "|  |"),
  );

  const result = runValidator(fixtureRoot);
  assert.notEqual(result.status, 0, result.stdout);
});

test("rejects malformed or missing matrix rows", (t) => {
  const fixtureRoot = makeFixture();
  t.after(() => fs.rmSync(fixtureRoot, { recursive: true, force: true }));
  assertBaselinePasses(fixtureRoot);

  rewrite(inventoryRelative, fixtureRoot, (text) => {
    const line = text
      .split("\n")
      .find((candidate) => candidate.startsWith("| Legacy chat default assistant |"));
    assert.ok(line);
    return text.replace(`${line}\n`, "");
  });

  const result = runValidator(fixtureRoot);
  assert.notEqual(result.status, 0, result.stdout);
});
