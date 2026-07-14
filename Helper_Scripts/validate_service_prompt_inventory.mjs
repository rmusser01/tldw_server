#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] ?? process.cwd());
const inventoryPath = path.join(root, "Docs/Design/service-prompt-inventory.md");
const inventory = fs.readFileSync(inventoryPath, "utf8");

const EXPECTED_MATRIX_ROWS = 232;
const EXPECTED_DECISIONS = { eligible: 73, deferred: 75, excluded: 84 };
const EXPECTED_SUBSETS = {
  fileBacked: { eligible: 5, deferred: 14, excluded: 17 },
  hardCoded: { eligible: 68, deferred: 61, excluded: 67 },
};
const EXPECTED_MATRIX_HEADER = [
  "candidate",
  "source",
  "runtime consumer/call sites",
  "data owner",
  "workflow owner",
  "explicit-field literal/template semantics",
  "variables",
  "assembly",
  "locked fragments/visibility",
  "output dependency",
  "contract sensitivity",
  "decision",
  "reason",
  "service_prompt_id",
  "parts",
  "rollout slice",
];
const EXPECTED_IDS_BY_DOMAIN = {
  "summarization/media/audio": [
    "media.analysis.critical",
    "media.analysis.executive",
    "media.analysis.qa",
    "media.audio.analysis",
    "media.review.bullets",
    "media.review.critical",
    "media.review.qa",
    "media.review.summary.bullets",
    "media.review.summary.detailed",
    "media.review.summary.executive",
    "media.transcript.clean",
    "media.transcript.correction",
    "media.transcript.headings",
    "media.transcript.speakerturns",
    "research.studio.audio",
    "slides.source.summary",
  ],
  "documents/web": [
    "chat.document.briefing",
    "chat.document.meetingnotes",
    "chat.document.qa",
    "chat.document.studyguide",
    "chat.document.summary",
    "chat.document.timeline",
    "documents.copilot.explain",
    "documents.copilot.rephrase",
    "documents.copilot.summary",
    "documents.copilot.translate",
    "image.prompt.refinement",
    "media.document.insights",
    "media.document.summary",
    "media.text.translation",
    "notes.title.generate",
    "web.search.client.answer",
    "web.search.snippet.digest",
    "workflow.book.analysis.chapter",
    "workflow.book.analysis.characters",
    "workflow.book.analysis.comprehensive",
    "workflow.book.analysis.concepts",
    "workflow.web.summary.brief",
    "workflow.web.summary.bullets",
    "workflow.web.summary.detailed",
    "writing.agent.brainstorm",
    "writing.agent.planning",
    "writing.agent.quick",
    "writing.annotation.scene",
    "writing.annotation.selection",
    "writing.continuation.fill",
    "writing.continuation.predict",
    "writing.feedback.echo",
  ],
  "RAG generation": ["rag.client.answer", "research.explainer.expansion"],
  "reports/digests/watchlists/outputs": [
    "chat.title.generation",
    "data.table.generation",
    "playground.disco.skill.comment",
    "research.studio.compare",
    "research.studio.corpus.gaps",
    "research.studio.executive",
    "research.studio.hypotheses",
    "research.studio.literature.matrix",
    "research.studio.proposal",
    "research.studio.report",
    "research.studio.slides",
    "research.studio.summary",
    "research.studio.timeline",
    "research.synthesis.report",
    "slides.deck.generation",
    "study.assistant.explain",
    "study.assistant.followup",
    "study.assistant.freeform",
    "study.assistant.mnemonic",
    "study.pack.generation",
    "web.search.report",
  ],
  "extraction/chunking": ["chunking.rolling.summary", "writing.feedback.mood"],
};
const EXPECTED_DOMAIN_BY_ID = new Map(
  Object.entries(EXPECTED_IDS_BY_DOMAIN).flatMap(([domain, ids]) =>
    ids.map((id) => [id, domain]),
  ),
);
if (EXPECTED_DOMAIN_BY_ID.size !== EXPECTED_DECISIONS.eligible) {
  throw new Error("invalid built-in eligible ID manifest");
}

function splitRow(line) {
  const cells = [];
  let value = "";
  let inCode = false;
  let escaped = false;
  for (let i = 1; i < line.length - 1; i += 1) {
    const ch = line[i];
    if (escaped) {
      value += ch;
      escaped = false;
    } else if (ch === "\\") {
      value += ch;
      escaped = true;
    } else if (ch === "`") {
      inCode = !inCode;
      value += ch;
    } else if (ch === "|" && !inCode) {
      cells.push(value.trim());
      value = "";
    } else {
      value += ch;
    }
  }
  cells.push(value.trim());
  return cells;
}

function tableBetween(text, startMarker, endMarker) {
  const start = text.indexOf(startMarker);
  const end = text.indexOf(endMarker, start + startMarker.length);
  if (start < 0 || end < 0) throw new Error(`missing section ${startMarker}`);
  const lines = text
    .slice(start, end)
    .split("\n")
    .filter((line) => line.startsWith("|") && line.endsWith("|"));
  const headerAt = lines.findIndex((line) => !/^\|\s*---/.test(line));
  const header = splitRow(lines[headerAt]);
  const parsedRows = lines.slice(headerAt + 2).map(splitRow);
  const rows = parsedRows.filter((cells) => cells.length === header.length);
  const malformedRows = parsedRows.filter((cells) => cells.length !== header.length);
  return { header, rows, malformedRows };
}

const matrix = tableBetween(
  inventory,
  "## Inventory matrix",
  "## Eligible definition contract validation",
);
const col = Object.fromEntries(matrix.header.map((name, index) => [name, index]));
const decisions = { eligible: 0, deferred: 0, excluded: 0 };
const eligibleIds = new Map();
const candidateNames = new Set();
const errors = [];
if (matrix.header.join("\n") !== EXPECTED_MATRIX_HEADER.join("\n")) {
  errors.push("matrix header differs from the approved 16-column schema");
}
if (matrix.malformedRows.length) {
  errors.push(`matrix contains ${matrix.malformedRows.length} malformed rows`);
}
if (matrix.rows.length !== EXPECTED_MATRIX_ROWS) {
  errors.push(`expected ${EXPECTED_MATRIX_ROWS} matrix rows, found ${matrix.rows.length}`);
}
for (const row of matrix.rows) {
  const candidate = row[col.candidate];
  if (row.some((cell) => cell.length === 0)) {
    errors.push(`matrix row contains a blank cell: ${candidate || "<blank candidate>"}`);
  }
  if (candidateNames.has(candidate)) errors.push(`duplicate candidate identity: ${candidate}`);
  else candidateNames.add(candidate);
  const decision = row[col.decision];
  if (!(decision in decisions)) {
    errors.push(`unknown decision for ${row[col.candidate]}: ${decision}`);
    continue;
  }
  decisions[decision] += 1;
  const idMatch = row[col.service_prompt_id].match(/`([a-z0-9]+(?:\.[a-z0-9]+)+)`/);
  if (decision === "eligible") {
    if (!idMatch) errors.push(`eligible row lacks ID: ${row[col.candidate]}`);
    else if (eligibleIds.has(idMatch[1])) errors.push(`duplicate ID ${idMatch[1]}`);
    else eligibleIds.set(idMatch[1], row);
    if (!row[col.assembly].includes("Golden:")) {
      errors.push(`eligible row lacks Golden: ${row[col.candidate]}`);
    }
    if (!row[col.parts] || row[col.parts] === "—") {
      errors.push(`eligible row lacks parts: ${row[col.candidate]}`);
    }
  } else if (idMatch) {
    errors.push(`${decision} row unexpectedly has ID ${idMatch[1]}`);
  }
}
for (const [decision, expected] of Object.entries(EXPECTED_DECISIONS)) {
  if (decisions[decision] !== expected) {
    errors.push(`expected ${expected} ${decision} rows, found ${decisions[decision]}`);
  }
}
for (const [id, domain] of EXPECTED_DOMAIN_BY_ID) {
  const row = eligibleIds.get(id);
  if (!row) errors.push(`approved eligible ID missing from matrix: ${id}`);
  else if (row[col["rollout slice"]] !== domain) {
    errors.push(
      `approved domain mismatch for ${id}: expected ${domain}, found ${row[col["rollout slice"]]}`,
    );
  }
}
for (const id of eligibleIds.keys()) {
  if (!EXPECTED_DOMAIN_BY_ID.has(id)) errors.push(`unexpected eligible ID in matrix: ${id}`);
}

const contract = tableBetween(
  inventory,
  "### Sorted eligible ID and contract index",
  "### Async pinning topology for eligible definitions",
);
const contractIds = contract.rows
  .map((row) => row[0].match(/`([^`]+)`/)?.[1])
  .filter(Boolean);
if (contractIds.join("\n") !== [...contractIds].sort().join("\n")) {
  errors.push("contract index is not sorted");
}
const eligibleSorted = [...eligibleIds.keys()].sort();
if (eligibleSorted.join("\n") !== contractIds.join("\n")) {
  errors.push("candidate IDs and contract-index IDs differ");
}

const jobs = tableBetween(
  inventory,
  "### Async pinning topology for eligible definitions",
  "### File-backed inventory notes",
);
const jobIds = jobs.rows.flatMap((row) =>
  [...row[0].matchAll(/`([a-z0-9]+(?:\.[a-z0-9]+)+)`/g)].map((match) => match[1]),
);
const expectedJobIds = [
  "data.table.generation",
  "media.audio.analysis",
  "research.explainer.expansion",
  "research.synthesis.report",
  "study.pack.generation",
  "writing.annotation.scene",
];
const uniqueJobIds = [...new Set(jobIds)].sort();
if (uniqueJobIds.join("\n") !== expectedJobIds.join("\n")) {
  errors.push(
    `Jobs ID set mismatch expected=[${expectedJobIds}] found=[${uniqueJobIds}]`,
  );
}

const statedSubsetMatch = inventory.match(
  /file-backed subset is \*\*36 = (\d+) eligible, (\d+) deferred, (\d+) excluded\*\*; the hard-coded subset is \*\*196 = (\d+) eligible, (\d+) deferred, (\d+) excluded\*\*/,
);
if (!statedSubsetMatch) {
  errors.push("missing authoritative file-backed/hard-coded subset counts");
}
const statedSubsets = statedSubsetMatch
  ? {
      fileBacked: {
        eligible: Number(statedSubsetMatch[1]),
        deferred: Number(statedSubsetMatch[2]),
        excluded: Number(statedSubsetMatch[3]),
      },
      hardCoded: {
        eligible: Number(statedSubsetMatch[4]),
        deferred: Number(statedSubsetMatch[5]),
        excluded: Number(statedSubsetMatch[6]),
      },
    }
  : null;
if (statedSubsets) {
  for (const subset of Object.keys(EXPECTED_SUBSETS)) {
    for (const decision of Object.keys(EXPECTED_DECISIONS)) {
      if (statedSubsets[subset][decision] !== EXPECTED_SUBSETS[subset][decision]) {
        errors.push(
          `expected ${subset} ${decision}=${EXPECTED_SUBSETS[subset][decision]}, found ${statedSubsets[subset][decision]}`,
        );
      }
    }
  }
  for (const decision of Object.keys(decisions)) {
    if (
      statedSubsets.fileBacked[decision] + statedSubsets.hardCoded[decision] !==
      decisions[decision]
    ) {
      errors.push(`subset ${decision} counts do not sum to matrix count`);
    }
  }
  if (Object.values(statedSubsets.fileBacked).reduce((a, b) => a + b, 0) !== 36) {
    errors.push("file-backed subset does not total 36");
  }
  if (Object.values(statedSubsets.hardCoded).reduce((a, b) => a + b, 0) !== 196) {
    errors.push("hard-coded subset does not total 196");
  }
}

const allFiles = [];
function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === ".git" || entry.name === "node_modules" || entry.name === ".venv") {
      continue;
    }
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else allFiles.push(path.relative(root, full));
  }
}
walk(root);

const lineCountCache = new Map();
function resolveFile(raw) {
  const cleaned = raw.replace(/^\.\//, "");
  if (fs.existsSync(path.join(root, cleaned))) return cleaned;
  const matches = allFiles.filter((file) => file === cleaned || file.endsWith(`/${cleaned}`));
  return matches.length === 1 ? matches[0] : null;
}

const spanErrors = [];
let sourceSpans = 0;
let lineComponents = 0;
for (const match of inventory.matchAll(/`([^`\n]+)`/g)) {
  const token = match[1].replaceAll("&lt;", "<").replaceAll("&gt;", ">");
  const ref = token.match(
    /^([^:]+\.(?:py|ts|tsx|yaml|yml|json|md)):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)$/,
  );
  if (!ref) continue;
  sourceSpans += 1;
  const resolved = resolveFile(ref[1]);
  if (!resolved) {
    spanErrors.push(`unresolved ${ref[1]}`);
    continue;
  }
  let maxLines = lineCountCache.get(resolved);
  if (!maxLines) {
    maxLines = fs.readFileSync(path.join(root, resolved), "utf8").split("\n").length;
    lineCountCache.set(resolved, maxLines);
  }
  for (const component of ref[2].split(",")) {
    lineComponents += 1;
    const [a, b = a] = component.split("-").map(Number);
    if (a < 1 || b < a || b > maxLines) {
      spanErrors.push(`out of bounds ${resolved}:${component}/${maxLines}`);
    }
  }
}

const domainPlans = [
  [
    "summarization/media/audio",
    "Docs/superpowers/plans/2026-07-13-service-prompts-domain-summarization-media-audio.md",
    "## Exact scope lock",
    "Copy the inventory topology literally",
  ],
  [
    "documents/web",
    "Docs/superpowers/plans/2026-07-13-service-prompts-domain-documents-web.md",
    "## Exact scope lock: 32 definitions",
    "Do not register adjacent/deferred IDs",
  ],
  [
    "RAG generation",
    "Docs/superpowers/plans/2026-07-13-service-prompts-domain-rag-generation.md",
    "## Approved scope and exact contracts",
    "For Explainer, keep",
  ],
  [
    "reports/digests/watchlists/outputs",
    "Docs/superpowers/plans/2026-07-13-service-prompts-domain-reports-digests-watchlists-outputs.md",
    "## Exact approved ID set",
    "The watchlist item/group",
  ],
  [
    "extraction/chunking",
    "Docs/superpowers/plans/2026-07-13-service-prompts-domain-extraction-chunking.md",
    "## Approved scope and contracts",
    "OCR, proposition extraction",
  ],
];
const planCoverage = {};
for (const [domain, relativePath, startMarker, endMarker] of domainPlans) {
  const text = fs.readFileSync(path.join(root, relativePath), "utf8");
  const start = text.indexOf(startMarker);
  const end = text.indexOf(endMarker, start + startMarker.length);
  const scope = text.slice(start, end);
  const planIds = new Set(
    [...scope.matchAll(/`?([a-z0-9]+(?:\.[a-z0-9]+)+)`?/g)]
      .map((match) => match[1])
      .filter((id) => eligibleIds.has(id)),
  );
  const expected = new Set(EXPECTED_IDS_BY_DOMAIN[domain]);
  const missing = [...expected].filter((id) => !planIds.has(id));
  const extra = [...planIds].filter((id) => !expected.has(id));
  if (missing.length || extra.length) {
    errors.push(`${domain} scope mismatch missing=[${missing}] extra=[${extra}]`);
  }
  planCoverage[domain] = { expected: expected.size, found: planIds.size };
  const requiredHeader = /^# .+ Implementation Plan\n\n> \*\*For agentic workers:\*\* REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development \(recommended\) or superpowers:executing-plans to implement this plan task-by-task\. Steps use checkbox \(`- \[ \]`\) syntax for tracking\.\n\n\*\*Goal:\*\* .+\n\n\*\*Architecture:\*\* .+\n\n\*\*Tech Stack:\*\* .+\n\n---/s;
  if (!requiredHeader.test(text)) {
    errors.push(`${relativePath} lacks exact writing-plans header`);
  }
}

console.log(
  JSON.stringify(
    {
      rows: matrix.rows.length,
      decisions,
      fileBacked: statedSubsets?.fileBacked,
      hardCoded: statedSubsets?.hardCoded,
      eligibleIds: eligibleIds.size,
      contractIds: contractIds.length,
      jobsIds: uniqueJobIds,
      sourceSpans,
      lineComponents,
      unresolvedOrOutOfBounds: spanErrors.length,
      firstReferenceErrors: spanErrors.slice(0, 20),
      planCoverage,
      errors,
    },
    null,
    2,
  ),
);
if (errors.length || spanErrors.length) process.exitCode = 1;
