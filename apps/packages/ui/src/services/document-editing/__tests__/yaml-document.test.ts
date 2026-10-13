import { expect, it } from "vitest";
import { checkDocument, formatDocument } from "../engine";
import { requestFor } from "./request-fixture";
import cases from "./fixtures/v1/cases.json";
it.each(cases.filter((c) => c.language === "yaml"))("shared $id", (c) => {
  const req = requestFor(c.text, "yaml"),
    check = checkDocument(req);
  expect(check.state).toBe(
    c.codes.some((code) => code === "unsupported") ? "unsupported" : "checked",
  );
  expect(
    !check.diagnostics.some(
      (d) => d.code !== "unsupported" && d.severity === "error",
    ),
  ).toBe(c.valid);
  expect([...new Set(check.diagnostics.map((d) => d.code))]).toEqual(c.codes);
  expect(check.format_allowed).toBe(c.format === "allowed");
  const result = formatDocument(req);
  if (c.format === "blocked") expect(result.state).toBe("blocked");
  else {
    expect(["formatted", "unchanged"]).toContain(result.state);
    const output = result.replacement ?? c.text;
    for (const token of c.must_keep) expect(output).toContain(token);
    expect(formatDocument(requestFor(output, "yaml")).state).toBe("unchanged");
  }
});
it("reindents a real commented anchored merged multi-document stream without rewriting tokens", () => {
  const text =
    '# header\n%YAML 1.2\n---\nbase: &base\n    n:   9007199254740993 # exact\n    f: 1e400\ncopy:\n    <<: *base\n    n: -0\n    s: "\\u0061"\nbody: |-\n  keep  spaces\n...\n---\nnext:\n    - 0.12345678901234567890123456789\n    - yes\n';
  const expected = text.replaceAll("    ", "  ").replace("n:   900", "n: 900");
  const result = formatDocument(requestFor(text, "yaml"));
  expect(result.state).toBe("formatted");
  expect(result.replacement).toBe(expected);
  expect(formatDocument(requestFor(result.replacement!, "yaml")).state).toBe(
    "unchanged",
  );
});
it("reindents nested block sequences", () => {
  expect(
    formatDocument(
      requestFor("list:\n    - one\n    - nested:\n          - two\n", "yaml"),
    ).replacement,
  ).toBe("list:\n  - one\n  - nested:\n      - two\n");
});
it.each(["+0xB", "-0xB", "+0o13", "-0o13"])(
  "uses Core decimal-only signs for %s",
  (token) => {
    for (const prefix of ["", "%YAML 1.2\n---\n"]) {
      const distinct = requestFor(
        prefix + `${token}: 1\n${token.startsWith("-") ? "-11" : "11"}: 2\n`,
        "yaml",
      );
      expect(checkDocument(distinct).format_allowed).toBe(true);
      const repeated = requestFor(
        prefix + `${token}: 1\n"${token}": 2\n`,
        "yaml",
      );
      expect(checkDocument(repeated).diagnostics.map((d) => d.code)).toEqual([
        "duplicate_key",
      ]);
      expect(formatDocument(repeated).state).toBe("blocked");
    }
  },
);
it.each([
  ["yes: 1\ntrue: 2\n", false],
  ["%YAML 1.1\n---\nyes: 1\ntrue: 2\n", true],
  ["0xB: 1\n11: 2\n", true],
  ["0o13: 1\n11: 2\n", true],
  ["%YAML 1.1\n---\n+0xB: 1\n11: 2\n", true],
  ["%YAML 1.1\n---\n013: 1\n11: 2\n", true],
  ['1: a\n1.0: b\ntrue: c\n"1": d\n', false],
  ["1.00: a\n1e0: b\n", true],
  ["-0.0: a\n0.0: b\n", true],
  [".NaN: a\n.nan: b\n", true],
  ["9007199254740992: a\n9007199254740993: b\n", false],
  ["0.123456789012345678901: a\n0.123456789012345678902: b\n", false],
  ["1e400: a\n2e400: b\n", false],
  ["1e400: a\n10e399: b\n", true],
  ['"a": 1\n"\\u0061": 2\n', true],
  ['!!str 1: a\n"1": b\n', true],
  ["a: 1\nb: {a: 2}\n", false],
] as const)("qualifies exact key identities %s", (text, duplicate) => {
  const check = checkDocument(requestFor(text, "yaml"));
  expect(check.state).toBe("checked");
  expect(check.diagnostics.some((d) => d.code === "duplicate_key")).toBe(
    duplicate,
  );
});
it.each([
  "x: !<tag:yaml.org,2002:python/object/apply:os.system> [never]\n",
  "? [a,b]\n: c\n",
  "a: !!int wrong\n",
  "a: !!set {x: null}\n",
  "%YAML 1.3\n---\na: 2001-01-01\n",
  "a:\n- x\n",
  "a:\n    body: |\n      unchanged\n",
  "base: &a 1\ncopy: {<<: *a}\n",
])("explicitly refuses unqualified %s", (text) => {
  expect(checkDocument(requestFor(text, "yaml")).state).toBe("unsupported");
  expect(formatDocument(requestFor(text, "yaml")).replacement).toBeNull();
});
it("does not expand alias graphs or confuse merge overrides with duplicates", () => {
  const text =
    "base: &base {x: 1}\ncopy: {<<: *base, x: 2}\ncycle: &cycle [*cycle]\nwide: [" +
    Array(3000).fill("*base").join(",") +
    "]\n";
  expect(checkDocument(requestFor(text, "yaml")).format_allowed).toBe(true);
  expect(formatDocument(requestFor(text, "yaml")).state).toBe("unchanged");
});
it("reports an undefined alias without exposing the alias name", () => {
  const check = checkDocument(requestFor("value: *private_secret", "yaml"));
  expect(check.diagnostics.some((d) => d.code === "syntax")).toBe(true);
  expect(JSON.stringify(check)).not.toContain("private_secret");
});
it("supports empty streams, comments, directives and CRLF without final newline changes", () => {
  for (const text of [
    "",
    "# comment",
    "%YAML 1.1\r\n---\r\na: yes\r\n",
    "a: [1, 2]",
  ]) {
    const result = formatDocument(
      requestFor(text, "yaml", {
        newline: text.includes("\r") ? "\r\n" : "\n",
      }),
    );
    expect(result.state).toBe("unchanged");
  }
});
it.each([
  "%YAML 1.1\n---\na: 2001-01-01\n",
  "? &key foo\n: 1\n? *key\n: 2\n",
  "a: !!str [1,2]\n",
])("refuses unqualified interpretation %s", (text) => {
  expect(checkDocument(requestFor(text, "yaml")).state).toBe("unsupported");
});
it.each(["a: [1,2", 'a: "broken', "a:\n\tb: x\n"])(
  "rejects malformed YAML %s",
  (text) => {
    const check = checkDocument(requestFor(text, "yaml"));
    expect(check.diagnostics.some((d) => d.code === "syntax")).toBe(true);
    expect(formatDocument(requestFor(text, "yaml")).replacement).toBeNull();
  },
);
it("bounds block collection depth before composition", () => {
  const text =
    Array.from({ length: 101 }, (_, i) => "  ".repeat(i) + "a:").join("\n") +
    " 0\n";
  expect(checkDocument(requestFor(text, "yaml")).state).toBe(
    "resource_limited",
  );
});
it("keeps comments and literal bytes during CRLF formatting", () => {
  const text =
    "root:\r\n    value:  9007199254740993 # same\r\nbody: |+\r\n  one\r\n\r\n";
  const result = formatDocument(requestFor(text, "yaml", { newline: "\r\n" }));
  expect(result.replacement).toBe(
    "root:\r\n  value: 9007199254740993 # same\r\nbody: |+\r\n  one\r\n\r\n",
  );
});
it.each([
  "- a:  1\n  b: 2\n",
  'a: [ {n: 9007199254740993}, "\\u0061" ]\n',
  "empty: null\n~: first\nnull: second\n",
])("checks and preserves ordinary layouts %s", (text) => {
  const result = formatDocument(requestFor(text, "yaml"));
  if (text.startsWith("empty")) expect(result.state).toBe("blocked");
  else {
    expect(["unchanged", "formatted"]).toContain(result.state);
    expect(
      formatDocument(requestFor(result.replacement ?? text, "yaml")).state,
    ).toBe("unchanged");
  }
});
it("refuses inherited 1.1 interpretation when the next document has no directive", () => {
  const text = "%YAML 1.1\n---\na: yes\n...\n---\nyes: 1\ntrue: 2\n";
  const check = checkDocument(requestFor(text, "yaml"));
  expect(check.state).toBe("unsupported");
  expect(check.diagnostics.map((d) => d.code)).toEqual(["unsupported"]);
  expect(formatDocument(requestFor(text, "yaml")).state).toBe("blocked");
  expect(
    checkDocument(requestFor(text.replace("1.1", "1.2"), "yaml"))
      .format_allowed,
  ).toBe(true);
});
it("formats an explicit 1.2 document followed by an undirected 1.2 document", () => {
  const text =
    "%YAML 1.2\n---\na:\n    value: yes\n...\n---\nb:\n    yes: 1\n    true: 2\n";
  const result = formatDocument(requestFor(text, "yaml"));
  expect(result.state).toBe("formatted");
  expect(result.replacement).toBe(text.replaceAll("    ", "  "));
  expect(formatDocument(requestFor(result.replacement!, "yaml")).state).toBe(
    "unchanged",
  );
});
it.each([".", "+.", "-.", "._", "e3", ".e3"])(
  "declines candidate 1.1 floats without a mantissa digit: %s",
  (token) => {
    const req = requestFor(
      `%YAML 1.1\n---\n${token}: first\n0.0: second\n`,
      "yaml",
    );
    const check = checkDocument(req);
    expect(check.state).toBe("unsupported");
    expect(check.diagnostics).toHaveLength(1);
    expect(check.diagnostics.some((d) => d.code === "duplicate_key")).toBe(
      false,
    );
    expect(formatDocument(req).replacement).toBeNull();
  },
);
it("keeps default 1.2 and quoted dot keys as exact strings", () => {
  for (const text of [
    ".: first\n0.0: second\n",
    '%YAML 1.1\n---\n".": first\n0.0: second\n',
  ]) {
    expect(checkDocument(requestFor(text, "yaml")).format_allowed).toBe(true);
    expect(formatDocument(requestFor(text, "yaml")).state).toBe("unchanged");
  }
});
it.each(["<<", "!!merge <<"])(
  "blocks repeated merge declarations in one mapping: %s",
  (secondKey) => {
    const text = `a: &a {x: 1}\nb: &b {y: 2}\nc:\n    <<: *a\n    ${secondKey}: *b\n`;
    const req = requestFor(text, "yaml");
    const check = checkDocument(req);
    expect(check.state).toBe("checked");
    expect(check.diagnostics.map((d) => d.code)).toEqual(["duplicate_key"]);
    expect(
      text.slice(check.diagnostics[0].start, check.diagnostics[0].end),
    ).toBe("<<");
    expect(check.format_allowed).toBe(false);
    expect(formatDocument(req).state).toBe("blocked");
    expect(formatDocument(req).replacement).toBeNull();
  },
);
it.each(["*a", "[*a, *b]", "{x: 1}"])(
  "preserves a single merge target and explicit imported-member override: %s",
  (target) => {
    const text = `a: &a {x: 1}\nb: &b {y: 2}\nc:\n    <<: ${target}\n    x: 2\n`;
    const expected = `a: &a {x: 1}\nb: &b {y: 2}\nc:\n  <<: ${target}\n  x: 2\n`;
    const req = requestFor(text, "yaml");
    expect(checkDocument(req).diagnostics).toEqual([]);
    const result = formatDocument(req);
    expect(result.state).toBe("formatted");
    expect(result.replacement).toBe(expected);
    expect(formatDocument(requestFor(expected, "yaml")).state).toBe(
      "unchanged",
    );
  },
);
it("keeps merge declaration identities scoped to sibling mappings", () => {
  const text = "a: &a {x: 1}\nb: {<<: *a}\nc: {<<: *a}\n";
  expect(checkDocument(requestFor(text, "yaml")).format_allowed).toBe(true);
  expect(formatDocument(requestFor(text, "yaml")).state).toBe("unchanged");
});
it("distinguishes a merge declaration from a quoted string key", () => {
  const text = '<<: {x: 1}\n"<<": literal\n';
  expect(checkDocument(requestFor(text, "yaml")).diagnostics).toEqual([]);
  expect(formatDocument(requestFor(text, "yaml")).state).toBe("unchanged");
});
it.each([
  "!!merge x: {a: 1}\n!!merge x: {b: 2}\n",
  "!!merge x: {a: 1}\n",
  "!!merge x\n",
  "value: !!merge x\n",
  "!!merge x: 1\n",
])(
  "refuses an explicit merge tag with unqualified scalar spelling: %s",
  (text) => {
    const req = requestFor(text, "yaml");
    const check = checkDocument(req);
    expect(check.state).toBe("unsupported");
    expect(check.complete).toBe(false);
    expect(check.diagnostics.map((d) => d.code)).toEqual(
      text === "!!merge x: {a: 1}\n!!merge x: {b: 2}\n"
        ? ["unsupported", "unsupported"]
        : ["unsupported"],
    );
    expect(check.format_allowed).toBe(false);
    expect(formatDocument(req).state).toBe("blocked");
    expect(formatDocument(req).replacement).toBeNull();
  },
);
it.each(["!!merge <<", '!!merge "<<"', "!<tag:yaml.org,2002:merge> <<"])(
  "preserves a qualified explicit merge scalar tag: %s",
  (key) => {
    const text = `a: &a {x: 1}\nb:\n    ${key}: *a\n    x: 2\n`;
    const expected = `a: &a {x: 1}\nb:\n  ${key}: *a\n  x: 2\n`;
    const req = requestFor(text, "yaml");
    expect(checkDocument(req).diagnostics).toEqual([]);
    const result = formatDocument(req);
    expect(result.state).toBe("formatted");
    expect(result.replacement).toBe(expected);
    expect(formatDocument(requestFor(expected, "yaml")).state).toBe(
      "unchanged",
    );
  },
);
it.each(["!!str &key name", "&key !!str name"])(
  "reindents mapping key properties without rewriting their tokens: %s",
  (key) => {
    const text = `root:\n    ${key}:  value\n    copy: *key\nbody: |-\n  keep  spaces\n`;
    const expected = `root:\n  ${key}: value\n  copy: *key\nbody: |-\n  keep  spaces\n`;
    const result = formatDocument(requestFor(text, "yaml"));
    expect(result.state).toBe("formatted");
    expect(result.replacement).toBe(expected);
    expect(formatDocument(requestFor(expected, "yaml")).state).toBe(
      "unchanged",
    );
  },
);
it.each([
  "!!str key: !!str value\n",
  "root:\n  !!str key: &value raw\n  copy: *value\n",
  "root: {!!str key: !!str value}\n",
  "- !!str key: value\n  other: !!str value\n",
  "body: |-\n  !!str key:  literal bytes\n",
])(
  "retains inline properties, flow and block-scalar boundaries: %s",
  (text) => {
    expect(checkDocument(requestFor(text, "yaml")).format_allowed).toBe(true);
    const result = formatDocument(requestFor(text, "yaml"));
    expect(result.state).toBe("unchanged");
    expect(result.replacement).toBeNull();
  },
);
