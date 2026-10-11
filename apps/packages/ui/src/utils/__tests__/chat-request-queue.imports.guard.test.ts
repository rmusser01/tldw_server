import fs from "node:fs";
import path from "node:path";
import ts from "typescript";
import { expect, it } from "vitest";

const runtimeImports = (filename: string) => {
  const source = ts.createSourceFile(
    filename,
    fs.readFileSync(filename, "utf8"),
    ts.ScriptTarget.Latest,
    true,
  );
  return source.statements
    .filter(ts.isImportDeclaration)
    .filter((node) => !node.importClause?.isTypeOnly)
    .map((node) => (node.moduleSpecifier as ts.StringLiteral).text);
};

it("keeps database initialization out of the app-shell queue ID import", () => {
  expect(
    runtimeImports(path.resolve(__dirname, "..", "chat-request-queue.ts")),
  ).not.toContain("@/db/dexie/helpers");
});

it("keeps the shared ID generator free of runtime dependencies", () => {
  expect(
    runtimeImports(path.resolve(__dirname, "..", "generate-id.ts")),
  ).toEqual([]);
});
