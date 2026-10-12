import { cpSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const apps = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const web = resolve(apps, "tldw-frontend");
const require = createRequire(resolve(web, "package.json"));
const monacoRequire = createRequire(require.resolve("@monaco-editor/react"));
const installed = dirname(monacoRequire.resolve("monaco-editor/package.json"));
const source = resolve(installed, "min/vs");

/** Fixed build-owned paths; runtime serving never imports this module. */
export function prepareMonacoAssets() {
  for (const target of [
    resolve(web, "public/monaco"),
    resolve(apps, "packages/ui/src/public/monaco"),
  ]) {
    mkdirSync(target, { recursive: true });
    cpSync(source, resolve(target, "vs"), { recursive: true });
    for (const name of ["LICENSE", "ThirdPartyNotices.txt"])
      cpSync(resolve(installed, name), resolve(target, name));
    for (const label of ["editor", "json", "css", "html", "ts"]) {
      const name = readdirSync(resolve(source, "assets")).find(
        (file) => file.startsWith(`${label}.worker-`) && file.endsWith(".js"),
      );
      if (!name) throw new Error(`Missing installed Monaco ${label} worker`);
      writeFileSync(
        resolve(target, `${label}-worker.js`),
        `importScripts('./vs/assets/${name}');\nglobalThis.postMessage({type:'vscode-worker-ready'});\n`,
      );
    }
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  prepareMonacoAssets();
