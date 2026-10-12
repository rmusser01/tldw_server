import { loader } from "@monaco-editor/react";

// Configure the single loader before any lazy consumer can start its CDN default.
if (typeof window !== "undefined") {
  loader.config({
    paths: { vs: new URL("/monaco/vs", window.location.href).href },
  });
}
function installNativeWorkers() {
  (
    globalThis as typeof globalThis & {
      MonacoEnvironment?: {
        getWorker: (moduleId: string, label: string) => Worker;
      };
    }
  ).MonacoEnvironment = {
    getWorker: (_moduleId, label) =>
      new Worker(
        new URL(
          `/monaco/${label === "json" ? "json" : ["css", "scss", "less"].includes(label) ? "css" : ["html", "handlebars", "razor"].includes(label) ? "html" : ["typescript", "javascript"].includes(label) ? "ts" : "editor"}-worker.js`,
          window.location.href,
        ),
      ),
  };
}
export async function loadLocalMonaco() {
  const monaco = await loader.init();
  // Installed AMD entry assigns its own blob factory during initialization.
  installNativeWorkers();
  return monaco;
}
