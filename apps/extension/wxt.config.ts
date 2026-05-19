import path from "node:path"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import { defineConfig } from "wxt"

import { getWxtTargetName, runPostBuildTasks } from "./scripts/post-build-tasks.mjs"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const sharedRoot = path.resolve(__dirname, "../packages/ui/src")
const require = createRequire(import.meta.url)
const wxtBrowserEntry = require.resolve("wxt/browser")

export default defineConfig({
  srcDir: sharedRoot,
  entrypointsDir: path.join(__dirname, "entrypoints"),
  publicDir: path.join(sharedRoot, "public"),
  hooks: {
    "build:done"(wxt) {
      if (wxt.config.command === "serve") {
        return
      }

      runPostBuildTasks({
        cwd: wxt.config.root,
        outDir: wxt.config.outDir,
        targetName: getWxtTargetName(
          wxt.config.browser,
          wxt.config.manifestVersion
        ),
      })
    },
  },
  manifest: {
    default_locale: "en",
    options_ui: {
      page: "options.html",
      open_in_tab: true
    },
    permissions: [
      "storage",
      "contextMenus",
      "activeTab",
      "notifications",
      "sidePanel",
      "tabs",
      "scripting"
    ],
    host_permissions: [
      "https://api.github.com/*"
    ],
    action: {
      default_title: "tldw Assistant",
      default_icon: {
        16: "icon/16.png",
        32: "icon/32.png",
        48: "icon/48.png",
        64: "icon/64.png",
        128: "icon/128.png"
      }
    },
    web_accessible_resources: [
      {
        resources: ["pdf.worker.min.mjs"],
        matches: ["<all_urls>"]
      }
    ]
  },
  vite: () => ({
    resolve: {
      alias: {
        "@": sharedRoot,
        "~": sharedRoot,
        "@tldw/ui": sharedRoot,
        "pa-tesseract.js": path.join(__dirname, "node_modules/pa-tesseract.js"),
        "wxt/browser": wxtBrowserEntry
      }
    },
    build: {
      modulePreload: false,
      rollupOptions: {
        onwarn(warning, warn) {
          if (
            warning.code === "MODULE_LEVEL_DIRECTIVE" &&
            warning.message?.includes('"use client"') &&
            warning.id?.includes("node_modules")
          ) {
            return
          }
          warn(warning)
        }
      }
    }
  })
})
