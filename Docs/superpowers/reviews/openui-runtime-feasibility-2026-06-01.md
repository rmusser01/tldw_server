# OpenUI Runtime Feasibility Review

Date: 2026-06-01
Backlog: TASK-495 (references TASK-493 plan)

## Packages Checked

| Package | Version | License | Unpacked size | React peer range |
| --- | --- | --- | --- | --- |
| `@openuidev/react-lang` | `0.2.6` | MIT | 188,362 bytes | `^18.3.1 || ^19.0.0` |
| `@openuidev/react-ui` | `0.11.8` | MIT | 12,163,175 bytes | `^18.3.1 || ^19.0.0` |
| `@openuidev/react-headless` | `0.8.2` | MIT | 329,688 bytes | `^18.3.1 || ^19.0.0` |

## Findings

- Package metadata: all three candidate packages exist, use the MIT license, and have React peer ranges compatible with `apps/tldw-frontend` (`react`/`react-dom` `^18.3.1`).
- React peer compatibility: `apps/extension` currently uses `react`/`react-dom` `18.2.0`, which does not satisfy the OpenUI peer range. OpenUI runtime dependencies are not added to the extension in Task 0; extension and sidepanel surfaces must remain source fallback until a later task verifies build compatibility or upgrades React.
- Zod peer compatibility: `@openuidev/react-lang` and `@openuidev/react-ui` require `zod` `^3.25.0 || ^4.0.0`; add `zod` `^4.0.0` with the web runtime dependency and as a shared UI peer.
- Zustand peer compatibility: `@openuidev/react-ui` and `@openuidev/react-headless` expect `zustand` `^4.5.5`. `apps/tldw-frontend` currently has `zustand` `^5.0.10`; this is a runtime risk for root OpenUI package imports. Task 4 should prefer `@openuidev/react-ui/genui-lib` imports rather than root chat/shell components unless the root import path is explicitly reviewed.
- Dynamic evaluation / CSP: verified tarball scan returned no matches for `eval(`, `new Function`, or `Function(` in `lang`, `ui`, or `headless` with source maps, README files, stories, and package manifests excluded.
- `dangerouslySetInnerHTML` / `innerHTML`: the only verified matches are OpenUI chart style injection paths:
  - `ui/package/dist/index.mjs:1982`
  - `ui/package/dist/index.cjs:2031`
  - `ui/package/dist/genui-lib/index.cjs:2112`
  - `ui/package/dist/genui-lib/index.mjs:2045`
  - `ui/package/dist/components/Charts/Charts.js:37`
  - `ui/package/dist/components/Charts/index.cjs:58`
- Chart style injection assessment: the matched paths create a `<style>` tag from chart theme/color configuration rather than arbitrary HTML. This is acceptable for the feasibility gate only if Task 4 keeps a chat/component subset or allowlist that avoids unsafe chart/style paths until explicitly reviewed.
- Bundle impact: `@openuidev/react-ui` is the dominant package at roughly 12.2 MB unpacked and brings Radix, recharts, react-markdown, syntax highlighting, KaTeX/remark/rehype, and chart dependencies. Runtime rendering must stay lazy-loaded and limited to the opted-in web chat surface.
- Component allowlist: initial production use should allow only reviewed chat-safe `genui-lib` components and should exclude chart/style-injection components until a separate review.
- Extension build risk: do not add OpenUI packages to `apps/extension/package.json` in Task 0. Later tasks must verify extension build behavior before any shared UI import path requires extension resolution.
- Install behavior: `bun install` resolved and extracted dependencies but hung in existing workspace postinstall behavior (`apps/extension` `node scripts/wxt-prepare.mjs`). The lockfile was updated with `bun install --ignore-scripts`; later build tasks must run the normal frontend/extension build checks explicitly rather than relying on postinstall completion.
- Lockfile result: `apps/bun.lock` now contains OpenUI package entries plus peer-isolated `zustand@4.5.7` entries for `@openuidev/react-ui` and `@openuidev/react-headless`, while the existing app `zustand` dependency remains unchanged.

## Decision

PASS

## Notes

- Feasibility PASS is conditional on the Task 4 mitigation: import from `@openuidev/react-ui/genui-lib` where possible, keep OpenUI behind capability/surface checks, lazy-load only on the web chat surface, and keep extension/workspace surfaces in source fallback unless explicitly enabled after build and CSP review.
- Verified baseline expectation for Task 1: `bunx vitest run packages/ui/src/utils/__tests__/dynamic-ui.test.ts` failed with `No test files found` for `packages/ui/src/utils/__tests__/dynamic-ui.test.ts`, confirming the next task starts red.
