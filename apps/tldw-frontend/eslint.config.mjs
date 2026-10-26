import js from "@eslint/js"
import globals from "globals"
import nextPlugin from "@next/eslint-plugin-next"
import react from "eslint-plugin-react"
import reactHooks from "eslint-plugin-react-hooks"
import tsPlugin from "@typescript-eslint/eslint-plugin"
import tsParser from "@typescript-eslint/parser"

const stripConfigs = (plugin) => {
  const { configs: _configs, ...rest } = plugin
  return rest
}

const nextRules = nextPlugin.configs["core-web-vitals"]?.rules ?? {}
const reactHooksRules = reactHooks.configs.recommended?.rules ?? {}
const reactRules = react.configs.recommended?.rules ?? {}
const tsRules = tsPlugin.configs.recommended?.rules ?? {}
const codeFiles = ["**/*.{js,jsx,ts,tsx,mjs,cjs}"]
const cjsFiles = ["**/*.cjs"]
const tsFiles = ["**/*.{ts,tsx}"]
// packages/ui has no eslint of its own (eslint and these plugins do not resolve from
// there), so scripts/check-shared-hooks.mjs lints it with this config from apps/.
// Globs that name packages/ui are therefore relative to apps/; `bun run lint` here
// never reaches them.
const ui = "packages/ui/src"
const nextRulesOff = Object.fromEntries(Object.keys(nextRules).map((rule) => [rule, "off"]))
// Tests may mock services/background-proxy and assert request paths.
const uiTests = ["**/__tests__/**", "**/*.test.*", "**/*.spec.*"]

// Shared UI boundaries (TASK-13950.10). Errors that predate a rule are frozen in
// eslint-suppressions.packages-ui.json; new ones fail frontend-required.
const keepWithReason = (rule) =>
  ` If it must stay, disable that line with "// eslint-disable-next-line ${rule} -- <reason>".`
// @/, ~/ and @tldw/ui/ all resolve to packages/ui/src. Only components/ has nested
// components/ or entries/ folders, so a relative ../ path to one is the top-level folder.
const boundary = (target, message) => ({
  regex: `^(?:(?:@|~|@tldw/ui)/|(?:\\.\\./)+)${target}`,
  message
})
const noComponents = boundary(
  "components(?:/|$)",
  "utils/, services/, store/ and hooks/ must not import components/: move what you need down into hooks/, utils/, services/ or types/ and import it from there."
)
const noEntries = boundary(
  "entries(?:/|$)",
  "components/ must not import entries/: move the shared piece into components/ or utils/ and let entries/ import it."
)
const noBackgroundProxy = boundary(
  "services/background-proxy$",
  "components/, hooks/ and routes/ must not import services/background-proxy: add or reuse a function in a services/ module (for example services/tldw/TldwApiClient.ts) and call that."
)
const importBoundaries = {
  components: [noEntries, noBackgroundProxy],
  hooks: [noComponents, noBackgroundProxy],
  routes: [noBackgroundProxy],
  services: [noComponents],
  store: [noComponents],
  utils: [noComponents]
}
const apiPathAsAny = [
  "TSAsExpression[typeAnnotation.type='TSAnyKeyword'] > Literal.expression[value=/^\\/api\\//]",
  "TSAsExpression[typeAnnotation.type='TSAnyKeyword'] > TemplateLiteral.expression[quasis.0.value.raw=/^\\/api\\//]"
].map((selector) => ({
  selector,
  message:
    "Do not cast an API path `as any`: it defeats the ClientPath guard in services/tldw/openapi-guard.ts. Drop the cast, or type the path as ClientPathRuntimeWithQuery; if the endpoint is missing, regenerate @tldw/api-types (`bun run generate:api-types`)." +
    keepWithReason("no-restricted-syntax")
}))
// Runtime strings that start with /api/v1/ (or a template part that does, as in
// `${base}/api/v1/x`); prose that mentions a path and type-level paths do not match.
const apiV1InComponents = [
  ":not(TSLiteralType) > Literal[value=/^\\/api\\/v1\\//]",
  "TemplateLiteral > TemplateElement[value.raw=/^\\/api\\/v1\\//]"
].map((selector) => ({
  selector,
  message:
    "Do not hard-code an /api/v1/ path in a component: call the endpoint through a services/ module, where the path is checked against ClientPath." +
    keepWithReason("no-restricted-syntax")
}))
const uiFolderRules = (folder, boundaries) => ({
  "no-restricted-imports": [
    "error",
    {
      patterns: boundaries.map(({ regex, message }) => ({
        regex,
        caseSensitive: true,
        message: message + keepWithReason("no-restricted-imports")
      }))
    }
  ],
  "no-restricted-syntax": [
    "error",
    ...apiPathAsAny,
    ...(folder === "components" ? apiV1InComponents : []),
    // no-restricted-imports does not see import(); this closes that gap.
    ...boundaries.map(({ regex, message }) => ({
      selector: `ImportExpression > Literal.source[value=/${regex.replaceAll("/", "\\/")}/]`,
      message: message + keepWithReason("no-restricted-syntax")
    }))
  ]
})

// Frozen chat monoliths (AC#4). Ceilings in code lines, measured on dev 39093fc827:
// [file, function]. Shrinking needs no edit. Raising one is a reviewed exception: change
// the number and add a "bump: <reason>" comment on that line.
const frozenUiFiles = {
  "components/Option/Playground/PlaygroundForm.tsx": [7552, 7067],
  "components/Sidepanel/Chat/form.tsx": [4829, 4601],
  "hooks/chat/useChatActions.ts": [5820, 5297],
  "hooks/useMessage.tsx": [4012, 3848],
  "routes/sidepanel-chat.tsx": [3274, 2774]
}
export const frozenUiFilePaths = Object.keys(frozenUiFiles).map((file) => `${ui}/${file}`)

const configFiles = [
  "**/*.config.js",
  "**/*.config.cjs",
  "**/*.config.mjs",
  "tailwind.config.js"
]

export default [
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      "public/**",
      // Vendored, minified assets (pdf.worker.min.mjs, theme-bootstrap.js).
      `${ui}/public/**`
    ]
  },
  {
    ...js.configs.recommended,
    files: codeFiles
  },
  {
    files: codeFiles,
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: { jsx: true }
      },
      globals: {
        ...globals.browser,
        ...globals.node,
        browser: "readonly",
        chrome: "readonly",
        React: "readonly",
        NodeJS: "readonly",
        JSX: "readonly",
        fetch: "readonly",
        process: "readonly"
      }
    },
    plugins: {
      "@next/next": stripConfigs(nextPlugin),
      "react": stripConfigs(react),
      "react-hooks": stripConfigs(reactHooks),
      "@typescript-eslint": stripConfigs(tsPlugin)
    },
    settings: {
      react: { version: "detect" }
    },
    rules: {
      ...tsRules,
      ...reactRules,
      ...reactHooksRules,
      ...nextRules,
      "react/display-name": "off",
      "react/no-unescaped-entities": "off",
      "react/prop-types": "off",
      "react-hooks/immutability": "off",
      "react-hooks/purity": "error",
      "react-hooks/preserve-manual-memoization": "off",
      "react-hooks/refs": "off",
      "react-hooks/set-state-in-effect": "off",
      "react-hooks/static-components": "error",
      "react-hooks/use-memo": "error",
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }
      ],
      "no-unused-vars": "off",
      "no-empty": ["error", { allowEmptyCatch: true }],
      "no-useless-escape": "warn",
      "no-useless-catch": "warn",
      "@typescript-eslint/triple-slash-reference": "off"
    }
  },
  {
    files: tsFiles,
    rules: {
      "no-undef": "off"
    }
  },
  {
    files: configFiles,
    rules: {
      "@typescript-eslint/no-require-imports": "off"
    }
  },
  {
    files: cjsFiles,
    languageOptions: {
      parserOptions: { sourceType: "script" }
    }
  },
  {
    // Shared UI is not a Next.js app (it also ships in the extension). Every folder
    // bans `as any` on API paths; folders with import boundaries extend this below.
    files: [`${ui}/**`],
    rules: {
      ...nextRulesOff,
      "no-restricted-syntax": ["error", ...apiPathAsAny]
    }
  },
  ...Object.entries(importBoundaries).map(([folder, boundaries]) => ({
    files: [`${ui}/${folder}/**`],
    rules: uiFolderRules(folder, boundaries)
  })),
  {
    files: uiTests.map((pattern) => `${ui}/${pattern}`),
    rules: { "no-restricted-imports": "off", "no-restricted-syntax": "off" }
  },
  ...Object.entries(frozenUiFiles).map(([file, [lines, functionLines]]) => ({
    files: [`${ui}/${file}`],
    rules: {
      "max-lines": ["error", { max: lines, skipBlankLines: true, skipComments: true }],
      "max-lines-per-function": ["error", { max: functionLines, skipBlankLines: true, skipComments: true }]
    }
  })),
  {
    files: ["e2e/**/*.ts"],
    rules: {
      "react-hooks/rules-of-hooks": "off",
      "no-empty-pattern": "off"
    }
  }
]
