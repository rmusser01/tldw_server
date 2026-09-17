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
      "public/**"
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
      "react-hooks/purity": "off",
      "react-hooks/preserve-manual-memoization": "off",
      "react-hooks/refs": "off",
      "react-hooks/set-state-in-effect": "off",
      "react-hooks/static-components": "off",
      "react-hooks/use-memo": "off",
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
    files: ["e2e/**/*.ts"],
    rules: {
      "react-hooks/rules-of-hooks": "off",
      "no-empty-pattern": "off"
    }
  }
]
