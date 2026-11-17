// @vitest-environment jsdom
/**
 * CM-08 (#3101, G17): inline code in chat answers, user messages and the
 * Notes preview must read as code (monospace on a tinted pill) and must not
 * show the literal backticks Tailwind Typography adds around `.prose code`.
 *
 * jsdom has no stylesheet, so this compiles the clients' real Tailwind config
 * against the rendered markup and checks which emitted rules reach the
 * inline <code> element.
 */
import React from "react"
import { createRequire } from "node:module"
import { render } from "@testing-library/react"
import postcss, { type Root, type Rule } from "postcss"
import tailwindcss from "tailwindcss"
import { describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

const require = createRequire(import.meta.url)
const clients = ["tldw-frontend", "extension"] as const

/** Wrapper classes the chat message bubble passes to Markdown. */
const CHAT_MARKDOWN_CLASSES =
  "prose break-words text-message dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark max-w-none"

const reply =
  "Parse section numbering (`1.`, `1.1`, `(a)`) and prepend the breadcrumb."

const compile = async (client: (typeof clients)[number], html: string) => {
  const config = require(`../../../../../../${client}/tailwind.config.js`)
  const result = await postcss([
    tailwindcss({ ...config, content: [{ raw: html, extension: "html" }] })
  ]).process("@tailwind base; @tailwind components; @tailwind utilities;", {
    from: undefined
  })
  return result.root
}

/** Selectors of `rule` that match `element` once a pseudo-element is removed. */
const matchingSelectors = (rule: Rule, element: Element, pseudo: string | null) =>
  rule.selectors.filter((selector) => {
    const hasPseudo = pseudo ? selector.includes(pseudo) : !selector.includes("::")
    if (!hasPseudo) return false
    const plain = pseudo ? selector.split(pseudo).join("") : selector
    try {
      return element.matches(plain)
    } catch {
      return false
    }
  })

const declarationsFor = (
  root: Root,
  element: Element,
  property: string,
  pseudo: string | null = null
) => {
  const values: string[] = []
  root.walkRules((rule) => {
    if (rule.parent?.type === "atrule") return
    if (matchingSelectors(rule, element, pseudo).length === 0) return
    rule.walkDecls(property, (decl) => {
      values.push(decl.value)
    })
  })
  return values
}

const renderInlineCode = (className?: string) => {
  const view = render(
    <Markdown message={reply} codeBlockVariant="github" className={className} />
  )
  const codes = Array.from(view.container.querySelectorAll("code"))
  expect(codes.map((code) => code.textContent)).toEqual(["1.", "1.1", "(a)"])
  return { view, code: codes[0] }
}

describe.each(clients)("Markdown inline code in %s", (client) => {
  it.each([
    ["a chat reply", CHAT_MARKDOWN_CLASSES],
    ["the Notes preview (default classes)", undefined]
  ])(
    "does not add backticks around inline code in %s",
    async (_surface, className) => {
      const { view, code } = renderInlineCode(className)
      const root = await compile(client, view.container.innerHTML)

      for (const pseudo of ["::before", "::after"]) {
        const contents = declarationsFor(root, code, "content", pseudo)
        expect(
          contents.filter((value) => value.includes("`")),
          `${pseudo} content`
        ).toEqual([])
      }
    }
  )

  it("styles inline code as a monospace pill, not bold text", async () => {
    const { view, code } = renderInlineCode(CHAT_MARKDOWN_CLASSES)
    const root = await compile(client, view.container.innerHTML)

    expect(declarationsFor(root, code, "font-family").join(" ")).toMatch(/mono/i)
    expect(declarationsFor(root, code, "background-color")).not.toEqual([])
    expect(declarationsFor(root, code, "font-weight")).not.toContain("600")
  })
})
