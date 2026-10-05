import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { render, renderHook, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Switch, Tooltip } from "antd"
import ts from "typescript"
import { describe, expect, it, vi } from "vitest"
import english from "@/assets/locale/en/playground.json"
import { usePersistenceMode } from "../usePersistenceMode"

const translate = (key: string, fallback?: string) => {
  let value: unknown = english
  for (const segment of key.replace("playground:", "").split(".")) {
    value =
      value && typeof value === "object"
        ? (value as Record<string, unknown>)[segment]
        : undefined
  }
  return typeof value === "string" ? value : fallback ?? key
}
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: translate })
}))

const historyHelp = "New messages can be stored in chat history."
const temporaryHelp =
  "New messages are not added to saved chat history. Draft text may remain on this device."

type ModeProps = {
  temporaryChat: boolean
  isConnectionReady: boolean
  serverChatId: string | null
  historyId: string | null
  onToggleTemporaryChat?: (next: boolean) => void
}
const initial: ModeProps = {
  temporaryChat: false,
  isConnectionReady: true,
  serverChatId: null,
  historyId: null
}

// Exercise the actual compact declarations/effect without mounting the whole
// multi-mode composer or replacing the production warning with a mock copy.
const source = ts.createSourceFile(
  "form.tsx",
  readFileSync(
    path.resolve(__dirname, "../../../components/Sidepanel/Chat/form.tsx"),
    "utf8"
  ),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
const snippets: Array<{ position: number; text: string }> = []
let actualToggle: ts.JsxElement | undefined
const modeNames = new Set([
  "previousServerChatIdRef",
  "persistenceModeLabel",
  "persistencePillLabel",
  "persistenceTooltip"
])
const visit = (node: ts.Node) => {
  if (
    ts.isVariableStatement(node) &&
    node.declarationList.declarations.some((declaration) =>
      ts.isIdentifier(declaration.name)
        ? modeNames.has(declaration.name.text)
        : ts.isObjectBindingPattern(declaration.name) &&
          declaration.name.elements.some(
            (element) =>
              ts.isIdentifier(element.name) &&
              element.name.text === "persistenceTooltip"
          )
    )
  )
    snippets.push({ position: node.pos, text: node.getText(source) })
  if (
    ts.isExpressionStatement(node) &&
    ts.isCallExpression(node.expression) &&
    node.expression.expression.getText(source) === "React.useEffect" &&
    node.expression.arguments[0]
      ?.getText(source)
      .includes("previousServerChatIdRef")
  )
    snippets.push({ position: node.pos, text: node.getText(source) })
  if (
    ts.isJsxElement(node) &&
    node.openingElement.tagName.getText(source) === "Tooltip" &&
    node.openingElement.attributes.properties.some(
      (attribute) =>
        ts.isJsxAttribute(attribute) &&
        attribute.name.getText(source) === "title" &&
        attribute.initializer?.getText(source) === "{persistenceTooltip}"
    )
  )
    actualToggle = node
  ts.forEachChild(node, visit)
}
visit(source)
if (!actualToggle) throw new Error("Missing actual compact history toggle")
if (!snippets.some((snippet) => snippet.text.includes("persistenceTooltip")))
  throw new Error("Missing actual compact persistence control")
const code = ts.transpileModule(
  `return function useCompactMode(props, notification) {
    const { temporaryChat, isConnectionReady, serverChatId, historyId } = props;
    const t = translate;
    const temporaryChatLocked = false, isFireFoxPrivateMode = false;
    const handleToggleTemporaryChat = props.onToggleTemporaryChat || (() => {});
    ${snippets
      .sort((a, b) => a.position - b.position)
      .map((snippet) => snippet.text)
      .join("\n")}
    return { persistenceTooltip, toggle: ${actualToggle.getText(source)} };
  }`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React } }
).outputText
const useCompactMode = new Function(
  "React",
  "translate",
  "usePersistenceMode",
  "Switch",
  "Tooltip",
  code
)(React, translate, usePersistenceMode, Switch, Tooltip) as (
  props: ModeProps,
  notification: { warning: ReturnType<typeof vi.fn> }
) => {
  persistenceTooltip: React.ReactElement
  toggle: React.ReactElement
}

const tooltipText = (node: React.ReactNode): string => {
  if (typeof node === "string" || typeof node === "number") return String(node)
  if (Array.isArray(node)) return node.map(tooltipText).join(" ")
  if (React.isValidElement<{ children?: React.ReactNode }>(node))
    return tooltipText(node.props.children)
  return ""
}

describe("chat history mode is independent of a save outcome", () => {
  it.each([
    { label: "connected fresh chat before the first turn", ...initial },
    { label: "disconnected fresh chat", ...initial, isConnectionReady: false },
    { label: "bare server reference", ...initial, serverChatId: "server-chat" },
    { label: "connected local reference", ...initial, historyId: "local-chat" },
    {
      label: "disconnected server reference",
      ...initial,
      isConnectionReady: false,
      serverChatId: "server-chat"
    }
  ])("describes $label as a mode without claiming a write", (props) => {
    const hook = renderHook(() => usePersistenceMode(props))
    expect({
      title: hook.result.current.persistencePillLabel,
      help: hook.result.current.persistenceModeLabel
    }).toEqual({
      title: "History on",
      help: historyHelp
    })
  })

  it("explains Temporary without promising transcript retention or deletion on close", () => {
    const hook = renderHook(() =>
      usePersistenceMode({
        ...initial,
        temporaryChat: true,
        serverChatId: "previous-chat"
      })
    )
    expect({
      title: hook.result.current.persistencePillLabel,
      help: hook.result.current.persistenceModeLabel
    }).toEqual({
      title: "Temporary",
      help: temporaryHelp
    })
  })

  it.each([null, "local-chat"])(
    "leaving server history for %s produces no save-failure warning",
    (historyId) => {
      const notification = { warning: vi.fn() }
      const hook = renderHook(
        (props: ModeProps) => useCompactMode(props, notification),
        {
          initialProps: { ...initial, serverChatId: "server-chat" }
        }
      )
      hook.rerender({ ...initial, historyId })
      expect(notification.warning).not.toHaveBeenCalled()
      expect(tooltipText(hook.result.current.persistenceTooltip)).toBe(
        `History on ${historyHelp}`
      )
    }
  )

  it("the compact control uses the same translated Temporary help", () => {
    const hook = renderHook(() =>
      useCompactMode({ ...initial, temporaryChat: true }, { warning: vi.fn() })
    )
    expect(tooltipText(hook.result.current.persistenceTooltip)).toBe(
      `Temporary ${temporaryHelp}`
    )
  })
})

describe("actual compact history switch semantics", () => {
  it.each([false, true])(
    "Temporary=%s agrees with checked state and keyboard activation",
    async (temporaryChat) => {
      const onToggleTemporaryChat = vi.fn()
      const props = { ...initial, temporaryChat, onToggleTemporaryChat }
      const CompactControl = () =>
        useCompactMode(props, { warning: vi.fn() }).toggle
      render(<CompactControl />)
      const switchControl = screen.getByRole("switch", {
        name: `Chat history: ${temporaryChat ? "Temporary" : "History on"}`
      })
      expect(switchControl).toHaveAttribute(
        "aria-checked",
        String(!temporaryChat)
      )
      switchControl.focus()
      await userEvent.setup().keyboard("{Enter}")
      expect(onToggleTemporaryChat).toHaveBeenCalledWith(!temporaryChat)
    }
  )
})
