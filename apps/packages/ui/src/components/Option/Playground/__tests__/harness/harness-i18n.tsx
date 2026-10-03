/**
 * `react-i18next` stand-in for the Playground harness.
 *
 * Resolves keys against the shipped English locale files first (so controls
 * carry the labels users see, e.g. "Regenerate", "Save & Send"), then the
 * caller's default value, then the key. `t` is stable per namespace so hooks
 * that list it as an effect dependency do not loop.
 */
import React from "react"

const localeModules = import.meta.glob("../../../../../assets/locale/en/*.json", {
  eager: true,
  import: "default"
}) as Record<string, Record<string, unknown>>

const resources: Record<string, Record<string, unknown>> = Object.fromEntries(
  Object.entries(localeModules).map(([path, value]) => [
    path.split("/").pop()!.replace(/\.json$/, ""),
    value
  ])
)

const lookup = (namespace: string, key: string): string | undefined => {
  let value: unknown = resources[namespace]
  for (const part of key.split(".")) {
    if (!value || typeof value !== "object") return undefined
    value = (value as Record<string, unknown>)[part]
  }
  return typeof value === "string" ? value : undefined
}

const interpolate = (template: string, options?: Record<string, unknown>) =>
  options
    ? template.replace(/\{\{\s*([\w.]+)\s*\}\}/g, (_match, token: string) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    : template

const createT = (defaultNamespace: string) =>
  (key: string | string[], fallbackOrOptions?: unknown, maybeOptions?: Record<string, unknown>) => {
    const rawKey = Array.isArray(key) ? key[0] : key
    let fallback: string | undefined
    let options: Record<string, unknown> | undefined
    if (typeof fallbackOrOptions === "string") {
      fallback = fallbackOrOptions
      options = maybeOptions
    } else if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
      options = fallbackOrOptions as Record<string, unknown>
      if (typeof options.defaultValue === "string") fallback = options.defaultValue
    }
    const separator = rawKey.indexOf(":")
    const namespace = separator > 0 ? rawKey.slice(0, separator) : defaultNamespace
    const path = separator > 0 ? rawKey.slice(separator + 1) : rawKey
    const count = options?.count
    const plural =
      typeof count === "number"
        ? lookup(namespace, `${path}_${count === 1 ? "one" : "other"}`)
        : undefined
    const template = plural ?? lookup(namespace, path) ?? fallback ?? path
    return interpolate(template, options)
  }

const translators = new Map<string, ReturnType<typeof createT>>()
const translatorFor = (namespace: string) => {
  let t = translators.get(namespace)
  if (!t) {
    t = createT(namespace)
    translators.set(namespace, t)
  }
  return t
}

const i18n = {
  language: "en",
  resolvedLanguage: "en",
  languages: ["en"],
  isInitialized: true,
  changeLanguage: async () => undefined,
  on: () => undefined,
  off: () => undefined,
  t: translatorFor("common"),
  exists: () => true,
  hasLoadedNamespace: () => true,
  loadNamespaces: async () => undefined
}

export const reactI18nextStandIn = {
  useTranslation: (namespace?: string | string[]) => {
    const resolved = Array.isArray(namespace) ? namespace[0] : namespace
    return { t: translatorFor(resolved || "common"), i18n, ready: true }
  },
  Trans: ({
    i18nKey,
    defaults,
    children,
    ns
  }: {
    i18nKey?: string
    defaults?: string
    children?: React.ReactNode
    ns?: string
  }) =>
    React.createElement(
      React.Fragment,
      null,
      children ?? (i18nKey ? translatorFor(ns || "common")(i18nKey, defaults) : defaults ?? null)
    ),
  withTranslation: () => (Component: React.ComponentType<Record<string, unknown>>) => (props: Record<string, unknown>) =>
    React.createElement(Component, { ...props, t: translatorFor("common"), i18n }),
  initReactI18next: { type: "3rdParty", init: () => undefined },
  I18nextProvider: ({ children }: { children: React.ReactNode }) =>
    React.createElement(React.Fragment, null, children)
}
