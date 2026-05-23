// @vitest-environment jsdom

import React from "react"
import axe from "axe-core"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TemplatePreviewPane } from "../TemplatePreviewPane"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return key
    }
  })
}))

vi.mock("antd", () => {
  const RadioButton = ({ value, children, checked, onSelect }: any) => (
    <button
      type="button"
      data-testid={`template-preview-mode-${value}`}
      aria-pressed={Boolean(checked)}
      onClick={() => onSelect?.(value)}
    >
      {children}
    </button>
  )

  const RadioGroup = ({ value, onChange, children }: any) => (
    <div role="radiogroup">
      {React.Children.map(children, (child: any) =>
        React.cloneElement(child, {
          checked: child.props.value === value,
          onSelect: (nextValue: string) => onChange?.({ target: { value: nextValue } })
        })
      )}
    </div>
  )

  const Select = ({ value, onChange, options = [], allowClear }: any) => (
    <select
      data-testid="template-preview-run-select"
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const next = event.currentTarget.value
        onChange?.(next === "" ? undefined : Number(next))
      }}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option: { label: string; value: number }) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )

  const Button = ({ children, onClick, loading: _loading, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  return {
    Alert: ({ title, description }: any) => (
      <div role="alert">
        <div>{title}</div>
        {description ? <div>{description}</div> : null}
      </div>
    ),
    Button,
    Radio: {
      Group: RadioGroup,
      Button: RadioButton
    },
    Select,
    Spin: () => <div aria-label="loading">loading</div>
  }
})

vi.mock("@/services/watchlists", () => ({
  previewWatchlistTemplate: vi.fn()
}))

const runA11yBaselineRules = async (context: Element) =>
  axe.run(context, {
    runOnly: {
      type: "rule",
      values: [
        "button-name",
        "link-name",
        "label",
        "aria-valid-attr",
        "aria-valid-attr-value",
        "aria-required-attr"
      ]
    },
    resultTypes: ["violations"]
  })

const expectNoInvalidAriaViolations = (
  violations: Array<{
    id: string
  }>
) => {
  const disallowedIds = new Set([
    "aria-valid-attr",
    "aria-valid-attr-value",
    "aria-required-attr"
  ])

  const disallowedViolations = violations.filter((violation) =>
    disallowedIds.has(violation.id)
  )

  expect(disallowedViolations).toEqual([])
}

describe("TemplatePreviewPane accessibility stage-1 baseline", () => {
  it("enforces aria-name baseline checks for preview mode controls", async () => {
    const { container } = render(
      <TemplatePreviewPane
        content="# Digest"
        format="md"
        runs={[{ id: 12, label: "Run #12" }]}
      />
    )

    fireEvent.click(screen.getByTestId("template-preview-mode-live"))
    expect(screen.getByTestId("template-preview-run-select")).toBeInTheDocument()

    const results = await runA11yBaselineRules(container)
    expectNoInvalidAriaViolations(results.violations)
    expect(results.violations.map((violation) => violation.id)).not.toContain("button-name")
    expect(results.violations.map((violation) => violation.id)).not.toContain("link-name")
  }, 15000)
})
