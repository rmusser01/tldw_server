import { screen } from "@testing-library/react"
import { expect } from "vitest"

const expectDesignSystemAlertAncestor = (node: HTMLElement) => {
  const alert = node.closest('[data-ds-component="Alert"]')
  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("data-ds-component", "Alert")
  return alertEl
}

export const expectInsideDesignSystemAlert = (text: string | RegExp) => {
  return expectDesignSystemAlertAncestor(screen.getByText(text))
}

export const expectInsideDesignSystemAlertAsync = async (
  text: string | RegExp
) => {
  return expectDesignSystemAlertAncestor(await screen.findByText(text))
}
