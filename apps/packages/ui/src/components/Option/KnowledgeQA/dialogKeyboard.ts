import type { KeyboardEvent } from "react";

// AntD restores focus, but its last Tab can leave the document before focusin fires.
export function containDialogTab(event: KeyboardEvent<HTMLElement>): void {
  if (event.key !== "Tab" || event.defaultPrevented) return;
  const dialog = (event.target as HTMLElement).closest('[role="dialog"]');
  if (!dialog) return;
  const controls = Array.from(
    dialog.querySelectorAll<HTMLElement>(
      "a[href], button, textarea, input, select, [tabindex]",
    ),
  ).filter(
    (node) =>
      !node.matches(":disabled") &&
      node.tabIndex >= 0 &&
      !node.closest('[inert], [aria-hidden="true"]') &&
      getComputedStyle(node).visibility !== "hidden" &&
      Boolean(
        node.getBoundingClientRect().width ||
        node.getBoundingClientRect().height,
      ),
  );
  const first = controls[0];
  const last = controls[controls.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last?.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first?.focus();
  }
}
