const isFocusableElement = (element: HTMLElement): boolean => {
  if ("disabled" in element && Boolean((element as HTMLButtonElement).disabled)) {
    return false;
  }
  return true;
};

const isHiddenBySelfOrAncestor = (element: HTMLElement): boolean => {
  let current: HTMLElement | null = element;
  while (current) {
    if (current.hidden || current.getAttribute("aria-hidden") === "true") {
      return true;
    }
    if (typeof window !== "undefined" && window.getComputedStyle) {
      const style = window.getComputedStyle(current);
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        style.visibility === "collapse"
      ) {
        return true;
      }
    }
    current = current.parentElement;
  }
  return false;
};

export const normalizeFocusSelector = (
  selector: unknown,
): string | null => {
  if (typeof selector !== "string") return null;
  const trimmedSelector = selector.trim();
  return trimmedSelector.length > 0 ? trimmedSelector : null;
};

export const getFirstVisibleFocusableElement = (
  selector: string,
  root?: ParentNode,
): HTMLElement | null => {
  if (typeof document === "undefined" && !root) return null;

  const normalizedSelector = normalizeFocusSelector(selector);
  if (!normalizedSelector) return null;

  const searchRoot = root ?? document;
  let candidates: HTMLElement[] = [];
  try {
    candidates = Array.from(
      searchRoot.querySelectorAll<HTMLElement>(normalizedSelector),
    );
  } catch {
    return null;
  }
  return (
    candidates.find(
      (candidate) =>
        isFocusableElement(candidate) && !isHiddenBySelfOrAncestor(candidate),
    ) ||
    candidates.find((candidate) => isFocusableElement(candidate)) ||
    candidates[0] ||
    null
  );
};

export const focusFirstVisibleElement = (
  selector: string,
  root?: ParentNode,
): boolean => {
  if (typeof document === "undefined") return false;
  const target = getFirstVisibleFocusableElement(selector, root ?? document);
  target?.focus();
  return Boolean(target);
};

export const scheduleFocusFirstVisibleElement = (
  selector: string,
  root?: ParentNode,
): void => {
  const focusTarget = () => {
    focusFirstVisibleElement(selector, root);
  };

  if (typeof window !== "undefined" && window.requestAnimationFrame) {
    window.requestAnimationFrame(focusTarget);
    return;
  }

  globalThis.setTimeout(focusTarget, 0);
};
