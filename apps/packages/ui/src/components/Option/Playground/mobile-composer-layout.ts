const DEFAULT_KEYBOARD_INSET_THRESHOLD = 90;

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value;
};

export type MobileComposerViewportState = {
  keyboardInsetPx: number;
  keyboardOpen: boolean;
};

export type ComposerDockLayoutMetrics = {
  occupiedHeightPx: number;
  keyboardInsetPx: number;
};

export const computeKeyboardInsetPx = (params: {
  layoutViewportHeight: number;
  visualViewportHeight: number;
  visualViewportOffsetTop?: number;
}): number => {
  const layoutHeight = toFiniteNumber(params.layoutViewportHeight);
  const visualHeight = toFiniteNumber(params.visualViewportHeight);
  const viewportOffsetTop = toFiniteNumber(params.visualViewportOffsetTop) ?? 0;

  if (
    layoutHeight == null ||
    visualHeight == null ||
    layoutHeight <= 0 ||
    visualHeight <= 0
  ) {
    return 0;
  }

  const inset = layoutHeight - (visualHeight + viewportOffsetTop);
  if (!Number.isFinite(inset) || inset <= 0) {
    return 0;
  }
  return Math.max(0, Math.round(inset));
};

export const isKeyboardLikelyOpen = (params: {
  keyboardInsetPx: number;
  thresholdPx?: number;
}): boolean => {
  const inset = toFiniteNumber(params.keyboardInsetPx);
  const threshold =
    toFiniteNumber(params.thresholdPx) ?? DEFAULT_KEYBOARD_INSET_THRESHOLD;
  if (inset == null) return false;
  return inset >= Math.max(32, Math.round(threshold));
};

export const resolveMobileComposerViewportState = (params: {
  layoutViewportHeight: number;
  visualViewportHeight: number;
  visualViewportOffsetTop?: number;
  thresholdPx?: number;
}): MobileComposerViewportState => {
  const keyboardInsetPx = computeKeyboardInsetPx(params);
  return {
    keyboardInsetPx,
    keyboardOpen: isKeyboardLikelyOpen({
      keyboardInsetPx,
      thresholdPx: params.thresholdPx,
    }),
  };
};

export const resolveStickyComposerTextareaMaxHeight = (params: {
  viewportHeightPx: number;
  keyboardInsetPx?: number;
  isMobileViewport: boolean;
  defaultMaxHeightPx: number;
}): number => {
  const viewportHeight = toFiniteNumber(params.viewportHeightPx);
  const keyboardInsetPx = toFiniteNumber(params.keyboardInsetPx) ?? 0;
  const defaultMaxHeightPx = Math.max(
    0,
    Math.round(toFiniteNumber(params.defaultMaxHeightPx) ?? 0),
  );
  const maxCap = params.isMobileViewport ? 220 : 320;
  const clampedDefaultMaxHeightPx = Math.min(maxCap, defaultMaxHeightPx);

  if (viewportHeight == null || viewportHeight <= 0) {
    return clampedDefaultMaxHeightPx;
  }

  const availableViewportHeight = Math.max(
    0,
    Math.round(
      viewportHeight - (params.isMobileViewport ? keyboardInsetPx : 0),
    ),
  );
  const viewportRatio = params.isMobileViewport ? 0.22 : 0.33;
  const targetHeight = Math.round(availableViewportHeight * viewportRatio);

  return Math.max(
    clampedDefaultMaxHeightPx,
    Math.min(maxCap, targetHeight),
  );
};

export const resolveComposerBottomOffsetPx = (
  metrics: ComposerDockLayoutMetrics | null | undefined,
): number => {
  if (!metrics) {
    return 0;
  }

  const occupiedHeightPx = Math.max(
    0,
    Math.round(toFiniteNumber(metrics.occupiedHeightPx) ?? 0),
  );
  const keyboardInsetPx = Math.max(
    0,
    Math.round(toFiniteNumber(metrics.keyboardInsetPx) ?? 0),
  );

  return occupiedHeightPx + keyboardInsetPx;
};
