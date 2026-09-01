'use client';

import { useEffect, useRef } from 'react';

type NavigateEventLike = Event & {
  destination?: { url?: string };
};

type NavigationLike = EventTarget & {
  addEventListener(type: 'navigate', listener: EventListener): void;
  removeEventListener(type: 'navigate', listener: EventListener): void;
};

const HISTORY_GUARD_KEY = '__tldwSensitiveNavigationGuard';
const blockers = new Map<symbol, () => void>();
let removeGlobalGuard: (() => void) | null = null;

const notifyBlocked = () => {
  for (const callback of blockers.values()) {
    callback();
  }
};

const destinationDiffers = (url: string | URL | null | undefined): boolean => {
  if (url === null || url === undefined) return false;
  try {
    return new URL(String(url), window.location.href).href !== window.location.href;
  } catch {
    return true;
  }
};

export const requestSensitiveNavigation = (
  url: string | URL | null | undefined,
): boolean => {
  if (blockers.size === 0 || !destinationDiffers(url)) return true;
  notifyBlocked();
  return false;
};

const installGlobalGuard = (): (() => void) => {
  const handleBeforeUnload = (event: BeforeUnloadEvent) => {
    event.preventDefault();
    event.returnValue = '';
  };
  const handleClick = (event: MouseEvent) => {
    if (
      event.defaultPrevented
      || event.button !== 0
      || event.metaKey
      || event.ctrlKey
      || event.shiftKey
      || event.altKey
    ) return;
    const target = event.target;
    const anchor = target instanceof Element ? target.closest('a[href]') : null;
    if (!(anchor instanceof HTMLAnchorElement)) return;
    if (anchor.target && anchor.target.toLowerCase() !== '_self') return;
    if (anchor.hasAttribute('download') || !destinationDiffers(anchor.href)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    notifyBlocked();
  };

  window.addEventListener('beforeunload', handleBeforeUnload);
  document.addEventListener('click', handleClick, true);

  const navigation = (window as Window & { navigation?: NavigationLike }).navigation;
  if (navigation) {
    const handleNavigate: EventListener = (rawEvent) => {
      const event = rawEvent as NavigateEventLike;
      if (!destinationDiffers(event.destination?.url)) return;
      if (event.cancelable) event.preventDefault();
      notifyBlocked();
    };
    navigation.addEventListener('navigate', handleNavigate);
    return () => {
      navigation.removeEventListener('navigate', handleNavigate);
      document.removeEventListener('click', handleClick, true);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }

  const marker = `guard-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;
  const stateWithMarker = (state: unknown) => ({
    ...(state !== null && typeof state === 'object' ? state : {}),
    [HISTORY_GUARD_KEY]: marker,
  });
  originalPushState.call(history, stateWithMarker(history.state), '', window.location.href);

  const guardedPushState: History['pushState'] = function guardedPushState(
    this: History,
    data,
    unused,
    url,
  ) {
    if (destinationDiffers(url)) {
      notifyBlocked();
      return;
    }
    originalPushState.call(this, data, unused, url);
  };
  const guardedReplaceState: History['replaceState'] = function guardedReplaceState(
    this: History,
    data,
    unused,
    url,
  ) {
    if (destinationDiffers(url)) {
      notifyBlocked();
      return;
    }
    originalReplaceState.call(this, data, unused, url);
  };
  history.pushState = guardedPushState;
  history.replaceState = guardedReplaceState;

  const handlePopState = (event: PopStateEvent) => {
    if (event.state?.[HISTORY_GUARD_KEY] === marker) return;
    event.stopImmediatePropagation();
    originalPushState.call(
      history,
      stateWithMarker(event.state),
      '',
      window.location.href,
    );
    notifyBlocked();
  };
  window.addEventListener('popstate', handlePopState);

  return () => {
    window.removeEventListener('popstate', handlePopState);
    if (history.pushState === guardedPushState) history.pushState = originalPushState;
    if (history.replaceState === guardedReplaceState) {
      history.replaceState = originalReplaceState;
    }
    if (history.state?.[HISTORY_GUARD_KEY] === marker) {
      const cleanState = { ...history.state };
      delete cleanState[HISTORY_GUARD_KEY];
      originalReplaceState.call(history, cleanState, '', window.location.href);
    }
    document.removeEventListener('click', handleClick, true);
    window.removeEventListener('beforeunload', handleBeforeUnload);
  };
};

const registerNavigationBlocker = (callback: () => void): (() => void) => {
  const token = Symbol('sensitive-navigation-blocker');
  blockers.set(token, callback);
  if (!removeGlobalGuard) removeGlobalGuard = installGlobalGuard();
  return () => {
    blockers.delete(token);
    if (blockers.size === 0 && removeGlobalGuard) {
      removeGlobalGuard();
      removeGlobalGuard = null;
    }
  };
};

export const useSensitiveNavigationGuard = (
  active: boolean,
  onBlocked: () => void,
) => {
  const callbackRef = useRef(onBlocked);

  useEffect(() => {
    callbackRef.current = onBlocked;
  }, [onBlocked]);

  useEffect(() => {
    if (!active) return;
    return registerNavigationBlocker(() => callbackRef.current());
  }, [active]);
};
