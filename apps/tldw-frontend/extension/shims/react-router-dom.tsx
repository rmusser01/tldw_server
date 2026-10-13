import React from "react"
import NextLink from "next/link"
import { useRouter } from "next/router"

type NextLinkProps = React.ComponentProps<typeof NextLink>

type LinkProps = Omit<NextLinkProps, "href"> & {
  to?: string
  href?: NextLinkProps["href"]
}

type NavLinkClassName =
  | string
  | ((props: { isActive: boolean }) => string | undefined)

type NavLinkProps = Omit<LinkProps, "className"> & {
  className?: NavLinkClassName
}

type NavigateOptions = {
  replace?: boolean
  state?: unknown
  flushSync?: boolean
}

type NavigateTo =
  | string
  | number
  | {
      pathname?: string
      search?: string
      hash?: string
    }

export type NavigateFunction = (to: NavigateTo, options?: NavigateOptions) => void

type RouteParams = Record<string, string | undefined>

type BlockerLocation = {
  pathname: string
  search: string
  hash: string
  state: null
  key: string
}

type BlockerFunctionArgs = {
  currentLocation: BlockerLocation
  nextLocation: BlockerLocation
  historyAction: "PUSH" | "POP" | "REPLACE"
}

type BlockerHookArg = boolean | ((args: BlockerFunctionArgs) => boolean)

type ShimBlocker = {
  state: "unblocked" | "blocked" | "proceeding"
  location?: BlockerLocation
  proceed: () => void
  reset: () => void
}

type PromptOptions = {
  when: boolean
  message: string
}

type CancelledNavigationError = Error & { cancelled: true }

const createCancelledNavigationError = (): CancelledNavigationError =>
  Object.assign(new Error("Navigation cancelled by route guard"), { cancelled: true as const })

const isCancelledNavigationError = (error: unknown): error is CancelledNavigationError =>
  error instanceof Error && (error as Partial<CancelledNavigationError>).cancelled === true

const runNavigationTransition = (
  update: () => void,
  options?: { flushSync?: boolean }
) => {
  if (options?.flushSync) {
    update()
    return
  }
  if (typeof React.startTransition === "function") {
    React.startTransition(update)
    return
  }
  update()
}

const normalizeDelimitedSegment = (
  value: string | undefined,
  delimiter: "?" | "#"
) => {
  const trimmed = value?.trim() ?? ""
  if (!trimmed) return ""
  return trimmed.startsWith(delimiter) ? trimmed : `${delimiter}${trimmed}`
}

const formatNavigateHref = (
  to: Exclude<NavigateTo, number>,
  fallbackPath: string
) => {
  if (typeof to === "string") return to
  const href = `${to.pathname ?? ""}${normalizeDelimitedSegment(
    to.search,
    "?"
  )}${normalizeDelimitedSegment(to.hash, "#")}`
  return href || fallbackPath
}

const noop = () => {}

export const UNSAFE_DataRouterContext = React.createContext<unknown | null>(null)

export const Link = React.forwardRef<HTMLAnchorElement, LinkProps>(
  function Link({ to, href, onClick, target, ...rest }, ref) {
    const navigate = useNavigate()
    const resolvedHref = href ?? to ?? "#"
    return (
      <NextLink
        ref={ref}
        href={resolvedHref}
        target={target}
        onClick={(event) => {
          onClick?.(event)
          if (
            event.defaultPrevented ||
            event.button !== 0 ||
            event.metaKey ||
            event.ctrlKey ||
            event.shiftKey ||
            event.altKey ||
            (target && target !== "_self") ||
            typeof resolvedHref !== "string"
          ) {
            return
          }
          event.preventDefault()
          navigate(resolvedHref)
        }}
        {...rest}
      />
    )
  }
)
Link.displayName = "Link"

export const NavLink = React.forwardRef<HTMLAnchorElement, NavLinkProps>(
  function NavLink({ to, href, className, ...rest }, ref) {
    const router = useRouter()
    const resolvedHref = href ?? to ?? "#"
    const targetPathSource =
      typeof resolvedHref === "string" ? resolvedHref : resolvedHref?.pathname ?? "#"
    const currentPath = router.asPath.split("?")[0]
    const targetPath = targetPathSource.split("?")[0]
    const isActive = currentPath === targetPath
    const resolvedClassName =
      typeof className === "function" ? className({ isActive }) : className

    return (
      <Link
        ref={ref}
        to={typeof resolvedHref === "string" ? resolvedHref : undefined}
        href={resolvedHref}
        className={resolvedClassName}
        {...rest}
      />
    )
  }
)
NavLink.displayName = "NavLink"

export const useNavigate = () => {
  const currentRouter = useRouter()
  const routerRef = React.useRef(currentRouter)
  routerRef.current = currentRouter
  // Next republishes its public router during hydration/Fast Refresh. Keep
  // effect dependencies stable without retaining an obsolete router snapshot.
  return React.useCallback((to: NavigateTo, options?: NavigateOptions) => {
    const router = routerRef.current
    if (typeof to === "number") {
      if (to < 0) {
        runNavigationTransition(
          () => {
            router.back()
          },
          { flushSync: options?.flushSync }
        )
      }
      return
    }
    const href = formatNavigateHref(to, router.asPath)
    const doFallback = () => {
      if (typeof window === "undefined") return
      const proto = window.location.protocol
      if (proto === "chrome-extension:" || proto === "moz-extension:") {
        window.location.hash = `#${href}`
        return
      }
      window.location.assign(href)
    }

    try {
      runNavigationTransition(
        () => {
          const navigation = options?.replace
            ? router.replace(href)
            : router.push(href)
          void navigation.catch((err) => {
            if (isCancelledNavigationError(err)) return
            console.error("[useNavigate shim] Navigation failed:", err)
            doFallback()
          })
        },
        { flushSync: options?.flushSync }
      )
    } catch (err) {
      if (isCancelledNavigationError(err)) return
      console.error("[useNavigate shim] Navigation failed:", err)
      doFallback()
    }
  }, [])
}

const useUnstablePrompt = ({ when, message }: PromptOptions): void => {
  const router = useRouter()
  const whenRef = React.useRef(when)
  const messageRef = React.useRef(message)
  const popRouteBypassRef = React.useRef<string | null>(null)
  whenRef.current = when
  messageRef.current = message

  React.useEffect(() => {
    const handleRouteStart = (url: string, options?: unknown) => {
      const bypassRoute = popRouteBypassRef.current
      popRouteBypassRef.current = null
      if (bypassRoute === url) return
      if (!whenRef.current || window.confirm(messageRef.current)) return
      const error = createCancelledNavigationError()
      router.events.emit("routeChangeError", error, url, options)
      throw error
    }
    const handleBeforePopState = (state: { url: string; as: string }) => {
      if (!whenRef.current) return true
      if (!window.confirm(messageRef.current)) return false
      popRouteBypassRef.current = state.as || state.url
      return true
    }

    router.events.on("routeChangeStart", handleRouteStart)
    router.events.on("hashChangeStart", handleRouteStart)
    router.beforePopState(handleBeforePopState)
    return () => {
      popRouteBypassRef.current = null
      router.events.off("routeChangeStart", handleRouteStart)
      router.events.off("hashChangeStart", handleRouteStart)
      router.beforePopState(() => true)
    }
  }, [router])
}

export { useUnstablePrompt as unstable_usePrompt }

export const useLocation = () => {
  const router = useRouter()
  return React.useMemo(() => {
    // Router and browser URLs can advance separately during a transition.
    const url = new URL(router.asPath || router.pathname, "http://localhost")
    return {
      pathname: url.pathname,
      search: url.search,
      hash: url.hash,
      state: null,
      key: router.asPath
    }
  }, [router.asPath, router.pathname])
}

export const useParams = <
  TParams extends Record<string, string | undefined> = Record<string, string | undefined>
>() => {
  const router = useRouter()

  return React.useMemo(() => {
    const params: Record<string, string | undefined> = {}
    for (const [key, value] of Object.entries(router.query || {})) {
      params[key] = Array.isArray(value) ? value[0] : value
    }
    return params as Readonly<TParams>
  }, [router.query])
}

export const useSearchParams = (): [
  URLSearchParams,
  (next: URLSearchParams | Record<string, string>, options?: NavigateOptions) => void
] => {
  const router = useRouter()
  const params = React.useMemo(() => {
    const queryString = router.asPath.split("?")[1] || ""
    return new URLSearchParams(queryString)
  }, [router.asPath])

  const setSearchParams = React.useCallback(
    (
      next: URLSearchParams | Record<string, string>,
      options?: NavigateOptions
    ) => {
      const nextParams =
        next instanceof URLSearchParams ? next : new URLSearchParams(next)
      const queryString = nextParams.toString()
      // Use the *actual* current path, not router.pathname (which is the
      // `[bracket]` dynamic-route pattern) so setSearchParams works on routes
      // like /sources/[id].
      const currentPath = router.asPath.split("?")[0].split("#")[0]
      const nextPath = queryString
        ? `${currentPath}?${queryString}`
        : currentPath
      runNavigationTransition(
        () => {
          const navigation = options?.replace
            ? router.replace(nextPath)
            : router.push(nextPath)
          void navigation.catch((error) => {
            if (isCancelledNavigationError(error)) return
            console.error("[useSearchParams shim] Navigation failed:", error)
          })
        },
        { flushSync: options?.flushSync }
      )
    },
    [router]
  )

  return [params, setSearchParams]
}

const toBlockerLocation = (href: string): BlockerLocation => {
  const url = new URL(href || "/", "http://localhost")
  return { pathname: url.pathname, search: url.search, hash: url.hash, state: null, key: href }
}

type PendingNavigation = { href: string; action: "PUSH" | "POP" }

const UNBLOCKED: ShimBlocker = { state: "unblocked", proceed: noop, reset: noop }

/**
 * react-router's useBlocker for the Next pages router. A blocked push is
 * cancelled from routeChangeStart and a blocked Back/Forward from
 * beforePopState; proceed() replays it once and reset() stays (restoring the
 * URL after a held Back press). `useBlocker(false)` stays inert so it never
 * replaces another guard's beforePopState handler.
 */
const useNextRouterBlocker = (shouldBlock: BlockerHookArg): ShimBlocker => {
  const router = useRouter()
  const routerRef = React.useRef(router)
  routerRef.current = router
  const shouldBlockRef = React.useRef(shouldBlock)
  shouldBlockRef.current = shouldBlock
  const pendingRef = React.useRef<PendingNavigation | null>(null)
  const bypassRef = React.useRef<string | null>(null)
  const [pending, setPending] = React.useState<PendingNavigation | null>(null)
  const active = shouldBlock !== false

  React.useEffect(() => {
    if (!active) return
    const shouldHold = (href: string, action: PendingNavigation["action"]) => {
      // Once one navigation is held, later ones wait behind it.
      if (pendingRef.current) return true
      const value = shouldBlockRef.current
      if (typeof value !== "function") return Boolean(value)
      return Boolean(
        value({
          currentLocation: toBlockerLocation(routerRef.current.asPath),
          nextLocation: toBlockerLocation(href),
          historyAction: action
        })
      )
    }
    const hold = (navigation: PendingNavigation) => {
      pendingRef.current = navigation
      setPending(navigation)
    }
    const handleRouteStart = (url: string, options?: unknown) => {
      if (bypassRef.current === url) {
        bypassRef.current = null
        return
      }
      if (!shouldHold(url, "PUSH")) return
      hold({ href: url, action: "PUSH" })
      const error = createCancelledNavigationError()
      router.events.emit("routeChangeError", error, url, options)
      throw error
    }
    const handleBeforePopState = (state: { url: string; as: string }) => {
      const href = state.as || state.url
      if (!shouldHold(href, "POP")) return true
      hold({ href, action: "POP" })
      return false
    }

    router.events.on("routeChangeStart", handleRouteStart)
    router.events.on("hashChangeStart", handleRouteStart)
    router.beforePopState(handleBeforePopState)
    return () => {
      router.events.off("routeChangeStart", handleRouteStart)
      router.events.off("hashChangeStart", handleRouteStart)
      router.beforePopState(() => true)
    }
  }, [active, router])

  const replay = React.useCallback((href: string, method: "push" | "replace") => {
    bypassRef.current = href
    const navigation = routerRef.current[method](href)
    void Promise.resolve(navigation).catch((error) => {
      if (bypassRef.current === href) bypassRef.current = null
      if (isCancelledNavigationError(error)) return
      console.error("[useBlocker shim] Navigation failed:", error)
    })
  }, [])

  const proceed = React.useCallback(() => {
    const held = pendingRef.current
    if (!held) return
    pendingRef.current = null
    setPending(null)
    // Next finishes a Back press itself with a replace; do the same.
    replay(held.href, held.action === "POP" ? "replace" : "push")
  }, [replay])

  const reset = React.useCallback(() => {
    const held = pendingRef.current
    if (!held) return
    pendingRef.current = null
    setPending(null)
    // After a held Back press the address bar already shows the target.
    if (held.action === "POP") replay(routerRef.current.asPath, "push")
  }, [replay])

  return React.useMemo(
    () =>
      pending
        ? { state: "blocked", location: toBlockerLocation(pending.href), proceed, reset }
        : UNBLOCKED,
    [pending, proceed, reset]
  )
}

/** Shared code checks this flag to know the shim can block (see route-leave-guard). */
export const useBlocker = Object.assign(useNextRouterBlocker, { tldwNextRouterShim: true as const })

export const useInRouterContext = () => true

export const Routes: React.FC<{ children?: React.ReactNode }> = ({
  children
}) => <>{children}</>

export const Route: React.FC<{
  element?: React.ReactNode
  path?: string
  index?: boolean
  children?: React.ReactNode
}> = ({ element, children }) => <>{element ?? children ?? null}</>

export const HashRouter: React.FC<{ children?: React.ReactNode }> = ({
  children
}) => <>{children}</>

export const MemoryRouter: React.FC<{ children?: React.ReactNode }> = ({
  children
}) => <>{children}</>

type NavigateProps = {
  to: string
  replace?: boolean
  state?: unknown
}

export const Navigate: React.FC<NavigateProps> = ({ to, replace }) => {
  const router = useRouter()
  React.useEffect(() => {
    runNavigationTransition(() => {
      const navigation = replace ? router.replace(to) : router.push(to)
      void navigation.catch((error) => {
        if (isCancelledNavigationError(error)) return
        console.error("[Navigate shim] Navigation failed:", error)
      })
    })
  }, [router, to, replace])
  return null
}
