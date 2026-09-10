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

type BlockerHookArg = boolean | ((...args: unknown[]) => boolean)

type ShimBlocker = {
  state: "unblocked" | "blocked" | "proceeding"
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
  const router = useRouter()
  return (to: NavigateTo, options?: NavigateOptions) => {
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
  }
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
  const search =
    typeof window === "undefined" ? "" : window.location.search || ""
  const hash = typeof window === "undefined" ? "" : window.location.hash || ""
  const pathname = router.asPath.split("?")[0].split("#")[0] || router.pathname
  return React.useMemo(
    () => ({
      pathname,
      search,
      hash,
      state: null,
      key: router.asPath
    }),
    [pathname, router.asPath, search, hash]
  )
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

export const useBlocker = (_when: BlockerHookArg): ShimBlocker =>
  React.useMemo(
    () => ({
      state: "unblocked",
      proceed: noop,
      reset: noop
    }),
    []
  )

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
