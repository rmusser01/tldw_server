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

const runNavigationTransition = (update: () => void) => {
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
  function Link({ to, href, ...rest }, ref) {
    const resolvedHref = href ?? to ?? "#"
    return <NextLink ref={ref} href={resolvedHref} {...rest} />
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
      <NextLink
        ref={ref}
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
        runNavigationTransition(() => {
          router.back()
        })
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
      runNavigationTransition(() => {
        const navigation = options?.replace
          ? router.replace(href)
          : router.push(href)
        void navigation.catch((err) => {
          console.error("[useNavigate shim] Navigation failed:", err)
          doFallback()
        })
      })
    } catch (err) {
      console.error("[useNavigate shim] Navigation failed:", err)
      doFallback()
    }
  }
}

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
      const nextPath = queryString
        ? `${router.pathname}?${queryString}`
        : router.pathname
      runNavigationTransition(() => {
        const navigation = options?.replace
          ? router.replace(nextPath)
          : router.push(nextPath)
        void navigation.catch((error) => {
          console.error("[useSearchParams shim] Navigation failed:", error)
        })
      })
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
        console.error("[Navigate shim] Navigation failed:", error)
      })
    })
  }, [router, to, replace])
  return null
}
