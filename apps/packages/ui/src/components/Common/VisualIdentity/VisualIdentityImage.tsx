import React from "react"

const usePrefersReducedMotion = (): boolean => {
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false)

  React.useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return
    }

    const query = window.matchMedia("(prefers-reduced-motion: reduce)")
    setPrefersReducedMotion(query.matches)

    const handleChange = (event: MediaQueryListEvent) => {
      setPrefersReducedMotion(event.matches)
    }

    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", handleChange)
      return () => query.removeEventListener("change", handleChange)
    }

    query.addListener(handleChange)
    return () => query.removeListener(handleChange)
  }, [])

  return prefersReducedMotion
}

export type VisualIdentityImageProps = {
  assetUrl: string
  previewUrl?: string | null
  isAnimated?: boolean
  alt?: string
  className?: string
  style?: React.CSSProperties
  loading?: "eager" | "lazy"
  onClick?: () => void
}

export const VisualIdentityImage = ({
  assetUrl,
  previewUrl,
  isAnimated = false,
  alt = "",
  className = "h-full w-full object-cover",
  style,
  loading = "lazy",
  onClick
}: VisualIdentityImageProps) => {
  const prefersReducedMotion = usePrefersReducedMotion()
  const resolvedSrc =
    isAnimated && prefersReducedMotion && previewUrl ? previewUrl : assetUrl

  return (
    <img
      src={resolvedSrc}
      alt={alt}
      className={className}
      style={style}
      loading={loading}
      onClick={onClick}
    />
  )
}

export default VisualIdentityImage
