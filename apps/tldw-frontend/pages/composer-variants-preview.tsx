import type { GetServerSideProps } from "next"
import dynamic from "next/dynamic"

/**
 * Dev-only preview harness for the Primer composer redesign variants.
 * Gated to non-production builds via `getServerSideProps` — the route
 * returns 404 when `NODE_ENV === "production"` so it doesn't ship to
 * end-users and search engines can't index it.
 */
export const getServerSideProps: GetServerSideProps = async () => {
  if (process.env.NODE_ENV === "production") {
    return { notFound: true }
  }
  return { props: {} }
}

export default dynamic(
  () => import("@/routes/composer-variants-preview"),
  { ssr: false }
)
