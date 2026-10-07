import dynamic from "next/dynamic"

// Loads `routes/document-workspace.ts`, which co-locates the react-pdf layer
// CSS with this async chunk instead of importing it app-wide from `_app`.
export default dynamic(
  () => import("@web/routes/document-workspace").then((m) => m.default),
  { ssr: false }
)
