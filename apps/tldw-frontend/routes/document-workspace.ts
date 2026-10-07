// Lazy entry for the document-workspace route.
//
// react-pdf's text/annotation layer CSS is imported here — inside the
// dynamically imported chunk — instead of `pages/_app.tsx` so ~10KB of PDF
// layer styles ship with the document-workspace route chunk rather than every
// page's global CSS (perf remediation W4 CSS co-location). The PDF viewer
// components that rely on these styles (packages/ui DocumentViewer) are only
// mounted through this route.
import "@/assets/react-pdf.css"

export { default } from "@/routes/option-document-workspace"
