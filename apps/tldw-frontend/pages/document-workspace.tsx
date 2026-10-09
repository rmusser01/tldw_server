import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-document-workspace"), {
  ssr: false
})
