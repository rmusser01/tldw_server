import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-media-collection"), {
  ssr: false,
})
