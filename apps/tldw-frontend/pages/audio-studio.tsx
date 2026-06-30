import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-audio-studio"), {
  ssr: false
})
