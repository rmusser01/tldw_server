import dynamic from "next/dynamic"
import Head from "next/head"

const MediaAnalysis = dynamic(() => import("@/routes/option-media-multi"), { ssr: false })

export default function MediaAnalysisPage() {
  return <><Head><title>Media Analysis | tldw</title></Head><MediaAnalysis /></>
}
