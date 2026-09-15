import dynamic from "next/dynamic"
import Head from "next/head"

const Knowledge = dynamic(() => import("@/routes/option-knowledge"), { ssr: false })

export default function KnowledgePage() {
  return <><Head><title>Knowledge | tldw</title></Head><Knowledge /></>
}
