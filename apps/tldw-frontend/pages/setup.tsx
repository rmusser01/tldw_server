import dynamic from "next/dynamic"
import Head from "next/head"

const Setup = dynamic(() => import("@/routes/option-setup"), { ssr: false })

export default function SetupPage() {
  return <><Head><title>Setup | tldw</title></Head><Setup /></>
}
