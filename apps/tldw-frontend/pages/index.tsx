import dynamic from "next/dynamic"
import Head from "next/head"

const Home = dynamic(() => import("@/routes/option-index"), { ssr: false })

export default function HomePage() {
  return <><Head><title>Home | tldw</title></Head><Home /></>
}
