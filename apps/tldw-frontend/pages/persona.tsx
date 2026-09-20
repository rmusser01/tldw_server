import dynamic from 'next/dynamic'

const PersonaGarden = dynamic(() => import('@/routes/sidepanel-persona'), { ssr: false })

export default function PersonaPage() {
  return <PersonaGarden shell="options" />
}
