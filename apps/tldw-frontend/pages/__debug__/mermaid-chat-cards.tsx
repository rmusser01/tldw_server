import dynamic from "next/dynamic"
import React from "react"

import { Markdown } from "@/components/Common/Markdown"
import { MermaidDiagramBlock } from "@/components/Common/MermaidDiagramBlock"

const validMermaidSource = `flowchart TD
  A["Assistant response"] --> B["MermaidDiagramBlock"]
  B --> C["Browser QA"]`

const assistantMarkdown = `Here is the assistant-facing Mermaid fixture.

\`\`\`mermaid
${validMermaidSource}
\`\`\`
`

const userMessageSource = `\`\`\`mermaid
${validMermaidSource}
\`\`\``

const disabledMarkdown = `Mermaid disabled fallback:

\`\`\`mermaid
${validMermaidSource}
\`\`\`
`

const invalidMermaidMarkdown = `Invalid Mermaid fallback:

\`\`\`mermaid
not a valid mermaid diagram @@@
\`\`\`
`

const graphvizMarkdown = `Graphviz should remain code:

\`\`\`dot
digraph G {
  A -> B;
}
\`\`\`
`

type HarnessSectionProps = {
  children: React.ReactNode
  testId: string
  title: string
}

const HarnessSection = ({ children, testId, title }: HarnessSectionProps) => (
  <section
    className="rounded-lg border border-border bg-surface p-4 shadow-sm"
    data-testid={testId}
  >
    <h2 className="mb-3 text-sm font-semibold text-text">{title}</h2>
    {children}
  </section>
)

const MermaidChatCardsHarness = () => (
  <main
    className="min-h-screen bg-bg px-6 py-8 text-text"
    data-testid="mermaid-chat-card-harness"
  >
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <header>
        <h1 className="text-xl font-semibold">Mermaid Chat Card QA</h1>
      </header>

      <HarnessSection
        testId="mermaid-harness-assistant"
        title="Assistant Mermaid render"
      >
        <Markdown enableMermaidDiagrams message={assistantMarkdown} />
      </HarnessSection>

      <HarnessSection
        testId="mermaid-harness-user"
        title="User message unchanged"
      >
        <pre className="overflow-auto whitespace-pre-wrap rounded-md bg-surface2 p-3 text-xs text-text">
          {userMessageSource}
        </pre>
      </HarnessSection>

      <HarnessSection
        testId="mermaid-harness-disabled"
        title="Setting-off fallback"
      >
        <Markdown message={disabledMarkdown} />
      </HarnessSection>

      <HarnessSection
        testId="mermaid-harness-invalid"
        title="Invalid Mermaid fallback"
      >
        <Markdown enableMermaidDiagrams message={invalidMermaidMarkdown} />
      </HarnessSection>

      <HarnessSection
        testId="mermaid-harness-graphviz"
        title="Graphviz/DOT fallback"
      >
        <Markdown enableMermaidDiagrams message={graphvizMarkdown} />
      </HarnessSection>

      <HarnessSection
        testId="mermaid-harness-artifact"
        title="Artifact-style Mermaid card"
      >
        <MermaidDiagramBlock
          artifactContextId="debug-mermaid-chat-card"
          blockIndex={0}
          enableArtifactAction
          source={validMermaidSource}
        />
      </HarnessSection>
    </div>
  </main>
)

export default dynamic(() => Promise.resolve(MermaidChatCardsHarness), {
  ssr: false
})
