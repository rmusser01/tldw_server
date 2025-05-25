# Documentation regarding construction of a 'coding' page/endpoint

## Introduction




### Link Dump:
repo2txt integration/clone https://github.com/abinthomasonline/repo2txt
https://www.anthropic.com/engineering/claude-code-best-practices
https://github.com/ymichael/open-codex
https://github.com/brandondocusen/CntxtPY
https://github.com/cyclotruc/gitdigest
https://github.com/simonw/files-to-prompt
https://github.com/yamadashy/repomix/tree/main
https://github.com/chanhx/crabviz
https://carper.ai/diff-models-a-new-way-to-edit-code/
https://www.partywave.site/show/research/Graph-Based%20Source%20Code%20Analysis
https://github.com/abinthomasonline/repo2txt
https://github.com/charmandercha/ArchiDoc
https://github.com/microsandbox/microsandbox
https://github.com/jezweb/roo-commander
https://pythontutor.com/c.html#mode=edit
https://pythontutor.com/articles/c-cpp-visualizer.html
https://pmbanugo.me/blog/peer-programming-with-llms
https://gitingest.com/
https://www.qodo.ai/blog/rag-for-large-scale-code-repos/
https://blog.voyageai.com/2024/12/04/code-retrieval-eval/
https://github.com/sammcj/ingest
https://gitdiagram.com/
https://www.ilograph.com/blog/posts/diagrams-ai-can-and-cannot-generate/#system-diagramming-with-ai
https://github.com/osanseviero/geminicoder
https://blog.val.town/blog/fast-follow/
https://glean.software/
https://github.com/AtakanTekparmak/exec-python/tree/main
https://huggingface.co/blog/andthattoo/dpab-a
https://github.com/firstbatchxyz/function-calling-eval
https://codemirror.net/
https://github.com/codemirror/dev
https://github.com/google/diff-match-patch
https://highlightjs.org/
https://github.com/The-Pocket/Tutorial-Codebase-Knowledge/blob/main/docs/design.md
https://github.com/The-Pocket/PocketFlow
https://ariana.dev/blog/introducing-ariana#how-we-design-it
https://github.com/The-Pocket/Tutorial-Codebase-Knowledge
https://ampcode.com/how-to-build-an-agent



```
Claude Code – Hidden‑Gem Power Tips

Seed every repo with a tuned CLAUDE.md – Claude pulls it into every prompt; use # in‑session to append live notes, add "IMPORTANT / YOU MUST" for stronger adherence, and run /init to scaffold one instantly.
Escalate Claude's reasoning on demand – insert the words "think" → "think hard" → "think harder" → "ultrathink" to unlock larger thinking budgets before it acts.
Stop or rewind in one keystroke – Esc halts any phase without losing context; double‑Esc lets you edit an earlier prompt and branch a new path.
Bullet‑proof autonomy with Safe YOLO – run claude --dangerously‑skip‑permissions only inside an isolated container (no internet) to let Claude mass‑fix lint errors or boilerplate unattended.
Lightning CLAUDE.md hierarchy – supports *root, parent, child, or *~/.claude locations, so monorepos inherit both global and per‑package guidance automatically.
One‑word tool whitelisting – /allowed-tools Edit Bash(git commit:*) (or --allowedTools) removes future permission prompts for trusted actions.
Turbo‑charge context with images – paste screenshots or give file paths; Claude compares mock vs. output visually and iterates UI until pixels match.
Use the gh CLI as a super‑power – Claude speaks gh fluently: drafts PRs, fixes review comments, triages issues, and writes commit messages that cite history.
Sub‑agent sanity checks – ask Claude to spin up sub‑agents that independently verify plans or guard against over‑fitting to your tests.
Markdown checklists as scratchpads – have Claude dump giant lint/problem lists to a .md file, tick off items as it auto‑fixes, and keep progress auditable.
Shift‑Tab toggles "auto‑accept" – let Claude run without pauses when you trust the current context, then flip it off to resume step‑wise control.
Parallel velocity via git worktrees – spawn several worktrees, one Claude per terminal, each on a separate branch; no merge conflicts, maximum throughput.
Headless mode for CI bots – claude -p "<prompt>" --output-format stream-json plugs into pipelines to label new GitHub issues or run subjective linting.
Invoke the golden workflow – Explore → Plan (no code) → Implement → Commit; the upfront plan request dramatically lifts success rates.
TDD on steroids – let Claude write failing tests, commit them, then iterate code until green; ask a second Claude to review the implementation cold.
Power‑phrase file targeting – tab‑complete file names in prompts so Claude jumps to exactly the right locations, saving context budget.
**Debug MCP setups with **--mcp-debug – prints a verbose trace of server discovery and tool hand‑offs to surface config mistakes instantly.
Keep these in your muscle memory and Claude Code jumps from helpful to game‑changing.
```


