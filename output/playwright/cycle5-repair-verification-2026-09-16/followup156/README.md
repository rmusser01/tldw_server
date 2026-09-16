# UAT156: canonical chronology with preserved local content

Matched saved rows use valid canonical creation times even when raw RAG questions or protected edits remain local. This prevents an acknowledged question from moving below its answer after reload.

Author and independent required-scope runs each pass128tests/7suites. Exact baseline replay reproduces13failures/4controls. Author ESLint0errors/0warnings; root full compiler retains90existing signatures with0added/removed. TypeScript-only Bandit is not applicable.

Independent extra probes expose an existing invalid captured-local NaN fallback limit on both baseline and current source; those failure logs are retained. The actual loader maps invalid dates to undefined and that path passes. No currently reachable native corruption path was established; this repair does not sanitize every corrupt local record.

Native source send/reload acceptance remains pending. UAT013 wrong answers and UAT103 history-promotion identity are separate unresolved findings.
