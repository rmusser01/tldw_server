# UAT221 independent blocking review finding

TASK13260.159. Initial frozen manifest65c283d8932874125b721ed08b049f98487102f0d694198335365d3e683c36d9. Main89/8PASS, but additional actual-hook/cache regression1FAIL/12PASS. Allfour frozen hashes matched before/after; exactcopies are retained in initial-snapshot/.

A graph request for radius1 is already running when radius2 receives403. Clearing existing authority-scoped paginated cache does not retire that request. Its late200 repopulates the radius1 cache; the current mounted graph stays hidden by local denial, but after unmount/reopen at radius1 the private graph is immediately returned from cache before revalidation. This violates the same denied-cache/reopen contract covered by221. The private probe uses the actual hook, QueryClient and graph service with only API transport doubled, appended through a nonmutating Vite loader.

Evidence: late-success-probe.tsx.txt, probes.config.ts, late-success-probe.log (1FAIL/12filtered), late-success-full-red.log (1FAIL/12PASS), source-before.json and first-release-source-after.json. The failing assertion is reopened.result.current.graph expectednull but received the private graph. Initial12permanentcontrols remain green. No production edits by reviewer.

Parent and author notified before repair. Parent approved reassessment of cancellation/cache boundaries and correction within221; review remains open until refreeze. Ordinary transient errors, explicit recovery, other authorities and existing ownership boundaries must remain intact. No native acceptance claim is withdrawn by this synthetic race; it is additional coverage.
