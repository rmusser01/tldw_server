# Reviewed UAT232 ordinary-chat error correction

The previous targeted native run on a7d3155 still rendered generic recovery for an actual structured model_not_available400. Ordinary TldwChatService wraps the HTTP error in Error.cause. The minimal formatter correction walks that cause chain with a cycle guard and sanitizes the matched model error, retaining unrelated error handling.

Independent review is CLEAR:203tests/7files pass with zero skips; production-baseline replay fails exactly2guidance assertions with18controls; scoped ESLint0errors/0warnings; full differential compiler90unchangeddiagnostics. Initial i18n/storage fixture corrections, incorrect per-suite config launches and compiler memory failure are retained and explained by the review. Bandit cannot parse the three TypeScript files and provides no meaningful security coverage.

41reviewed inputs plus review artifacts are retained byte-for-byte with original/stored hashes and lossless gzip where needed. Known credential/JWT scans pass before retention. Native source-upgrade acceptance is still required; this packet does not close232 or release the full48-outcome matrix. The review test uses controlled HTTP/storage boundaries and does not establish native persistence.
