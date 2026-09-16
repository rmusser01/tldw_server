# UAT160 narrow asset tag wrapping

TASK13260.97. One scoped asset-list class change bounds AntD Space children and wraps full tag text. Existing LatestBriefing and AudioReadinessStrip use the same wrapping convention. No content, actions or runtime logic changed.

Native RED: real GGUF list at390px had clientWidth306/scrollWidth483 and visibly clipped two warnings. After:306/306 with both complete warnings readable; all five inspected lists fit306px. Desktop908/908 with full metadata and warnings. Root inspected the after screenshot. Before screenshots and geometry remain in ../followup158/. Independent source/layout review is clear; existing real-AntD and asset action tests15/2pass, scoped lint0errors/0warnings.

This is a reversible CSS change with a real browser layout regression; no test that merely matches CSS strings was added. No new functions/Python; Bandit cannot analyze TSX. Full compiler run on this source plus103 retains90existing signatures, zero additions/removals (../followup103/independent-final-typecheck-comparison.json). Native download execution and broad admin certification are not claimed.
