# Round0 snapshot timing correction

The file named `round0-VisualIdentityPackPanel.test.tsx` was copied after the author had already applied the requested type annotation, before the author's completion message was read. Its actual SHA-256 `39cc5f9fa86a7bc96da973271e537c54c65c2d9c8c15263f87662dc2309c7522` is correctly recorded in `round0-audit.json`; it must not be treated as the original failing test bytes. That file, original audit, report and failing diagnostic remain unchanged.

`round0-frozen-test-reconstructed.tsx` reverses exactly the two annotation edits and independently reproduces the previously observed frozen SHA-256 `cbfbd96797cc5a4edfafd80c482bca2bcd6f85d7de219eb9ecc74c2c31fa41ea`. This reconstructed artifact, the original frozen hash inventory and original AST projection document the failing test. The production snapshot is unchanged in both rounds.
