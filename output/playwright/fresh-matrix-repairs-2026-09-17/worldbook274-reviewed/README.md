# Reviewed World Book entry identifier repair (UAT274)

The entry manager maps canonical API id into its existing internal entry_id at one boundary. The same numeric identifier reaches edit, delete, selected bulk actions and relationship calculations. The existing transport and backend contract are unchanged.

A retained baseline UI regression reproduces the undefined edit identifier. Current focused tests and independent review are included with exact scope and static-check limitations. This packet is implementation review only; original PostgreSQL book3/entry1 native acceptance remains pending. UAT275 summary refresh is a separate repair. No full-matrix acceptance is claimed.

Evidence is exact bytes or lossless gzip with credential scanning; excluded test-dependency links and any omitted inputs are listed in the manifest.
