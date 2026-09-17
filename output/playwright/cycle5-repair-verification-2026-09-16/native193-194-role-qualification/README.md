# UAT193/194 PostgreSQL role qualification

Read-only inspection of the owned native profile on2026-09-17 confirms rolsuper=true and rolbypassrls=true. Existing forced row-security policies do not protect queries run under that role. The isolated two-owner authorization failures are likewise demonstrated under a bypass role; they do not prove leakage under an ordinary restricted role. Separate verified NOSUPERUSER/NOBYPASSRLS causal controls pass foreign-row hiding and owner-changing-update rejection. Agent reports and exact control evidence will be retained with the character repair.

UAT193 still has a distinct global character-name uniqueness/bootstrap problem. Supported privileged service-role behavior and application predicates remain under review for194. Native evidence only observed public/default characters; no private-character attack was performed. No native database writes were made for this diagnostic.

Known runtime credentials and JWT/PEM patterns scanned, zero matches. Original evidence remains unchanged.
