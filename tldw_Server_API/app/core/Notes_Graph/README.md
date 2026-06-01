# Notes Graph

Notes_Graph builds bounded graph views over notes using manual note links, wikilinks, backlinks, tag relationships, source relationships, and recency constraints. It also provides a small in-memory cache and Cytoscape formatter for API consumers that need graph-shaped note data.

## Start Here

- `graph_service.py` builds note graph responses with node, edge, radius, tag, source, and time filters.
- `wikilink_parser.py` extracts supported note id wikilinks from note content.
- `graph_cache.py` provides a TTL cache for graph responses.
- `formatters.py` converts graph responses to Cytoscape-compatible JSON.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`.
- Related tests: `tldw_Server_API/tests/Notes_Graph/unit/` and `tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py`.

## Responsibilities

- Parse `[[id:<UUID>]]` wikilinks from note text.
- Build note graph nodes and edges for manual links, wikilinks, backlinks, tag membership, and source membership.
- Enforce graph caps for node count, edge count, and per-node degree.
- Support radius-limited graph expansion and neighbor lookups.
- Optionally cache graph responses by request parameters.
- Format graph responses for Cytoscape consumers.

## Module Map

- `graph_service.py`: graph expansion, pruning, filtering, edge construction, and metrics hooks.
- `wikilink_parser.py`: supported wikilink extraction.
- `graph_cache.py`: thread-safe TTL cache with max-key eviction.
- `formatters.py`: Cytoscape response conversion.
- `__init__.py`: package marker.

## How It Connects

- `notes_graph.py` exposes graph routes under the notes API surface: `/notes/graph`, `/notes/{note_id}/neighbors`, `POST /notes/{note_id}/links`, and `DELETE /notes/links/{edge_id}`.
- The endpoint uses ChaChaNotes DB dependencies, AuthNZ permissions, token-scope guards, and rate limiting.
- Graph data comes from the notes database, note link rows, note content, keywords, and source metadata.
- Environment variables such as `NOTES_GRAPH_ENABLED`, `NOTES_GRAPH_MAX_NODES`, `NOTES_GRAPH_MAX_EDGES`, `NOTES_GRAPH_MAX_DEGREE`, and cache settings tune runtime behavior.

## Extension Points

- Add an edge type in `graph_service.py`, schemas, and endpoint parsing together.
- Change wikilink syntax in `wikilink_parser.py` and update parser tests.
- Add response formats in `formatters.py` and route handling in `notes_graph.py`.
- Enable or tune caching by injecting `GraphCache` where the service is constructed.
- Adjust graph caps in `graph_service.py` and verify pruning behavior.

## Testing

- Unit tests for the parser, cache, and graph service live under `tldw_Server_API/tests/Notes_Graph/unit/`.
- Endpoint integration coverage for `/graph` and `/neighbors` lives in `tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py`.

## Gotchas

- The parser intentionally supports `[[id:<UUID>]]` links, not arbitrary title links.
- Radius 2 requests apply stricter built-in caps than caller-supplied maximums.
- The graph feature can be disabled with `NOTES_GRAPH_ENABLED`.
