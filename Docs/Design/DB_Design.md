# DB Design

## Table of Contents
- [General DB](#general-db)
- [SQLite](#sqlite)
- [SQLite DB Design](#sqlite-db-design)
- [Interesting/Relevant Later](#interesting-relevant-later)



SQLite
    https://highperformancesqlite.com/watch/dot-commands
    https://www.youtube.com/watch?v=XP-h304N06I
    https://rivet.gg/blog/2025-02-16-sqlite-on-the-server-is-misunderstood
    https://vlcn.io/docs/cr-sqlite/intro
    https://devina.io/sqlite-tester

Migrating to sqlite-vec
    https://www.youtube.com/live/xmdiwdom6Vk?t=1740s
    https://alexgarcia.xyz/blog/2024/sqlite-vec-metadata-release/index.html
    https://alexgarcia.xyz/sqlite-vec/features/vec0.html
    https://docs.google.com/document/d/1sJ_S2ggfFmtPJupxIO3C1EZAFuDMUfNYcAytissbFMs/edit?tab=t.0#heading=h.xyau1jyb6vyx
    https://github.com/Mozilla-Ocho/llamafile/pull/644


https://github.com/erikbern/ann-benchmarks/
https://github.com/dicroce/hnsw

https://github.com/GreenmaskIO/greenmask
https://blog.vectorchord.ai/postgresql-full-text-search-fast-when-done-right-debunking-the-slow-myth
https://github.com/1yefuwang1/vectorlite



https://github.com/integral-business-intelligence/chroma-auditor
https://ai.plainenglish.io/top-interview-questions-on-data-modeling-concepts-3d1587c86214
https://briandouglas.ie/sqlite-defaults/
https://phiresky.github.io/blog/2020/sqlite-performance-tuning/
https://kerkour.com/sqlite-for-servers
https://wafris.org/blog/rearchitecting-for-sqlite
    General DB:
    https://en.wikipedia.org/wiki/Database_normalization
    https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/
    
    Gradio:
    https://www.gradio.app/guides/connecting-to-a-database
    https://www.gradio.app/docs/gradio/dataset
    
    SQLite:
    https://www.sqlite.org/queryplanner.html
    https://www.sqlite.org/optoverview.html
    https://www.sqlite.org/queryplanner-ng.html
    https://docs.python.org/3/library/sqlite3.html
    
    SQLite Vector Search:
    https://github.com/asg017/sqlite-vec
    https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html
    https://news.ycombinator.com/item?id=40243168
    [test-tokenizer-random.py](..%2F..%2F..%2FGithub%2Fllama.cpp%2Ftests%2Ftest-tokenizer-random.py)
    SQLite DB Design:
    https://stackoverflow.com/questions/66293837/smart-way-to-structure-my-sqlite-database?rq=3
    https://stackoverflow.com/questions/7235435/sqlite-structure-advice?rq=3
    https://stackoverflow.com/questions/19368506/very-basic-sqlite-table-design?rq=3
    https://stackoverflow.com/questions/7665735/how-do-i-organize-such-database-in-sqlite?rq=3
    https://stackoverflow.com/questions/29055263/sql-database-layout-design?rq=3
    
    Seems like it might be interesting/relevant later:
    https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/
    
    
    https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
    https://www.youtube.com/watch?v=r09tJfON6kE
    
    
    error handling, vacuuming, zipping transcriptions above X size, and external storage for documents and said zipped transcriptions
