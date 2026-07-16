# RAG Benchmarking — Quick Samples

These JSON payloads are ready to POST to the unified Evaluations API to run a small RAG pipeline benchmark against a tiny inline dataset.

Prereqs
- Server running: `uvicorn tldw_Server_API.app.main:app --reload`
- Auth: set `X-API-KEY` for single-user mode (or JWT for multi-user)

Create evaluation (inline dataset) and start a run

1) Create the evaluation
```
curl -sS -X POST http://127.0.0.1:8000/api/v1/evaluations \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  --data @Docs/Evals/samples/rag_pipeline_eval_inline.json
```

Note the `id` in the response as `EVAL_ID`.

2) Start a run for that evaluation
```
curl -sS -X POST http://127.0.0.1:8000/api/v1/evaluations/$EVAL_ID/runs \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  --data @Docs/Evals/samples/run_request.json
```

3) Check run status
```
curl -sS -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://127.0.0.1:8000/api/v1/evaluations/runs/<RUN_ID>
```

Optional: Create a dataset separately
```
curl -sS -X POST http://127.0.0.1:8000/api/v1/evaluations/datasets \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  --data @Docs/Evals/samples/dataset_quick.json
```
