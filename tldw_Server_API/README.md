# tldw server Rewrite

- Page to cover API.


- Code Calling Pipeline:
  * You write the logic in `/app/core/<library>` and any backgroundable service processing in `/app/services`
  * Then you call into the library/ies via `/api/v1/<route>`
  * Which the routes themselves are defined in `main.py`

- So to add a new route/API Endpoint,
  - Write the endpoint in `main.py`
  - Write the handling logic of the endpoint in `/api/v1/<route>`
  - Write the majority/main of the processing logic in `/app/core/<library>`
  - Write any background-able service processing in `/app/services`
  - 


FastAPI has a bug, which is caused by starlette, caused by python.
- The gist is that you can't kill the python server on windows without killing the process itself, or issuing a 'stop' command from within the process.
- 


- Launch the API:
  - `python -m uvicorn tldw_Server_API.app.main:app --reload`
  - Visit the API via `127.0.0.1:8000/docs`

- Launching tests
  - `pip install pytest httpx`
  - `python -m pytest test_media_versions.py -v`
  - `python -m pytest .\tldw_Server_API\tests\Media_Ingestion_Modification\test_media_versions.py -v
`


- API Providers/Key checks defined in `/app/api/v1/schemas/chat_request_schemas.py`

## Endpoints


