# tldw API Rewrite

- Page to cover API.


- Code Calling Pipeline:
  * You write the logic in `/app/core/<library>` and any backgroundable service processing in `/app/services`
  * Then you call into the library/ies via `/api/v1/<route>`
  * Which the routes themselves are defined in `main.py`

- So to add a new route/API Endpoint,
  - Write the endpoint in `main.py`
  - Write the handling logic of the endpoint in `/api/v1/<route>`
  - Write the majority/main of the processing logic in `/app/core/<library>`
  - Write any backgroundable service processing in `/app/services`
  - 



- Launch the API:
  - `uvicorn main:app --reload`