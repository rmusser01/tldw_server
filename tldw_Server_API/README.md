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


FastAPI has a bug, which is caused by starlette, caused by python.
- The gist is that you can't kill the python server on windows without killing the process itself, or issuing a 'stop' command from within the process.
- 


- Launch the API:
  - `python -m uvicorn tldw_Server_API.app.main:app --reload`
  - Visit the API via `127.0.0.1:8000/docs`



## Endpoints

### /media

- `GET /media`
  - Returns a list of all media items in the database.
    - Example response:
      ```json
      {
        "items": [
          {
            "id": 4,
            "title": "ARCHIVE INTERVIEW: Robert Leyland - Jumpin' Jack Software - Early Saturn Development",
            "url": "/api/v1/media/4"
          },
          {
            "id": 3,
            "title": "Everything You Wanted to Know About LLM Post-Training, with Nathan Lambert of Allen Institute for AI",
            "url": "/api/v1/media/3"
          },
          {
            "id": 111,
            "title": "How One Of NYC's Best Chefs Makes Pancakes | Made to Order | Bon Appétit",
            "url": "/api/v1/media/111"
          },
        "pagination": {
          "page": 1,
          "results_per_page": 10,
          "total_pages": 12
        }
      }
    ```
  
- `POST /media`
-  Adds a new media item to the database.
  - Request body:
    ```json
    {
    }
    ```
  - Example response:
    ```json
    {
    }
    ```
    
- `GET /media/{media_id}`
-  Retrieves a specific media item by its ID.
  - Example response:
    ```json
    {
    }
    ```
    
- `PUT /media/{media_id}`
- Updates a specific media item by its ID.
  - Request body:
    ```json
    {
    }
    ```
  - Example response:
    ```json
    {
    }
    ```
    

- `DELETE /media/{media_id}`
- Deletes a specific media item by its ID.
  - Example response:
    ```json
    {
    }
    ```

- `GET /media/search/{query}`
  - Searches for media items based on a query string.
  - Search Fields:
      - `search_query`: A string to search for in the media title, description, or keywords.
      - `keywords`: A comma-separated list of keywords to filter the search results.
      - `page`: The page number for pagination (default is 1).
      - `results_per_page`: The number of results per page (default is 10).
    - Example Queries:
      1. Basic text search:
         - ```bash
           curl -X GET "http://<URL>/api/v1/media/search?search_query=climate%20change" -H "Accept: application/json"
           ```
         * `/search?search_query=climate%20change`
      2. Search by keywords only:
        - ```bash
          curl -X GET "http://<URL>/api/v1/media/search?keywords=technology,innovation" -H "Accept: application/json"
          ```
        * `/search?keywords=technology,innovation`
      3. Combined search with pagination:
        - ```bash
               curl -X GET "http://<URL>/api/v1/media/search?search_query=renewable%20energy&keywords=solar,wind&page=2&results_per_page=15" -H "Accept: application/json"
             ```
        * `/search?search_query=renewable%20energy&keywords=solar,wind&page=2&results_per_page=15`
- Example response:
     ```json
     {
       "results": [
         {
           "id": 109,
           "url": "https://www.youtube.com/watch?v=WpoYqs9Hsa4",
           "title": "Why You Can't Learn from Mistakes",
           "type": "video",
           "content_preview": "{\n  \"webpage_url\": \"https://www.youtube.com/watch?v=WpoYqs9Hsa4\",\n  \"title\": \"Why You Can't Learn from Mistakes\",\n  \"description\": \"Learn more in Dr. K's Guide to Mental Health: https://bit.ly/3B53...",
           "author": "HealthyGamerGG",
           "date": "2025-03-21T00:00:00",
           "keywords": [
             "default"
           ]
         }
       ],
       "pagination": {
         "page": 1,
         "per_page": 10,
         "total": 13,
         "total_pages": 2
       }
     }
     ```
- Empty response if no results are found:
  ```json
  {
    "results": [],
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total": 0,
      "total_pages": 0
    }
  }
  ```

