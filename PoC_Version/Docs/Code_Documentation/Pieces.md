# Pieces

## Table of Contents
- [What is this page?](#what-is-this-page)
- [Introduction](#introduction)
- [How Does it All Start?](#how-does-it-start)
- [Transcription / Summarization / Ingestion Tab](#transcription--summarization--ingestion-tab)
- [Audio File Transcription / Summarization](#audio_file_transcription)
- [](#)


Needs to be updated - 7/31

------------------------------------------------------------------------------------------------------------------
### <a name="what-is-this-page"></a>What is this page?
- Goal of this page is to help provide documentation and guidance for the various pieces of the project.
- This page will be updated as the project progresses and more pieces are added.
- Specifically, this page will walk through each of the functions exposed by the project and provide a brief overview of what they do and how they work by mapping the code functions used throughout the process.

- **What's in the Repo currently?**
  1. `summarize.py` - Main script for downloading, transcribing, and summarizing videos, audio files, books and documents.
  2. `config.txt` - Config file used for settings for main app.
  3. `requirements.txt` - Packages to install for Nvidia GPUs
  4. `AMD_requirements.txt` - Packages to install for AMD GPUs
  5. `llamafile` - Llama.cpp wrapper for local LLM inference, is multi-platform and multi-LLM compatible.
  6. `media_summary.db` - SQLite DB that stores all the data ingested, transcribed, and summarized.
  7. `prompts.db` - SQLite DB that stores all the prompts.
  8. `App_Function_Libraries` Folder - Folder containing the applications function libraries
  9. `Docs` - Folder containing documentation for the application
  10. `Tests` Folder - Folder containing tests for the application (ha.)
  11. `Helper_Scripts` - Folder containing helper scripts for the application
        * `DB-Related` folder
        * `Installer_Scripts` folder
        * `Parsing_Files` folder
        * `Prompts` folder
  12. `models` - Folder containing the models for the speaker diarization LLMs
  13. `tldw-original-scripts` - Original scripts from the original repo
- **What's in the original repo?**
  - `summarize.py` - download, transcribe and summarize audio
    1. First uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to download audio(optionally video) from supplied URL
    2. Next, it uses [ffmpeg](https://github.com/FFmpeg/FFmpeg) to convert the resulting `.m4a` file to `.wav`
    3. Then it uses [faster_whisper](https://github.com/SYSTRAN/faster-whisper) to transcribe the `.wav` file to `.txt`
    4. After that, it uses [pyannote](https://github.com/pyannote/pyannote-audio) to perform 'diarization'
    5. Finally, it'll send the resulting txt to an LLM endpoint of your choice for summarization of the text.
  - `chunker.py` - break text into parts and prepare each part for LLM summarization
  - `roller-*.py` - rolling summarization
    - [can-ai-code](https://github.com/the-crypt-keeper/can-ai-code) - interview executors to run LLM inference
  - `compare.py` - prepare LLM outputs for webapp
  - `compare-app.py` - summary viewer webapp

------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------
### <a name="introduction"></a>Introduction
- The project is broken down into several pieces, each of which is responsible for a specific task.
- The pieces are designed to be:
    - Modular and can be used independently of each other.
    - Used in conjunction with each other to provide a seamless experience for the user.
    - Easily extendable and can be modified to suit the needs of the user.
- The pieces are:
  - Ingestion
    - video/audio: yt-dlp, ffmpeg, faster_whisper, ctranslate2
    - text/ebook: pypandoc
    - website: requests, BeautifulSoup/trafilatura
    - pdf: marker pipeline
  - LLM Chat
    - plain web requests
  - Database Management
    - sqlite3 + python sqlite3 module
  - GUI
    - Gradio
------------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------------
### <a name="how-does-it-start"></a> How does it all start?
1. Execution starts with the `summarize.py` file.
2. The `summarize.py` file is responsible for starting the GUI and handling the user input.
3. Depending on the user input, the appropriate function is called to handle the user request.
   - This occurs on lines 696-910 of the `summarize.py` file.
4. For this, we are going to assume the user has passed the `-gui` argument to the script with no other arguments presented.
5. The GUI is started by calling the `launch_ui()` function on line 838.
   - The script first checks to see if the `--local_llm` argument was passed, first running it if it was, and then `launch_ui(share_public=False)`
   - The `launch_ui()` function is defined in `Gradio_Related.py` on line 3712.
6. The `launch_ui()` function is responsible for starting the GUI using the Gradio library.
   - As part of this function, the various tabs are aligned and ordered.
   - Each of the tabs are created using the appropriate `create_X_tab()` function.
7. From here, all functions are called as needed by the user from each individual tab within the Gradio UI.
8. The following sections will walk through each of the tabs and the functions they call to handle the user input.
------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------
### <a name="transcription--summarization--ingestion-tab"></a> Transcription / Summarization / Ingestion Tab
#### Video Transcription + Summarization
1. The tab for video transcription is the `create_video_transcription_tab()`
2. The function is defined in `Gradio_Related.py` on line 561.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Markdown()` -> `gr.Row():` functions.
5. The function then creates the input fields for the user to input the video URL and the desired summary length.
   - `custom_prompt_checkbox`, `preset_prompt_checkbox`, `preset_prompt`, `use_time_input`, `chunking_options_checkbox`, and `use_cookies_input` are used to dynamically show their corresponding items depending on the user selection.
6. The function then creates the output fields for the user to view the summary and the transcription.
7. Finally, the button `Process Videos` is created to handle the user input with the variable `process_button`
8. `process_button` when clicked, calls the `process_videos_wrapper()` function.
9. `process_videos_wrapper()` is defined on line 903 of `Gradio_Related.py`
    - The function first sets up file paths for `all_summaries.json` and `all_transcripts.json` and then proceeds to delete them, making sure each session starts clean.
    - Then validates the filenames passed in, and then proceeds to call `process_videos_with_error_handling()`
10. The parent function then defines the `def process_videos_with_error_handling(<args>):` function to handle the user input.
    - `def process_videos_with_error_handling(<args>):`:
      - The function first checks to see if there's any input in the URL field.
      - The function then checks/sets the batch size
      - Then separates URLs and local files
      - Validates existence of local files
      - Sets `all_inputs` variable as result of validation
      - Starts a `for` loop iterating through every item in the current batch
      - If the item is a URL, metadata extraction is attempted (using `yt-dlp`)
      - Chunking options are then set using the variables fed into the function
      - Line 765: `result = process_url_with_metadata<args>` - Stuff
        - Args: `input_item, 2, whisper_model, custom_prompt if custom_prompt_checkbox else None, start_seconds, api_name, api_key, False, False, False, False, 0.01, None, keywords, None, diarize, end_time=end_seconds, include_timestamps=(timestamp_option == "Include Timestamps"), metadata=video_metadata, use_chunking=chunking_options_checkbox, chunk_options=chunk_options, keep_original_video=keep_original_video`
      - If/else decision based on returned result, success or failure
        - If success, secondary sanity check, then: `transcription, summary = result`
            * (`batch_results` is an array)
            * `batch_results.append((input_item, transcription, "Success", result_metadata, json_file, summary_file))`
        - If failure, `batch_results.append((input_item, error_message, "Error", result_metadata, None, None))errors.append(error_message)`
      - Next, the length of the results is checked and printed to the screen
      - Then Lines 809-853 create the HTML to display the results nicely
      - Lines 857-858, 860-861 save the transcriptions + summaries to JSON
      - Finally Line 866 holds the following line which returns the final results to the user:
        - `return ( f"Processed {total_inputs} videos. {len(errors)} errors occurred.", error_summary, results_html, 'all_transcriptions.json', 'all_summaries.txt')`
- **Let's now dig a bit deeper into Line 765: `process_url_with_metadata<args>`**
  - `process_url_with_metadata<args>`:
    - Defined on line 934: `def process_url_with_metadata(input_item, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video_flag, download_audio, rolling_summarization, detail_level, question_box, keywords, local_file_path, diarize, end_time=None, include_timestamps=True, metadata=None, use_chunking=False, chunk_options=None, keep_original_video=False):```
    - First creates the folder `Video_Downloads` if it doesn't already exist in the current folder.
    - We then create the `infodict` dict
    - We then check if the input item is a URL or a local file
      - If it's a file we simply set the `video_metadata` to `None` with the `video_description` set to `Local file`
      - If it's a URL, we extract the metadata using `yt-dlp`
    - Next metadata is set,
    - Video is downloaded 1002: 
      - `Line video_file_path = download_video(input_item, download_path, full_info, download_video_flag)`
    - Video is transcribed Line 1011:
      - `audio_file_path, segments = perform_transcription(video_file_path, offset, whisper_model, vad_filter, diarize)`
    - Line 1027 We save the transcription to a JSON file
    - Lines 1032-1039 We delete the wav file 
    - Lines 1043-1054 we delete the video file if the user doesn't want to keep it
    - Line 1063 we create the `transcription_text` variable from compiling all the transcription segments
    - Lines 1076-1084 we summarize the transcription
      - `perform_summarization(api_name, full_text_with_metadata, custom_prompt, api_key)`
    - Lines 1086-1089 we save the summary to a text file
      - `save_transcription_and_summary(full_text_with_metadata, summary_text, download_path, info_dict)`
    - Lines 1095-1102 We prep the keywords
    - Lines 1104-1108 We save the media to the DB:
      - `add_media_to_database(info_dict['webpage_url'], info_dict, full_text_with_metadata, summary_text, keywords_list, custom_prompt, whisper_model)`
    - Finally, we return the transcription and summary on lines 1110-1111:
      - `return info_dict['webpage_url'], full_text_with_metadata, summary_text, json_file_path, summary_file_path, info_dict`
- **Let's now dig deeper into Line 1002: `download_video()`**
  - This function is imported from `Video_DL_Ingestion_Lib.py` and is defined on Line 97 of that file: 
    - `def download_video(video_url, download_path, info_dict, download_video_flag):` 
  - It first checks to see if the video has a valid name set in the info_dict{} dict
  - Does path handling for calling ffmpeg
  - Uses yt-dlp to download the audio of the video, or the full video if the user wants
- **Let's now dig deeper into Line 1011: `perform_transcription()`**
  - This function is imported from `Summarization_General.py` and is defined on Line 831 of that file:
    - `def perform_transcription(video_path, offset, whisper_model, vad_filter, diarize=False):`
  - It itself calls `audio_file_path = convert_to_wav(video_path, offset)`
  - Then checks for diarization and performs it if set
  - Then checks for a segments JSON file and loads it if it exists
  - Returns:
    - `return audio_file_path, diarized_segments`
- **Let's now dig deper into `convert_to_wav()` from the `Audio_Transcription_Lib.py` Library**
  - Sets the out-path for the file as the same path as the input file, but with a .wav extension
    - If the file already exists it will skip the conversion process
    - If the file doesn't exist, it will call ffmpeg to convert the file to a .wav file
  - Returns:
    - `return out_path`
- **Let's now dig deeper into Line 1076: `perform_summarization()`**
  - This function is defined in `Summarization_General.py` on Line 999:
    * `def perform_summarization(api_name, input_data, custom_prompt_input, api_key, recursive_summarization=False):`
  - First evaluates if a custom prompt was passed, if not it uses the default prompt
  - Then extracts the metadata + content from the input_data
    - `extract_metadata_and_content(input_data)`
  - Input is structured for summarization:
    - `structured_input = format_input_with_metadata(metadata, content)`
  - Then its checked if `recursive_summarization` was set (Line 1035), 
    - if so, it first performs chunking, calling `improved_chunking_process()`
      - `chunks = improved_chunking_process(structured_input, chunk_options)`
    - Finally, it performs recursive summarization using `recursive_summarize_chunks()`
      - `summary = recursive_summarize_chunks([chunk['text'] for chunk in chunks], lambda x: summarize_chunk(api_name, x, custom_prompt_input, api_key), custom_prompt_input)`      
  - If `recursive_summarization` was not set, then `summarize_chunk()` is called directly
    - `summary = summarize_chunk(api_name, structured_input, custom_prompt_input, api_key)`
  - Lines 1051-1081: It then evaluates if a summary was generated, and if so, saves it as a `.txt` file.
  - Once complete, it returns the summary
    - `return summary`
- **Let's now dig deeper into `improved_chunking_process()`:**
  - This function is defined in `Chunking_Lib.py` on Line 52:
    - `def improved_chunking_process(text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:`
    - First sets the various chunking options:
    - `chunk_method`, `max_chunk_size`, `overlap`, `language`, `adaptive`, `multi_level`
  - It then proceeds to perform chunking on the input text based on the selected chunking method: (All these functions are defined in the `Chunking_Lib.py` file)
    - `adaptive` - `adpativ_chunk_size()`
    - `multi_level` - `multi_level_chunking()`
    - `chunk_method == words` - `chunk_text_by_words()`
    - `chunk_method == sentences` - `chunk_text_by_sentences()`
    - `chunk_method == paragraps` - `chunk_text_by_paragraphs()`
    - `chunk_method == tokens` - `chunk_text_by_tokens()`
    - `chunks = [text]` - If no chunking method is selected / No chunking is performed.
  - Finally, it returns the chunks:
    - `return [{'text': chunk, 'metadata': get_chunk_metadata(chunk, text)} for chunk in chunks]`
- **Let's now dig deeper into `recursive_summarize_chunks()`:**
  - Defined on Line 215 of `Chunking_Lib.py`:
    - `def recursive_summarize_chunks(chunks, summarize_func, custom_prompt):`
  - First initializes the variables: `summarized_chunks = []`, `current_summary = ""`
  - Then iterates through each chunk in the chunks list
  - If it's the first chunk, it summarizes it using the `summarize_func()` function
  - If it's not the first chunk, it combines the current summary with the chunk and summarizes the combined text
  - Next it appends the current summary to the `summarized_chunks` list
  - Finally, it returns the summarized chunks list:
    - `return summarized_chunks`
- **Let's now dig deeper into `summarize_chunk()`:**
  - This function is defined on line 922 of `Summarization_General.py`:
    - `def summarize_chunk(api_name, text, custom_prompt_input, api_key):`
  - This function is simply one long if/else statement matching the `api_name` with the appropriate summarization function to call.
    - i.e. `if api_name == "openai":` -> `return summarize_with_openai(api_key, text, custom_prompt_input)`
    - Each of these functions are defined in the `Summarization_Lib.py` file.
    - The first one can be found at Line 66 of that file: `def summarize_with_openai(api_key, input_data, custom_prompt_arg):`
      - It continues downward from there.
  - If no api_name is matched, and none was passed, it simply returns `None`
  - If no api_name is matched and an argument for it was passed, it returns an error message.
- **Let's now dig deeper into Line 1086-1089 `save_transcription_and_summary()`**
  - The function is defined in `Summarization_General.py` on Line 900:
    - `def save_transcription_and_summary(transcription_text, summary_text, download_path, info_dict):`
  - The function first sanitizes the video title
    - `video_title = sanitize_filename(info_dict.get('title', 'Untitled'))`
    - `sanitize_filename()` is defined on Line 138 in `Utils.py`.
  - Then, Lines 904-907 it saves the transcription to a `.txt` file
    - `transcription_file_path = os.path.join(download_path, f"{video_title}_transcription.txt")`
  - Then, Lines 909-914 it saves the summary to a `.txt` file if it exists
    - `summary_file_path = os.path.join(download_path, f"{video_title}_summary.txt")`
  - Finally, it returns the paths to the transcription and summary files:
    - `return transcription_file_path, summary_file_path`
- **Let's now dig deeper into Line 1104-1108 `add_media_to_database()`**
  - Function is defined in `SQLite_DB.py` on Line 721:
    - `def add_media_to_database(url, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model, media_type='video'):`
  - The function first extracts the media contents from the `segments` list
  - It then sets the custom prompt if not set
    - ```
          result = add_media_with_keywords(
                    url=url,
                    title=info_dict.get('title', 'Untitled'),
                    media_type=media_type,
                    content=content,
                    keywords=','.join(keywords) if isinstance(keywords, list) else keywords,
                    prompt=custom_prompt_input or 'No prompt provided',
                    summary=summary or 'No summary provided',
                    transcription_model=whisper_model,
                    author=info_dict.get('uploader', 'Unknown'),
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
         )
      ```
  - Finally, it returns `result`
    - `return result`
- **Let's now dig deeper into `add_media_with_keywords()`**
  - It is defined on Lines 303-304 of `SQLite_DB.py`:
    - `def add_media_with_keywords(url, title, media_type, content, keywords, prompt, summary, transcription_model, author, ingestion_date):`
  - The function first sets default values for all the variables
  - Next it ensures the URL is valid and if it isn't, it sets it to `localhost`
  - It then performs a check of the media type, seeing if it matches one of the following, raising an error if it doesn't:
    - `if media_type not in ['article', 'audio', 'document', 'obsidian_note', 'podcast', 'text', 'video', 'unknown']:`
  - Next it checks to ensure the ingestion date/time is correct, raising an error if it isn't:
    - `if ingestion_date and not is_valid_date(ingestion_date):`
  - Next it ensure proper formatting of the keywords
  - Next, Lines 344-399 it attempts to connect to the DB and insert the generated items into the database.
  - Finally, it returns the following string on success (throwing appropriate errors if it fails at any part of the process):
    - `return f"Media '{title}' added/updated successfully with keywords: {', '.join(keyword_list)}"`

------------------------------------------------------------------------------------------------------------------

#### <a name="audio_file_transcription"></a>Audio File Transcription + Summarization
1. The tab for audio file transcription is the `create_audio_processing_tab()`
2. The function is defined in `Gradio_Related.py` on line 1157.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Row():` -> `gr.Column()` functions.
5. Then sets up inputs for the various variables the user can input
6. It then calls the `process_audio_files()` funcion when the `Process Audio Files` button is clicked.
7. `process_audio_files()` is defined on line 249 of `Audio_Files.py`
    - `def process_audio_files(audio_urls, audio_file, whisper_model, api_name, api_key, use_cookies, cookies, keep_original, custom_keywords, custom_prompt_input, chunk_method, max_chunk_size, chunk_overlap, use_adaptive_chunking, use_multi_level_chunking, chunk_language, diarize)`
    - The function first validates the filenames passed in, and then proceeds to call `process_audio_files_with_error_handling()`
    - Setups some arrays for usage
    - Then defines the `update_progress()` function to update the progress bar
    - Then defines the `cleanup_files()` function to cleanup the files after processing
    - Then defines the `reencode_mp3(mp3_file_path)` function to re-encode the mp3 files - ran into an issue where the mp3 files were not being processed correctly, this seemed to fix it...
    - Then defines the `convert_mp3_to_wav(mp3_file_path)` function, which does as it says.
    - Next it checks to see if ffmpeg is in the system path, if not it raises an error
    - Defines the chunking options
    - Handles multiple urls and splits them, proceeds to process one-by-one
    - URL Processing:
      - First the audio file is downloaded from the URL (if it's a URL) - `download_audio_file(url, use_cookies, cookies)`
      - UI is updated to show the progress
      - The audio file is then re-encoded as mp3
      - The audio file is then converted to wav
      - The audio file is then transcribed - diarization is handled here
      - Validation of segments
      - Chunking is then performed, `improved_chunking_process(transcription, chunk_options)` 
      - After chunking, the chunks are summarized: `perform_summarization(api_name, chunked_text, custom_prompt_input, api_key)`
      - Finally, the results are stored in the db with `add_media_with_keywords(url=url, title=os.path.basename(wav_file_path), media_type='audio', content=transcription, keywords=custom_keywords, prompt=custom_prompt_input, summary=summary, transcription_model=whisper_model, author="Unknown", ingestion_date=datetime.now().strftime('%Y-%m-%d'))`
    - Local File Processing:
      - Max file size is checked
      - Audio file is re-encoded as mp3
      - Audio file is converted to wav
      - The audio file is then transcribed - diarization is handled here
      - Validation of segments
      - Chunking is then performed, `improved_chunking_process(transcription, chunk_options)` 
      - After chunking, the chunks are summarized: `perform_summarization(api_name, chunked_text, custom_prompt_input, api_key)`
      - Finally, the results are stored in the db with `add_media_with_keywords(url=url, title=os.path.basename(wav_file_path), media_type='audio', content=transcription, keywords=custom_keywords, prompt=custom_prompt_input, summary=summary, transcription_model=whisper_model, author="Unknown", ingestion_date=datetime.now().strftime('%Y-%m-%d'))`
8. The results are then displayed to the user in the Gradio UI.
- **Let's now dig deeper into `download_audio_file(url, use_cookies, cookies)`**
    - Defined on line 48 of `Audio_Files.py`
      - `def download_audio_file(url, use_cookies=False, cookies=None):`
    - First sets up `header` dict
    - Then sets cookies if they were passed
    - Makes the request
    - Checks the file size
    - Generates a unique filename
    - Ensures the download path exists
    - Then downloads the file
    - Finally, returns the file path
      - `return save_path`

------------------------------------------------------------------------------------------------------------------

#### Podcast Transcription + Summarization
1. The tab for podcast transcription is the `create_podcast_tab()`
2. The function is defined in `Gradio_Related.py` on line 1254.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Row():` -> `gr.Column()` functions.
5. Then sets up inputs for the various variables the user can input
6. It then calls the `process_podcast()` function when the `Process Podcast` button is clicked.
7. `process_podcast()` is defined on lin 501 of `Audio_Files.py`
    - `def process_podcast(url, title, author, keywords, custom_prompt, api_name, api_key, whisper_model, keep_original=False, enable_diarization=False, use_cookies=False, cookies=None, chunk_method=None, max_chunk_size=300, chunk_overlap=0, use_adaptive_chunking=False, use_multi_level_chunking=False, chunk_language='english')`
    - Sets up some variables, and defines `update_progress()` and `cleanup_files()` functions (Same as above)
    - Then attempts to download the podcast file - `download_audio_file(url, use_cookies, cookies)`
    - Progress is updated, metadata is extracted and formatted
    - Keywords setup
    - Podcast is transcribed - `segments = speech_to_text(audio_file, whisper_model=whisper_model, diarize=True)`
    - Chunking is performed - `chunked_text = improved_chunking_process(transcription, chunk_options)`
    - Metadata and content is combined
    - The content is then summarized - `summary = perform_summarization(api_name, chunked_text, custom_prompt, api_key)`
    - Results are then stored in the db - `add_media_with_keywords( url=url, title=title, media_type='podcast', content=full_content, keywords=keywords, prompt=custom_prompt, summary=summary or "No summary available", transcription_model=whisper_model, author=author, ingestion_date=datetime.now().strftime('%Y-%m-%d'))`
8. Finally, the results are returned from `process_podcast()`
    - `return (update_progress("Processing complete."), full_content, summary or "No summary generated.", title, author, keywords, error_message)`

------------------------------------------------------------------------------------------------------------------

#### Website Scraping
1. The tab for podcast transcription is the `create_website_scraping_tab()`
2. The function is defined in `Gradio_Related.py` on line 1254.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Row():` -> `gr.Column()` functions.
5. Then sets up inputs for the various variables the user can input
6. It then calls the `scrape_and_summarize_multiple()` function when the `Scrape and Summarize` button is clicked.
7. `scrape_and_summarize_multiple()` is defined on line 110 of `Article_Summarization_Lib.py`
    - `def scrape_and_summarize_multiple(urls, custom_prompt_arg, api_name, api_key, keywords, custom_article_titles):`
    - First strips the URLs and sets the title for each
    - Sets up the `results` and `errors` arrays
    - Sets up a progress bar
    - Iterates through each URL, scraping the content and summarizing it
      - `scrape_and_summarize(url, custom_prompt_arg, api_name, api_key, keywords, custom_title)`
    - Updates the progress/GUI as it cycles through
    - Finally, returns the results:
      - `return combined_output`
8. Results are then displayed to the user in the Gradio UI.
- **Let's now dig deeper into `scrape_and_summarize()`**
    - Defined on line 141 of `Article_Summarization_Lib.py`
        - `def scrape_and_summarize(url, custom_prompt_arg, api_name, api_key, keywords, custom_article_title):`
    1. First scrapes the article content
        * `article_data = scrape_article(url)`
    2. Sets up the metadata
    3. Then sets up the custom prompt
    4. Then performs summarization of the content:
        * First sanitizes the filename - `sanitized_title = sanitize_filename(title)`
        * Runs down if/else statement to determine the API to use for summarization
    5. It then stores the results in the DB with the following statement:
        * `ingestion_result = ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date, article_custom_prompt)`
    6. Finally, it returns the results:
        * `return f"Title: {title}\nAuthor: {author}\nIngestion Result: {ingestion_result}\n\nSummary: {summary}\n\nArticle Contents: {content}"`
- **Let's now dig deeper into `scrape_article()`**
    - Defined on line 49 of `Article_Extractor_Lib.py`
    1. First defines the `fetch_html(url)` function
        * Function uses a headless chrome instance to fetch the HTML of the page
        * Returns the HTML - `return content`
    2. Then defines the `extract_article_data(html):` function
        * Function uses `trafilatura` to extract the article content from the HTML
        * returns metadata+article content
    3. Then defines the `convert_html_to_markdown()` function
        * `def convert_html_to_markdown(html):`
        * Uses beautifulsoup to convert the HTML to markdown
        * `return text`
    4. Then defines the `fetch_and_extract_article(url):` function
        * This function performs: `article_data = extract_article_data(html)`
        * Then `article_data['content'] = convert_html_to_markdown(article_data['content'])`
    5. Finally, it returns `article_data`
        * `return article_data`

------------------------------------------------------------------------------------------------------------------

#### Import .epub/ebook Files
1. The tab for epub/ebook ingestion is the `def create_import_book_tab():` functioned defined on line 3238 of the `Gradio_Related.py` file. 
2. The function is defined in `Gradio_Related.py` on line 1254.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Row():` -> `gr.Column()` functions.
5. Then sets up inputs for the various variables the user can input
6. It then calls the `import_epub()` function when the `Import eBook` button is clicked.
7. The function `import_epub()` is defined on line 3317 of `Gradio_Related.py`
    * `def import_epub(epub_file, title, author, keywords, custom_prompt, summary, auto_summarize, api_name, api_key):`
    * First creates a temp directory to store the epub file
    * Then uses pypandoc to convert the epub file to markdown
    * returns the following:
      * `return import_data(content, title, author, keywords, custom_prompt, summary, auto_summarize, api_name, api_key)`
8. The function `import_data()` is defined on line 2948 of `Gradio_Related.py`
    * First checks to verify a file was uploaded/passed
    * Performs various checks of the file
    * Creates the `info_dict` dict
    * Creates segments list
    * Sets up keywords list
    * Performs summarization: `perform_summarization(api_name, file_content, custom_prompt, api_key)`
    * Adds the media to the DB:
      * `add_media_to_database(url=file_name, info_dict=info_dict, segments=segments, summary=summary, keywords=keyword_list, custom_prompt_input=custom_prompt, whisper_model="Imported",  # Indicating this was an imported file, media_type = "document")`
    * Finally, returns the following:
      * `return f"File '{file_name}' successfully imported with title '{title}' and author '{author}'."`


------------------------------------------------------------------------------------------------------------------

#### PDF Ingestion
1. The tab for PDF ingestion is the `def create_pdf_ingestion_tab()` function.
2. The function is defined in `Gradio_Related.py` on line 1418.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Row():` -> `gr.Column()` functions.
5. Then sets up inputs for the various variables the user can input
6. It then calls the `process_and_cleanup_pdf()` function when the `Ingest PDF` button is clicked.
7. The `process_and_cleanup_pdf()` function is defined on line 136 in the `PDF_Ingestion_Lib.py` file
    * `def process_and_cleanup_pdf(file, title, author, keywords):`
    * First validates that a pdf has been uploaded.
    * Temp directory setup to store the pdf
    * PDF stored in the temp directory
    * PDF is then processed using the `ingest_pdf_file()` function.
8. The `ingest_pdf_file()` function is defined on line 94:
    * `def ingest_pdf_file(file_path, title=None, author=None, keywords=None):`
    * The function first attempts to convert the PDF to markdown - `markdown_content = convert_pdf_to_markdown(file_path)`
    * Filename and author are set
    * Keywords are set
    * Media is added to the DB:
      * `add_media_with_keywords(url=file_path, title=title, media_type='document', content=markdown_content, keywords=keywords, prompt='No prompt for PDF files', summary='No summary for PDF files', transcription_model='None', author=author, ingestion_date=datetime.now().strftime('%Y-%m-%d'))`
    * Finally, it returns the following string: 
      * `return f"PDF file '{title}' converted to Markdown and ingested successfully.", file_path`
10. The `convert_pdf_to_markdown(file_path)` function is defined on line 40 of `PDF_Ingestion_Lib.py`
    * `def convert_pdf_to_markdown(pdf_path)`
    * This function's whole purpose is to act as a shim callout to the Marker pipeline to convert the PDF to markdown. (`PDF_Converter.py` script)
    * Marker unfortunately has conflicting dependencies with the rest of the project, so it's been separated out into its own pipeline.
    * The function simply calls the Marker pipeline and returns the markdown content.
    * `return result.stdout`
11. The `PDF_Converter.py` script does the following:
    * Takes the passed in PDF file, and does the following: `markdown_content = marker_pdf.convert(pdf_file)`
    * It then returns the markdown content.
    * That's it.
12. This is then returned all the way back to step 8, where it's stored in the DB and the user is notified of the success.



------------------------------------------------------------------------------------------------------------------

#### Re-Summarize
1. The tab for Re-Summarization is the `def create_resummary_tab()` function.
2. The function is defined in `Gradio_Related.py` on line 1474.
3. The function is responsible for creating the tab and handling the user input.
4. The function first creates the tab using the `gr.Tabitem()` -> `gr.Row():` -> `gr.Column()` functions.
5. Then sets up inputs for the various variables the user can input
6. The function itself then calls `resummarize_content_wrapper` when the `Re-Summarize` button is clicked.
7. The `resummarize_content_wrapper()` function is defined on line 1573 in `Gradio_Related.py`
    * `def resummarize_content_wrapper(selected_item, item_mapping, api_name, api_key, chunking_options_checkbox, chunk_method, max_chunk_size, chunk_overlap, custom_prompt_checkbox, custom_prompt)`
    * The function first validates all necessary selections have been made
    * Then verifies the media selection is valid
    * Fetches the old content: `content, old_prompt, old_summary = fetch_item_details(media_id)`
    * Sets up the chunking options
    * Sets up the custom prompt: `summarization_prompt = custom_prompt if custom_prompt_checkbox and custom_prompt else None`
    * Then calls `resummarize_content()` with the appropriate arguments - `result = resummarize_content(media_id, content, api_name, api_key, chunk_options, summarization_prompt)`
    * Finally, it returns the results of the re-summarization.
      * `return result`
8. The `resummarize_content()` function is defined on line 1606 in `Summarization_General.py`
    * `def resummarize_content(selected_item, item_mapping, api_name, api_key, chunking_options_checkbox, chunk_method, max_chunk_size, chunk_overlap, custom_prompt_checkbox, custom_prompt):`
    * Verifies passed args are valid/media exists
    * Sets up chunking options -> Performs chunking using `summarize_chunk(api_name, chunk_text, summarization_prompt, api_key)`
    * Joins the resulting chunks (if chunking performed)
    * Then calls `update_media_content()`
      * `update_result = update_media_content(selected_item, item_mapping, content, summarization_prompt, new_summary)`
    * Finally, it returns the results of the re-summarization to the Gradio UI.
      * `return f"Re-summarization complete. New summary: {new_summary[:500]}..."`
- **Let's take a deeper look at `update_media_content()`**
    * Defined on line 852 of `SQLite_DB.py`
      * `def update_media_content(selected_item, item_mapping, content_input, prompt_input, summary_input):`
    * Performs various checks and updates the DB with the new records.