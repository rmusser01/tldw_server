# CLI Reference

Last updated ?

```
usage: summarize.py [-h] [-v] [-api API_NAME] [-key API_KEY] [-ns NUM_SPEAKERS] [-wm WHISPER_MODEL] [-off OFFSET] [-vad] [-log {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-gui] [-demo] [-prompt CUSTOM_PROMPT] [-overwrite] [-roll] [-detail DETAIL_LEVEL] [-model LLM_MODEL]
                    [-k KEYWORDS [KEYWORDS ...]] [--log_file LOG_FILE] [--local_llm] [--server_mode] [--share_public SHARE_PUBLIC] [--port PORT] [--ingest_text_file] [--text_title TEXT_TITLE] [--text_author TEXT_AUTHOR] [--diarize]
                    [input_path]

positional arguments:
  input_path            Path or URL of the video

options:
  -h, --help            show this help message and exit
  -v, --video           Download the video instead of just the audio
  -api API_NAME, --api_name API_NAME
                        API name for summarization (optional)
  -key API_KEY, --api_key API_KEY
                        API key for summarization (optional)
  -ns NUM_SPEAKERS, --num_speakers NUM_SPEAKERS
                        Number of speakers (default: 2)
  -wm WHISPER_MODEL, --whisper_model WHISPER_MODEL
                        Whisper model (default: small)| Options: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en
  -off OFFSET, --offset OFFSET
                        Offset in seconds (default: 0)
  -vad, --vad_filter    Enable VAD filter
  -log {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Log level (default: INFO)
  -gui, --user_interface
                        Launch the Gradio user interface
  -demo, --demo_mode    Enable demo mode
  -prompt CUSTOM_PROMPT, --custom_prompt CUSTOM_PROMPT
                        Pass in a custom prompt to be used in place of the existing one.
                         (Probably should just modify the script itself...)
  -overwrite, --overwrite
                        Overwrite existing files
  -roll, --rolling_summarization
                        Enable rolling summarization
  -detail DETAIL_LEVEL, --detail_level DETAIL_LEVEL
                        Mandatory if rolling summarization is enabled, defines the chunk  size.
                         Default is 0.01(lots of chunks) -> 1.00 (few chunks)
                         Currently only OpenAI works.
  -k KEYWORDS [KEYWORDS ...], --keywords KEYWORDS [KEYWORDS ...]
                        Keywords for tagging the media, can use multiple separated by spaces (default: cli_ingest_no_tag)
  --log_file LOG_FILE   Where to save logfile (non-default)
  --local_llm           Use a local LLM from the script(Downloads llamafile from github and 'mistral-7b-instruct-v0.2.Q8' - 8GB model from Huggingface)
  --server_mode         Run in server mode (This exposes the GUI/Server to the network)
  --share_public SHARE_PUBLIC
                        This will use Gradio's built-in ngrok tunneling to share the server publicly on the internet. Specify the port to use (default: 7860)
  --port PORT           Port to run the server on
  --ingest_text_file    Ingest .txt files as content instead of treating them as URL lists
  --text_title TEXT_TITLE
                        Title for the text file being ingested
  --text_author TEXT_AUTHOR
                        Author of the text file being ingested
  --diarize             Enable speaker diarization


Sample commands:
    1. Simple Sample command structure:
        summarize.py <path_to_video> -api openai -k tag_one tag_two tag_three

    2. Rolling Summary Sample command structure:
        summarize.py <path_to_video> -api openai -prompt "custom_prompt_goes_here-is-appended-after-transcription" -roll -detail 0.01 -k tag_one tag_two tag_three

    3. FULL Sample command structure:
        summarize.py <path_to_video> -api openai -ns 2 -wm small.en -off 0 -vad -log INFO -prompt "custom_prompt" -overwrite -roll -detail 0.01 -k tag_one tag_two tag_three

    4. Sample command structure for UI debug logging printed to console:
        summarize.py -gui -log DEBUG
```