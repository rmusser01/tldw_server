#!/usr/bin/env python3
# Std Lib Imports
import argparse
import atexit
import json
from logging.handlers import RotatingFileHandler
import os
import signal
import sys
import threading
import time
#
# 3rd-Party Imports
import nltk
from loguru import logger

from App_Function_Libraries.Metrics.logger_config import setup_logger

#
# Local Library Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'App_Function_Libraries')))
from App_Function_Libraries.Books.Book_Ingestion_Lib import ingest_folder, ingest_text_file
from App_Function_Libraries.Chunk_Lib import  semantic_chunk_long_file#, rolling_summarize_function,
from App_Function_Libraries.Gradio_Related import launch_ui
from App_Function_Libraries.Local_LLM.Local_LLM_Inference_Engine_Lib import cleanup_process
from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_local_llm
from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_openai, summarize_with_anthropic, \
    summarize_with_cohere, summarize_with_groq, perform_transcription, perform_summarization
from App_Function_Libraries.Audio.Audio_Transcription_Lib import speech_to_text
from App_Function_Libraries.Local_File_Processing_Lib import read_paths_from_file, process_local_file
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
from App_Function_Libraries.Utils.System_Checks_Lib import cuda_check, platform_check, check_ffmpeg
from App_Function_Libraries.Utils.Utils import load_and_log_configs, create_download_directory, \
    extract_text_from_segments, cleanup_downloads, logging
from App_Function_Libraries.Video_DL_Ingestion_Lib import download_video, extract_video_info
#
# Code responsible for launching GUI and leading to most functionality on line 838-862: launch UI launches the Gradio UI, which starts in the `Gradio_Related.py` file, where every tab it loads proceeds to load that page in a chain,
# this means that the `Gradio_Related.py` file is the main file for the UI, and then calls out to all the other pieces, through the individual tabs.
# So if you're trying to understand the codebase, start with `Gradio_Related.py` and then follow the chain of calls to understand how the UI is built/works on the backend as I've isolated/grouped most things together.
#######################
# Stop Gradio Analytics
#
#log_level = "DEBUG"
#logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
#
#############
#
# Global variables setup
whisper_models = ["small", "medium", "small.en", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3",
                  "distil-large-v2", "distil-medium.en", "distil-small.en", ]
server_mode = False
share_public = False
running_in_debug_mode = False

# FIXME - add to config.txt
log_file_path = os.getenv("tldw_LOG_FILE_PATH", "./Logs/tldw_app_logs.json")
max_bytes = int(os.getenv("tldw_LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
backup_count = int(os.getenv("tldw_LOG_BACKUP_COUNT", 5))

# logger.verbose("This is a VERBOSE message.")

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=max_bytes, backupCount=backup_count
)

logging.debug("Checking for nltk install...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

#
#######################

#######################
# Function Sections
#
abc_xyz = """
    Database Setup
    Config Loading
    System Checks
    DataBase Functions
    Processing Paths and local file handling
    Video Download/Handling
    Audio Transcription
    Diarization
    Chunking-related Techniques & Functions
    Tokenization-related Techniques & Functions
    Summarizers
    Gradio UI
    Main
"""
#
#
#######################
#######################
#
#       TL/DW: Too Long Didn't Watch
#
#  Project originally created by https://github.com/the-crypt-keeper
#  Modifications made by https://github.com/rmusser01
#  All credit to the original authors, I've just glued shit together.
#
#
# Usage:
#
#   Download Audio only from URL -> Transcribe audio:
#       python summarize.py https://www.youtube.com/watch?v=4nd1CDZP21s`
#
#   Download Audio+Video from URL -> Transcribe audio from Video:**
#       python summarize.py -v https://www.youtube.com/watch?v=4nd1CDZP21s`
#
#   Download Audio only from URL -> Transcribe audio -> Summarize using (`anthropic`/`cohere`/`openai`/`llama` (llama.cpp)/`ooba` (oobabooga/text-gen-webui)/`kobold` (kobold.cpp)/`tabby` (Tabbyapi)) API:**
#       python summarize.py -v https://www.youtube.com/watch?v=4nd1CDZP21s -api <your choice of API>` - Make sure to put your API key into `config.txt` under the appropriate API variable
#
#   Download Audio+Video from a list of videos in a text file (can be file paths or URLs) and have them all summarized:**
#       python summarize.py ./local/file_on_your/system --api_name <API_name>`
#
#   Run it as a WebApp**
#       python summarize.py -gui` - This requires you to either stuff your API keys into the `config.txt` file, or pass them into the app every time you want to use it.
#           Can be helpful for setting up a shared instance, but not wanting people to perform inference on your server.
#
#######################


#######################
# Random issues I've encountered and how I solved them:
#   1. Something about cuda nn library missing, even though cuda is installed...
#       https://github.com/tensorflow/tensorflow/issues/54784 - Basically, installing zlib made it go away. idk.
#       Or https://github.com/SYSTRAN/faster-whisper/issues/85
#
#   2. ERROR: Could not install packages due to an OSError: [WinError 2] The system cannot find the file specified: 'C:\\Python312\\Scripts\\dateparser-download.exe' -> 'C:\\Python312\\Scripts\\dateparser-download.exe.deleteme'
#       Resolved through adding --user to the pip install command
#
#   3. Windows: Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!
#
#   4.
#
#   5.
#
#
#
#######################


#######################
# DB Setup

# Handled by SQLite_DB.py

#######################


#######################
# System Startup Notice
#

# Dirty hack - sue me. - FIXME - fix this...
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
}
source_language_list = [key[0] for key in source_languages.items()]


def print_hello():
    print(r"""_____  _          ________  _    _                                 
|_   _|| |        / /|  _  \| |  | | _                              
  | |  | |       / / | | | || |  | |(_)                             
  | |  | |      / /  | | | || |/\| |                                
  | |  | |____ / /   | |/ / \  /\  / _                              
  \_/  \_____//_/    |___/   \/  \/ (_)                             


 _                   _                                              
| |                 | |                                             
| |_   ___    ___   | |  ___   _ __    __ _                         
| __| / _ \  / _ \  | | / _ \ | '_ \  / _` |                        
| |_ | (_) || (_) | | || (_) || | | || (_| | _                      
 \__| \___/  \___/  |_| \___/ |_| |_| \__, |( )                     
                                       __/ ||/                      
                                      |___/                         
     _  _      _         _  _                      _          _     
    | |(_)    | |       ( )| |                    | |        | |    
  __| | _   __| | _ __  |/ | |_  __      __  __ _ | |_   ___ | |__  
 / _` || | / _` || '_ \    | __| \ \ /\ / / / _` || __| / __|| '_ \ 
| (_| || || (_| || | | |   | |_   \ V  V / | (_| || |_ | (__ | | | |
 \__,_||_| \__,_||_| |_|    \__|   \_/\_/   \__,_| \__| \___||_| |_|
""")
    time.sleep(1)
    return


#
#
#######################


#######################
# System Check Functions
#
# 1. platform_check()
# 2. cuda_check()
# 3. decide_cpugpu()
# 4. check_ffmpeg()
# 5. download_ffmpeg()
#
#######################


#######################
# DB Functions
#
#     create_tables()
#     add_keyword()
#     delete_keyword()
#     add_keyword()
#     add_media_with_keywords()
#     search_db()
#     format_results()
#     search_and_display()
#     export_to_csv()
#     is_valid_url()
#     is_valid_date()
#
########################################################################################################################


########################################################################################################################
# Processing Paths and local file handling
#
# Function List
# 1. read_paths_from_file(file_path)
# 2. process_path(path)
# 3. process_local_file(file_path)
# 4. read_paths_from_file(file_path: str) -> List[str]
#
#
########################################################################################################################


#######################################################################################################################
# Online Article Extraction / Handling
#
# Function List
# 1. get_page_title(url)
# 2. get_article_text(url)
# 3. get_article_title(article_url_arg)
#
#
#######################################################################################################################


#######################################################################################################################
# Video Download/Handling
# Video-DL-Ingestion-Lib
#
# Function List
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, ) # FIXME - UPDATE
#
#
#######################################################################################################################


#######################################################################################################################
# Audio Transcription
#
# Function List
# 1. convert_to_wav(video_file_path, offset=0, overwrite=False)
# 2. speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False)
#
#
#######################################################################################################################


#######################################################################################################################
# Diarization
#
# Function List 1. speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding",
#                                   embedding_size=512, num_speakers=0)
#
#
#######################################################################################################################


#######################################################################################################################
# Chunking-related Techniques & Functions
#
#
# FIXME
#
#
#######################################################################################################################


#######################################################################################################################
# Tokenization-related Functions
#
#

# FIXME

#
#
#######################################################################################################################


#######################################################################################################################
# Website-related Techniques & Functions
#
#

#
#
#######################################################################################################################


#######################################################################################################################
# Summarizers
#
# Function List
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#################################
# Local Summarization
#
# Function List
#
# 1. summarize_with_local_llm(file_path, custom_prompt_arg)
# 2. summarize_with_llama(api_url, file_path, token, custom_prompt)
# 3. summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#######################################################################################################################


#######################################################################################################################
# Summarization with Detail
#


#
#
#######################################################################################################################


#######################################################################################################################
# Gradio UI
#
#
#
#
#
#################################################################################################################
#
#######################################################################################################################
# Local LLM Setup / Running
#
# Function List
# 1. download_latest_llamafile(repo, asset_name_prefix, output_filename)
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. cleanup_process()
# 5. signal_handler(sig, frame)
# 6. local_llm_function()
# 7. launch_in_new_terminal_windows(executable, args)
# 8. launch_in_new_terminal_linux(executable, args)
# 9. launch_in_new_terminal_mac(executable, args)
#
#
#######################################################################################################################


#######################################################################################################################
# Helper Functions for Main() & process_url()
#
#
#
#######################################################################################################################


######################################################################################################################
# Main()
#

def main(input_path, api_name=None, api_key=None,
         num_speakers=2,
         whisper_model="small.en",
         offset=0,
         vad_filter=False,
         download_video_flag=False,
         custom_prompt=None,
         overwrite=False,
         rolling_summarization=False,
         detail=0.01,
         keywords=None,
         llm_model=None,
         time_based=False,
         set_chunk_txt_by_words=False,
         set_max_txt_chunk_words=0,
         set_chunk_txt_by_sentences=False,
         set_max_txt_chunk_sentences=0,
         set_chunk_txt_by_paragraphs=False,
         set_max_txt_chunk_paragraphs=0,
         set_chunk_txt_by_tokens=False,
         set_max_txt_chunk_tokens=0,
         ingest_text_file=False,
         chunk=False,
         max_chunk_size=2000,
         chunk_overlap=100,
         chunk_unit='tokens',
         summarize_chunks=None,
         diarize=False,
         system_message=None):
    global detail_level_number, summary, audio_file, transcription_text, info_dict

    detail_level = detail

    print(f"Keywords: {keywords}")

    if not input_path:
        return []

    start_time = time.monotonic()
    paths = [input_path] if not os.path.isfile(input_path) else read_paths_from_file(input_path)
    results = []

    for path in paths:
        try:
            if path.startswith('http'):
                info_dict, title = extract_video_info(path)
                download_path = create_download_directory(title)
                video_path = download_video(path, download_path, info_dict, download_video_flag)

                if video_path:
                    if diarize:
                        audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter, diarize=True)
                        transcription_text = {'audio_file': audio_file, 'transcription': segments}
                    else:
                        audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)
                        transcription_text = {'audio_file': audio_file, 'transcription': segments}

                    # FIXME rolling summarization
                    if rolling_summarization:
                        pass
                    #     text = extract_text_from_segments(segments)
                    #     detail = detail_level
                    #     additional_instructions = custom_prompt_input
                    #     chunk_text_by_words = set_chunk_txt_by_words
                    #     max_words = set_max_txt_chunk_words
                    #     chunk_text_by_sentences = set_chunk_txt_by_sentences
                    #     max_sentences = set_max_txt_chunk_sentences
                    #     chunk_text_by_paragraphs = set_chunk_txt_by_paragraphs
                    #     max_paragraphs = set_max_txt_chunk_paragraphs
                    #     chunk_text_by_tokens = set_chunk_txt_by_tokens
                    #     max_tokens = set_max_txt_chunk_tokens
                    #     # FIXME
                    #     summarize_recursively = rolling_summarization
                    #     verbose = False
                    #     model = None
                    #     summary = rolling_summarize_function(text, detail, api_name, api_key, model, custom_prompt_input,
                    #                                          chunk_text_by_words,
                    #                                          max_words, chunk_text_by_sentences,
                    #                                          max_sentences, chunk_text_by_paragraphs,
                    #                                          max_paragraphs, chunk_text_by_tokens,
                    #                                          max_tokens, summarize_recursively, verbose
                    #                                          )

                    elif api_name:
                        summary = perform_summarization(api_name, transcription_text, custom_prompt_input, api_key)
                    else:
                        summary = None

                    if summary:
                        # Save the summary file in the download_path directory
                        summary_file_path = os.path.join(download_path, f"{transcription_text}_summary.txt")
                        with open(summary_file_path, 'w') as file:
                            file.write(summary)

                    add_media_to_database(path, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)
                else:
                    logging.error(f"Failed to download video: {path}")

            # FIXME - make sure this doesn't break ingesting multiple videos vs multiple text files
            # FIXME - Need to update so that chunking is fully handled.
            elif chunk and path.lower().endswith('.txt'):
                chunks = semantic_chunk_long_file(path, max_chunk_size, chunk_overlap)
                if chunks:
                    chunks_data = {
                        "file_path": path,
                        "chunk_unit": chunk_unit,
                        "max_chunk_size": max_chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunks": []
                    }
                    summaries_data = {
                        "file_path": path,
                        "summarization_method": summarize_chunks,
                        "summaries": []
                    }

                    for i, chunk_text in enumerate(chunks):
                        chunk_info = {
                            "chunk_id": i + 1,
                            "text": chunk_text
                        }
                        chunks_data["chunks"].append(chunk_info)

                        if summarize_chunks:
                            summary = None
                            if summarize_chunks == 'openai':
                                summary = summarize_with_openai(api_key, chunk_text, custom_prompt, system_message)
                            elif summarize_chunks == 'anthropic':
                                summary = summarize_with_anthropic(api_key, chunk_text, custom_prompt, system_message)
                            elif summarize_chunks == 'cohere':
                                summary = summarize_with_cohere(api_key, chunk_text, custom_prompt, system_message)
                            elif summarize_chunks == 'groq':
                                summary = summarize_with_groq(api_key, chunk_text, custom_prompt, system_message)
                            elif summarize_chunks == 'local-llm':
                                summary = summarize_with_local_llm(chunk_text, custom_prompt, system_message)
                            # FIXME - Add more summarization methods as needed

                            if summary:
                                summary_info = {
                                    "chunk_id": i + 1,
                                    "summary": summary
                                }
                                summaries_data["summaries"].append(summary_info)
                            else:
                                logging.warning(f"Failed to generate summary for chunk {i + 1}")

                    # Save chunks to a single JSON file
                    chunks_file_path = f"{path}_chunks.json"
                    with open(chunks_file_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                    logging.info(f"All chunks saved to {chunks_file_path}")

                    # Save summaries to a single JSON file (if summarization was performed)
                    if summarize_chunks:
                        summaries_file_path = f"{path}_summaries.json"
                        with open(summaries_file_path, 'w', encoding='utf-8') as f:
                            json.dump(summaries_data, f, ensure_ascii=False, indent=2)
                        logging.info(f"All summaries saved to {summaries_file_path}")

                    logging.info(f"File {path} chunked into {len(chunks)} parts using {chunk_unit} as the unit.")
                else:
                    logging.error(f"Failed to chunk file {path}")

            # Handle downloading of URLs from a text file or processing local video/audio files
            else:
                download_path, info_dict, urls_or_media_file = process_local_file(path)
                if isinstance(urls_or_media_file, list):
                    # Text file containing URLs
                    for url in urls_or_media_file:
                        for item in urls_or_media_file:
                            if item.startswith(('http://', 'https://')):
                                info_dict, title = extract_video_info(url)
                                download_path = create_download_directory(title)
                                video_path = download_video(url, download_path, info_dict, download_video_flag)

                                if video_path:
                                    if diarize:
                                        audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter, diarize=True)
                                    else:
                                        audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)

                                    transcription_text = {'audio_file': audio_file, 'transcription': segments}
                                    if rolling_summarization:
                                        text = extract_text_from_segments(segments)
                                        # FIXME
                                        #summary = summarize_with_detail_openai(text, detail=detail)
                                    elif api_name:
                                        summary = perform_summarization(api_name, transcription_text, custom_prompt_input, api_key)
                                    else:
                                        summary = None

                                    if summary:
                                        # Save the summary file in the download_path directory
                                        summary_file_path = os.path.join(download_path, f"{transcription_text}_summary.txt")
                                        with open(summary_file_path, 'w') as file:
                                            file.write(summary)

                                    add_media_to_database(url, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)
                                else:
                                    logging.error(f"Failed to download video: {url}")

                else:
                    # Video or audio or txt file
                    media_path = urls_or_media_file

                    if media_path.lower().endswith(('.txt', '.md')):
                        if media_path.lower().endswith('.txt'):
                            # Handle text file ingestion
                            result = ingest_text_file
                            logging.info(result)
                    elif media_path.lower().endswith(('.mp4', '.avi', '.mov')):
                        if diarize:
                            audio_file, segments = perform_transcription(media_path, offset, whisper_model, vad_filter, diarize=True)
                        else:
                            audio_file, segments = perform_transcription(media_path, offset, whisper_model, vad_filter)
                    elif media_path.lower().endswith(('.wav', '.mp3', '.m4a')):
                        if diarize:
                            segments = speech_to_text(media_path, whisper_model=whisper_model, vad_filter=vad_filter, diarize=True)
                        else:
                            segments = speech_to_text(media_path, whisper_model=whisper_model, vad_filter=vad_filter)
                    else:
                        logging.error(f"Unsupported media file format: {media_path}")
                        continue

                    transcription_text = {'media_path': path, 'audio_file': media_path, 'transcription': segments}

                    # FIXME
                    if rolling_summarization:
                        #text = extract_text_from_segments(segments)
                        #summary = summarize_with_detail_openai(text, detail=detail)
                        pass
                    elif api_name:
                        summary = perform_summarization(api_name, transcription_text, custom_prompt_input, api_key)
                    else:
                        summary = None

                    if summary:
                        # Save the summary file in the download_path directory
                        summary_file_path = os.path.join(download_path, f"{transcription_text}_summary.txt")
                        with open(summary_file_path, 'w') as file:
                            file.write(summary)

                    add_media_to_database(path, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)

        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            continue

    logging.debug(f"Total time taken: {time.monotonic() - start_time} seconds")
    logging.info("MAIN: returing transcription_text.")
    return transcription_text


def signal_handler(sig, frame):
    logging.info("Ctrl-C pressed, shutting down...")
    # Check for active threads before shutdown
    logging.debug(f"Active threads before shutdown: {threading.enumerate()}")
    # Check for active threads after shutdown
    logging.debug(f"Active threads after shutdown: {threading.enumerate()}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


####################################################################################################################
#
# MAIN

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load Config
    loaded_config_data = load_and_log_configs()

    if loaded_config_data:
        logging.info("Main: Configuration loaded successfully")
        # You can access the configuration data like this:
        # print(f"OpenAI API Key: {config_data['api_keys']['openai']}")
        # print(f"Anthropic Model: {config_data['models']['anthropic']}")
        # print(f"Kobold API IP: {config_data['local_apis']['kobold']['ip']}")
        # print(f"Output Path: {config_data['output_path']}")
        # print(f"Processing Choice: {config_data['processing_choice']}")
    else:
        print("Failed to load configuration")

    # Print ascii_art
    print_hello()

    transcription_text = None

    parser = argparse.ArgumentParser(
        description='Transcribe and summarize videos.',
        epilog='''
Sample commands:
    1. Simple Sample command structure:
        summarize.py <path_to_video> -api openai -k tag_one tag_two tag_three

    2. Rolling Summary Sample command structure:
        summarize.py <path_to_video> -api openai -prompt "custom_prompt_goes_here-is-appended-after-transcription" -roll -detail 0.01 -k tag_one tag_two tag_three

    3. FULL Sample command structure:
        summarize.py <path_to_video> -api openai -ns 2 -wm small.en -off 0 -vad -log INFO -prompt "custom_prompt" -overwrite -roll -detail 0.01 -k tag_one tag_two tag_three

    4. Sample command structure for UI:
        summarize.py -gui -log DEBUG
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', type=str, help='Path or URL of the video', nargs='?')
    parser.add_argument('-v', '--video', action='store_true', help='Download the video instead of just the audio')
    parser.add_argument('-api', '--api_name', type=str, help='API name for summarization (optional)')
    parser.add_argument('-key', '--api_key', type=str, help='API key for summarization (optional)')
    parser.add_argument('-ns', '--num_speakers', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('-wm', '--whisper_model', type=str, default='small',
                        help='Whisper model (default: small)| Options: tiny.en, tiny, base.en, base, small.en, small, medium.en, '
                             'medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, '
                             'distil-small.en')
    parser.add_argument('-off', '--offset', type=int, default=0, help='Offset in seconds (default: 0)')
    parser.add_argument('-vad', '--vad_filter', action='store_true', help='Enable VAD filter')
    parser.add_argument('-log', '--log_level', type=str, default='INFO',
                        choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level (default: INFO)')
    parser.add_argument('-gui', '--user_interface', action='store_true', help="Launch the Gradio user interface")
    parser.add_argument('-demo', '--demo_mode', action='store_true', help='Enable demo mode')
    parser.add_argument('-prompt', '--custom_prompt', type=str,
                        help='Pass in a custom prompt to be used in place of the existing one.\n (Probably should just '
                             'modify the script itself...)')
    parser.add_argument('-overwrite', '--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('-roll', '--rolling_summarization', action='store_true', help='Enable rolling summarization')
    parser.add_argument('-detail', '--detail_level', type=float, help='Mandatory if rolling summarization is enabled, '
                                                                      'defines the chunk  size.\n Default is 0.01(lots '
                                                                      'of chunks) -> 1.00 (few chunks)\n Currently '
                                                                      'only OpenAI works. ',
                        default=0.01, )
    parser.add_argument('-model', '--llm_model', type=str, default='',
                        help='Model to use for LLM summarization (only used for vLLM/TabbyAPI)')
    parser.add_argument('-k', '--keywords', nargs='+', default=['cli_ingest_no_tag'],
                        help='Keywords for tagging the media, can use multiple separated by spaces (default: cli_ingest_no_tag)')
    parser.add_argument('--log_file', type=str, help='Where to save logfile (non-default)')
    parser.add_argument('--local_llm', action='store_true',
                        help="Use a local LLM from the script(Downloads llamafile from github and 'mistral-7b-instruct-v0.2.Q8' - 8GB model from Huggingface)")
    parser.add_argument('--server_mode', action='store_true',
                        help='Run in server mode (This exposes the GUI/Server to the network)')
    parser.add_argument('-share', '--share_public', action='store_true', help="This will use Gradio's built-in ngrok tunneling to share the server publicly on the internet."),
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--ingest_text_file', action='store_true',
                        help='Ingest .txt files as content instead of treating them as URL lists')
    parser.add_argument('--text_title', type=str, help='Title for the text file being ingested')
    parser.add_argument('--text_author', type=str, help='Author of the text file being ingested')
    parser.add_argument('--diarize', action='store_true', help='Enable speaker diarization')
    # parser.add_argument('--offload', type=int, default=20, help='Numbers of layers to offload to GPU for Llamafile usage')
    # parser.add_argument('-o', '--output_path', type=str, help='Path to save the output file')

    args = parser.parse_args()

    # Set Chunking values/variables
    set_chunk_txt_by_words = False
    set_max_txt_chunk_words = 0
    set_chunk_txt_by_sentences = False
    set_max_txt_chunk_sentences = 0
    set_chunk_txt_by_paragraphs = False
    set_max_txt_chunk_paragraphs = 0
    set_chunk_txt_by_tokens = False
    set_max_txt_chunk_tokens = 0
    custom_prompt_input = args.custom_prompt

    if args.server_mode:
        server_mode = args.server_mode
    else:
        server_mode = None
    if args.server_mode is True:
        server_mode = True
    if args.port:
        server_port = args.port
    else:
        server_port = None

    ########## Logging setup
    # Convert the provided log level to uppercase
    log_level = args.log_level.upper()

    # Remove the default Loguru handler so we can add our own sinks
    logger.remove()
    # '%(asctime)s - %(levelname)s - %(message)s' is mapped to Loguru’s {time} - {level} - {message}.

    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    )
    if args.log_file:
        # Add a file sink with the same level and format.
        logger.add(
            args.log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
        )
        logger.info(f"Log file created at: {args.log_file}")
    else:
        logger.info(f"No Logfile declared. Using Standard logfile")
        logger = setup_logger(args)

    # Check if the user wants to use the local LLM from the script
    local_llm = args.local_llm
    logging.info(f'Local LLM flag: {local_llm}')

    # Check if the user wants to ingest a text file (singular or multiple from a folder)
    if args.input_path is not None:
        if os.path.isdir(args.input_path) and args.ingest_text_file:
            results = ingest_folder(args.input_path, keywords=args.keywords)
            for result in results:
                print(result)
        elif args.input_path.lower().endswith('.txt') and args.ingest_text_file:
            result = ingest_text_file(args.input_path, title=args.text_title, author=args.text_author,
                                      keywords=args.keywords)
            print(result)
        sys.exit(0)
########################################################################################################################
#
#   Launch the UI
    # Launch the GUI
    if args.user_interface:
        if args.share_public:
            if args.local_llm:
                time.sleep(2)
                launch_ui(share_public=True)
            else:
                launch_ui(share_public=True)
        else:
            launch_ui(share_public=False)
    elif not args.input_path:
        parser.print_help()
        sys.exit(1)
########################################################################################################################
    else:
        logging.info('Starting the transcription and summarization process.')
        logging.info(f'Input path: {args.input_path}')
        logging.info(f'API Name: {args.api_name}')
        logging.info(f'Number of speakers: {args.num_speakers}')
        logging.info(f'Whisper model: {args.whisper_model}')
        logging.info(f'Offset: {args.offset}')
        logging.info(f'VAD filter: {args.vad_filter}')
        logging.info(f'Log Level: {args.log_level}')
        logging.info(f'Demo Mode: {args.demo_mode}')
        logging.info(f'Custom Prompt: {args.custom_prompt}')
        logging.info(f'Overwrite: {args.overwrite}')
        logging.info(f'Rolling Summarization: {args.rolling_summarization}')
        logging.info(f'User Interface: {args.user_interface}')
        logging.info(f'Video Download: {args.video}')
        # logging.info(f'Save File location: {args.output_path}')
        # logging.info(f'Log File location: {args.log_file}')

        global api_name
        api_name = args.api_name

        summary = None  # Initialize to ensure it's always defined
        if args.detail_level is None:
            args.detail_level = 0.01

        # FIXME
        # if args.api_name and args.rolling_summarization and any(
        #         key.startswith(args.api_name) and value is not None for key, value in api_keys.items()):
        #     logging.info(f'MAIN: API used: {args.api_name}')
        #     logging.info('MAIN: Rolling Summarization will be performed.')

        elif args.api_name:
            logging.info(f'MAIN: API used: {args.api_name}')
            logging.info('MAIN: Summarization (not rolling) will be performed.')

        else:
            logging.info('No API specified. Summarization will not be performed.')

        logging.debug("Platform check being performed...")
        platform_check()
        logging.debug("CUDA check being performed...")
        cuda_check()
        processing_choice = "cpu"
        logging.debug("ffmpeg check being performed...")
        check_ffmpeg()
        # download_ffmpeg()

        llm_model = args.llm_model or None
        # FIXME - dirty hack
        args.time_based = False

        ########## Custom Prompt setup
        custom_prompt_input = args.custom_prompt

        if not args.custom_prompt:
            logging.debug("No custom prompt defined, will use default")
            args.custom_prompt_input = (
                "\n\nThis is the transcript of a video. "
                "Please read through the transcript carefully. Identify the main topics that are "
                "discussed over the course of the transcript. Then, summarize the key points about each "
                "main topic in a concise bullet point. The bullet points should cover the key "
                "information conveyed about each topic in the video, but should be much shorter than "
                "the full transcript. Please output your bullet point summary inside <bulletpoints> "
                "tags."
            )
            print("No custom prompt defined, will use default")

            custom_prompt_input = args.custom_prompt

        try:
            results = main(args.input_path, api_name=args.api_name, api_key=args.api_key,
                           num_speakers=args.num_speakers, whisper_model=args.whisper_model, offset=args.offset,
                           vad_filter=args.vad_filter, download_video_flag=args.video, custom_prompt=args.custom_prompt_input,
                           overwrite=args.overwrite, rolling_summarization=args.rolling_summarization,
                           detail=args.detail_level, keywords=args.keywords, llm_model=args.llm_model,
                           time_based=args.time_based, set_chunk_txt_by_words=set_chunk_txt_by_words,
                           set_max_txt_chunk_words=set_max_txt_chunk_words,
                           set_chunk_txt_by_sentences=set_chunk_txt_by_sentences,
                           set_max_txt_chunk_sentences=set_max_txt_chunk_sentences,
                           set_chunk_txt_by_paragraphs=set_chunk_txt_by_paragraphs,
                           set_max_txt_chunk_paragraphs=set_max_txt_chunk_paragraphs,
                           set_chunk_txt_by_tokens=set_chunk_txt_by_tokens,
                           set_max_txt_chunk_tokens=set_max_txt_chunk_tokens)

            logging.info('Transcription process completed.')
            atexit.register(cleanup_process)

        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)

    def cleanup():
        logging.info("Cleanup function called. Script is exiting.")

    atexit.register(cleanup)
    # Register the cleanup function to run on exit
    atexit.register(cleanup_downloads)

#
# End of summarize.py
#######################################################################################################################
