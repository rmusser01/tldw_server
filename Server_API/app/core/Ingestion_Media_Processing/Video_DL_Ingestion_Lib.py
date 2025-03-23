# Video_DL_Ingestion_Lib.py
#########################################
# Video Downloader and Ingestion Library
# This library is used to handle downloading videos from YouTube and other platforms.
# It also handles the ingestion of the videos into the database.
# It uses yt-dlp to extract video information and download the videos.
####
import json
####################
# Function List
#
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, chunk_summarization, chunk_duration_input, words_per_second_input)
#
#
####################
# Import necessary libraries to run solo for testing
import os
import re
import sys
from urllib.parse import urlparse, parse_qs

# 3rd-Party Imports
import yt_dlp
import unicodedata
# Import Local
from App_Function_Libraries.DB.DB_Manager import check_media_and_whisper_model
from App_Function_Libraries.Utils.Utils import logging


#
#######################################################################################################################
# Function Definitions
#

def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '').replace('*', '').replace('?',
                                                                                                                   '').replace(
        '<', '').replace('>', '').replace('|', '')
    return title

def get_video_info(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict
        except Exception as e:
            logging.error(f"Error extracting video info: {e}")
            return None


def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'noplaylist': False,
        'quiet': True,
        'extract_flat': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logging.debug("About to extract youtube info")
        info_dict = ydl.extract_info(video_url, download=False)
        logging.debug("Youtube info successfully extracted")
    return info_dict


def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None


def download_video(video_url, download_path, info_dict, download_video_flag, current_whisper_model):
    global video_file_path, ffmpeg_path
    global audio_file_path

    # Normalize Video Title name
    logging.debug("About to normalize downloaded video title")
    if 'title' not in info_dict or 'ext' not in info_dict:
        logging.error("info_dict is missing 'title' or 'ext'")
        return None

    normalized_video_title = normalize_title(info_dict['title'])

    # FIXME - make sure this works/checks against hte current model
    # Check if media already exists in the database and compare whisper models
    should_download, reason = check_media_and_whisper_model(
        title=normalized_video_title,
        url=video_url,
        current_whisper_model=current_whisper_model
    )

    if not should_download:
        logging.info(f"Skipping download: {reason}")
        return None

    logging.info(f"Proceeding with download: {reason}")

    video_file_path = os.path.join(download_path, f"{normalized_video_title}.{info_dict['ext']}")

    # Check for existence of video file
    if os.path.exists(video_file_path):
        logging.info(f"Video file already exists: {video_file_path}")
        return video_file_path

    # Setup path handling for ffmpeg on different OSs
    if sys.platform.startswith('win'):
        ffmpeg_path = os.path.join(os.getcwd(), 'Bin', 'ffmpeg.exe')
    elif sys.platform.startswith('linux'):
        ffmpeg_path = 'ffmpeg'
    elif sys.platform.startswith('darwin'):
        ffmpeg_path = 'ffmpeg'

    if download_video_flag:
        video_file_path = os.path.join(download_path, f"{normalized_video_title}.mp4")
        ydl_opts_video = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]',
            'outtmpl': video_file_path,
            'ffmpeg_location': ffmpeg_path
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                logging.debug("yt_dlp: About to download video with youtube-dl")
                ydl.download([video_url])
                logging.debug("yt_dlp: Video successfully downloaded with youtube-dl")
                if os.path.exists(video_file_path):
                    return video_file_path
                else:
                    logging.error("yt_dlp: Video file not found after download")
                    return None
        except Exception as e:
            logging.error(f"yt_dlp: Error downloading video: {e}")
            return None
    elif not download_video_flag:
        video_file_path = os.path.join(download_path, f"{normalized_video_title}.mp4")
        # Set options for video and audio
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]',
            'quiet': True,
            'outtmpl': video_file_path
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logging.debug("yt_dlp: About to download video with youtube-dl")
                ydl.download([video_url])
                logging.debug("yt_dlp: Video successfully downloaded with youtube-dl")
                if os.path.exists(video_file_path):
                    return video_file_path
                else:
                    logging.error("yt_dlp: Video file not found after download")
                    return None
        except Exception as e:
            logging.error(f"yt_dlp: Error downloading video: {e}")
            return None

    else:
        logging.debug("download_video: Download video flag is set to False and video file path is not found")
        return None


def extract_video_info(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)

            # Log only a subset of the info to avoid overwhelming the logs
            log_info = {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'upload_date': info.get('upload_date')
            }
            logging.debug(f"Extracted info for {url}: {log_info}")

            return info
    except Exception as e:
        logging.error(f"Error extracting video info for {url}: {str(e)}", exc_info=True)
        return None


def get_youtube_playlist_urls(playlist_id):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f'https://www.youtube.com/playlist?list={playlist_id}', download=False)
        return [entry['url'] for entry in result['entries'] if entry.get('url')]


def parse_and_expand_urls(urls):
    logging.info(f"Starting parse_and_expand_urls with input: {urls}")
    expanded_urls = []

    for url in urls:
        try:
            logging.info(f"Processing URL: {url}")
            parsed_url = urlparse(url)
            logging.debug(f"Parsed URL components: {parsed_url}")

            # YouTube playlist handling
            if 'youtube.com' in parsed_url.netloc and 'list' in parsed_url.query:
                playlist_id = parse_qs(parsed_url.query)['list'][0]
                logging.info(f"Detected YouTube playlist with ID: {playlist_id}")
                playlist_urls = get_youtube_playlist_urls(playlist_id)
                logging.info(f"Expanded playlist URLs: {playlist_urls}")
                expanded_urls.extend(playlist_urls)

            # YouTube short URL handling
            elif 'youtu.be' in parsed_url.netloc:
                video_id = parsed_url.path.lstrip('/')
                full_url = f'https://www.youtube.com/watch?v={video_id}'
                logging.info(f"Expanded YouTube short URL to: {full_url}")
                expanded_urls.append(full_url)

            # Vimeo handling
            elif 'vimeo.com' in parsed_url.netloc:
                video_id = parsed_url.path.lstrip('/')
                full_url = f'https://vimeo.com/{video_id}'
                logging.info(f"Processed Vimeo URL: {full_url}")
                expanded_urls.append(full_url)

            # Add more platform-specific handling here

            else:
                logging.info(f"URL not recognized as special case, adding as-is: {url}")
                expanded_urls.append(url)

        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
            # Optionally, you might want to add the problematic URL to expanded_urls
            # expanded_urls.append(url)

    logging.info(f"Final expanded URLs: {expanded_urls}")
    return expanded_urls


def extract_metadata(url, use_cookies=False, cookies=None):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }

    if use_cookies and cookies:
        try:
            cookie_dict = json.loads(cookies)
            ydl_opts['cookiefile'] = cookie_dict
        except json.JSONDecodeError:
            logging.warning("Invalid cookie format. Proceeding without cookies.")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            metadata = {
                'title': info.get('title'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'duration': info.get('duration'),
                'tags': info.get('tags'),
                'description': info.get('description')
            }

            # Create a safe subset of metadata to log
            safe_metadata = {
                'title': metadata.get('title', 'No title'),
                'duration': metadata.get('duration', 'Unknown duration'),
                'upload_date': metadata.get('upload_date', 'Unknown upload date'),
                'uploader': metadata.get('uploader', 'Unknown uploader')
            }

            logging.info(f"Successfully extracted metadata for {url}: {safe_metadata}")
            return metadata
        except Exception as e:
            logging.error(f"Error extracting metadata for {url}: {str(e)}", exc_info=True)
            return None


def generate_timestamped_url(url, hours, minutes, seconds):
    # Extract video ID from the URL
    video_id_match = re.search(r'(?:v=|)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return "Invalid YouTube URL"

    video_id = video_id_match.group(1)

    # Calculate total seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

    # Generate the new URL
    new_url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}s"

    return new_url

#
#
#######################################################################################################################
