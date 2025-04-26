import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import yt_dlp

from PoC_Version.App_Function_Libraries.Utils.Utils import sanitize_filename, downloaded_files


def create_utilities_yt_video_tab():
    with gr.TabItem("YouTube Video Downloader", id='youtube_dl', visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "<h3>Youtube Video Downloader</h3><p>This Input takes a Youtube URL as input and creates a webm file for you to download. </br><em>If you want a full-featured one:</em> <strong><em>https://github.com/StefanLobbenmeier/youtube-dl-gui</strong></em> or <strong><em>https://github.com/yt-dlg/yt-dlg</em></strong></p>")
                youtube_url_input = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here")
                download_button = gr.Button("Download Video")
            with gr.Column():
                output_file = gr.File(label="Download Video")
                output_message = gr.Textbox(label="Status")

        download_button.click(
            fn=gradio_download_youtube_video,
            inputs=youtube_url_input,
            outputs=[output_file, output_message]
        )

def create_utilities_yt_audio_tab():
    with gr.TabItem("YouTube Audio Downloader", id="youtube audio downloader", visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "<h3>Youtube Audio Downloader</h3><p>This Input takes a Youtube URL as input and creates an audio file for you to download.</p>"
                    +"\n<em>If you want a full-featured one:</em> <strong><em>https://github.com/StefanLobbenmeier/youtube-dl-gui</strong></em>\n or \n<strong><em>https://github.com/yt-dlg/yt-dlg</em></strong></p>")
                youtube_url_input_audio = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here")
                download_button_audio = gr.Button("Download Audio")
            with gr.Column():
                output_file_audio = gr.File(label="Download Audio")
                output_message_audio = gr.Textbox(label="Status")

        from PoC_Version.App_Function_Libraries.Audio.Audio_Files import download_youtube_audio
        download_button_audio.click(
            fn=download_youtube_audio,
            inputs=youtube_url_input_audio,
            outputs=[output_file_audio, output_message_audio]
        )

def create_utilities_yt_timestamp_tab():
    with gr.TabItem("YouTube Timestamp URL Generator", id="timestamp-gen", visible=True):
        gr.Markdown("## Generate YouTube URL with Timestamp")
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="YouTube URL")
                hours_input = gr.Number(label="Hours", value=0, minimum=0, precision=0)
                minutes_input = gr.Number(label="Minutes", value=0, minimum=0, maximum=59, precision=0)
                seconds_input = gr.Number(label="Seconds", value=0, minimum=0, maximum=59, precision=0)
                generate_button = gr.Button("Generate URL")
            with gr.Column():
                output_url = gr.Textbox(label="Timestamped URL")

        from PoC_Version.App_Function_Libraries.Video_DL_Ingestion_Lib import generate_timestamped_url
        generate_button.click(
            fn=generate_timestamped_url,
            inputs=[url_input, hours_input, minutes_input, seconds_input],
            outputs=output_url
        )


def gradio_download_youtube_video(url):
    try:
        # Determine ffmpeg path based on the operating system.
        ffmpeg_path = './Bin/ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract information about the video
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                sanitized_title = sanitize_filename(info_dict['title'])
                original_ext = info_dict['ext']

            # Setup the temporary filename
            temp_file_path = Path(temp_dir) / f"{sanitized_title}.{original_ext}"

            # Initialize yt-dlp with generic options and the output template
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'ffmpeg_location': ffmpeg_path,
                'outtmpl': str(temp_file_path),
                'noplaylist': True,
                'quiet': True
            }

            # Execute yt-dlp to download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Final check to ensure file exists
            if not temp_file_path.exists():
                raise FileNotFoundError(f"Expected file was not found: {temp_file_path}")

            # Create a persistent directory for the download if it doesn't exist
            persistent_dir = Path("downloads")
            persistent_dir.mkdir(exist_ok=True)

            # Move the file from the temporary directory to the persistent directory
            persistent_file_path = persistent_dir / f"{sanitized_title}.{original_ext}"
            shutil.move(str(temp_file_path), str(persistent_file_path))

            # Add the file to the list of downloaded files
            downloaded_files.append(str(persistent_file_path))

            return str(persistent_file_path), f"Video downloaded successfully: {sanitized_title}.{original_ext}"
    except Exception as e:
        return None, f"Error downloading video: {str(e)}"

