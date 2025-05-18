# test_Video_DL_ingestion.py
# Purpose: Unit test for the Video_DL_Ingestion_Lib module
# Run them: python -m unittest test_Video_DL_Ingestion_Lib.py
#############################################################################################################
#
# Imports
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Add the parent directory to the Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App_Function_Libraries.Video_DL_Ingestion_Lib import (
    normalize_title,
    get_video_info,
    parse_and_expand_urls,
    extract_metadata,
    generate_timestamped_url,
    download_video,
    get_youtube_playlist_urls
)
#
#
#############################################################################################################
#
# Functions:

class TestVideoDLIngestionLib(unittest.TestCase):

    def test_normalize_title(self):
        test_cases = [
            ("Test: Video Title", "Test_ Video Title"),
            ("Test/ Video\\Title", "Test_ Video_Title"),
            ("Test: Video? Title*", "Test_ Video_ Title"),
            ("áéíóú", "aeiou"),  # Test handling of accented characters
            ("", ""),  # Test empty string
            ("   Spaces   ", "   Spaces   "),  # Test handling of spaces
        ]
        for input_title, expected_output in test_cases:
            with self.subTest(input_title=input_title):
                self.assertEqual(normalize_title(input_title), expected_output)

    @patch('yt_dlp.YoutubeDL')
    def test_get_video_info(self, mock_ytdl):
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {"title": "Test Video", "duration": 300}
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        result = get_video_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(result, {"title": "Test Video", "duration": 300})

        # Test error handling
        mock_instance.extract_info.side_effect = Exception("Network error")
        result = get_video_info("https://www.youtube.com/watch?v=invalid")
        self.assertIsNone(result)

    @patch('Video_DL_Ingestion_Lib.get_youtube_playlist_urls')
    def test_parse_and_expand_urls(self, mock_get_playlist):
        mock_get_playlist.return_value = [
            "https://www.youtube.com/watch?v=video1",
            "https://www.youtube.com/watch?v=video2"
        ]

        urls = [
            "https://www.youtube.com/playlist?list=PLtest123",
            "https://youtu.be/shorturl",
            "https://vimeo.com/123456789",
            "https://www.example.com/video"
        ]

        result = parse_and_expand_urls(urls)

        expected = [
            "https://www.youtube.com/watch?v=video1",
            "https://www.youtube.com/watch?v=video2",
            "https://www.youtube.com/watch?v=shorturl",
            "https://vimeo.com/123456789",
            "https://www.example.com/video"
        ]

        self.assertEqual(result, expected)

    @patch('yt_dlp.YoutubeDL')
    def test_extract_metadata(self, mock_ytdl):
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            "title": "Test Video",
            "uploader": "Test Channel",
            "upload_date": "20230101",
            "view_count": 1000,
            "like_count": 100,
            "duration": 300,
            "tags": ["test", "video"],
            "description": "This is a test video"
        }
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        result = extract_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        expected = {
            "title": "Test Video",
            "uploader": "Test Channel",
            "upload_date": "20230101",
            "view_count": 1000,
            "like_count": 100,
            "duration": 300,
            "tags": ["test", "video"],
            "description": "This is a test video"
        }
        self.assertEqual(result, expected)

        # Test with cookies
        cookies = json.dumps({"cookie_name": "cookie_value"})
        result_with_cookies = extract_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ", use_cookies=True,
                                               cookies=cookies)
        self.assertEqual(result_with_cookies, expected)

        # Test error handling
        mock_instance.extract_info.side_effect = Exception("Network error")
        result_error = extract_metadata("https://www.youtube.com/watch?v=invalid")
        self.assertIsNone(result_error)

    def test_generate_timestamped_url(self):
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0, 1, 30,
             "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s"),
            ("https://youtu.be/dQw4w9WgXcQ", 1, 0, 0, "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3600s"),
            (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0, 0, 0, "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s"),
            ("https://www.example.com/video", 0, 1, 30, "Invalid YouTube URL"),
        ]
        for url, hours, minutes, seconds, expected in test_cases:
            with self.subTest(url=url, hours=hours, minutes=minutes, seconds=seconds):
                result = generate_timestamped_url(url, hours, minutes, seconds)
                self.assertEqual(result, expected)

    @patch('Video_DL_Ingestion_Lib.check_media_and_whisper_model')
    @patch('yt_dlp.YoutubeDL')
    @patch('os.path.exists')
    def test_download_video(self, mock_exists, mock_ytdl, mock_check):
        mock_check.return_value = (True, "Proceeding with download")
        mock_exists.return_value = False
        mock_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        info_dict = {'title': 'Test Video', 'ext': 'mp4'}
        result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "/tmp", info_dict, True, "whisper_model")

        self.assertIsNotNone(result)
        mock_instance.download.assert_called_once()

        # Test when file already exists
        mock_exists.return_value = True
        result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "/tmp", info_dict, True, "whisper_model")
        self.assertIsNotNone(result)

        # Test when download is not needed
        mock_check.return_value = (False, "Skipping download")
        result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "/tmp", info_dict, True, "whisper_model")
        self.assertIsNone(result)

    @patch('yt_dlp.YoutubeDL')
    def test_get_youtube_playlist_urls(self, mock_ytdl):
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            'entries': [
                {'url': 'https://www.youtube.com/watch?v=video1'},
                {'url': 'https://www.youtube.com/watch?v=video2'},
                {'_type': 'url', 'url': 'https://www.youtube.com/watch?v=video3'},
            ]
        }
        mock_ytdl.return_value.__enter__.return_value = mock_instance

        result = get_youtube_playlist_urls("PLtest123")
        expected = [
            'https://www.youtube.com/watch?v=video1',
            'https://www.youtube.com/watch?v=video2',
            'https://www.youtube.com/watch?v=video3'
        ]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()