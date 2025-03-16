# test_summarize.py
# Description: This file contains the unit tests for the summarize.py script.
#
# Usage: python -m unittest test_summarize.py
#
# Imports
import io
import json
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import sys

#
# Local Imports
from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize
from summarize import main, load_and_log_configs, platform_check, cuda_check, check_ffmpeg, extract_video_info, download_video, perform_transcription, perform_summarization, add_media_to_database, semantic_chunk_long_file, ingest_text_file, summarize_with_local_llm
#
#
####################################################################################################
#
# Unit Tests
#
# Test Status:
# FIXME
### IMPORTANT: This is a workaround to allow the test to run from the Tests directory
# Add the directory containing summarize.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the Workflows.json file
json_path = os.path.join(current_dir, '..', '..', 'Helper_Scripts', 'Workflows', 'Workflows.json')


class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.test_config = {
            "api_keys": {"openai": "test_key"},
            "models": {"anthropic": "test_model"},
            "local_apis": {"kobold": {"ip": "127.0.0.1"}},
            "output_path": self.temp_dir,
            "processing_choice": "cpu"
        }
        self.mock_workflows_json = [
            {
                "name": "5 Whys Analysis",
                "context": "A problem-solving technique that involves asking 'why' five times to get to the root cause of an issue.",
                "prompts": [
                    "What is the problem?",
                    "Why did this happen?",
                    "Why did that happen?",
                    "Why did this subsequent event happen?",
                    "Why did this further event happen?",
                    "Why did this final event happen?"
                ]
            },
            {
                "name": "Summarization",
                "context": "A process of condensing information into a concise form while retaining the main points.",
                "prompts": [
                    "Summarize the following text:",
                    "Refine the summary to be more concise:"
                ]
            }
        ]

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)


    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_config(self, mock_json_load, mock_file):
        mock_json_load.return_value = self.mock_workflows_json
        config = load_and_log_configs()
        self.assertEqual(config, self.mock_workflows_json)
        mock_file.assert_called_once_with(json_path, 'r')
        mock_json_load.assert_called_once_with(mock_file())

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_config_file_not_found(self, mock_open):
        with self.assertRaises(FileNotFoundError):
            load_and_log_configs()

    @patch('summarize.platform.system')
    @patch('summarize.platform.machine')
    def test_platform_check(self, mock_machine, mock_system):
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        self.assertIsNone(platform_check())

    @patch('summarize.platform.system')
    def test_platform_check_unsupported(self, mock_system):
        mock_system.return_value = "Unsupported"
        with self.assertRaises(SystemExit):
            platform_check()

    @patch('summarize.torch.cuda.is_available')
    def test_cuda_check(self, mock_cuda_available):
        mock_cuda_available.return_value = True
        self.assertTrue(cuda_check())

    @patch('summarize.subprocess.run')
    def test_check_ffmpeg(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(check_ffmpeg())

    @patch('summarize.subprocess.run')
    def test_check_ffmpeg_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        self.assertFalse(check_ffmpeg())

    @patch('summarize.yt_dlp.YoutubeDL')
    def test_extract_video_info(self, mock_YoutubeDL):
        mock_YoutubeDL.return_value.extract_info.return_value = {
            'title': 'Test Video',
            'duration': 180,
            'view_count': 1000000
        }
        info_dict, title = extract_video_info(self.test_video_url)
        self.assertEqual(title, 'Test Video')
        self.assertEqual(info_dict['duration'], 180)

    @patch('summarize.yt_dlp.YoutubeDL')
    def test_extract_video_info_error(self, mock_YoutubeDL):
        mock_YoutubeDL.return_value.extract_info.side_effect = Exception("Download Error")
        with self.assertRaises(Exception):
            extract_video_info(self.test_video_url)

    @patch('summarize.yt_dlp.YoutubeDL')
    def test_download_video(self, mock_YoutubeDL):
        mock_YoutubeDL.return_value.download.return_value = 0
        result = download_video(self.test_video_url, self.temp_dir, {'title': 'Test Video'}, True, current_whisper_model='small')
        self.assertIsNotNone(result)

    @patch('summarize.speech_to_text')
    def test_perform_transcription(self, mock_speech_to_text):
        mock_speech_to_text.return_value = [{'text': 'Test transcription'}]
        audio_file, segments = perform_transcription('test.mp4', 0, 'small', False)
        self.assertEqual(segments[0]['text'], 'Test transcription')

    @patch('summarize.speech_to_text')
    def test_perform_transcription_with_diarization(self, mock_speech_to_text):
        mock_speech_to_text.return_value = [{'text': 'Test transcription', 'speaker': 'Speaker 1'}]
        audio_file, segments = perform_transcription('test.mp4', 0, 'small', False, diarize=True)
        self.assertEqual(segments[0]['speaker'], 'Speaker 1')

    @patch('summarize.summarize_with_openai')
    def test_perform_summarization(self, mock_summarize):
        mock_summarize.return_value = "Test summary"
        summary = perform_summarization('openai', {'transcription': 'Test'}, "Custom prompt", "test_key")
        self.assertEqual(summary, "Test summary")

    @patch('summarize.sqlite3.connect')
    def test_add_media_to_database(self, mock_connect):
        mock_cursor = MagicMock()
        mock_connect.return_value.cursor.return_value = mock_cursor
        add_media_to_database('test_url', {'title': 'Test'}, [{'text': 'Test'}], 'Summary', ['tag1', 'tag2'],
                              'Custom prompt', 'small')
        mock_cursor.execute.assert_called()

    def test_semantic_chunk_long_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("This is a test file. It contains multiple sentences. We will chunk it.")

        chunks = semantic_chunk_long_file(temp_file.name, max_chunk_size=20)
        self.assertGreater(len(chunks), 1)
        os.unlink(temp_file.name)

    @patch('summarize.add_media_to_database')
    def test_ingest_text_file(self, mock_add_to_db):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("This is a test file for ingestion.")

        result = ingest_text_file(temp_file.name, title="Test", author="Author", keywords=["test"])
        self.assertIn("successfully ingested", result)
        mock_add_to_db.assert_called_once()
        os.unlink(temp_file.name)

    @patch('summarize.subprocess.Popen')
    @patch('summarize.requests.get')
    def test_local_llm_function(self, mock_get, mock_popen):
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"test content"
        mock_popen.return_value.communicate.return_value = (b"Started", b"")
        local_llm_function()
        mock_popen.assert_called_once()

        @patch('summarize.extract_video_info')
        @patch('summarize.download_video')
        @patch('summarize.perform_transcription')
        @patch('summarize.perform_summarization')
        @patch('summarize.add_media_to_database')
        def test_main_integration(self, mock_add_to_db, mock_summarize, mock_transcribe, mock_download,
                                  mock_extract_info):
            mock_extract_info.return_value = ({'title': 'Test Video'}, 'Test Video')
            mock_download.return_value = 'test_video.mp4'
            mock_transcribe.return_value = ('test_audio.wav', [{'text': 'Test transcription'}])
            mock_summarize.return_value = 'Test summary'

            result = main(
                input_path=self.test_video_url,
                api_name='openai',
                api_key='test_key',
                num_speakers=2,
                whisper_model='small',
                offset=0,
                vad_filter=False,
                download_video_flag=False,
                custom_prompt=None,
                overwrite=False,
                rolling_summarization=False,
                detail=0.01,
                keywords=['test'],
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
                system_message=None
            )

            mock_extract_info.assert_called_once_with(self.test_video_url)
            mock_download.assert_called_once()
            mock_transcribe.assert_called_once()
            mock_summarize.assert_called_once()
            mock_add_to_db.assert_called_once()

            self.assertIsNotNone(result)
            self.assertIn('transcription', result)

            @patch('sys.argv')
            @patch('summarize.main')
            def test_command_line_arguments(self, mock_main, mock_argv):
                # Simulate command line arguments
                test_args = [
                    'summarize.py',
                    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                    '--api_name', 'openai',
                    '--num_speakers', '2',
                    '--whisper_model', 'small',
                    '--offset', '0',
                    '--keywords', 'test1', 'test2',
                    '--rolling_summarization',
                    '--detail_level', '0.01'
                ]
                mock_argv.__getitem__.side_effect = lambda s: test_args[s]
                mock_argv.__len__.return_value = len(test_args)

                # Redirect stdout to capture print statements
                captured_output = io.StringIO()
                sys.stdout = captured_output

                # Run the script
                with patch.object(summarize, '__name__', '__main__'):
                    summarize.main()

                # Restore stdout
                sys.stdout = sys.__stdout__

                # Check if main was called with correct arguments
                mock_main.assert_called_once()
                args, kwargs = mock_main.call_args
                self.assertEqual(kwargs['input_path'], 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')
                self.assertEqual(kwargs['api_name'], 'openai')
                self.assertEqual(kwargs['num_speakers'], 2)
                self.assertEqual(kwargs['whisper_model'], 'small')
                self.assertEqual(kwargs['offset'], 0)
                self.assertEqual(kwargs['keywords'], ['test1', 'test2'])
                self.assertTrue(kwargs['rolling_summarization'])
                self.assertEqual(kwargs['detail'], 0.01)

                # Check if help text is printed when no arguments are provided
                mock_argv.__getitem__.side_effect = lambda s: ['summarize.py']
                mock_argv.__len__.return_value = 1

                with patch.object(summarize, '__name__', '__main__'):
                    with self.assertRaises(SystemExit):
                        summarize.main()

                self.assertIn('usage:', captured_output.getvalue())

        if __name__ == '__main__':
            unittest.main()
