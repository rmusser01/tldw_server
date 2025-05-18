# tests/test_media_functions.py
import pytest
from App_Function_Libraries.DB.DB_Manager import add_media_to_database, get_media_content, update_media_content, check_media_exists
#
####################################################################################################
# Test Status:
# FIXME

#
# @pytest.fixture
# def sample_media(empty_db):
#     info_dict = {
#         'title': 'Test Video',
#         'uploader': 'Test Uploader'
#     }
#     media_id = add_media_to_database('https://example.com/test', info_dict, [], 'Test summary', ['test'], 'Test prompt', 'whisper-1')
#     return media_id
#
# def test_add_media_to_database(empty_db):
#     info_dict = {
#         'title': 'Test Video',
#         'uploader': 'Test Uploader'
#     }
#     media_id = add_media_to_database('https://example.com/test', info_dict, [], 'Test summary', ['test'], 'Test prompt', 'whisper-1')
#     assert media_id is not None
#
# def test_get_media_content(empty_db, sample_media):
#     content = get_media_content(sample_media)
#     assert content is not None
#     assert 'Test Video' in content
#
# def test_update_media_content(empty_db, sample_media):
#     new_content = "Updated content for Test Video"
#     update_media_content(sample_media, {'Test Video': sample_media}, new_content, 'Updated prompt', 'Updated summary')
#     updated_content = get_media_content(sample_media)
#     assert new_content in updated_content
#
# def test_check_media_exists(empty_db, sample_media):
#     assert check_media_exists('Test Video', 'https://example.com/test') == sample_media
#     assert check_media_exists('Nonexistent Video', 'https://example.com/nonexistent') is None