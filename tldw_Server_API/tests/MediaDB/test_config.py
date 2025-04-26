# tests/test_config.py
from tldw_Server_API.app.core.Utils.Utils import load_comprehensive_config

def test_load_comprehensive_config():
    config = load_comprehensive_config()
    assert 'Database' in config
    assert 'sqlite_path' in config['Database']
    assert 'API' in config
    assert 'openai_api_key' in config['API']