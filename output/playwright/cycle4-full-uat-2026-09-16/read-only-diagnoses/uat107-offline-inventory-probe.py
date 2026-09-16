"""Read-only AST probe: execute current production functions without importing app/runtime."""
import ast
import configparser
import contextlib
from functools import lru_cache
import json
from pathlib import Path
import re

root = Path('/Users/macbook-dev/Documents/GitHub/tldw_server2')
chat_path = root / 'tldw_Server_API/app/core/Chat/chat_service.py'
config_path = root / 'tldw_Server_API/app/core/config.py'
setup_path = root / 'tldw_Server_API/app/api/v1/endpoints/setup.py'
startup = configparser.ConfigParser(interpolation=None)
startup.read('/private/tmp/tldw-onboarding-uat-cycle4-single-20260916/Config_Files/config.txt.pre-setup-20260916003245.bak')
saved = configparser.ConfigParser(interpolation=None)
saved.read('/private/tmp/tldw-onboarding-uat-cycle4-single-20260916/Config_Files/config.txt')
old_model = startup.get('API', 'custom_openai_api_model')
new_model = saved.get('API', 'custom_openai_api_model')
assert old_model != new_model

@lru_cache(maxsize=1)
def loader():
    return storage['config']

@lru_cache(maxsize=64)
def harmless(*args):
    return None

class LazyFixture:
    pass

storage = {'config': startup}
ns = dict(lru_cache=lru_cache, contextlib=contextlib, re=re,
          _CHAT_NONCRITICAL_EXCEPTIONS=Exception,
          _PROVIDER_MODEL_CONFIG_FIELDS={'custom-openai-api': ('API', 'custom_openai_api_model')},
          list_provider_models=lambda provider: [], load_comprehensive_config=loader,
          _config=loader(), _CONFIG_PARSER_CACHE=startup, _CONFIG_SOURCE_METADATA={},
          _route_toggle_policy=harmless, should_disable_cors=harmless,
          should_allow_cors_credentials=harmless, is_production_environment=harmless,
          settings=LazyFixture(), loaded_config_data=LazyFixture(),
          _load_models_with_case_cached=harmless, _load_alias_overrides_cached=harmless,
          _provider_has_model_cached=harmless, _find_catalog_providers_for_model_cached=harmless,
          _clear_openrouter_model_cache_shared=lambda: None)

def load_functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), ns)

load_functions(chat_path, ['_split_model_list', '_configured_models_for_provider_cached', 'known_models_for_provider_cached', 'is_model_known_for_provider', 'invalidate_model_alias_caches'])
load_functions(config_path, ['refresh_config_cache', 'clear_config_cache'])
# The successful setup branch calls clear_config_cache; exceptions/logging aren't invoked.
load_functions(setup_path, ['_refresh_runtime_config_cache'])
check = ns['is_model_known_for_provider']
assert check('custom-openai-api', old_model) is True
assert check('custom-openai-api', new_model) is False
storage['config'] = saved
assert ns['_refresh_runtime_config_cache']('offline fixture') is True
fresh_config_matches_new = loader().get('API', 'custom_openai_api_model') == new_model
stale_after_setup = check('custom-openai-api', new_model)
ns['invalidate_model_alias_caches']()
stale_after_chat_cache_clear = check('custom-openai-api', new_model)
# Restart analogue only: replace module startup parser, then clear model caches.
ns['_config'] = loader()
ns['invalidate_model_alias_caches']()
accepted_after_snapshot_refresh = check('custom-openai-api', new_model)
unknown_after_refresh = check('custom-openai-api', 'unadvertised-negative-control')
assert fresh_config_matches_new is True
assert stale_after_setup is False
assert stale_after_chat_cache_clear is False
assert accepted_after_snapshot_refresh is True
assert unknown_after_refresh is False
print(json.dumps({
    'probe': 'actual AST functions; no app import, network, runtime or data mutation',
    'startup_source': 'isolated actual pre-setup backup 20260916003245',
    'startup_model': old_model, 'saved_model': new_model,
    'fresh_config_matches_saved_model': fresh_config_matches_new,
    'saved_model_accepted_after_setup_refresh': stale_after_setup,
    'saved_model_accepted_after_additional_chat_lru_clear': stale_after_chat_cache_clear,
    'saved_model_accepted_after_startup_snapshot_refresh': accepted_after_snapshot_refresh,
    'unadvertised_model_accepted_after_refresh': unknown_after_refresh,
    'pricing_catalog': 'empty controlled fixture; actual 400 independently confirms no matching runtime inventory entry'
}, indent=2))
