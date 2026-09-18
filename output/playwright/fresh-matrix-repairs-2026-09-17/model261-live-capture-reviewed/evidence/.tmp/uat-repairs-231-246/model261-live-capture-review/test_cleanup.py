import importlib.util
from pathlib import Path
from types import SimpleNamespace

source = Path(__file__).parents[1] / 'model261-live-capture/uat261_capture.py'
spec = importlib.util.spec_from_file_location('review_capture', source)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

class ProviderStream:
    def __init__(self):
        self.closed = 0
    def __iter__(self):
        return self
    def __next__(self):
        return 'data: [DONE]\n\n'
    def close(self):
        self.closed += 1

def test_abandoned_unstarted_stream_closes_provider(tmp_path):
    provider = ProviderStream()
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kw: provider)
    directory = tmp_path/'capture'
    directory.mkdir(mode=0o700)
    _, observer = capture.install_capture(app=object(),sessions_module=sessions,capture_dir=directory,envelope_builder=lambda messages: SimpleNamespace(fingerprint_version='prompt-v1',aggregate_fingerprint='sha256:fixed',message_count=len(messages)))
    wrapped = sessions.perform_chat_api_call(messages_payload=[],streaming=True)
    wrapped.close()
    observer.close()
    assert provider.closed == 1

def test_provider_iteration_stays_deferred_until_consumer_iterates(tmp_path):
    class LazyStream(ProviderStream):
        def __init__(self):
            super().__init__()
            self.iterated = 0
        def __iter__(self):
            self.iterated += 1
            return self
    provider = LazyStream()
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kw: provider)
    directory = tmp_path/'capture'
    directory.mkdir(mode=0o700)
    _, observer = capture.install_capture(app=object(),sessions_module=sessions,capture_dir=directory,envelope_builder=lambda messages: SimpleNamespace(fingerprint_version='prompt-v1',aggregate_fingerprint='sha256:fixed',message_count=len(messages)))
    wrapped = sessions.perform_chat_api_call(messages_payload=[],streaming=True)
    before_iteration = provider.iterated
    wrapped.close()
    observer.close()
    assert before_iteration == 0
