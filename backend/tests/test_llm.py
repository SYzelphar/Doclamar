import pytest

from doclamar import llm as llm_module
from doclamar.llm import LLMClient, LLMConfig, LLMError, LLMNotConfigured


class StatusError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()


class ScriptedBackend:
    """Raises the scripted errors in order, then returns 'ok'."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def complete(self, *args):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _client(backend):
    client = LLMClient(max_retries=2)
    client._backend = backend
    client._config = LLMConfig("groq", "k", "m")
    return client


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    return sleeps


def test_retries_rate_limits_using_retry_after(no_sleep):
    backend = ScriptedBackend([StatusError(429, retry_after="3"), StatusError(503)])
    assert _client(backend).generate("hi") == "ok"
    assert backend.calls == 3
    assert no_sleep[0] == 3.0 and 1.0 <= no_sleep[1] <= 3.0


def test_gives_up_with_a_readable_message():
    backend = ScriptedBackend([StatusError(429)] * 5)
    with pytest.raises(LLMError, match="Groq rate limit reached"):
        _client(backend).generate("hi")


def test_bad_key_and_missing_model_fail_fast():
    with pytest.raises(LLMError, match="rejected the API key"):
        _client(ScriptedBackend([StatusError(401)])).generate("hi")
    with pytest.raises(LLMError, match="Model not found"):
        _client(ScriptedBackend([StatusError(404)])).generate("hi")


def test_not_configured():
    with pytest.raises(LLMNotConfigured):
        LLMClient().generate("hi")


def test_config_from_env(monkeypatch):
    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert LLMConfig.from_env() is None

    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = LLMConfig.from_env()
    assert (cfg.provider, cfg.model) == ("gemini", "gemini-2.5-flash")

    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "custom/model")
    cfg = LLMConfig.from_env()
    assert (cfg.provider, cfg.api_key, cfg.model) == ("groq", "q", "custom/model")


def test_configure_rejects_unknown_provider_and_empty_key():
    client = LLMClient()
    with pytest.raises(LLMError):
        client.configure("openai", "k", validate=False)
    with pytest.raises(LLMError):
        client.configure("groq", "  ", validate=False)
    assert client.configure("groq", "k", validate=False)["model"] == "openai/gpt-oss-120b"
