"""make_llm_client 工厂 + count_tokens 测试（方案 §3.4.1）。

不发起真实 LLM 调用：构造用 fake key（SDK 构造不发请求）。
count_tokens 通过 monkeypatch 注入 fake encoding，完全离线（避免 tiktoken
首次使用从 openaipublic.blob.core.windows.net 下载 BPE 文件）。
"""

from __future__ import annotations

import pytest
import tiktoken

from nanokb.config import Settings
from nanokb.llm.anthropic_client import AnthropicClient
from nanokb.llm.base import LLMClient, make_llm_client
from nanokb.llm.ollama_client import OllamaClient
from nanokb.llm.openai_client import OpenAIClient


class _FakeEncoding:
    """离线伪 tokenizer：每个字符计为 1 token（确定性）。"""

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encode_calls.append(text)
        return list(range(len(text)))


@pytest.fixture
def fake_tiktoken(monkeypatch: pytest.MonkeyPatch) -> _FakeEncoding:
    """全局替换 tiktoken 编码函数，返回同一个 fake encoding。"""
    fake = _FakeEncoding()
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: fake)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: fake)
    return fake


# ── factory 按 provider 分发 ───────────────────────────────────


def test_make_openai_client() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-fake",
        llm_model="gpt-4o-mini",
    )
    client = make_llm_client(settings)
    assert isinstance(client, OpenAIClient)
    assert isinstance(client, LLMClient)  # Protocol 合规（runtime_checkable）


def test_make_anthropic_client() -> None:
    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-fake",
        llm_model="claude-3-5-sonnet-20241022",
    )
    client = make_llm_client(settings)
    assert isinstance(client, AnthropicClient)
    assert isinstance(client, LLMClient)


def test_make_ollama_client() -> None:
    settings = Settings(
        llm_provider="ollama",
        llm_model="llama3",
        ollama_base_url="http://localhost:11434",
    )
    client = make_llm_client(settings)
    assert isinstance(client, OllamaClient)
    assert isinstance(client, LLMClient)


def test_factory_returns_three_distinct_provider_types() -> None:
    openai_client = make_llm_client(
        Settings(llm_provider="openai", openai_api_key="sk-fake")
    )
    anthropic_client = make_llm_client(
        Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-fake")
    )
    ollama_client = make_llm_client(Settings(llm_provider="ollama"))
    assert type(openai_client) is not type(anthropic_client)
    assert type(ollama_client) is not type(openai_client)


# ── API key 缺失 → exit code 2 ─────────────────────────────────


def test_openai_missing_key_exits_2() -> None:
    settings = Settings(llm_provider="openai")  # 无 key
    with pytest.raises(SystemExit) as exc:
        make_llm_client(settings)
    assert exc.value.code == 2


def test_anthropic_missing_key_exits_2() -> None:
    settings = Settings(llm_provider="anthropic")  # 无 key
    with pytest.raises(SystemExit) as exc:
        make_llm_client(settings)
    assert exc.value.code == 2


def test_openai_empty_key_exits_2() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(SystemExit) as exc:
        make_llm_client(settings)
    assert exc.value.code == 2


def test_anthropic_empty_key_exits_2() -> None:
    settings = Settings(llm_provider="anthropic", anthropic_api_key="")
    with pytest.raises(SystemExit) as exc:
        make_llm_client(settings)
    assert exc.value.code == 2


def test_ollama_needs_no_key() -> None:
    # ollama 不需要 API key，不应 exit
    settings = Settings(llm_provider="ollama")
    client = make_llm_client(settings)
    assert isinstance(client, OllamaClient)


# ── count_tokens 精确性（monkeypatch tiktoken，离线） ─────────


def test_openai_count_tokens_uses_encoding_for_model(fake_tiktoken: _FakeEncoding) -> None:
    client = OpenAIClient(
        api_key="sk-fake", model="gpt-4o-mini", embedding_model="text-embedding-3-small"
    )
    # fake encoding：每字符 1 token，count_tokens 应等于文本长度
    text = "Transformer 依赖 Attention"
    assert client.count_tokens(text) == len(text)
    assert fake_tiktoken.encode_calls == [text]


def test_openai_count_tokens_falls_back_on_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeEncoding()

    def raise_keyerror(model: str) -> _FakeEncoding:
        raise KeyError(model)

    monkeypatch.setattr(tiktoken, "encoding_for_model", raise_keyerror)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: fake)
    client = OpenAIClient(api_key="sk-fake", model="totally-unknown-model", embedding_model="x")
    assert client.count_tokens("abc") == 3


def test_openai_encoding_cached_after_first_call(fake_tiktoken: _FakeEncoding) -> None:
    client = OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x")
    client.count_tokens("first")
    client.count_tokens("second")
    # encoding_for_model 仅被 tiktoken 内部调用一次（懒加载缓存）
    assert fake_tiktoken.encode_calls == ["first", "second"]


def test_count_tokens_returns_zero_for_empty(fake_tiktoken: _FakeEncoding) -> None:
    clients: list[LLMClient] = [
        OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x"),
        AnthropicClient(
            api_key="sk-ant-fake", model="claude-3-5-sonnet-20241022", embedding_model="x"
        ),
        OllamaClient(base_url="http://localhost:11434", model="llama3", embedding_model="x"),
    ]
    for client in clients:
        assert client.count_tokens("") == 0


def test_count_tokens_positive_for_text(fake_tiktoken: _FakeEncoding) -> None:
    clients: list[LLMClient] = [
        OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x"),
        AnthropicClient(
            api_key="sk-ant-fake", model="claude-3-5-sonnet-20241022", embedding_model="x"
        ),
        OllamaClient(base_url="http://localhost:11434", model="llama3", embedding_model="x"),
    ]
    text = "Hello world, this is a token counting test."
    for client in clients:
        assert client.count_tokens(text) > 0


def test_count_tokens_stable_and_deterministic(fake_tiktoken: _FakeEncoding) -> None:
    client = OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x")
    text = "stability check 123"
    assert client.count_tokens(text) == client.count_tokens(text)


def test_openai_count_tokens_distinct_for_distinct_text(fake_tiktoken: _FakeEncoding) -> None:
    client = OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x")
    short = "a"
    long_text = "a" * 200
    assert client.count_tokens(short) < client.count_tokens(long_text)


# ── 构造不触发网络/下载（懒加载契约） ──────────────────────────


def test_client_construction_is_offline_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    # 构造期不应调用 tiktoken.encoding_for_model / get_encoding（懒加载）
    calls: list[str] = []

    def spy_encoding_for_model(model: str) -> _FakeEncoding:
        calls.append("encoding_for_model")
        return _FakeEncoding()

    def spy_get_encoding(name: str) -> _FakeEncoding:
        calls.append("get_encoding")
        return _FakeEncoding()

    monkeypatch.setattr(tiktoken, "encoding_for_model", spy_encoding_for_model)
    monkeypatch.setattr(tiktoken, "get_encoding", spy_get_encoding)

    OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x")
    AnthropicClient(
        api_key="sk-ant-fake", model="claude-3-5-sonnet-20241022", embedding_model="x"
    )
    OllamaClient(base_url="http://localhost:11434", model="llama3", embedding_model="x")
    assert calls == []  # 构造期零调用，下载延迟到 count_tokens
