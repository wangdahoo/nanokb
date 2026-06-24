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
from nanokb.llm.base import (
    EmbeddingClient,
    LLMClient,
    make_embedding_client,
    make_llm_client,
)
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
    openai_client = make_llm_client(Settings(llm_provider="openai", openai_api_key="sk-fake"))
    anthropic_client = make_llm_client(
        Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-fake")
    )
    ollama_client = make_llm_client(Settings(llm_provider="ollama"))
    assert type(openai_client) is not type(anthropic_client)
    assert type(ollama_client) is not type(openai_client)


# ── openai_base_url 透传（GLM/智谱等 OpenAI 兼容端点） ──────────


def test_openai_client_custom_base_url() -> None:
    client = OpenAIClient(
        api_key="sk-fake",
        model="glm-5.1",
        embedding_model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    assert str(client._client.base_url).startswith("https://open.bigmodel.cn")


def test_openai_client_default_base_url_when_unset() -> None:
    client = OpenAIClient(api_key="sk-fake", model="gpt-4o-mini", embedding_model="x")
    # 未指定 base_url → SDK 默认官方端点
    assert "api.openai.com" in str(client._client.base_url)


def test_factory_passes_base_url_to_openai() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-fake",
        llm_model="glm-5.1",
        openai_base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    client = make_llm_client(settings)
    assert isinstance(client, OpenAIClient)
    assert str(client._client.base_url).startswith("https://open.bigmodel.cn")


def test_factory_default_base_url_when_setting_absent() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="sk-fake")
    client = make_llm_client(settings)
    assert isinstance(client, OpenAIClient)
    assert "api.openai.com" in str(client._client.base_url)


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
    AnthropicClient(api_key="sk-ant-fake", model="claude-3-5-sonnet-20241022", embedding_model="x")
    OllamaClient(base_url="http://localhost:11434", model="llama3", embedding_model="x")
    assert calls == []  # 构造期零调用，下载延迟到 count_tokens


# ══════════════════════════════════════════════════════════════════════
# make_embedding_client 工厂（生文与向量解耦）
# ══════════════════════════════════════════════════════════════════════


def test_make_embedding_client_openai_default() -> None:
    """embedding_provider=openai，回退 openai_api_key/base_url → OpenAIClient。"""
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-fake",
        embedding_model="text-embedding-3-small",
    )
    client = make_embedding_client(settings)
    assert isinstance(client, OpenAIClient)
    assert isinstance(client, EmbeddingClient)  # Protocol 合规
    assert isinstance(client, LLMClient)  # 超集，兼容


def test_make_embedding_client_uses_dedicated_key_and_base_url() -> None:
    """配置独立 embedding_api_key / embedding_base_url 时透传到独立端点。"""
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-deepseek",  # 生文 key（不应被 embedding 使用）
        openai_base_url="https://api.deepseek.com",
        embedding_api_key="sk-glm",
        embedding_base_url="https://open.bigmodel.cn/api/paas/v4",
        embedding_model="embedding-3",
    )
    client = make_embedding_client(settings)
    assert isinstance(client, OpenAIClient)
    assert str(client._client.base_url).startswith("https://open.bigmodel.cn")


def test_make_embedding_client_ollama() -> None:
    """embedding_provider=ollama → OllamaClient（无需 key，本地端点）。"""
    settings = Settings(
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
    )
    client = make_embedding_client(settings)
    assert isinstance(client, OllamaClient)
    assert isinstance(client, EmbeddingClient)


def test_make_embedding_client_openai_missing_key_exits_2() -> None:
    """openai embedding 无任何可用 key（embedding_api_key / openai_api_key 均缺）→ exit 2。"""
    settings = Settings(embedding_provider="openai")  # 无任何 key
    with pytest.raises(SystemExit) as exc:
        make_embedding_client(settings)
    assert exc.value.code == 2


def test_make_embedding_client_openai_falls_back_to_chat_key() -> None:
    """未配 embedding_api_key 时回退 openai_api_key（向后兼容：生文与 embedding 共用）。"""
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-shared",
        embedding_model="text-embedding-3-small",
    )
    client = make_embedding_client(settings)
    assert isinstance(client, OpenAIClient)


def test_make_embedding_client_ollama_needs_no_key() -> None:
    """ollama embedding 不需要 key（本地服务）。"""
    settings = Settings(embedding_provider="ollama", embedding_model="nomic-embed-text")
    client = make_embedding_client(settings)
    assert isinstance(client, OllamaClient)


# ── _resolve_embedder 向后兼容回退（生文/向量解耦调度点） ─────────


def test_resolve_embedder_reuses_chat_llm_without_dedicated_config() -> None:
    """未配置独立 embedding 端点 → 复用 chat llm（同一对象，向后兼容）。"""
    from nanokb.pipeline import _resolve_embedder

    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-fake",
    )  # 无 embedding_api_key / embedding_base_url
    chat_llm = OpenAIClient(api_key="sk-fake", model="deepseek-chat", embedding_model="x")
    embedder = _resolve_embedder(settings, None, chat_llm)
    assert embedder is chat_llm  # 同一对象


def test_resolve_embedder_prefers_explicit_injection() -> None:
    """显式注入 embedding_client 时直接返回（测试 / 自定义场景）。"""
    from nanokb.pipeline import _resolve_embedder

    settings = Settings(embedding_provider="openai", openai_api_key="sk-fake")
    chat_llm = OpenAIClient(api_key="sk-fake", model="m", embedding_model="x")
    explicit = OllamaClient(base_url="http://localhost:11434", model="m", embedding_model="x")
    embedder = _resolve_embedder(settings, explicit, chat_llm)
    assert embedder is explicit


def test_resolve_embedder_builds_independent_client_for_dedicated_config() -> None:
    """配置了独立 embedding 端点 → 构造独立 client（不复用 chat llm）。"""
    from nanokb.pipeline import _resolve_embedder

    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-deepseek",
        openai_base_url="https://api.deepseek.com",
        embedding_api_key="sk-glm",
        embedding_base_url="https://open.bigmodel.cn/api/paas/v4",
        embedding_model="embedding-3",
    )
    chat_llm = OpenAIClient(api_key="sk-deepseek", model="deepseek-chat", embedding_model="x")
    embedder = _resolve_embedder(settings, None, chat_llm)
    assert embedder is not chat_llm
    assert isinstance(embedder, OpenAIClient)
    assert str(embedder._client.base_url).startswith("https://open.bigmodel.cn")


def test_resolve_embedder_builds_ollama_when_provider_is_ollama() -> None:
    """embedding_provider=ollama 即使有 openai_api_key 也走独立 ollama client。"""
    from nanokb.pipeline import _resolve_embedder

    settings = Settings(
        embedding_provider="ollama",
        openai_api_key="sk-fake",  # 生文仍可用 openai 兼容
        embedding_model="nomic-embed-text",
    )
    chat_llm = OpenAIClient(api_key="sk-fake", model="deepseek-chat", embedding_model="x")
    embedder = _resolve_embedder(settings, None, chat_llm)
    assert embedder is not chat_llm
    assert isinstance(embedder, OllamaClient)
