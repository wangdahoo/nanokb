"""Anthropic tool-use JSON 通道测试（方案 §3.4.1，Opt #4 v3）。

验证 complete(response_format='json') 走 emit_json tool-use 通道，
返回可 json.loads 的字符串。通过注入 fake SDK 客户端，不发起真实请求。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from nanokb.llm.anthropic_client import AnthropicClient


def _tool_use_response(payload: dict[str, Any]) -> Any:
    """构造包含单个 tool_use 块的伪 anthropic Message 响应。"""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use", name="emit_json", input=payload
            )
        ],
        stop_reason="tool_use",
    )


def _text_response(text: str) -> Any:
    """构造包含单个 text 块的伪 anthropic Message 响应。"""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _FakeMessages:
    """模拟 anthropic.Anthropic.messages：记录调用 kwargs 并返回预设响应。"""

    def __init__(self, response: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.response is not None:
            return self.response
        # 默认返回一个 tool_use 响应
        return _tool_use_response({"triples": [], "concepts": []})


def _make_client(response: Any = None) -> tuple[AnthropicClient, _FakeMessages]:
    client = AnthropicClient(
        api_key="sk-ant-fake",
        model="claude-3-5-sonnet-20241022",
        embedding_model="text-embedding-3-small",
    )
    fake = _FakeMessages(response=response)
    # 注入伪 SDK 客户端（duck-typed：仅用到 .messages.create）
    client._client = SimpleNamespace(messages=fake)  # type: ignore[assignment]
    return client, fake


def test_json_response_uses_tool_use_channel() -> None:
    client, fake = _make_client()
    result = client.complete("system prompt", "user prompt", response_format="json")

    # 返回值必须是可 json.loads 的字符串
    parsed = json.loads(result)
    assert parsed == {"triples": [], "concepts": []}

    # 验证走了 tool-use 通道
    assert len(fake.calls) == 1
    kwargs = fake.calls[0]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_json"}
    tools = kwargs["tools"]
    assert tools[0]["name"] == "emit_json"
    # Opt #4 v3：空 schema 仅保证通道，不校验结构
    assert tools[0]["input_schema"] == {"type": "object", "properties": {}}


def test_json_result_is_json_loadable_string() -> None:
    # tool_use.input 为复杂嵌套 dict 时，json.dumps 后仍可 json.loads 还原
    payload = {"deep": {"nested": [1, 2, {"k": "v"}]}, "ok": True}
    client, fake = _make_client(response=_tool_use_response(payload))
    result = client.complete("s", "u", response_format="json")
    assert json.loads(result) == payload


def test_text_response_skips_tool_use() -> None:
    # response_format='text' 时不应传 tools/tool_choice
    client, fake = _make_client(response=_text_response("plain answer"))
    result = client.complete("sys", "usr", response_format="text")
    assert result == "plain answer"
    assert len(fake.calls) == 1
    kwargs = fake.calls[0]
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_temperature_passed_through() -> None:
    client, fake = _make_client()
    client.complete("s", "u", response_format="json", temperature=0.3)
    assert fake.calls[0]["temperature"] == 0.3


def test_system_and_user_messages_passed() -> None:
    client, fake = _make_client()
    client.complete("my system", "my user", response_format="json")
    kwargs = fake.calls[0]
    assert kwargs["system"] == "my system"
    assert kwargs["messages"] == [{"role": "user", "content": "my user"}]


def test_fallback_to_text_when_no_tool_use_block() -> None:
    # 模型未返回 tool_use 块（降级场景）：应退回文本而非崩溃
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text='{"fallback": true}'),
            SimpleNamespace(type="text", text=" (extra)"),
        ]
    )
    client, fake = _make_client(response=response)
    result = client.complete("s", "u", response_format="json")
    # 文本拼接（上游可配合 parse_json_loose 容错）
    assert '{"fallback": true}' in result
    assert "(extra)" in result


def test_empty_embed_returns_empty_list() -> None:
    client = AnthropicClient(
        api_key="sk-ant-fake",
        model="claude-3-5-sonnet-20241022",
        embedding_model="x",
    )
    assert client.embed([]) == []


def test_embed_without_openai_key_raises() -> None:
    # Anthropic 无 embeddings API；未提供 openai_api_key 时应清晰报错
    client = AnthropicClient(
        api_key="sk-ant-fake",
        model="claude-3-5-sonnet-20241022",
        embedding_model="x",
        openai_api_key=None,
    )
    try:
        client.embed(["some text"])
    except RuntimeError as exc:
        assert "openai_api_key" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for embed without openai_api_key")
