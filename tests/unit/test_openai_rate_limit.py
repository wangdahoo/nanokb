"""OpenAIClient 速率限制回归测试（429 重试 + 请求间隔节流）。

不发起真实 API 调用：monkeypatch 替换 ``_client.chat.completions.create``，
用 mock 模拟 RateLimitError 与成功响应。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import httpx
import pytest
from openai import RateLimitError

from nanokb.llm.openai_client import OpenAIClient
from nanokb.llm.throttle import RateLimiter


def _make_rate_limit_error() -> RateLimitError:
    """构造一个最小 RateLimitError（429）。"""
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        headers={"content-type": "application/json"},
        content=b'{"error": {"code": "1302", "message": "rate limited"}}',
    )
    return RateLimitError(message="rate limited", response=response, body=None)


def _make_chat_response(content: str = '{"triples": [], "concepts": []}') -> MagicMock:
    """构造一个 mock chat completion 响应。"""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ── 429 应用层重试 ──────────────────────────────────────────────


def test_rate_limit_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次 429 → 应用层退避重试 → 第二次成功。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="x",
        rate_limit_retries=3,
    )
    create_mock = MagicMock(
        side_effect=[_make_rate_limit_error(), _make_chat_response('{"ok": true}')]
    )
    client._client.chat.completions.create = create_mock
    monkeypatch.setattr(time, "sleep", lambda _: None)  # 跳过真实 sleep

    result = client.complete("system", "user")
    assert result == '{"ok": true}'
    assert create_mock.call_count == 2


def test_rate_limit_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """连续 429 超过 rate_limit_retries 次后抛出 RateLimitError。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="x",
        rate_limit_retries=2,
    )
    create_mock = MagicMock(side_effect=_make_rate_limit_error())
    client._client.chat.completions.create = create_mock
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(RateLimitError):
        client.complete("system", "user")
    # 1 次初始 + 2 次重试 = 3 次
    assert create_mock.call_count == 3


def test_rate_limit_zero_retries_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rate_limit_retries=0 时首次 429 立即抛出，无重试。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="x",
        rate_limit_retries=0,
    )
    create_mock = MagicMock(side_effect=_make_rate_limit_error())
    client._client.chat.completions.create = create_mock
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(RateLimitError):
        client.complete("system", "user")
    assert create_mock.call_count == 1


def test_rate_limit_backoff_uses_exponential_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """退避时长应为指数增长（10s, 20s, 40s ...）。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="x",
        rate_limit_retries=3,
    )
    create_mock = MagicMock(side_effect=_make_rate_limit_error())
    client._client.chat.completions.create = create_mock

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    with pytest.raises(RateLimitError):
        client.complete("system", "user")

    # 3 次重试：10s, 20s, 40s
    assert sleep_calls == [10.0, 20.0, 40.0]


# ── 请求间隔节流 ───────────────────────────────────────────────


def test_throttle_enforces_min_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """注入 RateLimiter(interval>0) 时两次 complete 间至少间隔指定秒数。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="x",
        rate_limiter=RateLimiter(interval=2.0),
    )
    client._client.chat.completions.create = MagicMock(return_value=_make_chat_response())

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    # 用可控的 monotonic：每次调用推进 0s（模拟瞬时返回）
    mock_time = MagicMock(side_effect=[0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(time, "monotonic", mock_time)

    client.complete("s", "u")  # 首次：无上次调用时间，不 sleep
    client.complete("s", "u")  # 第二次：距上次 0s，应 sleep 2.0s

    # 第二次调用应触发一次 sleep(2.0)
    assert 2.0 in sleep_calls


def test_throttle_skips_when_interval_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注入 RateLimiter(interval=0) 时不触发任何 sleep（无限流快速返回）。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="x",
        rate_limiter=RateLimiter(interval=0.0),
    )
    client._client.chat.completions.create = MagicMock(return_value=_make_chat_response())

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    client.complete("s", "u")
    client.complete("s", "u")

    assert sleep_calls == []


# ── embed 也受节流保护 ─────────────────────────────────────────


def test_embed_respects_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed 调用前也应经过注入的 RateLimiter 节流。"""
    client = OpenAIClient(
        api_key="sk-fake",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        rate_limiter=RateLimiter(interval=1.5),
    )
    embed_resp = MagicMock()
    embed_resp.data = [MagicMock(embedding=[0.1, 0.2])]
    client._client.embeddings.create = MagicMock(return_value=embed_resp)

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    # acquire 首次调用：monotonic() ×1（_last_ts=None，跳过 sleep）
    # acquire 第二次调用：monotonic() ×1（计算 elapsed），触发 sleep，再 monotonic() ×1（更新 ts）
    mock_time = MagicMock(side_effect=[100.0, 100.0, 101.5])
    monkeypatch.setattr(time, "monotonic", mock_time)

    client.embed(["text"])  # 首次：不 sleep
    assert sleep_calls == []

    client.embed(["text2"])  # 第二次：应 sleep 1.5s
    assert 1.5 in sleep_calls
