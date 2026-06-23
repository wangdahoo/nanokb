"""count_tokens 单测（方案 §3.4.1 + Opt #6 + AC #4）。

验证：与 tiktoken 直接编码一致、非 chars/4 粗估、空串、未知 model fallback。
默认 model 为 glm-5.1，tiktoken 不识别 → fallback 到 cl100k_base（BPE 词表已内置
于 utils/_tiktoken_cache/，离线可用，无需访问 OpenAI CDN）。
"""

from __future__ import annotations

import tiktoken

from nanokb.utils.tokenize import DEFAULT_MODEL, count_tokens

# glm-5.1 对 tiktoken 是未知 model，统一 fallback 到 cl100k_base。
_ENC = tiktoken.get_encoding("cl100k_base")


def test_count_tokens_matches_tiktoken_cl100k_base() -> None:
    """中文+英文混合文本与 tiktoken cl100k_base 直接编码结果一致。"""
    text = "Hello world! 这是一个 mixed 中英文 token counting 测试。"
    expected = len(_ENC.encode(text))
    assert count_tokens(text, "glm-5.1") == expected
    assert expected > 0


def test_count_tokens_is_not_chars_div_4() -> None:
    """证明非 chars/4 粗估：与 chars//4 取值不同（多字节中文最易区分）。"""
    text = "你好世界 hello"
    assert count_tokens(text, "glm-5.1") != len(text) // 4


def test_count_tokens_empty_string_is_zero() -> None:
    assert count_tokens("", "glm-5.1") == 0


def test_count_tokens_default_model_is_glm_5_1() -> None:
    """不传 model 时默认走 glm-5.1（fallback cl100k_base）。"""
    assert DEFAULT_MODEL == "glm-5.1"
    text = "default model test"
    assert count_tokens(text) == len(_ENC.encode(text))


def test_count_tokens_unknown_model_falls_back_to_cl100k_base() -> None:
    """未知 model（如 anthropic/自定义）fallback 到 cl100k_base。"""
    text = "claude-3 opus some unknown model text 你好"
    expected = len(tiktoken.get_encoding("cl100k_base").encode(text))
    assert count_tokens(text, "claude-3-opus-20300101") == expected


def test_count_tokens_consistent_across_calls() -> None:
    """同一文本多次调用结果一致（LRU 缓存不引入不稳定性）。"""
    text = "稳定性测试 " * 50
    first = count_tokens(text, "glm-5.1")
    second = count_tokens(text, "glm-5.1")
    assert first == second
