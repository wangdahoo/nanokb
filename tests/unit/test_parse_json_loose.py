"""parse_json_loose 容错解析测试（方案 §3.4.1，Medium #4）。

纯函数测试，不依赖任何外部服务。
"""

from __future__ import annotations

from nanokb.llm.base import parse_json_loose


def test_clean_json_parsed_directly() -> None:
    assert parse_json_loose('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_markdown_fenced_json_extracted() -> None:
    raw = "Here you go:\n```json\n" + '{"triples": [], "concepts": []}' + "\n```"
    assert parse_json_loose(raw) == {"triples": [], "concepts": []}


def test_preamble_text_before_json() -> None:
    raw = 'Sure, the result is: {"name": "Transformer", "deps": ["Attn"]}'
    result = parse_json_loose(raw)
    assert result is not None
    assert result["name"] == "Transformer"
    assert result["deps"] == ["Attn"]


def test_nested_braces_balanced() -> None:
    raw = '{"outer": {"inner": [1, 2, {"k": 3}]}}'
    assert parse_json_loose(raw) == {"outer": {"inner": [1, 2, {"k": 3}]}}


def test_braces_inside_string_not_confused() -> None:
    # 字符串内部的花括号不应被误判为结构边界（引号感知扫描）
    raw = '{"code": "function() { return {}; }", "ok": true}'
    result = parse_json_loose(raw)
    assert result is not None
    assert result["ok"] is True
    assert result["code"] == "function() { return {}; }"


def test_escaped_quote_inside_string() -> None:
    raw = '{"text": "she said \\"hi {there}\\""}'
    result = parse_json_loose(raw)
    assert result is not None
    assert result["text"] == 'she said "hi {there}"'


def test_trailing_garbage_after_json() -> None:
    raw = '{"x": 1} Hope this helps!'
    assert parse_json_loose(raw) == {"x": 1}


def test_returns_first_dict_when_multiple_json_fragments() -> None:
    raw = 'first {"a": 1} then {"b": 2}'
    assert parse_json_loose(raw) == {"a": 1}


def test_no_json_returns_none() -> None:
    assert parse_json_loose("no json here at all") is None


def test_invalid_json_object_returns_none() -> None:
    assert parse_json_loose("{not valid json}") is None


def test_non_dict_json_returns_none() -> None:
    # 数组是合法 JSON 但非 dict → None
    assert parse_json_loose("[1, 2, 3]") is None


def test_empty_string_returns_none() -> None:
    assert parse_json_loose("") is None


def test_whitespace_padded_json() -> None:
    assert parse_json_loose('   \n  {"k": "v"}  \n') == {"k": "v"}


def test_unterminated_brace_returns_none() -> None:
    assert parse_json_loose('{"k": "v"') is None
