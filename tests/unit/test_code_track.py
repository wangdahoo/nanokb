"""CodeLoader + CodeTrack 单测（方案 §3.4.2 + §3.4.3，Feature s1-feat-010）。

覆盖 4 条验收标准：
- AC #1：.py 文件经 CodeLoader.load 返回源码原文。
- AC #2：含 ``def foo()`` 且 foo 内调用 ``bar()`` 的 .py 经 CodeTrack.extract 产出
  ``(foo, calls, bar)`` 三元组（confidence=EXTRACTED）+ foo/bar 节点签名派生描述。
- AC #3：.js/.java 经 CodeTrack.extract 产出 defines/calls/contains 关系。
- AC #4：CodeTrack 抽取零 LLM Token（确定性抽取，不注入也不调用 llm）。

全部用真实 tree-sitter grammar（确定性，无需 mock），tmp_path 隔离。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanokb.config import Settings
from nanokb.loaders import CodeLoader, DocumentLoader, UnsupportedFormatError
from nanokb.models import (
    Chunk,
    Confidence,
    Document,
    ExtractionResult,
    Track,
)
from nanokb.stage2_extract import build_default_extractor
from nanokb.stage2_extract.base import Extractor
from nanokb.stage2_extract.code_track import (
    CodeTrack,
    spec_for_suffix,
    supported_code_suffixes,
)

FIXTURES_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


# ── 测试 doubles ─────────────────────────────────────────────────────


class _CountingLLMClient:
    """计数 LLM：记录 complete/embed 调用次数，用于断言 CodeTrack 零 Token。"""

    def __init__(self) -> None:
        self.complete_calls: int = 0
        self.embed_calls: int = 0

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.complete_calls += 1
        return json.dumps({"triples": [], "concepts": []})

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _doc(path: str, content: str) -> Document:
    """构造不含 chunks 的 Document（CodeTrack 不依赖 chunks，直接解析 content）。"""
    return Document(
        path=Path(path),
        content=content,
        sha256="x",
        format=Path(path).suffix.lstrip("."),
        chunks=[],
    )


# ===================================================================== #
# AC #1：CodeLoader.load(.py) 返回源码原文
# ===================================================================== #


def test_code_loader_supports_code_extensions() -> None:
    loader = CodeLoader()
    for ext in (".py", ".js", ".java", ".PY", ".JS"):
        assert loader.supports(Path(f"src{ext}"))


def test_code_loader_does_not_support_other_extensions() -> None:
    loader = CodeLoader()
    for ext in (".md", ".txt", ".pdf", ".docx", ".xyz", ""):
        assert not loader.supports(Path(f"src{ext}"))


def test_code_loader_load_py_returns_source_verbatim(tmp_path: Path) -> None:
    """AC #1：.py 经 CodeLoader.load 返回与磁盘内容逐字一致的源码。"""
    loader = CodeLoader()
    f = tmp_path / "mod.py"
    payload = "def foo(a, b):\n    return a + b\n\nx = foo(1, 2)\n"
    f.write_text(payload, encoding="utf-8")

    assert loader.load(f) == payload


def test_code_loader_load_js_and_java_returns_source(tmp_path: Path) -> None:
    loader = CodeLoader()
    js = tmp_path / "a.js"
    js.write_text("function f(){ return 1; }\n", encoding="utf-8")
    java = tmp_path / "C.java"
    java.write_text("class C { int m() { return 1; } }\n", encoding="utf-8")

    assert loader.load(js) == js.read_text(encoding="utf-8")
    assert loader.load(java) == java.read_text(encoding="utf-8")


def test_code_loader_load_fixture_sample_py() -> None:
    """AC #1 集成：CodeLoader 加载真实 fixture sample.py。"""
    loader = CodeLoader()
    sample = FIXTURES_RAW / "sample.py"
    assert sample.exists(), f"fixture missing: {sample}"

    text = loader.load(sample)
    assert text == sample.read_text(encoding="utf-8")
    assert "def greet(name):" in text
    assert "class Calculator:" in text


def test_code_loader_load_unsupported_extension_raises(tmp_path: Path) -> None:
    loader = CodeLoader()
    f = tmp_path / "a.md"
    f.write_text("nope", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        loader.load(f)


def test_code_loader_satisfies_protocol() -> None:
    assert isinstance(CodeLoader(), DocumentLoader)


# ===================================================================== #
# AC #2：.py 的 calls 关系 + 签名派生节点描述
# ===================================================================== #


_PY_FOO_CALLS_BAR = """def foo():
    bar()


def bar():
    pass
"""


def test_python_extract_emits_calls_triple_with_extracted_confidence() -> None:
    """AC #2：foo 内调用 bar → (foo, calls, bar)，confidence=EXTRACTED，track=CODE。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))

    calls = [
        t for t in result.triples
        if t.head == "foo" and t.relation == "calls" and t.tail == "bar"
    ]
    assert len(calls) == 1
    triple = calls[0]
    assert triple.confidence == Confidence.EXTRACTED
    assert triple.track == Track.CODE
    assert triple.chunk_index is None
    assert triple.source_file == "mod.py"


def test_python_extract_concept_descriptions_are_signature_derived() -> None:
    """AC #2：foo/bar 节点描述由签名派生（foo 带参数签名，bar 仅有名）。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))

    concepts = {c.name: c for c in result.concepts}
    # foo 已定义 → 描述含函数签名
    assert "foo" in concepts
    assert concepts["foo"].description == "function foo()"
    # bar 已定义（有 def bar()）→ 描述含签名
    assert "bar" in concepts
    assert concepts["bar"].description == "function bar()"


def test_python_extract_callee_without_definition_gets_name_derived_description() -> None:
    """AC #2：仅被调用、无定义的实体获得名派生描述（function <name>）。"""
    src = "def foo():\n    bar()\n"
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", src))

    concepts = {c.name: c for c in result.concepts}
    assert "bar" in concepts
    assert concepts["bar"].description == "function bar"


def test_python_extract_function_signature_includes_parameters() -> None:
    """签名派生描述携带 parameters 字段（剥离外层括号后）。"""
    src = "def add(a, b):\n    return a + b\n"
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", src))

    concepts = {c.name: c for c in result.concepts}
    assert concepts["add"].description == "function add(a, b)"


def test_python_extract_emits_defines_for_top_level_function() -> None:
    """顶层函数 → (module, defines, foo)。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))

    defines = [
        t for t in result.triples
        if t.relation == "defines" and t.tail == "foo"
    ]
    assert len(defines) == 1
    assert defines[0].head == "mod"  # 文件 stem 作为模块节点


def test_python_extract_class_and_method_emits_contains() -> None:
    """Python 类内 function_definition 降级为 method，并产出 (Class, contains, method)。"""
    src = (
        "class Calc:\n"
        "    def add(self, x, y):\n"
        "        return helper(x, y)\n"
    )
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", src))

    contains = [
        t for t in result.triples
        if t.relation == "contains" and t.tail == "add"
    ]
    assert len(contains) == 1
    assert contains[0].head == "Calc"

    concepts = {c.name: c for c in result.concepts}
    assert concepts["add"].description == "method Calc.add(self, x, y)"
    assert concepts["Calc"].description == "class Calc"
    # method 内调用 helper → (add, calls, helper)
    calls = [t for t in result.triples if t.relation == "calls"]
    assert ("add", "helper") in {(t.head, t.tail) for t in calls}


def test_python_extract_member_call_callee_is_method_name() -> None:
    """obj.method() 调用 → callee 取方法名（attribute 末段）。"""
    src = (
        "def foo():\n"
        "    obj.helper(1)\n"
        "    bar.baz.qux(2)\n"
    )
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", src))

    callees = {t.tail for t in result.triples if t.relation == "calls"}
    assert "helper" in callees
    assert "qux" in callees  # 多级属性取末段


def test_python_extract_self_call_is_skipped() -> None:
    """函数自调用（foo 内调 foo）→ 自环跳过，不产生 (foo, calls, foo)。"""
    src = "def foo():\n    foo()\n"
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", src))

    self_calls = [
        t for t in result.triples
        if t.relation == "calls" and t.head == t.tail
    ]
    assert self_calls == []


def test_python_extract_returns_extraction_result_type() -> None:
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", "x = 1\n"))
    assert isinstance(result, ExtractionResult)


# ===================================================================== #
# AC #3：.js / .java 产出 defines / calls / contains
# ===================================================================== #


_JS_SRC = (
    "function greet(name) {\n"
    "  return hello(name);\n"
    "}\n"
    "class Greeter {\n"
    "  wave() { greet(this); }\n"
    "}\n"
)


def test_javascript_extract_emits_defines_calls_contains() -> None:
    """AC #3：.js 产出 defines / calls / contains 三类关系。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("a.js", _JS_SRC))

    relations = {(t.head, t.relation, t.tail) for t in result.triples}
    # function_declaration → defines
    assert ("a", "defines", "greet") in relations
    # call inside greet → calls
    assert ("greet", "calls", "hello") in relations
    # class_declaration → defines
    assert ("a", "defines", "Greeter") in relations
    # method_definition → contains
    assert ("Greeter", "contains", "wave") in relations
    # call inside method → calls（caller 为方法名）
    assert ("wave", "calls", "greet") in relations

    concepts = {c.name: c for c in result.concepts}
    assert concepts["greet"].description == "function greet(name)"
    assert concepts["Greeter"].description == "class Greeter"
    assert concepts["wave"].description == "method Greeter.wave()"


def test_javascript_member_call_callee_is_property() -> None:
    """JS obj.m() → callee 为 property 名 m。"""
    src = "function f(){ obj.m(1); }\n"
    track = CodeTrack(Settings())
    result = track.extract(_doc("a.js", src))

    callees = {t.tail for t in result.triples if t.relation == "calls"}
    assert "m" in callees


_JAVA_SRC = (
    "public class Calc {\n"
    "  public int compute(int a) { return square(a); }\n"
    "  private int square(int x) { return mult(x, x); }\n"
    "}\n"
)


def test_java_extract_emits_contains_and_calls() -> None:
    """AC #3：.java 产出 method_declaration 的 contains + method_invocation 的 calls。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("Calc.java", _JAVA_SRC))

    relations = {(t.head, t.relation, t.tail) for t in result.triples}
    # method_declaration → class contains method
    assert ("Calc", "contains", "compute") in relations
    assert ("Calc", "contains", "square") in relations
    # method body 内 method_invocation → calls
    assert ("compute", "calls", "square") in relations
    assert ("square", "calls", "mult") in relations

    concepts = {c.name: c for c in result.concepts}
    assert concepts["Calc"].description == "class Calc"
    assert concepts["compute"].description == "method Calc.compute(int a)"
    assert concepts["square"].description == "method Calc.square(int x)"


def test_java_file_named_after_class_has_no_defines_self_loop() -> None:
    """Java 文件名与 public 类同名（Calc.java 含 class Calc）→ 不产生 (Calc, defines, Calc)。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("Calc.java", _JAVA_SRC))

    self_loops = [
        t for t in result.triples
        if t.relation == "defines" and t.head == t.tail
    ]
    assert self_loops == []


def test_java_method_invocation_on_receiver_uses_name_field() -> None:
    """Java helper.run(a) → callee 为 name 字段 run（非 receiver helper）。"""
    src = (
        "class C {\n"
        "  void m() { obj.helper(1); }\n"
        "}\n"
    )
    track = CodeTrack(Settings())
    result = track.extract(_doc("C.java", src))

    callees = {t.tail for t in result.triples if t.relation == "calls"}
    assert "helper" in callees
    assert "obj" not in callees


# ===================================================================== #
# AC #4：CodeTrack 抽取零 LLM Token
# ===================================================================== #


def test_code_track_constructor_takes_no_llm() -> None:
    """AC #4：CodeTrack 构造不注入 llm（确定性抽取，签名上零 Token 依赖）。"""
    # 仅 settings，无 llm 参数 —— 构造成功即证明不依赖 llm
    track = CodeTrack(Settings())
    assert isinstance(track, Extractor)


def test_code_track_extract_does_not_invoke_llm() -> None:
    """AC #4：通过 DefaultExtractor 走代码轨时，LLM.complete / embed 零调用。"""
    llm = _CountingLLMClient()
    extractor = build_default_extractor(llm, Settings())

    result = extractor.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))

    assert llm.complete_calls == 0
    assert llm.embed_calls == 0
    # 仍然成功抽取
    calls = [t for t in result.triples if t.relation == "calls"]
    assert ("foo", "bar") in {(t.head, t.tail) for t in calls}


def test_code_track_all_triples_are_code_track_extracted_confidence() -> None:
    """AC #4：所有三元组 track=CODE / confidence=EXTRACTED（确定性抽取标注）。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))

    assert len(result.triples) > 0
    for triple in result.triples:
        assert triple.track == Track.CODE
        assert triple.confidence == Confidence.EXTRACTED


# ===================================================================== #
# 默认分发与边界
# ===================================================================== #


def test_default_extractor_routes_code_to_code_track() -> None:
    """DefaultExtractor 对 .py 走 CodeTrack（零 Token），对 .md 走 SemanticTrack。"""
    llm = _CountingLLMClient()
    extractor = build_default_extractor(llm, Settings())

    code_result = extractor.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))
    assert llm.complete_calls == 0  # 代码轨不调 LLM
    assert any(t.relation == "calls" for t in code_result.triples)

    # .md 走语义轨 → 需带 chunk（SemanticTrack 仅处理 chunks）→ 调用 LLM
    md_doc = Document(
        path=Path("doc.md"),
        content="Some text.",
        sha256="x",
        format="md",
        chunks=[Chunk(index=0, text="Some text.", token_count=2, source_file="doc.md")],
    )
    md_result = extractor.extract(md_doc)
    assert llm.complete_calls >= 1
    assert isinstance(md_result, ExtractionResult)


def test_default_extractor_satisfies_protocol() -> None:
    llm = _CountingLLMClient()
    assert isinstance(build_default_extractor(llm, Settings()), Extractor)


def test_code_track_respects_code_languages_filter() -> None:
    """settings.code_languages 排除 python 时，.py 返回空结果。"""
    settings = Settings(code_languages=["javascript", "java"])
    track = CodeTrack(settings)
    result = track.extract(_doc("mod.py", _PY_FOO_CALLS_BAR))
    # python 未启用 → 空结果
    assert result.triples == []
    assert result.concepts == []


def test_code_track_unsupported_suffix_returns_empty() -> None:
    track = CodeTrack(Settings())
    result = track.extract(_doc("doc.md", "# heading"))
    assert result.triples == []
    assert result.concepts == []


def test_code_track_empty_content_returns_empty_or_module_only() -> None:
    """空源码：tree-sitter 解析成功但无定义/调用 → 至多仅 module 节点 concept。"""
    track = CodeTrack(Settings())
    result = track.extract(_doc("empty.py", ""))

    assert result.triples == []
    # 仅 module 节点 concept（无定义/调用）
    assert all(c.node_type == "module" for c in result.concepts)


def test_supported_code_suffixes_and_spec_lookup() -> None:
    assert supported_code_suffixes() == frozenset({".py", ".js", ".java"})
    assert spec_for_suffix(".py").language == "python"
    assert spec_for_suffix(".JS").language == "javascript"
    assert spec_for_suffix(".md") is None


# ===================================================================== #
# 默认 registry 集成（build_default_registry 注册 CodeLoader）
# ===================================================================== #


def test_build_default_registry_loads_py() -> None:
    """默认 registry 现支持 .py（CodeLoader 已接入）。"""
    from nanokb.pipeline import build_default_registry

    registry = build_default_registry()
    sample = FIXTURES_RAW / "sample.py"
    assert sample.exists()

    text = registry.load(sample)
    assert "def greet(name):" in text


def test_build_default_registry_still_loads_md() -> None:
    """CodeLoader 接入后 UnstructuredLoader 仍正常处理 .md。"""
    from nanokb.pipeline import build_default_registry

    registry = build_default_registry()
    sample = FIXTURES_RAW / "sample.md"
    assert sample.exists()

    text = registry.load(sample)
    assert "Transformer" in text
